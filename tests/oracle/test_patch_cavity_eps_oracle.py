"""Dielectric ε-interface subpixel-smoothing oracle (committed lock for #325/#330).

Promoted from scripts/diagnostics/kottke_cavity_sanity.py — the clean, closed-PEC
cavity oracle that isolates the collocated-ε interface error the patch (#330)
residual is partly attributed to. cv05 (05_patch_antenna.py) was demoted to a
diagnostic-reporter (2026-07-15 first-principles review); its manifest gate_paths
delegate patch-accuracy evidence HERE, among others.

Two claims, with HONEST oracle labelling (per the review's B-Q2 correction):

  A) CONTROL — a PEC cavity FULLY filled with ε_r=4 has an EXACT analytic mode
     f_mnp = (c/2√ε)·√((m/a)²+(n/b)²+(p/d)²). There is NO dielectric interface,
     so Kottke dielectric subpixel smoothing must be a NO-OP (off == on) and both
     must match the exact analytic. This is a true analytic oracle.

  B) EFFECT — a PEC cavity HALF filled (ε_r=4 up to a mid-cell height, air above)
     puts the ε interface inside a cell, where the collocated scheme mis-registers
     it. Kottke subpixel smoothing must move the coarse-mesh resonance UP toward
     the converged value. This half-filled case is a DIRECTION+MAGNITUDE witness
     (the coarse on/off delta), NOT an exact-analytic oracle — a fully rigorous
     version would use the half-filled transcendental analytic (deferred).

Closed PEC cavity ⇒ energy-conserving, sharp modes, fast (no CPML/settling).
"""
import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from rfx import Simulation, Box  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402
from rfx.harminv import harminv  # noqa: E402

C0 = 299792458.0
A, B, D = 24e-3, 24e-3, 12e-3   # cavity dims (m)


def _fundamental(dx, kottke, h_diel, eps_r=4.0, fmax=12e9):
    """Lowest strong PEC-cavity resonance (GHz), or None."""
    sim = Simulation(freq_max=fmax, domain=(A, B, D), dx=dx, boundary="pec")
    if h_diel > 0:
        sim.add_material("diel", eps_r=eps_r, sigma=0.0)
        sim.add(Box((0, 0, 0), (A, B, h_diel)), material="diel")
    # off-symmetry source/probe couples to the low modes
    sim.add_source(position=(A * 0.3, B * 0.35, D * 0.5), component="ez",
                   waveform=GaussianPulse(f0=fmax * 0.45, bandwidth=1.6))
    sim.add_probe(position=(A * 0.7, B * 0.62, D * 0.5), component="ez")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(num_periods=180, subpixel_smoothing=kottke,
                      skip_preflight=True)
    ts = np.asarray(res.time_series).ravel()
    dt = float(res.dt)
    sig = ts[len(ts) // 4:]
    sig = sig - sig.mean()
    modes = [m for m in harminv(sig, dt, 2e9, fmax)
             if m.Q > 8 and m.amplitude > 1e-6]
    if not modes:
        return None
    modes.sort(key=lambda m: m.freq)
    return modes[0].freq / 1e9


def _analytic_full(eps_r=4.0):
    """Lowest full-fill PEC-box mode (exact)."""
    best = 1e99
    for m in range(3):
        for n in range(3):
            for p in range(3):
                if (m, n, p).count(0) >= 2:
                    continue
                f = (C0 / (2 * np.sqrt(eps_r))) * np.sqrt(
                    (m / A) ** 2 + (n / B) ** 2 + (p / D) ** 2)
                if 2e9 < f < best:
                    best = f
    return best / 1e9


def test_full_fill_kottke_is_noop_and_matches_exact_analytic():
    """CONTROL: no ε interface ⇒ Kottke is a no-op and both match the exact
    analytic cavity mode (subpixel smoothing must not perturb the bulk)."""
    f_an = _analytic_full()
    f_off = _fundamental(dx=1.0e-3, kottke=False, h_diel=D + 1e-6)
    f_on = _fundamental(dx=1.0e-3, kottke=True, h_diel=D + 1e-6)
    assert f_off is not None and f_on is not None
    # exact analytic oracle: coarse Yee is already very close with no interface
    assert abs(f_off - f_an) / f_an < 0.01, \
        f"full-fill off {f_off:.4f} vs analytic {f_an:.4f}"
    # Kottke must be a NO-OP with no dielectric interface present
    assert abs(f_on - f_off) / f_off < 1e-3, \
        f"Kottke must not change a homogeneous fill: off {f_off:.4f} on {f_on:.4f}"


def test_half_fill_kottke_moves_resonance_up_toward_truth():
    """EFFECT: a mid-cell ε interface ⇒ Kottke subpixel smoothing moves the
    coarse resonance UP (collocated over-estimates ε_eff → f too low → corrected).
    Direction+magnitude witness on the coarse on/off delta (not an exact oracle).
    """
    dx_c = 1.5e-3
    h = 5.25e-3   # interface at 3.5 cells (mid-cell, worst case) at dx=1.5mm
    f_off = _fundamental(dx=dx_c, kottke=False, h_diel=h)
    f_on = _fundamental(dx=dx_c, kottke=True, h_diel=h)
    assert f_off is not None and f_on is not None
    # Kottke moves the resonance UP (the ε-interface correction direction)
    assert f_on > f_off, \
        f"Kottke must raise the resonance at a sub-cell ε interface: off {f_off:.4f} on {f_on:.4f}"
    # and by a meaningful, not-noise amount (measured ~+3% on this fixture)
    shift_pct = 100 * (f_on - f_off) / f_off
    assert shift_pct > 0.5, \
        f"Kottke ε-interface correction too small to be real: {shift_pct:+.2f}%"


@pytest.mark.slow
def test_half_fill_kottke_closes_error_vs_fine_truth():
    """EFFECT (fine-truth, slow): the coarse kottke-ON resonance is closer to a
    fine-mesh converged reference than kottke-OFF — the same substrate-ε
    mechanism as the patch (#330), on a geometry whose answer converges.
    """
    dx_c = 1.5e-3
    h = 5.25e-3
    f_truth = _fundamental(dx=0.375e-3, kottke=False, h_diel=h)   # fine-mesh truth
    f_off = _fundamental(dx=dx_c, kottke=False, h_diel=h)
    f_on = _fundamental(dx=dx_c, kottke=True, h_diel=h)
    assert None not in (f_truth, f_off, f_on)
    err_off = abs(f_off - f_truth) / f_truth
    err_on = abs(f_on - f_truth) / f_truth
    assert err_on < err_off, \
        f"kottke-ON must be closer to fine truth {f_truth:.4f}: off err {err_off:.4f} on err {err_on:.4f}"

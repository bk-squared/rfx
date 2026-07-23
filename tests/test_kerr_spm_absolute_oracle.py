"""Reactive Kerr SPM — ABSOLUTE-magnitude oracle (#446, closes the fingerprint's open gate).

The sibling fingerprint oracle (test_kerr_spm_fingerprint.py) gates the *shape* of the reactive
Kerr SPM (positive sign, first-order ∝χ³A², A² scaling) but NOT the absolute magnitude vs the
textbook Δn=(3/8)χ³A². That was blocked (#446) by the pulsed-TFSF ⟨E²⟩ ambiguity: for a Gaussian
pulse ⟨E²⟩ is ill-defined (peak-A²/2 vs window-mean ≈ 6× spread), so the absolute ratio could not
be pinned. Two ingredients unblock it here:

  1. TRUE-CW source (``waveform='continuous_wave'``, normal incidence): at steady state the field
     is a single traveling wave of well-defined amplitude, so ⟨E²⟩ = A²/2 exactly.
  2. Rigorous local comparator: the measured wavenumber is the phase slope kx = ⟨k(x)⟩ across the
     probe span, so it integrates the LOCAL intensity. The weak reflection off the ε_eff step
     (Γ ≈ (3/16)χ³A²) leaves a small standing-wave ripple, so the target must use ⟨A(x)²⟩ (mean of
     the per-probe *squared* fundamental amplitude), NOT (mean A)². With ⟨A²⟩ the identity is exact:
         kx − kx0 = k0 · (3/8) · χ³ · ⟨A(x)²⟩          [ε_r=1 ⇒ n0=1, Δn=(3/8)χ³A²].

Physics: instantaneous Kerr D = ε0(ε_r E + χ³E³); for E=A cos ωt, E³=A³(¾cos ωt + ¼cos3ωt), so the
fundamental sees ε_eff = ε_r + ¾χ³A² ⇒ Δn = (3χ³A²)/(8n0). The 3/4 is the SPM degeneracy factor.

Validation (this session, CPU): the shipped D-based operator (#448) gives ratio 0.92–0.98 across
χ³∈{0.05,0.1,0.2,0.3} (mean 0.955±0.03), domain-invariant (0.60 m→0.925, 0.45 m→0.965), while the
pre-#448 increment operator gives 0.33 — so a gate at ratio∈[0.8,1.2] validates the D-based operator
and rejects the increment one. The residual ~5% is Yee numerical dispersion (the χ³=0 baseline kx0
itself sits +0.13% above the continuum k0). Harness:
docs/research_notes/experiments/kerr_cw_spm_absolute.py
"""
import numpy as np
import pytest

import rfx.materials.nonlinear as _nl
from rfx.api import Simulation
from rfx.grid import Grid
from rfx.geometry import Box

F0 = 5e9
W = 2 * np.pi * F0
C0 = 299792458.0
K0 = W / C0
DX = 0.002
NUM_PERIODS = 80
STEADY_FRAC = 0.45          # DFT / envelope over the last 45% (steady window, past ramp+fill)


def _measure(chi3, amp, domx, slab, probes):
    """(kx, ⟨A(x)²⟩, settle): discrete wavenumber from the f0-phasor phase slope over the steady CW
    window, the mean per-probe squared fundamental amplitude, and a settling witness (the fractional
    drift of the fundamental amplitude between the first and second half of the "steady" window —
    small ⇒ genuinely steady, not a transient) — χ³-only, ε_r=1 matched region."""
    dom = (domx, 0.06, 0.006)
    grid = Grid(freq_max=10e9, domain=dom, dx=DX, cpml_layers=10)
    dt = grid.dt
    sim = Simulation(freq_max=10e9, domain=dom, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_material("kerr", eps_r=1.0, chi3=chi3)
    sim.add(Box((slab[0], -1, -1), (slab[1], 1, 1)), material="kerr")
    sim.add_tfsf_source(f0=F0, amplitude=amp, polarization="ez", direction="+x",
                        waveform="continuous_wave")
    xprobes = np.arange(probes[0], probes[1], DX * 4)
    pidx = np.array([grid.position_to_index((float(x), 0.03, 0.003))[0] for x in xprobes], float)
    for x in xprobes:
        sim.add_probe((float(x), 0.03, 0.003), component="ez")
    ts = np.asarray(sim.run(num_periods=NUM_PERIODS, skip_preflight=True).time_series)
    nt = ts.shape[0]
    s0 = int((1.0 - STEADY_FRAC) * nt)
    tw = np.arange(s0, nt) * dt
    win = ts[s0:, :]
    T = win.shape[0] * dt
    phas = np.sum(win * (np.exp(-1j * W * tw) * dt)[:, None], axis=0)
    kx = abs(np.polyfit(pidx * DX, np.unwrap(np.angle(phas)), 1)[0])
    a_local = 2.0 * np.abs(phas) / T                     # per-probe fundamental amplitude
    # settling witness: the amplitude of the first vs second half of the window must agree
    h = win.shape[0] // 2
    amp_h1 = float(np.mean([np.max(np.abs(win[:h, i])) for i in range(win.shape[1])]))
    amp_h2 = float(np.mean([np.max(np.abs(win[h:, i])) for i in range(win.shape[1])]))
    settle = abs(amp_h2 - amp_h1) / amp_h1
    return kx, float(np.mean(a_local ** 2)), settle


def _ratio(chi3, amp, domx, slab, probes, kx0):
    kx, mean_a2, settle = _measure(chi3, amp, domx, slab, probes)
    dkx_meas = kx - kx0
    dkx_txt = (3.0 / 8.0) * chi3 * mean_a2 * K0
    return dkx_meas, dkx_txt, dkx_meas / dkx_txt, settle


# geometry: χ³-only slab spanning most of a CPML-terminated x-domain; probes strictly inside it
_LONG = dict(domx=0.60, slab=(0.06, 0.54), probes=(0.12, 0.44))
_SHORT = dict(domx=0.45, slab=(0.06, 0.39), probes=(0.12, 0.33))


@pytest.fixture(scope="module")
def oracle():
    # χ³=0 baseline is operator-independent (both operators are identity at χ³=0) → shared.
    kx0_L, _, _ = _measure(0.0, 1.0, **_LONG)
    dkx_meas, dkx_txt, r_dbased, settle = _ratio(0.20, 1.0, **_LONG, kx0=kx0_L)

    # falsifier: the pre-#448 increment operator (scale the E-update increment) — must NOT pass.
    shipped = _nl.apply_kerr_ade

    def _increment(state, e_prev, chi3_arr, eps_r_arr):
        ex_p, ey_p, ez_p = e_prev
        e2 = ex_p ** 2 + ey_p ** 2 + ez_p ** 2
        f = 1.0 / (1.0 + chi3_arr * e2 / eps_r_arr)
        return state._replace(ex=ex_p + (state.ex - ex_p) * f,
                              ey=ey_p + (state.ey - ey_p) * f,
                              ez=ez_p + (state.ez - ez_p) * f)
    try:
        _nl.apply_kerr_ade = _increment                  # picked up by the lazy import in run()
        _, _, r_increment, _ = _ratio(0.20, 1.0, **_LONG, kx0=kx0_L)
    finally:
        _nl.apply_kerr_ade = shipped

    # domain-size invariance (mandate: an ADDED gate needs a falsifier + domain-invariance)
    kx0_S, _, _ = _measure(0.0, 1.0, **_SHORT)
    _, _, r_short, _ = _ratio(0.20, 1.0, **_SHORT, kx0=kx0_S)

    return {"dkx_meas": dkx_meas, "dkx_txt": dkx_txt, "r_dbased": r_dbased,
            "r_increment": r_increment, "r_short": r_short, "settle": settle}


@pytest.mark.slow
def test_kerr_spm_absolute_magnitude(oracle):
    """The D-based operator reproduces the textbook Δk=(3/8)χ³⟨A²⟩k0 to within ±20% (measured ≈0.93,
    residual = Yee dispersion). Excludes the increment operator (0.33) AND the pulsed artifact (0.59)."""
    r = oracle["r_dbased"]
    assert 0.80 <= r <= 1.20, (
        f"absolute SPM ratio {r:.3f} outside [0.80,1.20]: "
        f"Δk_meas={oracle['dkx_meas']:.4f} Δk_txt={oracle['dkx_txt']:.4f}")


@pytest.mark.slow
def test_kerr_spm_absolute_positive_sign(oracle):
    """Δk > 0 — a reactive index INCREASE (self-focusing). A dissipative/linear operator gives ~0."""
    assert oracle["dkx_meas"] > 0.1, f"Δk_meas={oracle['dkx_meas']:.4f} not a clear positive shift"


@pytest.mark.slow
def test_kerr_spm_gate_discriminates_operator(oracle):
    """The gate BINDS PHYSICS, not any operator: the pre-#448 increment operator underestimates the
    magnitude (ratio ≈0.33) and FAILS the [0.8,1.2] gate — so a green here means the D-based
    constitutive solve, not merely 'some Kerr term fired'."""
    assert oracle["r_increment"] < 0.60, (
        f"increment falsifier ratio {oracle['r_increment']:.3f} should be well below the "
        f"passing band — the gate would not discriminate the operator")
    assert oracle["r_dbased"] - oracle["r_increment"] > 0.30, (
        f"D-based ({oracle['r_dbased']:.3f}) and increment ({oracle['r_increment']:.3f}) "
        f"not clearly separated")


@pytest.mark.slow
def test_kerr_spm_absolute_domain_invariant(oracle):
    """The ratio is a physical property, not a domain artifact: a 0.60 m and a 0.45 m domain agree."""
    assert abs(oracle["r_dbased"] - oracle["r_short"]) < 0.15, (
        f"ratio moved with domain size: long={oracle['r_dbased']:.3f} short={oracle['r_short']:.3f} "
        f"(an artifact would swing; physics is invariant)")


@pytest.mark.slow
def test_kerr_spm_measured_at_steady_state(oracle):
    """Settling witness: the DFT/phase-slope window is genuinely steady CW, not a transient — the
    fundamental amplitude drifts <5% between the first and second half of the window. Guards against
    a future domain/num_periods change silently sampling the ramp/fill transient."""
    assert oracle["settle"] < 0.05, (
        f"steady window not settled: amplitude drift {oracle['settle']:.3f} between window halves "
        f"(the ⟨E²⟩=A²/2 assumption requires a steady CW plateau)")

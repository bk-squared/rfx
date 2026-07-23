"""Reactive Kerr SPM fingerprint — quantitative-scale validation of the #440 reactive fix.

The reactive Kerr (#440) shifts the phase velocity (self-phase modulation). Measured in a χ³-ONLY
matched region (ε_r=1, so impedance-matched to vacuum ⇒ no standing wave ⇒ a clean traveling wave
of amplitude A, ⟨E²⟩=A²/2), via the discrete wavenumber k_x from the f0-phasor phase slope (the
numerical-dispersion machinery). This gates the FINGERPRINT — the aspects that are cleanly
established, discriminating, and robust:

  (1) REACTIVE, correct sign: Δk_x = k_x(χ³) − k_x(0) > 0 (an index INCREASE slows the wave). The
      pre-#440 dissipative operator was amplitude-only / phase-neutral ⇒ Δk_x ≈ 0; a linear solver
      gives 0. So Δk_x>0 discriminates the reactive Kerr.
  (2) FIRST-ORDER SPM: Δk_x/(χ³·A²) is CONSTANT across χ³ and A (measured ≈8.2, within 2.5%).

The ABSOLUTE magnitude vs textbook Δn=(3/8)χ³A² is gated separately in
test_kerr_spm_absolute_oracle.py (#446), which uses a TRUE-CW source (waveform='continuous_wave')
to remove the pulsed-⟨E²⟩ ambiguity and confirms the shipped D-based operator (#448) reaches
ratio ≈0.95 (the pre-#448 increment operator underestimated it at ~0.33×).
Harness: docs/research_notes/experiments/kerr_cw_spm_oracle.py
"""
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.geometry import Box

F0 = 5e9
W = 2 * np.pi * F0
DX = 0.002
DOM = (0.60, 0.06, 0.006)


def _measure(chi3, amp):
    """(k_x, A): discrete wavenumber from the phase slope + in-medium amplitude, χ³-only ε_r=1."""
    grid = Grid(freq_max=10e9, domain=DOM, dx=DX, cpml_layers=10)
    dt = grid.dt
    sim = Simulation(freq_max=10e9, domain=DOM, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_material("kerr", eps_r=1.0, chi3=chi3)
    sim.add(Box((0.06, -1, -1), (0.54, 1, 1)), material="kerr")
    sim.add_tfsf_source(f0=F0, bandwidth=0.3, amplitude=amp, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    xprobes = np.arange(0.12, 0.44, DX * 4)
    pidx = np.array([grid.position_to_index((float(x), 0.03, 0.003))[0] for x in xprobes], float)
    for x in xprobes:
        sim.add_probe((float(x), 0.03, 0.003), component="ez")
    ts = np.asarray(sim.run(num_periods=45, skip_preflight=True).time_series)
    t = np.arange(ts.shape[0]) * dt
    phas = np.sum(ts * (np.exp(-1j * W * t) * dt)[:, None], axis=0)
    kx = abs(np.polyfit(pidx * DX, np.unwrap(np.angle(phas)), 1)[0])
    A = float(np.mean([np.max(np.abs(ts[:, i])) for i in range(ts.shape[1])]))
    return kx, A


@pytest.fixture(scope="module")
def spm():
    kx0, _ = _measure(0.0, 1.0)                         # linear reference
    pts = {}
    for chi3, amp in [(0.05, 1.0), (0.10, 1.0), (0.10, 1.4)]:
        kx, A = _measure(chi3, amp)
        pts[(chi3, amp)] = {"kx": kx, "A": A, "dkx": kx - kx0,
                            "fp": (kx - kx0) / (chi3 * A ** 2)}
    return {"kx0": kx0, "pts": pts}


@pytest.mark.slow
def test_kerr_spm_is_reactive_positive(spm):
    """Δk_x > 0 for χ³>0 — a reactive index INCREASE (correct sign). The pre-#440 dissipative /
    a linear operator gives Δk_x ≈ 0, so this discriminates the reactive Kerr."""
    for key, p in spm["pts"].items():
        assert p["dkx"] > 0.1, f"{key}: Δk_x={p['dkx']:.4f} not a clear positive (reactive) shift"


@pytest.mark.slow
def test_kerr_spm_is_first_order(spm):
    """Δk_x/(χ³·A²) is constant across χ³ and A — the first-order SPM fingerprint (∝ χ³·intensity)."""
    fps = np.array([p["fp"] for p in spm["pts"].values()])
    assert fps.max() / fps.min() < 1.15, f"SPM not first-order: Δk_x/(χ³A²) spread {fps}"


@pytest.mark.slow
def test_kerr_spm_scales_with_intensity(spm):
    """At fixed χ³, a larger drive (A 1.0→1.4) gives a proportionally larger shift (Δk_x ∝ A²)."""
    p1 = spm["pts"][(0.10, 1.0)]
    p2 = spm["pts"][(0.10, 1.4)]
    ratio = (p2["dkx"] / p1["dkx"]) / (p2["A"] ** 2 / p1["A"] ** 2)
    assert abs(ratio - 1.0) < 0.15, f"Δk_x should scale as A²: normalized ratio {ratio:.3f}"

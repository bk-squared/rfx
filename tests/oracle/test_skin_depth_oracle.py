"""Skin depth in a GOOD CONDUCTOR (σ ≫ ωε) vs analytic (#403 blind spot).

test_lossy_material_validation.py gates plane-wave attenuation α(f) only in the LOW-LOSS regime
(tan δ ∈ [0.1, 0.3] ⇒ σ/ωε0 ≈ 0.4–1.2). The GOOD-CONDUCTOR limit (σ/ωε0 ≫ 1) is distinct physics:
    γ = (1 + j)/δ,   δ = √(2/(ω·μ·σ))   (skin depth)
so the field into the conductor is E(z) ∝ e^{-z/δ}·e^{-jz/δ} — amplitude AND phase share the SAME
length scale δ (the 45° signature α=β), unlike low loss where β ≫ α. This oracle measures both.

Method: TFSF plane wave into a thick conductor slab (σ/ωε0=20, δ≈10 cells); DFT the field at probes
INSIDE the conductor; fit ln|E(z)| slope ⇒ δ_amp and ∠E(z) slope ⇒ δ_phase. Measured (CPU): δ_amp
err 3.3%, δ_phase err 0.4%, δ_amp/δ_phase=1.04. Harness: docs/research_notes/experiments/skin_depth_oracle.py
"""
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.geometry import Box
from rfx.core.yee import EPS_0, MU_0

C0 = 299792458.0
F0 = 5e9
W = 2 * np.pi * F0
DX = 0.3e-3
X_ENTRY = 0.020
_ZD = np.arange(0.5, 3.6, 0.35)          # probe depths in units of δ


def _measure_delta(sigma_over_we0):
    """Return (δ_amp, δ_phase, δ_analytic) [m] for a conductor of the given σ/(ωε0)."""
    sigma = sigma_over_we0 * W * EPS_0
    delta = np.sqrt(2.0 / (W * MU_0 * sigma))
    slab = 7 * delta
    dom = (X_ENTRY + slab + 0.010, 0.02, 0.006)
    sim = Simulation(freq_max=10e9, domain=dom, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_material("cond", eps_r=1.0, sigma=sigma)
    sim.add(Box((X_ENTRY, -1, -1), (X_ENTRY + slab, 1, 1)), material="cond")
    sim.add_tfsf_source(f0=F0, bandwidth=0.3, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    zd = _ZD * delta
    for xp in X_ENTRY + zd:
        sim.add_probe((float(xp), 0.01, 0.003), component="ez")
    res = sim.run(num_periods=60, skip_preflight=True)
    ts = np.asarray(res.time_series)
    dt = float(res.dt)
    t = np.arange(ts.shape[0]) * dt
    E = np.sum(ts * (np.exp(-1j * W * t) * dt)[:, None], axis=0)
    d_amp = -1.0 / np.polyfit(zd, np.log(np.abs(E)), 1)[0]
    d_phase = -1.0 / np.polyfit(zd, np.unwrap(np.angle(E)), 1)[0]
    return d_amp, d_phase, delta


@pytest.fixture(scope="module")
def good_conductor():
    d_amp, d_phase, delta = _measure_delta(20.0)   # σ/ωε0 = 20 (good conductor)
    return {"amp": d_amp, "phase": d_phase, "analytic": delta}


@pytest.mark.slow
def test_skin_depth_amplitude_matches_analytic(good_conductor):
    """The field amplitude decays with the analytic skin depth δ=√(2/(ωμσ))."""
    g = good_conductor
    assert abs(g["amp"] - g["analytic"]) / g["analytic"] < 0.08, (
        f"δ_amp={g['amp']*1e3:.3f}mm vs analytic {g['analytic']*1e3:.3f}mm")


@pytest.mark.slow
def test_skin_depth_phase_matches_analytic(good_conductor):
    """The field phase advances with the same analytic skin depth (β = 1/δ)."""
    g = good_conductor
    assert abs(g["phase"] - g["analytic"]) / g["analytic"] < 0.08, (
        f"δ_phase={g['phase']*1e3:.3f}mm vs analytic {g['analytic']*1e3:.3f}mm")


@pytest.mark.slow
def test_skin_depth_45deg_signature_vs_lowloss_control(good_conductor):
    """DISCRIMINATING: a good conductor has α=β (δ_amp≈δ_phase, the 45° signature); a low-loss
    medium does NOT (β≫α ⇒ δ_phase≪δ_amp). Gating the equality catches a solver that models the
    conductive-loss path in the wrong regime."""
    g = good_conductor
    assert abs(g["amp"] / g["phase"] - 1.0) < 0.12, (
        f"good conductor should give δ_amp≈δ_phase, got ratio {g['amp']/g['phase']:.3f}")
    # low-loss control (σ/ωε0=0.3): β ≫ α ⇒ δ_amp/δ_phase far from 1 — proves the gate is not trivial
    a, p, _ = _measure_delta(0.3)
    assert a / p > 2.0, (
        f"low-loss control should have δ_amp≫δ_phase (β≫α), got ratio {a/p:.3f} — "
        "the 45° signature gate would be non-discriminating")

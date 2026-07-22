"""Reflection Γ off a LOSSY half-space vs the complex-ε Fresnel (#403 blind spot).

rfx already gates the two neighbouring physics:
  - bulk attenuation α(f) and cavity Q≈1/tanδ  → tests/test_lossy_material_validation.py
  - LOSSLESS slab reflection Γ=(1-√εr)/(1+√εr) → tests/test_forward_tfsf_fresnel_groundtruth.py
It did NOT gate the REFLECTION off a lossy interface — whether the σ loss term correctly
enters the surface impedance the wave reflects from. This is that binding oracle.

Physics (normal incidence, vacuum → lossy half-space, e^{+jωt} convention):
    ε_c = ε_r - j·σ/(ω·ε0)      (loss ⇒ Im(ε_c) < 0)
    n   = √ε_c                  (principal sqrt ⇒ Im(n) < 0, decaying wave)
    Γ   = (1 - n)/(1 + n)       (COMPLEX; both |Γ| and ∠Γ move with σ)

Method reuses the #404/#414-validated two-run plateau extractor
(``rfx.probes.fresnel_reflection_coefficient``), injecting σ via
``forward(sigma_override=…)``; a front-face time-gate makes it a half-space Fresnel and
the loss further suppresses the back-face echo. σ=0 is the CONTROL — it must recover the
lossless 0.333/180° anchor AND fixes the εr/σ-independent Yee half-cell reference-plane
phase offset (≈10.6°, matching the lossless sibling test's documented ~11°).

Measured envelope (CPU, εr=4): |Γ| err vs analytic 0.8/0.9/1.2% at σ/ωε0=0/2/4; calibrated
phase matches to ≤0.5°; |Γ| moves 0.333→0.385→0.452 — the σ>0 cases are ~13/36% above the
lossless value, so a loss-blind reflection (stuck at 0.333/180°) FAILS by far more than the
~1% discretization envelope. Harness: docs/research_notes/experiments/lossy_reflection_oracle.py
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.probes import fresnel_reflection_coefficient
from rfx.core.yee import EPS_0

F0 = 5e9
DOMAIN = (0.50, 0.12, 0.006)
DX = 0.002
X_IFACE, X_SLAB_END = 0.15, 0.40
PLATEAU = (0.05, 0.13)
C0 = 299792458.0
EPS_R = 4.0
_W = 2.0 * np.pi * F0
# σ/(ωε0) = 0 (control), 2, 4 — strong enough to move |Γ| well outside the Yee envelope
SIGMA_FACTORS = (0.0, 2.0, 4.0)
SIGMAS = tuple(fac * _W * EPS_0 for fac in SIGMA_FACTORS)


def _eps_c(sigma):
    return EPS_R - 1j * sigma / (_W * EPS_0)


def _gamma_analytic(sigma):
    n = np.sqrt(_eps_c(sigma))            # principal sqrt ⇒ Im(n) < 0
    return (1.0 - n) / (1.0 + n)


def _build():
    grid = Grid(freq_max=10e9, domain=DOMAIN, dx=DX, cpml_layers=10)
    xi = grid.position_to_index((X_IFACE, 0.06, 0.003))[0]
    xe = grid.position_to_index((X_SLAB_END, 0.06, 0.003))[0]
    xprobes = np.arange(PLATEAU[0], PLATEAU[1], DX)
    probe_idx = np.array([grid.position_to_index((float(xp), 0.06, 0.003))[0] for xp in xprobes])
    probe_distances = (xi - probe_idx).astype(np.float64) * DX
    # time-gate: stop before the FASTEST (εr=2-equivalent) back-face echo reaches the plateau
    n_fast = np.sqrt(2.0)
    t_back = (2 * (X_IFACE - PLATEAU[1]) / C0) + (2 * (X_SLAB_END - X_IFACE) / (C0 / n_fast))
    ns = int(0.9 * t_back / grid.dt)
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, bandwidth=0.5, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    for xp in xprobes:
        sim.add_probe((float(xp), 0.06, 0.003), component="ez")
    return sim, grid.shape, xi, xe, probe_distances, grid.dt, ns


@pytest.fixture(scope="module")
def lossy_run():
    sim, shape, xi, xe, dists, dt, ns = _build()
    inc = sim.forward(eps_override=jnp.ones(shape, jnp.float32),
                      sigma_override=jnp.zeros(shape, jnp.float32),
                      n_steps=ns, checkpoint=False, skip_preflight=True).time_series
    gamma = {}
    for sigma in SIGMAS:
        eps = jnp.ones(shape, jnp.float32).at[xi:xe, :, :].set(EPS_R)
        sig = jnp.zeros(shape, jnp.float32).at[xi:xe, :, :].set(sigma)
        tot = sim.forward(eps_override=eps, sigma_override=sig, n_steps=ns,
                          checkpoint=False, skip_preflight=True).time_series
        gamma[sigma] = complex(fresnel_reflection_coefficient(
            tot, inc, f0=F0, dt=dt, probe_distances=dists, n_gate=ns))
    return {"gamma": gamma}


@pytest.mark.slow
def test_lossless_control_recovers_fresnel(lossy_run):
    """σ=0 anchor: recovers the lossless Fresnel |Γ|=1/3 and is passive (<1)."""
    g = lossy_run["gamma"][SIGMAS[0]]
    assert abs(g) < 1.0, f"nonphysical |Γ|={abs(g):.3f}≥1"
    assert abs(abs(g) - 1.0 / 3.0) < 0.04, f"control |Γ|={abs(g):.3f} vs lossless 0.333"


@pytest.mark.slow
@pytest.mark.parametrize("sigma", SIGMAS)
def test_lossy_reflection_magnitude_vs_complex_fresnel(lossy_run, sigma):
    """|Γ| tracks the complex-ε Fresnel |(1-√ε_c)/(1+√ε_c)| as σ enters the surface impedance."""
    g = lossy_run["gamma"][sigma]
    ga = _gamma_analytic(sigma)
    assert abs(g) < 1.0, f"nonphysical |Γ|={abs(g):.3f}≥1 (σ={sigma:.4f})"
    assert abs(abs(g) - abs(ga)) < 0.03, (
        f"|Γ|={abs(g):.4f} vs complex-ε Fresnel {abs(ga):.4f} (σ/ωε0={sigma/(_W*EPS_0):.1f})")


@pytest.mark.slow
def test_loss_moves_reflection_discriminating(lossy_run):
    """DISCRIMINATING: loss must MOVE |Γ| away from the lossless value, matching the analytic
    shift. A solver that ignored σ in the reflection would keep |Γ|≈0.333 and fail here."""
    g0 = abs(lossy_run["gamma"][SIGMAS[0]])
    g4 = abs(lossy_run["gamma"][SIGMAS[2]])
    shift_meas = g4 - g0
    shift_ana = abs(_gamma_analytic(SIGMAS[2])) - abs(_gamma_analytic(SIGMAS[0]))
    assert shift_meas > 0.06, f"loss did not move |Γ| (Δ={shift_meas:.4f}); reflection looks loss-blind"
    assert abs(shift_meas - shift_ana) < 0.03, (
        f"measured |Γ| shift {shift_meas:.4f} ≠ analytic {shift_ana:.4f}")


@pytest.mark.slow
def test_lossy_reflection_phase_shift_vs_analytic(lossy_run):
    """After calibrating the εr/σ-independent Yee half-cell reference-plane offset at σ=0,
    the σ-dependent phase shift of Γ matches the complex-ε Fresnel prediction."""
    g0 = lossy_run["gamma"][SIGMAS[0]]
    ph0_meas, ph0_ana = np.degrees(np.angle(g0)), np.degrees(np.angle(_gamma_analytic(SIGMAS[0])))
    offset = ((ph0_meas - ph0_ana + 180) % 360) - 180   # reference-plane constant (~10.6°)
    for sigma in SIGMAS[1:]:
        g = lossy_run["gamma"][sigma]
        ph_cal = np.degrees(np.angle(g)) - offset
        ph_ana = np.degrees(np.angle(_gamma_analytic(sigma)))
        dph = abs(((ph_cal - ph_ana + 180) % 360) - 180)
        assert dph < 6.0, (
            f"calibrated ∠Γ off by {dph:.1f}° at σ/ωε0={sigma/(_W*EPS_0):.1f} "
            f"(meas {np.degrees(np.angle(g)):.1f}°, analytic {ph_ana:.1f}°)")

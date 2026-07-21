"""Differentiable OBLIQUE Fresnel |Γ|(θ) magnitude via forward() — ground-truth-gated.

Extends the normal-incidence complex Γ helper (#419) to oblique incidence (the RIS/absorber
condition), MAGNITUDE ONLY. forward() returns the raw complex Bloch envelope for oblique, so
``oblique_reflection_magnitude`` uses the +j (conjugate) DFT kernel and a de-embed-free
magnitude ratio (a de-embedded complex mean cancels at oblique — the reflected phase spans
many π across the plateau).

Validated vs analytic oblique Fresnel ``fresnel_r_te(θ,εr)``: |Γ| err 7.2% / 3.9% @ θ=30°/45°
(εr=4); d|Γ|/dε AD==FD. 60° is CPML-contaminated (memory) so it is NOT gated. PHASE (complex
Γ) is a documented open follow-up (needs the exact discrete k_x de-embed).
Harness: docs/research_notes/experiments/i404_oblique_20260720/oblique_gamma_validate.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.probes import oblique_reflection_magnitude, fresnel_r_te

F0, BW = 5e9, 0.15
DOMAIN = (0.60, 0.12, 0.006)
DX = 0.002
X_IFACE, X_SLAB_END = 0.15, 0.50
PLATEAU = (0.05, 0.13)
C0 = 299792458.0
EPS_R = 4.0


def _build(theta):
    grid = Grid(freq_max=10e9, domain=DOMAIN, dx=DX, cpml_layers=10)
    xi = grid.position_to_index((X_IFACE, 0.06, 0.003))[0]
    xe = grid.position_to_index((X_SLAB_END, 0.06, 0.003))[0]
    xprobes = np.arange(PLATEAU[0], PLATEAU[1], DX)
    n_slab = np.sqrt(EPS_R)
    t_back = (2 * (X_IFACE - PLATEAU[1]) / C0) + (2 * (X_SLAB_END - X_IFACE) / (C0 / n_slab))
    ns = min(int(0.9 * t_back / grid.dt), 1700)
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, bandwidth=BW, polarization="ez", direction="+x",
                        angle_deg=theta, waveform="modulated_gaussian")
    for xp in xprobes:
        sim.add_probe((float(xp), 0.06, 0.003), component="ez")
    return sim, grid.shape, xi, xe, grid.dt, ns


def _eps_slab(shape, xi, xe, eps):
    a = jnp.ones(shape, jnp.float32)
    return a if eps == 1.0 else a.at[xi:xe, :, :].set(eps)


def _series(sim, eps_arr, ns, ckpt=False):
    return sim.forward(eps_override=eps_arr, n_steps=ns, checkpoint=ckpt,
                       skip_preflight=True).time_series  # complex for oblique


@pytest.mark.slow
@pytest.mark.parametrize("theta,tol", [(30.0, 0.10), (45.0, 0.08)])
def test_oblique_fresnel_magnitude_vs_analytic(theta, tol):
    """|Γ|(θ) matches analytic oblique Fresnel R_TE and is passive; +j kernel is mandatory."""
    sim, shape, xi, xe, dt, ns = _build(theta)
    inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns)
    tot = _series(sim, _eps_slab(shape, xi, xe, EPS_R), ns)
    g = float(oblique_reflection_magnitude(tot, inc, f0=F0, dt=dt, n_gate=ns))
    ga = abs(fresnel_r_te(theta, EPS_R))
    assert g < 1.0, f"nonphysical |Γ|={g:.3f}≥1 (θ={theta})"
    assert abs(g - ga) / ga < tol, f"|Γ|={g:.3f} vs R_TE={ga:.3f} (θ={theta}, {abs(g-ga)/ga*100:.1f}%)"


@pytest.mark.slow
def test_oblique_reflection_magnitude_differentiable():
    """d|Γ|/dε flows through the checkpointed oblique forward and matches finite difference."""
    sim, shape, xi, xe, dt, ns = _build(30.0)
    inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns, ckpt=True)

    def absg(eps):
        tot = _series(sim, jnp.ones(shape, jnp.float32).at[xi:xe, :, :].set(eps), ns, ckpt=True)
        return oblique_reflection_magnitude(tot, inc, f0=F0, dt=dt, n_gate=ns)

    g_ad = float(jax.grad(absg)(EPS_R))
    h = 0.05
    g_fd = (float(absg(EPS_R + h)) - float(absg(EPS_R - h))) / (2 * h)
    assert np.isfinite(g_ad), "NaN/Inf gradient (P0)"
    assert abs(g_ad - g_fd) < 0.05 * abs(g_fd) + 1e-6, f"AD={g_ad:.4e} vs FD={g_fd:.4e}"


def test_oblique_magnitude_kernel_sign_and_shape_guard():
    """FAST synthetic contract: +j kernel on a complex envelope recovers a planted |Γ|; a
    -j kernel would average the incident to noise. Also the shape guard fires."""
    ns, nprobe, dt = 400, 6, 1.0 / (F0 * 20)
    t = np.arange(ns) * dt
    env = np.exp(-1j * 2 * np.pi * F0 * t)             # analytic envelope P ∝ e^{-j2πf0t}
    gmag_true = 0.4
    inc = np.outer(env, np.ones(nprobe))
    # reflected envelope with random per-probe phases (magnitude gmag_true)
    phases = np.linspace(0, 3 * np.pi, nprobe)
    refl = gmag_true * np.outer(env, np.exp(1j * phases))
    tot = inc + refl
    g = float(oblique_reflection_magnitude(jnp.asarray(tot), jnp.asarray(inc), f0=F0, dt=dt))
    assert abs(g - gmag_true) < 1e-3, f"planted |Γ|={gmag_true} recovered {g:.4f}"
    with pytest.raises(ValueError):
        oblique_reflection_magnitude(jnp.asarray(tot[:, 0]), jnp.asarray(inc[:, 0]), f0=F0, dt=dt)

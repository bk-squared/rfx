"""Differentiable oblique COMPLEX Fresnel Γ(θ) — magnitude AND PHASE (#423, follow-up to #422).

`oblique_reflection_coefficient` recovers the reflected PHASE (the RIS phase-steering knob) that
`oblique_reflection_magnitude` (#422) could not: the de-rotation slope is measured from the
INCIDENT wave itself (self-calibrating → the exact discrete k_x), so the complex mean no longer
cancels. Two gates:

  • FAST synthetic contract (no FDTD): a two-run signal with a planted complex Γ and a planted
    phase ramp — the helper recovers Γ (amplitude+phase) and stays on the AD tape.
  • SLOW FDTD: oblique forward() reflection off a dielectric slab vs analytic ``fresnel_r_te``:
    |Γ| ~4-7% and ∠Γ ~180° (up to the ~11° Yee reference-plane offset) at θ=30°/45°; the
    complex-target RIS loss |Γ-Γ_target|² is differentiable (AD==FD).

Harness: docs/research_notes/experiments/i404_oblique_20260720/oblique_complex_gamma_diff.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.probes import oblique_reflection_coefficient, fresnel_r_te

F0, BW = 5e9, 0.15
DOMAIN = (0.60, 0.12, 0.006)
DX = 0.002
X_IFACE, X_SLAB_END = 0.15, 0.50
PLATEAU = (0.05, 0.13)
C0 = 299792458.0
EPS_R = 4.0


# --------------------------------------------------------------------------- #
# FAST synthetic contract: extraction math + differentiability, no FDTD.
# --------------------------------------------------------------------------- #
def _synth(gamma_true, kx, idx, iface, f0=F0, n_periods=6, spp=24):
    """Complex Bloch-envelope two-run series carrying a planted Γ and phase ramp e^{-j k_x x}."""
    dt = 1.0 / (f0 * spp)
    n = n_periods * spp
    env = np.exp(-1j * 2 * np.pi * f0 * np.arange(n) * dt)   # analytic envelope P ∝ e^{-j2πf0t}
    ip = np.exp(-1j * kx * idx)                              # incident phasor (phase ramp)
    rp = gamma_true * np.exp(-1j * kx * (2 * iface - idx))   # reflected: round-trip to interface
    inc = ip[None, :] * env[:, None]
    tot = (ip + rp)[None, :] * env[:, None]
    return jnp.asarray(tot), jnp.asarray(inc), dt


def test_oblique_phase_recovers_planted_complex_gamma():
    """Helper recovers a planted complex Γ (amplitude+phase) and is grad-safe."""
    idx = np.arange(6, dtype=float)
    iface = 14.0
    kx = 0.35
    for g_true in (-0.4 + 0j, 0.3j, 0.25 - 0.15j):
        tot, inc, dt = _synth(g_true, kx, idx, iface)
        g = complex(oblique_reflection_coefficient(
            tot, inc, f0=F0, dt=dt, probe_index=idx, interface_index=iface))
        assert abs(g - g_true) < 2e-2, f"Γ={g:.4f} vs planted {g_true:.4f}"

    tot, inc, dt = _synth(-0.4 + 0j, kx, idx, iface)
    refl = tot - inc
    g_target = 0.2 + 0.1j

    def loss(theta):
        g = oblique_reflection_coefficient(inc + theta * refl, inc, f0=F0, dt=dt,
                                           probe_index=idx, interface_index=iface)
        return jnp.abs(g - g_target) ** 2

    g_ad = float(jax.grad(loss)(1.0))
    h = 1e-3
    g_fd = (float(loss(1.0 + h)) - float(loss(1.0 - h))) / (2 * h)
    assert np.isfinite(g_ad), "NaN/Inf gradient (P0)"
    assert abs(g_ad - g_fd) < 1e-2 * abs(g_fd) + 1e-6, f"AD={g_ad:.4e} vs FD={g_fd:.4e}"

    with pytest.raises(ValueError):
        oblique_reflection_coefficient(tot[:, 0], inc[:, 0], f0=F0, dt=dt,
                                       probe_index=idx, interface_index=iface)


# --------------------------------------------------------------------------- #
# SLOW physics: oblique forward() reflection vs analytic Fresnel (magnitude + phase).
# --------------------------------------------------------------------------- #
def _build(theta):
    grid = Grid(freq_max=10e9, domain=DOMAIN, dx=DX, cpml_layers=10)
    xi = grid.position_to_index((X_IFACE, 0.06, 0.003))[0]
    xe = grid.position_to_index((X_SLAB_END, 0.06, 0.003))[0]
    xprobes = np.arange(PLATEAU[0], PLATEAU[1], DX)
    probe_idx = np.array([grid.position_to_index((float(xp), 0.06, 0.003))[0] for xp in xprobes], float)
    n = np.sqrt(EPS_R)
    t_back = (2 * (X_IFACE - PLATEAU[1]) / C0) + (2 * (X_SLAB_END - X_IFACE) / (C0 / n))
    ns = min(int(0.9 * t_back / grid.dt), 1700)
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, bandwidth=BW, polarization="ez", direction="+x",
                        angle_deg=theta, waveform="modulated_gaussian")
    for xp in xprobes:
        sim.add_probe((float(xp), 0.06, 0.003), component="ez")
    return sim, grid.shape, xi, xe, probe_idx, grid.dt, ns


def _build_lean(theta):
    """Small oblique cell for the GRADIENT test only (AD=FD is config-independent). The
    full-size reverse tape OOMs the GPU (~5.6GB); this keeps the AD peak under GPU memory."""
    dom = (0.40, 0.08, 0.006)
    xif, xend, plat = 0.10, 0.30, (0.04, 0.08)
    yc, zc = dom[1] / 2, dom[2] / 2
    grid = Grid(freq_max=10e9, domain=dom, dx=DX, cpml_layers=10)
    xi = grid.position_to_index((xif, yc, zc))[0]
    xe = grid.position_to_index((xend, yc, zc))[0]
    xprobes = np.arange(plat[0], plat[1], DX)
    n = np.sqrt(EPS_R)
    t_back = (2 * (xif - plat[1]) / C0) + (2 * (xend - xif) / (C0 / n))
    ns = min(int(0.9 * t_back / grid.dt), 1000)
    sim = Simulation(freq_max=10e9, domain=dom, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, bandwidth=BW, polarization="ez", direction="+x",
                        angle_deg=theta, waveform="modulated_gaussian")
    for xp in xprobes:
        sim.add_probe((float(xp), yc, zc), component="ez")
    probe_idx = np.array([grid.position_to_index((float(xp), yc, zc))[0] for xp in xprobes], float)
    return sim, grid.shape, xi, xe, probe_idx, grid.dt, ns


def _series(sim, eps_arr, ns, ckpt=False):
    return sim.forward(eps_override=eps_arr, n_steps=ns, checkpoint=ckpt,
                       skip_preflight=True).time_series


def _eps_slab(shape, xi, xe, eps):
    a = jnp.ones(shape, jnp.float32)
    return a if eps == 1.0 else a.at[xi:xe, :, :].set(eps)


@pytest.mark.slow
@pytest.mark.parametrize("theta,magtol", [(30.0, 0.10), (45.0, 0.08)])
def test_oblique_complex_gamma_vs_analytic(theta, magtol):
    """|Γ| and ∠Γ from oblique forward() match analytic Fresnel (phase up to the Yee offset)."""
    sim, shape, xi, xe, idx, dt, ns = _build(theta)
    inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns)
    tot = _series(sim, _eps_slab(shape, xi, xe, EPS_R), ns)
    g = complex(oblique_reflection_coefficient(
        tot, inc, f0=F0, dt=dt, probe_index=idx, interface_index=float(xi), n_gate=ns))
    ga = fresnel_r_te(theta, EPS_R)
    assert abs(g) < 1.0, f"nonphysical |Γ|={abs(g):.3f} (θ={theta})"
    assert abs(abs(g) - abs(ga)) / abs(ga) < magtol, f"|Γ|={abs(g):.3f} vs {abs(ga):.3f} (θ={theta})"
    dph = abs(((np.degrees(np.angle(g)) - 180.0 + 180) % 360) - 180)
    assert dph < 20.0, f"∠Γ={np.degrees(np.angle(g)):.1f}° vs 180° (θ={theta}, off {dph:.1f}°)"


@pytest.mark.slow
def test_oblique_complex_gamma_differentiable():
    """The complex-target RIS loss |Γ-Γ_target|² is differentiable through oblique forward()."""
    sim, shape, xi, xe, idx, dt, ns = _build_lean(30.0)  # lean config: AD tape fits GPU memory
    inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns, ckpt=True)
    g_target = 0.30 * np.exp(1j * 2.5)

    def loss(eps):
        tot = _series(sim, jnp.ones(shape, jnp.float32).at[xi:xe, :, :].set(eps), ns, ckpt=True)
        g = oblique_reflection_coefficient(tot, inc, f0=F0, dt=dt, probe_index=idx,
                                           interface_index=float(xi), n_gate=ns)
        return jnp.abs(g - g_target) ** 2

    g_ad = float(jax.grad(loss)(EPS_R))
    h = 0.05
    g_fd = (float(loss(EPS_R + h)) - float(loss(EPS_R - h))) / (2 * h)
    assert np.isfinite(g_ad), "NaN/Inf gradient (P0)"
    assert abs(g_ad - g_fd) < 0.08 * abs(g_fd) + 1e-6, f"AD={g_ad:.4e} vs FD={g_fd:.4e}"

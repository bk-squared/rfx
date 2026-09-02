"""Differentiable COMPLEX Fresnel reflection Γ(f0) — public helper + GROUND-TRUTH gate.

Exercises ``rfx.probes.fresnel_reflection_coefficient`` (the differentiable two-run
plateau + de-embed extractor) two ways:

  • ``test_reflection_helper_recovers_planted_gamma`` (FAST, no FDTD): a synthetic
    two-run signal with a PLANTED complex Γ — pins the extraction math and that the
    helper stays on the AD tape (grad matches closed form). This is the cheap contract.
  • ``test_fresnel_*`` (SLOW, FDTD): the physics ground truth — a real
    ``Simulation.forward()`` reflection off a dielectric slab, extracted with the
    helper, compared to analytic Fresnel Γ=(1−√εr)/(1+√εr) in amplitude AND phase,
    plus the end-to-end d|Γ|/dε gradient.

Replaces the retracted ``test_forward_tfsf_reflection_objective.py`` (#417), which used
SINGLE-POINT (T−I)/I extraction (the #404-defective method → |Γ|>1, 170–290% vs analytic).
Measured envelope (CPU, this config): |Γ| err 1.7/2.4/3.6% @ εr=2/4/9; phase 10.7° off 180°
(the εr-independent Yee half-cell reference-plane constant); d|Γ|/dε AD==FD 0.01%.
Validation harness: docs/research_notes/experiments/i404_oblique_20260720/diffrefl_v3_complex_and_grad.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.probes import fresnel_reflection_coefficient

F0 = 5e9
DOMAIN = (0.50, 0.12, 0.006)
DX = 0.002
X_IFACE, X_SLAB_END = 0.15, 0.40      # finite slab (0.25 m), ends well before the +x CPML
PLATEAU = (0.05, 0.13)                # probe line, TF region in front of the slab
C0 = 299792458.0


def _fresnel(eps):
    n = np.sqrt(eps)
    return (1.0 - n) / (1.0 + n)      # normal incidence, vacuum->dielectric (negative real)


# --------------------------------------------------------------------------- #
# FAST synthetic contract: the extractor math + differentiability, no FDTD.
# --------------------------------------------------------------------------- #
def _synthetic_two_run(gamma_true, dists, f0=F0, n_periods=8, spp=24):
    """Build (total, incident) real time series carrying a planted complex Γ.

    Incident phasor at probe p: I_p = e^{-j k d_p}. Reflected phasor chosen so that
    (R_p/I_p)·e^{+j2k d_p} = Γ_true exactly ⇒ R_p = I_p·Γ_true·e^{-j2k d_p}.
    Real time series over integer periods make the DFT clean; the common (½, N·dt)
    factor cancels in the ratio, so the helper must return Γ_true.
    """
    k = 2 * np.pi * f0 / C0
    dt = 1.0 / (f0 * spp)
    n = n_periods * spp
    t = np.arange(n) * dt
    ip = np.exp(-1j * k * dists)                       # (n_probes,)
    rp = ip * gamma_true * np.exp(-1j * 2 * k * dists)
    osc = np.exp(1j * 2 * np.pi * f0 * t)              # (n,)
    inc = np.real(ip[None, :] * osc[:, None])          # (n, n_probes)
    total = np.real((ip + rp)[None, :] * osc[:, None])
    return jnp.asarray(total), jnp.asarray(inc), dt


def test_reflection_helper_recovers_planted_gamma():
    """Helper recovers a planted complex Γ (amplitude+phase) and is grad-safe."""
    dists = np.array([0.010, 0.014, 0.018, 0.022])     # metres in front of the ref plane
    for g_true in (-0.4 + 0j, 0.3j, 0.25 - 0.15j):
        total, inc, dt = _synthetic_two_run(g_true, dists)
        g = complex(fresnel_reflection_coefficient(
            total, inc, f0=F0, dt=dt, probe_distances=dists))
        assert abs(g - g_true) < 2e-2, f"Γ={g:.4f} vs planted {g_true:.4f}"

    # differentiability: total = inc + θ·reflected ⇒ |Γ|(θ)=θ·|Γ_true|, grad=|Γ_true|
    total, inc, dt = _synthetic_two_run(-0.4 + 0j, dists)
    refl = total - inc

    def absg(theta):
        return jnp.abs(fresnel_reflection_coefficient(
            inc + theta * refl, inc, f0=F0, dt=dt, probe_distances=jnp.asarray(dists)))

    g_ad = float(jax.grad(absg)(1.0))
    assert np.isfinite(g_ad), "NaN/Inf gradient (P0)"
    assert abs(g_ad - 0.4) < 1e-3, f"d|Γ|/dθ={g_ad:.5f} vs closed form 0.4"


# --------------------------------------------------------------------------- #
# SLOW physics ground truth: forward() reflection off a slab vs analytic Fresnel.
# --------------------------------------------------------------------------- #
def _build():
    grid = Grid(freq_max=10e9, domain=DOMAIN, dx=DX, cpml_layers=10)
    xi = grid.position_to_index((X_IFACE, 0.06, 0.003))[0]
    xe = grid.position_to_index((X_SLAB_END, 0.06, 0.003))[0]
    xprobes = np.arange(PLATEAU[0], PLATEAU[1], DX)
    probe_idx = np.array([grid.position_to_index((float(xp), 0.06, 0.003))[0] for xp in xprobes])
    # reference plane = the interface; probe distances in metres (CONSISTENT cell-index space)
    probe_distances = (xi - probe_idx).astype(np.float64) * DX
    # time-gate: stop before the FASTEST (εr=2) back-face reflection reaches the plateau
    n_fast = np.sqrt(2.0)
    t_back = (2 * (X_IFACE - PLATEAU[1]) / C0) + (2 * (X_SLAB_END - X_IFACE) / (C0 / n_fast))
    ns = int(0.9 * t_back / grid.dt)

    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, bandwidth=0.5, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    for xp in xprobes:
        sim.add_probe((float(xp), 0.06, 0.003), component="ez")
    return sim, grid.shape, xi, xe, probe_distances, grid.dt, ns


def _series(sim, eps_arr, ns, ckpt=False):
    return sim.forward(eps_override=eps_arr, n_steps=ns, checkpoint=ckpt,
                       skip_preflight=True).time_series  # (n_steps, n_probes)


def _eps_slab(shape, xi, xe, eps):
    a = jnp.ones(shape, jnp.float32)
    return a if eps == 1.0 else a.at[xi:xe, :, :].set(eps)


@pytest.fixture(scope="module")
def fresnel_run():
    sim, shape, xi, xe, dists, dt, ns = _build()
    inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns)          # vacuum incident
    out = {"sim": sim, "shape": shape, "xi": xi, "xe": xe, "dists": dists,
           "dt": dt, "ns": ns, "inc": inc, "gamma": {}}
    for eps in (2.0, 4.0, 9.0):
        tot = _series(sim, _eps_slab(shape, xi, xe, eps), ns)
        out["gamma"][eps] = complex(fresnel_reflection_coefficient(
            tot, inc, f0=F0, dt=dt, probe_distances=dists, n_gate=ns))
    return out


@pytest.mark.slow
@pytest.mark.parametrize("eps", [2.0, 4.0, 9.0])
def test_fresnel_magnitude_vs_analytic(fresnel_run, eps):
    """|Γ| matches analytic Fresnel to the Yee discretization envelope AND is passive (<1)."""
    g = fresnel_run["gamma"][eps]
    ga = _fresnel(eps)
    assert abs(g) < 1.0, f"nonphysical |Γ|={abs(g):.3f}≥1 (the retracted single-point failure mode)"
    assert abs(abs(g) - abs(ga)) < 0.04, f"|Γ|={abs(g):.3f} vs analytic {abs(ga):.3f} (εr={eps})"


@pytest.mark.slow
def test_fresnel_phase_vs_analytic(fresnel_run):
    """De-embedded phase ~180° (up to the constant Yee half-cell reference-plane offset)."""
    for eps in (2.0, 4.0, 9.0):
        g = fresnel_run["gamma"][eps]
        dph = abs(((np.degrees(np.angle(g)) - 180.0 + 180) % 360) - 180)
        assert dph < 20.0, f"∠Γ={np.degrees(np.angle(g)):.1f}° vs 180° (εr={eps}, off {dph:.1f}°)"


@pytest.mark.slow
def test_fresnel_phase_offset_is_reference_plane_constant(fresnel_run):
    """The ~11° phase residual is a reference-plane CONSTANT (εr-independent), not a physics error."""
    phases = [np.degrees(np.angle(fresnel_run["gamma"][e])) for e in (2.0, 4.0, 9.0)]
    assert (max(phases) - min(phases)) < 3.0, f"phase offset varies with εr {phases} — not a constant"


@pytest.mark.slow
def test_fresnel_gamma_differentiable(fresnel_run):
    """d|Γ|/dε flows through the checkpointed forward and matches finite difference."""
    sim, shape, xi, xe = fresnel_run["sim"], fresnel_run["shape"], fresnel_run["xi"], fresnel_run["xe"]
    dists, dt, ns = fresnel_run["dists"], fresnel_run["dt"], fresnel_run["ns"]
    inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns, ckpt=True)

    def abs_gamma(eps):
        tot = _series(sim, jnp.ones(shape, jnp.float32).at[xi:xe, :, :].set(eps), ns, ckpt=True)
        return jnp.abs(fresnel_reflection_coefficient(
            tot, inc, f0=F0, dt=dt, probe_distances=dists, n_gate=ns))

    g_ad = float(jax.grad(abs_gamma)(4.0))
    h = 0.05
    g_fd = (float(abs_gamma(4.0 + h)) - float(abs_gamma(4.0 - h))) / (2 * h)
    assert np.isfinite(g_ad), "NaN/Inf gradient (P0)"
    assert abs(g_ad - g_fd) < 0.05 * abs(g_fd) + 1e-6, f"AD={g_ad:.4e} vs FD={g_fd:.4e}"


@pytest.mark.slow
def test_fresnel_complex_target_gradient(fresnel_run):
    """ch07 conjugation: the RIS loss |Γ−Γ_target|² (phase-sensitive) differentiates correctly.

    Unlike |Γ|, a complex-target loss depends on BOTH Re(Γ) and Im(Γ), so its gradient
    exercises the phase path — where a mishandled e^{±jωt} DFT-sign / Wirtinger conjugation
    would silently corrupt the gradient (the #404 trap). FD is the ground truth.
    """
    sim, shape, xi, xe = fresnel_run["sim"], fresnel_run["shape"], fresnel_run["xi"], fresnel_run["xe"]
    dists, dt, ns = fresnel_run["dists"], fresnel_run["dt"], fresnel_run["ns"]
    inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns, ckpt=True)
    g_target = 0.30 * np.exp(1j * 2.5)  # arbitrary complex RIS target (amplitude + phase)

    def loss(eps):
        tot = _series(sim, jnp.ones(shape, jnp.float32).at[xi:xe, :, :].set(eps), ns, ckpt=True)
        g = fresnel_reflection_coefficient(tot, inc, f0=F0, dt=dt, probe_distances=dists, n_gate=ns)
        return jnp.abs(g - g_target) ** 2

    g_ad = float(jax.grad(loss)(4.0))
    h = 0.05
    g_fd = (float(loss(4.0 + h)) - float(loss(4.0 - h))) / (2 * h)
    assert np.isfinite(g_ad), "NaN/Inf gradient (P0)"
    assert abs(g_ad - g_fd) < 0.06 * abs(g_fd) + 1e-6, f"complex-target AD={g_ad:.4e} vs FD={g_fd:.4e}"

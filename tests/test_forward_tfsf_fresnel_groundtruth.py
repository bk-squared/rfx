"""Differentiable COMPLEX Fresnel reflection Γ(f0) — GROUND-TRUTH gated against analytic Fresnel.

Replaces the retracted `test_forward_tfsf_reflection_objective.py` (#417). That test extracted Γ
with a SINGLE-POINT (T−I)/I ratio — the exact method #404 already proved defective (a standing
wave makes the per-cell ratio oscillate; it gave |Γ|>1, 170–290% vs analytic). Its passivity/
phase assertions passed only by coincidence at a thin slab: a gate binding an artifact.

The CORRECT extraction (the #404/#414-validated recipe, here ported to the differentiable
`forward()` and validated vs the analytic oracle):
  • FINITE dielectric slab ending BEFORE the +x CPML — a CPML filled with dielectric is NOT
    impedance-matched and reflects (that inflated the retracted |Γ|>1).
  • TIME-GATE the DFT window to the FRONT-face reflection only (stop before the back-face
    reflection returns) ⇒ a simple half-space Fresnel Γ, no Fabry–Pérot.
  • Broadband source (bandwidth=0.5) develops fast, so the window can be short — this resolves
    the narrowband(clean-f0) ↔ time-gating(front-face) tension that sank the earlier attempts.
  • SPATIAL PLATEAU: a LINE of probes in the scattered-field-free TF region; two-run reference
    subtraction (slab − vacuum) gives the pure reflected traveling wave; average over the line.
  • PHASE: de-embed each probe to the interface plane (Γ = (refl/inc)·e^{+j2k d}). Analytic
    normal-incidence Γ=(1−√εr)/(1+√εr) is NEGATIVE real ⇒ 180°. A residual ~11° is the Yee
    half-cell reference-plane offset (constant across εr — a reference-plane constant, not a
    physics error; calibratable).

GROUND TRUTH: analytic Fresnel Γ=(1−√εr)/(1+√εr) — amplitude AND phase. Measured envelope
(CPU, this config): |Γ| err 1.7/2.4/3.6% @ εr=2/4/9; phase 10.7° off 180°; d|Γ|/dε AD==FD 0.01%.
Validation harness: docs/research_notes/experiments/i404_oblique_20260720/diffrefl_v3_complex_and_grad.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid

F0 = 5e9
DOMAIN = (0.50, 0.12, 0.006)
DX = 0.002
X_IFACE, X_SLAB_END = 0.15, 0.40      # finite slab (0.25 m), ends well before the +x CPML
PLATEAU = (0.05, 0.13)                # probe line, TF region in front of the slab
C0 = 299792458.0


def _fresnel(eps):
    n = np.sqrt(eps)
    return (1.0 - n) / (1.0 + n)      # normal incidence, vacuum->dielectric (negative real)


def _build():
    grid = Grid(freq_max=10e9, domain=DOMAIN, dx=DX, cpml_layers=10)
    xi = grid.position_to_index((X_IFACE, 0.06, 0.003))[0]
    xe = grid.position_to_index((X_SLAB_END, 0.06, 0.003))[0]
    xprobes = np.arange(PLATEAU[0], PLATEAU[1], DX)
    probe_idx = np.array([grid.position_to_index((float(xp), 0.06, 0.003))[0] for xp in xprobes])
    k0 = 2 * np.pi * F0 / C0
    # de-embed probe -> interface in CONSISTENT cell-index space (CPML pad offset cancels)
    deembed = np.exp(+1j * 2 * k0 * (xi - probe_idx).astype(np.float64) * DX)
    # time-gate: stop before the FASTEST (εr=2) back-face reflection reaches the plateau
    n_fast = np.sqrt(2.0)
    t_back = (2 * (X_IFACE - PLATEAU[1]) / C0) + (2 * (X_SLAB_END - X_IFACE) / (C0 / n_fast))
    ns = int(0.9 * t_back / grid.dt)
    kern = jnp.exp(-1j * 2 * jnp.pi * F0 * jnp.arange(ns) * grid.dt) * grid.dt

    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, bandwidth=0.5, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    for xp in xprobes:
        sim.add_probe((float(xp), 0.06, 0.003), component="ez")
    return sim, grid.shape, xi, xe, deembed, kern, ns


def _eps_slab(shape, xi, xe, eps):
    a = jnp.ones(shape, jnp.float32)
    return a if eps == 1.0 else a.at[xi:xe, :, :].set(eps)


def _phasors(sim, eps_arr, kern, ns, ckpt=False):
    fr = sim.forward(eps_override=eps_arr, n_steps=ns, checkpoint=ckpt, skip_preflight=True)
    return jnp.sum(fr.time_series.astype(jnp.complex64) * kern[:, None], axis=0)  # (n_probes,)


def _gamma(sim, shape, xi, xe, deembed, kern, ns, inc, eps):
    T = np.asarray(_phasors(sim, _eps_slab(shape, xi, xe, eps), kern, ns))
    gamma_p = (T - inc) / inc * deembed              # per-probe, de-embedded to the interface
    return gamma_p


@pytest.fixture(scope="module")
def fresnel_run():
    sim, shape, xi, xe, deembed, kern, ns = _build()
    inc = np.asarray(_phasors(sim, _eps_slab(shape, xi, xe, 1.0), kern, ns))  # vacuum incident
    out = {"sim": sim, "shape": shape, "xi": xi, "xe": xe, "deembed": deembed,
           "kern": kern, "ns": ns, "inc": inc, "gamma": {}}
    for eps in (2.0, 4.0, 9.0):
        out["gamma"][eps] = _gamma(sim, shape, xi, xe, deembed, kern, ns, inc, eps)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("eps", [2.0, 4.0, 9.0])
def test_fresnel_magnitude_vs_analytic(fresnel_run, eps):
    """|Γ| matches analytic Fresnel to the Yee discretization envelope AND is passive (<1)."""
    g = complex(np.mean(fresnel_run["gamma"][eps]))
    ga = _fresnel(eps)
    assert abs(g) < 1.0, f"nonphysical |Γ|={abs(g):.3f}≥1 (the retracted single-point failure mode)"
    assert abs(abs(g) - abs(ga)) < 0.04, f"|Γ|={abs(g):.3f} vs analytic {abs(ga):.3f} (εr={eps})"


@pytest.mark.slow
def test_fresnel_phase_vs_analytic(fresnel_run):
    """De-embedded phase ~180° (up to the constant Yee half-cell reference-plane offset)."""
    for eps in (2.0, 4.0, 9.0):
        gamma_p = fresnel_run["gamma"][eps]
        g = complex(np.mean(gamma_p))
        dph = abs(((np.degrees(np.angle(g)) - 180.0 + 180) % 360) - 180)
        spread = float(np.std(np.degrees(np.angle(gamma_p))))
        assert dph < 20.0, f"∠Γ={np.degrees(np.angle(g)):.1f}° vs 180° (εr={eps}, off {dph:.1f}°)"
        # de-embed consistency: a right reference plane => probes agree
        assert spread < 12.0, f"per-probe phase spread {spread:.1f}° too large (εr={eps})"


@pytest.mark.slow
def test_fresnel_phase_offset_is_reference_plane_constant(fresnel_run):
    """The ~11° phase residual is a reference-plane CONSTANT (εr-independent), not a physics error."""
    phases = [np.degrees(np.angle(complex(np.mean(fresnel_run["gamma"][e])))) for e in (2.0, 4.0, 9.0)]
    assert (max(phases) - min(phases)) < 3.0, f"phase offset varies with εr {phases} — not a constant"


@pytest.mark.slow
def test_fresnel_gamma_differentiable(fresnel_run):
    """d|Γ|/dε flows through the checkpointed forward and matches finite difference."""
    sim, shape, xi, xe = fresnel_run["sim"], fresnel_run["shape"], fresnel_run["xi"], fresnel_run["xe"]
    deembed, kern, ns = fresnel_run["deembed"], fresnel_run["kern"], fresnel_run["ns"]
    inc = _phasors(sim, _eps_slab(shape, xi, xe, 1.0), kern, ns, ckpt=True)

    def abs_gamma(eps):
        T = _phasors(sim, jnp.ones(shape, jnp.float32).at[xi:xe, :, :].set(eps), kern, ns, ckpt=True)
        return jnp.abs(jnp.mean((T - inc) / inc * jnp.asarray(deembed)))

    g_ad = float(jax.grad(abs_gamma)(4.0))
    h = 0.05
    g_fd = (float(abs_gamma(4.0 + h)) - float(abs_gamma(4.0 - h))) / (2 * h)
    assert np.isfinite(g_ad), "NaN/Inf gradient (P0)"
    assert abs(g_ad - g_fd) < 0.05 * abs(g_fd) + 1e-6, f"AD={g_ad:.4e} vs FD={g_fd:.4e}"

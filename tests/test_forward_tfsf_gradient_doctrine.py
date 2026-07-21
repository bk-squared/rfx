"""Gradient-doctrine contract for differentiable TFSF plane-wave forward.

Hardens the merged differentiable-TFSF forward (PR #415) per the anyscale
differentiability doctrine (rfx-adv/docs/plans/anyscale-book/07_differentiability.md
§5 composition contract + §6 verification):

  * FD STEP-SWEEP, not a single h — assert the AD gradient matches the
    finite-difference PLATEAU across h in {2e-2 .. 3e-4} (ch07 §6: "never a
    single h; pick the plateau").
  * INVARIANCE WITNESS (ch07 §6): a y-translation-symmetric setup (normal
    incidence, y-periodic, y-uniform scatterer) must give a y-invariant field,
    so the gradient of the difference between two y-shifted probes is ~0.
  * End-to-end at realistic parameter values; a NaN/Inf in the gradient is a P0.

TFSF-driven gradients are the T1 (full-wave rfx) differentiable primitive the
PEFM lab uses for RIS / FSS / RCS inverse design.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation

_DOMAIN = (0.24, 0.08, 0.006)
_DX = 0.002


def _sim(angle=0.0, extra_probe=None):
    sim = Simulation(freq_max=10e9, domain=_DOMAIN, dx=_DX,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add_tfsf_source(f0=5e9, bandwidth=0.5, polarization="ez", direction="+x",
                        angle_deg=angle, waveform="differentiated_gaussian")
    sim.add_probe((0.0, 0.0, 0.0), component="ez")
    if extra_probe is not None:
        sim.add_probe(extra_probe, component="ez")
    return sim


@pytest.mark.slow
def test_forward_tfsf_gradient_fd_step_sweep():
    """AD gradient of a TFSF-driven forward objective matches the FD PLATEAU
    across a step sweep (anyscale ch07 §6) — not just one lucky h."""
    sim = _sim(0.0)
    shp = sim.run(n_steps=1).state.ez.shape
    xi = shp[0] // 2

    def obj_j(eps_val):
        eps = jnp.ones(shp, jnp.float32).at[xi:xi + 12, :, :].set(eps_val)
        fr = sim.forward(eps_override=eps, n_steps=500, checkpoint=True,
                         skip_preflight=True)
        return jnp.sum(jnp.abs(fr.time_series) ** 2).real

    e0 = 4.0
    g_ad = float(jax.grad(obj_j)(e0))
    assert np.isfinite(g_ad), f"gradient not finite (P0): {g_ad}"

    fds = {}
    for h in (2e-2, 1e-2, 3e-3, 1e-3, 3e-4):
        fds[h] = (float(obj_j(e0 + h)) - float(obj_j(e0 - h))) / (2 * h)
    hs = sorted(fds)  # ascending h
    # plateau = the adjacent-h pair with the smallest successive change
    k = min(range(len(hs) - 1), key=lambda i: abs(fds[hs[i]] - fds[hs[i + 1]]))
    fd_plateau = 0.5 * (fds[hs[k]] + fds[hs[k + 1]])
    rel = abs(g_ad - fd_plateau) / max(abs(fd_plateau), 1e-30)
    assert rel < 0.05, (
        f"AD {g_ad:.5f} vs FD plateau {fd_plateau:.5f} (h~{hs[k]:.0e}) "
        f"rel {rel*100:.1f}%; sweep={ {f'{h:.0e}': round(v, 4) for h, v in fds.items()} }"
    )


@pytest.mark.slow
def test_forward_tfsf_gradient_y_translation_invariance():
    """Invariance witness (anyscale ch07 §6): normal incidence + y-periodic +
    y-uniform scatterer is y-translation-symmetric, so two y-shifted probes read
    the same field and the GRADIENT of their power difference is ~0."""
    sim = _sim(0.0, extra_probe=(0.0, 0.02, 0.0))  # 2nd probe shifted in y
    shp = sim.run(n_steps=1).state.ez.shape
    xi = shp[0] // 2

    def parts(eps_val):
        eps = jnp.ones(shp, jnp.float32).at[xi:xi + 12, :, :].set(eps_val)
        fr = sim.forward(eps_override=eps, n_steps=500, checkpoint=True,
                         skip_preflight=True)
        p0 = jnp.sum(jnp.abs(fr.time_series[:, 0]) ** 2).real
        p1 = jnp.sum(jnp.abs(fr.time_series[:, 1]) ** 2).real
        return p0, p1

    p0, p1 = parts(4.0)
    scale = float(p0 + p1) / 2
    assert scale > 1e-3, "probes recorded no field"
    # the two y-shifted probes must agree (y-uniform field)
    assert abs(float(p0) - float(p1)) / scale < 1e-3, "y-invariance broken in the forward"
    # ...and the gradient of the difference must be ~0 (respects y-symmetry)
    g_diff = float(jax.grad(lambda e: (parts(e)[0] - parts(e)[1]))(4.0))
    g_sum = float(jax.grad(lambda e: (parts(e)[0] + parts(e)[1]))(4.0))
    assert np.isfinite(g_diff)
    assert abs(g_diff) / max(abs(g_sum), 1e-30) < 1e-2, (
        f"gradient breaks y-translation invariance: d(diff)={g_diff:.3e} "
        f"vs d(sum)={g_sum:.3e}"
    )

"""Phase A pin: jax.checkpoint plumbing for the non-uniform scan body.

Issue #31 — memory-efficient inverse design. The NU forward path now
threads ``checkpoint`` into ``run_nonuniform``'s ``jax.lax.scan``
(``rfx/nonuniform.py``). Before this change the kwarg was silently
dropped (``del checkpoint`` in ``_forward_nonuniform_from_materials``)
so reverse-mode AD memory scaled linearly with ``n_steps``.

These tests pin the wiring:

1. ``forward(checkpoint=True)`` and ``forward(checkpoint=False)`` return
   bit-identical time-series.
2. ``jax.grad`` through both modes agrees on a simple scalar loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation


def _build_sim():
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3,
        dz_profile=dz,
        cpml_layers=4,
    )
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    return sim


def test_nu_forward_ckpt_matches_plain():
    sim = _build_sim()
    n_steps = 80
    ts_ckpt = np.asarray(sim.forward(n_steps=n_steps, checkpoint=True).time_series)
    ts_plain = np.asarray(sim.forward(n_steps=n_steps, checkpoint=False).time_series)
    assert ts_ckpt.shape == ts_plain.shape
    np.testing.assert_allclose(ts_ckpt, ts_plain, rtol=1e-6, atol=1e-12)


def test_nu_grad_ckpt_matches_plain():
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2
    n_steps = 60

    def loss(alpha, *, checkpoint):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        fr = sim.forward(eps_override=eps, n_steps=n_steps, checkpoint=checkpoint)
        return jnp.sum(fr.time_series ** 2)

    alpha0 = jnp.float32(2.0)
    g_ckpt = float(jax.grad(lambda a: loss(a, checkpoint=True))(alpha0))
    g_plain = float(jax.grad(lambda a: loss(a, checkpoint=False))(alpha0))
    assert np.isfinite(g_ckpt) and np.isfinite(g_plain)
    rel = abs(g_ckpt - g_plain) / max(abs(g_plain), 1e-30)
    assert rel < 1e-4, f"ckpt vs plain grad disagree: {g_ckpt} vs {g_plain} (rel={rel})"

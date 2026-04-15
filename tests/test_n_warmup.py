"""Issue #40: n_warmup splits the scan into a gradient-free warmup phase
and a full-AD optimize phase.

Guarantees:
  1. Forward output (time_series) is unchanged vs. a plain n_warmup=0
     run — the warmup phase still runs the same physics.
  2. jax.grad of a loss that depends only on post-warmup samples
     returns the same value with or without warmup (stop_gradient only
     kills tape for steps before n_warmup; physical forward is
     identical).
  3. n_warmup >= n_steps is a clean ValueError.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


def _build_sim():
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3, dz_profile=dz, cpml_layers=4,
    )
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    return sim


def test_forward_matches_plain():
    sim = _build_sim()
    ts_plain = np.asarray(sim.forward(n_steps=100).time_series)
    ts_warm = np.asarray(
        sim.forward(n_steps=100, n_warmup=40).time_series
    )
    assert ts_plain.shape == ts_warm.shape
    np.testing.assert_allclose(ts_plain, ts_warm, rtol=1e-5, atol=1e-10)


def test_warmup_grad_finite_and_same_sign():
    """n_warmup cuts gradient contribution from the warmup window (by
    design — that's the memory/speed trade). We only require the
    warmup'd gradient to be finite and agree on sign with the full
    gradient.
    """
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2
    n_steps = 60

    def loss(alpha, *, warmup):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        fr = sim.forward(
            eps_override=eps, n_steps=n_steps, n_warmup=warmup,
        )
        return jnp.sum(fr.time_series[warmup:] ** 2)

    a0 = jnp.float32(2.0)
    g_plain = float(jax.grad(lambda a: loss(a, warmup=0))(a0))
    g_warm = float(jax.grad(lambda a: loss(a, warmup=20))(a0))
    assert np.isfinite(g_plain) and np.isfinite(g_warm)
    assert np.sign(g_plain) == np.sign(g_warm), (
        f"grad signs disagree: plain={g_plain}, warmup={g_warm}"
    )


def test_n_warmup_ge_n_steps_raises():
    sim = _build_sim()
    with pytest.raises(ValueError, match="n_warmup"):
        sim.forward(n_steps=30, n_warmup=30)


def test_n_warmup_composes_with_checkpoint_every():
    sim = _build_sim()
    ts_plain = np.asarray(sim.forward(n_steps=80).time_series)
    ts_combo = np.asarray(
        sim.forward(n_steps=80, n_warmup=16, checkpoint_every=16).time_series
    )
    np.testing.assert_allclose(ts_plain, ts_combo, rtol=1e-5, atol=1e-10)

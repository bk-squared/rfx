"""Phase B (segmented scan) pin: checkpoint_every must produce identical
forward and gradient values to the plain whole-scan path.

Issue #31 follow-up: jax.checkpoint(step_fn) inside lax.scan only remats
*within* a step; the AD tape still stores the full carry stack at every
step (n_steps × carry_size). For FDTD where carry dominates, this means
Phase A alone OOMs on realistic 3D geometries.

Wrapping a *segment*-level scan-of-scan in jax.checkpoint forces XLA to
remat the inner scan during backward, so the tape only stores carry at
segment boundaries. With checkpoint_every ≈ sqrt(n_steps), AD memory
becomes O(sqrt(n_steps) × carry_size).

These tests pin: (1) bit-equivalent forward, (2) gradient agreement,
(3) divisible vs non-divisible n_steps both work.
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
        freq_max=10e9,
        domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3,
        dz_profile=dz,
        cpml_layers=4,
    )
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    return sim


@pytest.mark.parametrize("n_steps,chunk", [(60, 20), (60, 15), (50, 12)])
def test_segmented_forward_matches_plain(n_steps, chunk):
    sim = _build_sim()
    ts_plain = np.asarray(sim.forward(n_steps=n_steps).time_series)
    ts_seg = np.asarray(
        sim.forward(n_steps=n_steps, checkpoint_every=chunk).time_series
    )
    assert ts_plain.shape == ts_seg.shape
    np.testing.assert_allclose(ts_plain, ts_seg, rtol=1e-5, atol=1e-10)


def test_segmented_grad_matches_plain():
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2
    n_steps = 60

    def loss(alpha, *, chunk):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        fr = sim.forward(eps_override=eps, n_steps=n_steps,
                         checkpoint_every=chunk)
        return jnp.sum(fr.time_series ** 2)

    a0 = jnp.float32(2.0)
    g_plain = float(jax.grad(lambda a: loss(a, chunk=None))(a0))
    g_seg = float(jax.grad(lambda a: loss(a, chunk=20))(a0))
    rel = abs(g_plain - g_seg) / max(abs(g_plain), 1e-30)
    assert rel < 1e-4, f"segmented grad disagrees: {g_plain} vs {g_seg} (rel={rel})"


def test_segmented_emit_false_works():
    """Combined Phase A+C+segmented path."""
    sim = _build_sim()
    fr = sim.forward(n_steps=60, checkpoint_every=20, emit_time_series=False)
    assert np.asarray(fr.time_series).size == 0


def test_uniform_rejects_checkpoint_every():
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                     cpml_layers=4)
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    with pytest.raises(NotImplementedError, match="non-uniform"):
        sim.forward(n_steps=20, checkpoint_every=5)

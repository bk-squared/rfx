"""forward()/optimize() must be wrappable in an OUTER jax.jit.

Regression for the blind-docs finding: ``_assemble_materials`` decided whether to
return the PEC mask via ``bool(jnp.any(pec_mask))`` -- a host-side boolean
conversion on a geometry-derived device array. That is fine eagerly (and under a
bare ``jax.grad``, where materials stay concrete), but when the whole
``forward()`` is wrapped in an *outer* ``jax.jit`` the geometry-derived
``pec_mask`` becomes a tracer and the ``bool(...)`` raised
``TracerBoolConversionError`` deep in material assembly -- so a user JITing their
optimization step crashed with an opaque error.

Fix: the eager path keeps the exact ``jnp.any`` test (bit-identical results,
including the corner where a PEC shape's mask is empty); only under trace, where a
host bool is impossible, does it fall back to a static Python ``has_pec``
predicate. These tests lock the capability (outer-jit works) and the eager
invariant (a PEC shape entirely outside the grid stays a no-op).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import GaussianPulse, Simulation
from rfx.geometry import Box


def _pec_sim():
    s = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    s.add_source((0.02, 0.01, 0.01), "ez",
                 waveform=GaussianPulse(f0=3e9, bandwidth=3e9))
    s.add_probe((0.012, 0.008, 0.008), "ez")
    return s


def _reflected_energy_loss(sim):
    def loss(eps):
        r = sim.forward(eps_override=eps, n_steps=60, checkpoint=True,
                        skip_preflight=True)
        ts = r.time_series[:, 0]
        n = ts.shape[0]
        return jnp.sum(ts[n // 2:] ** 2) / (jnp.sum(ts[:n // 2] ** 2) + 1e-30)
    return loss


def test_forward_wrappable_in_outer_jit():
    """jax.jit(loss) must run (previously TracerBoolConversionError) and match eager."""
    sim = _pec_sim()
    shape = sim.run(n_steps=1, skip_preflight=True).grid.shape
    eps = jnp.ones(shape, dtype=jnp.float32) * 1.5
    loss = _reflected_energy_loss(sim)

    eager = float(loss(eps))
    jitted = float(jax.jit(loss)(eps))  # must not raise
    assert np.isfinite(jitted)
    assert abs(eager - jitted) <= 1e-6 * (abs(eager) + 1e-12), (eager, jitted)


def test_grad_wrappable_in_outer_jit():
    """jax.jit(jax.grad(loss)) must run and match the un-jitted gradient."""
    sim = _pec_sim()
    shape = sim.run(n_steps=1, skip_preflight=True).grid.shape
    eps = jnp.ones(shape, dtype=jnp.float32) * 1.5
    loss = _reflected_energy_loss(sim)

    g_plain = jax.grad(loss)(eps)
    g_jit = jax.jit(jax.grad(loss))(eps)  # must not raise
    assert np.all(np.isfinite(np.asarray(g_jit)))
    # float32 gradients through 60 FDTD steps: jit fuses ops differently, so
    # allow the usual XLA-reordering slack (agreement is to ~6 sig figs).
    assert np.allclose(np.asarray(g_plain), np.asarray(g_jit),
                       rtol=3e-4, atol=1e-6)


def test_real_interior_pec_under_outer_jit_matches_eager():
    """The point of the fix on the hot path: a sim with a REAL interior PEC
    obstacle (so ``has_pec_cells`` is True and the trace fallback returns the
    actual mask) must still be outer-jit-wrappable and match eager. `_pec_sim`
    has only a PEC *boundary* (separate grid path), which leaves ``pec_mask``
    empty — this exercises the non-trivial fallback branch."""
    s = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    s.add(Box((0.018, 0.006, 0.006), (0.024, 0.014, 0.014)), material="pec")
    s.add_source((0.008, 0.01, 0.01), "ez",
                 waveform=GaussianPulse(f0=3e9, bandwidth=3e9))
    s.add_probe((0.012, 0.008, 0.008), "ez")
    shape = s.run(n_steps=1, skip_preflight=True).grid.shape
    # confirm this sim really populates pec_mask (the branch under test)
    assert s._assemble_materials(s._build_grid())[3] is not None

    eps = jnp.ones(shape, dtype=jnp.float32) * 1.5
    loss = _reflected_energy_loss(s)
    eager = float(loss(eps))
    jitted = float(jax.jit(loss)(eps))
    assert abs(eager - jitted) <= 1e-5 * (abs(eager) + 1e-12), (eager, jitted)
    g_plain = jax.grad(loss)(eps)
    g_jit = jax.jit(jax.grad(loss))(eps)
    assert np.allclose(np.asarray(g_plain), np.asarray(g_jit), rtol=3e-4, atol=1e-6)


def test_assemble_materials_eager_pec_mask_none_when_empty():
    """Eager ``has_pec`` decision is byte-identical to the pre-fix ``jnp.any``:
    a PEC shape whose mask is empty (entirely outside the grid) still returns
    ``pec_mask=None``, while an interior PEC obstacle returns a real mask. This
    locks the corner the trace-only static fallback would otherwise
    over-approximate."""
    s_empty = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    s_empty.add(Box((10.0, 10.0, 10.0), (11.0, 11.0, 11.0)), material="pec")
    grid = s_empty._build_grid()
    pec_mask = s_empty._assemble_materials(grid)[3]
    assert pec_mask is None, "empty-mask PEC must stay None on the eager path"

    s_real = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    s_real.add(Box((0.018, 0.006, 0.006), (0.024, 0.014, 0.014)), material="pec")
    grid_r = s_real._build_grid()
    pec_mask_r = s_real._assemble_materials(grid_r)[3]
    assert pec_mask_r is not None and bool(pec_mask_r.any()), \
        "interior PEC obstacle must return a non-empty mask"

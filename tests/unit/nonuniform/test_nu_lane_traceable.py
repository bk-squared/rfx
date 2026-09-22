"""A non-uniform run can be traced as a whole (``jax.make_jaxpr`` / ``jax.jit``).

Eager ``jax.grad`` through ``forward()`` on the graded lane always worked, but
wrapping the whole call in ``jax.jit``, ``jax.make_jaxpr`` or the saved-residual
inspector raised ``ConcretizationTypeError`` inside ``init_cpml``:
``_get_axis_cell_sizes`` did ``float(dz_arr[0])`` on a CONCRETE jnp array, and
indexing a jnp array under a trace is itself a jnp op that yields a tracer.
The fix reads the concrete array on the host first. Pinned here as the
invariant — the lane traces, and the traced gradient equals the eager one —
not as a byte count.

Mutation that must turn this red: put ``float(dz_arr[0])`` back.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from rfx import GaussianPulse, Simulation


def _sim():
    dz = np.array([1e-3] * 6 + [0.5e-3] * 6 + [1e-3] * 6)
    sim = Simulation(freq_max=16e9, domain=(16e-3, 14e-3, float(dz.sum())), dx=1e-3,
                     dz_profile=dz, boundary="cpml", cpml_layers=4)
    sim.add_source((4e-3, 7e-3, 6e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
    sim.add_probe((12e-3, 7e-3, 6e-3), "ez")
    return sim


def test_graded_forward_traces_as_a_whole_and_the_jit_gradient_is_the_eager_one():
    sim = _sim()
    shape = tuple(sim.forward(n_steps=2, skip_preflight=True).grid.shape)
    nx, ny, nz = shape
    sl = (slice(nx // 2 - 1, nx // 2 + 2), slice(ny // 2 - 1, ny // 2 + 2), slice(nz // 2 - 1, nz // 2 + 2))

    def loss(e):
        full = jnp.ones(shape, jnp.float32).at[sl].set(e)
        r = sim.forward(eps_override=full, n_steps=12, skip_preflight=True)
        return jnp.sum(r.time_series ** 2)

    x = jnp.full((3, 3, 3), 3.0, jnp.float32)
    jax.make_jaxpr(loss)(x)                       # raised ConcretizationTypeError before the fix
    g_eager = np.asarray(jax.grad(loss)(x))
    g_jit = np.asarray(jax.jit(jax.grad(loss))(x))
    assert np.isfinite(g_eager).all() and np.linalg.norm(g_eager) > 0
    np.testing.assert_allclose(g_jit, g_eager, rtol=1e-6, atol=0)

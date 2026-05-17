"""Stage 1.5b regression — distributed_v2 final-state gather traceability.

`distributed_v2.run_distributed` ends by gathering the x-sharded device
slabs back into whole-domain arrays via the closure `_unstack_and_gather`.
Pre-fix that closure did ``np.array(sharded_arr)`` — a host pull that
raised ``TracerArrayConversionError`` whenever the gather ran inside a
JAX trace (e.g. a caller differentiating an objective off the gathered
``final_state``). The fix mirrors the already-correct
`distributed_nu.py::_unstack_and_gather`: pure ``jnp.reshape`` +
``gather_array_x`` (itself JAX-friendly), keeping the gather in-trace.

`_unstack_and_gather` is a runner-local closure and cannot be imported,
so these tests pin the exact pattern the fix installs: the post-fix
``jnp.reshape`` + ``gather_array_x`` body is differentiable, and the
pre-fix ``np.array`` host-pull is provably not. The runner's forward
output is covered (unchanged) by the distributed_v2 forward suites
(test_boundary_pmc_distributed / test_boundary_pmc_composition).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.runners.distributed import gather_array_x

_N_DEVICES, _NX_LOCAL, _NY, _NZ, _GHOST = 2, 6, 3, 3, 1


def _post_fix_gather(sharded_arr):
    """The post-fix `_unstack_and_gather` body (pure JAX)."""
    stacked = jnp.reshape(
        sharded_arr,
        (_N_DEVICES, _NX_LOCAL) + tuple(sharded_arr.shape[1:]),
    )
    return gather_array_x(stacked, _GHOST)


def test_distributed_v2_gather_pattern_is_differentiable():
    """The post-fix gather body flows a gradient end-to-end."""
    x = jnp.ones((_N_DEVICES * _NX_LOCAL, _NY, _NZ))

    def loss(sharded_arr):
        return jnp.sum(_post_fix_gather(sharded_arr) ** 2)

    grad = jax.grad(loss)(x)
    assert grad.shape == x.shape
    assert np.all(np.isfinite(np.asarray(grad)))
    # A non-trivial gradient — the gather actually couples input to output.
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_distributed_v2_gather_pattern_is_jit_traceable():
    """The post-fix gather body traces cleanly under jax.jit."""
    x = jnp.arange(_N_DEVICES * _NX_LOCAL * _NY * _NZ, dtype=jnp.float32)
    x = x.reshape(_N_DEVICES * _NX_LOCAL, _NY, _NZ)
    gathered = jax.jit(_post_fix_gather)(x)
    assert np.all(np.isfinite(np.asarray(gathered)))


def test_old_np_array_host_pull_breaks_tracing():
    """Contrast: the removed ``np.array(sharded_arr)`` host pull raises
    under a JAX trace — this is the exact break Stage 1.5b removed."""
    x = jnp.ones((_N_DEVICES * _NX_LOCAL, _NY, _NZ))

    def bad_gather(sharded_arr):
        arr = np.array(sharded_arr)  # the removed pre-fix line
        return jnp.sum(jnp.asarray(arr) ** 2)

    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.grad(bad_gather)(x)

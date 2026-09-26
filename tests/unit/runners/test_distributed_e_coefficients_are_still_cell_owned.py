"""The distributed lane was NOT converted to the edge average (#1210).

This test documents a refusal, not a property we want. A conductor or a
dielectric interface that lies where a distributed run's E update reads it is
realized one cell differently there than on the single-device lane: the
distributed slab bodies (``rfx/runners/distributed.py::_update_e_local`` and
``rfx/runners/_distributed_common.py::_update_e_local_nu``) keep their own
inline copy of the lossy coefficient and still take it from the cell that owns
the edge.

Why it was left: the slab's material arrays DO carry the one-cell x halo
(``_split_materials`` -> ``split_array_x`` with ``ghost=1``), so the seam cells
have the neighbour material they need. What they do not have is the right
material in the halo at the PHYSICAL x faces, where ``split_array_x`` pads with
vacuum (``eps_r = 1``, ``sigma = 0``) rather than replicating the boundary
cell. Averaging locally would therefore agree with the single-device lane
everywhere except the two x faces, and fixing that means changing what the
ghost material IS -- which also feeds the ghost cells' own H update and the
domain-face boundary condition on the lane the ledger already has open defects
on. That is a separate change with its own GPU witness.

When someone does it, this test goes red and must be DELETED, not loosened.

Not covered by this refusal: a model with a Debye or Lorentz pole. Its E update
on these lanes is the dispersive slab body, whose coefficients are built per E
component from the edge mean since #1260 (the staging reads each slab's
backward neighbour and replicates the boundary cell at a physical face), so a
dispersive distributed run agrees with the single-device one.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.core.yee import (
    FDTDState, MaterialArrays, e_update_coeffs,
    edge_averaged_e_update_coeffs,
)
from rfx.runners._distributed_common import _update_e_local

DT = 1.0e-12
DX = 1.0e-3
SHAPE = (6, 7, 8)


def test_the_distributed_slab_update_still_uses_the_owning_cell():
    idx = np.indices(SHAPE)[2]
    eps = jnp.asarray(np.where(idx < 4, 1.0, 9.0).astype(np.float32))
    sigma = jnp.zeros(SHAPE, jnp.float32)
    rng = np.random.default_rng(1210)
    h = [jnp.asarray(rng.standard_normal(SHAPE).astype(np.float32))
         for _ in range(3)]
    z = jnp.zeros(SHAPE, jnp.float32)
    st = FDTDState(ex=z, ey=z, ez=z, hx=h[0], hy=h[1], hz=h[2],
                   step=jnp.array(0, jnp.int32))

    out = _update_e_local(st, MaterialArrays(eps, sigma, jnp.ones(SHAPE)),
                          DT, DX)

    # The curl the slab body builds, spelled the way it spells it.
    from rfx.core.yee import _shift_bwd
    curl_x = ((h[2] - _shift_bwd(h[2], 1)) / DX
              - (h[1] - _shift_bwd(h[1], 2)) / DX)
    _, cb_owned = e_update_coeffs(eps, sigma, DT)
    _, (cb_edge, _, _) = edge_averaged_e_update_coeffs(eps, sigma, DT)

    got = np.asarray(out.ex)
    np.testing.assert_allclose(
        got, np.asarray(cb_owned * curl_x), rtol=1e-6,
        err_msg="the distributed slab body no longer uses the owning cell's "
                "coefficient -- if #1210 reached this lane, delete this test")
    rel = (np.abs(got - np.asarray(cb_edge * curl_x))
           / np.maximum(np.abs(got), 1e-30))
    assert rel.max() > 0.5, (
        "the distributed slab body and the single-device edge average agree "
        "on an eps = 1 / eps = 9 step, which they should not while this "
        "refusal stands")


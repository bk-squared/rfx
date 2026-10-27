"""Distributed slab exchange."""
from __future__ import annotations

from functools import partial
from jax import lax
from jax.sharding import PartitionSpec as P
from rfx.stepping.rank import rank_pair_exchange
from rfx.stepping.rank import rank_shard_map
import jax.numpy as jnp


def exchange_h_yee_shmap(state, mesh, n_devices, *, ranks, local=False):
    """Send packed Hy/Hz last-real rows right into the live LEFT ghosts.

    The second-order Yee E curl reads only these two x-neighbours. Keep
    physical-boundary ghosts verbatim; there is no wraparound on this lane.
    Refill only the Hy/Hz left ghosts consumed by the E curl.
    """
    if n_devices == 1:
        return state

    def _exchange(hy, hz, *, rank):
        packed = jnp.stack((hy[-2], hz[-2]))
        received = lax.ppermute(
            packed, "x", perm=[(i, i + 1) for i in range(n_devices - 1)])
        interior = rank > 0
        hy = hy.at[0].set(jnp.where(interior, received[0], hy[0]))
        hz = hz.at[0].set(jnp.where(interior, received[1], hz[0]))
        return hy, hz

    hy, hz = rank_pair_exchange(_exchange, mesh, ranks, state.hy, state.hz, local=local)
    return state._replace(hy=hy, hz=hz)


def exchange_e_yee_shmap(state, mesh, n_devices, *, ranks, local=False):
    """Send packed Ey/Ez first-real rows left into the live RIGHT ghosts.

    The H curl reads only these two x-neighbours; NTFF y/z faces also read the RIGHT Hx
    ghost, valid as every admitted H writer updates it like the owner row, except closed-box
    TF/SF, whose module sends it (another owner-only H writer needs a ghost-invariant test).

    Refill only Ey/Ez right ghosts from the neighbour ranks' real rows.

    Placement contract: this is the LAST stage of the E half-step in
    :func:`run_nonuniform_distributed_pec` — after sources and after every
    PEC stage — so a ghost row is a copy of the owner's FINISHED real row.
    The H update at a rank's last real cell reads ``Ey``/``Ez`` on the
    right ghost plane; a copy taken before the owner zeroed its PEC edges
    there is stale (#931 seam-cell divergence).
    """
    if n_devices == 1:
        return state

    def _exchange(ey, ez, *, rank):
        packed = jnp.stack((ey[1], ez[1]))
        received = lax.ppermute(
            packed, "x", perm=[(i, i - 1) for i in range(1, n_devices)])
        interior = rank < n_devices - 1
        ey = ey.at[-1].set(jnp.where(interior, received[0], ey[-1]))
        ez = ez.at[-1].set(jnp.where(interior, received[1], ez[-1]))
        return ey, ez

    ey, ez = rank_pair_exchange(_exchange, mesh, ranks, state.ey, state.ez, local=local)
    return state._replace(ey=ey, ez=ez)


def exchange_component_shmap(field, mesh, n_devices, *, ranks):
    """Exchange ghost cells for one field component using ``shard_map``.

    Extracted verbatim from ``distributed_v2.py::_exchange_component_shmap``
    and ``distributed_nu.py::_exchange_component_nu_shmap`` — those two
    bodies were byte-for-byte identical.

    ``field`` has global shape ``(nx_with_ghost*n_devices, ny, nz)`` when
    viewed from outside ``shard_map``; inside each shard sees
    ``(nx_local_with_ghost, ny, nz)``.

    Convention (also used by the retired pmap runner):
    - ``field[0]``  = left ghost  <- left neighbour's rightmost real cell
    - ``field[-1]`` = right ghost <- right neighbour's leftmost real cell
    - real cells: ``field[1:-1]``
    """

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=P("x"),
        out_specs=P("x"),
        check_rep=False,
    )
    def _exchange(f, *, rank):
        right_boundary = f[-2:-1, :, :]   # last real cell -> right neighbour's left ghost
        left_boundary = f[1:2, :, :]      # first real cell -> left neighbour's right ghost

        perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
        left_ghost_recv = lax.ppermute(right_boundary, "x", perm=perm_right)

        perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
        right_ghost_recv = lax.ppermute(left_boundary, "x", perm=perm_left)

        device_idx = rank

        left_ghost_val = jnp.where(device_idx > 0,
                                   left_ghost_recv,
                                   f[0:1, :, :])
        right_ghost_val = jnp.where(device_idx < n_devices - 1,
                                    right_ghost_recv,
                                    f[-1:, :, :])

        f = f.at[0:1, :, :].set(left_ghost_val)
        f = f.at[-1:, :, :].set(right_ghost_val)
        return f

    return _exchange(ranks, field)

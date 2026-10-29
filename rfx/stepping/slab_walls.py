"""Distributed slab walls."""
from __future__ import annotations

from functools import partial
from jax.sharding import Mesh, PartitionSpec as P
from rfx.core.yee import FDTDState
from rfx.model.conductors import distributed_mask_edges
from rfx.model.conductors import distributed_mask_spec
from rfx.stepping.rank import rank_shard_map
import jax.numpy as jnp


def apply_pec_face_shmap(state: FDTDState, mesh: Mesh, n_devices: int,
                         nx_local_with_ghost: int,
                         pad_x: int = 0, pec_faces: frozenset | None = None, *, ranks) -> FDTDState:
    """Apply PEC on the physical domain faces (x_lo, x_hi, y, z) under
    ``shard_map``, using device identity for the two x faces.

    ``pec_faces`` selects domain walls; None retains the all-face default.
    Y- and Z-face PEC is local to every rank.
    X-face PEC is rank-conditional: only rank 0 zeroes x_lo, only rank
    N-1 zeroes x_hi.

    The X faces act on the **first real cell** (``ghost``) and the
    **last real cell** (``nx_local_with_ghost - 1 - ghost - pad_x``),
    NOT on the seam ghost cells (which belong to neighbouring ranks)
    and NOT on the alignment-pad cells. ``pad_x`` (#622): when the last
    rank's slab carries alignment-pad cells past the real x-hi face
    (global node ``nx - 1``), the face must act on that real node, not
    on the padded slab end. This is V3 bullet 7 ("Hard PEC only acts on
    physical boundary or masked cells, not on seam ghosts") and V3
    bullet 6 ("Hard PEC does not re-zero interior seam cells of
    neighbouring ranks").

    ``nx_local_with_ghost`` is the per-rank slab length INCLUDING both
    ghost layers -- ``nx_padded // n_devices + 2 * ghost`` on both
    runners; ``ShardedNUGrid.nx_local`` is that same quantity and
    ``ShardedNUGrid.nx_per_rank`` is NOT (#1038 leg 3 derives this at
    every call site).

    Issue #1038 leg 3. This is ``distributed_v2.py::_apply_pec_shmap`` verbatim from the ``@partial`` down. ``distributed_nu.py`` carried a
    copy whose kernel differed in exactly one token -- its own parameter
    spelling -- plus three comments; the rename in the preceding commit
    made the two normalised-AST hashes equal
    (``29aeb6858fb45082...``) and the NU copy's richer face comments are
    folded into this docstring.

    **This function does not encode a hook point.** Callers own their
    ordering; this function only applies the faces. Both runners now call
    it AFTER source injection and BEFORE the E ghost exchange
    (``distributed_v2.step_fn_pec`` stage 5, ``distributed_nu`` stage 7).
    They diverged until #1041: v2 called it after its exchange. The
    measurement (``scripts/diagnostics/issue1041_v2_step_order.py``) found
    the ``exchange``/``face`` half of that divergence INERT on v2 -- with
    the source away from the seam the two orderings are bit-identical on
    every probe sample and all six final field arrays -- because the y/z
    faces above are written on every x row INCLUDING the ghosts, and the
    x_lo / x_hi faces act on rank 0's first and rank N-1's last real cell,
    whose exchanged copies the receiving rank discards. What was NOT inert
    was the ``exchange``/``source`` half, and the exchange moved for that.

    The retired pmap lane had its own ``_apply_pec_local``; since #1055
    that implementation sat at the same point in its step body, for the
    same measured reason. The remaining shard_map runners agree on that hook.
    """

    if pec_faces is None:
        pec_faces = frozenset(f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi"))
    if not pec_faces:
        return state

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(
            P("x"),  # ex
            P("x"),  # ey
            P("x"),  # ez
        ),
        out_specs=(
            P("x"),
            P("x"),
            P("x"),
        ),
        check_rep=False,
    )
    def _pec(ex, ey, ez, *, rank):
        ghost = 1

        if "y_lo" in pec_faces:
            ex = ex.at[:, 0, :].set(0.0)
            ez = ez.at[:, 0, :].set(0.0)
        if "y_hi" in pec_faces:
            ex = ex.at[:, -1, :].set(0.0)
            ez = ez.at[:, -1, :].set(0.0)
        if "z_lo" in pec_faces:
            ex = ex.at[:, :, 0].set(0.0)
            ey = ey.at[:, :, 0].set(0.0)
        if "z_hi" in pec_faces:
            ex = ex.at[:, :, -1].set(0.0)
            ey = ey.at[:, :, -1].set(0.0)

        device_idx = rank
        if "x_lo" in pec_faces:
            is_first = (device_idx == 0)
            ey_xlo = jnp.where(is_first, 0.0, ey[ghost, :, :])
            ez_xlo = jnp.where(is_first, 0.0, ez[ghost, :, :])
            ey = ey.at[ghost, :, :].set(ey_xlo)
            ez = ez.at[ghost, :, :].set(ez_xlo)
        if "x_hi" in pec_faces:
            is_last = (device_idx == n_devices - 1)
            last_real = nx_local_with_ghost - 1 - ghost - pad_x
            ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
            ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
            ey = ey.at[last_real, :, :].set(ey_xhi)
            ez = ez.at[last_real, :, :].set(ez_xhi)

        return ex, ey, ez

    ex, ey, ez = _pec(ranks, state.ex, state.ey, state.ez)
    return state._replace(ex=ex, ey=ey, ez=ez)


def apply_pec_mask_shmap(state: FDTDState, sharded_pec_mask, mesh,
                         n_devices: int, nx_local: int, *, ranks) -> FDTDState:
    """Apply geometry-defined PEC mask zeroing on x-sharded fields.

    Each rank owns the PEC cells inside its real-cell range
    ``[ghost, ghost + nx_per_rank)``.  Per V3 bullet 6, we must NOT
    re-zero PEC cells that live in another rank's slab; per V3 bullet 7,
    seam ghost cells must not be acted on.

    For legacy cell masks (instead of precomputed edge tuples), this:
      * computes the per-component edge masks on the local slab
        including ghost cells by CALLING
        ``rfx.boundaries.pec.realized_pec_edge_masks`` — the same
        function every single-device lane calls (#931 §1.7), so the two
        lanes cannot drift apart again (they did twice: an inlined
        ``jnp.roll`` copy through #689, then the pre-#931 sheet rule
        after the single-device lanes moved to the volume rule);
      * gates the mask so ghost-cell rows are forced to ``False`` before
        zeroing the field — interior real cells use their slab-local
        neighbour computation, and the **first/last real cells** see the
        ghost neighbour (which carries the seam-neighbour's PEC status
        because ``shard_pec_mask_x_slab`` populated it).

    The ghost rows are therefore NOT zeroed here.  They are refilled from
    the owner rank's (already zeroed) real row by the E ghost exchange,
    which the scan body runs AFTER this step (stage 9).  That order is
    load-bearing: the next H update at a rank's last real cell reads
    ``Ey``/``Ez`` on its right ghost plane, and when a body's cell is the
    neighbour's first real cell those edges are PEC only in the
    neighbour's copy.  Exchanging first handed this rank the un-zeroed
    value (#931 seam-cell divergence, 2.107e-01 final-step error).
    """
    if sharded_pec_mask is None:
        return state

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"), distributed_mask_spec(sharded_pec_mask)),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _pec_mask(ex, ey, ez, mask, *, rank):
        # ONE neighbour rule for both lanes: call the shared helper rather
        # than re-spelling it (#689 changed the rule and this copy did not
        # follow, so the two lanes disagreed at a y/z domain face —
        # measured on two 4x4 plates in a (6,6,10) domain, real cells only:
        #
        #   placement           single-device      this lane, inlined roll
        #   z faces  k=0 & k=9  [32, 32,  0]       [32, 32, 32]
        #   y faces  j=0 & j=5  [32,  0, 32]       [32, 32, 32]
        #   z interior k=1,k=8  [32, 32,  0]       [32, 32,  0]   (agreed)
        #
        # ``periodic=(True, False, False)``: the SHARDED axis keeps the
        # wrap, which is what the inlined copy relied on and is correct
        # here — a slab's ghost rows carry the seam neighbour's PEC status
        # (``shard_pec_mask_x_slab``), and the wrap only ever reaches
        # indices 0 and nx_local-1, both ghosts, both forced False below.
        # So no REAL cell sees the x wrap and the x behaviour is unchanged.
        # y and z have no ghosts and no periodic BC on this lane (the NU
        # runners install none), so they take the same zero-pad convention
        # ``rfx/nonuniform.py``'s ``apply_pec_mask(st, pec_mask)`` takes.
        mask_ex, mask_ey, mask_ez = distributed_mask_edges(mask)

        # Force ghost rows to False so we never touch a neighbour rank's
        # cells.  Real cells span [ghost, nx_local - ghost).
        ghost = 1
        ghost_zero = jnp.zeros_like(mask_ex[0:1, :, :])
        mask_ex = mask_ex.at[0:ghost, :, :].set(ghost_zero[0:ghost, :, :])
        mask_ex = mask_ex.at[nx_local - ghost:nx_local, :, :].set(
            ghost_zero[0:ghost, :, :])
        mask_ey = mask_ey.at[0:ghost, :, :].set(ghost_zero[0:ghost, :, :])
        mask_ey = mask_ey.at[nx_local - ghost:nx_local, :, :].set(
            ghost_zero[0:ghost, :, :])
        mask_ez = mask_ez.at[0:ghost, :, :].set(ghost_zero[0:ghost, :, :])
        mask_ez = mask_ez.at[nx_local - ghost:nx_local, :, :].set(
            ghost_zero[0:ghost, :, :])

        ex = ex * (1.0 - mask_ex.astype(ex.dtype))
        ey = ey * (1.0 - mask_ey.astype(ey.dtype))
        ez = ez * (1.0 - mask_ez.astype(ez.dtype))
        return ex, ey, ez

    ex, ey, ez = _pec_mask(ranks, state.ex, state.ey, state.ez, sharded_pec_mask)
    return state._replace(ex=ex, ey=ey, ez=ez)

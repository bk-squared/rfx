"""Shared scaffolding for the distributed FDTD runners.

This module holds shared helpers for ``distributed_v2.py`` and
``distributed_nu.py``. It also contains the uniform CPML helpers moved here
when the pmap runner in ``distributed.py`` was retired. The non-uniform
CPML kernel retains its per-axis, per-face spacing implementation.
"""

from __future__ import annotations

from rfx.stepping.slab import (
    Slab, Fill, cut, cut_poles, cut_pole_coeffs, fill_pole_rows, stage_poles,
    _vacuum_dispersion_values, vacuum_dispersion_values, shard_stacked, shard_stacked_poles,
)

from functools import partial

from rfx import _realized

import jax
import numpy as np
import jax.numpy as jnp

from rfx.core.drives import drive_layout, inject_drives
from jax import lax
from rfx.stepping.rank import mesh_ranks, rank_pair_exchange, rank_shard_map
from jax.sharding import Mesh, PartitionSpec as P

from rfx.model.conductors import distributed_mask_edges, distributed_mask_spec
from rfx.core.yee import (
    EPS_0,
    MU_0,
    FDTDState,
    MaterialArrays,
    ade_state_dtype,
    cell_owned_component_materials,
    component_e_materials,
    component_h_materials,
    edge_mean_components,
    e_update_coeffs,
    map_lumped,
    _shift_fwd,
    _shift_bwd,
)
from rfx.materials.debye import (
    DebyeCoeffs, DebyePole, DebyeState, debye_e_component, debye_e_coeffs,
    debye_pole_coeffs, init_debye, per_component,
)
from rfx.materials.lorentz import (
    LorentzCoeffs, LorentzPole, LorentzState, init_lorentz, lorentz_e_coeffs,
    lorentz_pole_coeffs, lorentz_e_component,
    lorentz_p_component, mixed_e_component_coeffs,
)

__all__ = [
    "split_array_x",
    "gather_array_x",
    "split_poles_x",
    "unstack_and_gather",
    "exchange_component_shmap",
    "apply_pec_face_shmap",
    "apply_pmc_face_shmap",
    "apply_pec_mask_shmap",
    "shard_stacked",
    "shard_stacked_poles",
    "inject_sources_shmap",
    "sample_probes_shmap",
    "_update_h_local_nu",
    "_update_e_local_nu",
    "slab_e_component_materials",
    "slab_e_materials_shmap",
    "update_h_nu_shmap",
    "update_e_nu_shmap",
    "_split_state",
    "_split_materials",
    "_split_debye_coeffs",
    "_split_debye_state",
    "_split_lorentz_coeffs",
    "_split_lorentz_state",
    "_update_h_local",
    "_update_e_local",
    "_update_e_debye_local",
    "_update_e_lorentz_local",
    "_update_e_local_with_dispersion",
    "_distributed_boundary_layers",
]


# ---------------------------------------------------------------------------
# Domain splitting / gathering (x-axis slabs with ghost cells)
# ---------------------------------------------------------------------------
#
# #1038 leg 2. Moved VERBATIM from ``distributed.py``, where they sat at
# L58/L107 -- ten lines AFTER that module's own import of this one. Nothing
# about them was duplicated; they move because they are the x-slab primitives
# every shared helper in this module that touches slabs has to call, and a
# shared body could not reach them at their old address without a circular
# import. ``distributed.py`` re-exports both at the position they were
# defined, so ``rfx.runners.distributed.split_array_x`` / ``.gather_array_x``
# still resolve for its external importers.
#
# Spatial placement delegates to rfx.stepping.slab; gather retains its
# original shape-only operation. The public re-exports remain identical.


def split_array_x(arr, n_devices, ghost=1, pad_value=0.0):
    if arr.shape[0] % n_devices:
        raise ValueError('split_array_x requires x length divisible by n_devices')
    return cut(arr, Slab(arr.shape[0], n_devices, ghost), Fill(pad_value, pad_value), mesh=None)


def shard_x_slabs(arr, n_devices, nx_per, ghost, pad_value, sharding):
    if arr.ndim != 3:
        raise ValueError('shard_x_slabs requires a 3-D array')
    return cut(arr, Slab(n_devices * nx_per, n_devices, ghost), Fill(pad_value, pad_value), mesh=sharding)


def stage_dispersion_slabs(materials, dt, debye_spec, lorentz_spec,
                           n_devices, nx_per, ghost, sharding, *, nx):
    """Place pole masks slab by slab, then build only the fixed ADE terms.

    E coefficients are deferred to the loop, where they read the same slab
    component materials as the plain E update and CPML. No whole-domain
    coefficient array or device-axis stack is constructed.
    """
    out = []
    for kind, spec in zip(("debye", "lorentz"), (debye_spec, lorentz_spec)):
        if spec is None:
            out.append(None)
            continue
        poles, masks = spec
        masks = jax.tree.map(
            lambda a: cut(a, Slab(nx, n_devices, ghost), "pole_mask", mesh=sharding),
            masks)
        out.append(stage_slab_pole_coeffs(
            poles, masks, dt, kind, sharding.mesh, nx_per, nx,
            (n_devices * (nx_per + 2 * ghost),) + materials.eps_r.shape[1:]))
    return tuple(out)


from rfx.model.electric_metrics import slab_cell_sizes, slab_metric_kwargs


def stage_slab_pole_coeffs(poles, masks, dt, kind, mesh, nx_per, nx, shape, *, grid=None):
    """Fixed ADE terms and zero carry; E coefficient slots stay ``None``.

    The compiled setup contains only pole-mask means and their fixed ADE
    factors, never differentiable epsilon/sigma coefficients. In particular,
    forward does not execute a large eager shard_map on every call (#1260).
    """
    def local(local_masks, *, rank):
        shape_local = (nx_per + 2,) + shape[1:]
        fractions = [None] * len(poles) if local_masks is None else [
            slab_pole_fractions(m, nx_per, nx, rank=rank, cell_sizes=slab_cell_sizes(grid, rank)) for m in
            (local_masks if isinstance(local_masks, (tuple, list)) else [local_masks] * len(poles))]
        if len(fractions) != len(poles):
            raise ValueError(f"Expected {len(poles)} {kind} masks, got {len(fractions)}")
        if kind == "debye":
            alpha, beta = debye_pole_coeffs(poles, dt, shape_local, fractions)
            coeffs = DebyeCoeffs(None, None, None, alpha, beta)
        else:
            a, b, c = lorentz_pole_coeffs(poles, dt, shape_local, fractions)
            coeffs = LorentzCoeffs(ca=None, cb=None, cc=None, a=a, b=b, c=c)
        return _pad_slab_dispersion_coeffs(coeffs, dt, nx_per, nx, rank=rank)

    state_type = DebyeState if kind == "debye" else LorentzState
    return stage_poles(local, masks, Slab(nx, mesh.size), shape[1:], mesh=mesh,
                       n_poles=len(poles), state_type=state_type,
                       dtype=ade_state_dtype(), kind=kind)



def _pad_slab_dispersion_coeffs(coeffs, dt, nx_per, nx, rank=None):
    return fill_pole_rows(coeffs, dt, nx_per, nx, rank=rank)


def slab_dispersion_coeffs(coeffs, e_materials, dt, nx_per, nx, rank=None):
    """Finish ADE E coefficients inside the loop from the shared edge means."""
    if isinstance(coeffs, DebyeCoeffs):
        full = debye_e_coeffs(e_materials, dt, coeffs.alpha, coeffs.beta)
    else:
        full = lorentz_e_coeffs(e_materials, dt, coeffs.a, coeffs.b, coeffs.c)
    return _pad_slab_dispersion_coeffs(full, dt, nx_per, nx, rank)


def gather_array_x(slabs, ghost=1):
    """Gather slabs back into a single array, stripping ghost cells.

    Parameters
    ----------
    slabs : ndarray, shape (n_devices, nx_local + 2*ghost, ny, nz)
    ghost : int

    Returns
    -------
    arr : ndarray, shape (nx, ny, nz)
    """
    # Strip ghost cells from each slab and concatenate
    inner = slabs[:, ghost:-ghost, :, :]  # (n_devices, nx_per, ny, nz)
    n_devices = inner.shape[0]
    # Reshape: merge device and x dims
    nx_per = inner.shape[1]
    ny = inner.shape[2]
    nz = inner.shape[3]
    return inner.reshape(n_devices * nx_per, ny, nz)


def split_poles_x(arr, n_poles, n_devices, ghost):
    return cut_poles(arr[:n_poles], Slab(arr.shape[1], n_devices, ghost), "state")


def unstack_and_gather(sharded_arr, n_devices, nx_local, ghost, pad_x, nx):
    """Flatten a sharded x-slab array back to one full-domain array.

    The shard_map runners hold fields as ``(n_devices * nx_local, ny, nz)``
    -- rank slabs concatenated along x, each still carrying its ghost cells.
    This reshapes that into ``(n_devices, nx_local, ny, nz)``, hands it to
    :func:`gather_array_x` (which strips the ghosts and merges the rank and x
    axes), and trims the high-x PEC padding cells that were added to make
    ``nx`` divisible by ``n_devices``.

    Parameters
    ----------
    sharded_arr : jax array, shape ``(n_devices * nx_local, ny, nz)``
    n_devices, nx_local, ghost, pad_x : int
        The slab decomposition the array was built with.
    nx : int
        Trim bound: the UNPADDED global x cell count. ``distributed_v2``
        passes its local ``nx`` (``grid.shape[0]``); ``distributed_nu``
        passes ``sharded_grid.nx``. Those are the same quantity -- both
        equal ``n_devices * (nx_local - 2 * ghost) - pad_x``, i.e. the
        gathered length less the padding -- which is why the two runners'
        copies of this body could merge; see the #1038 leg 2b commit
        message for the binding-by-binding derivation. It stays an explicit
        parameter rather than being recomputed here so that neither caller's
        expression changes.

    Must remain pure JAX: callers wrap the runners in ``jax.grad`` to drive
    an objective from the gathered ``final_state``, and an earlier
    ``np.array(sharded_arr)`` host-pull here raised
    ``TracerArrayConversionError``. Guarded by
    tests/unit/runners/test_distributed_v2_gather_traceable.py.
    """
    total_x = sharded_arr.shape[0]
    assert total_x == n_devices * nx_local, (
        f"unstack: total_x={total_x} != n_devices*nx_local={n_devices * nx_local}"
    )
    stacked = jnp.reshape(
        sharded_arr,
        (n_devices, nx_local) + tuple(sharded_arr.shape[1:]),
    )
    gathered = gather_array_x(stacked, ghost)
    # Trim padding cells if nx was padded
    if pad_x > 0:
        gathered = gathered[:nx]
    return gathered


# ---------------------------------------------------------------------------
# Vacuum CPML field-update coefficients
# ---------------------------------------------------------------------------
#
# Architect residual (a): the vacuum assumption is encoded in the NAME.
# The distributed CPML correction is applied where ``eps_r``/``mu_r`` are
# NOT in scope (``_apply_cpml_*_distributed`` and ``_apply_cpml_*_local_nu``
# receive only field arrays + profile coefficients). The vacuum form is
# therefore structural; these helpers + the guard test in
# ``tests/locks/test_cpml_axis_params_refactor_bit_identical.py`` make that
# explicit and prevent a CORE-C2-class hidden-assumption helper. A
# per-cell-``eps_r`` CPML coefficient is a separate future effort and is
# deliberately NOT supported by this signature.






# ---------------------------------------------------------------------------
# Stacked CPML psi allocation
# ---------------------------------------------------------------------------
#
# #1038 leg 2 (a). ``distributed._init_cpml_distributed`` and
# ``distributed_nu.init_cpml_for_sharded_nu`` each built the 24 psi arrays of
# a ``CPMLState`` through a nested ``_zeros`` closure over ``n_devices`` and
# the CPML layer count ``n``. The two bodies were the same statement spelled
# with different parameter names (``dim1, dim2`` vs ``d1, d2``) -- inventory
# §2.3(b). ``dim1, dim2`` is the spelling that landed; the second parameter
# keeps the terse name ``n`` the two closures used so the moved statement is
# byte-identical to the one it replaced.
#
# Both callers run at setup time, outside any jit/shard_map trace, so this
# move carries no jaxpr-shape change.




# ---------------------------------------------------------------------------
# shard_map ghost-cell exchange (single field component)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Domain-face boundary conditions inside shard_map
# ---------------------------------------------------------------------------
#
# Unlike the leg-1/leg-2 helpers, these were already TOP-LEVEL functions
# taking every value they need as an explicit parameter, on both runners.
# Nothing became a parameter that was not one before, so the inner kernel
# closes over exactly the names it closed over at its old address and the
# traced jaxpr is structurally unchanged -- the 13 baseline fixtures are
# what says so, not this comment.

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


def apply_pmc_face_shmap(state: FDTDState, mesh: Mesh, n_devices: int,
                         nx_local_with_ghost: int,
                         pmc_faces: frozenset, pad_x: int = 0, *, ranks) -> FDTDState:
    """Canonicalize only owned PMC high ghosts; retain every physical H row.

    The image path zeroes only the global high ghost, which no exchange sends,
    so the order relative to the exchange does not change any value.
    """
    if not pmc_faces:
        return state

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(
            P("x"),  # hx
            P("x"),  # hy
            P("x"),  # hz
        ),
        out_specs=(
            P("x"),
            P("x"),
            P("x"),
        ),
        check_rep=False,
    )
    def _pmc(hx, hy, hz, *, rank):
        from rfx.boundaries.pmc import apply_pmc_faces
        local = FDTDState(hx, hx, hx, hx, hy, hz, 0)
        local = apply_pmc_faces(
            local, pmc_faces, image=True, frame=cpml_slab_frame(hx.shape, 1, pad_x),
            owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})
        return local.hx, local.hy, local.hz

    hx, hy, hz = _pmc(ranks, state.hx, state.hy, state.hz)
    return state._replace(hx=hx, hy=hy, hz=hz)


# #1053 leg 2. The realized-PEC cell-mask kernel, moved here BY VALUE from
# ``distributed_nu.py`` so ``distributed_v2`` can call the same object rather
# than grow a second copy. It was already runner-agnostic -- ``(state,
# sharded_array, mesh, n_devices, nx_local)``, no grid, no NU spacing, no
# closure over anything -- so nothing became a parameter and the traced jaxpr
# is structurally unchanged. Only the def NAME differs from the pre-move
# source; the body from the docstring on is byte-identical, sha256
# eea6ee6aa20e5ea8e07869c0986a831d2a450143272fde6932887feb4d595e51 before and
# after.
#
# Re-spelling the neighbour rule here instead of calling
# ``rfx.boundaries.pec.realized_pec_edge_masks`` is forbidden by
# ``tests/contracts/test_pec_single_owner_lock.py``, and the repo has the
# scars: this lane's inlined ``jnp.roll`` copy drifted through #689, then the
# pre-#931 sheet rule drifted again after the single-device lanes moved to the
# volume rule. The call is the point of the function.


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


# ---------------------------------------------------------------------------
# CPML face-depth packing remains with the absorber adapters (#1535).




# ---------------------------------------------------------------------------
# Per-device source injection / probe sampling via shard_map
# ---------------------------------------------------------------------------
#
# Unlike the sharding helpers above, these two ARE traced: both are called
# from inside the jitted step body, and each builds a ``shard_map`` kernel
# whose Python loop is unrolled at trace time over ``*_local_specs`` and
# ``*_device_ids``.  Lifting them out of ``run_distributed`` /
# ``run_nonuniform_distributed_pec`` turns four closed-over locals into
# parameters, which is exactly the jaxpr-shape change the #1038 leg-0 lock
# was built to police.  It is done here because the lock says the arrays did
# not move: all 13 fixtures stay bit-identical, and the four that exercise
# these paths (distributed_v2_cpml, distributed_v2_pec,
# distributed_v2_nu_branch, distributed_nu_direct_pec) carry a non-empty,
# non-zero time_series, so the probe path is witnessed rather than assumed.
#
# The shape is ``exchange_component_shmap``'s: take ``mesh`` explicitly and
# build the ``shard_map`` inside.  The bodies are the pre-move bodies
# verbatim -- ``distributed_v2.py``'s copies differed only in a docstring,
# two lead comments and a one-spec-per-line ``in_specs`` layout, whose
# content is preserved in these docstrings instead.


def inject_sources_shmap(st, src_vals_step, mesh, n_src,
                         src_local_specs, src_device_ids, *, ranks, drives=None):
    """Inject sources on their owning device using ``shard_map``.

    ``shard_map`` gives each device its own slab; device identity inside
    the kernel comes from the explicit sharded rank input, and a source whose owner
    is some other device contributes ``jnp.where(... , 0.0)``.  So every
    device runs the same grouped add and only the owner's lands.

    ``in_specs`` are ``ex``, ``ey``, ``ez`` sharded on ``P("x")`` and
    ``src_vals_step`` slab-local (``P("x")``), shape (n_devices, n_sources).
    Each device receives its stored row, with all non-owned entries zero.

    ``src_local_specs[i]`` is ``(li, lj, lk, lc)``: the LOCAL index triple
    on the owning device plus the component name; ``src_device_ids[i]`` is
    that owner.  Both are Python data read at trace time, so the traced
    graph depends on their VALUES, not just their shapes.

    Shared by the uniform and graded runners. Their trace-time layout uses
    these local indices; unchanged waveform samples arrive as scan inputs.
    """
    if n_src == 0:
        return st

    if drives is None:
        drives = drive_layout(src_local_specs, src_vals_step.dtype,
                              electric_only_path="distributed")
    device_ids = np.asarray(src_device_ids, dtype=np.int32)

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"), P("x")),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _inject(ex, ey, ez, sv, *, rank):
        from rfx.core.yee import FDTDState
        local = FDTDState(ex, ey, ez, None, None, None, None)
        local = inject_drives(local, drives, jnp.where(rank == device_ids, sv[0], 0.0))
        return local.ex, local.ey, local.ez

    ex, ey, ez = _inject(ranks, st.ex, st.ey, st.ez, src_vals_step)
    return st._replace(ex=ex, ey=ey, ez=ez)


def sample_probes_shmap(st, mesh, n_prb, prb_local_specs, prb_device_ids,
                        *, reduce_devices=True, ranks):
    """Sample probes on their owning devices, then sum across devices.

    Mirror of :func:`inject_sources_shmap` on the read side: every device
    reads at the same local index, masks with ``jnp.where`` on device
    identity, and ``lax.psum`` over ``"x"`` leaves exactly the owner's
    value.  ``out_specs=P()`` because the psum result is replicated.

    With ``reduce_devices=False``, return masked samples of global shape
    ``(n_devices, n_prb)`` on ``P("x")`` without a collective. A scan can
    stack these and sum its device axis once after the time loop. The
    default retains the per-step replicated result for the NU runner.

    Extracted verbatim from
    ``distributed_nu.py::run_nonuniform_distributed_pec._sample_probes_shmap``
    and ``distributed_v2.py::run_distributed._sample_probes_shmap``.
    """
    if n_prb == 0:
        shape = (0,) if reduce_devices else (mesh.size, 0)
        return jnp.zeros(shape, dtype=jnp.float32)

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"),
                  P("x"), P("x"), P("x")),
        out_specs=P() if reduce_devices else P("x"),
        check_rep=False,
    )
    def _sample(ex, ey, ez, hx, hy, hz, *, rank):
        device_idx = rank
        samples = []
        for idx_p in range(n_prb):
            li, lj, lk, lc = prb_local_specs[idx_p]
            dev_id = prb_device_ids[idx_p]
            if lc == "ex":
                raw = ex[li, lj, lk]
            elif lc == "ey":
                raw = ey[li, lj, lk]
            elif lc == "ez":
                raw = ez[li, lj, lk]
            elif lc == "hx":
                raw = hx[li, lj, lk]
            elif lc == "hy":
                raw = hy[li, lj, lk]
            else:
                raw = hz[li, lj, lk]
            val = jnp.where(device_idx == dev_id, raw, 0.0)
            samples.append(val)
        samples = jnp.stack(samples)
        return lax.psum(samples, "x") if reduce_devices else samples[None, :]

    return _sample(ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz)


# ---------------------------------------------------------------------------
# Local NU update kernels (operate on per-device slab including ghosts)
# ---------------------------------------------------------------------------
#
# #1038 leg 4 (prerequisite). Moved VERBATIM -- name, signature, docstring and
# body byte-for-byte -- from ``distributed_nu.py`` L137/L169. Neither was
# duplicated; they move for the same reason leg 2 moved ``split_array_x`` /
# ``gather_array_x``: leg 4's shared NU shard wrappers below call them, and
# this module MUST NOT import from ``rfx.runners.distributed_nu`` (it is the
# leaf of the runner DAG, and ``distributed_nu`` imports it at L49). Their only
# dependencies are ``rfx.core.yee`` names, so the move is cycle-free.
#
# ``distributed_nu.py`` re-imports both at the position they were defined, so
# ``rfx.runners.distributed_nu._update_{h,e}_local_nu`` keeps resolving for
# ``tests/unit/runners/test_distributed_nu_kernel.py`` and for
# ``distributed_v2.py``'s function-local import. The two leading-underscore
# names in ``__all__`` above are deliberate: renaming a physics kernel to fit
# this module's public-name convention would put a rename inside a
# bit-identity leg for cosmetic reasons, so the exported spelling is the
# original one.

def _update_h_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full,
                      inv_dx_h_slab, inv_dy_h_full, inv_dz_h_full, *, cell_sizes=None):
    """H update on a local slab using NU inverse spacings.

    Mirrors ``rfx/core/yee.py::update_h_nu`` but accepts pre-sliced
    per-device ``inv_dx`` / ``inv_dx_h`` (length nx_local), while
    y/z spacings are replicated (full-axis).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    if cell_sizes is None:
        cell_sizes = tuple(1 / jnp.where(v > 0, v, v[-2])
                           for v in (inv_dx_h_slab, inv_dy_h_full, inv_dz_h_full))
    mu = component_h_materials(materials, cell_sizes=cell_sizes)
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H", cell_sizes=cell_sizes)
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch_x, ch_y, ch_z = dt / mu_x, dt / mu_y, dt / mu_z

    curl_x = (
        (_shift_fwd(ez, 1) - ez) * inv_dy_h_full[None, :, None]
        - (_shift_fwd(ey, 2) - ey) * inv_dz_h_full[None, None, :]
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) * inv_dz_h_full[None, None, :]
        - (_shift_fwd(ez, 0) - ez) * inv_dx_h_slab[:, None, None]
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) * inv_dx_h_slab[:, None, None]
        - (_shift_fwd(ex, 1) - ex) * inv_dy_h_full[None, :, None]
    )

    hx = state.hx - ch_x * curl_x
    hy = state.hy - ch_y * curl_y
    hz = state.hz - ch_z * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


from rfx.model.electric_metrics import _slab_x_lo_view, _slab_model_rows


from rfx.model.materials import slab_pole_fractions


def slab_e_component_materials(materials, nx_per, nx, rank=None, *, cell_sizes=None):
    """Per-E-component ``(eps_r, sigma)`` of one x slab, by the single-device
    rule (#1303).

    A Yee E component lies on an edge shared by four cells, and every lane
    gives it the mean of their eps and sigma: THE rule is
    :func:`rfx.core.yee.component_e_materials` (#1210, lumped stamps on their
    own edge #1236), and this function only hands it the right cells. The
    mean along x reads the row BEFORE the edge (Ey: i-1 and k-1, Ez: i-1 and
    j-1; Ex never reads x), so a slab needs one row below its first real row:

    * at a seam, the neighbour's last real cell. The staged material halo
      already holds it in the left ghost row; nothing is exchanged per step.
    * at the domain's x-lo face (rank 0), the boundary cell itself: outside
      the lattice the single-device rule edge-replicates. The staged ghost
      row there is vacuum padding, so the mean reads the first real row in
      its place. Only the mean does; the staged ghost keeps its value for
      every other reader (the #1317 drive reads it the same way).

    Rows that are not cells of the model -- the ghost rows at both slab ends
    and the high-x alignment pad -- keep the value of the cell they hold
    (:func:`rfx.core.yee.cell_owned_component_materials`), as they had before
    this lane was converted: their fields are replaced by the ghost exchange
    or stand in for the outside of the domain, and what the domain faces do
    is left as it was.

    ``materials`` : the slab's :class:`MaterialArrays`, ``(nx_local, ny, nz)``
        with one ghost row at each x end (lumped records optional).
    ``nx_per`` : real rows per slab (``nx_padded // n_devices``).
    ``nx`` : the unpadded global x cell count.
    ``rank`` : the slab's mesh position, supplied as data by the shard-map adapter.

    Returns ``((eps_x, eps_y, eps_z), (sig_x, sig_y, sig_z))``, slab-shaped.
    y and z are whole on every slab and non-periodic on these lanes (periodic
    axes are refused), so they take the helper's own edge replication.
    """
    if rank is None:
        raise ValueError("slab rank must be supplied as data")
    nx_local = materials.eps_r.shape[0]

    def x_lo_replicated(arr):
        return _slab_x_lo_view(arr, rank)

    view = MaterialArrays(
        eps_r=x_lo_replicated(materials.eps_r),
        sigma=x_lo_replicated(materials.sigma),
        mu_r=materials.mu_r,
        sigma_film=map_lumped(materials.sigma_film, x_lo_replicated),
        sigma_lumped=map_lumped(getattr(materials, "sigma_lumped", None),
                                x_lo_replicated),
        eps_r_lumped=map_lumped(getattr(materials, "eps_r_lumped", None),
                                x_lo_replicated))
    eps_edge, sig_edge = component_e_materials(view, (False, False, False), **({} if cell_sizes is None else dict(cell_sizes=cell_sizes)))
    eps_cell, sig_cell = cell_owned_component_materials(materials)
    model_cell = _slab_model_rows(nx_local, nx_per, nx, rank)

    def per_row(edge, cell):
        return tuple(jnp.where(model_cell, e, c) for e, c in zip(edge, cell))

    eps, sigma = per_row(eps_edge, eps_cell), per_row(sig_edge, sig_cell)
    if _realized.ACTIVE is not None:
        # Both coefficient paths consume these operands, including the
        # checkpointed shard_map. Record global ownership for the
        # full-grid reference.
        site = ("distributed_nu.E" if _realized.ACTIVE.lane == "fwd_distributed_nu"
                else "distributed.E")
        eps, sigma = _realized.electric(
            view, eps, sigma, site, owned_start=rank * nx_per,
            owned_count=jnp.minimum(nx_per, nx - rank * nx_per))
    return eps, sigma


def component_e_coeffs(e_materials, dt):
    """``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))`` from per-component
    ``((eps_x, eps_y, eps_z), (sig_x, sig_y, sig_z))``:
    :func:`rfx.core.yee.e_update_coeffs` of each component."""
    pairs = [e_update_coeffs(e, s, dt) for e, s in zip(*e_materials)]
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def slab_e_coeffs(materials, nx_per, nx, dt, rank=None):
    """The E coefficients of one x slab (#1303): :func:`component_e_coeffs`
    of :func:`slab_e_component_materials`. With :func:`slab_e_materials_shmap`
    the one path the multi-device E updates take to their coefficients."""
    return component_e_coeffs(
        slab_e_component_materials(materials, nx_per, nx, rank), dt)


def slab_e_materials_shmap(mat, mesh, nx_per, nx, *, ranks, grid=None):
    """:func:`slab_e_component_materials` of every slab, as x-sharded
    arrays, for a runner that averages once, before its time loop, and
    forms the coefficients (:func:`component_e_coeffs`) inside it.

    Why a runner would (#1303): inside the loop the mean is loop-invariant
    and XLA hoists it out of the loop anyway, so the forward program holds
    the same six coefficient slabs either way. Under ``jax.grad`` it does
    not: taken inside the loop (in the E update and again in the CPML), the
    four-cell mean is transposed at every step -- measured on CPU, 2.2x the
    cell-owned lane's time per step and 2.4x its backward scratch; taken
    here, once, about the cell-owned lane's. The coefficients stay inside
    the loop: built here instead, the permittivity gradient of a 1.5e16
    objective came back non-finite at four cells next to the current sources
    (``test_a_large_objective_gives_a_finite_gradient_on_both_lanes``; the
    mechanism was not established -- #1317 recorded a float32 overflow of
    ``Cb``'s reverse pass in the same test). ``jax.checkpoint`` recomputes
    this mean in the backward pass instead of keeping it on the tape.
    """
    @partial(rank_shard_map, mesh=mesh, in_specs=(P("x"),), out_specs=P("x"),
             check_rep=False)
    def _mean(local, *, rank):
        return slab_e_component_materials(local, nx_per, nx, rank=rank,
                                          **slab_metric_kwargs(grid, rank))

    return jax.checkpoint(_mean)(ranks, MaterialArrays(
        eps_r=mat.eps_r, sigma=mat.sigma, mu_r=mat.mu_r,
        sigma_film=mat.sigma_film,
        sigma_lumped=getattr(mat, "sigma_lumped", None),
        eps_r_lumped=getattr(mat, "eps_r_lumped", None)))


def _update_e_local_nu(state, e_coeffs,
                      inv_dx_slab, inv_dy_full, inv_dz_full, *, boundary=None, frame=None, owned=None):
    """E update on a local slab using NU inverse (cell-local) spacings.

    Mirrors ``rfx/core/yee.py::update_e_nu``. ``e_coeffs`` is the slab's
    per-component ``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))`` (#1303,
    :func:`slab_e_coeffs` / :func:`component_e_coeffs`).
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)
    (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = e_coeffs

    curl_x = (
        (hz - backward(hz, 1)) * inv_dy_full[None, :, None]
        - (hy - backward(hy, 2)) * inv_dz_full[None, None, :]
    )
    curl_y = (
        (hx - backward(hx, 2)) * inv_dz_full[None, None, :]
        - (hz - backward(hz, 0)) * inv_dx_slab[:, None, None]
    )
    curl_z = (
        (hy - backward(hy, 0)) * inv_dx_slab[:, None, None]
        - (hx - backward(hx, 1)) * inv_dy_full[None, :, None]
    )

    ex = ca_x * state.ex + cb_x * curl_x
    ey = ca_y * state.ey + cb_y * curl_y
    ez = ca_z * state.ez + cb_z * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


# ---------------------------------------------------------------------------
# NU shard_map update wrappers
# ---------------------------------------------------------------------------
#
# #1038 leg 4, inventory §2.4 -- THE renamed-clone pair the survey called the
# load-bearing finding. ``distributed_v2.py`` carried a copy of
# ``distributed_nu.py``'s NU shard wrappers under renamed inner functions
# (``_h_nu`` / ``_e_nu`` against ``_h`` / ``_e``), inside its ``if is_nu:``
# branch, and already imported the local kernels they wrap FROM
# ``distributed_nu``. The survey measured the two H copies at 0.884 and the two
# E copies at 0.947 similarity, the difference being the ``def`` name plus, for
# H, one re-wrapped argument list.
#
# Both copies were nested closures. The body below is ``distributed_nu.py``'s
# text, dedented, with the eight names it read from its enclosing
# ``run_nonuniform_distributed_pec`` turned into explicit parameters spelled
# exactly as they were spelled there -- so the statements are unchanged, only
# where the names come from. Both runners keep a same-named nested
# ``_update_h_shmap`` that forwards here, so no call site in either step body
# moved. This is the shape inventory §8 prescribes as the fusion-hazard
# mitigation, and the one ``exchange_component_shmap`` already uses: take the
# closed-over values explicitly and build the ``shard_map`` inside.
#
# TRACED, not eager. Unlike the leg-3 face kernels these are called from inside
# the jitted step body and lifting the locals to parameters changes the jaxpr's
# shape. Fixture 9 (``distributed_v2_nu_branch``) is the only fast-lane witness
# for v2's ``is_nu`` branch; fixtures 11-12 (``distributed_nu_*``) witness the
# NU runner. Those rows staying bit-identical is the evidence, not the argument.

def update_h_nu_shmap(st, mat, mesh, dt,
                      inv_dx_sharded, inv_dy_rep, inv_dz_rep,
                      inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep, *, ranks, cell_sizes=None):
    """H update on the NU distributed path, via ``shard_map``.

    Shared by ``distributed_nu.run_nonuniform_distributed_pec`` and by the
    ``is_nu`` branch of ``distributed_v2.run_distributed``.

    Parameters
    ----------
    st, mat
        The sharded :class:`FDTDState` and :class:`MaterialArrays`.
    mesh
        The 1-D x-axis :class:`jax.sharding.Mesh`.
    dt
        Timestep, closed over as a Python float by both original copies.
    inv_dx_sharded, inv_dx_h_sharded
        Per-device slabs of the inverse x spacings (cell-local and
        mean-spacing), sharded over ``"x"``.
    inv_dy_rep, inv_dz_rep, inv_dy_h_rep, inv_dz_h_rep
        Full-axis inverse y/z spacings, replicated on every device.

    Returns the state with ``hx``/``hy``/``hz``/``step`` replaced. The E-field
    components are passed in because the H curl reads them, and returned
    untouched by not being in ``out_specs``.
    """
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(mat, lane="distributed H shard", unsupported=True)
    if cell_sizes is None:
        cell_sizes = tuple(1 / jnp.where(v > 0, v, v[-2]) for v in
                           (inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep))
    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(
            P("x"), P("x"), P("x"),  # ex, ey, ez
            P("x"), P("x"), P("x"),  # hx, hy, hz
            P(),                     # step
            P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
            P("x"), P(None), P(None),  # inv_dx, inv_dy, inv_dz
            P("x"), P(None), P(None),  # inv_dx_h, inv_dy_h, inv_dz_h
            (P("x"), P(None), P(None)),  # original primal cell lengths
        ),
        out_specs=(P("x"), P("x"), P("x"), P()),
        check_rep=False,
    )
    def _h(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
           invdx, invdy, invdz, invdxh, invdyh, invdzh, widths, *, rank):
        _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
        _mat = MaterialArrays(eps_r=eps_r, sigma=sigma,
                             mu_r=magnetic_low_ghost(mu_r, rank))
        new_st = _update_h_local_nu(
            _st, _mat, dt, invdx, invdy, invdz, invdxh, invdyh, invdzh,
            cell_sizes=widths)
        return new_st.hx, new_st.hy, new_st.hz, new_st.step

    hx, hy, hz, step = _h(
        ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
        mat.eps_r, mat.sigma, mat.mu_r,
        inv_dx_sharded, inv_dy_rep, inv_dz_rep,
        inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep,
        cell_sizes,
    )
    return st._replace(hx=hx, hy=hy, hz=hz, step=step)


def update_e_nu_shmap(st, mat, mesh, dt,
                      inv_dx_sharded, inv_dy_rep, inv_dz_rep, nx_per, nx,
                      e_materials=None, *, ranks, boundary=None, ghost=1, pad_x=0):
    """E update on the NU distributed path, via ``shard_map``.

    The E sibling of :func:`update_h_nu_shmap`, shared by the same two
    runners and merged for the same reason -- inventory §2.4 measured the two
    inner kernels at 0.947 similarity, the entire residue being the ``def``
    name. It needs three inverse-spacing arrays rather than six because the E
    curl reads only the cell-local spacings, not the mean-spacing ones.

    Each component's eps and sigma are the mean over the four cells around
    its edge, the single-device rule (#1303): ``e_materials``, the x-sharded
    means :func:`slab_e_materials_shmap` built before the loop, or, when it
    is ``None``, :func:`slab_e_component_materials` of ``mat`` on each slab,
    from ``nx_per`` (real rows per slab) and ``nx`` (the unpadded global x
    count). The coefficients are formed here, in the loop, either way.

    Returns the state with ``ex``/``ey``/``ez``/``step`` replaced. Dispersion
    is NOT handled here: ``distributed_v2.run_distributed`` refuses Debye and
    Lorentz on the NU path upstream, and its caller passes the polarisation
    state straight back out.
    """
    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(
            P("x"), P("x"), P("x"),
            P("x"), P("x"), P("x"),
            P(),
            P("x"), P("x"), P("x"),
            P("x"), P("x"), P("x"),  # lumped and film component records
            P("x"), P(None), P(None),
            P("x"),                  # e_materials: None or ((eps...), (sig...))
        ),
        out_specs=(P("x"), P("x"), P("x"), P()),
        check_rep=False,
    )
    def _e(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
           sigma_lumped, eps_r_lumped, sigma_film, invdx, invdy, invdz, means, *, rank):
        _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
        if means is None:
            _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r,
                                  sigma_film=sigma_film,
                                  sigma_lumped=sigma_lumped,
                                  eps_r_lumped=eps_r_lumped)
            coeffs = slab_e_coeffs(_mat, nx_per, nx, dt, rank=rank)
        else:
            coeffs = component_e_coeffs(means, dt)
        new_st = _update_e_local_nu(_st, coeffs, invdx, invdy, invdz,
            **_slab_curl_kwargs(boundary, hx.shape, ghost, pad_x, rank, mesh.size))
        return new_st.ex, new_st.ey, new_st.ez, new_st.step

    ex, ey, ez, step = _e(
        ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
        mat.eps_r, mat.sigma, mat.mu_r,
        getattr(mat, "sigma_lumped", None), getattr(mat, "eps_r_lumped", None),
        mat.sigma_film, inv_dx_sharded, inv_dy_rep, inv_dz_rep, e_materials,
    )
    return st._replace(ex=ex, ey=ey, ez=ez, step=step)


# ---------------------------------------------------------------------------
# Uniform x-slab splitting, local Yee updates and x-slab CPML
# ---------------------------------------------------------------------------
#
# Moved VERBATIM from ``distributed.py`` ahead of retiring its ``jax.pmap``
# runner: ``distributed_v2`` imported twelve names from that module (the eleven
# below it uses, plus ``gather_array_x``, which leg 2 already moved here), and
# ``distributed_nu`` imported the four dispersive splitters from it lazily.
# ``_update_e_local``, ``_update_e_debye_local`` and ``_update_e_lorentz_local``
# ride along because ``_update_e_local_with_dispersion`` calls them. The names
# keep their leading underscore so every call site reads as it did;
# ``distributed.py`` re-exports all of them. Bodies are byte-identical to the
# ones they replace -- tests/locks/test_runner_split_bit_identity.py's baseline
# rows are what say the arrays did not move, and its _SHARED_HELPER_BINDINGS
# rows what says each importer holds this one object.


def _split_state(state, n_devices, ghost=1):
    layout = Slab(state.ex.shape[0], n_devices, ghost)
    return FDTDState(*(cut(field, layout, "state", mesh=None) for field in state[:6]),
                     step=jnp.broadcast_to(state.step, (n_devices,)))


def magnetic_low_ghost(mu_r, rank, ghost=1):
    """Replicate the physical low-x cell; preserve inter-rank material ghosts.

    Legacy staging pads outside with vacuum. That pad is not a neighbour
    cell of the new face mean. Only rank zero's exterior ghost is replaced.
    """
    return mu_r.at[:ghost].set(jnp.where(rank == 0, mu_r[ghost:ghost + 1],
                                       mu_r[:ghost]))


def _split_materials(materials, n_devices, ghost=1):
    """Legacy stacked material record; radius support is still refused."""
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(materials, lane="distributed material split", unsupported=True)
    return cut(materials, Slab(materials.eps_r.shape[0], n_devices, ghost),
               dict(eps_r="eps_r", sigma="sigma", mu_r="mu_r",
                    sigma_lumped="lumped", eps_r_lumped="lumped", sigma_film="lumped"), mesh=None)



# ---------------------------------------------------------------------------
# Dispersive material (Debye / Lorentz) splitting
# ---------------------------------------------------------------------------

def _split_debye_coeffs(coeffs: DebyeCoeffs, n_devices, ghost=1, *, dt):
    return cut_pole_coeffs(coeffs, Slab(jax.tree.leaves(coeffs.ca)[0].shape[0], n_devices, ghost),
                           "debye", dt=dt)


def _split_debye_state(state: DebyeState, n_devices, ghost=1):
    return cut_poles(state, Slab(state.px.shape[1], n_devices, ghost), "debye_state")


def _split_lorentz_coeffs(coeffs: LorentzCoeffs, n_devices, ghost=1, *, dt):
    return cut_pole_coeffs(coeffs, Slab(jax.tree.leaves(coeffs.ca)[0].shape[0], n_devices, ghost),
                           "lorentz", dt=dt)


def _split_lorentz_state(state: LorentzState, n_devices, ghost=1):
    return cut_poles(state, Slab(state.px.shape[1], n_devices, ghost), "lorentz_state")


# ---------------------------------------------------------------------------
# Local update functions (operate on per-device slab with ghosts)
# ---------------------------------------------------------------------------

def _update_h_local(state, materials, dt, dx):
    """H update on a local slab (including ghost cells).

    Identical to yee.update_h but without jit decorator and always
    non-periodic (ghost cells handle inter-device coupling).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = component_h_materials(materials)
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H")
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch_x, ch_y, ch_z = dt / mu_x, dt / mu_y, dt / mu_z

    curl_x = (
        (_shift_fwd(ez, 1) - ez) / dx
        - (_shift_fwd(ey, 2) - ey) / dx
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) / dx
        - (_shift_fwd(ez, 0) - ez) / dx
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) / dx
        - (_shift_fwd(ex, 1) - ex) / dx
    )

    hx = state.hx - ch_x * curl_x
    hy = state.hy - ch_y * curl_y
    hz = state.hz - ch_z * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local(state, e_coeffs, dx, *, boundary=None, frame=None, owned=None):
    """E update on a local slab (including ghost cells).

    Identical to yee.update_e but without jit decorator and always
    non-periodic. ``e_coeffs`` is the slab's per-component
    ``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))``, as :func:`slab_e_coeffs`
    builds them: from the mean eps and sigma over the four cells around each
    edge, with a lumped element on its own edge only -- the single-device
    rule (#1210, #1236, #1303).
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)
    (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = e_coeffs

    curl_x = (
        (hz - backward(hz, 1)) / dx
        - (hy - backward(hy, 2)) / dx
    )
    curl_y = (
        (hx - backward(hx, 2)) / dx
        - (hz - backward(hz, 0)) / dx
    )
    curl_z = (
        (hy - backward(hy, 0)) / dx
        - (hx - backward(hx, 1)) / dx
    )

    ex = ca_x * state.ex + cb_x * curl_x
    ey = ca_y * state.ey + cb_y * curl_y
    ez = ca_z * state.ez + cb_z * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


# ---------------------------------------------------------------------------
# Local dispersive E-update functions (operate on per-device slab)
# ---------------------------------------------------------------------------

def _update_e_debye_local(state, debye_coeffs, debye_state, dt, dx, *, boundary=None, frame=None, owned=None):
    """E update with Debye ADE on a local slab (always non-periodic).

    Per E component (#1260), the single-device body's arithmetic
    (``rfx.materials.debye.debye_e_component``) without its dtype narrowing.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)

    curl_x = ((hz - backward(hz, 1)) - (hy - backward(hy, 2))) / dx
    curl_y = ((hx - backward(hx, 2)) - (hz - backward(hz, 0))) / dx
    curl_z = ((hy - backward(hy, 0)) - (hx - backward(hx, 1))) / dx

    ex_new, px_new = debye_e_component(debye_coeffs, 0, state.ex, curl_x,
                                       debye_state.px)
    ey_new, py_new = debye_e_component(debye_coeffs, 1, state.ey, curl_y,
                                       debye_state.py)
    ez_new, pz_new = debye_e_component(debye_coeffs, 2, state.ez, curl_z,
                                       debye_state.pz)

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)
    return new_fdtd, new_debye


def _update_e_lorentz_local(state, lorentz_coeffs, lor_state, dt, dx, *, boundary=None, frame=None, owned=None):
    """E update with Lorentz/Drude ADE on a local slab (always non-periodic).

    Per E component (#1260), as ``rfx.stepping.poles.update_e_lorentz``
    without its dtype narrowing.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)

    curl_x = ((hz - backward(hz, 1)) - (hy - backward(hy, 2))) / dx
    curl_y = ((hx - backward(hx, 2)) - (hz - backward(hz, 0))) / dx
    curl_z = ((hy - backward(hy, 0)) - (hx - backward(hx, 1))) / dx

    px_new = lorentz_p_component(lorentz_coeffs, 0, state.ex, lor_state.px,
                                 lor_state.px_prev)
    py_new = lorentz_p_component(lorentz_coeffs, 1, state.ey, lor_state.py,
                                 lor_state.py_prev)
    pz_new = lorentz_p_component(lorentz_coeffs, 2, state.ez, lor_state.pz,
                                 lor_state.pz_prev)

    dpx = jnp.sum(px_new - lor_state.px, axis=0)
    dpy = jnp.sum(py_new - lor_state.py, axis=0)
    dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

    ex_new = lorentz_e_component(lorentz_coeffs, 0, state.ex, curl_x, dpx)
    ey_new = lorentz_e_component(lorentz_coeffs, 1, state.ey, curl_y, dpy)
    ez_new = lorentz_e_component(lorentz_coeffs, 2, state.ez, curl_z, dpz)

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_lor = LorentzState(
        px=px_new, py=py_new, pz=pz_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )
    return new_fdtd, new_lor


def _update_e_local_with_dispersion(state, materials, dt, dx,
                                     debye=None, lorentz=None, *, slab=None, e_materials=None, rank=None, boundary=None, frame=None, owned=None):
    """E update on a local slab with optional Debye/Lorentz dispersion.

    Every model uses the per-component edge means of
    :func:`slab_e_component_materials` (``slab = (nx_per, nx)``). Runners
    with dispersion pass the means formed before the loop and finish the
    material-dependent ADE coefficients here. Prebuilt coefficients remain
    accepted by grid-level callers.

    Returns (new_state, new_debye_state_or_None, new_lorentz_state_or_None).
    """
    if debye is None and lorentz is None:
        return _update_e_local(
            state, slab_e_coeffs(materials, *slab, dt, rank=rank), dx,
            boundary=boundary, frame=frame, owned=owned), None, None

    if e_materials is not None:
        if debye is not None:
            debye = (slab_dispersion_coeffs(debye[0], e_materials, dt, *slab, rank=rank), debye[1])
        if lorentz is not None:
            lorentz = (slab_dispersion_coeffs(lorentz[0], e_materials, dt, *slab, rank=rank), lorentz[1])

    if debye is not None and lorentz is None:
        debye_coeffs, debye_state = debye
        new_state, new_debye = _update_e_debye_local(
            state, debye_coeffs, debye_state, dt, dx, boundary=boundary, frame=frame, owned=owned)
        return new_state, new_debye, None

    if lorentz is not None and debye is None:
        lorentz_coeffs, lorentz_state = lorentz
        new_state, new_lorentz = _update_e_lorentz_local(
            state, lorentz_coeffs, lorentz_state, dt, dx, boundary=boundary, frame=frame, owned=owned)
        return new_state, None, new_lorentz

    # Mixed Debye + Lorentz: the single-device combined coefficients
    # (``mixed_e_component_coeffs``), per component (#1260). This body used to
    # take ca/cb/cc from the Debye build and the Lorentz dP coefficient as
    # 1/gamma_Lorentz; the combined update needs 1/gamma_total = 1/(gamma_Lorentz
    # + sum beta) there, which differs on every edge that carries both a Debye
    # and a Lorentz pole -- with the edge mean, every edge on a Debye|Lorentz
    # interface.
    debye_coeffs, debye_state = debye
    lorentz_coeffs, lorentz_state = lorentz
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)

    curl_x = ((hz - backward(hz, 1)) - (hy - backward(hy, 2))) / dx
    curl_y = ((hx - backward(hx, 2)) - (hz - backward(hz, 0))) / dx
    curl_z = ((hy - backward(hy, 0)) - (hx - backward(hx, 1))) / dx
    e_old = (state.ex, state.ey, state.ez)
    curls = (curl_x, curl_y, curl_z)
    p_l = (lorentz_state.px, lorentz_state.py, lorentz_state.pz)
    p_l_prev = (lorentz_state.px_prev, lorentz_state.py_prev,
                lorentz_state.pz_prev)
    p_d = (debye_state.px, debye_state.py, debye_state.pz)

    p_l_new = tuple(lorentz_p_component(lorentz_coeffs, c, e_old[c], p_l[c],
                                        p_l_prev[c]) for c in range(3))
    e_new, p_d_new = [], []
    for c in range(3):
        dp_l = jnp.sum(p_l_new[c] - p_l[c], axis=0)
        ca, cb, cc_d, cc_l = mixed_e_component_coeffs(
            debye_coeffs, lorentz_coeffs, c, dt)
        e_c = (ca * e_old[c] + cb * curls[c]
               + jnp.sum(cc_d * p_d[c], axis=0) - cc_l * dp_l)
        e_new.append(e_c)
        beta_c = per_component(debye_coeffs.beta, "beta")[c]
        p_d_new.append(debye_coeffs.alpha * p_d[c]
                       + beta_c * (e_c[None] + e_old[c][None]))

    new_fdtd = state._replace(ex=e_new[0], ey=e_new[1], ez=e_new[2],
                              step=state.step + 1)
    new_debye_st = DebyeState(px=p_d_new[0], py=p_d_new[1], pz=p_d_new[2])
    new_lor_st = LorentzState(
        px=p_l_new[0], py=p_l_new[1], pz=p_l_new[2],
        px_prev=lorentz_state.px, py_prev=lorentz_state.py, pz_prev=lorentz_state.pz,
    )
    return new_fdtd, new_debye_st, new_lor_st



# ---------------------------------------------------------------------------
# Distributed CPML support (Phase 2)
# ---------------------------------------------------------------------------



def _distributed_boundary_layers(grid, n_devices, *, pec_faces=None,
                                 pmc_faces=None, cpml_layers=None):
    """Keep absorbers and walls on owned rows, including boxes without CPML.

    PEC writes the end E row: one owned row. PMC-hi writes H one row
    inside it: two owned rows, otherwise the next H exchange replaces it.
    Two rows also keep PMC-lo away from the H row sent before the uniform
    runner applies the wall. An L-layer absorber needs L owned rows and
    the exchanged curl neighbour. Return the active depths for sizing.
    """
    pec_faces = set(getattr(grid, "pec_faces", None) or ()) if pec_faces is None else set(pec_faces)
    pmc_faces = set(getattr(grid, "pmc_faces", None) or ()) if pmc_faces is None else set(pmc_faces)
    from rfx.boundaries.depths import grid_face_depths

    layers = {face.name: face.realized for face in grid_face_depths(
        grid, pec_faces=pec_faces, pmc_faces=pmc_faces, budget=cpml_layers)}
    pad_x = (-grid.nx) % n_devices
    nx_per = (grid.nx + pad_x) // n_devices
    for face, rank, width in (("x_lo", 0, nx_per),
                              ("x_hi", n_devices - 1, max(0, nx_per - pad_x))):
        depth = layers[face]
        required = 2 if face in pmc_faces else 1
        if depth > 0 and width < depth:
            raise ValueError(
                f"distributed CPML face {face}: boundary rank {rank} has "
                f"{width} physical x cells, but {depth} absorber layers require "
                f"at least {depth} physical x cells plus 1 exchanged ghost cell "
                f"(nx={grid.nx}, devices={n_devices}, pad_x={pad_x}). "
                "Reduce devices or cpml_layers (or that face's "
                "lo_thickness/hi_thickness), or enlarge the x domain."
            )
        elif width < required:
            kind = "PMC" if face in pmc_faces else "PEC" if face in pec_faces else "boundary"
            raise ValueError(
                f"distributed {kind} face {face}: boundary rank {rank} has "
                f"{width} physical x cells, but {kind} requires at least "
                f"{required} physical x cells "
                f"(nx={grid.nx}, devices={n_devices}, pad_x={pad_x}). "
                "Reduce devices or enlarge the x domain."
            )
    return layers







def init_cpml_slabs(grid, nx_local, n_devices, *, mesh=None, **profile_options):
    """Build global profiles and the single-device psi layout on every rank.

    x faces are (s_x, Ny, Nz), y faces (L, s_y, Nz), and z faces
    (L, Ny, s_z). Merge the rank dimension into x when placing on a mesh.
    """
    from rfx.boundaries.cpml import CPMLState, init_cpml, _axis_buffer_depths
    from jax.sharding import NamedSharding

    _distributed_boundary_layers(grid, n_devices)
    params, _ = init_cpml(grid, **profile_options)
    depths = _axis_buffer_depths(grid, grid.cpml_layers)
    extents = (nx_local, grid.ny, grid.nz)
    state = {}
    for name in CPMLState._fields:
        axis = "xyz".index(name.split("_")[-1][0])
        shape = tuple(depths[i] if i == axis else extents[i] for i in range(3))
        if mesh is None:
            state[name] = jnp.zeros((n_devices, *shape), jnp.float32)
        else:
            state[name] = jnp.zeros((n_devices * shape[0], *shape[1:]), jnp.float32,
                                   device=NamedSharding(mesh, P("x")))
    return params, CPMLState(**state)


def cpml_h_materials(mu_r, rank, ghost):
    """Reuse the uniform slab's component means, including its low ghost."""
    from types import SimpleNamespace
    if mu_r is None:
        means = component_h_materials(MaterialArrays(None, None, 1.0))
    else:
        mu_r = magnetic_low_ghost(mu_r, rank, ghost)
        means = component_h_materials(MaterialArrays(None, None, mu_r))
    return SimpleNamespace(mu_r=mu_r, components=SimpleNamespace(mu_update=means))


def cpml_slab_frame(shape, ghost, pad_x):
    """The last rank's physical x end is static on every rank."""
    return {"x": (ghost, shape[0] - ghost - pad_x)}


def cpml_absorbing_faces(grid):
    """Select the realized absorbing faces of the global model."""
    from rfx.boundaries.depths import grid_face_depths
    return tuple(face.name for face in grid_face_depths(grid) if face.realized > 0)


def _h_backward(h, axis, *, boundary=None, frame=None, owned=None):
    """Keep nonmagnetic slab programs unchanged; share the one magnetic image."""
    if boundary is None or not boundary.pmc_faces:
        return _shift_bwd(h, axis)
    from rfx.core.yee import h_neighbor
    return h_neighbor(h, axis, boundary=boundary,
                      frame=None if frame is None else frame.get("xyz"[axis]), owned=owned)


def _slab_curl_kwargs(boundary, shape, ghost, pad_x, rank, n_devices):
    """Only magnetic slabs trace ownership; transverse axes use the full extent."""
    if boundary is None or not boundary.pmc_faces:
        return {}
    return dict(boundary=boundary, frame=cpml_slab_frame(shape, ghost, pad_x),
                owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})

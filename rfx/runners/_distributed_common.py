"""Shared scaffolding for the distributed FDTD runners.

This module holds the genuinely-common, bit-identical helpers used by
``distributed.py``, ``distributed_v2.py`` and ``distributed_nu.py``.
It is a *mechanical de-duplication* surface only — no algebra change.

Scope rationale (Stage 1.5a):

* ``distributed_v2.py`` already imports the CPML loop body from
  ``distributed.py`` (``_apply_cpml_{e,h}_distributed``), so the
  uniform-CPML scaffolding is NOT triplicated and is not re-extracted
  here.
* ``distributed_nu.py`` carries the *non-uniform* CPML variant
  (axis-aware per-face spacing). That is a genuinely different kernel
  and is intentionally NOT merged.
* What IS byte-for-byte duplicated, and is extracted here:
  - the vacuum CPML field-update coefficients, and
  - the ``shard_map`` ghost-exchange inner body (duplicated verbatim
    between ``distributed_v2.py`` and ``distributed_nu.py``).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.core.yee import (
    EPS_0,
    MU_0,
    FDTDState,
    MaterialArrays,
    ade_state_dtype,
    cell_owned_component_materials,
    lumped_components,
    map_lumped,
    _shift_fwd,
    _shift_bwd,
)
from rfx.materials.debye import DebyeCoeffs, DebyeState, init_debye
from rfx.materials.lorentz import LorentzCoeffs, LorentzState, init_lorentz

__all__ = [
    "cpml_coeff_e_vacuum",
    "cpml_coeff_h_vacuum",
    "split_array_x",
    "gather_array_x",
    "split_poles_x",
    "unstack_and_gather",
    "zeros_psi_stacked",
    "exchange_component_shmap",
    "apply_pec_face_shmap",
    "apply_pmc_face_shmap",
    "apply_pec_mask_shmap",
    "shard_stacked",
    "shard_stacked_poles",
    "shard_stacked_psi",
    "inject_sources_shmap",
    "sample_probes_shmap",
    "_update_h_local_nu",
    "_update_e_local_nu",
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
    "_init_cpml_distributed",
    "_apply_cpml_e_distributed",
    "_apply_cpml_h_distributed",
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
# Both are closure-free and use only ``jnp``, so this module stays the
# dependency-DAG leaf (inventory §2.5): it imports nothing from
# ``rfx.runners.*``.


def split_array_x(arr, n_devices, ghost=1, pad_value=0.0):
    """Split a 3D array into N slabs along x with ghost cells.

    Parameters
    ----------
    arr : ndarray, shape (nx, ny, nz)
    n_devices : int
    ghost : int
        Number of ghost cells on each side.
    pad_value : float
        Value used for ghost cells at the physical boundary (device 0
        left ghost and device N-1 right ghost).  Default 0.0 is correct
        for field arrays; use 1.0 for eps_r and mu_r to avoid division
        by zero in the Yee update.

    Returns
    -------
    slabs : ndarray, shape (n_devices, nx_local + 2*ghost, ny, nz)
    """
    nx = arr.shape[0]
    nx_per = nx // n_devices
    slabs = []
    for i in range(n_devices):
        x_start = i * nx_per
        x_end = x_start + nx_per

        # Desired range including ghosts
        want_lo = x_start - ghost
        want_hi = x_end + ghost

        # Clamp to valid array range
        g_lo = max(0, want_lo)
        g_hi = min(nx, want_hi)

        slab_data = arr[g_lo:g_hi]

        # Pad where the desired range exceeds array bounds
        pad_lo = g_lo - want_lo   # > 0 when want_lo < 0
        pad_hi = want_hi - g_hi   # > 0 when want_hi > nx

        if pad_lo > 0 or pad_hi > 0:
            pad_widths = [(pad_lo, pad_hi)] + [(0, 0)] * (arr.ndim - 1)
            slab_data = jnp.pad(slab_data, pad_widths, mode='constant',
                                constant_values=pad_value)

        slabs.append(slab_data)
    return jnp.stack(slabs)


def shard_x_slabs(arr, n_devices, nx_per, ghost, pad_value, sharding):
    """Place padded x slabs directly, constructing only addressable shards.

    ``arr`` already includes any high-x alignment padding. Physical-boundary
    ghosts use ``pad_value``; interior ghosts copy the adjacent slab's cells,
    exactly as in ``shard_stacked(split_array_x(...))``. No device-axis stack
    or whole-domain reshape is staged on the default device.
    """
    if arr.ndim != 3:
        raise ValueError(f"shard_x_slabs stages 3-D (x, y, z) arrays, got shape {arr.shape}")
    nx_local = nx_per + 2 * ghost
    shape = (n_devices * nx_local,) + arr.shape[1:]

    def slab(index):
        rank = (index[0].start or 0) // nx_local
        want_lo = rank * nx_per - ghost
        want_hi = (rank + 1) * nx_per + ghost
        lo, hi = max(0, want_lo), min(arr.shape[0], want_hi)
        data = arr[lo:hi]
        if lo != want_lo or hi != want_hi:
            data = jnp.pad(
                data, ((lo - want_lo, want_hi - hi), (0, 0), (0, 0)),
                constant_values=pad_value,
            )
        return data

    # No ``dtype=`` argument: jax.make_array_from_callback gained it only after
    # 0.5.0 (the VESSL image runs 0.4.33). Slicing and jnp.pad keep arr's dtype.
    return jax.make_array_from_callback(shape, sharding, slab)


def stage_dispersion_slabs(materials, dt, debye_spec, lorentz_spec,
                           n_devices, nx_per, ghost, sharding):
    """Initialize ADE data one addressable x slab at a time.

    Interior ghosts use neighbouring cells, just like ``shard_x_slabs``.
    Physical ghosts are padded *after* initialization: their coefficients
    must match the legacy splitters, not coefficients of vacuum materials.
    No whole-domain dispersion array or device-axis stack is constructed.
    """
    nx, ny, nz = materials.eps_r.shape
    nx_local = nx_per + 2 * ghost
    shape = (n_devices * nx_local, ny, nz)
    specs = (debye_spec, lorentz_spec)
    coefficient_slabs = ([], [])

    def stage_slab(device, index):
        rank = (index[0].start or 0) // nx_local
        want_lo = rank * nx_per - ghost
        want_hi = (rank + 1) * nx_per + ghost
        lo, hi = max(0, want_lo), min(nx, want_hi)
        padding = ((lo - want_lo, want_hi - hi), (0, 0), (0, 0))
        # Move only the clipped inputs. Coefficients and temporary ADE zeros
        # are then built on their destination, including on remote-process
        # meshes where only this process's addressable devices are visited.
        # The lumped-stamp records (#1210) are None or, since #1236, a
        # per-component 3-tuple: slice each array they hold, keep the shape.
        def _local(arr):
            return jax.device_put(arr[lo:hi], device)

        local_materials = MaterialArrays(
            eps_r=_local(materials.eps_r), sigma=_local(materials.sigma),
            mu_r=_local(materials.mu_r),
            sigma_lumped=map_lumped(materials.sigma_lumped, _local),
            eps_r_lumped=map_lumped(materials.eps_r_lumped, _local))
        for spec, init, slabs in zip(
                specs, (init_debye, init_lorentz), coefficient_slabs):
            if spec is None:
                continue
            poles, masks = spec
            local_masks = jax.tree.map(
                lambda mask: jax.device_put(mask[lo:hi], device), masks)
            # Slice before changing the default device, so an uncommitted
            # whole-domain input cannot migrate just to execute its slice.
            with jax.default_device(device):
                # Keep the runner's existing field_dtype=None policy.
                coeffs, state = init(poles, local_materials, dt, mask=local_masks)
                del state  # the global carry is allocated directly sharded below

                def place(name, arr):
                    pad_value = float(1.0 / EPS_0) if init is init_lorentz and name == "cc" else 0.0
                    if lo != want_lo or hi != want_hi:
                        widths = padding if arr.ndim == 3 else ((0, 0),) + padding
                        arr = jnp.pad(arr, widths, constant_values=pad_value)
                    return jax.device_put(arr, device)

                placed = type(coeffs)(*(place(name, arr) for name, arr in
                                       zip(coeffs._fields, coeffs)))
                # Finish the handoff before dropping slab temporaries and
                # starting another slab; asynchronous dispatch must not pile
                # up setup buffers on the source device.
                jax.block_until_ready(placed)
                slabs.append(placed)
                del coeffs, placed, local_masks

    if any(spec is not None for spec in specs):
        for device, index in sharding.addressable_devices_indices_map(shape).items():
            stage_slab(device, index)

    def assemble(spec, slabs, state_type):
        if spec is None:
            return None
        coeffs = type(slabs[0])(*(
            jax.make_array_from_single_device_arrays(
                (n_devices * parts[0].shape[0],) + parts[0].shape[1:],
                sharding, list(parts))
            for parts in zip(*slabs)))
        state_shape = (n_devices * len(spec[0]), nx_local, ny, nz)
        state = state_type(*(jnp.zeros(state_shape, dtype=ade_state_dtype(), device=sharding)
                             for _ in state_type._fields))
        return coeffs, state

    return (assemble(debye_spec, coefficient_slabs[0], DebyeState),
            assemble(lorentz_spec, coefficient_slabs[1], LorentzState))


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
    """Split a per-pole array ``(n_poles, nx, ny, nz)`` into x-slabs.

    Applies :func:`split_array_x` to each pole and stacks the results on
    axis 1, giving ``(n_devices, n_poles, nx_local, ny, nz)``. Ghost cells
    at the physical x boundaries are padded with ``0.0``, which is the
    correct fill for every array this is used on: the polarization state
    fields of Debye/Lorentz, and Lorentz's ``a``/``b``/``c`` coefficients.

    NOT for a coefficient whose ghost fill must be non-zero --
    ``_split_lorentz_coeffs`` splits ``cc`` with ``pad_value=1/EPS_0``
    through :func:`split_array_x` directly, and its comment records the
    NaN-in-backward reason. That call is deliberately not routed here.
    """
    return jnp.stack([
        split_array_x(arr[p], n_devices, ghost, pad_value=0.0)
        for p in range(n_poles)
    ], axis=1)


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


def cpml_coeff_e_vacuum(dt: float) -> float:
    """Vacuum CPML E-field update coefficient ``dt / eps_0``.

    Returns ``dt / EPS_0`` using the canonical ``rfx.core.yee.EPS_0``
    (post-2019 SI value 8.8541878128e-12). The pre-refactor distributed
    runners held the literal ``8.854187817e-12`` inline; Stage 3.5b
    consolidated every EPS_0 site onto the canonical constant.

    Takes ONLY ``dt`` — no ``eps_r``/``materials`` argument. The vacuum
    assumption is intentional and load-bearing: see the module docstring.
    """
    return dt / EPS_0


def cpml_coeff_h_vacuum(dt: float) -> float:
    """Vacuum CPML H-field update coefficient ``dt / mu_0``.

    Returns ``dt / MU_0`` using the canonical ``rfx.core.yee.MU_0``
    (post-2019 SI value 1.25663706212e-6). The pre-refactor distributed
    runners held the literal ``1.2566370614e-6`` inline; Stage 3.5b
    consolidated every MU_0 site onto the canonical post-2019 constant.

    Takes ONLY ``dt`` — no ``mu_r``/``materials`` argument. The vacuum
    assumption is intentional and load-bearing: see the module docstring.
    """
    return dt / MU_0


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


def zeros_psi_stacked(n_devices, n, dim1, dim2):
    """Zero CPML psi array stacked over ranks: ``(n_devices, n, dim1, dim2)``.

    Parameters
    ----------
    n_devices : int
        Number of ranks the psi array is stacked over (axis 0).
    n : int
        CPML layer count (``grid.cpml_layers``) -- the psi depth on axis 1.
    dim1, dim2 : int
        The two face-parallel extents. Which grid dimensions these are
        depends on the face; callers pass ``nx_local`` for a y-/z-face psi
        that indexes along the rank-local x slab, and the global ``ny``/``nz``
        for an x-face psi.

    ``float32`` is fixed here, matching the single-device
    ``rfx.boundaries.cpml.init_cpml`` convention both callers mirror.
    """
    return jnp.zeros((n_devices, n, dim1, dim2), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# shard_map ghost-cell exchange (single field component)
# ---------------------------------------------------------------------------


def exchange_component_shmap(field, mesh, n_devices):
    """Exchange ghost cells for one field component using ``shard_map``.

    Extracted verbatim from ``distributed_v2.py::_exchange_component_shmap``
    and ``distributed_nu.py::_exchange_component_nu_shmap`` — those two
    bodies were byte-for-byte identical.

    ``field`` has global shape ``(nx_with_ghost*n_devices, ny, nz)`` when
    viewed from outside ``shard_map``; inside each shard sees
    ``(nx_local_with_ghost, ny, nz)``.

    Convention (matches the pmap version in ``distributed.py``):
    - ``field[0]``  = left ghost  <- left neighbour's rightmost real cell
    - ``field[-1]`` = right ghost <- right neighbour's leftmost real cell
    - real cells: ``field[1:-1]``
    """

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=P("x"),
        out_specs=P("x"),
        check_rep=False,
    )
    def _exchange(f):
        right_boundary = f[-2:-1, :, :]   # last real cell -> right neighbour's left ghost
        left_boundary = f[1:2, :, :]      # first real cell -> left neighbour's right ghost

        perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
        left_ghost_recv = lax.ppermute(right_boundary, "x", perm=perm_right)

        perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
        right_ghost_recv = lax.ppermute(left_boundary, "x", perm=perm_left)

        device_idx = lax.axis_index("x")

        left_ghost_val = jnp.where(device_idx > 0,
                                   left_ghost_recv,
                                   f[0:1, :, :])
        right_ghost_val = jnp.where(device_idx < n_devices - 1,
                                    right_ghost_recv,
                                    f[-1:, :, :])

        f = f.at[0:1, :, :].set(left_ghost_val)
        f = f.at[-1:, :, :].set(right_ghost_val)
        return f

    return _exchange(field)


def exchange_h_yee_shmap(state, mesh, n_devices):
    """Send packed Hy/Hz last-real rows right into the live LEFT ghosts.

    The second-order Yee E curl reads only these two x-neighbours. Keep
    physical-boundary ghosts verbatim; there is no wraparound on this lane.
    """
    if n_devices == 1:
        return state

    @partial(shard_map, mesh=mesh, in_specs=(P("x"), P("x")),
             out_specs=(P("x"), P("x")), check_rep=False)
    def _exchange(hy, hz):
        packed = jnp.stack((hy[-2], hz[-2]))
        received = lax.ppermute(
            packed, "x", perm=[(i, i + 1) for i in range(n_devices - 1)])
        interior = lax.axis_index("x") > 0
        hy = hy.at[0].set(jnp.where(interior, received[0], hy[0]))
        hz = hz.at[0].set(jnp.where(interior, received[1], hz[0]))
        return hy, hz

    hy, hz = _exchange(state.hy, state.hz)
    return state._replace(hy=hy, hz=hz)


def exchange_e_yee_shmap(state, mesh, n_devices):
    """Send packed Ey/Ez first-real rows left into the live RIGHT ghosts.

    The second-order Yee H curl reads only these two x-neighbours. Ex/Hx
    and the opposite ghosts are dead: local ghost updates are overwritten
    by the next live exchange before a real cell can consume them.
    """
    if n_devices == 1:
        return state

    @partial(shard_map, mesh=mesh, in_specs=(P("x"), P("x")),
             out_specs=(P("x"), P("x")), check_rep=False)
    def _exchange(ey, ez):
        packed = jnp.stack((ey[1], ez[1]))
        received = lax.ppermute(
            packed, "x", perm=[(i, i - 1) for i in range(1, n_devices)])
        interior = lax.axis_index("x") < n_devices - 1
        ey = ey.at[-1].set(jnp.where(interior, received[0], ey[-1]))
        ez = ez.at[-1].set(jnp.where(interior, received[1], ez[-1]))
        return ey, ez

    ey, ez = _exchange(state.ey, state.ez)
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
                         pad_x: int = 0) -> FDTDState:
    """Apply PEC on the physical domain faces (x_lo, x_hi, y, z) under
    ``shard_map``, using device identity for the two x faces.

    Y- and Z-face PEC is local to every rank and runs unconditionally.
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

    #1038 leg 3. This is ``distributed_v2.py::_apply_pec_shmap``
    verbatim from the ``@partial`` down. ``distributed_nu.py`` carried a
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

    The legacy pmap lane does not call this function -- it has its own
    ``distributed.py::_apply_pec_local`` -- but since #1055 that one sits at
    the same point in its step body, for the same measured reason. All three
    lanes now agree on the hook point.
    """

    @partial(
        shard_map,
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
    def _pec(ex, ey, ez):
        ghost = 1

        # Y-axis PEC (all devices)
        ex = ex.at[:, 0, :].set(0.0)
        ex = ex.at[:, -1, :].set(0.0)
        ez = ez.at[:, 0, :].set(0.0)
        ez = ez.at[:, -1, :].set(0.0)

        # Z-axis PEC (all devices)
        ex = ex.at[:, :, 0].set(0.0)
        ex = ex.at[:, :, -1].set(0.0)
        ey = ey.at[:, :, 0].set(0.0)
        ey = ey.at[:, :, -1].set(0.0)

        device_idx = lax.axis_index("x")

        # X-lo PEC: device 0 only
        is_first = (device_idx == 0)
        ey_xlo = jnp.where(is_first, 0.0, ey[ghost, :, :])
        ez_xlo = jnp.where(is_first, 0.0, ez[ghost, :, :])
        ey = ey.at[ghost, :, :].set(ey_xlo)
        ez = ez.at[ghost, :, :].set(ez_xlo)

        # X-hi PEC: device N-1 only (skip ghost AND alignment pad, #622)
        is_last = (device_idx == n_devices - 1)
        last_real = nx_local_with_ghost - 1 - ghost - pad_x
        ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
        ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
        ey = ey.at[last_real, :, :].set(ey_xhi)
        ez = ez.at[last_real, :, :].set(ez_xhi)

        return ex, ey, ez

    ex, ey, ez = _pec(state.ex, state.ey, state.ez)
    return state._replace(ex=ex, ey=ey, ez=ez)


def apply_pmc_face_shmap(state: FDTDState, mesh: Mesh, n_devices: int,
                         nx_local_with_ghost: int,
                         pmc_faces: frozenset, pad_x: int = 0) -> FDTDState:
    """Apply PMC (``H_tangential = 0``) on the physical domain faces under
    ``shard_map``.

    Electromagnetic dual of :func:`apply_pec_face_shmap`: PEC zeroes
    tangential E, PMC zeroes tangential H. Y- and Z-face PMC is local to
    every rank; X-face PMC is rank-conditional (only rank 0 zeroes
    x_lo, only rank N-1 zeroes x_hi).

    Yee convention: a ``_hi`` face acts on index ``-2`` (half a cell
    INSIDE the wall), not ``-1`` (the ghost outside) -- see
    ``rfx/boundaries/pmc.py``. On the x axis that is ``last_inside``,
    one cell in from the last real cell
    ``nx_local_with_ghost - 1 - ghost - pad_x``; the faces never act on
    a seam ghost, nor on the alignment-pad cells when ``pad_x > 0``
    (#622, same class as the PEC face fix).

    ``pmc_faces`` is a Python ``frozenset`` read at trace time, so each
    face's ``if`` is resolved while the kernel is staged out, not at
    run time. An empty set short-circuits before the kernel is built.

    #1038 leg 3. ``distributed_v2.py::_apply_pmc_shmap`` and
    ``distributed_nu.py::_apply_pmc_face_nu_shmap`` were the SAME kernel
    with one token of difference, each runner's own spelling of the
    slab length; after the rename in this leg's first commit the two
    inner bodies hashed byte-identical
    (``c6baaf31de52299c39c564403a6fda4fbebdaea26014bbceb7f12d30983f0b12``).

    **This function does not encode a hook point, and the two runners
    disagree about one.** ``distributed_v2`` calls it AFTER the H ghost
    exchange (``step_fn_cpml`` 3b / ``step_fn_pec`` 2b);
    ``distributed_nu`` calls it BEFORE the H ghost exchange (step 2b),
    so that the zero propagates to neighbour ranks through the
    exchange. Both cite the same OQ9 directive for the H-half placement
    and reach opposite conclusions about the exchange. That is inventory
    §3.2 / leg 5 territory -- a physics question with its own gate --
    and merging the kernel settles nothing about it. Callers own their
    ordering.
    """
    if not pmc_faces:
        return state

    @partial(
        shard_map,
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
    def _pmc(hx, hy, hz):
        ghost = 1

        # Yee convention: _hi PMC acts on index -2 (0.5·dx INSIDE the
        # wall), not -1 (ghost outside). See rfx/boundaries/pmc.py.
        if "y_lo" in pmc_faces:
            hx = hx.at[:, 0, :].set(0.0)
            hz = hz.at[:, 0, :].set(0.0)
        if "y_hi" in pmc_faces:
            hx = hx.at[:, -2, :].set(0.0)
            hz = hz.at[:, -2, :].set(0.0)
        if "z_lo" in pmc_faces:
            hx = hx.at[:, :, 0].set(0.0)
            hy = hy.at[:, :, 0].set(0.0)
        if "z_hi" in pmc_faces:
            hx = hx.at[:, :, -2].set(0.0)
            hy = hy.at[:, :, -2].set(0.0)

        device_idx = lax.axis_index("x")
        is_first = (device_idx == 0)
        is_last = (device_idx == n_devices - 1)
        last_real = nx_local_with_ghost - 1 - ghost - pad_x
        last_inside = last_real - 1

        if "x_lo" in pmc_faces:
            hy_new = jnp.where(is_first, 0.0, hy[ghost, :, :])
            hz_new = jnp.where(is_first, 0.0, hz[ghost, :, :])
            hy = hy.at[ghost, :, :].set(hy_new)
            hz = hz.at[ghost, :, :].set(hz_new)
        if "x_hi" in pmc_faces:
            hy_new = jnp.where(is_last, 0.0, hy[last_inside, :, :])
            hz_new = jnp.where(is_last, 0.0, hz[last_inside, :, :])
            hy = hy.at[last_inside, :, :].set(hy_new)
            hz = hz.at[last_inside, :, :].set(hz_new)

        return hx, hy, hz

    hx, hy, hz = _pmc(state.hx, state.hy, state.hz)
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
                         n_devices: int, nx_local: int) -> FDTDState:
    """Apply geometry-defined PEC mask zeroing on x-sharded fields.

    Each rank owns the PEC cells inside its real-cell range
    ``[ghost, ghost + nx_per_rank)``.  Per V3 bullet 6, we must NOT
    re-zero PEC cells that live in another rank's slab; per V3 bullet 7,
    seam ghost cells must not be acted on.

    The implementation:
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
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"), P("x")),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _pec_mask(ex, ey, ez, mask):
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
        mask_ex, mask_ey, mask_ez = realized_pec_edge_masks(
            mask, periodic=(True, False, False))

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

    ex, ey, ez = _pec_mask(state.ex, state.ey, state.ez, sharded_pec_mask)
    return state._replace(ex=ex, ey=ey, ez=ez)


# ---------------------------------------------------------------------------
# Device-axis merge + x-slab sharding (host side, eager)
# ---------------------------------------------------------------------------
#
# These three take an array whose leading axis is the DEVICE axis, fold that
# axis into the next one, and ``jax.device_put`` the result onto an
# x-sharding.  They run at setup time, outside any ``jit``/``shard_map``
# trace, so lifting them out of their enclosing closures cannot change a
# jaxpr -- the only thing that moved into the signature is ``shd``, a
# ``NamedSharding`` the caller already had in hand (#1038 leg 1).
#
# The bodies are the pre-move bodies verbatim.  They are three FUNCTIONS and
# not one, on purpose: ``shard_stacked``'s generic ``*rest`` form would
# numerically subsume the other two, but that would be an algebra change on a
# leg whose whole contract is byte-identical motion.  Merging them is a later
# decision with its own evidence, not a side effect of de-duplication.


def shard_stacked(arr, shd):
    """Merge the device axis into x, then shard.

    ``(n_devices, nx_local, ny, nz) -> (n_devices*nx_local, ny, nz)``.

    Extracted verbatim from four byte-identical copies that carried two
    different names: ``distributed_nu.py::shard_debye_coeffs_x_slab._shard_3d``,
    ``distributed_nu.py::shard_lorentz_coeffs_x_slab._shard_3d``,
    ``distributed_nu.py::run_nonuniform_distributed_pec._shard_stacked`` and
    ``distributed_v2.py::run_distributed._shard_stacked``.  All four bodies
    hashed ``ab71430ea8cb``.
    """
    n_dev = arr.shape[0]
    rest = arr.shape[1:]
    return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)


def shard_stacked_poles(arr, shd):
    """Merge the device axis into the pole axis, then shard.

    ``(n_devices, n_poles, nx_local, ny, nz) ->``
    ``(n_devices*n_poles, nx_local, ny, nz)``, so ``P("x")`` hands each
    device ``(n_poles, nx_local, ny, nz)``.

    Extracted verbatim from the four ``_shard_4d`` copies in
    ``distributed_nu.py`` (``shard_debye_coeffs_x_slab``,
    ``shard_debye_state_x_slab``, ``shard_lorentz_coeffs_x_slab``,
    ``shard_lorentz_state_x_slab``), body ``2883a7c93bd2`` on all four.

    NOT the same text as ``distributed_v2.py::_shard_stacked_5d``, which
    computes the same thing through a named intermediate; that copy is
    outside this leg's byte-identical set and is deliberately left alone.
    """
    n_dev, n_poles, nx_loc, ny_a, nz_a = arr.shape
    return jax.device_put(
        arr.reshape(n_dev * n_poles, nx_loc, ny_a, nz_a), shd,
    )


def shard_stacked_psi(arr, shd):
    """Merge the device axis into the CPML-depth axis, then shard.

    ``arr``: ``(n_devices, n_cpml, d1, d2)`` -- re-interpreted as
    ``(n_devices * n_cpml, d1, d2)`` so x-sharding distributes the first
    axis across devices correctly.  Each device owns ``n_cpml`` rows.

    Extracted verbatim from ``distributed_nu.py::shard_cpml_state_x_slab.
    _shard_psi`` and ``distributed_v2.py::_init_cpml_sharded._shard_psi``,
    body ``6ae90bb58fb5`` on both (the copies differed only in comments).
    """
    n_dev, n_c, d1, d2 = arr.shape
    merged = arr.reshape(n_dev * n_c, d1, d2)
    return jax.device_put(merged, shd)


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
                         src_local_specs, src_device_ids):
    """Inject sources on their owning device using ``shard_map``.

    ``shard_map`` gives each device its own slab; device identity inside
    the kernel comes from ``lax.axis_index("x")``, and a source whose owner
    is some other device contributes ``jnp.where(... , 0.0)``.  So every
    device runs the same unrolled add and only the owner's lands.

    ``in_specs`` are ``ex``, ``ey``, ``ez`` sharded on ``P("x")`` and
    ``src_vals_step`` replicated (``P()``) -- it is a per-step scalar vector
    indexed by source, not a field.

    ``src_local_specs[i]`` is ``(li, lj, lk, lc)``: the LOCAL index triple
    on the owning device plus the component name; ``src_device_ids[i]`` is
    that owner.  Both are Python data read at trace time, so the traced
    graph depends on their VALUES, not just their shapes.

    Extracted verbatim from
    ``distributed_nu.py::run_nonuniform_distributed_pec._inject_sources_shmap``
    and ``distributed_v2.py::run_distributed._inject_sources_shmap``.
    """
    if n_src == 0:
        return st

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"), P()),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _inject(ex, ey, ez, sv):
        device_idx = lax.axis_index("x")
        for idx_s in range(n_src):
            li, lj, lk, lc = src_local_specs[idx_s]
            dev_id = src_device_ids[idx_s]
            val = jnp.where(device_idx == dev_id, sv[idx_s], 0.0)
            if lc == "ex":
                ex = ex.at[li, lj, lk].add(val)
            elif lc == "ey":
                ey = ey.at[li, lj, lk].add(val)
            elif lc == "ez":
                ez = ez.at[li, lj, lk].add(val)
        return ex, ey, ez

    ex, ey, ez = _inject(st.ex, st.ey, st.ez, src_vals_step)
    return st._replace(ex=ex, ey=ey, ez=ez)


def sample_probes_shmap(st, mesh, n_prb, prb_local_specs, prb_device_ids,
                        *, reduce_devices=True):
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
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"),
                  P("x"), P("x"), P("x")),
        out_specs=P() if reduce_devices else P("x"),
        check_rep=False,
    )
    def _sample(ex, ey, ez, hx, hy, hz):
        device_idx = lax.axis_index("x")
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

    return _sample(st.ex, st.ey, st.ez, st.hx, st.hy, st.hz)


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
                      inv_dx_h_slab, inv_dy_h_full, inv_dz_h_full):
    """H update on a local slab using NU inverse spacings.

    Mirrors ``rfx/core/yee.py::update_h_nu`` but accepts pre-sliced
    per-device ``inv_dx`` / ``inv_dx_h`` (length nx_local), while
    y/z spacings are replicated (full-axis).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = materials.mu_r * MU_0

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

    hx = state.hx - (dt / mu) * curl_x
    hy = state.hy - (dt / mu) * curl_y
    hz = state.hz - (dt / mu) * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full):
    """E update on a local slab using NU inverse (cell-local) spacings.

    Mirrors ``rfx/core/yee.py::update_e_nu``.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy_full[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz_full[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz_full[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx_slab[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx_slab[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy_full[None, :, None]
    )

    ex = ca * state.ex + cb * curl_x
    ey = ca * state.ey + cb * curl_y
    ez = ca * state.ez + cb * curl_z

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
                      inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep):
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
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("x"), P("x"), P("x"),  # ex, ey, ez
            P("x"), P("x"), P("x"),  # hx, hy, hz
            P(),                     # step
            P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
            P("x"), P(None), P(None),  # inv_dx, inv_dy, inv_dz
            P("x"), P(None), P(None),  # inv_dx_h, inv_dy_h, inv_dz_h
        ),
        out_specs=(P("x"), P("x"), P("x"), P()),
        check_rep=False,
    )
    def _h(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
           invdx, invdy, invdz, invdxh, invdyh, invdzh):
        _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
        _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        new_st = _update_h_local_nu(
            _st, _mat, dt, invdx, invdy, invdz, invdxh, invdyh, invdzh)
        return new_st.hx, new_st.hy, new_st.hz, new_st.step

    hx, hy, hz, step = _h(
        st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
        mat.eps_r, mat.sigma, mat.mu_r,
        inv_dx_sharded, inv_dy_rep, inv_dz_rep,
        inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep,
    )
    return st._replace(hx=hx, hy=hy, hz=hz, step=step)


def update_e_nu_shmap(st, mat, mesh, dt,
                      inv_dx_sharded, inv_dy_rep, inv_dz_rep):
    """E update on the NU distributed path, via ``shard_map``.

    The E sibling of :func:`update_h_nu_shmap`, shared by the same two
    runners and merged for the same reason -- inventory §2.4 measured the two
    inner kernels at 0.947 similarity, the entire residue being the ``def``
    name. It needs three inverse-spacing arrays rather than six because the E
    curl reads only the cell-local spacings, not the mean-spacing ones.

    Returns the state with ``ex``/``ey``/``ez``/``step`` replaced. Dispersion
    is NOT handled here: ``distributed_v2.run_distributed`` refuses Debye and
    Lorentz on the NU path upstream, and its caller passes the polarisation
    state straight back out.
    """
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("x"), P("x"), P("x"),
            P("x"), P("x"), P("x"),
            P(),
            P("x"), P("x"), P("x"),
            P("x"), P(None), P(None),
        ),
        out_specs=(P("x"), P("x"), P("x"), P()),
        check_rep=False,
    )
    def _e(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
           invdx, invdy, invdz):
        _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
        _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        new_st = _update_e_local_nu(_st, _mat, dt, invdx, invdy, invdz)
        return new_st.ex, new_st.ey, new_st.ez, new_st.step

    ex, ey, ez, step = _e(
        st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
        mat.eps_r, mat.sigma, mat.mu_r,
        inv_dx_sharded, inv_dy_rep, inv_dz_rep,
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
    """Split an FDTDState into per-device slabs with ghost cells."""
    return FDTDState(
        ex=split_array_x(state.ex, n_devices, ghost),
        ey=split_array_x(state.ey, n_devices, ghost),
        ez=split_array_x(state.ez, n_devices, ghost),
        hx=split_array_x(state.hx, n_devices, ghost),
        hy=split_array_x(state.hy, n_devices, ghost),
        hz=split_array_x(state.hz, n_devices, ghost),
        step=jnp.broadcast_to(state.step, (n_devices,)),
    )


def _split_materials(materials, n_devices, ghost=1):
    """Split MaterialArrays into per-device slabs with ghost cells.

    Uses pad_value=1.0 for eps_r and mu_r (vacuum) to prevent
    division-by-zero in Yee updates at boundary ghost cells, and
    pad_value=0.0 for sigma (lossless). The per-component lumped records
    (#1236) are split like ``sigma`` (a ghost holds no stamp).
    """
    def _split_lumped(arr):
        return split_array_x(arr, n_devices, ghost, pad_value=0.0)

    return MaterialArrays(
        eps_r=split_array_x(materials.eps_r, n_devices, ghost, pad_value=1.0),
        sigma=split_array_x(materials.sigma, n_devices, ghost, pad_value=0.0),
        mu_r=split_array_x(materials.mu_r, n_devices, ghost, pad_value=1.0),
        sigma_lumped=map_lumped(
            getattr(materials, "sigma_lumped", None), _split_lumped),
        eps_r_lumped=map_lumped(
            getattr(materials, "eps_r_lumped", None), _split_lumped),
    )



# ---------------------------------------------------------------------------
# Dispersive material (Debye / Lorentz) splitting
# ---------------------------------------------------------------------------

def _split_debye_coeffs(coeffs: DebyeCoeffs, n_devices, ghost=1):
    """Split DebyeCoeffs arrays along x into per-device slabs.

    3-D arrays (nx, ny, nz) are split with ghost cells.
    4-D arrays (n_poles, nx, ny, nz) are split per-pole along x.
    """
    # ca, cb are (nx, ny, nz)
    ca = split_array_x(coeffs.ca, n_devices, ghost, pad_value=0.0)
    cb = split_array_x(coeffs.cb, n_devices, ghost, pad_value=0.0)

    # cc, alpha, beta are (n_poles, nx, ny, nz) — split each pole along x
    n_poles = coeffs.alpha.shape[0]
    cc_slabs = jnp.stack([
        split_array_x(coeffs.cc[p], n_devices, ghost, pad_value=0.0)
        for p in range(n_poles)
    ], axis=1)  # (n_devices, n_poles, nx_local, ny, nz)
    alpha_slabs = jnp.stack([
        split_array_x(coeffs.alpha[p], n_devices, ghost, pad_value=0.0)
        for p in range(n_poles)
    ], axis=1)
    beta_slabs = jnp.stack([
        split_array_x(coeffs.beta[p], n_devices, ghost, pad_value=0.0)
        for p in range(n_poles)
    ], axis=1)

    return DebyeCoeffs(ca=ca, cb=cb, cc=cc_slabs, alpha=alpha_slabs, beta=beta_slabs)


def _split_debye_state(state: DebyeState, n_devices, ghost=1):
    """Split DebyeState arrays along x into per-device slabs.

    Each field is (n_poles, nx, ny, nz) — split each pole along x.
    """
    n_poles = state.px.shape[0]

    def _split_poles(arr):
        # (n_devices, n_poles, nx_local, ny, nz)
        return split_poles_x(arr, n_poles, n_devices, ghost)

    return DebyeState(
        px=_split_poles(state.px),
        py=_split_poles(state.py),
        pz=_split_poles(state.pz),
    )


def _split_lorentz_coeffs(coeffs: LorentzCoeffs, n_devices, ghost=1):
    """Split LorentzCoeffs arrays along x into per-device slabs."""
    ca = split_array_x(coeffs.ca, n_devices, ghost, pad_value=0.0)
    cb = split_array_x(coeffs.cb, n_devices, ghost, pad_value=0.0)
    # cc = 1 / safe_gamma; the mixed Debye+Lorentz path computes
    # `gamma_base = 1 / cc`, so padding cc=0 at physical-boundary ghosts
    # yields gamma_base=inf and `numer_base = ca*gamma_base = 0*inf = NaN`.
    # Forward stays bit-perfect (ghost cells drop before output assembly),
    # but in backward `cb_mixed[ghost] * curl = 0 * NaN = NaN` leaks the
    # NaN from the ghost's cb_mixed into real cells' hy/hz gradients and
    # ultimately into d_loss/d_eps at the first/last x cells.  Pad with
    # the vacuum cc value (1/EPS_0) so gamma_base=EPS_0 is finite at
    # ghosts; ca/alpha/beta stay padded with 0 so the ghost update is a
    # no-op forward.
    cc = split_array_x(coeffs.cc, n_devices, ghost, pad_value=float(1.0 / EPS_0))

    n_poles = coeffs.a.shape[0]

    def _split_poles(arr):
        return split_poles_x(arr, n_poles, n_devices, ghost)

    return LorentzCoeffs(
        ca=ca, cb=cb,
        a=_split_poles(coeffs.a),
        b=_split_poles(coeffs.b),
        c=_split_poles(coeffs.c),
        cc=cc,
    )


def _split_lorentz_state(state: LorentzState, n_devices, ghost=1):
    """Split LorentzState arrays along x into per-device slabs."""
    n_poles = state.px.shape[0]

    def _split_poles(arr):
        return split_poles_x(arr, n_poles, n_devices, ghost)

    return LorentzState(
        px=_split_poles(state.px),
        py=_split_poles(state.py),
        pz=_split_poles(state.pz),
        px_prev=_split_poles(state.px_prev),
        py_prev=_split_poles(state.py_prev),
        pz_prev=_split_poles(state.pz_prev),
    )


# ---------------------------------------------------------------------------
# Local update functions (operate on per-device slab with ghosts)
# ---------------------------------------------------------------------------

def _update_h_local(state, materials, dt, dx):
    """H update on a local slab (including ghost cells).

    Identical to yee.update_h but without jit decorator and always
    non-periodic (ghost cells handle inter-device coupling).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = materials.mu_r * MU_0

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

    hx = state.hx - (dt / mu) * curl_x
    hy = state.hy - (dt / mu) * curl_y
    hz = state.hz - (dt / mu) * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local(state, materials, dt, dx):
    """E update on a local slab (including ghost cells).

    Identical to yee.update_e but without jit decorator and always
    non-periodic. The coefficients are CELL-owned (#1210 not converted on
    this lane, ``test_distributed_e_coefficients_are_still_cell_owned.py``),
    except for a lumped element: it loads its own E edge only (#1236), so
    when the slab carries a lumped record each component takes the cell
    total minus the stamps of the other two components
    (:func:`rfx.core.yee.cell_owned_component_materials`). With no record
    this is the single-coefficient update it always was.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    has_lumped = any(
        part is not None
        for rec in (getattr(materials, "sigma_lumped", None),
                    getattr(materials, "eps_r_lumped", None))
        for part in lumped_components(rec))
    if has_lumped:
        eps_c, sig_c = cell_owned_component_materials(materials)

        def _coeffs(eps_r, sigma):
            eps = eps_r * EPS_0
            sigma_dt_2eps = sigma * dt / (2.0 * eps)
            return ((1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps),
                    (dt / eps) / (1.0 + sigma_dt_2eps))

        (ca_x, cb_x), (ca_y, cb_y), (ca_z, cb_z) = (
            _coeffs(e, s_) for e, s_ in zip(eps_c, sig_c))
    else:
        eps = materials.eps_r * EPS_0
        sigma = materials.sigma

        sigma_dt_2eps = sigma * dt / (2.0 * eps)
        ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
        cb = (dt / eps) / (1.0 + sigma_dt_2eps)
        ca_x = ca_y = ca_z = ca
        cb_x = cb_y = cb_z = cb

    curl_x = (
        (hz - _shift_bwd(hz, 1)) / dx
        - (hy - _shift_bwd(hy, 2)) / dx
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) / dx
        - (hz - _shift_bwd(hz, 0)) / dx
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) / dx
        - (hx - _shift_bwd(hx, 1)) / dx
    )

    ex = ca_x * state.ex + cb_x * curl_x
    ey = ca_y * state.ey + cb_y * curl_y
    ez = ca_z * state.ez + cb_z * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


# ---------------------------------------------------------------------------
# Local dispersive E-update functions (operate on per-device slab)
# ---------------------------------------------------------------------------

def _update_e_debye_local(state, debye_coeffs, debye_state, dt, dx):
    """E update with Debye ADE on a local slab (always non-periodic)."""
    hx, hy, hz = state.hx, state.hy, state.hz
    ca, cb, cc = debye_coeffs.ca, debye_coeffs.cb, debye_coeffs.cc
    alpha, beta = debye_coeffs.alpha, debye_coeffs.beta

    curl_x = ((hz - _shift_bwd(hz, 1)) - (hy - _shift_bwd(hy, 2))) / dx
    curl_y = ((hx - _shift_bwd(hx, 2)) - (hz - _shift_bwd(hz, 0))) / dx
    curl_z = ((hy - _shift_bwd(hy, 0)) - (hx - _shift_bwd(hx, 1))) / dx

    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    ex_new = ca * ex_old + cb * curl_x + jnp.sum(cc * debye_state.px, axis=0)
    ey_new = ca * ey_old + cb * curl_y + jnp.sum(cc * debye_state.py, axis=0)
    ez_new = ca * ez_old + cb * curl_z + jnp.sum(cc * debye_state.pz, axis=0)

    px_new = alpha * debye_state.px + beta * (ex_new[None] + ex_old[None])
    py_new = alpha * debye_state.py + beta * (ey_new[None] + ey_old[None])
    pz_new = alpha * debye_state.pz + beta * (ez_new[None] + ez_old[None])

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)
    return new_fdtd, new_debye


def _update_e_lorentz_local(state, lorentz_coeffs, lor_state, dt, dx):
    """E update with Lorentz/Drude ADE on a local slab (always non-periodic)."""
    hx, hy, hz = state.hx, state.hy, state.hz
    ca, cb, cc = lorentz_coeffs.ca, lorentz_coeffs.cb, lorentz_coeffs.cc
    a, b, c = lorentz_coeffs.a, lorentz_coeffs.b, lorentz_coeffs.c

    curl_x = ((hz - _shift_bwd(hz, 1)) - (hy - _shift_bwd(hy, 2))) / dx
    curl_y = ((hx - _shift_bwd(hx, 2)) - (hz - _shift_bwd(hz, 0))) / dx
    curl_z = ((hy - _shift_bwd(hy, 0)) - (hx - _shift_bwd(hx, 1))) / dx

    px_new = a * lor_state.px + b * lor_state.px_prev + c * state.ex[None]
    py_new = a * lor_state.py + b * lor_state.py_prev + c * state.ey[None]
    pz_new = a * lor_state.pz + b * lor_state.pz_prev + c * state.ez[None]

    dpx = jnp.sum(px_new - lor_state.px, axis=0)
    dpy = jnp.sum(py_new - lor_state.py, axis=0)
    dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

    ex_new = ca * state.ex + cb * curl_x - cc * dpx
    ey_new = ca * state.ey + cb * curl_y - cc * dpy
    ez_new = ca * state.ez + cb * curl_z - cc * dpz

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_lor = LorentzState(
        px=px_new, py=py_new, pz=pz_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )
    return new_fdtd, new_lor


def _update_e_local_with_dispersion(state, materials, dt, dx,
                                     debye=None, lorentz=None):
    """E update on a local slab with optional Debye/Lorentz dispersion.

    Returns (new_state, new_debye_state_or_None, new_lorentz_state_or_None).
    """
    if debye is None and lorentz is None:
        return _update_e_local(state, materials, dt, dx), None, None

    if debye is not None and lorentz is None:
        debye_coeffs, debye_state = debye
        new_state, new_debye = _update_e_debye_local(
            state, debye_coeffs, debye_state, dt, dx)
        return new_state, new_debye, None

    if lorentz is not None and debye is None:
        lorentz_coeffs, lorentz_state = lorentz
        new_state, new_lorentz = _update_e_lorentz_local(
            state, lorentz_coeffs, lorentz_state, dt, dx)
        return new_state, None, new_lorentz

    # Mixed Debye + Lorentz
    debye_coeffs, debye_state = debye
    lorentz_coeffs, lorentz_state = lorentz
    hx, hy, hz = state.hx, state.hy, state.hz

    curl_x = ((hz - _shift_bwd(hz, 1)) - (hy - _shift_bwd(hy, 2))) / dx
    curl_y = ((hx - _shift_bwd(hx, 2)) - (hz - _shift_bwd(hz, 0))) / dx
    curl_z = ((hy - _shift_bwd(hy, 0)) - (hx - _shift_bwd(hx, 1))) / dx
    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    px_l_new = (lorentz_coeffs.a * lorentz_state.px
                + lorentz_coeffs.b * lorentz_state.px_prev
                + lorentz_coeffs.c * ex_old[None])
    py_l_new = (lorentz_coeffs.a * lorentz_state.py
                + lorentz_coeffs.b * lorentz_state.py_prev
                + lorentz_coeffs.c * ey_old[None])
    pz_l_new = (lorentz_coeffs.a * lorentz_state.pz
                + lorentz_coeffs.b * lorentz_state.pz_prev
                + lorentz_coeffs.c * ez_old[None])
    dpx_l = jnp.sum(px_l_new - lorentz_state.px, axis=0)
    dpy_l = jnp.sum(py_l_new - lorentz_state.py, axis=0)
    dpz_l = jnp.sum(pz_l_new - lorentz_state.pz, axis=0)

    ca_d, cb_d, cc_d = debye_coeffs.ca, debye_coeffs.cb, debye_coeffs.cc
    alpha_d, beta_d = debye_coeffs.alpha, debye_coeffs.beta
    cc_l = lorentz_coeffs.cc

    ex_new = ca_d * ex_old + cb_d * curl_x + jnp.sum(cc_d * debye_state.px, axis=0) - cc_l * dpx_l
    ey_new = ca_d * ey_old + cb_d * curl_y + jnp.sum(cc_d * debye_state.py, axis=0) - cc_l * dpy_l
    ez_new = ca_d * ez_old + cb_d * curl_z + jnp.sum(cc_d * debye_state.pz, axis=0) - cc_l * dpz_l

    px_d_new = alpha_d * debye_state.px + beta_d * (ex_new[None] + ex_old[None])
    py_d_new = alpha_d * debye_state.py + beta_d * (ey_new[None] + ey_old[None])
    pz_d_new = alpha_d * debye_state.pz + beta_d * (ez_new[None] + ez_old[None])

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_debye_st = DebyeState(px=px_d_new, py=py_d_new, pz=pz_d_new)
    new_lor_st = LorentzState(
        px=px_l_new, py=py_l_new, pz=pz_l_new,
        px_prev=lorentz_state.px, py_prev=lorentz_state.py, pz_prev=lorentz_state.pz,
    )
    return new_fdtd, new_debye_st, new_lor_st



# ---------------------------------------------------------------------------
# Distributed CPML support (Phase 2)
# ---------------------------------------------------------------------------

def _init_cpml_distributed(grid, nx_local, n_devices):
    """Initialize CPML params and per-device state for distributed slabs.

    In 1D slab decomposition along x:
    - Y/Z CPML faces: applied on ALL devices (each owns full y/z extent).
      Psi arrays that index along x use ``nx_local`` (slab width with ghosts).
    - X CPML faces: applied only on boundary devices (device 0 for x-lo,
      device N-1 for x-hi). Psi arrays allocated on all devices but
      masked to zero on non-boundary devices via ``jnp.where``.

    Parameters
    ----------
    grid : Grid
        Full-domain grid with CPML configuration.
    nx_local : int
        Per-device slab width including ghost cells.
    n_devices : int
        Number of devices.

    Returns
    -------
    cpml_params : CPMLParams
        Shared CPML profile coefficients.
    cpml_state_stacked : dict
        Per-device CPML state arrays stacked along axis 0,
        shape ``(n_devices, ...)``.
    """
    from rfx.boundaries.cpml import _cpml_profile, _flip_profile, CPMLAxisParams, CPMLState

    kappa_max = getattr(grid, "kappa_max", None) or 1.0
    n = grid.cpml_layers
    dx = grid.dx
    electric = _cpml_profile(n, grid.dt, dx, kappa_max=kappa_max)
    magnetic_lo = _cpml_profile(n, grid.dt, dx, kappa_max=kappa_max, sample_offset=.5)
    magnetic_hi = _flip_profile(
        _cpml_profile(n, grid.dt, dx, kappa_max=kappa_max, sample_offset=-.5))
    sizes = dict(dx_x_lo=dx, dx_x_hi=dx, dx_y_lo=dx,
                 dx_y_hi=dx, dz_lo=dx, dz_hi=dx)
    params = CPMLAxisParams(
        electric, _flip_profile(electric), electric, _flip_profile(electric),
        electric, _flip_profile(electric), **sizes,
        magnetic=CPMLAxisParams(magnetic_lo, magnetic_hi, magnetic_lo, magnetic_hi,
                                magnetic_lo, magnetic_hi, **sizes))

    ny, nz = grid.ny, grid.nz

    def _zeros(dim1, dim2):
        """Zero psi array: (n_devices, n_cpml, dim1, dim2)."""
        return zeros_psi_stacked(n_devices, n, dim1, dim2)

    # X-face psi: perpendicular dims are (ny, nz) or transposed
    # Y/Z-face psi: perpendicular dims include nx_local (slab-local x)
    state_stacked = CPMLState(
        # E-field psi (12 faces)
        psi_ex_ylo=_zeros(nx_local, nz),
        psi_ex_yhi=_zeros(nx_local, nz),
        psi_ex_zlo=_zeros(nx_local, ny),
        psi_ex_zhi=_zeros(nx_local, ny),
        psi_ey_xlo=_zeros(ny, nz),
        psi_ey_xhi=_zeros(ny, nz),
        psi_ey_zlo=_zeros(ny, nx_local),
        psi_ey_zhi=_zeros(ny, nx_local),
        psi_ez_xlo=_zeros(nz, ny),
        psi_ez_xhi=_zeros(nz, ny),
        psi_ez_ylo=_zeros(nz, nx_local),
        psi_ez_yhi=_zeros(nz, nx_local),
        # H-field psi (12 faces)
        psi_hx_ylo=_zeros(nx_local, nz),
        psi_hx_yhi=_zeros(nx_local, nz),
        psi_hx_zlo=_zeros(nx_local, ny),
        psi_hx_zhi=_zeros(nx_local, ny),
        psi_hy_xlo=_zeros(ny, nz),
        psi_hy_xhi=_zeros(ny, nz),
        psi_hy_zlo=_zeros(ny, nx_local),
        psi_hy_zhi=_zeros(ny, nx_local),
        psi_hz_xlo=_zeros(nz, ny),
        psi_hz_xhi=_zeros(nz, ny),
        psi_hz_ylo=_zeros(nz, nx_local),
        psi_hz_yhi=_zeros(nz, nx_local),
    )

    return params, state_stacked


def _apply_cpml_e_distributed(
    state, cpml_params, cpml_state, n_cpml, dt, dx,
    n_devices, ghost=1, axis_name="devices", eps_r=None, pad_x: int = 0,
):
    """Apply CPML E-field correction on a distributed slab.

    Y/Z faces: applied on all devices (each owns full y/z extent).
    X faces: x-lo on device 0 only, x-hi on device N-1 only.
    Non-active faces are masked to zero via jnp.where.

    X-axis indexing accounts for ghost cells: x-lo uses
    ``[ghost:ghost+n]``, x-hi uses ``[-(ghost+pad_x+n):-(ghost+pad_x)]``.

    Parameters
    ----------
    state : FDTDState
        Per-device slab state (nx_local, ny, nz).
    cpml_params : CPMLParams
        Shared CPML profile coefficients.
    cpml_state : CPMLState
        Per-device CPML auxiliary state.
    n_cpml : int
        Number of CPML layers.
    dt, dx : float
        Timestep and cell size.
    n_devices : int
    ghost : int
        Number of ghost cells on each side of x (default 1).
    axis_name : str
    eps_r : jnp.ndarray or None
        Per-cell relative permittivity slab (nx_local+2*ghost, ny, nz),
        same ghost layout as the field slab.  When provided, the CPML
        E-coefficient is the material-aware ``dt / (eps_r * eps_0)`` so the
        absorber stays impedance-matched inside a dielectric (mirrors the
        single-device ``apply_cpml_e``).  ``None`` falls back to the vacuum
        scalar ``dt / eps_0`` (bit-identical to the pre-#205 behaviour).
    pad_x : int
        Number of alignment-pad cells appended to the high-x end of the
        last rank's slab so that ``(nx + pad_x) % n_devices == 0``
        (#623, same class as #622). The x-hi CPML window is shifted left
        by ``pad_x`` so it covers the real physical face (global node
        ``nx - 1``), matching single-device ``cpml.py``'s ``[nx-n, nx)``
        (a no-op on other ranks / when ``pad_x == 0``).
    """
    n = n_cpml
    g = ghost
    # #623: on the last rank, pad_x alignment cells sit past the real
    # x-hi face — shift the window left by pad_x.
    x_hi_edge = g + pad_x
    # Per-face E-coefficient (material-aware when eps_r is supplied).
    # Each face slice broadcasts element-wise against its correction array
    # (x faces account for the ghost offset; y/z faces have no x-ghost).
    if eps_r is not None:
        _ce = dt / (eps_r * EPS_0)  # (nx_local+2g, ny, nz)
        ce_xlo = _ce[g:g + n, :, :]
        ce_xhi = (
            _ce[-(x_hi_edge + n):-x_hi_edge, :, :]
            if x_hi_edge > 0 else _ce[-n:, :, :]
        )
        ce_ylo = _ce[:, :n, :]
        ce_yhi = _ce[:, -n:, :]
        ce_zlo = _ce[:, :, :n]
        ce_zhi = _ce[:, :, -n:]
    else:
        ce_xlo = ce_xhi = ce_ylo = ce_yhi = ce_zlo = ce_zhi = cpml_coeff_e_vacuum(dt)

    from rfx.boundaries.cpml import CPMLAxisParams
    if isinstance(cpml_params, CPMLAxisParams):
        cpml_params = cpml_params.x_lo
    b = cpml_params.b
    c = cpml_params.c
    kappa = cpml_params.kappa
    b_r = jnp.flip(b)
    c_r = jnp.flip(c)
    kappa_r = jnp.flip(kappa)

    ex = state.ex
    ey = state.ey
    ez = state.ez

    device_idx = lax.axis_index(axis_name)
    is_first = (device_idx == 0)
    is_last = (device_idx == n_devices - 1)

    # X-axis slice helpers (account for ghost offset AND alignment pad,
    # #623 — xhi shifted left by pad_x so it covers the real x-hi face)
    xlo = slice(g, g + n)          # first n real cells
    xhi = slice(-(x_hi_edge + n), -x_hi_edge) if x_hi_edge > 0 else slice(-n, None)

    # =======================================================================
    # X-axis CPML (device-conditional)
    # =======================================================================
    b_x = b[:, None, None]
    c_x = c[:, None, None]
    b_xr = b_r[:, None, None]
    c_xr = c_r[:, None, None]
    k_x = kappa[:, None, None]
    k_xr = kappa_r[:, None, None]

    # --- X-lo: Ey correction from dHz/dx (device 0 only) ---
    hz_xlo = state.hz[xlo, :, :]
    hz_shifted_xlo = _shift_bwd(state.hz, 0)[xlo, :, :]
    curl_hz_dx_xlo = (hz_xlo - hz_shifted_xlo) / dx

    new_psi_ey_xlo = b_x * cpml_state.psi_ey_xlo + c_x * curl_hz_dx_xlo
    ey_corr_xlo = -ce_xlo * new_psi_ey_xlo - ce_xlo * (1.0 / k_x - 1.0) * curl_hz_dx_xlo
    # Mask: only device 0
    ey_corr_xlo = jnp.where(is_first, ey_corr_xlo, 0.0)
    ey = ey.at[xlo, :, :].add(ey_corr_xlo)
    new_psi_ey_xlo = jnp.where(is_first, new_psi_ey_xlo, cpml_state.psi_ey_xlo)

    # --- X-hi: Ey correction from dHz/dx (device N-1 only) ---
    hz_xhi = state.hz[xhi, :, :]
    hz_shifted_xhi = _shift_bwd(state.hz, 0)[xhi, :, :]
    curl_hz_dx_xhi = (hz_xhi - hz_shifted_xhi) / dx

    new_psi_ey_xhi = b_xr * cpml_state.psi_ey_xhi + c_xr * curl_hz_dx_xhi
    ey_corr_xhi = -ce_xhi * new_psi_ey_xhi - ce_xhi * (1.0 / k_xr - 1.0) * curl_hz_dx_xhi
    ey_corr_xhi = jnp.where(is_last, ey_corr_xhi, 0.0)
    ey = ey.at[xhi, :, :].add(ey_corr_xhi)
    new_psi_ey_xhi = jnp.where(is_last, new_psi_ey_xhi, cpml_state.psi_ey_xhi)

    # --- X-lo: Ez correction from dHy/dx (device 0 only) ---
    hy_xlo = state.hy[xlo, :, :]
    hy_shifted_xlo = _shift_bwd(state.hy, 0)[xlo, :, :]
    curl_hy_dx_xlo = (hy_xlo - hy_shifted_xlo) / dx
    curl_hy_dx_xlo_t = jnp.transpose(curl_hy_dx_xlo, (0, 2, 1))

    new_psi_ez_xlo = b_x * cpml_state.psi_ez_xlo + c_x * curl_hy_dx_xlo_t
    correction_ez_xlo = jnp.transpose(new_psi_ez_xlo, (0, 2, 1))
    ez_corr_xlo = ce_xlo * correction_ez_xlo + ce_xlo * (1.0 / k_x - 1.0) * curl_hy_dx_xlo
    ez_corr_xlo = jnp.where(is_first, ez_corr_xlo, 0.0)
    ez = ez.at[xlo, :, :].add(ez_corr_xlo)
    new_psi_ez_xlo = jnp.where(is_first, new_psi_ez_xlo, cpml_state.psi_ez_xlo)

    # --- X-hi: Ez correction from dHy/dx (device N-1 only) ---
    hy_xhi = state.hy[xhi, :, :]
    hy_shifted_xhi = _shift_bwd(state.hy, 0)[xhi, :, :]
    curl_hy_dx_xhi = (hy_xhi - hy_shifted_xhi) / dx
    curl_hy_dx_xhi_t = jnp.transpose(curl_hy_dx_xhi, (0, 2, 1))

    new_psi_ez_xhi = b_xr * cpml_state.psi_ez_xhi + c_xr * curl_hy_dx_xhi_t
    correction_ez_xhi = jnp.transpose(new_psi_ez_xhi, (0, 2, 1))
    ez_corr_xhi = ce_xhi * correction_ez_xhi + ce_xhi * (1.0 / k_xr - 1.0) * curl_hy_dx_xhi
    ez_corr_xhi = jnp.where(is_last, ez_corr_xhi, 0.0)
    ez = ez.at[xhi, :, :].add(ez_corr_xhi)
    new_psi_ez_xhi = jnp.where(is_last, new_psi_ez_xhi, cpml_state.psi_ez_xhi)

    # =======================================================================
    # Y-axis CPML (all devices)
    # =======================================================================
    b_yn = b[:, None, None]
    c_yn = c[:, None, None]
    b_yrn = b_r[:, None, None]
    c_yrn = c_r[:, None, None]
    k_yn = kappa[:, None, None]
    k_yrn = kappa_r[:, None, None]

    # --- Y-lo: Ex correction from dHz/dy ---
    hz_ylo = state.hz[:, :n, :]
    hz_shifted_ylo = _shift_bwd(state.hz, 1)[:, :n, :]
    curl_hz_dy_ylo = (hz_ylo - hz_shifted_ylo) / dx
    curl_hz_dy_ylo_t = jnp.transpose(curl_hz_dy_ylo, (1, 0, 2))

    new_psi_ex_ylo = b_yn * cpml_state.psi_ex_ylo + c_yn * curl_hz_dy_ylo_t
    correction_ex_ylo = jnp.transpose(new_psi_ex_ylo, (1, 0, 2))
    ex = ex.at[:, :n, :].add(ce_ylo * correction_ex_ylo)
    kappa_corr_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hz_dy_ylo_t, (1, 0, 2))
    ex = ex.at[:, :n, :].add(ce_ylo * kappa_corr_ylo)

    # --- Y-hi: Ex correction from dHz/dy ---
    hz_yhi = state.hz[:, -n:, :]
    hz_shifted_yhi = _shift_bwd(state.hz, 1)[:, -n:, :]
    curl_hz_dy_yhi = (hz_yhi - hz_shifted_yhi) / dx
    curl_hz_dy_yhi_t = jnp.transpose(curl_hz_dy_yhi, (1, 0, 2))

    new_psi_ex_yhi = b_yrn * cpml_state.psi_ex_yhi + c_yrn * curl_hz_dy_yhi_t
    correction_ex_yhi = jnp.transpose(new_psi_ex_yhi, (1, 0, 2))
    ex = ex.at[:, -n:, :].add(ce_yhi * correction_ex_yhi)
    kappa_corr_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hz_dy_yhi_t, (1, 0, 2))
    ex = ex.at[:, -n:, :].add(ce_yhi * kappa_corr_yhi)

    # --- Y-lo: Ez correction from dHx/dy ---
    hx_ylo = state.hx[:, :n, :]
    hx_shifted_ylo = _shift_bwd(state.hx, 1)[:, :n, :]
    curl_hx_dy_ylo = (hx_ylo - hx_shifted_ylo) / dx
    curl_hx_dy_ylo_t = jnp.transpose(curl_hx_dy_ylo, (1, 2, 0))

    new_psi_ez_ylo = b_yn * cpml_state.psi_ez_ylo + c_yn * curl_hx_dy_ylo_t
    correction_ez_ylo = jnp.transpose(new_psi_ez_ylo, (2, 0, 1))
    ez = ez.at[:, :n, :].add(-ce_ylo * correction_ez_ylo)
    kappa_corr_ez_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hx_dy_ylo_t, (2, 0, 1))
    ez = ez.at[:, :n, :].add(-ce_ylo * kappa_corr_ez_ylo)

    # --- Y-hi: Ez correction from dHx/dy ---
    hx_yhi = state.hx[:, -n:, :]
    hx_shifted_yhi = _shift_bwd(state.hx, 1)[:, -n:, :]
    curl_hx_dy_yhi = (hx_yhi - hx_shifted_yhi) / dx
    curl_hx_dy_yhi_t = jnp.transpose(curl_hx_dy_yhi, (1, 2, 0))

    new_psi_ez_yhi = b_yrn * cpml_state.psi_ez_yhi + c_yrn * curl_hx_dy_yhi_t
    correction_ez_yhi = jnp.transpose(new_psi_ez_yhi, (2, 0, 1))
    ez = ez.at[:, -n:, :].add(-ce_yhi * correction_ez_yhi)
    kappa_corr_ez_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hx_dy_yhi_t, (2, 0, 1))
    ez = ez.at[:, -n:, :].add(-ce_yhi * kappa_corr_ez_yhi)

    # =======================================================================
    # Z-axis CPML (all devices)
    # =======================================================================

    # --- Z-lo: Ex correction from dHy/dz ---
    hy_zlo = state.hy[:, :, :n]
    hy_shifted_zlo = _shift_bwd(state.hy, 2)[:, :, :n]
    curl_hy_dz_zlo = (hy_zlo - hy_shifted_zlo) / dx
    curl_hy_dz_zlo_t = jnp.transpose(curl_hy_dz_zlo, (2, 0, 1))

    new_psi_ex_zlo = b_yn * cpml_state.psi_ex_zlo + c_yn * curl_hy_dz_zlo_t
    correction_ex_zlo = jnp.transpose(new_psi_ex_zlo, (1, 2, 0))
    ex = ex.at[:, :, :n].add(-ce_zlo * correction_ex_zlo)
    kappa_corr_ex_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hy_dz_zlo_t, (1, 2, 0))
    ex = ex.at[:, :, :n].add(-ce_zlo * kappa_corr_ex_zlo)

    # --- Z-hi: Ex correction from dHy/dz ---
    hy_zhi = state.hy[:, :, -n:]
    hy_shifted_zhi = _shift_bwd(state.hy, 2)[:, :, -n:]
    curl_hy_dz_zhi = (hy_zhi - hy_shifted_zhi) / dx
    curl_hy_dz_zhi_t = jnp.transpose(curl_hy_dz_zhi, (2, 0, 1))

    new_psi_ex_zhi = b_yrn * cpml_state.psi_ex_zhi + c_yrn * curl_hy_dz_zhi_t
    correction_ex_zhi = jnp.transpose(new_psi_ex_zhi, (1, 2, 0))
    ex = ex.at[:, :, -n:].add(-ce_zhi * correction_ex_zhi)
    kappa_corr_ex_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hy_dz_zhi_t, (1, 2, 0))
    ex = ex.at[:, :, -n:].add(-ce_zhi * kappa_corr_ex_zhi)

    # --- Z-lo: Ey correction from dHx/dz ---
    hx_zlo = state.hx[:, :, :n]
    hx_shifted_zlo = _shift_bwd(state.hx, 2)[:, :, :n]
    curl_hx_dz_zlo = (hx_zlo - hx_shifted_zlo) / dx
    curl_hx_dz_zlo_t = jnp.transpose(curl_hx_dz_zlo, (2, 1, 0))

    new_psi_ey_zlo = b_yn * cpml_state.psi_ey_zlo + c_yn * curl_hx_dz_zlo_t
    correction_ey_zlo = jnp.transpose(new_psi_ey_zlo, (2, 1, 0))
    ey = ey.at[:, :, :n].add(ce_zlo * correction_ey_zlo)
    kappa_corr_ey_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hx_dz_zlo_t, (2, 1, 0))
    ey = ey.at[:, :, :n].add(ce_zlo * kappa_corr_ey_zlo)

    # --- Z-hi: Ey correction from dHx/dz ---
    hx_zhi = state.hx[:, :, -n:]
    hx_shifted_zhi = _shift_bwd(state.hx, 2)[:, :, -n:]
    curl_hx_dz_zhi = (hx_zhi - hx_shifted_zhi) / dx
    curl_hx_dz_zhi_t = jnp.transpose(curl_hx_dz_zhi, (2, 1, 0))

    new_psi_ey_zhi = b_yrn * cpml_state.psi_ey_zhi + c_yrn * curl_hx_dz_zhi_t
    correction_ey_zhi = jnp.transpose(new_psi_ey_zhi, (2, 1, 0))
    ey = ey.at[:, :, -n:].add(ce_zhi * correction_ey_zhi)
    kappa_corr_ey_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hx_dz_zhi_t, (2, 1, 0))
    ey = ey.at[:, :, -n:].add(ce_zhi * kappa_corr_ey_zhi)

    state = state._replace(ex=ex, ey=ey, ez=ez)
    cpml_state = cpml_state._replace(
        # x-axis
        psi_ey_xlo=new_psi_ey_xlo,
        psi_ey_xhi=new_psi_ey_xhi,
        psi_ez_xlo=new_psi_ez_xlo,
        psi_ez_xhi=new_psi_ez_xhi,
        # y-axis
        psi_ex_ylo=new_psi_ex_ylo,
        psi_ex_yhi=new_psi_ex_yhi,
        psi_ez_ylo=new_psi_ez_ylo,
        psi_ez_yhi=new_psi_ez_yhi,
        # z-axis
        psi_ex_zlo=new_psi_ex_zlo,
        psi_ex_zhi=new_psi_ex_zhi,
        psi_ey_zlo=new_psi_ey_zlo,
        psi_ey_zhi=new_psi_ey_zhi,
    )

    return state, cpml_state


def _apply_cpml_h_distributed(
    state, cpml_params, cpml_state, n_cpml, dt, dx,
    n_devices, ghost=1, axis_name="devices", mu_r=None, pad_x: int = 0,
):
    """Apply CPML H-field correction on a distributed slab.

    Y/Z faces: applied on all devices.
    X faces: x-lo on device 0 only, x-hi on device N-1 only.

    X-axis indexing accounts for ghost cells: x-lo uses
    ``[ghost:ghost+n]``, x-hi uses
    ``[-(ghost+pad_x+n):-(ghost+pad_x)]``.

    Parameters
    ----------
    state : FDTDState
        Per-device slab state (nx_local, ny, nz).
    cpml_params : CPMLParams
        Shared CPML profile coefficients.
    cpml_state : CPMLState
        Per-device CPML auxiliary state.
    n_cpml : int
        Number of CPML layers.
    dt, dx : float
    n_devices : int
    ghost : int
        Number of ghost cells on each side of x (default 1).
    axis_name : str
    mu_r : jnp.ndarray or None
        Per-cell relative permeability slab (nx_local+2*ghost, ny, nz),
        same ghost layout as the field slab.  When provided, the CPML
        H-coefficient is the material-aware ``dt / (mu_r * mu_0)`` (mirrors
        the single-device ``apply_cpml_h``).  ``None`` falls back to the
        vacuum scalar ``dt / mu_0`` (bit-identical to pre-#205 behaviour).
    pad_x : int
        Number of alignment-pad cells appended to the high-x end of the
        last rank's slab so that ``(nx + pad_x) % n_devices == 0``
        (#623, same class as #622). The x-hi CPML window is shifted left
        by ``pad_x`` so it covers the real physical face (global node
        ``nx - 1``), matching single-device ``cpml.py``'s ``[nx-n, nx)``
        (a no-op on other ranks / when ``pad_x == 0``).
    """
    n = n_cpml
    g = ghost
    # #623: on the last rank, pad_x alignment cells sit past the real
    # x-hi face — shift the window left by pad_x.
    x_hi_edge = g + pad_x
    # Per-face H-coefficient (material-aware when mu_r is supplied).
    if mu_r is not None:
        _ch = dt / (mu_r * MU_0)  # (nx_local+2g, ny, nz)
        ch_xlo = _ch[g:g + n, :, :]
        ch_xhi = (
            _ch[-(x_hi_edge + n):-x_hi_edge, :, :]
            if x_hi_edge > 0 else _ch[-n:, :, :]
        )
        ch_ylo = _ch[:, :n, :]
        ch_yhi = _ch[:, -n:, :]
        ch_zlo = _ch[:, :, :n]
        ch_zhi = _ch[:, :, -n:]
    else:
        ch_xlo = ch_xhi = ch_ylo = ch_yhi = ch_zlo = ch_zhi = cpml_coeff_h_vacuum(dt)

    from rfx.boundaries.cpml import CPMLAxisParams
    if isinstance(cpml_params, CPMLAxisParams):
        profiles = cpml_params.magnetic if cpml_params.magnetic is not None else cpml_params
        b, c, kappa = profiles.x_lo.b, profiles.x_lo.c, profiles.x_lo.kappa
        b_r, c_r, kappa_r = profiles.x_hi.b, profiles.x_hi.c, profiles.x_hi.kappa
    else:
        b, c, kappa = cpml_params.b, cpml_params.c, cpml_params.kappa
        b_r, c_r, kappa_r = jnp.flip(b), jnp.flip(c), jnp.flip(kappa)

    hx = state.hx
    hy = state.hy
    hz = state.hz

    device_idx = lax.axis_index(axis_name)
    is_first = (device_idx == 0)
    is_last = (device_idx == n_devices - 1)

    # X-axis slice helpers (account for ghost offset AND alignment pad,
    # #623 — xhi shifted left by pad_x so it covers the real x-hi face)
    xlo = slice(g, g + n)          # first n real cells
    xhi = slice(-(x_hi_edge + n), -x_hi_edge) if x_hi_edge > 0 else slice(-n, None)

    # =======================================================================
    # X-axis CPML (device-conditional)
    # =======================================================================
    b_x = b[:, None, None]
    c_x = c[:, None, None]
    b_xr = b_r[:, None, None]
    c_xr = c_r[:, None, None]
    k_x = kappa[:, None, None]
    k_xr = kappa_r[:, None, None]

    # --- X-lo: Hy correction from dEz/dx (device 0 only) ---
    ez_xlo = state.ez[xlo, :, :]
    ez_shifted_xlo = _shift_fwd(state.ez, 0)[xlo, :, :]
    curl_ez_dx_xlo = (ez_shifted_xlo - ez_xlo) / dx

    new_psi_hy_xlo = b_x * cpml_state.psi_hy_xlo + c_x * curl_ez_dx_xlo
    hy_corr_xlo = ch_xlo * new_psi_hy_xlo + ch_xlo * (1.0 / k_x - 1.0) * curl_ez_dx_xlo
    hy_corr_xlo = jnp.where(is_first, hy_corr_xlo, 0.0)
    hy = hy.at[xlo, :, :].add(hy_corr_xlo)
    new_psi_hy_xlo = jnp.where(is_first, new_psi_hy_xlo, cpml_state.psi_hy_xlo)

    # --- X-hi: Hy correction from dEz/dx (device N-1 only) ---
    ez_xhi = state.ez[xhi, :, :]
    ez_shifted_xhi = _shift_fwd(state.ez, 0)[xhi, :, :]
    curl_ez_dx_xhi = (ez_shifted_xhi - ez_xhi) / dx

    new_psi_hy_xhi = b_xr * cpml_state.psi_hy_xhi + c_xr * curl_ez_dx_xhi
    hy_corr_xhi = ch_xhi * new_psi_hy_xhi + ch_xhi * (1.0 / k_xr - 1.0) * curl_ez_dx_xhi
    hy_corr_xhi = jnp.where(is_last, hy_corr_xhi, 0.0)
    hy = hy.at[xhi, :, :].add(hy_corr_xhi)
    new_psi_hy_xhi = jnp.where(is_last, new_psi_hy_xhi, cpml_state.psi_hy_xhi)

    # --- X-lo: Hz correction from dEy/dx (device 0 only) ---
    ey_xlo = state.ey[xlo, :, :]
    ey_shifted_xlo = _shift_fwd(state.ey, 0)[xlo, :, :]
    curl_ey_dx_xlo = (ey_shifted_xlo - ey_xlo) / dx
    curl_ey_dx_xlo_t = jnp.transpose(curl_ey_dx_xlo, (0, 2, 1))

    new_psi_hz_xlo = b_x * cpml_state.psi_hz_xlo + c_x * curl_ey_dx_xlo_t
    correction_hz_xlo = jnp.transpose(new_psi_hz_xlo, (0, 2, 1))
    hz_corr_xlo = -ch_xlo * correction_hz_xlo - ch_xlo * (1.0 / k_x - 1.0) * curl_ey_dx_xlo
    hz_corr_xlo = jnp.where(is_first, hz_corr_xlo, 0.0)
    hz = hz.at[xlo, :, :].add(hz_corr_xlo)
    new_psi_hz_xlo = jnp.where(is_first, new_psi_hz_xlo, cpml_state.psi_hz_xlo)

    # --- X-hi: Hz correction from dEy/dx (device N-1 only) ---
    ey_xhi = state.ey[xhi, :, :]
    ey_shifted_xhi = _shift_fwd(state.ey, 0)[xhi, :, :]
    curl_ey_dx_xhi = (ey_shifted_xhi - ey_xhi) / dx
    curl_ey_dx_xhi_t = jnp.transpose(curl_ey_dx_xhi, (0, 2, 1))

    new_psi_hz_xhi = b_xr * cpml_state.psi_hz_xhi + c_xr * curl_ey_dx_xhi_t
    correction_hz_xhi = jnp.transpose(new_psi_hz_xhi, (0, 2, 1))
    hz_corr_xhi = -ch_xhi * correction_hz_xhi - ch_xhi * (1.0 / k_xr - 1.0) * curl_ey_dx_xhi
    hz_corr_xhi = jnp.where(is_last, hz_corr_xhi, 0.0)
    hz = hz.at[xhi, :, :].add(hz_corr_xhi)
    new_psi_hz_xhi = jnp.where(is_last, new_psi_hz_xhi, cpml_state.psi_hz_xhi)

    # =======================================================================
    # Y-axis CPML (all devices)
    # =======================================================================
    b_yn = b[:, None, None]
    c_yn = c[:, None, None]
    b_yrn = b_r[:, None, None]
    c_yrn = c_r[:, None, None]
    k_yn = kappa[:, None, None]
    k_yrn = kappa_r[:, None, None]

    # --- Y-lo: Hx correction from dEz/dy ---
    ez_ylo = state.ez[:, :n, :]
    ez_shifted_ylo = _shift_fwd(state.ez, 1)[:, :n, :]
    curl_ez_dy_ylo = (ez_shifted_ylo - ez_ylo) / dx
    curl_ez_dy_ylo_t = jnp.transpose(curl_ez_dy_ylo, (1, 0, 2))

    new_psi_hx_ylo = b_yn * cpml_state.psi_hx_ylo + c_yn * curl_ez_dy_ylo_t
    correction_hx_ylo = jnp.transpose(new_psi_hx_ylo, (1, 0, 2))
    hx = hx.at[:, :n, :].add(-ch_ylo * correction_hx_ylo)
    kappa_corr_hx_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ez_dy_ylo_t, (1, 0, 2))
    hx = hx.at[:, :n, :].add(-ch_ylo * kappa_corr_hx_ylo)

    # --- Y-hi: Hx correction from dEz/dy ---
    ez_yhi = state.ez[:, -n:, :]
    ez_shifted_yhi = _shift_fwd(state.ez, 1)[:, -n:, :]
    curl_ez_dy_yhi = (ez_shifted_yhi - ez_yhi) / dx
    curl_ez_dy_yhi_t = jnp.transpose(curl_ez_dy_yhi, (1, 0, 2))

    new_psi_hx_yhi = b_yrn * cpml_state.psi_hx_yhi + c_yrn * curl_ez_dy_yhi_t
    correction_hx_yhi = jnp.transpose(new_psi_hx_yhi, (1, 0, 2))
    hx = hx.at[:, -n:, :].add(-ch_yhi * correction_hx_yhi)
    kappa_corr_hx_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ez_dy_yhi_t, (1, 0, 2))
    hx = hx.at[:, -n:, :].add(-ch_yhi * kappa_corr_hx_yhi)

    # --- Y-lo: Hz correction from dEx/dy ---
    ex_ylo = state.ex[:, :n, :]
    ex_shifted_ylo = _shift_fwd(state.ex, 1)[:, :n, :]
    curl_ex_dy_ylo = (ex_shifted_ylo - ex_ylo) / dx
    curl_ex_dy_ylo_t = jnp.transpose(curl_ex_dy_ylo, (1, 2, 0))

    new_psi_hz_ylo = b_yn * cpml_state.psi_hz_ylo + c_yn * curl_ex_dy_ylo_t
    correction_hz_ylo = jnp.transpose(new_psi_hz_ylo, (2, 0, 1))
    hz = hz.at[:, :n, :].add(ch_ylo * correction_hz_ylo)
    kappa_corr_hz_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ex_dy_ylo_t, (2, 0, 1))
    hz = hz.at[:, :n, :].add(ch_ylo * kappa_corr_hz_ylo)

    # --- Y-hi: Hz correction from dEx/dy ---
    ex_yhi = state.ex[:, -n:, :]
    ex_shifted_yhi = _shift_fwd(state.ex, 1)[:, -n:, :]
    curl_ex_dy_yhi = (ex_shifted_yhi - ex_yhi) / dx
    curl_ex_dy_yhi_t = jnp.transpose(curl_ex_dy_yhi, (1, 2, 0))

    new_psi_hz_yhi = b_yrn * cpml_state.psi_hz_yhi + c_yrn * curl_ex_dy_yhi_t
    correction_hz_yhi = jnp.transpose(new_psi_hz_yhi, (2, 0, 1))
    hz = hz.at[:, -n:, :].add(ch_yhi * correction_hz_yhi)
    kappa_corr_hz_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ex_dy_yhi_t, (2, 0, 1))
    hz = hz.at[:, -n:, :].add(ch_yhi * kappa_corr_hz_yhi)

    # =======================================================================
    # Z-axis CPML (all devices)
    # =======================================================================

    # --- Z-lo: Hx correction from dEy/dz ---
    ey_zlo = state.ey[:, :, :n]
    ey_shifted_zlo = _shift_fwd(state.ey, 2)[:, :, :n]
    curl_ey_dz_zlo = (ey_shifted_zlo - ey_zlo) / dx
    curl_ey_dz_zlo_t = jnp.transpose(curl_ey_dz_zlo, (2, 0, 1))

    new_psi_hx_zlo = b_yn * cpml_state.psi_hx_zlo + c_yn * curl_ey_dz_zlo_t
    correction_hx_zlo = jnp.transpose(new_psi_hx_zlo, (1, 2, 0))
    hx = hx.at[:, :, :n].add(ch_zlo * correction_hx_zlo)
    kappa_corr_hx_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ey_dz_zlo_t, (1, 2, 0))
    hx = hx.at[:, :, :n].add(ch_zlo * kappa_corr_hx_zlo)

    # --- Z-hi: Hx correction from dEy/dz ---
    ey_zhi = state.ey[:, :, -n:]
    ey_shifted_zhi = _shift_fwd(state.ey, 2)[:, :, -n:]
    curl_ey_dz_zhi = (ey_shifted_zhi - ey_zhi) / dx
    curl_ey_dz_zhi_t = jnp.transpose(curl_ey_dz_zhi, (2, 0, 1))

    new_psi_hx_zhi = b_yrn * cpml_state.psi_hx_zhi + c_yrn * curl_ey_dz_zhi_t
    correction_hx_zhi = jnp.transpose(new_psi_hx_zhi, (1, 2, 0))
    hx = hx.at[:, :, -n:].add(ch_zhi * correction_hx_zhi)
    kappa_corr_hx_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ey_dz_zhi_t, (1, 2, 0))
    hx = hx.at[:, :, -n:].add(ch_zhi * kappa_corr_hx_zhi)

    # --- Z-lo: Hy correction from dEx/dz ---
    ex_zlo = state.ex[:, :, :n]
    ex_shifted_zlo = _shift_fwd(state.ex, 2)[:, :, :n]
    curl_ex_dz_zlo = (ex_shifted_zlo - ex_zlo) / dx
    curl_ex_dz_zlo_t = jnp.transpose(curl_ex_dz_zlo, (2, 1, 0))

    new_psi_hy_zlo = b_yn * cpml_state.psi_hy_zlo + c_yn * curl_ex_dz_zlo_t
    correction_hy_zlo = jnp.transpose(new_psi_hy_zlo, (2, 1, 0))
    hy = hy.at[:, :, :n].add(-ch_zlo * correction_hy_zlo)
    kappa_corr_hy_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ex_dz_zlo_t, (2, 1, 0))
    hy = hy.at[:, :, :n].add(-ch_zlo * kappa_corr_hy_zlo)

    # --- Z-hi: Hy correction from dEx/dz ---
    ex_zhi = state.ex[:, :, -n:]
    ex_shifted_zhi = _shift_fwd(state.ex, 2)[:, :, -n:]
    curl_ex_dz_zhi = (ex_shifted_zhi - ex_zhi) / dx
    curl_ex_dz_zhi_t = jnp.transpose(curl_ex_dz_zhi, (2, 1, 0))

    new_psi_hy_zhi = b_yrn * cpml_state.psi_hy_zhi + c_yrn * curl_ex_dz_zhi_t
    correction_hy_zhi = jnp.transpose(new_psi_hy_zhi, (2, 1, 0))
    hy = hy.at[:, :, -n:].add(-ch_zhi * correction_hy_zhi)
    kappa_corr_hy_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ex_dz_zhi_t, (2, 1, 0))
    hy = hy.at[:, :, -n:].add(-ch_zhi * kappa_corr_hy_zhi)

    state = state._replace(hx=hx, hy=hy, hz=hz)
    cpml_state = cpml_state._replace(
        # x-axis
        psi_hy_xlo=new_psi_hy_xlo,
        psi_hy_xhi=new_psi_hy_xhi,
        psi_hz_xlo=new_psi_hz_xlo,
        psi_hz_xhi=new_psi_hz_xhi,
        # y-axis
        psi_hx_ylo=new_psi_hx_ylo,
        psi_hx_yhi=new_psi_hx_yhi,
        psi_hz_ylo=new_psi_hz_ylo,
        psi_hz_yhi=new_psi_hz_yhi,
        # z-axis
        psi_hx_zlo=new_psi_hx_zlo,
        psi_hx_zhi=new_psi_hx_zhi,
        psi_hy_zlo=new_psi_hy_zlo,
        psi_hy_zhi=new_psi_hy_zhi,
    )

    return state, cpml_state

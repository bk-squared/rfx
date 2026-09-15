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

from rfx.core.yee import EPS_0, MU_0, FDTDState

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
    "shard_stacked",
    "shard_stacked_poles",
    "shard_stacked_psi",
    "inject_sources_shmap",
    "sample_probes_shmap",
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

    **This function does not encode a hook point.** The two runners call
    it at different places in their step bodies: ``distributed_v2`` after
    the E ghost exchange and BEFORE source injection
    (``step_fn_pec`` step 5), ``distributed_nu`` AFTER source injection
    (step 7). That ordering divergence is inventory §3.2 / leg 5
    territory -- a physics question with its own gate, deliberately left
    untouched here. Callers own their ordering; this function only
    applies the faces.
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


def sample_probes_shmap(st, mesh, n_prb, prb_local_specs, prb_device_ids):
    """Sample probes on their owning devices, then sum across devices.

    Mirror of :func:`inject_sources_shmap` on the read side: every device
    reads at the same local index, masks with ``jnp.where`` on device
    identity, and ``lax.psum`` over ``"x"`` leaves exactly the owner's
    value.  ``out_specs=P()`` because the psum result is replicated.

    Returns an empty ``float32`` vector when there are no probes, which is
    what keeps the caller's scan carry shape stable.

    Extracted verbatim from
    ``distributed_nu.py::run_nonuniform_distributed_pec._sample_probes_shmap``
    and ``distributed_v2.py::run_distributed._sample_probes_shmap``.
    """
    if n_prb == 0:
        return jnp.zeros(0, dtype=jnp.float32)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"),
                  P("x"), P("x"), P("x")),
        out_specs=P(),
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
        return lax.psum(jnp.stack(samples), "x")

    return _sample(st.ex, st.ey, st.ez, st.hx, st.hy, st.hz)

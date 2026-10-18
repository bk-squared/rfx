"""Phase B: Non-uniform FDTD kernels for the shard_map distributed runner.

Lane policy (#1053, PI decision 2026-09-15). ``distributed_v2.py`` is the
distributed development trunk: ``Simulation.run(devices=...)`` dispatches
there for uniform AND non-uniform grids, and since #1065/#1068 it realizes
declared PEC volumes through the shared ``apply_pec_mask_shmap`` stage at the
#1041 ordering. THIS module is the ``forward(distributed=True)`` NU lane and
stays: it uniquely carries NU-grid CPML / Debye / Lorentz, ``checkpoint_every``,
``n_warmup``, ``emit_time_series`` and tracer-safe array inputs, none of which
v2 has (v2 raises on NU dispersion). Retiring it would require porting those,
not the two stages #1053 moved. New distributed features go to v2; kernels
both lanes need go to ``_distributed_common.py`` with an ``is``-identity guard
in ``tests/locks/test_runner_split_bit_identity.py``. Soft occupancy
is deliberately NOT ported to v2: ``run()``
has no ``pec_occupancy_override`` caller; revisit when v2 serves ``forward()``.

This module supplies the NU analogues of the uniform kernels used by
``rfx/runners/distributed_v2.py`` so the distributed path can accept
``NonUniformGrid`` without silently dropping the profile. The uniform
kernels in ``distributed.py`` remain the reference baseline and must
not be modified from this module.

Scope (Phase B minimal):
- x-axis shard only (1D slab decomposition).
- Global grading ratio <= 5:1 (shared single dt).
- x-axis CPML cells are uniform (guaranteed by make_nonuniform_grid
  boundary padding).
- TFSF single-device only (enforced upstream).
- Dispersive (Debye/Lorentz) E on NU distributed is NOT implemented
  here. The public entry point in ``distributed_v2`` falls back when
  dispersion is active.

Key helper: ``_build_sharded_inv_dx_arrays`` builds the x-axis
inverse-spacing arrays on the global padded profile (E update: mean
``2/(d[k-1]+d[k])``; H update: local ``1/d[k]`` — CORE-C2) so the
per-device slabs stay correct across the shard boundary.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h, cpml_grid_metadata
from rfx.runners._distributed_common import (
    init_cpml_slabs, cpml_slab_frame, cpml_absorbing_faces, cpml_h_materials,
)

from rfx import _realized
from rfx.stepping import segmented_scan
from rfx.stepping.slab_wrap import record_step_kind, runner_step_kind, validate_scan_controls
from rfx.stepping.slab import (
    Slab, Fill, cut, cut_poles, cut_pole_coeffs, owned_values, forward_sharding as _forward_sharding,
)

import jax
import jax.numpy as jnp

from rfx.core.drives import drive_layout
import numpy as np
from jax import lax
from rfx.runners._rank import mesh_ranks, rank_shard_map
from jax.sharding import PartitionSpec as P

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    MU_0,
    component_h_materials,
    EPS_0,
    _shift_fwd,
    _shift_bwd,
    map_lumped,
)
from rfx.core.jax_utils import is_tracer  # noqa: F401  (Phase 2C reuse target)
from rfx.model.occupancy import publish_slab_occupancy
from rfx.runners._distributed_common import (
    _update_e_local_nu,
    _update_h_local_nu,
    magnetic_low_ghost,
    apply_pec_face_shmap,
    apply_pec_mask_shmap,
    apply_pmc_face_shmap, _slab_curl_kwargs,
    exchange_component_shmap,
    exchange_h_yee_shmap,
    exchange_e_yee_shmap,
    inject_sources_shmap,
    sample_probes_shmap,
    shard_stacked,
    shard_stacked_poles,
    slab_e_component_materials,
    slab_dispersion_coeffs,
    stage_slab_pole_coeffs,
    slab_e_materials_shmap,
    unstack_and_gather,
    update_e_nu_shmap,
    update_h_nu_shmap,
    _distributed_boundary_layers,
)


# ---------------------------------------------------------------------------
# Sharded inv-spacing arrays
# ---------------------------------------------------------------------------

def _build_sharded_inv_dx_arrays(grid, n_devices, pad_x=0):
    """Build per-device x-axis inverse-spacing slabs for the shard_map runner.

    The caller has already decided to pad the global x-extent by
    ``pad_x`` cells (to align ``nx`` on ``n_devices``).  Unlike the
    pre-#622 version, this does NOT re-derive ``inv_dx``/``inv_dx_h``
    from a padded cell-size profile — it reads ``grid.inv_dx`` /
    ``grid.inv_dx_h`` directly (the single-device reference arrays,
    which already carry the correct trailing H zero at the REAL
    boundary node ``nx - 1``, see ``nonuniform.py:219``) and appends
    ``pad_x`` zero coefficients.  The pad region is therefore
    STRUCTURALLY INERT: with every E and H coefficient zero there, no
    field evolves in the pad cells regardless of what the PEC-mask
    padding does, and the wall stays exactly where the single-device
    grid put it (#622 — the old re-derivation moved the trailing H
    zero from real node ``nx-1`` to padded index ``nx_padded-1``,
    leaving the real boundary cell's H term live).

    Parameters
    ----------
    grid : NonUniformGrid
    n_devices : int
    pad_x : int
        Number of inert cells appended to the high-x end of the domain
        so that ``(nx + pad_x) % n_devices == 0``.

    Returns
    -------
    inv_dx_e_global : (nx_padded,) np.ndarray
        E-update inverse spacing — mean ``2/(d[k-1]+d[k])`` (leading
        ``1/d[0]``), zero in the pad region. Consumed by
        ``_update_e_local_nu``.
    inv_dx_h_global : (nx_padded,) np.ndarray
        H-update inverse spacing — local ``1/d[k]`` (zero at real
        boundary node ``nx-1`` AND throughout the pad region).
        Consumed by ``_update_h_local_nu``.
    dx_padded : (nx_padded,) np.ndarray
        The cell-size profile (float32), pad region filled with the
        boundary cell size for diagnostics (e.g. CPML dx_boundary) —
        this is geometry bookkeeping only; it does NOT feed the E/H
        inverse-spacing arrays above.
    """
    dx_arr = np.asarray(grid.dx_arr, dtype=np.float64)
    inv_dx_e = np.asarray(grid.inv_dx, dtype=np.float64)
    inv_dx_h = np.asarray(grid.inv_dx_h, dtype=np.float64)
    if pad_x > 0:
        dx_arr = np.concatenate(
            [dx_arr, np.full(pad_x, float(dx_arr[-1]))]
        )
        inv_dx_e = np.concatenate([inv_dx_e, np.zeros(pad_x, dtype=np.float64)])
        inv_dx_h = np.concatenate([inv_dx_h, np.zeros(pad_x, dtype=np.float64)])
    nx = inv_dx_e.shape[0]
    if nx % n_devices != 0:
        raise ValueError(
            f"After padding nx={nx} is not divisible by n_devices={n_devices}"
        )

    return (
        inv_dx_e.astype(np.float32),
        inv_dx_h.astype(np.float32),
        dx_arr.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Local NU update kernels (operate on per-device slab including ghosts)
# ---------------------------------------------------------------------------
#
# #1038 leg 4 (prerequisite): ``_update_h_local_nu`` and ``_update_e_local_nu``
# moved VERBATIM to ``rfx/runners/_distributed_common.py`` and are re-imported
# at the top of this module, so ``rfx.runners.distributed_nu._update_h_local_nu``
# and ``..._update_e_local_nu`` keep resolving for every existing importer.
# They had to move because leg 4's shared NU shard wrappers call them and this
# module sits ABOVE ``_distributed_common`` in the runner DAG -- the shared
# module may not import from here. See that module for the full rationale.


# ---------------------------------------------------------------------------
# Sharded NU grid metadata — Phase 2A
# ---------------------------------------------------------------------------

from typing import NamedTuple as _NamedTuple  # noqa: E402


class ShardedNUGrid(_NamedTuple):
    """Metadata describing a non-uniform grid that has been sliced into
    x-axis slabs for the shard_map distributed runner.

    **Coordinate mapping convention** (used by Phase 3 probe/source routing):

    Physical positions are always resolved on the *full-domain*
    ``NonUniformGrid`` first via ``position_to_index(grid, pos)`` which
    returns a global triple ``(i_global, j, k)``.  The x-index is then
    mapped to a rank and a local index:

        rank      = i_global // nx_per_rank
        local_i   = (i_global % nx_per_rank) + ghost_width

    where ``nx_per_rank = nx_local_real`` (the per-rank real cell count,
    *not* the padded/ghost count) and ``ghost_width`` is the ghost cell
    offset stored in this object.  No per-rank physical coordinate system
    is introduced — the global cumulative x-positions are the only
    reference frame.

    Fields
    ------
    nx : int
        Original (unpadded) global x cell count.
    ny : int
        Global y cell count (unchanged by sharding).
    nz : int
        Global z cell count (unchanged by sharding).
    n_devices : int
        Number of ranks / devices.
    nx_padded : int
        Global x count after PEC padding so ``nx_padded % n_devices == 0``.
    pad_x : int
        Number of PEC cells appended at the high-x end
        (``nx_padded - nx``).
    nx_per_rank : int
        Real cells per rank (``nx_padded // n_devices``).
    nx_local : int
        Per-rank cell count including ghost cells
        (``nx_per_rank + 2 * ghost_width``).
    ghost_width : int
        Number of ghost cells on each side of a rank's slab (always 1).
    cpml_layers : int
        CPML layer count from the source grid (replicated, same on every rank).
    dt : float
        Global shared timestep (same on every rank; not recomputed).
    inv_dx_global : np.ndarray  shape (nx_padded,)
        Cell-local inverse x-spacings for the padded full domain.
    inv_dx_h_global : np.ndarray  shape (nx_padded,)
        Mean-spacing inverse x-spacings (seam-aware) for the padded full domain.
    dx_padded : np.ndarray  shape (nx_padded,)
        Padded cell-size profile (float32); useful for diagnostics.
    inv_dy : np.ndarray  shape (ny,)
        Replicated y inverse spacings (every rank receives the full array).
    inv_dy_h : np.ndarray  shape (ny,)
        Replicated y mean-spacing inverse spacings.
    inv_dz : np.ndarray  shape (nz,)
        Replicated z inverse spacings.
    inv_dz_h : np.ndarray  shape (nz,)
        Replicated z mean-spacing inverse spacings.
    rank_has_high_x_pad : int
        Index of the rank that owns the high-x PEC padding cells
        (always ``n_devices - 1``; stored for Phase 3 trim logic).
    nx_trim : int
        Number of padded cells that must be trimmed from the high-x rank's
        slab when assembling the full-domain result (equals ``pad_x``).
    x_starts : tuple[int, ...]
        Global x start index (inclusive) of the real cells for each rank.
    x_stops : tuple[int, ...]
        Global x stop index (exclusive) of the real cells for each rank.
    """

    nx: int
    ny: int
    nz: int
    n_devices: int
    nx_padded: int
    pad_x: int
    nx_per_rank: int
    nx_local: int
    ghost_width: int
    cpml_layers: int
    dt: float
    inv_dx_global: object   # np.ndarray (nx_padded,) float32
    inv_dx_h_global: object  # np.ndarray (nx_padded,) float32
    dx_padded: object        # np.ndarray (nx_padded,) float32
    inv_dy: object           # np.ndarray (ny,) float32
    inv_dy_h: object         # np.ndarray (ny,) float32
    inv_dz: object           # np.ndarray (nz,) float32
    inv_dz_h: object         # np.ndarray (nz,) float32
    rank_has_high_x_pad: int
    nx_trim: int
    x_starts: tuple
    x_stops: tuple
    face_layers: dict[str, int] | None = None
    dy_cells: object = None
    dz_cells: object = None
    e_cell_sizes: object = None
    layout: Slab | None = None
    global_grid: object = None
    step_kind: str | None = None

    select_step = record_step_kind  # Record the runner's choice before any slab operands are staged.


def split_1d_with_ghost(arr: "np.ndarray", n_devices: int, nx_per: int,
                        nx_local: int, ghost: int,
                        pad_value: float) -> "np.ndarray":
    return np.asarray(cut(arr, Slab(n_devices * nx_per, n_devices, ghost), Fill(pad_value, pad_value), mesh=None, host=True))


def build_sharded_nu_grid(
    grid,
    n_devices: int,
    exchange_interval: int = 1,
) -> ShardedNUGrid:
    """Build a :class:`ShardedNUGrid` from a full-domain :class:`NonUniformGrid`.

    This is the Phase 2A metadata-only helper.  It does **not** touch
    JAX device placement or shard_map; callers (e.g. the Phase 2B scan
    body) are responsible for calling ``jax.device_put`` on the returned
    arrays.

    Parameters
    ----------
    grid : NonUniformGrid
        Full-domain non-uniform grid produced by ``make_nonuniform_grid``.
    n_devices : int
        Number of ranks / devices for the x-slab decomposition.
    exchange_interval : int
        Ghost exchange interval.  Currently only ``exchange_interval == 1``
        is supported (one exchange per FDTD step).  The parameter is
        accepted for forward-compatibility with Phase 2E batched exchange.
        Ghost width is always ``1 * exchange_interval`` cells, so passing
        a larger value will increase ``ghost_width`` accordingly if support
        is added in a later phase.

    Returns
    -------
    ShardedNUGrid
        Immutable metadata object.  All numpy arrays are float32 and live
        on the host (CPU) at this stage.

    Notes
    -----
    **Coordinate mapping convention** (important for Phase 3):

    Probe and source physical positions must be converted to
    ``(i_global, j, k)`` using ``position_to_index(grid, pos)`` on the
    *full-domain* grid **before** sharding.  The resulting global ``i``
    is then mapped to a (rank, local_i) pair as::

        rank    = i_global // sharded.nx_per_rank
        local_i = (i_global % sharded.nx_per_rank) + sharded.ghost_width

    No per-rank physical coordinate system should be created.
    """
    if exchange_interval != 1:
        raise NotImplementedError(
            "exchange_interval > 1 is refused (a one-cell ghost layer with skipped exchanges grows exponentially in a lossless box; see tests/fixtures/distributed_exchange_interval); "
            "only exchange_interval=1 is supported in Phase 2A."
        )

    ghost = exchange_interval  # ghost_width = exchange_interval cells

    nx, ny, nz = grid.nx, grid.ny, grid.nz

    layout = Slab(nx, n_devices, ghost)
    pad_x, nx_padded = layout.pad_x, layout.nx_padded
    nx_per, nx_local = layout.nx_per_rank, layout.nx_local

    # Build padded inverse-spacing arrays (reuses existing Phase B helper)
    inv_dx_global, inv_dx_h_global, dx_padded = _build_sharded_inv_dx_arrays(
        grid, n_devices, pad_x=pad_x
    )

    # Replicate y/z inverse spacings (unchanged by x-sharding)
    inv_dy = np.asarray(grid.inv_dy, dtype=np.float32)
    inv_dy_h = np.asarray(grid.inv_dy_h, dtype=np.float32)
    inv_dz = np.asarray(grid.inv_dz, dtype=np.float32)
    inv_dz_h = np.asarray(grid.inv_dz_h, dtype=np.float32)

    # Rank x-range bookkeeping
    x_starts, x_stops = zip(*layout.owned)

    from rfx.model.materials import electric_cell_sizes
    return ShardedNUGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        n_devices=n_devices,
        nx_padded=nx_padded,
        pad_x=pad_x,
        nx_per_rank=nx_per,
        nx_local=nx_local,
        ghost_width=ghost,
        cpml_layers=grid.cpml_layers,
        dt=float(grid.dt),
        inv_dx_global=inv_dx_global,
        inv_dx_h_global=inv_dx_h_global,
        dx_padded=dx_padded,
        inv_dy=inv_dy,
        inv_dy_h=inv_dy_h,
        inv_dz=inv_dz,
        inv_dz_h=inv_dz_h,
        rank_has_high_x_pad=n_devices - 1,
        nx_trim=pad_x,
        x_starts=x_starts,
        x_stops=x_stops,
        face_layers=grid.face_layers,
        dy_cells=np.asarray(grid.dy_arr),
        dz_cells=np.asarray(grid.dz),
        e_cell_sizes=electric_cell_sizes(grid),
        layout=layout,
        global_grid=grid,
    )


# ---------------------------------------------------------------------------
# Phase 2B: hard-PEC + ghost-exchange sharded NU scan body
# ---------------------------------------------------------------------------

def shard_pec_mask_x_slab(global_mask, sharded_grid: ShardedNUGrid):
    return None if global_mask is None else cut(global_mask, Slab.from_grid(sharded_grid), "pec_cell", mesh=None).reshape(-1, *global_mask.shape[1:])


# Ghost-exchange one field component on an x-sharded array.
#
# The NU scan body uses the same ghost-exchange contract as the uniform
# distributed runner; the body lived here verbatim and is now the shared
# ``exchange_component_shmap`` in ``_distributed_common.py``. Kept as a
# module-local alias for legacy callers and the shared-kernel identity lock.
_exchange_component_nu_shmap = exchange_component_shmap


# Module-local aliases of the shared exchanges (placement contract on exchange_e_yee_shmap);
# tests patch these names on this module.
_exchange_h_ghosts_nu = exchange_h_yee_shmap
_exchange_e_ghosts_nu = exchange_e_yee_shmap


# #1038 leg 3. ``_apply_pec_face_nu_shmap``'s body now lives as
# ``apply_pec_face_shmap`` in ``_distributed_common.py``; it was a copy of
# ``distributed_v2.py::_apply_pec_shmap`` differing only in the parameter
# spelling (renamed in the preceding commit) and three comments. Kept as a
# module-local alias so the existing call sites are unchanged.
_apply_pec_face_nu_shmap = apply_pec_face_shmap


# #1038 leg 3. ``_apply_pmc_face_nu_shmap``'s body now lives as
# ``apply_pmc_face_shmap`` in ``_distributed_common.py``; once the slab-length
# parameter was spelled the same on both runners the two kernels were byte
# identical. Kept as a module-local alias so the existing call sites are
# unchanged.
_apply_pmc_face_nu_shmap = apply_pmc_face_shmap


# #1053 leg 2. ``_apply_pec_mask_nu_shmap``'s body now lives as
# ``apply_pec_mask_shmap`` in ``_distributed_common.py``, so ``distributed_v2``
# can apply the same realized-PEC rule from the same function object. The
# kernel was already runner-agnostic, so nothing but the def name changed.
# Kept as a module-local alias: this lane's own call site (stage 8 of
# ``run_nonuniform_distributed_pec.step_fn``) and the #1038 lock fixture
# ``distributed_nu_pec_mask_seam`` both import it under this name.
_apply_pec_mask_nu_shmap = apply_pec_mask_shmap


# ---------------------------------------------------------------------------
# Phase 2C: CPML on x-slabs (rank-conditional x-faces, sliced y/z faces)
# ---------------------------------------------------------------------------

def init_cpml_for_sharded_nu(sharded_grid: ShardedNUGrid, n_devices: int,
                             *, kappa_max=None, pec_faces=None,
                             pmc_faces=None, mesh=None):
    """Use the global realized grid's profiles and field-order psi buffers."""
    params, state = init_cpml_slabs(
        sharded_grid.global_grid, sharded_grid.nx_local, n_devices, mesh=mesh,
        kappa_max=kappa_max, pec_faces=pec_faces, pmc_faces=pmc_faces)
    return params, state if sharded_grid.cpml_layers > 0 else None


def shard_cpml_state_x_slab(cpml_state_stacked, sharded_grid: ShardedNUGrid,
                            mesh):
    """Merge rank into the field-order x dimension, then shard P("x")."""
    from jax.sharding import NamedSharding
    if cpml_state_stacked is None:
        return None
    shd = NamedSharding(mesh, P("x"))
    return jax.tree.map(lambda value: shard_stacked(value, shd), cpml_state_stacked)


# ---------------------------------------------------------------------------
# Phase 2D: Debye / Lorentz x-slab sharding helpers
# ---------------------------------------------------------------------------
#
# These mirror ``shard_cpml_state_x_slab``: take a full-domain
# ``DebyeCoeffs`` / ``DebyeState`` / ``LorentzCoeffs`` / ``LorentzState``
# (as produced by ``init_debye`` / ``init_lorentz`` on the unsharded
# grid), split them along x with ghost cells using the canonical helpers
# from ``rfx.runners.distributed``, and then merge ``(n_devices, n_poles,
# nx_local, ...)`` -> ``(n_devices * n_poles, nx_local, ...)`` and place
# on the mesh with ``P("x")``.  Inside ``shard_map`` each device sees
# ``(n_poles, nx_local, ny, nz)`` for pole-axis arrays, exactly the
# layout the local helper expects.
#
def _concrete_on_mesh(arr, sharding):
    """Place concrete data on addressable devices only (also for replication)."""
    if isinstance(arr, jax.core.Tracer):
        return jax.device_put(arr, sharding)
    if isinstance(arr, jax.Array) and arr.sharding == sharding:
        return arr
    host = np.asarray(arr)
    return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])




def is_forward_sharded_override(arr, sharded_grid, mesh):
    """Validate an override's global layout without fetching any device data."""
    sharding = _forward_sharding(arr)
    if sharding is None:
        return False
    spec = tuple(sharding.spec) + (None,) * (3 - len(sharding.spec))
    if spec != ("x", None, None):
        # Any other placement (replicated, or split along y/z or another
        # mesh axis) is a local whole-domain array, as it was before.
        return False
    expected = (sharded_grid.nx_padded, sharded_grid.ny, sharded_grid.nz)
    if (arr.shape != expected
            or sharding.mesh.devices.shape != mesh.devices.shape
            or not np.array_equal(sharding.mesh.devices, mesh.devices)):
        raise ValueError(
            f"sharded forward override must have shape {expected}, P('x'), "
            "and the same ordered devices as forward(devices=...).")
    return True


def stage_sharded_forward_override(arr, sharded_grid, mesh, pad_value=0.0):
    return cut(arr, Slab.from_grid(sharded_grid), Fill(pad_value, pad_value), mesh=mesh)


def stage_concrete_forward_array(arr, sharded_grid, mesh, pad_value=0.0,
                                 ghost_value=None):
    return cut(arr, Slab.from_grid(sharded_grid), Fill(pad_value, pad_value if ghost_value is None else ghost_value), mesh=mesh)


def stage_forward_array_x_slab(arr, sharded_grid, mesh, pad_value=0.0):
    return cut(arr, Slab.from_grid(sharded_grid), Fill(pad_value, pad_value), mesh=mesh)


def shard_debye_coeffs_x_slab(debye_coeffs, sharded_grid: ShardedNUGrid, mesh):
    return cut_pole_coeffs(debye_coeffs, Slab.from_grid(sharded_grid), "debye",
                           dt=sharded_grid.dt, mesh=mesh)


def shard_debye_state_x_slab(debye_state, sharded_grid: ShardedNUGrid, mesh):
    return cut_poles(debye_state, Slab.from_grid(sharded_grid), "debye_state", mesh=mesh)


def shard_lorentz_coeffs_x_slab(lorentz_coeffs, sharded_grid: ShardedNUGrid, mesh):
    return cut_pole_coeffs(lorentz_coeffs, Slab.from_grid(sharded_grid), "lorentz",
                           dt=sharded_grid.dt, mesh=mesh)


def shard_lorentz_state_x_slab(lorentz_state, sharded_grid: ShardedNUGrid, mesh):
    return cut_poles(lorentz_state, Slab.from_grid(sharded_grid), "lorentz_state", mesh=mesh)


def _update_e_dispersive_local_nu(
    state: FDTDState,
    materials: MaterialArrays,
    dt: float,
    inv_dx: jnp.ndarray,
    inv_dy: jnp.ndarray,
    inv_dz: jnp.ndarray,
    *,
    debye=None,
    lorentz=None,
    e_old: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None, boundary=None, frame=None, owned=None,
):
    """Per-rank slab-aware NU dispersive E update.

    Thin wrapper around :func:`rfx.nonuniform._update_e_nu_dispersive`
    so the shard_map call site stays uncluttered and so the ADE
    Ordering Contract has a single chokepoint.

    Phase 2D ADE Ordering Contract: ``e_old`` MUST be the snapshot of
    ``state.ex/ey/ez`` taken BEFORE the H ghost exchange.  See the
    docstring of :func:`run_nonuniform_distributed_pec` for the full
    sequence; passing ``e_old=None`` here would let the helper fall
    back to ``state.ex/ey/ez``, but on the distributed path that is
    NEVER what the caller wants — pass the snapshot explicitly.
    """
    from rfx.stepping.graded_setup import _update_e_nu_dispersive

    return _update_e_nu_dispersive(
        state, materials, dt, inv_dx, inv_dy, inv_dz,
        debye=debye, lorentz=lorentz, e_old=e_old, boundary=boundary, frame=frame, owned=owned,
    )






from rfx.model.electric_metrics import material_drive_scales as _material_drive_scales, SlabElectricMetrics, stage_forward_dispersion_x_slab
material_drive_scales = partial(_material_drive_scales, reduce_devices=False)


def run_nonuniform_distributed_pec(
    sharded_grid: ShardedNUGrid,
    sharded_materials: MaterialArrays,
    sharded_pec_mask,
    n_steps: int,
    sources: list = None,
    probes: list = None,
    *,
    n_devices: int,
    exchange_interval: int = 1,
    debye=None,
    lorentz=None,
    devices=None,
    cpml_params=None,
    cpml_state=None,
    sharded_pec_occupancy=None,
    checkpoint_every: int | None = None,
    n_warmup: int = 0,
    emit_time_series: bool = True,
    pmc_faces: frozenset = frozenset(),
    gather_final_state: bool = True,
    material_drive=None,
    port_plan=None,
    prepare_only=False,
) -> dict:
    """Phase 2B/2C/2D sharded NU scan body — hard PEC, ghost exchange,
    optional CPML, and optional Debye/Lorentz dispersion on x-slabs.

    Phase 2B contract (V3 plan lines 621-632): H/E NU updates on x-slabs
    with ghost-cell exchange at slab seams, domain-face PEC, and
    geometry/override-union PEC mask zeroing.  Phase 2C extension
    (V3 plan lines 633-657): when ``cpml_params`` and ``cpml_state``
    are supplied, hook ``apply_cpml_h`` after the H update and
    ``apply_cpml_e`` after the E update.  X-face CPML is rank-conditional
    (rank 0 owns x-lo, rank N-1 owns x-hi); y- and z-face CPML run on
    every rank with psi arrays sliced over the local x extent.

    Phase 2D extension (V3 plan lines 659-720): when ``debye`` and/or
    ``lorentz`` are supplied, the per-rank E update dispatches into
    :func:`_update_e_dispersive_local_nu` (which wraps the canonical
    NU dispersive helper :func:`rfx.nonuniform._update_e_nu_dispersive`).
    The dispatch shape mirrors the uniform path's
    ``_update_e_with_optional_dispersion`` (no-disp / Debye-only /
    Lorentz-only / mixed); mixed Debye+Lorentz is fully supported (DP4
    explicit requirement).  See the **ADE Ordering Contract** below.

    Per-rank scan body ordering (mirrors single-device
    ``rfx/nonuniform.py::run_nonuniform`` plus Phase 2B's seam-aware
    ghost exchange, Phase 2C's CPML, and Phase 2D's ADE)::

        0. Snapshot ex_old/ey_old/ez_old (Phase 2D ADE Ordering Contract)
        1. H update (update_h_nu)                    via shard_map
        2. apply_cpml_h (Phase 2C, NU + slab-aware)  via shard_map  [if CPML]
        3. Ghost exchange of H                       via lax.ppermute
        4a. E update (no-dispersion path)            via shard_map  [if no Debye/Lorentz]
        4b. E update + ADE polarisation update       via shard_map  [if Debye/Lorentz]
            (uses snapshotted ex_old/ey_old/ez_old, not post-exchange E)
        5. apply_cpml_e (Phase 2C, NU + slab-aware)  via shard_map  [if CPML]
        5b. Multiply published soft-PEC factors                    [if occupancy]
        6. Source injection (rank-conditional)       via shard_map
        7. apply_pec on physical domain faces        via shard_map
        8. apply_pec_mask (geometry + override)      via shard_map
        9. Ghost exchange of E                       via lax.ppermute
       10. Probe accumulation (rank-conditional sum) via lax.psum

    Exchange E after all electric writes. PMC canonicalization follows CPML H and changes only the global
    high ghost, which no exchange sends. Only Hy/Hz left and Ey/Ez right ghosts are refreshed from owners.
    Other field, CPML and dispersion ghost entries are locally updated
    scratch values, can grow, and are never read by real cells. Exclude them
    from physical energy calculations; ``forward()`` discards this state.
    Measured on the #931 seam fixtures (16x8x8, 2 ranks, 30 steps, one
    PEC cell at ``(nx_per_rank, ny//2, nz//2)`` = rank 1's first real
    cell): with the exchange BEFORE the PEC stages, rank 0's right ghost
    kept the un-zeroed ``Ey``/``Ez`` on the seam plane that rank 1 zeroed
    a stage later, rank 0's next H update at its last real cell read that
    stale plane, and the Class B final-step error was 2.107e-01 against a
    5e-5 gate (hard mask and soft occupancy alike).  With the exchange
    after the PEC stages the seam row reads 7.8e-08 at the final step and
    the whole probe series agrees to 3 float32 ulp (max |diff| 1.5 on a
    4.0e6 peak) — the same rounding-level residual the away-from-seam
    bodies show before and after.  A body whose seam-plane edges
    are owned by the LEFT rank (or a body straddling the seam) was never
    visibly wrong, because the corrupted H then sat inside the body and
    fed only PEC edges — which is why the old three-cell fixtures passed.

    ADE Ordering Contract (Phase 2D — V3 plan lines 679-698)
    -------------------------------------------------------
    The trapezoidal Debye polarisation update is::

        px_new = alpha * debye_state.px + beta * (ex_new + ex_old)

    On a distributed slab decomposition, ``ex_old`` at a slab-boundary
    cell could be silently overwritten by a ghost exchange that brings
    in the neighbour rank's new ``ex`` BEFORE the ADE update runs —
    that would corrupt the polarisation update at the seam.  This
    runner enforces the following ordering:

      1. **Snapshot** ``ex_old, ey_old, ez_old`` from the carry at the
         very top of the scan body, BEFORE any ghost exchange or
         field update.
      2. Ghost-exchange H.
      3. Compute ``ex_new, ey_new, ez_new`` using exchanged H.  When
         dispersion is active, the ADE polarisation update inside
         ``_update_e_nu_dispersive`` receives the snapshotted
         ``e_old`` tuple via the ``e_old=`` keyword — NOT
         ``state.ex/ey/ez`` (which would be post-exchange E from the
         previous step's ninth stage).
      4. Continue with CPML-E, sources, PEC, and LAST the E ghost
         exchange (see the ordering note above).

    The snapshot lives only as a Python local within ``step_fn``; it
    costs nothing to carry (just three ``jnp.array`` references), and
    every dispersive E call inside the scan body must thread it
    through.  The seam-isolated unit test
    (``test_distributed_debye_seam_ade_ordering_isolated``) is the
    concrete merge gate for this contract.

    Parameters
    ----------
    sharded_grid : ShardedNUGrid
        Output of :func:`build_sharded_nu_grid`.
    sharded_materials : MaterialArrays
        x-slab sharded ``(eps_r, sigma, mu_r)``; each component already
        placed on the mesh with ``P("x")`` sharding.  Layout:
        ``(n_devices * nx_local, ny, nz)``.
    sharded_pec_mask : jnp.ndarray bool or None
        x-slab sharded PEC mask, same layout as a material array, or
        ``None`` to skip mask zeroing.  Use :func:`shard_pec_mask_x_slab`.
    n_steps : int
        Total FDTD steps.
    sources : list of SourceSpec, optional
        Each ``SourceSpec.i`` is a global x-index (full-domain).  Routing
        to per-rank handlers uses
        ``rank = i // nx_per_rank,
         local_i = (i % nx_per_rank) + ghost_width``.
    probes : list of ProbeSpec, optional
        Same routing as sources.
    n_devices : int
        Required.  Must match ``sharded_grid.n_devices``.
    exchange_interval : int, optional
        Reserved for Phase 2E batched exchange; only ``1`` is supported.
    debye : (DebyeCoeffs, DebyeState) tuple or None, optional
        Phase 2D Debye dispersion, already x-slab-sharded. Forward stages
        fixed pole terms and zero carry with :func:`stage_forward_dispersion_x_slab`;
        prebuilt bundles from :func:`shard_debye_coeffs_x_slab` /
        :func:`shard_debye_state_x_slab` are also accepted. The E coefficients
        are formed from the shared component means inside the scan.
        ``None`` selects the non-dispersive path.
    lorentz : (LorentzCoeffs, LorentzState) tuple or None, optional
        Phase 2D Lorentz/Drude dispersion, already x-slab-sharded, with
        fixed pole terms or a prebuilt bundle as for ``debye``. ``None`` selects the
        non-dispersive path.  When BOTH ``debye`` and ``lorentz`` are
        non-None, the runner takes the mixed Debye+Lorentz branch
        (V3 mandatory; DP4 makes mixed dispersive scope required).
    devices : list of jax.Device, optional
        If ``None``, uses ``jax.devices()[:n_devices]``.
    cpml_params : CPMLAxisParams or None, optional
        Per-axis CPML profiles for the x-, y-, and z-faces.  Pass the
        first return value of :func:`init_cpml_for_sharded_nu`.  When
        ``None``, the runner takes the Phase 2B PEC-only path.
    cpml_state : CPMLState or None, optional
        Sharded CPML auxiliary state (output of
        :func:`shard_cpml_state_x_slab` applied to the second return
        value of :func:`init_cpml_for_sharded_nu`).  Required when
        ``cpml_params`` is not None.
    sharded_pec_occupancy : jnp.ndarray float or None, optional
        x-slab sharded soft-PEC occupancy field (Phase 2E).  Same layout
        as ``sharded_pec_mask`` but float-valued in ``[0, 1]``.  Use
        :func:`rfx.stepping.slab.cut` with kind ``pec_occupancy`` to build it.  Its published
        edge factor multiplies E after the E update and CPML, before source
        injection; each source series carries its own edge's factor, as in
        :func:`rfx.nonuniform.run_nonuniform`.
    checkpoint_every : int or None, optional
        Phase 2F segmented remat.  When set to a positive integer ``K``
        with ``0 < K < n_optimize``, the optimize-phase scan is
        refactored as scan-of-scan: an outer scan over
        ``ceil(n_optimize / K)`` segments wraps an inner scan over
        ``K`` per-segment timesteps, with the segment body wrapped in
        :func:`jax.checkpoint`.  This forces XLA to remat the inner
        scan during backward, so the AD tape only stores carry at
        segment boundaries (``~ sqrt(n_steps) * carry_size`` when
        ``K ~ sqrt(n_steps)``).  Forward results are bit-identical;
        gradients are exact (segmented remat preserves AD).  ``None``
        (default) takes the single-scan Phase 2A-2E path.
    n_warmup : int, optional
        Phase 2F warmup splitting (mirrors single-device :issue:`#40`).
        When ``> 0``, run the first ``n_warmup`` steps with the carry
        ``stop_gradient``'d so AD builds no tape for that transient.
        Only the trailing ``n_steps - n_warmup`` steps participate in
        reverse-mode autodiff.  Must satisfy ``n_warmup < n_steps``.
        Truncation error DEPENDS ON DISTANCE FROM THE SOURCE TO THE
        DESIGN REGION, not merely on ``n_warmup``'s value: it is
        (near-)exact while ``n_warmup`` stays below the wavefront's
        arrival time at the design region (``K_safe ~= floor(
        min_distance(source, design_region) / (C0 * dt))``) and grows
        increasingly biased (magnitude-underestimated) beyond it -- for a
        NEAR-SOURCE placement specifically, measured ~6-7% at half the
        pre-loss-window length, 58% at the loss-window boundary itself,
        exactly zero once far enough beyond it; a far-from-source
        placement stays within this repo's ~1.5% AD-vs-FD noise floor for
        every ``n_warmup <= K_safe`` (growing smoothly, not a sharp
        cliff, from <0.01% well below ``K_safe`` to ~1.2% AT ``K_safe``
        itself). See the ``n_warmup split`` comment in
        :func:`rfx.nonuniform.run_nonuniform` for both measured curves,
        the ``K_safe`` formula, and the source scripts (issue #626 part 2
        / addendum).
    emit_time_series : bool, optional
        Phase 2F.  When ``True`` (default), the runner accumulates per-
        step probe samples and returns ``time_series`` as a
        ``(n_steps, n_probes)`` array (gradient-available when
        differentiating through it).  When ``False``, no probe
        accumulation is performed and ``time_series`` in the result
        dict is ``None``; callers must drive their objective from a
        non-time-series quantity (e.g. final state, sharded DFT
        accumulator).

    Returns
    -------
    dict
        ``{"time_series": (n_steps, n_probes) ndarray or None
                          (None when ``emit_time_series=False``),
           "final_state":  FDTDState (gathered to full-domain),
           "final_state_sharded": FDTDState (sharded, in-mesh layout),
           "cpml_state_sharded": CPMLState or None (final psi arrays,
                                  sharded on x),
           "debye_state_sharded": DebyeState or None,
           "lorentz_state_sharded": LorentzState or None}``
    ``gather_final_state=False`` skips the full-domain field gather and
    returns ``None`` for ``final_state``. Forward uses this mode; the
    ``final_state_sharded`` entry is always available in the slab layout.

    material_drive : sequence of (float or None), optional
        One entry per source (#1279); ``None`` (the default) is all-None.
        ``None``: the source's ``waveform`` is its complete per-step E
        increment, already normalised (``make_current_source``). A float
        ``dV``: the ``waveform`` is the current ``I(t)`` in amperes, and
        the runner multiplies it by ``Cb/dV`` inside its jitted program,
        ``Cb`` the coefficient of the source edge built from
        ``sharded_materials`` -- the arrays its E update receives, an
        override when there is one, traced when it is -- by
        :func:`material_drive_scales`; ``(dV, 2*sum(beta)/dt)`` on a Debye edge.
    """
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(sharded_materials, lane="distributed_nu", unsupported=True)
    if n_devices != sharded_grid.n_devices:
        raise ValueError(
            f"n_devices={n_devices} != sharded_grid.n_devices="
            f"{sharded_grid.n_devices}"
        )
    if exchange_interval != 1:
        raise NotImplementedError(
            "exchange_interval > 1 is refused (see tests/fixtures/distributed_exchange_interval); only 1 supported."
        )
    # Phase 2D: Debye / Lorentz dispatch (was NotImplementedError before).
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    if use_debye:
        debye_coeffs, debye_state_init = debye
    else:
        debye_coeffs = None
        debye_state_init = None
    if use_lorentz:
        lorentz_coeffs, lorentz_state_init = lorentz
    else:
        lorentz_coeffs = None
        lorentz_state_init = None
    from rfx.boundaries.depths import distributed_wall_faces
    e_wall_faces, curl_boundary = distributed_wall_faces(sharded_grid.global_grid, pmc_faces)
    use_dispersion = use_debye or use_lorentz
    use_cpml = cpml_params is not None
    if not use_cpml:
        # The all-wall scan never initializes CPML; it still owns both ends.
        _distributed_boundary_layers(
            sharded_grid, n_devices, cpml_layers=0,
            pec_faces={f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi")},
            pmc_faces=pmc_faces,
        )
    elif pmc_faces:
        # The graded grid keeps no PMC declaration, so init_cpml_slabs cannot see these faces.
        _distributed_boundary_layers(sharded_grid, n_devices, pmc_faces=pmc_faces)
    if use_cpml and cpml_state is None:
        raise ValueError(
            "cpml_params was provided but cpml_state is None; pass the "
            "second return value of init_cpml_for_sharded_nu through "
            "shard_cpml_state_x_slab."
        )
    if use_cpml and sharded_grid.cpml_layers <= 0:
        raise ValueError(
            "cpml_params provided but sharded_grid.cpml_layers <= 0; "
            "rebuild the grid with a non-zero CPML layer count."
        )

    validate_scan_controls(n_steps, n_warmup, checkpoint_every)

    sources = list(sources) if sources is not None else []
    for source in sources:
        if source.component not in ("ex", "ey", "ez"):
            raise ValueError(f"distributed_nu: unsupported source component {source.component}")
    probes = list(probes) if probes is not None else []

    if runner_step_kind(sharded_grid, cpml=use_cpml, debye=use_debye, lorentz=use_lorentz, pmc=bool(pmc_faces), ports=port_plan is not None) == "wrapped":
        from rfx.runners._distributed_nu_wrapped import run_wrapped
        return run_wrapped(sharded_grid, sharded_materials, sharded_pec_mask,
            n_steps, sources, probes, devices=devices, occupancy=sharded_pec_occupancy,
            checkpoint_every=checkpoint_every, n_warmup=n_warmup,
            emit_time_series=emit_time_series, gather_final_state=gather_final_state,
            material_drive=material_drive, debye=debye, lorentz=lorentz, cpml=use_cpml, cpml_state=cpml_state, port_plan=port_plan,
            prepare_only=prepare_only)

    # Defer the `Mesh` import to runtime so module import remains
    # lightweight (matches distributed_v2.py's pattern).
    from jax.sharding import Mesh, NamedSharding

    if devices is None:
        devices = jax.devices()[:n_devices]
    if len(devices) < n_devices:
        raise ValueError(
            f"Only {len(devices)} JAX devices available; need {n_devices}."
        )
    mesh = Mesh(np.array(devices), axis_names=("x",))
    shd = NamedSharding(mesh, P("x"))
    rep = NamedSharding(mesh, P())

    nx_local = sharded_grid.nx_local
    nx_per = sharded_grid.nx_per_rank
    nx_real = sharded_grid.nx   # an int for the step closures (#1303)
    ghost = sharded_grid.ghost_width
    ny = sharded_grid.ny
    nz = sharded_grid.nz
    pad_x = sharded_grid.pad_x
    dt = sharded_grid.dt

    # ------------------------------------------------------------------
    # Sharded inverse-spacing arrays (1-D)
    # ------------------------------------------------------------------
    layout = Slab.from_grid(sharded_grid)
    inv_dx_sharded = cut(sharded_grid.inv_dx_global, layout, "inv_dx", mesh=mesh)
    inv_dx_h_sharded = cut(sharded_grid.inv_dx_h_global, layout, "inv_dx_h", mesh=mesh)
    inv_dy_rep = _concrete_on_mesh(sharded_grid.inv_dy, rep)
    inv_dy_h_rep = _concrete_on_mesh(sharded_grid.inv_dy_h, rep)
    inv_dz_rep = _concrete_on_mesh(sharded_grid.inv_dz, rep)
    inv_dz_h_rep = _concrete_on_mesh(sharded_grid.inv_dz_h, rep)
    # The face permeability uses original cell lengths, including the true
    # neighbour across each x split, rather than inverse-metric round trips.
    cell_sizes = (
        cut(sharded_grid.dx_padded, layout, "width", mesh=mesh),
        _concrete_on_mesh(sharded_grid.dy_cells if sharded_grid.dy_cells is not None
                          else 1 / np.where(sharded_grid.inv_dy_h > 0,
                                            sharded_grid.inv_dy_h, sharded_grid.inv_dy_h[-2]), rep),
        _concrete_on_mesh(sharded_grid.dz_cells if sharded_grid.dz_cells is not None
                          else 1 / np.where(sharded_grid.inv_dz_h > 0,
                                            sharded_grid.inv_dz_h, sharded_grid.inv_dz_h[-2]), rep))

    # ------------------------------------------------------------------
    # Initial state — zeros created directly on the slab sharding
    # ------------------------------------------------------------------
    # Building the whole padded domain and its stacked slab copies on the
    # first device set that device's setup peak (six whole-domain buffers);
    # every value is +0.0, so creating each slab in place is bit-identical.
    def _zeros():
        return jnp.zeros((n_devices * nx_local, ny, nz), dtype=jnp.float32,
                         device=shd)

    sharded_state = FDTDState(
        ex=_zeros(), ey=_zeros(), ez=_zeros(),
        hx=_zeros(), hy=_zeros(), hz=_zeros(),
        step=_concrete_on_mesh(jnp.int32(0), rep),
    )

    # ------------------------------------------------------------------
    # Source / probe routing — V3 bullet (Phase 2A coordinate convention)
    # ------------------------------------------------------------------
    src_device_ids = []
    src_local_specs = []
    for s in sources:
        dev_id = layout.owner(s.i)
        local_i = layout.local_index(s.i)
        src_device_ids.append(int(dev_id))
        src_local_specs.append((int(local_i), int(s.j), int(s.k), s.component))

    prb_device_ids = []
    prb_local_specs = []
    for p in probes:
        dev_id = layout.owner(p.i)
        local_i = layout.local_index(p.i)
        prb_device_ids.append(int(dev_id))
        prb_local_specs.append((int(local_i), int(p.j), int(p.k), p.component))

    source_nodes = tuple((owner, *spec) for owner, spec in
                         zip(src_device_ids, src_local_specs))
    n_src = len(sources)
    n_prb = len(probes)

    # #1279: material-driven sources. Their table column is the current
    # I(t); run_fn scales it by Cb/dV read from the staged materials.
    material_drive = (tuple(material_drive) if material_drive is not None
                      else (None,) * n_src)
    if len(material_drive) != n_src:
        raise ValueError(
            f"material_drive has {len(material_drive)} entries for "
            f"{n_src} sources")
    drives = []
    drive_columns = []
    for col, (s, dV) in enumerate(zip(sources, material_drive)):
        if dV is None:
            continue
        # The x-lo face's left ghost row is vacuum padding, not a cell:
        # read that slab from its first real row (see material_drive_scales).
        row0 = ghost if s.i == 0 else 0
        drives.append((src_device_ids[col], row0,
                       (src_local_specs[col][0] - row0, int(s.j), int(s.k)),
                       s.component, *(dV if isinstance(dV, tuple) else (float(dV), 0.0))))
        drive_columns.append(col)
    drives = tuple(drives)

    if n_src > 0:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    src_waveforms_rep = _concrete_on_mesh(src_waveforms, rep)

    # ------------------------------------------------------------------
    # Per-step shmap-wrapped helpers
    # ------------------------------------------------------------------
    def _update_h_shmap(st, mat, spacings, *, ranks):
        (inv_dx_sharded, inv_dy_rep, inv_dz_rep,
         inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep) = spacings[:6]
        # #1038 leg 4: body moved VERBATIM to
        # _distributed_common.update_h_nu_shmap, which distributed_v2.py's
        # `if is_nu:` branch also calls now. The eight locals this closure read
        # are passed explicitly; the call sites below did not move.
        return update_h_nu_shmap(
            st, mat, mesh, dt,
            inv_dx_sharded, inv_dy_rep, inv_dz_rep,
            inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep,
            ranks=ranks, cell_sizes=spacings[6:])

    def _update_e_shmap(st, mat, spacings, e_materials, *, ranks):
        inv_dx_sharded, inv_dy_rep, inv_dz_rep = spacings[:3]
        # #1038 leg 4: body moved VERBATIM to
        # _distributed_common.update_e_nu_shmap, shared with the `is_nu`
        # branch of distributed_v2.run_distributed. #1303: this lane hands it
        # the four-cell means run_fn built once, before the loop.
        return update_e_nu_shmap(
            st, mat, mesh, dt,
            inv_dx_sharded, inv_dy_rep, inv_dz_rep,
            nx_per, nx_real, e_materials=e_materials,
            ranks=ranks, boundary=curl_boundary, ghost=ghost, pad_x=pad_x)

    # ------------------------------------------------------------------
    # Phase 2D: dispersive E shmap helper
    # ------------------------------------------------------------------
    # Mirrors ``_update_e_shmap`` but routes the local update through
    # :func:`_update_e_dispersive_local_nu` (which delegates to
    # :func:`rfx.nonuniform._update_e_nu_dispersive`).  Critical: the
    # caller MUST pass the pre-snapshot ``e_old_*`` arrays so the ADE
    # polarisation update reads pre-exchange E values (see ADE
    # Ordering Contract in the docstring of
    # ``run_nonuniform_distributed_pec``).
    #
    # Pole arrays use the (n_devices*n_poles, nx_local, ny, nz) layout
    # documented on ``shard_debye_*`` / ``shard_lorentz_*``.  Inside
    # ``shard_map`` each device sees ``(n_poles, nx_local, ny, nz)``.
    if use_dispersion:
        def _update_e_dispersive_shmap(
            st, mat, db_st_in, lr_st_in, e_old_ex, e_old_ey, e_old_ez,
            debye_coeffs, lorentz_coeffs, spacings, e_materials,
            *, ranks,
        ):
            inv_dx_sharded, inv_dy_rep, inv_dz_rep = spacings[:3]
            # Static booleans captured by closure: which poles are active.
            _has_db = use_debye
            _has_lr = use_lorentz

            # Build the in_specs / out_specs based on which dispersion
            # branches are active so we never wire dummy poles into
            # shard_map (keeps the JIT graph honest).

            db_in_count = 0
            if _has_db:
                # 5 coeff arrays + 3 state arrays = 8 P("x") inputs
                db_in_count = 8
            lr_in_count = 0
            if _has_lr:
                # 6 coeff arrays + 6 state arrays = 12 P("x") inputs
                lr_in_count = 12

            # Build in_specs as a flat tuple
            base_in = (
                P("x"), P("x"), P("x"),       # ex, ey, ez (current)
                P("x"), P("x"), P("x"),       # hx, hy, hz (curl source)
                P(),                          # step
                P("x"), P("x"), P("x"),       # eps_r, sigma, mu_r
                P("x"), P(None), P(None),     # inv_dx, inv_dy, inv_dz
                P("x"), P("x"), P("x"),       # ex_old, ey_old, ez_old (snapshots)
                P("x"),                      # shared component materials
            )
            disp_in = tuple(P("x") for _ in range(db_in_count + lr_in_count))
            in_specs_full = base_in + disp_in

            # Out: ex, ey, ez, step, [+ db_px, db_py, db_pz if Debye]
            #      [+ lr_px, lr_py, lr_pz, lr_pxp, lr_pyp, lr_pzp if Lorentz]
            base_out = (P("x"), P("x"), P("x"), P())
            db_out = (P("x"), P("x"), P("x")) if _has_db else ()
            lr_out = (
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"),
            ) if _has_lr else ()
            out_specs_full = base_out + db_out + lr_out + (P("x"),)

            @partial(
                rank_shard_map,
                mesh=mesh,
                in_specs=in_specs_full,
                out_specs=out_specs_full,
                check_rep=False,
            )
            def _e_disp(*args, rank):
                # Unpack base inputs
                idx = 0
                ex = args[idx]
                idx += 1
                ey = args[idx]
                idx += 1
                ez = args[idx]
                idx += 1
                hx = args[idx]
                idx += 1
                hy = args[idx]
                idx += 1
                hz = args[idx]
                idx += 1
                step = args[idx]
                idx += 1
                eps_r = args[idx]
                idx += 1
                sigma = args[idx]
                idx += 1
                mu_r = args[idx]
                idx += 1
                invdx = args[idx]
                idx += 1
                invdy = args[idx]
                idx += 1
                invdz = args[idx]
                idx += 1
                ex_old_local = args[idx]
                idx += 1
                ey_old_local = args[idx]
                idx += 1
                ez_old_local = args[idx]
                idx += 1
                means = args[idx]
                idx += 1

                # Debye
                if _has_db:
                    from rfx.materials.debye import DebyeCoeffs, DebyeState
                    d_ca = args[idx]
                    idx += 1
                    d_cb = args[idx]
                    idx += 1
                    d_cc = args[idx]
                    idx += 1
                    d_alpha = args[idx]
                    idx += 1
                    d_beta = args[idx]
                    idx += 1
                    d_px = args[idx]
                    idx += 1
                    d_py = args[idx]
                    idx += 1
                    d_pz = args[idx]
                    idx += 1
                    db_local = (
                        slab_dispersion_coeffs(
                            DebyeCoeffs(ca=d_ca, cb=d_cb, cc=d_cc,
                                        alpha=d_alpha, beta=d_beta),
                            means, dt, nx_per, nx_real, rank=rank),
                        DebyeState(px=d_px, py=d_py, pz=d_pz),
                    )
                else:
                    db_local = None

                if _has_lr:
                    from rfx.materials.lorentz import (
                        LorentzCoeffs, LorentzState,
                    )
                    l_ca = args[idx]
                    idx += 1
                    l_cb = args[idx]
                    idx += 1
                    l_cc = args[idx]
                    idx += 1
                    l_a = args[idx]
                    idx += 1
                    l_b = args[idx]
                    idx += 1
                    l_c = args[idx]
                    idx += 1
                    l_px = args[idx]
                    idx += 1
                    l_py = args[idx]
                    idx += 1
                    l_pz = args[idx]
                    idx += 1
                    l_pxp = args[idx]
                    idx += 1
                    l_pyp = args[idx]
                    idx += 1
                    l_pzp = args[idx]
                    idx += 1
                    lr_local = (
                        slab_dispersion_coeffs(
                            LorentzCoeffs(ca=l_ca, cb=l_cb, cc=l_cc,
                                          a=l_a, b=l_b, c=l_c),
                            means, dt, nx_per, nx_real, rank=rank),
                        LorentzState(px=l_px, py=l_py, pz=l_pz,
                                     px_prev=l_pxp, py_prev=l_pyp,
                                     pz_prev=l_pzp),
                    )
                else:
                    lr_local = None

                _st = FDTDState(ex=ex, ey=ey, ez=ez,
                                hx=hx, hy=hy, hz=hz, step=step)
                _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

                new_st, new_db, new_lr = _update_e_dispersive_local_nu(
                    _st, _mat, dt, invdx, invdy, invdz,
                    debye=db_local, lorentz=lr_local,
                    e_old=(ex_old_local, ey_old_local, ez_old_local),
                    **_slab_curl_kwargs(curl_boundary, hx.shape, ghost, pad_x, rank, n_devices),
                )

                from rfx.boundaries.electric_coefficient import dispersive_curl
                absorber_curl = dispersive_curl(
                    db_local[0] if db_local is not None else None,
                    lr_local[0] if lr_local is not None else None, dt)
                outs = (new_st.ex, new_st.ey, new_st.ez, new_st.step)
                if _has_db:
                    outs = outs + (new_db.px, new_db.py, new_db.pz)
                if _has_lr:
                    outs = outs + (
                        new_lr.px, new_lr.py, new_lr.pz,
                        new_lr.px_prev, new_lr.py_prev, new_lr.pz_prev,
                    )
                return outs + (absorber_curl,)

            # Build flat input tuple matching in_specs_full ordering
            call_args = [
                st.ex, st.ey, st.ez,
                st.hx, st.hy, st.hz,
                st.step,
                mat.eps_r, mat.sigma, mat.mu_r,
                inv_dx_sharded, inv_dy_rep, inv_dz_rep,
                e_old_ex, e_old_ey, e_old_ez,
                e_materials,
            ]
            if _has_db:
                call_args.extend([
                    debye_coeffs.ca, debye_coeffs.cb, debye_coeffs.cc,
                    debye_coeffs.alpha, debye_coeffs.beta,
                    db_st_in.px, db_st_in.py, db_st_in.pz,
                ])
            if _has_lr:
                call_args.extend([
                    lorentz_coeffs.ca, lorentz_coeffs.cb, lorentz_coeffs.cc,
                    lorentz_coeffs.a, lorentz_coeffs.b, lorentz_coeffs.c,
                    lr_st_in.px, lr_st_in.py, lr_st_in.pz,
                    lr_st_in.px_prev, lr_st_in.py_prev, lr_st_in.pz_prev,
                ])

            results = _e_disp(ranks, *call_args)

            ridx = 0
            new_ex = results[ridx]
            ridx += 1
            new_ey = results[ridx]
            ridx += 1
            new_ez = results[ridx]
            ridx += 1
            new_step = results[ridx]
            ridx += 1

            new_db_st = None
            new_lr_st = None
            if _has_db:
                from rfx.materials.debye import DebyeState
                new_db_st = DebyeState(
                    px=results[ridx], py=results[ridx + 1], pz=results[ridx + 2],
                )
                ridx += 3
            if _has_lr:
                from rfx.materials.lorentz import LorentzState
                new_lr_st = LorentzState(
                    px=results[ridx], py=results[ridx + 1], pz=results[ridx + 2],
                    px_prev=results[ridx + 3], py_prev=results[ridx + 4],
                    pz_prev=results[ridx + 5],
                )
                ridx += 6

            new_state = st._replace(ex=new_ex, ey=new_ey, ez=new_ez,
                                    step=new_step)
            return new_state, new_db_st, new_lr_st, results[-1]

    def _inject_sources_shmap(st, src_vals_step, *, ranks):
        # Construct host constants while tracing: the outer scan closes over
        # Python metadata only, never concrete device or NumPy arrays.
        source_drives = drive_layout(src_local_specs, src_vals_step.dtype,
                                     electric_only_path="distributed_nu")
        return inject_sources_shmap(
            st, src_vals_step, mesh, n_src,
            src_local_specs, src_device_ids,
            ranks=ranks, drives=source_drives)

    def _sample_probes_shmap(st, *, ranks):
        return sample_probes_shmap(
            st, mesh, n_prb, prb_local_specs, prb_device_ids,
            reduce_devices=False,
            ranks=ranks)

    # ------------------------------------------------------------------
    # Phase 2C: shmap-wrapped CPML helpers (no-op when CPML disabled)
    # ------------------------------------------------------------------
    global_grid = cpml_grid_metadata(sharded_grid.global_grid)

    # The face profiles are dynamic jit inputs; only the Python cell sizes
    # remain static, as required by the local kernels' float conversions.
    from rfx.boundaries.cpml import CPMLAxisParams

    cpml_spacings = {}
    magnetic_cpml_spacings = {}
    if isinstance(cpml_params, CPMLAxisParams):
        cpml_spacings = {name: getattr(cpml_params, name) for name in (
            "dx_x_lo", "dx_x_hi", "dx_y_lo", "dx_y_hi", "dz_lo", "dz_hi",
        )}
        cpml_params = cpml_params._replace(**dict.fromkeys(cpml_spacings))
        if cpml_params.magnetic is not None:
            magnetic_cpml_spacings = {
                name: getattr(cpml_params.magnetic, name) for name in cpml_spacings
            }
            cpml_params = cpml_params._replace(
                magnetic=cpml_params.magnetic._replace(
                    **dict.fromkeys(magnetic_cpml_spacings)))
        cpml_params = jax.tree_util.tree_map(
            lambda arr: _concrete_on_mesh(arr, rep), cpml_params)

    def _apply_cpml_h_shmap(st, cs, cpml_params, mu_r, widths, *, ranks):
        @partial(
            rank_shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),  # ex, ey, ez
                P("x"), P("x"), P("x"),  # hx, hy, hz
                # x-face psi (rank-conditional inside)
                P("x"), P("x"), P("x"), P("x"),
                # y-face psi (sliced per rank)
                P("x"), P("x"), P("x"), P("x"),
                # z-face psi (sliced per rank)
                P("x"), P("x"), P("x"), P("x"),
                P("x"),  # mu_r slab (material-aware CPML coeff, #205)
                (P("x"), P(None), P(None)),
            ),
            out_specs=(
                P("x"), P("x"), P("x"),               # hx, hy, hz
                P("x"), P("x"), P("x"), P("x"),       # x-face psi
                P("x"), P("x"), P("x"), P("x"),       # y-face psi
                P("x"), P("x"), P("x"), P("x"),       # z-face psi
            ),
            check_rep=False,
        )
        def _h(ex, ey, ez, hx, hy, hz,
               psi_hy_xlo, psi_hy_xhi, psi_hz_xlo, psi_hz_xhi,
               psi_hx_ylo, psi_hx_yhi, psi_hz_ylo, psi_hz_yhi,
               psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi,
               mu_r_slab, cell_sizes_slab, *, rank):
            _st = FDTDState(ex=ex, ey=ey, ez=ez,
                            hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
            _cs = cs._replace(
                psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
                psi_hz_xlo=psi_hz_xlo, psi_hz_xhi=psi_hz_xhi,
                psi_hx_ylo=psi_hx_ylo, psi_hx_yhi=psi_hx_yhi,
                psi_hz_ylo=psi_hz_ylo, psi_hz_yhi=psi_hz_yhi,
                psi_hx_zlo=psi_hx_zlo, psi_hx_zhi=psi_hx_zhi,
                psi_hy_zlo=psi_hy_zlo, psi_hy_zhi=psi_hy_zhi,
            )
            mu = magnetic_low_ghost(mu_r_slab, rank, ghost)
            means = component_h_materials(MaterialArrays(None, None, mu), cell_sizes=cell_sizes_slab)
            material_view = SimpleNamespace(mu_r=mu, components=SimpleNamespace(mu_update=means))
            new_st, new_cs = apply_cpml_h(
                _st, cpml_params, _cs, global_grid, materials=material_view,
                faces=cpml_absorbing_faces(global_grid), frame=cpml_slab_frame(ex.shape, ghost, pad_x),
                owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})
            return (new_st.hx, new_st.hy, new_st.hz,
                    new_cs.psi_hy_xlo, new_cs.psi_hy_xhi,
                    new_cs.psi_hz_xlo, new_cs.psi_hz_xhi,
                    new_cs.psi_hx_ylo, new_cs.psi_hx_yhi,
                    new_cs.psi_hz_ylo, new_cs.psi_hz_yhi,
                    new_cs.psi_hx_zlo, new_cs.psi_hx_zhi,
                    new_cs.psi_hy_zlo, new_cs.psi_hy_zhi)

        (hx, hy, hz,
         psi_hy_xlo, psi_hy_xhi, psi_hz_xlo, psi_hz_xhi,
         psi_hx_ylo, psi_hx_yhi, psi_hz_ylo, psi_hz_yhi,
         psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi) = _h(
            ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
            cs.psi_hy_xlo, cs.psi_hy_xhi, cs.psi_hz_xlo, cs.psi_hz_xhi,
            cs.psi_hx_ylo, cs.psi_hx_yhi, cs.psi_hz_ylo, cs.psi_hz_yhi,
            cs.psi_hx_zlo, cs.psi_hx_zhi, cs.psi_hy_zlo, cs.psi_hy_zhi,
            mu_r, widths,
        )
        new_st = st._replace(hx=hx, hy=hy, hz=hz)
        new_cs = cs._replace(
            psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
            psi_hz_xlo=psi_hz_xlo, psi_hz_xhi=psi_hz_xhi,
            psi_hx_ylo=psi_hx_ylo, psi_hx_yhi=psi_hx_yhi,
            psi_hz_ylo=psi_hz_ylo, psi_hz_yhi=psi_hz_yhi,
            psi_hx_zlo=psi_hx_zlo, psi_hx_zhi=psi_hx_zhi,
            psi_hy_zlo=psi_hy_zlo, psi_hy_zhi=psi_hy_zhi,
        )
        return new_st, new_cs

    def _apply_cpml_e_shmap(st, cs, cpml_params, materials, e_materials, e_curl_coeff, *, ranks):
        # The psi coefficient takes the permittivity the E update of this
        # step used (#1043): each component's four-cell edge mean -- the one
        # run_fn built before the loop, including dispersive models.
        @partial(
            rank_shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"), P("x"),  # x-face psi
                P("x"), P("x"), P("x"), P("x"),  # y-face psi
                P("x"), P("x"), P("x"), P("x"),  # z-face psi
                P("x"),  # materials (material-aware CPML coeff, #205)
                P("x"),  # the four-cell eps/sigma means
                P("x"),  # the E update's in-loop dispersive curl coefficient
            ),
            out_specs=(
                P("x"), P("x"), P("x"),               # ex, ey, ez
                P("x"), P("x"), P("x"), P("x"),       # x-face psi
                P("x"), P("x"), P("x"), P("x"),       # y-face psi
                P("x"), P("x"), P("x"), P("x"),       # z-face psi
            ),
            check_rep=False,
        )
        def _e(ex, ey, ez, hx, hy, hz,
               psi_ey_xlo, psi_ey_xhi, psi_ez_xlo, psi_ez_xhi,
               psi_ex_ylo, psi_ex_yhi, psi_ez_ylo, psi_ez_yhi,
               psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi,
               mat_slab, means, curl, *, rank):
            _st = FDTDState(ex=ex, ey=ey, ez=ez,
                            hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
            from rfx.boundaries.electric_coefficient import loss_terms
            eps_r_slab = means[0]
            loss = loss_terms(means, dt) if curl is None else None
            _cs = cs._replace(
                psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
                psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
                psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
                psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
                psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
                psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
            )
            new_st, new_cs = apply_cpml_e(
                _st, cpml_params, _cs, global_grid, materials=mat_slab,
                inv_eps_r_update=tuple(1 / e for e in eps_r_slab), e_loss=loss, e_curl_coeff=curl, boundary=curl_boundary,
                faces=cpml_absorbing_faces(global_grid), frame=cpml_slab_frame(ex.shape, ghost, pad_x),
                owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})
            return (new_st.ex, new_st.ey, new_st.ez,
                    new_cs.psi_ey_xlo, new_cs.psi_ey_xhi,
                    new_cs.psi_ez_xlo, new_cs.psi_ez_xhi,
                    new_cs.psi_ex_ylo, new_cs.psi_ex_yhi,
                    new_cs.psi_ez_ylo, new_cs.psi_ez_yhi,
                    new_cs.psi_ex_zlo, new_cs.psi_ex_zhi,
                    new_cs.psi_ey_zlo, new_cs.psi_ey_zhi)

        (ex, ey, ez,
         psi_ey_xlo, psi_ey_xhi, psi_ez_xlo, psi_ez_xhi,
         psi_ex_ylo, psi_ex_yhi, psi_ez_ylo, psi_ez_yhi,
         psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi) = _e(
            ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
            cs.psi_ey_xlo, cs.psi_ey_xhi, cs.psi_ez_xlo, cs.psi_ez_xhi,
            cs.psi_ex_ylo, cs.psi_ex_yhi, cs.psi_ez_ylo, cs.psi_ez_yhi,
            cs.psi_ex_zlo, cs.psi_ex_zhi, cs.psi_ey_zlo, cs.psi_ey_zhi,
            materials, e_materials, e_curl_coeff,
        )
        new_st = st._replace(ex=ex, ey=ey, ez=ez)
        new_cs = cs._replace(
            psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
            psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
            psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
            psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
            psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
            psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
        )
        return new_st, new_cs

    # ------------------------------------------------------------------
    # Per-step scan body (Phase 2B/2C/2D ordering — see docstring)
    # ------------------------------------------------------------------
    def step_fn(carry, xs, invariants, e_materials=None, *, ranks):
        (sharded_materials, sharded_pec_mask, edge_keep,
         debye_coeffs, lorentz_coeffs, cpml_params, spacings) = invariants
        if cpml_spacings:
            cpml_params = cpml_params._replace(**cpml_spacings)
        if magnetic_cpml_spacings:
            cpml_params = cpml_params._replace(
                magnetic=cpml_params.magnetic._replace(**magnetic_cpml_spacings))
        _step_idx, src_vals = xs
        st = carry["fdtd"]
        cs = carry.get("cpml")
        db_st = carry.get("debye")
        lr_st = carry.get("lorentz")

        # 0. Phase 2D ADE Ordering Contract — snapshot E BEFORE any
        #    ghost exchange / field update.  These references stay valid
        #    even when ``st`` is replaced because jnp arrays are
        #    immutable.  When dispersion is inactive the snapshot is
        #    unused (and the JIT will dead-code-eliminate it).
        ex_old_snapshot = st.ex
        ey_old_snapshot = st.ey
        ez_old_snapshot = st.ez

        # 1. H update (NU)
        st = _update_h_shmap(st, sharded_materials, spacings, ranks=ranks)

        # 2. Phase 2C: CPML H correction (after H, before E exchange).
        if use_cpml:
            st, cs = _apply_cpml_h_shmap(st, cs, cpml_params, sharded_materials.mu_r,
                                        spacings[6:], ranks=ranks)

        # Canonicalize PMC high ghosts after the final H update/correction.
        st = _apply_pmc_face_nu_shmap(
            st, mesh, n_devices, nx_local, pmc_faces, pad_x=pad_x, ranks=ranks)

        # 3. Ghost exchange of H so the upcoming E update sees the
        #    neighbour rank's H at the seam.
        st = _exchange_h_ghosts_nu(st, mesh, n_devices, ranks=ranks)

        # 4. E update — Phase 2D dispatch:
        #    - no dispersion: standard NU E update (Phase 2B path)
        #    - Debye-only / Lorentz-only / mixed: dispersive NU ADE
        #      with the snapshotted ex_old/ey_old/ez_old (NEVER the
        #      post-ghost-exchange E from the previous step's stage 9).
        absorber_curl = None
        if use_dispersion:
            st, db_st, lr_st, absorber_curl = _update_e_dispersive_shmap(
                st, sharded_materials, db_st, lr_st,
                ex_old_snapshot, ey_old_snapshot, ez_old_snapshot,
                debye_coeffs, lorentz_coeffs, spacings, e_materials,
                ranks=ranks)
        else:
            st = _update_e_shmap(st, sharded_materials, spacings, e_materials, ranks=ranks)

        # 5. Phase 2C: CPML E correction (after E, before sources/PEC).
        if use_cpml:
            st, cs = _apply_cpml_e_shmap(
                st, cs, cpml_params, sharded_materials,
                e_materials, absorber_curl, ranks=ranks)

        # 5b. Published soft-PEC factors; both ghost rows carry one.
        if edge_keep is not None:
            keep_x, keep_y, keep_z = edge_keep
            st = st._replace(ex=st.ex * keep_x, ey=st.ey * keep_y,
                             ez=st.ez * keep_z)

        # 6. Source injection (rank-conditional via shard_map)
        st = _inject_sources_shmap(st, src_vals, ranks=ranks)

        # 7. PEC on physical domain faces (X-faces are rank-conditional).
        st = _apply_pec_face_nu_shmap(st, mesh, n_devices, nx_local, pad_x=pad_x, pec_faces=e_wall_faces, ranks=ranks)

        # 8. PEC mask zeroing (geometry + override union).  No-op when
        #    sharded_pec_mask is None.
        if sharded_pec_mask is not None:
            st = _apply_pec_mask_nu_shmap(
                st, sharded_pec_mask, mesh, n_devices, nx_local, ranks=ranks)

        # 9. Ghost exchange of E so the next step's H update sees the
        #    neighbour rank's E at the seam.  LAST in the E half-step, after
        #    the PEC stages: the H update at a rank's last real cell reads
        #    Ey/Ez on its right ghost plane, and that plane's PEC edges are
        #    zeroed by the OWNER rank (stages 7/8 impose the hard conductors).
        #    Exchanging before the PEC stages handed the ghost the
        #    un-zeroed value — the #931 seam-cell divergence (2.107e-01
        #    final-step error on a one-cell body at rank 1's first real
        #    cell; see the docstring).
        st = _exchange_e_ghosts_nu(st, mesh, n_devices, ranks=ranks)

        # 10. Owner-masked samples; reduce devices once after all scans.
        #     Phase 2F emit_time_series=False: skip the probe-sample
        #     accumulation entirely so the AD tape is not loaded with
        #     per-step probe entries.  ``probe_out`` becomes a 0-length
        #     array (still a valid scan output, just empty).
        if emit_time_series:
            probe_out = _sample_probes_shmap(st, ranks=ranks)
        else:
            probe_out = jnp.zeros(0, dtype=jnp.float32)

        new_carry = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cs
        if use_debye:
            new_carry["debye"] = db_st
        if use_lorentz:
            new_carry["lorentz"] = lr_st
        return new_carry, probe_out

    # ------------------------------------------------------------------
    # JIT-compiled scan over n_steps
    # ------------------------------------------------------------------
    # Composition note (Phase 2F / M5 spike): an eager
    # ``jax.checkpoint(shard_map(...))`` raises
    # ``NotImplementedError: Eager evaluation of closed_call inside
    # shard_map isn't yet supported``.  The fix is to keep the outer
    # ``jax.jit`` boundary so that ``jax.checkpoint(segment_body)``
    # composes cleanly inside JIT (the shard_map calls live inside
    # ``step_fn``, which is reached via the ``lax.scan(seg_body, ...)``
    # call inside the JIT'd ``run_fn``).  Both Pattern D
    # ``jit(checkpoint(shard_map))`` and Pattern E
    # ``checkpoint(jit(shard_map))`` were verified to work; we use
    # Pattern D (jit-around-checkpoint) because the outer ``jax.jit``
    # boundary is already required for the scan dispatch and adding it
    # avoids re-jitting the shard_map step body once per call.
    step_indices = _concrete_on_mesh(np.arange(n_steps, dtype=np.int32), rep)
    xs_full = (step_indices, src_waveforms_rep)

    carry_init = {"fdtd": sharded_state}
    if use_cpml:
        carry_init["cpml"] = cpml_state
    if use_debye:
        carry_init["debye"] = debye_state_init
    if use_lorentz:
        carry_init["lorentz"] = lorentz_state_init

    # Phase 2F warmup split: first ``n_warmup`` steps run with
    # ``stop_gradient`` carry boundary so AD never sees that transient.
    # The optimize phase (``n_steps - n_warmup`` steps) runs afterwards
    # and may itself be segmented via ``checkpoint_every``.  When
    # ``n_warmup == 0`` this branch reduces to the prior single-scan
    # path (Phase 2A-2E semantics preserved).
    if n_warmup > 0:
        warmup_xs = (
            xs_full[0][:n_warmup],
            xs_full[1][:n_warmup],
        )
        opt_xs = (
            xs_full[0][n_warmup:],
            xs_full[1][n_warmup:],
        )
        n_opt = n_steps - n_warmup
    else:
        warmup_xs = None
        opt_xs = xs_full
        n_opt = n_steps

    use_segmented = (
        checkpoint_every is not None
        and 0 < int(checkpoint_every) < n_opt
    )

    # Per-cell arrays must enter the jit as arguments: captured concrete
    # arrays become whole-domain compiled constants on every device. Keep
    # their tracer-valued counterparts shared by all scans, including remat.
    def _drive_xs(xs, scales, ranks, weights=None):
        from rfx.model.source_coefficients import slab_drive_table
        return slab_drive_table(xs, mesh, src_device_ids, src_local_specs,
                                ranks=ranks, scales=scales, columns=drive_columns,
                                weights=weights)

    edge_keep = None
    if sharded_pec_occupancy is not None:
        edge_keep = publish_slab_occupancy(
            sharded_materials, sharded_pec_occupancy, mesh).edge_keep

    @jax.jit
    def run_fn(c0, invariants, warmup_xs, opt_xs, *, ranks, e_cell_sizes):
        electric_grid = SlabElectricMetrics(e_cell_sizes, nx_real, nx_per, nx_local)
        scales = None
        if drives:
            # #1279: the drive sees the materials the E update sees -- the
            # override, on the tape when traced -- read in the program.
            scales = material_drive_scales(invariants[0].eps_r, invariants[0].sigma, mesh, drives, dt,
                ranks=ranks, grid=electric_grid, sigma_film=invariants[0].sigma_film)
            if _realized.ACTIVE is not None:
                scales = jnp.broadcast_to(_realized.runtime_drive(jnp.sum(scales, axis=0), tuple(drive_columns)), scales.shape)
        # Occupancy is the last float32 drive factor, read slab-locally on the
        # source's owner; scans only add the stored samples.
        weights = None
        edge_keep = invariants[2]
        if edge_keep is not None and n_src:
            weights = owned_values(edge_keep, mesh, source_nodes, ranks=ranks)
        if warmup_xs is not None:
            warmup_xs = _drive_xs(warmup_xs, scales, ranks, weights)
        opt_xs = _drive_xs(opt_xs, scales, ranks, weights)
        # #1303: each component's four-cell mean, built once for the
        # warm-up and the optimize scans (and every remat segment) -- see
        # slab_e_materials_shmap for why not inside the loop. Dispersive E
        # coefficients are finished from these same means inside the loop.
        e_materials = slab_e_materials_shmap(
            invariants[0], mesh, nx_per, nx_real, ranks=ranks, grid=electric_grid)
        scan_step = partial(step_fn, invariants=invariants, e_materials=e_materials, ranks=ranks)
        # Optional warmup scan: stop_gradient the carry at boundary so
        # AD does not see the warmup steps.  Probe samples from the
        # warmup phase are also stop_gradient'd (they're just metadata
        # at this point — gradient through them would be a tape leak).
        if warmup_xs is not None:
            warmup_final, warmup_ys = lax.scan(scan_step, c0, warmup_xs)
            c0_opt = jax.tree_util.tree_map(lax.stop_gradient, warmup_final)
            warmup_ys = lax.stop_gradient(warmup_ys)
        else:
            c0_opt = c0
            warmup_ys = None

        # Checkpoint full segments and the real, unpadded tail.
        if use_segmented:
            final_, opt_ys = segmented_scan(
                scan_step, c0_opt, opt_xs, int(checkpoint_every), scan=lax.scan
            )
        else:
            final_, opt_ys = lax.scan(scan_step, c0_opt, opt_xs)

        if warmup_ys is not None:
            full_ys = jnp.concatenate([warmup_ys, opt_ys], axis=0)
        else:
            full_ys = opt_ys
        if emit_time_series:
            # (time, device, probe) -> replicated (time, probe), including
            # empty probes. Keep warmup stop_gradient above.
            full_ys = jnp.sum(full_ys, axis=1)
        return final_, full_ys

    final_carry, probe_ts = run_fn(
        carry_init,
        (sharded_materials, sharded_pec_mask, edge_keep,
         debye_coeffs, lorentz_coeffs, cpml_params,
         (inv_dx_sharded, inv_dy_rep, inv_dz_rep,
          inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep, *cell_sizes)),
        warmup_xs, opt_xs,
        ranks=mesh_ranks(mesh), e_cell_sizes=sharded_grid.e_cell_sizes)
    final_state_sharded = final_carry["fdtd"]
    final_cpml_sharded = final_carry.get("cpml") if use_cpml else None
    final_debye_sharded = final_carry.get("debye") if use_debye else None
    final_lorentz_sharded = final_carry.get("lorentz") if use_lorentz else None

    # ------------------------------------------------------------------
    # Gather sharded state -> full-domain (nx, ny, nz)
    # ------------------------------------------------------------------
    def _unstack_and_gather(sharded_arr):
        # Phase 2F: must remain traceable under ``jax.grad``.  The prior
        # ``np.array(sharded_arr)`` call broke whenever a caller wrapped
        # the runner in ``jax.grad`` to drive an objective from the
        # gathered ``final_state`` arrays.  Use pure JAX reshape +
        # ``gather_array_x`` (already JAX-friendly) so the gather stays
        # in the JAX trace.
        return unstack_and_gather(
            sharded_arr, n_devices, nx_local, ghost, pad_x, sharded_grid.nx,
        )

    final_state = None
    if gather_final_state:
        final_state = FDTDState(
            ex=_unstack_and_gather(final_state_sharded.ex),
            ey=_unstack_and_gather(final_state_sharded.ey),
            ez=_unstack_and_gather(final_state_sharded.ez),
            hx=_unstack_and_gather(final_state_sharded.hx),
            hy=_unstack_and_gather(final_state_sharded.hy),
            hz=_unstack_and_gather(final_state_sharded.hz),
            # Step counter is a replicated scalar; keep as a JAX array so
            # the gather works under ``jax.grad`` (no Python ``int()`` cast).
            step=jnp.asarray(final_state_sharded.step, dtype=jnp.int32),
        )

    if not emit_time_series:
        # Phase 2F: probe samples were skipped per-step (probe_out is
        # zero-length per step) so surface ``None`` to make absent
        # time series explicit.  Callers that drive a time-domain
        # objective must request ``emit_time_series=True``.
        time_series = None
    elif n_prb > 0:
        time_series = jnp.array(probe_ts)
    else:
        time_series = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    return {
        "time_series": time_series,
        "final_state": final_state,
        "final_state_sharded": final_state_sharded,
        "cpml_state_sharded": final_cpml_sharded,
        "debye_state_sharded": final_debye_sharded,
        "lorentz_state_sharded": final_lorentz_sharded,
    }

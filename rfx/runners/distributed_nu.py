"""Phase B: Non-uniform FDTD kernels for the shard_map distributed runner.

Lane policy (#1053, PI decision 2026-09-15). ``distributed_v2.py`` is the
distributed development trunk: ``Simulation.run(devices=...)`` dispatches
there for uniform AND non-uniform grids, and since #1065/#1068 it realizes
declared PEC volumes through the shared ``apply_pec_mask_shmap`` stage at the #1041 ordering. THIS module is the ``forward(distributed=True)`` NU lane and
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
from rfx.model.electric_metrics import _build_sharded_inv_dx_arrays
from rfx.model.occupancy import publish_slab_occupancy
from rfx.runners._distributed_common import (
    _update_e_local_nu,
    _update_h_local_nu,
    magnetic_low_ghost,
    apply_pec_face_shmap,
    apply_pec_mask_shmap,
    _slab_curl_kwargs,
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
#
# ``_build_sharded_inv_dx_arrays`` is a slab metric rule and lives in
# ``rfx/model/electric_metrics.py``; it is imported at the top of this module.


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
    high_wall: tuple[bool, bool, bool]
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
    from rfx.geometry.rasterize_grid import unpadded_high_walls
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
        high_wall=unpadded_high_walls(grid),
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
from rfx.stepping.slab_update import _concrete_on_mesh




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


from rfx.stepping.slab_update import _update_e_dispersive_local_nu






from rfx.model.electric_metrics import material_drive_scales as _material_drive_scales, SlabElectricMetrics, stage_forward_dispersion_x_slab
from rfx.stepping.slab_update import material_drive_scales


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
    prepare_only=False, slab_records=None,
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

    Exchange E after all electric writes. Zero high H inverse spacings retain the global
    PMC high ghost at zero, which no exchange sends. Only Hy/Hz left and Ey/Ez right ghosts are refreshed from owners.
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

    validate_scan_controls(n_steps, n_warmup, checkpoint_every, path="graded multi-device distributed_nu")

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
            prepare_only=prepare_only, slab_records=slab_records)

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
    high_wall = sharded_grid.high_wall
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
    # I(t); run_fn reads Cb from staged materials and finishes (1/dV)*(Cb*I).
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
                       s.component, *(dV if isinstance(dV, tuple) else (dV, 0.0))))
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
    from rfx.stepping.slab_graded_legacy_loop import run_graded_legacy_slab_loop
    final_cpml_sharded, final_debye_sharded, final_lorentz_sharded, final_state_sharded, probe_ts = run_graded_legacy_slab_loop(
        cell_sizes=cell_sizes,
        checkpoint_every=checkpoint_every,
        cpml_params=cpml_params,
        cpml_state=cpml_state,
        curl_boundary=curl_boundary,
        debye_coeffs=debye_coeffs,
        debye_state_init=debye_state_init,
        drive_columns=drive_columns,
        drives=drives,
        dt=dt,
        e_wall_faces=e_wall_faces,
        emit_time_series=emit_time_series,
        ghost=ghost,
        high_wall=high_wall,
        inv_dx_h_sharded=inv_dx_h_sharded,
        inv_dx_sharded=inv_dx_sharded,
        inv_dy_h_rep=inv_dy_h_rep,
        inv_dy_rep=inv_dy_rep,
        inv_dz_h_rep=inv_dz_h_rep,
        inv_dz_rep=inv_dz_rep,
        lorentz_coeffs=lorentz_coeffs,
        lorentz_state_init=lorentz_state_init,
        mesh=mesh,
        n_devices=n_devices,
        n_prb=n_prb,
        n_src=n_src,
        n_steps=n_steps,
        n_warmup=n_warmup,
        nx_local=nx_local,
        nx_per=nx_per,
        nx_real=nx_real,
        pad_x=pad_x,
        prb_device_ids=prb_device_ids,
        prb_local_specs=prb_local_specs,
        rep=rep,
        sharded_grid=sharded_grid,
        sharded_materials=sharded_materials,
        sharded_pec_mask=sharded_pec_mask,
        sharded_pec_occupancy=sharded_pec_occupancy,
        sharded_state=sharded_state,
        source_nodes=source_nodes,
        src_device_ids=src_device_ids,
        src_local_specs=src_local_specs,
        src_waveforms_rep=src_waveforms_rep,
        use_cpml=use_cpml,
        use_debye=use_debye,
        use_dispersion=use_dispersion,
        use_lorentz=use_lorentz,
    )

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

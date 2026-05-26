"""Phase B: Non-uniform FDTD kernels for the shard_map distributed runner.

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

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    MU_0,
    EPS_0,
    _shift_fwd,
    _shift_bwd,
)
from rfx.core.jax_utils import is_tracer  # noqa: F401  (Phase 2C reuse target)
from rfx.runners._distributed_common import (
    cpml_coeff_e_vacuum,
    cpml_coeff_h_vacuum,
    exchange_component_shmap,
)


# ---------------------------------------------------------------------------
# Sharded inv-spacing arrays
# ---------------------------------------------------------------------------

def _build_sharded_inv_dx_arrays(grid, n_devices, pad_x=0):
    """Build per-device x-axis inverse-spacing slabs for the shard_map runner.

    The caller has already padded the global x-extent by ``pad_x`` cells
    (to align ``nx`` on ``n_devices``).  We replicate that padding onto
    the cell-size profile using the boundary cell value (matching how
    ``make_nonuniform_grid`` pads CPML cells) and rebuild the global
    ``inv_dx`` and ``inv_dx_h`` from the padded profile, then reshape to
    per-device slabs.

    The two inverse-spacing arrays are built on the *global* padded
    profile, so slicing them into per-device slabs (with ghosts) keeps
    every entry — including the slab-boundary ones — globally correct.

    Parameters
    ----------
    grid : NonUniformGrid
    n_devices : int
    pad_x : int
        Number of PEC-padded cells appended to the high-x end of the
        domain so that ``(nx + pad_x) % n_devices == 0``.

    Returns
    -------
    inv_dx_e_global : (nx_padded,) np.ndarray
        E-update inverse spacing — mean ``2/(d[k-1]+d[k])`` (leading
        ``1/d[0]``). Consumed by ``_update_e_local_nu``.
    inv_dx_h_global : (nx_padded,) np.ndarray
        H-update inverse spacing — local ``1/d[k]`` (trailing ``0``).
        Consumed by ``_update_h_local_nu``.
    dx_padded : (nx_padded,) np.ndarray
        The padded cell-size profile (float32) — useful for diagnostics
        and the unit test.
    """
    dx_arr = np.asarray(grid.dx_arr, dtype=np.float64)
    if pad_x > 0:
        # pad at the high-x end with boundary-cell-size value
        dx_arr = np.concatenate(
            [dx_arr, np.full(pad_x, float(dx_arr[-1]))]
        )
    nx = dx_arr.shape[0]
    if nx % n_devices != 0:
        raise ValueError(
            f"After padding nx={nx} is not divisible by n_devices={n_devices}"
        )

    # CORE-C2: the E update needs the MEAN spacing 2/(d[k-1]+d[k]); the
    # H update needs the LOCAL cell width 1/d[k]. (Mirrors the
    # single-device rfx.nonuniform._profile_to_inv_arrays.) Built on the
    # global padded profile, so each per-device slab is correct after
    # the P("x") shard — the E-mean seam straddles the lower neighbour
    # and is resolved here, globally, before sharding.
    inv_local = 1.0 / dx_arr                          # 1/d[k]
    inv_mean = 2.0 / (dx_arr[:-1] + dx_arr[1:])       # 2/(d[k]+d[k+1])
    # First return -> E update: mean of (d[k-1], d[k]); leading 1/d[0].
    inv_dx_e = np.concatenate([inv_local[:1], inv_mean])
    # Second return -> H update: local cell width; trailing 0.
    inv_dx_h = np.concatenate([inv_local[:-1], np.zeros(1, dtype=np.float64)])

    return (
        inv_dx_e.astype(np.float32),
        inv_dx_h.astype(np.float32),
        dx_arr.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Local NU update kernels (operate on per-device slab including ghosts)
# ---------------------------------------------------------------------------

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


def split_1d_with_ghost(arr: "np.ndarray", n_devices: int, nx_per: int,
                        nx_local: int, ghost: int,
                        pad_value: float) -> "np.ndarray":
    """Split a 1-D inverse-spacing array into per-device slabs with ghost cells.

    This is the canonical split helper shared between the NU metadata builder
    and the distributed_v2 runner.  It produces a ``(n_devices, nx_local)``
    NumPy array where each row is one rank's slab including ``ghost`` cells on
    each side.

    Parameters
    ----------
    arr : np.ndarray  shape (n_devices * nx_per,)
        Padded global inverse-spacing array (output of
        ``_build_sharded_inv_dx_arrays``).
    n_devices : int
    nx_per : int
        Real cells per device (``arr.shape[0] // n_devices``).
    nx_local : int
        ``nx_per + 2 * ghost``.
    ghost : int
        Ghost width (typically 1).
    pad_value : float
        Value to fill boundary ghost cells (1.0 for inv_dx, 0.0 for inv_dx_h).

    Returns
    -------
    slabs : np.ndarray  shape (n_devices, nx_local)
    """
    slabs = np.zeros((n_devices, nx_local), dtype=arr.dtype)
    for d in range(n_devices):
        lo = d * nx_per
        hi = lo + nx_per
        slabs[d, ghost:ghost + nx_per] = arr[lo:hi]
        # left ghost
        if d > 0:
            slabs[d, 0] = arr[lo - 1]
        else:
            slabs[d, 0] = pad_value
        # right ghost
        if d < n_devices - 1:
            slabs[d, -1] = arr[hi]
        else:
            slabs[d, -1] = pad_value
    return slabs


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
            "exchange_interval > 1 is reserved for Phase 2E; "
            "only exchange_interval=1 is supported in Phase 2A."
        )

    ghost = exchange_interval  # ghost_width = exchange_interval cells

    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Pad nx to nearest multiple of n_devices (PEC cells on high-x end)
    pad_x = 0
    if nx % n_devices != 0:
        pad_x = n_devices - (nx % n_devices)
    nx_padded = nx + pad_x

    nx_per = nx_padded // n_devices
    nx_local = nx_per + 2 * ghost

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
    x_starts = tuple(d * nx_per for d in range(n_devices))
    x_stops = tuple(min((d + 1) * nx_per, nx) for d in range(n_devices))

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
    )


# ---------------------------------------------------------------------------
# Phase 2B: hard-PEC + ghost-exchange sharded NU scan body
# ---------------------------------------------------------------------------

def shard_pec_occupancy_x_slab(global_occupancy, sharded_grid: ShardedNUGrid):
    """Slice a global PEC occupancy field along x using the slab ownership of
    :class:`ShardedNUGrid`.

    The PEC occupancy field is a float ``(nx, ny, nz)`` array with values in
    ``[0, 1]`` (0 = no conductor, 1 = full PEC).  This is the soft / relaxed
    analogue of :func:`shard_pec_mask_x_slab` and mirrors its slab layout
    convention exactly: each rank owns the cells in its real-cell range, and
    the ghost cells at the slab seam carry the neighbour rank's occupancy so
    that the per-component tangential occupancy (built via
    ``occ * jnp.maximum(roll(occ, +1), roll(occ, -1))``) sees the correct
    neighbour at the first / last real cell.

    Parameters
    ----------
    global_occupancy : (nx, ny, nz) jnp.ndarray, or None
        Full-domain PEC occupancy.  Returns ``None`` if ``global_occupancy``
        is ``None`` so callers can do an unconditional call.
    sharded_grid : ShardedNUGrid

    Returns
    -------
    sharded_occupancy : (n_devices * nx_local, ny, nz) jnp.ndarray, or None
        x-sharded occupancy with ``P("x")`` layout.  Each device sees
        ``(nx_local, ny, nz)``.

    Notes
    -----
    Ghost cells at physical domain boundaries are padded with ``0.0``
    (no occupancy).  This differs from :func:`shard_pec_mask_x_slab`,
    which pads with ``True`` so :func:`apply_pec_mask`'s tangential rule
    still recognises the boundary face as PEC.  For the occupancy
    primitive, the boundary face is enforced separately by
    :func:`_apply_pec_face_nu_shmap`, so the soft-PEC ghost padding stays
    at ``0.0`` to avoid biasing the soft-occupancy contribution at the
    domain edge.
    """
    if global_occupancy is None:
        return None

    n_devices = sharded_grid.n_devices
    nx_per = sharded_grid.nx_per_rank
    nx_local = sharded_grid.nx_local
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x
    ny = sharded_grid.ny
    nz = sharded_grid.nz

    # Pad along x with occupancy=0.0 (no soft PEC in the high-x pad cells)
    if pad_x > 0:
        pad_widths = [(0, pad_x), (0, 0), (0, 0)]
        global_occupancy = jnp.pad(
            global_occupancy, pad_widths, constant_values=0.0
        )

    nx_padded = global_occupancy.shape[0]
    assert nx_padded == n_devices * nx_per, (
        f"sharded_pec_occupancy: padded shape {nx_padded} != "
        f"n_devices*nx_per_rank = {n_devices * nx_per}"
    )

    dtype = global_occupancy.dtype
    slabs = jnp.zeros((n_devices, nx_local, ny, nz), dtype=dtype)
    for d in range(n_devices):
        lo = d * nx_per
        hi = lo + nx_per
        slabs = slabs.at[d, ghost:ghost + nx_per, :, :].set(
            global_occupancy[lo:hi]
        )
        if d > 0:
            slabs = slabs.at[d, 0, :, :].set(global_occupancy[lo - 1])
        # else: domain boundary at x_lo — leave ghost row at 0.0
        if d < n_devices - 1:
            slabs = slabs.at[d, -1, :, :].set(global_occupancy[hi])
        # else: domain boundary at x_hi — leave ghost row at 0.0

    return slabs.reshape(n_devices * nx_local, ny, nz)


def shard_design_mask_x_slab(global_mask, sharded_grid: ShardedNUGrid):
    """Slice a global design mask along x using the slab ownership of
    :class:`ShardedNUGrid`.

    The design mask is a boolean ``(nx, ny, nz)`` array where ``True``
    marks cells whose ``eps`` participates in the optimisation variable
    (i.e. cells that should keep gradient).  Cells outside the mask
    will be replaced by ``stop_gradient(field)`` at the end of each
    step (mirrors :issue:`#41` semantics in
    :func:`rfx.nonuniform.run_nonuniform`).

    The same slab+ghost layout convention as
    :func:`shard_pec_mask_x_slab` is used.  Ghost rows at the physical
    domain boundary are padded ``False`` (no-design) so the
    ``stop_gradient`` filter is applied there, which is harmless when
    the boundary is PEC (field is zero anyway) but matches the safer
    "no-design outside the user-marked region" semantics.

    Parameters
    ----------
    global_mask : (nx, ny, nz) jnp.ndarray bool, or None
        Full-domain design mask.  Returns ``None`` if ``global_mask`` is
        ``None`` so callers can do an unconditional call.
    sharded_grid : ShardedNUGrid

    Returns
    -------
    sharded_mask : (n_devices * nx_local, ny, nz) jnp.ndarray bool, or None
    """
    if global_mask is None:
        return None

    n_devices = sharded_grid.n_devices
    nx_per = sharded_grid.nx_per_rank
    nx_local = sharded_grid.nx_local
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x
    ny = sharded_grid.ny
    nz = sharded_grid.nz

    if pad_x > 0:
        pad_widths = [(0, pad_x), (0, 0), (0, 0)]
        global_mask = jnp.pad(global_mask, pad_widths, constant_values=False)

    nx_padded = global_mask.shape[0]
    assert nx_padded == n_devices * nx_per, (
        f"sharded_design_mask: padded mask shape {nx_padded} != "
        f"n_devices*nx_per_rank = {n_devices * nx_per}"
    )

    slabs = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.bool_)
    for d in range(n_devices):
        lo = d * nx_per
        hi = lo + nx_per
        slabs = slabs.at[d, ghost:ghost + nx_per, :, :].set(global_mask[lo:hi])
        if d > 0:
            slabs = slabs.at[d, 0, :, :].set(global_mask[lo - 1])
        # else: domain boundary; ghost stays False
        if d < n_devices - 1:
            slabs = slabs.at[d, -1, :, :].set(global_mask[hi])
        # else: domain boundary; ghost stays False

    return slabs.reshape(n_devices * nx_local, ny, nz)


def shard_pec_mask_x_slab(global_mask, sharded_grid: ShardedNUGrid):
    """Slice a global PEC mask along x using the slab ownership of
    :class:`ShardedNUGrid`.

    The PEC mask is a boolean ``(nx, ny, nz)`` array.  Sharding it along
    the same x-slab partition as ``eps_r`` / ``sigma`` ensures that a
    PEC cell at global x-index ``i`` is owned by exactly one rank
    (``rank = i // nx_per_rank``) and not double-zeroed by the union of
    multiple ranks' ``apply_pec_mask`` calls.

    This helper handles ``pad_x`` padding (PEC=True for high-x padded
    cells) and the canonical split-with-ghost slabbing convention used
    by ``_split_state`` / ``_split_materials``.

    Parameters
    ----------
    global_mask : (nx, ny, nz) jnp.ndarray bool, or None
        Full-domain PEC mask.  Returns ``None`` if ``global_mask`` is
        ``None`` so callers can do an unconditional call.
    sharded_grid : ShardedNUGrid

    Returns
    -------
    sharded_mask : (n_devices * nx_local, ny, nz) jnp.ndarray bool, or None
        The mask reshaped to be x-sharded with ``P("x")``.  Each device
        sees ``(nx_local, ny, nz)``.
    """
    if global_mask is None:
        return None

    n_devices = sharded_grid.n_devices
    nx_per = sharded_grid.nx_per_rank
    nx_local = sharded_grid.nx_local
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x
    ny = sharded_grid.ny
    nz = sharded_grid.nz

    # Pad along x with PEC=True (consistent with high-x PEC padding)
    if pad_x > 0:
        pad_widths = [(0, pad_x), (0, 0), (0, 0)]
        global_mask = jnp.pad(global_mask, pad_widths, constant_values=True)

    nx_padded = global_mask.shape[0]
    assert nx_padded == n_devices * nx_per, (
        f"sharded_pec_mask: padded mask shape {nx_padded} != "
        f"n_devices*nx_per_rank = {n_devices * nx_per}"
    )

    # Build per-device slabs with ghost cells.  Ghost cells at the
    # physical boundary are PEC=True (matches the high-x pad and is
    # consistent with apply_pec on the domain face); interior ghosts
    # carry the neighbour's PEC status so apply_pec_mask sees a
    # correct neighbour-set when computing tangential masks.
    slabs = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.bool_)
    for d in range(n_devices):
        lo = d * nx_per
        hi = lo + nx_per
        slabs = slabs.at[d, ghost:ghost + nx_per, :, :].set(global_mask[lo:hi])
        if d > 0:
            slabs = slabs.at[d, 0, :, :].set(global_mask[lo - 1])
        else:
            # Domain boundary at x_lo: ghost is PEC=True (matches apply_pec)
            slabs = slabs.at[d, 0, :, :].set(True)
        if d < n_devices - 1:
            slabs = slabs.at[d, -1, :, :].set(global_mask[hi])
        else:
            # Domain boundary at x_hi: ghost is PEC=True
            slabs = slabs.at[d, -1, :, :].set(True)

    # Reshape to sharded layout: (n_devices * nx_local, ny, nz)
    return slabs.reshape(n_devices * nx_local, ny, nz)


# Ghost-exchange one field component on an x-sharded array.
#
# The NU scan body uses the same ghost-exchange contract as the uniform
# distributed runner; the body lived here verbatim and is now the shared
# ``exchange_component_shmap`` in ``_distributed_common.py``. Kept as a
# module-local alias so the existing call sites are unchanged.
_exchange_component_nu_shmap = exchange_component_shmap


def _exchange_h_ghosts_nu(state: FDTDState, mesh, n_devices: int) -> FDTDState:
    return state._replace(
        hx=_exchange_component_nu_shmap(state.hx, mesh, n_devices),
        hy=_exchange_component_nu_shmap(state.hy, mesh, n_devices),
        hz=_exchange_component_nu_shmap(state.hz, mesh, n_devices),
    )


def _exchange_e_ghosts_nu(state: FDTDState, mesh, n_devices: int) -> FDTDState:
    return state._replace(
        ex=_exchange_component_nu_shmap(state.ex, mesh, n_devices),
        ey=_exchange_component_nu_shmap(state.ey, mesh, n_devices),
        ez=_exchange_component_nu_shmap(state.ez, mesh, n_devices),
    )


def _apply_pec_face_nu_shmap(state: FDTDState, mesh, n_devices: int,
                             nx_local: int) -> FDTDState:
    """Apply PEC on physical domain faces (x_lo, x_hi, y, z) using shard_map.

    Mirrors ``rfx/runners/distributed_v2.py::_apply_pec_shmap`` exactly:
    Y- and Z-face PEC is local to every rank; X-face PEC is rank-
    conditional (only rank 0 zeroes x_lo, only rank N-1 zeroes x_hi).

    Critically, the X-face PEC acts on the **first real cell**
    (``ghost``) and the **last real cell** (``nx_local - 1 - ghost``),
    NOT on the seam ghost cells (which belong to neighbouring ranks).
    This is V3 bullet 7 ("Hard PEC only acts on physical boundary or
    masked cells, not on seam ghosts") and V3 bullet 6 ("Hard PEC does
    not re-zero interior seam cells of neighbouring ranks").
    """
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x")),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _pec(ex, ey, ez):
        ghost = 1

        # Y-axis PEC (every rank — the y boundary is local to all ranks)
        ex = ex.at[:, 0, :].set(0.0)
        ex = ex.at[:, -1, :].set(0.0)
        ez = ez.at[:, 0, :].set(0.0)
        ez = ez.at[:, -1, :].set(0.0)

        # Z-axis PEC (every rank — z boundary is local to all ranks)
        ex = ex.at[:, :, 0].set(0.0)
        ex = ex.at[:, :, -1].set(0.0)
        ey = ey.at[:, :, 0].set(0.0)
        ey = ey.at[:, :, -1].set(0.0)

        device_idx = lax.axis_index("x")

        # X-lo PEC: only rank 0; act on first REAL cell (skip ghost)
        is_first = (device_idx == 0)
        ey_xlo = jnp.where(is_first, 0.0, ey[ghost, :, :])
        ez_xlo = jnp.where(is_first, 0.0, ez[ghost, :, :])
        ey = ey.at[ghost, :, :].set(ey_xlo)
        ez = ez.at[ghost, :, :].set(ez_xlo)

        # X-hi PEC: only rank N-1; act on last REAL cell (skip ghost)
        is_last = (device_idx == n_devices - 1)
        last_real = nx_local - 1 - ghost
        ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
        ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
        ey = ey.at[last_real, :, :].set(ey_xhi)
        ez = ez.at[last_real, :, :].set(ez_xhi)

        return ex, ey, ez

    ex, ey, ez = _pec(state.ex, state.ey, state.ez)
    return state._replace(ex=ex, ey=ey, ez=ez)


def _apply_pmc_face_nu_shmap(state: FDTDState, mesh, n_devices: int,
                             nx_local: int, pmc_faces: frozenset) -> FDTDState:
    """Apply PMC (``H_tan = 0``) on physical domain faces using shard_map.

    Electromagnetic dual of :func:`_apply_pec_face_nu_shmap`: PEC zeroes
    tangential E, PMC zeroes tangential H. Hook point is the H-half of
    the scan body (before the H ghost exchange) so the zero propagates
    to neighbour ranks via the exchange — matching single-device
    ordering at ``rfx/simulation.py:699-705``
    (``apply_cpml_h`` -> ``apply_pmc_faces``).

    Y- and Z-face PMC is local to every rank; X-face PMC is rank-
    conditional (only rank 0 zeroes x_lo, only rank N-1 zeroes x_hi),
    and acts on the **first real cell** (``ghost``) / **last real cell**
    (``nx_local - 1 - ghost``) — NOT the seam ghost.
    """
    if not pmc_faces:
        return state

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x")),
        out_specs=(P("x"), P("x"), P("x")),
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
        last_real = nx_local - 1 - ghost
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


def _apply_pec_mask_nu_shmap(state: FDTDState, sharded_pec_mask, mesh,
                             n_devices: int, nx_local: int) -> FDTDState:
    """Apply geometry-defined PEC mask zeroing on x-sharded fields.

    Each rank owns the PEC cells inside its real-cell range
    ``[ghost, ghost + nx_per_rank)``.  Per V3 bullet 6, we must NOT
    re-zero PEC cells that live in another rank's slab; per V3 bullet 7,
    seam ghost cells must not be acted on.

    The implementation:
      * computes the per-component tangential mask (PEC AND has-PEC-
        neighbour-in-tangential-axis) on the local slab including ghost
        cells, identical math to ``rfx/boundaries/pec.py::apply_pec_mask``;
      * gates the mask so ghost-cell rows are forced to ``False`` before
        zeroing the field — interior real cells use their slab-local
        neighbour computation, and the **first/last real cells** see the
        ghost neighbour (which carries the seam-neighbour's PEC status
        because ``shard_pec_mask_x_slab`` populated it).
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
        # Tangential masks (per-component): PEC AND has-PEC-neighbour
        # in the component's own direction.
        # Use _shift_fwd / _shift_bwd-like inline rolls without wrap so
        # ghost cells do not introduce wrap artefacts.  jnp.roll wraps
        # in pure JAX, but inside a slab that is fine because the slab
        # already includes the seam neighbours via ghost cells; ghost
        # rows are forced to False below.
        mask_ex = mask & (jnp.roll(mask, 1, axis=0) | jnp.roll(mask, -1, axis=0))
        mask_ey = mask & (jnp.roll(mask, 1, axis=1) | jnp.roll(mask, -1, axis=1))
        mask_ez = mask & (jnp.roll(mask, 1, axis=2) | jnp.roll(mask, -1, axis=2))

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


def _apply_pec_occupancy_nu_shmap(state: FDTDState, sharded_pec_occupancy,
                                  mesh, n_devices: int,
                                  nx_local: int) -> FDTDState:
    """Apply soft PEC occupancy to x-sharded fields.

    This is the differentiable analogue of :func:`_apply_pec_mask_nu_shmap`
    and mirrors :func:`rfx.boundaries.pec.apply_pec_occupancy` on a slab
    decomposition.  Each rank owns the occupancy contribution inside its
    real-cell range ``[ghost, ghost + nx_per_rank)``; ghost rows are
    forced to ``0.0`` before the soft-zero is applied so a single seam
    cell is never folded into both neighbouring ranks.

    The occupancy at the first / last real cell sees the seam-neighbour's
    occupancy via the ghost row populated by
    :func:`shard_pec_occupancy_x_slab`, which keeps the
    ``jnp.maximum(roll(+1), roll(-1))`` neighbour rule consistent with
    the single-device path.
    """
    if sharded_pec_occupancy is None:
        return state

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"), P("x")),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _pec_occ(ex, ey, ez, occ):
        occ = jnp.clip(occ.astype(ex.dtype), 0.0, 1.0)

        occ_ex = occ * jnp.maximum(
            jnp.roll(occ, 1, axis=0), jnp.roll(occ, -1, axis=0))
        occ_ey = occ * jnp.maximum(
            jnp.roll(occ, 1, axis=1), jnp.roll(occ, -1, axis=1))
        occ_ez = occ * jnp.maximum(
            jnp.roll(occ, 1, axis=2), jnp.roll(occ, -1, axis=2))

        # Force ghost rows to 0.0 so seam cells in another rank's slab are
        # not double-applied; real cells span [ghost, nx_local - ghost).
        ghost = 1
        zero_row = jnp.zeros_like(occ_ex[0:ghost, :, :])
        occ_ex = occ_ex.at[0:ghost, :, :].set(zero_row)
        occ_ex = occ_ex.at[nx_local - ghost:nx_local, :, :].set(zero_row)
        occ_ey = occ_ey.at[0:ghost, :, :].set(zero_row)
        occ_ey = occ_ey.at[nx_local - ghost:nx_local, :, :].set(zero_row)
        occ_ez = occ_ez.at[0:ghost, :, :].set(zero_row)
        occ_ez = occ_ez.at[nx_local - ghost:nx_local, :, :].set(zero_row)

        ex = ex * (1.0 - occ_ex)
        ey = ey * (1.0 - occ_ey)
        ez = ez * (1.0 - occ_ez)
        return ex, ey, ez

    ex, ey, ez = _pec_occ(state.ex, state.ey, state.ez, sharded_pec_occupancy)
    return state._replace(ex=ex, ey=ey, ez=ez)


# ---------------------------------------------------------------------------
# Phase 2C: CPML on x-slabs (rank-conditional x-faces, sliced y/z faces)
# ---------------------------------------------------------------------------

def init_cpml_for_sharded_nu(sharded_grid: ShardedNUGrid, n_devices: int,
                             *, kappa_max=None, pec_faces=None,
                             pmc_faces=None):
    """Build CPMLAxisParams + a stacked per-rank CPMLState for the sharded
    NU runner.

    The CPML profile calibration is **face-physical** (V3 Phase 2C bullet 5):
    rank 0 uses the true x-lo boundary cell size, rank N-1 uses the true
    x-hi boundary cell size, interior ranks have x-face state that is
    initialised to zero and never updated (gated rank-conditionally inside
    the apply call).  All ranks own their local y- and z-face profiles.

    Because ``make_nonuniform_grid`` pads CPML cells with the boundary
    spacing (verified in Phase 2A), the x-axis CPML cells are uniform —
    so a single ``CPMLAxisParams.x`` profile correctly describes both
    the x-lo and x-hi faces and is shareable across ranks.

    Parameters
    ----------
    sharded_grid : ShardedNUGrid
        Output of :func:`build_sharded_nu_grid`.
    n_devices : int
    kappa_max, pec_faces : forwarded to :func:`init_cpml`.

    Returns
    -------
    cpml_params : CPMLAxisParams
        Per-axis CPML profile + boundary cell sizes (Python floats).
    cpml_state_stacked : CPMLState
        Per-rank CPML state with arrays of shape ``(n_devices, n_cpml, d1, d2)``.
        x-face psi arrays use ``(n_devices, n_cpml, ny, nz)`` (or transposed)
        and start at zero on every rank; only rank 0 / rank N-1 actually
        update them inside the scan body.  y-/z-face psi arrays use
        ``nx_local`` for their x-extent dimension so each rank owns its
        local slab portion of the y/z absorbing layer.
    """
    from rfx.boundaries.cpml import init_cpml

    # Build a duck-typed full-domain grid view that ``init_cpml`` can
    # consume.  We need ``dx``, ``dy``, ``dz`` (cell-size arrays for NU),
    # ``cpml_layers``, ``dt``, and ``shape``.  We construct a minimal
    # adapter object pulling the cached metadata off ``sharded_grid``;
    # this avoids re-building the full NonUniformGrid here.
    nx = sharded_grid.nx
    ny = sharded_grid.ny
    nz = sharded_grid.nz

    # Recover boundary cell sizes for x and y from the padded dx profile
    # (constant in the CPML layers thanks to make_nonuniform_grid padding).
    dx_padded = np.asarray(sharded_grid.dx_padded)
    dx_boundary = float(dx_padded[0])
    # Approximate dy boundary from inv_dy (1 / dy_arr[0]); inv_dy is
    # length-ny float32.
    inv_dy = np.asarray(sharded_grid.inv_dy)
    dy_boundary = float(1.0 / inv_dy[0])
    # dz array (length nz) reconstructed from inv_dz.
    inv_dz = np.asarray(sharded_grid.inv_dz)
    dz_arr = (1.0 / inv_dz).astype(np.float32)

    class _SharedNUGridView:
        """Minimal duck-typed view consumed by ``init_cpml``."""

        def __init__(self):
            # ``init_cpml`` reads grid.dx, grid.dy (optional), grid.dz
            # (optional), grid.cpml_layers, grid.dt, grid.shape (or
            # grid.nx/ny/nz).
            self.dx = dx_boundary
            self.dy = dy_boundary
            self.dz = jnp.asarray(dz_arr)
            self.cpml_layers = sharded_grid.cpml_layers
            self.dt = sharded_grid.dt
            self.nx = nx
            self.ny = ny
            self.nz = nz
            # Optional kappa_max / pec_faces / pmc_faces; init_cpml
            # falls back to these attrs when the kwargs are missing.
            self.kappa_max = kappa_max
            self.pec_faces = pec_faces
            self.pmc_faces = pmc_faces

        @property
        def shape(self):
            return (self.nx, self.ny, self.nz)

    grid_view = _SharedNUGridView()
    cpml_params, _single_state = init_cpml(
        grid_view, kappa_max=kappa_max, pec_faces=pec_faces,
        pmc_faces=pmc_faces,
    )

    # Build per-rank stacked CPMLState arrays.  Shapes follow the single-
    # device convention from rfx/boundaries/cpml.py:init_cpml, but with
    # per-rank ``nx_local`` substituted for ``nx`` on y-/z-face psi.
    n = sharded_grid.cpml_layers
    nx_local = sharded_grid.nx_local
    if n <= 0:
        # Caller selected pec boundary; build a degenerate state.  Returning
        # ``None`` forces the runner to take the PEC-only path.
        return cpml_params, None

    from rfx.boundaries.cpml import CPMLState

    def _zeros(d1, d2):
        # (n_devices, n_cpml, d1, d2) — same dtype as single-device init
        return jnp.zeros((n_devices, n, d1, d2), dtype=jnp.float32)

    cpml_state_stacked = CPMLState(
        # E-field psi arrays
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
        # H-field psi arrays
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
    return cpml_params, cpml_state_stacked


def shard_cpml_state_x_slab(cpml_state_stacked, sharded_grid: ShardedNUGrid,
                            mesh):
    """Shard a stacked CPMLState across the x-axis mesh.

    Each ``psi_*`` array has stacked shape ``(n_devices, n_cpml, d1, d2)``;
    we merge the leading two dims into one (shape
    ``(n_devices*n_cpml, d1, d2)``) and place it onto the mesh with
    ``P("x")`` so each device owns ``(n_cpml, d1, d2)``.

    For x-face psi (``psi_*_xlo`` / ``psi_*_xhi``), ``d1`` / ``d2`` are
    the global ``(ny, nz)`` (or transposed) extents; every rank holds a
    full copy but only rank 0 / rank N-1 ever update theirs (the apply
    helper gates the update with ``lax.axis_index``).  This satisfies
    V3 Phase 2C bullets 1-2: outer x-face CPML is rank-conditional;
    interior ranks' x-face psi remain at zero (Class C assertion).

    For y-/z-face psi, the ``d1`` / ``d2`` dimension that indexes into x
    is sized to the per-rank ``nx_local``, so each rank's slice covers
    only the local slab — V3 bullets 3-4.
    """
    from jax.sharding import NamedSharding, PartitionSpec as _P
    from rfx.boundaries.cpml import CPMLState

    if cpml_state_stacked is None:
        return None

    shd = NamedSharding(mesh, _P("x"))

    def _shard_psi(arr):
        n_dev, n_c, d1, d2 = arr.shape
        merged = arr.reshape(n_dev * n_c, d1, d2)
        return jax.device_put(merged, shd)

    return CPMLState(
        psi_ex_ylo=_shard_psi(cpml_state_stacked.psi_ex_ylo),
        psi_ex_yhi=_shard_psi(cpml_state_stacked.psi_ex_yhi),
        psi_ex_zlo=_shard_psi(cpml_state_stacked.psi_ex_zlo),
        psi_ex_zhi=_shard_psi(cpml_state_stacked.psi_ex_zhi),
        psi_ey_xlo=_shard_psi(cpml_state_stacked.psi_ey_xlo),
        psi_ey_xhi=_shard_psi(cpml_state_stacked.psi_ey_xhi),
        psi_ey_zlo=_shard_psi(cpml_state_stacked.psi_ey_zlo),
        psi_ey_zhi=_shard_psi(cpml_state_stacked.psi_ey_zhi),
        psi_ez_xlo=_shard_psi(cpml_state_stacked.psi_ez_xlo),
        psi_ez_xhi=_shard_psi(cpml_state_stacked.psi_ez_xhi),
        psi_ez_ylo=_shard_psi(cpml_state_stacked.psi_ez_ylo),
        psi_ez_yhi=_shard_psi(cpml_state_stacked.psi_ez_yhi),
        psi_hx_ylo=_shard_psi(cpml_state_stacked.psi_hx_ylo),
        psi_hx_yhi=_shard_psi(cpml_state_stacked.psi_hx_yhi),
        psi_hx_zlo=_shard_psi(cpml_state_stacked.psi_hx_zlo),
        psi_hx_zhi=_shard_psi(cpml_state_stacked.psi_hx_zhi),
        psi_hy_xlo=_shard_psi(cpml_state_stacked.psi_hy_xlo),
        psi_hy_xhi=_shard_psi(cpml_state_stacked.psi_hy_xhi),
        psi_hy_zlo=_shard_psi(cpml_state_stacked.psi_hy_zlo),
        psi_hy_zhi=_shard_psi(cpml_state_stacked.psi_hy_zhi),
        psi_hz_xlo=_shard_psi(cpml_state_stacked.psi_hz_xlo),
        psi_hz_xhi=_shard_psi(cpml_state_stacked.psi_hz_xhi),
        psi_hz_ylo=_shard_psi(cpml_state_stacked.psi_hz_ylo),
        psi_hz_yhi=_shard_psi(cpml_state_stacked.psi_hz_yhi),
    )


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
# Padding policy (matches the no-Debye / no-Lorentz dummy paths in
# ``rfx/runners/distributed_v2.py``):
#   * Debye coeffs: ca, cb, cc -> pad with 0; alpha, beta -> pad with 0
#   * Lorentz coeffs: ca, cb, cc -> pad with 0; a, b, c -> pad with 0
# Pad cells correspond to vacuum + high-x PEC padding; with ca=0/alpha=0
# the polarisation update is a no-op there, which matches single-device
# behaviour for cells that have no Debye/Lorentz pole.

def shard_debye_coeffs_x_slab(debye_coeffs, sharded_grid: ShardedNUGrid,
                              mesh):
    """Slab-shard a full-domain ``DebyeCoeffs`` along x.

    Layout
    ------
    Inputs:
        ca, cb : (nx, ny, nz)
        cc, alpha, beta : (n_poles, nx, ny, nz)
    After pad-to-divisible (high-x PEC padding handled by
    ``_split_debye_coeffs``) and reshape:
        ca, cb : sharded along ``P("x")`` over (n_devices*nx_local, ny, nz)
        cc, alpha, beta : sharded along ``P("x")`` over
                          (n_devices*n_poles, nx_local, ny, nz)
    """
    if debye_coeffs is None:
        return None

    from jax.sharding import NamedSharding, PartitionSpec as _P
    from rfx.materials.debye import DebyeCoeffs
    from rfx.runners.distributed import _split_debye_coeffs

    n_devices = sharded_grid.n_devices
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x

    # Pad along x to nx_padded so the canonical splitter sees a clean
    # multiple of n_devices.  For ca/cb pad with 0 (vacuum equivalent;
    # ADE update is no-op when ca=0 -> see rfx/materials/debye.py
    # init_debye for the vacuum-cell coefficients).  cc/alpha/beta also
    # pad with 0.  This matches the high-x PEC pad rationale: those
    # cells will be hard-zeroed by ``_apply_pec_face_nu_shmap``.
    if pad_x > 0:
        pad3 = ((0, pad_x), (0, 0), (0, 0))
        pad4 = ((0, 0), (0, pad_x), (0, 0), (0, 0))
        ca = jnp.pad(debye_coeffs.ca, pad3, constant_values=0.0)
        cb = jnp.pad(debye_coeffs.cb, pad3, constant_values=0.0)
        cc = jnp.pad(debye_coeffs.cc, pad4, constant_values=0.0)
        alpha = jnp.pad(debye_coeffs.alpha, pad4, constant_values=0.0)
        beta = jnp.pad(debye_coeffs.beta, pad4, constant_values=0.0)
        debye_coeffs_padded = DebyeCoeffs(ca=ca, cb=cb, cc=cc,
                                          alpha=alpha, beta=beta)
    else:
        debye_coeffs_padded = debye_coeffs

    coeffs_slabs = _split_debye_coeffs(
        debye_coeffs_padded, n_devices, ghost,
    )

    shd = NamedSharding(mesh, _P("x"))

    def _shard_3d(arr):
        # (n_devices, nx_local, ny, nz) -> (n_devices*nx_local, ny, nz)
        n_dev = arr.shape[0]
        rest = arr.shape[1:]
        return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)

    def _shard_4d(arr):
        # (n_devices, n_poles, nx_local, ny, nz) ->
        # (n_devices*n_poles, nx_local, ny, nz)
        n_dev, n_poles, nx_loc, ny_a, nz_a = arr.shape
        return jax.device_put(
            arr.reshape(n_dev * n_poles, nx_loc, ny_a, nz_a), shd,
        )

    return DebyeCoeffs(
        ca=_shard_3d(coeffs_slabs.ca),
        cb=_shard_3d(coeffs_slabs.cb),
        cc=_shard_4d(coeffs_slabs.cc),
        alpha=_shard_4d(coeffs_slabs.alpha),
        beta=_shard_4d(coeffs_slabs.beta),
    )


def shard_debye_state_x_slab(debye_state, sharded_grid: ShardedNUGrid,
                             mesh):
    """Slab-shard a full-domain ``DebyeState`` along x.

    Each ``p[xyz]`` has shape ``(n_poles, nx, ny, nz)`` and ends up
    sharded along ``P("x")`` over ``(n_devices*n_poles, nx_local, ny, nz)``.
    """
    if debye_state is None:
        return None

    from jax.sharding import NamedSharding, PartitionSpec as _P
    from rfx.materials.debye import DebyeState
    from rfx.runners.distributed import _split_debye_state

    n_devices = sharded_grid.n_devices
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x

    if pad_x > 0:
        pad4 = ((0, 0), (0, pad_x), (0, 0), (0, 0))
        debye_state_padded = DebyeState(
            px=jnp.pad(debye_state.px, pad4, constant_values=0.0),
            py=jnp.pad(debye_state.py, pad4, constant_values=0.0),
            pz=jnp.pad(debye_state.pz, pad4, constant_values=0.0),
        )
    else:
        debye_state_padded = debye_state

    state_slabs = _split_debye_state(debye_state_padded, n_devices, ghost)

    shd = NamedSharding(mesh, _P("x"))

    def _shard_4d(arr):
        n_dev, n_poles, nx_loc, ny_a, nz_a = arr.shape
        return jax.device_put(
            arr.reshape(n_dev * n_poles, nx_loc, ny_a, nz_a), shd,
        )

    return DebyeState(
        px=_shard_4d(state_slabs.px),
        py=_shard_4d(state_slabs.py),
        pz=_shard_4d(state_slabs.pz),
    )


def shard_lorentz_coeffs_x_slab(lorentz_coeffs, sharded_grid: ShardedNUGrid,
                                mesh):
    """Slab-shard a full-domain ``LorentzCoeffs`` along x.

    Layout
    ------
    Inputs:
        ca, cb, cc : (nx, ny, nz)
        a, b, c : (n_poles, nx, ny, nz)
    """
    if lorentz_coeffs is None:
        return None

    from jax.sharding import NamedSharding, PartitionSpec as _P
    from rfx.materials.lorentz import LorentzCoeffs
    from rfx.runners.distributed import _split_lorentz_coeffs

    n_devices = sharded_grid.n_devices
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x

    if pad_x > 0:
        pad3 = ((0, pad_x), (0, 0), (0, 0))
        pad4 = ((0, 0), (0, pad_x), (0, 0), (0, 0))
        # See `_split_lorentz_coeffs` docstring for the rationale: cc must
        # pad with the vacuum 1/EPS_0 so `gamma_base = 1/cc` is finite in
        # the mixed Debye+Lorentz path (avoids 0*inf = NaN leaking into
        # backward gradients at the x-boundary).
        lorentz_coeffs_padded = LorentzCoeffs(
            ca=jnp.pad(lorentz_coeffs.ca, pad3, constant_values=0.0),
            cb=jnp.pad(lorentz_coeffs.cb, pad3, constant_values=0.0),
            cc=jnp.pad(lorentz_coeffs.cc, pad3,
                       constant_values=float(1.0 / EPS_0)),
            a=jnp.pad(lorentz_coeffs.a, pad4, constant_values=0.0),
            b=jnp.pad(lorentz_coeffs.b, pad4, constant_values=0.0),
            c=jnp.pad(lorentz_coeffs.c, pad4, constant_values=0.0),
        )
    else:
        lorentz_coeffs_padded = lorentz_coeffs

    coeffs_slabs = _split_lorentz_coeffs(
        lorentz_coeffs_padded, n_devices, ghost,
    )

    shd = NamedSharding(mesh, _P("x"))

    def _shard_3d(arr):
        n_dev = arr.shape[0]
        rest = arr.shape[1:]
        return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)

    def _shard_4d(arr):
        n_dev, n_poles, nx_loc, ny_a, nz_a = arr.shape
        return jax.device_put(
            arr.reshape(n_dev * n_poles, nx_loc, ny_a, nz_a), shd,
        )

    return LorentzCoeffs(
        ca=_shard_3d(coeffs_slabs.ca),
        cb=_shard_3d(coeffs_slabs.cb),
        cc=_shard_3d(coeffs_slabs.cc),
        a=_shard_4d(coeffs_slabs.a),
        b=_shard_4d(coeffs_slabs.b),
        c=_shard_4d(coeffs_slabs.c),
    )


def shard_lorentz_state_x_slab(lorentz_state, sharded_grid: ShardedNUGrid,
                               mesh):
    """Slab-shard a full-domain ``LorentzState`` along x.

    Each ``p[xyz]`` and ``p[xyz]_prev`` has shape
    ``(n_poles, nx, ny, nz)`` and ends up sharded along ``P("x")`` over
    ``(n_devices*n_poles, nx_local, ny, nz)``.
    """
    if lorentz_state is None:
        return None

    from jax.sharding import NamedSharding, PartitionSpec as _P
    from rfx.materials.lorentz import LorentzState
    from rfx.runners.distributed import _split_lorentz_state

    n_devices = sharded_grid.n_devices
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x

    if pad_x > 0:
        pad4 = ((0, 0), (0, pad_x), (0, 0), (0, 0))
        lorentz_state_padded = LorentzState(
            px=jnp.pad(lorentz_state.px, pad4, constant_values=0.0),
            py=jnp.pad(lorentz_state.py, pad4, constant_values=0.0),
            pz=jnp.pad(lorentz_state.pz, pad4, constant_values=0.0),
            px_prev=jnp.pad(lorentz_state.px_prev, pad4, constant_values=0.0),
            py_prev=jnp.pad(lorentz_state.py_prev, pad4, constant_values=0.0),
            pz_prev=jnp.pad(lorentz_state.pz_prev, pad4, constant_values=0.0),
        )
    else:
        lorentz_state_padded = lorentz_state

    state_slabs = _split_lorentz_state(lorentz_state_padded, n_devices, ghost)

    shd = NamedSharding(mesh, _P("x"))

    def _shard_4d(arr):
        n_dev, n_poles, nx_loc, ny_a, nz_a = arr.shape
        return jax.device_put(
            arr.reshape(n_dev * n_poles, nx_loc, ny_a, nz_a), shd,
        )

    return LorentzState(
        px=_shard_4d(state_slabs.px),
        py=_shard_4d(state_slabs.py),
        pz=_shard_4d(state_slabs.pz),
        px_prev=_shard_4d(state_slabs.px_prev),
        py_prev=_shard_4d(state_slabs.py_prev),
        pz_prev=_shard_4d(state_slabs.pz_prev),
    )


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
    e_old: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
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
    from rfx.nonuniform import _update_e_nu_dispersive

    return _update_e_nu_dispersive(
        state, materials, dt, inv_dx, inv_dy, inv_dz,
        debye=debye, lorentz=lorentz, e_old=e_old,
    )


def _apply_cpml_e_local_nu(state: FDTDState, cpml_params, cpml_state,
                           n_cpml: int, dt: float, ghost: int,
                           n_devices: int):
    """Per-rank slab-aware CPML E-field correction with NU per-axis dx.

    Mirrors :func:`rfx.boundaries.cpml.apply_cpml_e` but operates on a
    slab including ``ghost`` cells, uses :class:`CPMLAxisParams` for
    face-physical per-axis profiles (NU), and is x-face rank-conditional
    via ``lax.axis_index``.

    Must be called inside ``shard_map`` so ``lax.axis_index("x")`` is
    available.
    """
    from rfx.boundaries.cpml import CPMLAxisParams

    coeff_e = cpml_coeff_e_vacuum(dt)  # vacuum coefficient (matches uniform path)

    if isinstance(cpml_params, CPMLAxisParams):
        # T7 PR1: read lo-face profile; scan body synthesises the hi-face
        # inline via jnp.flip(px.b) which preserves bit-identity with pre-PR1.
        px, py, pz_lo, pz_hi = (
            cpml_params.x_lo, cpml_params.y_lo,
            cpml_params.z_lo, cpml_params.z_hi)
        dx_x = float(cpml_params.dx_x_lo)
        dx_y = float(cpml_params.dx_y_lo)
        dz_lo = float(cpml_params.dz_lo)
        dz_hi = float(cpml_params.dz_hi)
    else:
        # Legacy single-profile path (uniform).
        px = py = pz_lo = pz_hi = cpml_params
        dx_x = dx_y = dz_lo = dz_hi = float(cpml_params.b.shape[0])  # placeholder

    # Profile coefficients (broadcast on x-axis index 0).
    b_x = px.b[:, None, None]
    c_x = px.c[:, None, None]
    k_x = px.kappa[:, None, None]
    b_xr = jnp.flip(px.b)[:, None, None]
    c_xr = jnp.flip(px.c)[:, None, None]
    k_xr = jnp.flip(px.kappa)[:, None, None]
    b_y = py.b[:, None, None]
    c_y = py.c[:, None, None]
    k_y = py.kappa[:, None, None]
    b_yr = jnp.flip(py.b)[:, None, None]
    c_yr = jnp.flip(py.c)[:, None, None]
    k_yr = jnp.flip(py.kappa)[:, None, None]
    b_zl = pz_lo.b[:, None, None]
    c_zl = pz_lo.c[:, None, None]
    k_zl = pz_lo.kappa[:, None, None]
    b_zh = pz_hi.b[:, None, None]
    c_zh = pz_hi.c[:, None, None]
    k_zh = pz_hi.kappa[:, None, None]

    n = n_cpml
    g = ghost
    xlo = slice(g, g + n)
    xhi = slice(-(g + n), -g) if g > 0 else slice(-n, None)

    device_idx = lax.axis_index("x")
    is_first = (device_idx == 0)
    is_last = (device_idx == n_devices - 1)

    ex = state.ex
    ey = state.ey
    ez = state.ez

    # =========================================================
    # X-axis CPML — rank-conditional (rank 0 owns x-lo, rank N-1 owns x-hi).
    # Internal slab seams are NOT physical CPML boundaries (V3 bullet 2);
    # interior ranks compute the candidate update but the where-mask
    # discards both the field correction and the psi-state update so
    # interior x-face psi stays exactly zero (Class C assertion).
    # =========================================================
    # --- X-lo: Ey from dHz/dx ---
    hz_xlo = state.hz[xlo, :, :]
    hz_shifted_xlo = _shift_bwd(state.hz, 0)[xlo, :, :]
    curl_hz_dx_xlo = (hz_xlo - hz_shifted_xlo) / dx_x
    new_psi_ey_xlo = b_x * cpml_state.psi_ey_xlo + c_x * curl_hz_dx_xlo
    ey_corr_xlo = (
        -coeff_e * new_psi_ey_xlo
        - coeff_e * (1.0 / k_x - 1.0) * curl_hz_dx_xlo
    )
    ey_corr_xlo = jnp.where(is_first, ey_corr_xlo, 0.0)
    ey = ey.at[xlo, :, :].add(ey_corr_xlo)
    new_psi_ey_xlo = jnp.where(is_first, new_psi_ey_xlo, cpml_state.psi_ey_xlo)

    # --- X-hi: Ey from dHz/dx ---
    hz_xhi = state.hz[xhi, :, :]
    hz_shifted_xhi = _shift_bwd(state.hz, 0)[xhi, :, :]
    curl_hz_dx_xhi = (hz_xhi - hz_shifted_xhi) / dx_x
    new_psi_ey_xhi = b_xr * cpml_state.psi_ey_xhi + c_xr * curl_hz_dx_xhi
    ey_corr_xhi = (
        -coeff_e * new_psi_ey_xhi
        - coeff_e * (1.0 / k_xr - 1.0) * curl_hz_dx_xhi
    )
    ey_corr_xhi = jnp.where(is_last, ey_corr_xhi, 0.0)
    ey = ey.at[xhi, :, :].add(ey_corr_xhi)
    new_psi_ey_xhi = jnp.where(is_last, new_psi_ey_xhi, cpml_state.psi_ey_xhi)

    # --- X-lo: Ez from dHy/dx ---
    hy_xlo = state.hy[xlo, :, :]
    hy_shifted_xlo = _shift_bwd(state.hy, 0)[xlo, :, :]
    curl_hy_dx_xlo = (hy_xlo - hy_shifted_xlo) / dx_x
    curl_hy_dx_xlo_t = jnp.transpose(curl_hy_dx_xlo, (0, 2, 1))
    new_psi_ez_xlo = b_x * cpml_state.psi_ez_xlo + c_x * curl_hy_dx_xlo_t
    correction_ez_xlo = jnp.transpose(new_psi_ez_xlo, (0, 2, 1))
    ez_corr_xlo = (
        coeff_e * correction_ez_xlo
        + coeff_e * (1.0 / k_x - 1.0) * curl_hy_dx_xlo
    )
    ez_corr_xlo = jnp.where(is_first, ez_corr_xlo, 0.0)
    ez = ez.at[xlo, :, :].add(ez_corr_xlo)
    new_psi_ez_xlo = jnp.where(is_first, new_psi_ez_xlo, cpml_state.psi_ez_xlo)

    # --- X-hi: Ez from dHy/dx ---
    hy_xhi = state.hy[xhi, :, :]
    hy_shifted_xhi = _shift_bwd(state.hy, 0)[xhi, :, :]
    curl_hy_dx_xhi = (hy_xhi - hy_shifted_xhi) / dx_x
    curl_hy_dx_xhi_t = jnp.transpose(curl_hy_dx_xhi, (0, 2, 1))
    new_psi_ez_xhi = b_xr * cpml_state.psi_ez_xhi + c_xr * curl_hy_dx_xhi_t
    correction_ez_xhi = jnp.transpose(new_psi_ez_xhi, (0, 2, 1))
    ez_corr_xhi = (
        coeff_e * correction_ez_xhi
        + coeff_e * (1.0 / k_xr - 1.0) * curl_hy_dx_xhi
    )
    ez_corr_xhi = jnp.where(is_last, ez_corr_xhi, 0.0)
    ez = ez.at[xhi, :, :].add(ez_corr_xhi)
    new_psi_ez_xhi = jnp.where(is_last, new_psi_ez_xhi, cpml_state.psi_ez_xhi)

    # =========================================================
    # Y-axis CPML — every rank, sliced over local x extent.
    # =========================================================
    # --- Y-lo: Ex from dHz/dy ---
    hz_ylo = state.hz[:, :n, :]
    hz_shifted_ylo = _shift_bwd(state.hz, 1)[:, :n, :]
    curl_hz_dy_ylo = (hz_ylo - hz_shifted_ylo) / dx_y
    curl_hz_dy_ylo_t = jnp.transpose(curl_hz_dy_ylo, (1, 0, 2))
    new_psi_ex_ylo = b_y * cpml_state.psi_ex_ylo + c_y * curl_hz_dy_ylo_t
    correction_ex_ylo = jnp.transpose(new_psi_ex_ylo, (1, 0, 2))
    ex = ex.at[:, :n, :].add(coeff_e * correction_ex_ylo)
    kappa_corr_ylo = jnp.transpose((1.0 / k_y - 1.0) * curl_hz_dy_ylo_t, (1, 0, 2))
    ex = ex.at[:, :n, :].add(coeff_e * kappa_corr_ylo)

    # --- Y-hi: Ex from dHz/dy ---
    hz_yhi = state.hz[:, -n:, :]
    hz_shifted_yhi = _shift_bwd(state.hz, 1)[:, -n:, :]
    curl_hz_dy_yhi = (hz_yhi - hz_shifted_yhi) / dx_y
    curl_hz_dy_yhi_t = jnp.transpose(curl_hz_dy_yhi, (1, 0, 2))
    new_psi_ex_yhi = b_yr * cpml_state.psi_ex_yhi + c_yr * curl_hz_dy_yhi_t
    correction_ex_yhi = jnp.transpose(new_psi_ex_yhi, (1, 0, 2))
    ex = ex.at[:, -n:, :].add(coeff_e * correction_ex_yhi)
    kappa_corr_yhi = jnp.transpose((1.0 / k_yr - 1.0) * curl_hz_dy_yhi_t, (1, 0, 2))
    ex = ex.at[:, -n:, :].add(coeff_e * kappa_corr_yhi)

    # --- Y-lo: Ez from dHx/dy ---
    hx_ylo = state.hx[:, :n, :]
    hx_shifted_ylo = _shift_bwd(state.hx, 1)[:, :n, :]
    curl_hx_dy_ylo = (hx_ylo - hx_shifted_ylo) / dx_y
    curl_hx_dy_ylo_t = jnp.transpose(curl_hx_dy_ylo, (1, 2, 0))
    new_psi_ez_ylo = b_y * cpml_state.psi_ez_ylo + c_y * curl_hx_dy_ylo_t
    correction_ez_ylo = jnp.transpose(new_psi_ez_ylo, (2, 0, 1))
    ez = ez.at[:, :n, :].add(-coeff_e * correction_ez_ylo)
    kappa_corr_ez_ylo = jnp.transpose((1.0 / k_y - 1.0) * curl_hx_dy_ylo_t, (2, 0, 1))
    ez = ez.at[:, :n, :].add(-coeff_e * kappa_corr_ez_ylo)

    # --- Y-hi: Ez from dHx/dy ---
    hx_yhi = state.hx[:, -n:, :]
    hx_shifted_yhi = _shift_bwd(state.hx, 1)[:, -n:, :]
    curl_hx_dy_yhi = (hx_yhi - hx_shifted_yhi) / dx_y
    curl_hx_dy_yhi_t = jnp.transpose(curl_hx_dy_yhi, (1, 2, 0))
    new_psi_ez_yhi = b_yr * cpml_state.psi_ez_yhi + c_yr * curl_hx_dy_yhi_t
    correction_ez_yhi = jnp.transpose(new_psi_ez_yhi, (2, 0, 1))
    ez = ez.at[:, -n:, :].add(-coeff_e * correction_ez_yhi)
    kappa_corr_ez_yhi = jnp.transpose((1.0 / k_yr - 1.0) * curl_hx_dy_yhi_t, (2, 0, 1))
    ez = ez.at[:, -n:, :].add(-coeff_e * kappa_corr_ez_yhi)

    # =========================================================
    # Z-axis CPML — every rank, sliced over local x extent.
    # =========================================================
    # --- Z-lo: Ex from dHy/dz ---
    hy_zlo = state.hy[:, :, :n]
    hy_shifted_zlo = _shift_bwd(state.hy, 2)[:, :, :n]
    curl_hy_dz_zlo = (hy_zlo - hy_shifted_zlo) / dz_lo
    curl_hy_dz_zlo_t = jnp.transpose(curl_hy_dz_zlo, (2, 0, 1))
    new_psi_ex_zlo = b_zl * cpml_state.psi_ex_zlo + c_zl * curl_hy_dz_zlo_t
    correction_ex_zlo = jnp.transpose(new_psi_ex_zlo, (1, 2, 0))
    ex = ex.at[:, :, :n].add(-coeff_e * correction_ex_zlo)
    kappa_corr_ex_zlo = jnp.transpose((1.0 / k_zl - 1.0) * curl_hy_dz_zlo_t, (1, 2, 0))
    ex = ex.at[:, :, :n].add(-coeff_e * kappa_corr_ex_zlo)

    # --- Z-hi: Ex from dHy/dz ---
    hy_zhi = state.hy[:, :, -n:]
    hy_shifted_zhi = _shift_bwd(state.hy, 2)[:, :, -n:]
    curl_hy_dz_zhi = (hy_zhi - hy_shifted_zhi) / dz_hi
    curl_hy_dz_zhi_t = jnp.transpose(curl_hy_dz_zhi, (2, 0, 1))
    new_psi_ex_zhi = b_zh * cpml_state.psi_ex_zhi + c_zh * curl_hy_dz_zhi_t
    correction_ex_zhi = jnp.transpose(new_psi_ex_zhi, (1, 2, 0))
    ex = ex.at[:, :, -n:].add(-coeff_e * correction_ex_zhi)
    kappa_corr_ex_zhi = jnp.transpose((1.0 / k_zh - 1.0) * curl_hy_dz_zhi_t, (1, 2, 0))
    ex = ex.at[:, :, -n:].add(-coeff_e * kappa_corr_ex_zhi)

    # --- Z-lo: Ey from dHx/dz ---
    hx_zlo = state.hx[:, :, :n]
    hx_shifted_zlo = _shift_bwd(state.hx, 2)[:, :, :n]
    curl_hx_dz_zlo = (hx_zlo - hx_shifted_zlo) / dz_lo
    curl_hx_dz_zlo_t = jnp.transpose(curl_hx_dz_zlo, (2, 1, 0))
    new_psi_ey_zlo = b_zl * cpml_state.psi_ey_zlo + c_zl * curl_hx_dz_zlo_t
    correction_ey_zlo = jnp.transpose(new_psi_ey_zlo, (2, 1, 0))
    ey = ey.at[:, :, :n].add(coeff_e * correction_ey_zlo)
    kappa_corr_ey_zlo = jnp.transpose((1.0 / k_zl - 1.0) * curl_hx_dz_zlo_t, (2, 1, 0))
    ey = ey.at[:, :, :n].add(coeff_e * kappa_corr_ey_zlo)

    # --- Z-hi: Ey from dHx/dz ---
    hx_zhi = state.hx[:, :, -n:]
    hx_shifted_zhi = _shift_bwd(state.hx, 2)[:, :, -n:]
    curl_hx_dz_zhi = (hx_zhi - hx_shifted_zhi) / dz_hi
    curl_hx_dz_zhi_t = jnp.transpose(curl_hx_dz_zhi, (2, 1, 0))
    new_psi_ey_zhi = b_zh * cpml_state.psi_ey_zhi + c_zh * curl_hx_dz_zhi_t
    correction_ey_zhi = jnp.transpose(new_psi_ey_zhi, (2, 1, 0))
    ey = ey.at[:, :, -n:].add(coeff_e * correction_ey_zhi)
    kappa_corr_ey_zhi = jnp.transpose((1.0 / k_zh - 1.0) * curl_hx_dz_zhi_t, (2, 1, 0))
    ey = ey.at[:, :, -n:].add(coeff_e * kappa_corr_ey_zhi)

    new_state = state._replace(ex=ex, ey=ey, ez=ez)
    new_cpml = cpml_state._replace(
        psi_ey_xlo=new_psi_ey_xlo, psi_ey_xhi=new_psi_ey_xhi,
        psi_ez_xlo=new_psi_ez_xlo, psi_ez_xhi=new_psi_ez_xhi,
        psi_ex_ylo=new_psi_ex_ylo, psi_ex_yhi=new_psi_ex_yhi,
        psi_ez_ylo=new_psi_ez_ylo, psi_ez_yhi=new_psi_ez_yhi,
        psi_ex_zlo=new_psi_ex_zlo, psi_ex_zhi=new_psi_ex_zhi,
        psi_ey_zlo=new_psi_ey_zlo, psi_ey_zhi=new_psi_ey_zhi,
    )
    return new_state, new_cpml


def _apply_cpml_h_local_nu(state: FDTDState, cpml_params, cpml_state,
                           n_cpml: int, dt: float, ghost: int,
                           n_devices: int):
    """Per-rank slab-aware CPML H-field correction with NU per-axis dx.

    Mirror of :func:`_apply_cpml_e_local_nu` for the H field.  Same
    ghost-aware slicing, same x-face rank-conditional gating.
    """
    from rfx.boundaries.cpml import CPMLAxisParams

    coeff_h = cpml_coeff_h_vacuum(dt)  # vacuum coefficient (matches uniform path)

    if isinstance(cpml_params, CPMLAxisParams):
        # T7 PR1: read lo-face; scan body jnp.flip(px.b) preserves bit-identity.
        px, py, pz_lo, pz_hi = (
            cpml_params.x_lo, cpml_params.y_lo,
            cpml_params.z_lo, cpml_params.z_hi)
        dx_x = float(cpml_params.dx_x_lo)
        dx_y = float(cpml_params.dx_y_lo)
        dz_lo = float(cpml_params.dz_lo)
        dz_hi = float(cpml_params.dz_hi)
    else:
        px = py = pz_lo = pz_hi = cpml_params
        dx_x = dx_y = dz_lo = dz_hi = 1.0

    b_x = px.b[:, None, None]
    c_x = px.c[:, None, None]
    k_x = px.kappa[:, None, None]
    b_xr = jnp.flip(px.b)[:, None, None]
    c_xr = jnp.flip(px.c)[:, None, None]
    k_xr = jnp.flip(px.kappa)[:, None, None]
    b_y = py.b[:, None, None]
    c_y = py.c[:, None, None]
    k_y = py.kappa[:, None, None]
    b_yr = jnp.flip(py.b)[:, None, None]
    c_yr = jnp.flip(py.c)[:, None, None]
    k_yr = jnp.flip(py.kappa)[:, None, None]
    b_zl = pz_lo.b[:, None, None]
    c_zl = pz_lo.c[:, None, None]
    k_zl = pz_lo.kappa[:, None, None]
    b_zh = pz_hi.b[:, None, None]
    c_zh = pz_hi.c[:, None, None]
    k_zh = pz_hi.kappa[:, None, None]

    n = n_cpml
    g = ghost
    xlo = slice(g, g + n)
    xhi = slice(-(g + n), -g) if g > 0 else slice(-n, None)

    device_idx = lax.axis_index("x")
    is_first = (device_idx == 0)
    is_last = (device_idx == n_devices - 1)

    hx = state.hx
    hy = state.hy
    hz = state.hz

    # X-axis (rank-conditional)
    # --- X-lo: Hy from dEz/dx ---
    ez_xlo = state.ez[xlo, :, :]
    ez_shifted_xlo = _shift_fwd(state.ez, 0)[xlo, :, :]
    curl_ez_dx_xlo = (ez_shifted_xlo - ez_xlo) / dx_x
    new_psi_hy_xlo = b_x * cpml_state.psi_hy_xlo + c_x * curl_ez_dx_xlo
    hy_corr_xlo = (
        coeff_h * new_psi_hy_xlo
        + coeff_h * (1.0 / k_x - 1.0) * curl_ez_dx_xlo
    )
    hy_corr_xlo = jnp.where(is_first, hy_corr_xlo, 0.0)
    hy = hy.at[xlo, :, :].add(hy_corr_xlo)
    new_psi_hy_xlo = jnp.where(is_first, new_psi_hy_xlo, cpml_state.psi_hy_xlo)

    # --- X-hi: Hy from dEz/dx ---
    ez_xhi = state.ez[xhi, :, :]
    ez_shifted_xhi = _shift_fwd(state.ez, 0)[xhi, :, :]
    curl_ez_dx_xhi = (ez_shifted_xhi - ez_xhi) / dx_x
    new_psi_hy_xhi = b_xr * cpml_state.psi_hy_xhi + c_xr * curl_ez_dx_xhi
    hy_corr_xhi = (
        coeff_h * new_psi_hy_xhi
        + coeff_h * (1.0 / k_xr - 1.0) * curl_ez_dx_xhi
    )
    hy_corr_xhi = jnp.where(is_last, hy_corr_xhi, 0.0)
    hy = hy.at[xhi, :, :].add(hy_corr_xhi)
    new_psi_hy_xhi = jnp.where(is_last, new_psi_hy_xhi, cpml_state.psi_hy_xhi)

    # --- X-lo: Hz from dEy/dx ---
    ey_xlo = state.ey[xlo, :, :]
    ey_shifted_xlo = _shift_fwd(state.ey, 0)[xlo, :, :]
    curl_ey_dx_xlo = (ey_shifted_xlo - ey_xlo) / dx_x
    curl_ey_dx_xlo_t = jnp.transpose(curl_ey_dx_xlo, (0, 2, 1))
    new_psi_hz_xlo = b_x * cpml_state.psi_hz_xlo + c_x * curl_ey_dx_xlo_t
    correction_hz_xlo = jnp.transpose(new_psi_hz_xlo, (0, 2, 1))
    hz_corr_xlo = (
        -coeff_h * correction_hz_xlo
        - coeff_h * (1.0 / k_x - 1.0) * curl_ey_dx_xlo
    )
    hz_corr_xlo = jnp.where(is_first, hz_corr_xlo, 0.0)
    hz = hz.at[xlo, :, :].add(hz_corr_xlo)
    new_psi_hz_xlo = jnp.where(is_first, new_psi_hz_xlo, cpml_state.psi_hz_xlo)

    # --- X-hi: Hz from dEy/dx ---
    ey_xhi = state.ey[xhi, :, :]
    ey_shifted_xhi = _shift_fwd(state.ey, 0)[xhi, :, :]
    curl_ey_dx_xhi = (ey_shifted_xhi - ey_xhi) / dx_x
    curl_ey_dx_xhi_t = jnp.transpose(curl_ey_dx_xhi, (0, 2, 1))
    new_psi_hz_xhi = b_xr * cpml_state.psi_hz_xhi + c_xr * curl_ey_dx_xhi_t
    correction_hz_xhi = jnp.transpose(new_psi_hz_xhi, (0, 2, 1))
    hz_corr_xhi = (
        -coeff_h * correction_hz_xhi
        - coeff_h * (1.0 / k_xr - 1.0) * curl_ey_dx_xhi
    )
    hz_corr_xhi = jnp.where(is_last, hz_corr_xhi, 0.0)
    hz = hz.at[xhi, :, :].add(hz_corr_xhi)
    new_psi_hz_xhi = jnp.where(is_last, new_psi_hz_xhi, cpml_state.psi_hz_xhi)

    # Y-axis (every rank)
    # --- Y-lo: Hx from dEz/dy ---
    ez_ylo = state.ez[:, :n, :]
    ez_shifted_ylo = _shift_fwd(state.ez, 1)[:, :n, :]
    curl_ez_dy_ylo = (ez_shifted_ylo - ez_ylo) / dx_y
    curl_ez_dy_ylo_t = jnp.transpose(curl_ez_dy_ylo, (1, 0, 2))
    new_psi_hx_ylo = b_y * cpml_state.psi_hx_ylo + c_y * curl_ez_dy_ylo_t
    correction_hx_ylo = jnp.transpose(new_psi_hx_ylo, (1, 0, 2))
    hx = hx.at[:, :n, :].add(-coeff_h * correction_hx_ylo)
    kappa_corr_hx_ylo = jnp.transpose((1.0 / k_y - 1.0) * curl_ez_dy_ylo_t, (1, 0, 2))
    hx = hx.at[:, :n, :].add(-coeff_h * kappa_corr_hx_ylo)

    # --- Y-hi: Hx from dEz/dy ---
    ez_yhi = state.ez[:, -n:, :]
    ez_shifted_yhi = _shift_fwd(state.ez, 1)[:, -n:, :]
    curl_ez_dy_yhi = (ez_shifted_yhi - ez_yhi) / dx_y
    curl_ez_dy_yhi_t = jnp.transpose(curl_ez_dy_yhi, (1, 0, 2))
    new_psi_hx_yhi = b_yr * cpml_state.psi_hx_yhi + c_yr * curl_ez_dy_yhi_t
    correction_hx_yhi = jnp.transpose(new_psi_hx_yhi, (1, 0, 2))
    hx = hx.at[:, -n:, :].add(-coeff_h * correction_hx_yhi)
    kappa_corr_hx_yhi = jnp.transpose((1.0 / k_yr - 1.0) * curl_ez_dy_yhi_t, (1, 0, 2))
    hx = hx.at[:, -n:, :].add(-coeff_h * kappa_corr_hx_yhi)

    # --- Y-lo: Hz from dEx/dy ---
    ex_ylo = state.ex[:, :n, :]
    ex_shifted_ylo = _shift_fwd(state.ex, 1)[:, :n, :]
    curl_ex_dy_ylo = (ex_shifted_ylo - ex_ylo) / dx_y
    curl_ex_dy_ylo_t = jnp.transpose(curl_ex_dy_ylo, (1, 2, 0))
    new_psi_hz_ylo = b_y * cpml_state.psi_hz_ylo + c_y * curl_ex_dy_ylo_t
    correction_hz_ylo = jnp.transpose(new_psi_hz_ylo, (2, 0, 1))
    hz = hz.at[:, :n, :].add(coeff_h * correction_hz_ylo)
    kappa_corr_hz_ylo = jnp.transpose((1.0 / k_y - 1.0) * curl_ex_dy_ylo_t, (2, 0, 1))
    hz = hz.at[:, :n, :].add(coeff_h * kappa_corr_hz_ylo)

    # --- Y-hi: Hz from dEx/dy ---
    ex_yhi = state.ex[:, -n:, :]
    ex_shifted_yhi = _shift_fwd(state.ex, 1)[:, -n:, :]
    curl_ex_dy_yhi = (ex_shifted_yhi - ex_yhi) / dx_y
    curl_ex_dy_yhi_t = jnp.transpose(curl_ex_dy_yhi, (1, 2, 0))
    new_psi_hz_yhi = b_yr * cpml_state.psi_hz_yhi + c_yr * curl_ex_dy_yhi_t
    correction_hz_yhi = jnp.transpose(new_psi_hz_yhi, (2, 0, 1))
    hz = hz.at[:, -n:, :].add(coeff_h * correction_hz_yhi)
    kappa_corr_hz_yhi = jnp.transpose((1.0 / k_yr - 1.0) * curl_ex_dy_yhi_t, (2, 0, 1))
    hz = hz.at[:, -n:, :].add(coeff_h * kappa_corr_hz_yhi)

    # Z-axis (every rank)
    # --- Z-lo: Hx from dEy/dz ---
    ey_zlo = state.ey[:, :, :n]
    ey_shifted_zlo = _shift_fwd(state.ey, 2)[:, :, :n]
    curl_ey_dz_zlo = (ey_shifted_zlo - ey_zlo) / dz_lo
    curl_ey_dz_zlo_t = jnp.transpose(curl_ey_dz_zlo, (2, 0, 1))
    new_psi_hx_zlo = b_zl * cpml_state.psi_hx_zlo + c_zl * curl_ey_dz_zlo_t
    correction_hx_zlo = jnp.transpose(new_psi_hx_zlo, (1, 2, 0))
    hx = hx.at[:, :, :n].add(coeff_h * correction_hx_zlo)
    kappa_corr_hx_zlo = jnp.transpose((1.0 / k_zl - 1.0) * curl_ey_dz_zlo_t, (1, 2, 0))
    hx = hx.at[:, :, :n].add(coeff_h * kappa_corr_hx_zlo)

    # --- Z-hi: Hx from dEy/dz ---
    ey_zhi = state.ey[:, :, -n:]
    ey_shifted_zhi = _shift_fwd(state.ey, 2)[:, :, -n:]
    curl_ey_dz_zhi = (ey_shifted_zhi - ey_zhi) / dz_hi
    curl_ey_dz_zhi_t = jnp.transpose(curl_ey_dz_zhi, (2, 0, 1))
    new_psi_hx_zhi = b_zh * cpml_state.psi_hx_zhi + c_zh * curl_ey_dz_zhi_t
    correction_hx_zhi = jnp.transpose(new_psi_hx_zhi, (1, 2, 0))
    hx = hx.at[:, :, -n:].add(coeff_h * correction_hx_zhi)
    kappa_corr_hx_zhi = jnp.transpose((1.0 / k_zh - 1.0) * curl_ey_dz_zhi_t, (1, 2, 0))
    hx = hx.at[:, :, -n:].add(coeff_h * kappa_corr_hx_zhi)

    # --- Z-lo: Hy from dEx/dz ---
    ex_zlo = state.ex[:, :, :n]
    ex_shifted_zlo = _shift_fwd(state.ex, 2)[:, :, :n]
    curl_ex_dz_zlo = (ex_shifted_zlo - ex_zlo) / dz_lo
    curl_ex_dz_zlo_t = jnp.transpose(curl_ex_dz_zlo, (2, 1, 0))
    new_psi_hy_zlo = b_zl * cpml_state.psi_hy_zlo + c_zl * curl_ex_dz_zlo_t
    correction_hy_zlo = jnp.transpose(new_psi_hy_zlo, (2, 1, 0))
    hy = hy.at[:, :, :n].add(-coeff_h * correction_hy_zlo)
    kappa_corr_hy_zlo = jnp.transpose((1.0 / k_zl - 1.0) * curl_ex_dz_zlo_t, (2, 1, 0))
    hy = hy.at[:, :, :n].add(-coeff_h * kappa_corr_hy_zlo)

    # --- Z-hi: Hy from dEx/dz ---
    ex_zhi = state.ex[:, :, -n:]
    ex_shifted_zhi = _shift_fwd(state.ex, 2)[:, :, -n:]
    curl_ex_dz_zhi = (ex_shifted_zhi - ex_zhi) / dz_hi
    curl_ex_dz_zhi_t = jnp.transpose(curl_ex_dz_zhi, (2, 1, 0))
    new_psi_hy_zhi = b_zh * cpml_state.psi_hy_zhi + c_zh * curl_ex_dz_zhi_t
    correction_hy_zhi = jnp.transpose(new_psi_hy_zhi, (2, 1, 0))
    hy = hy.at[:, :, -n:].add(-coeff_h * correction_hy_zhi)
    kappa_corr_hy_zhi = jnp.transpose((1.0 / k_zh - 1.0) * curl_ex_dz_zhi_t, (2, 1, 0))
    hy = hy.at[:, :, -n:].add(-coeff_h * kappa_corr_hy_zhi)

    new_state = state._replace(hx=hx, hy=hy, hz=hz)
    new_cpml = cpml_state._replace(
        psi_hy_xlo=new_psi_hy_xlo, psi_hy_xhi=new_psi_hy_xhi,
        psi_hz_xlo=new_psi_hz_xlo, psi_hz_xhi=new_psi_hz_xhi,
        psi_hx_ylo=new_psi_hx_ylo, psi_hx_yhi=new_psi_hx_yhi,
        psi_hz_ylo=new_psi_hz_ylo, psi_hz_yhi=new_psi_hz_yhi,
        psi_hx_zlo=new_psi_hx_zlo, psi_hx_zhi=new_psi_hx_zhi,
        psi_hy_zlo=new_psi_hy_zlo, psi_hy_zhi=new_psi_hy_zhi,
    )
    return new_state, new_cpml


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
    sharded_design_mask=None,
    emit_time_series: bool = True,
    pmc_faces: frozenset = frozenset(),
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
        6. Source injection (rank-conditional)       via shard_map
        7. Ghost exchange of E                       via lax.ppermute
        8. apply_pec on physical domain faces        via shard_map
        9. apply_pec_mask (geometry + override)      via shard_map
        9b. apply_pec_occupancy (soft PEC)           via shard_map  [if occupancy]
       10. Probe accumulation (rank-conditional sum) via lax.psum

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
         previous step's seventh stage).
      4. Continue with CPML-E, sources, E ghost exchange, PEC.

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
        Phase 2D Debye dispersion.  Both arrays must already be
        x-slab-sharded via :func:`shard_debye_coeffs_x_slab` /
        :func:`shard_debye_state_x_slab`.  ``None`` selects the
        non-dispersive path.
    lorentz : (LorentzCoeffs, LorentzState) tuple or None, optional
        Phase 2D Lorentz/Drude dispersion.  Both arrays must already be
        x-slab-sharded via :func:`shard_lorentz_coeffs_x_slab` /
        :func:`shard_lorentz_state_x_slab`.  ``None`` selects the
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
        :func:`shard_pec_occupancy_x_slab` to build it.  Applied after
        the hard ``sharded_pec_mask`` and before probe accumulation,
        mirroring the single-device ordering in
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
    sharded_design_mask : jnp.ndarray bool or None, optional
        Phase 2F design-mask stop-gradient (mirrors single-device
        :issue:`#41`).  x-slab sharded boolean mask with the same
        layout as ``sharded_pec_mask`` (use
        :func:`shard_design_mask_x_slab` to build it).  Cells where the
        mask is ``True`` keep gradient; cells where the mask is
        ``False`` have ``stop_gradient`` applied at each step before
        the carry-out.  Forward physics is bit-identical
        (``stop_gradient`` is forward-identity); backward memory and
        wall-time scale with mask occupancy instead of grid volume.
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
    """
    if n_devices != sharded_grid.n_devices:
        raise ValueError(
            f"n_devices={n_devices} != sharded_grid.n_devices="
            f"{sharded_grid.n_devices}"
        )
    if exchange_interval != 1:
        raise NotImplementedError(
            "exchange_interval > 1 reserved for Phase 2E; only 1 supported."
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
    use_dispersion = use_debye or use_lorentz
    use_cpml = cpml_params is not None
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

    # Phase 2F validation
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0; got {n_warmup}")
    if n_warmup >= n_steps:
        raise ValueError(
            f"n_warmup ({n_warmup}) must be < n_steps ({n_steps})"
        )
    if checkpoint_every is not None and int(checkpoint_every) <= 0:
        raise ValueError(
            f"checkpoint_every must be a positive int or None; "
            f"got {checkpoint_every}"
        )

    sources = list(sources) if sources is not None else []
    probes = list(probes) if probes is not None else []

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
    ghost = sharded_grid.ghost_width
    ny = sharded_grid.ny
    nz = sharded_grid.nz
    nx_padded = sharded_grid.nx_padded
    pad_x = sharded_grid.pad_x
    dt = sharded_grid.dt

    # ------------------------------------------------------------------
    # Sharded inverse-spacing arrays (1-D)
    # ------------------------------------------------------------------
    inv_dx_slabs = split_1d_with_ghost(
        sharded_grid.inv_dx_global, n_devices, nx_per, nx_local, ghost,
        pad_value=1.0,
    )
    inv_dx_h_slabs = split_1d_with_ghost(
        sharded_grid.inv_dx_h_global, n_devices, nx_per, nx_local, ghost,
        pad_value=0.0,
    )
    inv_dx_sharded = jax.device_put(
        inv_dx_slabs.reshape(n_devices * nx_local), shd)
    inv_dx_h_sharded = jax.device_put(
        inv_dx_h_slabs.reshape(n_devices * nx_local), shd)
    inv_dy_rep = jax.device_put(sharded_grid.inv_dy, rep)
    inv_dy_h_rep = jax.device_put(sharded_grid.inv_dy_h, rep)
    inv_dz_rep = jax.device_put(sharded_grid.inv_dz, rep)
    inv_dz_h_rep = jax.device_put(sharded_grid.inv_dz_h, rep)

    # ------------------------------------------------------------------
    # Initial state — full padded domain, then slab-split + shard
    # ------------------------------------------------------------------
    from rfx.core.yee import init_state
    from rfx.runners.distributed import _split_state

    full_state = init_state((nx_padded, ny, nz))
    state_slabs = _split_state(full_state, n_devices, ghost)

    def _shard_stacked(arr):
        n_dev = arr.shape[0]
        rest = arr.shape[1:]
        return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)

    sharded_state = FDTDState(
        ex=_shard_stacked(state_slabs.ex),
        ey=_shard_stacked(state_slabs.ey),
        ez=_shard_stacked(state_slabs.ez),
        hx=_shard_stacked(state_slabs.hx),
        hy=_shard_stacked(state_slabs.hy),
        hz=_shard_stacked(state_slabs.hz),
        step=jax.device_put(jnp.int32(0), rep),
    )

    # ------------------------------------------------------------------
    # Source / probe routing — V3 bullet (Phase 2A coordinate convention)
    # ------------------------------------------------------------------
    src_device_ids = []
    src_local_specs = []
    for s in sources:
        dev_id = s.i // nx_per
        local_i = (s.i % nx_per) + ghost
        src_device_ids.append(int(dev_id))
        src_local_specs.append((int(local_i), int(s.j), int(s.k), s.component))

    prb_device_ids = []
    prb_local_specs = []
    for p in probes:
        dev_id = p.i // nx_per
        local_i = (p.i % nx_per) + ghost
        prb_device_ids.append(int(dev_id))
        prb_local_specs.append((int(local_i), int(p.j), int(p.k), p.component))

    n_src = len(sources)
    n_prb = len(probes)

    if n_src > 0:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    src_waveforms_rep = jax.device_put(src_waveforms, rep)

    # ------------------------------------------------------------------
    # Per-step shmap-wrapped helpers
    # ------------------------------------------------------------------
    def _update_h_shmap(st, mat):
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

    def _update_e_shmap(st, mat):
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
        ):
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
            out_specs_full = base_out + db_out + lr_out

            @partial(
                shard_map,
                mesh=mesh,
                in_specs=in_specs_full,
                out_specs=out_specs_full,
                check_rep=False,
            )
            def _e_disp(*args):
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
                        DebyeCoeffs(ca=d_ca, cb=d_cb, cc=d_cc,
                                    alpha=d_alpha, beta=d_beta),
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
                        LorentzCoeffs(ca=l_ca, cb=l_cb, cc=l_cc,
                                      a=l_a, b=l_b, c=l_c),
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
                )

                outs = (new_st.ex, new_st.ey, new_st.ez, new_st.step)
                if _has_db:
                    outs = outs + (new_db.px, new_db.py, new_db.pz)
                if _has_lr:
                    outs = outs + (
                        new_lr.px, new_lr.py, new_lr.pz,
                        new_lr.px_prev, new_lr.py_prev, new_lr.pz_prev,
                    )
                return outs

            # Build flat input tuple matching in_specs_full ordering
            call_args = [
                st.ex, st.ey, st.ez,
                st.hx, st.hy, st.hz,
                st.step,
                mat.eps_r, mat.sigma, mat.mu_r,
                inv_dx_sharded, inv_dy_rep, inv_dz_rep,
                e_old_ex, e_old_ey, e_old_ez,
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

            results = _e_disp(*call_args)

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
            return new_state, new_db_st, new_lr_st

    def _inject_sources_shmap(st, src_vals_step):
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

    def _sample_probes_shmap(st):
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

    # ------------------------------------------------------------------
    # Phase 2C: shmap-wrapped CPML helpers (no-op when CPML disabled)
    # ------------------------------------------------------------------
    n_cpml_local = int(sharded_grid.cpml_layers) if use_cpml else 0

    def _apply_cpml_h_shmap(st, cs):
        @partial(
            shard_map,
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
               psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi):
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
            new_st, new_cs = _apply_cpml_h_local_nu(
                _st, cpml_params, _cs, n_cpml_local, dt, ghost, n_devices,
            )
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
            st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
            cs.psi_hy_xlo, cs.psi_hy_xhi, cs.psi_hz_xlo, cs.psi_hz_xhi,
            cs.psi_hx_ylo, cs.psi_hx_yhi, cs.psi_hz_ylo, cs.psi_hz_yhi,
            cs.psi_hx_zlo, cs.psi_hx_zhi, cs.psi_hy_zlo, cs.psi_hy_zhi,
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

    def _apply_cpml_e_shmap(st, cs):
        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"), P("x"),  # x-face psi
                P("x"), P("x"), P("x"), P("x"),  # y-face psi
                P("x"), P("x"), P("x"), P("x"),  # z-face psi
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
               psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi):
            _st = FDTDState(ex=ex, ey=ey, ez=ez,
                            hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
            _cs = cs._replace(
                psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
                psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
                psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
                psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
                psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
                psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
            )
            new_st, new_cs = _apply_cpml_e_local_nu(
                _st, cpml_params, _cs, n_cpml_local, dt, ghost, n_devices,
            )
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
            st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
            cs.psi_ey_xlo, cs.psi_ey_xhi, cs.psi_ez_xlo, cs.psi_ez_xhi,
            cs.psi_ex_ylo, cs.psi_ex_yhi, cs.psi_ez_ylo, cs.psi_ez_yhi,
            cs.psi_ex_zlo, cs.psi_ex_zhi, cs.psi_ey_zlo, cs.psi_ey_zhi,
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
    def step_fn(carry, xs):
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
        st = _update_h_shmap(st, sharded_materials)

        # 2. Phase 2C: CPML H correction (after H, before E exchange).
        if use_cpml:
            st, cs = _apply_cpml_h_shmap(st, cs)

        # 2b. PMC face (H-tangential = 0) before H ghost exchange so the
        #     zero propagates to neighbours via the exchange. H-half hook
        #     per OQ9 — mirrors single-device ordering in simulation.py
        #     (apply_cpml_h -> apply_pmc_faces). NOT the E-half position
        #     of PEC: PMC must fire before the next E-update reads H via
        #     curl.
        st = _apply_pmc_face_nu_shmap(
            st, mesh, n_devices, nx_local, pmc_faces)

        # 3. Ghost exchange of H so the upcoming E update sees the
        #    neighbour rank's H at the seam.
        st = _exchange_h_ghosts_nu(st, mesh, n_devices)

        # 4. E update — Phase 2D dispatch:
        #    - no dispersion: standard NU E update (Phase 2B path)
        #    - Debye-only / Lorentz-only / mixed: dispersive NU ADE
        #      with the snapshotted ex_old/ey_old/ez_old (NEVER the
        #      post-ghost-exchange E from the previous step's stage 7).
        if use_dispersion:
            st, db_st, lr_st = _update_e_dispersive_shmap(
                st, sharded_materials, db_st, lr_st,
                ex_old_snapshot, ey_old_snapshot, ez_old_snapshot,
            )
        else:
            st = _update_e_shmap(st, sharded_materials)

        # 5. Phase 2C: CPML E correction (after E, before sources/PEC).
        if use_cpml:
            st, cs = _apply_cpml_e_shmap(st, cs)

        # 6. Source injection (rank-conditional via shard_map)
        st = _inject_sources_shmap(st, src_vals)

        # 7. Ghost exchange of E so the next step's H update sees the
        #    neighbour rank's E at the seam.
        st = _exchange_e_ghosts_nu(st, mesh, n_devices)

        # 8. PEC on physical domain faces (X-faces are rank-conditional).
        st = _apply_pec_face_nu_shmap(st, mesh, n_devices, nx_local)

        # 9. PEC mask zeroing (geometry + override union).  No-op when
        #    sharded_pec_mask is None.
        if sharded_pec_mask is not None:
            st = _apply_pec_mask_nu_shmap(
                st, sharded_pec_mask, mesh, n_devices, nx_local)

        # 9b. Phase 2E: soft-PEC occupancy (differentiable analogue of the
        #     hard mask).  Mirrors single-device ordering in
        #     ``rfx.nonuniform.run_nonuniform``: applied after the hard
        #     mask and before probe accumulation.  Seam ghost rows are
        #     zeroed inside the helper so a seam cell is applied exactly
        #     once.
        if sharded_pec_occupancy is not None:
            st = _apply_pec_occupancy_nu_shmap(
                st, sharded_pec_occupancy, mesh, n_devices, nx_local)

        # 10. Probe accumulation (rank-conditional sample + lax.psum).
        #     Phase 2F emit_time_series=False: skip the probe-sample
        #     accumulation entirely so the AD tape is not loaded with
        #     per-step probe entries.  ``probe_out`` becomes a 0-length
        #     array (still a valid scan output, just empty).
        if emit_time_series:
            probe_out = _sample_probes_shmap(st)
        else:
            probe_out = jnp.zeros(0, dtype=jnp.float32)

        # 9c. Phase 2F design-mask stop-gradient (mirrors single-device
        #     :issue:`#41`).  Apply ``stop_gradient`` to fields outside
        #     the design region BEFORE carry-out so the backward tape
        #     never accumulates entries for cells whose ``eps`` does not
        #     depend on the optimisation variable.  Forward physics is
        #     unchanged (``stop_gradient`` is forward-identity); backward
        #     memory + wall-time scale with mask occupancy.  The mask
        #     itself is x-sharded with the same slab+ghost layout as the
        #     state, so the elementwise ``jnp.where`` runs locally on
        #     each rank with no cross-rank communication.
        if sharded_design_mask is not None:
            sg = jax.lax.stop_gradient
            st = st._replace(
                ex=jnp.where(sharded_design_mask, st.ex, sg(st.ex)),
                ey=jnp.where(sharded_design_mask, st.ey, sg(st.ey)),
                ez=jnp.where(sharded_design_mask, st.ez, sg(st.ez)),
                hx=jnp.where(sharded_design_mask, st.hx, sg(st.hx)),
                hy=jnp.where(sharded_design_mask, st.hy, sg(st.hy)),
                hz=jnp.where(sharded_design_mask, st.hz, sg(st.hz)),
            )

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
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
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

    @jax.jit
    def run_fn(c0):
        # Optional warmup scan: stop_gradient the carry at boundary so
        # AD does not see the warmup steps.  Probe samples from the
        # warmup phase are also stop_gradient'd (they're just metadata
        # at this point — gradient through them would be a tape leak).
        if warmup_xs is not None:
            warmup_final, warmup_ys = lax.scan(step_fn, c0, warmup_xs)
            c0_opt = jax.tree_util.tree_map(lax.stop_gradient, warmup_final)
            warmup_ys = lax.stop_gradient(warmup_ys)
        else:
            c0_opt = c0
            warmup_ys = None

        # Optimize scan: either single scan (Phase 2A-2E path) or
        # scan-of-scan with checkpoint(segment_body) (Phase 2F segmented
        # remat).  The pad-+-tail-discard semantics mirror single-device
        # ``run_nonuniform``.
        if use_segmented:
            chunk = int(checkpoint_every)
            n_seg = (n_opt + chunk - 1) // chunk
            pad = n_seg * chunk - n_opt
            opt_steps = opt_xs[0]
            opt_src = opt_xs[1]
            if pad > 0:
                start_step = jnp.int32(opt_steps[0])
                steps_padded = jnp.arange(
                    start_step, start_step + n_seg * chunk,
                    dtype=jnp.int32,
                )
                n_sources_local = opt_src.shape[1]
                src_pad = jnp.zeros(
                    (pad, n_sources_local), dtype=opt_src.dtype)
                src_padded = jnp.concatenate([opt_src, src_pad], axis=0)
            else:
                steps_padded = opt_steps
                src_padded = opt_src

            seg_steps = steps_padded.reshape(n_seg, chunk)
            seg_src = src_padded.reshape(n_seg, chunk, src_padded.shape[1])

            def segment_body(carry, segment_xs):
                # Inner scan over a single segment of ``chunk`` steps.
                return lax.scan(step_fn, carry, segment_xs)

            seg_body = jax.checkpoint(segment_body)
            final_, seg_ys = lax.scan(
                seg_body, c0_opt, (seg_steps, seg_src)
            )
            # seg_ys shape: (n_seg, chunk, ...).  Flatten + tail-discard.
            opt_ys_flat = seg_ys.reshape(
                (n_seg * chunk,) + seg_ys.shape[2:]
            )
            opt_ys = opt_ys_flat[:n_opt]
        else:
            final_, opt_ys = lax.scan(step_fn, c0_opt, opt_xs)

        if warmup_ys is not None:
            full_ys = jnp.concatenate([warmup_ys, opt_ys], axis=0)
        else:
            full_ys = opt_ys
        return final_, full_ys

    final_carry, probe_ts = run_fn(carry_init)
    final_state_sharded = final_carry["fdtd"]
    final_cpml_sharded = final_carry.get("cpml") if use_cpml else None
    final_debye_sharded = final_carry.get("debye") if use_debye else None
    final_lorentz_sharded = final_carry.get("lorentz") if use_lorentz else None

    # ------------------------------------------------------------------
    # Gather sharded state -> full-domain (nx, ny, nz)
    # ------------------------------------------------------------------
    from rfx.runners.distributed import gather_array_x

    def _unstack_and_gather(sharded_arr):
        # Phase 2F: must remain traceable under ``jax.grad``.  The prior
        # ``np.array(sharded_arr)`` call broke whenever a caller wrapped
        # the runner in ``jax.grad`` to drive an objective from the
        # gathered ``final_state`` arrays.  Use pure JAX reshape +
        # ``gather_array_x`` (already JAX-friendly) so the gather stays
        # in the JAX trace.
        total_x = sharded_arr.shape[0]
        assert total_x == n_devices * nx_local, (
            f"unstack: total_x={total_x} != n_devices*nx_local={n_devices * nx_local}"
        )
        stacked = jnp.reshape(
            sharded_arr,
            (n_devices, nx_local) + tuple(sharded_arr.shape[1:]),
        )
        gathered = gather_array_x(stacked, ghost)
        if pad_x > 0:
            gathered = gathered[: sharded_grid.nx]
        return gathered

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

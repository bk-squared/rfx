"""Non-uniform Yee grid FDTD runner.

Supports spatially-varying dx, dy, dz profiles. The most common use
is dz-graded meshes for thin-substrate structures where z-resolution
must be fine near the substrate but coarse in the air region; dx/dy
graded meshes additionally allow fine cells near metal edges
(fringing-field physics in patch antennas, microstrip filters, etc.)
without paying the cost of a uniform fine mesh everywhere.

Back-compat: `make_nonuniform_grid(domain_xy, dz_profile, dx, ...)`
with a scalar `dx` still produces a uniform-xy grid; only `dz` is
graded. Pass `dx_profile=` / `dy_profile=` to enable per-cell x/y
spacing.

Uses update_h_nu / update_e_nu from core/yee.py with pre-computed
per-axis inverse spacing arrays. Fully JIT-compiled via jax.lax.scan.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    update_h_nu, update_e_nu, EPS_0, MU_0,
)
from rfx.boundaries.pec import apply_pec, apply_pec_mask, apply_pec_occupancy
from rfx.core.jax_utils import is_tracer

C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))


class NonUniformGrid(NamedTuple):
    """Non-uniform grid with per-axis cell-size arrays.

    ``dx`` / ``dy`` are the BOUNDARY cell sizes (used by CPML and any
    legacy code that reads a scalar spacing); ``dx_arr`` / ``dy_arr`` /
    ``dz`` hold the per-cell spacings. In the uniform-xy case,
    ``dx_arr`` is ``jnp.full(nx, dx)`` and ``dy_arr`` is analogous.

    ``pad_{axis}_{lo|hi}`` mirror ``rfx.grid.Grid`` per-face padding —
    a face whose token is ``pmc``/``pec``/``periodic`` gets 0 cells on
    that side even when the axis as a whole uses CPML. For back-compat
    the six fields default to ``cpml_layers`` on every face (the
    pre-per-face-allocation symmetric layout).
    """
    nx: int
    ny: int
    nz: int
    dx: float              # BOUNDARY x cell size (CPML + legacy scalars)
    dy: float              # BOUNDARY y cell size
    dx_arr: jnp.ndarray    # (nx,) — per-cell dx (includes CPML padding)
    dy_arr: jnp.ndarray    # (ny,) — per-cell dy
    dz: jnp.ndarray        # (nz,) z cell sizes (already per-cell)
    dt: float | jax.Array  # timestep; eager float in normal use, traced
                           # jax.Array on the run_nonuniform traced path.
                           # Readers must not apply float() / .item() without
                           # an is_tracer(dt) guard or a host-boundary context.
    cpml_layers: int
    # Cell-size arrays are padded AND carry one trailing duplicate cell so N
    # cells are bounded by N+1 E-nodes (#562, see _append_bounding_node). The
    # duplicate is a node-provider, not physical extent: its H term is the one
    # the stencil zeroes. Read extents from node positions
    # (coords_from_nonuniform_grid, or cumsum edges up to index nx-1) — NEVER
    # from sum(dx_arr), which overshoots by that trailing cell.
    # Pre-computed inverse spacing arrays (length N, padded).
    # CORE-C2: inv_d* feed the E update (mean spacing), inv_d*_h feed
    # the H update (local cell width). See _profile_to_inv_arrays.
    inv_dx: jnp.ndarray    # (nx,) — E update: 2/(dx[i-1]+dx[i]); [0]=1/dx[0]
    inv_dy: jnp.ndarray    # (ny,) — E update: 2/(dy[j-1]+dy[j]); [0]=1/dy[0]
    inv_dz: jnp.ndarray    # (nz,) — E update: 2/(dz[k-1]+dz[k]); [0]=1/dz[0]
    inv_dx_h: jnp.ndarray  # (nx,) — H update: 1/dx[i]; [nx-1]=0
    inv_dy_h: jnp.ndarray  # (ny,) — H update: 1/dy[j]; [ny-1]=0
    inv_dz_h: jnp.ndarray  # (nz,) — H update: 1/dz[k]; [nz-1]=0
    # Per-face CPML padding (PMC+CPML composition fix, 2026-04)
    pad_x_lo: int = 0
    pad_x_hi: int = 0
    pad_y_lo: int = 0
    pad_y_hi: int = 0
    pad_z_lo: int = 0
    pad_z_hi: int = 0

    @property
    def shape(self):
        """Grid shape (nx, ny, nz) — duck-typing compatible with Grid."""
        return (self.nx, self.ny, self.nz)

    @property
    def axis_pads(self):
        """Leading (``lo``) per-axis pad — the number coordinate-offset
        callers subtract from array indices to recover user coords."""
        return (self.pad_x_lo, self.pad_y_lo, self.pad_z_lo)




def _pad_profile(profile, pad_lo: int, pad_hi: int | None = None):
    """Pad a 1-D cell-size profile with CPML cells on each end.

    ``pad_lo`` and ``pad_hi`` may differ (per-face allocation, 2026-04 —
    a PMC/PEC face gets 0 cells on that side while the opposing CPML
    face keeps its allocation). If ``pad_hi`` is omitted the symmetric
    ``pad_lo`` count is used on both ends (pre-per-face-allocation behaviour).

    CPML uses constant spacing matching the boundary cell size, so the
    ``pad_lo`` cells on the leading side carry ``profile[0]`` and the
    ``pad_hi`` cells on the trailing side carry ``profile[-1]``.

    When ``profile`` is a JAX tracer the padding stays in-trace (needed
    for ``jax.grad`` w.r.t. ``dz_profile`` — mesh-as-design-variable).
    Otherwise the numpy path is used, preserving the Python-float ``dt``
    that Simulation-level callers depend on.
    """
    if pad_hi is None:
        pad_hi = pad_lo
    if is_tracer(profile):
        prof = jnp.asarray(profile, dtype=jnp.float32)
        lo_pad = jnp.full(pad_lo, prof[0])
        hi_pad = jnp.full(pad_hi, prof[-1])
        return jnp.concatenate([lo_pad, prof, hi_pad])
    lo_pad = np.full(pad_lo, float(profile[0]))
    hi_pad = np.full(pad_hi, float(profile[-1]))
    return np.concatenate([lo_pad, np.asarray(profile, dtype=np.float64), hi_pad])


def _append_bounding_node(profile_full):
    """Append one duplicate boundary cell so an N-cell profile yields N+1
    E-nodes — the node count the uniform ``Grid`` allocates for N cells.

    N cells are bounded by N+1 nodes, and the E-nodes this grid steps sit on
    cell EDGES (``inv_d_e[i] = 2/(d[i-1]+d[i])`` is the dual spacing of an
    edge-centred node). Without this the array carried only N nodes, the
    stencil zeroed the last cell's H term (``inv_d_h[N-1] = 0``), and the
    realized wall-to-wall extent came out ``sum(d) - d[-1]`` — one cell
    short of what the caller asked for (#562: +2.47 % of TM110 on a
    45 x 39-cell PEC box, +37 MHz of WR-90 TE101 centre frequency from the
    guide-width axis alone).

    Duplicating the boundary cell is what makes the far node's coefficient
    the mirror of the near one's: ``inv_d_e[N] = 2/(d[N-1]+d[N])``, which is
    ``1/d[N-1]`` exactly when ``d[N] == d[N-1]``, matching the existing
    ``inv_d_e[0] = 1/d[0]`` boundary treatment. The appended cell's own H
    term is the one the stencil zeroes, so it adds a node without adding a
    cell of physical extent.
    """
    if is_tracer(profile_full):
        return jnp.concatenate([profile_full, profile_full[-1:]])
    return np.concatenate([profile_full, profile_full[-1:]])


def interior_cells(d_full, pad_lo: int, pad_hi: int):
    """The physical interior cells of a padded NU cell-size array.

    The array is ``[lo pad | interior | hi pad | bounding-node duplicate]``
    (#562, see ``_append_bounding_node``), so the interior ends one entry
    before ``len - pad_hi``. Slicing without that ``-1`` pulls in a pad cell
    — or the duplicate itself on a face with no pad — and overstates the
    extent by one cell, which is the same class of error #562 was.

    Returns the CELLS; ``np.insert(np.cumsum(cells), 0, 0.0)`` then gives the
    ``N+1`` interior NODE positions, the last of which lands exactly on the
    requested domain face.
    """
    return d_full[pad_lo : len(d_full) - pad_hi - 1]


def node_positions_from_profile(profile):
    """Interior E-node positions for a RAW (unpadded) cell-size profile.

    ``N`` cells give ``N+1`` nodes, the first at 0 and the last exactly on the
    profile's total extent — the same positions
    ``coords_from_nonuniform_grid`` produces for the interior of a built grid.
    Public because callers outside this module legitimately need the
    convention (the #325 graded-rasterization preflight models what the
    rasterizer will do, and modelling it with a private-import composition or,
    worse, a hand-rolled copy, is how that check drifted in the first place —
    #562 review F2, #568).
    """
    from rfx.geometry.rasterize_grid import _axis_node_positions
    return _axis_node_positions(_append_bounding_node(profile), 0)


def _profile_to_inv_arrays(profile_full: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(inv_d_e, inv_d_h)`` — the E-update and H-update inverse
    cell-spacing arrays for a padded 1-D cell-size profile ``d``.

    CORE-C2 fix (2026-05-16). The non-uniform Yee stencil needs a
    *different* metric for each update:

    * **H update** — the H-curl differences two E nodes that straddle
      one cell, so it divides by the **local cell width**::

          inv_d_h[k] = 1/d[k]       (k < N-1);   inv_d_h[N-1] = 0

    * **E update** — the E-curl differences two H cell-centres, whose
      separation is the **mean** of the two adjacent cell widths::

          inv_d_e[k] = 2/(d[k-1]+d[k])   (k >= 1);   inv_d_e[0] = 1/d[0]

    On a uniform mesh both collapse to ``1/d`` so the uniform path is
    bit-identical. The pre-fix code had the two swapped (H got the mean,
    E got the local width) — a stencil inconsistency that scaled the
    curl by ``2 d[k]/(d[k]+d[k±1])`` on every graded cell. The boundary
    entries (``inv_d_h[N-1]=0``, ``inv_d_e[0]=1/d[0]``) reproduce the
    prior boundary behaviour exactly, so PEC/CPML faces are untouched.

    The return order ``(inv_d_e, inv_d_h)`` matches the caller's
    ``inv_dx, inv_dx_h = _profile_to_inv_arrays(...)`` unpacking: the
    first slot feeds the E update, the second feeds ``update_h_nu``.
    """
    arr = jnp.asarray(profile_full, dtype=jnp.float32)
    inv_local = 1.0 / arr                          # 1/d[k]
    inv_mean = 2.0 / (arr[:-1] + arr[1:])          # 2/(d[k]+d[k+1])
    # H update: local cell width; trailing 0 (forward-diff has no d[N]).
    inv_d_h = jnp.concatenate([inv_local[:-1], jnp.zeros(1, dtype=jnp.float32)])
    # E update: mean of (d[k-1], d[k]); leading 1/d[0] (backward-diff
    # boundary — reproduces the pre-fix value so the face cell is
    # unchanged).
    inv_d_e = jnp.concatenate([inv_local[:1], inv_mean])
    return inv_d_e, inv_d_h


def e_node_dual_spacings(profile_full):
    """Per-node DUAL spacing — the length an E node's material acts over.

    ``inv_d_e[k] = 2/(d[k-1]+d[k])`` (``inv_d_e[0] = 1/d[0]``) is the metric
    the non-uniform E update divides the curl by, so the control volume of
    the E node at index ``k`` extends half a cell either side::

        dual[k] = (d[k-1] + d[k]) / 2     (k >= 1);   dual[0] = d[0]

    which is exactly ``1/inv_d_e``. Anything folded into ``materials.sigma``
    at an E node is a volumetric quantity whose realized SHEET value is
    ``sigma * dual[k]`` — never ``sigma * d[k]``. Dividing a sheet fold by
    the primal cell ``d[k]`` instead realizes ``Rs * d[k]/dual[k]``, which
    is right only where the two adjacent cells are equal; on a grading
    transition it is wrong by up to the local cell ratio (issue #669 review:
    a Leontovich sheet on a 0.25/0.50 mm transition node measured an
    attenuation ratio of 1.2021 against the matched-mesh case, and 0.6214 on
    a 1.00/0.25 mm one, where a mesh-independent sheet must give 1.000).

    Computed as ``(d[k-1]+d[k])/2`` rather than ``1/inv_d_e`` so a uniform
    profile returns the cell size bit-exactly (``0.5*(d+d) == d`` in
    floating point, while ``1/(2/(d+d))`` need not be). ``jnp`` throughout so
    a traced ``dz_profile`` (mesh-as-design-variable) stays differentiable.
    """
    arr = jnp.asarray(profile_full)
    return jnp.concatenate([arr[:1], 0.5 * (arr[:-1] + arr[1:])])


def e_node_dual_spacing_at(profile_full, k: int):
    """Scalar dual spacing at node ``k`` — same rule as
    :func:`e_node_dual_spacings`, evaluated on the caller's array WITHOUT the
    dtype-normalizing ``jnp.asarray``.

    Two callers need that: a float64 numpy profile must keep float64 so a
    uniform mesh stays bit-identical to the old primal spelling (a float32
    round-trip alone moves the last bits), and a traced profile must stay a
    tracer so the mesh-as-design-variable path keeps its gradient.

    ``tests/test_nonuniform_source_port_dual_spacing.py`` pins this against
    :func:`e_node_dual_spacings` element-by-element, so the two spellings
    cannot drift.
    """
    if k <= 0:
        return profile_full[0]
    return 0.5 * (profile_full[k - 1] + profile_full[k])


def make_nonuniform_grid(
    domain_xy: tuple[float, float],
    dz_profile: np.ndarray,
    dx: float,
    cpml_layers: int = 8,
    *,
    dx_profile: np.ndarray | None = None,
    dy_profile: np.ndarray | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    cpml_axes: str = "xyz",
) -> NonUniformGrid:
    """Create a non-uniform Yee grid.

    Parameters
    ----------
    domain_xy : (Lx, Ly) in metres
        Only used when ``dx_profile`` / ``dy_profile`` are None — in
        that case the xy mesh is uniform with spacing ``dx``.
    dz_profile : 1D array of z cell sizes in metres (physical domain only)
    dx : float
        Boundary cell size (also used for CPML padding and as the
        uniform-xy spacing when no xy profile is provided).
    cpml_layers : int
        Number of CPML cells added to each face (when that face is
        absorbing — see ``pec_faces`` / ``pmc_faces``).
    dx_profile, dy_profile : 1D arrays or None
        Optional per-cell x / y spacings for the physical (non-CPML)
        interior. When provided, the first and last values must match
        ``dx`` (they set the boundary cell size used by the CPML).
    pec_faces, pmc_faces : set of str or None
        Face labels (``x_lo``, ``x_hi``, ``y_lo``, ``y_hi``, ``z_lo``,
        ``z_hi``) where the boundary is PEC / PMC. Per-face pad count
        is forced to 0 on these faces so the reflector plane aligns
        with the user domain edge. Added 2026-04 to close the
        PMC+CPML composition gap on the non-uniform mesh path.
    cpml_axes : str
        Axes that participate in CPML allocation (default ``"xyz"``).
        Per-face allocation still applies; an axis absent from
        ``cpml_axes`` gets pad=0 on both faces.
    """
    _pec = pec_faces or set()
    _pmc = pmc_faces or set()

    def _face_pad(axis: str, side: str) -> int:
        face = f"{axis}_{side}"
        if face in _pec or face in _pmc:
            return 0
        if axis not in cpml_axes:
            return 0
        return int(cpml_layers)

    pad_x_lo = _face_pad("x", "lo")
    pad_x_hi = _face_pad("x", "hi")
    pad_y_lo = _face_pad("y", "lo")
    pad_y_hi = _face_pad("y", "hi")
    pad_z_lo = _face_pad("z", "lo")
    pad_z_hi = _face_pad("z", "hi")
    # --- x profile (uniform or provided) ---
    if dx_profile is None:
        nx_interior = int(round(domain_xy[0] / dx))
        dx_prof_phys = np.full(nx_interior, float(dx))
    elif is_tracer(dx_profile):
        # Tracer path (mesh-as-design-variable): stay in jnp and skip the
        # concrete boundary validation. Caller is responsible for keeping
        # `dx_profile[0] == dx_profile[-1] == dx` (the CPML uses the
        # boundary scalar). Mirrors the 2026-04-17 dz tracer refactor.
        dx_prof_phys = jnp.asarray(dx_profile, dtype=jnp.float32)
    else:
        dx_prof_phys = np.asarray(dx_profile, dtype=np.float64)
        # Guard: CPML cells must have the same size as the boundary interior
        # cells, otherwise the CPML σ/κ profile is miscalibrated.
        if abs(float(dx_prof_phys[0]) - float(dx)) > 1e-12:
            raise ValueError(
                f"dx_profile[0]={float(dx_prof_phys[0])} must equal boundary "
                f"dx={float(dx)} (CPML cells use the boundary spacing)."
            )
        if abs(float(dx_prof_phys[-1]) - float(dx)) > 1e-12:
            raise ValueError(
                f"dx_profile[-1]={float(dx_prof_phys[-1])} must equal boundary "
                f"dx={float(dx)}."
            )
    dx_full = _append_bounding_node(_pad_profile(dx_prof_phys, pad_x_lo, pad_x_hi))
    nx = int(dx_full.shape[0])

    # --- y profile ---
    if dy_profile is None:
        # Uniform y uses `dx` as the cell size (legacy behaviour)
        ny_interior = int(round(domain_xy[1] / dx))
        dy_prof_phys = np.full(ny_interior, float(dx))
        dy_boundary = float(dx)
    elif is_tracer(dy_profile):
        # Tracer path: stay in jnp. Use the concrete scalar `dx` as the
        # boundary cell size — the caller must align `dy_profile[0]` and
        # `dy_profile[-1]` with `dx` (same contract as the concrete path).
        dy_prof_phys = jnp.asarray(dy_profile, dtype=jnp.float32)
        dy_boundary = float(dx)
    else:
        dy_prof_phys = np.asarray(dy_profile, dtype=np.float64)
        dy_boundary = float(dy_prof_phys[0])
        if abs(float(dy_prof_phys[-1]) - dy_boundary) > 1e-12:
            raise ValueError(
                f"dy_profile boundary cells must match each other "
                f"(got lo={dy_boundary}, hi={float(dy_prof_phys[-1])})."
            )
    dy_full = _append_bounding_node(_pad_profile(dy_prof_phys, pad_y_lo, pad_y_hi))
    ny = int(dy_full.shape[0])

    # --- z profile ---
    dz_full = _append_bounding_node(_pad_profile(dz_profile, pad_z_lo, pad_z_hi))
    nz = int(dz_full.shape[0])

    # --- CFL from minimum cell size on every axis ---
    def _axis_min(d_full):
        return jnp.min(d_full) if is_tracer(d_full) else float(np.min(d_full))

    any_traced = (
        is_tracer(dx_full) or is_tracer(dy_full) or is_tracer(dz_full)
    )
    dx_min = _axis_min(dx_full)
    dy_min = _axis_min(dy_full)
    dz_min = _axis_min(dz_full)
    if any_traced:
        dt = 0.99 / (C0 * jnp.sqrt(1 / dx_min ** 2 + 1 / dy_min ** 2 + 1 / dz_min ** 2))
    else:
        dt = float(
            0.99 / (C0 * np.sqrt(1 / dx_min ** 2 + 1 / dy_min ** 2 + 1 / dz_min ** 2))
        )

    # --- Per-cell arrays + inverse spacings ---
    dx_arr = jnp.asarray(dx_full, dtype=jnp.float32)
    dy_arr = jnp.asarray(dy_full, dtype=jnp.float32)
    dz_arr = jnp.asarray(dz_full, dtype=jnp.float32)

    inv_dx, inv_dx_h = _profile_to_inv_arrays(dx_full)
    inv_dy, inv_dy_h = _profile_to_inv_arrays(dy_full)
    inv_dz, inv_dz_h = _profile_to_inv_arrays(dz_full)

    return NonUniformGrid(
        nx=nx, ny=ny, nz=nz,
        dx=float(dx), dy=dy_boundary,
        dx_arr=dx_arr, dy_arr=dy_arr, dz=dz_arr,
        dt=dt, cpml_layers=cpml_layers,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
        inv_dx_h=inv_dx_h, inv_dy_h=inv_dy_h, inv_dz_h=inv_dz_h,
        pad_x_lo=pad_x_lo, pad_x_hi=pad_x_hi,
        pad_y_lo=pad_y_lo, pad_y_hi=pad_y_hi,
        pad_z_lo=pad_z_lo, pad_z_hi=pad_z_hi,
    )


def _interior_line_positions(
    d_arr_np: np.ndarray, pad_lo: int, pad_hi: int | None = None,
) -> np.ndarray:
    """Return cell-edge positions (0 at first interior face) for a padded
    cell-size array. Length = n_interior + 1.

    ``pad_lo`` and ``pad_hi`` may differ (per-face allocation, 2026-04).
    Back-compat: a single-argument call treats the value as symmetric.
    """
    if pad_hi is None:
        pad_hi = pad_lo
    interior = interior_cells(d_arr_np, pad_lo, pad_hi)
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    return edges


def _nominal_edges_or_actual(
    d_arr, total_pad: int,
    *, pad_lo: int | None = None,
    fallback_dx: float | None = None,
) -> np.ndarray:
    """Return concrete cell-edge positions for index lookup.

    ``total_pad`` is ``pad_lo + pad_hi`` — the cells removed when
    slicing to the interior. ``pad_lo`` (defaulting to ``total_pad/2``
    for the legacy symmetric case) is needed by the tracer path
    to reconstruct ``n_interior`` and by the concrete path to pick
    the right interior slice.

    When ``d_arr`` is a JAX tracer (mesh-as-design-variable path), we
    fall back to a uniform ``fallback_dx`` reference mesh so that
    physical-coordinate lookup of source / probe / port positions still
    yields a concrete integer index. The traced cell sizes drive the
    FDTD physics downstream; only the structural index is resolved
    from the nominal mesh.
    """
    if pad_lo is None:
        pad_lo = total_pad // 2
    pad_hi = total_pad - pad_lo
    if is_tracer(d_arr):
        if fallback_dx is None or fallback_dx <= 0:
            raise ValueError(
                "tracer-valued cell-size profile requires a concrete "
                "fallback_dx for position->index resolution."
            )
        n_total = int(d_arr.shape[0])
        n_interior = n_total - total_pad
        interior = np.full(n_interior, float(fallback_dx), dtype=np.float64)
        return np.insert(np.cumsum(interior), 0, 0.0)
    return _interior_line_positions(np.asarray(d_arr), pad_lo, pad_hi)


def z_position_to_index(grid: NonUniformGrid, z_phys: float) -> int:
    """Convert physical z-coordinate to (cpml-offset) grid index."""
    edges = _nominal_edges_or_actual(
        grid.dz, grid.pad_z_lo + grid.pad_z_hi,
        pad_lo=grid.pad_z_lo, fallback_dx=float(grid.dx),
    )
    idx = int(np.argmin(np.abs(edges - float(z_phys))))
    return idx + grid.pad_z_lo


def _axis_position_to_index(
    d_arr: jnp.ndarray,
    pad_lo: int,
    pad_hi: int,
    pos: float,
    fallback_dx: float | None = None,
) -> int:
    """Generic non-uniform axis lookup.

    Uses cell-edge positions (same convention as z_position_to_index):
    position 0 is the first interior face, position ``sum(interior)`` is
    the last interior face.
    """
    edges = _nominal_edges_or_actual(
        d_arr, pad_lo + pad_hi, pad_lo=pad_lo, fallback_dx=fallback_dx,
    )
    idx = int(np.argmin(np.abs(edges - float(pos))))
    return idx + pad_lo


def position_to_index(grid: NonUniformGrid, pos: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert physical (x, y, z) to grid indices for NonUniformGrid.

    Accounts for per-face CPML padding (``pad_{axis}_lo`` leading offset).
    All three axes use cumulative cell-size lookup. In the uniform-xy
    case (``dx_arr`` constant) this reduces to the legacy
    ``round(pos[0]/dx) + pad_{axis}_lo`` behaviour within one cell.
    """
    i = _axis_position_to_index(
        grid.dx_arr, grid.pad_x_lo, grid.pad_x_hi, pos[0],
        fallback_dx=float(grid.dx),
    )
    j = _axis_position_to_index(
        grid.dy_arr, grid.pad_y_lo, grid.pad_y_hi, pos[1],
        fallback_dx=float(grid.dy),
    )
    k = z_position_to_index(grid, pos[2])
    return (i, j, k)


def make_z_profile(
    features: list[float],
    domain_z: float,
    dx_fine: float,
    dx_coarse: float | None = None,
    grading: float = 1.4,
) -> np.ndarray:
    """Generate z-profile that snaps to feature boundaries.

    Fine cells are used near feature boundaries; coarse cells fill the
    remaining space.  Adjacent cells differ by at most ``grading``.

    Parameters
    ----------
    features : list of z-positions that must align to cell boundaries
    domain_z : total z domain height
    dx_fine : fine cell size (near features)
    dx_coarse : coarse cell size (away from features). If None, uses dx_fine
        everywhere (no grading).
    grading : max ratio between adjacent cells (default 1.4)
    """
    if dx_coarse is None:
        dx_coarse = dx_fine

    features = sorted(set(features + [0, domain_z]))

    cells = []
    for i in range(len(features) - 1):
        span = features[i + 1] - features[i]
        if span <= 0:
            continue

        if dx_coarse <= dx_fine * 1.01 or span <= 4 * dx_fine:
            # Uniform fine cells for thin segments or when no grading needed
            n = max(1, int(round(span / dx_fine)))
            dz = span / n
            cells.extend([dz] * n)
        else:
            # Graded transition: fine → coarse → fine
            # Build from both ends toward the middle
            left = []
            dz = dx_fine
            remaining = span
            while remaining > 0 and dz < dx_coarse:
                dz_use = min(dz, remaining)
                left.append(dz_use)
                remaining -= dz_use
                dz = min(dz * grading, dx_coarse)

            # Fill middle with coarse cells
            if remaining > dx_coarse * 0.5:
                n_mid = max(1, int(round(remaining / dx_coarse)))
                mid = [remaining / n_mid] * n_mid
            else:
                mid = [remaining] if remaining > 1e-15 else []

            cells.extend(left + mid)

    return np.array(cells)


def make_current_source(grid: NonUniformGrid, position_ijk, component,
                        waveform_fn, n_steps, materials, amplitude_kind=None):
    """Create a properly normalized current source for non-uniform grid.

    The waveform specifies CURRENT (Amperes). The E-field addition is:
    E += (dt/ε) × I_source / dV
    where dV is the E node's control volume: the PRIMAL per-cell width on
    the component's own axis and the DUAL spacing ``(d[k-1]+d[k])/2`` on the
    two transverse axes (issue #672). On a uniform mesh the two coincide and
    this is the familiar ``dx × dy × dz``.

    This gives resolution-independent injected POWER regardless of cell size.
    Same approach as Meep's internal source normalization.

    Native amplitude convention (issue #571): ``'current'`` —
    ``amplitude_kind`` of None or ``'current'`` is bit-identical to the
    historical output (no extra multiply). ``amplitude_kind='field'``
    rescales by ``dV/Cb`` via
    ``rfx.api._source_semantics.source_amplitude_scale``, computed AFTER
    the tracer-safe cb/dV resolution below so the GEO-C3
    differentiable-material path is preserved unchanged.
    """
    import jax
    i, j, k = position_ijk

    # GEO-C3: on the differentiable-material path ``materials.eps_r`` /
    # ``materials.sigma`` are tracers — ``float()`` raised
    # TracerArrayConversionError. Stay in jnp when traced so the gradient
    # propagates into the waveform normalisation; keep the exact ``float()``
    # path otherwise so non-traced output stays bit-identical.
    materials_traced = (
        is_tracer(materials.eps_r) or is_tracer(materials.sigma)
    )
    if materials_traced:
        eps = jnp.asarray(materials.eps_r[i, j, k]) * EPS_0
        sigma = jnp.asarray(materials.sigma[i, j, k])
    else:
        eps = float(materials.eps_r[i, j, k]) * EPS_0
        sigma = float(materials.sigma[i, j, k])
    loss = sigma * grid.dt / (2.0 * eps)

    # Cb = dt / (eps * (1 + loss))
    cb = (grid.dt / eps) / (1.0 + loss)

    # Control volume of the E node this source injects into (issue #672).
    # An E_a component is an EDGE along its own axis a and sits ON a node on
    # the two transverse axes, so its control volume is MIXED:
    #   axis == a  -> primal per-cell width d[idx]
    #   axis != a  -> dual spacing (d[idx-1]+d[idx])/2, which is exactly the
    #                 metric the NU E update divides curl H by (CORE-C2,
    #                 ``_profile_to_inv_arrays``); the Ex update, for
    #                 instance, uses inv_dy and inv_dz and never inv_dx.
    # Using the primal width on all three axes (pre-#672) is right only where
    # both neighbour pairs are equal. On a 2:1 grading step it mis-normalizes
    # the injected current moment by the local cell ratio on each graded
    # TRANSVERSE axis (measured: dV too large by 1.333 with one graded
    # transverse axis, 1.778 with two).
    # Stay in jnp when the profile is traced so a mesh-as-design variable
    # propagates the gradient into the waveform normalisation.
    _axis_of = {"ex": 0, "ey": 1, "ez": 2}
    if component not in _axis_of:
        raise ValueError(
            f"make_current_source: unknown component {component!r} "
            f"(expected one of {sorted(_axis_of)}). The control volume is "
            f"component-dependent on a non-uniform mesh, so there is no safe "
            f"fallback (issue #672)."
        )
    p_axis = _axis_of[component]
    grid_traced = (
        is_tracer(grid.dx_arr) or is_tracer(grid.dy_arr) or is_tracer(grid.dz)
    )
    any_traced = materials_traced or grid_traced
    _profiles = (grid.dx_arr, grid.dy_arr, grid.dz)
    _idx = (i, j, k)
    _widths = []
    for _a in range(3):
        if _a == p_axis:
            _w = (jnp.asarray(_profiles[_a])[_idx[_a]] if grid_traced
                  else float(np.asarray(_profiles[_a])[_idx[_a]]))
        elif grid_traced:
            _w = e_node_dual_spacing_at(jnp.asarray(_profiles[_a]), _idx[_a])
        else:
            _w = float(e_node_dual_spacing_at(
                np.asarray(_profiles[_a], dtype=np.float64), _idx[_a]))
        _widths.append(_w)
    dx_local, dy_local, dz_local = _widths
    dV = dx_local * dy_local * dz_local

    # Normalized waveform: Cb * I(t) / dV
    # This ensures power = ∫(J·E)dV is independent of cell size
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = (cb / dV) * jax.vmap(waveform_fn)(times)

    from rfx.api._source_semantics import needs_scale, source_amplitude_scale
    if needs_scale(amplitude_kind, "cb_over_dv"):
        # only 'field' lands here (scale dV/Cb). Python-level dispatch on
        # the kind string (issue #571 dossier): cb/dV may be tracers on
        # the GEO-C3 / mesh-as-design-variable paths, so never gate the
        # multiply on the scale VALUE.
        waveform = source_amplitude_scale(
            amplitude_kind, "cb_over_dv", cb=cb, dV=dV) * waveform

    waveform_out = waveform if any_traced else np.array(waveform)
    return (i, j, k, component, waveform_out)


def _bwd_neighbor(h, idx, axis):
    """``h`` one cell back along ``axis``, using the SAME out-of-domain
    convention the NU E-update uses.

    ``rfx.core.yee._shift_bwd`` (rfx/core/yee.py) pads explicitly with zero,
    so ``_curl_h_nu`` reads 0 for H outside the domain. Before #689 this
    helper spelled the backward read as a raw ``h[i - 1]``, which at
    ``i == 0`` is Python's negative index — H at the OPPOSITE face of the
    domain. That made a 4-term local Ampere loop depend on a field cell it
    does not enclose, and it disagreed with the very curl the port current
    is supposed to be the loop integral of.

    A length-1 axis carries no variation along itself, so the difference
    along it is 0 (return the cell itself, matching the uniform lane's
    2-D behaviour). NU grids are 3-D today, so that branch is defensive.
    """
    i = int(idx[axis])
    if h.shape[axis] == 1:
        return h[idx]
    if i == 0:
        return jnp.zeros_like(h[idx])
    back = list(idx)
    back[axis] = i - 1
    return h[tuple(back)]


def _build_wp_meta(wire_ports, grid):
    """Static per-port metadata for the wire-port DFT scan body.

    Pre-computes the port cell's metrics OUTSIDE the scan (repo engineering
    principle 4). Two metric families, and they are not interchangeable
    (issue #672):
      PRIMAL  d[idx]                 -> V = -E_a * d_parallel, the length of
                                        the E_a edge itself.
      DUAL    (d[idx-1]+d[idx])/2    -> the Ampere-loop legs, each weighted
                                        by the dual spacing on that H
                                        component's own axis.
    x/y stay float() as before (np.asarray already refuses a traced xy
    profile here); the z metrics are indexed without np.asarray so a traced
    dz_profile keeps its gradient through the extractor. `excite` and
    `direction` are carried for the post-processing; `direction` no longer
    enters the wave split (issue #673).

    Slot order (guarded by
    tests/test_nonuniform_source_port_dual_spacing.py):
      0..2  mid_i, mid_j, mid_k     (midpoint of the LIVE run, issue #764)
      3     component
      4     impedance (whole-port Z0)
      5, 6  primal dx[mid_i], dy[mid_j]
      7     excite
      8     direction
      9,10  dual x/y spacings at mid
      11    primal dz[mid_k]        (traced-safe)
      12    dual z spacing at mid
      13    live_cells              (static tuple of (i, j, k), issue #764)
      14    d_par per live cell     (primal width on the component's own
                                     axis at each live cell, same order as
                                     slot 13; z entries traced-safe)
    Slots 13/14 feed the whole-port gap voltage
    V_port = sum_live(-E_c * d_par,c) — the #672 PRIMAL family, per cell.
    """
    _dx_arr_np = np.asarray(grid.dx_arr, dtype=np.float64)
    _dy_arr_np = np.asarray(grid.dy_arr, dtype=np.float64)

    def _d_par(cell, comp):
        ci, cj, ck = cell
        if comp == "ex":
            return float(_dx_arr_np[ci])
        if comp == "ey":
            return float(_dy_arr_np[cj])
        # ez: index grid.dz WITHOUT np.asarray so a traced dz_profile
        # keeps its gradient through the extractor.
        return grid.dz[ck]

    meta = []
    for wp in wire_ports:
        live_cells = tuple(
            tuple(int(x) for x in c)
            for c in wp.get('live_cells',
                            ((wp['mid_i'], wp['mid_j'], wp['mid_k']),))
        )
        comp = wp['component']
        meta.append((
            wp['mid_i'], wp['mid_j'], wp['mid_k'],
            comp, wp['impedance'],
            float(_dx_arr_np[wp['mid_i']]),
            float(_dy_arr_np[wp['mid_j']]),
            bool(wp.get('excite', True)),
            str(wp.get('direction', '-x')),
            float(e_node_dual_spacing_at(_dx_arr_np, wp['mid_i'])),
            float(e_node_dual_spacing_at(_dy_arr_np, wp['mid_j'])),
            grid.dz[wp['mid_k']],
            e_node_dual_spacing_at(grid.dz, wp['mid_k']),
            live_cells,
            tuple(_d_par(c, comp) for c in live_cells),
        ))
    return meta


def wire_port_current(hx, hy, hz, comp, mi, mj, mk,
                      dual_x, dual_y, dual_z):
    """Enclosed current from the discrete Ampere loop at a wire-port cell.

    For component ``a`` with ``(a, b, c)`` cyclic,
    ``curl_a = dH_c/db - dH_b/dc`` and the pierced DUAL face has area
    ``dual_b * dual_c``, so

        I_a = (H_c[b] - H_c[b-1]) * dual_c - (H_b[c] - H_b[c-1]) * dual_b

    i.e. each H-difference leg is weighted by the DUAL spacing along THAT H
    COMPONENT'S OWN axis (the leg's own length as a contour segment) — never
    by the axis it is differenced along, and never by a primal cell width.

    Before issue #672 every branch got both of those wrong at once: one leg
    carried the primal width of the right axis, the other carried a width
    from an unrelated axis. Both errors are invisible on a uniform cubic
    mesh, which is why the uniform lane (``rfx/simulation.py``) can spell the
    whole thing with a single ``dx`` and still be correct there.

    Signs match the pre-#672 code exactly; only the metrics changed.

    Out-of-domain H is ZERO (#689). Each backward read goes through
    :func:`_bwd_neighbor`, which pads the way ``_curl_h_nu`` /
    ``rfx.core.yee._shift_bwd`` pad, so the port current is the loop
    integral of the same curl the E-update integrates. The previous raw
    ``[mi - 1]`` spelling was a NEGATIVE index at ``mi == 0``, i.e. H at
    the opposite domain face, which a 4-term LOCAL loop must never touch.

    Scope, stated because #689's commit body overstated it: this is a
    STENCIL correctness fix, not a repair of a measured wrong number. The
    cell it used to read, the last index of the axis, holds 0.0 at runtime
    anyway — ``inv_d_h[N-1] = 0`` by construction
    (``_profile_to_inv_arrays``) and the last row is grid pad. Measured on
    a 21-cell NU run with an ez wire port flush against x_lo (``mi == 0``),
    sampling every H component's last plane on every axis at every one of
    300 steps: max|H| there was 0.000000e+00 throughout, under
    ``boundary="pec"`` and under ``boundary="cpml"``, against global
    max|H| of 1.9e+07 and 1.8e+04. So the two spellings agree on any state
    the stepper can produce, and the +1000 perturbation the #689 oracles
    use is not reachable by the solver. Keep the fix — a local loop should
    not read a cell it does not enclose, and the two spellings agreeing
    only by a coincidence of the pad arrays is not a property worth
    depending on — but do not describe it as a live defect.
    """
    idx = (mi, mj, mk)
    if comp == "ez":
        return ((hy[idx] - _bwd_neighbor(hy, idx, 0)) * dual_y
                - (hx[idx] - _bwd_neighbor(hx, idx, 1)) * dual_x)
    if comp == "ex":
        return ((hz[idx] - _bwd_neighbor(hz, idx, 1)) * dual_z
                - (hy[idx] - _bwd_neighbor(hy, idx, 2)) * dual_y)
    if comp == "ey":
        return ((hx[idx] - _bwd_neighbor(hx, idx, 2)) * dual_x
                - (hz[idx] - _bwd_neighbor(hz, idx, 0)) * dual_z)
    raise ValueError(f"wire_port_current: unknown component {comp!r}")


def _curl_h_nu(state, inv_dx, inv_dy, inv_dz):
    """Compute curl(H) using non-uniform backward differences.

    Shared by both plain and dispersive E updates on non-uniform grids.
    """
    from rfx.core.yee import _shift_bwd
    hx, hy, hz = state.hx, state.hy, state.hz

    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy[None, :, None]
    )
    return curl_x, curl_y, curl_z


def _update_e_nu_dispersive(
    state: FDTDState,
    materials: MaterialArrays,
    dt: float,
    inv_dx: jnp.ndarray,
    inv_dy: jnp.ndarray,
    inv_dz: jnp.ndarray,
    *,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    e_old: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[FDTDState, object | None, object | None]:
    """E-field update with ADE dispersion on non-uniform grid.

    Uses per-axis inverse spacing arrays for curl(H), then applies the
    same ADE coefficient math as the uniform path. The ADE coefficients
    (ca, cb, cc, alpha, beta, etc.) are pre-baked spatial arrays that do
    not depend on dx, so they work unchanged on non-uniform grids.

    Mirrors the structure of ``_update_e_with_optional_dispersion`` in
    ``rfx/simulation.py`` but replaces the uniform ``curl / dx`` with
    non-uniform ``curl * inv_d[axis]``.

    Parameters
    ----------
    e_old : (ex_old, ey_old, ez_old) tuple of arrays, optional
        Explicit pre-step E snapshots to use as ``*_old`` in the ADE
        polarisation update.  When ``None`` (single-device default), the
        helper reads ``state.ex/ey/ez`` directly — these are the fields
        from the start of the step in the single-device runner because
        no E ghost exchange occurs before this call.

        On the distributed NU path (V3 Phase 2D), the caller MUST pass
        explicit snapshots taken BEFORE any ghost exchange so that the
        polarisation update uses pre-exchange E values.  See the ADE
        Ordering Contract documented in ``run_nonuniform_distributed``.
    """
    from rfx.materials.debye import DebyeState
    from rfx.materials.lorentz import LorentzState

    curl_x, curl_y, curl_z = _curl_h_nu(state, inv_dx, inv_dy, inv_dz)
    if e_old is not None:
        ex_old, ey_old, ez_old = e_old
    else:
        ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    # Narrow every output back to the dtype of the carry it came from
    # (issue #656) — the NU lane runs the same ADE recurrences off the same
    # ``init_debye``/``init_lorentz`` coefficients, which are built from
    # ``grid.dt`` (an np.float64 scalar, strongly typed in JAX). The full
    # rationale lives on ``rfx.materials.lorentz.update_e_lorentz``.
    _fdtype = state.ex.dtype

    # --- Debye only ---
    if debye is not None and lorentz is None:
        debye_coeffs, debye_state = debye
        ca, cb, cc = debye_coeffs.ca, debye_coeffs.cb, debye_coeffs.cc
        alpha, beta = debye_coeffs.alpha, debye_coeffs.beta
        _pdtype = jnp.promote_types(debye_state.px.dtype, _fdtype)

        ex_new = (ca * ex_old + cb * curl_x
                  + jnp.sum(cc * debye_state.px, axis=0)).astype(_fdtype)
        ey_new = (ca * ey_old + cb * curl_y
                  + jnp.sum(cc * debye_state.py, axis=0)).astype(_fdtype)
        ez_new = (ca * ez_old + cb * curl_z
                  + jnp.sum(cc * debye_state.pz, axis=0)).astype(_fdtype)

        px_new = (alpha * debye_state.px
                  + beta * (ex_new[None] + ex_old[None])).astype(_pdtype)
        py_new = (alpha * debye_state.py
                  + beta * (ey_new[None] + ey_old[None])).astype(_pdtype)
        pz_new = (alpha * debye_state.pz
                  + beta * (ez_new[None] + ez_old[None])).astype(_pdtype)

        new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                                  step=state.step + 1)
        new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)
        return new_fdtd, new_debye, None

    # --- Lorentz only ---
    if lorentz is not None and debye is None:
        lorentz_coeffs, lor_state = lorentz
        ca, cb, cc = lorentz_coeffs.ca, lorentz_coeffs.cb, lorentz_coeffs.cc
        a, b, c = lorentz_coeffs.a, lorentz_coeffs.b, lorentz_coeffs.c
        _pdtype = jnp.promote_types(lor_state.px.dtype, _fdtype)

        px_new = (a * lor_state.px + b * lor_state.px_prev
                  + c * ex_old[None]).astype(_pdtype)
        py_new = (a * lor_state.py + b * lor_state.py_prev
                  + c * ey_old[None]).astype(_pdtype)
        pz_new = (a * lor_state.pz + b * lor_state.pz_prev
                  + c * ez_old[None]).astype(_pdtype)

        dpx = jnp.sum(px_new - lor_state.px, axis=0)
        dpy = jnp.sum(py_new - lor_state.py, axis=0)
        dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

        ex_new = (ca * ex_old + cb * curl_x - cc * dpx).astype(_fdtype)
        ey_new = (ca * ey_old + cb * curl_y - cc * dpy).astype(_fdtype)
        ez_new = (ca * ez_old + cb * curl_z - cc * dpz).astype(_fdtype)

        new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                                  step=state.step + 1)
        new_lor = LorentzState(
            px=px_new, py=py_new, pz=pz_new,
            px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
        )
        return new_fdtd, None, new_lor

    # --- Mixed Debye + Lorentz ---
    debye_coeffs, debye_state = debye
    lorentz_coeffs, lor_state = lorentz
    _dpdtype = jnp.promote_types(debye_state.px.dtype, _fdtype)
    _lpdtype = jnp.promote_types(lor_state.px.dtype, _fdtype)

    # Explicit Lorentz polarization update first
    px_l_new = (lorentz_coeffs.a * lor_state.px
                + lorentz_coeffs.b * lor_state.px_prev
                + lorentz_coeffs.c * ex_old[None]).astype(_lpdtype)
    py_l_new = (lorentz_coeffs.a * lor_state.py
                + lorentz_coeffs.b * lor_state.py_prev
                + lorentz_coeffs.c * ey_old[None]).astype(_lpdtype)
    pz_l_new = (lorentz_coeffs.a * lor_state.pz
                + lorentz_coeffs.b * lor_state.pz_prev
                + lorentz_coeffs.c * ez_old[None]).astype(_lpdtype)

    dpx_l = jnp.sum(px_l_new - lor_state.px, axis=0)
    dpy_l = jnp.sum(py_l_new - lor_state.py, axis=0)
    dpz_l = jnp.sum(pz_l_new - lor_state.pz, axis=0)

    beta_sum = jnp.sum(debye_coeffs.beta, axis=0)
    gamma_base = 1.0 / lorentz_coeffs.cc
    gamma_total = jnp.maximum(gamma_base + beta_sum, EPS_0 * 1e-10)
    numer_base = lorentz_coeffs.ca * gamma_base

    ca = (numer_base - beta_sum) / gamma_total
    cb = dt / gamma_total
    cc_debye = (1.0 - debye_coeffs.alpha) / gamma_total
    cc_lorentz = 1.0 / gamma_total

    ex_new = (ca * ex_old + cb * curl_x
              + jnp.sum(cc_debye * debye_state.px, axis=0)
              - cc_lorentz * dpx_l).astype(_fdtype)
    ey_new = (ca * ey_old + cb * curl_y
              + jnp.sum(cc_debye * debye_state.py, axis=0)
              - cc_lorentz * dpy_l).astype(_fdtype)
    ez_new = (ca * ez_old + cb * curl_z
              + jnp.sum(cc_debye * debye_state.pz, axis=0)
              - cc_lorentz * dpz_l).astype(_fdtype)

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                              step=state.step + 1)
    new_debye = DebyeState(
        px=(debye_coeffs.alpha * debye_state.px
            + debye_coeffs.beta * (ex_new[None] + ex_old[None])).astype(_dpdtype),
        py=(debye_coeffs.alpha * debye_state.py
            + debye_coeffs.beta * (ey_new[None] + ey_old[None])).astype(_dpdtype),
        pz=(debye_coeffs.alpha * debye_state.pz
            + debye_coeffs.beta * (ez_new[None] + ez_old[None])).astype(_dpdtype),
    )
    new_lor = LorentzState(
        px=px_l_new, py=py_l_new, pz=pz_l_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )
    return new_fdtd, new_debye, new_lor


class _NUScanSetup(NamedTuple):
    """Host-side bundle built by :func:`_build_nu_scan` (#383 code motion).

    Carries the scan step function, its initial carry, the stacked source
    table, and the post-scan assembly metadata shared by
    :func:`run_nonuniform` (single ``lax.scan``) and
    :func:`run_nonuniform_until_decay` (chunked host loop). This tuple
    never crosses a JAX transform boundary — ``step_fn`` is a Python
    closure and the ``use_*`` flags are Python bools.
    """
    step_fn: object
    carry_init: dict
    src_waveforms: jnp.ndarray
    dt: object
    sources: list
    probes: list
    wire_ports: list
    dft_planes: list
    flux_monitors: list
    waveguide_meta: tuple
    wp_meta: list
    sp_freqs: object
    use_wire_ports: bool
    use_dft_planes: bool
    use_flux_monitors: bool
    use_lumped_rlc: bool
    use_ntff: bool
    use_waveguide_ports: bool


def _build_nu_scan(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    pec_mask=None,
    pec_two_plane_mask=None,
    pec_occupancy=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    dft_planes: list | None = None,
    rlc_metas: tuple = (),
    rlc_states: tuple = (),
    ntff_box=None,
    ntff_data=None,
    waveguide_ports: list | None = None,
    tfsf: tuple | None = None,
    flux_monitors: list | None = None,
    emit_time_series: bool = True,
    aniso_eps: tuple | None = None,
    sheet_impedance=None,
) -> _NUScanSetup:
    """Build the NU scan carry + step function (pure code motion, #383).

    Extracted verbatim from :func:`run_nonuniform` so the until-decay
    entry point can drive the SAME ``step_fn`` / carry through a chunked
    host loop. ``n_steps`` is used only to size the zero-source
    placeholder table; the source waveform arrays themselves define the
    table length when sources are present.
    """
    sources = sources or []
    probes = probes or []
    wire_ports = wire_ports or []
    dft_planes = dft_planes or []
    waveguide_ports = waveguide_ports or []
    flux_monitors = flux_monitors or []
    dt = grid.dt
    use_wire_ports = len(wire_ports) > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_dft_planes = len(dft_planes) > 0
    use_flux_monitors = len(flux_monitors) > 0
    use_lumped_rlc = len(rlc_metas) > 0
    use_ntff = ntff_box is not None and ntff_data is not None
    use_waveguide_ports = len(waveguide_ports) > 0
    use_tfsf = tfsf is not None

    # CPML: only initialize when cpml_layers > 0 (skip for PEC boundary)
    use_cpml = grid.cpml_layers > 0

    cpml_params = None
    cpml_state_init = None
    cpml_grid = None
    # Effective CPML axes after PEC/PMC closure. Axes whose lo+hi pad is
    # zero are fully closed and the apply path's `state.e*[:, :, :n]`
    # slices clip to the (small) axis length, breaking the broadcast
    # against the (cpml_layers,) profile coefficients. Drop those axes
    # from `apply_cpml_*` so the no-op branch passes psi through
    # unchanged. Mirrors the uniform runner, which already threads
    # `cpml_axes` from the grid (rfx/runners/uniform.py).
    cpml_axes_eff = "xyz"

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e

        # Pass NonUniformGrid directly — init_cpml duck-types dx/dy/dz.
        # NonUniformGrid does not carry pmc_faces / pec_faces attrs
        # (frozen dataclass, pytree-registered), so the sets must be
        # threaded through from the caller.
        cpml_params, cpml_state_init = init_cpml(
            grid, pec_faces=pec_faces, pmc_faces=pmc_faces,
        )
        cpml_grid = grid
        cpml_axes_eff = "".join(
            ax for ax, lo, hi in (
                ("x", grid.pad_x_lo, grid.pad_x_hi),
                ("y", grid.pad_y_lo, grid.pad_y_hi),
                ("z", grid.pad_z_lo, grid.pad_z_hi),
            )
            if (lo + hi) > 0
        )

    # PMC enforcement (2026-04). The NU scan body previously never
    # zeroed H_tan on PMC faces, so a half-symmetric configuration
    # that relied on the mirror plane was running with an effectively
    # free boundary. Frozen set gives JIT cache a stable hash; empty
    # set short-circuits the apply to a no-op.
    use_pmc_faces = bool(pmc_faces)
    _pmc_faces_frozen = frozenset(pmc_faces) if pmc_faces else frozenset()

    use_pec_mask = pec_mask is not None
    use_pec_occupancy = pec_occupancy is not None

    # #677 surface-impedance sheet: exponential-stepping A/B built once
    # from the FINAL scan materials; applied per step at tangential edges
    # in step_fn (after apply_pec_mask, before sources/DFT sampling).
    use_sheet_impedance = sheet_impedance is not None
    if use_sheet_impedance:
        from rfx.materials.thin_conductor import sheet_update_coeffs
        sheet_coeffs = sheet_update_coeffs(
            sheet_impedance.sigma_sheet, materials, dt)

    if sources:
        src_waveforms = jnp.stack([jnp.array(s[4]) for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    src_meta = [(s[0], s[1], s[2], s[3]) for s in sources]
    prb_meta = [(p[0], p[1], p[2], p[3]) for p in probes]

    state = init_state((grid.nx, grid.ny, grid.nz))

    inv_dx_h = grid.inv_dx_h
    inv_dy_h = grid.inv_dy_h
    inv_dz_h = grid.inv_dz_h
    inv_dx = grid.inv_dx
    inv_dy = grid.inv_dy
    inv_dz = grid.inv_dz

    carry_init = {"fdtd": state}
    if use_cpml:
        carry_init["cpml"] = cpml_state_init

    # Debye/Lorentz ADE state
    if use_debye:
        debye_coeffs, debye_state = debye
        carry_init["debye"] = debye_state

    if use_lorentz:
        lorentz_coeffs, lorentz_state = lorentz
        carry_init["lorentz"] = lorentz_state

    # Wire port S-param DFT accumulators
    # (defaults bound unconditionally so _NUScanSetup can carry them;
    # both stay unused unless the branch below runs — no behavior change)
    sp_freqs = None
    wp_meta: list = []
    if use_wire_ports and s_param_freqs is not None:
        sp_freqs = jnp.asarray(s_param_freqs, dtype=jnp.float32)
        nf = len(sp_freqs)
        carry_init["wire_sparams"] = tuple(
            (jnp.zeros(nf, dtype=jnp.complex64),  # v_dft
             jnp.zeros(nf, dtype=jnp.complex64),  # i_dft
             jnp.zeros(nf, dtype=jnp.complex64),  # v_inc_dft
             jnp.zeros(nf, dtype=jnp.complex64))  # v_port_dft (issue #764)
            for _ in wire_ports
        )
        wp_meta = _build_wp_meta(wire_ports, grid)
    else:
        use_wire_ports = False

    # DFT plane probe carry + static metadata
    if use_dft_planes:
        carry_init["dft_planes"] = tuple(probe.accumulator for probe in dft_planes)
        dft_meta = tuple(
            (probe.component, probe.axis, probe.index, probe.freqs, probe.region)
            for probe in dft_planes
        )
    else:
        dft_meta = ()

    # Flux monitor carry + static metadata (mirrors the uniform scan body
    # in rfx/simulation.py). The NU dt is scalar, so the DFT kernels need
    # no per-axis time weighting; the axis-aware area element dA already
    # lives on each FluxMonitor (handles graded tangential cells).
    if use_flux_monitors:
        from rfx.probes.probes import _FLUX_COMPONENTS as _FC
        flux_meta = tuple(
            (fm.axis, fm.index, fm.freqs, _FC[fm.axis],
             fm.lo1, fm.hi1, fm.lo2, fm.hi2,
             fm.total_steps, fm.window, fm.window_alpha)
            for fm in flux_monitors
        )
        carry_init["flux_monitors"] = tuple(
            (fm.e1_dft, fm.e2_dft, fm.h1_dft, fm.h2_dft)
            for fm in flux_monitors
        )
    else:
        flux_meta = ()

    # Lumped RLC ADE state (one per element) — metas are Python-static
    if use_lumped_rlc:
        carry_init["rlc_states"] = tuple(rlc_states)

    # NTFF accumulators — seeded from caller, updated per step via
    # accumulate_ntff. Box indices and freqs are Python-static.
    if use_ntff:
        carry_init["ntff"] = ntff_data

    # Waveguide-port time-series carry (mirrors uniform path).
    # Phase 2 cleanup (2026-04-25) removed in-scan DFT accumulators;
    # spectra are computed POST-SCAN by a rect full-record DFT on the
    # recorded modal V/I time series.
    if use_waveguide_ports:
        carry_init["waveguide_port_accs"] = tuple(
            (
                cfg.v_probe_t,
                cfg.v_ref_t,
                cfg.i_probe_t,
                cfg.i_ref_t,
                cfg.v_inc_t,
                cfg.n_steps_recorded,
            )
            for cfg in waveguide_ports
        )
        waveguide_meta = tuple(waveguide_ports)
    else:
        waveguide_meta = ()

    # TFSF 1D auxiliary state carry. Injection axis is x (uniform on
    # NU paths we support — dz-only nonuniformity), so the 1D aux runs
    # with grid.dx spacing and the E/H corrections use scalar
    # coeff = dt / (EPS_0 * dx) etc. Oblique / +z,-z cases are
    # rejected upstream (see rfx/runners/nonuniform.py and rfx/api.py).
    if use_tfsf:
        from rfx.sources.tfsf import is_tfsf_2d as _is_tfsf_2d
        tfsf_cfg, tfsf_state = tfsf
        if _is_tfsf_2d(tfsf_cfg):
            raise ValueError(
                "TFSF oblique incidence (2D auxiliary grid) is not yet "
                "supported on nonuniform z mesh. Use angle_deg=0 along x."
            )
        if tfsf_cfg.direction not in ("+x", "-x"):
            raise ValueError(
                "TFSF on nonuniform mesh supports only direction='+x' or "
                f"'-x' (injection along uniform x axis); got {tfsf_cfg.direction!r}."
            )
        carry_init["tfsf"] = tfsf_state

    def step_fn(carry, xs):
        step_idx, src_vals = xs
        st = carry["fdtd"]

        # H update (non-uniform)
        st = update_h_nu(st, materials, dt, inv_dx_h, inv_dy_h, inv_dz_h)
        tfsf_h_state = None
        if use_tfsf:
            from rfx.sources.tfsf import apply_tfsf_h
            st = apply_tfsf_h(st, tfsf_cfg, carry["tfsf"], grid.dx, dt)
        if use_waveguide_ports:
            from rfx.sources.waveguide_port import apply_waveguide_port_h as _apply_wg_h_nu
            for cfg_meta in waveguide_meta:
                st = _apply_wg_h_nu(st, cfg_meta, step_idx, dt, grid.dx)
        if use_cpml:
            st, cpml_new = apply_cpml_h(st, cpml_params, carry["cpml"],
                                         cpml_grid, cpml_axes_eff,
                                         materials=materials)
        else:
            cpml_new = None
        if use_pmc_faces:
            from rfx.boundaries.pmc import apply_pmc_faces
            st = apply_pmc_faces(st, _pmc_faces_frozen)
        if use_tfsf:
            from rfx.sources.tfsf import update_tfsf_1d_h
            tfsf_h_state = update_tfsf_1d_h(tfsf_cfg, carry["tfsf"], grid.dx, dt)

        # Snapshot E^n for the #677 sheet operator (it REPLACES the
        # standard update at masked tangential edges with A*E^n + B*curlH).
        e_prev_sheet = (st.ex, st.ey, st.ez) if use_sheet_impedance else None

        # E update: use ADE-aware path when dispersive materials are present
        debye_new = None
        lorentz_new = None
        if use_debye or use_lorentz:
            st, debye_new, lorentz_new = _update_e_nu_dispersive(
                st, materials, dt, inv_dx, inv_dy, inv_dz,
                debye=(debye_coeffs, carry["debye"]) if use_debye else None,
                lorentz=(lorentz_coeffs, carry["lorentz"]) if use_lorentz else None,
            )
        elif aniso_eps is not None:
            from rfx.core.yee import update_e_nu_aniso
            _eex, _eey, _eez = aniso_eps
            st = update_e_nu_aniso(
                st, materials, _eex, _eey, _eez, dt,
                inv_dx, inv_dy, inv_dz,
            )
        else:
            st = update_e_nu(st, materials, dt, inv_dx, inv_dy, inv_dz)

        if use_tfsf:
            from rfx.sources.tfsf import apply_tfsf_e
            st = apply_tfsf_e(st, tfsf_cfg, tfsf_h_state, grid.dx, dt)
        if use_waveguide_ports:
            from rfx.sources.waveguide_port import apply_waveguide_port_e as _apply_wg_e_nu
            for cfg_meta in waveguide_meta:
                st = _apply_wg_e_nu(st, cfg_meta, step_idx, dt, grid.dx)
        if use_cpml:
            st, cpml_new = apply_cpml_e(st, cpml_params, cpml_new,
                                         cpml_grid, cpml_axes_eff,
                                         materials=materials)

        # PEC
        st = apply_pec(st)
        if use_pec_mask:
            # #689: default (non-periodic) is correct here — the NU
            # stepper installs no periodic BC at all, and NU grids are
            # 3-D, so both of the wrap-keeping guards are inert.
            st = apply_pec_mask(st, pec_mask,
                                two_plane_mask=pec_two_plane_mask)
        if use_pec_occupancy:
            st = apply_pec_occupancy(st, pec_occupancy)

        # #677 node-thin surface-impedance sheet operator. Contract slot:
        # AFTER apply_pec_mask/apply_pec_occupancy (PEC wins on overlap),
        # BEFORE sources and the port DFT sampling below. curlH comes from
        # the SAME shared stencil helper update_e_nu uses, on the same
        # H^{n+1/2} the E update consumed.
        if use_sheet_impedance:
            from rfx.core.yee import curl_h_nu as _curl_h_nu
            from rfx.materials.thin_conductor import (
                apply_sheet_impedance_e as _apply_sheet_e)
            _scd = jnp.promote_types(st.ex.dtype, jnp.float32)
            _curls = _curl_h_nu(
                st.hx.astype(_scd), st.hy.astype(_scd), st.hz.astype(_scd),
                inv_dx, inv_dy, inv_dz)
            st = _apply_sheet_e(st, e_prev_sheet, _curls,
                                sheet_impedance, sheet_coeffs)

        # Lumped RLC ADE update (after E update + boundaries, before sources)
        new_rlc_states = None
        if use_lumped_rlc:
            from rfx.lumped import update_rlc_element
            new_rlc_states = []
            for rlc_st, meta in zip(carry["rlc_states"], rlc_metas):
                st, rlc_st_new = update_rlc_element(st, rlc_st, meta)
                new_rlc_states.append(rlc_st_new)

        # Sources (point sources + wire port excitation)
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s])
            st = st._replace(**{sc: field})

        # Waveguide-port injection + DFT probe accumulation. The dx
        # arg is unused by the per-cell-weighted integrals (cfg already
        # stores u_widths/v_widths), but kept in the function signature
        # for back-compat.
        new_waveguide_port_accs = None
        if use_waveguide_ports:
            from rfx.sources.waveguide_port import (
                update_waveguide_port_probe,
            )
            new_waveguide_port_accs = []
            for accs, cfg_meta in zip(
                carry["waveguide_port_accs"], waveguide_meta
            ):
                cfg = cfg_meta._replace(
                    v_probe_t=accs[0],
                    v_ref_t=accs[1],
                    i_probe_t=accs[2],
                    i_ref_t=accs[3],
                    v_inc_t=accs[4],
                    n_steps_recorded=accs[5],
                )
                # TFSF-style corrections applied earlier at canonical slots.
                cfg_updated = update_waveguide_port_probe(cfg, st, dt, grid.dx)
                new_waveguide_port_accs.append((
                    cfg_updated.v_probe_t,
                    cfg_updated.v_ref_t,
                    cfg_updated.i_probe_t,
                    cfg_updated.i_ref_t,
                    cfg_updated.v_inc_t,
                    cfg_updated.n_steps_recorded,
                ))

        # Wire port V/I DFT accumulation.
        #
        # Ordering: AFTER source injection.  Issue #683 is DECIDED (by
        # measurement, 2026-08-29 — docs/design_notes/
        # issue683_sampling_order_decision_protocol.md section 9): this
        # slot is the terminal-consistent, physically correct one.  The
        # known-load circuit law holds only for the post-injection pair
        # (n*a = +0.9987/+0.9950, n*|b| = 0.08/0.32 Ohm over a six-point
        # R_L sweep at both decision bins), and the post-injection E is
        # the true field level E^{n+1} of the discrete update
        # (Ampere-identity residual 2.3e-7 vs 3.25 pre-injection).  The
        # once-CONTESTED known-load sweep was settled by that protocol run
        # (gates G0-G2 all passed; the earlier independent repro's fixture
        # had failed to load the port), and the once-UNEXAMINED Ampere
        # identity was examined in the same run (section 7).
        #
        # The uniform lane (rfx/simulation.py wire block) flipped to this
        # same POST slot for its physical V/I/V_port channels, with the
        # pre-injection drive sample kept as a separate calibration
        # reference channel for its #308/#313 off-diagonal decomposition
        # (issue #683 x #764,
        # docs/design_notes/issue683_decomposer_flip_predeclaration.md).
        # This lane's decomposition was calibrated in the POST frame
        # directly, so it needs NO reference channel and NO change.
        #
        # PASSIVE ports are unaffected either way: I reads H only, and the
        # source loop writes E only, at the SOURCE cell.
        t = step_idx.astype(jnp.float32) * dt
        new_wire_sp = None
        if use_wire_ports:
            new_wire_sp = []
            for (v_dft, i_dft, vinc_dft, v_port_dft), \
                (mi, mj, mk, comp, z0, dxi, dyj,
                 _excite, _dir, dual_xi, dual_yj,
                 dz_local, dual_zk, live_cells, d_par_cells) in \
                    zip(carry.get("wire_sparams", ()), wp_meta):
                # V = -E_comp * (PRIMAL width on the component's own axis):
                # the E edge's own length. I = the Ampere loop on the DUAL
                # face (issue #672); both metric families are precomputed in
                # wp_meta above.
                field_c = getattr(st, comp)
                v = -field_c[mi, mj, mk] * (
                    dz_local if comp == "ez" else
                    dxi if comp == "ex" else dyj)
                # Whole-port gap voltage (issue #764): the discrete line
                # integral of E across the LIVE run,
                # V_port = sum_live(-E_c * d_par,c), same PRIMAL metric
                # family per cell. Static unroll — live_cells is a static
                # tuple, so this resolves at trace time.
                v_port = -sum(
                    field_c[ci, cj, ck] * dp
                    for (ci, cj, ck), dp in zip(live_cells, d_par_cells))
                i_val = wire_port_current(
                    st.hx, st.hy, st.hz, comp, mi, mj, mk,
                    dual_xi, dual_yj, dual_zk)
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase = jnp.exp(-1j * 2.0 * jnp.pi * sp_freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_wire_sp.append((
                    v_dft + v * phase,
                    i_dft + i_val * phase,
                    vinc_dft,
                    v_port_dft + v_port * phase,
                ))

        # DFT plane probe accumulation (identical math to uniform path)
        new_dft_planes = None
        if use_dft_planes:
            t_plane = step_idx.astype(jnp.float32) * dt
            new_dft_planes = []
            for acc, (component, axis, index, freqs, region) in zip(
                carry["dft_planes"], dft_meta
            ):
                field = getattr(st, component)
                if region is None:
                    lo1 = lo2 = 0
                    if axis == 0:
                        hi1, hi2 = field.shape[1], field.shape[2]
                    elif axis == 1:
                        hi1, hi2 = field.shape[0], field.shape[2]
                    else:
                        hi1, hi2 = field.shape[0], field.shape[1]
                else:
                    lo1, hi1, lo2, hi2 = region
                if axis == 0:
                    plane = field[index, lo1:hi1, lo2:hi2]
                elif axis == 1:
                    plane = field[lo1:hi1, index, lo2:hi2]
                else:
                    plane = field[lo1:hi1, lo2:hi2, index]
                phase = jnp.exp(-1j * 2.0 * jnp.pi * freqs * t_plane)
                new_dft_planes.append(
                    acc + plane[None, :, :] * phase[:, None, None] * dt
                )

        # Flux monitor accumulation (mirrors uniform scan body in
        # rfx/simulation.py). H is offset +dx/2 along the normal axis on
        # the Yee grid; average H at idx-1 and idx to co-locate with E at
        # idx for a correct Poynting cross-product. E is sampled at
        # t=step*dt, H at t-dt/2.
        new_flux_monitors = None
        if use_flux_monitors:
            from rfx.core.dft_utils import dft_window_weight as _dft_w
            t_flux = step_idx.astype(jnp.float32) * dt
            new_flux_monitors = []
            for (e1_acc, e2_acc, h1_acc, h2_acc), (
                ax, idx, fqs, comp_names, _lo1, _hi1, _lo2, _hi2,
                _tot_steps, _win_name, _win_alpha,
            ) in zip(carry["flux_monitors"], flux_meta):
                e1n, e2n, h1n, h2n = comp_names
                idx_m1 = max(idx - 1, 0)
                if ax == 0:
                    e1 = getattr(st, e1n)[idx, _lo1:_hi1, _lo2:_hi2]
                    e2 = getattr(st, e2n)[idx, _lo1:_hi1, _lo2:_hi2]
                    h1 = (getattr(st, h1n)[idx_m1, _lo1:_hi1, _lo2:_hi2] + getattr(st, h1n)[idx, _lo1:_hi1, _lo2:_hi2]) * 0.5
                    h2 = (getattr(st, h2n)[idx_m1, _lo1:_hi1, _lo2:_hi2] + getattr(st, h2n)[idx, _lo1:_hi1, _lo2:_hi2]) * 0.5
                elif ax == 1:
                    e1 = getattr(st, e1n)[_lo1:_hi1, idx, _lo2:_hi2]
                    e2 = getattr(st, e2n)[_lo1:_hi1, idx, _lo2:_hi2]
                    h1 = (getattr(st, h1n)[_lo1:_hi1, idx_m1, _lo2:_hi2] + getattr(st, h1n)[_lo1:_hi1, idx, _lo2:_hi2]) * 0.5
                    h2 = (getattr(st, h2n)[_lo1:_hi1, idx_m1, _lo2:_hi2] + getattr(st, h2n)[_lo1:_hi1, idx, _lo2:_hi2]) * 0.5
                else:
                    e1 = getattr(st, e1n)[_lo1:_hi1, _lo2:_hi2, idx]
                    e2 = getattr(st, e2n)[_lo1:_hi1, _lo2:_hi2, idx]
                    h1 = (getattr(st, h1n)[_lo1:_hi1, _lo2:_hi2, idx_m1] + getattr(st, h1n)[_lo1:_hi1, _lo2:_hi2, idx]) * 0.5
                    h2 = (getattr(st, h2n)[_lo1:_hi1, _lo2:_hi2, idx_m1] + getattr(st, h2n)[_lo1:_hi1, _lo2:_hi2, idx]) * 0.5
                t_f64 = t_flux.astype(jnp.float64)
                fqs64 = fqs.astype(jnp.float64)
                _w = _dft_w(step_idx, _tot_steps, _win_name, _win_alpha).astype(jnp.float64)
                phase_e = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * t_f64)
                phase_h = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * (t_f64 - jnp.float64(dt * 0.5)))
                kernel_e = (phase_e[:, None, None] * dt * _w).astype(jnp.complex128)
                kernel_h = (phase_h[:, None, None] * dt * _w).astype(jnp.complex128)
                new_flux_monitors.append((
                    e1_acc + e1.astype(jnp.float64)[None, :, :] * kernel_e,
                    e2_acc + e2.astype(jnp.float64)[None, :, :] * kernel_e,
                    h1_acc + h1.astype(jnp.float64)[None, :, :] * kernel_h,
                    h2_acc + h2.astype(jnp.float64)[None, :, :] * kernel_h,
                ))

        # NTFF: accumulate tangential E/H DFT on 6 box faces
        new_ntff = None
        if use_ntff:
            from rfx.farfield import accumulate_ntff
            new_ntff = accumulate_ntff(carry["ntff"], st, ntff_box, dt, step_idx)

        # TFSF 1D auxiliary E-field update (mirrors uniform scan body:
        # called AFTER sources, closes the leapfrog step).
        tfsf_new = None
        if use_tfsf:
            from rfx.sources.tfsf import update_tfsf_1d_e
            t_tfsf = step_idx.astype(jnp.float32) * dt
            tfsf_new = update_tfsf_1d_e(tfsf_cfg, tfsf_h_state, grid.dx, dt, t_tfsf)

        # Probes
        if emit_time_series and prb_meta:
            samples = [getattr(st, pc)[pi, pj, pk] for pi, pj, pk, pc in prb_meta]
            probe_out = jnp.stack(samples)
        else:
            probe_out = jnp.zeros(0)

        new_carry = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cpml_new
        if use_debye and debye_new is not None:
            new_carry["debye"] = debye_new
        if use_lorentz and lorentz_new is not None:
            new_carry["lorentz"] = lorentz_new
        if use_wire_ports and new_wire_sp is not None:
            new_carry["wire_sparams"] = tuple(new_wire_sp)
        if use_dft_planes and new_dft_planes is not None:
            new_carry["dft_planes"] = tuple(new_dft_planes)
        if use_flux_monitors and new_flux_monitors is not None:
            new_carry["flux_monitors"] = tuple(new_flux_monitors)
        if use_lumped_rlc and new_rlc_states is not None:
            new_carry["rlc_states"] = tuple(new_rlc_states)
        if use_ntff and new_ntff is not None:
            new_carry["ntff"] = new_ntff
        if use_waveguide_ports and new_waveguide_port_accs is not None:
            new_carry["waveguide_port_accs"] = tuple(new_waveguide_port_accs)
        if use_tfsf and tfsf_new is not None:
            new_carry["tfsf"] = tfsf_new
        return new_carry, probe_out

    return _NUScanSetup(
        step_fn=step_fn,
        carry_init=carry_init,
        src_waveforms=src_waveforms,
        dt=dt,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        dft_planes=dft_planes,
        flux_monitors=flux_monitors,
        waveguide_meta=waveguide_meta,
        wp_meta=wp_meta,
        sp_freqs=sp_freqs,
        use_wire_ports=use_wire_ports,
        use_dft_planes=use_dft_planes,
        use_flux_monitors=use_flux_monitors,
        use_lumped_rlc=use_lumped_rlc,
        use_ntff=use_ntff,
        use_waveguide_ports=use_waveguide_ports,
    )


def run_nonuniform(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    pec_mask=None,
    pec_two_plane_mask=None,
    pec_occupancy=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    dft_planes: list | None = None,
    rlc_metas: tuple = (),
    rlc_states: tuple = (),
    ntff_box=None,
    ntff_data=None,
    waveguide_ports: list | None = None,
    tfsf: tuple | None = None,
    flux_monitors: list | None = None,
    checkpoint: bool = False,
    emit_time_series: bool = True,
    checkpoint_every: int | None = None,
    n_warmup: int = 0,
    aniso_eps: tuple | None = None,
    sheet_impedance=None,
) -> dict:
    """Run non-uniform FDTD via jax.lax.scan.

    Parameters
    ----------
    sources : list of (i, j, k, component, waveform_array)
    probes : list of (i, j, k, component)
    wire_ports : list of dict with keys:
        mid_i, mid_j, mid_k, component, impedance, waveform_array
    s_param_freqs : (n_freqs,) array for S-param DFT
    debye : (DebyeCoeffs, DebyeState) or None
    lorentz : (LorentzCoeffs, LorentzState) or None
    dft_planes : list of DFTPlaneProbe or None
        Frequency-domain plane accumulators. The accumulation is
        identical to the uniform path (acc += field * exp(-j2pi f t) * dt);
        dt is scalar on both paths so no per-axis weighting is required.
    """
    setup = _build_nu_scan(
        grid, materials, n_steps,
        pec_mask=pec_mask,
        pec_two_plane_mask=pec_two_plane_mask,
        pec_occupancy=pec_occupancy,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        s_param_freqs=s_param_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=pec_faces,
        pmc_faces=pmc_faces,
        dft_planes=dft_planes,
        rlc_metas=rlc_metas,
        rlc_states=rlc_states,
        ntff_box=ntff_box,
        ntff_data=ntff_data,
        waveguide_ports=waveguide_ports,
        tfsf=tfsf,
        flux_monitors=flux_monitors,
        emit_time_series=emit_time_series,
        aniso_eps=aniso_eps,
        sheet_impedance=sheet_impedance,
    )
    step_fn = setup.step_fn
    carry_init = setup.carry_init
    src_waveforms = setup.src_waveforms

    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)

    # ---- n_warmup split --------------------------------------------------
    # When n_warmup > 0, run the first n_warmup steps with the carry
    # stop_gradient'd so AD builds no tape for that transient lead-in
    # (issue #40). Only the trailing n_optimize = n_steps - n_warmup
    # steps participate in reverse-mode autodiff.
    #
    # The truncation error DEPENDS ON DISTANCE FROM THE SOURCE TO THE
    # DESIGN REGION -- it is not a blanket property of n_warmup itself
    # (corrects an earlier version of this comment that said the
    # design-relevant support "generally" extends into the warmup window;
    # see the #626 addendum below for why that was an overgeneralization
    # from a single, worst-case fixture). Mechanism: severing the carry
    # severs every gradient path from a design variable's influence during
    # steps < n_warmup back into the loss -- but only steps during which
    # the WAVEFRONT HAS ALREADY REACHED the design region carry any such
    # influence to sever. Before the wavefront arrives, the field there
    # (and hence the loss's sensitivity to that cell) is ~0, so severing
    # those steps' carry discards ~nothing and the gradient is
    # (near-)exact -- n_warmup is genuinely free compute/memory relief in
    # that regime, not merely an approximation. Define
    #     K_safe ~= floor(min_distance(source, design_region) / (C0 * dt))
    # where min_distance is the MINIMUM over EVERY active source AND over
    # each source's own spatial extent (not just its nominal position) --
    # a TFSF plane-wave source, for example, illuminates from an entire
    # box face, not a point, so "distance from the source" means distance
    # from the NEAREST point of that box to the nearest point of the
    # design region. (grid steps for the wavefront to reach the closest
    # design cell, C0 = vacuum lightspeed -- a conservative floor valid
    # for any slower, non-vacuum propagation). For n_warmup <= K_safe
    # truncation error is negligible; beyond it, error grows and can
    # reach the full gradient magnitude by the time n_warmup reaches the
    # loss window.
    #
    # Forward output is exactly n_warmup-invariant (bit-identical) in every
    # placement measured. Two measured regimes (issue #626 part 2 /
    # addendum, both vs an independent central-FD oracle):
    #   - NEAR-SOURCE placement (design cell ~3 cells from the source, so
    #     K_safe ~ 0 -- the wavefront is already present at every step):
    #     error grows smoothly and monotonically from a ~1.5% noise floor
    #     at small n_warmup up to 58% at the loss-window boundary itself,
    #     and EXACTLY zero (gradient fully vanished) once n_warmup extends
    #     far enough into the loss window (fixture:
    #     tests/test_n_warmup.py::test_warmup_truncation_error_grows_with_k,
    #     n_steps=100, loss window [80,100)).
    #   - FAR-FROM-SOURCE placement (design cell 62 cells from the source,
    #     K_safe=108 measured from the grid's own dt): AD matches the K=0
    #     FD oracle to <0.01% rel_err through K=88 (deep sub-wavefront,
    #     K_safe-20), then grows SMOOTHLY -- not a sharp cliff -- as K
    #     approaches K_safe: 0.015% at K=98, 0.098% at K=102, 0.259% at
    #     K=104, 0.601% at K=106, 1.186% AT K=108 (=K_safe itself). Every
    #     one of those K<=K_safe values sits comfortably inside this
    #     repo's own established ~1.5% AD-vs-FD noise floor. Past K_safe
    #     the curve keeps growing (1.56% at K=109, ~2.5% around K=112),
    #     though non-monotonically farther out (numerical-dispersion
    #     ripple, not noise) before the broader trend continues upward
    #     toward the loss window (75% by K=200). K_safe is therefore a
    #     BOUND on where truncation stays within the established noise
    #     floor, not a literal discontinuity -- an earlier version of
    #     this comment read the original coarse (multiples-of-20) sweep,
    #     which skipped every point in [101,119], as "exactly the
    #     wavefront-arrival prediction" and overstated how sharp the
    #     transition is (fixture:
    #     scripts/diagnostics/i626_n_warmup_wavefront_locality.py,
    #     n_steps=220, loss window [180,220), finely sampled around
    #     K_safe).
    # The near-source curve is the WORST case, not the general case --
    # use K_safe to decide whether a given source/design placement is in
    # the exact-or-noise-floor or the truncated regime; do not read
    # "n_warmup truncates the gradient" as true unconditionally.
    if n_warmup > 0:
        if n_warmup >= n_steps:
            raise ValueError(
                f"n_warmup ({n_warmup}) must be < n_steps ({n_steps})"
            )
        warmup_steps = jnp.arange(n_warmup, dtype=jnp.int32)
        warmup_xs = (warmup_steps, src_waveforms[:n_warmup])
        warmup_final, warmup_ys = jax.lax.scan(step_fn, carry_init, warmup_xs)
        carry_init = jax.tree_util.tree_map(
            jax.lax.stop_gradient, warmup_final
        )
        warmup_ys = jax.lax.stop_gradient(warmup_ys)
        xs = (
            jnp.arange(n_warmup, n_steps, dtype=jnp.int32),
            src_waveforms[n_warmup:],
        )
        n_steps_opt = n_steps - n_warmup
    else:
        warmup_ys = None
        n_steps_opt = n_steps

    use_segmented = (
        checkpoint_every is not None
        and 0 < int(checkpoint_every) < n_steps_opt
    )
    if use_segmented:
        # Scan-of-scan: outer scan over segments wrapped in jax.checkpoint
        # forces XLA to remat the inner scan during backward, so the AD
        # tape only stores carry at segment boundaries (≈ sqrt(n_steps)
        # × carry_size when checkpoint_every ≈ sqrt(n_steps)).
        chunk = int(checkpoint_every)
        n_segments = (n_steps_opt + chunk - 1) // chunk
        pad = n_segments * chunk - n_steps_opt
        opt_steps = xs[0]
        opt_src = xs[1]
        if pad > 0:
            steps_padded = jnp.arange(
                int(opt_steps[0]),
                int(opt_steps[0]) + n_segments * chunk,
                dtype=jnp.int32,
            )
            n_sources = opt_src.shape[1]
            src_pad = jnp.zeros((pad, n_sources), dtype=opt_src.dtype)
            src_padded = jnp.concatenate([opt_src, src_pad], axis=0)
        else:
            steps_padded = opt_steps
            src_padded = opt_src

        seg_steps = steps_padded.reshape(n_segments, chunk)
        seg_src = src_padded.reshape(n_segments, chunk, src_padded.shape[1])

        def segment_body(carry, segment_xs):
            return jax.lax.scan(step_fn, carry, segment_xs)

        seg_body = jax.checkpoint(segment_body)
        final, segment_ys = jax.lax.scan(
            seg_body, carry_init, (seg_steps, seg_src)
        )
        flat = segment_ys.reshape((n_segments * chunk,) + segment_ys.shape[2:])
        opt_ys = flat[:n_steps_opt]
    else:
        body = jax.checkpoint(step_fn) if checkpoint else step_fn
        final, opt_ys = jax.lax.scan(body, carry_init, xs)
    # Merge warmup + optimize outputs back into one time_series so the
    # downstream result shape stays (n_steps, n_probes).
    if warmup_ys is not None:
        time_series = jnp.concatenate([warmup_ys, opt_ys], axis=0)
    else:
        time_series = opt_ys

    return _assemble_nu_result(setup, final, time_series)


def _assemble_nu_result(setup: _NUScanSetup, final: dict, time_series) -> dict:
    """Assemble the NU result dict (pure code motion from run_nonuniform,
    #383). Shared by :func:`run_nonuniform` and
    :func:`run_nonuniform_until_decay` so the two paths return the exact
    same schema (state, time_series, dt, conditional dft_planes /
    flux_monitors / ntff_data / waveguide_ports / s_params...)."""
    dt = setup.dt
    dft_planes = setup.dft_planes
    flux_monitors = setup.flux_monitors
    wire_ports = setup.wire_ports
    waveguide_meta = setup.waveguide_meta
    wp_meta = setup.wp_meta
    sp_freqs = setup.sp_freqs
    use_wire_ports = setup.use_wire_ports
    use_dft_planes = setup.use_dft_planes
    use_flux_monitors = setup.use_flux_monitors
    use_lumped_rlc = setup.use_lumped_rlc
    use_ntff = setup.use_ntff
    use_waveguide_ports = setup.use_waveguide_ports

    result = {
        "state": final["fdtd"],
        "time_series": time_series,
        "dt": dt,
    }

    # Surface final RLC ADE states (per element).
    if use_lumped_rlc:
        result["rlc_states"] = tuple(final["rlc_states"])

    # Repack DFT plane probes with their final accumulators.
    if use_dft_planes:
        result["dft_planes"] = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, final["dft_planes"])
        )

    # Repack flux monitors with their final E/H DFT accumulators so the
    # caller can call flux_spectrum() on them (same schema as uniform).
    if use_flux_monitors:
        result["flux_monitors"] = tuple(
            mon._replace(e1_dft=e1, e2_dft=e2, h1_dft=h1, h2_dft=h2)
            for mon, (e1, e2, h1, h2) in zip(
                flux_monitors, final["flux_monitors"]
            )
        )

    # Surface final NTFF DFT accumulators
    if use_ntff:
        result["ntff_data"] = final["ntff"]

    # Surface final waveguide-port configs (with recorded modal V/I
    # time series; spectra are extracted post-scan via rect-DFT).
    if use_waveguide_ports:
        result["waveguide_ports"] = tuple(
            cfg_meta._replace(
                v_probe_t=accs[0],
                v_ref_t=accs[1],
                i_probe_t=accs[2],
                i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(
                waveguide_meta, final["waveguide_port_accs"]
            )
        )

    # ---- Extract full S-matrix column from wire port DFTs ----
    #
    # Each port has a V/I DFT pair. The wave decomposition below is
    # DIRECTION-FREE, and that is a physics statement, not a shortcut
    # (issue #673). The port's outward normal is not a degree of freedom
    # for a lumped gap V/I pair:
    #
    #   V = -E_comp * d_parallel      (the E edge's own length)
    #   I = (curl H)_comp * A_perp    (right-hand loop about the SAME axis)
    #
    # Neither expression contains the port's position or an outward
    # normal, so their RELATIVE sign is fixed by the sampling code and no
    # port placement can flip it. The E update at that edge enforces
    # eps*dE/dt + sigma*E = (curl H)_comp, so in phasor form, with
    # C = eps*A/d and the runner's sigma = n_live*d/(Z0*dp1*dp2),
    #
    #       V / I = -1 / (n_live/Z0 + j*w*C + Y_ext)
    #
    # for any passive Y_ext attached across the gap (Re Y_ext >= 0), at a
    # port that is NOT itself driven. Hence Z_in = -V/I sits in the right
    # half plane and
    #
    #       S = (Z_in - Z0)/(Z_in + Z0) = (V + Z0*I)/(V - Z0*I),
    #
    # which is the same ratio as a = (-V + Z0*I)/2, b = (-V - Z0*I)/2.
    # The reciprocal form maps that passive disk to its EXTERIOR, i.e. it
    # returns |S| >= 1 at EVERY bin on ANY passive structure — which is
    # what the pre-#673 direction switch did on "-x"/"-y". That is what
    # fixes the CONVENTION; the convention is then applied unconditionally,
    # exactly as the uniform lane does.
    # Scope note, measured: the |S| <= 1 consequence is a statement about an
    # UNDRIVEN port, and it is a property of the port CELL rather than of the
    # structure across the gap (vacuum, PEC plates shorting the gap and an
    # eps_r=10 slab filling it all read S11 = -0.600000 at n_live = 4).
    # That is exactly why the all-passive fallback below is INTENTIONALLY
    # frozen on this per-cell convention: no physical falsifier can gate a
    # load-independent reading, so it is a diagnostic/self-consistency
    # channel, not S_jj (S_jj requires driving port j) — and the
    # excite=False locks (lane-parity closed forms, sigma ORACLE 1/2) keep
    # their n_live sensitivity as witnesses. Do not "unify" it with the
    # driven diagonal below.
    #
    # DRIVEN diagonal (issue #764). At a genuinely excited port the per-cell
    # convention above stops tracking the load entirely — PROVENANCE, all
    # measured before the #764 fix: one geometry with only `excite` flipped
    # read S11(0.2 GHz) = -0.600000 passive against +0.999670-0.022145j
    # driven though the quasi-static input impedance is identical; the
    # 2-port MSL stub in tests/test_twoport_wire_port.py reached
    # max|S11| = 4.648 (the reciprocal class that (-V - Z0*I)/(-V + Z0*I)
    # applied to a driven port yields); a matched 50 ohm load read
    # S11 = +0.35426 and a PEC short +0.26780. Root cause was the #313/#318
    # frame mismatch: V and I sampled at ONE cell measure the per-cell
    # Z0/n_live while the reflection formula references the whole-port Z0.
    # The fix is frame-consistent terminal quantities:
    #
    #   V_port = sum_live(-E_c * d_par,c)   (whole-gap line integral,
    #                                        PRIMAL metric per cell, #672)
    #   I      = Ampere loop at the LIVE-run midpoint cell (DUAL metrics)
    #   S_kk   = (V_port - Z0*I) / (V_port + Z0*I)
    #
    # where Z0 is the whole-port port.impedance — the impedance #318
    # physically realizes as the series termination (n_live cells of
    # sigma-folded Z0/n_live in series). Sense: from the pinned per-cell law
    # with equal per-cell injection I0, V_port + Z0*I = Z0*I0 is a
    # drive-only constant (the incident wave), and closing the loop through
    # the external DUT gives V_port = +Z_L*I and I = Z0*I0/(Z0+Z_L) — the
    # measured #683 circuit law. The opposite branch sign (V_port = -Z_L*I)
    # would put I = Z0*I0/(Z0-Z_L), divergent at a matched load — refuted.
    # Only the SUM is KVL/Faraday-constrained on the staggered grid; a
    # single cell is not (a PEC short forces sum V_c = 0 while V_mid stays
    # finite — why the short used to read +0.268: a structural error, not a
    # scale factor).
    # Quasi-static check on the PASSIVE path: with Y_ext -> 0 and
    # w*C*Z0/n_live << 1 the per-cell convention gives
    # S11 -> (1 - n_live)/(1 + n_live), measured to 5 decimals on both
    # lanes (tests/test_nu_wire_port_lane_parity.py).
    #
    # `direction` is still carried on the port spec — the reference-plane
    # path (add_port(reference_plane_cells=...)) needs it for the outboard
    # sign — but it must NOT enter this wave split.
    #
    # Given these, with ONE excited port `k` the full k-th column of
    # the S-matrix is
    #       S[j, k] = b_j / a_k          (j = every port)
    # which reduces to the familiar S11 = b_k/a_k for j==k (reflection)
    # and to S21 = b_2/a_1 for j != k (transmission). Other columns of
    # the S matrix stay zero — callers need to run additional sims
    # with different excited ports to fill them (reciprocity lets us
    # infer S12 from S21 for passive networks).
    #
    # Which (a, b) pair fills a GENUINELY excited column is the issue
    # #770 whole-port pair (see the in-loop derivation below); the
    # per-cell `_ab` split above survives only in the all-passive
    # diagnostic fallback.
    if use_wire_ports and "wire_sparams" in final:
        import numpy as _np
        n_wp = len(wire_ports)
        nf = len(sp_freqs)
        # Tracer-safe accumulation (issue #70): build S with jnp + .at[].set
        # so jax.grad on an objective that pulls through the wire_sparams
        # extractor does not hit TracerArrayConversionError. Concrete-path
        # callers still get a jnp.ndarray that numpy consumers accept via
        # np.asarray(...) (jnp arrays implement __array__).
        S = jnp.zeros((n_wp, n_wp, nf), dtype=jnp.complex64)

        # Pick the excited ports as the filled columns. If no port is
        # excited (all passive), fall back to the legacy diagnostic
        # extraction (no meaningful S-matrix in that case).
        # ``genuinely_excited`` (issues #764 + #770) keys the whole-port
        # wave pair to meta[7] ONLY — the all-passive fallback stays on
        # the frozen per-cell convention (see the scope note above).
        genuinely_excited = [
            idx for idx, meta in enumerate(wp_meta) if meta[7]]
        excited_idx = genuinely_excited
        if not excited_idx:
            excited_idx = list(range(n_wp))   # legacy: treat all as self-excited

        def _ab(v_dft, i_dft, z0):
            """Return (a_incoming, b_outgoing) at one port.

            Direction-free, written in the uniform lane's sign so it reads
            line-for-line against rfx/runners/uniform.py (the overall -1
            cancels in the ratio b/a). See the derivation above (#673).
            """
            zi = z0 * i_dft
            return (-v_dft + zi) / 2.0, (-v_dft - zi) / 2.0

        ab_per_port = []
        for (v_dft, i_dft, _, _vp), meta in zip(final["wire_sparams"],
                                                wp_meta):
            z0 = meta[4]
            a, b = _ab(v_dft, i_dft, z0)
            ab_per_port.append((a, b))

        # All-passive fallback ONLY: the frozen per-cell diagnostic
        # convention (see the scope note above — no physical falsifier
        # can gate a load-independent reading; do not "unify").
        if not genuinely_excited:
            for k in excited_idx:
                a_k = ab_per_port[k][0]
                safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k,
                                     jnp.ones_like(a_k))
                for j in range(n_wp):
                    b_j = ab_per_port[j][1]
                    S = S.at[j, k, :].set(b_j / safe_a_k)

        # Issue #764 + #770: every GENUINELY excited column is the
        # whole-port wave pair, frame-consistent across diagonal and
        # off-diagonal (adjudicated against external physics 2026-08-29,
        # docs/design_notes/issue770_offdiag_adjudication_predeclaration.md
        # + results note):
        #
        #   a_k = (V_port,k + Z0*I_k) / (2*sqrt(Z0))   # measured drive-only
        #                                              # constant Z0*I0 (#683)
        #   b_j = (V_port,j - Z0*I_j) / (2*sqrt(Z0))   # outgoing at port j
        #   S[j, k] = b_j / a_k    (diagonal reduces to the #764
        #                           (V_port - Z0*I)/(V_port + Z0*I))
        #
        # The old mixed split (per-cell midpoint v against the whole-port
        # Z0, the open defect #764 section 6 named) measured neither
        # frame on the canonical thru (|S21| 0.65-0.97, drifting); the
        # whole-port pair measured |S21| = 0.934-0.995 vs the external
        # flux referee 0.971-0.997, power closure deficit 0.9-4.4% vs
        # the 0.2-4.0% flux gap, reciprocity 2.67e-4, and lane parity
        # 3.97e-6 against the uniform decomposer (#770 harness).
        # Global receive sign: +1, DC-witness pinned (S21(DC) -> +1,
        # dev -0.058/-0.115 rad at 0.5/1 GHz; flipped channel at ~pi).
        for k in genuinely_excited:
            v_port_k = final["wire_sparams"][k][3]
            i_k = final["wire_sparams"][k][1]
            z0_k = wp_meta[k][4]
            a_k = v_port_k + z0_k * i_k
            safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k, jnp.ones_like(a_k))
            sqrt_z0_k = jnp.sqrt(z0_k)
            for j in range(n_wp):
                v_port_j = final["wire_sparams"][j][3]
                i_j = final["wire_sparams"][j][1]
                z0_j = wp_meta[j][4]
                b_j = (v_port_j - z0_j * i_j) * (
                    sqrt_z0_k / jnp.sqrt(z0_j))
                S = S.at[j, k, :].set(b_j / safe_a_k)

        result["s_params"] = S
        result["s_param_freqs"] = _np.array(sp_freqs)
        # Raw per-port DFT accumulators (v_dft, i_dft, v_inc_dft,
        # v_port_dft), surfaced for diagnostics/validation harnesses
        # (issue #764; mirrors the uniform lane's raw-acc access via
        # forward()'s wire_port_sparams).
        result["wire_sparams_raw"] = final["wire_sparams"]

    return result


def run_nonuniform_until_decay(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    *,
    decay_by: float = 1e-3,
    check_interval: int = 50,
    min_steps: int = 100,
    max_steps: int = 50_000,
    decay_energy_consecutive: int = 2,
    radiated_flux_box: tuple | None = None,
    flux_env_checks: int = 4,
    report_every: int | None = None,
    report_label: str = "",
    pec_mask=None,
    pec_two_plane_mask=None,
    pec_occupancy=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    dft_planes: list | None = None,
    rlc_metas: tuple = (),
    rlc_states: tuple = (),
    ntff_box=None,
    ntff_data=None,
    waveguide_ports: list | None = None,
    tfsf: tuple | None = None,
    flux_monitors: list | None = None,
    emit_time_series: bool = True,
    aniso_eps: tuple | None = None,
    sheet_impedance=None,
) -> dict:
    """Run non-uniform FDTD until the interior-domain energy decays (#383).

    NU port of the issue #169 total-domain-energy stop criterion: a
    Python loop drives constant-length jitted chunks (``chunk =
    check_interval`` steps) of the SAME ``step_fn`` that
    :func:`run_nonuniform` scans, threading the SAME carry (NTFF Kahan
    compensation arrays included) across chunks. Per-chunk ``xs``
    continue the GLOBAL step indices, exactly like the ``n_warmup``
    split in :func:`run_nonuniform`, so source waveforms / DFT phases
    are identical to a fixed-step run of the same length.

    Stop criterion (absorbing boundaries — the ONLY supported class on
    this entry point; the routing layer in ``rfx/api/_execute.py`` gates
    on ``boundary in ('cpml', 'upml')``):

    * At each chunk boundary (after ``k * check_interval`` steps), once
      ``steps_done >= min_steps``, compute the host-side dV-weighted
      interior energy ``U = sum((Ex^2 + ... + Hz^2) * dV)`` over the
      non-CPML interior slice, with ``dV = dx[i] * dy[j] * dz[k]`` the
      per-cell primal volume (all six components weighted by the primal
      cell dV — the same staggering simplification the uniform path
      makes; on a uniform mesh this reduces to the uniform criterion up
      to the constant dV factor).
    * ``peak_U`` is updated at checks only, BEFORE the compare.
    * The stop fires after ``decay_energy_consecutive`` CONSECUTIVE
      sub-threshold checks (``U < decay_by * peak_U``); any
      above-threshold check resets the counter. ``>= 2`` is mandatory
      for the same reason as the uniform path: the interior energy is
      not null-free (transient inter-packet minima recover).
    * ``decay_by = 0.0`` preserves the forced-N escape (``U < 0`` is
      never true for ``U >= 0``), so the loop runs exactly
      ``max_steps`` steps.
    * The loop is hard-bounded by ``max_steps``; a final partial chunk
      (shorter than ``check_interval``) is allowed so the bound is
      exact.

    Check cadence note: the uniform ``run_until_decay`` checks after
    steps ``k * interval + 1`` (its 0-based ``step % interval == 0``
    fires at step index ``k * interval``); the chunked NU loop checks
    after ``k * interval`` steps. The <= 1-step cadence difference is
    deliberate (one XLA program per chunk length) — cross-lane stop-step
    equality is NOT a contract (host/device float order already differs
    between the lanes).

    On a closed domain (no absorbing pads) the interior energy does not
    decay and this function runs to ``max_steps``; there is NO
    point-field fallback on the NU lane (no legacy behavior to
    preserve). The routing layer warn-and-drops ``until_decay`` for
    non-absorbing NU boundaries instead of calling this.

    Forward-only: this is a host loop (not ``lax.scan``); it is not
    reachable from ``forward()``/``optimize()`` and carries no AD tape
    contract. ``checkpoint`` / ``checkpoint_every`` / ``n_warmup`` are
    deliberately absent from the signature (the caller raises before
    dispatching here).

    Returns
    -------
    dict
        Same schema as :func:`run_nonuniform` (assembled by the shared
        :func:`_assemble_nu_result`), plus a ``"decay_checks"``
        diagnostic: a list of ``(step, U, peak_U)`` host-float tuples,
        one per eligible energy check. The trace exists only on THIS
        runner-dict return — ``run_nonuniform_path`` does not copy it
        onto the public ``Result`` — so consumers that need to verify
        the fire condition actually held (workspace rule R5) must call
        this module function directly.
    """
    if check_interval < 1:
        raise ValueError(
            f"check_interval must be >= 1, got {check_interval}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    setup = _build_nu_scan(
        grid, materials, max_steps,
        pec_mask=pec_mask,
        pec_two_plane_mask=pec_two_plane_mask,
        pec_occupancy=pec_occupancy,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        s_param_freqs=s_param_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=pec_faces,
        pmc_faces=pmc_faces,
        dft_planes=dft_planes,
        rlc_metas=rlc_metas,
        rlc_states=rlc_states,
        ntff_box=ntff_box,
        ntff_data=ntff_data,
        waveguide_ports=waveguide_ports,
        tfsf=tfsf,
        flux_monitors=flux_monitors,
        emit_time_series=emit_time_series,
        aniso_eps=aniso_eps,
        sheet_impedance=sheet_impedance,
    )
    step_fn = setup.step_fn
    carry = setup.carry_init

    # Source table: pad/truncate to max_steps so every chunk slice is
    # full-length (mirrors the uniform run_until_decay pad/truncate).
    src_waveforms = jnp.asarray(setup.src_waveforms)
    n_have = int(src_waveforms.shape[0])
    if n_have < max_steps:
        src_waveforms = jnp.pad(
            src_waveforms, ((0, max_steps - n_have), (0, 0)))
    elif n_have > max_steps:
        src_waveforms = src_waveforms[:max_steps]

    # One compiled program per chunk length: full chunks share one XLA
    # executable; the final partial chunk (if any) compiles once more.
    @jax.jit
    def _run_chunk(carry_in, xs):
        return jax.lax.scan(step_fn, carry_in, xs)

    # Non-CPML interior slice bounds + per-cell primal volume dV
    # (Python ints / concrete arrays — the reduction below is host-side;
    # NonUniformGrid is always 3D).
    _ix0, _ix1 = grid.pad_x_lo, grid.nx - grid.pad_x_hi
    _iy0, _iy1 = grid.pad_y_lo, grid.ny - grid.pad_y_hi
    _iz0, _iz1 = grid.pad_z_lo, grid.nz - grid.pad_z_hi
    _dV = (
        jnp.asarray(grid.dx_arr)[_ix0:_ix1, None, None]
        * jnp.asarray(grid.dy_arr)[None, _iy0:_iy1, None]
        * jnp.asarray(grid.dz)[None, None, _iz0:_iz1]
    )

    def _interior_energy(state) -> float:
        """dV-weighted interior field energy (host float)."""
        sx, sy, sz = (slice(_ix0, _ix1), slice(_iy0, _iy1), slice(_iz0, _iz1))
        u = (state.ex[sx, sy, sz] ** 2 + state.ey[sx, sy, sz] ** 2
             + state.ez[sx, sy, sz] ** 2 + state.hx[sx, sy, sz] ** 2
             + state.hy[sx, sy, sz] ** 2 + state.hz[sx, sy, sz] ** 2)
        return float(jnp.sum(u * _dV))

    # #388 opt-in RADIATED-FLUX stop (NU lane; mirrors the uniform run_until_decay path):
    # stop on the outgoing Poynting flux through a Huygens box instead of the interior energy,
    # so the non-radiating static soft-source charge (which FLOORS the energy criterion) and
    # near-Nyquist buzz do not stall the stop. Opt-in; default keeps the energy criterion.
    use_flux_stop = radiated_flux_box is not None
    if use_flux_stop:
        _flo = position_to_index(grid, radiated_flux_box[0])
        _fhi = position_to_index(grid, radiated_flux_box[1])
        _bl = (min(_flo[0], _fhi[0]), max(_flo[0], _fhi[0]),
               min(_flo[1], _fhi[1]), max(_flo[1], _fhi[1]),
               min(_flo[2], _fhi[2]), max(_flo[2], _fhi[2]))

    def _radiated_power(state) -> float:
        """Net outgoing Poynting flux ∮ (E×H)·n̂ over the box (co-located approx; a stop
        criterion needs only the DECAY, so the half-cell stagger and per-cell dA are neglected
        — they are fixed weightings that do not change the decay rate)."""
        ex, ey, ez = state.ex, state.ey, state.ez
        hx, hy, hz = state.hx, state.hy, state.hz
        il, ih, jl, jh, kl, kh = _bl
        jj, kk, ii = slice(jl, jh), slice(kl, kh), slice(il, ih)
        p = jnp.sum(ey[ih, jj, kk] * hz[ih, jj, kk] - ez[ih, jj, kk] * hy[ih, jj, kk])
        p -= jnp.sum(ey[il, jj, kk] * hz[il, jj, kk] - ez[il, jj, kk] * hy[il, jj, kk])
        p += jnp.sum(ez[ii, jh, kk] * hx[ii, jh, kk] - ex[ii, jh, kk] * hz[ii, jh, kk])
        p -= jnp.sum(ez[ii, jl, kk] * hx[ii, jl, kk] - ex[ii, jl, kk] * hz[ii, jl, kk])
        p += jnp.sum(ex[ii, jj, kh] * hy[ii, jj, kh] - ey[ii, jj, kh] * hx[ii, jj, kh])
        p -= jnp.sum(ex[ii, jj, kl] * hy[ii, jj, kl] - ey[ii, jj, kl] * hx[ii, jj, kl])
        return float(p)

    peak_U = 0.0           # running peak, updated at checks only
    energy_below = 0       # consecutive sub-threshold energy checks
    peak_flux = 0.0        # #388 flux-stop: running peak of the |P| envelope (from first check)
    flux_below = 0         # #388 flux-stop: consecutive sub-threshold checks
    flux_hist: list = []   # recent |P| samples for the max-envelope
    decayed_fired = False  # #388: did the energy criterion fire (vs cap-hit)?
    decay_checks: list = []
    ys_chunks = []
    steps_done = 0
    reporter = None
    if report_every:
        from rfx.progress import ProgressReporter
        # decay_by == 0.0 is the documented forced-N escape: max_steps is then
        # the exact run length, not a cap, and the line should not say "(cap)".
        reporter = ProgressReporter(max_steps, label=report_label,
                                    total_is_cap=(decay_by > 0.0))

    while steps_done < max_steps:
        this_chunk = min(int(check_interval), max_steps - steps_done)
        xs = (
            jnp.arange(steps_done, steps_done + this_chunk, dtype=jnp.int32),
            src_waveforms[steps_done:steps_done + this_chunk],
        )
        carry, ys = _run_chunk(carry, xs)
        ys_chunks.append(ys)
        steps_done += this_chunk
        if reporter is not None and (
                steps_done - reporter.last_reported >= int(report_every)
                or steps_done >= max_steps):
            jax.block_until_ready(carry["fdtd"])   # honest wall-clock rate
            reporter.report(steps_done)

        # Stop check at the chunk boundary (check-step-only — the whole-domain
        # reduction / surface integral is the expensive part).
        if use_flux_stop:
            # RADIATED-FLUX stop: track the flux peak from the FIRST check (radiation peaks
            # during the source drive), gate only the STOP on min_steps.
            flux_hist.append(abs(_radiated_power(carry["fdtd"])))
            env = max(flux_hist[-flux_env_checks:])
            if env > peak_flux:
                peak_flux = env
            if steps_done >= min_steps and peak_flux > 0.0 and env < decay_by * peak_flux:
                flux_below += 1
                if flux_below >= decay_energy_consecutive:
                    decayed_fired = True
                    break
            else:
                flux_below = 0
        elif steps_done >= min_steps:
            U = _interior_energy(carry["fdtd"])
            if U > peak_U:
                peak_U = U
            decay_checks.append((steps_done, U, peak_U))
            # decay_by=0.0 forced-N escape: U >= 0 so U < 0 never fires.
            if U < decay_by * peak_U:
                energy_below += 1
                if energy_below >= decay_energy_consecutive:
                    decayed_fired = True
                    break
            else:
                energy_below = 0

    # #388: measured static-remnant advisory on cap-hit (until_decay is absorbing-only on
    # the NU lane too, so a cap-hit without firing means the energy criterion could not
    # self-terminate — warn if the remnant is electrostatic). Function-local import avoids
    # a module-level simulation<->nonuniform cycle.
    # decay_by == 0.0 is the forced-N escape (fixed-step progress route):
    # running to max_steps is the DESIGN there, not a failed stop — the
    # static-remnant advisory would be a false alarm.
    if not decayed_fired and not use_flux_stop and decay_by > 0.0:
        from rfx.simulation import _warn_static_remnant_cap_hit
        _warn_static_remnant_cap_hit(carry["fdtd"], materials, grid)

    # Per-chunk ys concatenate to exactly steps_done rows; the
    # emit_time_series=False convention ((steps, 0)) is preserved
    # because each chunk emits (this_chunk, 0).
    time_series = jnp.concatenate(ys_chunks, axis=0)

    result = _assemble_nu_result(setup, carry, time_series)
    result["decay_checks"] = decay_checks
    return result

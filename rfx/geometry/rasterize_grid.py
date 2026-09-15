"""Unified geometry rasterization for all grid types.

Extracts the material-assembly loop from api.py / nonuniform.py /
subgridded.py into a single function that accepts any grid type
via a coordinate-provider abstraction.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer
from rfx.core.yee import MaterialArrays
from rfx.geometry._pole_keying import (
    _accumulate_pole_mask,
    _spec_from_pole_masks,
)


class GridCoords(NamedTuple):
    """Physical sample coordinates for rasterization.

    **Two conventions live behind this type, deliberately.**
    ``coords_from_uniform_grid`` and ``coords_from_nonuniform_grid`` return
    E-NODE positions (cell edges) — the samples the Yee stencil differences and
    that PEC/geometry act on. ``coords_from_fine_grid`` genuinely returns cell
    CENTRES for the subgrid fine region (subgrid-fenced, unchanged).

    Reading the wrong one is not cosmetic. This type's docstring said
    "cell-center" for every producer until #562, and a consumer
    (``compute_smoothed_eps_nonuniform``) named its variables ``centers_*``
    accordingly and derived the node as ``centre - d/2``; when the NU producer
    became node-based, that derivation silently inverted and displaced every
    smoothed voxel by half a cell. Consumers that need both should derive the
    centre FROM the node (``centre = node + d/2``), and any new producer must
    say which convention it returns.

    **Dtype contract (#802/#807)**: concrete producers return HOST float64
    numpy arrays — exact node positions, independent of ``jax_enable_x64``.
    Do not ``jnp.asarray`` them before a comparison; under x64=0 that
    silently downcasts to float32, which flips inclusion at node-aligned
    faces (the #802 defect). Only the traced NU producer (mesh as design
    variable) returns a float32 jnp array.
    """
    x: jnp.ndarray  # (nx,)
    y: jnp.ndarray  # (ny,)
    z: jnp.ndarray  # (nz,)
    shape: tuple[int, int, int]


class ExactNodeSpineMissingWarning(UserWarning):
    """A concrete NonUniformGrid axis has no float64 cell-size spine.

    ``coords_from_nonuniform_grid`` then widens the solver's float32 store
    to build node positions — the pre-#802 values, ~1e-10 m off the exact
    node line, which flips half-open inclusion at node-aligned faces (a
    node-aligned Box on a graded 0.3048 mm profile realized 2000 cells from
    the widened store against 1800 from the exact spine, measured
    2026-09-02). ``make_nonuniform_grid`` always populates the spine on
    concrete profiles, so this fires only for grids assembled by hand or
    whose spine fields were replaced by non-float64 arrays.
    """


def _uniform_axis_nodes(n: int, pad: int, dx: float) -> np.ndarray:
    """Exact E-node positions for a uniform axis: host float64, flag-independent.

    ``(i - pad) * dx`` per node in float64 — bit-identical to the x64=1
    realization verified against the Box half-open convention (#802). This is
    THE uniform node formula: ``csg._grid_coords``, ``coords_from_uniform_grid``
    and the uniform-valued branch of ``_axis_node_positions`` all call it, so
    the uniform and non-uniform lanes cannot rasterize a uniform-valued axis
    differently (#807 class). Before this builder existed the repo carried
    three hand-copies that each rounded differently in float32.
    """
    return (np.arange(n, dtype=np.float64) - pad) * float(dx)


def coords_from_uniform_grid(grid) -> GridCoords:
    """Extract E-NODE coordinates from a uniform Grid.

    ``(arange - pad) * dx`` — node i at i*dx from the first interior node,
    despite the historical "cell-center" wording this docstring carried.
    Returns HOST float64 numpy arrays (see ``_uniform_axis_nodes``); do not
    ``jnp.asarray`` them before a comparison — under x64=0 that silently
    downcasts to float32, which is the #802 defect.
    """
    nx, ny, nz = grid.shape
    dx = grid.dx
    pad_x, pad_y, pad_z = grid.axis_pads
    return GridCoords(x=_uniform_axis_nodes(nx, pad_x, dx),
                      y=_uniform_axis_nodes(ny, pad_y, dx),
                      z=_uniform_axis_nodes(nz, pad_z, dx),
                      shape=(nx, ny, nz))


def _axis_node_positions(d_arr: np.ndarray, cpml: int) -> np.ndarray:
    """E-node positions for a padded cell-size array.

    The nodes this grid steps sit on cell EDGES, not centres: the
    non-uniform E update divides by ``2/(d[i-1]+d[i])``, the dual spacing
    of a node straddling cells ``i-1`` and ``i``. Node ``cpml`` (the first
    interior one) is the origin, so the interior spans
    ``[0, sum(interior d)]`` and the last interior node lands exactly on
    the requested domain face — the same convention
    ``coords_from_uniform_grid`` uses (``(arange - pad) * dx``).

    Until #562 this returned cell CENTRES, half a cell off the nodes the
    stencil differences and half a cell off the uniform builder, which put
    every rasterized material half a cell away from the fields acting on
    it and (with the missing bounding node) made a PEC-bounded guide one
    cell narrower than requested.
    """
    d = np.asarray(d_arr, dtype=np.float64)
    if d.size and bool(np.all(d == d[0])):
        # Uniform-valued axis: closed form, bitwise-equal to the uniform
        # builder by construction — the lane-equality guarantee (#807).
        return _uniform_axis_nodes(d.size, cpml, float(d[0]))
    edges = np.insert(np.cumsum(d), 0, 0.0)           # len = n+1
    return edges[:-1] - edges[cpml]                   # n nodes, origin at cpml


def coords_from_nonuniform_grid(grid) -> GridCoords:
    """Extract E-NODE coordinates from a NonUniformGrid (#562).

    All three axes use the per-cell spacing arrays (``dx_arr``,
    ``dy_arr``, ``dz``). The first interior cell on each axis is
    placed at physical position 0, matching the convention that a
    ``Box((0,0,0), (Lx,Ly,Lz))`` should tile the interior domain
    exactly.
    """
    # Per-axis pad — respects PEC/PMC faces which have pad=0 even when
    # ``grid.cpml_layers`` is nonzero. Using the scalar ``cpml_layers``
    # here hit IndexError on axes that are PEC on both sides and shorter
    # than ``cpml_layers + 1`` cells (e.g. WR-90's narrow b-axis at
    # dx=1 mm: 11 cells, cpml_layers=20 → edges[20] out of bounds).
    pad_x_lo = int(getattr(grid, "pad_x_lo", grid.cpml_layers))
    pad_y_lo = int(getattr(grid, "pad_y_lo", grid.cpml_layers))
    pad_z_lo = int(getattr(grid, "pad_z_lo", grid.cpml_layers))
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    spine_missing = []

    def _axis_nodes(d_arr, pad_lo, d_exact, axis_name):
        # Mesh-as-design-variable path: any axis cell-size profile may be
        # a JAX tracer. Route the cumsum / offset arithmetic through jnp
        # in-trace; fall back to the numpy path on concrete inputs to keep
        # the host-float behaviour the rest of the codebase depends on.
        if is_tracer(d_arr):
            d_j = jnp.asarray(d_arr)
            cum = jnp.concatenate([jnp.zeros((1,), dtype=d_j.dtype),
                                   jnp.cumsum(d_j)])
            nodes = cum[:-1] - cum[pad_lo]
            return nodes.astype(jnp.float32)
        # Concrete path: HOST float64, no f32 cast (#802/#807 — the cast
        # here re-quantized every node position and made the NU lane land
        # one plane away from the uniform lane at node-aligned faces).
        # Read the grid's exact float64 profile: ``dx_arr``/``dy_arr``/``dz``
        # are float32 stores, and a cumsum of f32-widened cell sizes drifts
        # off the exact node positions by the same 1e-10 m class this
        # function exists to eliminate. A grid without the spine (built by
        # hand, or a spine field replaced by a non-float64 array) gets the
        # widened store AND a warning: silently falling back reproduced the
        # #802 realization with no trace of why.
        if d_exact is None or np.asarray(d_exact).dtype != np.float64:
            spine_missing.append(axis_name)
            d_np = np.asarray(d_arr)
        else:
            d_np = np.asarray(d_exact)
        return _axis_node_positions(d_np, pad_lo)

    x = _axis_nodes(grid.dx_arr, pad_x_lo, getattr(grid, "dx_arr_f64", None),
                    "x")
    y = _axis_nodes(grid.dy_arr, pad_y_lo, getattr(grid, "dy_arr_f64", None),
                    "y")
    z = _axis_nodes(grid.dz, pad_z_lo, getattr(grid, "dz_f64", None), "z")
    if spine_missing:
        warnings.warn(
            "NonUniformGrid carries no exact float64 cell-size spine on axis "
            f"{', '.join(spine_missing)} (dx_arr_f64 / dy_arr_f64 / dz_f64 "
            "is None or not float64): node positions were widened from the "
            "float32 store and can sit ~1e-10 m off the exact node line, "
            "which flips half-open inclusion at node-aligned faces (#802 "
            "class). Build the grid with make_nonuniform_grid, or set the "
            "spine fields from the float64 profile.",
            ExactNodeSpineMissingWarning, stacklevel=2)

    return GridCoords(x=x, y=y, z=z, shape=(nx, ny, nz))


def coords_from_fine_grid(nx_f, ny_f, nz_f, dx_f, x_off, y_off, z_off) -> GridCoords:
    """Extract cell-center coordinates for a subgridded fine region.

    Uses cell centers (offset by dx_f/2), not cell edges. Host float64 like
    every other concrete producer (#802 policy) — the fine region is
    cv12/13-fenced experimental, but its coordinates follow the same
    exactness rule so a face landing on a sample point is not decided by
    float32 rounding.
    """
    x = x_off + (np.arange(nx_f, dtype=np.float64) + 0.5) * float(dx_f)
    y = y_off + (np.arange(ny_f, dtype=np.float64) + 0.5) * float(dx_f)
    z = z_off + (np.arange(nz_f, dtype=np.float64) + 0.5) * float(dx_f)
    return GridCoords(x=x, y=y, z=z, shape=(nx_f, ny_f, nz_f))


# ---------------------------------------------------------------------------
# Lattice ownership contract (#931): cell centres, PEC classification, sheets
#
# Normative text: docs/design_notes/20260906_plan_realign_lattice_ownership.md
# §1.1 (centre sampling for PEC volumes), §1.3 (sheet footprint / plane),
# §1.4 (wires), §1.5 (what ``sim.add(Box, material=pec)`` refuses).
# Dielectric sampling is untouched: it stays the node, half-open sampler the
# shapes implement in ``csg.py``.
# ---------------------------------------------------------------------------

_REL_TOL = 1e-9   # relative to the local cell: "on the lattice" tolerance


def _uniform_axis_centres(n: int, pad: int, dx: float) -> np.ndarray:
    """Exact primal-cell CENTRES for a uniform axis: ``(i - pad + 1/2) * dx``.

    One float64 rounding, the same route a user takes when spelling a
    corner on a cell midpoint (``(m + 0.5) * dx``, the cv18 "midpoint
    recipe"), so a corner drawn there lands on the tie the contract
    documents (lo inclusive, hi exclusive).
    """
    return (np.arange(n, dtype=np.float64) - pad + 0.5) * float(dx)


def axis_cell_sizes(nodes) -> object:
    """Per-cell sizes from a node line: ``d[i] = x_{i+1} - x_i``; the last
    cell repeats the previous size (the last node opens a cell past the
    array).  Host float64 on concrete input, ``jnp`` on a tracer."""
    if is_tracer(nodes):
        d = jnp.diff(jnp.asarray(nodes))
        return jnp.concatenate([d, d[-1:]]) if d.size else jnp.ones((1,), dtype=jnp.asarray(nodes).dtype)
    x = np.asarray(nodes, dtype=np.float64)
    if x.size < 2:
        return np.ones((x.size,), dtype=np.float64)
    d = np.diff(x)
    return np.concatenate([d, d[-1:]])


def cell_sizes_from_uniform_grid(grid):
    dx = float(grid.dx)
    return tuple(np.full((n,), dx, dtype=np.float64) for n in grid.shape)


def cell_sizes_from_nonuniform_grid(grid):
    """Per-cell sizes of a NonUniformGrid: the float64 spine when present
    (``dx_arr_f64`` / ``dy_arr_f64`` / ``dz_f64``), else the store."""
    out = []
    for store, exact in ((grid.dx_arr, getattr(grid, "dx_arr_f64", None)),
                         (grid.dy_arr, getattr(grid, "dy_arr_f64", None)),
                         (grid.dz, getattr(grid, "dz_f64", None))):
        if is_tracer(store):
            out.append(jnp.asarray(store))
        elif exact is not None and np.asarray(exact).dtype == np.float64:
            out.append(np.asarray(exact, dtype=np.float64))
        else:
            out.append(np.asarray(store, dtype=np.float64))
    return tuple(out)


def cell_centres_from_nodes(coords: GridCoords, cell_sizes=None) -> GridCoords:
    """Primal-cell centres ``node + d/2`` per axis (§1.1).

    ``cell_sizes`` is the per-cell size triple (from the grid's spine);
    when None it is derived from the node line.  A uniform-valued axis
    takes the closed form :func:`_uniform_axis_centres` so the uniform and
    non-uniform lanes cannot centre-sample a uniform axis differently
    (#807 class).  Traced input keeps traced arithmetic.
    """
    axes = []
    for nodes, d in zip((coords.x, coords.y, coords.z),
                        cell_sizes if cell_sizes is not None else (None, None, None)):
        if d is None:
            d = axis_cell_sizes(nodes)
        if is_tracer(nodes) or is_tracer(d):
            nj = jnp.asarray(nodes)
            axes.append(nj + 0.5 * jnp.asarray(d, dtype=nj.dtype))
            continue
        x = np.asarray(nodes, dtype=np.float64)
        dd = np.asarray(d, dtype=np.float64)
        closed = None
        if x.size and dd.size and bool(np.all(dd == dd[0])):
            # uniform-valued axis: IF the nodes are (i - pad) * dx exactly
            # (every in-repo producer goes through ``_uniform_axis_nodes``),
            # ``pad`` is recoverable from the first node and the closed form
            # keeps the uniform and non-uniform lanes bit-identical (#807).
            # A caller-supplied node line with a fractional origin is NOT
            # that axis, and the closed form would silently move its
            # centres by the fractional offset — take the exact fallback.
            dx = float(dd[0])
            pad = int(round(-x[0] / dx)) if dx > 0 else 0
            if np.array_equal(x, _uniform_axis_nodes(x.size, pad, dx)):
                closed = _uniform_axis_centres(x.size, pad, dx)
        axes.append(closed if closed is not None else x + 0.5 * dd)
    return GridCoords(x=axes[0], y=axes[1], z=axes[2], shape=coords.shape)


def centres_from_uniform_grid(grid) -> GridCoords:
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    dx = grid.dx
    return GridCoords(x=_uniform_axis_centres(nx, pad_x, dx),
                      y=_uniform_axis_centres(ny, pad_y, dx),
                      z=_uniform_axis_centres(nz, pad_z, dx),
                      shape=(nx, ny, nz))


def centres_from_nonuniform_grid(grid, coords: GridCoords | None = None) -> GridCoords:
    if coords is None:
        coords = coords_from_nonuniform_grid(grid)
    return cell_centres_from_nodes(coords, cell_sizes_from_nonuniform_grid(grid))


def _local_cell(nodes, d, pos: float) -> float:
    """Size of the cell containing (or nearest to) physical position ``pos``."""
    x = np.asarray(nodes, dtype=np.float64)
    dd = np.asarray(d, dtype=np.float64)
    if x.size == 0:
        return 1.0
    k = int(np.clip(np.searchsorted(x, pos, side="right") - 1, 0, dd.size - 1))
    return float(dd[k])


def _nearest_plane(nodes, pos: float, d_local: float, *, what: str = "sheet",
                   name=None, axis: int = 0) -> int:
    """Node plane nearest ``pos``; an exact half-cell tie resolves LOWER
    (§1.3, today's ``n_vol == 1`` rule for a face-registered 1-cell Box).

    "Nearest" means within half a local cell.  A declaration further out
    than that is not on this node line at all — ``argmin`` would clamp it
    onto the end plane and the caller would get a conductor it never drew
    (the #369 silently-relocated-metal class).  That raises.
    """
    x = np.asarray(nodes, dtype=np.float64)
    dist = np.abs(x - pos)
    k = int(np.argmin(dist))
    if dist[k] > 0.5 * d_local * (1.0 + _REL_TOL):
        raise ValueError(
            f"{what} {name!r}: the declared plane {'xyz'[axis]} = {pos:.6g} m "
            f"lies {dist[k]:.6g} m from the nearest node line "
            f"({x[k]:.6g} m), more than half the local cell "
            f"({d_local:.6g} m) — it is outside this grid's node range "
            f"[{x[0]:.6g}, {x[-1]:.6g}] m, so it would be silently clamped "
            "onto an end plane. Move the declaration inside the domain or "
            "enlarge the domain.")
    if k - 1 >= 0 and abs(dist[k - 1] - dist[k]) <= _REL_TOL * d_local:
        k = k - 1                  # exact half-cell tie resolves LOWER
    return k


def _box_axis_volume(centres, lo: float, hi: float):
    """§1.1 half-open volume rule on CELL CENTRES: cell ``i`` occupied iff
    ``lo <= c_i < hi``."""
    if is_tracer(centres):
        c = jnp.asarray(centres)
        return (c >= lo) & (c < hi)
    c = np.asarray(centres, dtype=np.float64)
    return (c >= lo) & (c < hi)


def _box_axis_closed(nodes, lo: float, hi: float, d_local: float):
    """§1.3 CLOSED footprint sampling on NODES: ``lo <= x_i <= hi`` with an
    on-lattice tolerance of ``1e-9`` cell so a corner spelled through a
    different f64 route (``a + b`` vs ``m*dx``) keeps its row."""
    tol = _REL_TOL * d_local
    if is_tracer(nodes):
        x = jnp.asarray(nodes)
        return (x >= lo - tol) & (x <= hi + tol)
    x = np.asarray(nodes, dtype=np.float64)
    return (x >= lo - tol) & (x <= hi + tol)


def _is_traced_coords(coords) -> bool:
    return any(is_tracer(c) for c in (coords.x, coords.y, coords.z))


def pec_volume_cell_mask(shape, centres: GridCoords):
    """CELL occupancy of a PEC VOLUME (§1.1): centre-sampled.

    Box: half-open ``lo <= c < hi`` per axis, spelled here directly — the
    Box's own node sampler carries a thin-sheet branch that must never
    decide a PEC volume.  Sphere / Cylinder / any other shape: the centre
    lies inside the shape (``mask_on_coords`` on the centre coordinates).
    """
    lo = getattr(shape, "corner_lo", None)
    hi = getattr(shape, "corner_hi", None)
    if lo is not None and hi is not None:
        mx = _box_axis_volume(centres.x, float(lo[0]), float(hi[0]))
        my = _box_axis_volume(centres.y, float(lo[1]), float(hi[1]))
        mz = _box_axis_volume(centres.z, float(lo[2]), float(hi[2]))
        return jnp.asarray(mx[:, None, None] & my[None, :, None] & mz[None, None, :])
    return jnp.asarray(shape.mask_on_coords(centres.x, centres.y, centres.z))


def _volume_is_empty(shape, centres: GridCoords, mask) -> bool | None:
    """Zero-cell test that never converts a jnp mask to bool (§1.5 refusal).

    Inside an outer ``jax.jit`` the grid coordinates are still concrete host
    arrays, but every jnp array built from them is a tracer, so a Python
    bool of ``jnp.any(mask)`` raises ``TracerBoolConversionError`` — the
    #642-class defect ``rfx/api/_compile.py`` documents and the first cut
    of this refusal re-introduced one frame down (found by the #931 phase-2
    T1 test ``test_real_interior_pec_under_outer_jit_matches_eager``).

    Decision order: a Box is decided on the HOST from its corners and the
    centre lines (exact, cheap); any other shape is decided from its mask
    only when that mask is concrete; a traced mask is undecidable here and
    returns ``None`` (the eager path already refused it, if it is empty).
    """
    lo = getattr(shape, "corner_lo", None)
    hi = getattr(shape, "corner_hi", None)
    axes = (centres.x, centres.y, centres.z)
    if lo is not None and hi is not None and not any(is_tracer(c) for c in axes):
        for i in range(3):
            c = np.asarray(axes[i], dtype=np.float64)
            if not np.any((c >= float(lo[i])) & (c < float(hi[i]))):
                return True
        return False
    if is_tracer(mask):
        return None
    return not bool(np.any(np.asarray(mask, dtype=bool)))


def _refuse_zero_cells(shape, name, what: str):
    raise ValueError(
        f"{what} {name!r} ({type(shape).__name__}) rasterizes to ZERO cells "
        "on this grid: no primal-cell centre lies inside it, so it would "
        "silently vanish (the #369 vaporized-metal class, now an error). A "
        "filament (a via or post thinner than a cell) is a PolylineWire, "
        "not a volume; a volume needs a radius of at least ~0.87 of the "
        "local cell (half the cell diagonal) to be sure of one centre — "
        "resolve the mesh or redraw the body.")


def sheet_spec_from_shape(shape, coords: GridCoords, cell_sizes=None, *,
                          normal_axis=None, name=None, lane: str = "",
                          refuse_thick: bool = False):
    """A :class:`SheetSpec` for a sheet declaration (§1.3).

    * plane = the node plane nearest the shape's mid-plane along
      ``normal_axis`` (default: the thinnest bounding-box axis); an exact
      half-cell tie resolves to the LOWER plane;
    * Box footprint: sampled CLOSED ``[lo, hi]`` on the two in-plane axes;
    * any other shape: its cross-section at its own mid-plane,
      ``shape.mask_on_coords(x, y, [mid])``, placed on ``plane``;
    * ``refuse_thick`` (``add_thin_conductor``): a shape thicker than one
      local cell along its normal is not a sheet.

    The plane is a static int, so a traced mesh (node coordinates are
    tracers) cannot declare a PEC sheet — the caller decides what to do.
    Returns the spec; raises ``ValueError`` on a zero footprint.
    """
    from rfx.boundaries.pec import SheetSpec
    from rfx.materials.thin_conductor import sheet_bounds

    if _is_traced_coords(coords):
        raise ValueError(
            f"sheet {name!r}: a sheet's plane is a static integer, and this "
            "mesh is a JAX tracer (mesh-as-design-variable), so the nearest "
            "node plane cannot be resolved. Pin the mesh, or declare the "
            "conductor as a volume.")
    lo, hi = sheet_bounds(shape)
    if lo is None or hi is None:
        raise ValueError(
            f"sheet {name!r}: shape {type(shape).__name__} has no axis-aligned "
            "bounding box (Box corner_lo/corner_hi or Shape.bounding_box()), "
            "so its normal and plane cannot be located.")
    lo = tuple(float(v) for v in lo)
    hi = tuple(float(v) for v in hi)
    extents = [hi[i] - lo[i] for i in range(3)]
    if normal_axis is None:
        normal_axis = min(range(3), key=lambda i: extents[i])
    a = int(normal_axis)
    node_axes = (coords.x, coords.y, coords.z)
    if cell_sizes is None:
        cell_sizes = tuple(axis_cell_sizes(n) for n in node_axes)
    mid = 0.5 * (lo[a] + hi[a])
    d_local = _local_cell(node_axes[a], cell_sizes[a], mid)
    if refuse_thick and extents[a] > d_local * (1.0 + _REL_TOL):
        raise ValueError(
            f"add_thin_conductor: shape {name!r} is {extents[a]:.6g} m thick "
            f"along {'xyz'[a]} against a local cell of {d_local:.6g} m — not a "
            "sheet; use add() for a volume (the lattice ownership contract "
            "realizes a volume with both faces and a shorted interior).")
    plane = _nearest_plane(node_axes[a], mid, d_local,
                           what="sheet", name=name, axis=a)
    shape_3 = tuple(coords.shape)
    is_box = getattr(shape, "corner_lo", None) is not None
    if is_box:
        axes = []
        for t in range(3):
            if t == a:
                m = np.zeros((shape_3[t],), dtype=bool)
                m[plane] = True
                axes.append(m)
            else:
                d_t = _local_cell(node_axes[t], cell_sizes[t],
                                  0.5 * (lo[t] + hi[t]))
                axes.append(np.asarray(_box_axis_closed(node_axes[t], lo[t], hi[t], d_t)))
        fp = axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]
    else:
        sample = [np.asarray(n, dtype=np.float64) for n in node_axes]
        sample[a] = np.asarray([mid], dtype=np.float64)
        cross = np.asarray(shape.mask_on_coords(*sample), dtype=bool)
        fp = np.zeros(shape_3, dtype=bool)
        idx = [slice(None)] * 3
        idx[a] = slice(plane, plane + 1)
        fp[tuple(idx)] = cross
    if not fp.any():
        raise ValueError(
            f"sheet {name!r}: the footprint rasterizes to ZERO nodes on this "
            f"grid{(' (' + lane + ' lane)') if lane else ''} at plane "
            f"{'xyz'[a]}={plane}; it would silently vanish (#369 class). "
            "Widen the footprint to reach a node line or refine the mesh.")
    return SheetSpec(normal_axis=a, plane=plane, footprint=jnp.asarray(fp),
                     name=name)



def refuse_vaporized_sheets(sheets, *, lane: str = "",
                            periodic=(False, False, False)):
    """Refuse a sheet PLANE that realizes no PEC edge (#931 §1.5, #369 class).

    §1.3 unions every footprint on one ``(normal_axis, plane)`` BEFORE the
    edge rule, so the unit that must carry current is the plane, not the
    declaration: a patterned ground drawn as twenty half-cell boxes has
    single-node rows that realize nothing alone and one connected plane
    together. What cannot stand is a plane whose whole union realizes ZERO
    edges — metal that reaches node lines but never two adjacent ones, and
    so carries no current, the same silent vanishing a sub-cell Box is
    refused for on the volume side.

    Asked of the single owner (:func:`realized_pec_edge_masks`) rather than
    re-derived, so it cannot disagree with the solve — INCLUDING the run's
    ``periodic`` flags: a footprint whose only two occupied nodes sit either
    side of a periodic seam realizes one edge THROUGH the seam, and asking
    with the default non-periodic padding refused a conductor the solve
    realizes (measured on an x-periodic (7,7,7) grid with nodes (0,3,3) and
    (6,3,3): 1 Ex edge with ``(True, False, False)``, 0 with the default).
    """
    sheets = list(sheets or ())
    if not sheets:
        return
    from rfx.boundaries.pec import realized_pec_edge_masks as _rpem
    by_plane: dict = {}
    for sp in sheets:
        by_plane.setdefault((int(sp.normal_axis), int(sp.plane)), []).append(sp)
    for (axis, plane), group in by_plane.items():
        if any(is_tracer(sp.footprint) for sp in group):
            continue
        if any(bool(np.asarray(m).any())
               for m in _rpem(None, sheets=tuple(group), periodic=periodic)):
            continue
        names = ", ".join(repr(getattr(sp, "name", None)) for sp in group)
        raise ValueError(
            f"PEC sheet plane {'xyz'[axis]}={plane} realizes ZERO PEC edges on "
            f"this grid{(' (' + lane + ' lane)') if lane else ''}: its whole "
            f"footprint ({len(group)} declaration(s): {names}) reaches node "
            "line(s) but never two ADJACENT ones, so the metal carries no "
            "current and would silently vanish (#369 class; the volume side "
            "refuses the same drawing, #931 §1.5). Widen the footprint to span "
            "at least one cell in an in-plane direction, or refine the mesh.")


def sheet_footprint_traced(shape, coords: GridCoords, normal_axis: int):
    """Sheet footprint on a TRACED mesh (mesh-as-design-variable), no plane.

    The plane of a sheet is a static integer under the contract, which a
    traced node line cannot provide, so a PEC sheet is refused there
    (:func:`sheet_spec_from_shape`). The f0 sheet only needs the footprint
    mask; this is the traced twin of the concrete rule — a Box footprint is
    CLOSED on the in-plane axes and one-hot at the nearest node along the
    normal (``argmin`` first occurrence = the lower plane on an exact tie),
    any other shape is its own ``mask_on_coords`` — so eager and traced
    builds agree on the footprint.
    """
    from rfx.materials.thin_conductor import sheet_bounds
    lo, hi = sheet_bounds(shape)
    if getattr(shape, "corner_lo", None) is None or lo is None:
        return shape.mask_on_coords(coords.x, coords.y, coords.z)
    a = int(normal_axis)
    axes = []
    for t, nodes in enumerate((coords.x, coords.y, coords.z)):
        c = jnp.asarray(nodes)
        if t == a:
            mid = 0.5 * (float(lo[a]) + float(hi[a]))
            axes.append(jnp.zeros(c.shape, dtype=bool).at[
                jnp.argmin(jnp.abs(c - mid))].set(True))
        else:
            axes.append((c >= float(lo[t])) & (c <= float(hi[t])))
    return axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]


def _box_zero_axes(lo, hi):
    return [i for i in range(3) if float(hi[i]) - float(lo[i]) == 0.0]


def _subcell_axes(lo, hi, node_axes, cell_sizes):
    """Axes where the drawn extent is ``0 < extent < one local cell`` (§1.5).

    One spelling for every shape: the test is on the DRAWN extent (the
    shape's axis-aligned bounding box), never on what the raster happened
    to produce — "nothing is inferred from raster thickness or drawing
    direction".  A zero extent is not sub-cell: on a Box it is the sheet
    declaration, and on any other shape it falls through to the zero-cell
    refusal.
    """
    out = []
    for i in range(3):
        ext = float(hi[i]) - float(lo[i])
        if ext <= 0.0:
            continue
        mid = 0.5 * (float(lo[i]) + float(hi[i]))
        d_local = _local_cell(node_axes[i], cell_sizes[i], mid)
        if ext < d_local * (1.0 - _REL_TOL):
            out.append((i, ext, d_local))
    return out


def _refuse_subcell(subcell, shape, name):
    where = "; ".join(
        f"{'xyz'[i]} ({ext:.6g} m against a local cell of {d:.6g} m)"
        for i, ext, d in subcell)
    kind = type(shape).__name__
    raise ValueError(
        f"PEC {kind} {name!r} is thinner than one cell along {where}: "
        f"a {kind} passed to sim.add() is a VOLUME; declare a sheet (a "
        "zero-thickness Box or add_thin_conductor) or resolve the "
        "thickness. Nothing is inferred from raster thickness or drawing "
        "direction (lattice ownership contract §1.5).")


def classify_pec_entry(shape, coords: GridCoords, centres: GridCoords,
                       cell_sizes=None, *, name=None):
    """Classify one PEC geometry entry (``sim.add(shape, material=pec)``).

    Returns ``(cell_mask, sheet, wire)`` with exactly one of the three set:
    a centre-sampled VOLUME cell mask, a :class:`SheetSpec`, or a
    :class:`WireSpec` (§1.5 / §1.4).  Refusals (all ``ValueError``):

    * Box with two or three zero-extent axes (a line / a point);
    * Box with ``0 < extent < one local cell`` along any axis — a Box is a
      volume; declare a sheet (a zero-thickness Box or add_thin_conductor)
      or resolve the thickness;
    * any shape whose centre-sampled volume is empty (concrete only).
    """
    from rfx.boundaries.pec import WireSpec, wire_path_edge_masks

    traced = _is_traced_coords(coords) or _is_traced_coords(centres)
    node_axes = (coords.x, coords.y, coords.z)
    if cell_sizes is None:
        cell_sizes = tuple(axis_cell_sizes(n) for n in node_axes)
    lo = getattr(shape, "corner_lo", None)
    hi = getattr(shape, "corner_hi", None)
    if lo is not None and hi is not None:
        zero = _box_zero_axes(lo, hi)
        if len(zero) >= 2:
            raise ValueError(
                f"PEC Box {name!r} has zero extent along "
                f"{'/'.join('xyz'[i] for i in zero)}: a line or a point is not "
                "a conductor. A filament is a PolylineWire; a sheet has exactly "
                "one zero-extent axis.")
        if len(zero) == 1:
            return None, sheet_spec_from_shape(
                shape, coords, cell_sizes, normal_axis=zero[0], name=name), None
        if not traced:
            subcell = _subcell_axes(lo, hi, node_axes, cell_sizes)
            if subcell:
                _refuse_subcell(subcell, shape, name)
        mask = pec_volume_cell_mask(shape, centres)
        if not traced and _volume_is_empty(shape, centres, mask):
            _refuse_zero_cells(shape, name, "PEC volume")
        return mask, None, None
    pts = getattr(shape, "points", None)
    radius = getattr(shape, "radius", None)
    if pts is not None and radius is not None and not traced:
        # PolylineWire (§1.4): radius >= half the local cell is a volume;
        # below that it is a filament on the axis-aligned lattice path
        # joining the nearest nodes of consecutive vertices.
        nodes = []
        d_min = None
        for p in pts:
            idx = []
            for t in range(3):
                x = np.asarray(node_axes[t], dtype=np.float64)
                k = int(np.argmin(np.abs(x - float(p[t]))))
                idx.append(k)
                d_here = float(np.asarray(cell_sizes[t], dtype=np.float64)[k])
                d_min = d_here if d_min is None else min(d_min, d_here)
            nodes.append(tuple(idx))
        if float(radius) < 0.5 * d_min:
            edges = wire_path_edge_masks(nodes, coords.shape)
            return None, None, WireSpec(edges=edges, name=name)
    elif not traced:
        # §1.5 for every OTHER shape with an axis-aligned bounding box —
        # a Cylinder via pad, a thin Sphere, an imported outline.  The
        # refusal was Box-only, so a 0.3-cell Cylinder pad was realized as
        # a one-cell slab with two faces (or refused as zero cells) purely
        # according to where it fell between two centres: the #369/#702
        # raster-dependent thickness the contract exists to make an error.
        # PolylineWire is excluded on purpose: §1.4 gives it its own
        # filament/volume rule on the radius, decided above.
        from rfx.materials.thin_conductor import sheet_bounds
        bb_lo, bb_hi = sheet_bounds(shape)
        if bb_lo is not None and bb_hi is not None:
            subcell = _subcell_axes(bb_lo, bb_hi, node_axes, cell_sizes)
            if subcell:
                _refuse_subcell(subcell, shape, name)
    mask = pec_volume_cell_mask(shape, centres)
    if not traced and _volume_is_empty(shape, centres, mask):
        _refuse_zero_cells(shape, name, "PEC volume")
    return mask, None, None


def rasterize_geometry(
    geometry_entries,
    material_resolver,
    coords: GridCoords,
    *,
    pec_sigma_threshold: float = 1e6,
    thin_conductors=None,
    thin_conductor_applier=None,
    grid=None,
    centres: GridCoords | None = None,
    cell_sizes=None,
    sheets: list | None = None,
    wires: list | None = None,
    periodic=(False, False, False),
):
    """Rasterize geometry entries onto material arrays.

    This is the single shared implementation used by all runner paths
    (uniform, non-uniform, subgridded).

    Parameters
    ----------
    geometry_entries : list of _GeometryEntry
        Each has .shape (Shape) and .material_name (str).
    material_resolver : callable(name) -> MaterialSpec
        Resolves material name to MaterialSpec.
    coords : GridCoords
        Sample coordinates from any grid type — E-NODES for the uniform and
        non-uniform builders, cell centres for the subgrid fine region (see
        ``GridCoords``).  Dielectrics are sampled here (node, half-open —
        untouched by #931).
    pec_sigma_threshold : float
        Conductivity above which a material is treated as PEC.
    thin_conductors : list or None
        ThinConductor entries to apply after geometry.
    thin_conductor_applier : callable or None
        Function(grid, tc, materials, pec_mask, sheets) -> (materials, pec_mask).
    grid : Grid or NonUniformGrid or None
        Original grid object, needed by thin_conductor_applier.
    centres : GridCoords or None
        Primal-cell CENTRES for PEC volume sampling (§1.1).  ``None``
        derives them from ``coords`` as ``node + d/2``; the subgrid fine
        lane, whose ``coords`` already are centres, passes them explicitly.
    cell_sizes : (dx, dy, dz) per-cell arrays or None
        The grid's per-cell sizes (spine); ``None`` derives them from the
        node line.
    sheets, wires : list or None
        Out-parameters: PEC sheets (:class:`SheetSpec`) and PEC filaments
        (:class:`WireSpec`) classified from the PEC entries are appended
        here.  They own no cell and are NOT in ``pec_mask``; a caller that
        passes no collector and has such an entry gets a ``ValueError``
        rather than a silently vanished conductor.

    Returns
    -------
    materials : MaterialArrays
    debye_spec : (poles, masks) or None
    lorentz_spec : (poles, masks) or None
    pec_mask : bool array or None
    pec_shapes : list of Shape
    kerr_chi3 : float array or None
    """
    from rfx.materials.debye import DebyePole
    from rfx.materials.lorentz import LorentzPole

    shape = coords.shape
    eps_r = jnp.ones(shape, dtype=jnp.float32)
    sigma = jnp.zeros(shape, dtype=jnp.float32)
    mu_r = jnp.ones(shape, dtype=jnp.float32)
    chi3_arr = jnp.zeros(shape, dtype=jnp.float32)
    pec_mask = jnp.zeros(shape, dtype=jnp.bool_)
    pec_shapes = []
    has_pec_cells = False
    has_kerr = False
    if centres is None:
        centres = cell_centres_from_nodes(coords, cell_sizes)

    # Keyed per _pole_key (#274): pole value when hashable, id(pole) for
    # traced poles. Values are (pole, mask) pairs.
    debye_masks_by_pole: dict[DebyePole | int, tuple[DebyePole, jnp.ndarray]] = {}
    lorentz_masks_by_pole: dict[LorentzPole | int, tuple[LorentzPole, jnp.ndarray]] = {}

    for entry in geometry_entries:
        mat = material_resolver(entry.material_name)
        mask = entry.shape.mask_on_coords(coords.x, coords.y, coords.z)

        if mat.sigma >= pec_sigma_threshold:
            cells, sheet, wire = classify_pec_entry(
                entry.shape, coords, centres, cell_sizes,
                name=entry.material_name)
            if cells is not None:
                has_pec_cells = True
                pec_mask = pec_mask | cells
                mask = cells
            elif sheet is not None:
                if sheets is None:
                    raise ValueError(
                        "PEC sheet declared by a zero-thickness Box, but this "
                        "lane collects no sheets (rasterize_geometry(sheets=None)); "
                        "refusing to drop it silently.")
                sheets.append(sheet)
            else:
                if wires is None:
                    raise ValueError(
                        "PEC PolylineWire filament declared, but this lane "
                        "collects no wires (rasterize_geometry(wires=None)); "
                        "refusing to drop it silently.")
                wires.append(wire)
            pec_shapes.append(entry.shape)
        else:
            eps_r = jnp.where(mask, mat.eps_r, eps_r)
            sigma = jnp.where(mask, mat.sigma, sigma)
            mu_r = jnp.where(mask, mat.mu_r, mu_r)

        if mat.chi3 != 0.0:
            chi3_arr = jnp.where(mask, mat.chi3, chi3_arr)
            has_kerr = True

        if mat.debye_poles:
            for pole in mat.debye_poles:
                _accumulate_pole_mask(debye_masks_by_pole, pole, mask)

        if mat.lorentz_poles:
            for pole in mat.lorentz_poles:
                _accumulate_pole_mask(lorentz_masks_by_pole, pole, mask)

    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    # Thin conductors (P4)
    if thin_conductors and thin_conductor_applier and grid is not None:
        for tc in thin_conductors:
            materials, pec_mask = thin_conductor_applier(
                grid, tc, materials, pec_mask=pec_mask, sheets=sheets)
            if tc.is_pec:
                pec_shapes.append(tc.shape)
                # A legacy applier can materialize a 2-D sheet as cells.
                has_pec_cells = True

    debye_spec = _spec_from_pole_masks(debye_masks_by_pole)
    lorentz_spec = _spec_from_pole_masks(lorentz_masks_by_pole)

    # Match the uniform assembler: only the optional-mask decision is
    # static under jit; classification/refusals above still use the grid.
    has_pec = has_pec_cells if is_tracer(pec_mask) else bool(jnp.any(pec_mask))
    kerr_chi3 = chi3_arr if has_kerr else None
    refuse_vaporized_sheets(sheets, lane="non-uniform", periodic=periodic)
    return materials, debye_spec, lorentz_spec, pec_mask if has_pec else None, pec_shapes, kerr_chi3


def periodic_flags_from_axes(periodic_axes) -> tuple[bool, bool, bool]:
    """``"xy"`` -> ``(True, True, False)``; ``None``/``""`` -> all False."""
    s = periodic_axes or ""
    return tuple(ax in s for ax in "xyz")


def extend_cpml_pad_materials(
    eps_r: jnp.ndarray,
    sigma: jnp.ndarray,
    mu_r: jnp.ndarray,
    plx: int, phx: int,
    ply: int, phy: int,
    plz: int, phz: int,
    dispersion_pole_mask: jnp.ndarray | None = None,
):
    """Extend eps_r/sigma/mu_r into the CPML padding so guided modes see an
    impedance-matched absorber, equivalent to UPML. Each CPML face copies
    the interior-edge slice outward, as if the geometry continued beyond
    the domain.

    Single shared implementation for the uniform (``rfx/api/_compile.py``)
    and non-uniform (``rfx/runners/nonuniform.py``) assemblers — issue #627
    found the two hand-duplicated copies (#582 mirrored one onto the other)
    both carrying the same gap, so the fix lives once, here, and both call
    sites use it instead of keeping duplicated pad-extension code that can
    drift.

    **Hi-face fallback (#627a).** ``rfx.geometry.csg.Box``'s volume-branch
    rasterization is deliberately half-open, ``[lo, hi)``, over node
    coordinates (see that class's docstring — the convention is load-
    bearing across the package, e.g. every WR-90 aperture/iris
    measurement). Its documented consequence is that the ``hi`` face of a
    box "contributes no node": a structure whose hi face lands on (or
    inside) the domain's last interior node loses exactly that one node
    from its own rasterized mask. The naive interior-edge source for a hi
    pad — literally the outermost interior column — therefore reads
    vacuum for such a structure even though its real material sits one
    column further in, and copying that vacuum outward gives the pad a
    Fresnel step instead of a match (measured on the #582 fixture: x-lo
    pad eps_r 4.0, x-hi pad eps_r 1.0, for a slab spanning the full x
    extent).

    The fix does NOT touch the rasterizer (out of scope — it would move
    geometry everywhere in the package, and the convention is correct and
    intentional for the shape mask itself). Instead, per transverse cell:
    if the naive interior-edge column is vacuum (``eps_r==1 & sigma==0 &
    mu_r==1``) but the column immediately inside it is not, replicate from
    that inner column instead. This is bounded to exactly one column
    inward — the rasterizer's hi-face shortfall for a single box is
    deterministically one node (Box docstring: "the shortfall is entirely
    at the hi face"), never more — so a genuine multi-cell vacuum buffer
    between a structure and the CPML pad (the overwhelmingly common case:
    almost every example leaves an air gap before the absorber) is left
    completely alone and still replicates plain vacuum, exactly as before.
    An unbounded backward scan for "the last non-vacuum column" was
    considered and rejected: it would bridge that common air gap and smear
    an unrelated interior structure's material into the pad.

    **The dropped node itself is repaired too (#655).** #627a fixed where
    the pad SOURCES its material but still wrote it only to the pad, so the
    dropped node stayed vacuum and became a one-cell film sandwiched
    between the structure and its own matched absorber:
    ``[material][vacuum][pad = material]``. Measured on a 1-D plane-wave
    fixture (eps_r=4 filling the domain, reflection isolated by field-level
    DFT subtraction against the same fixture with the box drawn half a cell
    past the face, so only that node differs): ``|r|`` = 0.238 at rfx's own
    default mesh ``dx = c0/freq_max/20`` (20 cells/lambda0), 0.191 at 30,
    0.083 at 60, 0.032 at 120 — matching thin-film theory
    ``2*pi*(dx/lambda0)*(eps_m-1)/(2*sqrt(eps_m))`` to within 22 % over the
    range and 0.8 % at the default. The error GROWS as the mesh coarsens,
    the opposite of the direction a convergence check looks in.

    Where the fallback fires, ``src`` is therefore written to the outermost
    interior column as well as to the pad. That is what makes the hi face
    behave like the lo face: ``_extend_lo`` replicates the boundary node
    itself, so its pad and its boundary node cannot disagree by
    construction. It inherits the one-column bound above unchanged, so a
    genuine air gap is still not bridged, on either side of the boundary.
    Where the fallback does NOT fire the write is value-for-value what is
    already there — ``run()`` is byte-identical for geometry that does not
    touch a face (verified over PEC / CPML / lossy+CPML / mu_r+CPML /
    non-uniform+CPML / periodic-mixed, SHA-256 over raw field bytes).

    This is deliberately NOT a rasterizer change: the half-open convention
    is load-bearing (see the ``Box`` docstring), and the defect is not even
    Box-specific — ``Sphere`` and ``Cylinder`` have closed axis predicates
    and reach the same state through the float32 knife edge instead, which
    an array-pattern repair covers and a rasterizer edit would not.

    **Dispersion-pole extension was tried and reverted (#627b).** An
    earlier revision of this function extended Debye/Lorentz pole masks
    into the pad the same way, via an ``extra_masks`` parameter, so a
    dispersive edge-touching material would be impedance-matched across
    the band, not just at DC. Review found this turns a stable simulation
    into a divergent one for an edge-touching structure carrying a
    high-Q Lorentz pole (Q~60): the same 20,000-step fixture that decays
    on both the shipped (statics-only) code and on the pre-#582 tree
    (last/mid energy ratio 0.12–0.16) grows without bound once the pole
    mask is also extended (last/mid ratio 649, no NaN and no exception —
    values stay finite and simply grow, so nothing downstream flags it).
    Extending the pole alone, on top of an otherwise-unpatched static
    extension, reproduces the divergence; the static extension alone,
    with the same high-Q pole left un-extended in the interior, decays
    cleanly. Do not re-add pole-mask extension here without a stability
    argument for the resonant-pole-in-a-CPML-pad regime — see the
    follow-up issue (filed separately from #627, tracking this factorial)
    and the physics-level regression lock in
    ``tests/unit/boundaries/test_cpml_pad_material_extension.py``, which reds if pole
    extension is naively reintroduced.

    **Pole-carrying columns get NO hi-face fallback promotion (#808).**
    With the pole half reverted (#627b), the hi-face fallback promoted a
    dispersive structure's STATICS anyway: the pad — and, after #655, the
    dropped boundary node itself — carried the material's eps_inf without
    its poles, a material that exists in no declared model. Issue #808
    measured the consequence on a committed observable: a face-touching
    Debye slab's differentiable recovery moved from its pinned
    delta_eps 3.330 (11% err) to 3.969 (32%, past the 20% gate), and a
    controlled pad-rule swap (geometry, observation and optimizer held
    fixed) toggled the result between the two states digit-for-digit.
    So when ``dispersion_pole_mask`` marks the fallback's SOURCE (inner)
    column as pole-carrying, the promotion and the #655 boundary write
    are suppressed for that transverse cell: the hi pad takes the naive
    outer-column copy (background) and the dropped node stays as
    rasterized — the pre-#638 hi-face state every committed dispersive
    gate was pinned against. Lo faces are deliberately NOT gated: the
    lo pad replicates a boundary column the material genuinely occupies,
    that behaviour predates #638, and the #808 discriminator's identity
    arm measured that removing it too moves the same recovery's tau to
    64% error — "less pad material" is not automatically better; the
    committed envelopes pin the lo-statics state. Callers that pass no
    mask keep the pre-#808 behaviour bit-for-bit.

    Parameters
    ----------
    dispersion_pole_mask : bool array or None
        OR of every Debye/Lorentz per-pole mask on the same padded grid
        as the material arrays, or None when the simulation declares no
        dispersive material (or a caller deliberately wants the ungated
        rule, e.g. the #636 factorial harness). The mask is replicated
        through the same lo/hi passes as the statics (as ``poleish``
        below), so a LO-pad copy of a pole-carrying column also refuses a
        later axis's hi-face promotion — without that, the y/z corner
        pads would promote the lo-pad chimera copy and realize a state
        the pinned pipeline never had.

    Returns
    -------
    (eps_r, sigma, mu_r)
    """
    arrays = [eps_r, sigma, mu_r]
    poleish = dispersion_pole_mask

    def _vacuum(e, s, m):
        return (e == 1.0) & (s == 0.0) & (m == 1.0)

    def _extend_lo(arrays, poleish, pad_lo, lo_src, lo_dst):
        if pad_lo <= 0:
            return arrays, poleish
        arrays = [a.at[lo_dst].set(a[lo_src]) for a in arrays]
        if poleish is not None:
            poleish = poleish.at[lo_dst].set(poleish[lo_src])
        return arrays, poleish

    def _extend_hi(arrays, poleish, n, pad_lo, pad_hi,
                   outer_sl, inner_sl, dst_sl):
        if pad_hi <= 0:
            return arrays, poleish
        e, s, m = arrays[0], arrays[1], arrays[2]
        outer_vac = _vacuum(e[outer_sl], s[outer_sl], m[outer_sl])
        use_inner = None
        if n - pad_lo - pad_hi >= 2:
            inner_vac = _vacuum(e[inner_sl], s[inner_sl], m[inner_sl])
            use_inner = outer_vac & (~inner_vac)
            if poleish is not None:
                # #808: never promote a pole-carrying column's statics —
                # the promoted material would carry eps_inf without its
                # poles (a material no declared model has), and #627b's
                # revert forbids extending the pole itself. Gating
                # ``use_inner`` kills both the pad promotion and the #655
                # boundary write below (``src`` stays the outer column,
                # so the write is value-for-value). ``poleish`` rather
                # than the raw mask, so a lo-pad replica of a pole column
                # is refused the same way (see the parameter docs).
                use_inner = use_inner & (~poleish[inner_sl])
        new_arrays = []
        for a in arrays:
            src = a[outer_sl]
            if use_inner is not None:
                src = jnp.where(use_inner, a[inner_sl], src)
                # #655: the pad is not the only place that lost the node.
                # Where the fallback fired, the LAST INTERIOR column is the
                # dropped node itself, so writing ``src`` only to the pad
                # leaves it vacuum and sandwiches a one-cell gap between the
                # structure and its own matched pad. Write ``src`` to the
                # boundary column as well, which is precisely what makes the
                # hi face behave like the lo face: ``_extend_lo`` replicates
                # the boundary node itself, so its pad and its boundary node
                # can never disagree. Where the fallback did NOT fire,
                # ``src is a[outer_sl]`` and this is a value-for-value
                # rewrite of what is already there — byte-identical.
                a = a.at[outer_sl].set(src)
            new_arrays.append(a.at[dst_sl].set(src))
        if poleish is not None:
            # Replicate the pole marking through the identical select, so
            # later axes see which pad cells are copies of pole columns.
            # Where the fallback fired the source column is non-pole by
            # construction (the gate above). Hi pads are NOT always False:
            # a shape drawn PAST the hi face rasterizes the boundary node
            # (and any overdrawn pad nodes) into its own mask, so the
            # naive outer-column copy replicates True outward there —
            # alongside the statics that copy also carries. Only for a
            # shape ending AT the face is the dropped outer node outside
            # every rasterized mask (hi pad False; lo pads carry True).
            psrc = poleish[outer_sl]
            if use_inner is not None:
                psrc = jnp.where(use_inner, poleish[inner_sl], psrc)
                poleish = poleish.at[outer_sl].set(psrc)
            poleish = poleish.at[dst_sl].set(psrc)
        return new_arrays, poleish

    # ---- x ----
    arrays, poleish = _extend_lo(
        arrays, poleish, plx, np.s_[plx:plx + 1, :, :], np.s_[:plx, :, :])
    nx = arrays[0].shape[0]
    arrays, poleish = _extend_hi(
        arrays, poleish, nx, plx, phx,
        np.s_[nx - phx - 1:nx - phx, :, :],
        np.s_[nx - phx - 2:nx - phx - 1, :, :],
        np.s_[nx - phx:nx, :, :],
    )

    # ---- y ----
    arrays, poleish = _extend_lo(
        arrays, poleish, ply, np.s_[:, ply:ply + 1, :], np.s_[:, :ply, :])
    ny = arrays[0].shape[1]
    arrays, poleish = _extend_hi(
        arrays, poleish, ny, ply, phy,
        np.s_[:, ny - phy - 1:ny - phy, :],
        np.s_[:, ny - phy - 2:ny - phy - 1, :],
        np.s_[:, ny - phy:ny, :],
    )

    # ---- z ----
    arrays, poleish = _extend_lo(
        arrays, poleish, plz, np.s_[:, :, plz:plz + 1], np.s_[:, :, :plz])
    nz = arrays[0].shape[2]
    arrays, poleish = _extend_hi(
        arrays, poleish, nz, plz, phz,
        np.s_[:, :, nz - phz - 1:nz - phz],
        np.s_[:, :, nz - phz - 2:nz - phz - 1],
        np.s_[:, :, nz - phz:nz],
    )

    eps_r, sigma, mu_r = arrays[0], arrays[1], arrays[2]
    return eps_r, sigma, mu_r

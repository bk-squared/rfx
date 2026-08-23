"""Unified geometry rasterization for all grid types.

Extracts the material-assembly loop from api.py / nonuniform.py /
subgridded.py into a single function that accepts any grid type
via a coordinate-provider abstraction.
"""

from __future__ import annotations

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
    """
    x: jnp.ndarray  # (nx,)
    y: jnp.ndarray  # (ny,)
    z: jnp.ndarray  # (nz,)
    shape: tuple[int, int, int]


def coords_from_uniform_grid(grid) -> GridCoords:
    """Extract E-NODE coordinates from a uniform Grid.

    ``(arange - pad) * dx`` — node i at i*dx from the first interior node,
    despite the historical "cell-center" wording this docstring carried.
    """
    nx, ny, nz = grid.shape
    dx = grid.dx
    pad_x, pad_y, pad_z = grid.axis_pads
    x = jnp.asarray((np.arange(nx) - pad_x) * dx, dtype=jnp.float32)
    y = jnp.asarray((np.arange(ny) - pad_y) * dx, dtype=jnp.float32)
    z = jnp.asarray((np.arange(nz) - pad_z) * dx, dtype=jnp.float32)
    return GridCoords(x=x, y=y, z=z, shape=(nx, ny, nz))


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

    def _axis_nodes(d_arr, pad_lo):
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
        d_np = np.asarray(d_arr)
        return jnp.asarray(_axis_node_positions(d_np, pad_lo), dtype=jnp.float32)

    x = _axis_nodes(grid.dx_arr, pad_x_lo)
    y = _axis_nodes(grid.dy_arr, pad_y_lo)
    z = _axis_nodes(grid.dz, pad_z_lo)

    return GridCoords(x=x, y=y, z=z, shape=(nx, ny, nz))


def coords_from_fine_grid(nx_f, ny_f, nz_f, dx_f, x_off, y_off, z_off) -> GridCoords:
    """Extract cell-center coordinates for a subgridded fine region.

    Uses cell centers (offset by dx_f/2), not cell edges.
    """
    x = jnp.asarray(x_off + (np.arange(nx_f) + 0.5) * dx_f, dtype=jnp.float32)
    y = jnp.asarray(y_off + (np.arange(ny_f) + 0.5) * dx_f, dtype=jnp.float32)
    z = jnp.asarray(z_off + (np.arange(nz_f) + 0.5) * dx_f, dtype=jnp.float32)
    return GridCoords(x=x, y=y, z=z, shape=(nx_f, ny_f, nz_f))


def rasterize_geometry(
    geometry_entries,
    material_resolver,
    coords: GridCoords,
    *,
    pec_sigma_threshold: float = 1e6,
    thin_conductors=None,
    thin_conductor_applier=None,
    grid=None,
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
        ``GridCoords``).
    pec_sigma_threshold : float
        Conductivity above which a material is treated as PEC.
    thin_conductors : list or None
        ThinConductor entries to apply after geometry.
    thin_conductor_applier : callable or None
        Function(grid, tc, materials, pec_mask) -> (materials, pec_mask).
    grid : Grid or NonUniformGrid or None
        Original grid object, needed by thin_conductor_applier.

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
    has_kerr = False

    # Keyed per _pole_key (#274): pole value when hashable, id(pole) for
    # traced poles. Values are (pole, mask) pairs.
    debye_masks_by_pole: dict[DebyePole | int, tuple[DebyePole, jnp.ndarray]] = {}
    lorentz_masks_by_pole: dict[LorentzPole | int, tuple[LorentzPole, jnp.ndarray]] = {}

    for entry in geometry_entries:
        mat = material_resolver(entry.material_name)
        mask = entry.shape.mask_on_coords(coords.x, coords.y, coords.z)

        if mat.sigma >= pec_sigma_threshold:
            pec_mask = pec_mask | mask
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
                grid, tc, materials, pec_mask=pec_mask)
            if tc.is_pec:
                pec_shapes.append(tc.shape)

    debye_spec = _spec_from_pole_masks(debye_masks_by_pole)
    lorentz_spec = _spec_from_pole_masks(lorentz_masks_by_pole)

    has_pec = bool(jnp.any(pec_mask))
    kerr_chi3 = chi3_arr if has_kerr else None
    return materials, debye_spec, lorentz_spec, pec_mask if has_pec else None, pec_shapes, kerr_chi3


def collect_thin_conductor_sheet_inputs(thin_conductors, mask_fn):
    """Split thin conductors into the two inputs the sheet resample needs.

    One rule, both lanes: a PEC thin sheet joins the PEC cell union (its
    normal is then read from the union's own adjacency, exactly as
    ``apply_pec_mask`` reads it), while a surface-impedance (``f0``) sheet
    is NOT in ``pec_mask`` at all and carries its own declared normal axis
    (``sheet_normal_axis``, the same reader
    ``apply_thin_conductor``/``assemble_materials_nu`` use to pick which
    dual spacing normalizes it).

    A legacy DC-fold conductor is neither: it writes a VOLUMETRIC
    ``sigma``/``eps_r`` fold into its cell (``thin_conductor.py``'s
    ``eps_r = where(mask, conductor.eps_r, ...)``), so its cell is not a
    node-thin surface and the resample must not touch it.

    Parameters
    ----------
    thin_conductors : iterable or None
    mask_fn : callable(shape) -> bool array
        Lane's rasterizer for a shape (``shape.mask(grid)`` uniform,
        ``shape.mask_on_coords(...)`` non-uniform).

    Returns
    -------
    (pec_masks, declared_sheets)
        ``pec_masks`` is a list of boolean arrays to OR into the PEC cell
        union; ``declared_sheets`` is a list of ``(mask, normal_axis)``.
    """
    pec_masks = []
    declared_sheets = []
    if not thin_conductors:
        return pec_masks, declared_sheets
    from rfx.materials.thin_conductor import sheet_bounds, sheet_normal_axis
    for tc in thin_conductors:
        if getattr(tc, "is_pec", False):
            pec_masks.append(mask_fn(tc.shape))
            continue
        if getattr(tc, "surface_impedance_f0", None) is None:
            continue
        lo, hi = sheet_bounds(tc.shape)
        if lo is None or hi is None:
            # The f0 lanes raise on this; do not pre-empt their message.
            continue
        declared_sheets.append((mask_fn(tc.shape), sheet_normal_axis(lo, hi)))
    return pec_masks, declared_sheets


def periodic_flags_from_axes(periodic_axes) -> tuple[bool, bool, bool]:
    """``"xy"`` -> ``(True, True, False)``; ``None``/``""`` -> all False."""
    s = periodic_axes or ""
    return tuple(ax in s for ax in "xyz")


def sheet_normal_live_axis_masks(
    conductor_cell_mask,
    *,
    declared_sheets=(),
    periodic=(False, False, False),
):
    """Cells where a node-thin conductor leaves exactly ONE E component live.

    A conductor thinner than a cell is realized node-thin: it occupies one
    cell layer, ``rfx.boundaries.pec.apply_pec_mask`` zeroes the two
    in-plane (tangential) E edges of that cell and deliberately LEAVES the
    sheet-normal edge alone, because that edge carries the surface charge.
    This function returns, per axis, the cells for which that surviving
    component is the axis' own — i.e. the cells whose stored ``eps_r`` /
    ``sigma`` still feed a live field update.

    The classification is the SAME rule the operator uses
    (:func:`rfx.boundaries.pec.tangential_edge_masks`, on the SAME union
    cell mask, with the SAME ``periodic`` flags), so "this cell's eps is
    still live along axis n" cannot disagree with "apply_pec_mask left
    component n alone at this cell". Classifying per shape instead would
    disagree: a patterned plane drawn as abutting 1-cell boxes reads thin
    in the in-plane axes box-by-box and solid as a union (the #690
    measurement, in the other direction).

    Restricted to cells thin along EXACTLY ONE axis. A body thin along two
    (a sub-cell wire) or three (an isolated cell) axes keeps two or three
    live components, which sit at two or three different half-cell offsets;
    one isotropic scalar per cell cannot serve them, so those cells are
    left alone rather than served wrongly.

    Parameters
    ----------
    conductor_cell_mask : (nx, ny, nz) bool array or None
        Union of every PEC-like conductor cell (geometry PEC entries plus
        PEC thin conductors). ``None`` contributes nothing.
    declared_sheets : iterable of (mask, normal_axis)
        Surface-impedance (``surface_impedance_f0``) sheets, which are NOT
        in the PEC mask and carry their own declared normal axis (#690).
    periodic : (bool, bool, bool)
        The run's per-axis periodic flags, forwarded unchanged.

    Returns
    -------
    (mask_x, mask_y, mask_z) boolean arrays.
    """
    from rfx.boundaries.pec import tangential_edge_masks

    masks = None
    if conductor_cell_mask is not None:
        tang = tangential_edge_masks(conductor_cell_mask, periodic)
        masks = [conductor_cell_mask & ~t for t in tang]

    for m, ax in declared_sheets:
        if masks is None:
            masks = [jnp.zeros(m.shape, dtype=jnp.bool_) for _ in range(3)]
        masks[int(ax)] = masks[int(ax)] | m

    if masks is None:
        return None

    # ONE restriction, applied once to the assembled per-axis claims: keep a
    # cell only where exactly one axis claims it. That covers both ways a cell
    # can be ambiguous -- a body thin along two or three axes (a sub-cell wire,
    # an isolated cell), whose two or three live components sit at two or three
    # different half-cell offsets that one isotropic scalar cannot serve; and a
    # cell claimed by sheets of two different normals. It is deliberately NOT
    # split into a per-body guard plus an overlap guard: each covered the
    # other's cases, so neither could be falsified alone (measured -- mutating
    # either one left all 18 tests of
    # tests/test_sheet_node_permittivity.py green).
    claimed = sum(m.astype(jnp.int32) for m in masks)
    unique = claimed == 1
    return tuple(m & unique for m in masks)


def _subcell_box_axis_window(entry_shape, axis, node_coords, half_steps_axis):
    """``(lo, hi)`` if this shape is a BOX thinner than its local cell, else None.

    Why the resample needs this. ``Box.mask_on_coords`` has a thin branch: a
    shape thinner than the local cell claims the ONE node nearest its midpoint
    (``csg.py``), so it survives instead of vaporizing. That is right for the
    main rasterization, and wrong for a half-cell-shifted re-sample, where the
    branch re-runs against the SHIFTED nodes and re-snaps the shape onto
    whichever of them now happens to be nearest.

    Measured on a real board: two identical 17 um buried-level dielectric
    fills, both registered at their mid-plane, both with the shifted sample
    point ~7 um ABOVE the fill's top face. One re-snapped onto the shifted
    node and one did not — the two candidate shifted nodes are equidistant
    from the fill midpoint by construction, so float32 rounding decided
    (eps_r 3.520 at one level, 3.380 at the other, from the same geometry).
    A material value must not be decided that way.

    So for a sub-cell Box the resample asks the only question that has an
    answer at a point: is the shifted point inside ``[lo, hi)``? Restricted to
    Box because for a Box the bounding box IS the shape; for a Sphere or a
    Cylinder it is not, and the plain shifted mask stays exact there.
    """
    lo = getattr(entry_shape, "corner_lo", None)
    hi = getattr(entry_shape, "corner_hi", None)
    if lo is None or hi is None:
        return None
    if is_tracer(half_steps_axis) or is_tracer(node_coords):
        return None
    lo_ax, hi_ax = float(lo[axis]), float(hi[axis])
    extent = hi_ax - lo_ax
    nodes = np.asarray(node_coords, dtype=np.float64)
    half = np.asarray(half_steps_axis, dtype=np.float64)
    if nodes.size == 0:
        return None
    idx = int(np.argmin(np.abs(nodes - 0.5 * (lo_ax + hi_ax))))
    flat = half.reshape(-1)
    d_local = 2.0 * float(flat[min(idx, flat.size - 1)] if flat.size > 1
                          else flat[0])
    if extent > d_local * 1.01:
        return None
    return lo_ax, hi_ax


def _statics_on_coords(geometry_entries, material_resolver, coords_shifted,
                       coords_node, axis, half_steps_axis, shape,
                       pec_sigma_threshold):
    """``eps_r`` / ``sigma`` from the geometry entries at the shifted point.

    Same entry order and same PEC branch as :func:`rasterize_geometry` (a PEC
    entry writes neither), so this is that function's statics read at a
    different sample point. Sub-cell Boxes take the half-open window test
    described in :func:`_subcell_box_axis_window` instead of the thin branch's
    argmin.
    """
    eps_r = jnp.ones(shape, dtype=jnp.float32)
    sigma = jnp.zeros(shape, dtype=jnp.float32)
    sx, sy, sz = coords_shifted
    for entry in geometry_entries:
        mat = material_resolver(entry.material_name)
        if mat.sigma >= pec_sigma_threshold:
            continue
        mask = entry.shape.mask_on_coords(sx, sy, sz)
        window = _subcell_box_axis_window(
            entry.shape, axis, coords_node[axis], half_steps_axis)
        if window is not None:
            lo_ax, hi_ax = window
            ax_c = coords_shifted[axis]
            inside = (ax_c >= lo_ax) & (ax_c < hi_ax)
            bshape = [1, 1, 1]
            bshape[axis] = inside.shape[0]
            unshifted = entry.shape.mask_on_coords(*coords_node)
            mask = (mask | unshifted) & inside.reshape(bshape)
        eps_r = jnp.where(mask, mat.eps_r, eps_r)
        sigma = jnp.where(mask, mat.sigma, sigma)
    return eps_r, sigma


def resample_sheet_node_materials(
    geometry_entries,
    material_resolver,
    coords: GridCoords,
    eps_r,
    sigma,
    *,
    half_steps,
    conductor_cell_mask=None,
    declared_sheets=(),
    periodic=(False, False, False),
    pec_sigma_threshold: float = 1e6,
):
    """Sample a node-thin conductor cell's statics where its LIVE edge sits.

    **The defect.** ``eps_r[i,j,k]`` is a point sample at the E NODE
    ``z[k]`` (``_axis_node_positions``: ``edges[:-1]``). A conductor thinner
    than a cell has no volume: it is registered on one node, contributes a
    PEC/sheet cell there, and NOTHING writes ``eps_r`` at that node — a PEC
    entry deliberately writes only ``pec_mask`` (``rasterize_geometry``,
    ``_compile._build_materials``). Where the surrounding dielectric boxes
    abut the conductor's faces instead of spanning its thickness — which is
    what a real stackup or a CAD export gives, the metal layer being a slot
    no dielectric fills — that node keeps the default vacuum.

    That vacuum is not harmless, because the one E component the sheet
    leaves alive is the sheet-NORMAL one, and rfx's own staggering
    (``rfx/geometry/smoothing.py``: "Ez lives at (i, j, k+0.5)") puts it at
    ``z[k] + dz[k]/2`` — half a cell away from the sample point, inside the
    dielectric above, not inside the metal. So the cavity a stacked pair
    bounds carries one vacuum cell in series.

    Measured, mid-plane-registered 17 um copper between eps_r 3.52 below and
    3.38 above on a 31.43 um graded mesh: ``eps_r`` at the sheet node 1.000,
    and a 14-cell series sum ``sum(d/eps_r)`` of 149.72 um against the
    physical stack's 127.59 um — the gap reads 17.3 % wider and the coupling
    capacitance 14.8 % low. The whole error is that one cell:
    ``31.43*(1 - 1/3.38) = 22.13 um == 149.72 - 127.59``. Reproduces
    identically on the uniform lane, so it is not a graded-mesh artefact.

    **The fix.** For exactly the cells of
    :func:`sheet_normal_live_axis_masks`, re-sample ``eps_r`` and ``sigma``
    from the same geometry, on the same mesh, at ``coord + d/2`` along that
    cell's live axis. No geometry moves and no mesh changes; the sample
    point moves onto the field point it feeds.

    **It does not invent dielectric.** An OUTER conductor with air above
    resamples to air, because that is what is at its live edge. Only a cell
    whose live edge genuinely sits in a dielectric gets one.

    **A sub-cell DIELECTRIC needs its own rule**, see
    :func:`_subcell_box_axis_window`: ``Box``'s thin branch would re-snap it
    onto whichever shifted node is now nearest, which for a mid-plane
    registered fill is a float32 tie-break, so a sub-cell Box takes a
    half-open window test along the resampled axis instead.

    **Deliberately not resampled.**

    * ``mu_r`` — it feeds the H update, which is staggered differently and
      is masked by ``apply_pec_h_mask``, not by the tangential-edge rule.
    * Debye/Lorentz pole masks and ``chi3`` — a sheet node whose live edge
      lands in a dispersive dielectric takes that material's ``eps_r``
      (its ``eps_inf``) but not its poles, so it behaves as the lossless
      high-frequency limit of the right material instead of as vacuum.
      Moving a resonant pole mask onto a new cell is the change #627b
      measured turning a stable run divergent, so it is a separate decision
      with its own stability argument, not a side effect of this one.
      Pinned by ``tests/test_sheet_node_permittivity.py``.

    Parameters
    ----------
    geometry_entries, material_resolver, coords, pec_sigma_threshold
        As :func:`rasterize_geometry`.
    eps_r, sigma : arrays
        The node-sampled statics to correct.
    half_steps : sequence of 3
        Half the PRIMAL cell size per axis — scalar ``dx/2`` on the uniform
        lane, ``(dx_arr/2, dy_arr/2, dz/2)`` on the non-uniform one. The
        offset is always ``+``: node ``k`` is the LOWER edge of primal cell
        ``k`` and the normal E edge sits at ``+d[k]/2`` from it.
    conductor_cell_mask, declared_sheets, periodic
        As :func:`sheet_normal_live_axis_masks`.

    Returns
    -------
    (eps_r, sigma)
    """
    axis_masks = sheet_normal_live_axis_masks(
        conductor_cell_mask,
        declared_sheets=declared_sheets,
        periodic=periodic,
    )
    if axis_masks is None:
        return eps_r, sigma

    base = [coords.x, coords.y, coords.z]
    for axis in range(3):
        m = axis_masks[axis]
        # Eager lanes skip an axis with no sheet outright (one extra
        # rasterization pass per axis otherwise). Under jit the predicate is
        # a tracer, so all three axes are taken and the result is identical.
        if not is_tracer(m) and not bool(jnp.any(m)):
            continue
        shifted = list(base)
        shifted[axis] = base[axis] + jnp.asarray(half_steps[axis],
                                                 dtype=base[axis].dtype)
        eps_s, sigma_s = _statics_on_coords(
            geometry_entries, material_resolver,
            tuple(shifted), tuple(base), axis, half_steps[axis],
            coords.shape, pec_sigma_threshold,
        )
        eps_r = jnp.where(m, eps_s, eps_r)
        sigma = jnp.where(m, sigma_s, sigma)
    return eps_r, sigma


def extend_cpml_pad_materials(
    eps_r: jnp.ndarray,
    sigma: jnp.ndarray,
    mu_r: jnp.ndarray,
    plx: int, phx: int,
    ply: int, phy: int,
    plz: int, phz: int,
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
    ``tests/test_cpml_pad_material_extension.py``, which reds if pole
    extension is naively reintroduced.

    Returns
    -------
    (eps_r, sigma, mu_r)
    """
    arrays = [eps_r, sigma, mu_r]

    def _vacuum(e, s, m):
        return (e == 1.0) & (s == 0.0) & (m == 1.0)

    def _extend_lo(arrays, pad_lo, lo_src, lo_dst):
        if pad_lo <= 0:
            return arrays
        return [a.at[lo_dst].set(a[lo_src]) for a in arrays]

    def _extend_hi(arrays, n, pad_lo, pad_hi, outer_sl, inner_sl, dst_sl):
        if pad_hi <= 0:
            return arrays
        e, s, m = arrays[0], arrays[1], arrays[2]
        outer_vac = _vacuum(e[outer_sl], s[outer_sl], m[outer_sl])
        use_inner = None
        if n - pad_lo - pad_hi >= 2:
            inner_vac = _vacuum(e[inner_sl], s[inner_sl], m[inner_sl])
            use_inner = outer_vac & (~inner_vac)
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
        return new_arrays

    # ---- x ----
    arrays = _extend_lo(arrays, plx, np.s_[plx:plx + 1, :, :], np.s_[:plx, :, :])
    nx = arrays[0].shape[0]
    arrays = _extend_hi(
        arrays, nx, plx, phx,
        np.s_[nx - phx - 1:nx - phx, :, :],
        np.s_[nx - phx - 2:nx - phx - 1, :, :],
        np.s_[nx - phx:nx, :, :],
    )

    # ---- y ----
    arrays = _extend_lo(arrays, ply, np.s_[:, ply:ply + 1, :], np.s_[:, :ply, :])
    ny = arrays[0].shape[1]
    arrays = _extend_hi(
        arrays, ny, ply, phy,
        np.s_[:, ny - phy - 1:ny - phy, :],
        np.s_[:, ny - phy - 2:ny - phy - 1, :],
        np.s_[:, ny - phy:ny, :],
    )

    # ---- z ----
    arrays = _extend_lo(arrays, plz, np.s_[:, :, plz:plz + 1], np.s_[:, :, :plz])
    nz = arrays[0].shape[2]
    arrays = _extend_hi(
        arrays, nz, plz, phz,
        np.s_[:, :, nz - phz - 1:nz - phz],
        np.s_[:, :, nz - phz - 2:nz - phz - 1],
        np.s_[:, :, nz - phz:nz],
    )

    eps_r, sigma, mu_r = arrays[0], arrays[1], arrays[2]
    return eps_r, sigma, mu_r

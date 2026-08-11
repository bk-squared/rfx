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


def extend_cpml_pad_materials(
    eps_r: jnp.ndarray,
    sigma: jnp.ndarray,
    mu_r: jnp.ndarray,
    plx: int, phx: int,
    ply: int, phy: int,
    plz: int, phz: int,
    extra_masks=(),
):
    """Extend material properties (and dispersion-pole masks) into the CPML
    padding so guided modes see an impedance-matched absorber, equivalent
    to UPML. Each CPML face copies the interior-edge slice outward, as if
    the geometry continued beyond the domain.

    Single shared implementation for the uniform (``rfx/api/_compile.py``)
    and non-uniform (``rfx/runners/nonuniform.py``) assemblers — issue #627
    found the two hand-duplicated copies (#582 mirrored one onto the other)
    both carrying the same two gaps, so the fix lives once, here, and both
    call sites use it instead of keeping duplicated pad-extension code that
    can drift.

    ``extra_masks`` is a sequence of additional per-cell arrays (e.g.
    Debye/Lorentz dispersion-pole boolean masks, #627b — previously never
    extended into the pad at all, so a dispersive edge-touching material
    was matched at DC but not across the band) that get the identical
    lo/hi replication as ``eps_r``/``sigma``/``mu_r``, using the SAME
    hi-face source column decided below (same box, same dropped node, for
    every property of that box).

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

    Returns
    -------
    (eps_r, sigma, mu_r, extended_extra_masks) — ``extended_extra_masks``
    is a list in the same order as ``extra_masks``.
    """
    arrays = [eps_r, sigma, mu_r] + list(extra_masks)

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
    return eps_r, sigma, mu_r, arrays[3:]

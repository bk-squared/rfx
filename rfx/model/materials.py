"""Cell-material assembly shared by uniform and profiled grids.

The legacy entry points adapt only the return tuple. Grid-type branches keep
existing lane-specific sampling, sheet folds and refusals; no new physics rule.
Conductors own their realized products in rfx.model.conductors.
"""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer
from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import Box, _grid_coords
from rfx.geometry._pole_keying import _accumulate_pole_mask, _spec_from_pole_masks
from rfx.geometry.rasterize_grid import (
    GridCoords, _material_cell_mask, assert_declared_span_is_filled,
    cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
    centres_from_nonuniform_grid, centres_from_uniform_grid,
    classify_pec_entry, coords_from_nonuniform_grid, extend_cpml_pad_materials,
    refuse_vaporized_sheets, sheet_footprint_traced, sheet_spec_from_shape,
)
from rfx.geometry.smoothing import continued_conductor_shape
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.materials.thin_conductor import (
    apply_thin_conductor, check_sheet_occupancy, sheet_bounds,
)
from rfx.nonuniform import NonUniformGrid, e_node_dual_spacings


def assemble_cells(
    sim, grid, *, include_thin_conductors=True, include_cpml_pad_extension=True,
    sheet_specs=None, pec_sheets=None, pec_wires=None, pad_fill_findings=None,
    geometry_masks=None, assembly_entries=None,
):
    """Return materials, Debye/Lorentz specs, PEC cells/shapes/walls and Kerr.

    Collectors append to caller-owned lists; sheet/wire declarations require
    their collectors even for cell-only readers. The include flags expose the
    uniform vmap observation points before pad extension and thin folds.
    pad_fill_findings collects uniform declared-span shortfalls instead of
    raising. Materials/poles retain declaration order and float32 initialization.
    Overrides and design boxes remain downstream until M4.
    """
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    chi3_arr = jnp.zeros(grid.shape, dtype=jnp.float32)
    pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)
    pec_shapes = []
    has_kerr = False
    has_pec_cells = False

    debye_masks_by_pole: dict[DebyePole | int, tuple[DebyePole, jnp.ndarray]] = {}
    lorentz_masks_by_pole: dict[LorentzPole | int, tuple[LorentzPole, jnp.ndarray]] = {}

    # Grid type, not constant widths: constant-profile NU retains its sampler
    # and carrier restrictions. Uniform custom shapes may only implement mask(grid).
    nonuniform = isinstance(grid, NonUniformGrid)
    lane = "non-uniform" if nonuniform else "uniform"
    sampling_grid = None if nonuniform else grid
    if nonuniform:
        _coords = coords_from_nonuniform_grid(grid)
        _cell_sizes = cell_sizes_from_nonuniform_grid(grid)
        _centres = centres_from_nonuniform_grid(grid, _coords)
    else:
        _cx, _cy, _cz = _grid_coords(grid)
        _coords = GridCoords(x=_cx, y=_cy, z=_cz, shape=grid.shape)
        _centres = centres_from_uniform_grid(grid)
        _cell_sizes = cell_sizes_from_uniform_grid(grid)
    _pec_sheets = pec_sheets if pec_sheets is not None else []
    _pec_wires = pec_wires if pec_wires is not None else []

    # The declared-span audit is a uniform-only contract (#1070).
    _check_pad_fill = (not nonuniform and include_cpml_pad_extension
                       and sim._boundary in ("cpml", "upml")
                       and sim._cpml_layers > 0)
    from rfx.geometry.smoothing import continued_conductor_shape, warn_unextendable_shapes
    conductor_findings = []

    # NU historically continues all PEC declarations before sampling any entry.
    geometry = [replace(entry, shape=continued_conductor_shape(
                    sim, grid, entry.shape, entry=entry, unextendable=conductor_findings))
                if sim._resolve_material(entry.material_name).sigma >= sim._PEC_SIGMA_THRESHOLD
                else entry for entry in sim._geometry] if nonuniform else sim._geometry
    for declared_entry, entry in zip(sim._geometry, geometry, strict=True):
        mat = sim._resolve_material(entry.material_name)
        mask = _material_cell_mask(entry.shape, _coords, _centres, grid=sampling_grid)

        if mat.sigma >= sim._PEC_SIGMA_THRESHOLD:
            solved_shape = entry.shape if nonuniform else continued_conductor_shape(
                sim, grid, entry.shape, entry=entry, unextendable=conductor_findings)
            cells, sheet, wire = classify_pec_entry(
                solved_shape, _coords, _centres, _cell_sizes,
                name=entry.material_name, grid=sampling_grid)
            if assembly_entries is not None:
                assembly_entries.append((id(declared_entry), cells, sheet, wire, solved_shape))
            if cells is not None:
                pec_mask = pec_mask | cells
                has_pec_cells = True
                mask = cells
            elif sheet is not None:
                _pec_sheets.append(sheet)
                # Uniform chi3 uses the sheet footprint; NU retains its sampler.
                if not nonuniform:
                    mask = sheet.footprint
            else:
                _pec_wires.append(wire)
            pec_shapes.append(solved_shape)
        else:
            if _check_pad_fill and not is_tracer(mask):
                assert_declared_span_is_filled(
                    entry.material_name, entry.shape, mask, grid,
                    sim._unresolved_domain,
                    record=pad_fill_findings)
            eps_r = jnp.where(mask, mat.eps_r, eps_r)
            sigma = jnp.where(mask, mat.sigma, sigma)
            mu_r = jnp.where(mask, mat.mu_r, mu_r)

        if geometry_masks is not None and mat.sigma < sim._PEC_SIGMA_THRESHOLD:
            geometry_masks.append((id(declared_entry), mask))

        if mat.chi3 != 0.0:
            chi3_arr = jnp.where(mask, mat.chi3, chi3_arr)
            has_kerr = True

        pole_mask = mask
        if mat.debye_poles or mat.lorentz_poles:
            pole_mask = _material_cell_mask(declared_entry.shape, _coords, _centres, grid=sampling_grid)
            if mat.sigma >= sim._PEC_SIGMA_THRESHOLD and cells is not None:
                from rfx.geometry.rasterize_grid import pec_volume_cell_mask
                pole_mask = pec_volume_cell_mask(declared_entry.shape, _centres, _cell_sizes, grid=sampling_grid)

        if mat.debye_poles:
            for pole in mat.debye_poles:
                _accumulate_pole_mask(debye_masks_by_pole, pole, pole_mask)

        if mat.lorentz_poles:
            for pole in mat.lorentz_poles:
                _accumulate_pole_mask(lorentz_masks_by_pole, pole, pole_mask)

    # NU's geometry-sheet guard precedes pads and thin sheets. Its stepper
    # installs no periodic boundary; preserve both the timing and those flags.
    if nonuniform:
        pec_mask = _optional_pec_mask(pec_mask, has_pec_cells)
        refuse_vaporized_sheets(_pec_sheets, lane=lane, periodic=(False, False, False))

    # Extend statics only, before thin folds. Never extend poles or Kerr (#627b);
    # pole occupancy gates hi-face static promotion (#808).
    if (include_cpml_pad_extension
            and sim._boundary in ("cpml", "upml")
            and sim._cpml_layers > 0):
        plx, phx = grid.pad_x_lo, grid.pad_x_hi
        ply, phy = grid.pad_y_lo, grid.pad_y_hi
        plz, phz = grid.pad_z_lo, grid.pad_z_hi
        _pole_mask_any = None
        for _, _pmask in (list(debye_masks_by_pole.values())
                          + list(lorentz_masks_by_pole.values())):
            _pole_mask_any = (_pmask if _pole_mask_any is None
                              else (_pole_mask_any | _pmask))
        eps_r, sigma, mu_r = extend_cpml_pad_materials(
            eps_r, sigma, mu_r, plx, phx, ply, phy, plz, phz,
            dispersion_pole_mask=_pole_mask_any,
        )

    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    if include_thin_conductors and nonuniform:
        # NU uses local dual spacing and PEC-first order, and refuses non-Box DC.
        materials = _fold_nonuniform_thin_conductors(
            sim, grid, materials, _coords, _cell_sizes, _pec_sheets,
            sheet_specs, geometry_masks, assembly_entries, conductor_findings)
    elif include_thin_conductors:
        # Uniform preserves declaration order and its existing sheet sampler.
        for tc in sim._thin_conductors:
            geometry_key = id(tc)
            tc = replace(tc, shape=continued_conductor_shape(
                sim, grid, tc.shape, entry=tc, unextendable=conductor_findings))
            materials, pec_mask = apply_thin_conductor(
                grid, tc, materials, pec_mask=pec_mask,
                sheet_specs=sheet_specs, sheets=_pec_sheets,
                geometry_masks=geometry_masks, geometry_key=geometry_key)
            if assembly_entries is not None:
                assembly_entries.append((geometry_key, None,
                                         _pec_sheets[-1] if tc.is_pec else None, None, tc.shape))
            if tc.is_pec:
                pec_shapes.append(tc.shape)

    if include_thin_conductors:
        from rfx.materials.thin_conductor import pinned_sheet_spec
        for _ps in getattr(sim, "_pinned_sheets", ()) or ():
            _pec_sheets.append(pinned_sheet_spec(grid, _ps))

    boundary_pec_shapes: list = []
    # Only the uniform Dey-Mittra consumer carries boundary half-space shapes.
    conformal_faces = () if nonuniform else sim._boundary_spec.conformal_faces()
    if conformal_faces:
        _conformal_domain = sim._unresolved_domain
        big = max(_conformal_domain) * 100.0
        for face in conformal_faces:
            axis_name, side = face.split("_")
            axis_idx = "xyz".index(axis_name)
            wall_lo = 0.0
            wall_hi = float(_conformal_domain[axis_idx])
            for entry in sim._waveguide_ports:
                if entry.direction[1] == axis_name:
                    continue
                rng = (entry.x_range, entry.y_range,
                       entry.z_range)[axis_idx]
                if rng is None:
                    continue
                wall_lo = max(wall_lo, float(rng[0]))
                wall_hi = min(wall_hi, float(rng[1]))

            corner_lo = [-big, -big, -big]
            corner_hi = [big, big, big]
            if side == "lo":
                if wall_lo <= 0.0:
                    continue
                corner_hi[axis_idx] = wall_lo
            else:  # hi
                corner_lo[axis_idx] = wall_hi
            _bpec_box = Box(tuple(corner_lo), tuple(corner_hi))
            pec_shapes.append(_bpec_box)
            boundary_pec_shapes.append(_bpec_box)

    debye_spec = _spec_from_pole_masks(debye_masks_by_pole)
    lorentz_spec = _spec_from_pole_masks(lorentz_masks_by_pole)

    if not nonuniform:
        pec_mask = _optional_pec_mask(pec_mask, has_pec_cells)
    kerr_chi3 = chi3_arr if has_kerr else None
    from rfx.materials.thin_conductor import (
        warn_sheet_planes_inside_dielectric,
    )
    warn_sheet_planes_inside_dielectric(_pec_sheets, materials.eps_r)
    from rfx.api._compile import _refuse_uncollected_pec
    _refuse_uncollected_pec(_pec_sheets if pec_sheets is None else (),
                            _pec_wires if pec_wires is None else (),
                            lane=lane)
    if not nonuniform:
        refuse_vaporized_sheets(_pec_sheets, lane=lane, periodic=sim._periodic_flags())
    warn_unextendable_shapes(conductor_findings)
    return materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, boundary_pec_shapes, kerr_chi3


def _optional_pec_mask(mask, has_pec_cells):
    """Keep eager empty masks absent and the outer-jit decision static."""
    has_pec = has_pec_cells if is_tracer(mask) else bool(jnp.any(mask))
    return mask if has_pec else None


def _fold_nonuniform_thin_conductors(
    sim, grid, materials, coords, cell_sizes, _pec_sheets,
    sheet_specs, geometry_masks, assembly_entries, conductor_findings,
):
    """Preserve the profiled-grid fold: PEC first, lossy local E-node duals."""
    from rfx.runners.nonuniform import nu_thin_conductor_refusal
    if sim._thin_conductors:
        conductors = [replace(tc, shape=continued_conductor_shape(
                        sim, grid, tc.shape, entry=tc, unextendable=conductor_findings))
                      for tc in sim._thin_conductors]
        conductor_keys = {id(tc): id(original)
                          for tc, original in zip(conductors, sim._thin_conductors, strict=True)}
        pec_tcs = [tc for tc in conductors
                   if getattr(tc, "is_pec", False)]
        lossy_tcs = [tc for tc in conductors
                     if not getattr(tc, "is_pec", False)]
        for tc in pec_tcs:
            _pec_sheets.append(sheet_spec_from_shape(
                tc.shape, coords, cell_sizes, name="thin_conductor",
                lane="non-uniform", refuse_thick=True))
            if assembly_entries is not None:
                assembly_entries.append((conductor_keys[id(tc)], None, _pec_sheets[-1], None, tc.shape))
        for tc in lossy_tcs:
            if assembly_entries is not None:
                assembly_entries.append((conductor_keys[id(tc)], None, None, None, tc.shape))
            _f0 = getattr(tc, "surface_impedance_f0", None)
            if _f0 is not None:
                lo, hi = sheet_bounds(tc.shape)
                if lo is None or hi is None:
                    raise ValueError(
                        "surface-impedance (surface_impedance_f0) thin "
                        "conductor requires a shape with an axis-aligned "
                        "bounding box (Box corner_lo/corner_hi, or "
                        "Shape.bounding_box()) to locate its normal; "
                        "refusing to skip it on the non-uniform path.")
            else:
                _refusal = nu_thin_conductor_refusal(tc)
                if _refusal is not None:
                    raise NotImplementedError(_refusal)
                lo = tc.shape.corner_lo
                hi = tc.shape.corner_hi
            extents = [float(hi[i]) - float(lo[i]) for i in range(3)]
            n_axis = min(range(3), key=lambda i: extents[i])  # sheet normal axis
            d_norm = e_node_dual_spacings(
                (grid.dx_arr, grid.dy_arr, grid.dz)[n_axis])
            bshape = [1, 1, 1]
            bshape[n_axis] = int(d_norm.shape[0])
            _plane = None
            if _f0 is not None:
                if any(is_tracer(c) for c in (coords.x, coords.y, coords.z)):
                    m = sheet_footprint_traced(tc.shape, coords, n_axis)
                else:
                    _spec = sheet_spec_from_shape(
                        tc.shape, coords, cell_sizes, normal_axis=n_axis,
                        name="thin_conductor", lane="non-uniform",
                        refuse_thick=True)
                    m = _spec.footprint
                    _plane = _spec.plane
                check_sheet_occupancy(m, n_axis, lane="non-uniform")
                if geometry_masks is not None:
                    geometry_masks.append((conductor_keys[id(tc)], m))
                from rfx.materials.thin_conductor import (
                    SheetImpedanceSpec, leontovich_rs)
                rs0 = leontovich_rs(_f0, tc.sigma_bulk)
                g_sheet = 1.0 / rs0
                if sheet_specs is not None:
                    sigma_sheet = jnp.where(
                        m, (g_sheet / d_norm).reshape(bshape) *
                        jnp.ones_like(materials.sigma), 0.0)
                    sheet_specs.append(SheetImpedanceSpec(
                        mask=m, normal_axis=n_axis, g_sheet=g_sheet,
                        sigma_sheet=sigma_sheet, plane=_plane))
                continue
            m = tc.shape.mask_on_coords(coords.x, coords.y, coords.z)
            if geometry_masks is not None:
                geometry_masks.append((conductor_keys[id(tc)], m))
            sigma_eff = tc.sigma_bulk * (tc.thickness / d_norm.reshape(bshape))
            materials = materials._replace(
                eps_r=jnp.where(m, tc.eps_r, materials.eps_r),
                sigma=jnp.where(m, sigma_eff, materials.sigma),
            )
    return materials

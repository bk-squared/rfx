"""Nominal conductor continuation with checks on the realized mesh."""

from copy import copy

import jax
import jax.numpy as jnp
import numpy as np

from rfx._traced_check import traced_check
from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box, Cylinder, declared_bounds


def _candidate_faces(sim, grid, bounds):
    from rfx.geometry.smoothing import _reaches_pad_face

    faces, missing, unpadded_missing = [], [], []
    nominal = sim._nominal_profiles or {}
    for axis, name in enumerate("xyz"):
        profile = getattr(sim, f"_d{name}_profile")
        if is_tracer(profile):
            if name not in nominal:
                padded = any(getattr(grid, f"pad_{name}_{side}") > 0
                             for side in ("lo", "hi"))
                (missing if padded else unpadded_missing).append(name)
                continue
            entry = nominal[name]
            profile = getattr(entry, "cells", entry)
        if any(is_tracer(bound[axis]) for bound in bounds):
            continue
        for side in ("lo", "hi"):
            if getattr(grid, f"pad_{name}_{side}") <= 0:
                continue
            cell = sim._dx if profile is None else float(np.asarray(profile)[0 if side == "lo" else -1])
            within_cell = (float(bounds[0][axis]) <= cell if side == "lo" else
                           float(bounds[1][axis]) >= sim._unresolved_domain[axis] - cell)
            if within_cell or _reaches_pad_face(
                    bounds, axis, side, 0., cell, sim._unresolved_domain, occupied_faces=()):
                faces.append((axis, side))
    if faces or missing:
        missing.extend(unpadded_missing)
    return faces, missing


def refuse_traced_conductor_at_absorber(sim, grid, shape, *, traced_node_axes):
    """Keep declaration-based refusal for continuation without a nominal lattice."""
    bounds = declared_bounds(shape)
    if bounds is None:
        return
    faces, missing = _candidate_faces(sim, grid, bounds)
    traced = [name for name in "xyz" if is_tracer(getattr(sim, f"_d{name}_profile"))]
    if not traced:
        traced = ["xyz"[axis] for axis in traced_node_axes]
    profiles = ", ".join(f"d{name}_profile" for name in traced)
    if missing:
        raise TracedStructureRefusal(
            "Cannot continue a conductor with finite declared extent on "
            f"traced axis {', '.join(missing)}: the edge "
            "cell has no concrete declaration. Declare nominal_profiles for "
            f"axis {', '.join(missing)}, or keep it concrete by passing "
            f"that axis's profile ({profiles}) as a concrete NumPy array for this run.")
    if faces:
        names = ", ".join(f"{'xyz'[axis]}-{side}" for axis, side in faces)
        raise TracedStructureRefusal(
            f"Cannot continue {type(shape).__name__} at absorbing face(s) {names} "
            f"while mesh axis {', '.join(traced)} is traced: its continuation "
            "requires a concrete lattice. Declare it as a Box volume, or keep "
            f"the axis concrete by passing {profiles} as a concrete NumPy array.")


def _nominal_grid(sim):
    """Reuse the grid builder with concrete declarations on every profile axis."""
    nominal_sim = copy(sim)
    for name in "xyz":
        profile = getattr(sim, f"_d{name}_profile")
        if is_tracer(profile):
            entry = sim._nominal_profiles[name]
            setattr(nominal_sim, f"_d{name}_profile", getattr(entry, "cells", entry))
    nominal_sim._nominal_profiles = None
    with jax.ensure_compile_time_eval():
        return nominal_sim._build_nonuniform_grid()


def _realized_lattice(sim, grid, shape, coords):
    from rfx.geometry.rasterize_grid import (
        cell_centres_from_nodes, cell_sizes_from_nonuniform_grid,
        classify_pec_entry, sheet_spec_from_shape)

    sizes = cell_sizes_from_nonuniform_grid(grid)
    thin = next((tc for tc in sim._thin_conductors if tc.shape is shape), None)
    if thin is not None:
        if thin.is_pec or thin.surface_impedance_f0 is not None:
            sheet = sheet_spec_from_shape(shape, coords, sizes, grid=grid, refuse_thick=True)
            return [(sheet.footprint, (False,) * 3)]
        from rfx.materials.thin_conductor import _thin_conductor_cell_mask
        return [(_thin_conductor_cell_mask(shape, grid), (False,) * 3)]
    centres = cell_centres_from_nodes(coords, sizes)
    cells, sheet, wire = classify_pec_entry(shape, coords, centres, sizes, grid=grid)
    if cells is not None:
        return [(cells, (True,) * 3)]
    if sheet is not None:
        return [(sheet.footprint, (False,) * 3)]
    return [(mask, tuple(a == component for a in range(3)))
            for component, mask in enumerate(wire.edges)]


def _check_faces(sim, grid, shape, nominal_occupied, targets):
    from rfx.boundaries.pec import SheetFootprint
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid

    coords = coords_from_nonuniform_grid(grid)
    lattice = _realized_lattice(sim, grid, shape, coords)
    faces, agrees = [], []
    for (axis, side), target in targets.items():
        pad = int(getattr(grid, f"pad_{'xyz'[axis]}_{side}"))
        occupied = False
        for mask, cell_axes in lattice:
            index = pad if side == "lo" else grid.shape[axis] - 1 - pad - int(cell_axes[axis])
            if isinstance(mask, SheetFootprint):
                section = (mask.mask if index == mask.plane else False) if axis == mask.normal_axis else jnp.take(
                    mask.mask, index, axis=axis - (axis > mask.normal_axis))
            else:
                section = jnp.take(mask, index, axis=axis)
            occupied = occupied | jnp.any(section)
        line = coords[axis]
        inside = line[0] >= target if side == "lo" else line[-1] <= target
        agrees.append((occupied == ((axis, side) in nominal_occupied)) & inside)
        faces.append(f"{'xyz'[axis]}-{side}")

    def check(values):
        for face, valid in zip(faces, np.asarray(values)):
            if not valid:
                raise ValueError(
                    f"conductor continuation: outermost layer at {face} differs "
                    "from the nominal occupancy or the realized pad exceeds its "
                    "continuation. Re-declare nominal_profiles at the current "
                    "design, or keep the axis concrete.")

    traced_check("conductor continuation", "outermost layer", check, jnp.stack(agrees))


def continued_traced_conductor_shape(sim, grid, shape, *, traced_node_axes,
                                     entry=None, unextendable=None):
    """Continue from the concrete nominal decision without traced shape coordinates."""
    from rfx.geometry import smoothing as sm
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid

    bounds = declared_bounds(shape)
    if bounds is None:
        return shape
    faces, missing = _candidate_faces(sim, grid, bounds)
    if missing:
        refuse_traced_conductor_at_absorber(sim, grid, shape, traced_node_axes=traced_node_axes)
    if not faces:
        return shape
    nominal = _nominal_grid(sim)
    with jax.ensure_compile_time_eval():
        solved = sm.continued_conductor_shape(
            sim, nominal, shape, entry=entry, unextendable=unextendable)
        coords = coords_from_nonuniform_grid(nominal)
        occupied = sm._occupied_conductor_faces(
            sm._declared_conductor_lattice(sim, nominal, shape, coords), nominal)
    if isinstance(solved, sm._ContinuedSheetShape):
        refuse_traced_conductor_at_absorber(sim, grid, shape, traced_node_axes=traced_node_axes)
    changed = []
    if isinstance(solved, (Box, Cylinder)):
        new_bounds = declared_bounds(solved)
        # Reconstructing a cylinder's centre and height can round its held end.
        tolerance = (16 * np.finfo(float).eps * np.max(np.abs(bounds))
                     if isinstance(shape, Cylinder) else 0.)
        changed = [(axis, side) for axis in range(3) for i, side in enumerate(("lo", "hi"))
                   if abs(new_bounds[i][axis] - bounds[i][axis]) > tolerance]
    if not changed and not occupied:
        return shape
    targets = {}
    for axis, name in enumerate("xyz"):
        widths = np.asarray(getattr(nominal, ("dx_arr_f64", "dy_arr_f64", "dz_f64")[axis]))
        for side in ("lo", "hi"):
            pad = int(getattr(grid, f"pad_{name}_{side}"))
            if pad:
                reach = 2 * pad * float(np.max(widths))
                targets[axis, side] = -reach if side == "lo" else sim._unresolved_domain[axis] + reach
    _check_faces(sim, grid, shape, occupied, targets)
    result = shape
    for axis, side in changed:
        grow = sm._continue_box if isinstance(shape, Box) else sm._continue_cylinder
        result = grow(result, axis, side, targets[axis, side])
    return result

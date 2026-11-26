"""Magnetic curl coefficients formed before the uniform time loop."""
from typing import NamedTuple

import jax.numpy as jnp

from rfx.core.yee import MU_0, component_h_materials


class CPMLHCoefficients(NamedTuple):
    dt: object
    faces: dict


def yee_h_coefficients(materials, dt, periodic, high_wall, factors):
    """Retain the Yee material read, audit/replay and arithmetic order."""
    from rfx._internal import _realized
    from rfx.boundaries.pmc import magnetic_update_coefficients
    mu = (materials.components.mu_update if materials.components is not None
          else component_h_materials(materials, periodic, high_wall=high_wall))
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H", periodic=periodic)
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch = dt / mu_x, dt / mu_y, dt / mu_z
    return magnetic_update_coefficients(ch, factors)


def cpml_h_coefficients(materials, grid, shape, dtype, periodic, axes, factors, frame=None):
    """Form only the tangential coefficients in the selected face windows."""
    from rfx.boundaries.cpml import _axis_buffer_depths
    from rfx.boundaries._cpml_slab import face_windows, selected_faces, ALL_FACES
    from rfx.boundaries.electric_coefficient import coefficient_dt
    from rfx.model.materials import h_components
    depths = _axis_buffer_depths(grid, grid.cpml_layers)
    _, windows, _, face_depths = face_windows(shape, depths, frame, periodic, grid)
    dt = coefficient_dt(grid.dt, dtype)
    mu = h_components(materials, grid, periodic=periodic)
    ch = tuple(dt / (m * MU_0) for m in mu)
    result = {}
    for face in sorted(selected_faces(axes, ALL_FACES)):
        if not face_depths[face]:
            continue
        axis = "xyz".index(face[0])
        window = windows[face]
        for component in range(3):
            if component == axis:
                continue
            arr = ch[component]
            if jnp.ndim(arr) != 0:
                arr = arr[window]
            for b, factor in enumerate(factors):
                if b != component and factor is not None:
                    broadcast = [1, 1, 1]
                    broadcast[b] = -1
                    arr = arr * jnp.asarray(factor[window[b]], dtype=arr.dtype).reshape(broadcast)
            result[component, axis, face.endswith("lo")] = arr
    return CPMLHCoefficients(dt, result)


def uniform_h_coefficients(invariants, dtype):
    """Fold after padding so coefficient shapes follow the actual operands."""
    from rfx.geometry.rasterize_grid import unpadded_high_walls
    ctx = invariants["ctx"]
    if ctx.use_fast_he or ctx.use_upml:
        return invariants
    factors = invariants["h_row_factors"]
    invariants["h_coefficients"] = yee_h_coefficients(
        ctx.materials, ctx.dt, ctx.periodic, unpadded_high_walls(ctx.grid), factors)
    if ctx.use_cpml:
        plan = ctx.pad_plan
        if dtype is None:
            dtype = ctx.cpml_params.x_lo.b.dtype
        invariants["cpml_h_coefficients"] = cpml_h_coefficients(
            ctx.materials, ctx.grid, ctx.grid.shape if plan is None else plan.padded,
            dtype, ctx.periodic, ctx.cpml_axes, factors,
            None if plan is None else plan.frame)
    return invariants

"""Step-local precision operands; realized geometry retains its storage policy."""
import jax.numpy as jnp
import jax


def field_grid(grid, field_dtype):
    """Widen graded metrics once, preserving the float32 operator verbatim."""
    if jnp.dtype(field_dtype) != jnp.dtype('float64'):
        return grid
    names = ('dt', 'dx', 'dy', 'dx_arr', 'dy_arr', 'dz',
             'inv_dx', 'inv_dy', 'inv_dz', 'inv_dx_h', 'inv_dy_h', 'inv_dz_h')
    return grid._replace(**{name: jnp.asarray(getattr(grid, name), dtype=field_dtype)
                            for name in names})


def field_operands(tree, field_dtype):
    """Widen precomputed profiles/metrics before per-step coefficient math."""
    if jnp.dtype(field_dtype) != jnp.dtype('float64'):
        return tree
    return jax.tree.map(lambda value: jnp.asarray(value, dtype=field_dtype), tree)


def init_graded_dispersion(materials, dt, periodic_axes, debye_spec, lorentz_spec, field_dtype):
    """Build ADE operands after final stamps using the graded field time."""
    from rfx.materials.debye import init_debye
    from rfx.materials.lorentz import init_lorentz
    result = []
    for spec, init in ((debye_spec, init_debye), (lorentz_spec, init_lorentz)):
        result.append(None if spec is None else init(
            spec[0], materials, dt, mask=spec[1], field_dtype=field_dtype,
            periodic=tuple(a in periodic_axes for a in "xyz")))
    return tuple(result)


def field_poles(materials, grid, debye_spec, lorentz_spec, field_dtype):
    """Re-form cached pole terms with field time, retaining realized edges."""
    if jnp.dtype(field_dtype) != jnp.dtype('float64'):
        return materials
    from dataclasses import replace
    from rfx.model.materials import _realize_poles, electric_cell_sizes, unpadded_high_walls
    poles = {name: _realize_poles(spec, materials.components.periodic, kind, grid,
             materials.eps_r.shape, high_wall=unpadded_high_walls(grid),
             cell_sizes=electric_cell_sizes(grid))
             for name, spec, kind in (('debye', debye_spec, 'Debye'),
                                      ('lorentz', lorentz_spec, 'Lorentz'))}
    return materials._replace(components=replace(materials.components, **poles))

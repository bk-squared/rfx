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

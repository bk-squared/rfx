"""Shared JAX utility helpers used across rfx sub-modules."""
import jax


def is_tracer(x: object) -> bool:
    """Return True if *x* is a JAX abstract tracer value.

    Use before any host-side coercion (float(), np.array(), etc.).
    """
    return isinstance(x, jax.core.Tracer)


def _is_cell_array(x, grid_shape) -> bool:
    """A concrete array with as many axes longer than one as the grid has.

    That is what scales with the grid: per-cell materials, masks and
    coefficients (``(nx, ny, nz)``, ``(n_poles, nx, ny, nz)``; ``(nx, ny, 1)``
    in 2-D mode) and box-shaped design coefficients. CPML profiles
    (``(layers,)`` or ``(layers, 1, 1)``), spacing vectors and port planes
    have fewer, and are not lifted.
    """
    import numpy as np
    if is_tracer(x) or not isinstance(x, (jax.Array, np.ndarray)):
        return False
    grid_axes = sum(1 for n in grid_shape if n > 1)
    return sum(1 for n in x.shape if n > 1) >= max(grid_axes, 2)


def _all_equal(x) -> bool:
    import numpy as np
    if isinstance(x, np.ndarray):
        return bool(np.all(x == x.reshape(-1)[0]))
    import jax.numpy as jnp
    return bool(jnp.all(x == x.reshape(-1)[0]))


def _const(value):
    return lambda a: value


def split_loop_invariants(tree, grid_shape):
    """Split the per-cell arrays out of a time-step closure's context.

    A concrete array that a jitted function closes over is compiled into the
    program as a literal. For a whole-grid array that literal is the size of
    the grid, and XLA holds copies of it while it compiles: on a large
    non-uniform board (~1e8 cells) the first chunk's compile took +27.8 GiB
    of host memory and 67.6 s with the materials, masks and coefficients
    closed over, against +0.16 GiB and 5.1 s with them as arguments (the
    same outputs, bit for bit). #1211 and #1217 made the same change for the
    distributed runners.

    Returns ``(arrays, rebuild)``. ``arrays`` is the list of per-cell arrays in
    ``tree`` (see :func:`_is_cell_array`; ``grid_shape`` is the grid's
    ``(nx, ny, nz)``) whose elements are not all equal;
    pass it to the jitted function as an argument and call ``rebuild(arrays)``
    inside it to get ``tree`` back with those leaves replaced. Everything
    else stays as it was, including:

    * an all-equal per-cell array (a vacuum ``eps_r``, an all-false mask):
      it is lowered as one scalar broadcast, so it costs the compile
      nothing, and as a constant the compiler keeps simplifying arithmetic
      on it. As an argument it does not, which moved uniform-permittivity
      results by up to 7 float32 ULP in #1211.
    * CPML profiles and other small arrays, for the same reason (as
      arguments they moved results by float32 rounding).
    * Python scalars and tuples (slice bounds, flags), which must stay static.

    ``tree`` may nest dicts, tuples, lists, NamedTuples and dataclasses.
    """
    import dataclasses

    arrays: list = []

    def split(value):
        # Returns a rebuild function, or None when nothing below is lifted
        # (the object is then kept as it is, never reconstructed). A rebuild
        # function holds the objects it keeps, never a lifted array.
        if _is_cell_array(value, grid_shape) and not _all_equal(value):
            k = len(arrays)
            arrays.append(value)
            return lambda a: a[k]
        if isinstance(value, dict):
            parts = {key: split(v) for key, v in value.items()}
            if not any(parts.values()):
                return None
            keep = {key: (p or _const(value[key])) for key, p in parts.items()}
            return lambda a: {key: p(a) for key, p in keep.items()}
        if isinstance(value, (tuple, list)):
            parts = [split(v) for v in value]
            if not any(parts):
                return None
            keep = [p or _const(v) for p, v in zip(parts, value)]
            cls = type(value)
            if hasattr(value, "_fields"):
                return lambda a: cls(*(p(a) for p in keep))
            return lambda a: cls(p(a) for p in keep)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            names = [f.name for f in dataclasses.fields(value) if f.init]
            parts = {n: split(getattr(value, n)) for n in names}
            if not any(parts.values()):
                return None
            keep = {n: (p or _const(getattr(value, n))) for n, p in parts.items()}
            cls = type(value)
            return lambda a: cls(**{n: p(a) for n, p in keep.items()})
        return None

    rebuild = split(tree)
    if rebuild is None:
        return arrays, lambda a: tree
    return arrays, rebuild

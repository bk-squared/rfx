"""Shared JAX utility helpers used across rfx sub-modules."""
import contextlib
import contextvars
import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from rfx.backend import backend_class

def is_tracer(x: object) -> bool:
    """Return True if *x* is a JAX abstract tracer value.

    Use before any host-side coercion (float(), np.array(), etc.).
    """
    return isinstance(x, jax.core.Tracer)


# ---------------------------------------------------------------------------
# The set-up / time-stepping boundary under an outer trace (#1367, #1354)
# ---------------------------------------------------------------------------
#
# Rule: set-up that reads only the declaration is always concrete, and the
# time stepping is always recorded. Under ``jax.jit`` (or make_jaxpr,
# eval_shape) JAX records every operation, including the ones that read only
# the model, so a host read of a realized PEC mask or of an NTFF frequency
# list fails. ``declaration_setup()`` evaluates everything that does not read
# a traced input while it traces (``jax.ensure_compile_time_eval``);
# ``recorded_scan`` is the way back out for the time-stepping loop, so the
# solve is compiled into the caller's program instead of run while it traces,
# over operands the compiler cannot read, as in a plain call.


class _Region:
    """A ``declaration_setup()`` in progress.

    ``anchor`` is a tracer of the outer trace, made before the region was
    entered, when JAX's switch is missing (the fallback), else None.
    """

    __slots__ = ("anchor",)

    def __init__(self, anchor):
        self.anchor = anchor


#: The ``declaration_setup()`` in progress, or None.
_REGION = contextvars.ContextVar("rfx_declaration_setup", default=None)


@functools.lru_cache(maxsize=None)
def compile_time_switch():
    """JAX's switch behind ``jax.ensure_compile_time_eval``, or ``None``.

    On JAX 0.10.2, the only version rfx runs in required CI, the region is the
    configuration state ``eager_constant_folding`` set to True, and setting
    it back to False inside the region records operations again. That state
    is private API, so it is trusted by behaviour, not by version number,
    probed once: inside a ``jax.make_jaxpr`` trace a constant must be
    concrete in the region, with the state on, and a tracer once the state
    is turned off. A JAX that fails the probe in any way (the declared
    0.4.20 floor implements the region differently) gets ``None``, and
    :func:`recorded_scan` uses its fallback. Every plain ``run()`` reaches
    this probe, so no exception it raises may escape.
    """
    try:
        from jax._src import config as _config
        switch = _config.eager_constant_folding
    except Exception:
        return None
    seen = []

    def probe():
        with jax.ensure_compile_time_eval():
            folded = not is_tracer(jnp.zeros(())) and bool(switch.value)
            with switch(False):
                staged = is_tracer(jnp.zeros(()))
        seen.append(folded and staged)
        return jnp.zeros(())

    try:
        jax.make_jaxpr(probe)()
    except Exception:
        return None
    return switch if seen == [True] else None


@contextlib.contextmanager
def declaration_setup():
    """Evaluate what reads only the declaration now; the outer trace records the rest.

    Entered only while an outer trace stages constants. Operations that read
    a traced input are still recorded; everything built from the model
    alone (grid, realized PEC edge masks, port cells, NTFF frequencies) is
    computed at trace time and becomes a constant of the caller's program.
    The time stepping inside must go through :func:`recorded_scan`.
    """
    anchor = None
    if compile_time_switch() is None:
        # Fallback: an operand of the outer trace, made before the region,
        # that lifts recorded_scan's concrete operands into the outer trace.
        anchor = jnp.ones((), jnp.float32)
    token = _REGION.set(_Region(anchor))
    try:
        with jax.ensure_compile_time_eval():
            yield
    finally:
        _REGION.reset(token)


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


def _one_bit_pattern(x):
    """The single element of an all-equal strong-typed ``jax.Array``, else None.

    ``==`` is not enough to rebuild an array from one element: +0.0 equals
    -0.0. So the signs must agree as well (NaN never compares equal, so an
    array holding one is not all-equal to begin with). A NumPy leaf and a
    weak-typed array are left as they are: the first stays a host constant
    its consumers may read while tracing, the second would come back
    strong-typed.
    """
    import numpy as np
    import jax.numpy as jnp
    if not isinstance(x, jax.Array) or getattr(x, "weak_type", False):
        return None
    first = x.reshape(-1)[0]
    if jnp.issubdtype(x.dtype, jnp.inexact):
        parts = (jnp.real, jnp.imag) if jnp.iscomplexobj(x) else (lambda v: v,)
        if not all(bool(jnp.all(jnp.signbit(p(x)) == jnp.signbit(p(first))))
                   for p in parts):
            return None
    return np.asarray(first)[()]


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
      results by up to 7 float32 ULP in #1211. It stays a constant, but not
      a closed-over array: when it is a ``jax.Array`` with one bit pattern
      everywhere, ``rebuild`` forms it as ``jnp.full(shape, value)`` inside
      the traced function (see :func:`_one_bit_pattern`), so the function
      holds no grid-sized object (rfx tracker 1606).
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
        if _is_cell_array(value, grid_shape):
            if not _all_equal(value):
                k = len(arrays)
                arrays.append(value)
                return lambda a: a[k]
            # One bit pattern everywhere: the program gets the same constant
            # as a broadcast of that value, formed where the step is traced,
            # so no array of the grid's size is closed over. A jitted step
            # that closes over one and is called inside a scan stays in JAX's
            # caches with it after the run (rfx tracker 1606).
            one = _one_bit_pattern(value)
            if one is not None:
                import jax.numpy as jnp
                shape = tuple(value.shape)
                return lambda a: jnp.full(shape, one)
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


# Complex linear solves (#1829). The jax-mps plugin (Apple-silicon GPUs) has no
# complex triangular-solve kernel, so complex solves go through the real
# 2n x 2n block form there; every other backend runs the native op unchanged.
def _is_complex(*arrays):
    return any(jnp.iscomplexobj(a) for a in arrays)


def _block_solve(a, b):
    """Solve complex matrix RHSs (or an unbatched vector) in the real dtype."""
    if not _is_complex(a, b):
        return jnp.linalg.solve(a, b)
    dtype = jnp.asarray(a).dtype if jnp.iscomplexobj(a) else jnp.result_type(a, b)
    a, b = jnp.asarray(a, dtype=dtype), jnp.asarray(b, dtype=dtype)
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    n = a.shape[-1]
    ar, ai = jnp.real(a), jnp.imag(a)
    matrix = jnp.concatenate([
        jnp.concatenate([ar, -ai], axis=-1),
        jnp.concatenate([ai, ar], axis=-1),
    ], axis=-2)
    rhs = jnp.concatenate([jnp.real(b), jnp.imag(b)], axis=-2)
    y = jnp.linalg.solve(matrix, rhs)
    x = (y[..., :n, :] + 1j * y[..., n:, :]).astype(dtype)
    return x[..., 0] if vector else x


def complex_solve(a, b):
    """Use the native solve except for complex operands on mps at trace time."""
    if backend_class().name == "mps" and _is_complex(a, b):
        return _block_solve(a, b)
    return jnp.linalg.solve(a, b)


def complex_inv(a):
    """Use a real block solve for a complex inverse on mps."""
    if backend_class().name == "mps" and _is_complex(a):
        return _block_solve(a, jnp.eye(a.shape[-1], dtype=a.dtype))
    return jnp.linalg.inv(a)


def complex_solve_triangular(r, b, *, lower):
    """The real block form is general, even when the complex input is triangular."""
    if backend_class().name == "mps" and _is_complex(r, b):
        # The native op reads only one triangle; so must the block form.
        return _block_solve(jnp.tril(r) if lower else jnp.triu(r), b)
    return jax.scipy.linalg.solve_triangular(r, b, lower=lower)


# Moved to the stepping layer; the old import path resolves lazily (a module-level import would be circular).
_LAZY_EXPORTS = {
    "recorded_scan": "rfx.stepping.scan",
    "_scan_with_opaque_operands": "rfx.stepping.scan",
    "_opaque": "rfx.stepping.scan",
    "_optimization_barrier": "rfx.stepping.scan",
}


def __getattr__(name):
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _LAZY_EXPORTS, name)

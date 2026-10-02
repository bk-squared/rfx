"""Shared JAX utility helpers used across rfx sub-modules."""
import contextlib
import contextvars
import functools

import jax
import jax.numpy as jnp


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


def _optimization_barrier(values):
    """``jax.lax.optimization_barrier``; before JAX made it public (the
    declared 0.4.20 floor) the same primitive is ``jax._src.ad_checkpoint``'s."""
    barrier = getattr(jax.lax, "optimization_barrier", None)
    if barrier is None:
        from jax._src.ad_checkpoint import _optimization_barrier as barrier
    return barrier(values)


def _opaque(tree, anchor):
    """``tree`` with every concrete array behind one ``optimization_barrier``.

    The barrier is the identity on values, bit for bit and for every dtype;
    its outputs are operands of the outer trace that the compiler cannot
    read. With ``anchor`` (the fallback, where the region still evaluates
    concrete operations at once) the barrier also takes the anchor, a tracer
    of the outer trace, so it is recorded there and not evaluated.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    picked = [i for i, leaf in enumerate(leaves)
              if hasattr(leaf, "dtype") and not is_tracer(leaf)]
    if picked:
        values = [leaves[i] for i in picked]
        if anchor is None:
            values = _optimization_barrier(values)
        else:
            _, values = _optimization_barrier((anchor, values))
        for i, value in zip(picked, values):
            leaves[i] = value
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _scan_with_opaque_operands(f, init, xs, length, kwargs, anchor):
    """``jax.lax.scan(f, init, xs)`` recorded over operands the compiler cannot read.

    A scan's operands are its carry, its ``xs`` and the arrays its body
    closes over. A plain call passes every one of them to the compiled loop
    as an argument. Here they are concrete, and recorded as they are they
    would become constants of the caller's program that XLA rewrites the
    loop around: on an absorbing (CPML) board the jitted probe record then
    drifted 20-170 float32 ULP at its peak from the plain call's over 200
    steps. So the body is traced once to find the arrays it closes over, and
    those, the carry and ``xs`` enter the loop through :func:`_opaque`; the
    loop then compiles as in the plain call.
    """
    leading = [a.shape[0] for a in jax.tree_util.tree_leaves(xs)
               if hasattr(a, "shape")]
    if (length == 0) or (leading and leading[0] == 0):
        # A zero-length scan runs nothing; its outputs are empty.
        return jax.lax.scan(f, init, xs, length=length, **kwargs)
    x0 = None if xs is None else jax.tree_util.tree_map(lambda a: a[0], xs)
    closed, out_shape = jax.make_jaxpr(f, return_shape=True)(init, x0)
    consts, init, xs = _opaque((list(closed.consts), init, xs), anchor)
    jaxpr = closed.jaxpr
    out_tree = jax.tree_util.tree_structure(out_shape)

    def body(carry, x):
        out = jax.core.eval_jaxpr(
            jaxpr, consts, *jax.tree_util.tree_leaves((carry, x)))
        return jax.tree_util.tree_unflatten(out_tree, out)

    return jax.lax.scan(body, init, xs, length=length, **kwargs)


def recorded_scan(f, init, xs=None, length=None, **kwargs):
    """``jax.lax.scan`` for a time-stepping loop: recorded, never run while tracing.

    Every time-stepping scan entry (uniform, graded, ADI) calls this instead
    of ``jax.lax.scan``. Inside :func:`declaration_setup` a scan whose
    operands are all constants would otherwise be evaluated at trace time:
    JAX 0.6.2 raises ``Evaluation rule for 'empty' not implemented`` (#1354),
    and 0.10.2 runs the solve uncompiled while it traces the caller's
    program. Its body would also be traced under eager constant folding.

    Inside the region the scan is traced with JAX's switch
    (:func:`compile_time_switch`) off, exactly as outside it, and every
    concrete operand reaches the loop through an ``optimization_barrier``
    (:func:`_scan_with_opaque_operands`), so the compiled loop is the one a
    plain call compiles. Without the switch the barrier also takes a tracer
    of the outer trace, which records it. A scan inside the body is a plain
    scan. Outside the region this is ``jax.lax.scan``.
    """
    region = _REGION.get()
    switch = compile_time_switch()
    if region is None:
        if switch is not None and switch.value:
            with switch(False):
                return jax.lax.scan(f, init, xs, length=length, **kwargs)
        return jax.lax.scan(f, init, xs, length=length, **kwargs)
    token = _REGION.set(None)
    try:
        if switch is not None:
            with switch(False):
                return _scan_with_opaque_operands(
                    f, init, xs, length, kwargs, None)
        return _scan_with_opaque_operands(
            f, init, xs, length, kwargs, region.anchor)
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

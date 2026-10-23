"""Record time-stepping scans across the declaration setup boundary."""
import jax

from rfx.core.jax_utils import _REGION, compile_time_switch, is_tracer


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

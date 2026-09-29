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
# solve is compiled into the caller's program instead of run while it traces.

#: A tracer of the outer trace, created before the compile-time region is
#: entered. Only the fallback of ``recorded_scan`` reads it.
_OUTER_ANCHOR = contextvars.ContextVar("rfx_outer_trace_anchor", default=None)


@functools.lru_cache(maxsize=None)
def compile_time_switch():
    """JAX's switch behind ``jax.ensure_compile_time_eval``, or ``None``.

    On the JAX versions rfx runs in CI (0.6.2 and 0.10.2) the region is the
    configuration state ``eager_constant_folding`` set to True, and setting
    it back to False inside the region records operations again. That state
    is private API, so it is trusted by behaviour, not by version number,
    probed once: inside a ``jax.make_jaxpr`` trace a constant must be
    concrete in the region, with the state on, and a tracer once the state
    is turned off. A JAX that fails the probe (the declared 0.4.20 floor
    implements the region differently) gets ``None``, and
    :func:`recorded_scan` uses its fallback.
    """
    try:
        from jax._src import config as _config
        switch = _config.eager_constant_folding
    except (ImportError, AttributeError):
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
    except (AttributeError, TypeError):
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
        # that recorded_scan threads into an all-constant carry.
        anchor = jnp.ones((), jnp.float32)
    token = _OUTER_ANCHOR.set(anchor)
    try:
        with jax.ensure_compile_time_eval():
            yield
    finally:
        _OUTER_ANCHOR.reset(token)


def _anchored(anchor, tree):
    """``tree`` with its first inexact leaf multiplied by ``anchor`` (1.0).

    ``x * 1.0`` is exact for every float, -0.0 and inf included, so the
    carry keeps its values bit for bit; it now holds a tracer of the outer
    trace, so a scan over it is recorded, not evaluated.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    for i, leaf in enumerate(leaves):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            leaves[i] = leaf * anchor.astype(dtype)
            return jax.tree_util.tree_unflatten(treedef, leaves)
    return tree


def recorded_scan(f, init, xs=None, length=None, **kwargs):
    """``jax.lax.scan`` for a time-stepping loop: recorded, never run while tracing.

    Every time-stepping scan entry (uniform, graded, ADI) calls this instead
    of ``jax.lax.scan``. Inside :func:`declaration_setup` a scan whose
    operands are all constants would otherwise be evaluated at trace time:
    JAX 0.6.2 raises ``Evaluation rule for 'empty' not implemented`` (#1354),
    and 0.10.2 runs the solve uncompiled while it traces the caller's
    program. Its body would also be traced under eager constant folding.

    With JAX's switch (:func:`compile_time_switch`) the scan is traced with
    the switch off, exactly as outside the region. Without it, a carry and
    ``xs`` that hold no tracer are anchored to the outer trace
    (``x * 1.0``). Outside a compile-time region this is ``jax.lax.scan``.
    """
    switch = compile_time_switch()
    if switch is not None:
        if not switch.value:
            return jax.lax.scan(f, init, xs, length=length, **kwargs)
        with switch(False):
            return jax.lax.scan(f, init, xs, length=length, **kwargs)
    anchor = _OUTER_ANCHOR.get()
    if anchor is not None and not any(
            is_tracer(leaf) for leaf in jax.tree_util.tree_leaves((init, xs))):
        init = _anchored(anchor, init)
    return jax.lax.scan(f, init, xs, length=length, **kwargs)

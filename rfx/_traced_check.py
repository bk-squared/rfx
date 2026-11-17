"""Host checks for concrete operands and backend admission for traced ones."""

import os

import jax

from rfx.backend import backend_class, host_cpu_device

_STAGED = object()


def require_host_callbacks(feature, check_name):
    """Refuse at trace time a backend that cannot run a host callback.

    A GPU process runs callbacks on its CPU device; with JAX_PLATFORMS
    lacking ``cpu`` it has none, and the callback fails only after the solve.
    """
    backend = backend_class()
    if not backend.host_callbacks:
        raise NotImplementedError(
            f"{feature}: {check_name}: this backend ({backend.name}) cannot run "
            "a host check inside compiled code")
    if backend.name == "cpu":
        return
    try:
        host_cpu_device()
    except RuntimeError as exc:
        raise RuntimeError(
            f"{feature}: {check_name}: a host check inside compiled code runs on "
            f"this process's CPU device, and JAX has none (JAX_PLATFORMS="
            f"{os.environ.get('JAX_PLATFORMS', '')!r}); add cpu, e.g. "
            "JAX_PLATFORMS=cuda,cpu") from exc


def _host_with_concrete_leaves(check, leaves, tree):
    """Split leaves: tracers go through the callback, the rest stay as given.

    Returns the host function and the staged leaves. The host function puts
    each staged value back at its position and hands every concrete leaf to
    ``check`` as the same object the caller passed, so a traced call sees the
    concrete operands an eager call sees. The closure holds no tracer.
    """
    positions = tuple(i for i, value in enumerate(leaves)
                      if isinstance(value, jax.core.Tracer))
    held = [_STAGED if i in positions else value for i, value in enumerate(leaves)]
    staged = [leaves[i] for i in positions]

    def host(*values):
        filled = list(held)
        for i, value in zip(positions, values):
            filled[i] = value
        return check(*jax.tree.unflatten(tree, filled))

    return host, staged


def traced_check(feature, check_name, check, *operands, ordered=False,
                 result_shape=None):
    """Run concrete checks now; retain callbacks (and optional tokens) in traces.

    Inspect pytree leaves: records can contain arrays nested in dictionaries
    or material tuples. ``result_shape`` retains an IO token's data dependency.
    Only tracer leaves pass through the callback; concrete leaves reach the
    host function unchanged, with their own types and dtypes.
    """
    leaves, tree = jax.tree.flatten(operands)
    if not any(isinstance(value, jax.core.Tracer) for value in leaves):
        return check(*operands)
    require_host_callbacks(feature, check_name)
    host, staged = _host_with_concrete_leaves(check, leaves, tree)
    if result_shape is not None:
        from jax.experimental import io_callback
        return io_callback(host, result_shape, *staged, ordered=ordered)
    return jax.debug.callback(host, *staged, ordered=ordered)

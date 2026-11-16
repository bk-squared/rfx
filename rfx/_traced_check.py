"""Host checks for concrete operands and backend admission for traced ones."""

import os

import jax

from rfx.backend import backend_class, host_cpu_device


def _host_callbacks_refusal(feature, check_name):
    """Return the refusal for a backend that cannot run a host callback.

    A GPU process runs callbacks on its CPU device; with JAX_PLATFORMS
    lacking ``cpu`` it has none, and the callback fails only after the solve.
    """
    backend = backend_class()
    if not backend.host_callbacks:
        return NotImplementedError(
            f"{feature}: {check_name}: this backend ({backend.name}) cannot run "
            "a host check inside compiled code")
    if backend.name == "cpu":
        return None
    try:
        host_cpu_device()
    except RuntimeError as exc:
        refusal = RuntimeError(
            f"{feature}: {check_name}: a host check inside compiled code runs on "
            f"this process's CPU device, and JAX has none (JAX_PLATFORMS="
            f"{os.environ.get('JAX_PLATFORMS', '')!r}); add cpu, e.g. "
            "JAX_PLATFORMS=cuda,cpu")
        refusal.__cause__ = exc
        return refusal
    return None


def host_callbacks_admitted():
    """Whether this process can run a host callback inside compiled code."""
    return _host_callbacks_refusal("", "") is None


def require_host_callbacks(feature, check_name):
    """Refuse at trace time a backend that cannot run a host callback."""
    refusal = _host_callbacks_refusal(feature, check_name)
    if refusal is not None:
        raise refusal


def traced_check(feature, check_name, check, *operands, ordered=False,
                 result_shape=None):
    """Run concrete checks now; retain callbacks (and optional tokens) in traces.

    Inspect pytree leaves: records can contain arrays nested in dictionaries
    or material tuples. ``result_shape`` retains an IO token's data dependency.
    """
    if not any(isinstance(value, jax.core.Tracer)
               for value in jax.tree.leaves(operands)):
        return check(*operands)
    require_host_callbacks(feature, check_name)
    if result_shape is not None:
        from jax.experimental import io_callback
        return io_callback(check, result_shape, *operands, ordered=ordered)
    return jax.debug.callback(check, *operands, ordered=ordered)

"""Host checks for concrete operands and backend admission for traced ones."""

import jax

from rfx.backend import backend_class


def traced_check(feature, check_name, check, *operands, ordered=False,
                 result_shape=None):
    """Run concrete checks now; retain callbacks (and optional tokens) in traces.

    Inspect pytree leaves: records can contain arrays nested in dictionaries
    or material tuples. ``result_shape`` retains an IO token's data dependency.
    """
    if not any(isinstance(value, jax.core.Tracer)
               for value in jax.tree.leaves(operands)):
        return check(*operands)
    backend = backend_class()
    if not backend.host_callbacks:
        raise NotImplementedError(
            f"{feature}: {check_name}: this backend ({backend.name}) cannot run "
            "a host check inside compiled code")
    if result_shape is not None:
        from jax.experimental import io_callback
        return io_callback(check, result_shape, *operands, ordered=ordered)
    return jax.debug.callback(check, *operands, ordered=ordered)

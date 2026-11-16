"""Record a multi-device advance across the declaration setup boundary."""
from rfx.core.jax_utils import _REGION, compile_time_switch
from rfx.stepping.scan import _opaque


def recorded_sharded_call(run_fn, *args, **kwargs):
    """Call an advance over opaque operands, leaving setup only for the call.

    Like ``recorded_scan``, constant operands must neither execute the loop
    while tracing nor let the compiler specialize its arithmetic. Both the
    wrapped slab step and the graded legacy step pass all their operands
    here. Outside declaration setup this is the original call.
    """
    region = _REGION.get()
    if region is None:
        return run_fn(*args, **kwargs)
    token = _REGION.set(None)
    try:
        switch = compile_time_switch()
        if switch is not None:
            with switch(False):
                args, kwargs = _opaque((args, kwargs), None)
                return run_fn(*args, **kwargs)
        args, kwargs = _opaque((args, kwargs), region.anchor)
        return run_fn(*args, **kwargs)
    finally:
        _REGION.reset(token)

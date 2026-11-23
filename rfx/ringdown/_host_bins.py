"""Concrete frequency bins for host-side ring-down identification."""
import numpy as np


def host_bins(bins, dtype, *, caller, argument):
    """Keep pole identification and its host report on concrete frequencies."""
    import jax

    if isinstance(bins, jax.core.Tracer):
        raise jax.errors.ConcretizationTypeError(
            bins, f"{caller}(ringdown=...) identifies poles on the host and needs "
            f"{argument} as concrete values; pass a numpy array built outside "
            "the traced function")
    return np.asarray(bins, dtype)

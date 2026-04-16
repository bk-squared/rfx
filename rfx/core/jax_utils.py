"""Shared JAX utility helpers used across rfx sub-modules."""
import jax


def is_tracer(x: object) -> bool:
    """Return True if *x* is a JAX abstract tracer value.

    Use before any host-side coercion (float(), np.array(), etc.).
    """
    return isinstance(x, jax.core.Tracer)

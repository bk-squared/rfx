"""The stability judge's matrix is host arithmetic: under an outer jax.jit with concrete pole records it must return the eager value, not a staged tracer (review B round 2; main df3d97f87 was pure NumPy)."""
import jax

from rfx.model.pole_stability import largest_growth


def test_largest_growth_is_host_arithmetic_under_outer_jit():
    eager = largest_growth(1.0, 0.0, [(0.0, 0.0, 3.0)], 1.0, 0.0)
    seen = []

    @jax.jit
    def f(x):
        seen.append(largest_growth(1.0, 0.0, [(0.0, 0.0, 3.0)], 1.0, 0.0))
        return x * 2.0
    assert float(f(1.0)) == 2.0 and seen == [eager]

"""optimize(jit=True) falls back on rfx's traced-structure refusal type itself."""
import warnings

import jax.numpy as jnp
import pytest

from rfx._traced_refusal import TracedStructureRefusal
from rfx.optimize import _value_and_grad_with_fallback


class _OnlyRfxRefusal(TracedStructureRefusal):
    """Shares no JAX error base: the fallback must key on the rfx type."""


def test_fallback_keys_on_traced_structure_refusal():
    calls = []

    def objective(x):
        calls.append(None)
        if len(calls) == 1:  # the compiled trace reads the structure on the host
            raise _OnlyRfxRefusal("traced structure input")
        return jnp.sum(x ** 2)

    evaluate = _value_and_grad_with_fallback(objective, jit=True, caller="optimize")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        value, grad = evaluate(jnp.array([1.0, 2.0]))
    fallback = [r for r in records if "every iteration now recompiles" in str(r.message)]
    assert len(fallback) == 1 and "_OnlyRfxRefusal" in str(fallback[0].message)
    assert float(value) == pytest.approx(5.0)
    assert grad.tolist() == pytest.approx([2.0, 4.0])


def test_every_named_traced_refusal_is_the_rfx_type():
    from rfx.geometry.rasterize_grid import TracedShapeDimensionError
    from rfx.model.graded_ports import TracedWireLiveCellsError
    assert issubclass(TracedShapeDimensionError, TracedStructureRefusal)
    assert issubclass(TracedWireLiveCellsError, TracedStructureRefusal)

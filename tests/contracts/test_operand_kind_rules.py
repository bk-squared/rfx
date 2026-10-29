"""Axis C coverage and the shared operand query's assembly/setup interface."""
import numpy as np
import pytest

# Import each rule owner so its table is registered even in an isolated run.
from rfx.model.operands import OperandKind, OperandRecord, operands_at


@pytest.mark.parametrize("shape,trailing,expected", [
    ((), 0, ()), ((4, 5), 0, (4, 5)),
    ((4, 5, 6), 0, (2, 2, 2)),
    ((2, 4, 5, 6), 0, (2, 2, 2, 2)),
    ((2, 1, 5, 1), 0, (2, 1, 2, 1)),
    ((4, 5, 6, 3, 3), 2, (2, 2, 2, 3, 3)),
    ((1, 5, 1, 3, 3), 2, (1, 2, 1, 3, 3)),
])
def test_query_spatial_axes(shape, trailing, expected):
    array = np.arange(np.prod(shape, dtype=int)).reshape(shape)
    record = OperandRecord("operand", frozenset({OperandKind.EPS}), array,
                           "identity" if trailing else 1, "material", None, None, trailing, None)
    slices = (slice(1, 3), slice(2, 4), slice(3, 5))
    ((found, cut),) = operands_at((record,), lambda _: slices)
    assert found is record
    assert cut.shape == expected
    if shape == (4, 5, 6):
        np.testing.assert_array_equal(cut, array[1:3, 2:4, 3:5])
    if len(shape) - trailing < 3:
        assert cut is array
    assert operands_at((record,), lambda _: None) == ()

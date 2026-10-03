"""Message-independent snapshot payloads and numeric comparisons."""
import re
from numbers import Real


# Fidelity digests round to 12 significant digits (up to 5e-12 relative).
# Allow two rounding quanta; near zero use the existing rasterization test's
# 1e-9 um bound (1e-15 m). Other units get a 1e-12 absolute floor.
RTOL = 1e-11
UM_ATOL = 1e-9
# Message tokens are already presentation-rounded. This permits float64 and
# digest rounding noise, while detecting changes above one part per billion;
# the unit-independent 1e-12 floor also protects near-zero reported values.
MESSAGE_RTOL = 1e-9
MESSAGE_ATOL = 1e-12
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
# numpy >= 2 prints scalars as ``np.float64(0.005)``, numpy 1 as ``0.005``;
# strip the wrapper first so the type's digits are not read as a number.
_NUMPY_SCALAR = re.compile(r"np\.(?:float|int|uint|complex|bool_?)\d*\(([^()]*)\)")


def message_numbers(message):
    """Numeric tokens in emission order, including signed/scientific values."""
    return [float(token) for token in _NUMBER.findall(_NUMPY_SCALAR.sub(r"\1", message))]


def without_prose(value):
    """Remove presentation fields, retaining identifiers and numeric data."""
    if isinstance(value, dict):
        result = {k: without_prose(v) for k, v in value.items()
                  if k not in {"message", "detail", "remedy", "note", "realization"}}
        if "message" in value:
            result["numbers"] = message_numbers(value["message"])
        return result
    if isinstance(value, (list, tuple)):
        return [without_prose(v) for v in value]
    return value


def assert_structured_close(actual, expected, path="snapshot", *, field=None):
    """Exact schema/identifiers/counts; tolerant real-valued measurements."""
    import pytest
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_structured_close(actual[key], expected[key], f"{path}.{key}", field=key)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), path
        assert len(actual) == len(expected), path
        for i, (have, want) in enumerate(zip(actual, expected, strict=True)):
            assert_structured_close(have, want, f"{path}[{i}]", field=field)
    elif isinstance(expected, float):
        assert isinstance(actual, Real) and not isinstance(actual, bool), path
        atol = UM_ATOL if field is not None and field.endswith("_um") else 1e-12
        rtol = RTOL
        if field == "numbers":
            rtol, atol = MESSAGE_RTOL, MESSAGE_ATOL
        assert actual == pytest.approx(expected, rel=rtol, abs=atol), path
    else:
        assert actual == expected, path

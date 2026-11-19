"""Message-independent snapshot payloads and numeric comparisons.
The unit-independent 1e-12 floor protects near-zero reported message values."""
import re
from numbers import Real


RTOL = 1e-11
UM_ATOL = 1e-9
OTHER_UNITS_ATOL = 1e-12
MESSAGE_RTOL = 1e-9
MESSAGE_ATOL = 1e-12
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_NUMPY_SCALAR = re.compile(r"np\.(?:float|int|uint|complex|bool_?)\d*\(([^()]*)\)")


def message_numbers(message):
    """Numeric tokens in emission order, including signed/scientific values."""
    return [float(token) for token in _NUMBER.findall(_NUMPY_SCALAR.sub(r"\1", message))]


def without_prose(value):
    """Remove presentation fields, retaining identifiers and numeric data."""
    if isinstance(value, dict):
        ignored = {"message", "detail", "remedy", "note", "realization"}
        if {"code", "message", "scope", "subjects"} <= value.keys():
            ignored.update(("scope", "subjects", "region"))
        result = {k: without_prose(v) for k, v in value.items()
                  if k not in ignored}
        if "message" in value:
            result["numbers"] = message_numbers(value["message"])
        return result
    if isinstance(value, (list, tuple)):
        return [without_prose(v) for v in value]
    return value


def assert_structured_close(actual, expected, path="snapshot", *, field=None):
    """Exact schema/identifiers/counts; tolerant real-valued measurements.
    Allow two rounding quanta for fidelity digests rounded to 12 significant digits."""
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
        atol = UM_ATOL if field is not None and field.endswith("_um") else OTHER_UNITS_ATOL
        rtol = RTOL
        if field == "numbers":
            rtol, atol = MESSAGE_RTOL, MESSAGE_ATOL
        assert actual == pytest.approx(expected, rel=rtol, abs=atol), path
    else:
        assert actual == expected, path

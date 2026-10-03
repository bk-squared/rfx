"""Message-independent snapshot payloads and numeric comparisons."""
from numbers import Real



# Fidelity digests round to 12 significant digits (up to 5e-12 relative).
# Allow two rounding quanta; near zero use the existing rasterization test's
# 1e-9 um bound (1e-15 m). Other units get a 1e-12 absolute floor.
RTOL = 1e-11
UM_ATOL = 1e-9


def without_prose(value):
    """Remove presentation fields, retaining identifiers and numeric data."""
    if isinstance(value, dict):
        return {k: without_prose(v) for k, v in value.items()
                if k not in {"message", "detail", "remedy", "note", "realization"}}
    if isinstance(value, (list, tuple)):
        return [without_prose(v) for v in value]
    return value


def assert_structured_close(actual, expected, path="snapshot"):
    """Exact schema/identifiers/counts; tolerant real-valued measurements."""
    import pytest
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_structured_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), path
        assert len(actual) == len(expected), path
        for i, (have, want) in enumerate(zip(actual, expected, strict=True)):
            assert_structured_close(have, want, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert isinstance(actual, Real) and not isinstance(actual, bool), path
        atol = UM_ATOL if "_um" in path else 1e-12
        assert actual == pytest.approx(expected, rel=RTOL, abs=atol), path
    else:
        assert actual == expected, path

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import rfx.simulation as simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/showcase"))
import _lens_filter_common as common


def test_lens_fd_dtype(monkeypatch):
    def forbidden_run(*args, **kwargs):
        raise AssertionError("dtype observation must use the model response")

    monkeypatch.setattr(simulation, "run", forbidden_run)
    calls = []
    steps = []
    accumulator_dtype = np.complex128

    def run(e):
        calls.append(e)
        return SimpleNamespace(ntff_data=SimpleNamespace(
            x_lo=np.zeros(1, dtype=accumulator_dtype)))

    def response_fn(n):
        steps.append(n)
        return SimpleNamespace(run=run)

    model = SimpleNamespace(response_fn=response_fn)
    e = np.array([2.0], dtype=np.float32)
    seen = common.observe_field_dtypes(model, e, 7, "float64")
    assert seen == [{"requested": "float64", "observed": "complex128"}]
    assert steps == [7]
    assert len(calls) == 1
    assert calls[0].dtype == np.float64
    np.testing.assert_array_equal(calls[0], [2.0])
    common.require_field_dtypes(seen, "float64")

    accumulator_dtype = np.complex64
    seen = common.observe_field_dtypes(model, e, 7, "float64")
    assert seen == [{"requested": "float64", "observed": "complex64"}]
    assert steps == [7, 7]
    assert len(calls) == 2
    assert calls[1].dtype == np.float64
    with pytest.raises(RuntimeError, match="float64 FD refused: observed field dtypes") as exc:
        common.require_field_dtypes(seen, "float64")
    assert str(seen) in str(exc.value)

"""Public method selector and its 2.x compatibility spelling."""

import inspect
import warnings

import numpy as np
import pytest

from tests._pec_short_advisory_fixture import build


METHOD = "compute_waveguide_s_matrix"


def _simulation():
    # One-period uniform waveguide: the dispatch suite's cheapest real solve.
    return build(np.linspace(4e9, 6e9, 6), dx=2e-3, cpml=8,
                 boundary={"x": "cpml", "y": "pec", "z": "pec"})


def _record_all_except_dependency_deprecation():
    warnings.simplefilter("always")
    # Lazy solver import emits this independent JAX API migration warning.
    warnings.filterwarnings("ignore", message=r"jax\.experimental\.shard_map is deprecated.*",
                            category=DeprecationWarning)


def _deprecations(record):
    # Only the alias's own warning: another library's deprecation on the solve path is not counted.
    return [w for w in record if issubclass(w.category, DeprecationWarning) and "lane=" in str(w.message)]


def test_real_solve_alias_is_byte_identical_and_warns_once():
    with warnings.catch_warnings(record=True) as old_warnings:
        _record_all_except_dependency_deprecation()
        call_line = inspect.currentframe().f_lineno + 1
        old = _simulation().compute_s_matrix(lane=METHOD, normalize=False, num_periods=1)
    deprecated = _deprecations(old_warnings)
    assert len(deprecated) == 1
    assert "method=" in str(deprecated[0].message)
    assert deprecated[0].filename == __file__
    assert deprecated[0].lineno == call_line
    with warnings.catch_warnings(record=True) as new_warnings:
        _record_all_except_dependency_deprecation()
        new = _simulation().compute_s_matrix(method=METHOD, normalize=False, num_periods=1)
    assert not _deprecations(new_warnings)
    for field in ("s_params", "freqs"):
        before, after = np.asarray(getattr(old, field)), np.asarray(getattr(new, field))
        assert before.size > 0
        assert np.isfinite(before).all()
        assert before.shape == after.shape
        assert before.dtype == after.dtype
        assert before.tobytes() == after.tobytes()
    assert old.port_names == new.port_names


def test_selector_alias_warns_at_user_call_once():
    sim = _simulation()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call_line = inspect.currentframe().f_lineno + 1
        assert sim.s_matrix_lane(lane=METHOD) == METHOD
    deprecated = _deprecations(record)
    assert len(deprecated) == 1
    assert "method=" in str(deprecated[0].message)
    assert deprecated[0].filename == __file__
    assert deprecated[0].lineno == call_line
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert sim.s_matrix_lane(method=METHOD) == METHOD
    assert not _deprecations(record)


@pytest.mark.parametrize("entry", ["compute_s_matrix", "s_matrix_lane"])
@pytest.mark.parametrize("alias", [METHOD, "compute_coaxial_two_port"])
def test_both_spellings_are_rejected(entry, alias):
    with pytest.raises(TypeError, match=r"method=.*lane="):
        getattr(_simulation(), entry)(method=METHOD, lane=alias)


@pytest.mark.parametrize("entry", ["compute_s_matrix", "s_matrix_lane"])
def test_invalid_method_names_method(entry):
    with pytest.raises(ValueError, match="method='invalid' is not a method"):
        getattr(_simulation(), entry)(method="invalid")


@pytest.mark.parametrize("entry", ["compute_s_matrix", "s_matrix_lane"])
def test_alias_value_is_not_ignored(entry):
    # Auto-selection would accept this registration; the explicit mismatch must fail.
    with pytest.warns(DeprecationWarning, match="method=") as record:
        with pytest.raises(ValueError, match="method=.*contradicts the registrations"):
            getattr(_simulation(), entry)(lane="compute_coaxial_two_port")
    assert len(_deprecations(record)) == 1

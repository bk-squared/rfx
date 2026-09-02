"""Tests for RF port-observable validation helpers.

These are pure metadata / algebra tests.  They do not run FDTD and do not
claim additional solver physics validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import (
    PortSMatrixObservable,
    PortValidationIssue,
    PortValidationReport,
    assert_port_smatrix_valid,
    normalize_port_smatrix,
    validate_port_smatrix,
)
from rfx.api import MSLSMatrixResult, Result, WaveguideSMatrixResult


def _passive_reciprocal_smatrix() -> tuple[np.ndarray, np.ndarray]:
    freqs = np.array([1.0e9, 2.0e9, 3.0e9])
    s = np.zeros((2, 2, freqs.size), dtype=np.complex128)
    s[0, 0, :] = [0.10, 0.12, 0.08]
    s[1, 1, :] = [0.09, 0.11, 0.07]
    s[0, 1, :] = [0.70, 0.65, 0.60]
    s[1, 0, :] = s[0, 1, :]
    return s, freqs


def test_validation_api_importable_from_top_level():
    assert PortSMatrixObservable.__name__ == "PortSMatrixObservable"
    assert PortValidationIssue.__name__ == "PortValidationIssue"
    assert PortValidationReport.__name__ == "PortValidationReport"


def test_validate_passive_reciprocal_full_matrix_passes():
    s, freqs = _passive_reciprocal_smatrix()

    report = validate_port_smatrix(
        s_params=s,
        freqs=freqs,
        port_names=("left", "right"),
        check_reciprocity=True,
    )

    assert report.ok, report.summary()
    assert report.n_ports == 2
    assert report.n_freqs == 3
    assert report.port_names == ("left", "right")
    assert report.metrics["max_column_power"] < 1.0
    assert report.metrics["max_reciprocity_abs_diff"] == pytest.approx(0.0)


def test_passivity_violation_reports_column_power_diagnostic():
    freqs = np.array([1.0e9, 2.0e9])
    s = np.zeros((2, 2, 2), dtype=np.complex128)
    s[0, 0, :] = [1.05, 0.8]
    s[1, 0, :] = [0.25, 0.1]

    report = validate_port_smatrix(
        s_params=s,
        freqs=freqs,
        port_names=("p1", "p2"),
        passivity_tol=0.0,
    )

    assert not report.ok
    issues = report.by_code("passivity_violation")
    assert len(issues) == 1
    assert issues[0].port_indices == (0,)
    assert issues[0].frequency_index == 0
    assert issues[0].value > 1.0


def test_reciprocity_violation_reports_port_pair_diagnostic():
    s, freqs = _passive_reciprocal_smatrix()
    s[0, 1, 1] = 0.1
    s[1, 0, 1] = 0.7

    report = validate_port_smatrix(
        s_params=s,
        freqs=freqs,
        port_names=("p1", "p2"),
        check_reciprocity=True,
        reciprocity_atol=1e-12,
        reciprocity_rtol=0.0,
    )

    assert not report.ok
    issues = report.by_code("reciprocity_violation")
    assert len(issues) == 1
    assert set(issues[0].port_indices or ()) == {0, 1}
    assert issues[0].frequency_index == 1


def test_invalid_shape_reports_schema_diagnostic_without_raising():
    report = validate_port_smatrix(
        s_params=np.zeros((2, 3, 4, 5)),
        freqs=np.array([1.0e9, 2.0e9]),
    )

    assert not report.ok
    assert report.by_code("invalid_schema")
    assert "1D, 2D, or 3D" in report.summary()


def test_invalid_frequency_metadata_reports_diagnostics():
    s = np.zeros((1, 1, 3), dtype=np.complex128)
    freqs = np.array([2.0e9, np.nan, 1.0e9])

    report = validate_port_smatrix(s_params=s, freqs=freqs)

    assert not report.ok
    assert report.by_code("nonfinite_freqs")

    report = validate_port_smatrix(
        s_params=s,
        freqs=np.array([2.0e9, 2.0e9, 1.0e9]),
    )
    assert report.by_code("nonmonotonic_freqs")

    report = validate_port_smatrix(
        s_params=s,
        freqs=np.array([0.0, 1.0e9, 2.0e9]),
    )
    assert report.by_code("nonpositive_freqs")


def test_empty_frequency_axis_reports_diagnostic_without_invariant_crash():
    report = validate_port_smatrix(
        s_params=np.zeros((1, 1, 0), dtype=np.complex128),
        freqs=np.array([]),
    )

    assert not report.ok
    assert report.by_code("empty_freqs")
    assert "max_column_power" not in report.metrics


def test_nonfinite_sparams_reports_diagnostic_and_skips_invariants():
    s = np.zeros((1, 1, 2), dtype=np.complex128)
    s[0, 0, 1] = np.nan

    report = validate_port_smatrix(s_params=s, freqs=np.array([1.0e9, 2.0e9]))

    assert not report.ok
    assert report.by_code("nonfinite_sparams")
    assert "max_column_power" not in report.metrics


def test_assert_helper_raises_on_failure_and_returns_report_on_success():
    s, freqs = _passive_reciprocal_smatrix()
    report = assert_port_smatrix_valid(s_params=s, freqs=freqs)
    assert report.ok

    bad = np.array([1.2 + 0j])
    with pytest.raises(AssertionError, match="passivity_violation"):
        assert_port_smatrix_valid(s_params=bad, freqs=np.array([1.0e9]))


def test_one_dimensional_s11_normalizes_to_single_port_observable():
    obs = normalize_port_smatrix(
        s_params=np.array([0.1, 0.2]),
        freqs=np.array([1.0e9, 2.0e9]),
    )

    assert obs.s_params.shape == (1, 1, 2)
    np.testing.assert_allclose(obs.s_params[0, 0, :], [0.1, 0.2])
    assert obs.port_names == ("port0",)


def test_reflection_stack_normalizes_to_diagonal_smatrix():
    obs = normalize_port_smatrix(
        s_params=np.array([[0.1, 0.2], [0.3, 0.4]]),
        freqs=np.array([1.0e9, 2.0e9]),
        port_names=("a", "b"),
    )

    assert obs.s_params.shape == (2, 2, 2)
    np.testing.assert_allclose(obs.s_params[0, 0, :], [0.1, 0.2])
    np.testing.assert_allclose(obs.s_params[1, 1, :], [0.3, 0.4])
    np.testing.assert_allclose(obs.s_params[0, 1, :], [0.0, 0.0])


def test_normalizes_existing_result_structures_without_fdtd():
    s, freqs = _passive_reciprocal_smatrix()

    run_result = Result(
        state=None,
        time_series=np.zeros((0, 0)),
        s_params=s,
        freqs=freqs,
    )
    run_obs = normalize_port_smatrix(run_result)
    assert run_obs.s_params.shape == s.shape
    assert run_obs.port_names == ("port0", "port1")

    wg_result = WaveguideSMatrixResult(
        s_params=s,
        freqs=freqs,
        port_names=("left", "right"),
        port_directions=("+x", "-x"),
        reference_planes=np.array([0.01, 0.09]),
    )
    wg_obs = normalize_port_smatrix(wg_result)
    assert wg_obs.port_names == ("left", "right")
    assert validate_port_smatrix(wg_result, check_reciprocity=True).ok

    msl_result = MSLSMatrixResult(
        S=s,
        freqs=freqs,
        Z0=np.full((2, freqs.size), 50.0 + 0.0j),
        beta=np.ones(freqs.size, dtype=np.complex128),
        port_names=("in", "out"),
    )
    msl_obs = normalize_port_smatrix(msl_result)
    assert msl_obs.port_names == ("in", "out")
    assert validate_port_smatrix(msl_result, check_reciprocity=True).ok


def test_port_name_count_and_uniqueness_are_checked():
    s, freqs = _passive_reciprocal_smatrix()

    count_report = validate_port_smatrix(s_params=s, freqs=freqs, port_names=("only",))
    assert count_report.by_code("port_name_count")

    dup_report = validate_port_smatrix(s_params=s, freqs=freqs, port_names=("p", "p"))
    assert dup_report.by_code("duplicate_port_names")

"""Regression coverage for issues #337 and #342 passivity wiring."""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.api._spec import MSLSMatrixResult
from rfx.api._sparams import _warn_if_nonpassive_smatrix
from rfx.sources.sources import GaussianPulse


def _result(s_params):
    return SimpleNamespace(
        s_params=np.asarray(s_params, dtype=complex),
        freqs=np.array([1.0e9, 2.0e9, 3.0e9]),
        port_names=("port0",),
    )


def test_per_frequency_amplitude_advisory_reports_worst_bin():
    s = np.array([[[0.2, 1.36, 0.4]]])
    with pytest.warns(UserWarning) as recorded:
        _warn_if_nonpassive_smatrix(
            _result(s),
            extractor="compute_waveguide_s_matrix",
            passivity_tol=2.0,
        )
    messages = [str(item.message) for item in recorded]
    amplitude = [message for message in messages if "frequency index" in message]
    assert len(amplitude) == 1
    assert "frequency index 1" in amplitude[0]
    assert "1.36" in amplitude[0]
    assert "issue #337" in amplitude[0]


def test_pec_short_like_1_03_column_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_nonpassive_smatrix(
            _result([[[1.03, 1.02, 1.01]]]),
            extractor="compute_waveguide_s_matrix",
            passivity_tol=2.0,
        )


def test_msl_result_uses_shared_epilogue_and_warns():
    result = MSLSMatrixResult(
        S=np.array([[[0.2, 1.36, 0.4]]]),
        freqs=np.array([1.0e9, 2.0e9, 3.0e9]),
        Z0=np.full((1, 3), 50.0),
        beta=np.ones(3),
        port_names=("msl0",),
    )
    with pytest.warns(UserWarning, match="compute_msl_s_matrix.*1.36"):
        _warn_if_nonpassive_smatrix(
            result,
            extractor="compute_msl_s_matrix",
            passivity_tol=2.0,
        )


def test_run_uniform_epilogue_warns_for_patched_extractor(monkeypatch):
    sim = Simulation(
        freq_max=3.0e9,
        domain=(0.004, 0.004, 0.004),
        dx=1.0e-3,
        boundary="pec",
    )
    sim.add_port(
        position=(0.002, 0.002, 0.002),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=2.0e9, bandwidth=0.5),
    )
    fake_result = SimpleNamespace(
        s_params=np.array([0.2, 1.36, 0.4]),
        freqs=np.array([1.0e9, 2.0e9, 3.0e9]),
    )

    monkeypatch.setattr("rfx.runners.uniform.run_uniform", lambda *a, **k: fake_result)
    monkeypatch.setattr("rfx.api._execute._warn_if_nonfinite_result", lambda *a, **k: None)

    with pytest.warns(UserWarning, match=r"run\(compute_s_params=True\).*non-passive"):
        result = sim.run(
            n_steps=1,
            compute_s_params=True,
            s_param_freqs=fake_result.freqs,
            skip_preflight=True,
        )
    assert result is fake_result

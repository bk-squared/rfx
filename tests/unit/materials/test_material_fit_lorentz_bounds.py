"""Lorentz fits on flat-loss tables and resonances near the table's top."""

import numpy as np
import pytest

from rfx.material_fit import fit_lorentz
from rfx.model.pole_stability import largest_growth


def _lorentz_sum(freqs, eps_inf, poles):
    omega = 2 * np.pi * freqs
    return eps_inf + sum(
        p.kappa / (p.omega_0 ** 2 - omega ** 2 + 2j * omega * p.delta)
        for p in poles)


def _parameters(result):
    poles = sorted(result.poles, key=lambda p: (p.omega_0, 2 * p.delta))
    return np.array([result.eps_inf] + [
        value for p in poles
        for value in (p.omega_0, 2 * p.delta, p.kappa / p.omega_0 ** 2)])


@pytest.fixture(scope="module", params=[2, 3])
def fr4_fit(request):
    freqs = np.geomspace(1e9, 40e9, 100)
    results = [fit_lorentz(freqs, np.full(freqs.shape, target), request.param)
               for target in (4.4 - 0.088j, 4.4 - 1j * 4.4 * 0.02)]
    return request.param, results


def test_fr4_floor_and_span(fr4_fit):
    _, results = fr4_fit
    for result in results:
        assert result.eps_inf >= 1.0
        assert all(2 * np.pi * 1e9 * (1 - 1e-12) <= p.omega_0
                   <= 2 * np.pi * 40e9 * (1 + 1e-12) for p in result.poles)


def test_fr4_loss_tangent(fr4_fit):
    n_poles, results = fr4_fit
    inner = np.geomspace(2e9, 20e9, 10001)
    for result in results:
        fitted = _lorentz_sum(inner, result.eps_inf, result.poles)
        tan_delta = -fitted.imag / fitted.real
        error = np.max(np.abs(tan_delta / 0.02 - 1))
        assert error <= {2: 0.12, 3: 0.035}[n_poles], error


def test_fr4_growth(fr4_fit):
    _, results = fr4_fit
    budget = 3 * 299792458.0 ** 2 / 1e-3 ** 2
    dt = 0.99 / np.sqrt(budget)
    for result in results:
        growth = largest_growth(result.eps_inf, 0, result.poles, dt, 4 * dt ** 2 * budget)
        assert growth <= 1 + 1e-9, growth


def test_fr4_reproducibility(fr4_fit):
    _, results = fr4_fit
    a, b = map(_parameters, results)
    scale = np.maximum(np.abs(a), np.abs(b))
    relative = np.divide(np.abs(a - b), scale, out=np.zeros_like(scale), where=scale != 0)
    assert np.all(relative <= 0.01), relative


@pytest.mark.parametrize("resonance", [7.94e9, 8.5e9])
def test_top_of_band_resonance(resonance):
    freqs = np.linspace(1e9, 10e9, 60)
    omega = 2 * np.pi * freqs
    omega_0 = 2 * np.pi * resonance
    gamma = 2 * np.pi * 0.5e9
    measured = 1.5 + 2.0 * omega_0 ** 2 / (omega_0 ** 2 - omega ** 2 + 1j * omega * gamma)
    result = fit_lorentz(freqs, measured)
    assert abs(result.poles[0].omega_0 / omega_0 - 1) < 0.05
    fitted = _lorentz_sum(freqs, result.eps_inf, result.poles)
    expected_error = np.sqrt(np.mean(np.abs(fitted - measured) ** 2)) / np.sqrt(
        np.mean(np.abs(measured) ** 2))
    assert result.fit_error == pytest.approx(expected_error, rel=1e-12, abs=1e-15)
    assert result.fit_error < 0.05


@pytest.mark.parametrize("freqs", [[2e9], [2e9, 2e9], [0.1, 0.5, 1.0]])
def test_no_frequency_span_refused(freqs):
    with pytest.raises(ValueError, match="frequency span"):
        fit_lorentz(freqs, np.full(len(freqs), 4.4 - 0.088j))

"""Bounded Debye fits on constant-loss tables and placed-pole contracts."""

import warnings

import numpy as np
import pytest

from rfx.errors import AccuracyWarning
from rfx.material_fit import fit_debye


@pytest.mark.parametrize("eps_real,tan_delta", [(1.1, 0.1), (4.4, 0.02)])
@pytest.mark.parametrize("n_poles", [2, 3])
@pytest.mark.parametrize("mode", ["free", "placed"])
def test_constant_loss_floor_and_span(eps_real, tan_delta, n_poles, mode):
    freqs = np.geomspace(1e9, 40e9, 100)
    measured = np.full(freqs.shape, eps_real - 1j * eps_real * tan_delta)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AccuracyWarning)
        result = fit_debye(freqs, measured, n_poles,
                           **({} if mode == "free" else {"tau": mode}))
    assert result.eps_inf >= 1.0
    tau_lo, tau_hi = 1 / (2 * np.pi * 40e9), 1 / (2 * np.pi * 1e9)
    assert all(tau_lo * (1 - 1e-12) <= p.tau <= tau_hi * (1 + 1e-12)
               for p in result.poles)
    accuracy = [w for w in caught if issubclass(w.category, AccuracyWarning)]
    if eps_real == 1.1 and mode == "free":
        assert result.eps_inf == 1.0
        assert len(accuracy) == 1
        assert "eps_inf=1 is at the floor 1" in str(accuracy[0].message)
    if eps_real == 4.4 and n_poles == 3:
        inner = np.geomspace(2e9, 20e9, 10001)
        fitted = result.eps_inf + sum(
            p.delta_eps / (1 + 2j * np.pi * inner * p.tau) for p in result.poles)
        fitted_tan_delta = -fitted.imag / fitted.real
        assert np.max(np.abs(fitted_tan_delta / tan_delta - 1)) <= 0.025
    fitted_table = result.eps_inf + sum(
        p.delta_eps / (1 + 2j * np.pi * freqs * p.tau) for p in result.poles)
    expected_error = np.sqrt(np.mean(np.abs(fitted_table - measured) ** 2)) / np.sqrt(
        np.mean(np.abs(measured) ** 2))
    assert result.fit_error == pytest.approx(expected_error, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("mode", ["free", "placed"])
def test_fixed_subunit_permittivity_warns_once(mode):
    freqs = np.geomspace(1e9, 40e9, 100)
    measured = 0.8 + 0.3 / (1 + 2j * np.pi * freqs * 2e-11)
    with pytest.warns(AccuracyWarning, match="sqrt\\(eps_inf\\).*vacuum Courant") as caught:
        result = fit_debye(freqs, measured, eps_inf=0.8, tau=mode)
    assert result.eps_inf == 0.8
    assert len(caught) == 1


def test_placed_single_pole_and_zero_strength_warning():
    freqs = np.geomspace(1e9, 40e9, 100)
    centre = 1 / (2 * np.pi * np.sqrt(1e9 * 40e9))
    measured = 2.0 + 1.5 / (1 + 2j * np.pi * freqs * centre)
    result = fit_debye(freqs, measured, eps_inf=2.0, tau="placed")
    assert result.poles[0].tau == pytest.approx(centre, rel=1e-12)
    assert result.poles[0].delta_eps == pytest.approx(1.5, rel=1e-12)
    with pytest.warns(AccuracyWarning, match="pole 1.*fewer poles") as caught:
        result = fit_debye(freqs, np.full(freqs.shape, 0.9 + 0j), tau="placed")
    assert len(caught) == 1
    assert "eps_inf=1 is at the floor 1" in str(caught[0].message)
    assert result.poles[0].delta_eps == 0


@pytest.mark.parametrize("tau", ["invalid", None, [1e-11], np.array([1e-11])])
def test_invalid_tau(tau):
    with pytest.raises(ValueError, match='"free" or "placed"'):
        fit_debye(np.array([1e9, 2e9]), np.array([2 - 0.1j, 2 - 0.1j]), tau=tau)

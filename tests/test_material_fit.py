"""Tests for frequency-dependent material import and Debye/Lorentz fitting.

Validates:
1. CSV loading (complex permittivity and tan-delta formats)
2. Single and multi-pole Debye fitting on synthetic data
3. Lorentz fitting on synthetic resonance data
4. Loss-tangent input handling
5. Plot output
"""

import io
import numpy as np
import pytest

from rfx.material_fit import (
    load_material_csv,
    fit_debye,
    fit_lorentz,
    eval_debye,
    eval_lorentz,
    plot_material_fit,
    DebyeFitResult,
    LorentzFitResult,
)
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_debye_data(freqs, eps_inf, poles):
    """Generate synthetic Debye permittivity data."""
    omega = 2.0 * np.pi * freqs
    eps = np.full_like(omega, eps_inf, dtype=complex)
    for delta_eps, tau in poles:
        eps = eps + delta_eps / (1.0 + 1j * omega * tau)
    return eps


def _make_lorentz_data(freqs, eps_inf, poles):
    """Generate synthetic Lorentz permittivity data."""
    omega = 2.0 * np.pi * freqs
    eps = np.full_like(omega, eps_inf, dtype=complex)
    for delta_eps, omega_0, gamma in poles:
        eps = eps + (delta_eps * omega_0 ** 2 /
                     (omega_0 ** 2 - omega ** 2 + 1j * omega * gamma))
    return eps


# ---------------------------------------------------------------------------
# test_load_csv
# ---------------------------------------------------------------------------

def test_load_csv():
    """Load complex-permittivity CSV with inline data."""
    csv_text = (
        "freq_hz,eps_r,eps_i\n"
        "1e9,4.0,0.1\n"
        "5e9,3.8,0.15\n"
        "10e9,3.5,0.2\n"
    )
    buf = io.StringIO(csv_text)
    freqs, eps = load_material_csv(buf)

    assert len(freqs) == 3
    assert freqs[0] == pytest.approx(1e9)
    assert freqs[2] == pytest.approx(10e9)
    # eps = eps_r - j*eps_i
    assert np.real(eps[0]) == pytest.approx(4.0)
    assert -np.imag(eps[0]) == pytest.approx(0.1)
    assert np.real(eps[2]) == pytest.approx(3.5)


def test_load_csv_tan_delta():
    """Load CSV with loss-tangent column."""
    csv_text = (
        "freq_hz,eps_r,tan_d\n"
        "1e9,4.0,0.02\n"
        "5e9,3.8,0.03\n"
    )
    buf = io.StringIO(csv_text)
    freqs, eps = load_material_csv(buf, tan_delta_col="tan_d")

    assert len(freqs) == 2
    # Imaginary part = eps_r * tan_delta
    expected_imag_0 = 4.0 * 0.02
    assert -np.imag(eps[0]) == pytest.approx(expected_imag_0, rel=1e-10)
    expected_imag_1 = 3.8 * 0.03
    assert -np.imag(eps[1]) == pytest.approx(expected_imag_1, rel=1e-10)


# ---------------------------------------------------------------------------
# test_fit_single_debye_pole
# ---------------------------------------------------------------------------

def test_fit_single_debye_pole():
    """Fit a single Debye pole to synthetic data; expect < 5% relative error."""
    freqs = np.logspace(8, 11, 50)  # 100 MHz to 100 GHz
    eps_inf_true = 2.5
    delta_eps_true = 1.5
    tau_true = 5e-11  # 50 ps

    eps_meas = _make_debye_data(freqs, eps_inf_true,
                                [(delta_eps_true, tau_true)])

    result = fit_debye(freqs, eps_meas, n_poles=1)

    assert isinstance(result, DebyeFitResult)
    assert len(result.poles) == 1
    assert result.fit_error < 0.05, (
        f"Fit error {result.fit_error:.4f} exceeds 5% threshold"
    )

    # Check recovered parameters are close
    assert result.eps_inf == pytest.approx(eps_inf_true, rel=0.1)
    assert result.poles[0].delta_eps == pytest.approx(delta_eps_true, rel=0.15)
    assert result.poles[0].tau == pytest.approx(tau_true, rel=0.15)

    # Verify eval_debye reproduces the fit
    eps_eval = eval_debye(freqs, result.eps_inf, result.poles)
    rms = np.sqrt(np.mean(np.abs(eps_eval - eps_meas) ** 2))
    norm = np.sqrt(np.mean(np.abs(eps_meas) ** 2))
    assert rms / norm < 0.05


# ---------------------------------------------------------------------------
# test_fit_two_debye_poles
# ---------------------------------------------------------------------------

def test_fit_two_debye_poles():
    """Fit two Debye poles to synthetic two-pole data."""
    freqs = np.logspace(7, 12, 80)
    eps_inf_true = 2.0
    poles_true = [(1.0, 1e-10), (2.0, 1e-8)]

    eps_meas = _make_debye_data(freqs, eps_inf_true, poles_true)

    result = fit_debye(freqs, eps_meas, n_poles=2)

    assert len(result.poles) == 2
    assert result.fit_error < 0.05, (
        f"Two-pole fit error {result.fit_error:.4f} exceeds 5% threshold"
    )

    # Verify the fitted model matches measurement
    eps_fitted = eval_debye(freqs, result.eps_inf, result.poles)
    rel_err = np.abs(eps_fitted - eps_meas) / np.abs(eps_meas)
    assert np.max(rel_err) < 0.1, (
        f"Max pointwise relative error {np.max(rel_err):.4f} too large"
    )


# ---------------------------------------------------------------------------
# test_fit_lorentz
# ---------------------------------------------------------------------------

def test_fit_lorentz():
    """Fit a single Lorentz pole to synthetic resonance data; < 5% error."""
    f_center = 5e9  # 5 GHz resonance
    freqs = np.linspace(1e9, 10e9, 60)

    eps_inf_true = 1.5
    delta_eps_true = 2.0
    omega_0_true = 2 * np.pi * f_center
    gamma_true = 2 * np.pi * 0.5e9  # 500 MHz damping

    eps_meas = _make_lorentz_data(
        freqs, eps_inf_true,
        [(delta_eps_true, omega_0_true, gamma_true)]
    )

    result = fit_lorentz(freqs, eps_meas, n_poles=1)

    assert isinstance(result, LorentzFitResult)
    assert len(result.poles) == 1
    assert result.fit_error < 0.05, (
        f"Lorentz fit error {result.fit_error:.4f} exceeds 5% threshold"
    )

    # Verify eval_lorentz reproduces the fit
    eps_eval = eval_lorentz(freqs, result.eps_inf, result.poles)
    rms = np.sqrt(np.mean(np.abs(eps_eval - eps_meas) ** 2))
    norm = np.sqrt(np.mean(np.abs(eps_meas) ** 2))
    assert rms / norm < 0.05


# ---------------------------------------------------------------------------
# test_fit_tan_delta_input
# ---------------------------------------------------------------------------

def test_fit_tan_delta_input():
    """End-to-end: load tan-delta CSV then fit Debye model."""
    # Generate synthetic Debye material
    freqs_true = np.logspace(9, 10.5, 30)
    eps_inf_true = 3.0
    delta_eps_true = 1.0
    tau_true = 2e-11

    eps_true = _make_debye_data(freqs_true, eps_inf_true,
                                [(delta_eps_true, tau_true)])

    # Build CSV with tan_delta = eps''/eps'
    eps_real = np.real(eps_true)
    eps_imag = -np.imag(eps_true)  # positive loss
    tan_delta = eps_imag / eps_real

    lines = ["freq_hz,eps_r,tan_d"]
    for f, er, td in zip(freqs_true, eps_real, tan_delta):
        lines.append(f"{f:.6e},{er:.8f},{td:.8f}")
    csv_text = "\n".join(lines) + "\n"

    buf = io.StringIO(csv_text)
    freqs, eps_loaded = load_material_csv(buf, tan_delta_col="tan_d")

    # Loaded data should match original
    np.testing.assert_allclose(np.real(eps_loaded), eps_real, rtol=1e-6)
    np.testing.assert_allclose(-np.imag(eps_loaded), eps_imag, rtol=1e-6)

    # Fit
    result = fit_debye(freqs, eps_loaded, n_poles=1)
    assert result.fit_error < 0.05


# ---------------------------------------------------------------------------
# test_plot_creates_figure
# ---------------------------------------------------------------------------

def test_plot_creates_figure():
    """plot_material_fit returns a matplotlib Figure."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    freqs = np.logspace(9, 11, 20)
    eps_meas = _make_debye_data(freqs, 2.5, [(1.5, 5e-11)])
    eps_fit = eps_meas * 1.01  # slightly perturbed

    fig = plot_material_fit(freqs, eps_meas, eps_fit)
    assert fig is not None
    assert hasattr(fig, "savefig")  # quacks like a Figure

    # Verify axes were created
    axes = fig.get_axes()
    assert len(axes) == 2  # real + imaginary panels

    plt.close(fig)


# ---------------------------------------------------------------------------
# test_debye_pole_types
# ---------------------------------------------------------------------------

def test_debye_pole_types():
    """Verify returned poles are DebyePole NamedTuples."""
    freqs = np.logspace(9, 11, 30)
    eps_meas = _make_debye_data(freqs, 2.0, [(1.0, 1e-10)])
    result = fit_debye(freqs, eps_meas, n_poles=1)

    pole = result.poles[0]
    assert isinstance(pole, DebyePole)
    assert hasattr(pole, "delta_eps")
    assert hasattr(pole, "tau")
    assert pole.delta_eps > 0
    assert pole.tau > 0


def test_lorentz_pole_types():
    """Verify returned poles are LorentzPole NamedTuples."""
    freqs = np.linspace(1e9, 10e9, 40)
    omega_0 = 2 * np.pi * 5e9
    gamma = 2 * np.pi * 0.5e9
    eps_meas = _make_lorentz_data(freqs, 1.5, [(2.0, omega_0, gamma)])
    result = fit_lorentz(freqs, eps_meas, n_poles=1)

    pole = result.poles[0]
    assert isinstance(pole, LorentzPole)
    assert hasattr(pole, "omega_0")
    assert hasattr(pole, "delta")
    assert hasattr(pole, "kappa")
    assert pole.omega_0 > 0
    assert pole.kappa > 0


# ---------------------------------------------------------------------------
# test_fixed_eps_inf
# ---------------------------------------------------------------------------

def test_fit_debye_fixed_eps_inf():
    """Fit with eps_inf fixed should still converge."""
    freqs = np.logspace(8, 11, 40)
    eps_inf_true = 2.5
    eps_meas = _make_debye_data(freqs, eps_inf_true, [(1.5, 5e-11)])

    result = fit_debye(freqs, eps_meas, n_poles=1, eps_inf=eps_inf_true)
    assert result.eps_inf == pytest.approx(eps_inf_true)
    assert result.fit_error < 0.05


# ---------------------------------------------------------------------------
# Robustness tests — noise, overfitting, outliers (GitHub issue #1)
# ---------------------------------------------------------------------------

class TestDebyeNoiseRobustness:
    """Debye fitting under measurement noise."""

    def test_fit_debye_with_5pct_noise(self):
        """Debye fit should recover tau within 30% under 5% Gaussian noise."""
        freqs = np.linspace(1e9, 10e9, 100)
        eps_inf, deps, tau = 4.0, 2.0, 5e-11
        omega = 2 * np.pi * freqs
        eps_clean = eps_inf + deps / (1 + 1j * omega * tau)

        rng = np.random.RandomState(42)
        noise = 0.05 * np.abs(eps_clean) * (
            rng.randn(100) + 1j * rng.randn(100)
        )
        eps_noisy = eps_clean + noise

        result = fit_debye(freqs, eps_noisy, n_poles=1)

        # tau recovery within 30%
        assert abs(result.poles[0].tau - tau) / tau < 0.30, (
            f"tau recovery failed: fitted {result.poles[0].tau:.3e} vs "
            f"true {tau:.3e}"
        )
        # eps_inf recovery within 20%
        assert abs(result.eps_inf - eps_inf) / eps_inf < 0.20
        # delta_eps recovery within 30%
        assert abs(result.poles[0].delta_eps - deps) / deps < 0.30
        # Fit error should still be reasonable (dominated by noise floor)
        assert result.fit_error < 0.15

    def test_fit_debye_with_10pct_noise(self):
        """Debye fit should not diverge under 10% noise."""
        freqs = np.logspace(8, 11, 80)
        eps_inf, deps, tau = 3.0, 1.5, 1e-10

        eps_clean = _make_debye_data(freqs, eps_inf, [(deps, tau)])

        rng = np.random.RandomState(99)
        noise = 0.10 * np.abs(eps_clean) * (
            rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))
        )
        eps_noisy = eps_clean + noise

        result = fit_debye(freqs, eps_noisy, n_poles=1)

        # Should not produce absurd values
        assert 0.5 < result.eps_inf < 10.0, (
            f"eps_inf out of plausible range: {result.eps_inf}"
        )
        assert result.poles[0].tau > 0
        assert result.poles[0].delta_eps > 0
        # Fit error bounded (noise floor ~ 10%)
        assert result.fit_error < 0.25

    def test_fit_debye_noise_determinism(self):
        """Same noise seed should give identical results."""
        freqs = np.linspace(1e9, 10e9, 60)
        eps_clean = _make_debye_data(freqs, 4.0, [(2.0, 5e-11)])

        results = []
        for _ in range(2):
            rng = np.random.RandomState(123)
            noise = 0.05 * np.abs(eps_clean) * (
                rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))
            )
            results.append(fit_debye(freqs, eps_clean + noise, n_poles=1))

        assert results[0].eps_inf == pytest.approx(results[1].eps_inf)
        assert results[0].poles[0].tau == pytest.approx(results[1].poles[0].tau)


class TestLorentzNoiseRobustness:
    """Lorentz fitting under measurement noise."""

    def test_fit_lorentz_with_5pct_noise(self):
        """Lorentz fit with 5% noise should still find the resonance."""
        f_center = 5e9
        freqs = np.linspace(1e9, 10e9, 100)
        eps_inf_true = 1.5
        delta_eps_true = 2.0
        omega_0_true = 2 * np.pi * f_center
        gamma_true = 2 * np.pi * 0.5e9

        eps_clean = _make_lorentz_data(
            freqs, eps_inf_true,
            [(delta_eps_true, omega_0_true, gamma_true)]
        )

        rng = np.random.RandomState(42)
        noise = 0.05 * np.abs(eps_clean) * (
            rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))
        )
        eps_noisy = eps_clean + noise

        result = fit_lorentz(freqs, eps_noisy, n_poles=1)

        # Resonance frequency recovery within 20%
        assert abs(result.poles[0].omega_0 - omega_0_true) / omega_0_true < 0.20, (
            f"omega_0 recovery failed: fitted {result.poles[0].omega_0:.3e} "
            f"vs true {omega_0_true:.3e}"
        )
        # Fit error reasonable
        assert result.fit_error < 0.15

    def test_fit_lorentz_with_10pct_noise(self):
        """Lorentz fit should not diverge under 10% noise."""
        f_center = 3e9
        freqs = np.linspace(0.5e9, 8e9, 80)
        omega_0 = 2 * np.pi * f_center
        gamma = 2 * np.pi * 0.3e9

        eps_clean = _make_lorentz_data(freqs, 2.0, [(1.5, omega_0, gamma)])

        rng = np.random.RandomState(77)
        noise = 0.10 * np.abs(eps_clean) * (
            rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))
        )
        eps_noisy = eps_clean + noise

        result = fit_lorentz(freqs, eps_noisy, n_poles=1)

        # Should produce finite, positive parameters
        assert result.poles[0].omega_0 > 0
        assert result.poles[0].kappa > 0
        assert np.isfinite(result.fit_error)
        assert result.fit_error < 0.30


class TestOverfitting:
    """Fitting more poles than the data warrants."""

    def test_debye_overfitting_3_poles_on_1_pole_data(self):
        """Fitting 3 poles to 1-pole data should not crash or give absurd results."""
        freqs = np.logspace(8, 11, 60)
        eps_inf_true = 3.0
        deps_true = 2.0
        tau_true = 5e-11

        eps_meas = _make_debye_data(freqs, eps_inf_true, [(deps_true, tau_true)])

        result = fit_debye(freqs, eps_meas, n_poles=3)

        assert len(result.poles) == 3
        # Fit error should be at least as good as the 1-pole case
        assert result.fit_error < 0.05, (
            f"Overfit model error {result.fit_error:.4f} unexpectedly large"
        )

        # All delta_eps should be non-negative (enforced by log parameterization)
        for i, pole in enumerate(result.poles):
            assert pole.delta_eps > 0, f"Pole {i} has non-positive delta_eps"
            assert pole.tau > 0, f"Pole {i} has non-positive tau"

        # Sum of delta_eps should approximate the true value (within 50%)
        total_deps = sum(p.delta_eps for p in result.poles)
        assert abs(total_deps - deps_true) / deps_true < 0.50, (
            f"Total delta_eps {total_deps:.3f} far from true {deps_true:.3f}"
        )

    def test_lorentz_overfitting_2_poles_on_1_pole_data(self):
        """Fitting 2 Lorentz poles to 1-pole data should still converge."""
        freqs = np.linspace(1e9, 10e9, 60)
        omega_0 = 2 * np.pi * 5e9
        gamma = 2 * np.pi * 0.5e9

        eps_meas = _make_lorentz_data(freqs, 1.5, [(2.0, omega_0, gamma)])

        result = fit_lorentz(freqs, eps_meas, n_poles=2)

        assert len(result.poles) == 2
        assert result.fit_error < 0.05
        # All poles should have positive physical parameters
        for i, pole in enumerate(result.poles):
            assert pole.omega_0 > 0, f"Pole {i} has non-positive omega_0"
            assert pole.kappa > 0, f"Pole {i} has non-positive kappa"

    def test_debye_overfit_does_not_degrade(self):
        """3-pole fit error should be <= 1-pole fit error on 1-pole data."""
        freqs = np.logspace(8, 11, 50)
        eps_meas = _make_debye_data(freqs, 2.5, [(1.5, 5e-11)])

        result_1 = fit_debye(freqs, eps_meas, n_poles=1)
        result_3 = fit_debye(freqs, eps_meas, n_poles=3)

        # More poles should not make fit worse (with small tolerance for
        # optimizer variability)
        assert result_3.fit_error <= result_1.fit_error + 0.01, (
            f"3-pole error {result_3.fit_error:.4f} worse than "
            f"1-pole error {result_1.fit_error:.4f}"
        )


class TestOutlierRobustness:
    """Fitting with outlier-corrupted data points."""

    def test_debye_with_outliers(self):
        """A few large outliers should not destroy the Debye fit."""
        freqs = np.linspace(1e9, 10e9, 100)
        eps_inf, deps, tau = 4.0, 2.0, 5e-11
        eps_clean = _make_debye_data(freqs, eps_inf, [(deps, tau)])
        eps_corrupted = eps_clean.copy()

        # Inject 3 large outliers at random positions
        rng = np.random.RandomState(42)
        outlier_idx = rng.choice(100, size=3, replace=False)
        eps_corrupted[outlier_idx] *= 5.0  # 5x magnitude spike

        result = fit_debye(freqs, eps_corrupted, n_poles=1)

        # Fit should not crash
        assert np.isfinite(result.fit_error)
        assert result.poles[0].tau > 0
        assert result.poles[0].delta_eps > 0
        # tau should still be in the right order of magnitude
        assert 1e-12 < result.poles[0].tau < 1e-8, (
            f"tau {result.poles[0].tau:.3e} out of plausible range"
        )

    def test_lorentz_with_outliers(self):
        """A few large outliers should not destroy the Lorentz fit."""
        freqs = np.linspace(1e9, 10e9, 80)
        omega_0 = 2 * np.pi * 5e9
        gamma = 2 * np.pi * 0.5e9

        eps_clean = _make_lorentz_data(freqs, 1.5, [(2.0, omega_0, gamma)])
        eps_corrupted = eps_clean.copy()

        rng = np.random.RandomState(17)
        outlier_idx = rng.choice(80, size=3, replace=False)
        eps_corrupted[outlier_idx] *= 5.0

        result = fit_lorentz(freqs, eps_corrupted, n_poles=1)

        # Should not crash or produce NaN
        assert np.isfinite(result.fit_error)
        assert result.poles[0].omega_0 > 0
        assert result.poles[0].kappa > 0

    def test_debye_single_outlier_at_edge(self):
        """Single outlier at the lowest frequency should not dominate fit."""
        freqs = np.logspace(8, 11, 50)
        eps_inf, deps, tau = 3.0, 1.5, 1e-10
        eps_clean = _make_debye_data(freqs, eps_inf, [(deps, tau)])
        eps_corrupted = eps_clean.copy()

        # Corrupt the first (lowest frequency) point — this is where
        # the static permittivity estimate comes from
        eps_corrupted[0] *= 10.0

        result = fit_debye(freqs, eps_corrupted, n_poles=1)

        # Evaluate fit quality on the clean data (excluding the outlier)
        eps_fitted = eval_debye(freqs, result.eps_inf, result.poles)
        residual_clean = np.abs(eps_fitted[1:] - eps_clean[1:])
        norm_clean = np.sqrt(np.mean(np.abs(eps_clean[1:]) ** 2))
        rms_on_clean = np.sqrt(np.mean(residual_clean ** 2)) / norm_clean

        # A 10x edge outlier biases the initial guess heavily; the fitter
        # should still keep the fit in a reasonable ballpark (< 60% RMS on
        # clean data).  This is intentionally lenient — it documents that
        # edge outliers do degrade the current L-BFGS-B fitter.
        assert rms_on_clean < 0.60, (
            f"RMS on clean portion {rms_on_clean:.4f} too large after "
            f"edge outlier"
        )

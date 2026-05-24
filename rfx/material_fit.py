"""Frequency-dependent material import and Debye/Lorentz pole fitting.

Tools to load measured material data from CSV files and fit Debye or Lorentz
dispersion models for use in rfx FDTD simulations.

Debye model:
    eps(f) = eps_inf + sum_k delta_eps_k / (1 + j*2*pi*f*tau_k)

Lorentz model:
    eps(f) = eps_inf + sum_k delta_eps_k * omega_k^2 /
             (omega_k^2 - omega^2 + j*omega*gamma_k)

The fitting uses scipy.optimize.minimize (L-BFGS-B) to find optimal pole
parameters that minimize the relative RMS error between measured and modeled
complex permittivity.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole


try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class DebyeFitResult(NamedTuple):
    """Result of a Debye pole fit.

    eps_inf : float
        High-frequency permittivity (dimensionless).
    poles : list[DebyePole]
        Fitted Debye poles.
    fit_error : float
        Relative RMS fit error.
    """
    eps_inf: float
    poles: list
    fit_error: float


class LorentzFitResult(NamedTuple):
    """Result of a Lorentz pole fit.

    eps_inf : float
        High-frequency permittivity (dimensionless).
    poles : list[LorentzPole]
        Fitted Lorentz poles.
    fit_error : float
        Relative RMS fit error.
    """
    eps_inf: float
    poles: list
    fit_error: float


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_material_csv(
    path: str,
    *,
    freq_col: str = "freq_hz",
    eps_real_col: str = "eps_r",
    eps_imag_col: str = "eps_i",
    tan_delta_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load measured material data from a CSV file.

    Supports two input formats:

    1. Complex permittivity: columns for frequency, real part, and imaginary
       part of permittivity.
    2. Loss tangent: columns for frequency, real part, and tan(delta).
       The imaginary part is computed as ``eps_r * tan_delta``.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    freq_col : str
        Column name for frequency in Hz.
    eps_real_col : str
        Column name for the real part of permittivity.
    eps_imag_col : str
        Column name for the imaginary part of permittivity.
    tan_delta_col : str or None
        Column name for loss tangent. If provided, ``eps_imag_col`` is ignored
        and the imaginary part is derived from ``eps_r * tan_delta``.

    Returns
    -------
    freqs : ndarray, shape (N,)
        Frequencies in Hz.
    eps_complex : ndarray, shape (N,), dtype complex128
        Complex relative permittivity ``eps' - j*eps''``.
    """
    import csv
    import io

    # Support both file paths and string buffers (for testing)
    if hasattr(path, "read"):
        text = path.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        reader_input = io.StringIO(text)
    else:
        reader_input = open(path, newline="", encoding="utf-8")

    try:
        reader = csv.DictReader(reader_input)
        rows = list(reader)
    finally:
        reader_input.close()

    if not rows:
        raise ValueError("CSV file contains no data rows")

    freqs = np.array([float(row[freq_col]) for row in rows])
    eps_real = np.array([float(row[eps_real_col]) for row in rows])

    if tan_delta_col is not None:
        tan_d = np.array([float(row[tan_delta_col]) for row in rows])
        eps_imag = eps_real * tan_d
    else:
        eps_imag = np.array([float(row[eps_imag_col]) for row in rows])

    # Convention: eps = eps' - j*eps'' (positive eps_imag means loss)
    eps_complex = eps_real - 1j * eps_imag

    return freqs, eps_complex


# ---------------------------------------------------------------------------
# Debye fitting
# ---------------------------------------------------------------------------

def _debye_model(freqs: np.ndarray, eps_inf: float,
                 poles: list[tuple[float, float]]) -> np.ndarray:
    """Evaluate Debye model at given frequencies.

    Parameters
    ----------
    freqs : (N,) array of frequencies in Hz
    eps_inf : high-frequency permittivity
    poles : list of (delta_eps, tau) tuples

    Returns
    -------
    eps_complex : (N,) complex array
    """
    omega = 2.0 * np.pi * freqs
    eps = np.full_like(omega, eps_inf, dtype=complex)
    for delta_eps, tau in poles:
        eps = eps + delta_eps / (1.0 + 1j * omega * tau)
    return eps


def fit_debye(
    freqs: np.ndarray,
    eps_measured: np.ndarray,
    n_poles: int = 1,
    *,
    eps_inf: float | None = None,
) -> DebyeFitResult:
    """Fit Debye relaxation model to measured permittivity data.

    Uses ``scipy.optimize.minimize`` (L-BFGS-B) to find optimal Debye pole
    parameters that minimize the relative RMS error.

    Model:
        eps(f) = eps_inf + sum_k delta_eps_k / (1 + j*2*pi*f*tau_k)

    Parameters
    ----------
    freqs : (N,) array
        Measurement frequencies in Hz.
    eps_measured : (N,) complex array
        Measured complex permittivity.
    n_poles : int
        Number of Debye poles to fit.
    eps_inf : float or None
        If provided, fix the high-frequency permittivity. If None, it is
        included as a free parameter (initialized from the highest-frequency
        measurement).

    Returns
    -------
    DebyeFitResult
        Named tuple with ``eps_inf``, ``poles`` (list of DebyePole), and
        ``fit_error`` (relative RMS).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    eps_measured = np.asarray(eps_measured, dtype=np.complex128)

    # Normalization for error
    norm = np.sqrt(np.mean(np.abs(eps_measured) ** 2))
    if norm == 0:
        norm = 1.0

    # Initial guesses
    fit_eps_inf = eps_inf is None
    if eps_inf is None:
        # Use real part at highest frequency as initial guess
        idx_max = np.argmax(freqs)
        eps_inf_init = float(np.real(eps_measured[idx_max]))
    else:
        eps_inf_init = eps_inf

    # Static permittivity estimate
    idx_min = np.argmin(freqs)
    eps_static = float(np.real(eps_measured[idx_min]))
    delta_eps_total = max(eps_static - eps_inf_init, 0.1)

    # Log-spaced tau initial guesses spanning the frequency range
    f_min = max(freqs.min(), 1.0)
    f_max = max(freqs.max(), f_min * 10)
    tau_guesses = np.logspace(
        np.log10(1.0 / (2 * np.pi * f_max)),
        np.log10(1.0 / (2 * np.pi * f_min)),
        n_poles,
    )

    # Pack parameters: [eps_inf?, log10(delta_eps_1), log10(tau_1), ...]
    x0 = []
    bounds = []

    if fit_eps_inf:
        x0.append(eps_inf_init)
        bounds.append((0.5, None))

    for k in range(n_poles):
        de_init = delta_eps_total / n_poles
        x0.append(np.log10(max(de_init, 1e-6)))
        bounds.append((np.log10(1e-6), np.log10(1e4)))
        x0.append(np.log10(tau_guesses[k]))
        bounds.append((np.log10(1e-15), np.log10(1e-3)))

    x0 = np.array(x0)

    def _unpack(x):
        idx = 0
        if fit_eps_inf:
            ei = x[idx]
            idx += 1
        else:
            ei = eps_inf
        poles_list = []
        for _ in range(n_poles):
            de = 10.0 ** x[idx]
            idx += 1
            tau = 10.0 ** x[idx]
            idx += 1
            poles_list.append((de, tau))
        return ei, poles_list

    def _objective(x):
        ei, poles_list = _unpack(x)
        eps_model = _debye_model(freqs, ei, poles_list)
        residual = eps_model - eps_measured
        return float(np.sqrt(np.mean(np.abs(residual) ** 2)) / norm)

    result = minimize(_objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 5000, "ftol": 1e-14})

    ei_opt, poles_opt = _unpack(result.x)
    debye_poles = [DebyePole(delta_eps=de, tau=tau) for de, tau in poles_opt]

    # Final error
    eps_fitted = _debye_model(freqs, ei_opt, poles_opt)
    fit_err = float(np.sqrt(np.mean(np.abs(eps_fitted - eps_measured) ** 2)) / norm)

    return DebyeFitResult(eps_inf=float(ei_opt), poles=debye_poles,
                          fit_error=fit_err)


# ---------------------------------------------------------------------------
# Lorentz fitting
# ---------------------------------------------------------------------------

def _lorentz_model(freqs: np.ndarray, eps_inf: float,
                   poles: list[tuple[float, float, float]]) -> np.ndarray:
    """Evaluate Lorentz model at given frequencies.

    Parameters
    ----------
    freqs : (N,) array of frequencies in Hz
    eps_inf : high-frequency permittivity
    poles : list of (delta_eps, omega_0, gamma) tuples

    Returns
    -------
    eps_complex : (N,) complex array
    """
    omega = 2.0 * np.pi * freqs
    eps = np.full_like(omega, eps_inf, dtype=complex)
    for delta_eps, omega_0, gamma in poles:
        eps = eps + (delta_eps * omega_0 ** 2 /
                     (omega_0 ** 2 - omega ** 2 + 1j * omega * gamma))
    return eps


def fit_lorentz(
    freqs: np.ndarray,
    eps_measured: np.ndarray,
    n_poles: int = 1,
    *,
    eps_inf: float | None = None,
) -> LorentzFitResult:
    """Fit Lorentz oscillator model to measured permittivity data.

    Uses ``scipy.optimize.minimize`` (L-BFGS-B) to find optimal Lorentz
    pole parameters.

    Model:
        eps(f) = eps_inf + sum_k delta_eps_k * omega_k^2 /
                 (omega_k^2 - omega^2 + j*omega*gamma_k)

    Parameters
    ----------
    freqs : (N,) array
        Measurement frequencies in Hz.
    eps_measured : (N,) complex array
        Measured complex permittivity.
    n_poles : int
        Number of Lorentz poles to fit.
    eps_inf : float or None
        If provided, fix the high-frequency permittivity. If None, it is
        included as a free parameter.

    Returns
    -------
    LorentzFitResult
        Named tuple with ``eps_inf``, ``poles`` (list of LorentzPole), and
        ``fit_error`` (relative RMS).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    eps_measured = np.asarray(eps_measured, dtype=np.complex128)

    norm = np.sqrt(np.mean(np.abs(eps_measured) ** 2))
    if norm == 0:
        norm = 1.0

    fit_eps_inf = eps_inf is None
    if eps_inf is None:
        idx_max = np.argmax(freqs)
        eps_inf_init = float(np.real(eps_measured[idx_max]))
    else:
        eps_inf_init = eps_inf

    # Initial guesses: resonances spread across the frequency band
    f_min = max(freqs.min(), 1.0)
    f_max = max(freqs.max(), f_min * 10)
    omega_min = 2 * np.pi * f_min
    omega_max = 2 * np.pi * f_max
    omega_guesses = np.logspace(
        np.log10(omega_min), np.log10(omega_max), n_poles
    )

    # Pack: [eps_inf?, log10(delta_eps_1), log10(omega_0_1), log10(gamma_1), ...]
    x0 = []
    bounds = []

    if fit_eps_inf:
        x0.append(eps_inf_init)
        bounds.append((0.1, None))

    for k in range(n_poles):
        # Initial delta_eps
        x0.append(np.log10(1.0))
        bounds.append((np.log10(1e-4), np.log10(1e4)))
        # Initial omega_0
        x0.append(np.log10(omega_guesses[k]))
        bounds.append((np.log10(omega_min * 0.1), np.log10(omega_max * 10)))
        # Initial gamma (damping ~ 10% of omega_0)
        x0.append(np.log10(omega_guesses[k] * 0.1))
        bounds.append((np.log10(omega_min * 0.001), np.log10(omega_max * 10)))

    x0 = np.array(x0)

    def _unpack(x):
        idx = 0
        if fit_eps_inf:
            ei = x[idx]
            idx += 1
        else:
            ei = eps_inf
        poles_list = []
        for _ in range(n_poles):
            de = 10.0 ** x[idx]
            idx += 1
            w0 = 10.0 ** x[idx]
            idx += 1
            gam = 10.0 ** x[idx]
            idx += 1
            poles_list.append((de, w0, gam))
        return ei, poles_list

    def _objective(x):
        ei, poles_list = _unpack(x)
        eps_model = _lorentz_model(freqs, ei, poles_list)
        residual = eps_model - eps_measured
        return float(np.sqrt(np.mean(np.abs(residual) ** 2)) / norm)

    # Run multiple restarts to find better optima
    best_result = None
    best_cost = np.inf

    for trial in range(3):
        if trial == 0:
            x_init = x0.copy()
        else:
            # Perturbed initial guess
            rng = np.random.RandomState(42 + trial)
            x_init = x0 + rng.randn(len(x0)) * 0.3

        res = minimize(_objective, x_init, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 5000, "ftol": 1e-14})
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res

    ei_opt, poles_opt = _unpack(best_result.x)

    # Convert to LorentzPole NamedTuples
    # LorentzPole uses (omega_0, delta, kappa) where delta = gamma/2
    # and kappa = delta_eps * omega_0^2
    lorentz_poles = [
        LorentzPole(omega_0=w0, delta=gam / 2.0, kappa=de * w0 ** 2)
        for de, w0, gam in poles_opt
    ]

    # Final error
    eps_fitted = _lorentz_model(freqs, ei_opt, poles_opt)
    fit_err = float(np.sqrt(np.mean(np.abs(eps_fitted - eps_measured) ** 2)) / norm)

    return LorentzFitResult(eps_inf=float(ei_opt), poles=lorentz_poles,
                            fit_error=fit_err)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_debye(freqs: np.ndarray, eps_inf: float,
               poles: list[DebyePole]) -> np.ndarray:
    """Evaluate a Debye model at given frequencies.

    Parameters
    ----------
    freqs : (N,) array in Hz
    eps_inf : high-frequency permittivity
    poles : list of DebyePole

    Returns
    -------
    eps_complex : (N,) complex array
    """
    return _debye_model(freqs, eps_inf,
                        [(p.delta_eps, p.tau) for p in poles])


def eval_lorentz(freqs: np.ndarray, eps_inf: float,
                 poles: list[LorentzPole]) -> np.ndarray:
    """Evaluate a Lorentz model at given frequencies.

    Parameters
    ----------
    freqs : (N,) array in Hz
    eps_inf : high-frequency permittivity
    poles : list of LorentzPole

    Returns
    -------
    eps_complex : (N,) complex array
    """
    pole_tuples = []
    for p in poles:
        # Recover delta_eps and gamma from LorentzPole fields
        w0 = p.omega_0
        gamma = 2.0 * p.delta
        if w0 > 0:
            delta_eps = p.kappa / (w0 ** 2)
        else:
            delta_eps = 0.0
        pole_tuples.append((delta_eps, w0, gamma))
    return _lorentz_model(freqs, eps_inf, pole_tuples)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_material_fit(
    freqs: np.ndarray,
    eps_measured: np.ndarray,
    eps_fitted: np.ndarray,
    *,
    ax: object | None = None,
) -> object:
    """Plot measured vs fitted complex permittivity.

    Creates a two-panel plot showing real and imaginary parts of permittivity
    as a function of frequency.

    Parameters
    ----------
    freqs : (N,) array
        Frequencies in Hz.
    eps_measured : (N,) complex array
        Measured permittivity.
    eps_fitted : (N,) complex array
        Fitted permittivity from the model.
    ax : matplotlib Axes or None
        If None, a new figure with two subplots is created.

    Returns
    -------
    fig : matplotlib Figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plot_material_fit")

    freqs = np.asarray(freqs)
    eps_measured = np.asarray(eps_measured)
    eps_fitted = np.asarray(eps_fitted)

    # Frequency in GHz for display
    f_ghz = freqs / 1e9

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = ax.get_figure()
        ax1 = ax
        ax2 = None

    # Real part
    ax1.plot(f_ghz, np.real(eps_measured), "o", ms=4, label="Measured",
             alpha=0.7)
    ax1.plot(f_ghz, np.real(eps_fitted), "-", lw=2, label="Fitted")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel(r"$\varepsilon'$ (real)")
    ax1.set_title("Real permittivity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Imaginary part (if we have a second axes)
    if ax2 is not None:
        ax2.plot(f_ghz, -np.imag(eps_measured), "o", ms=4, label="Measured",
                 alpha=0.7)
        ax2.plot(f_ghz, -np.imag(eps_fitted), "-", lw=2, label="Fitted")
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel(r"$\varepsilon''$ (loss)")
        ax2.set_title("Imaginary permittivity")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

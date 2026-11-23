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
import warnings

import numpy as np
from scipy.optimize import lsq_linear, minimize

from rfx.errors import AccuracyWarning
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole


from rfx._plotting import _pyplot


_DISPLAY_HZ_PER_GHZ = 1e9
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
    Positive input ``eps_imag`` denotes loss."""
    import csv
    import io

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

    eps_complex = eps_real - 1j * eps_imag

    return freqs, eps_complex


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


_EPS_INF_FLOOR = 1.0


def _fit_weights(eps_measured):
    """Scale each part by its table RMS, falling back for negligible loss."""
    w_re = float(np.sqrt(np.mean(eps_measured.real ** 2)))
    rms_im = float(np.sqrt(np.mean(eps_measured.imag ** 2)))
    w_im = rms_im if rms_im > 1e-6 * w_re else w_re
    return w_re, w_im


def _warn_fit_bounds(eps_inf, *, free_eps_inf=True, flags=(), suggest_placed=False):
    """Report all active fit bounds once, sharing the permittivity floor."""
    messages = list(flags)
    if free_eps_inf and eps_inf - _EPS_INF_FLOOR <= 1e-6:
        messages.insert(0, f"eps_inf={eps_inf:g} is at the floor {_EPS_INF_FLOOR:g}")
    elif not free_eps_inf and eps_inf < _EPS_INF_FLOOR:
        messages.append(
            f"fixed eps_inf={eps_inf:g} is below {_EPS_INF_FLOOR:g}; this model needs "
            "a time step at most sqrt(eps_inf) of the vacuum Courant limit")
    if messages:
        if suggest_placed:
            messages.append('Consider tau="placed" for a fit with placed relaxation times')
        warnings.warn("; ".join(messages), AccuracyWarning, stacklevel=3)


def fit_debye(
    freqs: np.ndarray,
    eps_measured: np.ndarray,
    n_poles: int = 1,
    *,
    eps_inf: float | None = None,
    tau: str = "free",
) -> DebyeFitResult:
    """Fit Debye relaxation model to measured permittivity data.

    The real and imaginary residuals are each divided by their table RMS;
    loss below 1e-6 of the real RMS uses the real RMS instead.
    A free ``eps_inf`` is bounded below at 1. Relaxation times lie between
    1/(2*pi*f_hi) and 1/(2*pi*f_lo), with f_lo = max(min(freqs), 1 Hz)
    and f_hi = max(freqs).

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
        measurement), with a lower bound of 1. A fixed value below 1 is
        retained with an AccuracyWarning about the required time step.
    tau : {"free", "placed"}
        "free" (default) optimizes relaxation times with L-BFGS-B.
        "placed" uses log-spaced times including both span ends (the
        geometric centre for one pole) and bounded linear least squares
        for eps_inf and nonnegative pole strengths.

    Returns
    -------
    DebyeFitResult
        Named tuple with ``eps_inf``, ``poles`` (list of DebyePole), and
        ``fit_error`` (RMS complex residual divided by RMS measured magnitude).

    Notes
    -----
    For constant eps' 1.1 and tan delta 0.1 over 1–40 GHz, neither mode
    fits within 10% over 2–20 GHz: at n_poles 3 and 4 the default's worst
    tan delta error is 13.1% and eps' error 15.6%, with eps_inf at 1;
    "placed" gives worst tan delta errors of 15.0% and 14.7%, respectively.
    """
    if not isinstance(tau, str) or tau not in ("free", "placed"):
        raise ValueError('tau must be "free" or "placed"')
    freqs = np.asarray(freqs, dtype=np.float64)
    eps_measured = np.asarray(eps_measured, dtype=np.complex128)

    norm = np.sqrt(np.mean(np.abs(eps_measured) ** 2))
    if norm == 0:
        norm = 1.0

    w_re, w_im = _fit_weights(eps_measured)
    tau_lo = 1.0 / (2 * np.pi * freqs.max())
    tau_hi = 1.0 / (2 * np.pi * max(freqs.min(), 1.0))

    if tau == "placed":
        taus = (np.geomspace(tau_lo, tau_hi, n_poles) if n_poles > 1
                else np.array([np.sqrt(tau_lo * tau_hi)]))
        design = 1 / (1 + 2j * np.pi * freqs[:, None] * taus[None, :])
        if eps_inf is None:
            design = np.column_stack([np.ones(len(freqs)), design])
            target = eps_measured
            lower = np.r_[_EPS_INF_FLOOR, np.zeros(n_poles)]
        else:
            target = eps_measured - eps_inf
            lower = np.zeros(n_poles)
        solution = lsq_linear(
            np.vstack([design.real / w_re, design.imag / w_im]),
            np.r_[target.real / w_re, target.imag / w_im],
            bounds=(lower, np.inf), method="bvls")
        ei = float(solution.x[0]) if eps_inf is None else float(eps_inf)
        strengths = solution.x[1:] if eps_inf is None else solution.x
        poles = [DebyePole(float(d), float(t)) for d, t in zip(strengths, taus)]
        flags = [f"pole {k + 1} delta_eps=0; fewer poles fit this table as well"
                 for k, d in enumerate(strengths) if d == 0]
        _warn_fit_bounds(ei, free_eps_inf=eps_inf is None, flags=flags)
        fitted = _debye_model(freqs, ei, [(p.delta_eps, p.tau) for p in poles])
        error = float(np.sqrt(np.mean(np.abs(fitted - eps_measured) ** 2)) / norm)
        return DebyeFitResult(ei, poles, error)

    fit_eps_inf = eps_inf is None
    if eps_inf is None:
        idx_max = np.argmax(freqs)
        eps_inf_init = float(np.real(eps_measured[idx_max]))
    else:
        eps_inf_init = eps_inf

    idx_min = np.argmin(freqs)
    eps_static = float(np.real(eps_measured[idx_min]))
    delta_eps_total = max(eps_static - eps_inf_init, 0.1)

    f_min = max(freqs.min(), 1.0)
    f_max = max(freqs.max(), f_min * 10)
    tau_guesses = np.logspace(
        np.log10(1.0 / (2 * np.pi * f_max)),
        np.log10(1.0 / (2 * np.pi * f_min)),
        n_poles,
    )

    x0 = []
    bounds = []

    if fit_eps_inf:
        x0.append(eps_inf_init)
        bounds.append((_EPS_INF_FLOOR, None))

    for k in range(n_poles):
        de_init = delta_eps_total / n_poles
        x0.append(np.log10(max(de_init, 1e-6)))
        bounds.append((np.log10(1e-6), np.log10(1e4)))
        x0.append(np.log10(tau_guesses[k]))
        bounds.append((np.log10(tau_lo), np.log10(tau_hi)))

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
        return float(np.sqrt(np.mean((residual.real / w_re) ** 2
                                     + (residual.imag / w_im) ** 2)))

    result = minimize(_objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 5000, "ftol": 1e-14})

    ei_opt, poles_opt = _unpack(result.x)
    debye_poles = [DebyePole(delta_eps=de, tau=tau) for de, tau in poles_opt]

    flags = [f"pole {k + 1} delta_eps={p.delta_eps:g} is at its floor; "
             "fewer poles fit this table as well"
             for k, p in enumerate(debye_poles) if p.delta_eps / 1e-6 - 1 <= 1e-6]
    _warn_fit_bounds(ei_opt, free_eps_inf=fit_eps_inf, flags=flags,
                     suggest_placed=True)

    eps_fitted = _debye_model(freqs, ei_opt, poles_opt)
    fit_err = float(np.sqrt(np.mean(np.abs(eps_fitted - eps_measured) ** 2)) / norm)

    return DebyeFitResult(eps_inf=float(ei_opt), poles=debye_poles,
                          fit_error=fit_err)


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
    pole parameters. Real and imaginary residuals are divided by their table
    RMS; loss below 1e-6 of the real RMS uses the real RMS instead.
    With f_lo = max(min(freqs), 1 Hz) and f_hi = max(freqs), resonances
    lie in [2*pi*f_lo, 2*pi*f_hi], damping in [0.001*2*pi*f_lo,
    10*2*pi*f_hi], and pole strengths in [1e-4, 1e4].
    A comb of log-spaced resonances covers the span with
    max(1, ceil(ln(f_hi/f_lo)/(2*ln(1.1)))) starts; the lowest objective
    wins, with relative ties within 1e-12 retaining the first start.

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
        included as a free parameter bounded below at 1. A fixed value
        below 1 is retained with an AccuracyWarning about the time step.

    Returns
    -------
    LorentzFitResult
        Named tuple with ``eps_inf``, ``poles`` (list of LorentzPole), and
        ``fit_error`` (RMS complex residual divided by RMS measured magnitude).

    Raises
    ------
    ValueError
        If the table has no frequency span after clipping f_lo to 1 Hz.

    Notes
    -----
    For constant eps' 1.1 and tan delta 0.1 over 1–40 GHz, the worst
    errors over 2–20 GHz at n_poles 2 and 3 are 13.6% and 14.6% in
    tan delta, and 12.8% and 11.6% in eps', with eps_inf at 1.
    A resonance above the table is held at its top frequency: on a
    1–10 GHz table of a 15 GHz resonance, the worst errors over the
    table are 25% in eps' and 46% in tan delta.
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

    f_min = max(freqs.min(), 1.0)
    f_max = freqs.max()
    if f_max <= f_min:
        raise ValueError("frequency span must have f_hi > f_lo after the 1 Hz clip")
    omega_min = 2 * np.pi * f_min
    omega_max = 2 * np.pi * f_max
    omega_guesses = np.logspace(
        np.log10(omega_min), np.log10(omega_max), n_poles
    )

    x0 = []
    bounds = []

    if fit_eps_inf:
        x0.append(eps_inf_init)
        bounds.append((_EPS_INF_FLOOR, None))

    for k in range(n_poles):
        x0.append(np.log10(1.0))
        bounds.append((np.log10(1e-4), np.log10(1e4)))
        x0.append(np.log10(omega_guesses[k]))
        bounds.append((np.log10(omega_min), np.log10(omega_max)))
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

    w_re, w_im = _fit_weights(eps_measured)

    def _objective(x):
        ei, poles_list = _unpack(x)
        eps_model = _lorentz_model(freqs, ei, poles_list)
        residual = eps_model - eps_measured
        return float(np.sqrt(np.mean((residual.real / w_re) ** 2
                                     + (residual.imag / w_im) ** 2)))

    starts = max(1, int(np.ceil(np.log(f_max / f_min) / (2 * np.log(1.1)))))
    results = []
    for k in range(starts):
        x_init = x0.copy()
        for j in range(n_poles):
            w0 = omega_min * (omega_max / omega_min) ** ((j + (k + 0.5) / starts) / n_poles)
            idx = int(fit_eps_inf) + 3 * j
            x_init[idx:idx + 3] = [np.log10(1.0), np.log10(w0), np.log10(w0 * 0.1)]
        res = minimize(_objective, x_init, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 5000, "ftol": 1e-14})
        results.append(res)

    best = 0
    for i in range(1, len(results)):
        a, b = results[i].fun, results[best].fun
        if a < b and abs(a - b) > 1e-12 * max(abs(a), abs(b)):
            best = i
    best_result = results[best]

    ei_opt, poles_opt = _unpack(best_result.x)

    lorentz_poles = [
        LorentzPole(omega_0=w0, delta=gam / 2.0, kappa=de * w0 ** 2)
        for de, w0, gam in poles_opt
    ]

    flags = [f"pole {k + 1} delta_eps={de:g} is at its floor; "
             "fewer poles fit this table as well"
             for k, (de, _, _) in enumerate(poles_opt) if de / 1e-4 - 1 <= 1e-6]
    _warn_fit_bounds(ei_opt, free_eps_inf=fit_eps_inf, flags=flags)

    eps_fitted = _lorentz_model(freqs, ei_opt, poles_opt)
    fit_err = float(np.sqrt(np.mean(np.abs(eps_fitted - eps_measured) ** 2)) / norm)

    return LorentzFitResult(eps_inf=float(ei_opt), poles=lorentz_poles,
                            fit_error=fit_err)


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
    """Evaluate a Lorentz or Drude pole model at given frequencies.

    Parameters
    ----------
    freqs : (N,) array in Hz
    eps_inf : high-frequency permittivity
    poles : list of LorentzPole (``lorentz_pole`` or ``drude_pole``; a Drude
        pole has ``omega_0 == 0`` and gives eps_inf - omega_p^2 / (w^2 - j*gamma*w))

    Returns
    -------
    eps_complex : (N,) complex array
    """
    from rfx.materials.pole_map import refuse_maps
    refuse_maps(poles, "material_fit.eval_lorentz")
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    eps = np.full(omega.shape, eps_inf, dtype=complex)
    for p in poles:
        from rfx.materials.relative_strength import RelativeStrength
        kappa = p.kappa.si if isinstance(p.kappa, RelativeStrength) else p.kappa
        eps = eps + kappa / (p.omega_0 ** 2 - omega ** 2 + 2j * p.delta * omega)
    return eps


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
    plt = _pyplot()

    freqs = np.asarray(freqs)
    eps_measured = np.asarray(eps_measured)
    eps_fitted = np.asarray(eps_fitted)

    f_ghz = freqs / _DISPLAY_HZ_PER_GHZ

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = ax.get_figure()
        ax1 = ax
        ax2 = None

    ax1.plot(f_ghz, np.real(eps_measured), "o", ms=4, label="Measured",
             alpha=0.7)
    ax1.plot(f_ghz, np.real(eps_fitted), "-", lw=2, label="Fitted")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel(r"$\varepsilon'$ (real)")
    ax1.set_title("Real permittivity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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

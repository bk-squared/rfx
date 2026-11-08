"""Harmonic inversion for resonance extraction.

Implements the Matrix Pencil Method (MPM) to extract resonance
frequencies, decay rates, and Q factors from time-domain signals.
Much more accurate than FFT for short time series.

Reference: T. K. Sarkar and O. Pereira, "Using the Matrix Pencil
Method to Estimate the Parameters of a Sum of Complex Exponentials",
IEEE AP Magazine, Feb 1995.

Usage
-----
>>> modes = matrix_pencil(signal, dt, f_min, f_max)
>>> for m in modes:
...     print(f"f={m.freq/1e9:.4f} GHz, Q={m.Q:.0f}")
"""

from __future__ import annotations

from typing import NamedTuple
import warnings

import numpy as np
from scipy.signal import decimate as scipy_decimate


from rfx.measurement.matrix_pencil import (
    DECIMATION_GUARD, _MAX_DECIMATION_STAGE, _FIR_HALF_OUTPUT_SAMPLES,
    _MIN_PENCIL_SAMPLES, _pencil_columns, _decimation_factors, _decimation_plan,
    _decimate, _pencil, _to_original_rate, _fit_residues,
)


def matrix_pencil_record_duration(n_samples: int, dt: float, f_max: float, *,
                           decimate: str | bool = "auto",
                           pencil_parameter: float = 0.33,
                           max_modes: int = 50) -> float:
    """Time between the first and last samples used by the matrix pencil method.

    Reference: Sarkar & Pereira (1995), "Using the Matrix Pencil Method to
    Estimate the Parameters of a Sum of Complex Exponentials", IEEE AP Magazine.

    Use this duration, rather than the unfiltered input length, for record-
    length admission or uncertainty calculations. Each zero-phase FIR stage
    excludes outputs whose filter support reaches outside its input record.
    This helper shares the estimator's sampling plan and performs no solve.
    Pass the same ``pencil_parameter`` and ``max_modes`` as the estimator:
    they determine when further decimation would reduce its model capacity.
    This describes the pole fit; amplitudes still use the original input.
    """
    factors, kept = _decimation_plan(n_samples, dt, f_max, decimate,
                                      pencil_parameter=pencil_parameter,
                                      max_modes=max_modes)
    for factor in factors:
        dt *= factor
    return max(0, kept - 1) * dt


class MatrixPencilMode(NamedTuple):
    """A single resonant mode extracted by the matrix pencil method."""
    freq: float       # frequency in Hz
    decay: float      # decay rate (1/s), positive = decaying
    Q: float          # quality factor
    amplitude: float  # amplitude magnitude
    phase: float      # phase in radians
    error: float      # relative error estimate


def _fit_mode_amplitudes(signal: np.ndarray, poles: np.ndarray,
                         dt: float) -> np.ndarray:
    """Joint complex coefficients at the original input's first sample.

    All finite candidate poles participate, including conjugates and modes
    outside the reporting band. Independent projections double-count their
    overlap on a finite record. Coefficients are probe-signal weights, not
    power-normalized modal energies or noise-confidence estimates.

    A growing pole is anchored at the last sample so every basis entry has
    magnitude <= 1; column normalization then removes norm disparities from
    the least-squares system. These scalings avoid overflow, but cannot make
    nearly coincident poles physically distinguishable. lstsq's numerical
    rank cutoff handles singular systems without forming normal equations.
    """
    return _fit_residues(np.asarray(signal)[:, None], poles, dt)[:, 0]


def matrix_pencil(
    signal: np.ndarray,
    dt: float,
    f_min: float,
    f_max: float,
    *,
    pencil_parameter: float = 0.33,
    min_Q: float = 1.0,
    max_modes: int = 50,
    sv_threshold: float = 1e-3,
    decimate: str | bool = "auto",
) -> list[MatrixPencilMode]:
    """Extract resonant modes via the matrix pencil method.

    Reference: Sarkar & Pereira (1995), "Using the Matrix Pencil Method to
    Estimate the Parameters of a Sum of Complex Exponentials", IEEE AP Magazine.

    .. warning::

       **Lossless closed cavities have infinite physical Q — do not read a Q
       off a finite window.** In a PEC-bounded domain with no material loss
       (an empty or lossless-filled cavity), a resonance never decays, so the
       Q returned here is a pure *windowing artefact*: it swings by orders of
       magnitude with the run length rather than tracking any physics. The
       resonant *frequency* is still meaningful; the *Q* is not. To measure a
       meaningful Q, add a realistic loss (a finite ``sigma`` / ``tan_delta``
       on the fill, or a lossy load) so the cavity has a finite physical Q, or
       use an open (CPML) boundary so radiation sets the Q.

    Parameters
    ----------
    signal : 1D real or complex array
        Time-domain signal.
    dt : float
        Timestep in seconds.
    f_min, f_max : float
        Frequency search range in Hz.
    pencil_parameter : float
        Fraction of signal length for pencil size L (0.2-0.5).
    min_Q : float
        Discard modes with Q < min_Q.
    max_modes : int
        Maximum complex-pole rank and maximum modes to return. Both members
        of a real signal's conjugate pair consume rank before reporting.
    sv_threshold : float
        Relative singular value threshold for rank determination (relative
        to the largest singular value). The same threshold is used after
        resampling; it is not a calibrated noise-confidence bound.
    decimate : {"auto", False}
        Automatically reduce oversampled, band-limited inputs before matrix
        pencil analysis. With the default ``"auto"``, decimation is applied
        when ``1 / dt > 8 * f_max`` using a target factor of
        ``int(1 / dt / (4 * f_max))``. Anti-aliased, zero-phase FIR stages of
        at most 13 are used, and the effective timestep is passed to the core
        algorithm. At each stage, outputs touching the filter's zero-padded
        boundary are excluded: filtering an interior exponential preserves
        its pole, while zero padding introduces transients. This SHORTENS
        the usable record; ``matrix_pencil_record_duration`` reports its duration.
        A stage is omitted if fewer than 10 interior samples would remain or
        its pencil could not hold ``max_modes`` poles plus one singular
        direction for rank selection. If the original pencil is smaller,
        its existing capacity is preserved instead. This is a dimension
        safeguard, not a guarantee that noisy/nearby modes are identifiable.
        Set to ``False`` to retain every input sample. Decimation avoids
        redundant work because the estimator scales approximately as
        :math:`O(N^{2.7})`.

    Returns
    -------
    list of MatrixPencilMode, sorted by amplitude (strongest first).
        Amplitude and phase describe jointly fitted complex coefficients at
        the original input's first sample. For a real cosine of peak A the
        positive-frequency coefficient has magnitude A/2; its conjugate is
        fitted too, then omitted from the returned list. Relative weights
        depend on source/probe placement and are not modal energies.
        Distinct poles are kept regardless of frequency spacing; for complex
        signals, only negative-frequency conjugate partners within 1e-9 in
        relative frequency and 1e-9 * 2*pi*f in decay are omitted.
        A complex record that is real up to noise can therefore list a mode
        twice; pass the real part. Check each mode's ``error``: weak
        neighbouring poles with a large error may be fitting artefacts.
    """
    y = np.asarray(signal, dtype=np.complex128).ravel()
    real_signal = not np.any(y.imag)
    amplitude_signal = y
    amplitude_dt = dt
    factors, _ = _decimation_plan(len(y), dt, f_max, decimate,
                                 pencil_parameter=pencil_parameter,
                                 max_modes=max_modes)
    Yd = _decimate(y[:, None], factors, decimate_fn=scipy_decimate)
    if len(Yd) < _MIN_PENCIL_SAMPLES:
        return []
    D = int(np.prod(factors)) if factors else 1
    try:
        kept, _, growing, _, _ = _pencil(
            Yd, sv_threshold, 0, pencil_parameter=pencil_parameter,
            max_rank=max_modes)
        poles, _, eigenvalues = _to_original_rate(
            np.concatenate([kept, growing]), D, dt, 0.9)
        amplitudes = _fit_mode_amplitudes(amplitude_signal, poles, amplitude_dt)
    except (ValueError, np.linalg.LinAlgError):
        return []

    modes = []
    mode_poles = []
    for lam, s, amp_complex in zip(eigenvalues, poles, amplitudes):
        # Real records have conjugate poles. Fit both, then report the
        # positive-frequency coefficient with a deterministic phase.
        if real_signal and s.imag < 0:
            continue

        # z = exp(s * dt) where s = -alpha + j*omega
        freq = abs(s.imag) / (2 * np.pi)
        decay = -s.real

        # Filter by frequency range
        if freq < f_min * 0.9 or freq > f_max * 1.1:
            continue

        # Quality factor
        if abs(decay) > 1e-30 and freq > 0:
            Q = np.pi * freq / abs(decay)
        else:
            Q = float('inf')

        if Q < min_Q:
            continue

        amplitude = abs(amp_complex)
        phase = np.angle(amp_complex)

        # Error estimate: how well does this mode fit?
        err = 1.0 - min(abs(lam), 1.0 / abs(lam)) if abs(lam) != 0 else 1.0

        modes.append(MatrixPencilMode(
            freq=float(freq),
            decay=float(decay),
            Q=float(Q),
            amplitude=float(amplitude),
            phase=float(phase),
            error=float(abs(err)),
        ))
        mode_poles.append(s)

    # Real signals already omit negative frequencies. For complex signals,
    # omit only a negative pole whose positive conjugate is also reported.
    if not real_signal:
        positive_modes = [m for m, s in zip(modes, mode_poles) if s.imag > 0]
        modes = [
            m for m, s in zip(modes, mode_poles)
            if s.imag >= 0 or not any(
                abs(m.freq - partner.freq) <= 1e-9 * partner.freq
                and abs(m.decay - partner.decay) <= 1e-9 * 2 * np.pi * partner.freq
                for partner in positive_modes
            )
        ]

    modes.sort(key=lambda m: m.freq)

    # Sort by amplitude
    modes.sort(key=lambda m: m.amplitude, reverse=True)
    return modes[:max_modes]


def _ring_down_modes(series, dt, f_min, f_max, source_decay_time, **matrix_pencil_kwargs):
    """Fit a mean-free ring-down, capping pencil size by truncating time."""
    series = np.asarray(series).ravel()
    start = min(int(np.ceil(source_decay_time / dt)), max(len(series) - 20, 0))
    windowed = series[start:]
    if not len(windowed):
        return []
    windowed = windowed - np.mean(windowed)
    plan_kwargs = {
        "decimate": matrix_pencil_kwargs.get("decimate", "auto"),
        "pencil_parameter": matrix_pencil_kwargs.get("pencil_parameter", 0.33),
        "max_modes": matrix_pencil_kwargs.get("max_modes", 50),
    }
    factors, kept = _decimation_plan(len(windowed), dt, f_max, **plan_kwargs)
    n_keep = len(windowed)

    def capacity(n):
        columns = _pencil_columns(n, plan_kwargs["pencil_parameter"])
        return min(plan_kwargs["max_modes"] + 1, columns, n - columns)

    while kept > 10_000:
        candidate = 10_000
        for factor in reversed(factors):
            candidate = (candidate + 2 * _FIR_HALF_OUTPUT_SAMPLES) * factor
        # At fixed capacity, shortening cannot admit another FIR stage.
        # The next possible solution is this inverse bound or the last
        # length with lower capacity (which can admit additional stages).
        low, high = 10_000, n_keep
        required = capacity(n_keep)
        while low + 1 < high:
            middle = (low + high) // 2
            if capacity(middle) < required:
                low = middle
            else:
                high = middle
        n_keep = max(candidate, low)
        factors, kept = _decimation_plan(n_keep, dt, f_max, **plan_kwargs)
    return matrix_pencil(windowed[:n_keep], dt, f_min, f_max, **matrix_pencil_kwargs)


def matrix_pencil_from_probe(
    time_series: np.ndarray,
    dt: float,
    freq_range: tuple[float, float],
    *,
    source_decay_time: float = 0.0,
    **kwargs: float | int | str | bool,
) -> list[MatrixPencilMode]:
    """Extract resonances from a simulation probe via the matrix pencil method.

    Reference: Sarkar & Pereira (1995), "Using the Matrix Pencil Method to
    Estimate the Parameters of a Sum of Complex Exponentials", IEEE AP Magazine.

    Automatically windows the signal to use only the ring-down
    portion (after source has decayed).

    .. warning::

       A **lossless closed (PEC) cavity** has infinite physical Q, so the Q
       returned here is a window-length artefact, not physics — add a
       realistic loss or use an open (CPML) boundary before trusting a Q.
       See :func:`matrix_pencil` for the full note. (The frequency is fine.)

    Parameters
    ----------
    time_series : 1D array
    dt : float
    freq_range : (f_min, f_max) in Hz
    source_decay_time : float
        Time at which source has decayed (seconds).
    max_samples : ignored, deprecated
        Unfiltered subsampling was removed because it could alias modes.
    """
    if "max_samples" in kwargs:
        kwargs.pop("max_samples")
        warnings.warn(
            "max_samples is ignored because unfiltered subsampling was removed "
            "to prevent aliased modes absent from the record.",
            DeprecationWarning, stacklevel=2,
        )
    return _ring_down_modes(time_series, dt, *freq_range, source_decay_time, **kwargs)


HarminvMode = MatrixPencilMode


def harminv(
    signal: np.ndarray,
    dt: float,
    f_min: float,
    f_max: float,
    *,
    pencil_parameter: float = 0.33,
    min_Q: float = 1.0,
    max_modes: int = 50,
    sv_threshold: float = 1e-3,
    decimate: str | bool = "auto",
) -> list[MatrixPencilMode]:
    """Deprecated alias for matrix_pencil."""
    warnings.warn(
        "rfx.harminv.harminv is deprecated; use rfx.matrix_pencil (same arguments, same result). "
        "It implements the matrix pencil method, not Harminv.",
        DeprecationWarning, stacklevel=2,
    )
    return matrix_pencil(
        signal, dt, f_min, f_max, pencil_parameter=pencil_parameter,
        min_Q=min_Q, max_modes=max_modes, sv_threshold=sv_threshold, decimate=decimate,
    )


def harminv_from_probe(
    time_series: np.ndarray,
    dt: float,
    freq_range: tuple[float, float],
    *,
    source_decay_time: float = 0.0,
    **kwargs: float | int | str | bool,
) -> list[MatrixPencilMode]:
    """Deprecated alias for matrix_pencil_from_probe."""
    warnings.warn(
        "rfx.harminv.harminv_from_probe is deprecated; use rfx.matrix_pencil_from_probe "
        "(same arguments, same result). It implements the matrix pencil method, not Harminv.",
        DeprecationWarning, stacklevel=2,
    )
    return matrix_pencil_from_probe(
        time_series, dt, freq_range, source_decay_time=source_decay_time, **kwargs,
    )


def harminv_record_duration(n_samples: int, dt: float, f_max: float, *,
                           decimate: str | bool = "auto",
                           pencil_parameter: float = 0.33,
                           max_modes: int = 50) -> float:
    """Deprecated alias for matrix_pencil_record_duration."""
    warnings.warn(
        "rfx.harminv.harminv_record_duration is deprecated; use rfx.harminv.matrix_pencil_record_duration "
        "(same arguments, same result). It implements the matrix pencil method, not Harminv.",
        DeprecationWarning, stacklevel=2,
    )
    return matrix_pencil_record_duration(
        n_samples, dt, f_max, decimate=decimate,
        pencil_parameter=pencil_parameter, max_modes=max_modes,
    )

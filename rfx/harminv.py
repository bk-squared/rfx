"""Harmonic inversion for resonance extraction.

Implements the Matrix Pencil Method (MPM) to extract resonance
frequencies, decay rates, and Q factors from time-domain signals.
Much more accurate than FFT for short time series.

Reference: T. K. Sarkar and O. Pereira, "Using the Matrix Pencil
Method to Estimate the Parameters of a Sum of Complex Exponentials",
IEEE AP Magazine, Feb 1995.

Usage
-----
>>> modes = harminv(signal, dt, f_min, f_max)
>>> for m in modes:
...     print(f"f={m.freq/1e9:.4f} GHz, Q={m.Q:.0f}")
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.signal import decimate as scipy_decimate


DECIMATION_GUARD = 8
_MAX_DECIMATION_STAGE = 13
_FIR_HALF_OUTPUT_SAMPLES = 10
_MIN_PENCIL_SAMPLES = 10


def _decimation_plan(n_samples: int, dt: float, f_max: float,
                     decimate: str | bool) -> tuple[tuple[int, ...], int]:
    """Stages that leave at least the core's minimum number of valid samples."""
    if decimate not in ("auto", False):
        raise ValueError("decimate must be 'auto' or False")
    factors = []
    if decimate == "auto" and 1.0 / dt > DECIMATION_GUARD * f_max:
        target = int(1.0 / dt / (4.0 * f_max))
        for factor in _decimation_factors(target):
            kept = (n_samples + factor - 1) // factor - 2 * _FIR_HALF_OUTPUT_SAMPLES
            if kept < _MIN_PENCIL_SAMPLES:
                # Stop before consuming the record. Since factor <= 13,
                # this fallback leaves fewer than 390 samples for the SVD.
                break
            factors.append(factor)
            n_samples = kept
    return tuple(factors), n_samples


def harminv_record_duration(n_samples: int, dt: float, f_max: float, *,
                           decimate: str | bool = "auto") -> float:
    """Time between the first and last samples used to estimate the poles.

    Use this duration, rather than the unfiltered input length, for record-
    length admission or uncertainty calculations. Each zero-phase FIR stage
    excludes outputs whose filter support reaches outside its input record.
    This helper shares the estimator's sampling plan and performs no solve.
    It describes the pole fit; amplitudes still use the original input.
    """
    factors, kept = _decimation_plan(n_samples, dt, f_max, decimate)
    for factor in factors:
        dt *= factor
    return max(0, kept - 1) * dt


class HarminvMode(NamedTuple):
    """A single resonant mode extracted by Harminv."""
    freq: float       # frequency in Hz
    decay: float      # decay rate (1/s), positive = decaying
    Q: float          # quality factor
    amplitude: float  # amplitude magnitude
    phase: float      # phase in radians
    error: float      # relative error estimate


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
) -> list[HarminvMode]:
    """Extract resonant modes via the Matrix Pencil Method.

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
        Maximum modes to return.
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
        the usable record; ``harminv_record_duration`` reports its duration.
        A stage is omitted if fewer than 10 interior samples would remain.
        Set to ``False`` to retain every input sample. Decimation avoids
        redundant work because the estimator scales approximately as
        :math:`O(N^{2.7})`.

    Returns
    -------
    list of HarminvMode, sorted by amplitude (strongest first).
    """
    y = np.asarray(signal, dtype=np.complex128).ravel()
    amplitude_signal = y
    amplitude_dt = dt
    factors, _ = _decimation_plan(len(y), dt, f_max, decimate)
    for factor in factors:
        # A symmetric FIR of order 20*q has support [-10*q, +10*q]
        # in input samples. At output k (input centre k*q), precisely
        # k=10 .. ceil(N/q)-11 have their whole support inside the record.
        # Pin the order explicitly rather than depending on SciPy defaults.
        half = _FIR_HALF_OUTPUT_SAMPLES
        y = scipy_decimate(y, factor, n=2 * half * factor,
                           ftype="fir", zero_phase=True)[half:-half]
        dt *= factor

    N = len(y)
    if N < _MIN_PENCIL_SAMPLES:
        return []

    # Pencil parameter L: matrix size
    L = max(4, min(int(N * pencil_parameter), N - 2))

    # Build Hankel matrices Y0 and Y1
    # Y0[i,j] = y[i+j],     i=0..N-L-1, j=0..L-1
    # Y1[i,j] = y[i+j+1],   same indices
    M = N - L
    Y0 = np.zeros((M, L), dtype=np.complex128)
    Y1 = np.zeros((M, L), dtype=np.complex128)
    for j in range(L):
        Y0[:, j] = y[j:j+M]
        Y1[:, j] = y[j+1:j+1+M]

    # SVD of Y0 to determine rank
    U, sv, Vh = np.linalg.svd(Y0, full_matrices=False)

    # Determine effective rank from singular values
    sv_norm = sv / sv[0] if sv[0] > 0 else sv
    rank = max(1, int(np.sum(sv_norm > sv_threshold)))
    rank = min(rank, max_modes, len(sv))

    # Truncate to rank
    U1 = U[:, :rank]
    sv_trunc = sv[:rank]
    sv_trunc = np.where(sv_trunc > 1e-30, sv_trunc, 1e-30)  # guard zeros
    V1h = Vh[:rank, :]

    # Matrix pencil: Z = S_inv @ U^H @ Y1 @ V^H @ S_inv
    # Eigenvalues of Z are z_k = exp(s_k * dt)
    S_inv = np.diag(1.0 / sv_trunc)
    Z_mat = S_inv @ U1.conj().T @ Y1 @ V1h.conj().T

    if not np.all(np.isfinite(Z_mat)):
        return []

    try:
        eigenvalues = np.linalg.eigvals(Z_mat)
    except np.linalg.LinAlgError:
        return []

    modes = []
    for lam in eigenvalues:
        if not np.isfinite(lam) or abs(lam) < 1e-30:
            continue

        # z = exp(s * dt) where s = -alpha + j*omega
        s = np.log(lam) / dt
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

        # Amplitude: project signal onto this mode
        n_arr = np.arange(len(amplitude_signal))
        amplitude_lam = np.exp(s * amplitude_dt)
        basis = amplitude_lam**n_arr  # z^n at the original sample rate
        amp_complex = np.dot(amplitude_signal, basis.conj()) / np.dot(
            basis, basis.conj()
        )
        amplitude = abs(amp_complex)
        phase = np.angle(amp_complex)

        # Error estimate: how well does this mode fit?
        err = 1.0 - min(abs(lam), 1.0 / abs(lam)) if abs(lam) != 0 else 1.0

        modes.append(HarminvMode(
            freq=float(freq),
            decay=float(decay),
            Q=float(Q),
            amplitude=float(amplitude),
            phase=float(phase),
            error=float(abs(err)),
        ))

    # Deduplicate conjugate pairs (keep positive frequency, merge close modes)
    modes.sort(key=lambda m: m.freq)
    deduped = []
    for m in modes:
        if m.freq < 0:
            continue
        if deduped and abs(m.freq - deduped[-1].freq) / max(m.freq, 1) < 0.01:
            # Merge: keep the one with higher amplitude
            if m.amplitude > deduped[-1].amplitude:
                deduped[-1] = m
        else:
            deduped.append(m)

    # Sort by amplitude
    deduped.sort(key=lambda m: m.amplitude, reverse=True)
    return deduped[:max_modes]


def _decimation_factors(target: int) -> list[int]:
    """Return the largest <= target factor composed of stages no larger than 13."""
    for effective_factor in range(target, 1, -1):
        remainder = effective_factor
        factors = []
        for factor in range(_MAX_DECIMATION_STAGE, 1, -1):
            while remainder % factor == 0:
                factors.append(factor)
                remainder //= factor
        if remainder == 1:
            return factors
    return []


def harminv_from_probe(
    time_series: np.ndarray,
    dt: float,
    freq_range: tuple[float, float],
    *,
    source_decay_time: float = 0.0,
    **kwargs,
) -> list[HarminvMode]:
    """Extract resonances from a simulation probe time series.

    Automatically windows the signal to use only the ring-down
    portion (after source has decayed).

    .. warning::

       A **lossless closed (PEC) cavity** has infinite physical Q, so the Q
       returned here is a window-length artefact, not physics — add a
       realistic loss or use an open (CPML) boundary before trusting a Q.
       See :func:`harminv` for the full note. (The frequency is fine.)

    Parameters
    ----------
    time_series : 1D array
    dt : float
    freq_range : (f_min, f_max) in Hz
    source_decay_time : float
        Time at which source has decayed (seconds).
    """
    ts = np.asarray(time_series).ravel()
    start_idx = int(np.ceil(source_decay_time / dt))
    start_idx = min(start_idx, max(len(ts) - 20, 0))
    windowed = ts[start_idx:]
    windowed = windowed - np.mean(windowed)

    if len(windowed) < 20:
        return []

    f_min, f_max = freq_range

    # FFT-based bandpass filter to isolate modes in freq_range.
    # This removes DC surface charge artifacts and out-of-band modes
    # that can overwhelm the target resonance (especially at fine dx).
    fft_data = np.fft.rfft(windowed)
    fft_freqs = np.fft.rfftfreq(len(windowed), d=dt)
    bp_mask = (fft_freqs >= f_min * 0.8) & (fft_freqs <= f_max * 1.2)
    windowed = np.fft.irfft(fft_data * bp_mask, n=len(windowed))

    # Subsample for Harminv speed (SVD scales as N³)
    max_samples = kwargs.pop("max_samples", 3000)
    if len(windowed) > max_samples:
        step = len(windowed) // max_samples
        windowed = windowed[::step][:max_samples]
        dt = dt * step

    return harminv(windowed, dt, f_min, f_max, **kwargs)

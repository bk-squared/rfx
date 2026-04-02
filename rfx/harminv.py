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
) -> list[HarminvMode]:
    """Extract resonant modes via the Matrix Pencil Method.

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
        Singular value threshold for rank determination.

    Returns
    -------
    list of HarminvMode, sorted by amplitude (strongest first).
    """
    y = np.asarray(signal, dtype=np.complex128).ravel()
    N = len(y)
    if N < 10:
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
        n_arr = np.arange(N)
        basis = lam ** n_arr  # z^n
        amp_complex = np.dot(y, basis.conj()) / np.dot(basis, basis.conj())
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

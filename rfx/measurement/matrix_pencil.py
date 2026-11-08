"""Shared host matrix-pencil extraction and original-rate residue fit."""

from __future__ import annotations

import numpy as np


DECIMATION_GUARD = 8
_MAX_DECIMATION_STAGE = 13
_FIR_HALF_OUTPUT_SAMPLES = 10
_MIN_PENCIL_SAMPLES = 10


def _pencil_columns(n_samples: int, pencil_parameter: float) -> int:
    """The same clamped pencil shape for planning and the actual solve."""
    return max(4, min(int(n_samples * pencil_parameter), n_samples - 2))


def _decimation_plan(n_samples: int, dt: float, f_max: float,
                     decimate: str | bool, *, pencil_parameter: float = 0.33,
                     max_modes: int = 50) -> tuple[tuple[int, ...], int]:
    """Preserve requested pencil capacity as well as finite FIR support.

    A K-pole pencil needs at least K rows and columns. Leave one additional
    singular direction for the relative rank cut, up to max_modes poles.
    If the original record/shape cannot provide that capacity, preserve what
    it already has; preprocessing cannot supply missing information. This
    also respects the core's historical clamp for extreme pencil fractions.
    """
    if decimate not in ("auto", False):
        raise ValueError("decimate must be 'auto' or False")
    if n_samples < _MIN_PENCIL_SAMPLES:
        return (), n_samples
    columns = _pencil_columns(n_samples, pencil_parameter)
    capacity = min(max_modes + 1, columns, n_samples - columns)
    factors = []
    if decimate == "auto" and 1.0 / dt > DECIMATION_GUARD * f_max:
        target = int(1.0 / dt / (4.0 * f_max))
        for factor in _decimation_factors(target):
            kept = (n_samples + factor - 1) // factor - 2 * _FIR_HALF_OUTPUT_SAMPLES
            columns = _pencil_columns(kept, pencil_parameter)
            if (kept < _MIN_PENCIL_SAMPLES
                    or min(columns, kept - columns) < capacity):
                # Default p=.33/K=50 needs 155 retained samples. A rejected
                # stage (q<=13) therefore leaves at most 2262 inputs to the
                # core. Other requested pencil shapes/ranks change that bound.
                break
            factors.append(factor)
            n_samples = kept
    return tuple(factors), n_samples


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


def _decimate(Y: np.ndarray, factors, *, decimate_fn) -> np.ndarray:
    """harminv's FIR stage, per channel and per factor. ``Y (n, C)`` -> ``(Nd, C)``."""
    half = int(_FIR_HALF_OUTPUT_SAMPLES)
    cols = []
    for ch in range(Y.shape[1]):
        y = Y[:, ch]
        for q in factors:
            y = decimate_fn(y, q, n=2 * half * q, ftype="fir",
                                        zero_phase=True)[half:-half]
        cols.append(y)
    return np.stack(cols, axis=1)


def _pencil(Yd: np.ndarray, sv_rel: float, unit_tol: float, *,
            pencil_parameter: float, max_rank=None):
    """Side-by-side stacked matrix pencil on ``Yd (Nd, C)``."""
    from numpy.lib.stride_tricks import sliding_window_view

    Nd, C = Yd.shape
    L = int(_pencil_columns(Nd, pencil_parameter))
    M = Nd - L
    if M < 1 or L < 1:
        raise ValueError(
            f"a window of {Nd} decimated samples is too short for a matrix "
            "pencil; use a longer record or an earlier window_start")
    rms = np.sqrt(np.mean(np.abs(Yd) ** 2, axis=0))
    if np.any(rms <= 0) or not np.all(np.isfinite(rms)):
        raise ValueError(
            f"a channel has zero or non-finite RMS over the window ({rms}); "
            "the identification channel saw no ringing to identify")
    b0, b1 = [], []
    for ch in range(C):
        y = Yd[:, ch].astype(np.complex128) / rms[ch]
        b0.append(sliding_window_view(y, L)[:M])        # [i, j] = y[i + j]
        b1.append(sliding_window_view(y[1:], L)[:M])    # [i, j] = y[i + j + 1]
    Y0 = np.concatenate(b0, axis=1)
    Y1 = np.concatenate(b1, axis=1)
    U, sv, Vh = np.linalg.svd(Y0, full_matrices=False)
    if not sv[0] > 0:
        raise ValueError("the stacked Hankel matrix of the window is zero")
    rank = max(int(np.sum(sv > sv_rel * sv[0])), 1)
    if max_rank is not None:
        rank = min(rank, max_rank)
    Ur, sr, Vr = U[:, :rank], sv[:rank], Vh[:rank].conj().T
    Z = (Ur.conj().T @ Y1 @ Vr) / sr[:, None]
    lam = np.linalg.eigvals(Z)
    valid = np.isfinite(lam) & (np.abs(lam) > 0)
    n_invalid = int(np.sum(~valid))
    lam = lam[valid]
    growing = np.abs(lam) > 1.0 + unit_tol
    return lam[~growing], rank, lam[growing], n_invalid, sv


def _to_original_rate(lam_dec: np.ndarray, D: int, dt: float, guard):
    """Method step 4 and the map ``s = log(lambda_dec) / (D dt)``.

    Returns ``(s, n_dropped_by_guard, kept_decimated_eigenvalues)``. With ``D > 1`` the principal log's
    cut (``arg lambda_dec = +-pi``, the decimated Nyquist frequency) is where a
    pole's original-rate frequency jumps by ``1 / (D dt)``; the guard drops
    every pole within ``(1 - guard) pi`` of it. With ``D == 1`` the tail uses
    ``exp(s dt) = lambda`` itself, so the branch does not matter.
    """
    lam_dec = np.asarray(lam_dec, dtype=np.complex128)
    n_guard = 0
    if guard is not None and D > 1:
        keep = np.abs(np.angle(lam_dec)) <= float(guard) * np.pi
        n_guard = int(np.sum(~keep))
        lam_dec = lam_dec[keep]
    return np.log(lam_dec) / (int(D) * float(dt)), n_guard, lam_dec


def _fit_residues(Y_win: np.ndarray, s: np.ndarray, dt: float, *,
                  column_reference: bool = False) -> np.ndarray:
    """Fit bounded exponential columns and return start- or column-referenced residues."""
    if s.size == 0:
        return np.zeros((0, Y_win.shape[1]), dtype=np.complex128)
    m = np.arange(Y_win.shape[0], dtype=np.float64) * float(dt)
    growing = s.real > 0
    exponents = np.outer(m, s)
    exponents[:, growing] = np.outer(m - m[-1], s[growing])
    basis = np.exp(exponents)
    norms = np.linalg.norm(basis, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    coef = np.linalg.lstsq(basis / norms, Y_win.astype(np.complex128), rcond=None)[0]
    residues = coef / norms[:, None]
    if not column_reference:
        residues[growing] *= np.exp(-m[-1] * s[growing])[:, None]
    return residues



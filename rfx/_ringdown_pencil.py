"""Identification numerics used by :mod:`rfx.ringdown`."""

from __future__ import annotations

from importlib import import_module
import math
from numbers import Number

import numpy as np

# Import the module rather than rfx's re-exported harminv function.
_harminv = import_module("rfx.harminv")

# Mac per-stage peaks / formula, maxima 1.780 and 1.626, plus 5 %:
# records 20261012-pencil-stage-peaks, decision g.
_STAGE1_FACTOR = 1.87
_STAGE2_FACTOR = 1.71


class IdentificationMemoryRefused(ValueError):
    """An identification stage would exceed the configured memory bound."""

    __module__ = "rfx.ringdown"

    def __init__(self, stage, needed_bytes, bound_bytes, *, n_win, Nd, C, L, M, k, r):
        self.stage = stage
        self.needed_bytes = needed_bytes
        self.bound_bytes = bound_bytes
        self.n_win, self.Nd, self.C = n_win, Nd, C
        self.L, self.M, self.k, self.r = L, M, k, r
        name = "pencil" if stage == 1 else "residue fit"
        order = f"k={k}" if stage == 1 else f"rank={r}"
        super().__init__(
            f"identification would need {needed_bytes / 2**30:.2f} GiB "
            f"at stage {stage} ({name}; n_win={n_win}, Nd={Nd}, C={C}, L={L}, "
            f"{order}), above the {bound_bytes / 2**30:.2f} GiB bound; "
            "shorten the identification window, identify on fewer channels, "
            "or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES")

    def __reduce__(self):
        dimensions = {name: getattr(self, name) for name in ("n_win", "Nd", "C", "L", "M", "k", "r")}
        return (_restore_memory_refusal, (self.stage, self.needed_bytes, self.bound_bytes, dimensions))


def _restore_memory_refusal(stage, needed_bytes, bound_bytes, dimensions):
    return IdentificationMemoryRefused(stage, needed_bytes, bound_bytes, **dimensions)


def _validate_memory_bound(bound):
    """Keep invalid public settings on the ValueError verdict path."""
    try:
        valid = (isinstance(bound, Number)
                 and not isinstance(bound, (bool, np.bool_, complex, np.complexfloating))
                 and 0 < bound <= math.inf)
    except (TypeError, ArithmeticError):
        valid = False
    if not valid:
        raise ValueError("rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES must be a positive finite number or math.inf")


def _check_identification_memory(n_win, shape, *, r=None, bound, pencil_parameter):
    """Refuse before the pencil or original-rate residue fits allocate."""
    _validate_memory_bound(bound)
    Nd, C = shape
    L = int(_harminv._pencil_columns(Nd, pencil_parameter))
    M = Nd - L
    k = min(M, L * C)
    stage = 1 if r is None else 2
    needed = (_STAGE1_FACTOR * 16 * (3*M*L*C + M*k + 2*k*L*C + 4*k*k)
              if r is None else _STAGE2_FACTOR * 16 * 5 * n_win * r)
    if needed > bound:
        raise IdentificationMemoryRefused(
            stage, needed, bound,
            n_win=n_win, Nd=Nd, C=C, L=L, M=M, k=k, r=r)


def _check_run_identification_memory(plan):
    """The fixed run's main pencil, before the lane takes any step."""
    from rfx import ringdown as rd

    bound = rd.IDENTIFY_MEMORY_BOUND_BYTES
    _validate_memory_bound(bound)
    bins = plan.bins
    # Default and graded-lane result bins are the accumulator's float32 bins.
    peak = (float(np.float32(plan.sim._freq_max)) if bins is None else
            float(np.max(np.asarray(bins, dtype=np.float32 if plan.lane == "graded" else np.float64))))
    n_win = plan.n_steps - plan.n_start
    _, Nd = rd.decimation_plan(n_win, float(plan.dt), max(plan.freq_max, peak))
    C = 2 * len(plan.wire_entries) + len(plan.identification_keys)
    rd._check_identification_memory(n_win, (Nd, C), bound=bound, pencil_parameter=rd.PENCIL_PARAMETER)


def _witness_identification(identify_window):
    """A long-window refusal is a failed witness, preserving the main result."""
    def identify_long_window(window):
        try:
            return identify_window(window)
        except IdentificationMemoryRefused as exc:
            raise ValueError(str(exc)) from exc
    return identify_long_window


def decimation_plan(n_samples: int, dt: float, freq_max: float) -> tuple:
    """harminv's plan for a window of ``n_samples``: ``(factors, kept samples)``."""
    from rfx.ringdown import HARMINV_MAX_MODES, PENCIL_PARAMETER

    factors, kept = _harminv._decimation_plan(
        int(n_samples), float(dt), float(freq_max), "auto",
        pencil_parameter=PENCIL_PARAMETER, max_modes=HARMINV_MAX_MODES)
    return tuple(int(q) for q in factors), int(kept)


def _decimate(Y: np.ndarray, factors) -> np.ndarray:
    """harminv's FIR stage, per channel and per factor. ``Y (n, C)`` -> ``(Nd, C)``."""
    half = int(_harminv._FIR_HALF_OUTPUT_SAMPLES)
    cols = []
    for ch in range(Y.shape[1]):
        y = Y[:, ch]
        for q in factors:
            y = _harminv.scipy_decimate(y, q, n=2 * half * q, ftype="fir",
                                        zero_phase=True)[half:-half]
        cols.append(y)
    return np.stack(cols, axis=1)


def _pencil(Yd: np.ndarray, sv_rel: float, unit_tol: float):
    """Side-by-side stacked matrix pencil on ``Yd (Nd, C)``."""
    from numpy.lib.stride_tricks import sliding_window_view
    from rfx.ringdown import PENCIL_PARAMETER

    Nd, C = Yd.shape
    L = int(_harminv._pencil_columns(Nd, PENCIL_PARAMETER))
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

    Returns ``(s, n_dropped_by_guard)``. With ``D > 1`` the principal log's
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
    return np.log(lam_dec) / (int(D) * float(dt)), n_guard


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

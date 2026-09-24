"""Ring-down completion of wire-port S-parameters (issue #1254).

Physics
-------
A resonator hit by a short pulse keeps ringing long after the pulse has
passed. A DFT of a record cut while it still rings misses the part the record
has not reached, and every S-parameter built from those DFTs is wrong by that
part. After the source has turned off, the discrete update is one fixed
linear map, so every port voltage and port current is a finite sum of the same
damped oscillations, ``y_n = sum_k r_k lambda_k**n``. Identify the slow ones
on the last part of the record and the unrecorded tail of each spectrum is a
closed form::

    Y_inf(f) = Y_T(f) + dt * sum_k r_k lambda_k**(N+1) z**(N+1) / (1 - lambda_k z)
    z = exp(-j 2 pi f dt),   N = index of the LAST recorded sample,
    Y_T(f) = dt * sum_{n=0..N} y_n z**n.

Sample ``n`` is paired with the time ``n dt``, as rfx's port accumulators pair
``step`` with ``t = step * dt``. ``1 - lambda_k z`` is evaluated as
``-expm1(s_k dt - j 2 pi f dt)`` so a slow pole next to a bin keeps its
digits.

The method (every rule fixed here, none tuned per board)
-------------------------------------------------------
1. Window: samples ``[n_start, n_stop)`` of the record.
2. Anti-alias decimation of every channel with ``rfx.harminv``'s own plan and
   FIR stage (``_decimation_plan`` toward ``4 x freq_max`` sampling, harminv's
   default pencil fraction and model capacity; ``scipy.signal.decimate`` with
   a zero-phase FIR of order ``20 q``, the ten outputs at each end dropped).
   The reference is the run's ``freq_max``: a fixed reference below the band
   filters the band's own modes out before the pencil sees them.
3. One multichannel matrix pencil: each channel's Hankel pair, scaled by that
   channel's RMS over the decimated window, stacked side by side; rank = number
   of singular values above ``sv_rel`` of the largest; eigenvalues with
   ``|lambda| > 1 + unit_tol`` are discarded and counted, eigenvalues that are
   not finite or exactly zero are dropped and counted.
4. Transition-band guard: when the window was decimated (``D > 1``), a
   decimated pole with ``|arg lambda| > guard * pi`` is dropped before the
   residue fit. Such a pole sits in the anti-alias filter's transition band
   (no in-band content) and its original-rate frequency lies on the branch cut
   of ``log(lambda) / (D dt)``, so which side it maps to is decided by rounding
   and the completed spectrum would jump with it.
5. Residues per channel by linear least squares at the ORIGINAL rate over the
   window, poles fixed, columns normalised (harminv's amplitude fit).
6. Completion by the closed form above, per channel.

The two-window witness ``W2`` identifies the poles a second time on the
shorter window ``[n_start, split * N)`` and completes the SAME record with
them; the largest difference of the two answers bounds how much the answer
still depends on the window. On the research boards it tracked the actual
error within a factor 10 (rfx #1254).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from importlib import import_module
from typing import NamedTuple

import numpy as np

# The MODULE: ``rfx/__init__.py`` re-exports the function ``harminv`` under
# the same name, so ``from rfx import harminv`` depends on import order.
_harminv = import_module("rfx.harminv")

# harminv's own defaults, read from its signature rather than re-typed, so the
# decimation plan and pencil shape here follow the estimator they came from.
_HARMINV_DEFAULTS = {
    k: v.default for k, v in inspect.signature(_harminv.harminv).parameters.items()
    if v.default is not inspect.Parameter.empty
}
PENCIL_PARAMETER = float(_HARMINV_DEFAULTS["pencil_parameter"])
HARMINV_MAX_MODES = int(_HARMINV_DEFAULTS["max_modes"])

__all__ = [
    "RingdownSpec",
    "RingdownModel",
    "RingdownPole",
    "RingdownWitness",
    "RingdownReport",
    "RingdownResult",
    "identify",
    "plain_dft",
    "tail_dft",
    "complete_spectra",
    "two_window_witness",
]


# ---------------------------------------------------------------------------
# Public records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RingdownSpec:
    """Opt-in ring-down completion for ``Simulation.run(..., ringdown=...)``.

    Parameters
    ----------
    window_start : float
        Start of the identification window as a fraction of the record; the
        window is ``[window_start * T, T]``. The source must be off there.
    guard : float
        Transition-band guard, as a fraction of the decimated Nyquist
        frequency (method step 4).
    witness_tol : float
        Bar of the two-window witness ``W2``: the largest ``|S_A - S_B|``
        over every bin and entry, in units of ``|S|`` (an S-parameter of a
        passive port is at most 1, so this is an absolute difference).
    freq_max : float or None
        Decimation reference in Hz; ``None`` uses the run's ``freq_max``.
    split : float
        End of the second, shorter window of ``W2`` as a fraction of the
        record: ``[window_start * T, split * T]``.
    sv_rel, unit_tol : float
        Pencil rank threshold and unit-circle tolerance (method step 3).
    passivity_tol : float
        Bar of the passivity witness: ``max |S_kk| <= 1 + passivity_tol``.
    source_off_tol : float
        Every source's waveform over the window must stay at or below this
        fraction of its peak over the record; ``run()`` refuses otherwise.
    """

    window_start: float = 0.5
    guard: float = 0.9
    witness_tol: float = 1.0e-3
    freq_max: float | None = None
    split: float = 0.9
    sv_rel: float = 1.0e-6
    unit_tol: float = 1.0e-6
    passivity_tol: float = 1.0e-3
    source_off_tol: float = 1.0e-6

    def __post_init__(self):
        if not (0.0 < float(self.window_start) < float(self.split) < 1.0):
            raise ValueError(
                "RingdownSpec needs 0 < window_start < split < 1 (the two "
                f"windows [window_start, 1] and [window_start, split] of the "
                f"record); got window_start={self.window_start}, "
                f"split={self.split}")
        if not (0.0 < float(self.guard) <= 1.0):
            raise ValueError(
                f"RingdownSpec.guard must lie in (0, 1], got {self.guard}")
        for name in ("witness_tol", "sv_rel", "unit_tol", "passivity_tol",
                     "source_off_tol"):
            v = float(getattr(self, name))
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"RingdownSpec.{name} must be positive, got {v}")
        if self.freq_max is not None and not (float(self.freq_max) > 0.0):
            raise ValueError(
                f"RingdownSpec.freq_max must be positive or None, got {self.freq_max}")


class RingdownPole(NamedTuple):
    """One identified pole: signed frequency, loaded Q, decay rate, ``|lambda|`` per step."""

    f_hz: float
    q: float
    decay_per_s: float
    abs_lambda: float


class RingdownWitness(NamedTuple):
    """One check of a completed number: the measured value, its bar, pass/fail, the rule."""

    name: str
    value: float
    bar: float
    ok: bool
    rule: str


@dataclass(frozen=True)
class RingdownModel:
    """Poles and residues identified on a window of a multichannel record.

    ``s`` are the poles at the ORIGINAL step (1/s); ``c (K, C)`` the residues
    referred to sample ``n_ref`` (the window's first sample, so a fast pole's
    residue is never extrapolated back over the whole record).
    """

    s: np.ndarray
    c: np.ndarray
    n_ref: int
    dt: float
    freq_max: float
    guard: float | None
    factors: tuple
    D: int
    rank: int
    n_window: int
    n_decimated: int
    n_growing_discarded: int
    n_invalid: int
    n_guard_dropped: int
    singular_values: np.ndarray

    @property
    def lam(self) -> np.ndarray:
        """Per-step eigenvalues ``exp(s dt)``."""
        return np.exp(self.s * self.dt)

    def predict(self, n) -> np.ndarray:
        """Model samples at absolute indices ``n`` -> ``(len(n), C)``."""
        n = np.asarray(n, dtype=np.float64)
        return np.exp(np.outer((n - self.n_ref) * self.dt, self.s)) @ self.c

    def poles(self) -> tuple:
        """Every kept pole as a :class:`RingdownPole`, sorted by frequency."""
        f = self.s.imag / (2.0 * np.pi)
        alpha = -self.s.real
        lam = np.abs(self.lam)
        rows = []
        for k in np.argsort(f, kind="stable"):
            q = (math.pi * abs(float(f[k])) / float(alpha[k])
                 if alpha[k] != 0 else math.inf)
            rows.append(RingdownPole(float(f[k]), float(q), float(alpha[k]),
                                     float(lam[k])))
        return tuple(rows)


@dataclass(frozen=True)
class RingdownReport:
    """What a completed S-matrix rests on: the window, the poles, the witnesses."""

    n_record: int
    dt: float
    window_steps: tuple
    window_s: tuple
    short_window_steps: tuple
    freq_max_hz: float
    decimation_factors: tuple
    rank: int
    n_kept: int
    n_growing_discarded: int
    n_invalid: int
    n_guard_dropped: int
    short_rank: int
    short_n_kept: int
    poles: tuple
    witnesses: tuple
    growing_pole_rule: str

    @property
    def ok(self) -> bool:
        """True when every witness reads ok."""
        return all(w.ok for w in self.witnesses)

    def witness(self, name: str) -> RingdownWitness:
        for w in self.witnesses:
            if w.name == name:
                return w
        raise KeyError(f"no witness named {name!r}; have "
                       f"{[w.name for w in self.witnesses]}")

    def summary(self) -> str:
        """A few lines for a log: window, poles kept, every witness."""
        lines = [
            f"ring-down completion: record {self.n_record} steps "
            f"({self.n_record * self.dt * 1e9:.4g} ns), window "
            f"[{self.window_s[0] * 1e9:.4g}, {self.window_s[1] * 1e9:.4g}] ns, "
            f"decimation {self.decimation_factors or (1,)}, rank {self.rank}, "
            f"{self.n_kept} poles kept ({self.n_guard_dropped} dropped by the "
            f"guard, {self.n_growing_discarded} growing discarded)"]
        for w in self.witnesses:
            lines.append(f"  {w.name}: {w.value:.3e} (bar {w.bar:.3e}) "
                         f"{'ok' if w.ok else 'FAILED'}")
        return "\n".join(lines)


class RingdownResult(NamedTuple):
    """``Result.ringdown``: the completed S-matrix, its bins and its report.

    ``s_params`` has the shape and dtype of ``Result.s_params`` and is
    evaluated at the same bins (``freqs`` is ``Result.freqs``; rfx accumulates
    at those bins rounded to float32, and so does the completion).
    """

    s_params: np.ndarray
    freqs: np.ndarray
    report: RingdownReport


# ---------------------------------------------------------------------------
# DFTs
# ---------------------------------------------------------------------------

def _as_2d(series) -> tuple[np.ndarray, bool]:
    y = np.asarray(series)
    if y.ndim == 1:
        return y[:, None], True
    if y.ndim != 2:
        raise ValueError(f"series must be (n,) or (n, C), got shape {y.shape}")
    return y, False


def plain_dft(series, dt, freqs, *, chunk: int = 4096) -> np.ndarray:
    """``dt * sum_n y_n exp(-j 2 pi f n dt)`` over the whole ``series``.

    ``series`` is ``(n,)`` or ``(n, C)``; returns ``(nf,)`` or ``(nf, C)``,
    complex128, float64 kernel.
    """
    Y, one_d = _as_2d(series)
    freqs = np.asarray(freqs, dtype=np.float64)
    w = -2.0 * np.pi * freqs * float(dt)
    acc = np.zeros((freqs.size, Y.shape[1]), dtype=np.complex128)
    for n0 in range(0, Y.shape[0], chunk):
        n1 = min(n0 + chunk, Y.shape[0])
        nn = np.arange(n0, n1, dtype=np.float64)
        acc += np.exp(1j * np.outer(w, nn)) @ Y[n0:n1].astype(np.complex128)
    out = acc * float(dt)
    return out[:, 0] if one_d else out


def tail_dft(model: RingdownModel, n_last: int, freqs) -> np.ndarray:
    """The unrecorded part of each spectrum: samples ``n_last + 1 .. inf``.

    ``dt * sum_k c_k lambda_k**(N+1-n_ref) z**(N+1) / (1 - lambda_k z)`` with
    ``N = n_last`` and ``1 - lambda_k z = -expm1(s_k dt - j 2 pi f dt)``.
    Returns ``(nf, C)``.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(model.s, dtype=np.complex128)
    c = np.asarray(model.c, dtype=np.complex128)
    dt = float(model.dt)
    if s.size == 0:
        return np.zeros((freqs.size, c.shape[1]), dtype=np.complex128)
    n1 = int(n_last) + 1
    wdt = 2.0 * np.pi * freqs * dt                                  # (nf,)
    amp = np.exp(s * dt * (n1 - int(model.n_ref)))[:, None] * c    # (K, C)
    zn1 = np.exp(-1j * wdt * n1)                                    # (nf,)
    one_minus = -np.expm1(s[None, :] * dt - 1j * wdt[:, None])      # (nf, K)
    return dt * ((zn1[:, None] / one_minus) @ amp)


def complete_spectra(series, dt, freqs, model: RingdownModel, n_record: int) -> np.ndarray:
    """Plain DFT of samples ``0 .. n_record - 1`` plus the closed-form tail after them."""
    Y, one_d = _as_2d(series)
    n_record = int(n_record)
    if not (0 < n_record <= Y.shape[0]):
        raise ValueError(f"n_record={n_record} outside a series of {Y.shape[0]} samples")
    if float(dt) != float(model.dt):
        raise ValueError(f"dt={dt} differs from the model's dt={model.dt}")
    out = plain_dft(Y[:n_record], dt, freqs) + tail_dft(model, n_record - 1, freqs)
    return out[:, 0] if one_d else out


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

def decimation_plan(n_samples: int, dt: float, freq_max: float) -> tuple:
    """harminv's plan for a window of ``n_samples``: ``(factors, kept samples)``."""
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
            "the port saw no ringing to identify")
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
    return lam[~growing], rank, int(np.sum(growing)), n_invalid, sv


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


def _fit_residues(Y_win: np.ndarray, s: np.ndarray, dt: float) -> np.ndarray:
    """Least-squares residues at the original rate, poles fixed, referred to the window start."""
    if s.size == 0:
        return np.zeros((0, Y_win.shape[1]), dtype=np.complex128)
    m = np.arange(Y_win.shape[0], dtype=np.float64) * float(dt)
    basis = np.exp(np.outer(m, s))
    norms = np.linalg.norm(basis, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    coef = np.linalg.lstsq(basis / norms, Y_win.astype(np.complex128), rcond=None)[0]
    return coef / norms[:, None]


def identify(series, dt, n_start, n_stop, *, freq_max, guard=0.9,
             sv_rel=1.0e-6, unit_tol=1.0e-6) -> RingdownModel:
    """Poles and residues of samples ``[n_start, n_stop)`` of ``series (n, C)``.

    Method steps 2-5 of the module docstring. ``guard=None`` turns the
    transition-band guard off (for tests of the guard itself).
    """
    Y, _ = _as_2d(series)
    n_start, n_stop = int(n_start), int(n_stop)
    if not (0 <= n_start < n_stop <= Y.shape[0]):
        raise ValueError(f"window [{n_start}, {n_stop}) outside {Y.shape[0]} samples")
    if guard is not None and not (0.0 < float(guard) <= 1.0):
        raise ValueError(f"guard must lie in (0, 1] or be None, got {guard}")
    dt, freq_max = float(dt), float(freq_max)
    win = Y[n_start:n_stop]
    win = win.astype(np.float64 if np.isrealobj(win) else np.complex128)
    factors, _kept = decimation_plan(win.shape[0], dt, freq_max)
    Yd = _decimate(win, factors) if factors else win
    D = int(np.prod(factors)) if factors else 1
    lam_dec, rank, n_growing, n_invalid, sv = _pencil(Yd, float(sv_rel), float(unit_tol))
    s, n_guard = _to_original_rate(lam_dec, D, dt, guard)
    c = _fit_residues(win, s, dt)
    return RingdownModel(
        s=s, c=c, n_ref=n_start, dt=dt, freq_max=freq_max,
        guard=None if guard is None else float(guard), factors=factors, D=D,
        rank=int(rank), n_window=int(win.shape[0]), n_decimated=int(Yd.shape[0]),
        n_growing_discarded=int(n_growing), n_invalid=int(n_invalid),
        n_guard_dropped=n_guard, singular_values=sv)


class TwoWindowWitness(NamedTuple):
    """``W2`` and the two completions it compares."""

    value: float
    spectra: np.ndarray
    spectra_short: np.ndarray
    model: RingdownModel
    model_short: RingdownModel


def two_window_witness(series, dt, freqs, n_record, n_start, *, freq_max,
                       split=0.9, guard=0.9, sv_rel=1.0e-6, unit_tol=1.0e-6,
                       observable=None) -> TwoWindowWitness:
    """Complete ``series[:n_record]`` from ``[n_start, n_record)`` and from ``[n_start, split*n_record)``.

    Both completions add their tail after the SAME last sample
    (``n_record - 1``), so the difference is the window's influence alone.
    ``observable`` maps a ``(nf, C)`` spectra array to the quantity compared
    (default: the spectra themselves); ``value`` is the largest absolute
    difference of the two observables.
    """
    n_record, n_start = int(n_record), int(n_start)
    n_split = int(round(float(split) * n_record))
    if not (n_start < n_split < n_record):
        raise ValueError(
            f"the short window [{n_start}, {n_split}) must end inside the record "
            f"of {n_record} samples")
    Y, _ = _as_2d(series)
    Y = Y[:n_record]
    kw = dict(freq_max=freq_max, guard=guard, sv_rel=sv_rel, unit_tol=unit_tol)
    model = identify(Y, dt, n_start, n_record, **kw)
    model_short = identify(Y, dt, n_start, n_split, **kw)
    plain = plain_dft(Y, dt, freqs)
    spectra = plain + tail_dft(model, n_record - 1, freqs)
    spectra_short = plain + tail_dft(model_short, n_record - 1, freqs)
    obs = (lambda a: a) if observable is None else observable
    value = float(np.max(np.abs(np.asarray(obs(spectra), dtype=np.complex128)
                                - np.asarray(obs(spectra_short), dtype=np.complex128))))
    return TwoWindowWitness(value, spectra, spectra_short, model, model_short)


def growing_poles(model: RingdownModel, record_s: float, unit_tol: float):
    """Kept poles with ``|lambda| > 1`` per step, split into (growing, exempt).

    Exempt: a pole within rounding of zero frequency -- ``|f| < 1 / record``
    (the record cannot tell it from DC) and ``|lambda| - 1 < unit_tol`` per
    step. A static field left behind in a closed box identifies as such a
    pole (rfx #1254, the Q 533 cavity).
    """
    grow, exempt = [], []
    for p in model.poles():
        if not p.abs_lambda > 1.0:
            continue
        if abs(p.f_hz) < 1.0 / float(record_s) and p.abs_lambda - 1.0 < float(unit_tol):
            exempt.append(p)
        else:
            grow.append(p)
    return tuple(grow), tuple(exempt)


GROWING_POLE_RULE = (
    "a kept pole with |lambda| > 1 per step fails the check unless |f| < "
    "1/record length and |lambda| - 1 < unit_tol (a zero-frequency pole within "
    "rounding of the unit circle)")

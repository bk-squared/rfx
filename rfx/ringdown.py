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

The error witness ``WE`` identifies the poles a second time on the window of
double span ``[n_start / 2, N)`` and completes the SAME record with them; the
largest difference of the two answers is the witness. On the research boards
it read the actual error within 0.84-3.2x wherever the record resolves the
structure (rfx #1254, R-i). It needs the sources off over its window too; when
they are not, the two-window witness ``W2`` (the shorter window
``[n_start, split * N)``, which read the actual error 0.003-22x there) is
judged in its place. Otherwise ``W2`` is reported, not judged.

What this module does inside ``Simulation.run(..., ringdown=RingdownSpec())``
--------------------------------------------------------------------------
For every wire port (``add_port(..., extent=...)``, impedance > 0) it places
read-only probes on the port's Ez (Ex, Ey) edges and on the H samples of its
Ampere loop before the run, and removes them from everything the run returns.
After the run it reads the port's REALIZED cells, metrics and DFT
accumulators from the run itself, rebuilds the port voltage and current step
by step, and checks that the rebuilt series, accumulated the way the run
accumulates them, reproduce the run's own accumulators to within
``W0_ULP_BAR`` ULPs (W0). It then completes the spectra and assembles the
S-matrix with the lane's own assembly (the graded lane's
``rfx.nonuniform._assemble_nu_result``, the uniform lane's
``rfx.probes.probes.driven_port_reflection``). When W0, W1 or the pole
identification fails, the run is returned as it is, uncompleted, with the
failed check in ``Result.ringdown.report`` and a warning.
"""

from __future__ import annotations

import inspect
import math
import warnings
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
    "double_span_witness",
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
        window is ``[window_start * T, T]``. The source must be off there
        (and over ``[window_start / 2 * T, T]`` for ``WE`` to be formed).
    guard : float
        Transition-band guard, as a fraction of the decimated Nyquist
        frequency (method step 4).
    witness_tol : float
        Bar of the error witness ``WE`` (``W2`` when ``WE`` cannot be
        formed): the largest ``|S_A - S_B|`` over every bin and entry, in
        units of ``|S|`` (an S-parameter of a passive port is at most 1, so
        this is an absolute difference).
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
    """One identified pole.

    Signed frequency, loaded Q, amplitude decay rate, ``|lambda|`` per step,
    and its amplitude: the largest ``|residue|`` over the channels at the
    window's first sample, relative to that channel's RMS over the window.
    """

    f_hz: float
    q: float
    decay_per_s: float
    abs_lambda: float
    amplitude: float = math.nan


class RingdownWitness(NamedTuple):
    """One check of a completed number: the measured value, its bar, pass/fail, the rule.

    ``judged`` is False for a witness reported as information: its ``ok``
    is its own reading against its bar and does not enter ``report.ok``.
    """

    name: str
    value: float
    bar: float
    ok: bool
    rule: str
    note: str = ""
    judged: bool = True


@dataclass(frozen=True)
class RingdownModel:
    """Poles and residues identified on a window of a multichannel record.

    ``s`` are the kept poles at the ORIGINAL step (1/s); ``c (K, C)`` their
    residues referred to sample ``n_ref`` (the window's first sample, so a
    fast pole's residue is never extrapolated back over the whole record).
    ``amplitude`` is each kept pole's largest ``|c| / RMS`` over the channels.
    ``s_growing`` are the poles the pencil discarded as growing, and
    ``growing_amplitude`` each one's size at the window's last sample,
    ``|r_k| |lambda_k|**(n_end - n_ref) / RMS`` (largest over the channels),
    from a residue fit made jointly with the kept poles.
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
    amplitude: np.ndarray = None
    s_growing: np.ndarray = None
    growing_amplitude: np.ndarray = None
    window_rms: np.ndarray = None

    @property
    def lam(self) -> np.ndarray:
        """Per-step eigenvalues ``exp(s dt)``."""
        return np.exp(self.s * self.dt)

    def predict(self, n) -> np.ndarray:
        """Model samples at absolute indices ``n`` -> ``(len(n), C)``."""
        n = np.asarray(n, dtype=np.float64)
        return np.exp(np.outer((n - self.n_ref) * self.dt, self.s)) @ self.c

    def poles(self) -> tuple:
        """Every kept pole as a :class:`RingdownPole`, largest amplitude first
        (by frequency when no amplitude is known)."""
        f = self.s.imag / (2.0 * np.pi)
        alpha = -self.s.real
        lam = np.abs(self.lam)
        amp = (np.full(self.s.size, math.nan) if self.amplitude is None
               else np.asarray(self.amplitude, dtype=np.float64))
        order = (np.argsort(f, kind="stable") if np.all(np.isnan(amp))
                 else np.argsort(-np.nan_to_num(amp, nan=-1.0), kind="stable"))
        rows = []
        for k in order:
            q = (math.pi * abs(float(f[k])) / float(alpha[k])
                 if alpha[k] != 0 else math.inf)
            rows.append(RingdownPole(float(f[k]), float(q), float(alpha[k]),
                                     float(lam[k]), float(amp[k])))
        return tuple(rows)


@dataclass(frozen=True)
class RingdownReport:
    """What a completed S-matrix rests on: the window, the poles, the witnesses.

    When a check that decides whether the completion can be trusted at all
    fails (W0, W1, the identification), nothing is completed: ``failure``
    names the check and its number, ``completed`` is False, and the fields that
    need a model stay ``None``. ``s_accumulator_roundoff`` is information, not
    a check: ``max |S of the float64 DFT of the rebuilt record - Result.s_params|``,
    the run's own complex64 accumulator round-off in ``Result.s_params``,
    which grows with the record.
    """

    n_record: int
    dt: float
    window_steps: tuple
    window_s: tuple
    short_window_steps: tuple
    freq_max_hz: float
    witnesses: tuple
    failure: str | None = None
    decimation_factors: tuple | None = None
    rank: int | None = None
    n_kept: int | None = None
    n_growing_discarded: int | None = None
    n_invalid: int | None = None
    n_guard_dropped: int | None = None
    short_rank: int | None = None
    short_n_kept: int | None = None
    long_window_steps: tuple | None = None
    long_rank: int | None = None
    long_n_kept: int | None = None
    poles: tuple = ()
    growing_pole_rule: str = ""
    w0_ulps: dict | None = None
    s_accumulator_roundoff: float | None = None
    tail_share: float | None = None
    slowest_decay_over_window: float | None = None
    discarded_growth_max: float | None = None
    discarded_growth_f_hz: float | None = None

    @property
    def completed(self) -> bool:
        """True when the completed S-matrix was produced."""
        return self.failure is None

    @property
    def ok(self) -> bool:
        """True when the completion was produced and every judged witness reads ok."""
        return self.completed and all(w.ok for w in self.witnesses if w.judged)

    def witness(self, name: str) -> RingdownWitness:
        for w in self.witnesses:
            if w.name == name:
                return w
        raise KeyError(f"no witness named {name!r}; have "
                       f"{[w.name for w in self.witnesses]}")

    def summary(self) -> str:
        """A few lines for a log: window, poles kept, every witness."""
        head = (f"ring-down completion: record {self.n_record} steps "
                f"({self.n_record * self.dt * 1e9:.4g} ns), window "
                f"[{self.window_s[0] * 1e9:.4g}, {self.window_s[1] * 1e9:.4g}] ns, "
                f"decimation reference {self.freq_max_hz / 1e9:.4g} GHz")
        if self.completed:
            head += (f", decimation {self.decimation_factors or (1,)}, rank "
                     f"{self.rank}, {self.n_kept} poles kept ({self.n_guard_dropped} "
                     f"dropped by the guard, {self.n_growing_discarded} growing "
                     f"discarded); tail share {self.tail_share:.3g}, slowest decay "
                     f"{self.slowest_decay_over_window:.3g} windows")
        if not self.completed:
            head += f"; NOT COMPLETED: {self.failure}"
        elif self.s_accumulator_roundoff is not None:
            head += (f"; Result.s_params' own accumulator round-off "
                     f"{self.s_accumulator_roundoff:.2e} (information)")
        lines = [head]
        for w in self.witnesses:
            status = ("ok" if w.ok else "FAILED") if w.judged else "not judged"
            lines.append(f"  {w.name}: {w.value:.3e} (bar {w.bar:.3e}) {status}"
                         + (f" -- {w.note}" if w.note else ""))
        return "\n".join(lines)


class RingdownResult(NamedTuple):
    """``Result.ringdown``: the completed S-matrix, its bins and its report.

    ``s_params`` has the shape and dtype of ``Result.s_params`` and is
    evaluated at the same bins (``freqs`` is ``Result.freqs``; rfx accumulates
    at those bins rounded to float32, and so does the completion). It is
    ``None`` when the report says the completion was not produced.
    """

    s_params: np.ndarray | None
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
    transition-band guard off (for tests of the guard itself). The poles the
    pencil discards as growing are kept aside (``s_growing``) and sized by a
    residue fit made jointly with the kept poles (``growing_amplitude``); the
    completion itself uses the kept poles' own fit.
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
    lam_dec, rank, lam_grow, n_invalid, sv = _pencil(Yd, float(sv_rel), float(unit_tol))
    s, n_guard = _to_original_rate(lam_dec, D, dt, guard)
    c = _fit_residues(win, s, dt)
    rms = np.sqrt(np.mean(np.abs(win) ** 2, axis=0))
    rms_safe = np.where(rms > 0, rms, 1.0)
    amplitude = (np.max(np.abs(c) / rms_safe[None, :], axis=1) if s.size
                 else np.zeros(0))
    s_grow = np.log(np.asarray(lam_grow, dtype=np.complex128)) / (D * dt)
    g = np.zeros(s_grow.size)
    if s_grow.size:
        c_joint = _fit_residues(win, np.concatenate([s, s_grow]), dt)[s.size:]
        n_end = win.shape[0] - 1
        at_end = np.abs(c_joint) * np.exp(s_grow.real * dt * n_end)[:, None]
        g = np.max(at_end / rms_safe[None, :], axis=1)
    return RingdownModel(
        s=s, c=c, n_ref=n_start, dt=dt, freq_max=freq_max,
        guard=None if guard is None else float(guard), factors=factors, D=D,
        rank=int(rank), n_window=int(win.shape[0]), n_decimated=int(Yd.shape[0]),
        n_growing_discarded=int(s_grow.size), n_invalid=int(n_invalid),
        n_guard_dropped=n_guard, singular_values=sv, amplitude=amplitude,
        s_growing=s_grow, growing_amplitude=g, window_rms=rms)


class TwoWindowWitness(NamedTuple):
    """``W2`` and the two completions it compares."""

    value: float
    spectra: np.ndarray
    spectra_short: np.ndarray
    model: RingdownModel
    model_short: RingdownModel


def two_window_witness(series, dt, freqs, n_record, n_start, *, freq_max,
                       split=0.9, guard=0.9, sv_rel=1.0e-6, unit_tol=1.0e-6,
                       observable=None, plain=None) -> TwoWindowWitness:
    """Complete ``series[:n_record]`` from ``[n_start, n_record)`` and from ``[n_start, split*n_record)``.

    Both completions add their tail after the SAME last sample
    (``n_record - 1``), so the difference is the window's influence alone.
    ``observable`` maps a ``(nf, C)`` spectra array to the quantity compared
    (default: the spectra themselves); ``value`` is the largest absolute
    difference of the two observables. ``plain`` may carry the plain DFT of
    ``series[:n_record]`` at ``freqs`` when the caller already has it.
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
    if plain is None:
        plain = plain_dft(Y, dt, freqs)
    spectra = plain + tail_dft(model, n_record - 1, freqs)
    spectra_short = plain + tail_dft(model_short, n_record - 1, freqs)
    obs = (lambda a: a) if observable is None else observable
    value = float(np.max(np.abs(np.asarray(obs(spectra), dtype=np.complex128)
                                - np.asarray(obs(spectra_short), dtype=np.complex128))))
    return TwoWindowWitness(value, spectra, spectra_short, model, model_short)


def _long_window_start(n_record, window_start) -> int:
    """First sample of ``WE``'s window ``[window_start / 2 * T, T]``."""
    return int(round(0.5 * float(window_start) * int(n_record)))


class DoubleSpanWitness(NamedTuple):
    """``WE`` and the completion it compares with the main one."""

    value: float
    spectra_long: np.ndarray
    model_long: RingdownModel
    n_start_long: int


def double_span_witness(series, dt, freqs, n_record, window_start, *, freq_max,
                        guard=0.9, sv_rel=1.0e-6, unit_tol=1.0e-6,
                        observable=None, plain=None, spectra=None) -> DoubleSpanWitness:
    """Complete ``series[:n_record]`` from ``[window_start T, T]`` and from ``[window_start/2 T, T]``.

    The second window has double the span of the first and contains it. Both
    completions add their tail after the SAME last sample (``n_record - 1``);
    ``value`` is the largest absolute difference of the two observables
    (``observable`` as in :func:`two_window_witness`). ``spectra`` may carry
    the completion from ``[window_start T, T]`` and ``plain`` the plain DFT of
    ``series[:n_record]`` when the caller already has them.
    """
    n_record = int(n_record)
    n_start = int(round(float(window_start) * n_record))
    n_long = _long_window_start(n_record, window_start)
    if not (0 <= n_long < n_start < n_record):
        raise ValueError(
            f"a record of {n_record} samples is too short for the windows "
            f"[{n_long}, {n_record}) and [{n_start}, {n_record})")
    Y, _ = _as_2d(series)
    Y = Y[:n_record]
    kw = dict(freq_max=freq_max, guard=guard, sv_rel=sv_rel, unit_tol=unit_tol)
    if plain is None:
        plain = plain_dft(Y, dt, freqs)
    if spectra is None:
        spectra = plain + tail_dft(identify(Y, dt, n_start, n_record, **kw),
                                   n_record - 1, freqs)
    model_long = identify(Y, dt, n_long, n_record, **kw)
    spectra_long = plain + tail_dft(model_long, n_record - 1, freqs)
    obs = (lambda a: a) if observable is None else observable
    value = float(np.max(np.abs(np.asarray(obs(spectra), dtype=np.complex128)
                                - np.asarray(obs(spectra_long), dtype=np.complex128))))
    return DoubleSpanWitness(value, spectra_long, model_long, n_long)


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


# ---------------------------------------------------------------------------
# Simulation.run(..., ringdown=...) integration
# ---------------------------------------------------------------------------

#: Ampere-loop legs of a port component: (H component, axis it is differenced
#: along). The same pairs ``rfx.probes.probes._ampere_loop`` (uniform lane)
#: and ``rfx.nonuniform.wire_port_current`` (graded lane) read.
_LOOP_LEGS = {
    "ez": (("hy", 0), ("hx", 1)),
    "ex": (("hz", 1), ("hy", 2)),
    "ey": (("hx", 2), ("hz", 0)),
}
_AXIS_OF = {"ex": 0, "ey": 1, "ez": 2}
_LOCAL = (1, 1, 1)


def refuse_run_request(sim, spec, *, devices, until_decay, compute_s_params) -> None:
    """Raise, with the reason, for every run ``ringdown=`` does not cover."""
    if not isinstance(spec, RingdownSpec):
        raise TypeError(
            f"ringdown= takes a rfx.ringdown.RingdownSpec, got {type(spec).__name__}")
    if devices is not None and len(devices) > 1:
        raise NotImplementedError(
            "run(ringdown=...) is not supported on the distributed lane "
            "(devices=): the port-channel probes and their check against the "
            "port accumulators are single-device only.")
    if until_decay is not None:
        raise NotImplementedError(
            "run(ringdown=...) completes a record of fixed length; until_decay= "
            "stops the record on field energy instead, and the two are not "
            "combined. Pass n_steps= or num_periods=.")
    if compute_s_params is False:
        raise ValueError(
            "run(ringdown=...) completes the port S-parameters, but "
            "compute_s_params=False asks for none.")
    if getattr(sim, "_mode", "3d") != "3d":
        raise NotImplementedError(
            f"run(ringdown=...) needs mode='3d'; this model is mode={sim._mode!r}.")
    if getattr(sim, "_solver", None) == "adi":
        raise NotImplementedError("run(ringdown=...) is not supported with solver='adi'.")
    if getattr(sim, "_refinement", None) is not None:
        raise NotImplementedError(
            "run(ringdown=...) is not supported on the subgridded lane.")
    if getattr(sim, "_periodic_axes", ""):
        raise NotImplementedError(
            "run(ringdown=...) is not supported with periodic axes: rfx refuses "
            "wire-port S-parameters there (#206).")
    if getattr(sim, "_tfsf", None) is not None:
        wf = getattr(sim._tfsf, "waveform", "")
        raise NotImplementedError(
            "run(ringdown=...) is not supported with a TFSF plane wave"
            + (" driven by a continuous wave, which never turns off"
               if wf == "continuous_wave" else
               ": the total-field boundary keeps injecting while the auxiliary "
               "line still carries the pulse, so the injection waveform alone "
               "does not say when the source is off") + ".")
    if any(float(getattr(m, "chi3", 0.0) or 0.0) != 0.0
           for m in getattr(sim, "_materials", {}).values()):
        raise NotImplementedError(
            "run(ringdown=...) is not supported with a Kerr (chi3) material: "
            "after the pulse the update is not linear, so the record is not a "
            "sum of fixed damped oscillations.")
    for attr, what in (("_msl_ports", "microstrip (add_msl_port)"),
                       ("_waveguide_ports", "waveguide (add_waveguide_port)"),
                       ("_coaxial_ports", "coaxial (add_coaxial_port)"),
                       ("_floquet_ports", "Floquet (add_floquet_port)")):
        if getattr(sim, attr, None):
            raise NotImplementedError(
                f"run(ringdown=...) completes wire-port S-parameters only; this "
                f"model has a {what} port.")
    wire = []
    for pe in sim._ports:
        if float(pe.impedance) == 0.0:
            continue            # a soft source: checked for being off, below
        if pe.extent is None:
            raise NotImplementedError(
                f"run(ringdown=...) completes wire-port S-parameters only; the "
                f"port at {tuple(pe.position)} is a one-cell lumped port "
                "(add_port without extent=).")
        if getattr(pe, "reference_plane_cells", None) is not None:
            raise NotImplementedError(
                f"run(ringdown=...) does not cover the reference-plane waves of "
                f"the wire port at {tuple(pe.position)} (reference_plane_cells=).")
        wire.append(pe)
    if not wire:
        raise ValueError(
            "run(ringdown=...) needs a wire port (add_port(..., extent=...), "
            "impedance > 0); this model has none.")
    if not any(bool(pe.excite) for pe in wire):
        raise ValueError(
            "run(ringdown=...) needs a driven wire port: with every port passive "
            "run() reports a per-cell diagnostic, not a scattering parameter.")


def refuse_run_lane(lane: str, n_wire_ports: int) -> None:
    """Lane-dependent refusals, once ``run()`` has chosen its lane."""
    if lane not in ("run_uniform", "run_nonuniform"):
        raise NotImplementedError(
            f"run(ringdown=...) supports the uniform and graded-mesh lanes; this "
            f"run takes the {lane!r} lane.")
    if lane == "run_uniform" and n_wire_ports > 1:
        raise NotImplementedError(
            "run(ringdown=...) on the uniform lane supports one wire port: with "
            "several, run() assembles the S-matrix from separate per-port drive "
            "runs that the port-channel probes do not see. Use a graded mesh "
            "(any dx/dy/dz profile), where one run fills the driven column.")


def _lane_resolver(lane: str, grid):
    """The position -> index function the lane's runner resolves probes with."""
    if lane == "uniform":
        return grid.position_to_index
    from rfx.runners.nonuniform import pos_to_nu_index
    return lambda pos: pos_to_nu_index(grid, pos)


def _node_position(grid, idx) -> tuple:
    return tuple(float(grid.node_of(ax, int(idx[ax]))) for ax in range(3))


class _PortMeta(NamedTuple):
    """One wire port as the run realized it (read from the run's own metadata)."""

    mid: tuple
    component: str
    z0: object
    excite: bool
    live_cells: tuple
    k_mid: int
    raw: dict            # the metric objects the run multiplied by
    f64: dict            # the same metrics as Python floats
    accs: tuple          # rfx's (v, i, v_port) accumulators
    freqs: object        # jnp float32 bins the accumulators used
    meta: object         # the run's own metadata entry


def _read_port_metas(lane: str, result, grid) -> list:
    """Realized cells, metrics and accumulators of every wire port, from the run."""
    import jax.numpy as jnp

    wps = getattr(result, "wire_port_sparams", None)
    if not wps:
        raise RuntimeError(
            "run(ringdown=...) found no wire-port accumulators on the run's "
            "result; the port metadata it needs is missing.")
    out = []
    for meta, accs in wps:
        if lane == "uniform":
            mid = (int(meta.mid_i), int(meta.mid_j), int(meta.mid_k))
            comp = str(meta.component)
            live = tuple(tuple(int(v) for v in c) for c in (meta.live_cells or (mid,)))
            dx = float(grid.cells(0)[mid[0]])
            raw = {"dx": dx}
            f64 = {"dx": dx}
            z0, excite, freqs = meta.impedance, bool(meta.excite), meta.freqs
            acc3 = (accs[0], accs[1], accs[3])
        else:
            mid = (int(meta[0]), int(meta[1]), int(meta[2]))
            comp = str(meta[3])
            live = tuple(tuple(int(v) for v in c) for c in meta[13])
            v_metric = meta[11] if comp == "ez" else meta[5] if comp == "ex" else meta[6]
            raw = {"v_metric": v_metric, "d_par": tuple(meta[14]),
                   "dual_x": meta[9], "dual_y": meta[10], "dual_z": meta[12]}
            f64 = {"v_metric": float(v_metric),
                   "d_par": tuple(float(d) for d in meta[14]),
                   "dual_x": float(meta[9]), "dual_y": float(meta[10]),
                   "dual_z": float(meta[12])}
            z0, excite = meta[4], bool(meta[7])
            freqs = jnp.asarray(np.asarray(result.freqs), dtype=jnp.float32)
            acc3 = (accs[0], accs[1], accs[3])
        if mid not in live:
            raise RuntimeError(
                f"wire port metadata: midpoint {mid} is not among its live cells {live}")
        out.append(_PortMeta(mid=mid, component=comp, z0=z0, excite=excite,
                             live_cells=live, k_mid=live.index(mid), raw=raw,
                             f64=f64, accs=acc3, freqs=freqs, meta=meta))
    return out


class _HLocal(NamedTuple):
    hx: object
    hy: object
    hz: object


def _port_vi(lane: str, pm: _PortMeta, metrics: dict, e, hx, hy, hz):
    """``(v_mid, v_port, i)`` from the port's E edges and local Ampere-loop H.

    ``e`` indexes the live edges on its FIRST axis (``e[c]``); ``hx, hy, hz``
    are 2x2x2 neighbourhoods (optionally with a trailing time axis) holding
    the loop's H samples around local index (1, 1, 1). The current goes
    through the lane's OWN loop function; the voltage is the lane's inline
    sum, spelled as the scan spells it (W0 checks the two against the run).
    """
    n_live = len(pm.live_cells)
    if lane == "uniform":
        from rfx.probes.probes import _ampere_loop
        dx = metrics["dx"]
        v = -e[pm.k_mid] * dx
        v_port = -sum(e[c] for c in range(n_live)) * dx
        i_val = _ampere_loop(_HLocal(hx, hy, hz), _LOCAL, pm.component, dx,
                             (False, False, False))
    else:
        from rfx.nonuniform import wire_port_current
        v = -e[pm.k_mid] * metrics["v_metric"]
        v_port = -sum(e[c] * dp for c, dp in zip(range(n_live), metrics["d_par"]))
        i_val = wire_port_current(hx, hy, hz, pm.component, *_LOCAL,
                                  metrics["dual_x"], metrics["dual_y"],
                                  metrics["dual_z"])
    return v, v_port, i_val


def _emulate_accumulators(lane: str, pm: _PortMeta, e32, h32, dt):
    """rfx's wire-port DFT accumulation re-run on the probe samples (W0).

    The scan body is the lane's wire block with the field reads replaced by
    the probe samples: ``t = float32(step) * dt``, the kernel
    ``exp(-j 2 pi f t)`` cast to complex64 and times ``dt``, the current's
    kernel times ``half_step_current_phase``, a sequential complex64 sum.
    Returns ``((v, i, v_port) accumulators as numpy arrays of the run's dtype,
    (v_port, i) per step)`` -- the per-step series are the values the scan
    accumulated, in the dtype it computed them in (float32 on a float32 run).
    """
    import jax
    import jax.numpy as jnp
    from rfx.core.dft_utils import half_step_current_phase as _half_i_phase

    freqs = pm.freqs

    def body(carry, xs):
        step_idx, e, hx, hy, hz = xs
        v_dft, i_dft, vp_dft = carry
        v, v_port, i_val = _port_vi(lane, pm, pm.raw, e, hx, hy, hz)
        t = step_idx.astype(jnp.float32) * dt
        t_f64 = t.astype(jnp.float64)
        phase = jnp.exp(-1j * 2.0 * jnp.pi * freqs.astype(jnp.float64)
                        * t_f64).astype(jnp.complex64) * dt
        i_phase = phase * _half_i_phase(
            freqs.astype(jnp.float64), dt).astype(jnp.complex64)
        return (v_dft + v * phase, i_dft + i_val * i_phase,
                vp_dft + v_port * phase), (v_port, i_val)

    nf = int(freqs.shape[0])
    # The run's own carry dtype: complex64, or complex128 on the uniform lane
    # under x64 (``rfx.simulation``'s ``_sparam_acc_dtype``).
    init = tuple(jnp.zeros(nf, dtype=np.asarray(a).dtype) for a in pm.accs)
    n = e32.shape[0]
    xs = (jnp.arange(n, dtype=jnp.int32), jnp.asarray(e32),
          jnp.asarray(h32["hx"]), jnp.asarray(h32["hy"]), jnp.asarray(h32["hz"]))
    with warnings.catch_warnings():
        # rfx's own kernel requests float64 that x64-off JAX serves as
        # float32; the emulation repeats the request, and its warning.
        warnings.simplefilter("ignore")
        final, per_step = jax.jit(lambda c, x: jax.lax.scan(body, c, x))(init, xs)
    return tuple(np.asarray(a) for a in final), tuple(np.asarray(a) for a in per_step)


def _assemble(lane: str, pms, spectra_vp, spectra_i, freqs32, dt, v_mid=None):
    """The lane's own S-matrix assembly on given (V_port, I) spectra per port.

    ``spectra_vp[p]`` and ``spectra_i[p]`` are the accumulator-convention
    spectra (the current already carries its half-step phase). They enter the
    assembly in the dtype of the run's own accumulators, as the run's do.
    Returns a numpy array shaped and typed like ``Result.s_params``.
    """
    import jax.numpy as jnp

    nf = int(np.asarray(freqs32).shape[0])
    acc_dtype = np.asarray(pms[0].accs[2]).dtype

    def as_acc(a):
        return jnp.asarray(np.asarray(a), dtype=acc_dtype)

    if lane == "uniform":
        from rfx.probes.probes import driven_port_reflection
        (pm,) = pms
        # the runner's container (rfx/runners/uniform.py, single-port wire path)
        S = np.zeros((1, 1, nf), dtype=np.complex64)
        S[0, 0, :] = np.array(driven_port_reflection(
            as_acc(spectra_vp[0]), as_acc(spectra_i[0]), pm.z0))
        return S
    from types import SimpleNamespace

    from rfx.nonuniform import _assemble_nu_result
    zeros = jnp.zeros(nf, dtype=acc_dtype)
    accs = tuple(
        (zeros if v_mid is None else as_acc(v_mid[p]), as_acc(spectra_i[p]),
         zeros, as_acc(spectra_vp[p]))
        for p in range(len(pms)))
    metas = [pm.meta for pm in pms]
    setup = SimpleNamespace(
        dt=float(dt), dft_planes=None, flux_monitors=None, wire_ports=metas,
        waveguide_meta=None, wp_meta=metas,
        sp_freqs=jnp.asarray(np.asarray(freqs32), dtype=jnp.float32),
        use_wire_ports=True, use_dft_planes=False, use_flux_monitors=False,
        use_lumped_rlc=False, use_ntff=False, use_waveguide_ports=False)
    r = _assemble_nu_result(setup, {"fdtd": None, "wire_sparams": accs}, None)
    return np.asarray(r["s_params"])


def _ulps(rebuilt, rfx_values) -> float:
    """``max |rebuilt - rfx|`` in units of the spacing of the rfx array's own
    real dtype at that array's peak (the cross-trace ULP reading)."""
    a = np.asarray(rebuilt).astype(np.complex128)
    b_raw = np.asarray(rfx_values)
    b = b_raw.astype(np.complex128)
    d = float(np.max(np.abs(a - b))) if b.size else 0.0
    if d == 0.0:
        return 0.0
    real = np.real(b_raw.ravel()[:1]).dtype
    peak = float(np.max(np.abs(b)))
    return d / float(np.spacing(np.asarray(peak, dtype=real)))


#: W0's bar: the rebuilt accumulators and the lane's S assembly on the run's
#: own accumulators may differ from the run by at most this many ULPs, read at
#: the peak of each array in the run's own real dtype (the PI's cross-trace
#: rule of 2026-09-23: aim bitwise, allow a single-digit ULP count).
W0_ULP_BAR = 9.0
#: W1's bar, absolute on S: the S the completion's channels give without a
#: tail -- the float64 DFT of the float64 rebuild of the port's V and I --
#: against the S the same DFT gives of the float32 per-step V and I that the
#: W0 replay accumulated (W0 ties those to the run's accumulators). Neither
#: side carries a float32 accumulation, so the difference is the two V/I
#: constructions alone: 1.7e-7 .. 6.0e-7 on clean runs of 1,500-60,000 steps,
#: 0.89 when the completion is fed the midpoint cell's voltage of a 3-cell port.
W1_BAR = 1.0e-4


class _NotCompleted(Exception):
    """A check that decides whether the completion can be trusted at all failed."""

    def __init__(self, name, value, bar, message):
        super().__init__(message)
        self.name, self.value, self.bar, self.message = name, value, bar, message


class RingdownRun:
    """One ``run(..., ringdown=spec)``: probes in, run, probes out, complete.

    Built by ``Simulation.run`` once the lane, the step count and the grid are
    known; :meth:`run` wraps the lane's runner call. Everything that can refuse
    the request does so here, before the run; after the run nothing raises: a
    failed check returns the plain result unchanged with the reason in
    ``Result.ringdown.report`` and a warning.
    """

    def __init__(self, sim, spec: RingdownSpec, *, lane: str, n_steps: int, grid):
        if lane not in ("uniform", "graded"):
            raise ValueError(f"lane must be 'uniform' or 'graded', got {lane!r}")
        self.sim, self.spec, self.lane = sim, spec, lane
        self.n_steps = int(n_steps)
        self.grid = grid
        self.dt = grid.dt
        self.n_start = int(round(float(spec.window_start) * self.n_steps))
        self.n_split = int(round(float(spec.split) * self.n_steps))
        self.n_long = _long_window_start(self.n_steps, spec.window_start)
        if not (0 < self.n_start < self.n_split < self.n_steps):
            raise ValueError(
                f"a record of {self.n_steps} steps is too short for the windows "
                f"[{self.n_start}, {self.n_steps}) and [{self.n_start}, {self.n_split})")
        self.freq_max = float(spec.freq_max if spec.freq_max is not None
                              else sim._freq_max)
        self.wire_entries = [pe for pe in sim._ports
                             if float(pe.impedance) > 0.0 and pe.extent is not None]
        self.source_off, self.source_ratio_long = self._check_sources_off()
        self.probe_keys = self._plan_probes()

    # -- before the run ------------------------------------------------------

    def _check_sources_off(self) -> tuple:
        """Every source's waveform over the window, as a fraction of its peak.

        The tables are evaluated the way the runners build them:
        ``waveform(float32(n) * dt)`` for ``n = 0 .. n_steps - 1``. Returns
        ``(rows, largest ratio over WE's window [n_long, n_steps))``; only
        the main window refuses.
        """
        import jax
        import jax.numpy as jnp

        times = jnp.arange(self.n_steps, dtype=jnp.float32) * self.dt
        rows = []
        ratio_long = 0.0
        for pe in self.sim._ports:
            driven = float(pe.impedance) == 0.0 or bool(pe.excite)
            if not driven or pe.waveform is None:
                continue
            w = np.abs(np.asarray(jax.vmap(pe.waveform)(times), dtype=np.float64))
            peak = float(np.max(w)) if w.size else 0.0
            ratio = 0.0 if peak == 0.0 else float(np.max(w[self.n_start:]) / peak)
            kind = "source" if float(pe.impedance) == 0.0 else "wire port"
            label = f"{kind} at {tuple(float(v) for v in pe.position)} ({pe.component})"
            if not ratio <= float(self.spec.source_off_tol):
                raise ValueError(
                    f"run(ringdown=...): the {label} is still on inside the "
                    f"identification window [{self.n_start * float(self.dt):.4g}, "
                    f"{self.n_steps * float(self.dt):.4g}] s: its waveform there "
                    f"reaches {ratio:.3e} of its peak, above source_off_tol = "
                    f"{self.spec.source_off_tol:.1e}. The window must hold free "
                    "ringing only; use a longer record or a later window_start.")
            rows.append((label, ratio))
            if peak != 0.0:
                ratio_long = max(ratio_long, float(np.max(w[self.n_long:]) / peak))
        return tuple(rows), ratio_long

    def _plan_probes(self) -> tuple:
        """(component, index) of every port-channel probe, deduplicated, in order.

        The E edges of each port's extent span and the H samples of an Ampere
        loop at each of them, with one extra edge below the span (a sub-cell
        extent drives the edge on either side of its node) unless that edge
        would lie in the absorber pad. Which of these are the port's live
        edges and midpoint is read from the run itself after it has run; this
        only has to cover them.
        """
        resolve = _lane_resolver(self.lane, self.grid)
        keys = []
        seen = set()

        def add(comp, idx):
            key = (comp, tuple(int(v) for v in idx))
            if key not in seen:
                seen.add(key)
                keys.append(key)

        pads = tuple(int(getattr(self.grid, f"pad_{ax}_lo", 0)) for ax in "xyz")
        for pe in self.wire_entries:
            comp = str(pe.component)
            axis = _AXIS_OF[comp]
            end = list(pe.position)
            end[axis] += float(pe.extent)
            i0 = tuple(int(v) for v in resolve(tuple(pe.position)))
            i1 = tuple(int(v) for v in resolve(tuple(end)))
            lo, hi = min(i0[axis], i1[axis]), max(i0[axis], i1[axis])
            for k in range(max(lo - 1, pads[axis]), hi + 1):
                cell = list(i0)
                cell[axis] = k
                add(comp, cell)
                for hname, back_axis in _LOOP_LEGS[comp]:
                    add(hname, cell)
                    if cell[back_axis] > 0:
                        back = list(cell)
                        back[back_axis] -= 1
                        if self.lane == "graded" and back[back_axis] < pads[back_axis]:
                            raise NotImplementedError(
                                f"run(ringdown=...): the wire port at "
                                f"{tuple(float(v) for v in pe.position)} sits on "
                                f"the {'xyz'[back_axis]}-low face, so its Ampere "
                                f"loop reads {hname} inside the absorber pad, "
                                "where the graded lane's probes cannot be placed. "
                                "Move the port at least one cell inside.")
                        add(hname, back)
        for comp, idx in keys:
            pos = _node_position(self.grid, idx)
            got = tuple(int(v) for v in resolve(pos))
            if got != idx:
                raise RuntimeError(
                    f"port-channel probe {comp}{idx}: its node position {pos} "
                    f"resolves to {got}")
        return tuple(keys)

    # -- the run -------------------------------------------------------------

    def run(self, call):
        """Attach the probes, call the lane's runner, detach, complete."""
        from rfx.api._spec import _ProbeEntry

        probes = self.sim._probes
        n_user = len(probes)
        probes.extend(_ProbeEntry(position=_node_position(self.grid, idx),
                                  component=comp)
                      for comp, idx in self.probe_keys)
        try:
            kwargs = {"keep_wire_port_sparams": True} if self.lane == "uniform" else {}
            result = call(**kwargs)
        finally:
            del probes[n_user:]
        return self._finish(result, n_user)

    # -- after the run -------------------------------------------------------

    def _finish(self, result, n_user: int):
        full = result.time_series
        ts_all = np.asarray(full)
        n_int = len(self.probe_keys)
        if ts_all.ndim != 2 or ts_all.shape != (self.n_steps, n_user + n_int):
            raise RuntimeError(
                f"run(ringdown=...): the run returned a time series of shape "
                f"{ts_all.shape}, expected ({self.n_steps}, {n_user} user + "
                f"{n_int} port-channel probes)")
        stripped = result._replace(
            time_series=full[:, :n_user],
            **({"wire_port_sparams": None} if self.lane == "uniform" else {}))
        dt = float(result.grid.dt)
        ref_hz = max(self.freq_max, float(np.max(np.asarray(result.freqs,
                                                            dtype=np.float64))))
        base = dict(n_record=self.n_steps, dt=dt,
                    window_steps=(self.n_start, self.n_steps),
                    window_s=(self.n_start * dt, self.n_steps * dt),
                    short_window_steps=(self.n_start, self.n_split),
                    long_window_steps=(self.n_long, self.n_steps),
                    freq_max_hz=ref_hz)
        witnesses, state = [], {}
        try:
            S, report_extra = self._complete(result, ts_all, n_user, dt, ref_hz,
                                             witnesses, state)
        except _NotCompleted as nc:
            witnesses.append(RingdownWitness(nc.name, float(nc.value), float(nc.bar),
                                             False, nc.message))
            report = RingdownReport(**base, witnesses=tuple(witnesses),
                                    failure=f"{nc.name}: {nc.message}",
                                    w0_ulps=state.get("w0_ulps"),
                                    s_accumulator_roundoff=state.get(
                                        "s_accumulator_roundoff"))
            warnings.warn(
                f"ring-down completion not produced -- {nc.name} failed: "
                f"{nc.message} Result.ringdown.s_params is None; every other "
                "output of this run is as without ringdown=.", stacklevel=3)
            return stripped._replace(ringdown=RingdownResult(
                s_params=None, freqs=result.freqs, report=report))
        report = RingdownReport(**base, witnesses=tuple(witnesses),
                                w0_ulps=state.get("w0_ulps"),
                                s_accumulator_roundoff=state.get(
                                    "s_accumulator_roundoff"), **report_extra)
        if not report.ok:
            failed = ", ".join(f"{w.name} {w.value:.3e} (bar {w.bar:.3e})"
                               + (f" -- {w.note}" if w.note else "")
                               for w in witnesses if w.judged and not w.ok)
            warnings.warn(
                f"ring-down completion: witness failed -- {failed}; "
                "Result.ringdown.s_params is not supported by its own check "
                "(see Result.ringdown.report).", stacklevel=3)
        return stripped._replace(ringdown=RingdownResult(
            s_params=S, freqs=result.freqs, report=report))

    def _rebuild_inputs(self, result, ts_all, n_user):
        """The port metadata the run realized and the probe samples it covers."""
        run_grid = result.grid
        resolve = _lane_resolver(self.lane, run_grid)
        for comp, idx in self.probe_keys:
            got = tuple(int(v) for v in resolve(_node_position(self.grid, idx)))
            if got != idx:
                raise _NotCompleted(
                    "W0", math.nan, W0_ULP_BAR,
                    f"port-channel probe {comp}{idx} resolved to {got} on the "
                    "run's own grid.")
        col = {key: n_user + j for j, key in enumerate(self.probe_keys)}
        try:
            pms = _read_port_metas(self.lane, result, run_grid)
        except RuntimeError as exc:
            raise _NotCompleted("W0", math.nan, W0_ULP_BAR, str(exc)) from None
        if len(pms) != len(self.wire_entries):
            raise _NotCompleted(
                "W0", math.nan, W0_ULP_BAR,
                f"the run reports {len(pms)} wire ports, the model declares "
                f"{len(self.wire_entries)}.")
        e_cols, h_cols = [], []
        for p, pm in enumerate(pms):
            missing = [c for c in pm.live_cells if (pm.component, c) not in col]
            if missing or any((hname, pm.mid) not in col
                              for hname, _ax in _LOOP_LEGS[pm.component]):
                raise _NotCompleted(
                    "W0", math.nan, W0_ULP_BAR,
                    f"wire port {p}: realized live edges {pm.live_cells} / "
                    f"midpoint {pm.mid} not all inside the probed span.")
            e = ts_all[:, [col[(pm.component, c)] for c in pm.live_cells]]
            h = {name: np.zeros((ts_all.shape[0], 2, 2, 2), dtype=ts_all.dtype)
                 for name in ("hx", "hy", "hz")}
            for hname, back_axis in _LOOP_LEGS[pm.component]:
                h[hname][:, 1, 1, 1] = ts_all[:, col[(hname, pm.mid)]]
                if pm.mid[back_axis] > 0:
                    back = list(pm.mid)
                    back[back_axis] -= 1
                    lb = [1, 1, 1]
                    lb[back_axis] = 0
                    h[hname][:, lb[0], lb[1], lb[2]] = ts_all[:, col[(hname, tuple(back))]]
                # at index 0 the runner reads zero there; the local cell stays 0
            e_cols.append(e)
            h_cols.append(h)
        return pms, e_cols, h_cols

    def _error_witnesses(self, Y, dt, f_bins, ref_hz, to_s, plain, w2, state) -> list:
        """``WE`` judged and ``W2`` reported, or ``W2`` judged when ``WE`` cannot be formed.

        ``WE`` compares the completion from ``[ws T, T]`` with the one from
        ``[ws/2 T, T]``; it is formed when every source is at or below
        ``source_off_tol`` of its peak over ``[ws/2 T, T]``. A failed
        identification on that window is a failed ``WE``.
        """
        spec = self.spec
        tol = float(spec.witness_tol)
        ws = float(spec.window_start)
        we_rule = (f"max |S| difference between the completions from [{ws:g} T, T] "
                   f"and [{0.5 * ws:g} T, T]")
        w2_rule = (f"max |S| difference between the completions from [{ws:g} T, T] "
                   f"and [{ws:g} T, {spec.split:g} T]")
        r_long = float(self.source_ratio_long)
        if not r_long <= float(spec.source_off_tol):
            why = (f"a source reaches {r_long:.3e} of its peak in [{0.5 * ws:g} T, T], "
                   f"above source_off_tol = {spec.source_off_tol:.1e}")
            return [
                RingdownWitness("WE", math.nan, tol, False, we_rule,
                                f"not formed: {why}", judged=False),
                RingdownWitness("W2", w2.value, tol, w2.value <= tol, w2_rule,
                                f"judged in place of WE, which could not be formed "
                                f"({why}); on the research boards W2 read the "
                                "actual error low by up to several times (rfx "
                                "#1254, R-i)"),
            ]
        try:
            we = double_span_witness(
                Y, dt, f_bins, self.n_steps, ws, freq_max=ref_hz, guard=spec.guard,
                sv_rel=spec.sv_rel, unit_tol=spec.unit_tol, observable=to_s,
                plain=plain, spectra=w2.spectra)
            we_value, we_note = we.value, ""
            state["long_rank"] = we.model_long.rank
            state["long_n_kept"] = int(we.model_long.s.size)
        except (ValueError, np.linalg.LinAlgError) as exc:
            we_value = math.nan
            we_note = (f"the identification on [{0.5 * ws:g} T, T] failed: {exc}; "
                       "a WE that cannot be read fails")
        return [
            RingdownWitness("WE", we_value, tol, we_value <= tol, we_rule, we_note),
            RingdownWitness("W2", w2.value, tol, w2.value <= tol, w2_rule,
                            "information (WE is the judged error witness)",
                            judged=False),
        ]

    def _complete(self, result, ts_all, n_user, dt, ref_hz, witnesses, state):
        pms, e_cols, h_cols = self._rebuild_inputs(result, ts_all, n_user)

        # ---- W0: the rebuilt series, accumulated the run's way, are the run's
        ulps = {}
        # the grid's own dt object: under x64 a numpy float64 and a Python
        # float promote the run's float32 step index differently
        dt_run = result.grid.dt
        replayed = []                    # per port: the replay's float32 (v_port, i)
        for p, pm in enumerate(pms):
            emu, per_step = _emulate_accumulators(self.lane, pm, e_cols[p],
                                                  h_cols[p], dt_run)
            replayed.append(per_step)
            for name, a, b in zip(("v_mid", "i", "v_port"), emu, pm.accs):
                ulps[f"port{p}/{name}"] = _ulps(a, b)
        freqs32 = pms[0].freqs
        S_run = np.asarray(result.s_params)
        S_check = _assemble(self.lane, pms, [pm.accs[2] for pm in pms],
                            [pm.accs[1] for pm in pms], freqs32, dt,
                            v_mid=[pm.accs[0] for pm in pms])
        for j in range(S_run.shape[0]):
            for k in range(S_run.shape[1]):
                ulps[f"S[{j},{k}]"] = _ulps(S_check[j, k], S_run[j, k])
        state["w0_ulps"] = ulps
        worst = max(ulps, key=ulps.get)
        w0 = ulps[worst]
        w0_rule = ("the port V/I rebuilt from probes, accumulated the run's way "
                   "in the run's dtype, and the lane's S assembly on the run's own "
                   "accumulators, against the run: max ULPs at each array's peak")
        if not w0 <= W0_ULP_BAR:
            raise _NotCompleted(
                "W0", w0, W0_ULP_BAR,
                f"{worst} differs from the run by {w0:.3g} ULPs (bar "
                f"{W0_ULP_BAR:g}): the rebuilt port channels are not the ones the "
                "run accumulated.")
        witnesses.append(RingdownWitness("W0", w0, W0_ULP_BAR, True, w0_rule))

        # ---- the rebuilt record, float64 ------------------------------------
        cols = []
        for p, pm in enumerate(pms):
            e64 = e_cols[p].astype(np.float64).T                      # (n_live, n)
            h64 = {k: np.moveaxis(v.astype(np.float64), 0, -1)       # (2, 2, 2, n)
                   for k, v in h_cols[p].items()}
            _v, v_port, i_val = _port_vi(self.lane, pm, pm.f64, e64,
                                         h64["hx"], h64["hy"], h64["hz"])
            cols += [np.asarray(v_port, dtype=np.float64),
                     np.asarray(i_val, dtype=np.float64)]
        Y = np.stack(cols, axis=1)
        f_bins = np.asarray(freqs32, dtype=np.float32).astype(np.float64)
        from rfx.core.dft_utils import half_step_current_phase
        half = np.asarray(half_step_current_phase(f_bins, dt)).astype(np.complex128)

        def to_s(spectra):
            vp = [spectra[:, 2 * p] for p in range(len(pms))]
            ii = [spectra[:, 2 * p + 1] * half for p in range(len(pms))]
            return _assemble(self.lane, pms, vp, ii, freqs32, dt)

        # ---- W1: the channels fed to the completion are the ones W0 checked --
        # float64 DFT of the float64 rebuild vs float64 DFT of the float32
        # per-step series the W0 replay accumulated; no float32 accumulation
        # on either side
        plain = plain_dft(Y, dt, f_bins)
        Y32 = np.stack([np.asarray(c, dtype=np.float64)
                        for per_step in replayed for c in per_step], axis=1)
        S_plain = to_s(plain).astype(np.complex128)
        w1 = float(np.max(np.abs(S_plain - to_s(plain_dft(Y32, dt, f_bins))
                                 .astype(np.complex128))))
        # information, not a check: the run's own complex64 accumulator
        # round-off in Result.s_params, which grows with the record
        roundoff = float(np.max(np.abs(S_plain - S_run.astype(np.complex128))))
        state["s_accumulator_roundoff"] = roundoff
        w1_rule = ("max |S of the float64 DFT of the float64 rebuild of the port "
                   "V/I - S of the float64 DFT of the float32 per-step V/I the W0 "
                   "replay accumulated| (no tail)")
        if not w1 <= W1_BAR:
            raise _NotCompleted(
                "W1", w1, W1_BAR,
                f"the port channels fed to the completion give an S that differs "
                f"by {w1:.3e} (bar {W1_BAR:.0e}) from the S of the channels W0 "
                "tied to the run: they are not the port's voltage and current.")
        witnesses.append(RingdownWitness("W1", w1, W1_BAR, True, w1_rule))

        # ---- identification and completion ----------------------------------
        spec = self.spec
        try:
            w2 = two_window_witness(
                Y, dt, f_bins, self.n_steps, self.n_start, freq_max=ref_hz,
                split=spec.split, guard=spec.guard, sv_rel=spec.sv_rel,
                unit_tol=spec.unit_tol, observable=to_s, plain=plain)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise _NotCompleted("identification", math.nan, math.nan,
                                f"{exc}.") from None
        S = to_s(w2.spectra)
        model = w2.model

        record_s = self.n_steps * dt
        grow, exempt = growing_poles(model, record_s, spec.unit_tol)
        g = np.asarray(model.growing_amplitude, dtype=np.float64)
        if g.size:
            kg = int(np.argmax(g))
            g_max = float(g[kg])
            g_f = float(np.asarray(model.s_growing)[kg].imag / (2.0 * np.pi))
        else:
            g_max, g_f = 0.0, math.nan
        driven = [p for p, pm in enumerate(pms) if pm.excite]
        s_diag = max(float(np.max(np.abs(S[p, p, :]))) for p in driven)
        s_plain = max(float(np.max(np.abs(S_run[p, p, :]))) for p in driven)
        p_bar = 1.0 + float(spec.passivity_tol)
        p_note = (f"plain record max |S_kk| = {s_plain:.6g}"
                  + ("; the plain record reads non-passive without the completion (the board or port itself, or a record cut while it still rings)"
                     if s_plain > p_bar else ""))
        src_ratio = max((r for _l, r in self.source_off), default=0.0)
        witnesses += self._error_witnesses(Y, dt, f_bins, ref_hz, to_s, plain, w2, state)
        witnesses += [
            RingdownWitness("growing_poles", float(len(grow)), 0.0, len(grow) == 0,
                            GROWING_POLE_RULE + ". Poles the pencil discarded as "
                            "growing are sized and reported (the largest, with its "
                            "frequency: its size at the record's end relative to "
                            "its channel's RMS over the window) but not judged",
                            f"{len(exempt)} zero-frequency exempt; {g.size} "
                            f"discarded as growing, the largest {g_max:.3g} at "
                            f"{g_f / 1e9:.4g} GHz (reported, not judged)"
                            if g.size else
                            f"{len(exempt)} zero-frequency exempt; none discarded "
                            "as growing"),
            RingdownWitness("passivity", s_diag, p_bar, s_diag <= p_bar,
                            "max |S_kk| over the driven ports", p_note),
            RingdownWitness("source_off", src_ratio, float(spec.source_off_tol),
                            src_ratio <= float(spec.source_off_tol),
                            "largest source waveform over the window, as a "
                            "fraction of its peak"),
        ]
        tail = w2.spectra - plain
        tail_share = float(np.max(np.max(np.abs(tail), axis=0)
                                  / np.maximum(np.max(np.abs(w2.spectra), axis=0),
                                               1e-300)))
        amp = np.asarray(model.amplitude, dtype=np.float64)
        alpha = -model.s.real
        strong = amp >= 1.0e-3
        window_s = (self.n_steps - self.n_start) * dt
        if np.any(strong):
            a_min = float(np.min(alpha[strong]))
            slowest = math.inf if a_min <= 0 else 1.0 / a_min / window_s
        else:
            slowest = 0.0
        extra = dict(
            decimation_factors=model.factors, rank=model.rank,
            n_kept=int(model.s.size),
            n_growing_discarded=model.n_growing_discarded,
            n_invalid=model.n_invalid, n_guard_dropped=model.n_guard_dropped,
            short_rank=w2.model_short.rank, short_n_kept=int(w2.model_short.s.size),
            long_rank=state.get("long_rank"), long_n_kept=state.get("long_n_kept"),
            poles=model.poles(), growing_pole_rule=GROWING_POLE_RULE,
            tail_share=tail_share, slowest_decay_over_window=slowest,
            discarded_growth_max=g_max, discarded_growth_f_hz=g_f)
        return S, extra

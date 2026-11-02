"""Judge kept growing ring-down poles by the tail they can add at each read bin."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from rfx.measurement.ringdown_resolution import pole_classes

if TYPE_CHECKING:
    from rfx.ringdown import RingdownModel


def growing_poles(model: RingdownModel, record_s: float, unit_tol: float):
    """Kept poles with ``|lambda| > 1`` per step, split into (growing, exempt).

    The verdict is :func:`judge_growing_poles`; this helper only counts.
    Exempt: a static field (:func:`_static_pole`) -- a pole within rounding
    of zero frequency, ``|f| < 1 / record`` (the record cannot tell it from
    DC) and ``|lambda| - 1 < unit_tol`` per step. A static field left behind
    in a closed box identifies as such a pole (rfx #1254, the Q 533 cavity).
    """
    from rfx.ringdown import _static_pole
    grow, exempt = [], []
    for p in model.poles():
        if not p.abs_lambda > 1.0:
            continue
        if _static_pole(p, record_s, unit_tol):
            exempt.append(p)
        else:
            grow.append(p)
    return tuple(grow), tuple(exempt)


GROWING_POLE_RULE = (
    "The symmetric resolution edge is |Re(s) dt| = log1p(unit_tol)/D; "
    "unresolved poles of either sign never set H and use the same tail bound as "
    "kept non-static growing poles: their tail share S_k through "
    "twice the judge horizon H (bar 1e-2); H is the larger of the record length "
    "and the hundredfold decay horizon of significant resolved decaying poles. "
    "The band condition (1 + unit_tol)^(2H/D) <= 2 must hold. The count is "
    "reported and never judged; static poles are exempt. A pole the pencil "
    "resolves as growing is discarded and judged where the caller judges "
    "discarded poles")


class GrowingPoleJudgement(NamedTuple):
    status: str
    ok: bool
    reason: str
    n_kept_growing: int
    n_exempt: int
    horizon_steps: int
    decay_horizon_steps: float
    band_factor: float
    share_per_bin: np.ndarray
    worst: tuple | None
    growing_share_per_bin: np.ndarray | None = None


def judge_growing_poles(model, n_record, freqs, denominator, *, unit_tol=1e-6,
                        bar=1e-2, significance=1e-4) -> GrowingPoleJudgement:
    """Judge kept non-static growing poles by their tail bound through ``2H``.

    ``H`` is at least the record length and covers a hundredfold decay of
    every significant resolved decaying pole. Unresolved poles of either sign
    receive the same bound as growing poles. ``denominator (nf, C)`` is the
    caller's plain DFT magnitude (``(nf,)`` for a one-channel record);
    infinity marks an inactive channel.
    """
    from rfx.measurement.ringdown_tail import _pole_tails
    from rfx.ringdown import _static_pole
    freqs = np.asarray(freqs, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if denominator.ndim == 1:
        denominator = denominator[:, None]
    H, Hd, band = int(n_record), 0.0, math.nan
    share, worst = np.zeros(freqs.size), None
    positive_share = np.zeros(freqs.size)
    grow, exempt = (growing_poles(model, n_record * model.dt, unit_tol)
                    if len(model.s) else ((), ()))

    def result(status, reason):
        return GrowingPoleJudgement(status, status == "pass", reason,
                                    len(grow), len(exempt), H, Hd, band, share, worst,
                                    positive_share)

    if (not np.isfinite(model.s).all() or not np.isfinite(model.c).all()
            or not np.isfinite(freqs).all()
            or not np.all(np.isfinite(denominator) | np.isposinf(denominator))):
        return result("cannot judge", "non-finite poles, residues, bins or denominator")
    decaying, growing = [], []
    classes = pole_classes(model, unit_tol)
    for k in range(len(model.s)):
        if _static_pole(model.pole(k), n_record * model.dt, unit_tol):
            continue
        if classes[k] >= 0:
            growing.append(k)
        elif classes[k] < 0:
            decaying.append(k)
    none_growing = "no kept non-static growing pole"
    horizon_pole = None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if decaying:
            dec = replace(model, s=model.s[decaying], c=model.c[decaying])
            own = np.abs(_pole_tails(dec, n_record - 1, freqs)) / denominator[:, None, :]
            if not np.isfinite(own).all():
                # The horizon only bounds a growing pole's tail; without one there is nothing to judge.
                return (result("cannot judge", "non-finite decaying pole tail share") if growing
                        else result("pass", none_growing))
            significant = np.any(own >= significance, axis=(0, 2))
            if np.any(significant):
                eligible = np.asarray(decaying)[significant]
                horizon_pole = int(eligible[np.argmin(-model.s.real[eligible] * model.dt)])
                Hd = float(np.log(100.0) / (-model.s[horizon_pole].real * model.dt))
                if not np.isfinite(Hd):
                    return (result("cannot judge", "non-finite decay horizon") if growing
                            else result("pass", none_growing))
                H = max(H, math.ceil(Hd))
        if not growing:
            return result("pass", none_growing)
        log_band = (2.0 * H / model.D) * np.log1p(unit_tol)
        log_max = np.log(np.finfo(float).max)
        band = float(np.exp(min(log_band, log_max)))
        s, c = model.s[growing], model.c[growing]
        rate = s.real * model.dt
        # |1-exp(z)| = exp(Re(z)) |1-exp(-z)| for a growing pole.
        # No positive exponential is formed, even far outside the band.
        z = s[None, :] * model.dt - 2j * np.pi * freqs[:, None] * model.dt
        log_divisor = np.maximum(rate, 0)[None, :] + np.log(np.abs(
            -np.expm1(np.where(rate[None, :] > 0, -z, z))))
        log_amp = (np.logaddexp(0., 2.0 * H * rate) + np.log(model.dt)
                   + (n_record - model.n_ref) * rate)[:, None]
        log_values = (log_amp[None, :, :] + np.log(np.abs(c))[None, :, :]
                      - log_divisor[:, :, None] - np.log(denominator[:, None, :]))
        # -inf represents an exact zero residue or inactive channel; +inf/NaN
        # here means a singular divisor or invalid denominator, not overflow.
        values = np.exp(np.minimum(log_values, log_max))
        share = np.max(values, axis=(1, 2))
        # The S-parameter report's kept-growing row retains its positive-rate
        # meaning; the judge's total share also covers unresolved decay and DC rate.
        if np.any(rate > 0):
            positive_share = np.max(values[:, rate > 0, :], axis=(1, 2))
    f, k, ch = np.unravel_index(np.argmax(log_values), values.shape)
    worst = (float(s[k].imag / (2.0 * np.pi)), float(np.expm1(min(rate[k], log_max))),
             float(values[f, k, ch]))
    if log_band > np.log(2.0):
        factor = f"{band:.12g}" if log_band <= log_max else f"exp({log_band:.12g})"
        setter = "record length"
        if horizon_pole is not None and math.ceil(Hd) >= n_record:
            p = model.s[horizon_pole]
            setter = f"pole {p.imag / (2*np.pi):.12g} Hz, decay per step={-p.real * model.dt:.12g}"
        return result("cannot judge", f"decay horizon is longer than the identification resolves growth: H={H}, D={model.D}, band_factor={factor} > 2; H set by {setter}")
    if np.any(np.isnan(log_values) | np.isposinf(log_values)):
        return result("cannot judge", "non-finite pole tail bound from zero divisor or denominator")
    if log_values[f, k, ch] > np.log(bar):
        return result("refused", f"kept growing or unresolved pole {worst[0]:.12g} Hz, |lambda|-1={worst[1]:.12g} per step, tail share {worst[2]:.12g} > {bar:g} at bin {freqs[f]:.12g} Hz")
    return result("pass", f"kept growing and unresolved pole tail shares <= {bar:g} at H={H}")

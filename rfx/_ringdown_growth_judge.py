"""Judge kept growing ring-down poles by the tail they can add at each read bin."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

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
    "kept non-static growing poles are judged by their tail share S_k through "
    "twice the judge horizon H (bar 1e-2); H is the larger of the record length "
    "and the hundredfold decay horizon of significant kept decaying poles. "
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


def judge_growing_poles(model, n_record, freqs, denominator, *, unit_tol=1e-6,
                        bar=1e-2, significance=1e-4) -> GrowingPoleJudgement:
    """Judge kept non-static growing poles by their tail bound through ``2H``.

    ``H`` is at least the record length and covers a hundredfold decay of
    every significant kept decaying pole. ``denominator (nf, C)`` is the
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
    grow, exempt = (growing_poles(model, n_record * model.dt, unit_tol)
                    if len(model.s) else ((), ()))

    def result(status, reason):
        return GrowingPoleJudgement(status, status == "pass", reason,
                                    len(grow), len(exempt), H, Hd, band, share, worst)

    if (not np.isfinite(model.s).all() or not np.isfinite(model.c).all()
            or not np.isfinite(freqs).all()
            or not np.all(np.isfinite(denominator) | np.isposinf(denominator))):
        return result("cannot judge", "non-finite poles, residues, bins or denominator")
    decaying, growing = [], []
    for k in range(len(model.s)):
        # Classed by the decay rate's sign, not by a rounded |lambda|: any growth is
        # judged here; a pole with |rate| < 1e-12 that does not grow is a non-decaying
        # sinusoid and keeps the caller's in-window rule. Static poles are exempt.
        if _static_pole(model.pole(k), n_record * model.dt, unit_tol):
            continue
        rate = model.s[k].real * model.dt
        if rate > 0.0:
            growing.append(k)
        elif rate <= -1e-12:
            decaying.append(k)
    none_growing = "no kept non-static growing pole"
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
                Hd = float(np.max(np.log(100.0) / (-dec.s.real[significant] * model.dt)))
                if not np.isfinite(Hd):
                    return (result("cannot judge", "non-finite decay horizon") if growing
                            else result("pass", none_growing))
                H = max(H, math.ceil(Hd))
        if not growing:
            return result("pass", none_growing)
        band = float(np.exp(2.0 * H / model.D * np.log1p(unit_tol)))
        s, c = model.s[growing], model.c[growing]
        rate = s.real * model.dt
        one_minus = np.abs(-np.expm1(s[None, :] * model.dt
                                    - 2j * np.pi * freqs[:, None] * model.dt))
        amp = ((1.0 + np.exp(2.0 * H * rate)) * model.dt
               * np.exp((n_record - model.n_ref) * rate))[:, None] * np.abs(c)
        values = amp[None, :, :] / one_minus[:, :, None] / denominator[:, None, :]
        share = np.max(values, axis=(1, 2))
    if not np.isfinite(band) or not np.isfinite(values).all():
        return result("cannot judge", f"non-finite growing pole share or band_factor; H={H}, D={model.D}, band_factor={band:g}")
    f, k, ch = np.unravel_index(np.argmax(values), values.shape)
    worst = (float(s[k].imag / (2.0 * np.pi)), float(np.expm1(rate[k])),
             float(values[f, k, ch]))
    if band > 2.0:
        return result("cannot judge", f"decay horizon is longer than the identification resolves growth: H={H}, D={model.D}, band_factor={band:.12g} > 2")
    if worst[2] > bar:
        return result("refused", f"kept growing pole {worst[0]:.12g} Hz, |lambda|-1={worst[1]:.12g} per step, tail share {worst[2]:.12g} > {bar:g} at bin {freqs[f]:.12g} Hz")
    return result("pass", f"kept growing pole tail shares <= {bar:g} at H={H}")

"""Paired-loss reference, separate from the frozen AD-Q/E7 order judges.

Protocol: docs/design_notes/20260921_nu_paired_gradient_predeclaration.md.
Host-side float64 arithmetic judges float32 evaluations. A verdict reports
finite-ladder consistency, not global correctness or a rigorous confidence bound.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Final, Literal, TypedDict

import numpy as np

MAX_RELATIVE_BAND: Final = 0.025
GAP_CONTRACTION: Final = 0.75
Verdict = Literal["HELD", "FIRED", "INCONCLUSIVE"]


class LossPair(TypedDict):
    h: float
    loss_plus: float
    loss_minus: float


class InvalidLadder(ValueError):
    """The samples cannot define the declared paired-loss reference."""

    field: str

    def __init__(self, field: str, detail: str) -> None:
        self.field = field
        super().__init__(f"{field}: {detail}")


@dataclass(frozen=True, slots=True)
class Window:
    index: int
    h: float
    reference: float
    roundoff: float
    truncation: float
    band: float
    richardson_ratio: float | None
    model: Literal["richardson", "plateau", "unresolved"]
    gaps: tuple[float, float, float]
    gap_uncertainties: tuple[float, float, float]
    smooth: bool
    resolving: bool
    eligible: bool
    error: float
    within_band: bool


@dataclass(frozen=True, slots=True)
class Judgment:
    verdict: Verdict
    reason: str
    loss0: float
    derivative: float
    sigma: float | None
    samples: tuple[tuple[float, float, float], ...]
    windows: tuple[Window, ...]
    eligible_indices: tuple[int, ...]
    narrowest_index: int | None
    reference_interval: tuple[float, float] | None


def judge_direction(
    points: Sequence[LossPair],
    loss0: float,
    derivative: float,
    *,
    sigma: float | None = None,
    floor_reliable: bool = True,
) -> Judgment:
    """Qualify a reference without consulting the proposed derivative."""
    if len(points) < 3:
        raise InvalidLadder("points", "at least three paired steps are required")
    if not math.isfinite(derivative):
        raise InvalidLadder("derivative", "must be finite")
    if sigma is not None and (not math.isfinite(sigma) or sigma < 0):
        raise InvalidLadder("sigma", "must be finite and nonnegative")
    samples = tuple(
        (float(p["h"]), float(p["loss_plus"]), float(p["loss_minus"]))
        for p in points
    )
    losses = (loss0, *(value for _, lp, lm in samples for value in (lp, lm)))
    if any(not math.isfinite(v) or abs(v) > np.finfo(np.float32).max for v in losses):
        raise InvalidLadder("loss", "must be finite in the float32 evaluation range")
    hs = tuple(p[0] for p in samples)
    if any(not math.isfinite(h) or h <= 0 for h in hs):
        raise InvalidLadder("h", "must be positive and finite")
    if any(hi != 2 * lo for lo, hi in zip(hs, hs[1:])):
        raise InvalidLadder("h", "must be increasing with ratio exactly two")

    measured = 0.0 if sigma is None else sigma
    ds, qs, ks = [], [], []
    for h, lp, lm in samples:
        quantum = max(
            measured,
            *(float(abs(np.spacing(np.float32(abs(v))))) for v in (loss0, lp, lm)),
        )
        ds.append((lp - lm) / (2 * h))
        qs.append(quantum / h)
        ks.append((lp + lm - 2 * loss0) / (2 * h))
    if any(not math.isfinite(v) for values in (ds, qs, ks) for v in values):
        raise InvalidLadder("arithmetic", "loss/step exceeds the host arithmetic range")

    rows = []
    for i in range(1, len(samples) - 1):
        left, right = ds[i] - ds[i - 1], ds[i + 1] - ds[i]
        left_noise, right_noise = 3 * (qs[i - 1] + qs[i]), 3 * (qs[i] + qs[i + 1])
        ratio = right / left if left else None
        plateau = abs(left) <= left_noise and abs(right) <= right_noise
        richardson = (
            abs(left) > left_noise and abs(right) > right_noise
            and ratio is not None and 2.0 <= ratio <= 8.0
        )
        model: Literal["richardson", "plateau", "unresolved"] = "unresolved"
        if richardson:
            model = "richardson"
        elif plateau:
            model = "plateau"
        gaps = (ks[i - 1], ks[i], ks[i + 1])
        gap_noise = (2 * qs[i - 1], 2 * qs[i], 2 * qs[i + 1])
        smooth = all(
            abs(fine) <= GAP_CONTRACTION * abs(coarse)
            + 3 * (fine_noise + GAP_CONTRACTION * coarse_noise)
            for fine, coarse, fine_noise, coarse_noise in zip(
                gaps, gaps[1:], gap_noise, gap_noise[1:]
            )
        )
        truncation = max(4 * abs(left) / 3, abs(right) / 3)
        band = 3 * (truncation + qs[i])
        error = abs(derivative - ds[i])
        if any(not math.isfinite(v) for v in (
            left, right, left_noise, right_noise, truncation, band, error,
            *(value for value in (ratio,) if value is not None),
        )):
            raise InvalidLadder("arithmetic", "triplet exceeds the host arithmetic range")
        resolving = ds[i] != 0 and band / abs(ds[i]) <= MAX_RELATIVE_BAND
        eligible = floor_reliable and smooth and resolving and (richardson or plateau)
        rows.append(Window(
            index=i, h=hs[i], reference=ds[i], roundoff=qs[i],
            truncation=truncation, band=band, richardson_ratio=ratio,
            model=model, gaps=gaps, gap_uncertainties=gap_noise,
            smooth=smooth, resolving=resolving, eligible=eligible,
            error=error, within_band=error <= band,
        ))

    qualified = tuple(row for row in rows if row.eligible)
    verdict: Verdict = "INCONCLUSIVE"
    reason = "unresolved_reference"
    narrowest = None
    interval = None
    if not floor_reliable:
        reason = "unreliable_floor"
    elif qualified:
        narrowest = min(qualified, key=lambda row: (row.band, row.index)).index
        lower = max(row.reference - row.band for row in qualified)
        upper = min(row.reference + row.band for row in qualified)
        interval = (lower, upper)
        if lower > upper:
            reason = "inconsistent_reference"
        elif all(row.within_band for row in qualified):
            verdict, reason = "HELD", "within_reference"
        else:
            verdict, reason = "FIRED", "outside_reference"
    return Judgment(
        verdict=verdict, reason=reason, loss0=loss0, derivative=derivative,
        sigma=sigma, samples=samples, windows=tuple(rows),
        eligible_indices=tuple(row.index for row in qualified),
        narrowest_index=narrowest, reference_interval=interval,
    )

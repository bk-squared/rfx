"""Fringe-resolved comparison of an etalon R(f) against a reference.

WHY THIS EXISTS (issue #812, cv04 row; audit pattern P2 "band-mean collapse").

``validation/crossval/04_multilayer_fresnel.py`` gated a Fabry-Perot
interference pattern with ``mean(|R_rfx - R_analytic|) < 0.05``. The mean's
null space is every zero-mean shape error, so the audit could move the measured
R_max 22.3% low, or build the FDTD slab with eps 12.33% too high, or with a
thickness 8% wrong, and the case still reported PASS. A band mean over fringes
is the wrong shape of metric for this observable.

This module implements the right shape: locate the fringe extrema, then gate
their POSITIONS and their VALUES one fringe at a time.

Two properties are load-bearing and must survive any edit:

1. **The search window cannot entail the verdict.** The detector is
   REFERENCE-ANCHORED, not reference-blind: the analytic slab partitions the
   band into half-fringe cells of width ``FSR/2``, one per extremum, and
   supplies each extremum's kind. What is *measured* is the arg-extremum
   inside its own cell, refined by a parabolic vertex fit; if it lands on a
   cell boundary the gate FAILS with "pinned at the edge of its half-fringe
   cell" rather than silently reporting the boundary. Anchoring is safe
   because the search half-width is far wider than the widest gate window --
   the ratio is
   ``_04_fresnel_results/fringe_gate_geometry.json::windows.non_entailment_ratio``
   for the committed cv04 config -- so "found in the cell" is much looser than
   "passes the gate" and cannot imply it, which is exactly how cv02's
   ``mean_err < 5%`` verdict turned out to be entailed by its own
   ``best_diff < 0.05`` matcher window (200,000 trials, zero failures). A
   defect that moves a fringe by more than ``W`` but less than the cell
   half-width is found and reported as a position failure; a defect that moves
   it out of its cell is reported as a pinning failure. Both regimes fire, and
   both are pinned as ``falsifiers[]`` entries in that same artifact.
2. **The windows are derived, not fitted.** ``position_window_hz`` is built from
   the spectral bin and the exact discrete-Yee dispersion relation;
   ``FRINGE_VALUE_LIMIT`` is built from the dispersion-induced contrast change
   plus the script's own committed run-truncation provenance. Both were frozen
   in ``docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md`` in a
   commit PRECEDING the measurement that judges them. Do not widen either after
   looking at a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

C0_DEFAULT = 2.998e8

# --- frozen thresholds (see the pre-declaration note, section 4/5/6) ---------
#
# SAFETY covers ONE un-derived systematic of the same magnitude as the largest
# derived term (the material-interface staircase / half-cell E-H stagger at the
# two slab faces). It is frozen at 2 and must not be raised after measurement.
SAFETY = 2.0

# Extremum-value window: dispersion-induced contrast change at the top gated
# fringe (0.0055) + 3x the committed rung-C4 band-mean truncation figure
# (3 x 0.005 = 0.015) + negligible vertex quantization (4.3e-5), all x SAFETY.
# = 0.041 -> 0.04, i.e. 11.1% of the analytic fringe contrast R_max - R_min.
FRINGE_VALUE_LIMIT = 0.04

# Each analytic extremum is searched for inside its own half-fringe cell,
# +/- FSR/4 around the analytic position. FSR/4 is the largest half-width that
# cannot contain two extrema of the same kind, so the cells tile the band
# without overlap. Pure geometry, no tuning.
CELL_HALF_WIDTHS_PER_FSR = 0.25

# The located vertex must be at least this many sampled bins away from either
# end of its (possibly band-truncated) cell, otherwise it is "pinned" and the
# gate fails rather than reporting a boundary as an extremum.
PIN_MARGIN_BINS = 2

# Meep leg (section 6). Pointwise |dR| induced by one position budget
# W(f_top) (= windows.max_position_window_hz in the evidence artifact, 234.9 MHz
# for the committed cv04 config, pinned by
# tests/test_crossval_gate_logic.py::test_cv04_fringe_windows_match_the_predeclared_values)
# at the maximum fringe slope |dR/df| = 0.151/GHz is 0.0355; plus the amplitude
# budget 0.04 -> 0.08. Two independent solvers differing from each other get
# twice that.
MEEP_ABS_LIMIT = 0.08
MEEP_CROSS_LIMIT = 0.16


@dataclass(frozen=True)
class Extremum:
    """One located fringe extremum."""

    kind: str  # "max" or "min"
    f_hz: float  # sub-bin refined vertex frequency
    value: float  # sub-bin refined vertex value
    index: int  # index of the sampled bin nearest the vertex


@dataclass(frozen=True)
class FringeRow:
    """One measured/reference extremum pair and its two verdicts."""

    kind: str
    f_ref_hz: float
    f_meas_hz: float
    df_hz: float
    f_window_hz: float
    value_ref: float
    value_meas: float
    dvalue: float
    value_limit: float

    @property
    def position_ok(self) -> bool:
        return abs(self.df_hz) <= self.f_window_hz

    @property
    def value_ok(self) -> bool:
        return abs(self.dvalue) <= self.value_limit


@dataclass(frozen=True)
class FringeVerdict:
    ok: bool
    rows: tuple[FringeRow, ...]
    reasons: tuple[str, ...]  # empty iff ok


# ---------------------------------------------------------------------------
# analytic side
# ---------------------------------------------------------------------------
def slab_R_max(eps_r: float, eps_out: float = 1.0) -> float:
    """Peak reflectance of a lossless slab at normal incidence."""
    return float(((eps_r - eps_out) / (eps_r + eps_out)) ** 2)


def analytic_slab_extrema(
    eps_r: float,
    d: float,
    f_lo: float,
    f_hi: float,
    *,
    c0: float = C0_DEFAULT,
) -> list[tuple[str, float, float]]:
    """Analytic (continuum) R(f) extrema of a slab, as (kind, f_hz, value).

    R maxima at delta = (m + 1/2) pi, R minima at delta = m pi, with
    delta = 2 pi f n d / c and n = sqrt(eps_r); free spectral range
    FSR = c / (2 n d). Only extrema strictly inside (f_lo, f_hi) are returned.
    """
    n = math.sqrt(eps_r)
    fsr = c0 / (2.0 * n * d)
    r_max = slab_R_max(eps_r)
    out: list[tuple[str, float, float]] = []
    m = 0
    while True:
        f_max = (m + 0.5) * fsr
        f_min = (m + 1.0) * fsr
        if f_max > f_hi and f_min > f_hi:
            break
        if f_lo < f_max < f_hi:
            out.append(("max", f_max, r_max))
        if f_lo < f_min < f_hi:
            out.append(("min", f_min, 0.0))
        m += 1
    out.sort(key=lambda row: row[1])
    return out


# ---------------------------------------------------------------------------
# discrete-Yee dispersion
# ---------------------------------------------------------------------------
def yee_numerical_wavenumber(
    f_hz: float, n_index: float, dx: float, dt: float, *, c0: float = C0_DEFAULT
) -> float:
    """k~ from the 1-D Yee relation sin(omega dt/2) = S_m sin(k~ dx/2)."""
    s_m = c0 * dt / (n_index * dx)
    arg = math.sin(math.pi * f_hz * dt) / s_m
    if not -1.0 <= arg <= 1.0:
        raise ValueError(
            f"Yee dispersion relation has no real solution at f={f_hz:.4e} Hz "
            f"(sin(omega dt/2)/S_m = {arg:.4f}); the sampled band is above the "
            "numerical cutoff of this grid."
        )
    return (2.0 / dx) * math.asin(arg)


def yee_dispersion_shift_hz(
    f_hz: float, n_index: float, dx: float, dt: float, *, c0: float = C0_DEFAULT
) -> float:
    """|f_numerical - f_analytic| for a fringe extremum of a slab of index n.

    A fringe extremum is fixed by the round-trip phase inside the slab, so the
    numerically-observed extremum sits where the DISCRETE wavenumber equals the
    continuum wavenumber of the analytic extremum: k~(f_num) = k_exact(f_an).
    Solved exactly (Brent), not by Taylor expansion.
    """
    target = 2.0 * math.pi * f_hz * n_index / c0

    def residual(f: float) -> float:
        return yee_numerical_wavenumber(f, n_index, dx, dt, c0=c0) - target

    lo, hi = 0.5 * f_hz, f_hz
    # k~ >= k always for the Yee scheme at S_m < 1, so the root is below f_hz.
    if residual(lo) * residual(hi) > 0.0:
        # Degenerate/near-continuum grid: no measurable shift.
        return 0.0
    f_num = brentq(residual, lo, hi, xtol=1.0)
    return abs(f_num - f_hz)


def position_window_hz(
    f_hz: float,
    *,
    n_index: float,
    dx: float,
    dt: float,
    df_bin_hz: float,
    c0: float = C0_DEFAULT,
    safety: float = SAFETY,
) -> float:
    """W(f) = safety * (df_bin/2 + |Yee dispersion shift(f)|).

    Derived in docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md
    section 4. The per-fringe values for the committed cv04 config are in
    ``validation/crossval/_04_fresnel_results/fringe_gate_geometry.json``
    under ``windows.fringes[].position_window_hz``, regenerated on every run of
    ``tests/test_crossval_gate_logic.py``.
    """
    return safety * (
        0.5 * df_bin_hz
        + yee_dispersion_shift_hz(f_hz, n_index, dx, dt, c0=c0)
    )


# ---------------------------------------------------------------------------
# cell-anchored extremum detection
#
# NOT reference-blind. The reference supplies the cell centre and the kind; the
# measurement supplies the arg-extremum inside the cell. See property 1 in the
# module docstring for why anchoring cannot entail the verdict, and the
# pre-declaration note section 9 for the reference-blind prominence detector
# that was implemented, measured, and withdrawn for failing criterion (A) on
# correct code.
# ---------------------------------------------------------------------------
def _parabolic_vertex(
    y_left: float, y_mid: float, y_right: float
) -> tuple[float, float]:
    """Sub-bin vertex (offset in bins, value) of the parabola through 3 points."""
    denom = y_left - 2.0 * y_mid + y_right
    if denom == 0.0:
        return 0.0, y_mid
    delta = 0.5 * (y_left - y_right) / denom
    delta = max(-0.5, min(0.5, delta))
    value = y_mid - 0.25 * (y_left - y_right) * delta
    return delta, value


def locate_extremum_in_cell(
    freqs_hz: np.ndarray,
    values: np.ndarray,
    *,
    kind: str,
    f_center_hz: float,
    half_width_hz: float,
    pin_margin_bins: int = PIN_MARGIN_BINS,
) -> tuple[Extremum | None, str]:
    """Arg-extremum of ``values`` inside one half-fringe cell, sub-bin refined.

    Returns ``(extremum, "")`` on success or ``(None, reason)`` when the cell
    holds too few samples or the arg-extremum is pinned within
    ``pin_margin_bins`` of a cell edge. Pinning is a FAILURE, never a silently
    reported boundary value: it is the signal that the real extremum has left
    its own half-fringe cell.

    ``freqs_hz`` must be uniformly sampled (checked).
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    values = np.asarray(values, dtype=float)
    if freqs_hz.shape != values.shape or freqs_hz.ndim != 1:
        raise ValueError("freqs_hz and values must be 1-D arrays of equal length")
    if freqs_hz.size >= 2:
        steps = np.diff(freqs_hz)
        df = float(steps.mean())
        if not np.allclose(steps, df, rtol=1e-6, atol=0.0):
            raise ValueError(
                "fringe location requires a uniformly sampled frequency axis"
            )
    else:
        raise ValueError("fringe location needs at least 2 samples")

    sel = np.nonzero(
        np.abs(freqs_hz - f_center_hz) <= half_width_hz
    )[0]
    need = 2 * pin_margin_bins + 1
    if sel.size < need:
        return None, (
            f"the {kind} cell centred at {f_center_hz/1e9:.4f} GHz holds only "
            f"{sel.size} sampled bins (needs {need}); the evaluated band does "
            "not resolve this fringe"
        )

    lo, hi = int(sel[0]), int(sel[-1])
    window = values[lo:hi + 1]
    j = int(np.argmax(window) if kind == "max" else np.argmin(window))
    i = lo + j

    if j < pin_margin_bins or (window.size - 1 - j) < pin_margin_bins:
        return None, (
            f"the measured {kind} is PINNED at the edge of its half-fringe "
            f"cell: arg-{kind} sits at {freqs_hz[i]/1e9:.4f} GHz, "
            f"{min(j, window.size - 1 - j)} bin(s) from the cell edge "
            f"[{freqs_hz[lo]/1e9:.4f}, {freqs_hz[hi]/1e9:.4f}] GHz around the "
            f"analytic {kind} at {f_center_hz/1e9:.4f} GHz. The extremum this "
            "gate exists to compare has left its own fringe cell."
        )

    sign = 1.0 if kind == "max" else -1.0
    delta, vertex = _parabolic_vertex(
        sign * values[i - 1], sign * values[i], sign * values[i + 1]
    )
    return (
        Extremum(
            kind=kind,
            f_hz=float(freqs_hz[i] + delta * df),
            value=float(sign * vertex),
            index=int(i),
        ),
        "",
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def compare_fringes(
    freqs_hz: np.ndarray,
    r_measured: np.ndarray,
    *,
    eps_r: float,
    d: float,
    n_index: float | None,
    dx: float,
    dt: float,
    df_bin_hz: float,
    c0: float = C0_DEFAULT,
    value_limit: float = FRINGE_VALUE_LIMIT,
    safety: float = SAFETY,
    label: str = "measured",
) -> FringeVerdict:
    """Gate a measured etalon R(f) against the analytic slab fringe structure.

    Three verdicts, all binding:
      * containment -- every analytic extremum the band resolves must have a
        measured arg-extremum strictly interior to its own half-fringe cell
        (a pinned one is a failure, not a reported boundary);
      * position -- each |f_measured - f_analytic| <= W(f_analytic);
      * value -- each |R_measured(vertex) - R_analytic(extremum)| <= value_limit.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    r_measured = np.asarray(r_measured, dtype=float)
    f_lo, f_hi = float(freqs_hz[0]), float(freqs_hz[-1])

    n_idx = math.sqrt(eps_r) if n_index is None else n_index
    fsr = c0 / (2.0 * n_idx * d)
    cell_half_width = CELL_HALF_WIDTHS_PER_FSR * fsr

    expected_all = analytic_slab_extrema(eps_r, d, f_lo, f_hi, c0=c0)
    # Keep only extrema whose analytic position plus its own gate window plus
    # one bin fits inside the evaluated band -- anything closer to an edge
    # cannot be judged at the declared resolution and is out of scope here.
    expected: list[tuple[str, float, float]] = []
    windows: list[float] = []
    for kind, f_an, v_an in expected_all:
        w = position_window_hz(
            f_an,
            n_index=n_idx,
            dx=dx,
            dt=dt,
            df_bin_hz=df_bin_hz,
            c0=c0,
            safety=safety,
        )
        if f_an - w - df_bin_hz < f_lo or f_an + w + df_bin_hz > f_hi:
            continue
        expected.append((kind, f_an, v_an))
        windows.append(w)

    reasons: list[str] = []
    if not expected:
        return FringeVerdict(
            ok=False,
            rows=(),
            reasons=(
                f"{label}: the evaluated band {f_lo/1e9:.3f}-{f_hi/1e9:.3f} GHz "
                "contains no analytic fringe extremum it can judge at the "
                "declared resolution -- this gate cannot decide, which is a "
                "FAIL, not a pass.",
            ),
        )

    found: list[Extremum] = []
    for kind, f_an, _v_an in expected:
        try:
            meas, why = locate_extremum_in_cell(
                freqs_hz, r_measured, kind=kind,
                f_center_hz=f_an, half_width_hz=cell_half_width,
            )
        except ValueError as exc:
            return FringeVerdict(
                ok=False,
                rows=(),
                reasons=(
                    f"{label}: the evaluated frequency axis is not uniformly "
                    f"sampled, so fringe extrema cannot be located ({exc}). "
                    "The spectral mask must select one contiguous band.",
                ),
            )
        if meas is None:
            reasons.append(f"{label}: fringe CONTAINMENT -- {why}")
        else:
            found.append(meas)

    if reasons:
        return FringeVerdict(ok=False, rows=(), reasons=tuple(reasons))

    rows: list[FringeRow] = []
    for (kind, f_an, v_an), w, meas in zip(expected, windows, found):
        rows.append(
            FringeRow(
                kind=kind,
                f_ref_hz=f_an,
                f_meas_hz=meas.f_hz,
                df_hz=meas.f_hz - f_an,
                f_window_hz=w,
                value_ref=v_an,
                value_meas=meas.value,
                dvalue=meas.value - v_an,
                value_limit=value_limit,
            )
        )

    for row in rows:
        if not row.position_ok:
            reasons.append(
                f"{label}: fringe POSITION -- {row.kind} at "
                f"{row.f_meas_hz/1e9:.4f} GHz vs analytic "
                f"{row.f_ref_hz/1e9:.4f} GHz: dev {row.df_hz/1e6:+.1f} MHz "
                f"exceeds window {row.f_window_hz/1e6:.1f} MHz "
                f"(= {abs(row.df_hz)/row.f_window_hz:.2f}x)."
            )
        if not row.value_ok:
            reasons.append(
                f"{label}: fringe VALUE -- {row.kind} at "
                f"{row.f_meas_hz/1e9:.4f} GHz reads R={row.value_meas:.4f} vs "
                f"analytic R={row.value_ref:.4f}: dev {row.dvalue:+.4f} exceeds "
                f"limit {row.value_limit:.4f} "
                f"(= {abs(row.dvalue)/row.value_limit:.2f}x)."
            )

    return FringeVerdict(ok=not reasons, rows=tuple(rows), reasons=tuple(reasons))


def format_fringe_table(verdict: FringeVerdict, label: str) -> str:
    """Human-readable per-fringe table for the crossval stdout."""
    lines = [
        f"  Fringe-resolved gate ({label}) — issue #812, per-extremum, "
        "not a band mean:",
        f"    {'kind':>4} {'f_ref(GHz)':>11} {'f_meas(GHz)':>12} {'df(MHz)':>9} "
        f"{'W(MHz)':>8} {'R_ref':>8} {'R_meas':>8} {'dR':>8} {'verdict':>9}",
    ]
    for row in verdict.rows:
        ok = row.position_ok and row.value_ok
        lines.append(
            f"    {row.kind:>4} {row.f_ref_hz/1e9:>11.4f} "
            f"{row.f_meas_hz/1e9:>12.4f} {row.df_hz/1e6:>+9.1f} "
            f"{row.f_window_hz/1e6:>8.1f} {row.value_ref:>8.4f} "
            f"{row.value_meas:>8.4f} {row.dvalue:>+8.4f} "
            f"{'ok' if ok else 'FAIL':>9}"
        )
    for reason in verdict.reasons:
        lines.append(f"    !! {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# external-solver (E4) leg
# ---------------------------------------------------------------------------
def external_pointwise_reasons(
    r_err_ref: np.ndarray,
    t_err_ref: np.ndarray,
    r_cross: np.ndarray,
    t_cross: np.ndarray,
    *,
    solver: str = "Meep",
    abs_limit: float = MEEP_ABS_LIMIT,
    cross_limit: float = MEEP_CROSS_LIMIT,
) -> list[str]:
    """Pointwise gates that put the EXTERNAL SOLVER'S NUMBERS in the verdict.

    ``r_err_ref``/``t_err_ref`` are |solver - analytic|; ``r_cross``/``t_cross``
    are |rfx - solver|, both over the solver's valid band. Everything is a
    per-bin MAXIMUM, never a band mean -- a band mean over an interference
    pattern is the shape of metric issue #812 flagged as pattern P2.

    Before #812 cv04 computed every one of these arrays and printed all of
    them, and the exit code depended only on whether ``import meep`` had
    succeeded: the E4 label was carried by an import, not by a verdict.
    """
    reasons: list[str] = []
    abs_worst = max(float(np.max(r_err_ref)), float(np.max(t_err_ref)))
    if abs_worst > abs_limit:
        reasons.append(
            f"{solver} vs analytic: max|dR|={np.max(r_err_ref):.4f}, "
            f"max|dT|={np.max(t_err_ref):.4f} — worst {abs_worst:.4f} exceeds "
            f"the pointwise limit {abs_limit:.4f} "
            f"(= {abs_worst / abs_limit:.2f}x)"
        )
    cross_worst = max(float(np.max(r_cross)), float(np.max(t_cross)))
    if cross_worst > cross_limit:
        reasons.append(
            f"rfx vs {solver}: max|dR|={np.max(r_cross):.4f}, "
            f"max|dT|={np.max(t_cross):.4f} — worst {cross_worst:.4f} exceeds "
            f"the cross-solver limit {cross_limit:.4f} "
            f"(= {cross_worst / cross_limit:.2f}x)"
        )
    return reasons

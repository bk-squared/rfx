"""Issue #786 — the ladder-reading precondition this lane's D5 licenses.

NOT a remedy for any mechanism, and NOT an rfx code change. Nothing in
rfx/ is touched by this lane. This is a research-package helper that
states, in code, the precondition the W4R ladder violated:

    An order fitted from a self-referential ladder is meaningless unless
    the observable is MONOTONE over the fit range, the anchor lies on the
    SAME monotone branch as the fit points, and the interval between the
    finest fit point and the anchor is actually SAMPLED.

The last clause is the one that matters in practice: on PR #785's own
five rungs the first two checks pass -- those five ARE monotone -- and
only the rungs D5 added reveal the turn. P5 is what would have caught it
without them: the W4R anchor sits a factor 2.0 below the finest fit
point while the ladder's own largest internal step is 1.5, so the whole
turn hid inside an interval the ladder never sampled.

D5 measured the P-C fixture's f(h) rising to a maximum at
h = dz_fine = 0.125 mm and then descending over four further rungs. The
PR #785 ladder fitted the ascending branch and anchored on a rung four
rungs down the DESCENDING branch, so |f - f_ref| was never an error
sequence at all -- which is the whole of the "~4e-3 floor".

Run the self-check:
    PYTHONPATH=. python -m validation.research.convergence_floor.ladder_guard
"""

from __future__ import annotations

import numpy as np


def check_ladder(h, f, h_ref=None, f_ref=None, instrument_hz=1.0e6,
                 min_points=3) -> dict:
    """Preconditions for quoting a convergence order from (h, f).

    Parameters
    ----------
    h, f : sequences
        Cell size and observable, in any consistent units (f in Hz for
        ``instrument_hz`` to mean anything).
    h_ref, f_ref : float or None
        The anchor rung, when the ladder is self-referential.
    instrument_hz : float
        The extraction instrument's own uncertainty, MEASURED (this lane
        measured 6-17 kHz for Harminv on the W4R record class via an
        exact-reference twin; the 1 MHz default is the frozen
        Cramer-Rao-derived 'sound' level, deliberately conservative).
    min_points : int
        Points required on the monotone branch.

    Returns
    -------
    dict with ``ok`` and a list of ``reasons`` (empty when ok).
    """
    h = np.asarray(h, float)
    f = np.asarray(f, float)
    order = np.argsort(-h)                 # coarse -> fine
    h, f = h[order], f[order]
    d = np.diff(f)
    signs = np.sign(d)
    changes = int(np.sum(signs[1:] != signs[:-1]))
    reasons = []

    if changes > 0:
        turn = int(np.argmax(np.abs(np.diff(signs))) + 1)
        reasons.append(
            "P1 NOT MONOTONE: f(h) turns over at h = %.6g (%d sign change(s) "
            "in the consecutive differences). No single-power-law order is "
            "defined across a turning point." % (h[turn], changes))
        branch = slice(turn, len(h))
    else:
        branch = slice(0, len(h))

    n_branch = len(h[branch])
    if n_branch < min_points:
        reasons.append("P3 TOO FEW POINTS: %d on the monotone branch, "
                       "need %d." % (n_branch, min_points))

    if f_ref is not None:
        lo, hi = float(min(f[branch])), float(max(f[branch]))
        if not (lo <= f_ref <= hi) and changes > 0:
            reasons.append(
                "P2 ANCHOR OFF-BRANCH: the reference f = %.6g lies outside "
                "the monotone branch's range [%.6g, %.6g]; |f - f_ref| is "
                "not an error sequence." % (f_ref, lo, hi))
        if h_ref is not None and changes > 0 and h_ref < h[-1]:
            # anchor finer than every fit point AND the curve turned:
            # the anchor is on the far side of the turn.
            reasons.append(
                "P2 ANCHOR PAST THE TURN: h_ref = %.6g is finer than every "
                "fit point and the curve has a turning point between them."
                % h_ref)
        if h_ref is not None and h_ref < h[-1]:
            gap = h[-1] / h_ref
            steps = h[:-1] / h[1:]
            worst_step = float(steps.max()) if len(steps) else 1.0
            if gap > worst_step * (1 + 1e-9):
                reasons.append(
                    "P5 ANCHOR GAP UNSAMPLED: the anchor sits a factor %.3f "
                    "below the finest fit point with NO rung in between, "
                    "while the ladder's own largest internal step is only "
                    "%.3f. A turning point inside an interval you never "
                    "sample is undetectable." % (gap, worst_step))
        gaps = np.abs(f[branch] - f_ref)
        if gaps.min() < 3 * instrument_hz:
            reasons.append(
                "P4 NO INSTRUMENT MARGIN: the closest fit point is %.6g from "
                "the anchor, under 3x the instrument's own %.6g."
                % (float(gaps.min()), instrument_hz))

    return {"ok": not reasons, "reasons": reasons,
            "sign_changes": changes,
            "monotone_branch_points": int(n_branch)}


def _selfcheck():
    # The W4R uniform ladder exactly as PR #785 fitted it (D0 reproduced
    # every value bit-identically), plus D5's three added rungs.
    s_fit = [1.5, 1.0, 0.75, 0.6, 0.5]
    f_fit = [5404546169.206672, 5502114526.260705, 5533066155.959281,
             5542444437.5726, 5543558293.505646]
    s_all = s_fit + [3 / 7, 0.375, 1 / 3]
    f_all = f_fit + [5541260000.0, 5537579000.0, 5533355000.0]
    f_ref, s_ref = 5520820807.318383, 0.25

    print("PR #785's own five rungs, anchored on s=0.25:")
    r = check_ladder([0.25e-3 * s for s in s_fit], f_fit,
                     h_ref=0.25e-3 * s_ref, f_ref=f_ref)
    for x in r["reasons"]:
        print("   FIRES:", x)
    print("   ok =", r["ok"])

    print("\nThe same ladder with D5's three added rungs:")
    r = check_ladder([0.25e-3 * s for s in s_all], f_all,
                     h_ref=0.25e-3 * s_ref, f_ref=f_ref)
    for x in r["reasons"]:
        print("   FIRES:", x)
    print("   ok =", r["ok"])
    assert not r["ok"], "the guard must fire on the W4R ladder"


if __name__ == "__main__":
    _selfcheck()

"""Pin `ladder_guard.check_ladder`'s two self-check outcomes (PR #788 review).

`validation/research/convergence_floor/ladder_guard.py` states the
precondition for quoting a convergence order off a self-referential
ladder (monotone fit range, anchor on the same branch, the anchor-to-
fit-point gap actually sampled). This is a research-lane helper, not
rfx/ code, but its self-check (``python -m
validation.research.convergence_floor.ladder_guard``) is not wired into
CI, so a change to `check_ladder` could silently stop firing on the two
committed cases the reviewer measured while working the PR. This test
locks both outcomes using the committed rung values from
`validation/research/convergence_floor/results/{d0_reproduction,d5_turnover}.json`
(no simulation, CPU-instant).
"""
from __future__ import annotations

import json
import os

from validation.research.convergence_floor import fixture as fx
from validation.research.convergence_floor.ladder_guard import check_ladder

RES = os.path.join(os.path.dirname(fx.__file__), "results")


def _load_uc_targets():
    d0 = json.load(open(os.path.join(RES, "d0_reproduction.json")))
    return {r["scale"]: r["f_target"] for r in d0["rows"] if not r["multiband"]}


def _load_d5_targets():
    d5 = json.load(open(os.path.join(RES, "d5_turnover.json")))
    return {r["scale"]: r["f_target"] for r in d5["rows"]}


def test_five_rung_ladder_fires_anchor_gap_unsampled():
    """PR #785's own five rungs, anchored on s=0.25: only P5 fires.

    The five fit rungs (s in fx.SCALES) are themselves monotone, so P1
    does not fire here -- the guard's only complaint is that the anchor
    sits past an interval the ladder never sampled, which is exactly
    what D5 (three added rungs) later showed hides the turn-over.
    """
    uc = _load_uc_targets()
    fit_scales = sorted(fx.SCALES, reverse=True)
    h_fit = [fx.PC_DZF0 * s for s in fit_scales]
    f_fit = [uc[s] for s in fit_scales]
    h_ref = fx.PC_DZF0 * fx.REF_SCALE
    f_ref = uc[fx.REF_SCALE]

    r = check_ladder(h_fit, f_fit, h_ref=h_ref, f_ref=f_ref)

    assert not r["ok"]
    assert any(reason.startswith("P5 ANCHOR GAP UNSAMPLED")
              for reason in r["reasons"]), r["reasons"]
    assert r["sign_changes"] == 0


def test_nine_rung_ladder_fires_not_monotone_and_anchor_off_branch():
    """With D5's three added rungs the ladder turns over: P1 and both
    P2 clauses fire, and the earlier P5 finding is superseded (the gap
    IS sampled now -- that's the point of D5)."""
    uc = _load_uc_targets()
    d5 = _load_d5_targets()
    fit_scales = sorted(fx.SCALES, reverse=True)
    all_scales = sorted(list(fit_scales) + list(d5), reverse=True)
    h_all = [fx.PC_DZF0 * s for s in all_scales]
    f_all = [uc[s] if s in uc else d5[s] for s in all_scales]
    h_ref = fx.PC_DZF0 * fx.REF_SCALE
    f_ref = uc[fx.REF_SCALE]

    r = check_ladder(h_all, f_all, h_ref=h_ref, f_ref=f_ref)

    assert not r["ok"]
    assert r["sign_changes"] == 1
    prefixes = ("P1 NOT MONOTONE", "P2 ANCHOR OFF-BRANCH",
               "P2 ANCHOR PAST THE TURN")
    for prefix in prefixes:
        assert any(reason.startswith(prefix) for reason in r["reasons"]), (
            "expected %r to fire; got %r" % (prefix, r["reasons"]))

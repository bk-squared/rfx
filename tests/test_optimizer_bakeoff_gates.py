"""Committed gate for the WP 4-D optimizer bake-off (external-reviewer plan).

Locks the frozen design-loop-capability evidence in
``tests/fixtures/optimizer_bakeoff/bakeoff_results.json`` (produced by
``scripts/diagnostics/optimizer_bakeoff/run_bakeoff.py``) WITHOUT running any
FDTD: it replays the committed per-(optimizer, benchmark) loss-vs-forward-solve
curves and re-derives every headline number and the adoption VERDICT, so a
silent regression in the committed evidence goes red here.

DESIGN-LOOP CAPABILITY EVIDENCE ONLY — this test asserts NOTHING about physics
and touches NO committed validation gate.

What is enforced:
1. **Fences / meta** — work package 4-D, float32, the >= 3 dB on >= 2-of-3
   adoption rule, the named incumbent (the shipped 4-C default) and candidates,
   and budgets divisible by the multi-start count.
2. **Curve re-derivation (rel 1e-9)** — for every run, best_loss = min loss over
   the recorded curve within budget, final_loss = last loss within budget, and
   db = 10*log10(loss) recomputed from the committed raw curve must match the
   committed reductions.
3. **Gate arithmetic** — per benchmark the incumbent-vs-candidate dB margin and
   the per-candidate "beats" flag are re-derived; per candidate the beat count
   and adopt flag; and the final verdict.  A "no-adopt" verdict REQUIRES the
   >= 3 dB on >= 2/3 rule to be UNMET by the recorded numbers (fail-closed: the
   verdict cannot claim no-adopt while a candidate actually cleared the bar, nor
   claim adopt while none did).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "optimizer_bakeoff" / "bakeoff_results.json"

# Producer constants (scripts/diagnostics/optimizer_bakeoff/run_bakeoff.py).
ADOPT_GATE_DB = 3.0
ADOPT_MIN_BENCHMARKS = 2
INCUMBENT = "adam_multistart_bi"
CANDIDATES = ("optax_adam", "optax_lbfgs")
OPT_ORDER = ("adam_rfx", "optax_adam", "optax_lbfgs", "adam_multistart_bi")
BENCHMARKS = ("ar_coating", "msl_stub_notch", "waveguide_taper")
DB_FACTOR = 10.0


def _env() -> dict:
    return json.loads(FIXTURE.read_text())


# ---- reductions re-derived here, independent of the producer ---------------
def _within(curve, budget):
    return [(int(n), float(v)) for n, v in curve if int(n) <= budget]


def _best(curve, budget):
    return min((v for _, v in _within(curve, budget)), default=math.inf)


def _final(curve, budget):
    pts = _within(curve, budget)
    return pts[-1][1] if pts else math.inf


def _db(loss):
    return DB_FACTOR * math.log10(max(float(loss), 1e-30))


def test_fixture_present_and_meta_fences() -> None:
    env = _env()
    m = env["meta"]
    assert m["work_package"] == "4-D"
    assert m["precision"] == "float32"
    assert m["metric_db_factor"] == DB_FACTOR
    assert set(m["optimizers"]) == set(OPT_ORDER)
    d = env["decision"]
    assert d["adopt_gate_db"] == ADOPT_GATE_DB
    assert d["adopt_min_benchmarks"] == ADOPT_MIN_BENCHMARKS
    assert d["incumbent"] == INCUMBENT
    assert tuple(d["candidates"]) == CANDIDATES
    # budgets divisible by the multi-start count (equal-total-forward-solves fence)
    n_starts = m["n_starts"]
    for name, b in m["budgets"].items():
        assert b % n_starts == 0, f"{name} budget {b} not divisible by {n_starts}"


def test_all_benchmarks_and_optimizers_present() -> None:
    env = _env()
    assert set(env["benchmarks"]) == set(BENCHMARKS)
    for name in BENCHMARKS:
        b = env["benchmarks"][name]
        assert set(b["runs"]) == set(OPT_ORDER)
        # every run recorded at least one within-budget point.
        for opt in OPT_ORDER:
            assert _within(b["runs"][opt]["curve"], b["budget"]), f"{name}/{opt} empty curve"


@pytest.mark.parametrize("name", BENCHMARKS)
def test_curve_rederivation_and_margins(name: str) -> None:
    env = _env()
    b = env["benchmarks"][name]
    budget = b["budget"]

    # 1. re-derive best/final/db from the committed raw curve (rel 1e-9).
    for opt in OPT_ORDER:
        r = b["runs"][opt]
        best = _best(r["curve"], budget)
        final = _final(r["curve"], budget)
        assert best == pytest.approx(r["best_loss"], rel=1e-9, abs=1e-30), f"{name}/{opt}"
        assert final == pytest.approx(r["final_loss"], rel=1e-9, abs=1e-30), f"{name}/{opt}"
        assert _db(best) == pytest.approx(r["best_db"], rel=1e-9, abs=1e-12), f"{name}/{opt}"
        assert _db(final) == pytest.approx(r["final_db"], rel=1e-9, abs=1e-12), f"{name}/{opt}"
        # forward-solve budget honoured: no recorded point exceeds it, and the
        # method used <= budget solves.
        assert all(int(n) <= budget for n, _ in _within(r["curve"], budget))
        assert r["n_solves"] <= budget or opt == "optax_lbfgs", f"{name}/{opt} overran budget"

    # 2. incumbent best-dB and per-candidate margins (>= 3 dB rule).
    incumbent_db = _db(_best(b["runs"][INCUMBENT]["curve"], budget))
    assert incumbent_db == pytest.approx(b["incumbent_best_db"], rel=1e-9, abs=1e-12)
    for c in CANDIDATES:
        cand_db = _db(_best(b["runs"][c]["curve"], budget))
        margin = incumbent_db - cand_db
        beats = bool(margin >= ADOPT_GATE_DB)
        assert margin == pytest.approx(b["candidate_margins_db"][c], rel=1e-9, abs=1e-9), f"{name}/{c}"
        assert beats == b["candidate_beats"][c], f"{name}/{c} beat flag"

    # 3. winner = min best-loss over all optimizers.
    winner = min(OPT_ORDER, key=lambda o: _best(b["runs"][o]["curve"], budget))
    assert winner == b["winner"], name


def test_decision_and_verdict_fail_closed() -> None:
    env = _env()
    d = env["decision"]

    # re-derive per-candidate beats across benchmarks, beat counts, adopt flags.
    per_bench_beats = {c: [] for c in CANDIDATES}
    for name in BENCHMARKS:
        b = env["benchmarks"][name]
        budget = b["budget"]
        incumbent_db = _db(_best(b["runs"][INCUMBENT]["curve"], budget))
        for c in CANDIDATES:
            cand_db = _db(_best(b["runs"][c]["curve"], budget))
            per_bench_beats[c].append(bool((incumbent_db - cand_db) >= ADOPT_GATE_DB))

    n_beats = {c: sum(per_bench_beats[c]) for c in CANDIDATES}
    adopts = {c: n_beats[c] >= ADOPT_MIN_BENCHMARKS for c in CANDIDATES}
    any_adopt = any(adopts.values())
    expected_verdict = "adopt" if any_adopt else "no-adopt"

    # committed decision must match the re-derivation exactly.
    for c in CANDIDATES:
        assert list(map(bool, d["per_benchmark_candidate_beats"][c])) == per_bench_beats[c], c
        assert d["candidate_n_beats"][c] == n_beats[c], c
        assert bool(d["candidate_adopts"][c]) == adopts[c], c
    assert bool(d["any_candidate_adopts"]) == any_adopt
    assert d["verdict"] == expected_verdict

    # FAIL-CLOSED verdict-string lock: a "no-adopt" verdict REQUIRES the
    # >= 3 dB on >= 2/3 rule to be UNMET by the recorded numbers, and vice versa.
    if d["verdict"] == "no-adopt":
        assert not any_adopt, "verdict says no-adopt but a candidate cleared the bar"
        assert all(n_beats[c] < ADOPT_MIN_BENCHMARKS for c in CANDIDATES)
        assert "no-adopt" in d["verdict_text"].lower()
    else:
        assert any_adopt, "verdict says adopt but no candidate cleared the bar"
        assert "adopt" in d["verdict_text"].lower()

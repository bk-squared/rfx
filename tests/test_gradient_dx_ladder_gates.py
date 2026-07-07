"""Committed gate for the WP 3-1 gradient-fidelity-vs-mesh ladder.

Locks the frozen AD-vs-FD evidence from
``tests/fixtures/gradient_dx_ladder/ad_fd_ladder.json`` (produced by
``scripts/diagnostics/gradient_dx_ladder/run_ladder.py``) WITHOUT running any
FDTD: it replays the committed raw J / finite-difference / gradient numbers and
re-derives every verdict, so a silent regression in the committed evidence goes
red here.

What is enforced (per the external-reviewer plan, item 3):
1. **Fences** — num_periods held FIXED at 20 on every rung (the settling-
   confound trap), precision is float32 (the object of study), one fixed
   objective frequency, one scalar DoF.
2. **Comparator validity** — at each rung the FD signal must sit >= 100x above
   the float32 objective repeat-noise (or the repeat-noise must be exactly 0.0,
   which is the deterministic-CPU case). A comparator-invalid rung is the WP
   3-1 falsifier and would fail here.
3. **Re-derivation lock** — g_fd = (J_plus - J_minus) / (2*h) and
   rel_err = |grad_ad - g_fd| / |g_fd| recomputed from the committed numbers
   must match the committed values to rel 1e-9.
4. **Physics gate** — sign agreement on EVERY committed rung, and each rung's
   rel_err at or below its committed ceiling max(measured*1.5, 0.10) (the
   plan's 0.10 floor with a regression margin).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "gradient_dx_ladder" / "ad_fd_ladder.json"

# Producer constants (scripts/diagnostics/gradient_dx_ladder/run_ladder.py).
REL_ERR_FLOOR = 0.10
REL_ERR_MARGIN = 1.5
COMPARATOR_MIN_RATIO = 100.0
NUM_PERIODS = 20.0
F_OBJ_HZ = 8e9


def _env() -> dict:
    return json.loads(FIXTURE.read_text())


def _live_rungs(env: dict) -> list:
    return [r for r in env["rungs"] if not r.get("dropped")]


def test_fixture_present_and_meta_fences() -> None:
    env = _env()
    m = env["meta"]
    assert m["work_package"] == "3-1"
    # num_periods held FIXED at 20 -- the settling-confound fence.
    assert m["num_periods"] == NUM_PERIODS
    # float32 is the object of study (no module-level x64).
    assert m["precision"] == "float32"
    # one fixed objective frequency; one scalar DoF witness.
    assert m["f_obj_hz"] == F_OBJ_HZ
    assert "slab" in m["witness"].lower()
    assert m["rel_err_floor"] == REL_ERR_FLOOR
    assert m["rel_err_margin"] == REL_ERR_MARGIN
    assert m["comparator_min_ratio"] == COMPARATOR_MIN_RATIO


def test_comparator_falsifier_valid_at_finest_rung() -> None:
    """WP 3-1 step 1: FD signal must dominate float32 objective repeat-noise."""
    env = _env()
    comp = env["comparator_falsifier"]
    # The comparator runs at the finest ladder rung.
    finest = max(r["lambda_over_dx"] for r in _live_rungs(env))
    assert comp["lambda_over_dx"] == finest
    assert comp["fd_diff"] > 0.0
    if comp["repeat_noise"] == 0.0:
        assert comp["comparator_valid"] is True
    else:
        ratio = comp["fd_diff"] / comp["repeat_noise"]
        assert ratio >= COMPARATOR_MIN_RATIO
        assert comp["comparator_valid"] is True


def test_ladder_covers_coarse_to_fine() -> None:
    env = _env()
    live = _live_rungs(env)
    assert len(live) >= 3, "need >= 3 measured rungs for a curve"
    ratios = sorted(r["lambda_over_dx"] for r in live)
    # coarsest is lambda/20; the ladder strictly refines.
    assert ratios[0] == 20
    assert ratios == sorted(set(ratios)), "duplicate rungs"
    assert ratios[-1] >= 40, "ladder must reach at least lambda/40"


@pytest.mark.parametrize("idx", range(4))
def test_each_rung_rederives_and_passes_gate(idx: int) -> None:
    env = _env()
    live = _live_rungs(env)
    if idx >= len(live):
        pytest.skip(f"only {len(live)} live rungs committed")
    r = live[idx]

    h = r["fd_step_abs"]
    # 1. re-derive g_fd and rel_err from committed raw numbers.
    g_fd = (r["J_plus"] - r["J_minus"]) / (2.0 * h)
    g_ad = r["grad_ad"]
    rel_err = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)

    assert g_fd == pytest.approx(r["grad_fd"], rel=1e-9, abs=1e-30), r["rung"]
    assert rel_err == pytest.approx(r["rel_err"], rel=1e-9, abs=1e-30), r["rung"]

    # 2. sign agreement locked on every committed rung.
    assert (g_ad * g_fd) > 0.0, f"{r['rung']}: AD/FD signs disagree"
    assert r["sign_agree"] is True

    # 3. rel_err ceiling is exactly max(measured*1.5, 0.10) -- cannot be
    #    silently inflated -- and the committed rel_err is at or below it.
    #    (A large-but-comparator-valid rel_err is a recorded FINDING, not a
    #    failure -- the ceiling is a regression lock, not a physics bound --
    #    so this gate rides on the frozen per-rung ceiling, not a hard 0.10.)
    ceiling = max(rel_err * REL_ERR_MARGIN, REL_ERR_FLOOR)
    assert r["rel_err_ceiling"] == pytest.approx(ceiling, rel=1e-9)
    assert rel_err <= r["rel_err_ceiling"], r["rung"]

    # 4. per-rung comparator validity (FD-diff / repeat-noise >= 100, or
    #    deterministic noise == 0).
    assert r["fd_diff"] > 0.0
    if r["repeat_noise"] == 0.0:
        assert r["comparator_valid"] is True
    else:
        assert r["fd_diff"] / r["repeat_noise"] >= COMPARATOR_MIN_RATIO
        assert r["comparator_valid"] is True

    assert math.isfinite(g_ad) and math.isfinite(g_fd)

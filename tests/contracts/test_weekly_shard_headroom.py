"""The weekly lane's largest shard must keep room under its budget.

On 2026-10-09 the weekly lane lost a shard to its 3 h limit with no test
failing, and nothing said how close the lane was. This check makes the lane's
recorded load against its budget a number that fails before a run does.

What it checks: pytest-split's own algorithm (duration_based_chunks, as
validation.yml runs it) splits the tests in `.test_durations_weekly` into the
workflow's shards; the LARGEST group's recorded seconds, scaled by the runner
spread below, plus ten minutes of install and import, must fit the fraction of
the job limit that the lane's budget step enforces (`--fraction` of
`timeout-minutes`). The weekly file contains only tests collected by the weekly
selection (`not gpu and not highmem and not docs_consistency`).

What it does NOT see:
- A test without a recorded duration (pytest-split prices it at the mean); a
  durations regeneration closes that, and scripts/ci/DURATIONS.md states how
  many collected tests have none. That was 2026-10-09's own cause.
- Stale prices: on run 40083662779 one shard's measured time over its priced
  time ranged 0.64-1.27 (part of that is the runner). The shard count was
  chosen on that run's measured times as well (validation.yml comment).
- Collection order inside a file: the file's ids are in sorted order, so the
  replayed group boundaries can differ from the run's by a few tests.

RUNNER_SPREAD is the largest ratio of one shard's summed test time between two
weekly runs over the same test ids (>= 95 % common), i.e. how much slower one
hosted runner was than another for identical work. Pairs measured so far
(runs on bk-squared/rfx, artifacts weekly-durations-<shard>):

  1.78  shard 5  39748623375 (909a9133f) vs 39878447063 (63ab0c9d1), 2065 ids, 78.1 / 138.7 min
  1.55  shard 4  39878447063 (63ab0c9d1) vs 40083662779 (f0c70c556), 4071 ids, 135.0 / 87.3 min
  1.52  shard 1  39748623375 vs 39878447063, 8238 ids, 152.9 / 100.7 min
  1.49  shard 3  39698601854 (f8c0361a2) vs 39801780906 (d246de39c), 2527 ids, 89.2 / 132.9 min
  1.49  shard 4  39801780906 vs 39878447063, 4246 ids, 93.1 / 138.6 min
  1.47  shard 6  39801780906 attempt 1 vs attempt 2, 719 ids, 154.9 / 105.7 min
  1.43  shard 6  40083662779 vs 39801780906 attempt 2, 718 ids, 146.3 / 102.7 min
  1.42  shard 1  39698601854 vs 39748623375, 8224 ids, 107.1 / 152.1 min
  1.42  shard 3  39801780906 vs 39878447063, 2546 ids, 134.6 / 95.0 min
  1.39  shard 3  39698601854 vs 39748623375, 2527 ids, 89.2 / 124.3 min

The 1.78 pair spans three merges (#1983, #1984, #1988). Over its 381 tests
longer than 1 s the per-test ratio is p10 1.21, median 1.39, p90 1.96: nearly
every test ran slower, so the runner was slower; how much the merges add is not
separated, and the pair is counted whole, the conservative side. A larger pair measured later
raises RUNNER_SPREAD (add its line here); it is never lowered to make room.
When this fails, add shards to `validation.yml` (`group:` and `--splits`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pytest_split.algorithms import Algorithms

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_SPREAD = 1.78
SETUP_S = 600.0


class _Item:
    __slots__ = ("nodeid",)

    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid


def _weekly_shards_limit_fraction() -> tuple[int, float, float]:
    text = (REPO_ROOT / ".github" / "workflows" / "validation.yml").read_text(encoding="utf-8")
    block = text[text.index("  slow-tests:"):]
    # The job ends where the next job key (two-space indent) begins.
    block = block[:re.search(r"\n  [\w-]+:[ \t]*(#.*)?\n", block).start()]
    splits = {int(n) for n in re.findall(r"--splits (\d+)", block)}
    limits = [int(n) for n in re.findall(r"^    timeout-minutes: (\d+)$", block, re.M)]
    budget = re.findall(r"--limit-minutes (\d+) --fraction (0\.\d+)", block)
    groups = re.search(r"group: \[([\d, ]+)\]", block)
    algorithm = re.findall(r"--splitting-algorithm[ =](\S+)", block)
    assert len(splits) == 1 and len(limits) == 1 and len(budget) == 1 and groups, (splits, limits, budget)
    assert algorithm in ([], ["duration_based_chunks"]), algorithm
    (n,) = splits
    assert [int(g) for g in groups.group(1).split(",")] == list(range(1, n + 1))
    assert int(budget[0][0]) == limits[0], (budget, limits)
    return n, limits[0] * 60.0, float(budget[0][1])


def largest_group_seconds(durations: dict[str, float], shards: int) -> float:
    items = [_Item(nodeid) for nodeid in durations]
    groups = Algorithms.duration_based_chunks.value(shards, items, durations)
    return max(sum(durations[item.nodeid] for item in group.selected) for group in groups)


def projected_shard_seconds(largest_recorded_s: float) -> float:
    return largest_recorded_s * RUNNER_SPREAD + SETUP_S


def test_the_largest_weekly_shard_fits_its_budget():
    shards, limit_s, fraction = _weekly_shards_limit_fraction()
    durations = json.loads((REPO_ROOT / ".test_durations_weekly").read_text(encoding="utf-8"))
    largest = largest_group_seconds(durations, shards)
    projected = projected_shard_seconds(largest)
    assert projected <= fraction * limit_s, (
        f"weekly lane: the largest of {shards} shards holds {largest / 60:.0f} min of recorded tests; "
        f"x{RUNNER_SPREAD} + setup projects {projected / 60:.0f} min against the "
        f"{fraction:.0%} budget of {limit_s / 60:.0f} min — add shards"
    )


def test_the_projection_uses_the_largest_group_not_the_mean():
    # Contiguous chunks: one group can hold far more than the mean.
    durations = {f"t{i}": s for i, s in enumerate([100.0, 100.0, 100.0, 7000.0, 100.0, 100.0])}
    assert largest_group_seconds(durations, 2) >= 7000.0
    # 153 min budget: 80 min recorded x 1.78 + 10 min is 152.4 min; 81 min is 154.2 min.
    assert projected_shard_seconds(80 * 60.0) <= 0.85 * 10_800.0
    assert projected_shard_seconds(81 * 60.0) > 0.85 * 10_800.0

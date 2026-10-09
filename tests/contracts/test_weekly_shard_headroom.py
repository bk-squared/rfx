"""The weekly lane's shards must keep room under their job limit.

On 2026-10-09 the weekly lane lost a shard to its 3 h limit with no test
failing, and nothing said how close the lane was. This check makes the lane's
recorded load against its limit a number that fails before a run does.

What it does NOT see is that day's own cause: a new 51-minute test file with no
recorded duration adds nothing to the total, so this check was green then. A
test without an entry is priced at the mean by pytest-split; only a durations
regeneration closes that, and scripts/ci/DURATIONS.md states how many collected
tests have none.

It does not predict a shard either. It bounds the MEAN: the recorded seconds of every
test, divided by the number of weekly shards, scaled by the runner spread the
fast lane was sized with (x1.22, `.github/workflows/pr-tests.yml`, fast-suite
timeout comment) plus ten minutes of install and import, must fit the weekly
job limit. The weekly selection runs every test except the gpu and highmem
ones, so the file's total is its recorded load to within those few entries.
When this fails, add a shard to `validation.yml` (and to `regen-durations.yml`);
do not raise the factor.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_SPREAD = 1.22
SETUP_S = 600.0


def _weekly_shards_and_limit_s() -> tuple[int, float]:
    text = (REPO_ROOT / ".github" / "workflows" / "validation.yml").read_text(encoding="utf-8")
    block = text[text.index("  slow-tests:"):]
    block = block[:block.index("\n  weekly-a6000-lane:")]
    splits = {int(n) for n in re.findall(r"--splits (\d+)", block)}
    limits = [int(n) for n in re.findall(r"^    timeout-minutes: (\d+)$", block, re.M)]
    groups = re.search(r"group: \[([\d, ]+)\]", block)
    assert len(splits) == 1 and len(limits) == 1 and groups, (splits, limits)
    (n,) = splits
    assert [int(g) for g in groups.group(1).split(",")] == list(range(1, n + 1))
    return n, limits[0] * 60.0


def projected_shard_seconds(total_recorded_s: float, shards: int) -> float:
    return total_recorded_s / shards * RUNNER_SPREAD + SETUP_S


def test_the_mean_weekly_shard_fits_its_job_limit():
    shards, limit_s = _weekly_shards_and_limit_s()
    total = sum(json.loads((REPO_ROOT / ".test_durations").read_text(encoding="utf-8")).values())
    projected = projected_shard_seconds(total, shards)
    assert projected <= limit_s, (
        f"weekly lane: {total:.0f} s recorded over {shards} shards projects to "
        f"{projected / 60:.0f} min per shard against a {limit_s / 60:.0f} min limit — add a shard"
    )


def test_the_projection_would_have_flagged_a_lane_at_its_limit():
    # Five shards at the 3 h limit hold 5 * (10800 - 600) / 1.22 = 41803 s of recorded tests.
    assert projected_shard_seconds(41_000.0, 5) <= 10_800.0
    assert projected_shard_seconds(42_500.0, 5) > 10_800.0
    assert projected_shard_seconds(42_500.0, 6) <= 10_800.0

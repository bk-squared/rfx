"""A file-wide census belongs to exactly one pytest-split shard."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from tests._whole_file_shards import keep_whole

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tests/contracts/test_finding_subjects.py"


@pytest.mark.parametrize("items,snapshot,kept,removed", [
    (["before", "a", "after"], ["a", "b", "c"],
     ["before", "a", "b", "c", "after"], []),
    (["before", "a", "between", "c", "after"], ["a", "b", "c"],
     ["before", "a", "b", "c", "between", "after"], []),
    (["before", "b", "between", "c", "after"], ["a", "b", "c"],
     ["before", "between", "after"], ["b", "c"]),
    (["before", "after"], ["a", "b", "c"], ["before", "after"], []),
    (["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"], []),
    ([], ["a", "b", "c"], [], []),
    (["before", "after"], [], ["before", "after"], []),
])
def test_keep_whole(items, snapshot, kept, removed):
    original_items, original_snapshot = list(items), list(snapshot)
    assert keep_whole(items, snapshot) == (kept, removed)
    assert items == original_items
    assert snapshot == original_snapshot


def _pytest(*args):
    env = os.environ.copy()
    env["PYTEST_ADDOPTS"] = ""
    # Child reports must not enter the parent shard's duration log.
    env.pop("RFX_DURATIONS_LOG", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
         "-o", "addopts=", "--color=no", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def _collected(result):
    return [line.strip() for line in result.stdout.splitlines()
            if line.startswith(MODULE + "::")]


def test_real_file_has_exactly_one_owner(tmp_path):
    durations = tmp_path / "empty-durations.json"
    durations.write_text("{}", encoding="utf-8")
    args = (MODULE, "--collect-only", "-q", "-m", "not gpu and not docs_consistency",
            "--durations-path", str(durations))
    baseline = _pytest(*args)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    expected = _collected(baseline)
    assert len(expected) > 1  # Equal durations force a cut inside this file.
    assert len(expected) == len(set(expected))

    groups = []
    for group in (1, 2):
        result = _pytest(*args, "--splits", "2", "--group", str(group))
        ids = _collected(result)
        assert result.returncode == (0 if ids else 5), result.stdout + result.stderr
        groups.append(ids)
    assert groups == [expected, []]


def test_partial_census_reports_selected_counts():
    result = _pytest(MODULE + "::test_scene_code_coverage", "-q", "-rs")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 skipped" in output
    reason = re.search(r"coverage census applies to the complete scene battery: "
                       r"1 of (\d+) tests selected", output)
    assert reason is not None, output
    assert int(reason.group(1)) > 1

"""Contract tests for the weekly validation lane's own failure signal (#717).

The scheduled ``scientific-validation`` workflow is this repo's only recurring
physics lane, and it ended not green for months with nobody told: the last green
scheduled run was 2026-06-29, and every scheduled run after it produced no
issue, no comment and no notification. The ``notify`` job added in the #717 work
is the watcher. It closes that hole only for as long as it actually watches the
whole lane, so this file pins the properties a later edit could break without
reddening anything:

1. ``notify.needs`` names every other job in the file. It is a hand-kept list;
   a job added to the workflow and not added there is invisible to the notifier,
   which is the same silent-gap class #717 exists to close.
2. No job gates itself on a copy of ``on.schedule.cron``. A duplicated cron
   literal stops matching the moment the schedule is edited, and the job then
   skips on every scheduled run forever with nothing saying so.
3. The workflow declares exactly one cron. Every job here runs on every
   trigger, so a second cron entry would put the slow suite and a GPU seat
   (``weekly-a6000-lane``) on that schedule too. Adding one is allowed; doing it
   without re-reading what it starts is not. (Until 2026-09-21 the reason was a
   120-minute Meep job, ``crossval-external``; it left with the last scheduled
   cross-validation case.)
4. The notifier counts ``skipped`` as not green. No job here is supposed to skip
   on either trigger, so a skip means the lane quietly stopped covering
   something.
5. A denied ``issues: write`` is annotated (``core.error``), not only warned.
   That branch degrades the notifier back to the pre-#717 status quo — a job
   summary nobody reads — so it has to be visible on the run page.
6. ``validation.yml``'s ``slow-tests`` matrix keeps ``fail-fast: false``. GitHub's
   default is ``true``, under which one failing shard cancels the others and the
   lane loses their results.
7. ``regen-durations.yml``'s ``slow`` job installs and selects what ``slow-tests``
   does, so the prices it measures are for the tests those shards actually run.
8. Six weekly shards store and upload their durations, record their start after
   checkout, and finish with an always-run check against 85 % of the job limit.

The tests parse the workflow instead of grepping it, so reformatting the file
cannot fool them. They run on every scheduled lane that carries this notifier:
``validation.yml`` and, since the full-ladder cases moved out of it (2026-09-27),
the monthly ``crossval-ladder.yml``.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
NOTIFY_JOB = "notify"
#: Each scheduled lane with the #717 notifier, and the one cron it declares.
LANES = {"validation.yml": "0 6 * * 1", "crossval-ladder.yml": "0 12 1 * *"}
lanes = pytest.mark.parametrize("name", sorted(LANES))


def _workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load((WORKFLOWS_DIR / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` block.

    YAML 1.1 reads the bare key ``on`` as the boolean ``True`` and PyYAML
    follows that, so accept either spelling rather than depending on the loader.
    """
    for key in ("on", True):
        if key in workflow:
            return workflow[key]
    raise AssertionError("the workflow declares no triggers")


def _crons(workflow: dict[str, Any]) -> list[str]:
    schedule = _triggers(workflow).get("schedule") or []
    return [entry["cron"] for entry in schedule]


def _notify_steps(workflow: dict[str, Any]) -> dict[str, str]:
    jobs = workflow["jobs"]
    assert NOTIFY_JOB in jobs, f"the workflow lost its {NOTIFY_JOB} job"
    steps = jobs[NOTIFY_JOB]["steps"]
    summary = next(step for step in steps if step.get("id") == "verdicts")
    issue = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/github-script")
    )
    return {"summary": summary["run"], "issue": issue["with"]["script"]}


@lanes
def test_notify_watches_every_other_job_in_the_lane(name: str) -> None:
    jobs = _workflow(name)["jobs"]
    watched = set(jobs[NOTIFY_JOB]["needs"])
    expected = set(jobs) - {NOTIFY_JOB}
    assert watched == expected, (
        f"notify.needs must list every other job in {name}; missing "
        f"{sorted(expected - watched)}, stale {sorted(watched - expected)}. "
        "A job the notifier does not depend on cannot appear in its verdict, "
        "so its failures go unreported (issue #717)."
    )


@lanes
def test_no_job_gates_itself_on_a_copy_of_the_cron(name: str) -> None:
    workflow = _workflow(name)
    crons = _crons(workflow)
    assert crons, f"{name} must keep a cron schedule"
    for name, job in workflow["jobs"].items():
        condition = str(job.get("if", ""))
        for cron in crons:
            assert cron not in condition, (
                f"job {name!r} gates itself on the cron literal {cron!r}. "
                "Editing on.schedule.cron would then make it skip on every "
                "scheduled run with no signal; gate on github.event_name "
                "instead (issue #717 review)."
            )
        assert "github.event.schedule" not in condition, (
            f"job {name!r} compares github.event.schedule. Use "
            "github.event_name == 'schedule' so the gate survives a cron edit "
            "(issue #717 review)."
        )


@lanes
def test_validation_declares_exactly_one_cron(name: str) -> None:
    crons = _crons(_workflow(name))
    assert crons == [LANES[name]], (
        f"{name} now declares {crons}. Every cron here starts the whole lane, "
        "and each of these lanes holds a GPU seat. Re-read what a new schedule "
        "starts before adding it, then update LANES."
    )


@lanes
def test_notify_counts_a_skipped_job_as_not_green(name: str) -> None:
    scripts = _notify_steps(_workflow(name))
    assert 'select(.value.result != "success")' in scripts["summary"], (
        "the notify summary's jq filter must treat anything that is not "
        "'success' as not green, skipped included (issue #717 review)."
    )
    assert "skipped" not in scripts["summary"], (
        "the notify summary must not exempt skipped jobs: no job in this "
        "workflow is supposed to skip on either trigger, so a skip means the "
        "lane silently stopped covering something."
    )
    assert "job.conclusion !== 'success'" in scripts["issue"], (
        "the per-job table in the tracking issue must list every non-success "
        "conclusion, skipped included (issue #717 review)."
    )
    assert "skipped" not in scripts["issue"], (
        "the tracking-issue script must not exempt skipped jobs."
    )


@lanes
def test_a_denied_issue_write_is_annotated_not_only_warned(name: str) -> None:
    issue_script = _notify_steps(_workflow(name))["issue"]
    assert "core.error(`could not file the tracking issue" in issue_script, (
        "a failure to file the tracking issue puts the lane back to the "
        "pre-#717 status quo (a job summary nobody reads), so it must annotate "
        "the run with core.error rather than only core.warning (issue #717 "
        "review)."
    )


def test_one_failing_shard_does_not_cancel_the_others() -> None:
    strategy = _workflow("validation.yml")["jobs"]["slow-tests"]["strategy"]
    assert strategy.get("fail-fast") is False, (
        "slow-tests must set fail-fast: false; the default (true) cancels every "
        "other shard when one fails, and their results are lost."
    )


def _slow_shards(job: dict[str, Any]) -> dict[str, Any]:
    """What a slow job installs, which tests it selects, and how it splits them."""
    runs = [step["run"] for step in job["steps"] if "run" in step]
    installs = [run for run in runs if run.startswith("pip install")]
    splits = [run for run in runs if "--splits" in run]
    assert len(installs) == 1 and len(splits) == 1, f"expected one install and one split pytest run: {runs}"
    run = splits[0]
    marker = re.search(r"""-m (["'])(.*?)\1""", run)
    n = re.search(r"--splits (\d+)", run)
    assert marker and n, f"no -m or --splits N in: {run}"
    return {"install": installs[0], "-m": marker[2], "ignores": sorted(re.findall(r"--ignore=(\S+)", run)),
            "splits": int(n[1]), "groups": job["strategy"]["matrix"]["group"],
            "-k": bool(re.search(r"(?:^|\s)-k(?:\s|=|$)", run))}


def test_the_durations_regeneration_prices_what_the_slow_shards_run() -> None:
    weekly = _slow_shards(_workflow("validation.yml")["jobs"]["slow-tests"])
    regen = _slow_shards(_workflow("regen-durations.yml")["jobs"]["slow"])
    assert not regen["-k"], "regen-durations slow narrows its selection with -k"
    for name, lane in (("validation slow-tests", weekly), ("regen-durations slow", regen)):
        assert lane["splits"] == len(lane["groups"]), f"{name}: --splits {lane['splits']} but groups {lane['groups']}"
    assert regen == weekly, f"regen-durations slow {regen} != validation slow-tests {weekly}"


def _weekly_test_step(job: dict[str, Any]) -> dict[str, Any]:
    steps = [step for step in job["steps"] if "--splits" in step.get("run", "")]
    assert len(steps) == 1
    return steps[0]


def test_weekly_uses_six_shards() -> None:
    job = _workflow("validation.yml")["jobs"]["slow-tests"]
    assert job["strategy"]["matrix"]["group"] == [1, 2, 3, 4, 5, 6]
    assert _slow_shards(job)["splits"] == 6
    step = _weekly_test_step(job)
    assert step["name"] == "Run slow suite — shard ${{ matrix.group }} of 6"
    assert "--group ${{ matrix.group }}" in step["run"]
    three_device = next(s for s in job["steps"] if s.get("name") == "Run three-device x-wall alignment cases")
    assert three_device["if"] == "matrix.group == 1"
    assert job["timeout-minutes"] == 180
    assert _workflow("regen-durations.yml")["jobs"]["slow"]["timeout-minutes"] == 360


def test_weekly_stores_and_always_uploads_durations() -> None:
    job = _workflow("validation.yml")["jobs"]["slow-tests"]
    step = _weekly_test_step(job)
    assert "--store-durations" in shlex.split(step["run"])
    assert "--durations-path durations/weekly_${{ matrix.group }}.json" in step["run"]
    # pytest-split reads prices from the path it stores to: an unseeded path
    # would price every test the same and undo the shard balance.
    assert "|| true" not in step["run"], "a failing test must turn the weekly shard red"
    seed = "cp .test_durations durations/weekly_${{ matrix.group }}.json &&"
    assert seed in step["run"]
    assert step["run"].index(seed) < step["run"].index("pytest ")
    assert step["env"]["RFX_WEEKLY_RSS"] == "1"
    assert step["env"]["RFX_S0_FULL"] == "1"
    uploads = [s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert len(uploads) == 1
    upload = uploads[0]
    assert upload["if"] == "always()"
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"] == {
        "name": "weekly-durations-${{ matrix.group }}",
        "path": "durations/weekly_${{ matrix.group }}.json",
        "if-no-files-found": "warn",
    }
    budget = next(s for s in job["steps"] if s.get("name") == "Shard time against the job limit")
    assert job["steps"][-1] is upload
    assert job["steps"].index(upload) == job["steps"].index(budget) + 1
    regen_steps = _workflow("regen-durations.yml")["jobs"]["slow"]["steps"]
    assert any(s.get("uses") == upload["uses"] for s in regen_steps)


def test_weekly_records_start_immediately_after_checkout() -> None:
    steps = _workflow("validation.yml")["jobs"]["slow-tests"]["steps"]
    assert steps[0]["uses"] == "actions/checkout@v4"
    assert steps[1]["run"] == 'echo "SHARD_T0=$(date +%s)" >> "$GITHUB_ENV"'
    assert "if" not in steps[1]


def test_weekly_budget_is_always_after_the_tests_and_matches_job_limit() -> None:
    job = _workflow("validation.yml")["jobs"]["slow-tests"]
    checks = [s for s in job["steps"] if s.get("name") == "Shard time against the job limit"]
    assert len(checks) == 1
    step = checks[0]
    assert job["steps"].index(step) == len(job["steps"]) - 2
    assert step["if"] == "always()"
    assert "continue-on-error" not in step and "continue-on-error" not in job
    # Resolve only the matrix expression so shell argument parsing sees one path.
    args = shlex.split(step["run"].replace("${{ matrix.group }}", "1"))
    assert args == [
        "python", "scripts/ci/weekly_shard_budget.py", "--start", "$SHARD_T0",
        "--limit-minutes", "180", "--fraction", "0.85",
        "--measured", "durations/weekly_1.json", "--recorded", ".test_durations",
        "--prune",
    ]
    assert int(args[args.index("--limit-minutes") + 1]) == job["timeout-minutes"]


def _run_weekly_budget(
    tmp_path: Path, elapsed: int, *, measured: bool = True, prune: bool = False
) -> subprocess.CompletedProcess[str]:
    recorded_path = tmp_path / "recorded.json"
    measured_path = tmp_path / "measured.json"
    recorded_path.write_text(json.dumps({"known": 10, "not-run": 20}), encoding="utf-8")
    if measured:
        # As the weekly step leaves it: the recorded file, with what ran overwritten.
        measured_path.write_text(json.dumps({"known": 12, "not-run": 20, "new": 7}), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/ci/weekly_shard_budget.py"),
         "--start", str(time.time() - elapsed), "--limit-minutes", "180",
         "--fraction", "0.85", "--measured", str(measured_path),
         "--recorded", str(recorded_path), *(["--prune"] if prune else [])],
        capture_output=True, text=True, timeout=10, check=False,
    )


def test_weekly_budget_under_limit(tmp_path: Path) -> None:
    result = _run_weekly_budget(tmp_path, 7000)
    assert result.returncode == 0, result.stdout + result.stderr
    # The clock keeps running while the subprocess starts: compare with a tolerance.
    m = re.search(r"shard wall time: ([\d.]+) min of a 180 min limit \(([\d.]+) %\)", result.stdout)
    assert m, result.stdout
    assert abs(float(m.group(1)) - 7000 / 60) < 0.5 and abs(float(m.group(2)) - 7000 / 108) < 0.5
    assert "FAIL:" not in result.stdout


def test_weekly_budget_over_85_percent(tmp_path: Path) -> None:
    result = _run_weekly_budget(tmp_path, 9300)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        "FAIL: shard used more than 85 % of its job limit — add a weekly shard or "
        "regenerate .test_durations (scripts/ci/DURATIONS.md)"
    ) in result.stdout.splitlines()


def test_weekly_budget_missing_measured_file(tmp_path: Path) -> None:
    result = _run_weekly_budget(tmp_path, 7000, measured=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "measured durations file missing" in result.stdout.splitlines()
    assert "tests this shard ran with no recorded duration: unknown" in result.stdout.splitlines()


def test_weekly_budget_reports_unpriced_count_without_failing(tmp_path: Path) -> None:
    result = _run_weekly_budget(tmp_path, 7000)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests this shard ran with no recorded duration: 1" in result.stdout.splitlines()


def test_weekly_budget_prunes_the_upload_to_what_the_shard_measured(tmp_path: Path) -> None:
    result = _run_weekly_budget(tmp_path, 7000, prune=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests this shard measured: 2" in result.stdout.splitlines()
    assert json.loads((tmp_path / "measured.json").read_text(encoding="utf-8")) == {"known": 12, "new": 7}
    untouched = _run_weekly_budget(tmp_path, 7000)
    assert untouched.returncode == 0
    assert json.loads((tmp_path / "measured.json").read_text(encoding="utf-8")) == {"known": 12, "not-run": 20, "new": 7}


def test_weekly_budget_prunes_also_when_the_shard_is_over_budget(tmp_path: Path) -> None:
    # An over-budget shard still uploads; an unpruned file would carry the old
    # prices of every other shard's tests into the next merge.
    result = _run_weekly_budget(tmp_path, 9300, prune=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads((tmp_path / "measured.json").read_text(encoding="utf-8")) == {"known": 12, "new": 7}


def test_weekly_budget_survives_a_truncated_measured_file(tmp_path: Path) -> None:
    (tmp_path / "recorded.json").write_text("{}", encoding="utf-8")
    (tmp_path / "measured.json").write_text('{"known": 1', encoding="utf-8")
    args = [sys.executable, str(REPO_ROOT / "scripts/ci/weekly_shard_budget.py"),
            "--limit-minutes", "180", "--fraction", "0.85", "--prune",
            "--measured", str(tmp_path / "measured.json"), "--recorded", str(tmp_path / "recorded.json")]
    under = subprocess.run([*args, "--start", str(time.time() - 7000)], capture_output=True, text=True, timeout=10, check=False)
    over = subprocess.run([*args, "--start", str(time.time() - 9300)], capture_output=True, text=True, timeout=10, check=False)
    assert (under.returncode, over.returncode) == (0, 1), under.stdout + under.stderr + over.stdout + over.stderr
    assert "measured durations file unreadable" in under.stdout.splitlines()


TWO_CARD_JOB = "weekly-a6000x2-multigpu"
TWO_CARD_FILE = REPO_ROOT / "scripts" / "vessl_validation_multigpu_a6000x2.yaml"


def test_two_card_job_follows_the_single_card_lane_even_when_it_fails() -> None:
    # One GPU job at a time, and a red single-card lane must not hide the two-card result.
    job = _workflow("validation.yml")["jobs"][TWO_CARD_JOB]
    assert job["needs"] == ["weekly-a6000-lane"]
    assert job["if"].replace(" ", "") == "${{!cancelled()}}"
    run = next(s["run"] for s in job["steps"] if "vessl run create" in s.get("run", ""))
    assert "scripts/vessl_validation_multigpu_a6000x2.yaml" in run
    assert 'RFX_SHA: \\"origin/main\\"' in run, "the step pins the SHA by rewriting this exact line"
    assert '[ "$ST" = completed ] || { echo "two-card job ended with status $ST"; exit 1; }' in run.splitlines()[-1]


def test_two_card_file_needs_two_cards_before_its_tests_run() -> None:
    spec = yaml.safe_load(TWO_CARD_FILE.read_text(encoding="utf-8"))
    assert spec["resources"]["preset"] == "gpu-a6000-2"
    assert spec["env"]["RFX_SHA"] == "origin/main"
    assert spec["mount"] == {"/results": "volume://remilab-fs/rfx-vessl-tests"}
    lines = [line.strip() for line in spec["run"].splitlines()]
    # Whole lines, so a commented-out guard, an appended `|| true` or a collect-only
    # pytest call does not pass. Without the guard a node with one visible card skips
    # every test and ends green.
    guard = lines.index(
        '"$PY" -c "import jax, sys; assert sys.version_info[:2] == (3, 11), sys.version; '
        "assert jax.__version__ == '0.10.2', jax.__version__; "
        "gpus = [d for d in jax.devices() if d.platform == 'gpu']; "
        'assert len(gpus) >= 2, jax.devices()"'
    )
    tests = lines.index(
        'timeout 3600 "$PY" -m pytest -v -ra -s -p no:cacheprovider -m multi_gpu tests > "$OUT/multi_gpu.log" 2>&1'
    )
    assert guard < tests
    assert lines[tests + 1] == "RC=$?"
    assert lines[-1] == '[ "$RC" -eq 0 ]'

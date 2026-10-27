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
   trigger, so a second cron entry would put the slow suite on that schedule
   too. Adding one is allowed; doing it without re-reading what it starts is
   not. (Until 2026-09-21 the reason was a 120-minute Meep job, ``crossval-external``; it left with the last scheduled
   cross-validation case.)
4. The notifier counts ``skipped`` as not green. No job here is supposed to skip
   on either trigger, so a skip means the lane quietly stopped covering
   something. The one exception is a dispatch with ``only_latest_jax`` set:
   ``slow-tests`` and ``three-device`` skip and so does the notifier, so no
   verdict is written for that run. Only that exact switch may skip a job.
5. A denied ``issues: write`` is annotated (``core.error``), not only warned.
   That branch degrades the notifier back to the pre-#717 status quo — a job
   summary nobody reads — so it has to be visible on the run page.
6. ``validation.yml``'s ``slow-tests`` matrix keeps ``fail-fast: false``. GitHub's
   default is ``true``, under which one failing shard cancels the others and the
   lane loses their results.
7. ``regen-durations.yml``'s ``slow`` job installs and selects what ``slow-tests``
   does, so the prices it measures are for the tests those shards actually run.
8. Eight weekly shards store and upload their durations, record their start after
   checkout, and finish with an always-run check against 85 % of the job limit.
9. The three-device cases run in their own job, outside the shards' balance, with
   the same kind of check against their own limit.

The notifier tests parse ``validation.yml`` so reformatting cannot fool them.
A separate contract scans every workflow for forbidden VESSL dependencies.
GPU and monthly full-ladder checks now run from the merge train's Mac runner
in agent-rules.
"""

from __future__ import annotations

import importlib.util
import json
import os
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
LANES = {"validation.yml": "0 6 * * 1"}
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
        "including every slow-tests shard. Re-read what a new schedule "
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
    for workflow, job in (("validation.yml", "slow-tests"), ("regen-durations.yml", "slow")):
        step = _weekly_test_step(_workflow(workflow)["jobs"][job])
        # The prices are measured with modules kept together; least_duration would split them.
        assert "--splitting-algorithm" not in step["run"]
    regen_step = _weekly_test_step(_workflow("regen-durations.yml")["jobs"]["slow"])
    assert "--clean-durations" in shlex.split(regen_step["run"])
    assert "--durations-path .test_durations_weekly" in regen_step["run"]
    assert 'cp .test_durations_weekly "$RUNNER_TEMP/weekly-seed"' in regen_step["run"]
    refuse = [s for s in _workflow("regen-durations.yml")["jobs"]["slow"]["steps"]
              if "cmp -s .test_durations_weekly" in s.get("run", "")]
    assert len(refuse) == 1 and "exit 1" in refuse[0]["run"]
    regen_upload = _workflow("regen-durations.yml")["jobs"]["slow"]["steps"][-1]
    assert regen_upload["with"]["path"] == ".test_durations_weekly"
    assert regen_upload["with"]["include-hidden-files"] is True
    assert regen == weekly, f"regen-durations slow {regen} != validation slow-tests {weekly}"


def _weekly_test_step(job: dict[str, Any]) -> dict[str, Any]:
    steps = [step for step in job["steps"] if "--splits" in step.get("run", "")]
    assert len(steps) == 1
    return steps[0]


def test_weekly_uses_sixteen_shards() -> None:
    # 2026-10-25: 16, was 8 (runner spread 1.78; tests/contracts/test_weekly_shard_headroom.py).
    job = _workflow("validation.yml")["jobs"]["slow-tests"]
    assert job["strategy"]["matrix"]["group"] == list(range(1, 17))
    assert _slow_shards(job)["splits"] == 16
    step = _weekly_test_step(job)
    assert step["name"] == "Run slow suite — shard ${{ matrix.group }} of 16"
    assert "--group ${{ matrix.group }}" in step["run"]
    assert job["timeout-minutes"] == 180
    assert _workflow("regen-durations.yml")["jobs"]["slow"]["timeout-minutes"] == 360


ONLY_LATEST_JAX_SKIP = "${{ !inputs.only_latest_jax }}"


def test_only_the_latest_jax_dispatch_switch_skips_a_job() -> None:
    """On a scheduled run ``inputs`` is empty, so ``!inputs.only_latest_jax`` is true and every job runs.

    A dispatch with the switch set runs latest-jax alone and skips the notifier with the slow shards,
    so that run writes no verdict and files no issue. Any other job-level condition is refused.
    """
    workflow = _workflow("validation.yml")
    switch = _triggers(workflow)["workflow_dispatch"]["inputs"]["only_latest_jax"]
    assert switch["type"] == "boolean" and switch["default"] is False
    conditions = {name: job.get("if") for name, job in workflow["jobs"].items()}
    assert conditions == {
        "slow-tests": ONLY_LATEST_JAX_SKIP,
        "three-device": ONLY_LATEST_JAX_SKIP,
        "latest-jax": None,
        "notify": "${{ always() && !inputs.only_latest_jax }}",
    }


def test_three_device_cases_run_in_their_own_job() -> None:
    # pytest-split balances only a shard's main step. On run 39748623375 shard 1 carried
    # these cases after its group and was cut at its limit with no test failed.
    jobs = _workflow("validation.yml")["jobs"]
    assert not [s for s in jobs["slow-tests"]["steps"] if "xla_force_host_platform_device_count" in str(s.get("env"))]
    job = jobs["three-device"]
    assert job.get("if") == ONLY_LATEST_JAX_SKIP
    assert not {"strategy", "needs", "continue-on-error"} & set(job)
    steps = job["steps"]
    assert steps[0]["uses"] == "actions/checkout@v4"
    assert steps[1]["run"] == 'echo "JOB_T0=$(date +%s)" >> "$GITHUB_ENV"'
    runs = [s for s in steps if "pytest" in s.get("run", "")]
    assert runs
    counts = {"three": 3, "four": 4, "five": 5}
    for step in runs:
        # A failing file must not skip the later steps.
        assert step["if"] == "${{ !cancelled() }}", step["name"]
        # Each step names its host-device count; the closed-box TF/SF cap needs 4 and 5.
        n = counts[step["name"].split()[1].removesuffix("-device")]
        assert step["env"] == {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": f"--xla_force_host_platform_device_count={n}"}
    assert sum("three-device" in s["name"] for s in runs) >= 4
    check = steps[-1]
    assert check["name"] == "Job time against the job limit" and check["if"] == "always()"
    limit = job["timeout-minutes"]
    assert f"$(( {limit} * 60 * 85 / 100 ))" in check["run"]
    assert f"of a {limit} min limit" in check["run"]


def test_weekly_stores_and_always_uploads_durations() -> None:
    job = _workflow("validation.yml")["jobs"]["slow-tests"]
    step = _weekly_test_step(job)
    assert "--store-durations" in shlex.split(step["run"])
    assert "--durations-path durations/weekly_${{ matrix.group }}.json" in step["run"]
    # pytest-split reads prices from the path it stores to: an unseeded path
    # would price every test the same and undo the shard balance.
    assert "|| true" not in step["run"], "a failing test must turn the weekly shard red"
    seed = "cp .test_durations_weekly durations/weekly_${{ matrix.group }}.json &&"
    assert seed in step["run"]
    assert step["run"].index(seed) < step["run"].index("pytest ")
    assert step["env"]["RFX_WEEKLY_RSS"] == "1"
    assert step["env"]["RFX_S0_FULL"] == "1"
    assert step["env"]["RFX_CLEAR_JAX_CACHES_PER_MODULE"] == "1"
    regen_slow = next(s for s in _workflow("regen-durations.yml")["jobs"]["slow"]["steps"]
                      if "--splits" in s.get("run", ""))
    assert regen_slow["env"]["RFX_CLEAR_JAX_CACHES_PER_MODULE"] == "1"
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
        "--measured", "durations/weekly_1.json", "--recorded", ".test_durations_weekly",
        "--log", "durations/weekly_1.log", "--prune",
    ]
    test_step = _weekly_test_step(job)
    # The log the budget reads is the one conftest writes during the test step.
    assert test_step["env"]["RFX_DURATIONS_LOG"] == "durations/weekly_${{ matrix.group }}.log"
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


def _crashed_shard(tmp_path: Path, log_lines: list[str] | None) -> subprocess.CompletedProcess[str]:
    # pytest aborted: the measured file is still the untouched copy of the recorded one.
    recorded = {"known": 10.0, "not-run": 20.0}
    (tmp_path / "recorded.json").write_text(json.dumps(recorded), encoding="utf-8")
    (tmp_path / "measured.json").write_text(json.dumps(recorded), encoding="utf-8")
    log = tmp_path / "weekly.log"
    if log_lines is not None:
        log.write_text("".join(line + "\n" for line in log_lines), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/ci/weekly_shard_budget.py"),
         "--start", str(time.time() - 7000), "--limit-minutes", "180", "--fraction", "0.85",
         "--measured", str(tmp_path / "measured.json"), "--recorded", str(tmp_path / "recorded.json"),
         "--log", str(log), "--prune"],
        capture_output=True, text=True, timeout=10, check=False,
    )


def test_weekly_budget_reads_the_per_test_log_when_pytest_stored_nothing(tmp_path: Path) -> None:
    # Phases summed per test as pytest-split does; a setup over 600 s is dropped as it
    # drops it; the line cut by the abort is skipped.
    lines = [json.dumps(["known", "setup", 1.0]), json.dumps(["known", "call", 11.5]),
             json.dumps(["known", "teardown", 0.5]), json.dumps(["new", "setup", 700.0]),
             json.dumps(["new", "call", 52.0]), '["cut", "ca']
    result = _crashed_shard(tmp_path, lines)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout.splitlines()
    assert "pytest stored no durations: the session did not reach its end" in out
    assert "tests this shard measured: 2" in out
    assert "tests this shard ran with no recorded duration: 1" in out
    assert json.loads((tmp_path / "measured.json").read_text(encoding="utf-8")) == {"known": 13.0, "new": 52.0}


def test_weekly_budget_says_unknown_when_pytest_stored_nothing_and_no_log(tmp_path: Path) -> None:
    result = _crashed_shard(tmp_path, None)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout.splitlines()
    assert "tests this shard measured: unknown (no per-test log either)" in out
    assert "tests this shard ran with no recorded duration: unknown" in out
    assert "tests this shard measured: 0" not in out


def test_durations_log_survives_a_test_that_changes_directory(tmp_path: Path) -> None:
    # A real pytest session with the plugin, the log path relative as validation.yml
    # sets it, and a test that changes directory before its call report: the log must
    # stay at the configure-time path and the session must run every test.
    session = tmp_path / "session"
    session.mkdir()
    (session / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (session / "test_chdir.py").write_text(
        "def test_moves(monkeypatch, tmp_path):\n    monkeypatch.chdir(tmp_path)\n\n"
        "def test_after():\n    pass\n", encoding="utf-8")
    env = {**os.environ, "RFX_DURATIONS_LOG": "durations/weekly.log",
           "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])}
    (session / "durations").mkdir()
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "tests._durations_log",
         "test_chdir.py"],
        cwd=session, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "2 passed" in run.stdout, run.stdout
    rows = [json.loads(line) for line in (session / "durations/weekly.log").read_text(encoding="utf-8").splitlines()]
    assert [(nodeid, when) for nodeid, when, _ in rows] == [
        (f"test_chdir.py::{name}", when) for name in ("test_moves", "test_after")
        for when in ("setup", "call", "teardown")]
    lines = (session / "durations/weekly.log").read_text(encoding="utf-8").splitlines()
    result = _crashed_shard(tmp_path, lines)
    assert result.returncode == 0, result.stdout + result.stderr
    assert set(json.loads((tmp_path / "measured.json").read_text(encoding="utf-8"))) == {
        "test_chdir.py::test_moves", "test_chdir.py::test_after"}


def test_budget_log_rule_matches_pytest_split() -> None:
    from pytest_split.plugin import STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD
    spec = importlib.util.spec_from_file_location("weekly_shard_budget", REPO_ROOT / "scripts/ci/weekly_shard_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SETUP_TEARDOWN_LIMIT_S == STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD


def test_no_workflow_uses_vessl() -> None:
    workflows = sorted(
        path for path in WORKFLOWS_DIR.rglob("*")
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )
    assert workflows, "no workflows found"
    forbidden = ("vessl run create", "VESSL_ACCESS_TOKEN", "vars.VESSL_", "environment: vessl")
    violations = [
        f"{path.relative_to(WORKFLOWS_DIR)}:{number}: {token}"
        for path in workflows
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for token in forbidden if token in line
    ]
    assert not violations, "VESSL workflow dependencies:\n" + "\n".join(violations)

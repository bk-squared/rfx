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

The tests parse the workflow instead of grepping it, so reformatting the file
cannot fool them. They run on every scheduled lane that carries this notifier:
``validation.yml`` and, since the full-ladder cases moved out of it (2026-09-27),
the monthly ``crossval-ladder.yml``.
"""

from __future__ import annotations

import re
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

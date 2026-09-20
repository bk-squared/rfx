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
3. The workflow declares exactly one cron. ``crossval-external`` is gated on the
   trigger (``github.event_name == 'schedule'``) rather than on a cron literal,
   so a second cron entry would put a 120-minute Meep job on that schedule too.
   Adding one is allowed; doing it without re-reading that gate is not.
4. The notifier counts ``skipped`` as not green. No job here is supposed to skip
   on either trigger, so a skip means the lane quietly stopped covering
   something.
5. A denied ``issues: write`` is annotated (``core.error``), not only warned.
   That branch degrades the notifier back to the pre-#717 status quo — a job
   summary nobody reads — so it has to be visible on the run page.

The tests parse the workflow instead of grepping it, so reformatting the file
cannot fool them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "validation.yml"
NOTIFY_JOB = "notify"


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` block.

    YAML 1.1 reads the bare key ``on`` as the boolean ``True`` and PyYAML
    follows that, so accept either spelling rather than depending on the loader.
    """
    for key in ("on", True):
        if key in workflow:
            return workflow[key]
    raise AssertionError(f"{WORKFLOW_PATH} declares no triggers")


def _crons(workflow: dict[str, Any]) -> list[str]:
    schedule = _triggers(workflow).get("schedule") or []
    return [entry["cron"] for entry in schedule]


def _notify_steps(workflow: dict[str, Any]) -> dict[str, str]:
    jobs = workflow["jobs"]
    assert NOTIFY_JOB in jobs, f"{WORKFLOW_PATH} lost its {NOTIFY_JOB} job"
    steps = jobs[NOTIFY_JOB]["steps"]
    summary = next(step for step in steps if step.get("id") == "verdicts")
    issue = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/github-script")
    )
    return {"summary": summary["run"], "issue": issue["with"]["script"]}


def test_notify_watches_every_other_job_in_the_lane() -> None:
    jobs = _workflow()["jobs"]
    watched = set(jobs[NOTIFY_JOB]["needs"])
    expected = set(jobs) - {NOTIFY_JOB}
    assert watched == expected, (
        "notify.needs must list every other job in validation.yml; missing "
        f"{sorted(expected - watched)}, stale {sorted(watched - expected)}. "
        "A job the notifier does not depend on cannot appear in its verdict, "
        "so its failures go unreported (issue #717)."
    )


def test_no_job_gates_itself_on_a_copy_of_the_cron() -> None:
    workflow = _workflow()
    crons = _crons(workflow)
    assert crons, "validation.yml must keep a cron schedule"
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


def test_validation_declares_exactly_one_cron() -> None:
    crons = _crons(_workflow())
    assert crons == ["0 6 * * 1"], (
        f"validation.yml now declares {crons}. crossval-external is gated on "
        "github.event_name == 'schedule', so every cron here runs a 120-minute "
        "Meep job and the weekly-a6000-lane holds a GPU seat. Re-read those two "
        "gates before adding a schedule, then update this test."
    )


def test_notify_counts_a_skipped_job_as_not_green() -> None:
    scripts = _notify_steps(_workflow())
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


def test_a_denied_issue_write_is_annotated_not_only_warned() -> None:
    issue_script = _notify_steps(_workflow())["issue"]
    assert "core.error(`could not file the tracking issue" in issue_script, (
        "a failure to file the tracking issue puts the lane back to the "
        "pre-#717 status quo (a job summary nobody reads), so it must annotate "
        "the run with core.error rather than only core.warning (issue #717 "
        "review)."
    )

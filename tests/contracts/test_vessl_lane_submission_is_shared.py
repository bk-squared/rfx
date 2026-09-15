"""Contract tests for the VESSL-submitting jobs in the weekly validation lane.

Two jobs here submit a VESSL lane on the weekly schedule — ``weekly-a6000-lane`` and
``crossval-solvers-lane`` — and the lab runs a two-seat GPU policy. The
submit/poll/terminate sequence used to live inline in the one job that had it; the
second job would have made that a copy, and the copy-sensitive part is the
``trap ... EXIT`` that releases the seat when the watcher goes away for any reason:
a give-up, a cancellation, or the 360-minute job ceiling. A watcher that dies without
terminating its run holds a seat until somebody notices by hand.

A second thing these tests pin is less obvious and was found by review rather than by
a failure: **almost nothing crosses from the pod to the workflow.** The lane writes its
per-node table and its machine-readable summary into the mounted NFS workspace, which a
GitHub runner cannot read, and the submitting step reads the run back with
``vessl run logs --tail 400 | tail -60``. So a lane whose verdict is not in that tail is
a lane whose result the workflow cannot act on. The shared script takes a grep pattern
for exactly that reason, and the solver lane is required to pass one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "validation.yml"
SHARED_SCRIPT = REPO_ROOT / "scripts" / "ci" / "vessl_submit_and_wait.sh"


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _run_steps(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            body = step.get("run")
            if isinstance(body, str):
                out.append((job_name, body))
    return out


def test_no_job_re_derives_vessl_run_create_inline() -> None:
    offenders = [
        job for job, body in _run_steps(_workflow())
        if "vessl run create" in body and "vessl_submit_and_wait.sh" not in body
    ]
    assert not offenders, (
        f"these jobs call `vessl run create` inline instead of through the shared "
        f"script: {sorted(set(offenders))}. It owns the EXIT trap that releases the "
        "seat; an inline copy is how that gets lost under the two-seat lab policy."
    )


def test_the_shared_script_releases_the_seat_and_checks_its_own_sha_pin() -> None:
    body = SHARED_SCRIPT.read_text(encoding="utf-8")
    assert "trap " in body and "EXIT" in body and "vessl run terminate" in body, (
        f"{SHARED_SCRIPT.name} no longer terminates the run it submitted on exit"
    )
    assert "grep -q" in body and "GITHUB_SHA" in body, (
        "the script must grep the pinned SHA back out of the rewritten lane file; "
        "without it a renamed env key makes the sed a no-op and the lane runs "
        "origin/main while the job reports the triggering commit"
    )


def _submitted_lanes(workflow: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(job, lane yaml, ref env key, verdict grep or '') per submitting step."""
    call = re.compile(
        r"vessl_submit_and_wait\.sh\s*\\?\s*\n?\s*(\S+\.yaml)\s*\\?\s*\n?\s*(\S+)"
    )
    found = []
    for job, body in _run_steps(workflow):
        if "vessl_submit_and_wait.sh" not in body:
            continue
        m = call.search(body)
        assert m, f"could not parse the submit call in job {job!r}:\n{body}"
        grep = ""
        g = re.search(r"'([^']*VERDICT[^']*)'", body)
        if g:
            grep = g.group(1)
        found.append((job, m.group(1), m.group(2), grep))
    return found


def test_every_submitted_lane_exists_and_declares_the_ref_key_the_job_passes() -> None:
    lanes = _submitted_lanes(_workflow())
    assert lanes, "no job in validation.yml submits a VESSL lane any more"
    for job, lane_path, ref_key, _ in lanes:
        lane_file = REPO_ROOT / lane_path
        assert lane_file.is_file(), f"job {job!r} submits missing lane file {lane_path}"
        env = yaml.safe_load(lane_file.read_text(encoding="utf-8")).get("env") or {}
        assert ref_key in env, (
            f"job {job!r} passes ref key {ref_key!r} but {lane_path} declares "
            f"{sorted(env)}. The sed in the shared script would silently no-op."
        )
        assert env[ref_key] == "origin/main", (
            f"{lane_path}'s {ref_key} is committed as {env[ref_key]!r}; the script "
            'rewrites the literal "origin/main", so any other committed value is '
            "never replaced and the lane ignores the triggering SHA."
        )


def test_the_solver_lane_surfaces_a_verdict_the_workflow_can_read() -> None:
    lanes = {job: grep for job, _, _, grep in _submitted_lanes(_workflow())}
    grep = lanes.get("crossval-solvers-lane")
    assert grep, (
        "crossval-solvers-lane passes no verdict grep to the shared script. Its "
        "per-node table and summary JSON land on the NFS mount, which a GitHub "
        "runner cannot read, so without this the job summary carries 60 lines of "
        "tail and nothing a reader can act on."
    )
    lane = yaml.safe_load(
        (REPO_ROOT / "scripts" / "vessl_crossval_solvers_lane.yaml").read_text(encoding="utf-8")
    )
    for token in re.findall(r"[A-Z_]{6,}", grep):
        assert token in lane["run"], (
            f"the job greps for {token!r} but the lane never prints it"
        )


def test_the_new_job_gates_on_the_trigger_and_never_skips() -> None:
    """A skipped job counts as not-green (see the notify comment at :386-393)."""
    job = _workflow()["jobs"]["crossval-solvers-lane"]
    condition = job.get("if", "")
    assert "github.event_name" in condition, (
        "crossval-solvers-lane must gate on the trigger so that adding a push or "
        "pull_request trigger to this workflow does not put a multi-hour solver job "
        "on every push"
    )
    for trigger in ("schedule", "workflow_dispatch"):
        assert trigger in condition, (
            f"the condition excludes {trigger!r}; this workflow has exactly those two "
            "triggers, so excluding one makes the job skip, and notify counts a skip "
            "as not green"
        )

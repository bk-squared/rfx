"""Contract tests for the two-solver crossval lane (#717 item 2).

The lane exists because the four gpu-marked crossval files carry 13 external legs
that have executed in no lane, ever. Every lane that has ever collected them lacked
a solver, so all 13 ``pytest.importorskip``-ed away and the run exited 0. #717 was
filed about exactly that: a coverage claim that is not honest.

Which makes the closing condition delicate. "A green scheduled run that collects the
four files" is satisfied by 18 skips, and a count-based gate cannot tell the
difference. So the gate is a frozen node-id set plus ``skipped == 0`` plus a duration
floor on each external leg, and these tests pin the three ways that gate could rot:

1. The frozen node list drifts away from what the four files actually collect — the
   gate would then demand tests that no longer exist, or silently stop covering ones
   that do.
2. The lane file loses ``-m gpu``, the solver probe, or the asserter call. Without
   ``-m gpu`` the pyproject addopts deselect every one of the 18 and the lane is green
   and empty; without the probe a solver-less pod is green with 13 skips.
3. The asserter stops failing on a skip. That is the whole point of it, and a planted
   skip is the cheapest way to keep proving it.

There is also a hard environment rule this lane must not break:
``_configs/.claude/rules/vessl-jobs.md`` forbids heredocs in a VESSL ``run:`` block —
VESSL re-indents the block inside its own wrapper, a heredoc terminator leaves column
0, and the job dies at parse time before any work runs. ``sh -n`` does not catch it,
so it is pinned here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE = REPO_ROOT / "scripts" / "vessl_crossval_solvers_lane.yaml"
FROZEN = REPO_ROOT / "scripts" / "ops" / "crossval_solver_lane_nodes.json"
ASSERTER = REPO_ROOT / "scripts" / "ops" / "assert_crossval_solver_lane.py"
PROBE = REPO_ROOT / "scripts" / "ops" / "probe_crossval_solvers.py"

FOUR_FILES = (
    "tests/crossval/test_crossval_comprehensive.py",
    "tests/crossval/test_meep_crossval.py",
    "tests/crossval/test_meep_crossval_dielectric_cavity.py",
    "tests/crossval/test_openems_crossval.py",
)


def _frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def _lane_run() -> str:
    return yaml.safe_load(LANE.read_text(encoding="utf-8"))["run"]


def test_the_frozen_node_list_matches_what_the_four_files_collect() -> None:
    frozen = _frozen()
    expected = set(frozen["external"]) | set(frozen["rfx_only"])

    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "--collect-only", "-q",
            "-o", "addopts=", "-p", "no:cacheprovider", *FOUR_FILES,
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    live = {line.strip() for line in proc.stdout.splitlines() if "::" in line}
    assert live, f"collection produced no node ids:\n{proc.stdout}\n{proc.stderr}"
    assert live == expected, (
        "scripts/ops/crossval_solver_lane_nodes.json has drifted from what the four "
        f"files collect.\n  only in the frozen list: {sorted(expected - live)}\n"
        f"  only in collection: {sorted(live - expected)}\n"
        "Regenerate it deliberately — widening the gate to match a drifted "
        "collection is how the lane stops covering what it claims to."
    )


def test_every_external_leg_is_classified_as_external() -> None:
    """The duration floor only applies to legs marked external, so the split matters."""
    frozen = _frozen()
    for node in frozen["external"]:
        assert "meep" in node or "openems" in node, (
            f"{node} is listed as an external leg but names neither solver"
        )
    for node in frozen["rfx_only"]:
        assert "rfx_vs_meep" not in node and "rfx_vs_openems" not in node, (
            f"{node} is listed as rfx-only but its name says it compares against a "
            "solver; misclassifying it exempts it from the duration floor"
        )
    assert len(frozen["external"]) == 13
    assert len(frozen["rfx_only"]) == 5


def test_the_lane_has_no_heredoc() -> None:
    run = _lane_run()
    assert "<<" not in run, (
        "scripts/vessl_crossval_solvers_lane.yaml uses a heredoc. VESSL re-indents "
        "the run: block inside its own wrapper, which moves the terminator off "
        "column 0; the job then dies at parse time before any work runs "
        "(_configs/.claude/rules/vessl-jobs.md, amc run 369367250654). `sh -n` on "
        "the extracted block does not catch this. Put the inline python in a repo "
        "script under scripts/ops/ and call it."
    )


def test_the_lane_keeps_the_gpu_marker_the_probe_and_the_asserter() -> None:
    run = _lane_run()
    for name in FOUR_FILES:
        assert name in run, f"the lane no longer names {name}"
    assert "-m gpu" in run, (
        "the lane must pass `-m gpu`. The pyproject addopts are "
        "-m 'not gpu and not slow and not slow_physics' and a node id does not "
        "override a marker filter, so without it the four files collect zero tests "
        "and the lane is green and empty."
    )
    assert "probe_crossval_solvers.py" in run, (
        "the lane lost its solver probe; a solver-less pod would report 13 clean "
        "importorskip SKIPs and exit 0"
    )
    assert "assert_crossval_solver_lane.py" in run, (
        "the lane lost its closing gate; the pytest exit code alone is satisfied by "
        "18 skips"
    )
    assert "commit.txt" in run, (
        "the lane must record the commit it actually exported "
        "(_configs/.claude/rules/vessl-jobs.md: provenance capture fails loudly)"
    )
    assert "--timeout" in run and "--timeout-method=thread" in run, (
        "every pytest lane here caps itself; a hang that prints nothing looks alive "
        "to VESSL and says nothing about which test stopped"
    )


def _junit(cases: list[tuple[str, str, str, float]]) -> str:
    body = "".join(
        f'<testcase classname="{cls}" name="{name}" time="{t}">'
        + ("" if outcome == "passed" else f"<{outcome} />")
        + "</testcase>"
        for cls, name, outcome, t in cases
    )
    return f'<?xml version="1.0"?><testsuites><testsuite name="pytest">{body}</testsuite></testsuites>'


def _all_cases(outcome_for_external: str, external_time: float) -> list[tuple[str, str, str, float]]:
    frozen = _frozen()
    cases = []
    for group, outcome, t in (
        ("external", outcome_for_external, external_time),
        ("rfx_only", "passed", 5.0),
    ):
        for node in frozen[group]:
            path, *rest = node.split("::")
            cls = path[: -len(".py")].replace("/", ".")
            if len(rest) == 2:
                cls = f"{cls}.{rest[0]}"
            cases.append((cls, rest[-1], outcome, t))
    return cases


def _run_asserter(tmp_path: Path, cases: list[tuple[str, str, str, float]]) -> subprocess.CompletedProcess:
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit(cases), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable, str(ASSERTER), str(junit),
            "--summary-json", str(tmp_path / "summary.json"), "--sha", "deadbeef",
        ],
        capture_output=True, text=True, timeout=120,
    )


def test_the_asserter_passes_a_run_where_every_leg_actually_ran(tmp_path: Path) -> None:
    proc = _run_asserter(tmp_path, _all_cases("passed", 42.0))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["collected"] == 18


@pytest.mark.parametrize(
    ("outcome", "time_s", "why"),
    [
        ("skipped", 0.0, "a skip means the leg did not run"),
        ("passed", 0.01, "a pass this fast is evidence no solver ran"),
        ("failure", 3.0, "a red physics gate is not a green lane"),
    ],
)
def test_the_asserter_rejects_the_ways_a_lane_can_look_green(
    tmp_path: Path, outcome: str, time_s: float, why: str
) -> None:
    proc = _run_asserter(tmp_path, _all_cases(outcome, time_s))
    assert proc.returncode != 0, f"the asserter accepted a run where {why}:\n{proc.stdout}"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is False
    assert summary["problems"], "the asserter failed without saying why"


def test_the_probe_exits_nonzero_when_a_solver_is_missing() -> None:
    """Runs on any machine without meep/openEMS — which is every dev box and CI runner."""
    pytest.importorskip("rfx")
    try:
        import meep  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("this box has meep; the probe's failure path cannot be exercised here")
    proc = subprocess.run(
        [sys.executable, str(PROBE)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0, (
        "the probe exited 0 on a box with no meep. Its whole job is to make a "
        "solver-less pod red instead of green-with-13-skips."
    )
    assert "FATAL" in proc.stderr


def test_the_verdict_crosses_the_log_tail(tmp_path: Path) -> None:
    """The one line the submitting GitHub job can actually see.

    The workflow that submits this lane reads it back with
    ``vessl run logs --tail 400 | tail -60``, and the machine-readable summary lands
    on the NFS mount, which a GitHub runner cannot read at all. So the asserter
    prints one compact JSON line LAST, on both the pass and the fail path, and the
    lane must not print anything verbose after it.
    """
    for outcome, time_s, expect_ok in (("passed", 42.0, True), ("skipped", 0.0, False)):
        proc = _run_asserter(tmp_path, _all_cases(outcome, time_s))
        tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        assert tail, "the asserter printed nothing"
        assert tail[-1].startswith("CROSSVAL_SOLVER_LANE_VERDICT "), (
            f"the verdict line is not last on the {outcome} path; last line was {tail[-1]!r}"
        )
        verdict = json.loads(tail[-1].split(" ", 1)[1])
        assert verdict["ok"] is expect_ok
        assert verdict["collected"] == 18
        assert verdict["skipped"] == (0 if expect_ok else 13)

    run = _lane_run()
    marker = "CROSSVAL_SOLVER_LANE_VERDICT"
    assert marker in run, (
        "the lane no longer mentions the verdict marker, so nothing pins that it "
        "stays inside the log tail the submitting job reads"
    )
    # The reds grep must happen BEFORE the gate, or up to 18 FAILED lines push the
    # verdict out of the tail exactly when the lane is red and it matters most.
    assert run.index("reds.txt") < run.index("assert_crossval_solver_lane.py"), (
        "the reds collection moved after the gate call; on a red run its output can "
        "push the verdict line out of `vessl run logs --tail 400 | tail -60`"
    )


def test_the_lane_captures_pytest_s_exit_code_and_not_tee_s() -> None:
    """`$?` after `cmd | tee file` is TEE's status, and tee always succeeds.

    The lane must tee (a plain redirect hides a hang behind an NFS file), and it
    cannot use ``set -o pipefail`` or ``${PIPESTATUS[0]}`` — both are bashisms this
    runner shell does not have, and the second has already killed a lane here
    (gpu_suite run 369367242339, "bad substitution"). So the rc is written from
    inside the pipeline's left side and read back from the file.

    Getting this wrong is silent and it inverts the lane's whole purpose: a red
    pytest would report rc=0 and the lane would be green.
    """
    run = _lane_run()

    # Anchor on the GATED calls. `-m pytest` alone also matches the collection
    # rehearsal, whose exit code is deliberately not consulted — anchoring there
    # would make this test pass while the call that matters stayed broken.
    #
    # And check the STRUCTURE, not the presence of two substrings. An earlier
    # revision of this test looked for `tee ` and `echo $? >` within a window and
    # called that pinned. It was not: moving `echo $?` OUT of the brace group to
    # after the pipeline — which is exactly how this bug comes back, and is what
    # the code looked like before it was fixed — leaves both substrings present
    # and in the window. Review demonstrated that mutant passing 12/12 while the
    # lane silently reported a red pytest as rc=0.
    #
    # The property that actually matters: the rc capture is INSIDE the left side
    # of the pipe, i.e. it appears before the `| tee` that follows it.
    for cmd in ("--junitxml", "assert_crossval_solver_lane.py"):
        idx = run.index(cmd)
        window = run[max(0, idx - 700): idx + 900]
        assert "tee " in window, f"the {cmd} call no longer tees; a hang would print nothing"
        capture = window.find("echo $? >")
        assert capture != -1, (
            f"the {cmd} call does not capture its own exit code. `$?` after a pipe "
            "to tee is tee's status, which is always 0."
        )
        pipe_to_tee = window.find("| tee", capture)
        assert pipe_to_tee != -1, (
            f"the {cmd} call captures an exit code but no `| tee` follows it, so "
            "the capture is not inside the pipeline's left side. Either the tee "
            "went away or the capture moved after the pipe — the second is the "
            "regression this test exists for."
        )
        between = window[capture:pipe_to_tee]
        assert "}" in between, (
            f"the {cmd} call's `echo $? >` is not inside a brace group that the "
            "pipe closes over. Written as `cmd | tee log` then `echo $? > rc` on "
            "the next line, the recorded code is TEE's and is always 0 — a red "
            "pytest reports green. It must read `{ cmd; echo $? > rc; } | tee log`."
        )
    # Comments are allowed to NAME the bashisms — the lane explains why it avoids
    # them — so strip comment lines before checking that none is actually used.
    code = "\n".join(
        ln for ln in run.split("\n") if not ln.lstrip().startswith("#")
    )
    assert "pipefail" not in code, (
        "`set -o pipefail` is a bashism; this runner shell is dash/busybox-ash"
    )
    assert "PIPESTATUS" not in code, (
        "${PIPESTATUS[0]} is a bashism that has already killed a lane on this "
        "cluster with 'bad substitution' (gpu_suite run 369367242339)"
    )
    assert 'rc=$(cat "$OUT/pytest.rc")' in run and 'gate_rc=$(cat "$OUT/gate.rc")' in run, (
        "the lane no longer reads the captured codes back out of their files"
    )


def test_the_lane_names_a_missing_junit_as_an_environment_failure() -> None:
    run = _lane_run()
    assert '[ ! -s "$OUT/junit.xml" ]' in run, (
        "a pytest that dies before writing junit leaves the gate with nothing to "
        "parse; the lane must say that is an environment failure rather than let an "
        "XML traceback stand in for a cross-solver verdict"
    )

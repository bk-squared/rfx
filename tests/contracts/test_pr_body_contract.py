"""The PR-body provenance gate: the accepted forms, and the ones that must fail.

``scripts/ci/check_pr_body.py`` is what makes the separate-instance review rule
enforceable instead of advisory. These tests pin the shape of the contract, so
a later edit to the regexes cannot quietly widen it -- in particular that
``REJECT`` never passes, that the PR template's own stub text does not satisfy
the check, that lines invisible on render do not count (a body that merely
QUOTES the required lines has not made the claim), and that the script's own
remediation text is not itself a passing body.

The workflow assertion is the second half: the check is worthless if the body
reaches the shell instead of the environment, because a PR body is
attacker-controlled text and ``${{ }}`` inside a ``run:`` block is an
injection.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
_SCRIPT = REPO / "scripts" / "ci" / "check_pr_body.py"
_WORKFLOW = REPO / ".github" / "workflows" / "pr-body.yml"
_TEMPLATE = REPO / ".github" / "pull_request_template.md"

_spec = importlib.util.spec_from_file_location("check_pr_body", _SCRIPT)
assert _spec is not None and _spec.loader is not None
cpb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cpb
_spec.loader.exec_module(cpb)

# A deliberately small allowed set: the point is that the check reads the live
# labels, not that it hardcodes today's ten.
ENV = {"LANE_LABELS": "lane:ci-infra\nlane:crossval\nlane:msl-port\n"}

ACCEPT = "Review: opus (separate instance) — ACCEPT"


def body(*lines: str) -> str:
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Bodies that must pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "review",
    [
        "Review: opus (separate instance) — ACCEPT",
        "Review: opus (separate instance) — ACCEPT WITH CHANGES",
        "Review: opus (separate instance) - ACCEPT",
        "Review: gpt-5.6-sol (separate instance) - ACCEPT WITH CHANGES",
        "Review: skipped — (a) pure docs change, one paragraph, trivially reverted",
        "Review: skipped - (b) PI instructed the skip on 2026-09-18",
    ],
    ids=[
        "accept-emdash",
        "accept-with-changes-emdash",
        "accept-hyphen",
        "accept-with-changes-hyphen",
        "skipped-a",
        "skipped-b",
    ],
)
def test_accepted_review_forms(review: str) -> None:
    assert cpb.check(body("Lane: lane:ci-infra", review), ENV) == []


def test_lines_may_sit_anywhere_in_a_longer_body() -> None:
    text = body(
        "## What",
        "Moved the decay peak tracker to the first check.",
        "",
        "Lane: lane:crossval",
        ACCEPT,
        "",
        "\U0001f916 Generated with Claude Code",
    )
    assert cpb.check(text, ENV) == []


def test_fallback_lane_list_is_used_when_the_env_is_absent() -> None:
    """A local run with no LANE_LABELS still checks against a real lane set."""
    assert cpb.check(body("Lane: lane:ci-infra", ACCEPT), {}) == []
    lanes, source = cpb.allowed_lanes({})
    assert lanes == cpb.FALLBACK_LANE_LABELS
    assert "fallback" in source
    assert all(re.fullmatch(r"lane:[a-z0-9-]+", label) for label in lanes)


def test_an_empty_lane_labels_env_falls_back_rather_than_rejecting_everything() -> None:
    """`gh label list` failing must not make every lane label unknown."""
    assert cpb.check(body("Lane: lane:absorber", ACCEPT), {"LANE_LABELS": "\n\n"}) == []


# --------------------------------------------------------------------------
# Bodies that must fail
# --------------------------------------------------------------------------


def test_reject_never_passes() -> None:
    problems = cpb.check(
        body("Lane: lane:ci-infra", "Review: opus (separate instance) — REJECT"),
        ENV,
    )
    assert any("Review:" in p for p in problems)


def test_missing_lane_fails() -> None:
    problems = cpb.check(body("Some description.", ACCEPT), ENV)
    assert len(problems) == 1
    assert "no `Lane:` line" in problems[0]


def test_missing_review_fails() -> None:
    problems = cpb.check(body("Lane: lane:ci-infra", "Reviewed by a friend."), ENV)
    assert len(problems) == 1
    assert "`Review:` line" in problems[0]


def test_unknown_lane_fails_and_lists_the_allowed_set() -> None:
    problems = cpb.check(body("Lane: lane:not-a-lane", ACCEPT), ENV)
    assert len(problems) == 1
    assert "lane:not-a-lane" in problems[0]
    assert "lane:crossval" in problems[0]


def test_two_lane_lines_fail() -> None:
    problems = cpb.check(
        body("Lane: lane:ci-infra", "Lane: lane:crossval", ACCEPT), ENV
    )
    assert len(problems) == 1
    assert "2 `Lane:` lines" in problems[0]


def test_two_review_lines_fail() -> None:
    problems = cpb.check(
        body(
            "Lane: lane:ci-infra",
            ACCEPT,
            "Review: opus (separate instance) - ACCEPT WITH CHANGES",
        ),
        ENV,
    )
    assert len(problems) == 1
    assert "2 `Review:` lines" in problems[0]


def test_an_empty_body_fails() -> None:
    assert len(cpb.check("", ENV)) == 2


def test_crlf_bodies_are_handled() -> None:
    """GitHub delivers bodies with CRLF; a stray \\r must not break the anchors."""
    assert cpb.check("Lane: lane:ci-infra\r\n" + ACCEPT + "\r\n", ENV) == []


# --------------------------------------------------------------------------
# A placeholder reviewer is not a review record
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "review",
    [
        "Review: <who reviewed> (separate instance) - ACCEPT",
        "Review: <who read it> (separate instance) — ACCEPT WITH CHANGES",
        "Review: <name> (separate instance) - ACCEPT",
    ],
)
def test_angle_bracket_placeholders_do_not_pass(review: str) -> None:
    problems = cpb.check(body("Lane: lane:ci-infra", review), ENV)
    assert len(problems) == 1
    assert "placeholder" in problems[0]


def test_the_remediation_text_is_not_itself_a_passing_body() -> None:
    """Pasting the failure message into a PR body must fail the same check.

    This is the trap the first cut of the script fell into: the "here is what
    to add" block was a literally valid body, so an author could satisfy the
    gate by pasting the error.
    """
    report = cpb.failure_report(cpb.check("", ENV), ENV)
    problems = cpb.check(report, ENV)
    assert len(problems) == 2, f"the remediation text passes its own gate: {problems}"
    # It must fail ON THE PLACEHOLDERS, not because some earlier line happened
    # to swallow the rest of the report. The first cut wrote a literal "<pre>"
    # in its own prose, which strip_invisible then ate to end-of-body -- the
    # test was green for a reason that had nothing to do with the placeholders.
    assert "placeholder" in problems[1], problems[1]
    assert "Review: <who read it> (separate instance) - ACCEPT" in cpb.body_lines(report)


def test_failure_report_contains_no_literal_pre_tag() -> None:
    """A literal `<pre>` in the report would swallow everything after it."""
    report = cpb.failure_report(cpb.check("", ENV), ENV)
    assert "<pre" not in report


def test_failure_report_names_the_contract_and_the_fields() -> None:
    report = cpb.failure_report(cpb.check("", ENV), ENV)
    assert "Lane: lane:<name>" in report
    assert "(separate instance) - ACCEPT" in report
    assert "Review: skipped - (a)" in report
    assert "scripts/ci/check_pr_body.py" in report


# --------------------------------------------------------------------------
# A skipped review must say WHICH exception
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "review",
    [
        "Review: skipped - x",
        "Review: skipped — docs only",
        "Review: skipped - a) docs only",
        "Review: skipped - (c) I did not feel like it",
        "Review: skipped -",
    ],
)
def test_skipped_without_a_named_exception_fails(review: str) -> None:
    problems = cpb.check(body("Lane: lane:ci-infra", review), ENV)
    assert len(problems) == 1
    assert "`Review:` line" in problems[0]


def test_a_bare_skip_is_told_which_exceptions_exist() -> None:
    problems = cpb.check(body("Lane: lane:ci-infra", "Review: skipped - x"), ENV)
    assert "(a)" in problems[0] and "(b)" in problems[0]


# --------------------------------------------------------------------------
# Lines that are invisible once GitHub renders the body do not count
# --------------------------------------------------------------------------


def test_lines_inside_a_code_fence_do_not_count() -> None:
    text = body(
        "Paste this into your body:",
        "```",
        "Lane: lane:ci-infra",
        ACCEPT,
        "```",
    )
    assert len(cpb.check(text, ENV)) == 2


def test_lines_inside_an_html_comment_do_not_count() -> None:
    text = body("<!--", "Lane: lane:ci-infra", ACCEPT, "-->")
    assert len(cpb.check(text, ENV)) == 2


def test_an_unterminated_comment_swallows_the_rest_of_the_body() -> None:
    """GitHub renders nothing after an unclosed `<!--`, so neither do we."""
    text = body("<!-- note to self", "Lane: lane:ci-infra", ACCEPT)
    assert len(cpb.check(text, ENV)) == 2


def test_a_comment_before_real_lines_does_not_swallow_them() -> None:
    text = body("<!-- hint -->", "Lane: lane:ci-infra", ACCEPT)
    assert cpb.check(text, ENV) == []


def test_lines_inside_a_pre_block_do_not_count() -> None:
    text = body("<pre>", "Lane: lane:ci-infra", ACCEPT, "</pre>")
    assert len(cpb.check(text, ENV)) == 2


def test_an_unterminated_pre_block_swallows_the_rest_of_the_body() -> None:
    text = body('<pre lang="text">', "Lane: lane:ci-infra", ACCEPT)
    assert len(cpb.check(text, ENV)) == 2


def test_a_fenced_copy_does_not_satisfy_a_body_that_also_has_a_real_lane_line() -> None:
    """The fence is skipped, so one real line plus a quoted copy is still one."""
    text = body(
        "Lane: lane:ci-infra",
        ACCEPT,
        "The gate wants:",
        "```text",
        "Lane: lane:crossval",
        "```",
    )
    assert cpb.check(text, ENV) == []


def test_the_pr_template_alone_does_not_pass() -> None:
    assert _TEMPLATE.is_file(), f"missing {_TEMPLATE}"
    problems = cpb.check(_TEMPLATE.read_text(encoding="utf-8"), ENV)
    assert len(problems) == 2, (
        "the PR template must not satisfy its own gate -- an author who submits "
        f"it unedited has to be told. Got: {problems}"
    )


# --------------------------------------------------------------------------
# An adorned line still fails, but the message says why
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adorned",
    ["- Lane: lane:ci-infra", "> Lane: lane:ci-infra", "**Lane: lane:ci-infra**",
     "  Lane: lane:ci-infra", "* Lane: lane:ci-infra"],
)
def test_an_adorned_lane_line_fails_and_says_so(adorned: str) -> None:
    problems = cpb.check(body(adorned, ACCEPT), ENV)
    assert len(problems) == 1
    assert "must start the line" in problems[0], problems[0]


def test_an_adorned_review_line_fails_and_says_so() -> None:
    problems = cpb.check(body("Lane: lane:ci-infra", "- " + ACCEPT), ENV)
    assert len(problems) == 1
    assert "must start the line" in problems[0], problems[0]


def test_a_plain_missing_line_does_not_get_the_adornment_hint() -> None:
    problems = cpb.check(body("Nothing relevant here.", ACCEPT), ENV)
    assert "must start the line" not in problems[0]


# --------------------------------------------------------------------------
# The command-line entry points
# --------------------------------------------------------------------------


PASSING_BODY = body("Lane: lane:ci-infra", ACCEPT)


def _run(args: list[str], extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANE_LABELS": ENV["LANE_LABELS"],
        **extra_env,
    }
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        env=env, capture_output=True, text=True, timeout=60,
    )


def test_cli_reads_pr_body_from_the_environment() -> None:
    ok = _run([], {"PR_BODY": PASSING_BODY})
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout == "" and ok.stderr == "", "success must be silent"

    bad = _run([], {"PR_BODY": "nothing here"})
    assert bad.returncode == 1
    assert "PR body contract FAILED" in bad.stderr


def test_cli_reads_a_file(tmp_path: Path) -> None:
    good = tmp_path / "good.md"
    good.write_text(PASSING_BODY, encoding="utf-8")
    ok = _run(["--file", str(good)], {})
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout == "" and ok.stderr == ""

    bad = tmp_path / "bad.md"
    bad.write_text("## What\n\nno provenance lines\n", encoding="utf-8")
    failed = _run(["--file", str(bad)], {})
    assert failed.returncode == 1
    assert "no `Lane:` line" in failed.stderr


def test_cli_with_no_body_at_all_fails_rather_than_passing_vacuously() -> None:
    result = _run([], {})
    assert result.returncode == 1
    assert "no body to check" in result.stderr


# --------------------------------------------------------------------------
# The workflow that runs it
# --------------------------------------------------------------------------


def _workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _run_blocks(node: object) -> list[str]:
    """Every `run:` script anywhere in the workflow."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                found.append(value)
            else:
                found += _run_blocks(value)
    elif isinstance(node, list):
        for item in node:
            found += _run_blocks(item)
    return found


def test_workflow_exists_and_runs_the_script() -> None:
    assert _WORKFLOW.is_file(), f"missing {_WORKFLOW}"
    data = _workflow()
    scripts = _run_blocks(data)
    assert any("scripts/ci/check_pr_body.py" in block for block in scripts)
    # `on:` parses as the boolean True in YAML 1.1; accept either spelling.
    triggers = data.get("on", data.get(True))
    assert set(triggers["pull_request"]["types"]) == {
        "opened", "edited", "reopened", "synchronize",
    }


def test_no_workflow_expression_is_interpolated_into_any_run_block() -> None:
    """A PR body is attacker-controlled; `${{ }}` in a `run:` block is injection.

    Constrained to every shell block, not only the lines that name the body:
    a later step that interpolates the title, the branch name or a label is
    the same defect.
    """
    offenders = [block for block in _run_blocks(_workflow()) if "${{" in block]
    assert not offenders, (
        "workflow expressions must reach a shell through `env:`, never by "
        f"interpolation into `run:`. Offending block(s): {offenders}"
    )


def test_workflow_passes_the_body_through_env() -> None:
    steps = _workflow()["jobs"]["pr-body-contract"]["steps"]
    env_values = [
        value
        for step in steps
        for value in (step.get("env") or {}).values()
        if isinstance(value, str)
    ]
    assert "${{ github.event.pull_request.body }}" in env_values


def test_workflow_asks_to_be_a_required_check() -> None:
    """A gate nobody made required is advisory, which is what prose already was."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "REQUIRED check in branch protection" in text

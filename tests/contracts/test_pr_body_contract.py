"""The PR-body provenance gate: the accepted forms, and the ones that must fail.

``scripts/ci/check_pr_body.py`` is what makes the separate-instance review rule
enforceable instead of advisory. These tests pin the shape of the contract, so
a later edit to the regexes cannot quietly widen it -- in particular that
``REJECT`` never passes, that the PR template's own stub text does not satisfy
the check, and that lines inside a code fence do not count (a body that merely
QUOTES the required lines has not made the claim).

The workflow assertion is the second half: the check is worthless if the body
reaches the shell instead of the environment, because a PR body is
attacker-controlled text and ``${{ }}`` inside a ``run:`` block is an
injection.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

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
        "skipped-emdash",
        "skipped-hyphen",
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


def test_lines_inside_a_code_fence_do_not_count() -> None:
    text = body(
        "Paste this into your body:",
        "```",
        "Lane: lane:ci-infra",
        ACCEPT,
        "```",
    )
    problems = cpb.check(text, ENV)
    assert len(problems) == 2, problems


def test_lines_inside_an_html_comment_do_not_count() -> None:
    text = body("<!--", "Lane: lane:ci-infra", ACCEPT, "-->")
    problems = cpb.check(text, ENV)
    assert len(problems) == 2, problems


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


def test_an_empty_body_fails() -> None:
    assert len(cpb.check("", ENV)) == 2


def test_crlf_bodies_are_handled() -> None:
    """GitHub delivers bodies with CRLF; a stray \\r must not break the anchors."""
    assert cpb.check("Lane: lane:ci-infra\r\n" + ACCEPT + "\r\n", ENV) == []


def test_failure_report_prints_pasteable_lines() -> None:
    report = cpb.failure_report(cpb.check("", ENV), ENV)
    assert "Lane: lane:ci-infra" in report
    assert "(separate instance) - ACCEPT" in report
    assert "scripts/ci/check_pr_body.py" in report


# --------------------------------------------------------------------------
# The workflow that runs it
# --------------------------------------------------------------------------


def test_workflow_exists_and_runs_the_script() -> None:
    assert _WORKFLOW.is_file(), f"missing {_WORKFLOW}"
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/ci/check_pr_body.py" in text
    assert "types: [opened, edited, reopened, synchronize]" in text


def test_workflow_passes_the_body_through_env_not_the_shell() -> None:
    """A PR body is attacker-controlled; `${{ }}` in a `run:` block is injection."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    uses = [line for line in text.splitlines() if "pull_request.body" in line]
    assert uses, "the workflow never reads the PR body"
    for line in uses:
        assert re.fullmatch(
            r"\s*PR_BODY: \$\{\{ github\.event\.pull_request\.body \}\}", line
        ), f"body must reach the script via env:, got {line!r}"
    assert re.search(r"^\s*env:\n\s*PR_BODY:", text, re.MULTILINE), (
        "PR_BODY must sit under an `env:` block"
    )

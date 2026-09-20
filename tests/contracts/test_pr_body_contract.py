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
        "Review: opus (separate instance) – ACCEPT",
        "Review: opus (separate instance) – ACCEPT WITH CHANGES",
        "Review: skipped — (a) pure docs change, one paragraph, trivially reverted",
        "Review: skipped - (b) PI instructed the skip on 2026-09-18",
        "Review: skipped – (a) one comment typo, reverted with one click",
    ],
    ids=[
        "accept-emdash",
        "accept-with-changes-emdash",
        "accept-hyphen",
        "accept-with-changes-hyphen",
        "accept-endash",
        "accept-with-changes-endash",
        "skipped-a",
        "skipped-b",
        "skipped-endash",
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
    assert len(problems) == 1
    assert "verdict must be ACCEPT" in problems[0], problems[0]


def test_an_unaccepted_separator_says_which_ones_are_accepted() -> None:
    """The en dash is accepted now; a fourth glyph must still explain itself."""
    problems = cpb.check(
        body("Lane: lane:ci-infra", "Review: opus (separate instance) : ACCEPT"), ENV
    )
    assert len(problems) == 1
    assert "em dash" in problems[0] and "en dash" in problems[0]
    assert "hyphen" in problems[0]


def test_the_failure_report_names_the_accepted_separators() -> None:
    report = cpb.failure_report(cpb.check("", ENV), ENV)
    assert "em dash" in report and "en dash" in report and "hyphen" in report


def test_a_literal_null_body_fails_cleanly() -> None:
    """A PR with no description delivers the string "null", not an empty body.

    `gh pr view --json body --jq .body` and the workflow context both render a
    missing body that way. It has to come out as the ordinary "no Lane: line"
    failure, not a crash and not a pass.
    """
    problems = cpb.check("null", ENV)
    assert len(problems) == 2
    assert "no `Lane:` line" in problems[0]
    assert "`Review:` line" in problems[1]


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


@pytest.mark.parametrize(
    "review",
    [
        "Review: claude <noreply@anthropic.com> (separate instance) - ACCEPT",
        "Review: Claude Opus 5 <noreply@anthropic.com> (separate instance) \u2014 ACCEPT",
        "Review: opus <1M context> (separate instance) - ACCEPT WITH CHANGES",
    ],
)
def test_a_name_that_merely_contains_angle_brackets_is_not_a_placeholder(
    review: str,
) -> None:
    """`claude <noreply@anthropic.com>` is a reviewer; `<who read it>` is not.

    Only a who-field that is nothing BUT a bracketed span is an unedited stub.
    Rejecting every angle bracket turned the repo's own commit-trailer spelling
    of a reviewer into a failure.
    """
    assert cpb.check(body("Lane: lane:ci-infra", review), ENV) == [], review


@pytest.mark.parametrize(
    "shown",
    ["<!-- note to self", "<pre>", "<pre lang=\"text\">", "<!--"],
    ids=["open-comment", "pre", "pre-with-info", "bare-open-comment"],
)
def test_an_unterminated_marker_inside_a_fence_does_not_swallow_the_body(
    shown: str,
) -> None:
    """A fence SHOWS markup; it does not apply it.

    A body that demonstrates an unterminated comment or an unclosed `<pre>`
    inside a code fence still has its own Lane and Review lines below the
    closing fence, and they render. Masking inline code alone was not enough:
    the marker inside the fence swallowed everything after it.
    """
    text = body(
        "For example:",
        "```",
        shown,
        "```",
        "",
        "Lane: lane:ci-infra",
        ACCEPT,
    )
    assert cpb.check(text, ENV) == [], shown


def test_a_tilde_fence_hides_markers_too() -> None:
    text = body("~~~", "<!-- note", "~~~", "", "Lane: lane:ci-infra", ACCEPT)
    assert cpb.check(text, ENV) == []


def test_a_real_unterminated_comment_outside_a_fence_still_swallows() -> None:
    """The fence masking must not disarm the rule it was narrowing."""
    text = body("<!-- note to self", "", "Lane: lane:ci-infra", ACCEPT)
    assert len(cpb.check(text, ENV)) == 2


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


@pytest.mark.parametrize(
    "prose",
    [
        "An unclosed `<!--` swallows the rest of the body on render.",
        "Fenced code, `<pre>` blocks and `<!-- -->` comments are all stripped.",
        "Write `<!--` when you mean a comment.",
    ],
)
def test_a_backticked_tag_in_prose_does_not_eat_the_body(prose: str) -> None:
    """Inline code is literal text on render, not markup.

    This is not hypothetical: the first version of this gate ate its own PR
    body. One backticked `<!--` in a sentence ABOUT unterminated comments
    swallowed the Lane and Review lines forty lines further down, and the job
    reported "no Lane: line" on a body that plainly had one.
    """
    text = body("## What", prose, "", "Lane: lane:ci-infra", ACCEPT)
    assert cpb.check(text, ENV) == [], prose


def test_the_prs_own_body_style_passes() -> None:
    """A body that documents the contract in prose still satisfies it."""
    text = body(
        "Exactly one `Lane: lane:<label>` line and one `Review: ...` line.",
        "Lines inside `<pre>` blocks and unterminated `<!--` comments do not count.",
        "",
        "```",
        "Lane: lane:crossval",
        "```",
        "",
        "Lane: lane:ci-infra",
        ACCEPT,
    )
    assert cpb.check(text, ENV) == []


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
        "opened", "edited", "reopened", "synchronize", "labeled", "unlabeled",
    }, (
        "`labeled`/`unlabeled` cover a lane label changed BY HAND. They do not "
        "cover .github/workflows/labeler.yml -- an event triggered by "
        "GITHUB_TOKEN creates no workflow run -- which is why the label set is "
        "read live rather than taken from the event payload"
    )


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


def test_workflow_reads_the_labels_live_not_from_the_event_payload() -> None:
    """The payload's label list is wrong in both directions.

    On `opened` it is empty, because .github/workflows/labeler.yml has not run
    yet. And when that job does run, its label change fires no event here at
    all: an event triggered by GITHUB_TOKEN creates no workflow run. A gate that
    cross-checks the `Lane:` line against labels it read from the payload would
    be cross-checking against nothing on most PRs.
    """
    steps = _workflow()["jobs"]["pr-body-contract"]["steps"]
    blocks = "\n".join(_run_blocks(_workflow()))
    env_values = [
        value
        for step in steps
        for value in (step.get("env") or {}).values()
        if isinstance(value, str)
    ]
    assert "${{ toJSON(github.event.pull_request.labels.*.name) }}" not in env_values
    assert "gh pr view" in blocks and "PR_LABELS_JSON" in blocks
    assert "${{ github.event.pull_request.number }}" in env_values


# --------------------------------------------------------------------------
# The lane: the line is required, and the labels cross-check it
# --------------------------------------------------------------------------

#: A PR that `.github/labeler.yml` put in one lane.
ONE_LABEL = dict(ENV, PR_LABELS_JSON='["lane:ci-infra", "release"]')
TWO_LABELS = dict(ENV, PR_LABELS_JSON='["lane:ci-infra", "lane:crossval"]')


def test_a_lane_label_does_not_replace_the_line() -> None:
    """Measured over 60 merged PRs: 32 earn two lane labels, 6 earn none.

    For 38 of 60 the labels cannot say which lane OWNS the change, so the line
    is the claim and the labels only cross-check it.
    """
    problems = cpb.check(body(ACCEPT), env=ONE_LABEL)
    assert len(problems) == 1 and "no `Lane:` line" in problems[0]


def test_the_missing_line_message_suggests_a_label_the_pr_carries() -> None:
    problems = cpb.check(body(ACCEPT), env=TWO_LABELS)
    assert "Lane: lane:ci-infra" in problems[0]
    assert "lane:crossval" in problems[0]


def test_no_line_and_no_label_fails() -> None:
    problems = cpb.check(body(ACCEPT), env=ENV)
    assert len(problems) == 1 and "no `Lane:` line" in problems[0]


def test_a_line_naming_one_of_two_labels_passes() -> None:
    assert cpb.check(body("Lane: lane:crossval", ACCEPT), env=TWO_LABELS) == []


def test_a_line_that_contradicts_the_labels_fails() -> None:
    """Either the line is wrong or labeler.yml is missing a path. Both matter."""
    problems = cpb.check(body("Lane: lane:msl-port", ACCEPT), env=ONE_LABEL)
    assert len(problems) == 1
    assert "not among the lane labels" in problems[0]
    assert "labeler.yml" in problems[0]


def test_a_line_that_agrees_with_the_single_label_passes() -> None:
    assert cpb.check(body("Lane: lane:ci-infra", ACCEPT), env=ONE_LABEL) == []


def test_a_line_still_works_with_no_labels_at_all() -> None:
    """A PR confined to unowned paths carries no label, and is not cross-checked."""
    assert cpb.check(body("Lane: lane:ci-infra", ACCEPT), env=ENV) == []


def test_two_lane_lines_fail_even_when_a_label_agrees() -> None:
    problems = cpb.check(
        body("Lane: lane:ci-infra", "Lane: lane:crossval", ACCEPT), env=ONE_LABEL
    )
    assert len(problems) == 1 and "2 `Lane:` lines" in problems[0]


def test_an_unknown_lane_in_the_line_fails_before_the_labels_are_consulted() -> None:
    problems = cpb.check(body("Lane: lane:invented", ACCEPT), env=ONE_LABEL)
    assert len(problems) == 1 and "is not a lane label on this repository" in problems[0]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "[]",
        "not json",
        '{"name": "lane:ci-infra"}',
        '["release", "bug"]',
    ],
)
def test_a_payload_carrying_no_lane_label_switches_the_cross_check_off(raw: str) -> None:
    """Missing, malformed and lane-free label sets all read as "no labels".

    Erroring instead would fail every PR the day GitHub changes the payload,
    and the `Lane:` line remains a complete answer on its own.
    """
    env = dict(ENV, PR_LABELS_JSON=raw)
    assert cpb.pr_lane_labels(env) == ()
    assert cpb.check(body("Lane: lane:crossval", ACCEPT), env=env) == []


def test_a_label_containing_a_comma_survives_the_json() -> None:
    env = dict(ENV, PR_LABELS_JSON='["lane:ci-infra", "needs,triage"]')
    assert cpb.pr_lane_labels(env) == ("lane:ci-infra",)


# --------------------------------------------------------------------------
# A retired label warns; it never fails
# --------------------------------------------------------------------------


def test_a_lane_label_the_repository_no_longer_has_is_only_a_warning() -> None:
    """Retiring a label must not turn every open PR red for nobody's mistake."""
    env = dict(ENV, PR_LABELS_JSON='["lane:ci-infra", "lane:retired"]')
    warnings = cpb.lane_label_warnings(env)
    assert len(warnings) == 1 and "lane:retired" in warnings[0]
    assert cpb.check(body("Lane: lane:ci-infra", ACCEPT), env=env) == []


def test_a_stale_label_is_warned_and_then_IGNORED() -> None:
    """Warned and ignored, on every path -- not warned and then failed on.

    A PR whose only lane label has been retired is a PR with no lane label. The
    earlier version cross-checked against the unfiltered set, so this body
    failed one branch after the warning promised it would not, and the failure
    text then suggested `Lane: lane:retired`, which fails the allowed-set
    branch. Two dead ends for something the author did not do.
    """
    env = dict(ENV, PR_LABELS_JSON='["lane:retired"]')
    warnings = cpb.lane_label_warnings(env)
    assert len(warnings) == 1 and "lane:retired" in warnings[0]
    assert cpb.check(body("Lane: lane:ci-infra", ACCEPT), env=env) == []


def test_a_stale_label_is_not_suggested_when_the_line_is_missing() -> None:
    """The suggestion has to be a lane the author can actually write."""
    env = dict(ENV, PR_LABELS_JSON='["lane:retired", "lane:crossval"]')
    problems = cpb.check(body(ACCEPT), env=env)
    assert len(problems) == 1
    assert "Lane: lane:crossval" in problems[0]
    assert "lane:retired" not in problems[0]


def test_a_pr_whose_only_lane_label_is_stale_is_not_cross_checked_at_all() -> None:
    env = dict(ENV, PR_LABELS_JSON='["lane:retired"]')
    for lane in ("lane:ci-infra", "lane:crossval", "lane:msl-port"):
        assert cpb.check(body(f"Lane: {lane}", ACCEPT), env=env) == [], lane


def test_a_current_label_set_warns_about_nothing() -> None:
    assert cpb.lane_label_warnings(ONE_LABEL) == []
    assert cpb.lane_label_warnings(ENV) == []


def test_the_remediation_text_says_the_label_is_not_a_substitute() -> None:
    report = cpb.failure_report(cpb.check(body(ACCEPT), env=ENV), env=ENV)
    assert "do not replace the line" in report



#: The ruff scope CI runs: the packages under test, plus every script that gates
#: a merge. The rest of `scripts/` has 197 findings under this selector and is
#: its own cleanup -- but an unlinted gate is a gate nobody notices breaking, so
#: a helper that can fail a PR is never outside the scope.
RUFF_SCOPE = (
    "rfx/",
    "tests/",
    "validation/",
    "scripts/ci/",
    "scripts/dev/",
    "scripts/changelog/",
)


def test_the_lint_scope_covers_this_gates_own_script() -> None:
    """An unlinted CI helper is one nobody notices breaking.

    The line lives in `scripts/ci/lint.sh` since 2026-09-18, not in the workflow:
    a gate whose only copy is inline yaml cannot be reproduced locally, which is
    how two red checks on one PR became un-debuggable. `lint.yml` calls the
    script; this pins what the script lints.
    """
    script = REPO / "scripts" / "ci" / "lint.sh"
    assert script.is_file(), f"missing {script}"
    ruff_lines = [
        line
        for line in script.read_text(encoding="utf-8").splitlines()
        if "ruff check" in line and not line.lstrip().startswith("#")
    ]
    assert len(ruff_lines) == 1, f"expected one ruff invocation, got {ruff_lines}"
    for path in RUFF_SCOPE:
        assert path in ruff_lines[0], (
            f"{path} dropped from the CI ruff scope: {ruff_lines[0].strip()!r}"
        )


def test_the_lint_workflow_calls_the_script() -> None:
    """The workflow must not grow its own copy of the ruff line again."""
    lint = (REPO / ".github" / "workflows" / "lint.yml").read_text(encoding="utf-8")
    assert "scripts/ci/lint.sh" in lint
    assert "ruff check" not in lint, "the ruff line is inline in the workflow again"


def test_workflow_asks_to_be_a_required_check() -> None:
    """A gate nobody made required is advisory, which is what prose already was."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "REQUIRED check in branch protection" in text

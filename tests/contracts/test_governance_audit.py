"""The weekly governance report's classifier, on fixtures rather than the API.

Everything that decides whether a row appears is a pure function over JSON the
`gh` calls returned, so the interesting cases can be written down instead of
waited for: a PR whose review line is fine, one that claimed an exception, an
issue filed outside the forms, a merged PR closing an issue in a closed
milestone.

The one that matters most is the first: the audit imports the PR gate's own
review parser, so a body CI accepted must not appear here as unreviewed. Two
copies of that rule would disagree within a week and the report would cry wolf
every Monday until somebody stopped reading it.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "governance_audit.py"

_SPEC = importlib.util.spec_from_file_location("governance_audit", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)

ACCEPTED = "Review: opus (separate instance) — ACCEPT"
SKIPPED = "Review: skipped — (a) docs only, one paragraph, trivially reverted"

FORM_ISSUE = "### What is wrong\n\nx\n\n### Lane\n\nlane:absorber\n"
HAND_ISSUE = "Filed by hand, no form, no lane.\n"


def pr(number: int, body: str, title: str = "a change") -> dict:
    return {"number": number, "title": title, "body": body,
            "mergedAt": "2026-09-17T00:00:00Z", "url": f"u/{number}"}


def issue(number: int, body: str, milestone: dict | None = None,
          labels: list | None = None) -> dict:
    """An issue record. *labels* defaults to one lane label, the healthy case."""
    return {"number": number, "title": f"issue {number}", "body": body,
            "createdAt": "2026-09-17T00:00:00Z", "milestone": milestone,
            "url": f"u/{number}",
            "labels": [{"name": "lane:absorber"}] if labels is None else labels}


OPEN_MILESTONE = {"title": "port/S-param basic support", "state": "open"}
CLOSED_MILESTONE = {"title": "v1.9 — MSL + coax S-chain", "state": "closed"}


def kinds(rows) -> list[str]:
    return [row.kind for row in rows]


# --------------------------------------------------------------------------
# Reviews: the audit and the gate must agree
# --------------------------------------------------------------------------


def test_an_accepted_review_line_produces_no_row() -> None:
    body = f"Lane: lane:absorber\n{ACCEPTED}\n\nFixes #10\n"
    rows = classify_one(body, {10: issue(10, "", OPEN_MILESTONE)})
    assert "review" not in kinds(rows)


def test_a_body_with_no_review_line_is_reported() -> None:
    rows = classify_one("Fixes #10\n", {10: issue(10, "", OPEN_MILESTONE)})
    assert kinds(rows) == ["review"]
    assert "no accepted" in rows[0].detail


def test_a_claimed_exception_is_reported_with_the_reason() -> None:
    """Allowed by the gate. Counted here, because an exception used weekly is not one."""
    body = f"Lane: lane:absorber\n{SKIPPED}\n\nFixes #10\n"
    rows = classify_one(body, {10: issue(10, "", OPEN_MILESTONE)})
    assert kinds(rows) == ["review"]
    assert "(a) docs only" in rows[0].detail


def test_a_reject_verdict_is_reported() -> None:
    body = "Lane: lane:absorber\nReview: opus (separate instance) — REJECT\n\nFixes #10\n"
    rows = classify_one(body, {10: issue(10, "", OPEN_MILESTONE)})
    assert kinds(rows) == ["review"]


def test_a_review_line_inside_a_code_fence_does_not_count() -> None:
    """Same reading as the gate: a body that QUOTES the line has not claimed it."""
    body = f"Fixes #10\n\n```\n{ACCEPTED}\n```\n"
    rows = classify_one(body, {10: issue(10, "", OPEN_MILESTONE)})
    assert kinds(rows) == ["review"]


# --------------------------------------------------------------------------
# Links and milestones
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("Fixes #10", [10]),
        ("fixed #10", [10]),
        ("Closes #10 and closes #11", [10, 11]),
        ("Resolves: #10", [10]),
        ("CLOSED #10", [10]),
        ("Fixes #10, fixes #10", [10]),
        ("Related to #10", []),
        ("See #10", []),
        ("prefixed #10", []),
        ("", []),
    ],
)
def test_only_githubs_closing_keywords_link_an_issue(phrase, expected) -> None:
    """"Related to #10" does not close #10, and its milestone does not apply."""
    assert audit.linked_issues(phrase) == expected


def test_an_issue_in_an_open_milestone_produces_no_row() -> None:
    body = f"{ACCEPTED}\n\nFixes #10\n"
    rows = classify_one(body, {10: issue(10, "", OPEN_MILESTONE)})
    assert rows == []


def test_an_issue_with_no_milestone_is_reported() -> None:
    body = f"{ACCEPTED}\n\nFixes #10\n"
    rows = classify_one(body, {10: issue(10, "", None)})
    assert kinds(rows) == ["milestone"]
    assert "no milestone" in rows[0].detail


def test_an_issue_in_a_closed_milestone_is_reported() -> None:
    body = f"{ACCEPTED}\n\nFixes #10\n"
    rows = classify_one(body, {10: issue(10, "", CLOSED_MILESTONE)})
    assert kinds(rows) == ["milestone"]
    assert "is closed" in rows[0].detail


def test_a_link_to_another_pull_request_is_skipped() -> None:
    """"Fixes #1042" pointing at a PR is a cross-reference between PRs.

    `gh issue view <PR number>` SUCCEEDS with a null milestone, which read as
    "an issue with no milestone" and put every cross-referenced PR in the
    report. The index marks a PR so classify can tell the two apart.
    """
    body = f"{ACCEPTED}\n\nFixes #1042\n"
    rows = classify_one(body, {1042: {"number": 1042, "title": "a PR",
                                      "milestone": None, "is_pull_request": True}})
    assert rows == []


def test_a_link_to_a_milestone_less_ISSUE_is_still_reported() -> None:
    """The skip is for pull requests only, not for anything without a milestone."""
    body = f"{ACCEPTED}\n\nFixes #1042\n"
    rows = classify_one(body, {1042: {"number": 1042, "title": "an issue",
                                      "milestone": None, "is_pull_request": False}})
    assert kinds(rows) == ["milestone"]


def test_a_link_the_audit_could_not_resolve_is_reported_not_skipped() -> None:
    """A number `gh issue view` refused is exactly the case worth a human look."""
    body = f"{ACCEPTED}\n\nFixes #999\n"
    rows = classify_one(body, {})
    assert kinds(rows) == ["milestone"]
    assert "not found" in rows[0].detail


def test_a_pr_closing_no_issue_is_reported_once() -> None:
    rows = classify_one(f"{ACCEPTED}\n\nhousekeeping\n", {})
    assert kinds(rows) == ["unlinked"]


def test_a_pr_closing_no_issue_does_not_make_the_run_red() -> None:
    """65 of 102 merged PRs in one week, and most were legitimate housekeeping.

    A report that is red every week for a thing that is usually fine is a report
    nobody opens by the third week, so this section is listed and never counted.
    """
    rows = classify_one(f"{ACCEPTED}\n\nhousekeeping\n", {})
    assert audit.failing_rows(rows) == []


@pytest.mark.parametrize("kind", ["review", "form", "milestone", "unlabelled"])
def test_the_other_three_sections_do_make_the_run_red(kind: str) -> None:
    assert audit.failing_rows([audit.Row(kind, "#1", "t", "d")])


def test_an_unlinked_pr_is_not_also_reported_for_a_milestone() -> None:
    """One row per problem: a PR with no link has no milestone question to answer."""
    rows = classify_one("no review, no link\n", {})
    assert kinds(rows) == ["review", "unlinked"]


def test_each_linked_issue_gets_its_own_row() -> None:
    body = f"{ACCEPTED}\n\nCloses #10, closes #11\n"
    rows = audit.classify(
        [pr(1, body)], [], {10: issue(10, "", None), 11: issue(11, "", CLOSED_MILESTONE)}
    )
    assert kinds(rows) == ["milestone", "milestone"]
    assert "#10" in rows[0].detail and "#11" in rows[1].detail


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------


def test_an_issue_filed_through_a_form_produces_no_row() -> None:
    rows = audit.classify([], [issue(20, FORM_ISSUE)], {})
    assert rows == []


def test_an_issue_filed_outside_the_forms_is_reported() -> None:
    rows = audit.classify([], [issue(20, HAND_ISSUE)], {})
    assert kinds(rows) == ["form"]
    assert rows[0].ref == "#20"


# --------------------------------------------------------------------------
# The label, not the heading
# --------------------------------------------------------------------------


def test_a_form_issue_that_never_got_a_lane_label_is_reported() -> None:
    """The gap the heading check could not see.

    .github/workflows/issue-lane-label.yml exits 0 without labelling anything
    when the dropdown was left unanswered or a second `### Lane` section was
    pasted in. The issue is then well-formed, invisible to every lane query, and
    nothing on it says so.
    """
    rows = audit.classify([], [issue(20, FORM_ISSUE, labels=[])], {})
    assert kinds(rows) == ["unlabelled"]
    assert rows[0].detail == "no `lane:*` label"


def test_an_unanswered_dropdown_is_caught_by_the_label_check() -> None:
    body = "### What is wrong\n\nx\n\n### Lane\n\n_No response_\n"
    rows = audit.classify([], [issue(21, body, labels=[])], {})
    assert kinds(rows) == ["unlabelled"]


def test_two_conflicting_lane_sections_are_caught_by_the_label_check() -> None:
    body = "### Lane\n\nlane:absorber\n\n### Lane\n\nlane:msl-port\n"
    assert audit._lane.resolve_lane(body)[0] is None  # the label job does nothing
    rows = audit.classify([], [issue(22, body, labels=[])], {})
    assert kinds(rows) == ["unlabelled"]


def test_a_non_lane_label_does_not_satisfy_the_check() -> None:
    rows = audit.classify([], [issue(20, FORM_ISSUE, labels=["bug", "release"])], {})
    assert kinds(rows) == ["unlabelled"]


def test_a_labelled_form_issue_produces_no_row() -> None:
    rows = audit.classify([], [issue(20, FORM_ISSUE, labels=["lane:absorber"])], {})
    assert rows == []


def test_an_issue_with_neither_appears_in_both_sections() -> None:
    """Two different facts: nobody was asked, and nothing can find it."""
    rows = audit.classify([], [issue(20, HAND_ISSUE, labels=[])], {})
    assert kinds(rows) == ["unlabelled", "form"]


def test_a_hand_filed_issue_someone_labelled_by_hand_is_only_a_form_row() -> None:
    rows = audit.classify([], [issue(20, HAND_ISSUE, labels=["lane:plan"])], {})
    assert kinds(rows) == ["form"]


@pytest.mark.parametrize(
    "labels,expected",
    [
        ([{"name": "lane:plan"}, {"name": "bug"}], ["lane:plan"]),
        (["lane:plan", "bug"], ["lane:plan"]),
        ([], []),
        (None, []),
        ([{"colour": "red"}], []),
    ],
)
def test_lane_labels_are_read_from_either_shape(labels, expected) -> None:
    """`gh issue list --json labels` wraps names; a fixture need not."""
    assert audit.lane_labels_of({"labels": labels}) == expected


def test_the_issue_listing_asks_github_for_labels() -> None:
    """Without the field every issue reads as unlabelled and every week is red."""
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "number,title,body,createdAt,url,labels" in source


def test_the_intake_check_uses_the_lane_parser_the_workflow_uses() -> None:
    """A form whose Lane heading moved must fail in ONE place, not diverge."""
    assert audit._lane.has_lane_section(FORM_ISSUE) is True


def test_an_issue_whose_dropdown_was_left_blank_is_not_a_FORM_row() -> None:
    """The form WAS used, so not that row -- the label check is what catches it."""
    body = "### What is wrong\n\nx\n\n### Lane\n\n_No response_\n"
    rows = audit.classify([], [issue(21, body, labels=["lane:absorber"])], {})
    assert rows == []


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render(rows) -> str:
    return audit.render(
        rows, dt.date(2026, 9, 11), dt.date(2026, 9, 18), {"prs": 3, "issues": 2}
    )


def test_an_empty_report_says_so_and_has_no_table() -> None:
    text = _render([])
    assert "Nothing to look at." in text
    assert "|---|" not in text


def test_every_row_kind_has_a_section_heading() -> None:
    """A kind with no entry in SECTIONS renders nowhere and is silently lost."""
    kinds_ = [kind for kind, _, _ in audit.SECTIONS]
    rows = [
        audit.Row(kind, f"#{index}", "t", "d")
        for index, kind in enumerate(kinds_)
    ]
    text = _render(rows)
    for index in range(len(kinds_)):
        assert f"| #{index} |" in text
    assert text.count("| ref | title | detail |") == len(kinds_)


def test_the_sections_cover_every_kind_classify_can_emit() -> None:
    """Pinned by name: a kind added to classify and not to SECTIONS vanishes."""
    assert {kind for kind, _, _ in audit.SECTIONS} == {
        "review", "unlabelled", "form", "milestone", "unlinked",
    }


def test_a_pipe_in_a_title_stays_in_its_cell() -> None:
    text = _render([audit.Row("review", "#1", "a | b", "d")])
    assert "| #1 | a \\| b | d |" in text


def test_a_newline_in_a_title_does_not_break_the_table() -> None:
    text = _render([audit.Row("review", "#1", "a\nb", "d")])
    assert "| #1 | a b | d |" in text


def test_the_report_says_it_is_not_a_gate() -> None:
    assert "not a gate" in _render([audit.Row("review", "#1", "t", "d")])


def test_the_footer_separates_failing_rows_from_informational_ones() -> None:
    text = _render([
        audit.Row("review", "#1", "t", "d"),
        audit.Row("unlinked", "#2", "t", "d"),
        audit.Row("unlinked", "#3", "t", "d"),
    ])
    assert "**1 row to look at**, plus 2 informational." in text
    assert "The run is red" in text


def test_a_report_of_only_informational_rows_does_not_claim_the_run_is_red() -> None:
    """It is not red: informational rows never set the exit code.

    A footer that said otherwise would teach a reader to distrust the one that
    does, which is the whole value of making the run red at all.
    """
    text = _render([
        audit.Row("unlinked", "#1", "t", "d"),
        audit.Row("unlinked", "#2", "t", "d"),
    ])
    assert "The run is red" not in text
    assert "**Nothing to look at.** The 2 informational rows above" in text
    assert "| #1 |" in text, "the rows are still listed"


def test_the_informational_footer_counts_one_row_in_the_singular() -> None:
    text = _render([audit.Row("unlinked", "#1", "t", "d")])
    assert "The 1 informational row above" in text


# --------------------------------------------------------------------------
# Fetching: a truncated window is worse than no report
# --------------------------------------------------------------------------


def test_hitting_the_fetch_cap_raises_instead_of_truncating() -> None:
    """A silent cap prints "200 merged PRs" for a window holding 900.

    Every count under that heading is then wrong, with nothing saying so.
    """
    with pytest.raises(RuntimeError) as excinfo:
        audit._capped([{}] * 5, 5, "merged PRs", "2026-08-19..2026-09-18")
    assert "at the 5 cap" in str(excinfo.value)
    assert "fewer" in str(excinfo.value)


def test_a_window_under_the_cap_passes_through() -> None:
    rows = [{}, {}]
    assert audit._capped(rows, 5, "merged PRs", "w") is rows


def test_the_cap_is_the_search_apis_own_ceiling() -> None:
    """1000 is where GitHub search stops; a wider window cannot be reported."""
    assert audit.FETCH_LIMIT == 1000


def test_fetch_marks_a_linked_number_that_is_really_a_pull_request(monkeypatch) -> None:
    """`gh issue view <PR number>` succeeds with a null milestone.

    That read as "an issue with no milestone" and put every cross-referenced PR
    in the report. The REST payload carries a `pull_request` key, which is the
    only thing that tells the two apart, so `fetch` has to use it.
    """
    calls: list[list[str]] = []

    def fake_gh(args):
        calls.append(list(args))
        if args[0] == "pr":
            return [pr(1, f"{ACCEPTED}\n\nFixes #10, closes #11\n")]
        return []

    def fake_gh_one(args):
        calls.append(list(args))
        number = int(args[-1].rsplit("/", 1)[-1])
        raw = {"number": number, "title": f"n{number}", "milestone": None,
               "html_url": f"u/{number}"}
        if number == 11:
            raw["pull_request"] = {"url": "..."}
        return raw

    monkeypatch.setattr(audit, "_gh", fake_gh)
    monkeypatch.setattr(audit, "_gh_one", fake_gh_one)
    data = audit.fetch(7)

    assert data["issue_index"]["10"]["is_pull_request"] is False
    assert data["issue_index"]["11"]["is_pull_request"] is True
    assert not any("view" in call for call in calls), calls

    rows = audit.classify(
        data["merged_prs"], [],
        {int(k): v for k, v in data["issue_index"].items()},
    )
    assert [row.detail for row in rows] == ["closes #10: no milestone"]


# --------------------------------------------------------------------------
# The CLI, replayed from a payload: no network
# --------------------------------------------------------------------------


def _payload(**overrides) -> dict:
    data = {
        "since": "2026-09-11",
        "until": "2026-09-18",
        "merged_prs": [],
        "opened_issues": [],
        "issue_index": {},
    }
    data.update(overrides)
    return data


def _run(payload: dict, tmp_path: Path, env: dict | None = None):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    child = dict(os.environ)
    child.pop("GITHUB_STEP_SUMMARY", None)
    child.update(env or {})
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(path)],
        env=child, capture_output=True, text=True, timeout=60,
    )


def test_a_clean_week_exits_zero(tmp_path: Path) -> None:
    result = _run(
        _payload(
            merged_prs=[pr(1, f"{ACCEPTED}\n\nFixes #10\n")],
            opened_issues=[issue(20, FORM_ISSUE)],
            issue_index={"10": issue(10, "", OPEN_MILESTONE)},
        ),
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Nothing to look at." in result.stdout


def test_any_failing_row_makes_the_run_red(tmp_path: Path) -> None:
    """A green weekly summary nobody opens reports nothing."""
    result = _run(_payload(opened_issues=[issue(20, HAND_ISSUE)]), tmp_path)
    assert result.returncode == 1
    assert "#20" in result.stdout


def test_a_week_of_only_informational_rows_stays_green(tmp_path: Path) -> None:
    """Otherwise the report is red every week, for infra PRs that close nothing."""
    result = _run(
        _payload(merged_prs=[pr(1, f"{ACCEPTED}\n\nhousekeeping\n")]), tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Merged PRs closing no issue (informational)" in result.stdout
    assert "#1" in result.stdout
    assert "The run is red" not in result.stdout, (
        "a green run must not print the sentence that explains a red one"
    )


def test_the_report_is_written_to_the_step_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("", encoding="utf-8")
    result = _run(
        _payload(opened_issues=[issue(20, HAND_ISSUE)]),
        tmp_path,
        {"GITHUB_STEP_SUMMARY": str(summary)},
    )
    assert result.returncode == 1
    written = summary.read_text(encoding="utf-8")
    assert "## Governance audit" in written and "#20" in written


def test_replaying_a_payload_makes_no_network_call(tmp_path: Path, monkeypatch) -> None:
    """`--input` is what makes this testable; a stray `gh` call would break that."""
    def explode(*args, **kwargs):  # pragma: no cover - the point is not reaching it
        raise AssertionError(f"governance_audit called out: {args}")

    monkeypatch.setattr(audit.subprocess, "run", explode)
    text, count = audit.report(_payload(opened_issues=[issue(20, HAND_ISSUE)]))
    assert count == 1 and "#20" in text


def classify_one(body: str, index: dict) -> list:
    """One merged PR, no opened issues."""
    return audit.classify([pr(1, body)], [], index)

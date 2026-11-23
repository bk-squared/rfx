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


def pr(number: int, body: str, title: str = "a change", labels: list | None = None) -> dict:
    return {"number": number, "title": title, "body": body,
            "mergedAt": "2026-09-17T00:00:00Z", "url": f"u/{number}",
            "labels": labels or []}


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


def test_an_accepted_review_line_produces_no_row() -> None:
    body = f"Lane: lane:absorber\n{ACCEPTED}\n\nFixes #10\n"
    rows = classify_one(body, {10: issue(10, "", OPEN_MILESTONE)})
    assert "review" not in kinds(rows)


def test_a_body_with_no_review_line_is_reported() -> None:
    rows = classify_one("Fixes #10\n", {10: issue(10, "", OPEN_MILESTONE)})
    assert kinds(rows) == ["review"]
    assert "no accepted" in rows[0].detail


def test_a_claimed_exception_is_reported_with_the_reason() -> None:
    """A claimed docs exception stays visible even when its files were unreadable."""
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


@pytest.mark.parametrize("kind", ["review", "data-budget", "unlabelled"])
def test_the_other_sections_do_make_the_run_red(kind: str) -> None:
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


@pytest.mark.parametrize("labels", [
    [{"name": "data-budget-exception"}, {"name": "lane:ci-infra"}],
    ["data-budget-exception"],
])
def test_a_pr_that_carried_the_data_budget_exception_is_listed(labels) -> None:
    """The gate lets it through with a warning on that PR only; the week sees it here."""
    body = f"{ACCEPTED}\n\nFixes #10\n"
    rows = audit.classify([pr(1, body, labels=labels)], [],
                          {10: issue(10, "", OPEN_MILESTONE)})
    assert kinds(rows) == ["data-budget"]
    assert "not enforced" in rows[0].detail
    assert audit.failing_rows(rows), "an exception claimed is a row to look at"


def test_the_audit_reads_the_label_name_from_the_gate() -> None:
    """One constant: a renamed label must not leave the audit counting the old one."""
    assert audit._budget.EXCEPTION_LABEL == "data-budget-exception"
    assert audit._budget.__file__.endswith("scripts/ci/check_data_budget.py")


@pytest.mark.parametrize("labels", [[], ["lane:ci-infra"], ["x,data-budget-exception"], None])
def test_a_pr_without_the_label_gets_no_data_budget_row(labels) -> None:
    body = f"{ACCEPTED}\n\nFixes #10\n"
    record = pr(1, body)
    record["labels"] = labels
    rows = audit.classify([record], [], {10: issue(10, "", OPEN_MILESTONE)})
    assert rows == []


def test_the_pr_listing_asks_github_for_labels() -> None:
    """Without the field no merged PR carries the label and the row never appears."""
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "number,title,body,mergedAt,url,labels" in source


def _frozen_pr(frozen) -> dict:
    record = pr(1, f"{ACCEPTED}\n\nFixes #10\n")
    record["frozen_bytes"] = frozen
    return record


@pytest.mark.parametrize("frozen,listed", [(1_000_001, True), (1_000_000, False), (0, False)])
def test_a_pr_adding_over_1_MB_of_frozen_data_is_listed(frozen: int, listed: bool) -> None:
    """The budget bounds each PR at 5 MB, not their sum; the sum is watched here."""
    rows = audit.classify([_frozen_pr(frozen)], [], {10: issue(10, "", OPEN_MILESTONE)})
    assert (kinds(rows) == ["frozen"]) is listed
    if listed:
        assert rows[0].detail == f"{frozen:,} bytes"


def test_the_frozen_section_never_makes_the_run_red() -> None:
    rows = audit.classify([_frozen_pr(4_900_000)], [], {10: issue(10, "", OPEN_MILESTONE)})
    assert kinds(rows) == ["frozen"] and audit.failing_rows(rows) == []


def test_an_unmeasured_pr_is_said_not_skipped() -> None:
    rows = audit.classify([_frozen_pr(None)], [], {10: issue(10, "", OPEN_MILESTONE)})
    assert kinds(rows) == ["frozen"] and "not measured" in rows[0].detail


def test_a_payload_without_a_measurement_gets_no_frozen_row() -> None:
    rows = audit.classify([pr(1, f"{ACCEPTED}\n\nFixes #10\n")], [],
                          {10: issue(10, "", OPEN_MILESTONE)})
    assert rows == []


def test_the_header_states_the_frozen_total_on_main() -> None:
    text = audit.render([], dt.date(2026, 9, 11), dt.date(2026, 9, 18),
                        {"prs": 3, "issues": 2, "frozen_main": [206, 21_345_678]})
    assert "Frozen test data on main: 206 files, 21,345,678 bytes" in text


def test_fetch_measures_each_merge_commit_with_the_gates_own_code(monkeypatch) -> None:
    """The bytes come from check_data_budget.frozen_bytes, the gate's rule-4 measure."""
    calls = []

    def fake_gh(args):
        if args[0] == "pr":
            record = pr(1, f"{ACCEPTED}\n\nhousekeeping\n")
            record["mergeCommit"] = {"oid": "abc123"}
            return [record, pr(2, f"{ACCEPTED}\n\nhousekeeping\n")]
        return []

    def fake_frozen_bytes(base, head, repo):
        calls.append((base, head))
        return 2_500_000

    monkeypatch.setattr(audit, "_gh", fake_gh)
    monkeypatch.setattr(audit._budget, "frozen_bytes", fake_frozen_bytes)
    monkeypatch.setattr(audit._budget, "frozen_on", lambda rev, repo: (7, 9_000_000))
    data = audit.fetch(7)
    assert calls == [("abc123^1", "abc123")]
    assert data["merged_prs"][0]["frozen_bytes"] == 2_500_000
    assert "frozen_bytes" not in data["merged_prs"][1]
    assert data["frozen_main"] == [7, 9_000_000]
    text, count = audit.report(data)
    assert "| #1 | a change | 2,500,000 bytes |" in text
    assert "Frozen test data on main: 7 files, 9,000,000 bytes" in text
    assert count == 0


def test_a_failed_measurement_is_recorded_as_none(monkeypatch) -> None:
    def fake_gh(args):
        if args[0] == "pr":
            record = pr(1, f"{ACCEPTED}\n\nhousekeeping\n")
            record["mergeCommit"] = {"oid": "0" * 40}
            return [record]
        return []

    monkeypatch.setattr(audit, "_gh", fake_gh)
    data = audit.fetch(7)
    assert data["merged_prs"][0]["frozen_bytes"] is None


def test_the_pr_listing_asks_github_for_the_merge_commit() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "number,title,body,mergedAt,url,labels,mergeCommit" in source


def test_the_audit_workflow_checks_out_full_history() -> None:
    """A shallow clone has no merge commits to measure."""
    workflow = (REPO / ".github" / "workflows" / "governance-audit.yml").read_text(encoding="utf-8")
    assert "fetch-depth: 0" in workflow


def test_an_issue_filed_through_a_form_produces_no_row() -> None:
    rows = audit.classify([], [issue(20, FORM_ISSUE)], {})
    assert rows == []


def test_a_labelled_issue_without_a_lane_section_produces_no_row() -> None:
    rows = audit.classify([], [issue(20, HAND_ISSUE)], {})
    assert rows == []


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
    rows = audit.classify([], [issue(22, body, labels=[])], {})
    assert kinds(rows) == ["unlabelled"]


def test_a_non_lane_label_does_not_satisfy_the_check() -> None:
    rows = audit.classify([], [issue(20, FORM_ISSUE, labels=["bug", "release"])], {})
    assert kinds(rows) == ["unlabelled"]


def test_a_labelled_form_issue_produces_no_row() -> None:
    rows = audit.classify([], [issue(20, FORM_ISSUE, labels=["lane:absorber"])], {})
    assert rows == []


def test_an_issue_without_a_section_or_label_has_only_an_unlabelled_row() -> None:
    """The lane label is the issue rule."""
    rows = audit.classify([], [issue(20, HAND_ISSUE, labels=[])], {})
    assert kinds(rows) == ["unlabelled"]


def test_a_hand_labelled_issue_produces_no_row() -> None:
    rows = audit.classify([], [issue(20, HAND_ISSUE, labels=["lane:plan"])], {})
    assert rows == []


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


def test_a_labelled_issue_with_a_blank_dropdown_produces_no_row() -> None:
    """A lane label satisfies the issue rule even when the dropdown is blank."""
    body = "### What is wrong\n\nx\n\n### Lane\n\n_No response_\n"
    rows = audit.classify([], [issue(21, body, labels=["lane:absorber"])], {})
    assert rows == []


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
        "review", "review-exceptions", "data-budget", "unlabelled", "milestone", "frozen", "unlinked",
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
    result = _run(_payload(opened_issues=[issue(20, HAND_ISSUE, labels=[])]), tmp_path)
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
        _payload(opened_issues=[issue(20, HAND_ISSUE, labels=[])]),
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
    text, count = audit.report(_payload(opened_issues=[issue(20, HAND_ISSUE, labels=[])]))
    assert count == 1 and "#20" in text


def classify_one(body: str, index: dict) -> list:
    """One merged PR, no opened issues."""
    return audit.classify([pr(1, body)], [], index)


@pytest.fixture
def batch_payload() -> dict:
    members = [2, 3]
    return _payload(
        merged_prs=[pr(1, "\n".join(
            f"Merge-train-member: #{number} {'a' * 40}" for number in members
        ))],
        member_index={str(number): {"body": ACCEPTED} for number in members},
    )


def test_a_batch_with_all_members_reviewed_has_no_failing_row(batch_payload) -> None:
    text, count = audit.report(batch_payload)
    assert count == 0
    assert "Merged PRs with no accepted review" not in text


def test_a_batch_with_an_unreviewed_member_names_that_member(batch_payload) -> None:
    number = next(iter(batch_payload["member_index"]))
    batch_payload["member_index"][number]["body"] = "No review recorded"
    text, count = audit.report(batch_payload)
    assert count == 1
    assert f"#{number}: no accepted `Review:` line" in text


@pytest.mark.parametrize("batch", [False, True])
def test_review_failures_and_permitted_exceptions_render_separately(batch, batch_payload) -> None:
    """Bound failures stay red; permitted claims have their own section.

    Expected table text comes from the fixture records, without classifier or
    rendering helpers. A mixed batch must preserve both member details under
    their respective headings while counting only one failing row.
    """
    members = list(batch_payload["member_index"].items())
    for (_, record), path in zip(members, ["rfx/core.py", "docs/guide.md"]):
        record.update(body=SKIPPED, diff={"files": [
            {"filename": path, "before": "value = 1", "after": "value = 2"},
        ]})
    if not batch:
        batch_payload["merged_prs"] = [
            {**pr(int(number), record["body"]), "diff": record["diff"]}
            for number, record in members
        ]

    expected_rows = []
    for index, (number, record) in enumerate(members):
        merged = batch_payload["merged_prs"][0 if batch else index]
        detail = (f"#{number}: " if batch else "") + record["body"]
        if index == 0:
            detail += ": " + record["diff"]["files"][0]["filename"]
        expected_rows.append(f"| #{merged['number']} | {merged['title']} | {detail} |")

    text, count = audit.report(batch_payload)
    sections = dict(section.split("\n", 1) for section in text.split("### ")[1:])
    review = sections["Merged PRs with no accepted review (1)"]
    exceptions = sections["Claimed review exceptions (informational) (1)"]
    assert expected_rows[0] in review
    assert expected_rows[1] not in review
    assert expected_rows[1] in exceptions
    assert expected_rows[0] not in exceptions
    informational = len(batch_payload["merged_prs"]) + 1
    assert f"**1 row to look at**, plus {informational} informational." in text
    assert count == 1
    assert "review-exceptions" in audit.INFORMATIONAL


@pytest.mark.parametrize("member_present", [False, True])
def test_an_unreadable_batch_member_fails(batch_payload, member_present) -> None:
    number = next(iter(batch_payload["member_index"]))
    if member_present:
        batch_payload["member_index"][number] = {}
    else:
        del batch_payload["member_index"][number]
    text, count = audit.report(batch_payload)
    assert count == 1
    assert f"#{number}: no accepted `Review:` line" in text


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "diff,expected,detail",
    [
        ({"files": [{"filename": "rfx/core.py", "before": '"old"\nvalue = 1',
                     "after": '"new"\nvalue = 1'}]}, 0, "(a) docs only"),
        ({"files": [{"filename": "tests/test_core.py", "before": "value = 1 # old",
                     "after": "value = 1 # new"}]}, 0, "(a) docs only"),
        ({"files": [{"filename": "rfx/core.py", "before": "value = 1",
                     "after": "value = 2"}]}, 1, "rfx/core.py"),
        ({"files": [{"filename": "rfx/config.json"}]}, 1, "rfx/config.json"),
        ({"files": [{"filename": "examples/demo.py", "before": "value = 1",
                     "after": "value = 2"}]}, 1, "examples/demo.py"),
        ({"files": [{"filename": "scripts/config.json"}]}, 0, "(a) docs only"),
        (None, 1, "diff not read"),
        ({"files": [{}]}, 1, "diff not read"),
        ({"files": [{"filename": "rfx/core.py", "before": "value = 1",
                     "after": "def broken("}]}, 1, "diff not read"),
        ({"files": [{"filename": "docs/guide.md"}]}, 0, "(a) docs only"),
        ({"files": []}, 0, "(a) docs only"),
    ],
    ids=["docstring", "comment", "code", "non-python-source", "example-code",
         "non-python-script", "unreadable",
         "malformed", "parse-error", "docs", "empty"],
)
def test_ast_exception_bound(diff, expected, detail, batch, batch_payload) -> None:
    """Expected outcomes are literal fixture values for both claim locations."""
    if batch:
        data = batch_payload
        number = next(iter(data["member_index"]))
        record = data["member_index"][number]
    else:
        record = pr(1, SKIPPED)
        data = _payload(merged_prs=[record])
    record.update(body=SKIPPED, diff=diff)
    text, count = audit.report(data)
    assert count == expected
    assert detail in text
    heading = ("Merged PRs with no accepted review (1)" if expected else
               "Claimed review exceptions (informational) (1)")
    assert heading in text
    if batch:
        assert f"#{number}: Review: skipped" in text


@pytest.mark.parametrize(
    "before,after,equal",
    [
        ('"old"\nvalue = 1', '"new"\nvalue = 1', True),
        ('class C:\n    "old"\n    value = 1', 'class C:\n    "new"\n    value = 1', True),
        ('def f():\n    "old"\n    return 1', 'def f():\n    "new"\n    return 1', True),
        ('async def f():\n    "old"\n    return 1', 'async def f():\n    return 1', True),
        ("value = 1 # old", "value = 1 # new", True),
        ("", '"documentation"', True),
        ('"documentation"', "", True),
        ("", "value = 1", False),
        ("value = 1", "", False),
        ("value = 1", "value = 2", False),
        ('value = 1\n"old"', 'value = 1\n"new"', False),
        ('if True:\n    "old"', 'if True:\n    "new"', False),
    ],
)
def test_source_comparison(before, after, equal) -> None:
    assert audit.same_code(before, after) is equal


def test_source_comparison_propagates_parse_errors() -> None:
    with pytest.raises(SyntaxError):
        audit.same_code("", "def broken(")


@pytest.mark.parametrize("batch", [False, True])
def test_maintainer_skip_is_informational(batch, batch_payload) -> None:
    body = "Review: skipped — (b) maintainer instruction"
    if batch:
        data = batch_payload
        next(iter(data["member_index"].values()))["body"] = body
    else:
        data = _payload(merged_prs=[pr(1, body)])
    text, count = audit.report(data)
    assert count == 0
    assert body in text


def test_milestone_row_is_informational() -> None:
    number = 10
    data = _payload(
        merged_prs=[pr(1, f"{ACCEPTED}\nCloses #{number}")],
        issue_index={str(number): {"milestone": None}},
    )
    text, count = audit.report(data)
    assert count == 0
    assert f"closes #{number}: no milestone" in text


def test_batch_input_replay_is_offline(batch_payload, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(audit.subprocess, "run", refuse)
    text, count = audit.report(json.loads(json.dumps(batch_payload)))
    assert count == 0
    assert "Merged PRs with no accepted review" not in text


def test_batch_and_ast_payload_exit_zero_offline(batch_payload, tmp_path) -> None:
    next(iter(batch_payload["member_index"].values())).update(
        body=SKIPPED, diff={"files": [{"filename": "rfx/core.py",
                                     "before": '"old"\nvalue = 1',
                                     "after": '"new"\nvalue = 1'}]},
    )
    result = _run(batch_payload, tmp_path, {"PATH": str(tmp_path)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert SKIPPED in result.stdout


def test_fetch_reads_closed_members_and_exception_refs(monkeypatch) -> None:
    numbers = [2, 3]
    head = "a" * 40
    base = "b" * 40
    batch = pr(1, "\n".join(f"Merge-train-member: #{n} {head}" for n in numbers))
    claimed = pr(4, SKIPPED)
    calls = []

    def listing(args):
        return [batch, claimed] if args[0] == "pr" else []

    def view(args):
        calls.append(args)
        if args[0] == "api":
            return {"merge_base_commit": {"sha": "c" * 40},
                    "files": [{"filename": "docs/guide.md", "status": "modified"}]}
        return {"body": SKIPPED, "baseRefOid": base, "headRefOid": "d" * 40}

    monkeypatch.setattr(audit, "_gh", listing)
    monkeypatch.setattr(audit, "_gh_one", view)
    monkeypatch.setattr(audit._budget, "frozen_on", lambda *args: (0, 0))
    data = audit.fetch(7)
    assert [call for call in calls if call[0] == "pr"] == [
        ["pr", "view", str(n), "--json", "body,baseRefOid,headRefOid"] for n in numbers
    ] + [["pr", "view", str(claimed["number"]), "--json", "baseRefOid,headRefOid"]]
    expected_files = [{"filename": "docs/guide.md", "status": "modified"}]
    for number in numbers:
        assert data["member_index"][str(number)]["diff"] == {
            "base": base, "head": head, "merge_base": "c" * 40, "files": expected_files,
        }
    assert claimed["diff"] == {
        "base": base, "head": "d" * 40, "merge_base": "c" * 40, "files": expected_files,
    }
    assert audit.report(json.loads(json.dumps(data)))[1] == 0


@pytest.mark.parametrize("stage", ["refs", "compare", "contents"])
def test_fetch_diff_failure_is_local_to_the_claim(monkeypatch, stage) -> None:
    head, base = "a" * 40, "b" * 40
    claims = [pr(1, SKIPPED), pr(2, SKIPPED)]

    def listing(args):
        return claims if args[0] == "pr" else []

    active = None

    def view(args):
        nonlocal active
        if args[0] == "pr":
            active = int(args[2])
            current = "refs"
        else:
            current = "compare" if "/compare/" in args[1] else "contents"
        if active == 1 and current == stage:
            raise subprocess.CalledProcessError(1, args)
        if current == "refs":
            return {"baseRefOid": base, "headRefOid": head}
        if current == "compare":
            return {"merge_base_commit": {"sha": "c" * 40}, "files": [
                {"filename": "rfx/core.py", "status": "modified"},
            ]}
        return {"type": "file", "encoding": "base64", "content": "dmFsdWUgPSAx"}

    monkeypatch.setattr(audit, "_gh", listing)
    monkeypatch.setattr(audit, "_gh_one", view)
    monkeypatch.setattr(audit._budget, "frozen_on", lambda *args: (0, 0))
    data = audit.fetch(7)
    assert claims[0]["diff"] is None
    assert claims[1]["diff"]["files"][0]["before"] == "value = 1"
    rows = audit.classify(data["merged_prs"], [], {})
    assert [(row.ref, row.kind) for row in rows if row.kind.startswith("review")] == [
        ("#1", "review"), ("#2", "review-exceptions"),
    ]
    assert next(row.detail for row in rows if row.kind == "review") == SKIPPED + ": diff not read"


@pytest.mark.parametrize("status,previous,path,before,after", [
    ("modified", "rfx/core.py", "rfx/core.py", "value = 1", "value = 1"),
    ("renamed", "rfx/old.py", "rfx/new.py", "value = 1", "value = 1"),
    ("added", "rfx/core.py", "rfx/core.py", "", "value = 1"),
    ("removed", "rfx/core.py", "rfx/core.py", "value = 1", ""),
])
def test_fetch_uses_merge_base_and_head_contents(monkeypatch, status, previous, path, before, after) -> None:
    base, head, merge_base = "a" * 40, "b" * 40, "c" * 40
    calls = []

    def view(args):
        calls.append(args)
        if "/compare/" in args[1]:
            return {"merge_base_commit": {"sha": merge_base}, "files": [
                {"filename": path, "previous_filename": previous, "status": status},
            ]}
        return {"type": "file", "encoding": "base64", "content": "dmFsdWUgPSAx"}

    monkeypatch.setattr(audit, "_gh_one", view)
    record = {"body": SKIPPED, "baseRefOid": base, "headRefOid": head}
    data = audit.fetch_diff(record)
    item = {"filename": path, "status": status, "before": before, "after": after}
    if previous != path:
        item["previous_filename"] = previous
    assert data == {"base": base, "head": head, "merge_base": merge_base, "files": [item]}
    expected = [["api", f"repos/{{owner}}/{{repo}}/compare/{base}...{head}"]]
    if status != "added":
        expected.append(["api", f"repos/{{owner}}/{{repo}}/contents/{previous}?ref={merge_base}"])
    if status != "removed":
        expected.append(["api", f"repos/{{owner}}/{{repo}}/contents/{path}?ref={head}"])
    assert calls == expected


def test_compare_file_cap_is_unreadable(monkeypatch) -> None:
    calls = []

    def view(args):
        calls.append(args)
        return {"merge_base_commit": {"sha": "c" * 40}, "files": [
            {"filename": "docs/guide.md", "status": "modified"},
        ] * 300}

    monkeypatch.setattr(audit, "_gh_one", view)
    record = {"body": SKIPPED, "baseRefOid": "a" * 40, "headRefOid": "b" * 40}
    assert audit.fetch_diff(record) is None
    assert len(calls) == 1
    assert audit.review_result({**record, "diff": {"files": [{}] * 300}}) == (
        "missing", SKIPPED + ": diff not read",
    )


def test_fetch_preserves_unreadable_members_as_failures(monkeypatch) -> None:
    number = 2
    batch = pr(1, f"Merge-train-member: #{number} {'a' * 40}")
    claimed = pr(3, SKIPPED)

    def listing(args):
        return [batch, claimed] if args[0] == "pr" else []

    def refuse(args):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(audit, "_gh", listing)
    monkeypatch.setattr(audit, "_gh_one", refuse)
    monkeypatch.setattr(audit._budget, "frozen_on", lambda *args: (0, 0))
    text, count = audit.report(audit.fetch(7))
    assert count == 2
    assert f"#{number}: no accepted `Review:` line" in text
    assert "diff not read" in text

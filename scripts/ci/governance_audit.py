#!/usr/bin/env python3
"""Weekly report on the intake rules that CI cannot enforce on a single PR.

Three of the repository's working rules are only checkable ACROSS pull requests
and issues, so no per-PR gate can see them:

* **Reviews.** ``scripts/ci/check_pr_body.py`` accepts ``Review: skipped - (a)``
  and ``(b)``, by design -- the rule has two exceptions. What it cannot see is
  how OFTEN the exception is claimed. An exception used every week is not an
  exception, and the only way to notice is to count them weekly.
* **Intake.** ``.github/ISSUE_TEMPLATE/`` makes "someone will DO or DECIDE
  something" a required field, which works only for issues filed THROUGH a
  form. An issue opened through the API or transferred from elsewhere skips it
  and has no ``### Lane`` section, so it is invisible to every lane query.
* **One campaign at a time.** The PI's standing instruction is that work runs
  one campaign at a time, and an open milestone is how that is visible. A merged
  PR that closes an issue outside every open milestone is work that happened off
  the current campaign; a merged PR that closes no issue at all may be fine
  (housekeeping) or may be an issue nobody filed.

None of this is a gate. A row here is a thing to look at, not a thing that was
forbidden -- the report is deliberately loud (a failing row exits 1, so the
scheduled run is RED and shows up in the Actions list) because a green weekly
summary nobody opens reports nothing. Do not make it a required check.

One section is INFORMATIONAL and never touches the exit code: merged PRs that
close no issue. Infrastructure and documentation work legitimately closes
nothing, and on the week this was written that section alone was 65 of 102
merged PRs -- a report that is red every week for a thing that is usually fine
is a report nobody opens by the third week.

Reads GitHub through ``gh``; classification is a pure function over the JSON so
the tests need no network. Stdlib only, Python 3.10.

Usage::

    python scripts/ci/governance_audit.py                 # last 7 days
    python scripts/ci/governance_audit.py --days 30
    python scripts/ci/governance_audit.py --input audit.json   # no network
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECK_PR_BODY = REPO_ROOT / "scripts" / "ci" / "check_pr_body.py"
_ISSUE_LANE = REPO_ROOT / "scripts" / "ci" / "issue_lane_label.py"

DEFAULT_DAYS = 7

#: GitHub's own closing keywords. A PR body saying "Fixes #123" links the issue;
#: "Related to #123" does not, and the difference is what decides whether the
#: issue's milestone applies to this PR.
LINK_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b[:\s]+#(\d+)",
    re.IGNORECASE,
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Imported, not reimplemented: the audit must agree with the gate about what an
# accepted `Review:` line looks like, or a body that passes CI shows up here as
# unreviewed every Monday.
_cpb = _load(_CHECK_PR_BODY, "rfx_check_pr_body")
_lane = _load(_ISSUE_LANE, "rfx_issue_lane_label")


class Row(NamedTuple):
    """One line of the report."""

    kind: str
    ref: str
    title: str
    detail: str


#: Heading, and the sentence under it, for each kind. Printed in this order.
#: Kinds that do NOT set the exit code. Listed and counted, never failed on.
INFORMATIONAL = ("unlinked",)

SECTIONS = (
    (
        "review",
        "Merged PRs with no accepted review, or a claimed exception",
        "`Review: skipped` is allowed for the two exceptions. An exception "
        "claimed often is not an exception.",
    ),
    (
        "form",
        "Issues opened outside the issue forms",
        "No `### Lane` section, so no lane label and no answer to \"who will "
        "DO or DECIDE something\".",
    ),
    (
        "milestone",
        "Merged PRs closing an issue outside every open milestone",
        "Work that happened off the current campaign, or an issue that should "
        "have been added to one.",
    ),
    (
        "unlinked",
        "Merged PRs closing no issue (informational)",
        "Housekeeping is fine here, and most of these are. A substantial change "
        "is an issue nobody filed. Does not make the run red.",
    ),
)


def _review_lines(body: str) -> List[str]:
    """Every visible line of *body* that opens with the review keyword."""
    return [line for line in _cpb.body_lines(body or "") if line.startswith("Review:")]


def review_state(body: str) -> tuple[str, str]:
    """``(state, detail)`` where state is ``ok``, ``skipped`` or ``missing``.

    Uses the gate's own parser, so an accepted body is accepted here.
    """
    problems = _cpb._check_review(_cpb.body_lines(body or ""))
    skipped = [line for line in _review_lines(body) if line.startswith("Review: skipped")]
    if problems:
        return "missing", "no accepted `Review:` line"
    if skipped:
        return "skipped", skipped[0].strip()
    return "ok", ""


def linked_issues(body: str) -> List[int]:
    """Issue numbers this PR body says it closes, in order, deduplicated."""
    seen: List[int] = []
    for match in LINK_RE.finditer(body or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def _in_open_milestone(issue: Optional[dict]) -> bool:
    if not issue:
        return False
    milestone = issue.get("milestone") or {}
    return bool(milestone) and milestone.get("state") == "open"


def classify(
    merged_prs: Sequence[dict],
    opened_issues: Sequence[dict],
    issue_index: Dict[int, dict],
) -> List[Row]:
    """The report rows. Pure: everything it needs is in the arguments.

    *issue_index* maps an issue number to its record, and must cover every
    number a merged PR links to. A record carrying ``is_pull_request`` is
    skipped: "Fixes #1042" pointing at a pull request is a cross-reference
    between PRs, and a PR has no milestone question to answer. A number MISSING
    from the index is reported rather than skipped -- a link the audit could not
    resolve at all is exactly the case worth a human look.
    """
    rows: List[Row] = []

    for pr in sorted(merged_prs, key=lambda item: item.get("number", 0)):
        ref = f"#{pr.get('number')}"
        title = pr.get("title", "")
        body = pr.get("body") or ""

        state, detail = review_state(body)
        if state != "ok":
            rows.append(Row("review", ref, title, detail or "no accepted `Review:` line"))

        links = linked_issues(body)
        if not links:
            rows.append(Row("unlinked", ref, title, "no Fixes/Closes link in the body"))
            continue
        for number in links:
            issue = issue_index.get(number)
            if issue is not None and issue.get("is_pull_request"):
                continue
            if _in_open_milestone(issue):
                continue
            if issue is None:
                where = "issue not found"
            elif not issue.get("milestone"):
                where = "no milestone"
            else:
                where = f"milestone {issue['milestone'].get('title')!r} is closed"
            rows.append(Row("milestone", ref, title, f"closes #{number}: {where}"))

    for issue in sorted(opened_issues, key=lambda item: item.get("number", 0)):
        # Asks only whether the heading is THERE. An unanswered dropdown, or two
        # sections that conflict, still means the form was used -- those are the
        # lane-label job's business, not intake's.
        if not _lane.has_lane_section(issue.get("body") or ""):
            rows.append(
                Row(
                    "form",
                    f"#{issue.get('number')}",
                    issue.get("title", ""),
                    "no `### Lane` section",
                )
            )

    return rows


def _escape(text: str) -> str:
    """Keep a title containing a pipe inside its own table cell."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def render(rows: Iterable[Row], since: dt.date, until: dt.date, counts: dict) -> str:
    """The markdown report, one table per section."""
    rows = list(rows)
    out = [
        "## Governance audit",
        "",
        f"{since.isoformat()} to {until.isoformat()}: "
        f"{counts.get('prs', 0)} merged PRs, {counts.get('issues', 0)} issues opened.",
        "",
    ]
    if not rows:
        out += ["Nothing to look at.", ""]
        return "\n".join(out)

    for kind, heading, blurb in SECTIONS:
        section = [row for row in rows if row.kind == kind]
        if not section:
            continue
        out += [
            f"### {heading} ({len(section)})",
            "",
            blurb,
            "",
            "| ref | title | detail |",
            "|---|---|---|",
        ]
        out += [
            f"| {row.ref} | {_escape(row.title)} | {_escape(row.detail)} |"
            for row in section
        ]
        out += [""]

    failing = failing_rows(rows)
    informational = len(rows) - len(failing)
    out += [
        f"**{len(failing)} rows to look at**, plus {informational} informational. "
        "This report is not a gate: a row is something to look at, not something "
        "that was forbidden. The run is red so the report gets opened.",
        "",
    ]
    return "\n".join(out)


def failing_rows(rows: Iterable[Row]) -> List[Row]:
    """The rows that set the exit code."""
    return [row for row in rows if row.kind not in INFORMATIONAL]


# --------------------------------------------------------------------------
# Fetching. Everything below talks to `gh`; nothing above it does.
# --------------------------------------------------------------------------


def _gh(args: Sequence[str]) -> list:
    out = subprocess.run(
        ["gh", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out or "[]")


def _gh_one(args: Sequence[str]) -> dict:
    """A single JSON object, for the endpoints that do not return a list."""
    out = subprocess.run(
        ["gh", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out or "{}")


#: `gh pr list --limit N` pages internally up to N. 1000 is the GitHub search
#: API's own ceiling, so a window that would return more cannot be reported
#: correctly by this tool at all -- which is why hitting it raises rather than
#: truncating.
FETCH_LIMIT = 1000


def _capped(rows: list, limit: int, what: str, window: str) -> list:
    """*rows*, or an error naming the cap it hit.

    A silent truncation is the worst outcome here: the report would print "200
    merged PRs" for a window holding 900 and every count under it would be
    wrong, with nothing saying so.
    """
    if len(rows) >= limit:
        raise RuntimeError(
            f"{what} for {window} returned {len(rows)} rows, at the {limit} cap. "
            f"The window is too wide to report accurately -- run it over fewer "
            f"days, or raise FETCH_LIMIT if the search API still allows it."
        )
    return rows


def fetch(days: int, limit: int = FETCH_LIMIT) -> dict:
    """Everything `classify` needs, as one JSON-serializable dict."""
    until = dt.datetime.now(dt.timezone.utc).date()
    since = until - dt.timedelta(days=days)
    window = f"{since.isoformat()}..{until.isoformat()}"

    prs = _capped(_gh([
        "pr", "list", "--state", "merged", "--limit", str(limit),
        "--search", f"merged:>={since.isoformat()}",
        "--json", "number,title,body,mergedAt,url",
    ]), limit, "merged PRs", window)
    issues = _capped(_gh([
        "issue", "list", "--state", "all", "--limit", str(limit),
        "--search", f"created:>={since.isoformat()}",
        "--json", "number,title,body,createdAt,url",
    ]), limit, "opened issues", window)

    wanted = {n for pr in prs for n in linked_issues(pr.get("body") or "")}
    index: Dict[int, dict] = {}
    for number in sorted(wanted):
        try:
            # `gh api .../issues/N`, not `gh issue view N`: the REST payload
            # carries a `pull_request` key for a number that is a PR, and
            # `gh issue view` on a PR SUCCEEDS with a null milestone, which read
            # as "an issue with no milestone" and put every cross-referenced PR
            # in the report.
            raw = _gh_one(["api", f"repos/{{owner}}/{{repo}}/issues/{number}"])
        except subprocess.CalledProcessError:
            # Not in this repository at all. Left out of the index, which
            # classify() reports as "issue not found".
            continue
        index[number] = {
            "number": raw.get("number"),
            "title": raw.get("title"),
            "milestone": raw.get("milestone"),
            "url": raw.get("html_url"),
            "is_pull_request": "pull_request" in raw,
        }

    return {
        "since": since.isoformat(),
        "until": until.isoformat(),
        "merged_prs": prs,
        "opened_issues": issues,
        "issue_index": {str(k): v for k, v in index.items()},
    }


def report(data: dict) -> tuple[str, int]:
    """``(markdown, failing row count)`` for a fetched or loaded payload.

    The count drives the exit code, so informational rows are left out of it.
    """
    index = {int(k): v for k, v in (data.get("issue_index") or {}).items()}
    rows = classify(
        data.get("merged_prs") or [], data.get("opened_issues") or [], index
    )
    text = render(
        rows,
        dt.date.fromisoformat(data["since"]),
        dt.date.fromisoformat(data["until"]),
        {
            "prs": len(data.get("merged_prs") or []),
            "issues": len(data.get("opened_issues") or []),
        },
    )
    return text, len(failing_rows(rows))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--input", help="read a previously fetched payload instead of calling gh"
    )
    parser.add_argument(
        "--dump", help="write the fetched payload here (for a later --input run)"
    )
    args = parser.parse_args(argv)

    if args.input:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        data = fetch(args.days)
    if args.dump:
        Path(args.dump).write_text(json.dumps(data, indent=2), encoding="utf-8")

    text, count = report(data)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())

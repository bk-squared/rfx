#!/usr/bin/env python3
"""Enforce the two provenance lines every rfx pull request body must carry.

Agents from several pods (Claude and Codex) push under one GitHub account, so
the PR body is the only place a reader can learn WHO reviewed a change and
WHICH owner lane it belongs to. The operating standard has required a
separate-instance review line in prose for weeks; an audit on 2026-09-18 found
35 of the 90 PRs merged since 2026-09-11 carried no review trace at all, and
none of the 17 open PRs carried the line. Prose is context, not enforcement --
so this is a CI check, which binds every author including the ones that never
read a CLAUDE.md.

Two lines, each exactly once, in the body text (not inside a code fence, not
inside an HTML comment):

    Lane: lane:<label>
    Review: <who> (separate instance) - ACCEPT
    Review: <who> (separate instance) - ACCEPT WITH CHANGES
    Review: skipped - <which exception, and why>

``Lane:`` must name a label that exists on the repository. The workflow passes
the live set in ``LANE_LABELS`` (newline-separated, from ``gh label list``); a
local run with no such env falls back to the set recorded in this file.

``Review: skipped`` covers the rule's only two exceptions: (a) an easily
reverted pure docs/comment change, (b) an explicit instruction from the PI to
skip. The free text has to name which one -- that is what makes the exception
auditable later.

A ``REJECT`` verdict never passes. Neither does a self-review: the line asserts
a SEPARATE instance did the reading, and the check can only enforce that the
claim is present and well-formed, not that it is true. Making the claim
explicit is what lets a later audit catch it.

Usage
-----
    PR_BODY="$(gh pr view 123 --json body --jq .body)" python scripts/ci/check_pr_body.py
    python scripts/ci/check_pr_body.py --file /tmp/body.md

Exit 0 silently on success. Exit 1 with the failing condition and the lines to
paste. Standard library only, Python 3.10.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Fallback owner lanes, used only when LANE_LABELS is absent or empty (a local
# run). Every open issue carries exactly one of these; the workflow reads the
# live set from the repository so a lane added after this file was written does
# not fail a PR. Keep in sync with `gh label list | grep '^lane:'`.
FALLBACK_LANE_LABELS = (
    "lane:crossval",
    "lane:waveguide-port",
    "lane:msl-port",
    "lane:coax-mixed-port",
    "lane:absorber",
    "lane:nu-mesh",
    "lane:distributed",
    "lane:ci-infra",
    "lane:examples-docs",
    "lane:plan",
)

# Em dash or a plain hyphen: Codex-written bodies use the hyphen, and failing a
# PR on a glyph teaches nothing.
_DASH = r"[—-]"

LANE_RE = re.compile(r"^Lane: (lane:[a-z0-9-]+)$")
REVIEW_ACCEPT_RE = re.compile(
    r"^Review: .+ \(separate instance\) " + _DASH + r" (ACCEPT|ACCEPT WITH CHANGES)$"
)
REVIEW_SKIPPED_RE = re.compile(r"^Review: skipped " + _DASH + r" .+$")

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def strip_comments(body: str) -> str:
    """Drop ``<!-- ... -->`` spans, keeping the line count so fences still pair."""
    return _COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), body)


def body_lines(body: str) -> list[str]:
    """Lines that count: outside HTML comments and outside fenced code blocks.

    The PR template's hint text lives in a comment and its examples live in a
    fence, so the template alone can never satisfy the check.
    """
    kept: list[str] = []
    fence: str | None = None
    for raw in strip_comments(body).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.rstrip()
        marker = _FENCE_RE.match(line)
        if marker is not None:
            token = marker.group(1)[0]
            if fence is None:
                fence = token
            elif token == fence:
                fence = None
            continue
        if fence is None:
            kept.append(line)
    return kept


def allowed_lanes(env: dict[str, str] | None = None) -> tuple[tuple[str, ...], str]:
    """The permitted lane labels and where they came from.

    Returns ``(labels, source)``. ``source`` is ``"LANE_LABELS"`` or
    ``"the fallback list in scripts/ci/check_pr_body.py"``; the failure message
    prints it, so an author who trips the check knows whether the set is live
    or hardcoded.
    """
    env = os.environ if env is None else env
    raw = env.get("LANE_LABELS", "")
    labels = tuple(item.strip() for item in raw.split("\n") if item.strip())
    if labels:
        return labels, "LANE_LABELS"
    return FALLBACK_LANE_LABELS, "the fallback list in scripts/ci/check_pr_body.py"


def check(body: str, env: dict[str, str] | None = None) -> list[str]:
    """Return the problems with *body*. Empty list means the body passes."""
    lanes, lane_source = allowed_lanes(env)
    lines = body_lines(body)

    problems: list[str] = []

    lane_matches = [m for m in (LANE_RE.match(line) for line in lines) if m]
    if len(lane_matches) == 0:
        problems.append(
            "no `Lane:` line. Exactly one line must read `Lane: lane:<label>` "
            "and name the owner lane this PR belongs to."
        )
    elif len(lane_matches) > 1:
        found = ", ".join(m.group(1) for m in lane_matches)
        problems.append(
            f"{len(lane_matches)} `Lane:` lines ({found}). A PR belongs to exactly "
            "one owner lane; keep one line and delete the rest."
        )
    else:
        label = lane_matches[0].group(1)
        if label not in lanes:
            problems.append(
                f"`Lane: {label}` is not a lane label on this repository. "
                f"Allowed ({lane_source}): {', '.join(sorted(lanes))}."
            )

    review_lines = [
        line
        for line in lines
        if REVIEW_ACCEPT_RE.match(line) or REVIEW_SKIPPED_RE.match(line)
    ]
    if len(review_lines) == 0:
        problems.append(
            "no well-formed `Review:` line. Every merged change is read by a "
            "SEPARATE agent instance, and the PR body is the only record of it. "
            "A `REJECT` verdict never passes -- fix the change, then get the "
            "re-read."
        )
    elif len(review_lines) > 1:
        problems.append(
            f"{len(review_lines)} `Review:` lines. Keep the one that reflects the "
            "verdict the PR is merging on."
        )

    return problems


def failure_report(problems: list[str], env: dict[str, str] | None = None) -> str:
    lanes, lane_source = allowed_lanes(env)
    example_lane = lanes[0] if lanes else "lane:ci-infra"
    out = ["PR body contract FAILED:"]
    out += [f"  - {problem}" for problem in problems]
    out += [
        "",
        "Add these two lines to the PR body, outside any code fence or HTML comment:",
        "",
        f"Lane: {example_lane}",
        "Review: <who reviewed> (separate instance) - ACCEPT",
        "",
        "Accepted `Review:` forms:",
        "  Review: <who reviewed> (separate instance) - ACCEPT",
        "  Review: <who reviewed> (separate instance) - ACCEPT WITH CHANGES",
        "  Review: skipped - <(a) easily reverted pure docs/comment change, or "
        "(b) PI instructed the skip>",
        "",
        f"Allowed lanes ({lane_source}):",
    ]
    out += [f"  {label}" for label in sorted(lanes)]
    out += ["", "Contract: scripts/ci/check_pr_body.py"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--file",
        help="read the PR body from this path instead of the PR_BODY environment variable",
    )
    args = parser.parse_args(argv)

    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            body = handle.read()
    elif "PR_BODY" in os.environ:
        body = os.environ["PR_BODY"]
    else:
        print(
            "PR body contract FAILED: no body to check. Set PR_BODY or pass --file PATH.",
            file=sys.stderr,
        )
        return 1

    problems = check(body)
    if problems:
        print(failure_report(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

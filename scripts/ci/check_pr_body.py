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

Two lines, each exactly once, starting flush at the left margin of the body
text (not inside a code fence, an HTML comment or a ``<pre>`` block):

    Lane: lane:<label>
    Review: <who> (separate instance) - ACCEPT
    Review: <who> (separate instance) - ACCEPT WITH CHANGES
    Review: skipped - (a) <why> | Review: skipped - (b) <why>

``Lane:`` must name a label that exists on the repository. The workflow passes
the live set in ``LANE_LABELS`` (newline-separated, from ``gh label list``); a
local run with no such env falls back to the set recorded in this file.

``<who>`` names the reviewing instance and may not contain ``<`` or ``>``: an
unedited placeholder is not a review record, and this file's own remediation
text would otherwise be a passing body.

``Review: skipped`` covers the rule's only two exceptions, and the line has to
say WHICH by opening the reason with the literal ``(a)`` or ``(b)``:

    (a) an easily reverted pure docs/comment change
    (b) an explicit instruction from the PI to skip

That is what makes the exception auditable later; a bare "skipped - x" records
nothing.

A ``REJECT`` verdict never passes. Neither does a self-review: the line asserts
a SEPARATE instance did the reading, and the check can only enforce that the
claim is present and well-formed, not that it is true. Making the claim
explicit is what lets a later audit catch it.

Usage
-----
    PR_BODY="$(gh pr view 123 --json body --jq .body)" python scripts/ci/check_pr_body.py
    python scripts/ci/check_pr_body.py --file /tmp/body.md

Exit 0 silently on success. Exit 1 with the failing condition and the lines to
fill in. Standard library only, Python 3.10.
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

# Em dash, en dash or a plain hyphen. Codex-written bodies use the hyphen and a
# Korean keyboard produces the en dash; failing a PR on a glyph teaches nothing,
# and the failure text names all three so a fourth one is a readable error
# rather than a silent "no Review: line".
_DASH = r"[—–-]"
DASH_NAMES = "an em dash (—), an en dash (–) or a plain hyphen (-)"

LANE_RE = re.compile(r"^Lane: (lane:[a-z0-9-]+)$")
REVIEW_ACCEPT_RE = re.compile(
    r"^Review: (?P<who>.+) \(separate instance\) "
    + _DASH
    + r" (?P<verdict>ACCEPT|ACCEPT WITH CHANGES)$"
)
REVIEW_SKIPPED_RE = re.compile(r"^Review: skipped " + _DASH + r" \((?:a|b)\) .+$")
# Right shape, wrong content: a REJECT verdict, a separator nobody accepts, a
# missing verdict. Worth naming, because "no `Review:` line" is a baffling
# thing to read on a body that visibly has one.
REVIEW_SHAPE_RE = re.compile(r"^Review: .*\(separate instance\)")
# An unedited template stub, as opposed to a name that merely CONTAINS angle
# brackets: `claude <noreply@anthropic.com>` is a reviewer, `<who read it>` is
# not. Only a who-field that is nothing but a bracketed span is a placeholder.
PLACEHOLDER_WHO_RE = re.compile(r"^<[^<>]*>$")

# Anything that is invisible once GitHub renders the body cannot carry the
# claim. Comments (terminated or not), fenced code, and <pre> blocks all
# qualify; an unterminated <!-- swallows the rest of the body on render, so it
# has to swallow the rest of the body here too.
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_OPEN_COMMENT_RE = re.compile(r"<!--.*\Z", re.DOTALL)
_PRE_RE = re.compile(r"<pre\b[^>]*>.*?(?:</pre>|\Z)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

# `<!--` inside an inline code span or a fenced block is literal text on
# render, not the start of a comment. Without this the gate eats the tail of
# any body that DESCRIBES the gate -- which is exactly how it first behaved on
# its own PR: one backticked `<!--` in a sentence about unterminated comments
# swallowed the Lane and Review lines 40 lines below it. Inline spans are
# single-line so a stray backtick cannot run away across the whole body.
_INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)([^\n]+?)(?<!`)\1(?!`)")

# A line that carries the keyword but not at the left margin: a list item, a
# block quote, or a bold run. The strict rule stands -- the line must start the
# claim -- but the failure text should say that instead of "no Lane: line".
_ADORNMENT_RE = re.compile(r"^[\s>*+\-]*")
_TRAILING_BOLD_RE = re.compile(r"[*_\s]+$")


def _mask_inline_code(text: str) -> str:
    """*text* with inline code spans blanked, character count preserved."""
    return _INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), text)


def _mask_fenced(text: str) -> str:
    """*text* with fenced code blocks blanked, character count preserved.

    Same reason as inline code, one level up: a `<!--` or a `<pre>` shown
    inside a fence is an EXAMPLE, not markup, and an unterminated one must not
    swallow the lines below the closing fence. Delimiters are masked too, so a
    fence whose content is an unclosed comment cannot leak either marker.
    """
    chars = list(text)
    offset = 0
    fence: str | None = None
    for line in text.split("\n"):
        marker = _FENCE_RE.match(line.rstrip())
        masked = fence is not None
        if marker is not None:
            token = marker.group(1)[0]
            if fence is None:
                fence, masked = token, True
            elif token == fence:
                fence, masked = None, True
        if masked:
            for index in range(offset, offset + len(line)):
                if chars[index] != "\n":
                    chars[index] = " "
        offset += len(line) + 1
    return "".join(chars)


def _mask_quoted(text: str) -> str:
    """*text* with everything that renders as literal code blanked out.

    Used only to LOCATE comments and ``<pre>`` blocks; the offsets have to keep
    lining up with the real text, so spans are overwritten rather than removed.
    """
    return _mask_inline_code(_mask_fenced(text))


def strip_invisible(body: str) -> str:
    """Blank out HTML comments and ``<pre>`` blocks, keeping the line count.

    Each stage looks for its markers in a copy where fenced blocks and inline
    code spans are masked, then blanks the matching range of the real text.
    Blanking with spaces rather than deleting keeps every later offset and
    every newline where it was, so the fence pairing that ``body_lines`` does
    on the RESULT is unaffected -- masking happens only in the throwaway copy
    used to find the markers.
    """
    text = body
    for pattern in (_COMMENT_RE, _PRE_RE, _OPEN_COMMENT_RE):
        masked = _mask_quoted(text)
        chars = list(text)
        for match in pattern.finditer(masked):
            for index in range(match.start(), match.end()):
                if chars[index] != "\n":
                    chars[index] = " "
        text = "".join(chars)
    return text


def body_lines(body: str) -> list[str]:
    """Lines that count: visible on render and outside fenced code blocks.

    The PR template's hint text lives in a comment and this file's remediation
    examples are placeholders, so neither can satisfy the check.
    """
    kept: list[str] = []
    fence: str | None = None
    normalized = strip_invisible(body).replace("\r\n", "\n").replace("\r", "\n")
    for raw in normalized.split("\n"):
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


def deadorn(line: str) -> str:
    """The line with list bullets, quote markers and bold runs removed.

    Only used to explain a failure. A body whose claim sits inside a bullet is
    still a failing body -- the two lines have to start their own line so a
    later audit can grep for them.
    """
    stripped = _ADORNMENT_RE.sub("", line)
    return _TRAILING_BOLD_RE.sub("", stripped)


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


def _adorned_hint(lines: list[str], keyword: str) -> str:
    """Explain a keyword that is present but indented, bulleted or quoted."""
    for line in lines:
        if line.startswith(keyword):
            continue
        if deadorn(line).startswith(keyword):
            return (
                f" Found `{keyword}` inside a list item, block quote or bold run "
                f"({line.strip()!r}) -- it must start the line, flush left."
            )
    return ""


def _check_lane(lines: list[str], lanes: tuple[str, ...], source: str) -> list[str]:
    matches = [m for m in (LANE_RE.match(line) for line in lines) if m]
    if len(matches) == 0:
        return [
            "no `Lane:` line. Exactly one line must read `Lane: lane:<label>` and "
            "name the owner lane this PR belongs to."
            + _adorned_hint(lines, "Lane:")
        ]
    if len(matches) > 1:
        found = ", ".join(m.group(1) for m in matches)
        return [
            f"{len(matches)} `Lane:` lines ({found}). A PR belongs to exactly one "
            "owner lane; keep one line and delete the rest."
        ]
    label = matches[0].group(1)
    if label not in lanes:
        return [
            f"`Lane: {label}` is not a lane label on this repository. "
            f"Allowed ({source}): {', '.join(sorted(lanes))}."
        ]
    return []


def _check_review(lines: list[str]) -> list[str]:
    valid: list[str] = []
    placeholder: list[str] = []
    bad_skip: list[str] = []
    malformed: list[str] = []

    for line in lines:
        accept = REVIEW_ACCEPT_RE.match(line)
        if accept is not None:
            who = accept.group("who")
            if PLACEHOLDER_WHO_RE.match(who.strip()):
                placeholder.append(line)
            else:
                valid.append(line)
            continue
        if REVIEW_SKIPPED_RE.match(line):
            valid.append(line)
        elif line.startswith("Review: skipped"):
            bad_skip.append(line)
        elif REVIEW_SHAPE_RE.match(line):
            malformed.append(line)

    if len(valid) > 1:
        return [
            f"{len(valid)} `Review:` lines. Keep the one that reflects the verdict "
            "the PR is merging on."
        ]
    if valid:
        return []

    problem = (
        "no well-formed `Review:` line. Every merged change is read by a SEPARATE "
        "agent instance, and the PR body is the only record of it. A `REJECT` "
        "verdict never passes -- fix the change, then get the re-read."
    )
    if placeholder:
        problem += (
            f" Found an unfilled placeholder ({placeholder[0].strip()!r}): replace "
            "the angle-bracketed text with the instance that actually read the diff."
        )
    if malformed:
        problem += (
            f" Found a `Review:` line whose verdict or separator is not accepted "
            f"({malformed[0].strip()!r}): the verdict must be ACCEPT or ACCEPT WITH "
            f"CHANGES, and the separator before it must be {DASH_NAMES}."
        )
    if bad_skip:
        problem += (
            f" Found a `Review: skipped` line that does not say which exception "
            f"({bad_skip[0].strip()!r}): open the reason with `(a)` for an easily "
            "reverted pure docs/comment change, or `(b)` for a skip the PI instructed."
        )
    problem += _adorned_hint(lines, "Review:")
    return [problem]


def check(body: str, env: dict[str, str] | None = None) -> list[str]:
    """Return the problems with *body*. Empty list means the body passes."""
    lanes, lane_source = allowed_lanes(env)
    lines = body_lines(body)
    return _check_lane(lines, lanes, lane_source) + _check_review(lines)


def failure_report(problems: list[str], env: dict[str, str] | None = None) -> str:
    """The remediation text.

    Every example below is a deliberate NON-example: `lane:<name>` is not a
    legal label and `<who read it>` is not a legal reviewer, so pasting this
    report into a PR body fails the check it came from. A contract test pins
    that.
    """
    lanes, lane_source = allowed_lanes(env)
    out = ["PR body contract FAILED:"]
    out += [f"  - {problem}" for problem in problems]
    out += [
        "",
        # No literal "<pre>" in this text: an unterminated one swallows the
        # rest of a body, which would make this report fail for the wrong
        # reason instead of on its placeholders.
        "Fill these two lines in, flush left, outside any code fence, HTML comment",
        "or preformatted block -- the placeholders below do not pass as written:",
        "",
        "Lane: lane:<name>",
        "Review: <who read it> (separate instance) - ACCEPT",
        "",
        f"The separator before the verdict may be {DASH_NAMES}.",
        "",
        "Accepted `Review:` forms:",
        "  Review: <who read it> (separate instance) - ACCEPT",
        "  Review: <who read it> (separate instance) - ACCEPT WITH CHANGES",
        "  Review: skipped - (a) <easily reverted pure docs/comment change>",
        "  Review: skipped - (b) <the PI instructed the skip>",
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

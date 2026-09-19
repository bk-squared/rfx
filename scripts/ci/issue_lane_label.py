#!/usr/bin/env python3
"""Read the Lane a GitHub issue form asked for, and say which labels to change.

Issue forms render every field as a markdown heading and the answer below it,
so the ``Lane`` dropdown of ``.github/ISSUE_TEMPLATE/{finding,decision}.yml``
reaches the API as::

    ### Lane

    lane:waveguide-port

That is the only machine-readable record of the lane an author chose, and a
label is what every other pod can query (``gh issue list -l lane:msl-port``).
``.github/workflows/issue-lane-label.yml`` runs this on ``issues: [opened,
edited]`` and applies what it prints.

Editing the dropdown has to MOVE the label, not add a second one, or an issue
ends up in two lanes and the query that tells a session what it owns starts
lying. So this reports both halves of the move: the label to add, and the
``lane:*`` labels to remove.

Inputs, all through the environment (issue text is author-controlled and never
reaches a shell):

``ISSUE_BODY``
    The rendered issue body.
``CURRENT_LABELS_JSON``
    The issue's labels, as GitHub's ``toJSON(...labels.*.name)`` array.
    Optional; absent means "no labels yet".
``LANE_LABELS``
    Newline-separated lane labels that exist on the repository, from
    ``gh label list``. Optional; absent falls back to the list in
    ``scripts/ci/check_pr_body.py`` so a local run still works.

An issue with no ``### Lane`` section was filed outside the forms. That is not
an error here -- the weekly governance audit is what reports it -- so this
prints nothing and exits 0, and the workflow step does nothing.

Neither is a body with MORE THAN ONE ``### Lane`` section. A body is text an
author controls, and pasting the heading into an earlier answer would otherwise
let the paste outrank the dropdown, which is silent and invisible. When the
sections disagree this labels nothing, prints the conflicting lines and exits 0
-- a body must never be able to turn the job red, because the job runs on every
edit of every issue.

A ``### Lane`` section naming something that is not a lane label is reported
the same way. The dropdown cannot produce it, so the body was hand-edited -- but
the author of an issue can edit their own body, and failing on it would put a
red check on somebody's issue for a word they typed. Nothing is hidden by that:
an issue this job could not label carries no ``lane:*`` label, and
``.github/workflows/governance-audit.yml`` lists exactly that, weekly.

**No body can make this job fail.** That is structural, not a rule to remember:
`Plan` has no field for a failure. The job runs on every edit of every issue, so
a red one would follow the issue around until somebody with write access noticed.
The only non-zero exit is having no ``ISSUE_BODY`` to read at all, which is a
broken workflow, not a broken issue.

Usage::

    ISSUE_BODY="$(gh issue view 1111 --json body --jq .body)" \\
        python scripts/ci/issue_lane_label.py

Standard library only, Python 3.10.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_PR_BODY_CHECK = REPO_ROOT / "scripts" / "ci" / "check_pr_body.py"

#: The field label in both issue forms. Change it in one place and this stops
#: matching, which `tests/contracts/test_governance_intake_contract.py` catches
#: by reading the forms rather than trusting this constant.
LANE_HEADING = "Lane"

LANE_PREFIX = "lane:"

# `###` exactly: issue forms always render a field label at heading level 3, and
# accepting `##` would let a prose heading in a hand-written body be read as an
# answer.
_HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")

# GitHub's placeholder for an unanswered optional field. A required dropdown
# cannot produce it, but a hand-edited body can.
_NO_RESPONSE = "_No response_"


def _fallback_lanes() -> Tuple[str, ...]:
    """The lane list `check_pr_body.py` already maintains.

    Imported rather than copied: two hardcoded lists drift, and the day they do
    the issue lane and the PR lane disagree about what exists.
    """
    spec = importlib.util.spec_from_file_location(
        "rfx_check_pr_body", _PR_BODY_CHECK
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {_PR_BODY_CHECK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.FALLBACK_LANE_LABELS)


def allowed_lanes(env: Optional[dict] = None) -> Tuple[str, ...]:
    """Lane labels this repository has, live if the workflow passed them."""
    env = os.environ if env is None else env
    raw = env.get("LANE_LABELS", "")
    labels = tuple(item.strip() for item in raw.split("\n") if item.strip())
    return labels or _fallback_lanes()


def parse_labels(raw: str) -> List[str]:
    """Label names from GitHub's ``toJSON`` array.

    Same reasoning as `check_changelog_fragment.parse_labels`: a comma-joined
    list cannot carry a label containing a comma.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError("CURRENT_LABELS_JSON must be a JSON array of label names")
    return [str(value) for value in values]


class Section(NamedTuple):
    """One ``### Lane`` heading found in a body."""

    line: int          #: 1-based line number of the heading
    value: Optional[str]  #: the answer under it, or None if it answered nothing


def lane_sections(body: str) -> List[Section]:
    """Every ``### Lane`` heading in *body*, with the answer under each.

    The answer is the first non-empty line after the heading. Issue forms put a
    blank line between the two and a single-select dropdown answers with one
    line, so nothing is gained by joining the paragraph -- and joining it would
    turn a hand-written note below the answer into part of the label.
    """
    found: List[Section] = []
    lines = (body or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        heading = _HEADING_RE.match(line)
        if heading is None or heading.group("title") != LANE_HEADING:
            continue
        if not line.startswith("### "):
            # A deeper or shallower heading is not what the form renders.
            continue
        value: Optional[str] = None
        for candidate in lines[index + 1:]:
            stripped = candidate.strip()
            if not stripped:
                continue
            if stripped != _NO_RESPONSE and _HEADING_RE.match(candidate) is None:
                value = stripped
            break
        found.append(Section(index + 1, value))
    return found


def has_lane_section(body: str) -> bool:
    """Whether *body* was rendered from a form at all.

    The weekly audit asks this and nothing more: an issue with no ``### Lane``
    heading anywhere was filed outside the forms.
    """
    return bool(lane_sections(body))


def lane_from_body(body: str) -> Optional[str]:
    """The single lane *body* asks for, or ``None``.

    ``None`` when nothing answered, and also when two sections both did: see
    `resolve_lane`, which says which of the two it was.
    """
    return resolve_lane(body)[0]


def resolve_lane(body: str) -> Tuple[Optional[str], List[str]]:
    """``(lane, conflicts)``.

    *conflicts* is empty unless more than one ``### Lane`` section answered.
    When it is not empty *lane* is ``None`` and nothing should be changed: the
    body is ambiguous, and guessing which section the author meant is how a
    pasted heading silently overrides a dropdown.
    """
    answered = [section for section in lane_sections(body) if section.value]
    if len(answered) <= 1:
        return (answered[0].value if answered else None), []
    lines = ", ".join(
        f"line {section.line} -> {section.value!r}" for section in answered
    )
    return None, [
        f"{len(answered)} `### {LANE_HEADING}` sections answer differently or "
        f"twice ({lines}). The form renders exactly one, so this body was edited "
        f"by hand. No label changed -- delete the extra section, or fix the "
        f"label by hand."
    ]


class Plan(NamedTuple):
    """What to do about one issue.

    There is deliberately no failure field. Everything this job can object to is
    something an issue author typed, and a red check on their issue helps nobody;
    the weekly audit's "no lane label" row is the net that catches every case
    where this job declined to act.
    """

    add: Optional[str]   #: the lane label to add, or None
    remove: List[str]    #: `lane:*` labels to take off
    notes: List[str]     #: printed on stderr; the exit code stays 0


def plan(body: str, current: Sequence[str], lanes: Sequence[str]) -> Plan:
    """What the labels should become for one issue.

    ``add`` is ``None`` when the body names no lane, when the sections conflict,
    when the value is not a lane, or when the label is already on the issue --
    an edit that did not change the dropdown must not churn the label and
    re-notify every watcher.
    """
    wanted, conflicts = resolve_lane(body)
    if conflicts:
        return Plan(None, [], conflicts)
    if wanted is None:
        return Plan(None, [], [])
    if wanted not in lanes:
        return Plan(None, [], [
            f"`### {LANE_HEADING}` names {wanted!r}, which is not a lane label on "
            f"this repository, so no label was applied. The dropdown cannot "
            f"produce it, so the body was hand-edited. Lanes: "
            f"{', '.join(sorted(lanes))}. The weekly governance audit lists this "
            f"issue until it carries a lane label."
        ])
    remove = sorted(
        label for label in set(current)
        if label.startswith(LANE_PREFIX) and label != wanted
    )
    add = None if wanted in current else wanted
    return Plan(add, remove, [])


def _write_output(add: Optional[str], remove: Sequence[str]) -> None:
    """Publish the plan as step outputs when running inside Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"lane={add or ''}\n")
        handle.write(f"remove={','.join(remove)}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--file",
        help="read the issue body from this path instead of the ISSUE_BODY variable",
    )
    args = parser.parse_args(argv)

    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    elif "ISSUE_BODY" in os.environ:
        body = os.environ["ISSUE_BODY"]
    else:
        print(
            "no issue body to read. Set ISSUE_BODY or pass --file PATH.",
            file=sys.stderr,
        )
        return 1

    current = parse_labels(os.environ.get("CURRENT_LABELS_JSON", ""))
    result = plan(body, current, allowed_lanes())
    for note in result.notes:
        print(note, file=sys.stderr)

    _write_output(result.add, result.remove)
    if result.add:
        print(f"add {result.add}")
    for label in result.remove:
        print(f"remove {label}")
    if not result.add and not result.remove:
        print("no lane change")
    return 0


if __name__ == "__main__":
    sys.exit(main())

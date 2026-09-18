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

A ``### Lane`` section naming something that is not a lane label IS an error:
the dropdown cannot produce it, so the body was hand-edited, and silently
labelling nothing would hide that.

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
from typing import List, Optional, Sequence, Tuple

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


def lane_from_body(body: str) -> Optional[str]:
    """The value under the ``### Lane`` heading, or ``None`` if there is none.

    The answer is the first non-empty line after the heading. Issue forms put a
    blank line between the two and a single-select dropdown answers with one
    line, so nothing is gained by joining the paragraph -- and joining it would
    turn a hand-written note below the answer into part of the label.
    """
    lines = (body or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        heading = _HEADING_RE.match(line)
        if heading is None or heading.group("title") != LANE_HEADING:
            continue
        if not line.startswith("### "):
            # A deeper or shallower heading is not what the form renders.
            continue
        for candidate in lines[index + 1:]:
            stripped = candidate.strip()
            if not stripped:
                continue
            if stripped == _NO_RESPONSE or _HEADING_RE.match(candidate):
                return None
            return stripped
    return None


def plan(body: str, current: Sequence[str], lanes: Sequence[str]) -> Tuple[Optional[str], List[str], List[str]]:
    """``(add, remove, problems)`` for one issue.

    ``add`` is ``None`` when the body names no lane, or when the label is
    already on the issue -- an edit that did not change the dropdown must not
    churn the label and re-notify every watcher.
    """
    wanted = lane_from_body(body)
    if wanted is None:
        return None, [], []
    if wanted not in lanes:
        return None, [], [
            f"`### {LANE_HEADING}` names {wanted!r}, which is not a lane label on "
            f"this repository. The dropdown cannot produce it, so the body was "
            f"hand-edited. Lanes: {', '.join(sorted(lanes))}."
        ]
    remove = sorted(
        label for label in set(current)
        if label.startswith(LANE_PREFIX) and label != wanted
    )
    add = None if wanted in current else wanted
    return add, remove, []


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
    add, remove, problems = plan(body, current, allowed_lanes())
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    _write_output(add, remove)
    if add:
        print(f"add {add}")
    for label in remove:
        print(f"remove {label}")
    if not add and not remove:
        print("no lane change")
    return 0


if __name__ == "__main__":
    sys.exit(main())

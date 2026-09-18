#!/usr/bin/env python3
"""CI gate for changelog fragments (issue #934).

Three rules, all mechanical:

1. ``CHANGELOG.md`` may only be edited by a PR labelled ``release``. Every
   other PR writes ``changelog.d/<number>.<type>.md`` instead, which is what
   removes the conflict class the file used to generate.
2. A PR that changes anything under ``rfx/`` must ADD a fragment. Docs, tests,
   scripts and records may add one; they are not required to.
3. Fragment names, heading lines and bodies must be well formed. The validator
   is the assembler's own, imported here, so CI and the release step cannot
   drift.

Inputs: ``--base``/``--head`` shas (the workflow passes the pull request's
``base.sha`` and ``head.sha``), ``PR_LABELS_JSON`` (the JSON array GitHub's
``toJSON(...labels.*.name)`` produces) and the optional ``PR_NUMBER`` used to
name the example file in the failure message.

Success is silent. Stdlib only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLER = REPO_ROOT / "scripts" / "changelog" / "assemble.py"

CHANGELOG = "CHANGELOG.md"
FRAGMENT_DIR = "changelog.d"
RELEASE_LABEL = "release"
GUARDED_PREFIX = "rfx/"

#: ``(status, path, present_at_head)``.
Entry = Tuple[str, str, bool]


def _load_assembler():
    """Import the assembler by path: ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("rfx_changelog_assemble", ASSEMBLER)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import the assembler from {ASSEMBLER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_labels(raw: str) -> List[str]:
    """Labels from GitHub's ``toJSON`` array.

    A comma-separated list cannot carry a label that contains a comma, and a
    label named ``pre,release`` would have satisfied a substring-free split on
    ``,`` -- the JSON form has no such reading.
    """
    raw = raw.strip()
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"PR_LABELS_JSON is not JSON: {exc}") from exc
    if not isinstance(values, list):
        raise ValueError("PR_LABELS_JSON must be a JSON array of label names")
    return [str(value) for value in values]


def changed_paths(base: str, head: str, repo: Path) -> List[Entry]:
    """``[(status, path, present_at_head), ...]`` for ``base...head``.

    A rename reports BOTH endpoints. Keeping only the destination let a PR move
    a file out of ``rfx/`` -- ``R100 rfx/mod.py docs/mod.py`` -- and owe no
    changelog entry for deleting a module.
    """
    out = subprocess.run(
        ["git", "diff", "--name-status", f"{base}...{head}"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    entries: List[Entry] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        status = fields[0][0]
        if status in ("R", "C") and len(fields) >= 3:
            source, destination = fields[1], fields[2]
            # A rename's source is gone at head; a copy's source is still there.
            entries.append((status, source, status == "C"))
            entries.append((status, destination, True))
        else:
            entries.append((status, fields[-1], status != "D"))
    return entries


def _is_fragment(path: str) -> bool:
    return (
        path.startswith(FRAGMENT_DIR + "/")
        and path.endswith(".md")
        and Path(path).name != "README.md"
    )


def _file_at(rev: str, path: str, repo: Path) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout


def check(base: str, head: str, labels: Sequence[str], repo: Path,
          pr_number: str = "") -> List[str]:
    """Return the failure messages; empty means the PR passes."""
    assemble = _load_assembler()
    try:
        entries = changed_paths(base, head, repo)
    except subprocess.CalledProcessError:
        return [
            f"cannot diff {base}...{head} in {repo}. Both commits must be "
            f"present -- check out with fetch-depth: 0."
        ]
    failures: List[str] = []

    example = f"{FRAGMENT_DIR}/{pr_number or '<PR number>'}.fixed.md"

    touched_changelog = any(path == CHANGELOG for _, path, _ in entries)
    if touched_changelog and RELEASE_LABEL not in labels:
        failures.append(
            f"{CHANGELOG} was modified without the '{RELEASE_LABEL}' label.\n"
            f"  Entries are written as fragments now, one file per PR, which is\n"
            f"  what keeps concurrent PRs from conflicting on this file.\n"
            f"  Revert the {CHANGELOG} edit and add {example} instead.\n"
            f"  Only a release PR (label '{RELEASE_LABEL}') assembles fragments\n"
            f"  into {CHANGELOG}, via scripts/changelog/assemble.py."
        )

    added_fragments = [path for status, path, _ in entries
                       if status == "A" and _is_fragment(path)]
    touched_package = any(path.startswith(GUARDED_PREFIX) for _, path, _ in entries)
    if touched_package and not added_fragments:
        failures.append(
            f"this PR changes {GUARDED_PREFIX} but adds no changelog fragment.\n"
            f"  Add {example} containing exactly the entry you would have pasted\n"
            f"  under '## [Unreleased]': a '### Fixed — <one sentence> "
            f"(#{pr_number or 'NNN'})'\n"
            f"  heading and its bullets. Types: added, breaking, changed,\n"
            f"  deprecated, fixed, removed. See {FRAGMENT_DIR}/README.md."
        )

    seen: set = set()
    for _, path, present in entries:
        if not present or path in seen:
            continue
        if not path.startswith(FRAGMENT_DIR + "/"):
            continue
        seen.add(path)
        name = Path(path).name
        if name == "README.md":
            continue
        try:
            text = _file_at(head, path, repo)
        except subprocess.CalledProcessError:  # pragma: no cover - defensive
            failures.append(f"{path}: cannot read the file at {head}")
            continue
        try:
            assemble.validate_fragment_text(name, text)
        except assemble.FragmentError as exc:
            failures.append(f"{exc}\n  See {FRAGMENT_DIR}/README.md for the shape.")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base", required=True, help="pull request base sha")
    parser.add_argument("--head", required=True, help="pull request head sha")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        labels = parse_labels(os.environ.get("PR_LABELS_JSON", ""))
    except ValueError as exc:
        print(f"changelog fragment check failed: {exc}", file=sys.stderr)
        return 1
    failures = check(args.base, args.head, labels, args.repo,
                     os.environ.get("PR_NUMBER", "").strip())
    if failures:
        print("changelog fragment check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

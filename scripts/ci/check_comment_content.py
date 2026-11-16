"""Check the content of Python comments on added working-tree lines."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tokenize


RULES = (
    ("date", re.compile(
        r"\b(?:19|20)\d\d[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])\b"
        r"|\b20\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\b"
    ), "a dated record in rfx-archive, or git history"),
    ("issue number", re.compile(r"(?<!&)#\d{3,}\b|(?i:\bPR) ?#\d+|/issues/\d+|/pull/\d+"),
     "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("predeclaration", re.compile(r"predeclar", re.I), "the predeclaration file"),
    ("move note", re.compile(r"\bmoved from\b", re.I), "git history (git log --follow)"),
    ("run id", re.compile(r"(?<!\d)(?<!\d\.)\d{10,}(?!\d)(?!\.\d)"), "the measurement record in rfx-archive"),
    ("quoted Korean", re.compile(
        r"[\"'“”‘’「」『』][^\"'“”‘’「」『』]*[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]"
        r"[^\"'“”‘’「」『』]*[\"'“”‘’「」『』]"
    ), "a named test or assertion states the rule; the rationale goes to rfx-archive"),
)
GUIDANCE = "a comment says what the code does, not where it came from"
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class CheckError(Exception):
    """The check could not inspect the requested changes."""


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="surrogateescape",
    )
    if check and result.returncode:
        raise CheckError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def ensure_commit(root: Path, revision: str) -> str:
    result = git(root, "rev-parse", "--verify", f"{revision}^{{commit}}", check=False)
    if result.returncode:
        git(root, "fetch", "--no-tags", "--depth=1", "origin", revision)
        result = git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    return result.stdout.strip()


def resolve_base(root: Path, explicit: str | None) -> str:
    if explicit:
        revision = ensure_commit(root, explicit)
        return git(root, "merge-base", revision, "HEAD").stdout.strip()
    header = git(root, "cat-file", "-p", "HEAD").stdout.split("\n\n", 1)[0]
    parents = [line.split()[1] for line in header.splitlines() if line.startswith("parent ")]
    if os.environ.get("GITHUB_EVENT_NAME") in {"pull_request", "pull_request_target", "merge_group"}:
        if not parents:
            raise CheckError("HEAD has no parent and no base could be resolved")
        return ensure_commit(root, parents[0])
    origin = git(root, "rev-parse", "--verify", "origin/main^{commit}", check=False)
    if origin.returncode == 0:
        base = git(root, "merge-base", "origin/main", "HEAD").stdout.strip()
        if base != git(root, "rev-parse", "HEAD").stdout.strip():
            return base
    if parents:
        return ensure_commit(root, parents[0])
    raise CheckError("HEAD has no parent and no base could be resolved")


def added_lines(root: Path, base: str) -> dict[str, set[int]]:
    patch = git(
        root, "diff", "--no-color", "--unified=0", "-M", "--diff-filter=AMR",
        base, "--", "*.py",
    ).stdout
    changed: dict[str, set[int]] = {}
    path = None
    line_number = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            path = None
            line_number = None
        elif line.startswith("+++ ") and line_number is None:
            name = line[4:]
            if name.startswith('"'):
                name = ast.literal_eval(name)
            else:
                name = name.removesuffix("\t")
            path = name.removeprefix("b/")
            changed.setdefault(path, set())
        elif match := HUNK.match(line):
            line_number = int(match.group(1))
        elif line_number is not None and path is not None:
            if line.startswith("+"):
                changed.setdefault(path, set()).add(line_number)
                line_number += 1
            elif line.startswith(" "):
                line_number += 1
    return changed


def check_comments(root: Path, changed: dict[str, set[int]]) -> list[str]:
    refused = []
    for path, lines in changed.items():
        try:
            with (root / path).open("rb") as source:
                for token in tokenize.tokenize(source.readline):
                    if token.type == tokenize.ERRORTOKEN and not token.string.isspace():
                        raise CheckError(f"{path}: cannot tokenize {token.string!r} at line {token.start[0]}")
                    if token.type != tokenize.COMMENT or token.start[0] not in lines:
                        continue
                    for rule, pattern, destination in RULES:
                        if pattern.search(token.string):
                            refused.append(
                                f"{path}:{token.start[0]}: {rule}: {token.string}"
                                f"  -> {destination}; {GUIDANCE}"
                            )
                            break
        except (OSError, SyntaxError, tokenize.TokenError, UnicodeError) as exc:
            raise CheckError(f"{path}: cannot tokenize: {exc}") from exc
    return refused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("BASE_SHA"))
    args = parser.parse_args(argv)
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.strip())
        base = resolve_base(root, args.base)
        changed = added_lines(root, base)
        refused = check_comments(root, changed)
        short = git(root, "rev-parse", "--short", base).stdout.strip()
    except (CheckError, OSError) as exc:
        print(f"comment content: {exc}", file=sys.stderr)
        return 2
    for diagnostic in refused:
        print(diagnostic)
    print(
        f"comment content: {len(refused)} refused on {sum(map(len, changed.values()))}"
        f" added lines (base {short})"
    )
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())

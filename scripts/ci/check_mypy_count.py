#!/usr/bin/env python3
"""Run the non-required mypy error-count ratchet with the repository config."""

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
ERROR_SUMMARY = re.compile(r"Found (\d+) errors? in \d+ files?(?: \((?:checked \d+ source files?|errors prevented further checking)\))?")
SUCCESS_SUMMARY = re.compile(r"Success: no issues found in \d+ source files?")
ERROR_LINE = re.compile(r"^(.+?):(?:\d+:(?:\d+:)?)?( error: .*)$")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True,
    )
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_base(rev: str | None) -> str:
    if rev is not None:
        return git("rev-parse", "--verify", f"{rev}^{{commit}}")
    headers = git("cat-file", "-p", "HEAD").split("\n\n", 1)[0].splitlines()
    parent = next((line.split()[1] for line in headers if line.startswith("parent ")), None)
    if parent is None:
        raise ValueError("HEAD has no first parent; specify --base <rev>")
    try:
        return git("rev-parse", "--verify", f"{parent}^{{commit}}")
    except ValueError:
        git("fetch", "--no-tags", "--depth=1", "origin", parent)
        return git("rev-parse", "--verify", f"{parent}^{{commit}}")


def run_mypy(tree: Path) -> tuple[int, list[tuple[str, str]]]:
    with tempfile.TemporaryDirectory(prefix="rfx-mypy-") as cache_dir:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "rfx", "--cache-dir", cache_dir], cwd=tree,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    lines = result.stdout.strip().splitlines()
    summary = lines[-1] if lines else ""
    errors = ERROR_SUMMARY.fullmatch(summary)
    success = SUCCESS_SUMMARY.fullmatch(summary)
    if result.returncode not in (0, 1) or not (errors or success):
        print(result.stdout, end="", file=sys.stderr)
        raise ValueError("no valid summary or unexpected exit code")

    count = int(errors[1]) if errors else 0
    error_lines = []
    for line in lines:
        match = ERROR_LINE.fullmatch(line)
        if match:
            error_lines.append((f"{match[1]}:{match[2]}", line))
    return count, error_lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="compare against this revision instead of HEAD's first parent")
    args = parser.parse_args(argv)
    try:
        base = resolve_base(args.base)
        short_sha = git("rev-parse", "--short", base)
        head_count, head_errors = run_mypy(REPO)
        with tempfile.TemporaryDirectory(prefix="rfx-mypy-base-") as temp_dir:
            tree = Path(temp_dir) / "tree"
            git("worktree", "add", "--detach", str(tree), base)
            try:
                base_count, base_errors = run_mypy(tree)
            finally:
                git("worktree", "remove", "--force", str(tree))
    except (OSError, ValueError) as error:
        print(f"mypy check failed: {error}", file=sys.stderr)
        return 2

    print(f"mypy errors: {head_count} (base {base_count} at {short_sha})")
    if head_count > base_count:
        remaining = Counter(key for key, _ in base_errors)
        for key, line in head_errors:
            if remaining[key]:
                remaining[key] -= 1
            else:
                print(line)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

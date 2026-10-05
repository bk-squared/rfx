#!/usr/bin/env python3
"""Enforce the down-only line-count baseline for rfx Python sources.

PI 2026-10-05 and the structure plan §11 put file structure in the 2.0 bar.
The only way to add lines to a listed file is to remove at least as many from
it. Split the file instead of raising its baseline; comparisons use numbers,
never source text. Counts are newline bytes, exactly like wc -l.

BASE_SHA/HEAD_SHA (or --base/--head) select Git snapshots. Without HEAD_SHA,
check the working tree, including uncommitted edits. An absent baseline at
BASE_SHA permits the initial adoption; an unreadable revision fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
BASELINE = "scripts/ci/file_size_baseline.json"


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
    ).stdout


def baseline(raw: bytes) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"cap", "files"}:
        raise ValueError("baseline must contain cap and files")
    if type(data["cap"]) is not int or data["cap"] < 0:
        raise ValueError("cap must be a nonnegative integer")
    if not isinstance(data["files"], dict):
        raise ValueError("files must be a mapping")
    for path, value in data["files"].items():
        if (not path.startswith("rfx/") or not path.endswith(".py")
                or ".." in Path(path).parts
                or type(value) is not int or value < 0):
            raise ValueError(f"invalid baseline entry: {path}")
    return data


def check(repo: Path, base: str = "", head: str = "") -> int:
    read = (lambda path: git(repo, "show", f"{head}:{path}")) if head else (
        lambda path: (repo / path).read_bytes()
    )
    current = baseline(read(BASELINE))
    failures = []
    if base:
        # ls-tree fails for a bad revision but returns empty for an absent file.
        if git(repo, "ls-tree", base, "--", BASELINE):
            previous = baseline(git(repo, "show", f"{base}:{BASELINE}"))
            if current["cap"] > previous["cap"]:
                failures.append(
                    f"{BASELINE}: cap baseline {previous['cap']}, count {current['cap']}"
                )
            for path, value in current["files"].items():
                old = previous["files"].get(path)
                if old is None or value > old:
                    failures.append(
                        f"{path}: baseline {old if old is not None else 'unlisted'}, "
                        f"count {value} in proposed baseline"
                    )
    if head:
        paths = [p.decode() for p in git(
            repo, "ls-tree", "-r", "--name-only", "-z", head, "--", "rfx/",
        ).split(b"\0") if p.endswith(b".py")]
    else:
        paths = [p.relative_to(repo).as_posix() for p in (repo / "rfx").rglob("*.py")]
    for path in sorted(paths):
        data = read(path)
        # Newlines as wc -l counts them, plus a bare CR, which Python also ends a line on.
        count = data.count(b"\n") + data.count(b"\r") - data.count(b"\r\n")
        limit = current["files"].get(path, current["cap"])
        if count > limit:
            failures.append(f"{path}: baseline {limit}, count {count}")
        elif path in current["files"] and count < limit:
            print(f"lower {path} to {count} in {BASELINE}", file=sys.stderr)
    for failure in failures:
        print(f"{failure}; split the file rather than raise the number.", file=sys.stderr)
    return int(bool(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=os.environ.get("BASE_SHA", ""))
    parser.add_argument("--head", default=os.environ.get("HEAD_SHA", ""))
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args(argv)
    try:
        return check(args.repo, args.base, args.head)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"file-size ratchet failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

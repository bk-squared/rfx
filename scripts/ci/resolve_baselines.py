#!/usr/bin/env python3
"""Resolve size/layer baseline conflicts after a stopped Git merge; never commit."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys

# Use the counters beside this script, including when imported by another caller.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_file_size_ratchet as size
import check_layer_structure as layer

BASELINES = (size.BASELINE, layer.BASELINE)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
    ).stdout


def git_path(repo: Path, name: str) -> Path:
    return repo / Path(git(repo, "rev-parse", "--git-path", name).decode().strip())


def eligible(repo: Path, explicit_theirs: bool) -> set[str] | None:
    if any(git_path(repo, name).exists()
           for name in ("REBASE_HEAD", "rebase-merge", "rebase-apply")):
        print("rebase in progress: abort it and use `git merge origin/main`, "
              "then run this script", file=sys.stderr)
        return None
    if not explicit_theirs and not git_path(repo, "MERGE_HEAD").exists():
        print("no merge in progress: supply --theirs or start a git merge", file=sys.stderr)
        return None
    paths = {p.decode() for p in git(
        repo, "diff", "--name-only", "--diff-filter=U", "-z",
    ).split(b"\0") if p}
    if not paths or not paths <= set(BASELINES):
        print("unmerged paths must be a nonempty subset of " + ", ".join(BASELINES)
              + (": " + ", ".join(sorted(paths)) if paths else " (none)"), file=sys.stderr)
        return None
    return paths


def encode(data: dict) -> bytes:
    return (json.dumps(data, indent=2) + "\n").encode("utf-8")


def regenerate_size(repo: Path, ours: dict, theirs: dict,
                    base_listed: set[str] | None = None) -> tuple[dict, list[str], list[str]]:
    """Merge ratchet ceilings; count sources to verify, without spending slack.

    An entry the --base side does not list is dropped (the checker refuses an entry its base
    does not list); its file is then governed by the merged cap."""
    cap = min(ours["cap"], theirs["cap"])
    listed = ours["files"].keys() | theirs["files"].keys()
    files, errors, notes = {}, [], []
    tracked = {p.decode() for p in git(repo, "ls-files", "-z", "--", "rfx/").split(b"\0")
               if p.endswith(b".py")}
    for path in sorted(listed | tracked):
        source = repo / path
        if not source.exists():
            continue
        count = size.code_lines(source.read_bytes())
        left = ours["files"].get(path, ours["cap"])
        right = theirs["files"].get(path, theirs["cap"])
        limit = min(left, right)
        keep = ((path in ours["files"] and path in theirs["files"]) or (
            path in listed and limit < cap
        )) and (base_listed is None or path in base_listed)
        if keep:
            files[path] = limit
            for side, data in (("ours", ours), ("theirs", theirs)):
                if path not in data["files"]:
                    notes.append(f"file_size_baseline.json {path}: kept entry omitted by {side} "
                                 f"at merged limit {limit}")
        else:
            # A removed entry at/above cap becomes unlisted, so the merged cap
            # governs it even when the other side had a larger explicit limit.
            limit = cap
            left, right = ours["cap"], theirs["cap"]
        if count > limit:
            errors.append(f"file_size_baseline.json {path}: merged count {count} exceeds "
                          f"min(ours {left}, theirs {right}) — real conflict")
    return {"counts": "code-lines", "cap": cap, "files": files}, errors, notes


def regenerate_layer(repo: Path, ours: dict, theirs: dict) -> tuple[dict, list[str]]:
    actual = layer.count(layer.Sources(repo, ""))["baseline"]
    errors = []
    for group, paths in actual.items():
        for path, rules in paths.items():
            for rule, names in rules.items():
                left = ours[group].get(path, {}).get(rule, [])
                right = theirs[group].get(path, {}).get(rule, [])
                extra = set(names) - (set(left) | set(right))
                if len(names) > min(len(left), len(right)) or extra:
                    detail = "; extra names " + ", ".join(sorted(extra)) if extra else ""
                    errors.append(f"layer_baseline.json {group}/{path}/{rule}: merged count "
                                  f"{len(names)}, min(ours {len(left)}, theirs {len(right)})"
                                  f"{detail} — real conflict")
    return actual, errors


def resolve(repo: Path, ours: str = "HEAD", theirs: str | None = None,
            base: str | None = None) -> int:
    repo = repo.resolve()
    try:
        conflicted = eligible(repo, theirs is not None)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"cannot inspect merge: {exc}", file=sys.stderr)
        return 2
    if conflicted is None:
        return 2
    theirs = "MERGE_HEAD" if theirs is None else theirs
    base = theirs if base is None else base
    # The final check must use the merged tree's own checkers: refuse a copy of this script
    # whose bundled checkers differ from the tree's (run the tree's own resolve_baselines.py).
    here = Path(__file__).resolve().parent
    for module in (size, layer):
        name = Path(module.__file__).name
        try:
            same = (here / name).read_bytes() == (repo / "scripts/ci" / name).read_bytes()
        except OSError:
            same = False
        if not same:
            print(f"scripts/ci/{name} in {repo} differs from the copy beside this script: "
                  "run the tree's own scripts/ci/resolve_baselines.py", file=sys.stderr)
            return 2
    try:
        size_sides = [size.baseline(git(repo, "show", f"{rev}:{size.BASELINE}"))
                      for rev in (ours, theirs)]
        layer_sides = [layer.baseline(git(repo, "show", f"{rev}:{layer.BASELINE}"))
                       for rev in (ours, theirs)]
        base_listed = set(size.baseline(git(repo, "show", f"{base}:{size.BASELINE}"))["files"])
        sized, errors, notes = regenerate_size(repo, *size_sides, base_listed=base_listed)
        layered, layer_errors = regenerate_layer(repo, *layer_sides)
        errors.extend(layer_errors)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        generated = {size.BASELINE: encode(sized), layer.BASELINE: encode(layered)}
        saved = {path: (repo / path).read_bytes() for path in BASELINES}
    except Exception as exc:
        print(f"baseline regeneration failed: {exc}", file=sys.stderr)
        return 1

    failed = True
    try:
        for path in sorted(conflicted):
            (repo / path).write_bytes(generated[path])
        failed = False
        for path, checker in ((size.BASELINE, size), (layer.BASELINE, layer)):
            out, err = io.StringIO(), io.StringIO()
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    status = checker.check(repo, base)
            except Exception as exc:
                status = 1
                print(f"{Path(path).name}: checker raised {exc}", file=err)
            if status:
                failed = True
                print(f"{Path(path).name}: checker failed against base {base}", file=sys.stderr)
                for message in (out.getvalue(), err.getvalue()):
                    if message:
                        print(message, end="", file=sys.stderr)
        if failed:
            return 1
        git(repo, "add", "--", *sorted(conflicted))
    except Exception as exc:
        failed = True
        print(f"baseline resolution failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # Both conflicted marker bytes and any unchanged baseline survive failure.
        if failed:
            for path, raw in saved.items():
                (repo / path).write_bytes(raw)

    for path in BASELINES:
        action = "regenerated and staged" if path in conflicted else "unchanged and checked"
        print(f"{Path(path).name}: {action}")
    if size.BASELINE in conflicted:
        for note in notes:
            print(note)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", default="HEAD")
    parser.add_argument("--theirs")
    parser.add_argument("--base")
    # Default: the repository of the CURRENT directory (the tree being merged), not the one
    # holding this script — a session may run another checkout's copy.
    parser.add_argument("--repo", type=Path, default=None)
    args = parser.parse_args(argv)
    repo = args.repo
    if repo is None:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if top.returncode:
            print("not inside a git working tree; pass --repo", file=sys.stderr)
            return 2
        repo = Path(top.stdout.strip())
    return resolve(repo, args.ours, args.theirs, args.base)


if __name__ == "__main__":
    raise SystemExit(main())

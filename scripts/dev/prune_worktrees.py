#!/usr/bin/env python3
"""Remove the git worktrees that agent sessions leave behind.

Dry run by default: every worktree of the repository containing the current
directory is classified and printed with the action that ``--apply`` would
take.  A worktree is removed only when its work is safe somewhere else --
merged into the main branch, or pushed to origin and idle.  A worktree with
uncommitted changes, an open pull request, or commits that exist nowhere but
that worktree is kept and listed.

Usage::

    python scripts/dev/prune_worktrees.py            # dry run, table only
    python scripts/dev/prune_worktrees.py --apply    # perform the removals
    python scripts/dev/prune_worktrees.py --no-gh    # offline classification

Python 3.10, standard library, ``git``, and optionally ``gh`` for pull-request
state.  ``git worktree list`` already knows every worktree of this repository
wherever it lives on disk, so no directory is ever scanned.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

CLASS_LOCKED = "locked"
CLASS_DIRTY = "dirty"
CLASS_DETACHED = "detached"
CLASS_DETACHED_ORPHAN = "detached-orphan"
CLASS_MERGED = "merged"
CLASS_OPEN_PR = "open-pr"
CLASS_UNPUSHED = "unpushed"
CLASS_PUSHED_IDLE = "pushed-idle"
CLASS_PUSHED_FRESH = "pushed-fresh"

# Local branches that this script never deletes, whatever their class.
PROTECTED_BRANCHES = {"main", "master"}


class GitError(RuntimeError):
    """A git invocation that the script cannot continue without failed."""


def git(cwd: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise GitError(
            "git {} failed in {} (exit {}):\n{}".format(
                " ".join(args), cwd, proc.returncode, proc.stderr.strip()
            )
        )
    return proc


def git_out(cwd: str, *args: str) -> str:
    return git(cwd, *args).stdout.strip()


def git_ok(cwd: str, *args: str) -> bool:
    return git(cwd, *args, check=False).returncode == 0


@dataclass
class Worktree:
    path: str
    head: str = ""
    branch: Optional[str] = None
    detached: bool = False
    bare: bool = False
    locked: bool = False
    prunable: bool = False


@dataclass
class Record:
    wt: Worktree
    klass: str
    action: str
    last_commit: Optional[dt.datetime] = None
    age_days: Optional[float] = None
    pr: str = "-"
    remove_worktree: bool = False
    delete_branch: bool = False
    push_branch: bool = False
    note: str = ""

    @property
    def branch_label(self) -> str:
        return self.wt.branch or "(detached)"


@dataclass
class PrLookup:
    """PR state for a branch, from ``gh``, cached and degrade-to-offline."""

    repo: str
    enabled: bool = True
    cache: dict = field(default_factory=dict)
    _warned: bool = False

    def __post_init__(self) -> None:
        if self.enabled and shutil.which("gh") is None:
            self._warn("gh is not on PATH")
            self.enabled = False

    def _warn(self, reason: str) -> None:
        if not self._warned:
            print(
                "warning: pull-request lookup unavailable ({}); only "
                "ancestor-of-main counts as merged".format(reason),
                file=sys.stderr,
            )
            self._warned = True

    def __call__(self, branch: str) -> Tuple[Optional[int], Optional[str]]:
        if not self.enabled:
            return (None, None)
        if branch in self.cache:
            return self.cache[branch]
        proc = subprocess.run(
            [
                "gh", "pr", "list",
                "--head", branch,
                "--state", "all",
                "--json", "number,state",
                "--limit", "20",
            ],
            cwd=self.repo,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()
            self._warn(tail[-1] if tail else "gh exited {}".format(proc.returncode))
            self.enabled = False
            return (None, None)
        try:
            prs = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            self._warn("gh returned output that is not JSON")
            self.enabled = False
            return (None, None)
        best: Tuple[Optional[int], Optional[str]] = (None, None)
        for wanted in ("OPEN", "MERGED", "CLOSED"):
            for pr in prs:
                if pr.get("state") == wanted:
                    best = (pr.get("number"), wanted)
                    break
            if best[1] is not None:
                break
        self.cache[branch] = best
        return best


def list_worktrees(cwd: str) -> list:
    out = git_out(cwd, "worktree", "list", "--porcelain")
    trees: list = []
    cur: Optional[Worktree] = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            cur = Worktree(path=line[len("worktree "):])
            trees.append(cur)
        elif cur is None:
            continue
        elif line.startswith("HEAD "):
            cur.head = line[len("HEAD "):]
        elif line.startswith("branch "):
            ref = line[len("branch "):]
            cur.branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        elif line == "detached":
            cur.detached = True
        elif line == "bare":
            cur.bare = True
        elif line == "locked" or line.startswith("locked "):
            cur.locked = True
        elif line == "prunable" or line.startswith("prunable "):
            cur.prunable = True
    return trees


def main_worktree_path(cwd: str) -> str:
    proc = git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir", check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        common = proc.stdout.strip()
    else:
        common = os.path.join(cwd, git_out(cwd, "rev-parse", "--git-common-dir"))
    common = os.path.realpath(common)
    if os.path.basename(common) == ".git":
        return os.path.dirname(common)
    return common


def is_dirty(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    proc = subprocess.run(
        ["git", "-C", path, "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Unreadable worktree: treat as dirty so it is never removed.
        return True
    return bool(proc.stdout.strip())


def reachable_from_a_ref(repo: str, rev: str) -> bool:
    """True when some branch or tag contains ``rev``.

    A detached worktree whose HEAD no ref contains holds commits that exist
    nowhere else; removing it would lose them.
    """
    proc = git(repo, "for-each-ref", "--contains", rev, "--count=1", "--format=%(refname)",
               check=False)
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def commit_time(repo: str, rev: str) -> Optional[dt.datetime]:
    proc = git(repo, "log", "-1", "--format=%cI", rev, check=False)
    stamp = proc.stdout.strip()
    if proc.returncode != 0 or not stamp:
        return None
    try:
        return dt.datetime.fromisoformat(stamp)
    except ValueError:
        return None


def branch_is_pushed(repo: str, branch: str, main_ref: str) -> bool:
    """True when the remote already has every commit of this local branch."""
    remote_ref = "refs/remotes/" + main_ref.split("/")[0] + "/" + branch
    if not git_ok(repo, "rev-parse", "--verify", "--quiet", remote_ref):
        return False
    return git_ok(repo, "merge-base", "--is-ancestor", "refs/heads/" + branch, remote_ref)


def classify(
    repo: str,
    wt: Worktree,
    *,
    main_ref: str,
    days: float,
    pr_lookup: Optional[PrLookup],
    archive: bool,
    now: dt.datetime,
) -> Record:
    last = commit_time(repo, wt.head) if wt.head else None
    age = None if last is None else (now - last).total_seconds() / 86400.0
    old = age is not None and age >= days
    rec = Record(wt=wt, klass="", action="", last_commit=last, age_days=age)

    if wt.locked:
        rec.klass = CLASS_LOCKED
        rec.action = "keep (locked)"
        return rec

    if is_dirty(wt.path):
        rec.klass = CLASS_DIRTY
        rec.action = "keep (uncommitted changes)"
        if wt.branch and not branch_is_pushed(repo, wt.branch, main_ref):
            rec.action = "KEEP (uncommitted changes; branch unpushed)"
        return rec

    if wt.branch is None:
        if wt.head and not reachable_from_a_ref(repo, wt.head):
            rec.klass = CLASS_DETACHED_ORPHAN
            rec.action = "KEEP -- detached HEAD on no ref; commits exist only here"
            return rec
        rec.klass = CLASS_DETACHED
        if old:
            rec.remove_worktree = True
            rec.action = "remove worktree"
        else:
            rec.action = "keep (detached, newer than {:g}d)".format(days)
        return rec

    branch = wt.branch
    pr_number, pr_state = pr_lookup(branch) if pr_lookup else (None, None)
    if pr_number is not None:
        rec.pr = "#{} {}".format(pr_number, pr_state)

    ancestor = git_ok(repo, "merge-base", "--is-ancestor", "refs/heads/" + branch, main_ref)
    if pr_state in ("MERGED", "CLOSED") or ancestor:
        rec.klass = CLASS_MERGED
        rec.remove_worktree = True
        if branch in PROTECTED_BRANCHES or branch == main_ref.split("/")[-1]:
            rec.action = "remove worktree, keep protected branch"
        else:
            rec.delete_branch = True
            rec.action = "remove worktree, delete local branch"
        return rec

    if pr_state == "OPEN":
        rec.klass = CLASS_OPEN_PR
        rec.action = "keep (open PR)"
        return rec

    remote = main_ref.split("/")[0]
    pushed = branch_is_pushed(repo, branch, main_ref)

    if not pushed:
        if archive:
            rec.push_branch = True
            if old:
                rec.klass = CLASS_PUSHED_IDLE
                rec.remove_worktree = True
                rec.action = "push to {}, then remove worktree".format(remote)
            else:
                rec.klass = CLASS_UNPUSHED
                rec.action = "push to {}, keep worktree (newer than {:g}d)".format(remote, days)
            return rec
        rec.klass = CLASS_UNPUSHED
        rec.action = "KEEP -- commits exist only here (use --archive to push)"
        return rec

    if old:
        rec.klass = CLASS_PUSHED_IDLE
        rec.remove_worktree = True
        rec.action = "remove worktree, keep branches"
        return rec

    rec.klass = CLASS_PUSHED_FRESH
    rec.action = "keep (pushed, newer than {:g}d)".format(days)
    return rec


def format_table(records: Sequence[Record]) -> str:
    header = ("PATH", "BRANCH", "CLASS", "LAST COMMIT", "AGE", "PR", "ACTION")
    rows = [header]
    for rec in records:
        rows.append(
            (
                rec.wt.path,
                rec.branch_label,
                rec.klass,
                rec.last_commit.strftime("%Y-%m-%d") if rec.last_commit else "-",
                "{:.0f}d".format(rec.age_days) if rec.age_days is not None else "-",
                rec.pr,
                rec.action,
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def apply_actions(repo: str, records: Sequence[Record], remote: str) -> int:
    failures = 0
    for rec in records:
        if rec.push_branch and rec.wt.branch:
            proc = git(repo, "push", remote, rec.wt.branch, check=False)
            if proc.returncode != 0:
                print(
                    "FAILED to push {}: {}".format(rec.wt.branch, proc.stderr.strip()),
                    file=sys.stderr,
                )
                failures += 1
                continue
            print("pushed {} to {}".format(rec.wt.branch, remote))
        if not rec.remove_worktree:
            continue
        proc = git(repo, "worktree", "remove", "--force", rec.wt.path, check=False)
        if proc.returncode != 0:
            print(
                "FAILED to remove worktree {}: {}".format(rec.wt.path, proc.stderr.strip()),
                file=sys.stderr,
            )
            failures += 1
            continue
        print("removed worktree {}".format(rec.wt.path))
        if rec.delete_branch and rec.wt.branch:
            proc = git(repo, "branch", "-D", rec.wt.branch, check=False)
            if proc.returncode != 0:
                print(
                    "FAILED to delete branch {}: {}".format(rec.wt.branch, proc.stderr.strip()),
                    file=sys.stderr,
                )
                failures += 1
            else:
                print("deleted local branch {}".format(rec.wt.branch))
    return failures


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and prune the git worktrees of this repository.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the planned actions (default: dry run, print only)",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=7.0,
        help="a pushed or detached worktree is idle once its last commit is "
             "this many days old (default: 7)",
    )
    parser.add_argument(
        "--no-gh",
        action="store_true",
        help="skip pull-request lookup; only ancestor-of-main counts as merged",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="push unpushed branches to the remote before treating them as idle",
    )
    parser.add_argument(
        "--main-ref",
        default="origin/main",
        help="remote ref that decides 'merged' (default: origin/main)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cwd = os.getcwd()
    try:
        repo = main_worktree_path(cwd)
    except GitError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not git_ok(repo, "rev-parse", "--verify", "--quiet", args.main_ref):
        print(
            "error: {} does not exist; pass --main-ref, or fetch first".format(args.main_ref),
            file=sys.stderr,
        )
        return 2
    remote = args.main_ref.split("/")[0]

    trees = list_worktrees(repo)
    prunable = [wt for wt in trees if wt.prunable]
    if args.apply:
        git(repo, "worktree", "prune", check=False)
        print("pruned {} worktree entries whose directory is gone".format(len(prunable)))
    else:
        print(
            "{} worktree entries point at a deleted directory and would be pruned".format(
                len(prunable)
            )
        )
    for wt in prunable:
        print("  prunable: {}".format(wt.path))

    pr_lookup = None if args.no_gh else PrLookup(repo=repo)
    now = dt.datetime.now(dt.timezone.utc)
    records = []
    for wt in trees:
        if wt.prunable or wt.bare:
            continue
        if os.path.realpath(wt.path) == os.path.realpath(repo):
            continue
        records.append(
            classify(
                repo,
                wt,
                main_ref=args.main_ref,
                days=args.days,
                pr_lookup=pr_lookup,
                archive=args.archive,
                now=now,
            )
        )

    print()
    if not records:
        print("no linked worktrees besides the main one")
        return 0
    print(format_table(records))
    print()
    counts: dict = {}
    for rec in records:
        counts[rec.klass] = counts.get(rec.klass, 0) + 1
    print(
        "{} linked worktrees: ".format(len(records))
        + ", ".join("{} {}".format(v, k) for k, v in sorted(counts.items()))
    )

    if not args.apply:
        planned = sum(1 for rec in records if rec.remove_worktree)
        print("dry run: {} would be removed. Re-run with --apply.".format(planned))
        return 0

    print()
    failures = apply_actions(repo, records, remote)
    git(repo, "worktree", "prune", check=False)
    if failures:
        print("{} action(s) failed".format(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

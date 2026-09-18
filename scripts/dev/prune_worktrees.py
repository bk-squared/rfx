#!/usr/bin/env python3
"""Remove the git worktrees that agent sessions leave behind.

Dry run by default: every worktree of the repository containing the current
directory is classified and printed with the action that ``--apply`` would
take.  Nothing is removed unless the work survives the removal -- on origin,
on the main branch, or on a local branch this script keeps.

Usage::

    python scripts/dev/prune_worktrees.py            # dry run, table only
    python scripts/dev/prune_worktrees.py --apply    # perform the removals
    python scripts/dev/prune_worktrees.py --no-gh    # offline, conservative

Python 3.10, standard library, ``git``, and optionally ``gh`` for pull-request
state.  ``git worktree list`` already knows every worktree of this repository
wherever it lives on disk, so no directory is ever scanned.

What is never removed, on any path:

* a branch whose tip is not on the remote and is not an ancestor of the main
  ref, and a branch whose tip no other ref contains -- the local branch stays
  even when its pull request is merged or closed;
* a worktree with uncommitted changes, or with ignored files that are not
  regenerable caches (``.omc/``, ``docs/research_notes/`` and the like);
* a detached HEAD that no ref contains;
* anything at all, once a ``gh`` lookup has failed: ``--apply`` aborts instead
  of acting on pull-request state it could not read.
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
from typing import List, Optional, Sequence

CLASS_LOCKED = "locked"
CLASS_DIRTY = "dirty"
CLASS_DIRTY_IGNORED = "dirty-ignored"
CLASS_DETACHED = "detached"
CLASS_DETACHED_ORPHAN = "detached-orphan"
CLASS_MERGED = "merged"
CLASS_OPEN_PR = "open-pr"
CLASS_UNPUSHED = "unpushed"
CLASS_PUSHED_IDLE = "pushed-idle"
CLASS_PUSHED_FRESH = "pushed-fresh"

# Local branches that this script never deletes, whatever their class.
PROTECTED_BRANCHES = {"main", "master"}

# Ignored paths that hold no work: a tool regenerates them from tracked files.
# Any OTHER ignored path makes the worktree `dirty-ignored` and keeps it, because
# this repository ignores working scratch that exists nowhere else -- `.omc/`,
# `docs/agent-memory/`, `docs/research_notes/`.
REGENERABLE_IGNORED = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
    ".ipynb_checkpoints",
    ".cache",
    ".tox",
    ".eggs",
    "htmlcov",
    ".coverage",
    "node_modules",
    ".venv",
    "venv",
}
REGENERABLE_SUFFIXES = (".pyc", ".pyo", ".egg-info", ".so")

PR_OK = "ok"
PR_OFF = "off"
PR_FAILED = "failed"


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
class PrInfo:
    """What is known about a branch's pull requests, and how well."""

    number: Optional[int] = None
    state: Optional[str] = None
    status: str = PR_OK
    checked: bool = False

    @property
    def trusted(self) -> bool:
        return self.status == PR_OK

    def label(self) -> str:
        """The PR column never lets "not looked at" read as "no PR exists"."""
        if self.status == PR_FAILED:
            return "?"
        if self.status == PR_OFF:
            return "off"
        if not self.checked:
            return "-"
        if self.number is None:
            return "none"
        return "#{} {}".format(self.number, self.state)


@dataclass
class Status:
    """What `git status` says about a worktree."""

    tracked: bool = False
    ignored: List[str] = field(default_factory=list)
    unreadable: bool = False


@dataclass
class Record:
    wt: Worktree
    klass: str = ""
    action: str = ""
    last_commit: Optional[dt.datetime] = None
    age_days: Optional[float] = None
    pr: PrInfo = field(default_factory=PrInfo)
    status: Status = field(default_factory=Status)
    remove_worktree: bool = False
    delete_branch: bool = False
    push_branch: bool = False

    @property
    def branch_label(self) -> str:
        return self.wt.branch or "(detached)"

    @property
    def unreliable(self) -> bool:
        return self.pr.status == PR_FAILED


@dataclass
class PrLookup:
    """Pull-request state for a branch, from ``gh``, cached.

    A failure is sticky and visible: every later branch reports ``PR_FAILED``
    too, and the caller refuses to act on what it could not read.
    """

    repo: str
    cache: dict = field(default_factory=dict)
    failed: bool = False
    reason: str = ""
    owner: Optional[str] = None

    def _fail(self, reason: str) -> PrInfo:
        if not self.failed:
            self.failed = True
            self.reason = reason
            print(
                "warning: pull-request lookup failed ({}); classification that "
                "depends on it is unreliable".format(reason),
                file=sys.stderr,
            )
        return PrInfo(status=PR_FAILED)

    def _gh(self, *args: str) -> Optional[object]:
        proc = subprocess.run(
            ["gh", *args], cwd=self.repo, capture_output=True, text=True
        )
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()
            self._fail(tail[-1] if tail else "gh exited {}".format(proc.returncode))
            return None
        try:
            return json.loads(proc.stdout or "null")
        except json.JSONDecodeError:
            self._fail("gh returned output that is not JSON")
            return None

    def _resolve_owner(self) -> bool:
        if self.owner is not None:
            return True
        if shutil.which("gh") is None:
            self._fail("gh is not on PATH")
            return False
        data = self._gh("repo", "view", "--json", "owner")
        if not isinstance(data, dict):
            if not self.failed:
                self._fail("gh repo view did not report an owner")
            return False
        login = (data.get("owner") or {}).get("login")
        if not login:
            self._fail("gh repo view did not report an owner")
            return False
        self.owner = login
        return True

    def __call__(self, branch: str) -> PrInfo:
        if self.failed:
            return PrInfo(status=PR_FAILED)
        if branch in self.cache:
            return self.cache[branch]
        if not self._resolve_owner():
            return PrInfo(status=PR_FAILED)
        data = self._gh(
            "pr", "list",
            "--head", branch,
            "--state", "all",
            "--json", "number,state,headRepositoryOwner",
            "--limit", "20",
        )
        if data is None:
            return PrInfo(status=PR_FAILED)
        # `--head` matches the branch NAME, so a fork's PR with the same head
        # name comes back too. Only this repository's own branches count.
        mine = [
            pr
            for pr in data
            if ((pr.get("headRepositoryOwner") or {}).get("login") == self.owner)
        ]
        info = PrInfo(checked=True)
        for wanted in ("OPEN", "MERGED", "CLOSED"):
            for pr in mine:
                if pr.get("state") == wanted:
                    info = PrInfo(number=pr.get("number"), state=wanted, checked=True)
                    break
            if info.state is not None:
                break
        self.cache[branch] = info
        return info


def list_worktrees(cwd: str) -> List[Worktree]:
    out = git_out(cwd, "worktree", "list", "--porcelain")
    trees: List[Worktree] = []
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


def is_regenerable(path: str) -> bool:
    """True when an ignored path is a cache a tool rebuilds from tracked files."""
    parts = [p for p in path.strip('"').rstrip("/").split("/") if p]
    for part in parts:
        if part in REGENERABLE_IGNORED or part.endswith(REGENERABLE_SUFFIXES):
            return True
    return False


def worktree_status(path: str) -> Status:
    """Uncommitted work in a worktree, including the ignored files git hides.

    `git worktree remove --force` deletes ignored files too, and this repository
    ignores directories that hold the only copy of working notes.
    """
    if not os.path.isdir(path):
        return Status(unreadable=True)
    proc = subprocess.run(
        ["git", "-C", path, "status", "--porcelain", "--ignored=matching"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Unreadable worktree: treat as dirty so it is never removed.
        return Status(unreadable=True)
    status = Status()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        if line.startswith("!! "):
            entry = line[3:]
            if not is_regenerable(entry):
                status.ignored.append(entry)
        else:
            status.tracked = True
    return status


def refs_containing(repo: str, rev: str, exclude: Sequence[str] = ()) -> List[str]:
    """Every ref whose history contains ``rev``, minus ``exclude``."""
    proc = git(repo, "for-each-ref", "--contains", rev, "--format=%(refname)", check=False)
    if proc.returncode != 0:
        return []
    return [ref for ref in proc.stdout.split() if ref not in exclude]


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
    pr_lookup,
    archive: bool,
    remove_ignored: bool,
    now: dt.datetime,
) -> Record:
    last = commit_time(repo, wt.head) if wt.head else None
    age = None if last is None else (now - last).total_seconds() / 86400.0
    old = age is not None and age >= days
    rec = Record(wt=wt, last_commit=last, age_days=age)

    if wt.locked:
        rec.klass = CLASS_LOCKED
        rec.action = "keep (locked)"
        return rec

    rec.status = worktree_status(wt.path)
    if rec.status.unreadable or rec.status.tracked:
        rec.klass = CLASS_DIRTY
        rec.action = "keep (uncommitted changes)"
        if rec.status.unreadable:
            rec.action = "keep (status unreadable)"
        elif wt.branch and not branch_is_pushed(repo, wt.branch, main_ref):
            rec.action = "KEEP (uncommitted changes; branch unpushed)"
        return rec

    if rec.status.ignored and not remove_ignored:
        rec.klass = CLASS_DIRTY_IGNORED
        rec.action = "keep (ignored files: {})".format(_sample(rec.status.ignored))
        return rec

    if wt.branch is None:
        if wt.head and not refs_containing(repo, wt.head):
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
    rec.pr = pr_lookup(branch) if pr_lookup else PrInfo(status=PR_OFF, checked=True)

    ancestor = git_ok(repo, "merge-base", "--is-ancestor", "refs/heads/" + branch, main_ref)
    pushed = branch_is_pushed(repo, branch, main_ref)
    remote = main_ref.split("/")[0]
    # The branch ref may hold the only copy of a commit even when its PR merged:
    # rfx squash-merges, so a merged branch is never an ancestor of main, and a
    # commit made after the merge is on no other ref.
    elsewhere = bool(refs_containing(repo, "refs/heads/" + branch, ("refs/heads/" + branch,)))
    protected = branch in PROTECTED_BRANCHES or branch == main_ref.split("/")[-1]
    safe_to_delete = (pushed or ancestor) and elsewhere and not protected

    def merged_record(reason: str) -> Record:
        rec.klass = CLASS_MERGED
        rec.remove_worktree = True
        if safe_to_delete:
            rec.delete_branch = True
            rec.action = "remove worktree, delete local branch ({})".format(reason)
        elif protected:
            rec.action = "remove worktree, keep protected branch"
        else:
            rec.action = "remove worktree, KEEP branch (commits not on {})".format(remote)
        return rec

    if rec.pr.state == "MERGED" or ancestor:
        return _finish(merged_record("merged"))

    if rec.pr.state == "CLOSED":
        if pushed or ancestor:
            return _finish(merged_record("PR closed"))
        # A closed PR whose commits reached no ref of ours is abandoned work that
        # exists only here. Keep all of it.
        rec.klass = CLASS_UNPUSHED
        rec.action = "KEEP -- PR closed and commits are not on {}".format(remote)
        return _finish(rec)

    if rec.pr.state == "OPEN":
        rec.klass = CLASS_OPEN_PR
        rec.action = "keep (open PR)"
        return _finish(rec)

    if not pushed:
        if archive:
            rec.push_branch = True
            if old and rec.pr.trusted:
                rec.klass = CLASS_PUSHED_IDLE
                rec.remove_worktree = True
                rec.action = "push to {}, then remove worktree".format(remote)
            elif old:
                rec.klass = CLASS_PUSHED_IDLE
                rec.action = "push to {}, keep (no PR state)".format(remote)
            else:
                rec.klass = CLASS_UNPUSHED
                rec.action = "push to {}, keep worktree (newer than {:g}d)".format(remote, days)
            return _finish(rec)
        rec.klass = CLASS_UNPUSHED
        rec.action = "KEEP -- commits exist only here (use --archive to push)"
        return _finish(rec)

    if old:
        rec.klass = CLASS_PUSHED_IDLE
        if rec.pr.trusted:
            rec.remove_worktree = True
            rec.action = "remove worktree, keep branches"
        else:
            rec.action = "keep (no PR state: an open PR cannot be ruled out)"
        return _finish(rec)

    rec.klass = CLASS_PUSHED_FRESH
    rec.action = "keep (pushed, newer than {:g}d)".format(days)
    return _finish(rec)


def _finish(rec: Record) -> Record:
    """A branch whose PR state could not be read is acted on by nobody."""
    if rec.unreliable:
        rec.remove_worktree = False
        rec.delete_branch = False
        rec.push_branch = False
        rec.action = "keep -- unreliable: gh failed, PR state unknown"
    return rec


def _sample(paths: Sequence[str], limit: int = 2) -> str:
    shown = ", ".join(paths[:limit])
    extra = len(paths) - limit
    return shown + (" +{} more".format(extra) if extra > 0 else "")


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
                rec.pr.label(),
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
        elif rec.wt.branch:
            print("kept local branch {} ({})".format(rec.wt.branch, rec.action))
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
        help="skip pull-request lookup; only an ancestor of the main ref counts "
             "as merged and no pushed-idle worktree is removed",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="push unpushed branches to the remote before treating them as idle",
    )
    parser.add_argument(
        "--remove-ignored",
        action="store_true",
        help="allow removing a worktree that holds ignored files which are not "
             "regenerable caches (they are deleted with it)",
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

    if args.no_gh:
        print(
            "--no-gh: pull-request state is unknown. Only a branch that is an "
            "ancestor of {} counts as merged, and no pushed-idle worktree will "
            "be removed.".format(args.main_ref)
        )

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
                remove_ignored=args.remove_ignored,
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

    unreliable = [rec for rec in records if rec.unreliable]
    if unreliable:
        print(
            "{} row(s) marked unreliable: gh failed ({}). Their pull-request "
            "state is unknown.".format(
                len(unreliable), pr_lookup.reason if pr_lookup else "unknown"
            ),
            file=sys.stderr,
        )

    if not args.apply:
        planned = sum(1 for rec in records if rec.remove_worktree)
        print("dry run: {} would be removed. Re-run with --apply.".format(planned))
        return 0

    if unreliable or (pr_lookup is not None and pr_lookup.failed):
        print(
            "aborting: --apply will not act on pull-request state it could not "
            "read. Fix gh (or pass --no-gh for the conservative offline rules). "
            "Nothing was removed.",
            file=sys.stderr,
        )
        return 2

    print()
    failures = apply_actions(repo, records, remote)
    git(repo, "worktree", "prune", check=False)
    if failures:
        print("{} action(s) failed".format(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Contract: the worktree pruner removes only work that is safe elsewhere.

Builds a throwaway bare "origin" plus a clone in ``tmp_path`` and drives
``scripts/dev/prune_worktrees.py`` over it with ``--no-gh``, so the test needs
no network and no GitHub token.  It pins the classification a reviewer cares
about: an unpushed branch and a dirty worktree survive ``--apply``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "prune_worktrees.py"

OLD_DATE = "2020-01-02T03:04:05+00:00"


def _env(tmp_path: Path) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(tmp_path),
            "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
            "GIT_CONFIG_SYSTEM": str(tmp_path / "gitsystem"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Prune Test",
            "GIT_AUTHOR_EMAIL": "prune@example.invalid",
            "GIT_COMMITTER_NAME": "Prune Test",
            "GIT_COMMITTER_EMAIL": "prune@example.invalid",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    for stale in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(stale, None)
    return env


def _git(cwd, *args, env, date=None) -> subprocess.CompletedProcess:
    run_env = dict(env)
    if date is not None:
        run_env["GIT_AUTHOR_DATE"] = date
        run_env["GIT_COMMITTER_DATE"] = date
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), env=run_env, capture_output=True, text=True
    )
    assert proc.returncode == 0, "git {} failed:\n{}".format(" ".join(args), proc.stderr)
    return proc


def _commit(cwd, name, env, date=None) -> None:
    (Path(cwd) / name).write_text(name + "\n", encoding="utf-8")
    _git(cwd, "add", name, env=env)
    _git(cwd, "commit", "-m", "add " + name, env=env, date=date)


def _run_script(clone, env, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--no-gh", *args],
        cwd=str(clone),
        env=env,
        capture_output=True,
        text=True,
    )


def _table(stdout: str) -> dict:
    """Parse the printed table into {path: {'branch':…, 'class':…, 'action':…}}."""
    rows = {}
    header_seen = False
    for line in stdout.splitlines():
        if line.startswith("PATH "):
            header_seen = True
            continue
        if not header_seen or not line.strip():
            continue
        if not line.startswith("/"):
            break
        cells = re.split(r"\s{2,}", line.strip())
        assert len(cells) == 7, "unexpected table row: {!r}".format(line)
        rows[cells[0]] = {
            "branch": cells[1],
            "class": cells[2],
            "last_commit": cells[3],
            "age": cells[4],
            "pr": cells[5],
            "action": cells[6],
        }
    return rows


@pytest.fixture()
def repo(tmp_path):
    """A clone with four linked worktrees: merged, pushed-idle, unpushed, dirty."""
    env = _env(tmp_path)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin), env=env)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone), env=env)
    _commit(clone, "base.txt", env)
    _git(clone, "push", "-u", "origin", "main", env=env)

    wt = tmp_path / "wt"
    wt.mkdir()

    # (a) merged: branch is an ancestor of origin/main.
    _git(clone, "branch", "feat/merged", "main", env=env)
    _git(clone, "worktree", "add", str(wt / "merged"), "feat/merged", env=env)

    # (b) pushed-idle: on origin, last commit backdated well past --days.
    _git(clone, "worktree", "add", "-b", "feat/pushed", str(wt / "pushed"), "main", env=env)
    _commit(wt / "pushed", "pushed.txt", env, date=OLD_DATE)
    _git(clone, "push", "-u", "origin", "feat/pushed", env=env)

    # (c) unpushed: commits exist nowhere but this worktree.
    _git(clone, "worktree", "add", "-b", "feat/unpushed", str(wt / "unpushed"), "main", env=env)
    _commit(wt / "unpushed", "unpushed.txt", env, date=OLD_DATE)

    # (d) dirty: would be pushed-idle, but has an uncommitted file.
    _git(clone, "worktree", "add", "-b", "feat/dirty", str(wt / "dirty"), "main", env=env)
    _commit(wt / "dirty", "dirty.txt", env, date=OLD_DATE)
    _git(clone, "push", "-u", "origin", "feat/dirty", env=env)
    (wt / "dirty" / "scratch.txt").write_text("work in progress\n", encoding="utf-8")

    return {
        "env": env,
        "clone": clone,
        "origin": origin,
        "paths": {
            "merged": str((wt / "merged").resolve()),
            "pushed": str((wt / "pushed").resolve()),
            "unpushed": str((wt / "unpushed").resolve()),
            "dirty": str((wt / "dirty").resolve()),
        },
    }


def test_dry_run_classifies_each_worktree(repo):
    proc = _run_script(repo["clone"], repo["env"])
    assert proc.returncode == 0, proc.stderr
    rows = _table(proc.stdout)
    paths = repo["paths"]
    assert set(rows) == set(paths.values()), proc.stdout

    assert rows[paths["merged"]]["class"] == "merged"
    assert rows[paths["merged"]]["branch"] == "feat/merged"
    assert "delete local branch" in rows[paths["merged"]]["action"]

    assert rows[paths["pushed"]]["class"] == "pushed-idle"
    assert "keep branches" in rows[paths["pushed"]]["action"]

    assert rows[paths["unpushed"]]["class"] == "unpushed"
    assert rows[paths["unpushed"]]["action"].startswith("KEEP")

    assert rows[paths["dirty"]]["class"] == "dirty"

    # A dry run touches nothing.
    for path in paths.values():
        assert os.path.isdir(path)


def test_dry_run_leaves_a_recent_pushed_branch_alone(repo):
    """The idle window is what removes a pushed branch, not the push itself."""
    proc = _run_script(repo["clone"], repo["env"], "--days", "100000")
    assert proc.returncode == 0, proc.stderr
    rows = _table(proc.stdout)
    assert rows[repo["paths"]["pushed"]]["class"] == "pushed-fresh"
    assert "keep (pushed" in rows[repo["paths"]["pushed"]]["action"]
    # Only the merged worktree stays removable: merge does not wait out a window.
    assert "dry run: 1 would be removed" in proc.stdout


def test_apply_removes_merged_and_idle_only(repo):
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    proc = _run_script(clone, env, "--apply")
    assert proc.returncode == 0, proc.stderr + proc.stdout

    assert not os.path.exists(paths["merged"])
    assert not os.path.exists(paths["pushed"])
    assert os.path.isdir(paths["unpushed"])
    assert os.path.isdir(paths["dirty"])
    assert (Path(paths["dirty"]) / "scratch.txt").exists()

    listed = _git(clone, "worktree", "list", "--porcelain", env=env).stdout
    assert paths["merged"] not in listed
    assert paths["pushed"] not in listed
    assert paths["unpushed"] in listed
    assert paths["dirty"] in listed

    branches = _git(clone, "branch", "--format=%(refname:short)", env=env).stdout.split()
    assert "feat/merged" not in branches, "a merged branch's worktree and branch both go"
    assert "feat/pushed" in branches, "pushed-idle keeps its branches"
    assert "feat/unpushed" in branches
    assert "feat/dirty" in branches

    # The remote branch of a removed worktree is untouched.
    remote = _git(clone, "ls-remote", "--heads", "origin", env=env).stdout
    assert "refs/heads/feat/pushed" in remote


def test_apply_is_idempotent(repo):
    env, clone = repo["env"], repo["clone"]
    first = _run_script(clone, env, "--apply")
    assert first.returncode == 0, first.stderr
    second = _run_script(clone, env, "--apply")
    assert second.returncode == 0, second.stderr
    assert "removed worktree" not in second.stdout


def test_detached_worktree_is_removed_only_when_idle(repo, tmp_path):
    env, clone = repo["env"], repo["clone"]
    head = _git(clone, "rev-parse", "HEAD", env=env).stdout.strip()
    detached = tmp_path / "wt" / "detached"
    _git(clone, "worktree", "add", "--detach", str(detached), head, env=env)
    path = str(detached.resolve())

    fresh = _run_script(clone, env)
    assert _table(fresh.stdout)[path]["class"] == "detached"
    assert _table(fresh.stdout)[path]["branch"] == "(detached)"
    assert "keep" in _table(fresh.stdout)[path]["action"]
    assert os.path.isdir(path)

    aged = _run_script(clone, env, "--days", "0", "--apply")
    assert aged.returncode == 0, aged.stderr
    assert not os.path.exists(path)


def test_archive_pushes_an_unpushed_branch_before_removing_it(repo):
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    proc = _run_script(clone, env, "--apply", "--archive")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    remote = _git(clone, "ls-remote", "--heads", "origin", env=env).stdout
    assert "refs/heads/feat/unpushed" in remote
    assert not os.path.exists(paths["unpushed"])
    # --archive does not make a dirty worktree removable.
    assert os.path.isdir(paths["dirty"])


def test_detached_worktree_on_no_ref_is_never_removed(repo, tmp_path):
    """A detached HEAD can hold the only copy of a commit, just like a branch."""
    env, clone = repo["env"], repo["clone"]
    head = _git(clone, "rev-parse", "HEAD", env=env).stdout.strip()
    orphan = tmp_path / "wt" / "orphan"
    _git(clone, "worktree", "add", "--detach", str(orphan), head, env=env)
    _commit(orphan, "orphan.txt", env, date=OLD_DATE)
    path = str(orphan.resolve())

    proc = _run_script(clone, env, "--days", "0", "--apply")
    assert proc.returncode == 0, proc.stderr
    assert os.path.isdir(path)
    listed = _git(clone, "worktree", "list", "--porcelain", env=env).stdout
    assert path in listed

    rows = _table(_run_script(clone, env).stdout)
    assert rows[path]["class"] == "detached-orphan"
    assert rows[path]["action"].startswith("KEEP")


def test_dirty_worktree_says_when_its_branch_is_also_unpushed(repo, tmp_path):
    env, clone = repo["env"], repo["clone"]
    both = tmp_path / "wt" / "dirty-unpushed"
    _git(clone, "worktree", "add", "-b", "feat/dirty-unpushed", str(both), "main", env=env)
    _commit(both, "local-only.txt", env, date=OLD_DATE)
    (both / "scratch.txt").write_text("work in progress\n", encoding="utf-8")

    rows = _table(_run_script(clone, env).stdout)
    row = rows[str(both.resolve())]
    assert row["class"] == "dirty"
    assert "branch unpushed" in row["action"]

"""Contract: the worktree pruner removes only work that survives the removal.

Builds a throwaway bare "origin" plus a clone in ``tmp_path`` and drives
``scripts/dev/prune_worktrees.py`` over it.  The pull-request paths run against
a stub ``gh`` that the test writes onto PATH, so nothing here touches the
network or a token.

What it pins, in one sentence: an unpushed branch, a dirty worktree, ignored
scratch and a failed ``gh`` lookup all survive ``--apply``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "prune_worktrees.py"

OLD_DATE = "2020-01-02T03:04:05+00:00"
OWNER = "origin-owner"

STUB_GH = '''#!/usr/bin/env python3
"""Stand-in for `gh`: answers from a JSON fixture, never touches the network."""
import json
import os
import sys

fixture = json.load(open(os.environ["STUB_GH_FIXTURE"]))
args = sys.argv[1:]
if fixture.get("fail"):
    sys.stderr.write("stub gh: forced failure\\n")
    sys.exit(1)
if args[:2] == ["repo", "view"]:
    print(json.dumps({"owner": {"login": fixture.get("owner", "origin-owner")}}))
    sys.exit(0)
if args[:2] == ["pr", "list"]:
    head = args[args.index("--head") + 1]
    print(json.dumps(fixture.get("prs", {}).get(head, [])))
    sys.exit(0)
sys.stderr.write("stub gh: unexpected args %r\\n" % (args,))
sys.exit(2)
'''


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
    path = Path(cwd) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name + "\n", encoding="utf-8")
    _git(cwd, "add", name, env=env)
    _git(cwd, "commit", "-m", "add " + name, env=env, date=date)


def _run_script(clone, env, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
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
    (clone / ".gitignore").write_text(".omc/\n__pycache__/\n", encoding="utf-8")
    _git(clone, "add", ".gitignore", env=env)
    _git(clone, "commit", "-m", "ignore scratch and caches", env=env)
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
        "wt": wt,
        "tmp_path": tmp_path,
        "paths": {
            "merged": str((wt / "merged").resolve()),
            "pushed": str((wt / "pushed").resolve()),
            "unpushed": str((wt / "unpushed").resolve()),
            "dirty": str((wt / "dirty").resolve()),
        },
    }


@pytest.fixture()
def gh_stub(repo):
    """Install a fake `gh` on PATH and return a setter for its answers."""
    tmp_path = repo["tmp_path"]
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(STUB_GH, encoding="utf-8")
    gh.chmod(0o755)
    fixture = tmp_path / "gh_fixture.json"

    def configure(prs=None, fail=False, owner=OWNER):
        fixture.write_text(
            json.dumps({"prs": prs or {}, "fail": fail, "owner": owner}), encoding="utf-8"
        )
        repo["env"]["PATH"] = str(bin_dir) + os.pathsep + repo["env"]["PATH"]
        repo["env"]["STUB_GH_FIXTURE"] = str(fixture)

    configure()
    return configure


def _pr(number, state, owner=OWNER):
    return {"number": number, "state": state, "headRepositoryOwner": {"login": owner}}


# --- offline classification -------------------------------------------------


def test_dry_run_classifies_each_worktree(repo):
    proc = _run_script(repo["clone"], repo["env"], "--no-gh")
    assert proc.returncode == 0, proc.stderr
    rows = _table(proc.stdout)
    paths = repo["paths"]
    assert set(rows) == set(paths.values()), proc.stdout

    assert rows[paths["merged"]]["class"] == "merged"
    assert rows[paths["merged"]]["branch"] == "feat/merged"
    assert rows[paths["merged"]]["pr"] == "off", "--no-gh never claims a PR is absent"
    assert "delete local branch" in rows[paths["merged"]]["action"]

    assert rows[paths["pushed"]]["class"] == "pushed-idle"
    assert rows[paths["unpushed"]]["class"] == "unpushed"
    assert rows[paths["unpushed"]]["action"].startswith("KEEP")
    assert rows[paths["dirty"]]["class"] == "dirty"

    # A dry run touches nothing.
    for path in paths.values():
        assert os.path.isdir(path)


def test_no_gh_refuses_to_remove_a_pushed_idle_worktree(repo):
    """Without PR state an open PR cannot be ruled out, so idle is not enough."""
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    proc = _run_script(clone, env, "--no-gh")
    assert "--no-gh: pull-request state is unknown" in proc.stdout
    row = _table(proc.stdout)[paths["pushed"]]
    assert row["class"] == "pushed-idle"
    assert row["action"].startswith("keep")

    applied = _run_script(clone, env, "--no-gh", "--apply")
    assert applied.returncode == 0, applied.stderr
    assert not os.path.exists(paths["merged"]), "an ancestor of main is still removable"
    assert os.path.isdir(paths["pushed"])


def test_dry_run_leaves_a_recent_pushed_branch_alone(repo):
    """The idle window is what removes a pushed branch, not the push itself."""
    proc = _run_script(repo["clone"], repo["env"], "--no-gh", "--days", "100000")
    assert proc.returncode == 0, proc.stderr
    rows = _table(proc.stdout)
    assert rows[repo["paths"]["pushed"]]["class"] == "pushed-fresh"
    # Only the merged worktree stays removable: merge does not wait out a window.
    assert "dry run: 1 would be removed" in proc.stdout


def test_apply_removes_merged_and_idle_only(repo, gh_stub):
    gh_stub(prs={})
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


def test_apply_is_idempotent(repo, gh_stub):
    gh_stub(prs={})
    env, clone = repo["env"], repo["clone"]
    first = _run_script(clone, env, "--apply")
    assert first.returncode == 0, first.stderr
    second = _run_script(clone, env, "--apply")
    assert second.returncode == 0, second.stderr
    assert "removed worktree" not in second.stdout


# --- detached ---------------------------------------------------------------


def test_detached_worktree_is_removed_only_when_idle(repo, tmp_path):
    env, clone = repo["env"], repo["clone"]
    head = _git(clone, "rev-parse", "HEAD", env=env).stdout.strip()
    detached = tmp_path / "wt" / "detached"
    _git(clone, "worktree", "add", "--detach", str(detached), head, env=env)
    path = str(detached.resolve())

    rows = _table(_run_script(clone, env, "--no-gh").stdout)
    assert rows[path]["class"] == "detached"
    assert rows[path]["branch"] == "(detached)"
    assert "keep" in rows[path]["action"]
    assert os.path.isdir(path)

    aged = _run_script(clone, env, "--no-gh", "--days", "0", "--apply")
    assert aged.returncode == 0, aged.stderr
    assert not os.path.exists(path)


def test_detached_worktree_on_no_ref_is_never_removed(repo, tmp_path):
    """A detached HEAD can hold the only copy of a commit, just like a branch."""
    env, clone = repo["env"], repo["clone"]
    head = _git(clone, "rev-parse", "HEAD", env=env).stdout.strip()
    orphan = tmp_path / "wt" / "orphan"
    _git(clone, "worktree", "add", "--detach", str(orphan), head, env=env)
    _commit(orphan, "orphan.txt", env, date=OLD_DATE)
    path = str(orphan.resolve())

    proc = _run_script(clone, env, "--no-gh", "--days", "0", "--apply")
    assert proc.returncode == 0, proc.stderr
    assert os.path.isdir(path)

    rows = _table(_run_script(clone, env, "--no-gh").stdout)
    assert rows[path]["class"] == "detached-orphan"
    assert rows[path]["action"].startswith("KEEP")


# --- uncommitted and ignored work ------------------------------------------


def test_dirty_worktree_says_when_its_branch_is_also_unpushed(repo, tmp_path):
    env, clone = repo["env"], repo["clone"]
    both = tmp_path / "wt" / "dirty-unpushed"
    _git(clone, "worktree", "add", "-b", "feat/dirty-unpushed", str(both), "main", env=env)
    _commit(both, "local-only.txt", env, date=OLD_DATE)
    (both / "scratch.txt").write_text("work in progress\n", encoding="utf-8")

    row = _table(_run_script(clone, env, "--no-gh").stdout)[str(both.resolve())]
    assert row["class"] == "dirty"
    assert row["pr"] == "-", "a dirty row is classified before any PR lookup"
    assert "branch unpushed" in row["action"]


def test_ignored_scratch_keeps_a_merged_worktree(repo):
    """`worktree remove --force` deletes ignored files; this repo ignores notes."""
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    scratch = Path(paths["merged"]) / ".omc" / "notepad.md"
    scratch.parent.mkdir()
    scratch.write_text("the only copy\n", encoding="utf-8")

    row = _table(_run_script(clone, env, "--no-gh").stdout)[paths["merged"]]
    assert row["class"] == "dirty-ignored"
    assert ".omc/" in row["action"]

    kept = _run_script(clone, env, "--no-gh", "--apply")
    assert kept.returncode == 0, kept.stderr
    assert scratch.exists()

    removed = _run_script(clone, env, "--no-gh", "--apply", "--remove-ignored")
    assert removed.returncode == 0, removed.stderr
    assert not os.path.exists(paths["merged"])


def test_regenerable_caches_do_not_block_removal(repo):
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    cache = Path(paths["merged"]) / "__pycache__"
    cache.mkdir()
    (cache / "thing.cpython-311.pyc").write_bytes(b"\x00")

    row = _table(_run_script(clone, env, "--no-gh").stdout)[paths["merged"]]
    assert row["class"] == "merged"
    proc = _run_script(clone, env, "--no-gh", "--apply")
    assert proc.returncode == 0, proc.stderr
    assert not os.path.exists(paths["merged"])


# --- pull-request paths, against a stub gh ---------------------------------


def test_merged_pr_keeps_a_branch_holding_a_local_commit(repo, gh_stub):
    """A squash merge leaves the branch off main; a later commit is only here."""
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    _commit(paths["pushed"], "after-the-merge.txt", env)
    tip = _git(clone, "rev-parse", "refs/heads/feat/pushed", env=env).stdout.strip()
    gh_stub(prs={"feat/pushed": [_pr(11, "MERGED")]})

    proc = _run_script(clone, env, "--apply")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not os.path.exists(paths["pushed"]), "the worktree of a merged PR goes"
    assert "kept local branch feat/pushed" in proc.stdout

    branches = _git(clone, "branch", "--format=%(refname:short)", env=env).stdout.split()
    assert "feat/pushed" in branches
    still_there = _git(clone, "rev-parse", "refs/heads/feat/pushed", env=env).stdout.strip()
    assert still_there == tip, "the commit that exists only locally survives"


def test_closed_pr_that_never_reached_origin_removes_nothing(repo, gh_stub):
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    gh_stub(prs={"feat/unpushed": [_pr(12, "CLOSED")]})

    row = _table(_run_script(clone, env).stdout)[paths["unpushed"]]
    assert row["class"] == "unpushed"
    assert row["pr"] == "#12 CLOSED"
    assert row["action"].startswith("KEEP")

    proc = _run_script(clone, env, "--apply")
    assert proc.returncode == 0, proc.stderr
    assert os.path.isdir(paths["unpushed"])
    branches = _git(clone, "branch", "--format=%(refname:short)", env=env).stdout.split()
    assert "feat/unpushed" in branches


def test_open_pr_is_kept(repo, gh_stub):
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    gh_stub(prs={"feat/pushed": [_pr(13, "OPEN")]})

    row = _table(_run_script(clone, env).stdout)[paths["pushed"]]
    assert row["class"] == "open-pr"
    assert row["pr"] == "#13 OPEN"

    proc = _run_script(clone, env, "--apply")
    assert proc.returncode == 0, proc.stderr
    assert os.path.isdir(paths["pushed"])


def test_gh_failure_aborts_apply_before_anything_is_removed(repo, gh_stub):
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    gh_stub(fail=True)

    proc = _run_script(clone, env, "--apply")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "aborting" in proc.stderr
    for path in paths.values():
        assert os.path.isdir(path), "a failed lookup must not cost a worktree"

    dry = _run_script(clone, env)
    assert dry.returncode == 0
    row = _table(dry.stdout)[paths["merged"]]
    assert row["pr"] == "?"
    assert "unreliable" in row["action"]


def test_a_forks_pr_with_the_same_head_name_is_ignored(repo, gh_stub):
    """`gh pr list --head` matches the branch NAME across forks."""
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    gh_stub(prs={"feat/pushed": [_pr(14, "OPEN", owner="a-fork")]})

    row = _table(_run_script(clone, env).stdout)[paths["pushed"]]
    assert row["class"] == "pushed-idle", "the fork's open PR is not ours"
    assert row["pr"] == "none", "looked, found no PR of ours"

    proc = _run_script(clone, env, "--apply")
    assert proc.returncode == 0, proc.stderr
    assert not os.path.exists(paths["pushed"])


def test_archive_pushes_an_unpushed_branch_before_removing_it(repo, gh_stub):
    gh_stub(prs={})
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    proc = _run_script(clone, env, "--apply", "--archive")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    remote = _git(clone, "ls-remote", "--heads", "origin", env=env).stdout
    assert "refs/heads/feat/unpushed" in remote
    assert not os.path.exists(paths["unpushed"])
    # --archive does not make a dirty worktree removable.
    assert os.path.isdir(paths["dirty"])


def test_archive_without_gh_pushes_but_keeps_the_worktree(repo):
    env, clone, paths = repo["env"], repo["clone"], repo["paths"]
    proc = _run_script(clone, env, "--no-gh", "--apply", "--archive")
    assert proc.returncode == 0, proc.stderr
    remote = _git(clone, "ls-remote", "--heads", "origin", env=env).stdout
    assert "refs/heads/feat/unpushed" in remote
    assert os.path.isdir(paths["unpushed"])

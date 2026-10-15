"""Real stopped merges using copied CI counters and a minimal, AST-only tree."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/ci/resolve_baselines.py"
SPEC = importlib.util.spec_from_file_location("baseline_resolver", REPO / SCRIPT)
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)
SIZE, LAYER = resolver.BASELINES
X, Y = "rfx/x.py", "rfx/y.py"
LEFT, RIGHT = "rfx/api/_execute.py", "rfx/simulation.py"


def git(repo, *args, expected=0):
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, timeout=30,
        env={**os.environ, "GIT_AUTHOR_NAME": "Baseline tests",
             "GIT_AUTHOR_EMAIL": "tests@example.invalid", "GIT_COMMITTER_NAME": "Baseline tests",
             "GIT_COMMITTER_EMAIL": "tests@example.invalid",
             "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
    )
    assert result.returncode == expected, result.stdout.decode() + result.stderr.decode()
    return result.stdout


def write(repo, path, text):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def baselines(repo, compact=False, limits=None, cap=20):
    """The size gate has no writer: use its counter, validator, and check."""
    data = {"counts": "code-lines", "cap": cap, "files": {
        path: resolver.size.code_lines((repo / path).read_bytes())
        for path in (X, Y) if (repo / path).exists()
    }}
    if limits is not None:
        data["files"].update(limits)
    (repo / SIZE).write_bytes(resolver.encode(data))
    resolver.size.baseline((repo / SIZE).read_bytes())
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        assert resolver.size.check(repo) == 0
        assert resolver.layer.check(repo, write_baseline=True) == 0
    if compact:
        # Put each side's valid baseline on one line so Git must conflict even
        # when the independent rule changes are farther apart than a diff hunk.
        for path in resolver.BASELINES:
            write(repo, path, json.dumps(json.loads((repo / path).read_bytes())) + "\n")


def commit(repo, message):
    git(repo, "add", "--all")
    git(repo, "commit", "-qm", message)


@pytest.fixture
def tree():
    with tempfile.TemporaryDirectory(prefix="baseline-resolver-") as directory:
        repo = Path(directory)
        git(repo, "init", "-q", "-b", "main")
        for path in resolver.layer.FIVE_FILES:
            write(repo, path, "# Minimal caller\n")
        for package in resolver.layer.LAYERS:
            write(repo, package.replace(".", "/") + "/__init__.py", "__all__ = []\n")
        write(repo, "rfx/__init__.py", "# AST-only package\n")
        for path in (LEFT, RIGHT):
            write(repo, path, "def step():\n    update_e()\n")
        for path in (X, Y):
            write(repo, path, "".join(f"value_{i} = {i}\n" for i in range(12)))
        for name in ("check_file_size_ratchet.py", "check_layer_structure.py", "resolve_baselines.py"):
            target = repo / "scripts/ci" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / "scripts/ci" / name, target)
        baselines(repo)
        commit(repo, "Initial counter baselines")
        yield repo


def merge(tree, ours=None, theirs=None, extra=False, layer_conflict=True, omit_ours=(), omit_theirs=(),
          ours_limits=None, theirs_limits=None, ours_cap=20, theirs_cap=20):
    git(tree, "checkout", "-qb", "side-a")
    write(tree, LEFT, "def step():\n    pass\n")
    if ours is None:
        write(tree, X, "value = 1\n" * 8)
    else:
        ours(tree)
    if extra:
        write(tree, "other.txt", "ours\n")
    baselines(tree, compact=True, limits=ours_limits, cap=ours_cap)
    if omit_ours:
        data = json.loads((tree / SIZE).read_bytes())
        for path in omit_ours:
            del data["files"][path]
        write(tree, SIZE, json.dumps(data) + "\n")
    commit(tree, "Lower first baseline")
    git(tree, "checkout", "-qb", "side-b", "main")
    if layer_conflict:
        write(tree, RIGHT, "def step():\n    pass\n")
    else:
        write(tree, LEFT, "def step():\n    pass\n")
    if theirs is None:
        write(tree, Y, "value = 2\n" * 9)
    else:
        theirs(tree)
    if extra:
        write(tree, "other.txt", "theirs\n")
    baselines(tree, compact=True, limits=theirs_limits, cap=theirs_cap)
    if omit_theirs:
        data = json.loads((tree / SIZE).read_bytes())
        for path in omit_theirs:
            del data["files"][path]
        write(tree, SIZE, json.dumps(data) + "\n")
    commit(tree, "Update second baseline")
    git(tree, "checkout", "-q", "side-a")
    git(tree, "merge", "--no-ff", "--no-commit", "side-b", expected=1)
    paths = git(tree, "diff", "--name-only", "--diff-filter=U", "-z").split(b"\0")
    assert SIZE.encode() in paths
    if layer_conflict:
        assert LAYER.encode() in paths


def snapshot(tree):
    return {path: (tree / path).read_bytes() for path in resolver.BASELINES}


def run(tree, *args):
    return subprocess.run(
        [sys.executable, str(tree / SCRIPT), "--repo", str(tree), *args],
        cwd=tree, capture_output=True, text=True, timeout=30,
    )


def unchanged(tree, before, index):
    assert snapshot(tree) == before
    assert git(tree, "ls-files", "--stage", "-z") == index


def test_independent_lowerings_resolve(tree):
    merge(tree)
    head = git(tree, "rev-parse", "HEAD")
    result = run(tree)
    assert result.returncode == 0, result.stderr
    assert "file_size_baseline.json: regenerated and staged" in result.stdout
    assert "layer_baseline.json: regenerated and staged" in result.stdout
    assert "lowered to" not in result.stdout
    assert git(tree, "diff", "--name-only", "--diff-filter=U") == b""
    assert git(tree, "rev-parse", "HEAD") == head  # The resolver never commits.
    resolved = snapshot(tree)
    for checker in (resolver.size, resolver.layer):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            assert checker.check(tree, "side-b") == 0
    baselines(tree)
    assert snapshot(tree) == resolved
    assert json.loads(resolved[SIZE])["files"] == {X: 8, Y: 9}
    assert json.loads(resolved[LAYER])["files"][LEFT]["S"] == []
    assert json.loads(resolved[LAYER])["files"][RIGHT]["S"] == []


def test_both_lowered_limits_keep_other_entries_untouched(tree):
    # Slack in both changed and untouched entries must survive the merge.
    low, high = "rfx/low.py", "rfx/high.py"
    write(tree, low, "value = 0\n")
    write(tree, high, "value = 0\n" * 2)
    original = {X: 18, Y: 19, low: 16, high: 25}
    baselines(tree, limits=original)
    commit(tree, "Keep existing ratchet slack")
    merge(tree, ours_limits={**original, X: 15}, theirs_limits={**original, Y: 17})
    # An older base permits both original ceilings, so this asserts the merge
    # rule itself rather than relying on the checker's base-ratchet rejection.
    result = run(tree, "--base", "main")
    assert result.returncode == 0, result.stderr
    data = json.loads((tree / SIZE).read_bytes())
    assert data == {"counts": "code-lines", "cap": 20,
                    "files": {**original, X: 15, Y: 17}}
    assert "lowered to" not in result.stdout
    for checker in (resolver.size, resolver.layer):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            assert checker.check(tree, "side-b") == 0


@pytest.mark.parametrize("failure", ["merged-count", "base-ratchet"])
def test_growth_preserves_conflicted_bytes(tree, failure):
    merge(tree, ours=lambda repo: write(repo, Y, "value = 1\n" * 9),
          theirs=lambda repo: write(repo, X, "value = 2\n" * 14))
    args = []
    if failure == "base-ratchet":
        # Both side ceilings now permit X, but an explicitly older base does
        # not. This reaches the final in-process checks and tests rollback.
        args = ["--ours", "side-b", "--theirs", "side-b", "--base", "main"]
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    assert all(b"<<<<<<<" in raw for raw in before.values())
    result = run(tree, *args)
    assert result.returncode == 1, result.stdout + result.stderr
    assert X in result.stderr
    assert "file_size_baseline.json" in result.stderr
    if failure == "merged-count":
        assert f"{X}: merged count 14 exceeds min(ours 12, theirs 14) — real conflict" in result.stderr
    else:
        assert "baseline 12, count 14 in proposed baseline" in result.stderr
    unchanged(tree, before, index)


def test_other_conflict_refused(tree):
    write(tree, "other.txt", "base\n")
    commit(tree, "Add shared file")
    merge(tree, extra=True)
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    other = (tree / "other.txt").read_bytes()
    result = run(tree)
    assert result.returncode == 2, result.stderr
    assert "other.txt" in result.stderr
    unchanged(tree, before, index)
    assert (tree / "other.txt").read_bytes() == other


def test_deleted_listed_file_dropped(tree):
    merge(tree, ours=lambda repo: write(repo, Y, "value = 1\n" * 8),
          theirs=lambda repo: (repo / X).unlink())
    result = run(tree)
    assert result.returncode == 0, result.stderr
    assert X not in json.loads((tree / SIZE).read_bytes())["files"]
    assert not (tree / X).exists()


def test_combined_additions_exceed_each_side_limit(tree):
    def add(repo, first, amount):
        path = repo / X
        lines = "extra = 0\n" * amount
        old = path.read_text()
        path.write_text(lines + old if first else old + lines)

    merge(tree, ours=lambda repo: add(repo, True, 2), theirs=lambda repo: add(repo, False, 3))
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = run(tree)
    assert result.returncode == 1, result.stderr
    assert f"{X}: merged count 17 exceeds min(ours 14, theirs 15)" in result.stderr
    unchanged(tree, before, index)


@pytest.mark.parametrize("marker", ["REBASE_HEAD", "rebase-merge", "rebase-apply"])
def test_rebase_refused(tree, marker):
    merge(tree)
    path = resolver.git_path(tree, marker)
    if marker == "REBASE_HEAD":
        path.write_bytes(git(tree, "rev-parse", "HEAD"))
    else:
        path.mkdir()
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = run(tree)
    assert result.returncode == 2
    assert "rebase in progress: abort it and use `git merge origin/main`" in result.stderr
    unchanged(tree, before, index)


def test_real_stopped_rebase_refused(tree):
    merge(tree)
    git(tree, "merge", "--abort")
    git(tree, "rebase", "side-b", expected=1)
    try:
        before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
        result = run(tree, "--theirs", "side-b")
        assert result.returncode == 2
        assert "rebase in progress" in result.stderr
        unchanged(tree, before, index)
    finally:
        git(tree, "rebase", "--abort")


def test_no_merge_or_no_conflicts_refused(tree):
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = run(tree)
    assert result.returncode == 2
    assert "no merge in progress" in result.stderr
    result = run(tree, "--theirs", "HEAD")
    assert result.returncode == 2
    assert "(none)" in result.stderr
    unchanged(tree, before, index)


def test_nonconflicted_baseline_keeps_bytes(tree):
    merge(tree, layer_conflict=False)
    before = (tree / LAYER).read_bytes()
    result = run(tree, "--ours", "HEAD", "--theirs", "side-b")
    assert result.returncode == 0, result.stderr
    assert (tree / LAYER).read_bytes() == before
    assert "layer_baseline.json: unchanged and checked" in result.stdout


def test_nonconflicted_baseline_is_still_checked(tree):
    merge(tree, layer_conflict=False)
    data = json.loads((tree / LAYER).read_bytes())
    data["files"][LEFT]["S"] = ["stale"]
    write(tree, LAYER, json.dumps(data) + "\n")
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = run(tree)
    assert result.returncode == 1
    assert "stale baseline entry stale" in result.stderr
    unchanged(tree, before, index)


def test_only_layer_conflicted_keeps_size_bytes(tree):
    merge(tree)
    # Resolve size ourselves with valid, deliberately noncanonical whitespace.
    data = {"counts": "code-lines", "cap": 20, "files": {X: 8, Y: 9}}
    write(tree, SIZE, json.dumps(data) + "\n")
    git(tree, "add", "--", SIZE)
    before = (tree / SIZE).read_bytes()
    result = run(tree)
    assert result.returncode == 0, result.stderr
    assert (tree / SIZE).read_bytes() == before
    assert "file_size_baseline.json: unchanged and checked" in result.stdout
    assert "lowered to" not in result.stdout


def test_retained_entry_is_reported(tree):
    merge(tree, omit_ours=(X,))
    result = run(tree)
    assert result.returncode == 0, result.stderr
    assert f"{X}: kept entry omitted by ours at merged limit 12" in result.stdout
    assert json.loads((tree / SIZE).read_bytes())["files"][X] == 12
    assert resolver.size.code_lines((tree / X).read_bytes()) == 8


@pytest.mark.parametrize("limit", [20, 25])
def test_removed_entry_at_or_above_cap_dropped(tree, limit):
    baselines(tree, limits={X: limit})
    commit(tree, "Set entry at or above cap")
    merge(tree, omit_ours=(X,), theirs_limits={X: limit})
    result = run(tree)
    assert result.returncode == 0, result.stderr
    assert X not in json.loads((tree / SIZE).read_bytes())["files"]
    assert "kept entry" not in result.stdout
    with redirect_stderr(io.StringIO()):
        assert resolver.size.check(tree, "side-b") == 0


@pytest.mark.parametrize("limit,kept", [(9, True), (10, False), (12, False)])
def test_removed_entry_compared_with_merged_cap(tree, limit, kept):
    merge(tree, ours_cap=10, theirs_cap=20, omit_ours=(X,), theirs_limits={X: limit},
          theirs=lambda repo: write(repo, X, "value = 1\n" * 8))
    result = run(tree)
    assert result.returncode == 0, result.stderr
    data = json.loads((tree / SIZE).read_bytes())
    assert data["cap"] == 10
    assert (X in data["files"]) == kept
    if kept:
        assert data["files"][X] == limit


def test_dropped_entry_count_checked_against_merged_cap(tree):
    # The surviving side's explicit limit is 25, but removal makes cap=10
    # govern X. A later working-tree edit must therefore fail at count 11.
    baselines(tree, limits={X: 25})
    commit(tree, "Prepare removed entry")
    merge(tree, omit_ours=(X,), ours_cap=10, theirs_limits={X: 25})
    write(tree, X, "value = 0\n" * 11)
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = run(tree)
    assert result.returncode == 1
    assert f"{X}: merged count 11 exceeds min(ours 10, theirs 20)" in result.stderr
    unchanged(tree, before, index)


def test_unlisted_tracked_file_exceeding_cap(tree):
    merge(tree)
    write(tree, 'rfx/quoted"file.py', "value = 0\n" * 21)
    git(tree, "add", "--", 'rfx/quoted"file.py')
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = run(tree)
    assert result.returncode == 1
    assert 'rfx/quoted"file.py: merged count 21 exceeds min(ours 20, theirs 20)' in result.stderr
    unchanged(tree, before, index)


def test_layer_growth_names_key_and_limits(tree):
    merge(tree, theirs=lambda repo: write(
        repo, "rfx/nonuniform.py", "def added():\n    update_e()\n",
    ))
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = run(tree)
    assert result.returncode == 1
    assert "layer_baseline.json files/rfx/nonuniform.py/S: merged count 1, min(ours 0, theirs 1)" in result.stderr
    unchanged(tree, before, index)


def test_layer_names_must_come_from_either_side(tree):
    write(tree, "rfx/nonuniform.py", "def old():\n    update_e()\n")
    baselines(tree)
    commit(tree, "Add existing layer violation")
    merge(tree)
    write(tree, "rfx/nonuniform.py", "def fresh():\n    update_e()\n")
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = run(tree)
    assert result.returncode == 1
    assert "min(ours 1, theirs 1); extra names fresh" in result.stderr
    unchanged(tree, before, index)


def test_linked_worktree_rebase_refused(tree):
    with tempfile.TemporaryDirectory(prefix="baseline-linked-") as directory:
        linked = Path(directory) / "repo"
        git(tree, "worktree", "add", "-q", "--detach", str(linked), "main")
        try:
            merge(linked)
            assert (linked / ".git").is_file()
            resolver.git_path(linked, "rebase-merge").mkdir()
            before, index = snapshot(linked), git(linked, "ls-files", "--stage", "-z")
            result = run(linked)
            assert result.returncode == 2
            assert "rebase in progress" in result.stderr
            unchanged(linked, before, index)
        finally:
            git(tree, "worktree", "remove", "--force", str(linked))


@pytest.mark.parametrize("checker_name", ["size", "layer"])
def test_checker_exception_restores_bytes(tree, monkeypatch, checker_name):
    merge(tree)
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")

    def broken(*args):
        raise RuntimeError("counter failure")

    monkeypatch.setattr(getattr(resolver, checker_name), "check", broken)
    assert resolver.resolve(tree) == 1
    unchanged(tree, before, index)


@pytest.mark.parametrize("kind", ["size", "layer"])
def test_main_tree_regeneration_is_byte_exact(kind):
    # Clone committed HEAD locally, without alternates or shared object links.
    # Counters only read its sources; all generation remains in memory.
    with tempfile.TemporaryDirectory(prefix="baseline-main-") as directory:
        clone = Path(directory) / "repo"
        git(REPO, "clone", "-q", "--no-hardlinks", "--no-checkout", str(REPO), str(clone))
        head = git(REPO, "rev-parse", "HEAD").decode().strip()
        git(clone, "checkout", "-q", "--detach", head)
        before = snapshot(clone)
        if kind == "size":
            path = SIZE
            sides = resolver.size.baseline(before[path])
            data, errors, _ = resolver.regenerate_size(clone, sides, sides)
        else:
            path = LAYER
            sides = resolver.layer.baseline(before[path])
            data, errors = resolver.regenerate_layer(clone, sides, sides)
        assert errors == []
        assert resolver.encode(data) == before[path]
        assert git(clone, "status", "--porcelain") == b""


def test_default_repo_is_the_current_directory_not_the_script_location(tree):
    # A session may run another checkout's copy of the script: the tree being merged is the
    # current directory's repository. Run THIS worktree's script (not the tree's copy) from inside
    # the tree with no --repo.
    merge(tree)
    result = subprocess.run(
        [sys.executable, str(REPO / SCRIPT), "--theirs", "side-b"],
        cwd=tree, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert git(tree, "diff", "--name-only", "--diff-filter=U") == b""


def test_checkers_differing_from_the_tree_refused(tree):
    # The final check must run the merged tree's own checkers, not this script's neighbours.
    merge(tree)
    checker = tree / "scripts/ci/check_file_size_ratchet.py"
    checker.write_bytes(checker.read_bytes() + b"\n# local edit\n")
    before, index = snapshot(tree), git(tree, "ls-files", "--stage", "-z")
    result = subprocess.run(
        [sys.executable, str(REPO / SCRIPT), "--theirs", "side-b"],
        cwd=tree, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "differs from the copy beside this script" in result.stderr
    unchanged(tree, before, index)


def test_entry_the_base_side_removed_is_dropped(tree):
    # Main (theirs, the --base) removed a below-cap entry the PR still lists: keeping it would fail
    # the checker against main ("unlisted"), so it is dropped and the cap governs the file.
    merge(tree, omit_theirs=(X,))
    result = run(tree, "--theirs", "side-b")
    assert result.returncode == 0, result.stderr
    assert X not in json.loads((tree / SIZE).read_bytes())["files"]
    with redirect_stderr(io.StringIO()):
        assert resolver.size.check(tree, "side-b") == 0

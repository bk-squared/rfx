"""Compare real Git trees while replacing only the mypy subprocess."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mypy_count", REPO / "scripts/ci/check_mypy_count.py",
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def git(repo, *args, check=True):
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=check,
    )
    return result.stdout.strip() if check else result


def commit(repo, message):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Independent repositories must not inherit the caller’s Git index or object overlays."""
    for name in ("GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                 "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Mypy contract")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "mypy@example.invalid")
    monkeypatch.setattr(gate.tempfile, "tempdir", str(tmp_path))
    tree = tmp_path / "repo"
    (tree / "rfx").mkdir(parents=True)
    git(tree, "init", "-q", "-b", "main")
    (tree / "rfx/mod.py").write_text("# error: alpha\n# error: beta\n# error: gamma\n")
    (tree / "pyproject.toml").write_text('[tool.mypy]\npython_version = "3.11"\n')
    commit(tree, "base")
    (tree / "rfx/mod.py").write_text(
        "# error: alpha\n# error: beta\n# error: gamma\n# error: delta\n",
    )
    (tree / "pyproject.toml").write_text('[tool.mypy]\npython_version = "3.12"\n')
    commit(tree, "head")
    monkeypatch.setattr(gate, "REPO", tree)
    return tree


@pytest.fixture
def fake_mypy(repo, monkeypatch):
    real_run = subprocess.run
    caches = []
    trees = []

    def install(respond):
        def run(args, **kwargs):
            if args[:3] != [sys.executable, "-m", "mypy"]:
                return real_run(args, **kwargs)
            assert args[:5] == [sys.executable, "-m", "mypy", "rfx", "--cache-dir"]
            assert len(args) == 6
            assert kwargs["stdout"] == subprocess.PIPE
            assert kwargs["stderr"] == subprocess.STDOUT
            assert kwargs["text"] is True
            cache = Path(args[5])
            assert cache.is_dir() and not list(cache.iterdir())
            assert cache not in caches
            caches.append(cache)
            (cache / "cache-entry").write_text("stale")
            tree = Path(kwargs["cwd"])
            trees.append(tree)
            returncode, output = respond(tree)
            return subprocess.CompletedProcess(args, returncode, output)

        monkeypatch.setattr(gate.subprocess, "run", run)

    yield install, caches, trees
    assert all(not cache.exists() for cache in caches)
    assert all(not tree.exists() for tree in trees if tree.name == "tree")
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


@pytest.fixture
def run_gate(repo, fake_mypy, capsys):
    install, _, _ = fake_mypy

    def run(head_output, head_code=1, base_output=None, base_code=1):
        if base_output is None:
            base_output = head_output
        install(lambda tree: (head_code, head_output) if tree == repo else (base_code, base_output))
        code = gate.main([])
        return code, capsys.readouterr()

    return run


@pytest.mark.parametrize("summary,count,returncode", [
    ("Found 1382 errors in 80 files (checked 900 source files)", 1382, 1),
    ("Found 1 error in 1 file (checked 1 source file)", 1, 1),
    ("Found 1382 errors in 80 files (errors prevented further checking)", 1382, 1),
    ("Success: no issues found in 1 source file", 0, 0),
])
def test_equal_counts_and_summary_forms_pass(repo, run_gate, summary, count, returncode):
    code, output = run_gate(summary + "\n", returncode, base_code=returncode)
    short = git(repo, "rev-parse", "--short", "HEAD^1")
    assert code == 0
    assert output.out == f"mypy errors: {count} (base {count} at {short})\n"
    assert output.err == ""


@pytest.mark.parametrize("summary,count,returncode", [
    ("Found 1 error in 1 file", 1, 1),
    ("Success: no issues found in 900 source files", 0, 0),
])
def test_lower_count_passes(run_gate, summary, count, returncode):
    code, output = run_gate(summary + "\n", returncode, "Found 2 errors in 1 file\n")
    assert code == 0
    assert output.out.startswith(f"mypy errors: {count} (base 2 at ")
    assert len(output.out.splitlines()) == 1


@pytest.mark.parametrize("side", ["head", "base"])
@pytest.mark.parametrize("output,returncode", [
    ("Traceback: crashed\n", 1),
    ("No summary\n", 0),
    ("Found 1 error in 1 file (checked 1 source file)\n", 2),
    ("Found 1 error in 1 file\ninternal error\n", 1),
])
def test_crash_preserves_output_and_exits_two(run_gate, side, output, returncode):
    valid = "Found 1 error in 1 file\n"
    if side == "head":
        code, captured = run_gate(output, returncode, valid)
    else:
        code, captured = run_gate(valid, 1, output, returncode)
    assert code == 2
    assert output in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("crash_side", [None, "head", "base"])
def test_fresh_caches_and_worktree_are_removed(repo, fake_mypy, capsys, crash_side):
    install, caches, trees = fake_mypy

    def respond(tree):
        side = "head" if tree == repo else "base"
        expected_version = "3.12" if side == "head" else "3.11"
        assert expected_version in (tree / "pyproject.toml").read_text()
        if side == crash_side:
            raise OSError("mypy failed to start")
        return 0, "Success: no issues found in 1 source file\n"

    install(respond)
    for _ in range(2):
        assert gate.main([]) == (2 if crash_side else 0)
        assert all(not cache.exists() for cache in caches)
        assert all(not tree.exists() for tree in trees if tree != repo)
        captured = capsys.readouterr()
        if crash_side:
            assert "mypy failed to start" in captured.err
    assert len(caches) == (2 if crash_side == "head" else 4)


def marker_mypy(tree):
    lines = (tree / "rfx/mod.py").read_text().splitlines()
    errors = [f"rfx/mod.py:{number}: error: {line.removeprefix('# error: ')} [misc]"
              for number, line in enumerate(lines, 1) if line.startswith("# error: ")]
    return 1, "\n".join([*errors, f"Found {len(errors)} errors in 1 file"]) + "\n"


def test_error_added_on_top_of_base_fails_even_below_old_total(repo, tmp_path, fake_mypy, monkeypatch, capsys):
    install, _, _ = fake_mypy
    install(marker_mypy)
    base = git(repo, "rev-parse", "HEAD^1")
    short = git(repo, "rev-parse", "--short", base)
    expected = f"mypy errors: 4 (base 3 at {short})\nrfx/mod.py:4: error: delta [misc]\n"
    assert gate.main([]) == 1
    assert capsys.readouterr().out == expected

    def shallow_clone(name):
        clone = tmp_path / name
        git(tmp_path, "clone", "--depth", "1", repo.as_uri(), str(clone))
        assert git(clone, "rev-parse", "--is-shallow-repository") == "true"
        assert git(clone, "rev-parse", "HEAD^1", check=False).returncode != 0
        assert git(clone, "cat-file", "-e", base, check=False).returncode != 0
        assert f"parent {base}" in git(clone, "cat-file", "-p", "HEAD").splitlines()
        return clone

    probe = shallow_clone("probe")
    git(probe, "fetch", "--no-tags", "--depth=1", "origin", base)
    probe_tree = tmp_path / "probe-tree"
    git(probe, "worktree", "add", "--detach", str(probe_tree), base)
    try:
        assert git(probe_tree, "rev-parse", "HEAD") == base
        assert git(probe_tree, "symbolic-ref", "-q", "HEAD", check=False).returncode != 0
        assert (probe_tree / "rfx/mod.py").read_text().splitlines() == [
            "# error: alpha", "# error: beta", "# error: gamma",
        ]
        (probe_tree / "untracked").write_text("force removal")
    finally:
        git(probe, "worktree", "remove", "--force", str(probe_tree))
    assert not probe_tree.exists()
    assert git(probe, "worktree", "list", "--porcelain").count("worktree ") == 1

    clone = shallow_clone("shallow")
    monkeypatch.setattr(gate, "REPO", clone)
    assert gate.main([]) == 1
    assert capsys.readouterr().out == expected
    assert git(clone, "cat-file", "-t", base) == "commit"
    assert git(clone, "worktree", "list", "--porcelain").count("worktree ") == 1

    unavailable = shallow_clone("no-origin")
    git(unavailable, "remote", "remove", "origin")
    monkeypatch.setattr(gate, "REPO", unavailable)
    assert gate.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fetch" in captured.err and "origin" in captured.err

    git(repo, "checkout", "--detach", base)
    (repo / "rfx/mod.py").write_text("\n# error: alpha\n\n# error: beta\n\n# error: gamma\n")
    commit(repo, "move markers")
    monkeypatch.setattr(gate, "REPO", repo)
    assert gate.main([]) == 0
    assert capsys.readouterr().out == f"mypy errors: 3 (base 3 at {short})\n"


def test_added_errors_are_a_multiset_ignoring_lines_and_columns(run_gate):
    base = "rfx/mod.py:1:2: error: repeated [misc]\nrfx/mod.py:2: error: moved [misc]\nFound 2 errors in 1 file\n"
    head = (
        "rfx/mod.py:10: error: repeated [misc]\n"
        "rfx/mod.py:20:30: error: moved [misc]\n"
        "rfx/mod.py:40:5: error: repeated [misc]\n"
        "rfx/mod.py:50: note: context\nFound 3 errors in 1 file\n"
    )
    code, output = run_gate(head, 1, base)
    assert code == 1
    assert output.out.splitlines()[1:] == ["rfx/mod.py:40:5: error: repeated [misc]"]


def test_explicit_base_revision(repo, fake_mypy, capsys):
    install, _, _ = fake_mypy
    install(marker_mypy)
    git(repo, "checkout", "--detach", "HEAD^1")
    git(repo, "branch", "comparison")
    short = git(repo, "rev-parse", "--short", "HEAD")
    assert gate.main(["--base", "comparison"]) == 0
    assert capsys.readouterr().out == f"mypy errors: 3 (base 3 at {short})\n"


@pytest.mark.parametrize("mode", ["root", "unreadable"])
def test_missing_base_exits_two(repo, fake_mypy, capsys, mode):
    install, _, trees = fake_mypy
    install(marker_mypy)
    if mode == "root":
        git(repo, "checkout", "--detach", "HEAD^1")
        args, reason = [], "no first parent"
    else:
        args, reason = ["--base", "not-a-revision"], "not-a-revision"
    assert gate.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert reason in captured.err
    assert trees == []

"""Numeric file-size gate: synthetic boundaries plus the live source tree."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "file_size_ratchet", REPO / "scripts/ci/check_file_size_ratchet.py",
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.delenv("HEAD_SHA", raising=False)
    (tmp_path / "scripts/ci").mkdir(parents=True)
    (tmp_path / "rfx/nested").mkdir(parents=True)
    return tmp_path


def write_tree(tree, files=None, count=3, cap=2):
    (tree / gate.BASELINE).write_text(json.dumps({
        "counts": "code-lines", "cap": cap, "files": {"rfx/nested/a.py": 3} if files is None else files,
    }))
    (tree / "rfx/nested/a.py").write_bytes(b"x\n" * count + b"unterminated")


def run(tree, capsys, *args):
    result = gate.main(["--repo", str(tree), *args])
    captured = capsys.readouterr()
    assert captured.out == ""
    return result, captured.err


def mock_base(monkeypatch, cap=2, files=None, legacy=False):
    previous = {
        "counts": "code-lines", "cap": cap, "files": {"rfx/nested/a.py": 3} if files is None else files,
    }
    if legacy:
        del previous["counts"]
    previous = json.dumps(previous).encode()

    def git(repo, *args):
        assert args in (("ls-tree", "base", "--", gate.BASELINE),
                        ("show", f"base:{gate.BASELINE}"))
        return b"present" if args[0] == "ls-tree" else previous

    monkeypatch.setattr(gate, "git", git)


def test_growth_fails(tree, capsys):
    write_tree(tree, count=4)
    code, err = run(tree, capsys)
    assert code == 1
    assert "rfx/nested/a.py: baseline 3, count 4" in err
    assert "split the file rather than raise the number" in err


def test_bare_carriage_returns_count_as_lines(tree, capsys):
    write_tree(tree)
    (tree / "rfx/nested/a.py").write_bytes(b"x\r" * 4 + b"y\r\n")
    code, err = run(tree, capsys)
    assert code == 1
    assert "rfx/nested/a.py: baseline 3, count 5" in err


def test_unlisted_cap_breach_fails(tree, capsys):
    write_tree(tree, files={})
    code, err = run(tree, capsys)
    assert code == 1
    assert "baseline 2, count 3" in err


@pytest.mark.parametrize("files,cap", [({"rfx/nested/a.py": 4}, 2), ({}, 3)])
def test_raised_baseline_or_cap_fails(tree, capsys, monkeypatch, files, cap):
    mock_base(monkeypatch)
    write_tree(tree, files=files, cap=cap, count=2)
    assert run(tree, capsys, "--base", "base")[0] == 1


def test_new_entry_fails(tree, capsys, monkeypatch):
    mock_base(monkeypatch, files={})
    write_tree(tree)
    code, err = run(tree, capsys, "--base", "base")
    assert code == 1
    assert "baseline unlisted" in err


def test_lowering_passes(tree, capsys, monkeypatch):
    mock_base(monkeypatch)
    write_tree(tree, files={"rfx/nested/a.py": 2}, count=2, cap=1)
    assert run(tree, capsys, "--base", "base") == (0, "")


@pytest.mark.parametrize("count,expected", [(2, 0), (3, 1), (None, 0)])
def test_removal_requires_deletion_or_count_within_cap(
    tree, capsys, monkeypatch, count, expected,
):
    mock_base(monkeypatch)
    write_tree(tree, files={}, count=count or 0)
    if count is None:
        (tree / "rfx/nested/a.py").unlink()
    assert run(tree, capsys, "--base", "base")[0] == expected


def test_lower_count_prints_notice(tree, capsys):
    write_tree(tree, count=2)
    assert run(tree, capsys) == (
        0, f"lower rfx/nested/a.py to 2 in {gate.BASELINE}\n",
    )


def test_missing_base_baseline_allows_adoption(tree, capsys, monkeypatch):
    write_tree(tree)
    monkeypatch.setattr(gate, "git", lambda *args: b"")
    assert run(tree, capsys, "--base", "base") == (0, "")


def test_invalid_base_fails_closed(tree, capsys, monkeypatch):
    write_tree(tree)

    def broken(*args):
        raise subprocess.CalledProcessError(128, ["git", "ls-tree"])

    monkeypatch.setattr(gate, "git", broken)
    assert run(tree, capsys, "--base", "missing")[0] == 1


def test_head_and_base_environment_select_snapshots(tree, capsys, monkeypatch):
    write_tree(tree, count=1)
    monkeypatch.setenv("BASE_SHA", "base")
    monkeypatch.setenv("HEAD_SHA", "head")
    snapshots = {
        ("ls-tree", "base", "--", gate.BASELINE): b"present",
        ("show", f"base:{gate.BASELINE}"): json.dumps({"counts": "code-lines", "cap": 2, "files": {}}).encode(),
        ("show", f"head:{gate.BASELINE}"): json.dumps({"counts": "code-lines", "cap": 2, "files": {}}).encode(),
        ("ls-tree", "-r", "--name-only", "-z", "head", "--", "rfx/"): b"rfx/a.py\0",
        ("show", "head:rfx/a.py"): b"x\nx\nx\n",
    }
    monkeypatch.setattr(gate, "git", lambda repo, *args: snapshots[args])
    code, err = run(tree, capsys)
    assert code == 1
    assert "rfx/a.py: baseline 2, count 3" in err


def test_baseline_matches_real_tree(capsys, monkeypatch):
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.delenv("HEAD_SHA", raising=False)
    code, err = run(REPO, capsys)
    assert code == 0, err


def test_required_job_and_local_runner_bind_ratchet():
    workflow = yaml.safe_load((REPO / ".github/workflows/pr-tests.yml").read_text())
    steps = workflow["jobs"]["guards-and-preflight"]["steps"]
    calls = [i for i, s in enumerate(steps)
             if "scripts/ci/check_file_size_ratchet.py" in s.get("run", "")]
    assert len(calls) == 1
    index = calls[0]
    step = steps[index]
    assert "check_data_budget.py" in steps[index - 1]["run"]
    assert "pip install" in steps[index + 1]["run"]
    assert not step.get("if") and not step.get("continue-on-error")
    assert step["run"] == "python scripts/ci/check_file_size_ratchet.py"
    # No HEAD_SHA: the checked-out merge with the base is what gets counted.
    assert step["env"] == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha || github.event.merge_group.base_sha }}",
    }
    local = (REPO / "scripts/ci/local.sh").read_text()
    assert 'BASE_SHA="$CHANGELOG_BASE" \\\n  "$PYTHON" scripts/ci/check_file_size_ratchet.py' in local
    assert "HEAD_SHA=" not in local.split("check_file_size_ratchet.py")[0].rsplit("begin", 1)[-1]
    assert '"$PYTHON" scripts/ci/check_file_size_ratchet.py || fail' in local


@pytest.mark.parametrize("padding", [b" \t\n", b"  # comment\n"], ids=["blank", "comment"])
def test_replacing_ignored_lines_with_code_fails(tree, capsys, padding):
    write_tree(tree)
    source = tree / "rfx/nested/a.py"
    source.write_bytes(b"x\n" * 3 + padding * 5)
    assert run(tree, capsys) == (0, "")
    source.write_bytes(b"x\n" * 8)
    code, err = run(tree, capsys)
    assert code == 1
    assert "baseline 3, count 8" in err


def test_adding_only_blank_and_comment_lines_passes(tree, capsys):
    write_tree(tree)
    (tree / "rfx/nested/a.py").write_bytes(b"x\n" * 3 + b" \t\n  # comment\n" * 5)
    assert run(tree, capsys) == (0, "")


def test_code_lines_byte_rules():
    assert gate.code_lines(b'x\r\n \t\r# comment\n  # comment\r\nx # inline\n') == 2
    assert gate.code_lines(b'"""docstring\ntext\n"""\nvalue = """\ntext\n"""\n') == 6
    assert gate.code_lines(b'\xff\n\v\n\f\nunterminated') == 1


@pytest.mark.parametrize("value,expected", [(2, 1), (3, 0), (4, 1)])
def test_transition_requires_exact_counts(tree, capsys, monkeypatch, value, expected):
    mock_base(monkeypatch, files={"rfx/nested/a.py": 8}, legacy=True)
    write_tree(tree, files={"rfx/nested/a.py": value})
    code, err = run(tree, capsys, "--base", "base")
    assert code == expected
    if expected:
        assert f"transition baseline {value} must equal actual count 3" in err
        assert "set the entry to 3" in err
        assert "split the file" not in err


def test_transition_allows_new_exact_entry(tree, capsys, monkeypatch):
    mock_base(monkeypatch, cap=3, files={}, legacy=True)
    write_tree(tree)
    assert run(tree, capsys, "--base", "base") == (0, "")


@pytest.mark.parametrize("case", ["cap", "listed", "unlisted"])
def test_transition_rejects_legacy_limit_increases(tree, capsys, monkeypatch, case):
    mock_base(monkeypatch, cap=10, files={"rfx/nested/a.py": 30}, legacy=True)
    files = {"rfx/nested/a.py": 50 if case == "listed" else 30}
    cap = 9999 if case == "cap" else 10
    if case == "unlisted":
        files["rfx/nested/b.py"] = 40
        (tree / "rfx/nested/b.py").write_bytes(b"x\n" * 40)
    write_tree(tree, files=files, count=files["rfx/nested/a.py"], cap=cap)
    code, err = run(tree, capsys, "--base", "base")
    assert code == 1
    expected = {
        "cap": f"{gate.BASELINE}: cap baseline 10, count 9999",
        "listed": "rfx/nested/a.py: baseline 30, count 50 in proposed baseline",
        "unlisted": "rfx/nested/b.py: baseline 10, count 40 in proposed baseline",
    }[case]
    assert err == expected + "; split the file rather than raise the number.\n"


def test_transition_lowering_legacy_limits_passes(tree, capsys, monkeypatch):
    mock_base(monkeypatch, cap=10, files={"rfx/nested/a.py": 30}, legacy=True)
    write_tree(tree, files={"rfx/nested/a.py": 20, "rfx/nested/b.py": 8}, count=20, cap=9)
    (tree / "rfx/nested/b.py").write_bytes(b"x\n" * 8)
    assert run(tree, capsys, "--base", "base") == (0, "")


def test_transition_rejects_missing_listed_source(tree, capsys, monkeypatch):
    mock_base(monkeypatch, legacy=True)
    write_tree(tree)
    (tree / "rfx/nested/a.py").unlink()
    code, err = run(tree, capsys, "--base", "base")
    assert code == 1
    assert "transition baseline entry has no source file" in err
    assert "split the file" not in err


@pytest.mark.parametrize("counts", [None, "newline-bytes", 1])
def test_invalid_count_mode_fails(tree, capsys, counts):
    write_tree(tree)
    path = tree / gate.BASELINE
    data = json.loads(path.read_bytes())
    data["counts"] = counts
    path.write_text(json.dumps(data))
    assert run(tree, capsys)[0] == 1


def test_current_legacy_baseline_fails(tree, capsys):
    write_tree(tree)
    path = tree / gate.BASELINE
    data = json.loads(path.read_bytes())
    del data["counts"]
    path.write_text(json.dumps(data))
    assert run(tree, capsys)[0] == 1

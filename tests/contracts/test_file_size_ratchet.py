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
        "counts": gate.UNIT, "cap": cap, "files": {"rfx/nested/a.py": 3} if files is None else files,
    }))
    (tree / "rfx/nested/a.py").write_bytes(b"x\n" * count + b"unterminated")


def run(tree, capsys, *args):
    result = gate.main(["--repo", str(tree), *args])
    captured = capsys.readouterr()
    assert captured.out == ""
    return result, captured.err


def mock_base(monkeypatch, cap=2, files=None, legacy=False):
    previous = {
        "counts": gate.UNIT, "cap": cap, "files": {"rfx/nested/a.py": 3} if files is None else files,
    }
    if legacy:
        del previous["counts"]
    previous = json.dumps(previous).encode()

    def git(repo, *args):
        if args in (("ls-tree", "base", "--", gate.PENDING),
                    ("diff", "-M90%", "--name-status", "--diff-filter=R", "-z", "base", "--", "rfx/")):
            return b""
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


@pytest.mark.parametrize("cap,count,expected,message", [
    (4, 2, 1, "rfx/nested/a.py: delisting gives the file the cap of 4 instead of its baseline 3; "
              "keep the entry, or delete the file.\n"),
    (3, 3, 0, ""),
    (2, 2, 0, ""),
    (2, 3, 1, "rfx/nested/a.py: baseline 2, count 3; split the file rather than raise the number.\n"),
    (4, None, 0, ""),
], ids=["base-below-cap", "base-at-cap", "base-above-cap", "count-above-cap", "deleted"])
def test_removal_requires_deletion_or_base_entry_at_or_above_cap(
    tree, capsys, monkeypatch, cap, count, expected, message,
):
    """Base lists a.py at 3 under cap 5; the head removes the entry."""
    mock_base(monkeypatch, cap=5)
    write_tree(tree, files={}, count=count or 0, cap=cap)
    if count is None:
        (tree / "rfx/nested/a.py").unlink()
    assert run(tree, capsys, "--base", "base") == (expected, message)


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
        ("ls-tree", "base", "--", gate.PENDING): b"",
        ("ls-tree", "head", "--", gate.PENDING): b"",
        ("ls-tree", "base", "--", gate.BASELINE): b"present",
        ("show", f"base:{gate.BASELINE}"): json.dumps({"counts": gate.UNIT, "cap": 2, "files": {}}).encode(),
        ("show", f"head:{gate.BASELINE}"): json.dumps({"counts": gate.UNIT, "cap": 2, "files": {}}).encode(),
        ("ls-tree", "-r", "--name-only", "-z", "head", "--", "rfx/"): b"rfx/a.py\0",
        ("show", "head:rfx/a.py"): b"x\nx\nx\n",
    }
    monkeypatch.setattr(gate, "git", lambda repo, *args: snapshots[args])
    code, err = run(tree, capsys)
    assert code == 1
    assert "rfx/a.py: baseline 2, count 3" in err


@pytest.mark.parametrize("head", ["", "head"], ids=["working-tree", "snapshot"])
@pytest.mark.parametrize("old,new,expected", [
    ("9", "10", 1), ("10", "9", 0), ("9", "9", 0),
    (None, "9", 0), ("unreadable", "9", 1), ("9", "True", 1),
    ("9", "9 + 1", 1), ("9", "(", 1),
], ids=["raised", "lowered", "equal", "adoption", "unreadable", "bool", "expression", "syntax"])
def test_pending_ceiling_ratchets_against_base(tree, capsys, monkeypatch, head, old, new, expected):
    write_tree(tree)
    pending = tree / gate.PENDING
    pending.parent.mkdir(parents=True)
    pending.write_text(f"PENDING_CEILING = {new if not head else '0'}\n")
    snapshots = {
        ("ls-tree", "base", "--", gate.BASELINE): b"",
        ("show", f"head:{gate.BASELINE}"): (tree / gate.BASELINE).read_bytes(),
        ("ls-tree", "-r", "--name-only", "-z", "head", "--", "rfx/"): b"rfx/nested/a.py\0",
        ("show", "head:rfx/nested/a.py"): (tree / "rfx/nested/a.py").read_bytes(),
        ("ls-tree", "base", "--", gate.PENDING): b"present" if old is not None else b"",
        ("ls-tree", "head", "--", gate.PENDING): b"present",
        ("show", f"base:{gate.PENDING}"): f"PENDING_CEILING = {old}\n".encode(),
        ("show", f"head:{gate.PENDING}"): f"PENDING_CEILING = {new}\n".encode(),
    }

    def git(repo, *args):
        if old == "unreadable" and args == ("ls-tree", "base", "--", gate.PENDING):
            raise subprocess.CalledProcessError(128, ["git", *args])
        return snapshots[args]

    monkeypatch.setattr(gate, "git", git)
    code, err = run(tree, capsys, "--base", "base", "--head", head)
    assert code == expected, err
    if old == "9" and new == "10":
        assert "PENDING_CEILING baseline 9, count 10" in err


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
    assert steps[index + 1]["run"] == "python scripts/ci/check_layer_structure.py"
    assert "pip install" in steps[index + 2]["run"]
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


@pytest.mark.parametrize("counts", [None, "newline-bytes", 1, "code-lines"])
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


@pytest.mark.parametrize("head", ["", "HEAD"], ids=["working-tree", "snapshot"])
@pytest.mark.parametrize("case,expected,message", [
    ("module", 0, ""),
    ("rename", 0, ""),
    ("unrelated", 1, "rfx/nested/b.py: baseline unlisted, count 20 in proposed baseline"),
    ("raised", 1, "rfx/nested/b.py: baseline 20, count 21 in proposed baseline"),
    ("below-similarity", 1, "rfx/nested/b.py: baseline unlisted, count 20 in proposed baseline"),
    ("source-present", 1, "rfx/nested/a/__init__.py: baseline unlisted, count 20 in proposed baseline"),
    ("reuse", 1, "rfx/nested/b.py: baseline unlisted, count 20 in proposed baseline"),
    ("count-growth", 1, "rfx/nested/b.py: baseline 20, count 21"),
])
def test_moved_entry_inherits_only_its_old_cap(tree, capsys, head, case, expected, message):
    write_tree(tree, files={"rfx/nested/a.py": 20}, count=20)
    original = b"".join(f"old_{i:02d} = {i}\n".encode() for i in range(20))
    source = tree / "rfx/nested/a.py"
    source.write_bytes(original)
    gate.git(tree, "init", "-q", "--template=")
    gate.git(tree, "config", "user.name", "Ratchet tests")
    gate.git(tree, "config", "user.email", "tests@example.invalid")
    gate.git(tree, "config", "commit.gpgsign", "false")
    gate.git(tree, "config", "core.hooksPath", "/dev/null")
    gate.git(tree, "add", ".")
    gate.git(tree, "commit", "-qm", "Base baseline")
    base = gate.git(tree, "rev-parse", "HEAD").decode().strip()

    path = ("rfx/nested/a/__init__.py" if case in ("module", "source-present", "reuse")
            else "rfx/nested/b.py")
    target = tree / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if case == "source-present":
        target.write_bytes(original)
    else:
        source.rename(target)
    if case in ("module", "unrelated", "reuse"):
        target.write_bytes(original.replace(b"old_", b"new_"))
    elif case == "rename":
        target.write_bytes(original.replace(b"old_19", b"new_19"))
    elif case == "below-similarity":
        target.write_bytes(original.replace(b"old_0", b"new_0"))
    elif case in ("raised", "count-growth"):
        target.write_bytes(original + b"extra = 1\n")
    files = {path: 21 if case == "raised" else 20}
    if case == "source-present":
        files["rfx/nested/a.py"] = 20
    elif case == "reuse":
        (tree / "rfx/nested/b.py").write_bytes(original)
        files["rfx/nested/b.py"] = 20
    (tree / gate.BASELINE).write_text(json.dumps({
        "counts": gate.UNIT, "cap": 2, "files": files,
    }))
    gate.git(tree, "add", ".")
    if head:
        gate.git(tree, "commit", "-qm", "Moved baseline")
        # Snapshot checks must ignore subsequent working-tree contents.
        target.write_bytes(b"dirty = 1\n" * 30)
    code, err = run(tree, capsys, "--base", base, "--head", head)
    assert code == expected
    assert err == (message + "; split the file rather than raise the number.\n" if message else "")


OWNERS = (
    b'"""Module\n'
    b'docstring."""\n'
    b"class A:\n"
    b'    """Class docstring."""\n'
    b"    def method(self):\n"
    b'        """Method\n'
    b'        docstring."""\n'
    b"        return 1\n"
    b"def f():\n"
    b'    """Def docstring."""\n'
    b"    def inner():\n"
    b'        """Nested def docstring."""\n'
    b"    return inner\n"
    b"async def g():\n"
    b'    """Async def docstring."""\n'
    b"if True:\n"
    b"    class B:\n"
    b"        def m(self):\n"
    b'            """Method of a class inside an if."""\n'
    b"            pass\n"
)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
def test_ratchet_lines_leaves_out_docstrings_of_every_owner(ending):
    """Eleven code lines remain: every docstring, nested or not, is left out."""
    source = OWNERS.replace(b"\n", ending)
    assert gate.code_lines(source) == 20
    assert gate.ratchet_lines(source, "p.py") == 11


def test_ratchet_lines_counts_strings_that_are_not_docstrings():
    """A bytes literal, an f-string, an assigned string and a second string statement count."""
    source = (
        b'"""Module."""\n'
        b"def a():\n"
        b'    b"""bytes"""\n'
        b"def b():\n"
        b'    f"""f-string"""\n'
        b"def c():\n"
        b'    s = """assigned"""\n'
        b"def d():\n"
        b"    pass\n"
        b'    """second"""\n'
    )
    assert gate.ratchet_lines(source, "p.py") == 9


@pytest.mark.parametrize("source,expected", [
    (b'def f(): """d"""\n', 1),
    (b'"""d"""; x = 1\n', 1),
    (b'"""d"""  # note\n', 0),
    (b'def f():\n    """d\n    e"""; x = 1\n', 2),
    (b'\xef\xbb\xbf"""d"""\n', 0),
    (b"# coding: latin-1\ndef f():\n    \"\xe9\xe9\xe9\xe9\"  # n\n", 1),
], ids=["after-def", "before-statement", "trailing-comment", "multiline-before-statement",
        "bom", "latin-1-cookie"])
def test_ratchet_lines_shared_first_and_last_lines(source, expected):
    """A docstring line shared with code counts; a trailing comment does not keep it."""
    assert gate.ratchet_lines(source, "p.py") == expected


def test_docstring_line_starting_with_hash_is_left_out_once():
    """code_lines skips the "# heading" line, so the docstring holds 2 counted lines, not 3."""
    source = b'"""Title.\n# heading\n"""\n'
    assert gate.code_lines(source) == 2
    assert gate.ratchet_lines(source, "p.py") == 0
    assert gate.docstring_code_lines(source, "p.py") == 2


def test_form_feed_inside_docstring_keeps_line_numbers():
    """A form feed is no line terminator for ast or code_lines, so the count does not move."""
    plain = b'def f():\n    """Title.\n\n    body text\n    """\n    return 1\n'
    feed = b'def f():\n    """Title.\n\x0c\n    body\x0c text\n    """\n    return 1\n'
    assert gate.ratchet_lines(plain, "p.py") == 2
    assert gate.ratchet_lines(feed, "p.py") == 2


def test_unparsable_file_fails_with_its_path(tree, capsys):
    write_tree(tree)
    (tree / "rfx/nested/a.py").write_bytes(b"def (\n")
    code, err = run(tree, capsys)
    assert code == 1
    assert "rfx/nested/a.py: does not parse, so its docstrings cannot be found" in err


A_BASE = b'def a():\n    """A.\n    one\n    two\n    """\n    return 1\n'
B_BASE = b'def b():\n    """B.\n    1\n    2\n    3\n    """\n    return 2\n'
C_BASE = b'def c():\n    """C.\n    """\n    return 3\n'
A_HEAD = b'def a():\n    """A."""\n    return 1\n'
B_HEAD = b'def b():\n    """B."""\n    return 2\n' + b"".join(b"x%d = %d\n" % (i, i) for i in range(4))
NEW_UNIT_BOUND = "with docstrings not counted it must be at most"


@pytest.mark.parametrize("case,expected,message", [
    ("listed-at-bound", 0, ""),
    ("listed-above-bound", 1,
     f"rfx/a.py: baseline 7; {NEW_UNIT_BOUND} 6 (10 minus 4 docstring lines at the base)."),
    ("cap-at-bound", 0, ""),
    ("cap-above-bound", 1,
     f"{gate.BASELINE}: cap 16; it must be at most 15 (20 minus 5, the docstring lines of "
     "rfx/b.py at the base)."),
    ("new-at-bound", 0, ""),
    ("new-above-bound", 1,
     f"rfx/b.py: baseline 16; {NEW_UNIT_BOUND} 15 (cap 20 minus 5 docstring lines at the base)."),
    ("new-absent-at-base", 1, "rfx/e.py: new entry 8, but the file is absent at the base"),
    ("new-within-cap", 1,
     "rfx/b.py: new entry 15, but its count 6 is within the cap 15; leave it unlisted."),
    ("delisted-present", 1,
     "rfx/a.py: entry removed while the file exists; with docstrings not counted the cap 15 "
     "must be at most 6 (10 minus 4 docstring lines at the base) to govern it, so keep the entry."),
    ("delisted-deleted", 0, ""),
    ("delisted-to-package", 1,
     "rfx/a.py: delisting moves the file to rfx/a/__init__.py, which gets the cap of 15 instead "
     "of the baseline 6 (10 minus 4 docstring lines at the base); keep rfx/a.py listed at most 6."),
])
def test_unit_change_bounds_come_from_the_base(tree, capsys, case, expected, message):
    """Base in code-lines: a.py listed at 10 with 4 docstring lines, cap 20, unlisted b.py
    with 5 and c.py with 2. The head drops docstring lines from a.py and b.py, so a bound
    read at the head would be looser than 6 (a.py), 15 (cap) and 15 (b.py)."""
    for name, data in (("a", A_BASE), ("b", B_BASE), ("c", C_BASE)):
        (tree / f"rfx/{name}.py").write_bytes(data)
    (tree / gate.BASELINE).write_text(json.dumps({
        "counts": "code-lines", "cap": 20, "files": {"rfx/a.py": 10},
    }))
    gate.git(tree, "init", "-q", "--template=")
    gate.git(tree, "config", "user.name", "Ratchet tests")
    gate.git(tree, "config", "user.email", "tests@example.invalid")
    gate.git(tree, "config", "commit.gpgsign", "false")
    gate.git(tree, "config", "core.hooksPath", "/dev/null")
    gate.git(tree, "add", ".")
    gate.git(tree, "commit", "-qm", "Base in code-lines")
    base = gate.git(tree, "rev-parse", "HEAD").decode().strip()

    (tree / "rfx/a.py").write_bytes(A_HEAD)
    (tree / "rfx/b.py").write_bytes(B_HEAD)
    files, cap = {"rfx/a.py": 6}, 15
    if case == "listed-above-bound":
        files["rfx/a.py"] = 7
    elif case == "cap-above-bound":
        cap = 16
    elif case in ("new-at-bound", "new-above-bound"):
        files["rfx/b.py"], cap = (15 if case == "new-at-bound" else 16), 3
    elif case == "new-absent-at-base":
        (tree / "rfx/e.py").write_bytes(b"".join(b"e%d = %d\n" % (i, i) for i in range(8)))
        files["rfx/e.py"], cap = 8, 6
    elif case == "new-within-cap":
        files["rfx/b.py"] = 15
    elif case.startswith("delisted"):
        del files["rfx/a.py"]
        if case in ("delisted-deleted", "delisted-to-package"):
            (tree / "rfx/a.py").unlink()
        if case == "delisted-to-package":
            (tree / "rfx/a").mkdir()
            (tree / "rfx/a/__init__.py").write_bytes(A_HEAD)
    (tree / gate.BASELINE).write_text(json.dumps({"counts": gate.UNIT, "cap": cap, "files": files}))
    code, err = run(tree, capsys, "--base", base)
    assert code == expected, err
    assert message in err
    assert "split the file" not in err


@pytest.mark.parametrize("case,expected,message", [
    ("package-unlisted", 1,
     "rfx/a.py: delisting moves the file to rfx/a/__init__.py, which gets the cap of 20 instead "
     "of the baseline 10; list rfx/a/__init__.py at most 10.\n"),
    ("package-listed", 0, ""),
    ("rename-unlisted", 1,
     "rfx/a.py: delisting moves the file to rfx/b.py, which gets the cap of 20 instead of the "
     "baseline 10; list rfx/b.py at most 10.\n"),
    ("rename-listed", 0, ""),
])
def test_delisted_file_moved_to_an_unlisted_path_fails(tree, capsys, case, expected, message):
    """Base lists a.py at 10 under cap 20. The head deletes a.py and continues it as its
    package form or as an unchanged rename, listing nothing or the new path at 10."""
    original = b"".join(f"line_{i:02d} = {i}\n".encode() for i in range(10))
    (tree / "rfx/a.py").write_bytes(original)
    (tree / gate.BASELINE).write_text(json.dumps({
        "counts": gate.UNIT, "cap": 20, "files": {"rfx/a.py": 10},
    }))
    gate.git(tree, "init", "-q", "--template=")
    gate.git(tree, "config", "user.name", "Ratchet tests")
    gate.git(tree, "config", "user.email", "tests@example.invalid")
    gate.git(tree, "config", "commit.gpgsign", "false")
    gate.git(tree, "config", "core.hooksPath", "/dev/null")
    gate.git(tree, "add", ".")
    gate.git(tree, "commit", "-qm", "Base baseline")
    base = gate.git(tree, "rev-parse", "HEAD").decode().strip()

    (tree / "rfx/a.py").unlink()
    if case.startswith("package"):
        target = "rfx/a/__init__.py"
        (tree / "rfx/a").mkdir()
        (tree / target).write_bytes(b"".join(f"new_{i:02d} = {i}\n".encode() for i in range(8)))
    else:
        target = "rfx/b.py"
        (tree / target).write_bytes(original)
    files = {target: 10} if case.endswith("-listed") else {}
    (tree / gate.BASELINE).write_text(json.dumps({"counts": gate.UNIT, "cap": 20, "files": files}))
    gate.git(tree, "add", "--all")
    code, err = run(tree, capsys, "--base", base)
    assert code == expected, err
    assert err.endswith(message)

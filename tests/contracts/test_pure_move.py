"""Offline CLI contracts, with hand-written expectations and two Git commits."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dev/check_pure_move.py"
OLD = '''
def run(x: "Coeffs") -> "Coeffs":
    return helper(x)

class Outer:
    class Inner:
        def method(self, x):
            return x + 1

def numeric(x):
    return x + 1

def imported(x):
    return x

def ordered(x):
    x += 1
    x *= 2
    return x

def attribute(x):
    return x.old

def keyword(x):
    return x(old=1)

def plain_string():
    return "old"

def imports():
    import oldmod as oldalias
    import pkg.oldmod
    from rfx.simulation import oldsymbol as oldlocal
    from rfx.simulation.extra import untouched
    return oldalias, oldlocal

class Before:
    class NestedBefore:
        async def method_before(self):
            def local_before():
                return Before
            return local_before()

def annotations(x: Optional["Coeffs"], /, *args: "Coeffs", y: "Coeffs", **kw: "Coeffs") -> list["Coeffs"]:
    local: Optional["Coeffs"] = x
    def inner(value: "Coeffs") -> "Coeffs":
        nested: "Coeffs" = value
        return nested
    return inner(local)

async def async_annotations(x: "Coeffs") -> "Coeffs":
    return x

def invalid_annotation(x: "not valid !"):
    return x

def chain(a, b):
    return a, b

def outer_function():
    def inaccessible():
        return 1
    return inaccessible()

def details(x=1):
    """original"""
    return x

def u_literal():
    return u"value"
'''

NEW = '''
# Moving and reformatting deliberately change line/column attributes.


def moved(x: "NewCoeffs") -> "NewCoeffs":
    return new_helper( x )

def unquoted(x: NewCoeffs) -> NewCoeffs:
    return new_helper(x)

def method(self, x):
    return x + 1

def numeric(x):
    return x + 2

def imported(x):
    import math
    return x

def ordered(x):
    x *= 2
    x += 1
    return x

def attribute(x):
    return x.new

def keyword(x):
    return x(new=1)

def plain_string():
    return "new"

def imports():
    import newmod as newalias
    import pkg.oldmod
    from rfx.stepping.uniform_loop import newsymbol as newlocal
    from rfx.simulation.extra import untouched
    return newalias, newlocal

class After:
    class NestedAfter:
        async def method_after(self):
            def local_after():
                return After
            return local_after()

def annotations(x: Optional["NewCoeffs"], /, *args: "NewCoeffs", y: "NewCoeffs", **kw: "NewCoeffs") -> list["NewCoeffs"]:
    local: Optional["NewCoeffs"] = x
    def inner(value: "NewCoeffs") -> "NewCoeffs":
        nested: "NewCoeffs" = value
        return nested
    return inner(local)

async def async_annotations(x: "NewCoeffs") -> "NewCoeffs":
    return x

def invalid_annotation(x: "not valid !"):
    return x

def changed_invalid_annotation(x: "still invalid !"):
    return x

def chain(a, b):
    return b, c

def details_docstring(x=1):
    """changed"""
    return x

def details_default(x=2):
    """original"""
    return x

@decorator
def details_decorator(x=1):
    """original"""
    return x

def u_literal():
    return "value"
'''

LISTING = '''
ONE = TWO = 1
typed: int = 2
(tuple_left, tuple_right) = (1, 2)
def kept():
    return 1
def moved():
    def local():
        return 1
    return local()
def lost():
    return 2
class Lost:
    class Inner:
        def method(self):
            pass
class Covered:
    class Inner:
        def method(self):
            pass
'''


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("pure-move")
    work = root / "work"
    work.mkdir()
    metadata = root / "git-data"
    (metadata / "objects").mkdir(parents=True)
    (metadata / "refs" / "heads").mkdir(parents=True)
    (metadata / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (metadata / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n", encoding="utf-8"
    )
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("GIT_"):
            env.pop(key)
    env.update(
        GIT_DIR=str(metadata), GIT_WORK_TREE=str(work),
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="Pure Move Test", GIT_AUTHOR_EMAIL="test@example.invalid",
        GIT_COMMITTER_NAME="Pure Move Test", GIT_COMMITTER_EMAIL="test@example.invalid",
        GIT_TERMINAL_PROMPT="0",
    )

    def git(*args):
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=" + os.devnull, *args],
            cwd=work, env=env, capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    for path, source in {
        "old.py": OLD, "listing.py": LISTING, "deleted.py": LISTING,
        "unchanged.py": "def keep():\n    return 1\n",
        "diff.py": "def many():\n" + "    x = 1\n" * 30 + "    return x\n",
        "hunks.py": "def hunks():\n    a = 1\n    middle = 0\n    b = 2\n",
    }.items():
        (work / path).write_text(source, encoding="utf-8")
    git("add", "--all")
    git("commit", "-m", "Record base units")
    base = git("rev-parse", "HEAD")
    (work / "old.py").write_text("# moved\n", encoding="utf-8")
    (work / "new.py").write_text(NEW, encoding="utf-8")
    (work / "listing.py").write_text("def kept():\n    return 1\n", encoding="utf-8")
    (work / "deleted.py").unlink()
    (work / "destination.py").write_text(
        "def moved():\n    def local():\n        return 1\n    return local()\n"
        "class Covered:\n    class Inner:\n        def method(self):\n            pass\n",
        encoding="utf-8",
    )
    (work / "diff.py").write_text(
        "def many():\n" + "    x = 2\n" * 30 + "    return x\n", encoding="utf-8"
    )
    (work / "hunks.py").write_text(
        "def hunks():\n    a = 3\n    middle = 0\n    b = 4\n", encoding="utf-8"
    )
    git("add", "--all")
    git("commit", "-m", "Move and change units")
    return work, env, base, git("rev-parse", "HEAD")


def run(repo, *args, cwd=None):
    work, env, base, head = repo
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base", base, "--head", head, *args],
        cwd=cwd or work, env=env, capture_output=True, text=True, timeout=5,
    )


def test_moved_function_with_string_annotation_rename(repo):
    result = run(repo, "--pair", "old.py:run=new.py:moved",
                 "--rename", "helper=new_helper", "--rename", "Coeffs=NewCoeffs")
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.splitlines()[0] == "old.py:run -> new.py:moved  IDENTICAL"


def test_annotation_name_absent_from_map(repo):
    result = run(repo, "--pair", "old.py:run=new.py:moved", "--rename", "helper=new_helper")
    assert result.returncode == 1
    assert result.stdout.splitlines()[0] == "old.py:run -> new.py:moved  DIFFERENT"


def test_quoted_annotation_equals_unquoted(repo):
    result = run(repo, "--pair", "old.py:run=new.py:unquoted",
                 "--rename", "helper=new_helper", "--rename", "Coeffs=NewCoeffs")
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.splitlines()[0] == "old.py:run -> new.py:unquoted  IDENTICAL"


def test_nested_class_method_moved_to_function(repo):
    result = run(repo, "--pair", "old.py:Outer.Inner.method=new.py:method")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "old.py:Outer.Inner.method -> new.py:method  IDENTICAL"


@pytest.mark.parametrize("name", ["numeric", "imported", "ordered", "attribute", "keyword", "plain_string"])
def test_changes_are_different(repo, name):
    result = run(repo, "--pair", f"old.py:{name}=new.py:{name}", "--rename", "old=new")
    assert result.returncode == 1, result.stderr + result.stdout
    assert result.stdout.splitlines()[0] == f"old.py:{name} -> new.py:{name}  DIFFERENT"
    assert result.stdout.splitlines()[1].startswith("@@")


def test_exact_import_names_and_aliases(repo):
    result = run(repo, "--pair", "old.py:imports=new.py:imports",
                 "--rename", "oldmod=newmod", "--rename", "oldalias=newalias",
                 "--rename", "rfx.simulation=rfx.stepping.uniform_loop",
                 "--rename", "oldsymbol=newsymbol", "--rename", "oldlocal=newlocal")
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.splitlines()[0] == "old.py:imports -> new.py:imports  IDENTICAL"


def test_all_nested_definition_names_and_overlapping_pairs(repo):
    result = run(repo, "--pair", "old.py:Before=new.py:After",
                 "--pair", "old.py:Before.NestedBefore=new.py:After.NestedAfter",
                 "--rename", "method_before=method_after", "--rename", "local_before=local_after")
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.splitlines()[:2] == [
        "old.py:Before -> new.py:After  IDENTICAL",
        "old.py:Before.NestedBefore -> new.py:After.NestedAfter  IDENTICAL",
    ]


@pytest.mark.parametrize("name", ["annotations", "async_annotations"])
def test_annotation_positions_and_nested_strings(repo, name):
    result = run(repo, "--pair", f"old.py:{name}=new.py:{name}", "--rename", "Coeffs=NewCoeffs")
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.splitlines()[0] == f"old.py:{name} -> new.py:{name}  IDENTICAL"


@pytest.mark.parametrize("new_name, verdict, code", [
    ("invalid_annotation", "IDENTICAL", 0), ("changed_invalid_annotation", "DIFFERENT", 1),
])
def test_unparsable_string_annotation_stays_literal(repo, new_name, verdict, code):
    result = run(repo, "--pair", f"old.py:invalid_annotation=new.py:{new_name}")
    assert result.returncode == code, result.stderr
    assert result.stdout.splitlines()[0] == f"old.py:invalid_annotation -> new.py:{new_name}  {verdict}"


def test_renames_are_simultaneous_and_duplicates_allowed(repo):
    result = run(repo, "--pair", "old.py:chain=new.py:chain", "--rename", "a=b",
                 "--rename", "b=c", "--rename", "a=b")
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.splitlines()[0] == "old.py:chain -> new.py:chain  IDENTICAL"


def test_pairs_file_and_implied_renames(repo, tmp_path):
    pairs = tmp_path / "pairs.txt"
    pairs.write_text(
        "# pairs and renames\n\nold.py:run=new.py:moved # inline comment\n"
        "rename Coeffs=NewCoeffs\nrename helper=new_helper\n"
        "old.py:Outer.Inner.method=new.py:method\n", encoding="utf-8",
    )
    result = run(repo, "--pairs-file", str(pairs), "--rename", "run=moved")
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.splitlines()[:2] == [
        "old.py:run -> new.py:moved  IDENTICAL",
        "old.py:Outer.Inner.method -> new.py:method  IDENTICAL",
    ]


@pytest.mark.parametrize("extra", [
    ["--rename", "run=other"], ["--pair", "old.py:run=new.py:unquoted"],
])
def test_conflicting_renames_are_usage_errors(repo, extra):
    result = run(repo, "--pair", "old.py:run=new.py:moved", *extra)
    assert result.returncode == 2
    assert "conflicting rename for run: moved and " in result.stderr
    assert ("other" if extra[0] == "--rename" else "unquoted") in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("pair, missing", [
    ("old.py:numeric=new.py:absent", "new"),
    ("old.py:absent=new.py:numeric", "old"),
    ("old.py:absent=new.py:absent", "both"),
    ("absent.py:numeric=new.py:numeric", "old"),
    ("old.py:numeric=absent.py:numeric", "new"),
    ("absent.py:numeric=absent.py:numeric", "both"),
    ("old.py:outer_function.inaccessible=new.py:numeric", "old"),
])
def test_missing_units_and_files(repo, pair, missing):
    result = run(repo, "--pair", pair)
    assert result.returncode == 1, result.stderr
    assert result.stdout.splitlines()[0].endswith(f"  MISSING ({missing})")


@pytest.mark.parametrize("path", ["listing.py", "deleted.py"])
def test_deleted_not_covered_including_deleted_file(repo, path):
    result = run(repo, "--pair", f"{path}:moved=destination.py:moved",
                 "--pair", f"{path}:Covered=destination.py:Covered")
    assert result.returncode == 0, result.stderr + result.stdout
    expected = [
        f"{path}:moved -> destination.py:moved  IDENTICAL",
        f"{path}:Covered -> destination.py:Covered  IDENTICAL",
        "deleted at HEAD, not in any pair:",
        f"{path}:Lost", f"{path}:Lost.Inner", f"{path}:Lost.Inner.method",
        f"{path}:ONE", f"{path}:TWO",
    ]
    if path == "deleted.py":
        expected.append("deleted.py:kept")
    expected.extend([f"{path}:lost", f"{path}:typed"])
    assert result.stdout.splitlines() == expected


def test_empty_deletion_listing_and_repo_subdirectory(repo):
    subdir = repo[0] / "subdir"
    subdir.mkdir(exist_ok=True)
    result = run(repo, "--pair", "unchanged.py:keep=unchanged.py:keep", cwd=subdir)
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "unchanged.py:keep -> unchanged.py:keep  IDENTICAL\n"
        "deleted at HEAD, not in any pair: none\n"
    )


@pytest.mark.parametrize("args", [
    [], ["--pair", "bad"], ["--pair", "old.py:=new.py:x"],
    ["--pair", "old.py:run=new.py:moved", "--rename", "bad"],
    ["--pairs-file", "/nonexistent-pure-move-pairs"],
])
def test_usage_errors(repo, args):
    result = run(repo, *args)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_git_invalid_ref_is_read_error(repo):
    result = run(repo, "--head", "does-not-exist", "--pair", "old.py:run=new.py:moved")
    assert result.returncode == 2
    assert "git read error:" in result.stderr
    assert "MISSING" not in result.stdout


def test_git_show_failure_is_read_error(repo, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("pure_move_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_run = subprocess.run

    def fail_show(command, **kwargs):
        if command[:2] == ["git", "show"]:
            return subprocess.CompletedProcess(command, 128, "", "forced git show failure")
        return real_run(command, **kwargs)

    monkeypatch.chdir(repo[0])
    for key, value in repo[1].items():
        if key.startswith("GIT_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(module.subprocess, "run", fail_show)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--base", repo[2], "--head", repo[3],
                                     "--pair", "old.py:run=new.py:moved"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "git read error: forced git show failure" in capsys.readouterr().err


@pytest.mark.parametrize("new_name", ["details_docstring", "details_default", "details_decorator"])
def test_docstrings_defaults_decorators_count(repo, new_name):
    result = run(repo, "--pair", f"old.py:details=new.py:{new_name}")
    assert result.returncode == 1
    assert result.stdout.splitlines()[0].endswith("  DIFFERENT")


def test_literal_kind_counts_even_when_unparse_matches(repo):
    result = run(repo, "--pair", "old.py:u_literal=new.py:u_literal")
    assert result.returncode == 1
    assert result.stdout.splitlines()[0] == "old.py:u_literal -> new.py:u_literal  DIFFERENT"


def test_diff_is_limited_to_twenty_lines(repo):
    result = run(repo, "--pair", "diff.py:many=diff.py:many")
    assert result.returncode == 1
    assert result.stdout.splitlines()[1:] == [
        "@@ -2,30 +2,30 @@", *["-    x = 1"] * 19,
        "deleted at HEAD, not in any pair: none",
    ]


def test_only_first_diff_hunk_is_printed(repo):
    result = run(repo, "--pair", "hunks.py:hunks=hunks.py:hunks")
    assert result.returncode == 1
    assert result.stdout.splitlines()[1:] == [
        "@@ -2 +2 @@", "-    a = 1", "+    a = 3",
        "deleted at HEAD, not in any pair: none",
    ]


def test_multiple_pairs_keep_nonzero_exit(repo):
    result = run(repo, "--pair", "old.py:numeric=new.py:numeric",
                 "--pair", "unchanged.py:keep=unchanged.py:keep")
    assert result.returncode == 1
    assert "unchanged.py:keep -> unchanged.py:keep  IDENTICAL" in result.stdout

"""Exercise the comment gate against real Git trees and working-tree edits."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/ci/check_comment_content.py"
GUIDANCE = "a comment says what the code does, not where it came from"


@pytest.fixture
def git_env():
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_") and key != "BASE_SHA"}
    env.pop("GITHUB_EVENT_NAME", None)
    env.update(
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="Contract", GIT_AUTHOR_EMAIL="contract@example.invalid",
        GIT_COMMITTER_NAME="Contract", GIT_COMMITTER_EMAIL="contract@example.invalid",
        GIT_TERMINAL_PROMPT="0",
    )
    return env


def git(repo, env, *args, check=True):
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        cwd=repo, env=env, capture_output=True, text=True, timeout=30,
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def commit(repo, env):
    git(repo, env, "add", "--all")
    git(repo, env, "commit", "-m", "fixture")
    return git(repo, env, "rev-parse", "HEAD").stdout.strip()


def make_repo(tmp_path, env, content="value = 0\n"):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, env, "init", "--initial-branch=main", "--template=")
    (repo / "mod.py").write_text(content, encoding="utf-8")
    return repo, commit(repo, env)


def check(repo, env, *args, swept=()):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--swept", *swept], cwd=repo, env=env,
        capture_output=True, text=True, timeout=30,
    )


def summary(repo, env, base, refused, added):
    short = git(repo, env, "rev-parse", "--short", base).stdout.strip()
    return f"comment content: {refused} refused on {added} added lines (base {short})"


def test_only_added_comment_lines_are_checked(tmp_path, git_env):
    original = ['# see #777', 'label = "original"']
    repo, base = make_repo(tmp_path, git_env, "\n".join(original) + "\n")
    additions = ['x = "#123"', '# fixes #123']
    (repo / "mod.py").write_text("\n".join(original + additions) + "\n", encoding="utf-8")
    commit(repo, git_env)
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        f"mod.py:{len(original) + len(additions)}: issue number: {additions[-1]}"
        "  -> the PR body or the issue; a strict-xfail reason (a string) may carry it; " + GUIDANCE,
        summary(repo, git_env, base, 1, len(additions)),
    ]
    (repo / "mod.py").write_text("\n".join(original + additions[:1]) + "\n", encoding="utf-8")
    commit(repo, git_env)
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, 1)]


@pytest.mark.parametrize("comment,rule,destination", [
    ("# 2026-11-17", "date", "a dated record, or git history"),
    ("# 1999/12/31", "date", "a dated record, or git history"),
    ("# 2026.01.02", "date", "a dated record, or git history"),
    ("# 20261117", "date", "a dated record, or git history"),
    ("# see #123", "issue number", "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("# see https://github.com/example/repo/issues/45", "issue number", "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("# https://github.com/example/repo/pull/2809", "issue number", "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("# rfx#2809", "issue number", "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("# see org/other-repo#123", "issue number", "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("# PR #12", "issue number", "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("# PR#7", "issue number", "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("# pr #3", "issue number", "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("# PREDECLARE", "predeclaration", "the predeclaration file"),
    ("# predeclaration", "predeclaration", "the predeclaration file"),
    ("# MoVeD FrOm mod.py", "move note", "git history (git log --follow)"),
    ("# 1234567890", "run id", "the measurement record"),
    ("# remi-123456789012", "run id", "the measurement record"),
    ("# from run 1234567890.", "run id", "the measurement record"),
    ('# "\uaddc\uce59"', "quoted Korean", "a named test or assertion states the rule; the rationale goes to the record"),
    ("# '\uaddc\uce59'", "quoted Korean", "a named test or assertion states the rule; the rationale goes to the record"),
    ("# “\uaddc\uce59”", "quoted Korean", "a named test or assertion states the rule; the rationale goes to the record"),
    ("# ‘\uaddc\uce59’", "quoted Korean", "a named test or assertion states the rule; the rationale goes to the record"),
    ("# 「\u1100」", "quoted Korean", "a named test or assertion states the rule; the rationale goes to the record"),
    ("# 『\u3131』", "quoted Korean", "a named test or assertion states the rule; the rationale goes to the record"),
])
def test_refused_comment_rules(tmp_path, git_env, comment, rule, destination):
    repo, base = make_repo(tmp_path, git_env)
    (repo / "mod.py").write_text(f"value = 0\n{comment}\n", encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"mod.py:2: {rule}: {comment}  -> {destination}; {GUIDANCE}",
        summary(repo, git_env, base, 1, 1),
    ]


@pytest.mark.parametrize("addition", [
    "# preview", "# 299792458 m/s", "# step 3", "# Compute the offset",
    "# case #1", "# colour #1f77b4",
    "# step #12", "# &#123;", "# .s2fix-review/main", "# review the offset", "# ReViEWed",
    "# dt=.5000510204081633", "# eps0 = 8.8541878128e-12 F/m", "# pi = 3.14159265358979", "# mu0 = 1.25663706212e-6",
    "# \uaddc\uce59", '# "offset" \uaddc\uce59', "# 2026-13-17", "# 20260017",
    'value = "#123 review PREDECLARE moved from 2026-11-17 1234567890"',
    '"""#123 review \uaddc\uce59"""', 'value = f"#123 {1}"',
])
def test_near_misses_and_strings_pass(tmp_path, git_env, addition):
    repo, base = make_repo(tmp_path, git_env)
    (repo / "mod.py").write_text(f"value = 0\n{addition}\n", encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, 1)]


def merge_repo(tmp_path, env):
    repo, ancestor = make_repo(tmp_path, env)
    git(repo, env, "checkout", "-b", "topic")
    (repo / "mod.py").write_text('value = 0\nx = "#123"\n# fixes #123\n', encoding="utf-8")
    topic = commit(repo, env)
    git(repo, env, "checkout", "main")
    (repo / "base.txt").write_text("base tip\n", encoding="utf-8")
    first = commit(repo, env)
    git(repo, env, "merge", "--no-ff", "topic", "-m", "fixture merge")
    return repo, ancestor, first, topic


def shallow_clone(tmp_path, env):
    repo, ancestor, first, topic = merge_repo(tmp_path, env)
    clone = tmp_path / "clone"
    git(tmp_path, env, "clone", "--depth", "1", repo.as_uri(), str(clone))
    assert git(clone, env, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    header = git(clone, env, "cat-file", "-p", "HEAD").stdout.split("\n\n", 1)[0]
    assert [line for line in header.splitlines() if line.startswith("parent ")] == [
        f"parent {first}", f"parent {topic}",
    ]
    assert len(git(clone, env, "rev-list", "--parents", "-1", "HEAD").stdout.split()) == 1
    for revision in (ancestor, first, topic):
        assert git(clone, env, "cat-file", "-e", revision, check=False).returncode != 0
    return repo, clone, ancestor, first, topic


def test_shallow_merge_fetches_only_base_and_diff_works(tmp_path, git_env):
    git_env["GITHUB_EVENT_NAME"] = "pull_request"
    repo, clone, ancestor, first, topic = shallow_clone(tmp_path, git_env)
    expected = check(repo, git_env)
    assert expected.returncode == 1, expected.stdout + expected.stderr
    result = check(clone, git_env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout == expected.stdout
    assert result.stderr == ""
    assert "mod.py:3: issue number: # fixes #123" in result.stdout
    assert result.stdout.splitlines()[-1] == summary(clone, git_env, first, 1, 2)
    assert git(clone, git_env, "cat-file", "-e", first).returncode == 0
    for revision in (ancestor, topic):
        assert git(clone, git_env, "cat-file", "-e", revision, check=False).returncode != 0
    shallow = (clone / ".git/shallow").read_text().splitlines()
    assert first in shallow
    patch = git(clone, git_env, "diff", "-U0", "-M", "--diff-filter=AMR", first, "--", "*.py").stdout
    assert [line for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")] == [
        '+x = "#123"', '+# fixes #123',
    ]


def test_shallow_merge_without_origin_fails(tmp_path, git_env):
    git_env["GITHUB_EVENT_NAME"] = "pull_request"
    _, clone, _, first, _ = shallow_clone(tmp_path, git_env)
    git(clone, git_env, "remote", "remove", "origin")
    result = check(clone, git_env)
    assert result.returncode == 2
    assert result.stdout == ""
    assert f"fetch --no-tags --depth=1 origin {first}" in result.stderr


@pytest.mark.parametrize("broken", ['value = (\n', '"""unfinished\n', "value = 'unfinished\n"])
def test_tokenization_failure_names_file(tmp_path, git_env, broken):
    repo, base = make_repo(tmp_path, git_env)
    (repo / "mod.py").write_text(broken, encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "mod.py: cannot tokenize" in result.stderr


def test_explicit_base_and_environment_use_merge_base(tmp_path, git_env):
    git_env["GITHUB_EVENT_NAME"] = "pull_request"
    repo, ancestor, first, topic = merge_repo(tmp_path, git_env)
    git(repo, git_env, "checkout", "topic")
    for args, env in [(("--base", first), git_env), ((), dict(git_env, BASE_SHA=first)),
                      (("--base", first), dict(git_env, BASE_SHA=topic))]:
        result = check(repo, env, *args)
        assert result.returncode == 1, result.stdout + result.stderr
        assert result.stdout.splitlines()[-1] == summary(repo, git_env, ancestor, 1, 2)


def test_explicit_base_overrides_merge_parent(tmp_path, git_env):
    git_env["GITHUB_EVENT_NAME"] = "pull_request"
    repo, _, _, topic = merge_repo(tmp_path, git_env)
    result = check(repo, git_env, "--base", topic)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, topic, 0, 0)]


def test_origin_main_uses_merge_base_before_single_parent(tmp_path, git_env):
    repo, ancestor, first, _ = merge_repo(tmp_path, git_env)
    git(repo, git_env, "checkout", "topic")
    (repo / "mod.py").write_text('value = 0\nx = "#123"\n# fixes #123\ny = 1\n', encoding="utf-8")
    commit(repo, git_env)
    git(repo, git_env, "update-ref", "refs/remotes/origin/main", first)
    result = check(repo, git_env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == summary(repo, git_env, ancestor, 1, 3)


@pytest.mark.parametrize("event", [None, "push"])
def test_origin_main_at_head_checks_landed_commit(tmp_path, git_env, event):
    if event is not None:
        git_env["GITHUB_EVENT_NAME"] = event
    original = ["value = 0"]
    addition = "# fixes #123"
    repo, base = make_repo(tmp_path, git_env, "\n".join(original) + "\n")
    (repo / "mod.py").write_text("\n".join(original + [addition]) + "\n", encoding="utf-8")
    head = commit(repo, git_env)
    git(repo, git_env, "update-ref", "refs/remotes/origin/main", head)
    result = check(repo, git_env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        f"mod.py:{len(original) + 1}: issue number: {addition}"
        "  -> the PR body or the issue; a strict-xfail reason (a string) may carry it; " + GUIDANCE,
        summary(repo, git_env, base, 1, 1),
    ]


@pytest.mark.parametrize("event", [None, "pull_request", "pull_request_target", "merge_group"])
def test_branch_merging_origin_main_selects_base_by_event(tmp_path, git_env, event):
    if event is not None:
        git_env["GITHUB_EVENT_NAME"] = event
    repo, _ = make_repo(tmp_path, git_env)
    branch_comment = "# fixes #123"
    main_comment = "# fixes #456"
    git(repo, git_env, "checkout", "-b", "topic")
    (repo / "branch.py").write_text(branch_comment + "\n", encoding="utf-8")
    first = commit(repo, git_env)
    git(repo, git_env, "checkout", "main")
    (repo / "main.py").write_text(main_comment + "\n", encoding="utf-8")
    main = commit(repo, git_env)
    git(repo, git_env, "update-ref", "refs/remotes/origin/main", main)
    git(repo, git_env, "checkout", "topic")
    git(repo, git_env, "merge", "--no-ff", "origin/main", "-m", "fixture merge")
    if event is None:
        base, path, comment = main, "branch.py", branch_comment
    else:
        base, path, comment = first, "main.py", main_comment
    result = check(repo, git_env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        f"{path}:1: issue number: {comment}"
        "  -> the PR body or the issue; a strict-xfail reason (a string) may carry it; " + GUIDANCE,
        summary(repo, git_env, base, 1, 1),
    ]


def test_single_parent_fallback_includes_working_tree(tmp_path, git_env):
    repo, base = make_repo(tmp_path, git_env)
    (repo / "mod.py").write_text("value = 0\nx = 1\n", encoding="utf-8")
    commit(repo, git_env)
    (repo / "mod.py").write_text("value = 0\nx = 1\n# fixes #123\n", encoding="utf-8")
    result = check(repo, git_env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == summary(repo, git_env, base, 1, 2)


@pytest.mark.parametrize("event", [None, "pull_request", "pull_request_target", "merge_group"])
def test_root_without_base_fails(tmp_path, git_env, event):
    if event is not None:
        git_env["GITHUB_EVENT_NAME"] = event
    repo, _ = make_repo(tmp_path, git_env)
    result = check(repo, git_env)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "no parent" in result.stderr


@pytest.mark.parametrize("use_origin", [False, True])
def test_failed_merge_base_is_an_error(tmp_path, git_env, use_origin):
    repo, unrelated = make_repo(tmp_path, git_env)
    git(repo, git_env, "checkout", "--orphan", "separate")
    (repo / "mod.py").write_text("value = 1\n", encoding="utf-8")
    commit(repo, git_env)
    if use_origin:
        git(repo, git_env, "update-ref", "refs/remotes/origin/main", unrelated)
    result = check(repo, git_env, *(() if use_origin else ("--base", unrelated)))
    assert result.returncode == 2
    assert result.stdout == ""
    assert "merge-base" in result.stderr


@pytest.mark.parametrize("name", ["renamed.py", "space name.py", 'quote"name.py', "\ud55c\uae00.py", "tab\tname.py"])
def test_rename_and_changed_inline_comment(tmp_path, git_env, name):
    repo, base = make_repo(tmp_path, git_env, "# see #77\n" + "".join(f"x{i} = {i}\n" for i in range(20)))
    git(repo, git_env, "mv", "mod.py", name)
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, 0)]
    path = repo / name
    path.write_text(path.read_text().replace("x5 = 5", "x5 = 5  # fixes #123"), encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.startswith(f"{name}:7: issue number: # fixes #123")
    assert result.stdout.splitlines()[-1] == summary(repo, git_env, base, 1, 1)


def test_moved_comment_counts_as_added(tmp_path, git_env):
    body = "".join(f"x{i} = {i}\n" for i in range(20))
    comment = "# fixes #123\n"
    repo, base = make_repo(tmp_path, git_env, comment + body)
    (repo / "mod.py").write_text(body + comment, encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.startswith("mod.py:21: issue number: # fixes #123")
    assert result.stdout.splitlines()[-1] == summary(repo, git_env, base, 1, 1)


def test_encoding_cookie_added_file_and_deleted_file(tmp_path, git_env):
    repo, base = make_repo(tmp_path, git_env, "# see #77\n")
    (repo / "mod.py").unlink()
    (repo / "encoded.py").write_bytes(b"# coding: latin-1\n# caf\xe9\nx = '#123'\n")
    (repo / "notes.txt").write_text("# fixes #123\n", encoding="utf-8")
    git(repo, git_env, "add", "--all")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, 3)]


def test_multiple_matches_emit_one_line_per_comment(tmp_path, git_env):
    repo, base = make_repo(tmp_path, git_env)
    (repo / "mod.py").write_text("value = 0\n# PREDECLARE #123\n# fixes #456\n", encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(result.stdout.splitlines()) == 3
    assert result.stdout.splitlines()[-1] == summary(repo, git_env, base, 2, 2)


FULL_BAN = (
    "a name, an assertion, a test, the docstring, or a dated record; under rfx/, tests/, examples/ and scripts/"
    " a comment may only be a tool directive (noqa, type:, pragma:, fmt:, isort:, ruff:, mypy:, pyright:, pylint:)"
)
PROSE = "# scale the offset by the cell size"


def add_file(repo, env, path, content):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(repo, env, "add", "--all")


@pytest.mark.parametrize("path", [
    "rfx/a.py", "tests/a.py", "examples/a.py", "scripts/a.py", "rfx/tests/nested.py",
])
def test_prose_comment_is_refused_in_full_ban_directories(tmp_path, git_env, path):
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, path, f"value = 0\nother = 1\n{PROSE}\n")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        f"{path}:3: comment: {PROSE}  -> {FULL_BAN}",
        summary(repo, git_env, base, 1, 3),
    ]


@pytest.mark.parametrize("path", [
    "docs/a.py", "a.py", "validation/a.py", "rfxlib/a.py", "Rfx/a.py", "src/tests/a.py",
])
def test_prose_comment_passes_outside_full_ban_directories(tmp_path, git_env, path):
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, path, f"value = 0\nother = 1\n{PROSE}\n")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, 3)]


def test_existing_prose_comment_on_unchanged_line_is_not_reported(tmp_path, git_env):
    repo, _ = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "rfx/a.py", "# an old prose comment\nvalue = 0\n")
    base = commit(repo, git_env)
    (repo / "rfx/a.py").write_text(
        "# an old prose comment\nvalue = 0\nother = 1\n# a new prose comment\n", encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"rfx/a.py:4: comment: # a new prose comment  -> {FULL_BAN}",
        summary(repo, git_env, base, 1, 2),
    ]


@pytest.mark.parametrize("line", [
    "# noqa", "# noqa: E402", "# noqa: BLE001", "# noqa: E402,F401", "# noqa: E402, F401", "# noqa: E402 F401",
    "#noqa", "#  noqa: E501", "# noqa:E501", "# NOQA: E402",
    "# flake8: noqa",
    "# type: ignore", "# type: ignore[attr-defined]", "# type: ignore[arg-type, union-attr]",
    "# type:ignore", "# type: ignore[ a , b ]",
    "# pragma: no cover", "# pragma: no branch",
    "# fmt: off", "# fmt: on", "# fmt: skip", "# fmt:off",
    "# isort: skip_file", "# isort: skip", "# isort: off", "# isort: on", "# isort: split",
    "# ruff: noqa", "# ruff: noqa: E402", "# ruff: noqa: E402, F401", "# ruff: isort: skip_file",
    "# mypy: ignore-errors", "# mypy: disable-error-code=attr-defined,arg-type",
    '# mypy: disable-error-code="attr-defined"', "# mypy: strict, warn-unused-ignores=True",
    '# mypy: disable-error-code="attr-defined, misc"',
    "# pyright: basic", "# pyright: standard", "# pyright: strict",
    "# pyright: ignore", "# pyright: ignore[reportPrivateUsage]",
    "# pyright: reportPrivateUsage=false",
    "# pyright: reportPrivateUsage=false, reportMissingImports=none",
    "# pylint: disable=unused-import", "# pylint: enable=too-many-locals,invalid-name",
    "# pylint: disable-next=broad-except", "# pylint: skip-file",
    "# type: ignore[attr-defined]  # noqa: E501", "# noqa: E402 # pragma: no cover",
    "import os  # noqa: F401", "value = 1  # type: ignore[assignment]  ",
])
def test_tool_directives_pass_in_full_ban_directories(tmp_path, git_env, line):
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "rfx/a.py", f"value = 0\nother = 1\n{line}\n")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, 3)]


@pytest.mark.parametrize("comment", [
    "#", "#:", "# %%", "# ---- setup ----", "# ==========", "# value = compute(1)",
    "# noqa: E501 long URL", "# noqa E501", "# noqa: e501",
    "# type: this is a note", "# type: List[int]", "# type: (int) -> str", "# type: not checked",
    "# pragma: no cover - reason", "# mypy: ignore-errors because", "# pylint: disable=x  why",
    "# noqa: E402,F401  (re-export)", "# noqa: E501#type: ignore",
    '# mypy: note="scale the offset by the cell size"', '# pyright: reportX="tuned by hand"',
    "# Type: ignore", "# PRAGMA: no cover", "# fmt: OFF",
])
def test_comments_other_than_directives_are_refused(tmp_path, git_env, comment):
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "rfx/a.py", f"value = 0\nother = 1\n{comment}\n")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"rfx/a.py:3: comment: {comment}  -> {FULL_BAN}",
        summary(repo, git_env, base, 1, 3),
    ]


@pytest.mark.parametrize("content,added", [
    ("#!/usr/bin/env python3\nvalue = 0\n", 2),
    ("#!/usr/bin/env python3\n# coding: utf-8\nvalue = 0\n", 3),
    ("# -*- coding: utf-8 -*-\nvalue = 0\n", 2),
    ("value = 0\n# vim: set fileencoding=utf-8 :\n", 2),
])
def test_shebang_and_coding_lines_pass_on_their_lines(tmp_path, git_env, content, added):
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "scripts/run.py", content)
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, added)]


@pytest.mark.parametrize("comment", [
    "#!/usr/bin/env python3", "# coding: utf-8", "# -*- coding: utf-8 -*-",
])
def test_shebang_and_coding_lines_are_refused_from_line_three(tmp_path, git_env, comment):
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "scripts/run.py", f"value = 0\nother = 1\n{comment}\n")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"scripts/run.py:3: comment: {comment}  -> {FULL_BAN}",
        summary(repo, git_env, base, 1, 3),
    ]


@pytest.mark.parametrize("line,comment", [
    ("value = 0  #! scale the offset", "#! scale the offset"),
    ("x = 1  # coding: utf-8 then prose", "# coding: utf-8 then prose"),
])
def test_trailing_shebang_or_coding_comment_on_line_one_is_refused(tmp_path, git_env, line, comment):
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "rfx/a.py", f"{line}\nother = 1\n")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"rfx/a.py:1: comment: {comment}  -> {FULL_BAN}",
        summary(repo, git_env, base, 1, 2),
    ]


def test_trailing_comment_on_changed_code_line_is_refused(tmp_path, git_env):
    repo, _ = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "tests/t.py", "".join(f"x{i} = {i}\n" for i in range(20)))
    base = commit(repo, git_env)
    path = repo / "tests/t.py"
    path.write_text(path.read_text().replace("x5 = 5", "x5 = 6  # tuned by hand"), encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"tests/t.py:6: comment: # tuned by hand  -> {FULL_BAN}",
        summary(repo, git_env, base, 1, 1),
    ]


def test_full_ban_comment_matching_a_narrow_rule_names_that_rule(tmp_path, git_env):
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "rfx/a.py", "value = 0\nother = 1\n# see #1234\n")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        "rfx/a.py:3: issue number: # see #1234"
        "  -> the PR body or the issue; a strict-xfail reason (a string) may carry it; " + GUIDANCE,
        summary(repo, git_env, base, 1, 3),
    ]


def test_directive_is_final_in_full_ban_directories(tmp_path, git_env):
    directive = "# noqa: A1234567890"
    repo, base = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "rfx/a.py", f"value = 0\nother = 1\n{directive}\n")
    add_file(repo, git_env, "mod.py", f"value = 0\nother = 1\n{directive}\n")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"mod.py:3: run id: {directive}  -> the measurement record; " + GUIDANCE,
        summary(repo, git_env, base, 1, 5),
    ]


def test_file_moved_into_full_ban_directory_judges_only_edited_lines(tmp_path, git_env):
    repo, _ = make_repo(tmp_path, git_env)
    add_file(repo, git_env, "docs/m.py", "# an old prose comment\n" + "".join(f"x{i} = {i}\n" for i in range(20)))
    base = commit(repo, git_env)
    (repo / "rfx").mkdir()
    git(repo, git_env, "mv", "docs/m.py", "rfx/m.py")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, 0)]
    path = repo / "rfx/m.py"
    path.write_text(path.read_text().replace("x5 = 5", "x5 = 5  # tuned by hand"), encoding="utf-8")
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"rfx/m.py:7: comment: # tuned by hand  -> {FULL_BAN}",
        summary(repo, git_env, base, 1, 1),
    ]


def swept_line(path, line, comment, entry):
    return (
        f"{path}:{line}: swept: {comment}  -> a name, an assertion, a test, the docstring, or a dated record; "
        f"{entry} is swept, so every comment in it may only be a tool directive"
    )


def swept_repo(tmp_path, env, files):
    repo, _ = make_repo(tmp_path, env)
    for path, content in files.items():
        add_file(repo, env, path, content)
    return repo, commit(repo, env)


def test_swept_file_with_prose_on_untouched_line_is_refused(tmp_path, git_env):
    repo, base = swept_repo(tmp_path, git_env, {
        "rfx/a.py": "# an old prose comment\nvalue = 0\n", "rfx/b.py": "value = 0\n",
    })
    result = check(repo, git_env, "--base", base, swept=("rfx/a.py",))
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        swept_line("rfx/a.py", 1, "# an old prose comment", "rfx/a.py"),
        summary(repo, git_env, base, 0, 0),
        "swept: 1 refused in 1 files (1 entries)",
    ]
    result = check(repo, git_env, "--base", base, swept=("rfx/b.py",))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        summary(repo, git_env, base, 0, 0),
        "swept: 0 refused in 1 files (1 entries)",
    ]
    result = check(repo, git_env, "--base", base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [summary(repo, git_env, base, 0, 0)]


def test_directives_pass_in_swept_file(tmp_path, git_env):
    repo, base = swept_repo(tmp_path, git_env, {"rfx/sub/x.py": (
        "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nimport os  # noqa: F401\n"
        "value = 1  # type: ignore[assignment]\nif value:  # pragma: no cover\n    pass\n"
    )})
    result = check(repo, git_env, "--base", base, swept=("rfx/sub/",))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        summary(repo, git_env, base, 0, 0),
        "swept: 0 refused in 1 files (1 entries)",
    ]


def test_swept_entries_cover_packages_and_single_files(tmp_path, git_env):
    note = "# a prose note"
    repo, base = swept_repo(tmp_path, git_env, {
        path: f"{note}\nvalue = 0\n"
        for path in ("rfx/sub/x.py", "rfx/sub/deep/y.py", "rfx/subgrid/z.py", "rfx/a.py", "rfx/b.py")
    })
    result = check(repo, git_env, "--base", base, swept=("rfx/sub/", "rfx/a.py"))
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        swept_line("rfx/a.py", 1, note, "rfx/a.py"),
        swept_line("rfx/sub/deep/y.py", 1, note, "rfx/sub/"),
        swept_line("rfx/sub/x.py", 1, note, "rfx/sub/"),
        summary(repo, git_env, base, 0, 0),
        "swept: 3 refused in 3 files (2 entries)",
    ]


@pytest.mark.parametrize("source", ["docs/m.py", "m.py"])
def test_file_moved_unchanged_into_swept_package_is_refused(tmp_path, git_env, source):
    repo, base = swept_repo(tmp_path, git_env, {
        source: "# an old prose comment\n" + "".join(f"x{i} = {i}\n" for i in range(20)),
    })
    (repo / "rfx/sub").mkdir(parents=True)
    git(repo, git_env, "mv", source, "rfx/sub/m.py")
    result = check(repo, git_env, "--base", base, swept=("rfx/sub/",))
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        swept_line("rfx/sub/m.py", 1, "# an old prose comment", "rfx/sub/"),
        summary(repo, git_env, base, 0, 0),
        "swept: 1 refused in 1 files (1 entries)",
    ]


@pytest.mark.parametrize("entry,message", [
    ("rfx/missing/", "rfx/missing/: swept entry matches no file; name the file's new path or remove the entry"),
    ("rfx/missing.py", "rfx/missing.py: swept entry matches no file; name the file's new path or remove the entry"),
    ("rfx/notes/", "rfx/notes/: swept entry matches no file; name the file's new path or remove the entry"),
    ("rfx/a", "rfx/a: swept entry must end with / (a package) or .py (one file)"),
    ("rfx/../docs/a.py", "rfx/../docs/a.py: swept entry must not contain .."),
    ("docs/a.py", "docs/a.py: swept entry must lie under rfx/, tests/, examples/ or scripts/"),
    ("./rfx/a.py", "./rfx/a.py: swept entry must lie under rfx/, tests/, examples/ or scripts/"),
])
def test_stale_and_malformed_swept_entries_are_refused(tmp_path, git_env, entry, message):
    repo, base = swept_repo(tmp_path, git_env, {
        "rfx/a.py": "value = 0\n", "docs/a.py": "value = 0\n", "rfx/notes/readme.txt": "text\n",
    })
    result = check(repo, git_env, "--base", base, swept=(entry,))
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        message,
        summary(repo, git_env, base, 0, 0),
        "swept: 1 refused in 0 files (1 entries)",
    ]


def load_checker():
    spec = importlib.util.spec_from_file_location("check_comment_content", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_swept_entries_are_well_formed_and_pass():
    checker = load_checker()
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, timeout=60, check=True,
    ).stdout.split("\0")
    for entry in checker.SWEPT_PREFIXES:
        assert entry.endswith(("/", ".py")), entry
        assert ".." not in entry, entry
        assert entry.startswith(("rfx/", "tests/", "examples/", "scripts/")), entry
        assert any(path.endswith(".py") and path.startswith(entry) for path in tracked), entry
    entry_refusals, refused, _ = checker.check_swept(REPO, checker.SWEPT_PREFIXES, set())
    assert entry_refusals == []
    assert [refusal.text for refusal in refused] == []


def test_token_refused_by_both_checks_is_printed_once(tmp_path, git_env):
    repo, base = swept_repo(tmp_path, git_env, {"rfx/a.py": "# an old prose comment\nvalue = 0\n"})
    (repo / "rfx/a.py").write_text(
        "# an old prose comment\nvalue = 0\n# a new prose comment\n", encoding="utf-8")
    result = check(repo, git_env, "--base", base, swept=("rfx/a.py",))
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"rfx/a.py:3: comment: # a new prose comment  -> {FULL_BAN}",
        swept_line("rfx/a.py", 1, "# an old prose comment", "rfx/a.py"),
        summary(repo, git_env, base, 1, 1),
        "swept: 1 refused in 1 files (1 entries)",
    ]


@pytest.mark.parametrize("entry,content,first", [
    ("rfx/missing/", "value = 0\n",
     "rfx/missing/: swept entry matches no file; name the file's new path or remove the entry"),
    ("rfx/a.py", "# an old prose comment\nvalue = 0\n",
     "rfx/a.py:1: swept: # an old prose comment  -> a name, an assertion, a test, the docstring, or a dated record; "
     "rfx/a.py is swept, so every comment in it may only be a tool directive"),
])
def test_main_without_swept_option_reads_swept_prefixes(tmp_path, git_env, monkeypatch, capsys, entry, content, first):
    repo, base = swept_repo(tmp_path, git_env, {"rfx/a.py": content})
    files = 0 if entry.endswith("/") else 1
    checker = load_checker()
    monkeypatch.setattr(checker, "SWEPT_PREFIXES", (entry,))
    for key in list(os.environ):
        if key.startswith("GIT_") or key in {"BASE_SHA", "GITHUB_EVENT_NAME"}:
            monkeypatch.delenv(key)
    for key, value in git_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(repo)
    status = checker.main(["--base", base])
    out = capsys.readouterr().out
    assert status == 1, out
    assert out.splitlines() == [
        first,
        summary(repo, git_env, base, 0, 0),
        f"swept: 1 refused in {files} files (1 entries)",
    ]

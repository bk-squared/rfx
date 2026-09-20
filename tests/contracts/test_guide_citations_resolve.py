"""Every relative LINK in a repo guide must resolve to a tracked file.

Issue #752 re-verification. ``docs/guides/msl_geometry_diagnostics.md`` linked
its "frozen protocol" at ``../research_notes/issue752/fresh/protocol.md``, and
``docs/.gitignore`` excludes ``/research_notes/`` repo-wide -- so the protocol
that page told a reader to consult for the geometry, the pulse coverage, the
0.4 % hypothesis and the rejection screens is not in a clean clone at all.
``docs/guides/sparameter_support_matrix.md`` linked the same ignored tree for
#729. Two instances in one scan is the repo's threshold for a gate rather than
two fixes.

Scope is deliberately narrow, so this stays a fact check and never a style
rule:

* ``docs/**/*.md``, EXCEPT ``docs/public/`` (rendered by a site that resolves
  its own slugs, so a relative target there is not a repo path) and
  ``docs/agent/`` (same, for the agent-facing site). ``docs/research_notes/``
  is ignored by design and so is never scanned -- it is the tree being linked
  INTO, not a scanned source. The first version scanned only ``docs/guides/``
  and missed two ``docs/design_notes/931_migration/`` pages doing the same
  thing.
* Markdown LINKS with a relative target, inline ``[t](p)`` and reference-style
  ``[id]: p`` alike -- the thing a reader clicks. A path
  NAMED in prose or backticks is out of scope on purpose: naming an
  out-of-repo artifact for provenance, and saying it is out of repo, is how
  both defects above were fixed, and a gate that forbade it would push writers
  back toward linking things that are not there.
* "Resolves" means ``git ls-files`` knows it, or it is a directory holding
  tracked files. Present-but-ignored is exactly the defect, so a filesystem
  check would have passed on the machine that shipped it.

WHERE THERE IS NO REPOSITORY. The GPU suite exports the tree with
``git archive`` and runs the tests against a directory that has no ``.git``.
The first version of this file called ``git ls-files`` at MODULE level with
``check=True``, so that export did not fail this gate -- it failed COLLECTION,
with ``returned non-zero exit status 128``, and took the whole GPU lane down
with ``Interrupted: 1 error during collection`` (VESSL run 369367262260). Two
things follow, and both are load-bearing:

* No git call happens at import or collection time. The tracked set is looked
  up inside the test, once, and cached.
* In a tree with no ``.git`` the oracle becomes filesystem existence. That is
  not a weakening in the place it applies: ``git archive`` writes ONLY tracked
  files, so in an export "present" and "tracked" are the same set, and the two
  dead links this gate exists to catch are absent there for the same reason
  they are untracked here. In a checkout the fallback never runs -- by
  construction, not by hope: a tree WITH ``.git`` must get an answer out of
  git and raises if it cannot (review of PR #1135, A).

WHY THIS FILE FALLS BACK AND ITS SIBLING SKIPS.
``tests/contracts/test_research_notes_are_not_tracked.py`` meets the same
missing repository and answers it differently, on purpose. It asks whether a
file is TRACKED though ignored, and nothing but git can answer that: in an
export the ignored files are simply absent, so there is no evidence either
way and the honest verdict is "unanswerable", which is a skip. This file asks
whether a link target is PRESENT for someone who clones, and an export is a
valid witness for exactly that, because it holds the tracked files and nothing
else. Same missing tool, two different questions, two different right answers.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._git_tracked import git_available, tracked_set

REPO = Path(__file__).resolve().parents[2]
_EXCLUDED = ("docs/public/", "docs/agent/")


def _scanned_pages() -> list[Path]:
    out = []
    for path in sorted((REPO / "docs").rglob("*.md")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(_EXCLUDED):
            continue
        out.append(path)
    return out


GUIDES = _scanned_pages()

# Inline ``[text](target)`` and reference-style ``[id]: target``. The second
# form slipped the first version of this gate entirely.
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# ``[^id]:`` is a FOOTNOTE definition, not a link -- its body is prose, and
# reading the first token of it as a target produced hits like "E[k+1]-E[k]".
_REF_LINK = re.compile(r"^\[(?!\^)[^\]]+\]:[ \t]*(\S+)", re.MULTILINE)
_SKIP_SCHEME = ("http://", "https://", "mailto:", "#", "/")


def _strip_angle_brackets(target: str) -> str:
    """CommonMark allows ``[text](<path with spaces>)``. Rejecting any target
    containing ``<`` let that whole form slip the gate."""
    if target.startswith("<") and target.endswith(">"):
        return target[1:-1]
    return target


def _looks_like_a_path(target: str) -> bool:
    """A link target that could name a file: it has a directory part or a
    suffix. Keeps prose and maths out of a gate about paths."""
    if any(c in target for c in "[]<>|"):
        return False
    return "/" in target or bool(Path(target).suffix)


#: Sentinel for "look the tracked set up yourself". ``None`` is a real value
#: here -- it means "no repository, use the filesystem" -- so it cannot double
#: as the default.
_LOOK_IT_UP = object()

_CACHE: dict[str, frozenset[str] | None] = {}


class GitCannotAnswerHere(RuntimeError):
    """A checkout whose git refuses to answer. Not a licence to guess."""


def tracked_or_none(repo: Path = REPO) -> frozenset[str] | None:
    """Every tracked path in *repo*, or ``None`` in a tree that has no ``.git``.

    Called from inside the tests, never at import: a repository question asked
    at collection time turns "this tree has no git" into a collection error
    that stops the whole session rather than one test.

    The two cases are decided by ``.git``, not by whether git happens to
    succeed (review of PR #1135, A). Keying on success conflated an EXPORT,
    where presence is a valid oracle, with a CHECKOUT whose git refused --
    exit 128 is a live condition here, dubious ownership being the reason the
    GPU harness sets ``safe.directory`` -- and in the second case the weaker
    oracle silently passes a present-but-untracked link that real git catches.
    So a tree with ``.git`` must get an answer from git, and raises if it
    cannot. That is what makes "in a checkout the fallback never runs", in the
    module docstring above, true by construction rather than by hope.

    ``.git`` is a directory in a clone and a file in a linked worktree;
    ``exists()`` covers both.
    """
    key = str(repo)
    if key not in _CACHE:
        if not (Path(repo) / ".git").exists():
            _CACHE[key] = None
            return _CACHE[key]
        tracked = tracked_set(repo) if git_available(repo) else frozenset()
        if not tracked:
            raise GitCannotAnswerHere(
                f"{repo} has a .git, so this is a checkout and the question "
                "'is this path committed' has a real answer -- but git did "
                "not give one. `git rev-parse --is-inside-work-tree` or "
                "`git ls-files` failed or came back empty. Exit 128 here is "
                "usually dubious ownership: run "
                f"`git config --global --add safe.directory {repo}`. Falling "
                "back to filesystem presence would pass a link that is "
                "present but ignored, which is the defect this gate exists "
                "to catch, so it is refused instead."
            )
        _CACHE[key] = tracked
    return _CACHE[key]


def unresolved_links(page_dir: Path, text: str, *,
                     tracked=_LOOK_IT_UP,
                     repo: Path = REPO) -> list[tuple[str, str]]:
    """Every relative link target on the page that a clean clone lacks.

    The single predicate: the per-page test and the self-checks all call it.

    *tracked* is the set git reports, or ``None`` to score by filesystem
    existence instead -- see the module docstring for when that is the right
    oracle. Left unset, it is looked up once and cached.
    """
    if tracked is _LOOK_IT_UP:
        tracked = tracked_or_none(repo)
    out: list[tuple[str, str]] = []
    for raw in _LINK.findall(text) + _REF_LINK.findall(text):
        if raw.startswith(_SKIP_SCHEME):
            continue
        target = _strip_angle_brackets(raw).split("#", 1)[0]
        if not target or not _looks_like_a_path(target):
            continue
        resolved = (page_dir / target).resolve()
        try:
            rel = resolved.relative_to(repo).as_posix()
        except ValueError:
            out.append((raw, "resolves outside the repository"))
            continue
        if tracked is None:
            if not resolved.exists():
                out.append((raw, "absent from this tree (no repository here, "
                                 "so presence is what can be checked)"))
            continue
        if rel in tracked or any(t.startswith(rel + "/") for t in tracked):
            continue
        out.append((raw, "present locally but NOT tracked (check "
                         "docs/.gitignore)" if resolved.exists() else "absent"))
    return out


@pytest.mark.parametrize("page", GUIDES, ids=lambda p: p.name)
def test_every_relative_link_is_in_the_repository(page: Path) -> None:
    assert GUIDES, "the glob found no guides"
    missing = unresolved_links(page.parent, page.read_text(encoding="utf-8"))
    assert not missing, (
        f"{page.relative_to(REPO)} links paths a clean clone does not have:\n"
        + "\n".join(f"  {raw}  -- {why}" for raw, why in missing)
        + "\n\nLink a tracked artifact instead, or name the path in prose and "
        "say it is not in the repository, so no claim rests on a reader "
        "opening it."
    )


def test_the_scan_covers_design_notes_too() -> None:
    """The first version scanned only docs/guides/ and missed two pages."""
    names = {p.relative_to(REPO).as_posix() for p in GUIDES}
    assert "docs/guides/msl_geometry_diagnostics.md" in names
    assert "docs/design_notes/931_migration/T6-RECOMPUTE.md" in names
    assert not any(n.startswith(_EXCLUDED) for n in names)


def test_a_reference_style_link_does_not_slip_the_gate() -> None:
    """``[id]: target`` is a link a reader follows, and the inline regex
    cannot see it."""
    guides = REPO / "docs" / "guides"
    ref = "See [the protocol][p].\n\n[p]: ../research_notes/issue752/fresh/protocol.md\n"
    assert len(unresolved_links(guides, ref)) == 1
    live = "See [the matrix][m].\n\n[m]: support_matrix.md\n"
    assert unresolved_links(guides, live) == []


def test_an_angle_bracket_target_does_not_slip_the_gate() -> None:
    """``[text](<path>)`` is valid CommonMark. The first version of the
    path-shape filter rejected any target containing ``<``, which meant this
    whole form was skipped rather than checked."""
    guides = REPO / "docs" / "guides"
    dead = "See [the protocol](<../research_notes/issue752/fresh/protocol.md>).\n"
    assert len(unresolved_links(guides, dead)) == 1
    live = "See [the matrix](<support_matrix.md>).\n"
    assert unresolved_links(guides, live) == []


def test_the_scan_catches_the_two_links_it_was_written_for() -> None:
    """Both shipped links, replayed through the real predicate.

    A scan that matches nothing is not a gate. The third case is the fix that
    replaced them -- a path NAMED in prose, which must NOT be reported.
    """
    guides = REPO / "docs" / "guides"
    shipped = (
        "The [frozen protocol](../research_notes/issue752/fresh/protocol.md) "
        "specifies geometry.\n"
        "[Diagnosis and limits](../research_notes/issue729/ad-referee-diagnosis.md).\n"
    )
    assert len(unresolved_links(guides, shipped)) == 2

    fixed = (
        "Its frozen protocol was written under `docs/research_notes/issue752/`, "
        "which `docs/.gitignore` excludes from the repository.\n"
    )
    assert unresolved_links(guides, fixed) == []

    live = "See [the support matrix](support_matrix.md) for the current state.\n"
    assert unresolved_links(guides, live) == []


def _git_shim(tmp_path, monkeypatch, exit_code: int) -> None:
    """Put a git on PATH that always exits *exit_code*, and nothing else."""
    import os
    import stat

    binary = tmp_path / "shimbin"
    binary.mkdir()
    shim = binary / "git"
    shim.write_text(f"#!/bin/sh\necho 'git: shimmed' >&2\nexit {exit_code}\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + "/usr/bin")


@pytest.fixture(autouse=True)
def _clear_tracked_cache():
    """The tracked set is cached per repo path; these tests build their own."""
    _CACHE.clear()
    yield
    _CACHE.clear()


def test_a_checkout_whose_git_refuses_is_an_error_not_a_fallback(
        tmp_path, monkeypatch) -> None:
    """Review of PR #1135, A. The one case that must NOT reach the fallback.

    Exit 128 inside a real checkout is a live condition -- dubious ownership
    is why the GPU harness sets ``safe.directory`` -- and treating it like an
    export would quietly swap the strong oracle for the weak one and pass a
    present-but-ignored link that real git catches.
    """
    repo = tmp_path / "checkout"
    (repo / ".git").mkdir(parents=True)
    _git_shim(tmp_path, monkeypatch, 128)

    with pytest.raises(GitCannotAnswerHere) as excinfo:
        tracked_or_none(repo)
    assert "safe.directory" in str(excinfo.value)


def test_a_missing_git_binary_in_a_checkout_is_also_an_error(
        tmp_path, monkeypatch) -> None:
    """Same verdict by the other route: ``.git`` is what decides, not whether
    the subprocess happened to run."""
    repo = tmp_path / "checkout"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(GitCannotAnswerHere):
        tracked_or_none(repo)


def test_a_git_file_counts_as_a_checkout(tmp_path, monkeypatch) -> None:
    """A linked worktree has ``.git`` as a FILE. This repository is often
    worked in one, so keying on a directory would put every worktree on the
    weak oracle."""
    repo = tmp_path / "linked"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/linked\n")
    _git_shim(tmp_path, monkeypatch, 128)
    with pytest.raises(GitCannotAnswerHere):
        tracked_or_none(repo)


def test_an_export_still_falls_back_when_git_is_broken(
        tmp_path, monkeypatch) -> None:
    """The harness case, with the shim in place: no ``.git``, so presence is
    the oracle and the gate runs rather than erroring."""
    guides = tmp_path / "docs" / "guides"
    guides.mkdir(parents=True)
    (guides / "present.md").write_text("the target that exists\n")
    _git_shim(tmp_path, monkeypatch, 128)

    assert tracked_or_none(tmp_path) is None
    found = unresolved_links(guides, "[kept](present.md) and [gone](missing.md)\n",
                             repo=tmp_path)
    assert [raw for raw, _ in found] == ["missing.md"], found


def test_without_a_repository_the_filesystem_is_the_oracle(tmp_path) -> None:
    """Criterion (B) for the archive export: the predicate still discriminates.

    A tree with no ``.git`` is what the GPU suite runs. The gate must not go
    silent there and must not go red on every link; it must report exactly the
    targets that are absent, because ``git archive`` writes only tracked files
    and so absence is the same finding that untrackedness is in a checkout.
    """
    guides = tmp_path / "docs" / "guides"
    guides.mkdir(parents=True)
    (guides / "present.md").write_text("the target that exists\n")

    assert tracked_or_none(tmp_path) is None, (
        "this fixture is only meaningful where git cannot answer; "
        f"{tmp_path} looks like a checkout")

    page = "[kept](present.md) and [dropped](missing.md)\n"
    found = unresolved_links(guides, page, repo=tmp_path)
    assert [raw for raw, _ in found] == ["missing.md"], found
    assert "no repository here" in found[0][1], found[0][1]


def test_the_same_page_is_scored_by_git_where_git_can_answer(tmp_path) -> None:
    """The fallback is not the behaviour anywhere git works.

    Same directory, same page, an explicit tracked set: the file that exists on
    disk but is not in that set is reported, which is the ignored-but-present
    defect the gate was written for and the one thing filesystem existence
    cannot see.
    """
    guides = tmp_path / "docs" / "guides"
    guides.mkdir(parents=True)
    (guides / "present.md").write_text("present but not committed\n")

    page = "[kept](present.md)\n"
    assert unresolved_links(guides, page, repo=tmp_path,
                            tracked=frozenset({"docs/guides/present.md"})) == []
    found = unresolved_links(guides, page, repo=tmp_path,
                             tracked=frozenset({"docs/guides/other.md"}))
    assert [raw for raw, _ in found] == ["present.md"], found
    assert "NOT tracked" in found[0][1], found[0][1]


def test_this_module_imports_where_there_is_no_git_binary() -> None:
    """What actually broke the GPU lane, pinned at its own level.

    The failure was not an assertion -- it was ``git ls-files`` raising at
    MODULE level, which pytest reports as a collection error and which stops
    the entire session, not just this file. A subprocess with an empty PATH
    has no git binary at all, so an import that still asked and did not
    swallow the error would fail here.

    Named for what it checks (review of PR #1135, C). It does NOT pin "no git
    call at import": a module-level ``tracked_set(REPO)`` would pass it,
    because that helper returns an empty set rather than raising. What it pins
    is that COLLECTION SURVIVES where git does not exist, which is the
    property the GPU lane needs and the one that was lost.
    """
    import subprocess
    import sys

    code = (
        "import importlib.util, pathlib, sys\n"
        f"spec = importlib.util.spec_from_file_location('probe', {str(Path(__file__).resolve())!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "spec.loader.exec_module(mod)\n"
        "print('imported', len(mod.GUIDES))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={"PATH": "", "PYTHONPATH": str(REPO),
             "HOME": str(Path.home()), "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, (
        "importing this module without a git binary failed -- something at "
        "module level is asking git and letting the failure out:\n"
        f"{result.stderr}")
    assert "imported" in result.stdout, result.stdout

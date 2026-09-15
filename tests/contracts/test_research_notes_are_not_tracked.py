"""docs/research_notes/ and docs/agent-memory/ are local-only. Keep them that way.

This is a PUBLIC repository. ``docs/.gitignore`` ignores both directories, and
the repo's own operating standard says so in prose: agent-memory is "NOT
version-controlled here", research_notes is "Local only -- gitignored in this
public repo". Neither statement is self-enforcing. ``git add -f`` overrides a
gitignore without warning, and between 2026-09-08 and 2026-09-14 fifteen
commits force-added 1060 files (373.6 MB, 344.8 MB of it .npz/.npy arrays)
under ``docs/research_notes/`` -- publishing internal chronology, and naming
1060 public paths after private issue numbers, without anyone deciding to.

Nobody chose that; the absence of a check let it accumulate. The content now
lives in the private workspace repo under ``docs/research-archive/rfx/``, kept
current by ``scripts/sync_research_archive.sh``; oversize artifacts the mirror
filters out are bundled on NFS. This gate keeps the public side clean.

If a file under either directory is load-bearing for a committed test or a
gated citation, that file is not a note -- move it to ``tests/fixtures/`` or
next to its consumer and cite it there. The numeric-provenance gate already
says as much when it refuses an untracked artifact: "Commit the artifact or
cite one that is committed."
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests._git_tracked import git_available

REPO = Path(__file__).resolve().parents[2]

LOCAL_ONLY = ("docs/research_notes", "docs/agent-memory")


def _tracked_under(repo: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Paths git tracks under *prefixes*, in *repo*. Empty when none."""
    result = subprocess.run(
        ["git", "ls-files", "--", *prefixes],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"git ls-files failed: {result.stderr.strip()}"
    return [line for line in result.stdout.splitlines() if line]


@pytest.mark.skipif(
    not git_available(REPO),
    reason="not a git checkout (wheel install, source tarball, or no git binary) -- "
           "tracking is not a question that can be answered here",
)
def test_local_only_doc_directories_are_not_tracked() -> None:
    tracked = _tracked_under(REPO, LOCAL_ONLY)
    assert not tracked, (
        f"{len(tracked)} file(s) under {LOCAL_ONLY} are git-tracked in this PUBLIC "
        f"repo. docs/.gitignore ignores both; these got in past it, which takes "
        f"`git add -f`. First five: {tracked[:5]}. Untrack with "
        f"`git rm -r --cached docs/research_notes docs/agent-memory` (the files stay "
        f"on disk). If one of them is load-bearing for a test or a gated citation, "
        f"move that file to tests/fixtures/ or beside its consumer and re-point the "
        f"reference -- do not re-add the directory."
    )


@pytest.mark.skipif(
    not git_available(REPO),
    reason="not a git checkout -- gitignore behaviour is not observable here",
)
def test_gitignore_still_ignores_both_directories() -> None:
    """Untracking is only durable while the ignore rule stands behind it.

    Checked through ``git check-ignore`` rather than by grepping the file, so a
    later negation elsewhere in the ignore chain is caught too.
    """
    for prefix in LOCAL_ONLY:
        probe = f"{prefix}/__ignore_probe__.md"
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", probe],
            cwd=REPO, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"{probe} is NOT ignored (git check-ignore returned "
            f"{result.returncode}). docs/.gitignore must keep ignoring {prefix}/ or "
            f"the next write there lands in the public repo by default."
        )


@pytest.mark.skipif(
    not git_available(REPO),
    reason="not a git checkout -- a scratch repo cannot be built without git",
)
def test_the_gate_fires_on_a_repository_that_does_track_them(tmp_path: Path) -> None:
    """Criterion (B): a gate that cannot go red is not a gate.

    The real check reads green on a clean tree, which is also what a broken
    check reads. This builds the failure case -- a repository that force-adds a
    file under each local-only directory -- and asserts the same predicate
    reports it.
    """
    for argv in (["git", "init", "-q"],
                 ["git", "config", "user.email", "probe@example.invalid"],
                 ["git", "config", "user.name", "probe"]):
        assert subprocess.run(argv, cwd=tmp_path, capture_output=True).returncode == 0

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / ".gitignore").write_text("/research_notes/\n/agent-memory/\n")
    for prefix in LOCAL_ONLY:
        target = tmp_path / prefix
        target.mkdir(parents=True)
        (target / "note.md").write_text("forced past the ignore rule\n")

    assert _tracked_under(tmp_path, LOCAL_ONLY) == [], (
        "the ignore rule should keep these out until someone forces them"
    )
    forced = subprocess.run(
        ["git", "add", "-f", "docs/research_notes", "docs/agent-memory"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert forced.returncode == 0, forced.stderr

    assert sorted(_tracked_under(tmp_path, LOCAL_ONLY)) == [
        "docs/agent-memory/note.md",
        "docs/research_notes/note.md",
    ], "the predicate did not see force-added files -- it cannot protect this repo"

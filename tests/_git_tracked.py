"""tests/_git_tracked.py

One shared answer to "is this path a COMMITTED artifact?" for the evidence
gates (issue #928).

``Path.exists()`` is not that answer. A results directory can hold a file that
is present in one checkout and in no other -- gitignored scratch, a pod's
leftover, a file someone forgot to add -- and every gate that only checks
existence goes green on it while a fresh clone has nothing to read. The
question a citation gate is really asking is "can a reviewer who clones this
repository open the file I am pointing at", and only git can answer it.

``_is_tracked`` was written first in
``tests/crossval/test_waveguide_tjunction_e4e5_gates.py``, whose fixtures had
to be genuinely committed rather than merely present; that lane imports it
back from here so the question is asked one way.

Callers must handle "git unavailable" themselves -- ``git_available()`` returns
False in a source tarball or a sandbox with no git, where an assertion about
tracking is unanswerable rather than false.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git_available(repo: Path | None = None) -> bool:
    """True when *repo* is inside a working tree git can talk about."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo or REPO, capture_output=True, text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def is_tracked(rel_path: str | Path, repo: Path | None = None) -> bool:
    """True when *rel_path* (repo-relative) is tracked by git in *repo*."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(rel_path)],
            cwd=repo or REPO, capture_output=True, text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def tracked_set(repo: Path | None = None) -> frozenset[str]:
    """Every tracked path in *repo*, once.

    A gate that resolves hundreds of citations should ask git once, not once
    per citation: the per-path form spawned a process per reference and made a
    1147-reference census measurably slower for no extra information.
    """
    try:
        result = subprocess.run(["git", "ls-files"], cwd=repo or REPO,
                                capture_output=True, text=True)
    except OSError:
        return frozenset()
    if result.returncode != 0:
        return frozenset()
    return frozenset(result.stdout.splitlines())

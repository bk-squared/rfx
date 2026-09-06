"""tests/_git_tracked.py

One shared answer to "is this path a COMMITTED artifact?" for the evidence
gates (issue #928).

``Path.exists()`` is not that answer. A results directory can hold a file that
is present in one checkout and in no other -- gitignored scratch, a VESSL pod's
leftover, a file someone forgot to ``git add`` -- and every gate that only
checks existence goes green on it while a fresh clone has nothing. That is the
gitignored-``.omx`` shape that lost the June numbers
(``tests/crossval/test_waveguide_tjunction_e4e5_gates.py``, which is where
``_is_tracked`` was written first and which now imports it from here).

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

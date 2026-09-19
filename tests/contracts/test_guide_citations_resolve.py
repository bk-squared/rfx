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

* ``docs/guides/*.md`` only. ``docs/public/**`` is rendered by a site that
  resolves its own slugs, and ``docs/research_notes/**`` is ignored by design.
* Markdown LINKS with a relative target -- the thing a reader clicks. A path
  NAMED in prose or backticks is out of scope on purpose: naming an
  out-of-repo artifact for provenance, and saying it is out of repo, is how
  both defects above were fixed, and a gate that forbade it would push writers
  back toward linking things that are not there.
* "Resolves" means ``git ls-files`` knows it, or it is a directory holding
  tracked files. Present-but-ignored is exactly the defect, so a filesystem
  check would have passed on the machine that shipped it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUIDES = sorted((REPO / "docs" / "guides").glob("*.md"))

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_SKIP_SCHEME = ("http://", "https://", "mailto:", "#", "/")


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    return set(out.splitlines())


TRACKED = _tracked()


def unresolved_links(page_dir: Path, text: str) -> list[tuple[str, str]]:
    """Every relative link target on the page that a clean clone lacks.

    The single predicate: the per-page test and the self-check both call it.
    """
    out: list[tuple[str, str]] = []
    for raw in _LINK.findall(text):
        if raw.startswith(_SKIP_SCHEME):
            continue
        target = raw.split("#", 1)[0]
        if not target:
            continue
        resolved = (page_dir / target).resolve()
        try:
            rel = resolved.relative_to(REPO).as_posix()
        except ValueError:
            out.append((raw, "resolves outside the repository"))
            continue
        if rel in TRACKED or any(t.startswith(rel + "/") for t in TRACKED):
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

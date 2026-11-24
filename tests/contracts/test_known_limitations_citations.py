"""Legacy citation inventory for the public known-limitations page.

These sets pin historical arrow citations and resolved references, not live
issue state. An issue can close through a fix, a characterization, or acceptance
of a standing limitation. The per-subject Tracker contract now lives in
``test_known_limitations_tracker.py``; ``scripts/ci/check_known_limitations.py``
checks PR closures and the weekly governance sweep reads live states.

Keep this inventory in sync when audited entries are removed or rewritten.
A closed issue or PR recording a fix, measurement or settled decision is allowed to appear without a citation line.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "guides" / "known_limitations.md"
README = REPO_ROOT / "README.md"
SUPPORT_MATRIX = REPO_ROOT / "docs" / "guides" / "support_matrix.md"

CITED_ISSUES = frozenset({737, 1373, 1512, 2150, 2473, 2977})

RESOLVED_REFERENCES = frozenset({726, 830, 838, 1100, 1122, 1181, 1186, 1255, 1381, 1419,
                                 1524, 1571, 1761})

CITATION_RE = re.compile(
    r"^→ \[#(\d+)\]\(https://github\.com/bk-squared/rfx/issues/(\d+)\)$",
    re.MULTILINE,
)
INLINE_RE = re.compile(r"(?<![\w/])#(\d+)\b")


def _page() -> str:
    assert PAGE.is_file(), f"{PAGE} is missing"
    return PAGE.read_text(encoding="utf-8")


@pytest.mark.reads_docs_for_gate(reason=(
    "a known-limitations entry must stay while its defect is open: the one page "
    "a PR must keep right"))
def test_every_pinned_issue_keeps_its_entry():
    """An entry that disappears while its issue is open hides a defect from users."""
    cited = {int(text) for text, _ in CITATION_RE.findall(_page())}
    missing = CITED_ISSUES - cited
    assert not missing, (
        f"known_limitations.md no longer cites {sorted(missing)}. An entry leaves "
        "this page when its issue closes and a test pins the fix; if that happened, "
        "drop the number from CITED_ISSUES in the same change, after checking its "
        "state with `gh issue view <N> --json state`."
    )


@pytest.mark.docs_consistency
def test_the_page_cites_no_issue_outside_the_pinned_set():
    """A new citation not yet added to the set is bookkeeping."""
    cited = {int(text) for text, _ in CITATION_RE.findall(_page())}
    added = cited - CITED_ISSUES
    assert not added, (
        f"known_limitations.md newly cites {sorted(added)}. Add each to CITED_ISSUES "
        "in this file in the same change, and check its state with "
        "`gh issue view <N> --json state` while you are there."
    )


@pytest.mark.docs_consistency
def test_every_citation_link_points_at_the_issue_it_names():
    mismatched = [
        (text, target) for text, target in CITATION_RE.findall(_page()) if text != target
    ]
    assert not mismatched, (
        f"a citation's link text and URL name different issues: {mismatched}. "
        "A reader clicking #N and landing on #M learns the wrong thing."
    )


@pytest.mark.docs_consistency
def test_no_entry_names_an_issue_it_does_not_cite():
    """An issue mentioned in prose must be cited, tracked, or classified as resolved.

    Otherwise a number can sit in a sentence, never appear in CITED_ISSUES, and
    outlive the issue it refers to. This caught #1043 on its first run, which is
    a real reference and now sits in RESOLVED_REFERENCES.
    """
    page = _page()
    cited = {int(text) for text, _ in CITATION_RE.findall(page)}
    tracked = {int(n) for line in page.splitlines() if line.startswith("Tracker: #")
               for n in INLINE_RE.findall(line)}
    prose_only = {int(n) for n in INLINE_RE.findall(page)} - cited - tracked - RESOLVED_REFERENCES
    assert not prose_only, (
        f"issues named in prose but never cited: {sorted(prose_only)}. Give the entry a "
        "`→ [#N](…)` line, or — if the number is a closed issue named for provenance — "
        "add it to RESOLVED_REFERENCES with the reason."
    )


@pytest.mark.docs_consistency
def test_the_page_is_reachable_from_the_readme_and_the_support_matrix():
    """The page is a repo doc, not a public-site page.

    `docs/guides/` is not in `docs/public/site_map.json` — the site map holds
    `rfx/guide/*` slugs and resolves none of the guides directory, exactly as
    for `support_matrix.md`. So these two links are the only way a reader
    arrives, and they are worth a check.
    """
    assert "docs/guides/known_limitations.md" in README.read_text(encoding="utf-8"), (
        "README.md no longer links the known-limitations page"
    )
    assert "known_limitations.md" in SUPPORT_MATRIX.read_text(encoding="utf-8"), (
        "docs/guides/support_matrix.md no longer links the known-limitations page"
    )

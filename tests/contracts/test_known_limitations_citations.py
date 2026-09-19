"""`docs/guides/known_limitations.md` cites exactly the issues it was pinned to.

The page's own rule is that an entry leaves when its defect is fixed and a
committed test pins the fix. Nothing enforced that, and the first revision of
the page shipped two entries whose issues were already closed on `main`: #1090
(fixed in 8bc6c084) and #1085 (fixed in 246cde5d). A public page that describes
a defect the user does not have is the same class of wrong as one that hides a
defect they do.

What this pins, and what it cannot
----------------------------------
The real invariant is "every cited issue is OPEN", and that cannot be checked
here: the CI lanes have no network, and a test that shells out to `gh` would be
a flake on every runner. So the set of cited numbers is pinned instead. Adding
or removing an entry then has to come with an edit to CITED_ISSUES, which is the
moment a reader is asked whether the tracker still agrees -- drift becomes a
deliberate act rather than an oversight.

The OPEN check itself belongs to the weekly audit, which has a network and a
schedule. Last run by hand on 2026-09-18 with `gh issue view <N> --json state`:
all eleven below OPEN, and the two closed ones removed in the same change.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "guides" / "known_limitations.md"
README = REPO_ROOT / "README.md"
SUPPORT_MATRIX = REPO_ROOT / "docs" / "guides" / "support_matrix.md"

# Every issue the page cites, verified OPEN on 2026-09-18. Edit this set in the
# same change that adds or removes an entry, and re-check the tracker when you
# do -- that re-check is the whole point of the set being written down.
# #1101 (the port_aperture_snap advisory naming the wrong source for the
# cutoff) left on 2026-09-19: the message now READS the built
# ``cfg.f_cutoff`` and names the profile it came from, pinned by
# tests/unit/ports/test_port_aperture_rasterization.py's two
# mode_profile fixtures -- the page's own rule for when an entry goes.
# The taper entry carries NO citation line on purpose. #1100 (the SMOKE half)
# closed with the commensurate mesh, and #1122 (the paper lane's published
# figures) closed on 2026-09-19 with a PI decision to disclose rather than
# re-mesh or issue an erratum. The limitation is still real and still belongs
# on this page, but there is no open work behind it, and an arrow line would
# present a settled decision as a task. Both numbers sit in
# RESOLVED_REFERENCES below, which is what this file already uses for a closed
# issue named for provenance.
CITED_ISSUES = frozenset({1070, 1066, 838, 830, 726, 820, 737, 715, 1022})

# Numbers the prose names for provenance rather than as a live defect: a CLOSED
# issue quoted to say what part of the problem is already fixed. #1043 (the
# runners' pad continuation, closed) is named inside the #1066 entry to mark the
# boundary of what remains. These are allowed to appear without a citation line;
# a number that is neither cited nor listed here fails the test below, which is
# what makes the exception a decision rather than a gap.
RESOLVED_REFERENCES = frozenset({1043, 1100, 1122})
# #1100 and #1122 join it together: the taper entry names both to record which
# half was fixed and what was decided about the other, and both are closed.

# The page's citation form: a line that is an arrow, then a link whose text is
# `#N` and whose target is that issue.
CITATION_RE = re.compile(
    r"^→ \[#(\d+)\]\(https://github\.com/bk-squared/rfx/issues/(\d+)\)$",
    re.MULTILINE,
)
# Any `#N` anywhere in the prose, to catch an entry that names an issue in a
# sentence without carrying it into a citation line.
INLINE_RE = re.compile(r"(?<![\w/])#(\d+)\b")


def _page() -> str:
    assert PAGE.is_file(), f"{PAGE} is missing"
    return PAGE.read_text(encoding="utf-8")


def test_the_page_cites_exactly_the_pinned_issues():
    cited = {int(text) for text, _ in CITATION_RE.findall(_page())}
    missing = CITED_ISSUES - cited
    added = cited - CITED_ISSUES
    assert not missing and not added, (
        f"known_limitations.md citations drifted from the pinned set.\n"
        f"  no longer cited: {sorted(missing)}\n"
        f"  newly cited:     {sorted(added)}\n"
        "Update CITED_ISSUES in this file in the same change, and check each number's "
        "state with `gh issue view <N> --json state` while you are there — an entry "
        "leaves this page when its issue closes and a test pins the fix."
    )


def test_every_citation_link_points_at_the_issue_it_names():
    mismatched = [
        (text, target) for text, target in CITATION_RE.findall(_page()) if text != target
    ]
    assert not mismatched, (
        f"a citation's link text and URL name different issues: {mismatched}. "
        "A reader clicking #N and landing on #M learns the wrong thing."
    )


def test_no_entry_names_an_issue_it_does_not_cite():
    """An issue mentioned in prose must be cited, or classified as resolved.

    Otherwise a number can sit in a sentence, never appear in CITED_ISSUES, and
    outlive the issue it refers to. This caught #1043 on its first run, which is
    a real reference and now sits in RESOLVED_REFERENCES.
    """
    page = _page()
    cited = {int(text) for text, _ in CITATION_RE.findall(page)}
    prose_only = {int(n) for n in INLINE_RE.findall(page)} - cited - RESOLVED_REFERENCES
    assert not prose_only, (
        f"issues named in prose but never cited: {sorted(prose_only)}. Give the entry a "
        "`→ [#N](…)` line, or — if the number is a closed issue named for provenance — "
        "add it to RESOLVED_REFERENCES with the reason."
    )


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

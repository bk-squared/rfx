"""Legacy citation inventory for the public known-limitations page.

These sets pin historical arrow citations and resolved references, not live
issue state. An issue can close through a fix, a characterization, or acceptance
of a standing limitation. The per-subject Tracker contract now lives in
``test_known_limitations_tracker.py``; ``scripts/ci/check_known_limitations.py``
checks PR closures and the weekly governance sweep reads live states.

Keep this inventory in sync when the leader removes or rewrites audited entries.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "guides" / "known_limitations.md"
README = REPO_ROOT / "README.md"
SUPPORT_MATRIX = REPO_ROOT / "docs" / "guides" / "support_matrix.md"

# Legacy arrow citations, not an assertion that these issues are still OPEN.
# Edit this set when an audited entry is added or removed; the Tracker gate
# and weekly sweep enforce live tracking separately.
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
# #726 left the cited set on 2026-09-19. The CONTRADICTION it names (the
# Z0 guard and preflight disagreeing about the same condition) is closed
# and pinned: one shared text, embedded by both sites plus the auto-offset
# resolver, with tests that fail if either site drops it or revives a
# retired claim. What remains on the page is the underlying LIMITATION --
# an N-probe fit near a reflector cannot be read -- which is inherent to
# the method, fully disclosed and gateable on probe_clearance /
# beta_railed. No open work stands behind it, so the entry carries no
# arrow and 726 sits in RESOLVED_REFERENCES.
# #1066 (the waveguide S-parameter lane not continuing a dielectric into
# its absorber pad) left on 2026-09-20: both of that lane's smoothing
# sites now call smoothed_shape_pairs, the one implementation the
# runners use, and tests/unit/sparams/test_waveguide_lane_pad_continuation.py
# pins the pad -- the page's own rule for when an entry goes.
# #1070 (a dielectric spanning the declared domain solved with vacuum in one
# absorber pad) left on 2026-09-20 with its whole section, which it was the
# only entry in. The arithmetic that bought the unfilled cell is gone
# (rfx.grid.cells_spanning) and the invariant behind it is asserted at
# assembly (rfx.geometry.rasterize_grid.assert_declared_span_is_filled), so
# a structure declared out to a padded face can no longer be rasterized more
# than the documented one node short of it. Pinned by
# tests/unit/grid/test_cell_count_ulp_snap.py and
# tests/unit/geometry/test_declared_span_reaches_padded_face.py.
# #830 (the fitted microstrip beta sitting ~0.9 % above the closed form) left on
# 2026-09-21: the PI closed it as a stated accuracy in the support matrix. The
# offset is still real and unattributed, so the prose stays without the arrow
# and 830 sits in RESOLVED_REFERENCES.
# #838 left on 2026-09-23: closed as not planned before 2.0 by PI decision.
# The coax-to-microstrip power over-read remains a standing limitation, with
# its closed issue named for provenance in RESOLVED_REFERENCES.
# #1381 left on 2026-09-30: closed as a stated limit by PI decision (both the
# completion of an unresolved pair and the early stop it can end). The two ring-down
# entries stay as standing limitations, with 1381 in RESOLVED_REFERENCES.
# #1230 replaces #801 on 2026-09-23 for the part PR #1178 does not fix: with a
# traced mesh axis no conductor is continued into the absorber. #801 is closed;
# the residual is tracked by #1230.
# #1181 left on 2026-09-23: the truncated-record gradient limitation remains,
# and the witness in PR #1186 is opt-in. Both closed numbers are provenance
# in RESOLVED_REFERENCES, not arrows to open work.
# #820 left on 2026-09-23: tests/unit/farfield/test_rcs_translation_invariance.py
# now pins the monostatic RCS spread under whole-cell target translations.
# #1221 (magnetic faces on distributed and ADI lanes, and the Yee half-cell wall)
# joined on 2026-09-23; OPEN checked with
# `gh issue view 1221 --repo bk-squared/rfx --json number,state,url,title,updatedAt`.
# #1260 (a Debye/Lorentz material anywhere puts the whole grid on the
# cell-owned E update, so every lumped element loads all three edges at its
# node) and #1257 (a graded mesh with a dispersive material leaves a port
# unterminated) joined on 2026-09-24 with the dispersive-lane entry of #1236;
# both OPEN, checked with `gh issue view <N> --json number,state`.
# Both left on 2026-09-25 with that entry: #1260's fix builds the Debye/Lorentz
# coefficients per E component from the edge mean, lumped stamps on their own
# component (tests/unit/materials/test_dispersive_edge_average.py), and #1257
# was already CLOSED (PR #1283, tests/unit/nonuniform/test_dispersive_port_load_1257.py)
# -- `gh issue view 1257 --json state` read CLOSED that day. With the entry gone
# the page no longer names #1236 either.
# #1373 (ADI refuses a material interface until 2.1) joined on 2026-10-03 with
# the "Solver lanes" section; OPEN, checked by the session leader that day.
CITED_ISSUES = frozenset({737, 1501, 1221, 1230, 1373, 1512})

# Numbers the prose names for provenance rather than as open work: a CLOSED
# issue or PR recording a fix, measurement or settled decision. These are
# allowed to appear without a citation line;
# a number that is neither cited nor listed here fails the test below, which is
# what makes the exception a decision rather than a gap.
RESOLVED_REFERENCES = frozenset({726, 830, 838, 1100, 1122, 1181, 1186, 1255, 1381, 1419})
# #1419 (weak-port ring-down gradient) is closed by the PR that added the
# identification-probe option; the entry stays because port-only identification
# is still the default, and names the issue as provenance.
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


# The pinned set is checked in two halves. An entry that disappears while its
# issue is open hides a defect from users, and the PI's documentation rule
# (2026-09-22) keeps that one blocking. A new citation not yet added to the set
# is bookkeeping, and runs with the other documentation checks.
@pytest.mark.reads_docs_for_gate(reason=(
    "a known-limitations entry must stay while its defect is open: the one page "
    "a PR must keep right (PI, 2026-09-22)"))
def test_every_pinned_issue_keeps_its_entry():
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
    # Tracker metadata is governed by the per-entry contract, not this legacy
    # arrow/provenance inventory. New live trackers need not add arrow links.
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

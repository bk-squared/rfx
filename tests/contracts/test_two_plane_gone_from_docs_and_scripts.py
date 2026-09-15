"""``two_plane`` is gone OUTSIDE ``rfx/`` too — docs, scripts, validation (#931).

``tests/contracts/test_lattice_ownership_contract.py::
test_two_plane_is_gone_from_the_package`` greps ``rfx/``. That scope is
deliberate — widening it there would fail on the ``#313`` two-PROBE-plane
instrument, which spells its wave-decomposition helper
``refplane_zc_two_plane`` and has nothing to do with the deleted realization
flag. But it also means the keyword can survive in a script, a validation
harness or a doc page, where a reader will copy it and get a ``TypeError`` —
or worse, read prose that describes a realization the solver no longer has.

This file is the other half of that gate. It scans the live trees and holds
three lists, each with a reason per entry:

``ALLOWED``  files that may name the removed surface forever. Two kinds only:
             dated history that documents the removal (CHANGELOG, the public
             changelog, the benchmarks page), and code that names it in order
             to refuse it.
``FROZEN``   committed run records — solver logs and their result JSONs —
             that quote pre-2.0 preflight text verbatim. They are evidence
             with a sha256 behind them; editing one destroys what it is for.
             Matched by path pattern, not by name.
``PENDING``  files whose #931 migration belongs to ANOTHER group. Listed so
             the gate stays green while the branches land, and paired with
             ``test_no_pending_entry_is_stale``: once the owner lands, the
             file stops matching and that test goes red telling you to delete
             the row. A pending list that cannot rot is a schedule; one that
             can is a hiding place.

``scripts/archive/`` and ``scripts/spikes/`` are excluded wholesale, as
history that is not run. (Neither contains the token today; the exclusion is
so that adding this gate never becomes a reason to rewrite an archive.)
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Trees this gate owns. rfx/ belongs to the contract file's own grep.
# NOT scanned, by design: rfx/ (the contract file's own grep owns it) and
# docs/design_notes/ + docs/agent/, which are dated records — a pre-declaration
# that quoted the old rule carries a SUPERSEDED header instead of an edit.
LIVE_ROOTS = ("scripts", "validation", "examples", "docs/public", "docs/guides")
LIVE_FILES = ("ROADMAP.md", "README.md", "CHANGELOG.md")

# Not scanned at all: history that is not executed.
EXCLUDED_TREES = (
    "scripts/archive",
    "scripts/spikes",
)

# Build products, never sources.
EXCLUDED_DIR_NAMES = ("__pycache__", ".pytest_cache")

# What the gate looks for: the deleted PUBLIC keyword, spelled as an
# identifier, and the deleted internal symbols by name. Deliberately NOT the
# prose phrase "two-plane" — that is the #313 two-PROBE-plane instrument and
# it is a different thing that happens to share two words. `two_plane_delta`
# and `refplane_zc_two_plane` are that instrument's identifiers and are
# excluded by the word boundaries.
TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])two_plane(?![A-Za-z0-9_])"
    r"|two_plane_extension_masks|_two_plane_cell_mask|_refuse_two_plane"
    r"|resample_sheet_node_materials|sheet_normal_live_axis_masks"
    r"|collect_thin_conductor_sheet_inputs|_subcell_box_axis_window"
    r"|(?<![A-Za-z0-9_])tangential_edge_masks(?![A-Za-z0-9_])")

# Frozen committed evidence: solver run logs and the result JSONs beside them.
FROZEN_PATTERNS = (
    re.compile(r"(^|/)_[A-Za-z0-9_]*_logs/"),      # scripts/diagnostics/_*_logs/
    re.compile(r"\.log$"),
    re.compile(r"(^|/)_[0-9]{2}_[A-Za-z0-9_]+_(results|logs)/"),  # validation/crossval/_15_patch_results/
)

ALLOWED = {
    # -- dated history that documents the removal by name
    "CHANGELOG.md":
        "release history; the 2.0.0 entry removes the keyword and says so",
    "ROADMAP.md":
        "the v2.0 row names #931 and the removed keyword",
    "docs/public/guide/changelog.mdx":
        "public release history; states two_plane is removed, not deprecated",
    "docs/public/guide/materials-geometry.mdx":
        "the canonical contract page; names the keyword to say it is gone",
    "docs/public/api/geometry-materials.mdx":
        "API page; names the keyword to say passing it raises TypeError",
    "docs/public/guide/benchmarks.mdx":
        "names the retired one-plane/two_plane A/B as history",
    "scripts/diagnostics/slow_931_farfield_attribution.py":
        "attribution diagnostic; its header cites the removed map by commit "
        "(a3e4dba4^:rfx/boundaries/pec.py tangential_edge_masks) to say which "
        "historical operator it reconstructs in-process. Dated history, and the "
        "citation is what makes the counterfactual checkable",
    # -- #313 two-PROBE-plane wave decomposition, same token, different thing
    "scripts/diagnostics/coax_msl_flux_adjudication.py":
        "#313 two-PROBE-plane decomposition (two_plane_delta, a local array)",
    # -- migrated by #931; they name the keyword only to record its removal
    "scripts/diagnostics/sheet_vs_volume_patch_radiation_ab.py":
        "replaced the two_plane A/B; its docstring says which arms are gone",
    "scripts/diagnostics/patch_sheet_realization_ladder.py":
        "arms are sheet/vol1/vol2; the pre-2.0 verdict is kept as history",
    "scripts/diagnostics/patch_tutorial_rfx.py":
        "PT_TWO_PLANE deleted; the comment records what replaced it",
    "scripts/diagnostics/build_patch_mode_pair_ratio_band_census.py":
        "resolves fixture legs by name; the old leg spelling is the fallback",
    "scripts/diagnostics/patch_edgefed_s11_band_repin.py":
        "its `retired` arm refuses to run and names the deleted "
        "resample_sheet_node_materials in the refusal",
    # -- crossval-C migration landed 2026-09-07 (#931 phase 2a merge)
    "validation/crossval/15_patch_antenna_rt5880.py":
        "cv15: ground/patch are sheets, the parameter is deleted; the "
        "docstrings name two_plane only to record what #740 did and #931 undid",
    "validation/crossval/manifest.json":
        "cv15's claim_scope names the deleted `two_plane=True` ground as the "
        "#740 repair the #931 sheet declaration replaced (dated history)",
    # -- crossval groups A/C/D landed 2026-09-07; dated history only
    "validation/crossval/05_patch_antenna.py":
        "cv05 docstring cites the 2026-08-28 two_plane A/B verdict as history",
    "validation/crossval/18_wr90_iris_modematch.py":
        "cv18 docstring/print record that the flag put the far face back for "
        "t = 1 only; the case's own irises are volumes",
    "scripts/diagnostics/cv15_before_after_931.py":
        "the BEFORE row of the cv15 decomposition names the #768 two_plane leg",
}

PENDING = {
    # (empty since 2026-09-07: the crossval-C migration landed; its two rows
    # moved to ALLOWED because both files now name the keyword only as dated
    # history. Keep the dict — it is the schedule for the next removal.)
}


def _rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def _scan():
    seen = {}
    roots = [REPO / r for r in LIVE_ROOTS] + [REPO / f for f in LIVE_FILES]
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for p in paths:
            if not p.is_file():
                continue
            rel = _rel(p)
            if any(rel.startswith(t + "/") for t in EXCLUDED_TREES):
                continue
            if any(part in EXCLUDED_DIR_NAMES for part in p.parts):
                continue
            if any(pat.search(rel) for pat in FROZEN_PATTERNS):
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:                                   # pragma: no cover
                continue
            hits = [f"{i}: {ln.strip()[:120]}"
                    for i, ln in enumerate(text.splitlines(), 1)
                    if TOKEN.search(ln)]
            if hits:
                seen[rel] = hits
    return seen


def test_no_live_two_plane_outside_the_allowlist():
    """A live doc or script must not carry the deleted keyword."""
    seen = _scan()
    offenders = {k: v for k, v in seen.items()
                 if k not in ALLOWED and k not in PENDING}
    assert offenders == {}, (
        "`two_plane` is removed (#931 §1.5) and these live files still carry "
        "it. Migrate the file, or add it to ALLOWED with the reason it may "
        "keep the token (dated history, or the #313 two-PROBE-plane "
        "instrument):\n"
        + "\n".join(f"  {k}\n    " + "\n    ".join(v[:4])
                    for k, v in sorted(offenders.items())))


def test_no_allowlist_entry_is_stale():
    """An allowlist naming a file that no longer has the token hides nothing —
    but it does teach the next reader that the entry is load-bearing when it
    is not. Delete it."""
    seen = _scan()
    stale = [k for k in ALLOWED
             if not (REPO / k).exists() or k not in seen]
    assert stale == [], (
        "these ALLOWED entries no longer carry `two_plane` (or the file is "
        f"gone) — delete the row: {stale}")


def test_no_pending_entry_is_stale():
    """PENDING is a schedule, and it must expire on its own.

    When the owning group's migration lands, its file stops carrying the
    token and this test goes red naming the row to delete. Without this, a
    pending list is indistinguishable from an allowlist.
    """
    seen = _scan()
    done = [k for k in PENDING if k not in seen]
    assert done == [], (
        "these #931 migrations have landed — delete their PENDING rows so the "
        f"gate covers them again: {done}")


def test_the_excluded_trees_still_exist():
    """An exclusion naming a deleted tree hides a real offender later."""
    for t in EXCLUDED_TREES:
        assert (REPO / t).is_dir(), f"excluded tree is gone: {t}"

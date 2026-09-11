"""Every number the crossval evidence documents *cite* must still be the number
the committed artifact holds.

Why this exists (#812, round-1 retrospective). Issue #812 shipped a wrong number
into a durable document four times: the audit's own cv11 claim, cv10's "1.6x /
at the arithmetic floor", cv17/cv18's sign-inverted measured claim in two
committed evidence JSONs, and cv15's regenerated leg whose ``s11_dip_db`` moved
without the prose moving with it. Of nine round-1 blocking findings, six were a
wrong or falsified claim in prose rather than a defective gate.

PR #814 closed the mechanical half of the same class for *pointers*: a
``file.py:NNN`` citation must land on a constant assignment whose NAME the
document also mentions. This gate is the same treatment for *values*.

The contract
------------
A document may cite a quantity by writing, inside a single-backtick code span,
an **artifact reference**::

    `validation/crossval/_18_wr90_iris_results/rfx.json::gates.fine_gate_abs`
    `validation/crossval/_18_wr90_iris_results/rfx.json::gates.fine_gate_abs = 0.04`
    `validation/crossval/_15_patch_results/rfx.json::f_primary_hz = 2.3139 GHz`

- the path is repo-relative and must name a committed JSON file;
- the key path walks that JSON (``a.b[2].c``);
- if a literal follows ``=``, the resolved value must equal it **to the
  precision written** -- tolerance is half a unit in the last decimal place of
  the literal as typed, times the (exact, tabulated) unit multiplier. Writing
  more digits is a tighter assertion; writing a sign is a signed assertion.

Any backtick span containing ``::`` that does not parse is an error, so a
malformed reference cannot be silently skipped.

What it does NOT catch (stated plainly, because the gap is the point)
--------------------------------------------------------------------
1. **Prose claims with no number** -- "the gate cannot fire", "this is at the
   arithmetic floor", "no committed run matches". Round 1's cv05 false absolute
   is exactly this shape and is invisible here.
2. **Wrong-but-self-consistent artifacts.** If the harness writes a wrong value
   and the prose quotes it faithfully, both agree and the gate is green. This
   gate checks provenance, never physics.
3. **A number that is correct today but describes the wrong quantity** -- citing
   ``mean_mag_abs_diff`` in a sentence about the max. The value resolves, the
   sentence is still false.
4. **Numbers nobody chose to reference.** Coverage is opt-in; the census below
   is what stops it from silently shrinking, not from being incomplete.
   Bare numerics in the opted-in sections are deliberately NOT flagged: these
   ``claim_scope`` strings carry dozens of legitimate bare numerics (issue and
   PR numbers, dates, drawn dimensions, cell counts, derived percentages), so a
   bare-numeric detector here is a false-positive storm, not a gate.

It asserts NO physics and changes NO gate value.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from tests._git_tracked import git_available, tracked_set

_REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Reference syntax
# --------------------------------------------------------------------------

# Exact multipliers. ``artifact_value == literal * UNITS[unit]``. Choosing the
# wrong one turns the gate red, never green, so the table is a convenience for
# readable prose and not a place a wrong claim can hide.
UNITS: dict[str, float] = {
    "": 1.0,
    "Hz": 1.0,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
    "dB": 1.0,
    "dBi": 1.0,
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "%": 1e-2,
}

_SPAN = re.compile(r"`([^`\n]+)`")

_REFERENCE = re.compile(
    r"^(?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*\.json)"
    r"::(?P<keypath>[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_][A-Za-z0-9_-]*|\[\d+\])*)"
    r"(?:\s*=\s*(?P<literal>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*(?P<unit>%|[A-Za-z]+))?)?$"
)

_STEP = re.compile(r"\[(\d+)\]|([A-Za-z0-9_][A-Za-z0-9_-]*)")


class Reference:
    """One ``path::keypath [= literal unit]`` citation found in a document."""

    def __init__(self, doc: str, site: str, raw: str, path: str, keypath: str,
                 literal: str | None, unit: str) -> None:
        self.doc = doc
        self.site = site
        self.raw = raw
        self.path = path
        self.keypath = keypath
        self.literal = literal
        self.unit = unit

    def __repr__(self) -> str:  # pragma: no cover - test ids only
        return f"{self.doc}::{self.site}::{self.raw}"


def strip_code_blocks(text: str) -> str:
    """Blank out fenced and indented markdown code blocks.

    A document must be able to *show* the reference syntax without the gate
    trying to resolve the illustration. Only whole code blocks are exempt;
    inline spans in running prose are always checked.
    """
    kept: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            kept.append("")
            continue
        if fenced or line.startswith("    ") or line.startswith("\t"):
            kept.append("")
            continue
        kept.append(line)
    return "\n".join(kept)


def parse_references(doc: str, site: str, text: str) -> list[Reference]:
    """Every backtick span containing ``::`` in *text*, parsed or rejected."""
    out: list[Reference] = []
    for span in _SPAN.finditer(text):
        inner = span.group(1).strip()
        if "::" not in inner:
            continue
        match = _REFERENCE.match(inner)
        if match is None:
            raise ValueError(
                f"{doc} [{site}] contains `{inner}`, which looks like an "
                f"artifact reference but does not parse. The syntax is "
                f"`path/to.json::a.b[2].c` optionally followed by "
                f"` = <number>[ <unit>]`."
            )
        if ".." in match.group("path"):
            raise ValueError(
                f"{doc} [{site}] cites `{inner}`, whose path escapes the "
                f"repository. Artifact references are repo-relative."
            )
        unit = match.group("unit") or ""
        if unit not in UNITS:
            raise ValueError(
                f"{doc} [{site}] cites `{inner}` with unit {unit!r}, which is "
                f"not in the declared multiplier table {sorted(UNITS)}."
            )
        out.append(Reference(doc, site, inner, match.group("path"),
                             match.group("keypath"), match.group("literal"), unit))
    return out


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

_JSON_CACHE: dict[Path, object] = {}
_TRACKED_CACHE: list[frozenset[str]] = []


def _TRACKED() -> frozenset[str]:
    """`git ls-files` once per session, not once per citation."""
    if not _TRACKED_CACHE:
        _TRACKED_CACHE.append(tracked_set(_REPO))
    return _TRACKED_CACHE[0]


def _load_json(target: Path):
    """Load once per path: the census reaches ~650 references over ~30 files."""
    if target not in _JSON_CACHE:
        _JSON_CACHE[target] = json.loads(target.read_text(encoding="utf-8"))
    return _JSON_CACHE[target]


def resolve(root: Path, ref: Reference):
    """Walk ``ref.keypath`` into the committed JSON at ``ref.path``."""
    target = root / ref.path
    if not target.exists():
        raise AssertionError(
            f"{ref.doc} [{ref.site}] cites `{ref.raw}`, but the artifact "
            f"{ref.path} does not exist."
        )
    # #928: existing is not committed. A results directory can hold a file that
    # lives in one checkout and no other (gitignored scratch, an un-added VESSL
    # leftover); a citation resolved against such a file is green here and
    # unresolvable in a fresh clone. Only asked of the repo tree -- the (B)-arm
    # falsifiers below resolve against scratch trees, where tracking is not a
    # question git can answer.
    if root == _REPO and git_available(_REPO) and ref.path not in _TRACKED():
        raise AssertionError(
            f"{ref.doc} [{ref.site}] cites `{ref.raw}`, but the artifact "
            f"{ref.path} is NOT git-tracked. It exists in this checkout only; "
            f"a fresh clone cannot resolve the citation. Commit the artifact "
            f"or cite one that is committed."
        )
    data = _load_json(target)
    node = data
    walked = ""
    for step in _STEP.finditer(ref.keypath):
        index, name = step.group(1), step.group(2)
        if index is not None:
            walked += f"[{index}]"
            if not isinstance(node, list) or int(index) >= len(node):
                raise AssertionError(
                    f"{ref.doc} [{ref.site}] cites `{ref.raw}`, but "
                    f"{ref.path}::{walked} is not a list index that exists "
                    f"(found {type(node).__name__})."
                )
            node = node[int(index)]
        else:
            walked = f"{walked}.{name}" if walked else name
            if not isinstance(node, dict) or name not in node:
                available = sorted(node)[:12] if isinstance(node, dict) else []
                raise AssertionError(
                    f"{ref.doc} [{ref.site}] cites `{ref.raw}`, but "
                    f"{ref.path}::{walked} does not exist. "
                    f"Keys available at that level: {available}"
                )
            node = node[name]
    return node


def tolerance(literal: str, unit: str) -> tuple[float, float]:
    """Expected value and half-a-last-place tolerance for *literal*."""
    dec = Decimal(literal)
    scale = UNITS[unit]
    expected = float(dec) * scale
    exponent = dec.as_tuple().exponent
    assert isinstance(exponent, int)
    tol = 0.5 * float(Decimal(1).scaleb(exponent)) * scale
    return expected, tol


def check(root: Path, ref: Reference) -> None:
    """Resolve *ref* and, if it carries a literal, assert the value matches."""
    value = resolve(root, ref)
    if ref.literal is None:
        if isinstance(value, (dict, list)) or value is None:
            raise AssertionError(
                f"{ref.doc} [{ref.site}] cites `{ref.raw}`, which resolves to "
                f"{type(value).__name__}, not a value a document can cite."
            )
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(
            f"{ref.doc} [{ref.site}] cites `{ref.raw}` with a numeric literal, "
            f"but {ref.path}::{ref.keypath} holds {value!r} "
            f"({type(value).__name__})."
        )
    expected, tol = tolerance(ref.literal, ref.unit)
    if isinstance(value, int) and Decimal(ref.literal).as_tuple().exponent >= 0 \
            and ref.unit in ("", "Hz", "dB", "dBi", "m"):
        if value != int(Decimal(ref.literal)):
            raise AssertionError(
                f"NUMERIC PROVENANCE FAILURE\n"
                f"  document : {ref.doc} [{ref.site}]\n"
                f"  reference: `{ref.raw}`\n"
                f"  document says: {ref.literal}\n"
                f"  artifact holds: {value}\n"
                f"  the artifact holds an integer; the citation must match it "
                f"exactly."
            )
        return
    if abs(value - expected) > tol * (1.0 + 1e-9):
        raise AssertionError(
            f"NUMERIC PROVENANCE FAILURE\n"
            f"  document : {ref.doc} [{ref.site}]\n"
            f"  reference: `{ref.raw}`\n"
            f"  document says: {ref.literal}{(' ' + ref.unit) if ref.unit else ''}"
            f"  (= {expected!r} in artifact units, +/- {tol!r})\n"
            f"  artifact holds: {value!r}\n"
            f"  at {ref.path}::{ref.keypath}\n"
            f"  |difference| = {abs(value - expected)!r}. Either the artifact "
            f"moved and the document was not updated with it, or the document "
            f"restates a number it never re-derived."
        )


# --------------------------------------------------------------------------
# The opted-in surface (coverage cannot silently shrink)
# --------------------------------------------------------------------------

MANIFEST = "validation/crossval/manifest.json"
CV11_NOTE = "docs/design_notes/20260831_cv11_broad_e4_artifact_provenance.md"
# 2026-09-03: the lattice-witness standard is an evidence document of exactly the
# class this gate exists for -- it quotes ~30 measured numbers out of committed
# artifacts, and its first draft carried several that were reconstructed rather
# than read (see that note's section 5.3, "These are the run's numbers, not a
# reconstruction"). Opting it in is what stops that recurring.
LATTICE_NOTE = "docs/design_notes/20260903_lattice_witness_standard.md"
CV19_WITNESS_NOTE = "docs/design_notes/20260903_cv19_fdfd_unitarity_witness.md"
# 2026-09-04 (#888): the auxiliary-echo record invariant. The note's whole claim
# is a table of per-rung ratios read out of three committed artifacts, and its
# falsifier argument is "no committed rung is near 1.0" -- exactly the shape that
# is worthless if the numbers stop resolving. Opted in with its section 3.
AUX_ECHO_NOTE = "docs/design_notes/20260904_aux_echo_record_invariant.md"
# 2026-09-06 (#928): the public benchmarks page is the single largest carrier of
# measured numbers in the repository (93 references) and was NOT under this gate.
# Every public "Validated comparison" row quotes an artifact value; a page that
# says what is validated, with numbers nobody re-resolves, is the exact shape
# this gate exists for.
BENCHMARKS = "docs/public/guide/benchmarks.mdx"
# 2026-09-06 (#928): the enumerate-and-classify pass over docs/public/**/*.mdx
# and docs/design_notes/*.md (see CLASSIFICATION below) found eight further
# notes whose references already resolve, with no unparseable `::` span to trip
# the reader. Leaving a document that cites artifacts outside the gate because
# nobody opted it in is how coverage stays accidental, so they are opted in
# here; all eight were green at the commit that added them.
NEWLY_GATED_NOTES = (
    "docs/design_notes/20260901_patch_mode_identification_predeclaration.md",
    "docs/design_notes/20260902_cv24_nu_cavity_predeclaration.md",
    "docs/design_notes/estimator_resolution_regate.md",
    "docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md",
    "docs/design_notes/issue812_cv03_dispersion_regate_results.md",
    "docs/design_notes/issue812_cv17_cv18_geometry_sensitivity_predeclaration.md",
    "docs/design_notes/issue812_phase_identity_predeclaration.md",
    "docs/design_notes/issue812_phase_identity_results.md",
)

# Markdown documents, with the regex that cuts them into named sites.
MARKDOWN_SITES: dict[str, str] = {
    "validation/README.md": r"^\|\s*`(crossval/[^`]+)`",
    CV11_NOTE: r"^#+\s+(.*\S)\s*$",
    "docs/design_notes/20260901_numeric_provenance_gate.md": r"^#+\s+(.*\S)\s*$",
    LATTICE_NOTE: r"^#+\s+(.*\S)\s*$",
    BENCHMARKS: r"^#+\s+(.*\S)\s*$",
    **{note: r"^#+\s+(.*\S)\s*$" for note in NEWLY_GATED_NOTES},
    # 2026-09-03 (#884): the cv19 unitarity-witness note argues *from* the
    # committed self-test scalars, so the committed scalars it quotes are
    # opted in here rather than retyped and trusted.
    CV19_WITNESS_NOTE: r"^#+\s+(.*\S)\s*$",
    # 2026-09-04 (#888): see AUX_ECHO_NOTE above.
    AUX_ECHO_NOTE: r"^#+\s+(.*\S)\s*$",
    # 2026-09-10 (#931 lattice-ownership merge): two notes opted in because
    # they now carry a resolvable citation each -- see the CLASSIFICATION
    # comments below at the same date for why each one moved. (A third,
    # v18_waveguide_s_chain_plan.md, also gained a resolvable citation but
    # stays OUT of DOCUMENTS: it carries a second `::` span,
    # `nu_flux_ad::..._grad_finite_and_fd_consistent`, a test-name reference
    # this parser rejects by construction -- SYMBOL_SPAN_PARSER_SCOPE, not
    # GATED. Adding it here breaks collection for every doc.)
    "docs/design_notes/20260908_docs_truth_audit.md": r"^#+\s+(.*\S)\s*$",
    "docs/design_notes/chain_closure_contract.md": r"^#+\s+(.*\S)\s*$",
    "docs/design_notes/20260911_harminv_record_support.md": r"^#+\s+(.*\S)\s*$",
}

DOCUMENTS = (MANIFEST, *MARKDOWN_SITES)

# Sites that MUST carry at least this many value-checked references. Lowering a
# floor is a deliberate act that belongs in the same commit as the reason.
REQUIRED_SITES: dict[tuple[str, str], int] = {
    ("docs/design_notes/20260911_harminv_record_support.md", "Actual FDTD records"): 3,
    (MANIFEST, "11_waveguide_port_wr90"): 4,
    (MANIFEST, "15_patch_antenna_rt5880"): 3,
    (MANIFEST, "17_dielectric_sphere_mie"): 2,
    (MANIFEST, "18_wr90_iris_modematch"): 4,
    (MANIFEST, "19_wr90_iris_filter_aghanim"): 4,
    ("validation/README.md", "crossval/11_waveguide_port_wr90.py"): 4,
    # 2026-09-03: floors for the cv15/cv18 README rows re-based to the rows that
    # #847 / #846 rewrote after this gate was drafted (see the note, "Re-basing
    # two floors"); those rows cite in the value-checked form and carry 2 each.
    ("validation/README.md", "crossval/15_patch_antenna_rt5880.py"): 2,
    ("validation/README.md", "crossval/17_dielectric_sphere_mie.py"): 3,
    ("validation/README.md", "crossval/18_wr90_iris_modematch.py"): 2,
    ("validation/README.md", "crossval/19_wr90_iris_filter_aghanim.py"): 6,
    (CV11_NOTE, "7. Numeric provenance (appended 2026-09-01, #812 round 2 \u2014 no finding changed)"): 6,
    # 2026-09-03, the lattice-witness standard. Section 5.3 is the one the review
    # found reconstructed numbers in; its floor is the point of opting the note in.
    (LATTICE_NOTE, "5.1 cv23 \u2014 nine committed entries, eight distinct meshes, all green"): 4,
    (LATTICE_NOTE, "5.2 cv22 \u2014 three rungs, all green; the pole lattice predicts the residual a priori"): 8,
    # 2026-09-10: cv04's settling-extension fix landed (PI override of this
    # section's own 8.3, "no new physics" -- see the commit message). Two
    # passes, both reproduced with this file's own parser rather than
    # asserted:
    #   Pass 1 (c9b86b5e): PRE-FIX the section carried 16 live citations.
    #   `lattice_witness.json` was regenerated in place and no longer holds
    #   the pre-fix 719-step values under those keys, so 10 were demoted to
    #   plain historical text (6 stayed live, unaffected keys -- that 6 is
    #   already inside the 16 - 10, not an addition to it) and 7 new live
    #   citations to the POST-FIX 990-step record were added:
    #   16 - 10 + 7 = 13 (the 6 that stayed live are the 16 - 10).
    #   Pass 2 (PR #974 adversarial review): re-checked the 10 demotions and
    #   found only 4 were forced -- W_witness,R/T and the worst-per-bin
    #   ratios R/T have no other committed home, so they stay plain text
    #   with a `git show e079b0b5:...` retrieval note. The other 6 were
    #   re-lived by pointing them at sources the fix does not change: the
    #   a-priori ceiling and \u0393 (ringdown rate) are geometry-derived and
    #   numerically unchanged, so they now cite the CURRENT artifact (+2);
    #   the four |rfx-lattice| gated-mean citations (dR, dT, each appearing
    #   twice) now cite the IMMUTABLE r1 revision of `envelope.json`, which
    #   archived them before the fix and does not move when the producer is
    #   re-run (+4, previously plain text under the same keys). The section
    #   also gained 2 live citations to the re-run F2/F3 falsifier
    #   separations at the settled rung, which now discriminate where they
    #   previously did not: 13 + 2 + 4 + 2 = 21.
    # Floor set to the reproduced count (21), not a round number.
    (LATTICE_NOTE, "5.3 cv04 \u2014 the witness was REPORTED, not gated; the derivation said why, and the settling-extension fix (2026-09-10) closed it"): 21,
    (LATTICE_NOTE, "8.1 cv22 Debye at a 3e-4 settling bar (the only rung a claim requires)"): 3,
    # 2026-09-03 (#884): the cv19 witness note's two load-bearing sections. §6.2
    # cites the committed unitarity that U3's floor is compared against; §6.3
    # cites the empty_s11 and r=2 anchor values it says were deliberately NOT
    # changed. A "we changed nothing" claim is worth exactly as much as its
    # numbers still resolving.
    (CV19_WITNESS_NOTE, "6.2 The three checks, with the shipped measurements"): 1,
    (CV19_WITNESS_NOTE, "6.3 What did NOT change"): 2,
    # 2026-09-04 (#888): section 3 IS the finding -- 13 per-rung ratios plus
    # cv04's arrival, record and fitted reflector index, every one of them read
    # back out of the artifact the run wrote rather than retyped from a note.
    (AUX_ECHO_NOTE, "3. The per-case ratios, read from the committed artifacts"): 18,
    # 2026-09-06 (#928): the public benchmarks table. This is the page a reader
    # takes "validated" from, so its measured numbers are the ones that must
    # keep resolving. 93 references parse there; 91 of them carry a value,
    # which is what this floor counts (the other two are existence-only).
    (BENCHMARKS, "Reference cases"): 85,
}

# Anti-vacuity census. A green gate must mean the references are right, not that
# somebody deleted them.
# 2026-09-03 (#884): +8, the cv19 unitarity-witness note's citations. Raised in
# the same commit that adds them, so the census tracks the opted-in surface.
# 2026-09-04 (#888): +18, the auxiliary-echo record invariant's section 3, and
# +1 distinct artifact (cv04's lattice_witness.json, cited here for the first
# time). Raised in the same commit that adds them.
# 2026-09-06 (#928): +benchmarks.mdx (93) and +8 design notes, all green at the
# commit that opted them in; the population went 566 -> 1147 references (+581)
# over 60 -> 64 distinct artifacts. Raised here in the same commit, as above.
MIN_REFERENCES = 1140
MIN_VALUE_CHECKED = 1095
MIN_DISTINCT_ARTIFACTS = 60


# --------------------------------------------------------------------------
# Enumerate-and-classify (methodology 2.3): which documents are under this
# gate, and -- for every one that is not -- a reason this file can CHECK.
#
# Before #928 the opted-in set was a hand-kept tuple and its complement was
# invisible: a new evidence document simply was not gated, and nothing said so.
# The surface is enumerated dynamically below (docs/public/**/*.mdx and
# docs/design_notes/*.md); an unclassified file fails.
#
# The three classifications, and what makes each one falsifiable:
#   GATED                     - in DOCUMENTS; every reference is resolved above.
#   NO_ARTIFACT_REFERENCE     - the document contains NO backtick span that
#                               parses as `<path>.json::<key>`. Verified here,
#                               so "it has no citations" cannot be an excuse a
#                               document quietly outgrows.
#   SYMBOL_SPAN_PARSER_SCOPE  - the document DOES carry resolvable artifact
#                               references AND at least one `::` span this
#                               parser rejects by construction (`module.py::sym`,
#                               `expected_metrics[id=...]`, an elided `…::key`).
#                               Opting it in would raise on the symbol spans, so
#                               the blocker is the reference syntax, not the
#                               document. Both halves are verified.
# There is deliberately NO "resolvable but not opted in" class: that would be a
# note saying the gate could have covered a document and chose not to.
# --------------------------------------------------------------------------

GATED = "gated"
NO_ARTIFACT_REFERENCE = "no-artifact-reference"
SYMBOL_SPAN_PARSER_SCOPE = "symbol-span-parser-scope"

CLASSIFIED_DOC_DIRS = ("docs/public/**/*.mdx", "docs/design_notes/*.md")

CLASSIFICATION: dict[str, str] = {
    "docs/public/api/automation.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/geometry-materials.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/results-observables.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/simulation.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/sources-ports.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/support-boundaries.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/examples/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/gallery/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/gallery/multilayer_fresnel.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/gallery/patch_antenna.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/gallery/waveguide_wr90.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/adi-solver.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/api-reference.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/autodiff-adjoint.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/benchmarks.mdx": GATED,
    "docs/public/guide/changelog.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/first-patch.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/installation.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/materials-geometry.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/memory-reduction.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/nonuniform-mesh.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/parametric-sweeps.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/probes-sparams.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/quickstart.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/sources-ports.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/studio-experiments.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/tutorial-convergence.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/tutorial-patch-antenna.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/validation.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/validation/cross-solver.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/validation/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/validation/recommended-configuration.mdx": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260829_spec01_multiband_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260830_issue786_convergence_floor.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260831_cv02_ring_judge_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260831_cv11_broad_e4_artifact_provenance.md": GATED,
    "docs/design_notes/20260901_numeric_provenance_gate.md": GATED,
    "docs/design_notes/20260901_patch_mode_identification_predeclaration.md": GATED,
    "docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md": SYMBOL_SPAN_PARSER_SCOPE,
    "docs/design_notes/20260902_cv23_lossy_slab_predeclaration.md": SYMBOL_SPAN_PARSER_SCOPE,
    "docs/design_notes/20260902_cv24_nu_cavity_predeclaration.md": GATED,
    "docs/design_notes/20260902_test_reorg_tier4b_plan.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260903_cv19_fdfd_unitarity_witness.md": GATED,
    "docs/design_notes/20260903_e4_all_solver_classes_plan.md": SYMBOL_SPAN_PARSER_SCOPE,
    "docs/design_notes/20260903_lattice_witness_standard.md": GATED,
    "docs/design_notes/20260903_test_reorg_tier3b_consolidation.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260904_aux_echo_record_invariant.md": GATED,
    "docs/design_notes/20260905_post_merge_review_20_prs.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260905_v18_close_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260906_issue928_ownership_decision.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260906_plan_realign_lattice_ownership.md": NO_ARTIFACT_REFERENCE,
    # 2026-09-11 (nu band-profile lane, PR #956): three `::` spans, all of the
    # form `rfx/auto_config.py::_make_dz_profile` -- module-symbol references,
    # not artifact paths (no `.json`), so they land in `others` and never in
    # `parses`. The note's F1-F8 numbers are not quoted out of a committed
    # artifact by `path::key` at all; they are replayed from
    # `validation/research/multiband_nu/results/w6_band_builder.json` by
    # tests/unit/nonuniform/test_band_builder_chain_model.py, which is a
    # stronger check than this gate performs. Same shape as
    # 20260910_cv05_crossval_disposition.md above.
    "docs/design_notes/20260907_nu_band_profile_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260908_adi_interior_pec_guard.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260908_automesh_regressions.md": NO_ARTIFACT_REFERENCE,
    # 2026-09-10 (#931 lattice-ownership merge): a single `::referee` span
    # cited a dict (DT-F01), which is what test_every_enumerated_document_is_
    # classified caught. Repointed to the leaf the finding's sentence
    # actually describes (argmin_first_null.note) and opted in -- an audit
    # document's whole job is naming real paths.
    "docs/design_notes/20260908_docs_truth_audit.md": GATED,
    "docs/design_notes/20260908_docs_truth_field_ledger.md": NO_ARTIFACT_REFERENCE,
    # 2026-09-10 (cv05 crossval disposition, issues #959/#965): one `::` span,
    # `test_patch_mode_identification.py::test_cv05_constants_...` -- a
    # test-name reference, not an artifact path (no `.json`), so it lands in
    # `others` and never `parses`; NO_ARTIFACT_REFERENCE only checks `parses`.
    "docs/design_notes/20260910_cv05_crossval_disposition.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260911_harminv_record_support.md": GATED,
    # 2026-09-10 (#931 lattice-ownership merge): 591e296e added a resolvable
    # citation to this note (cv18's Richardson envelope); opted in rather than
    # left failing NO_ARTIFACT_REFERENCE's own vacuity check.
    "docs/design_notes/chain_closure_contract.md": GATED,
    "docs/design_notes/cv10_pmc_realization_regate.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/cv14_rect_cavity_gate_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/estimator_resolution_regate.md": GATED,
    "docs/design_notes/geometry_setup_interop.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/graded_z_lowz_demo_closure.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/graded_z_lowz_demo_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/i489_stage2_two_port_fdtd_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/i636_cpml_pole_pad_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue683_decomposer_flip_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue683_sampling_order_decision_protocol.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue763_dz_profile_preserve_regions.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue764_wireport_norm_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue764_wireport_norm_results.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue770_offdiag_adjudication_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue770_offdiag_adjudication_results.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue782_retired_resonance_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue802_807_rasterization_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue808_debye_pad_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue811_dz_dispatch_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md": GATED,
    "docs/design_notes/issue812_cv03_dispersion_regate_results.md": GATED,
    "docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md": SYMBOL_SPAN_PARSER_SCOPE,
    "docs/design_notes/issue812_cv09_mirror_plane_regate.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/issue812_cv17_cv18_geometry_sensitivity_predeclaration.md": GATED,
    "docs/design_notes/issue812_phase_identity_predeclaration.md": GATED,
    "docs/design_notes/issue812_phase_identity_results.md": GATED,
    "docs/design_notes/mixed_refplane_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/portgrid_m0m1_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/portgrid_m0m1_results.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/portgrid_m1b_retry_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/preflight_lessons_from_a_long_crossval.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/thin_sheet_plane_bc_direction.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/thru_feedpost_deembed_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/thru_feedpost_joint_extraction_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/thru_feedpost_junction_windows_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/thru_feedpost_twoseg_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/thru_singular_value_dx_ladder_predeclaration.md": NO_ARTIFACT_REFERENCE,
    # 2026-09-10 (#931 lattice-ownership merge): 591e296e added a resolvable
    # cv19 citation here too (missing its tests/fixtures/ prefix, fixed in the
    # same pass) -- but unlike chain_closure_contract.md this document ALSO
    # carries `nu_flux_ad::..._grad_finite_and_fd_consistent` in its timing
    # table, a test-name reference the artifact-reference parser rejects by
    # construction. GATED would break collection for the whole test module;
    # SYMBOL_SPAN_PARSER_SCOPE is the classification that is actually true.
    "docs/design_notes/v18_waveguide_s_chain_plan.md": SYMBOL_SPAN_PARSER_SCOPE,
    "docs/design_notes/waveguide_chain_battery_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_chain_battery_remeasure_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_false_lane_column_power_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_false_lane_column_power_results.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_vi_envelope_sweep_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_vi_envelope_sweep_results.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/wp4e_lumped_component_value_ad_spike.md": NO_ARTIFACT_REFERENCE,
}


def _enumerate_classified_docs(root: Path) -> list[str]:
    found: list[str] = []
    for pattern in CLASSIFIED_DOC_DIRS:
        base, _, glob = pattern.partition("/**/")
        if glob:
            found += [str(p.relative_to(root)) for p in (root / base).rglob(glob)]
        else:
            base, _, glob = pattern.rpartition("/")
            found += [str(p.relative_to(root)) for p in (root / base).glob(glob)]
    return sorted(found)


def _reference_spans(root: Path, doc: str) -> tuple[list[str], list[str]]:
    """(parseable artifact references, other ``::`` spans) in *doc*."""
    text = strip_code_blocks((root / doc).read_text(encoding="utf-8"))
    spans = [s.group(1).strip() for s in _SPAN.finditer(text) if "::" in s.group(1)]
    parses = [s for s in spans if _REFERENCE.match(s)]
    others = [s for s in spans if not _REFERENCE.match(s)]
    return parses, others


def _sites(root: Path, doc: str) -> list[tuple[str, str]]:
    text = (root / doc).read_text(encoding="utf-8")
    if doc == MANIFEST:
        cases = json.loads(text)["cases"]
        return [(case["id"], json.dumps(case, ensure_ascii=False)) for case in cases]
    text = strip_code_blocks(text)
    pattern = re.compile(MARKDOWN_SITES[doc], re.MULTILINE)
    marks = list(pattern.finditer(text))
    out: list[tuple[str, str]] = []
    if marks and marks[0].start() > 0:
        out.append(("<preamble>", text[: marks[0].start()]))
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((mark.group(1), text[mark.start():end]))
    if not marks:
        out.append(("<whole document>", text))
    return out


def collect(root: Path) -> list[Reference]:
    refs: list[Reference] = []
    for doc in DOCUMENTS:
        for site, text in _sites(root, doc):
            refs.extend(parse_references(doc, site, text))
    return refs


_REFS = collect(_REPO)


@pytest.mark.parametrize(
    "ref", _REFS, ids=[f"{r.doc.split('/')[-1]}:{r.site}:{r.keypath}" for r in _REFS]
)
def test_every_cited_number_matches_its_artifact(ref: Reference) -> None:
    check(_REPO, ref)


def test_the_cited_population_is_still_present() -> None:
    value_checked = [r for r in _REFS if r.literal is not None]
    artifacts = {r.path for r in _REFS}
    assert len(_REFS) >= MIN_REFERENCES, (
        f"only {len(_REFS)} artifact references across {list(DOCUMENTS)}; "
        f"expected at least {MIN_REFERENCES}. If references were legitimately "
        f"removed, lower MIN_REFERENCES in the same commit and say why."
    )
    assert len(value_checked) >= MIN_VALUE_CHECKED, (
        f"only {len(value_checked)} of {len(_REFS)} references carry a value to "
        f"check; expected at least {MIN_VALUE_CHECKED}. Existence-only "
        f"references keep a key alive but assert no number."
    )
    assert len(artifacts) >= MIN_DISTINCT_ARTIFACTS, (
        f"references reach only {len(artifacts)} distinct artifacts "
        f"({sorted(artifacts)}); expected at least {MIN_DISTINCT_ARTIFACTS}."
    )


@pytest.mark.parametrize("site,floor", sorted(REQUIRED_SITES.items()),
                         ids=lambda v: str(v) if not isinstance(v, tuple) else f"{v[0].split('/')[-1]}:{v[1]}")
def test_each_registered_site_still_carries_its_references(site, floor) -> None:
    doc, name = site
    found = [r for r in _REFS if r.doc == doc and r.site == name and r.literal is not None]
    assert len(found) >= floor, (
        f"{doc} [{name}] carries {len(found)} value-checked artifact "
        f"references, below its declared floor of {floor}. A rewrite that drops "
        f"references drops the only mechanical link between this claim and its "
        f"evidence."
    )


def test_every_enumerated_document_is_classified() -> None:
    """No document in the two evidence-carrying trees may be unclassified.

    This is the half that stops the opted-in set from being accidental: adding
    a design note or a public page now fails here until someone says, in this
    table, whether it is gated and -- if not -- why the gate cannot reach it.
    """
    found = set(_enumerate_classified_docs(_REPO))
    listed = set(CLASSIFICATION)
    assert found - listed == set(), (
        f"unclassified documents: {sorted(found - listed)}. Add each to "
        f"CLASSIFICATION as {GATED!r} (and to MARKDOWN_SITES), "
        f"{NO_ARTIFACT_REFERENCE!r}, or {SYMBOL_SPAN_PARSER_SCOPE!r}."
    )
    assert listed - found == set(), (
        f"CLASSIFICATION lists documents that no longer exist: "
        f"{sorted(listed - found)}."
    )


@pytest.mark.parametrize("doc", sorted(CLASSIFICATION))
def test_each_classification_holds_mechanically(doc: str) -> None:
    """Each reason is checked, not asserted in prose."""
    kind = CLASSIFICATION[doc]
    parses, others = _reference_spans(_REPO, doc)
    if kind == GATED:
        assert doc in DOCUMENTS, f"{doc} is classified gated but is not in DOCUMENTS"
        return
    assert doc not in DOCUMENTS, f"{doc} is in DOCUMENTS but classified {kind}"
    if kind == NO_ARTIFACT_REFERENCE:
        assert not parses, (
            f"{doc} is classified {NO_ARTIFACT_REFERENCE!r} but now carries "
            f"{len(parses)} resolvable artifact reference(s), e.g. `{parses[0]}`. "
            f"Opt it into DOCUMENTS/MARKDOWN_SITES and re-classify it as gated."
        )
    elif kind == SYMBOL_SPAN_PARSER_SCOPE:
        assert parses, (
            f"{doc} is classified {SYMBOL_SPAN_PARSER_SCOPE!r} but carries no "
            f"resolvable artifact reference; it is {NO_ARTIFACT_REFERENCE!r}."
        )
        assert others, (
            f"{doc} is classified {SYMBOL_SPAN_PARSER_SCOPE!r} but every `::` "
            f"span in it now parses, so nothing blocks opting it in. Gate it."
        )
    else:  # pragma: no cover - the closed set is enforced here
        raise AssertionError(f"{doc}: unknown classification {kind!r}")


# --------------------------------------------------------------------------
# Criterion (B): the gate fails on the round-1 defects it exists for.
#
# These run against a scratch tree, never the repo, so they assert detection
# power without touching a committed artifact.
# --------------------------------------------------------------------------

# The measured round-1 cv15 finding: the lane regenerated the committed rfx leg
# and `s11_dip_db` moved from the value the documents quote to this one, with no
# prose moving with it.
CV15_ARTIFACT = "validation/crossval/_15_patch_results/rfx.json"
CV15_KEY = "s11_dip_db"
CV15_REGENERATED_VALUE = -0.3448069095611572


def _scratch(tmp_path: Path, artifact: str, key: str, value) -> Path:
    """A tree holding one artifact with one leaf replaced."""
    data = json.loads((_REPO / artifact).read_text(encoding="utf-8"))
    assert key in data, f"{artifact} no longer has a top-level {key}"
    data[key] = value
    dst = tmp_path / artifact
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def _cv15_dip_references() -> list[Reference]:
    return [r for r in _REFS
            if r.path == CV15_ARTIFACT and r.keypath == CV15_KEY and r.literal is not None]


def test_the_gate_fires_on_the_measured_cv15_regression(tmp_path: Path) -> None:
    """Round 1's cv15 defect, reproduced: artifact moves, prose does not."""
    refs = _cv15_dip_references()
    assert refs, (
        f"no committed document cites {CV15_ARTIFACT}::{CV15_KEY}; the (B) arm "
        f"of this gate's falsifier has nothing to fire on."
    )
    root = _scratch(tmp_path, CV15_ARTIFACT, CV15_KEY, CV15_REGENERATED_VALUE)
    for ref in refs:
        try:
            check(root, ref)  # (B): must go red on the regenerated leg
        except AssertionError as exc:
            message = str(exc)
        else:
            raise AssertionError(
                f"the gate did NOT fire: {ref.doc} [{ref.site}] now cites the "
                f"regenerated value {CV15_REGENERATED_VALUE!r} itself, so this "
                f"falsifier no longer falsifies. If cv15's leg was legitimately "
                f"regenerated, re-anchor criterion (B) on another measured "
                f"defect in the same commit that explains the move -- do not "
                f"delete the arm."
            )
        assert ref.doc in message
        assert ref.raw in message
        assert ref.literal in message                       # what the document says
        assert repr(CV15_REGENERATED_VALUE) in message      # what the artifact holds


def test_the_gate_fires_on_a_sign_inversion(tmp_path: Path) -> None:
    """Round 1's cv17/cv18 shape: the magnitude survives, the sign is inverted."""
    refs = _cv15_dip_references()
    assert refs
    ref = refs[0]
    expected, _tol = tolerance(ref.literal, ref.unit)
    root = _scratch(tmp_path, CV15_ARTIFACT, CV15_KEY, -expected)
    with pytest.raises(AssertionError) as caught:
        check(root, ref)
    assert repr(-expected) in str(caught.value)


def test_the_gate_fires_on_a_drift_below_the_last_digit_written(tmp_path: Path) -> None:
    """Precision is what was typed: a move under half a last place is silent,
    a move over it is not. This pins the boundary rather than assuming it.

    Anchored on the literal the document carries, not on the tree, so it
    measures the rule and not today's artifact."""
    ref = _cv15_dip_references()[0]
    expected, tol = tolerance(ref.literal, ref.unit)
    inside = _scratch(tmp_path / "in", CV15_ARTIFACT, CV15_KEY, expected + 0.4 * tol)
    check(inside, ref)
    outside = _scratch(tmp_path / "out", CV15_ARTIFACT, CV15_KEY, expected + 2.0 * tol)
    with pytest.raises(AssertionError):
        check(outside, ref)


def test_an_unresolvable_key_is_an_error_not_a_skip(tmp_path: Path) -> None:
    """A renamed key must fail loudly; a silent skip is how coverage rots."""
    ref = _cv15_dip_references()[0]
    data = json.loads((_REPO / CV15_ARTIFACT).read_text(encoding="utf-8"))
    data.pop(CV15_KEY)
    dst = tmp_path / CV15_ARTIFACT
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AssertionError, match="does not exist"):
        check(tmp_path, ref)


def test_the_gate_fires_on_a_present_but_untracked_artifact() -> None:
    """#928's (B) arm: an artifact that exists here and in no clone.

    Written into the repo tree deliberately -- the property under test is
    "present but untracked", which cannot be staged anywhere else -- and
    removed in the finally.
    """
    if not git_available(_REPO):
        pytest.skip("git unavailable; tracking is not a question git can answer")
    probe_rel = "docs/design_notes/.untracked_probe_928.json"
    probe = _REPO / probe_rel
    # A killed run used to leave this file behind and the next run failed its
    # own precondition (round-2 item 8). A leftover is removed and reported,
    # not treated as a failure: the file is this test's, and it is untracked by
    # construction.
    if probe.exists():
        print(f"removing a leftover probe from an interrupted run: {probe_rel}")
        probe.unlink()
    ref = Reference("doc", "site", f"{probe_rel}::value = 1", probe_rel, "value", "1", "")
    try:
        probe.write_text(json.dumps({"value": 1.0}), encoding="utf-8")
        assert probe_rel not in tracked_set(_REPO)
        with pytest.raises(AssertionError, match="NOT git-tracked"):
            check(_REPO, ref)
    finally:
        probe.unlink(missing_ok=True)
    # and the control: the same shape resolves when the artifact IS tracked.
    assert MANIFEST in tracked_set(_REPO)


def test_a_malformed_reference_is_rejected() -> None:
    """A typo in the syntax must not read as 'no reference here'."""
    with pytest.raises(ValueError, match="does not parse"):
        parse_references("doc", "site", "the value is `results/x.json::a b c`")
    with pytest.raises(ValueError, match="not in the declared multiplier table"):
        parse_references("doc", "site", "the value is `results/x.json::a = 1.0 furlongs`")

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

    `validation/crossval/_15_patch_results/openems.json::f_dip_hz`
    `validation/crossval/_15_patch_results/openems.json::f_dip_hz = 2330000000`
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

import functools
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from tests._git_tracked import git_available, tracked_set

pytestmark = pytest.mark.docs_consistency

_REPO = Path(__file__).resolve().parents[2]


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
    """Walk ``ref.keypath`` into the committed JSON at ``ref.path``.
    A citation resolved against a file that lives in one checkout and no other is unresolvable in a fresh clone."""
    target = root / ref.path
    if not target.exists():
        raise AssertionError(
            f"{ref.doc} [{ref.site}] cites `{ref.raw}`, but the artifact "
            f"{ref.path} does not exist."
        )
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


CV11_NOTE = "docs/design_notes/20260831_cv11_broad_e4_artifact_provenance.md"
LATTICE_NOTE = "docs/design_notes/20260903_lattice_witness_standard.md"
CV19_WITNESS_NOTE = "docs/design_notes/20260903_cv19_fdfd_unitarity_witness.md"
AUX_ECHO_NOTE = "docs/design_notes/20260904_aux_echo_record_invariant.md"
CPML_STAGGER_NOTE = "docs/design_notes/2026-09-13_cpml_yee_stagger_correction.md"
CV01_CPML_NOTE = "docs/design_notes/cv01_cpml_flux_selfcheck_predeclaration.md"
BENCHMARKS = "docs/public/guide/benchmarks.mdx"
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

ISSUE1043_STABILITY_NOTE = (
    "docs/design_notes/issue1043_cpml_subpixel_coefficient_results.md")
ISSUE1043_F1_NOTE = (
    "docs/design_notes/issue1043_f1_pec_short_gate_results.md")
ISSUE831_PREDECLARATION = (
    "docs/design_notes/issue831_far_end_return_predeclaration.md")
ISSUE831_RESULTS = "docs/design_notes/issue831_far_end_return_results.md"
ISSUE1043_PAD_CONTINUATION_NOTE = (
    "docs/design_notes/issue1043_pad_continuation_results.md")
TILT_RESULTS = (
    "docs/design_notes/waveguide_false_lane_transmission_tilt_results.md")
NEAR_FIELD_RESULTS = (
    "docs/design_notes/waveguide_driven_plane_near_field_composition_results.md")
OPEN_BOUNDARY_CONTRACT = "docs/design_notes/20260921_open_boundary_contract.md"

MARKDOWN_SITES: dict[str, str] = {
    "validation/README.md": r"^\|\s*`(crossval/[^`]+)`",
    CV11_NOTE: r"^#+\s+(.*\S)\s*$",
    "docs/design_notes/20260901_numeric_provenance_gate.md": r"^#+\s+(.*\S)\s*$",
    LATTICE_NOTE: r"^#+\s+(.*\S)\s*$",
    BENCHMARKS: r"^#+\s+(.*\S)\s*$",
    **{note: r"^#+\s+(.*\S)\s*$" for note in NEWLY_GATED_NOTES},
    CV19_WITNESS_NOTE: r"^#+\s+(.*\S)\s*$",
    AUX_ECHO_NOTE: r"^#+\s+(.*\S)\s*$",
    "docs/design_notes/20260908_docs_truth_audit.md": r"^#+\s+(.*\S)\s*$",
    "docs/design_notes/chain_closure_contract.md": r"^#+\s+(.*\S)\s*$",
    "docs/design_notes/20260911_harminv_record_support.md": r"^#+\s+(.*\S)\s*$",
    CV01_CPML_NOTE: r"^#+\s+(.*\S)\s*$",
    ISSUE1043_STABILITY_NOTE: r"^#+\s+(.*\S)\s*$",
    ISSUE1043_F1_NOTE: r"^#+\s+(.*\S)\s*$",
    ISSUE831_PREDECLARATION: r"^#+\s+(.*\S)\s*$",
    ISSUE831_RESULTS: r"^#+\s+(.*\S)\s*$",
    ISSUE1043_PAD_CONTINUATION_NOTE: r"^#+\s+(.*\S)\s*$",
    TILT_RESULTS: r"^#+\s+(.*\S)\s*$",
    NEAR_FIELD_RESULTS: r"^#+\s+(.*\S)\s*$",
    OPEN_BOUNDARY_CONTRACT: r"^#+\s+(.*\S)\s*$",
}

DOCUMENTS = tuple(MARKDOWN_SITES)

REQUIRED_SITES: dict[tuple[str, str], int] = {
    ("docs/design_notes/20260911_harminv_record_support.md", "Actual FDTD records"): 3,
    (TILT_RESULTS, "7. Numeric provenance"): 54,
    (CV11_NOTE, "7. Numeric provenance (appended 2026-09-01, #812 round 2 \u2014 no finding changed)"): 6,
}

MIN_REFERENCES = 277
MIN_VALUE_CHECKED = 264
MIN_DISTINCT_ARTIFACTS = 25


GATED = "gated"
NO_ARTIFACT_REFERENCE = "no-artifact-reference"
SYMBOL_SPAN_PARSER_SCOPE = "symbol-span-parser-scope"

CLASSIFIED_DOC_DIRS = ("docs/public/**/*.mdx", "docs/design_notes/*.md")

CLASSIFICATION: dict[str, str] = {
    "docs/design_notes/20260928_model_assembly_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20261004_showcase_lens_filter_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20261029_remove_adi_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20261010_absorber_curl_coefficient.md": NO_ARTIFACT_REFERENCE,
    "docs/public/api/generated/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/machine-readable.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/examples/teaching/boundary-reflection.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/examples/teaching/cavity-standing-wave.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/examples/teaching/dielectric-interface.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/first-run.mdx": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260927_public_docs_manual_examples_api_plan.md": NO_ARTIFACT_REFERENCE,
    "docs/public/api/functions.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/geometry-materials.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/results.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/simulation.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/api/sources-ports.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/support/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/examples/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/api-reference.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/design/gradients.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/benchmarks.mdx": GATED,
    "docs/public/project/changelog.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/project/deprecation-policy.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/project/structure.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/start/concepts.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/examples/first-patch.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/start/install.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/model/materials.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/run/memory-and-speed.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/model/domain-mesh.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/design/sweeps.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/design/pareto.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/measure/s-parameters.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/start/first-simulation.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/start/terms.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/start/upgrading.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/model/boundaries.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/model/conductors.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/model/sources.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/check/model-on-grid.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/check/refusals.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/run/run-and-forward.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/run/multi-device.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/measure/probes-and-fields.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/measure/resonances.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/check/simulation-time.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/check/gradients.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/check/gradient-simulation-time.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/check/agreement.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/design/design-regions.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/support/known-limitations.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/project/citation.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/model/ports.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/check/mesh-convergence.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/examples/patch-antenna.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/guide/validation.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/showcase/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/showcase/patch-calibration.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/showcase/grin-lens.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/showcase/waveguide-filter.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/validation/cross-solver.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/validation/index.mdx": NO_ARTIFACT_REFERENCE,
    "docs/public/validation/recommended-configuration.mdx": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/slab_family_per_arm_lattice_window_predeclaration.md": NO_ARTIFACT_REFERENCE,
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
    "docs/design_notes/20260902_cv26_oblique_fresnel_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260904_cv26_round3_close.md": SYMBOL_SPAN_PARSER_SCOPE,
    "docs/design_notes/20260904_aux_absorber_depth_derivation.md": SYMBOL_SPAN_PARSER_SCOPE,
    "docs/design_notes/20260905_post_merge_review_20_prs.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260905_v18_close_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260906_issue928_ownership_decision.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260906_plan_realign_lattice_ownership.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260907_nu_band_profile_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260910_cpml_localization_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260907_nu_cost_reduction_plan.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260907_nu_exp4_diff_stackup_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260907_nu_exp2_thin_layer_subpixel_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260907_nu_exp3_interface_eps_rule_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260907_nu_exp1_band_law_sweep_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260907_nu_band_accuracy_ad_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260913_nu_ad_directional_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260913_nu_ad_designvar_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260915_nu_lane2b_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260913_nu_full_functionality_program.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260913_nu_lane1_multilevel_xyz_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260913_nu_lane2_inplane_designvar_ad_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260908_adi_interior_pec_guard.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260908_automesh_regressions.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260908_docs_truth_audit.md": GATED,
    "docs/design_notes/20260908_docs_truth_field_ledger.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260910_cv05_crossval_disposition.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260911_harminv_record_support.md": GATED,
    "docs/design_notes/20260914_lattice_witness_gl1_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/717_crossval_lane_decision.md": NO_ARTIFACT_REFERENCE,
    CPML_STAGGER_NOTE: NO_ARTIFACT_REFERENCE,
    "docs/design_notes/chain_closure_contract.md": GATED,
    CV01_CPML_NOTE: GATED,
    "docs/design_notes/coax_conductor_realization.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/realized_geometry_record.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/cv10_pmc_realization_regate.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/cv14_rect_cavity_gate_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/estimator_resolution_regate.md": GATED,
    "docs/design_notes/geometry_setup_interop.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/graded_mesh_sparameter_accuracy_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/graded_mesh_sparameter_accuracy_results.md": NO_ARTIFACT_REFERENCE,
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
    "docs/design_notes/issue1043_cpml_subpixel_coefficient_predeclaration.md":
        NO_ARTIFACT_REFERENCE,
    ISSUE1043_STABILITY_NOTE: GATED,
    "docs/design_notes/issue1043_f1_pec_short_gate_predeclaration.md":
        NO_ARTIFACT_REFERENCE,
    ISSUE1043_F1_NOTE: GATED,
    ISSUE831_PREDECLARATION: GATED,
    ISSUE831_RESULTS: GATED,
    ISSUE1043_PAD_CONTINUATION_NOTE: GATED,
    "docs/design_notes/lumped_wire_chain_battery_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/mixed_refplane_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/msl_chain_battery_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/coax_chain_battery_predeclaration.md": NO_ARTIFACT_REFERENCE,
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
    "docs/design_notes/v18_waveguide_s_chain_plan.md": SYMBOL_SPAN_PARSER_SCOPE,
    "docs/design_notes/waveguide_chain_battery_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_chain_battery_remeasure_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_driven_plane_near_field_composition_predeclaration.md":
        NO_ARTIFACT_REFERENCE,
    NEAR_FIELD_RESULTS: GATED,
    OPEN_BOUNDARY_CONTRACT: GATED,
    "docs/design_notes/waveguide_false_lane_column_power_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_false_lane_column_power_results.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/waveguide_false_lane_transmission_tilt_predeclaration.md":
        NO_ARTIFACT_REFERENCE,
    TILT_RESULTS: GATED,
    "docs/design_notes/20260921_crossval_role_redesign.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260921_ntff_second_order.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260922_msl_notch_graded_mesh_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260922_msl_notch_fz_ladder_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260922_nu_grid_core_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260923_boundary_model_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260923_msl_notch_fz_after_1213_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260924_coax_open_closed_can_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260927_showcase_predeclaration.md": NO_ARTIFACT_REFERENCE,
    "docs/design_notes/20260927_lane_admission_allowlist_predeclaration.md": NO_ARTIFACT_REFERENCE,
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
    text = strip_code_blocks((root / doc).read_text(encoding="utf-8"))
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


REMOVED_ARTIFACT_PREFIXES: tuple[str, ...] = (
    "scripts/diagnostics/_artifacts/cv01_cpml_813/",
    "validation/crossval/_01_waveguide_bend_results/",
    "validation/crossval/_05_patch_results/",
    "tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json",
    "validation/crossval/_04_fresnel_results/",
    "validation/crossval/_04_fresnel_logs/",
    "validation/crossval/_17_dielectric_results/",
    "validation/crossval/_22_dispersive_results/",
    "validation/crossval/_22_dispersive_diag/",
    "validation/crossval/_23_lossy_results/",
    "validation/crossval/_24_nu_cavity_results/",
    "validation/crossval/_26_oblique_results/",
    "tests/fixtures/golden_workflows/",
    "tests/fixtures/rcs_dielectric_sphere_mie/",
    "tests/fixtures/slab_family_windows_baseline.json",
    "validation/crossval/_11_wr90_port_results/",
    "validation/crossval/_19_iris_filter_results/",
    "tests/fixtures/wr90_iris_filter/",
    "validation/crossval/_18_wr90_iris_results/",
    "tests/fixtures/wr90_iris_modematch/",
    "validation/crossval/_21_coax_two_port_referee_logs/",
    "validation/crossval/_06b_msl_notch_results/",
    "tests/fixtures/msl_notch_e4/",
    "tests/fixtures/cv06b_estimator_regate/",
    "validation/crossval/_07_sheen_results/",
    "tests/fixtures/cv07_estimator_regate/",
    "tests/fixtures/sheen_lpf_e4/",
    "validation/crossval/_15_patch_results/",
    "tests/fixtures/patch_mode_identification/",
)


def collect(root: Path, documents=DOCUMENTS) -> list[Reference]:
    refs: list[Reference] = []
    for doc in documents:
        for site, text in _sites(root, doc):
            refs.extend(r for r in parse_references(doc, site, text)
                        if not r.path.startswith(REMOVED_ARTIFACT_PREFIXES))
    return refs


FROZEN_CITATIONS = _REPO / "tests" / "fixtures" / "cited_record_values.json"
_NOTES = DOCUMENTS


@functools.lru_cache(maxsize=None)
def _refs() -> tuple[Reference, ...]:
    """Every citation in every gated document. Reads the notes, so only
    docs_consistency test bodies call it (tests/_prose_reads.py holds the rest
    to it)."""
    return tuple(collect(_REPO))


@functools.lru_cache(maxsize=None)
def _frozen_refs() -> tuple[Reference, ...]:
    with FROZEN_CITATIONS.open(encoding="utf-8") as fh:
        by_doc = json.load(fh)["citations"]
    return tuple(ref for doc, sites in by_doc.items() for site, spans in sites.items()
                 for span in spans for ref in parse_references(doc, site, f"`{span}`"))


def _ref_id(r: Reference) -> str:
    return f"{r.doc.split('/')[-1]}:{r.site}:{r.keypath}"


def _check_all(refs) -> None:
    failures = []
    for ref in refs:
        try:
            check(_REPO, ref)
        except AssertionError as exc:
            failures.append(str(exc))
    assert not failures, f"{len(failures)} citation(s) failed:\n\n" + "\n\n".join(failures)


@pytest.mark.docs_consistency
@pytest.mark.parametrize("doc", _NOTES)
def test_every_number_a_note_cites_matches_its_artifact(doc: str) -> None:
    _check_all(collect(_REPO, (doc,)))


@pytest.mark.parametrize("ref", _frozen_refs(), ids=[_ref_id(r) for r in _frozen_refs()])
def test_every_frozen_citation_still_matches_its_record(ref: Reference) -> None:
    """The record values the notes cited, to the precision they were written."""
    check(_REPO, ref)


@pytest.mark.docs_consistency
def test_the_cited_population_is_still_present() -> None:
    """The population floors preserve the merged-tree census measured earlier: 277 references, 264 value-checked references and 25 distinct artifacts."""
    refs = _refs()
    value_checked = [r for r in refs if r.literal is not None]
    artifacts = {r.path for r in refs}
    assert len(refs) >= MIN_REFERENCES, (
        f"only {len(refs)} artifact references across {list(DOCUMENTS)}; "
        f"expected at least {MIN_REFERENCES}. If references were legitimately "
        f"removed, lower MIN_REFERENCES in the same commit and say why."
    )
    assert len(value_checked) >= MIN_VALUE_CHECKED, (
        f"only {len(value_checked)} of {len(refs)} references carry a value to "
        f"check; expected at least {MIN_VALUE_CHECKED}. Existence-only "
        f"references keep a key alive but assert no number."
    )
    assert len(artifacts) >= MIN_DISTINCT_ARTIFACTS, (
        f"references reach only {len(artifacts)} distinct artifacts "
        f"({sorted(artifacts)}); expected at least {MIN_DISTINCT_ARTIFACTS}."
    )


@pytest.mark.parametrize("prefix", REMOVED_ARTIFACT_PREFIXES)
def test_a_skipped_artifact_prefix_is_really_gone(prefix: str) -> None:
    """A string that is no path at all would pass "is this path missing" and still silence citations to live files."""
    live = sorted(t for t in _TRACKED() if t.startswith(prefix))
    assert not live, (
        f"{prefix} is listed in REMOVED_ARTIFACT_PREFIXES, so citations under it "
        f"are skipped, but {len(live)} tracked file(s) start with it, e.g. "
        f"{live[0]}. Remove or narrow the prefix so those citations are checked."
    )


@pytest.mark.docs_consistency
@pytest.mark.parametrize("site,floor", sorted(REQUIRED_SITES.items()),
                         ids=lambda v: str(v) if not isinstance(v, tuple) else f"{v[0].split('/')[-1]}:{v[1]}")
def test_each_registered_site_still_carries_its_references(site, floor) -> None:
    """The transmission-tilt note's floor of 54 is the count of value-checked citations in its section 7, Numeric provenance."""
    doc, name = site
    found = [r for r in _refs() if r.doc == doc and r.site == name and r.literal is not None]
    assert len(found) >= floor, (
        f"{doc} [{name}] carries {len(found)} value-checked artifact "
        f"references, below its declared floor of {floor}. A rewrite that drops "
        f"references drops the only mechanical link between this claim and its "
        f"evidence."
    )


@pytest.mark.docs_consistency
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


@pytest.mark.docs_consistency
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


B_ANCHOR_ARTIFACT = (
    "scripts/diagnostics/open_boundary_contract/cont/n2_pad10_cpml4_a/result.json")
B_ANCHOR_KEY = "worst_rate_per_step"
B_ANCHOR_REGENERATED_VALUE = 0.0025469707764233167
B_ANCHOR_DEFECT_RECORD = "scripts/diagnostics/open_boundary_contract/result_main.json"


def _scratch(tmp_path: Path, artifact: str, key: str, value) -> Path:
    """A tree holding one artifact with one leaf replaced."""
    data = json.loads((_REPO / artifact).read_text(encoding="utf-8"))
    assert key in data, f"{artifact} no longer has a top-level {key}"
    data[key] = value
    dst = tmp_path / artifact
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def _anchor_references() -> list[Reference]:
    """The falsifiers must not read the notes to exercise the gate."""
    return [r for r in _frozen_refs()
            if r.path == B_ANCHOR_ARTIFACT and r.keypath == B_ANCHOR_KEY
            and r.literal is not None]


def test_the_anchor_is_the_measured_value() -> None:
    """The regenerated value the (B) arms plant is the one the defect tree
    measured for the same arm, read from its committed record -- not a number
    chosen to make the gate fire."""
    arms = json.loads((_REPO / B_ANCHOR_DEFECT_RECORD).read_text(encoding="utf-8"))["arms"]
    anchor = json.loads((_REPO / B_ANCHOR_ARTIFACT).read_text(encoding="utf-8"))
    same = [a for a in arms if a["arm"] == anchor["arm"]]
    assert len(same) == 1, (anchor["arm"], [a["arm"] for a in arms])
    assert anchor[B_ANCHOR_KEY] < 0 < same[0][B_ANCHOR_KEY], (
        "The continuation arm must decay while the matched defect arm grows."
    )
    assert same[0][B_ANCHOR_KEY] == B_ANCHOR_REGENERATED_VALUE


def test_the_gate_fires_on_a_measured_regression(tmp_path: Path) -> None:
    """A measured defect, reproduced: the artifact moves, the prose does not."""
    refs = _anchor_references()
    assert refs, (
        f"no committed document cites {B_ANCHOR_ARTIFACT}::{B_ANCHOR_KEY}; the (B) "
        f"arm of this gate's falsifier has nothing to fire on."
    )
    root = _scratch(tmp_path, B_ANCHOR_ARTIFACT, B_ANCHOR_KEY, B_ANCHOR_REGENERATED_VALUE)
    for ref in refs:
        try:
            check(root, ref)
        except AssertionError as exc:
            message = str(exc)
        else:
            raise AssertionError(
                f"the gate did NOT fire: {ref.doc} [{ref.site}] now cites the "
                f"regenerated value {B_ANCHOR_REGENERATED_VALUE!r} itself, so "
                f"this falsifier no longer falsifies. If the anchor record was "
                f"legitimately regenerated, re-anchor criterion (B) on another "
                f"measured defect in the same commit that explains the move -- "
                f"do not delete the arm."
            )
        assert ref.doc in message
        assert ref.raw in message
        assert ref.literal in message
        assert repr(B_ANCHOR_REGENERATED_VALUE) in message


def test_the_gate_fires_on_a_sign_inversion(tmp_path: Path) -> None:
    """Round 1's cv17/cv18 shape: the magnitude survives, the sign is inverted."""
    refs = _anchor_references()
    assert refs
    ref = refs[0]
    expected, _tol = tolerance(ref.literal, ref.unit)
    root = _scratch(tmp_path, B_ANCHOR_ARTIFACT, B_ANCHOR_KEY, -expected)
    with pytest.raises(AssertionError) as caught:
        check(root, ref)
    assert repr(-expected) in str(caught.value)


def test_the_gate_fires_on_a_drift_below_the_last_digit_written(tmp_path: Path) -> None:
    """Precision is what was typed: a move under half a last place is silent,
    a move over it is not. This pins the boundary rather than assuming it.

    Anchored on the literal the document carries, not on the tree, so it
    measures the rule and not today's artifact."""
    ref = _anchor_references()[0]
    expected, tol = tolerance(ref.literal, ref.unit)
    inside = _scratch(tmp_path / "in", B_ANCHOR_ARTIFACT, B_ANCHOR_KEY,
                      expected + 0.4 * tol)
    check(inside, ref)
    outside = _scratch(tmp_path / "out", B_ANCHOR_ARTIFACT, B_ANCHOR_KEY,
                       expected + 2.0 * tol)
    with pytest.raises(AssertionError):
        check(outside, ref)


def test_an_unresolvable_key_is_an_error_not_a_skip(tmp_path: Path) -> None:
    """A renamed key must fail loudly; a silent skip is how coverage rots."""
    ref = _anchor_references()[0]
    data = json.loads((_REPO / B_ANCHOR_ARTIFACT).read_text(encoding="utf-8"))
    data.pop(B_ANCHOR_KEY)
    dst = tmp_path / B_ANCHOR_ARTIFACT
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AssertionError, match="does not exist"):
        check(tmp_path, ref)


def test_the_gate_fires_on_a_present_but_untracked_artifact() -> None:
    """#928's (B) arm: an artifact that exists here and in no clone.

    Written into the repo tree deliberately -- the property under test is
    "present but untracked", which cannot be staged anywhere else -- and
    removed in the finally.
    A leftover is not treated as a failure because the file is this test's.
    """
    if not git_available(_REPO):
        pytest.skip("git unavailable; tracking is not a question git can answer")
    probe_rel = "docs/design_notes/.untracked_probe_928.json"
    probe = _REPO / probe_rel
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
    assert B_ANCHOR_ARTIFACT in tracked_set(_REPO)


def test_a_malformed_reference_is_rejected() -> None:
    """A typo in the syntax must not read as 'no reference here'."""
    with pytest.raises(ValueError, match="does not parse"):
        parse_references("doc", "site", "the value is `results/x.json::a b c`")
    with pytest.raises(ValueError, match="not in the declared multiplier table"):
        parse_references("doc", "site", "the value is `results/x.json::a = 1.0 furlongs`")

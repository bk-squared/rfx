"""Who owns a measured envelope, and what may hold a copy of one (issue #928).

The defect this exists for, measured on main: cv04 measured a band-mean R/T
envelope and never wrote it down. Its consumers (cv22, cv23) needed it to
derive their windows, so they cited the nearest committed copies -- a STUDIO UI
fixture and a source comment in cv04 -- and pinned themselves equal to those.
A display fixture had become a physics source, and a producer's evidence lived
in a module named after one of its consumers.

The three things this file keeps apart:

* **evidence**    -- what a producer measured, in the producer's own artifact,
                     append-only by revision, hashed;
* **calibration** -- which revision a consumer adopted, declared IN the
                     consumer with that revision's hash;
* **expectation** -- the window, derived from the two, written nowhere else.

The rules below are mechanical consequences of that split:

1. ownership -- an envelope artifact a comparator reads must be registered in
   the crossval manifest under the case whose id equals its ``producer``;
2. no self-certification in one step -- the artifact and an adoption record
   that names it may not change in the same branch diff (a re-run touches the
   artifact; an adoption touches the record), with a single bootstrap
   exemption per artifact for the revision that creates it;
3. fan-out -- every appearance of an adopted value inside the slab family is
   classified, and the classes are checked, not asserted in prose;
4. display copies -- a fixture that shows an adopted value must carry a
   reference to it that RESOLVES; equality is reported, not asserted, because
   a display fixture is allowed to lag a re-measurement (it is not a gate).

"Same measurement" (rule 4's scope) = the same realized-rig hash AND the same
estimator. Sibling artifacts that deliberately vary the rig -- cv23's dx2/dx4
ladders, cv22's longer-record diagnostic -- are NOT copies of each other and
are exempt by construction: their rigs differ, so nothing here asks them to
agree.

No FDTD, no physics, no gate value.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests._git_tracked import git_available, is_tracked

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "validation/crossval/manifest.json"
_COMPARATORS = _REPO / "validation/crossval/comparators"
_ENVELOPE_REL = "validation/crossval/_04_fresnel_results/envelope.json"
_ENVELOPE = _REPO / _ENVELOPE_REL
_GOLDEN = _REPO / "tests/fixtures/golden_workflows/multilayer_fresnel.json"


def _load_comparator(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_ownership_{name}", _COMPARATORS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _envelope_doc() -> dict:
    return json.loads(_ENVELOPE.read_text(encoding="utf-8"))


def _adoption_records() -> dict[str, dict]:
    """Every comparator module that declares a CV04_ADOPTION, by file name."""
    out: dict[str, dict] = {}
    for path in sorted(_COMPARATORS.glob("*.py")):
        if "CV04_ADOPTION" not in path.read_text(encoding="utf-8"):
            continue
        module = _load_comparator(path.stem)
        record = getattr(module, "CV04_ADOPTION", None)
        if isinstance(record, dict):
            out[str(path.relative_to(_REPO))] = record
    return out


# ---------------------------------------------------------------------------
# 1. Ownership: the manifest binds the artifact to its producer's case
# ---------------------------------------------------------------------------

def test_every_cited_envelope_artifact_is_registered_under_its_producer():
    records = _adoption_records()
    assert records, "no comparator declares an adoption record any more"
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    for consumer, record in records.items():
        rel = record["envelope"]
        assert (_REPO / rel).is_file(), f"{consumer} cites a missing artifact {rel}"
        producer = json.loads((_REPO / rel).read_text(encoding="utf-8"))["producer"]
        assert producer in cases, (
            f"{rel} names producer {producer!r}, which is not a crossval case")
        assert rel in cases[producer]["artifact_paths"], (
            f"{rel} is cited by {consumer} but is not listed under case "
            f"{producer} in the manifest. An artifact a consumer derives a gate "
            f"from is that case's committed evidence and must be registered as "
            f"such -- that registration is what makes the ownership structural "
            f"rather than a convention.")
        if git_available(_REPO):
            assert is_tracked(rel, _REPO)


def test_consumer_modules_carry_no_envelope_literal():
    """A consumer holds an adoption record, never a copy of the values.

    Before #928 both consumers held the four numbers inline. The measurable
    consequence of the split is that they now hold none: every window they
    enforce comes from the producer's artifact through the record.
    """
    for consumer in _adoption_records():
        text = (_REPO / consumer).read_text(encoding="utf-8")
        found = _LITERAL_RE.findall(text)
        assert not found, (
            f"{consumer} restates the adopted envelope value(s) {sorted(set(found))}. "
            f"A consumer derives from the artifact; it does not keep a copy.")


def test_the_artifact_declares_the_consumers_that_adopt_it():
    """Fan-out closure: the revision lists who derives what from it."""
    doc = _envelope_doc()
    records = _adoption_records()
    for name, block in doc["revisions"].items():
        listed = {entry["consumer"] for entry in block["adopted_by"]}
        for entry in block["adopted_by"]:
            assert (_REPO / entry["consumer"]).is_file(), entry["consumer"]
            assert entry["derives"], f"{name}/{entry['case']} derives nothing"
            assert (_REPO / entry["adopted_in"]).is_file(), entry["adopted_in"]
        adopters = {c for c, r in records.items() if r["adopted_revision"] == name}
        assert adopters == listed, (
            f"revision {name} says it is adopted by {sorted(listed)}, the "
            f"modules say {sorted(adopters)}")


# ---------------------------------------------------------------------------
# 2. No self-certification: evidence and calibration move in separate changes
# ---------------------------------------------------------------------------

def _diff_base() -> str | None:
    for ref in ("origin/main", "main"):
        result = subprocess.run(["git", "merge-base", ref, "HEAD"],
                                cwd=_REPO, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    result = subprocess.run(["git", "rev-parse", "HEAD~1"],
                            cwd=_REPO, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def test_bootstrap_is_used_at_most_once_per_artifact():
    doc = _envelope_doc()
    bootstraps = [n for n, b in doc["revisions"].items() if b.get("bootstrap")]
    assert len(bootstraps) <= 1, (
        f"{_ENVELOPE_REL} marks {bootstraps} as bootstrap revisions. The "
        f"exemption exists once, for the revision that creates the artifact; "
        f"every later revision lands without its adoptions.")
    if bootstraps:
        assert bootstraps == sorted(doc["revisions"])[:1], (
            "the bootstrap flag belongs to the FIRST revision, not a later one")


def test_the_producer_artifact_and_an_adoption_record_do_not_move_together():
    """A re-run touches the artifact. An adoption touches the record. A change
    that does both in one diff is a producer widening the gates that judge it.

    The bootstrap revision is exempt: the artifact and its first adoption
    cannot exist without each other.
    """
    if not git_available(_REPO):
        pytest.skip("git unavailable; the branch diff cannot be read")
    base = _diff_base()
    if base is None:
        pytest.skip("no diff base (shallow checkout?)")
    changed = subprocess.run(["git", "diff", "--name-only", f"{base}..HEAD"],
                             cwd=_REPO, capture_output=True, text=True)
    if changed.returncode != 0:
        pytest.skip("git diff unavailable")
    touched = set(changed.stdout.split())
    if _ENVELOPE_REL not in touched:
        return
    records = _adoption_records()
    also_touched = sorted(set(records) & touched)
    if not also_touched:
        return
    doc = _envelope_doc()
    bootstraps = {n for n, b in doc["revisions"].items() if b.get("bootstrap")}
    adopted = {r["adopted_revision"] for r in records.values()}
    assert adopted <= bootstraps, (
        f"this branch changes {_ENVELOPE_REL} AND the adoption record(s) in "
        f"{also_touched}, while the adopted revision(s) {sorted(adopted)} are "
        f"not the bootstrap revision. Split the change: append the revision in "
        f"one PR (evidence), adopt it in another (calibration, reviewed).")


# ---------------------------------------------------------------------------
# 3. Fan-out: every appearance of an adopted value, classified and checked
# ---------------------------------------------------------------------------

INPUT_SOURCE = "input-source"
RECORDED_OUTPUT = "recorded-output"
RESOLVING_REFERENCE = "resolving-reference"
ADOPTION_DECLARATION = "adoption-declaration"
PRODUCER_NOTE = "producer-note"
DISPLAY_WITH_SOURCE = "display-with-source"
BASELINE_IDENTITY = "baseline-identity"
HISTORICAL_PROSE = "historical-prose"
DIFFERENT_QUANTITY = "different-quantity"

# Produced artifacts record their OWN run's numbers, including the windows they
# were judged against; that is what makes those windows citable
# (`_22_dispersive_results/rfx.json::arms.debye.mean_window_R`, quoted by value
# in benchmarks.mdx). Recording a derived window as an OUTPUT is required, not
# forbidden -- the rule is one INPUT source. Everything under a case's results
# directory is therefore an output by construction.
RESULTS_DIR_RE = re.compile(r"^validation/crossval/_[0-9][0-9a-z]*_[a-z0-9_]+/")

# Files inside the slab family that carry one of the adopted literals, and what
# that appearance IS. A file that mentions the family and carries a literal but
# is neither under a results directory nor listed here fails the enumeration
# test below.
FANOUT: dict[str, str] = {
    _ENVELOPE_REL: INPUT_SOURCE,
    "validation/crossval/04_multilayer_fresnel.py": PRODUCER_NOTE,
    "validation/crossval/manifest.json": RESOLVING_REFERENCE,
    "validation/README.md": DIFFERENT_QUANTITY,
    "tests/fixtures/golden_workflows/multilayer_fresnel.json": DISPLAY_WITH_SOURCE,
    "tests/fixtures/slab_family_windows_baseline.json": BASELINE_IDENTITY,
    "tests/contracts/test_calibration_envelope_ownership.py": DIFFERENT_QUANTITY,
    "tests/crossval/test_cv22_dispersive_slab_gates.py": DIFFERENT_QUANTITY,
    "tests/crossval/test_crossval_gate_logic.py": DIFFERENT_QUANTITY,
    "tests/studio/test_interop_design_document.py": DIFFERENT_QUANTITY,
    "docs/public/gallery/assets/multilayer_fresnel/manifest.json": RECORDED_OUTPUT,
    "docs/public/gallery/multilayer_fresnel.mdx": DISPLAY_WITH_SOURCE,
    "docs/public/guide/benchmarks.mdx": RESOLVING_REFERENCE,
    "docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md": ADOPTION_DECLARATION,
    "docs/design_notes/20260902_cv23_lossy_slab_predeclaration.md": ADOPTION_DECLARATION,
    "docs/design_notes/20260903_lattice_witness_standard.md": RESOLVING_REFERENCE,
    "docs/design_notes/20260903_test_reorg_tier3b_consolidation.md": DIFFERENT_QUANTITY,
    "docs/design_notes/20260904_aux_echo_record_invariant.md": RESOLVING_REFERENCE,
    "docs/design_notes/20260905_post_merge_review_20_prs.md": HISTORICAL_PROSE,
    "docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md": HISTORICAL_PROSE,
}

# What a DISPLAY_WITH_SOURCE file must point at: the producer's envelope, or --
# for a page showing a DIFFERENT run of the same case -- that run's own
# artifact. Either way the number stops being a free-floating copy.
DISPLAY_SOURCE: dict[str, str] = {
    "tests/fixtures/golden_workflows/multilayer_fresnel.json": "envelope.json",
    "docs/public/gallery/multilayer_fresnel.mdx": "gallery/assets/multilayer_fresnel/manifest.json",
}

# Where the literals may legitimately appear without being this envelope: the
# reason, one line, per file. Only the DIFFERENT_QUANTITY class needs one.
DIFFERENT_QUANTITY_REASON: dict[str, str] = {
    "validation/README.md": "the public row quotes cv04's 0.05 gate limits and a 0.011 that is the case's own reported mean, not an input to any window",
    "tests/contracts/test_calibration_envelope_ownership.py": "this file lists the literals it searches for, and the reasons quote them",
    "tests/crossval/test_cv22_dispersive_slab_gates.py": "one docstring line naming the grep string that #928 deleted; every window in that file is re-derived from the artifact",
    "tests/crossval/test_crossval_gate_logic.py": "cv04's own per-bin closure ceiling test, 0.0487 against the 0.06 ceiling -- the producer's gate, not a consumer window",
    "tests/studio/test_interop_design_document.py": "a geometry centre coordinate that happens to read 0.011 m",
    "docs/design_notes/20260903_test_reorg_tier3b_consolidation.md": "a pytest node id containing a parametrized 0.011",
}

_LITERALS = ("0.0066", "0.011", "0.0487", "0.0091")
_LITERAL_RE = re.compile(r"(?<![0-9.])(0\.0066|0\.011|0\.0487|0\.0091)(?![0-9])")
_FAMILY_MARKERS = ("04_multilayer_fresnel", "_04_fresnel_results", "cv22", "cv23",
                   "_22_dispersive", "_23_lossy", "slab_rig", "slab_family",
                   "dispersive_slab", "lossy_slab", "multilayer_fresnel")
_TREES = ("validation", "tests", "docs/public", "docs/design_notes")
_SUFFIXES = {".py", ".json", ".md", ".mdx"}


def _family_files_with_literals() -> list[str]:
    found: list[str] = []
    for tree in _TREES:
        for path in sorted((_REPO / tree).rglob("*")):
            if path.suffix not in _SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not _LITERAL_RE.search(text):
                continue
            rel = str(path.relative_to(_REPO))
            if RESULTS_DIR_RE.match(rel) and rel != _ENVELOPE_REL:
                continue      # a produced artifact: output by construction
            if any(marker in text or marker in rel for marker in _FAMILY_MARKERS):
                found.append(rel)
    return found


def _results_artifacts_with_literals() -> list[str]:
    out = []
    for path in sorted((_REPO / "validation/crossval").rglob("*.json")):
        rel = str(path.relative_to(_REPO))
        if rel == _ENVELOPE_REL:
            continue
        if RESULTS_DIR_RE.match(rel) and _LITERAL_RE.search(
                path.read_text(encoding="utf-8", errors="replace")):
            out.append(rel)
    return out


def test_results_artifacts_carrying_an_adopted_value_are_produced_outputs():
    """The results-directory rule, checked rather than assumed.

    These files carry the adopted values because a run RECORDED the window it
    was judged against -- which is what makes those windows citable by value in
    the public docs. Each must look like a produced artifact (its own schema
    tag), not a hand-written table.
    """
    found = _results_artifacts_with_literals()
    assert found, "no produced artifact records a window any more"
    for rel in found:
        doc = json.loads((_REPO / rel).read_text(encoding="utf-8"))
        assert doc.get("schema"), f"{rel} carries an adopted value with no schema tag"
        assert rel != _ENVELOPE_REL


def test_every_family_appearance_of_an_adopted_value_is_classified():
    """Enumerate-and-classify (methodology 2.3) over the envelope's fan-out.

    Scope, stated so it is not mistaken for more: files that carry one of the
    adopted literals AND mention the slab family. A file outside the family
    carrying 0.011 is a different quantity by construction -- that is what
    "outside the family" means -- and the family markers are re-checked here,
    so a new consumer cannot join the family silently.
    """
    found = set(_family_files_with_literals())
    listed = set(FANOUT)
    assert found - listed == set(), (
        f"unclassified appearances of an adopted envelope value inside the slab "
        f"family: {sorted(found - listed)}. Each is one of: {INPUT_SOURCE} (the "
        f"producer artifact), {RECORDED_OUTPUT} (a produced artifact recording "
        f"its own run), {RESOLVING_REFERENCE} (a `path::key` citation), "
        f"{ADOPTION_DECLARATION}, {DISPLAY_WITH_SOURCE}, {BASELINE_IDENTITY}, or "
        f"{DIFFERENT_QUANTITY} with a written reason -- or it is a VIOLATION: a "
        f"second input source.")
    assert listed - found == set(), (
        f"classified files that no longer carry the value: {sorted(listed - found)}")


@pytest.mark.parametrize("rel", sorted(FANOUT))
def test_each_fanout_classification_holds(rel: str):
    kind = FANOUT[rel]
    path = _REPO / rel
    assert path.is_file(), rel
    text = path.read_text(encoding="utf-8", errors="replace")
    if kind == INPUT_SOURCE:
        assert rel == _ENVELOPE_REL
        assert json.loads(text)["producer"]
    elif kind == RECORDED_OUTPUT:
        # a produced artifact recording its OWN run's numbers, not a copy of
        # this envelope: it carries its own provenance block.
        assert json.loads(text).get("provenance"), rel
    elif kind == RESOLVING_REFERENCE:
        # every literal sits inside a `path::key = value` span, which
        # tests/contracts/test_evidence_numeric_provenance.py resolves.
        spans = re.findall(r"`[^`\n]*::[^`\n]*`", text)
        assert spans, f"{rel} carries no artifact reference span"
    elif kind == ADOPTION_DECLARATION:
        # a consumer module carries the adoption record and NOT the values; a
        # note is an adoption declaration only if a revision names it.
        if rel.endswith(".py"):
            assert "CV04_ADOPTION" in text
            assert not _LITERAL_RE.search(
                text.split("CV04_ADOPTION")[1].split("FALSIFIERS")[0]), (
                f"{rel} restates an envelope value next to its adoption record")
        else:
            declared = {entry["adopted_in"]
                        for block in _envelope_doc()["revisions"].values()
                        for entry in block["adopted_by"]}
            assert rel in declared, (
                f"{rel} is classified {ADOPTION_DECLARATION!r} but no revision "
                f"names it as the place an adoption was declared")
    elif kind == PRODUCER_NOTE:
        assert rel == _envelope_doc()["producer_script"], (
            f"only the producer's own script may state its measurement in a "
            f"comment; {rel} is not it")
    elif kind == DISPLAY_WITH_SOURCE:
        expected = DISPLAY_SOURCE[rel]
        assert expected in text, (
            f"{rel} shows an adopted value with no reference back to "
            f"{expected}")
    elif kind == HISTORICAL_PROSE:
        # a dated record of a past decision. Admissible only where nothing
        # derives from it: no revision names it as an adoption site.
        assert rel.startswith("docs/design_notes/"), rel
        declared = {entry["adopted_in"]
                    for block in _envelope_doc()["revisions"].values()
                    for entry in block["adopted_by"]}
        assert rel not in declared, (
            f"{rel} IS an adoption site; classify it {ADOPTION_DECLARATION!r}")
    elif kind == BASELINE_IDENTITY:
        assert json.loads(text)["schema"] == "rfx.slab_family_windows_baseline/v1"
    elif kind == DIFFERENT_QUANTITY:
        assert DIFFERENT_QUANTITY_REASON.get(rel, "").strip(), (
            f"{rel} is classified {DIFFERENT_QUANTITY!r} with no reason")
    else:  # pragma: no cover
        raise AssertionError(f"{rel}: unknown class {kind!r}")


# ---------------------------------------------------------------------------
# 4. Display copies carry a reference that resolves
# ---------------------------------------------------------------------------

def _resolve(reference: str):
    rel, _, keypath = reference.partition("::")
    node = json.loads((_REPO / rel).read_text(encoding="utf-8"))
    for step in keypath.split("."):
        assert isinstance(node, dict) and step in node, (
            f"{reference} does not resolve: {step!r} missing")
        node = node[step]
    return node


def test_the_ui_fixture_references_resolve_and_equality_is_reported():
    """The STUDIO fixture keeps its display numbers and gains a reference.

    Equality is REPORTED, not asserted: this fixture drives a UI journey, and
    a fixture allowed to gate a physics number is what #928 came from. If it
    ever disagrees with the artifact, the print says so and the artifact is
    right.
    """
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    doc = _envelope_doc()
    checked = 0
    for metric in golden["expected_metrics"]:
        reference = metric.get("observed_baseline_source")
        if reference is None:
            continue
        checked += 1
        assert metric.get("observed_baseline_source_revision") in doc["revisions"], (
            f"{metric['id']} names a source revision the artifact does not have")
        value = _resolve(reference)
        shown = metric["observed_baseline"]
        status = "same" if value == shown else f"DIFFERS (artifact {value})"
        print(f"ui-fixture {metric['id']}: shows {shown}, {reference} -> {status}")
    assert checked == 3, (
        f"{checked} of the fixture's metrics carry a source reference; the "
        f"three cv04 baselines must each name where they came from")

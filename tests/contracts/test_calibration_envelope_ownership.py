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
import ast
import json
import math
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


# Value matching cannot tell an adopted number from an unrelated constant that
# happens to equal it, so the coincidences are declared here -- with the reason
# and checked below, rather than silently tolerated by a narrower regex.
# Keyed by (file, SYMBOL), not (file, value): the value form exempted every
# occurrence of that number in the file, so a planted `W_MEAN_R = 0.010` in
# cv22 was covered by the exemption written for Meep's `a = 0.01` (round-2
# review). One symbol, one reason.
COINCIDENTAL_VALUES: dict[tuple[str, str], str] = {
    ("validation/crossval/comparators/cv22_dispersive_gates.py", "MEEP_A_M"):
        "Meep's length unit a = 1 cm, in metres. Equal to W_MEAN_R (0.010) by "
        "arithmetic coincidence; it is the reference leg's geometry scale and "
        "enters no window.",
}


def test_every_declared_coincidence_still_exists():
    """A declared coincidence that has gone away is a stale exemption."""
    targets = {**_adopted_values(), **_derived_windows()}
    for (module, symbol), reason in COINCIDENTAL_VALUES.items():
        assert reason.strip(), (module, symbol)
        named = _python_named_numbers(_REPO / module)
        hits = [v for name, v in named if name == symbol]
        assert hits, f"{module} no longer defines {symbol}; drop the exemption."
        assert any(_matches(v, targets) for v in hits), (
            f"{module}::{symbol} no longer collides with an adopted value or a "
            f"derived window; the exemption is stale.")


def test_consumer_modules_carry_no_envelope_literal():
    """A consumer holds an adoption record, never a copy of the values.

    Before #928 both consumers held the four numbers inline. The measurable
    consequence of the split is that they now hold none: every window they
    enforce comes from the producer's artifact through the record.
    """
    targets = {**_adopted_values(), **_derived_windows()}
    for consumer in _adoption_records():
        offenders = []
        for name, value in _python_named_numbers(_REPO / consumer):
            hit = _matches(value, targets)
            if hit is None:
                continue
            if COINCIDENTAL_VALUES.get((consumer, name)):
                continue
            offenders.append(f"{name} = {value!r} == {hit}")
        assert not offenders, (
            f"{consumer} restates {offenders}. A consumer derives from the "
            f"artifact; it keeps neither the adopted values nor the windows "
            f"derived from them, in any spelling.")


def test_the_consumer_literal_guard_catches_a_planted_value():
    """(B) arm: the two plants a review used to walk past the first version --
    a differently-spelled adopted value and a hard-coded derived window."""
    targets = {**_adopted_values(), **_derived_windows()}
    assert _matches(9.1e-3, targets) == "mean_closure"      # cv23's plant
    assert _matches(0.0110, targets) == "mean_dT"           # trailing-zero spelling
    assert _matches(1.1e-2, targets) == "mean_dT"           # exponent spelling
    assert _matches(0.074, targets) == "W_BIN"              # a derived window
    assert _matches(0.148, targets) == "W_BIN_A"            # cv23's doubled one
    assert _matches(0.027000000000000003, targets) == "W_MEAN_A"
    assert _matches(0.5, targets) is None                   # an unrelated number

    # ... and the expression forms a review walked past the first version with.
    import ast as _ast
    for source, expected in (("148 / 1000", "W_BIN_A"),
                             ("74 / 1000", "W_BIN"),
                             ("14 / 1000", "W_MEAN_A_TIGHT"),
                             ("2 * 0.037", "W_BIN"),
                             ("-0.011", None),
                             ("74e-3", "W_BIN")):
        folded = _fold(_ast.parse(source, mode="eval").body)
        assert folded is not None, source
        if expected is None:
            assert _matches(abs(folded), targets) == "mean_dT"
        else:
            assert _matches(folded, targets) == expected, source
    # a name-keyed exemption covers ONE symbol, not every occurrence of a value
    assert ("validation/crossval/comparators/cv22_dispersive_gates.py",
            0.01) not in COINCIDENTAL_VALUES


def test_the_artifact_does_not_record_its_citers():
    """Evidence does not know who cites it.

    `adopted_by` used to live INSIDE the hashed revision block, so a third
    consumer adopting an already-merged revision would have changed that
    revision's hash — tripping the immutability rule and invalidating the pin
    every existing consumer holds. Adoption is declared in the consumer; this
    file's own enumeration is what discovers the adopters.
    """
    doc = _envelope_doc()
    for name, block in doc["revisions"].items():
        assert "adopted_by" not in block, (
            f"revision {name} lists its citers inside the hashed block")
        assert "adopted_in" not in block, name
        assert "rig_provenance_keys" not in block, (
            f"revision {name} carries the provenance legend inside the hashed "
            f"block; it is a key to the vocabulary, not part of the measurement")
    assert "adopted_by" not in doc and "adopted_in" not in doc
    assert doc["adoption"].strip()
    # the adopters are discoverable, and they exist
    records = _adoption_records()
    assert records, "no comparator declares an adoption record any more"
    for consumer, record in records.items():
        assert record["adopted_revision"] in doc["revisions"], consumer
        assert (_REPO / record["adopted_in"]).is_file(), consumer


def test_a_third_adopter_needs_no_change_to_the_artifact():
    """The property the round-2 review asked for, exercised.

    A scratch third consumer adopts r1 with the pins the artifact already
    publishes. The artifact is byte-identical to the committed one, both
    existing consumers keep their pins, and the same-commit guard is silent —
    because adding an adopter is a change to the ADOPTER, not to the evidence.
    """
    doc = _envelope_doc()
    r1 = doc["revisions"]["r1"]
    third = {
        "envelope": _ENVELOPE_REL,
        "adopted_revision": "r1",
        "revision_sha256": r1["revision_hash"],
        "rig_hash": r1["rig_hash"],
        "gate_policy": {"multiplier": 1.5, "quantum": 1000},
        "adopted_in": "docs/design_notes/20260906_issue928_ownership_decision.md",
        "adopted_by_reviewer": "scratch third adopter (test)",
    }
    loaded = _load_comparator("slab_family").load_adopted_envelope(
        third, path=str(_ENVELOPE))
    assert loaded["revision"] == "r1"
    assert loaded["values"] == r1["values"]
    # nothing about the artifact changed to accommodate it
    assert json.loads(_ENVELOPE.read_text(encoding="utf-8")) == doc
    # and the guard says nothing about a change that touches only the adopter
    records = dict(_adoption_records())
    records["validation/crossval/comparators/_scratch_third_consumer.py"] = third
    assert same_commit_violation({"validation/crossval/comparators/_scratch_third_consumer.py"},
                                 records, doc, doc) is None


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


def _envelope_paths() -> set[str]:
    """Every producer envelope artifact under the crossval tree.

    The guard used to watch one hard-coded path, so a second case emitting its
    own `envelope.json` would have been unguarded from the day it landed.
    """
    return {str(p.relative_to(_REPO))
            for p in (_REPO / "validation/crossval").glob("*/envelope.json")}


def _git(*args) -> tuple[int, str]:
    result = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True)
    return result.returncode, result.stdout


def _doc_at(base: str, rel: str) -> dict | None:
    """The artifact as of *base*, or None when it did not exist there."""
    code, out = _git("show", f"{base}:{rel}")
    if code != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def same_commit_violation(touched, records: dict, doc: dict,
                          base_doc: dict | None) -> str | None:
    """The predicate, separated from the branch it runs on so it can be
    falsified. Returns a message when a change breaks either half of the
    evidence/calibration split, else None.

    Two independent rules, both needed (a review defeated the first version by
    satisfying neither and being waived anyway):

    (a) **A written revision is immutable.** If a revision already exists at
        the diff base, its block may not change -- not its values, not its rig,
        not its hash. New evidence is a NEW revision. Without this, "edit r1
        and re-stamp both hashes" is a silent gate move that every downstream
        check accepts, because every downstream check is derived from r1.
    (b) **Evidence and calibration do not move in one change.** If the artifact
        and an adoption record that cites it both change, that is a producer
        widening the gates that judge it -- UNLESS the artifact did not exist at
        the base at all (the bootstrap case: the artifact and its first adoption
        cannot exist without each other). "The adopted revision is marked
        bootstrap" is NOT the exemption; r1 stays marked bootstrap forever, and
        the first version of this guard therefore waived every later change too.
    """
    if not (set(touched) & _envelope_paths()):
        return None
    problems: list[str] = []

    if base_doc is not None:
        base_revisions = base_doc.get("revisions", {})
        now_revisions = doc.get("revisions", {})
        # Deleting a revision is not "no change": the first version of this rule
        # iterated the NEW document, so base {r0, r1} -> new {r1} said nothing
        # (round-2 review). Iterate the BASE and require each one to still be
        # there, byte-identical.
        for name, was in base_revisions.items():
            block = now_revisions.get(name)
            if block is None:
                problems.append(
                    f"revision {name} existed at the base and is GONE. A "
                    f"revision is append-only evidence; withdraw it by setting "
                    f"its status, never by deleting it.")
                continue
            if json.dumps(was, sort_keys=True) != json.dumps(block, sort_keys=True):
                changed = sorted(k for k in set(was) | set(block)
                                 if was.get(k) != block.get(k))
                problems.append(
                    f"revision {name} already existed at the base and its "
                    f"{changed} changed in place. A revision is append-only "
                    f"evidence: append a new one and adopt it deliberately.")

    also_touched = sorted(set(records) & set(touched))
    if also_touched:
        artifact_is_new = base_doc is None
        if not artifact_is_new:
            problems.append(
                f"this change touches {_ENVELOPE_REL} AND the adoption "
                f"record(s) in {also_touched}. Split it: append the revision in "
                f"one change (evidence), adopt it in another (calibration, "
                f"reviewed). The bootstrap exemption applies only when the "
                f"artifact does not exist at the base.")
    return " ".join(problems) or None


def test_the_producer_artifact_and_an_adoption_record_do_not_move_together():
    """The live branch, judged by the predicate above.

    The diff is taken against the base INCLUDING the working tree (``git diff
    <base>``, not ``<base>..HEAD``), so an uncommitted edit is seen by a local
    run rather than only after it is committed.
    """
    if not git_available(_REPO):
        pytest.skip("git unavailable; the branch diff cannot be read")
    base = _diff_base()
    if base is None:
        pytest.skip("no diff base (shallow checkout?)")
    code, out = _git("diff", "--name-only", base)
    if code != 0:
        pytest.skip("git diff unavailable")
    touched = set(out.split())
    violation = same_commit_violation(touched, _adoption_records(), _envelope_doc(),
                                      _doc_at(base, _ENVELOPE_REL))
    assert violation is None, violation


def test_the_guard_state_matches_this_branch():
    """What the branch IS, stated as a fact rather than assumed by the guard.

    On the branch that introduces the artifact, the artifact does not exist at
    the base and the exemption is live; on any later branch it does exist and
    the exemption is gone. This test reads which of the two it is instead of
    asserting one of them unconditionally -- the first version asserted the
    exemption always applied, and would have failed a legitimate r2 adoption.
    """
    if not git_available(_REPO):
        pytest.skip("git unavailable")
    base = _diff_base()
    if base is None:
        pytest.skip("no diff base")
    base_doc = _doc_at(base, _ENVELOPE_REL)
    doc = _envelope_doc()
    if base_doc is None:
        print(f"guard state: {_ENVELOPE_REL} is NEW in this branch -- bootstrap "
              f"exemption live; revisions {sorted(doc['revisions'])}")
        assert [n for n, b in doc["revisions"].items() if b.get("bootstrap")]
    else:
        print(f"guard state: {_ENVELOPE_REL} exists at {base[:8]} -- exemption "
              f"expired; every revision at base must be untouched")
        for name, block in doc["revisions"].items():
            was = base_doc["revisions"].get(name)
            if was is not None:
                assert json.dumps(was, sort_keys=True) == json.dumps(block, sort_keys=True), name


def test_the_guard_fires_on_the_reviewers_rewrite_probe():
    """(B) arm: the exact mutation two reviewers used to defeat the first guard.

    r1's values are rewritten in place, its hash re-stamped so the artifact is
    self-consistent, and both consumers re-pinned -- in one change. The windows
    moved and nothing complained. Both rules must catch it now, and each half
    of the change alone must still be allowed.
    """
    doc = _envelope_doc()
    records = _adoption_records()
    consumer = sorted(records)[0]
    base_doc = json.loads(json.dumps(doc))          # the artifact as it stands

    rewritten = json.loads(json.dumps(doc))
    block = rewritten["revisions"]["r1"]
    block["values"] = {k: v / 10.0 for k, v in block["values"].items()}
    block["revision_hash"] = "sha256:" + "0" * 64   # re-stamped, self-consistent
    both = {_ENVELOPE_REL, consumer}

    message = same_commit_violation(both, records, rewritten, base_doc)
    assert message is not None
    assert "append-only" in message and "r1" in message
    # rule (a) alone: the artifact edited, no consumer touched
    only_artifact = same_commit_violation({_ENVELOPE_REL}, records, rewritten, base_doc)
    assert only_artifact is not None and "append-only" in only_artifact
    # rule (b) alone: an APPENDED revision plus a re-adoption in one change
    appended = json.loads(json.dumps(doc))
    appended["revisions"]["r2"] = json.loads(json.dumps(doc["revisions"]["r1"]))
    appended["revisions"]["r2"]["bootstrap"] = False
    msg_b = same_commit_violation(both, records, appended, base_doc)
    assert msg_b is not None and "Split it" in msg_b
    # and the legitimate shapes stay legal
    assert same_commit_violation({_ENVELOPE_REL}, records, appended, base_doc) is None
    assert same_commit_violation({consumer}, records, appended, base_doc) is None
    # the bootstrap case: the artifact is new at the base
    assert same_commit_violation(both, records, doc, None) is None

    # A DELETED revision is not "no change" (round-2 item 6): the first version
    # of rule (a) iterated the new document, so base {r0, r1} -> new {r1}
    # returned None and a withdrawn-by-deletion revision passed.
    base_with_r0 = json.loads(json.dumps(doc))
    base_with_r0["revisions"]["r0"] = json.loads(json.dumps(doc["revisions"]["r1"]))
    deleted = same_commit_violation({_ENVELOPE_REL}, records, doc, base_with_r0)
    assert deleted is not None and "r0" in deleted and "GONE" in deleted


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
    "rfx/api/_preflight.py": DIFFERENT_QUANTITY,
    "docs/public/gallery/assets/multilayer_fresnel/manifest.json": RECORDED_OUTPUT,
    "docs/public/gallery/multilayer_fresnel.mdx": DISPLAY_WITH_SOURCE,
    "docs/public/guide/benchmarks.mdx": RESOLVING_REFERENCE,
    "docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md": ADOPTION_DECLARATION,
    "docs/design_notes/20260902_cv23_lossy_slab_predeclaration.md": ADOPTION_DECLARATION,
    "docs/design_notes/20260903_lattice_witness_standard.md": RESOLVING_REFERENCE,
    "docs/design_notes/20260903_test_reorg_tier3b_consolidation.md": DIFFERENT_QUANTITY,
    "docs/design_notes/20260904_aux_echo_record_invariant.md": RESOLVING_REFERENCE,
    "docs/design_notes/20260905_post_merge_review_20_prs.md": HISTORICAL_PROSE,
    "docs/design_notes/20260906_issue928_ownership_decision.md": HISTORICAL_PROSE,
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
    "rfx/api/_preflight.py": "a docstring worked example of the ceil(domain/dx) rounding rule, whose grid coordinate reads 0.011 m; the file enters this scan at all only because a preflight helper is named _waveguide_with_dispersive_slab",
    "docs/design_notes/20260903_test_reorg_tier3b_consolidation.md": "a pytest node id containing a parametrized 0.011",
}

# Detection is by VALUE, never by spelling. The first version of this file
# matched four decimal strings, and a review planted `CV04_MEAN_CLOSURE = 9.1e-3`
# in a consumer -- the same number, a different spelling -- past all thirty
# tests. Python files are walked as an AST (every numeric constant, including
# the ones inside expressions); other files are scanned for numeric tokens and
# compared as floats, so `0.0110`, `9.1e-3` and `0.011` are one thing.
_REL_TOL = 1e-12
_NUMBER_TOKEN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_FAMILY_MARKERS = ("04_multilayer_fresnel", "_04_fresnel_results", "cv22", "cv23",
                   "_22_dispersive", "_23_lossy", "slab_rig", "slab_family",
                   "dispersive_slab", "lossy_slab", "multilayer_fresnel")
# Wider than the four trees the first version looked at: a second copy of a
# producer's evidence is just as wrong in scripts/ or in the package itself.
_TREES = ("validation", "tests", "docs/public", "docs/design_notes",
          "docs/agent", "scripts", "rfx", "examples")
_SUFFIXES = {".py", ".json", ".md", ".mdx"}


def _adopted_values() -> dict[str, float]:
    return dict(_envelope_doc()["revisions"]["r1"]["values"])


def _derived_windows() -> dict[str, float]:
    """The windows a consumer must DERIVE, never restate: round-up(v x 1.5) at
    quantum 1000, their triangle sums, and the doubled per-bin window."""
    values = _adopted_values()
    policy = sorted(_adoption_records().values(),
                    key=lambda r: r["adopted_revision"])[0]["gate_policy"]
    mult, quantum = policy["multiplier"], policy["quantum"]

    def gate(v):
        return math.ceil(v * mult * quantum) / quantum

    w_bin = gate(values["per_bin_max_RT_closure"])
    w_mean_r = gate(values["mean_dR"])
    w_mean_t = gate(values["mean_dT"])
    return {
        "W_BIN": w_bin, "W_MEAN_R": w_mean_r, "W_MEAN_T": w_mean_t,
        "W_BIN_A": 2.0 * w_bin, "W_MEAN_A": w_mean_r + w_mean_t,
        "W_MEAN_A_TIGHT": gate(values["mean_closure"]),
    }


def _matches(value: float, targets: dict[str, float]) -> str | None:
    for name, target in targets.items():
        if target == 0:
            continue
        if abs(value - target) <= _REL_TOL * abs(target):
            return name
    return None


_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.Pow: lambda a, b: a ** b}


def _fold(node) -> float | None:
    """The value of a constant expression, or None if it is not one.

    A review appended `W_BIN_A = 148 / 1000` -- a GATED window, in expression
    form -- to a consumer and the whole suite stayed green, because the scan
    looked at `ast.Constant` nodes and never at the tree above them. Anything
    built only from numeric literals folds here: `148 / 1000`, `2 * 0.037`,
    `-0.011`, `74e-3`.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _fold(node.operand)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        handler = _BINOPS.get(type(node.op))
        if handler is None:
            return None
        left, right = _fold(node.left), _fold(node.right)
        if left is None or right is None:
            return None
        try:
            return float(handler(left, right))
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
    return None


def _python_number_literals(path: Path) -> list[float]:
    """Every numeric value a Python file states, folded."""
    out: list[float] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        value = _fold(node)
        if value is not None:
            out.append(value)
    return out


def _python_named_numbers(path: Path) -> list[tuple[str, float]]:
    """(key, value) for every numeric a Python file states.

    The key is the assignment target when there is one -- so an exemption can
    name `MEEP_A_M` rather than "every 0.01 in this file", which is what the
    (file, value) form meant and which let `W_MEAN_R = 0.010` through.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    named: list[tuple[str, float]] = []
    assigned: set[int] = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value_node = [node.target.id], node.value
        else:
            continue
        if value_node is None:
            continue
        value = _fold(value_node)
        if value is None:
            continue
        assigned.add(id(value_node))
        for name in targets or ["<unnamed>"]:
            named.append((name, value))
    for node in ast.walk(tree):
        if id(node) in assigned:
            continue
        value = _fold(node)
        if value is None:
            continue
        lineno = getattr(node, "lineno", 0)
        named.append((f"line:{lineno}", value))
    return named


def _number_tokens_in(text: str) -> list[float]:
    out: list[float] = []
    for token in _NUMBER_TOKEN.findall(text):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _carries_adopted_value(text: str) -> bool:
    """Any adopted value, in any spelling, anywhere in the file -- code,
    comment or prose. The fan-out question is "is a copy of the producer's
    number here", and a comment is a copy too (the deleted grep guard was
    aimed at exactly one)."""
    targets = _adopted_values()
    return any(_matches(v, targets) for v in _number_tokens_in(text))


def _family_files_with_literals() -> list[str]:
    found: list[str] = []
    for tree in _TREES:
        for path in sorted((_REPO / tree).rglob("*")):
            if path.suffix not in _SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = str(path.relative_to(_REPO))
            if not any(marker in text or marker in rel for marker in _FAMILY_MARKERS):
                continue
            if not _carries_adopted_value(text):
                continue
            if RESULTS_DIR_RE.match(rel) and rel != _ENVELOPE_REL:
                continue      # a produced artifact: output by construction
            found.append(rel)
    return found


def _results_artifacts_with_literals() -> list[str]:
    out = []
    for path in sorted((_REPO / "validation/crossval").rglob("*.json")):
        rel = str(path.relative_to(_REPO))
        if rel == _ENVELOPE_REL:
            continue
        if RESULTS_DIR_RE.match(rel) and _carries_adopted_value(
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
            # the module-wide, value-based version of this lives in
            # test_consumer_modules_carry_no_envelope_literal
            assert not _carries_adopted_value(
                text.split("CV04_ADOPTION")[1].split("FALSIFIERS")[0]), (
                f"{rel} restates an envelope value next to its adoption record")
        else:
            # the adoption SITES, read from the consumers (the artifact does
            # not record its citers any more -- round-2 item 2)
            declared = {r["adopted_in"] for r in _adoption_records().values()}
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
        declared = {r["adopted_in"] for r in _adoption_records().values()}
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

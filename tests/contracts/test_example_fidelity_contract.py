"""No-solve contract test for the #737 example-credibility gate (P4).

Builds every committed example/validation script that CAN be built without
solving, WITHOUT solving, and compares its ``preflight()`` +
``fidelity_report()`` output against a committed snapshot
(``tests/data/example_fidelity_snapshot.json``, regenerable with
``scripts/capture_example_fidelity_snapshot.py``). It is cheap precisely
because neither call time-steps: measured 2026-09-16 on this repo's CPU
lane at c6788ef7 + the #737-item-2 builders, ``205 passed, 20 warnings in
103.76s`` for all 62 script/builder/variant triples with optax installed
(the same lane measured ``194 passed, 13 warnings in 91.71s`` at c6788ef7
before those builders, i.e. 51 triples). On a machine without optax
beam_steering_superstrate.py declares it in OPTIONAL_DEPENDENCIES and
skips rather than failing -- that is CI's configuration.

EMISSION-DRIFT PIN PLUS A ZERO-*UNEXPLAINED*-ADVISORY BAR -- STILL NOT A
PHYSICS CHECK. Nothing here time-steps: what is pinned is the TEXT each example
EMITS at build time (its preflight rows and its fidelity/realization
report), so a green run says "this example's declared geometry and its
build-time advisories did not change since the snapshot was taken." It
says nothing about whether the example's OUTPUT numbers are right, and
nothing about whether today's advisories are correct or the examples are
clean. Never quote a green run here as example correctness.

WHAT DOES VERIFY EXAMPLE OUTPUTS (the other half of #737, and the reason
this gate's coverage table is not the coverage table):

* ``tests/contracts/test_tutorial_examples.py`` RUNS six committed
  tutorials end-to-end in a subprocess with time stepping and asserts
  physics (peak directivity within 0.3 dB of 1.76 dBi, a TE101 error
  bound, a backscatter/GO ratio window, preflight banner counts):
  materials_and_dispersion, antenna_farfield_pattern,
  ports_and_sparams_101, run_control_and_fields, rcs_scattering,
  resonance_harminv. All six are ALSO audited here, so a change in one of
  them shows up in two places: a snapshot diff (what it declares) and an
  end-to-end failure (what it computes). That file carries no pytest
  marker either, so it is in the same PR fast suite as this gate.
* ``tests/unit/api/test_diagnostics.py`` runs
  ``examples/quickstart/hello_world.py`` end-to-end via
  ``runpy.run_path``. Since #737 item 2 that script is ALSO audited here
  (``build_simulation``), so it is covered twice: what it declares and
  what it computes.
* the weekly ``crossval-external`` job (validation.yml) builds AND solves
  the eight scheduled crossval cases (01, 02, 03, 04, 09, 10, 22, 23).

EVERY PINNED ADVISORY ROW CARRIES A WRITTEN DISPOSITION (#737 item 1,
2026-09-16). ``tests/data/example_fidelity_advisories.json`` classifies
each (variant, code) group of pinned preflight rows as ``intended`` -- the
example deliberately shows or tolerates the condition, with its own
docstring/comment or the physics as the reason -- or ``defect-open``, with
the issue number tracking the fix; ``test_every_pinned_advisory_row_is_
classified`` asserts the mapping is a bijection with the row counts, so a
new advisory, a retired one, and a stale entry all fail here. That is the
"tighten once #742 closes" step the tracker's first comment planned: the
bar is zero UNEXPLAINED advisories, not zero advisories. It does NOT
assert that an advisory is correct, and it does not block a new one --
it blocks an unexplained one.

Measured on the snapshot as committed (2026-09-16, 62 variants): 91
preflight rows over 27 variants, 90 ``severity="warning"`` + 1 ``info``;
codes mesh_resolution 31, port_aperture_snap 12, pec_faces_finite_pec 10,
off_lattice_design_edges 8, lossless_q 5,
wire_port_dead_cell_classification_unavailable 5, msl_port_geometry 4,
port_evanescent 4, no_sources 2, ntff_small_ground_plane 2, and 8
singletons. The eleven variants #737 item 2 added carry 8 of those rows
(mesh_resolution 6 on 13_subgrid_material_validation's dielectric arms,
lossless_q 1 + off_lattice_design_edges 1 on nonuniform_patch_demo, which
are the two the tutorial's own text already explains); the other eight new
variants emit none. #742 (the false positives and
the never-emitted advisories that made a zero-advisory bar unworkable) is
CLOSED. The rows that remain are statements about the examples' own meshes
and ports, so getting to zero means changing those examples -- separate
work from this gate, which pins whatever they say. Whether that work
happens at all (tighten to a zero-UNEXPLAINED-advisory bar by classifying
all 83 rows) or the gate stays a drift pin is an open PI decision on #737;
until it is taken, this file is a drift pin and the tracker's first
comment's "tighten once #742 closes" plan has NOT been executed.

51 classification entries cover all 91 rows: 49 ``intended`` and 2
``defect-open`` (#928 for cv07's congruent-feed parity, #1100 for the taper's
declared-vs-realized WR-90 guide). The four rows the eleven #737-item-2 variants
added are classified against the examples' own text: the patch demo states that
it models FR4 lossless and quotes frequencies rather than Q, and that a graded
mesh trades accuracy for cells; the subgrid script's coarse dielectric arm exists
to be rejected, and its uniform twin reuses that mesh so the two are comparable
and is checked against the analytic TM110 resonance.

The #729 site-1 defect this header used to warn about (every ``domain``
row's ``realized_extent_um``/``n_cells`` one cell too large, because
``rfx/fidelity.py`` summed a NODE-count slice) was FIXED by PR #734
(merged a5a72280) and the snapshot was re-captured after it: cv11's WR-90
guide now reads 23000/11000 um, which is what #722 measures, not the old
24000/12000. The snapshot's own ``_comment`` records that re-capture and a
second one (2026-09-02, #833 item 2).

COVERAGE (measured 2026-09-16 at c6788ef7 + this PR; re-derive with
``test_discovery_matches_classification_table`` and a Counter over
``lib.CLASSIFICATION``). Discovery sees 138 scripts under examples/ +
validation/, all 138 classified: audited 39, builder_fused_with_solve 8,
module_level_solve 6, no_solve 4, no_simulation 81; the snapshot holds 62
variants over those 39 audited scripts.

Against the 47 scripts that existed when #737 was filed (ed3484c1 -- the
tracker's own denominator, kept here because the tracker's table is stated
in it): 30 audited / 4 builder_fused_with_solve / 5 module_level_solve /
8 no_simulation. The moves since the 2026-08-27 audit's 23/10/6/8: cv07
and cv15 grew build-only entry points (23->25, 10->8), and #737 item 2
(2026-09-16) added five more -- hello_world, boundary_spec_demo,
nu_cavity_gate_scan and 13_subgrid_material_validation out of
``builder_fused_with_solve``, and nonuniform_patch_demo out of
``module_level_solve`` (its module-scope body moved into ``main()``, so
importing it no longer solves).

* ``builder_fused_with_solve`` (build and solve share one function with
  no separable build-only path) in that 47-set, i.e. the scripts this
  gate does NOT reach: examples/tutorials/cad_mesh_import_demo.py, cv09,
  cv10, cv18. "Out of this snapshot" is not "unverified": cv09 is REBUILT
  build-only from its own constants and helpers in
  tests/crossval/test_cv09_cv10_body_contract_controls.py (cv10 only at
  the spec level there, via ``_common_spec()``) and both solve weekly;
  cv18's gates replay a frozen fixture
  (tests/crossval/test_wr90_iris_modematch_gates.py) and its realized
  geometry is checked build-time in test_wr90_iris_realized_is_shared.py.
  cad_mesh_import_demo is the one script of #737 item 2 that did NOT get a
  builder, and it is the one with no build or run coverage anywhere: its
  Simulation needs ``trimesh`` (the optional [cad] extra), which the lane
  running this gate does not install, and a build-time ModuleNotFoundError
  would red the gate rather than skip (OPTIONAL_DEPENDENCIES only covers
  an import-time miss). Its CLASSIFICATION entry carries that reason.
* ``module_level_solve`` (importing the module solves at module scope):
  cv01-cv05 -- also out of scope, and deliberately never imported by this
  test. cv01/cv02 have ``--replay`` re-judge subprocess tests and cv05 a
  ``RFX_CV05_BUILD_ONLY=1`` subprocess build test.
* 8 build no rfx ``Simulation`` at all (cv16/cv17/cv20/cv21, the two
  crossval comparator subdirs, and the two research RCWA scripts) -- out
  of scope BY CONSTRUCTION.

WHAT THIS DOES NOT COVER. Against #722's own list of eight scripts that
solved geometry other than what they declared (cv06b, cv20, cv11, cv16,
cv17, cv07, cv09, cv15), this gate now reaches FOUR: cv06b, cv11, cv07
and cv15 (the last two arrived with their build-only entry points).
cv20/cv16/cv17 are ``no_simulation`` and cv09 is
``builder_fused_with_solve``, so a no-solve gate cannot see them as those
scripts stand today. Nor does it reach cv21's fence-post error (#739, cv21
is ``no_simulation``; that one is pinned against a rebuilt grid in
tests/crossval/test_coax_two_port_referee_header.py instead). It DOES pin
differentiable_s11_design's two domain widths (#738); that script's third
declared width, the port aperture, falls under fidelity_report's own
out-of-scope port row. A green run here is not evidence that the #722
campaign's class is closed.

Every discovered script is EXPLICITLY classified in
``tests/_example_fidelity_lib.CLASSIFICATION`` (enumerate-and-classify: a
new script with no entry fails ``test_discovery_matches_classification_
table`` instead of silently passing uncovered), and all four out-of-scope
buckets are verified against each script's own AST in
``test_not_auditable_classifications_are_machine_checked`` below: a
classification that disagrees with what the script's source actually does
is a bug in the table, not a judgement call.

Discovery is a recursive, unfiltered glob (see ``_example_fidelity_lib``'s
module docstring) -- no one-level ``*/*.py`` pattern to miss a doubly-nested
comparator, no ``_``-prefix filter to hide a private helper.

Fidelity digests are keyed by entity IDENTITY (material + declared bounds)
and axis LETTER, never by list position: inserting one new Box anywhere in
a script's geometry does not relabel every entity after it in a failure's
diff (2026-08-27 review, required change #2 -- verified in this file's own
``test_snapshot_keys_survive_entity_insertion``).
"""
from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from typing import Any

import jax
import re

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _example_fidelity_lib as lib  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _preserve_crossval_evidence():
    """Detect even native/child-process writes the Python guard cannot see.

    Compare with the user's starting bytes, not HEAD: existing local work
    must remain valid and must never be restored or overwritten by a test.
    """
    root = lib.REPO_ROOT / "validation" / "crossval"

    def hashes():
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }

    before = hashes()
    with lib.build_only():
        yield
    after = hashes()
    changed = sorted(p for p in before.keys() | after.keys()
                     if before.get(p) != after.get(p))
    assert not changed, f"build-only audit changed crossval evidence: {changed}"


def _load_snapshot() -> dict[str, dict]:
    import json
    if not lib.SNAPSHOT_PATH.exists():
        pytest.fail(
            f"{lib.SNAPSHOT_PATH} does not exist -- regenerate with "
            "`JAX_ENABLE_X64=0 python scripts/capture_example_fidelity_"
            "snapshot.py`")
    return json.loads(lib.SNAPSHOT_PATH.read_text())["variants"]


def test_discovery_matches_classification_table() -> None:
    """Every committed script under examples/ or validation/ must have an
    explicit classification entry -- a new file with none must fail here,
    not silently pass with zero coverage (enumerate-and-classify)."""
    discovered = set(lib.discover_scripts())
    classified = set(lib.CLASSIFICATION)
    unknown = sorted(discovered - classified)
    missing = sorted(classified - discovered)
    assert not unknown, (
        f"script(s) with no classification entry: {unknown} -- add an "
        "entry to tests/_example_fidelity_lib.py CLASSIFICATION")
    assert not missing, (
        f"classification table entry for a script no longer on disk: "
        f"{missing} -- remove it from CLASSIFICATION")


@pytest.mark.parametrize(
    "relpath,entry", sorted(lib.CLASSIFICATION.items()),
    ids=[p for p in sorted(lib.CLASSIFICATION)],
)
def test_not_auditable_classifications_are_machine_checked(
    relpath: str, entry: lib.Entry,
) -> None:
    """Every bucket is checked against the script's own AST, not just a
    trusted reason string -- required for the two not-auditable buckets by
    the 2026-08-27 review (change #4); extended here to all four so a
    script that later gains (or loses) a separable builder cannot stay
    misclassified silently."""
    if entry.kind == "no_simulation":
        assert lib.real_simulation_call_count(relpath) == 0, (
            f"{relpath} is classified no_simulation but its AST now "
            "constructs a Simulation() -- reclassify")
    elif entry.kind == "module_level_solve":
        assert not lib.has_main_guard(relpath), (
            f"{relpath} is classified module_level_solve but now has a "
            "main guard -- it may be auditable, reclassify")
        assert lib.has_top_level_solve_call(relpath), (
            f"{relpath} is classified module_level_solve but no top-level "
            "solve call was found in its AST -- reclassify")
    elif entry.kind == "no_solve":
        # Builds a Simulation (so it is not ``no_simulation``) but never
        # solves (so ``builder_fused_with_solve`` cannot describe it either):
        # the #636 analysis scripts construct a Simulation only to read the
        # lock-test grid's dt. Pinned both ways so the bucket cannot absorb a
        # script that later gains a solve, or one that stops building at all.
        assert lib.real_simulation_call_count(relpath) > 0, (
            f"{relpath} is classified no_solve but its AST constructs no "
            "Simulation -- reclassify as no_simulation")
        assert not lib.has_any_solve_call(relpath), (
            f"{relpath} is classified no_solve but now calls a solve "
            "entrypoint -- reclassify (builder_fused_with_solve or audited)")
    elif entry.kind == "builder_fused_with_solve":
        fns = lib.functions_building_simulation(relpath)
        assert any(fns.values()), (
            f"{relpath} is classified builder_fused_with_solve but no "
            f"top-level function both builds and solves (found: {fns}) -- "
            "it may now have a separable builder, reclassify as 'audited'")
    elif entry.kind == "audited":
        # This test IMPORTS these scripts (load_module execs their top
        # level), so the "nothing solves at import" precondition is asserted
        # here per script rather than asserted once in prose: a main guard
        # and no module-scope solve call.
        assert lib.has_main_guard(relpath), (
            f"{relpath} is classified audited but has no "
            "`if __name__ == \"__main__\":` guard -- this gate imports it, "
            "so its module scope must not be a script body")
        assert not lib.has_top_level_solve_call(
                relpath, skip_main_guard=True), (
            f"{relpath} is classified audited but calls a solve entrypoint "
            "at module scope OUTSIDE its main guard -- importing it would "
            "SOLVE; move that call inside main() before auditing it")
        fns = lib.functions_building_simulation(relpath)
        for builder in entry.builders:
            assert builder.fn in fns, (
                f"{relpath}: declared builder {builder.fn!r} not found as "
                f"a top-level function constructing a Simulation (found: "
                f"{fns})")
            assert not fns[builder.fn], (
                f"{relpath}: builder {builder.fn!r} now calls a solve "
                "entrypoint in its own body -- reclassify as "
                "'builder_fused_with_solve'")
    else:
        raise AssertionError(f"unhandled classification kind {entry.kind!r}")


def test_precision_is_pinned() -> None:
    """The snapshot is pinned at JAX_ENABLE_X64=0. conftest.py's autouse
    ``_no_x64_leak`` anchors to the SESSION's starting value, not to False,
    because "the supported way to reproduce #646 is to run the suite under
    JAX_ENABLE_X64=1" (conftest.py; echoed at tests/unit/grid/test_x64_scan_carry_
    dtypes.py:123,161,363). A hard ``assert not x64`` would therefore fail
    this whole file under that documented workflow -- skip instead, so the
    guard's actual purpose (never silently bless a wrong-precision
    snapshot as correct) holds without breaking a supported configuration.
    """
    if jax.config.jax_enable_x64:
        pytest.skip(
            "example_fidelity_snapshot.json is pinned at JAX_ENABLE_X64=0; "
            "this run has jax_enable_x64=True (the supported #646 "
            "reproduction mode) so the snapshot comparison does not apply "
            "-- rerun under the default precision to exercise this gate")


def _flat_variants() -> list[tuple[str, str, lib.Builder, lib.Variant]]:
    rows = []
    for relpath, entry in sorted(lib.CLASSIFICATION.items()):
        if entry.kind != "audited":
            continue
        for builder in entry.builders:
            for variant in builder.variants:
                key = f"{relpath}::{builder.fn}::{variant.label}"
                rows.append((key, relpath, builder, variant))
    return rows


_FLAT_VARIANTS = _flat_variants()


def _diff_message(key: str, expected: dict, actual: dict) -> str:
    """Which dimension of which entity moved -- not a raw dict repr (the
    thing required change #2 exists to make possible)."""
    lines = [f"snapshot drift for {key}:"]
    exp_f: dict[str, Any] = expected.get("fidelity", {})
    act_f: dict[str, Any] = actual.get("fidelity", {})
    added = sorted(set(act_f) - set(exp_f))
    removed = sorted(set(exp_f) - set(act_f))
    for e in added:
        lines.append(f"  + entity added: {e}")
    for e in removed:
        lines.append(f"  - entity removed: {e}")
    for e in sorted(set(exp_f) & set(act_f)):
        exp_item, act_item = exp_f[e], act_f[e]
        if exp_item == act_item:
            continue
        exp_axes = exp_item.get("axes", {})
        act_axes = act_item.get("axes", {})
        for ax in sorted(set(exp_axes) | set(act_axes)):
            if exp_axes.get(ax) != act_axes.get(ax):
                lines.append(f"  ~ {e} axis {ax}: "
                             f"expected {exp_axes.get(ax)} "
                             f"actual {act_axes.get(ax)}")
        if exp_item.get("n_cells") != act_item.get("n_cells"):
            lines.append(f"  ~ {e} n_cells: expected {exp_item.get('n_cells')} "
                         f"actual {act_item.get('n_cells')}")
        if exp_item.get("findings") != act_item.get("findings"):
            lines.append(f"  ~ {e} findings: expected {exp_item.get('findings')} "
                         f"actual {act_item.get('findings')}")
    exp_p, act_p = expected.get("preflight", []), actual.get("preflight", [])
    if exp_p != act_p:
        lines.append(f"  preflight: expected {len(exp_p)} issue(s), "
                     f"actual {len(act_p)} issue(s)")
        for row in act_p:
            if row not in exp_p:
                lines.append(f"    + {row['code']} {row['loc']}: {row['message']}")
        for row in exp_p:
            if row not in act_p:
                lines.append(f"    - {row['code']} {row['loc']}: {row['message']}")
    return "\n".join(lines)


@pytest.mark.parametrize(
    "key,relpath,builder,variant", _FLAT_VARIANTS,
    ids=[row[0] for row in _FLAT_VARIANTS],
)
def test_example_matches_snapshot(
    key: str, relpath: str, builder: lib.Builder, variant: lib.Variant,
) -> None:
    snapshot = _load_snapshot()
    assert key in snapshot, (
        f"{key} is missing from the snapshot -- regenerate with "
        "scripts/capture_example_fidelity_snapshot.py")
    try:
        module = lib.load_module(relpath)
    except lib.MissingOptionalDependency as exc:
        # Visible SKIP, never green: this repo's exit-code convention
        # (development_methodology.md 2.7) says a missing reference is a
        # SKIP that shows, not a pass. The variant stays enumerated in
        # _FLAT_VARIANTS and pinned in the snapshot, so coverage is
        # reported as skipped rather than silently lost, and an
        # UNDECLARED missing module is still a hard error (see
        # lib.OPTIONAL_DEPENDENCIES).
        pytest.skip(str(exc))
    fn = getattr(module, builder.fn)
    kwargs = variant.kwargs(module)
    result = fn(**kwargs)
    sim = result if builder.result_index is None else result[builder.result_index]
    # Round-trip through JSON so tuples (live digest) compare equal to lists
    # (snapshot, loaded from JSON) -- the digest itself is unchanged either
    # way, only its Python container types are normalized for comparison.
    import json
    actual = json.loads(json.dumps(lib.digest_variant(sim)))
    expected = snapshot[key]
    assert actual == expected, _diff_message(key, expected, actual)


def test_every_pinned_advisory_row_is_classified() -> None:
    """Zero UNEXPLAINED advisories: the bar #737's first comment planned.

    The snapshot pins WHAT each audited example emits. This asserts that every
    pinned preflight row also has a WRITTEN disposition in
    ``tests/data/example_fidelity_advisories.json`` -- intended (the example
    deliberately shows or tolerates it, with its own docstring/comment or the
    physics as the reason) or defect-open (it should not emit it, with the
    issue that tracks the fix). The mapping is a bijection with the row counts
    checked, so all three drift directions fail here rather than silently:

    * a NEW advisory row on an audited example -> no entry -> fails;
    * a row that DISAPPEARS (an example fixed, or an advisory retired)
      -> stale entry -> fails;
    * the same (variant, code) firing more or fewer times -> count mismatch.

    This does NOT judge whether an advisory is correct, and it does not stop a
    new advisory from landing -- it stops one from landing unexplained.
    """
    snapshot = _load_snapshot()
    pinned = lib.snapshot_advisory_counts(snapshot)
    entries = lib.load_advisory_classification()

    seen: dict[tuple[str, str], int] = {}
    for entry in entries:
        key = (entry["variant"], entry["code"])
        assert key not in seen, (
            f"two classification entries for {key} -- one entry per "
            "(variant, code), with `rows` counting the pinned rows it covers")
        seen[key] = entry["rows"]

    missing = sorted(k for k in pinned if k not in seen)
    assert not missing, (
        "pinned preflight row(s) with no written disposition: "
        f"{missing} -- classify them in tests/data/"
        "example_fidelity_advisories.json (#737 item 1); a new advisory on an "
        "audited example is either intended (say why, citing the example) or "
        "an example defect (fix it, or file an issue and use 'defect-open')")
    stale = sorted(k for k in seen if k not in pinned)
    assert not stale, (
        f"classification entr(ies) for row(s) that are no longer pinned: "
        f"{stale} -- the example or the advisory changed; drop the entry (and "
        "say so in the PR) instead of leaving a claim about a row nobody emits")
    wrong = {k: (seen[k], pinned[k]) for k in pinned if seen[k] != pinned[k]}
    assert not wrong, (
        f"row-count mismatch (classified, pinned): {wrong} -- the same code "
        "now fires a different number of times on that variant")
    assert sum(seen.values()) == sum(pinned.values()), (
        "row totals disagree after the per-key check -- this cannot happen "
        "and means the two tables were built from different snapshots")


def test_advisory_classifications_are_well_formed() -> None:
    """A disposition has to carry its evidence, and defect-open its issue.

    Enforced per entry: a disposition from ``lib.ADVISORY_DISPOSITIONS``, a
    non-trivial ``reason`` and ``ref`` (the classification's value is the
    written reason -- an entry that just says 'intended' explains nothing),
    and for ``defect-open`` an integer issue number, so a row cannot be parked
    as a known defect with nothing tracking it.
    """
    for entry in lib.load_advisory_classification():
        where = f"{entry.get('variant')} :: {entry.get('code')}"
        for field in ("variant", "code", "rows", "disposition", "reason", "ref"):
            assert field in entry, f"{where}: classification entry lacks {field!r}"
        assert entry["disposition"] in lib.ADVISORY_DISPOSITIONS, (
            f"{where}: unknown disposition {entry['disposition']!r} -- "
            f"expected one of {sorted(lib.ADVISORY_DISPOSITIONS)}")
        assert isinstance(entry["rows"], int) and entry["rows"] >= 1, (
            f"{where}: rows must be a positive int, got {entry['rows']!r}")
        assert len(entry["reason"].strip()) >= 80, (
            f"{where}: the reason is the point of this file -- write why this "
            "row is there, citing the example's own docstring/comment or the "
            "physics")
        assert entry["ref"].strip(), (
            f"{where}: ref must point at the lines (or the issue) the reason "
            "is read from")
        if entry["disposition"] == "defect-open":
            assert isinstance(entry.get("issue"), int), (
                f"{where}: 'defect-open' requires an integer issue number -- "
                "file the issue rather than parking the row")


def test_optional_dependency_declarations_are_grounded() -> None:
    """The optional-dependency exemptions cannot rot in either direction.

    Forward: every declared key is an audited script, so the table cannot
    grant an exemption to something this gate does not even cover.
    Backward: every declared module is actually imported by that script, so
    a dependency that is dropped from a script cannot leave a standing
    excuse behind that would swallow a future, genuine import failure of
    the same name.
    """
    audited = {
        rel for rel, entry in lib.CLASSIFICATION.items()
        if entry.kind == "audited"
    }
    for relpath, modules in sorted(lib.OPTIONAL_DEPENDENCIES.items()):
        assert relpath in audited, (
            f"OPTIONAL_DEPENDENCIES declares {relpath}, which is not an "
            "'audited' script -- the exemption covers nothing")
        src = (lib.REPO_ROOT / relpath).read_text()
        for module in sorted(modules):
            assert re.search(
                rf"^\s*(?:import\s+{re.escape(module)}\b"
                rf"|from\s+{re.escape(module)}\b)", src, re.M), (
                f"OPTIONAL_DEPENDENCIES lists {module!r} for {relpath}, but "
                f"that script no longer imports it -- drop the stale "
                f"exemption before it swallows an unrelated ImportError")


def test_snapshot_keys_survive_entity_insertion() -> None:
    """Digest keys are entity IDENTITY, not list position.

    Inserting one Box at the FRONT of a model's geometry must leave every
    other entity's key untouched; keying on ``fidelity_report``'s own
    ``geometry[i] 'name'`` label instead would relabel all of them and turn a
    one-entity change into a whole-model diff (2026-08-27 review, required
    change #2 -- this is the test that docstring cites).
    """
    from rfx import Simulation
    from rfx.geometry import Box

    def _sim(with_extra: bool):
        sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3),
                         dx=1e-3, boundary="cpml", cpml_layers=4)
        sim.add_material("sub", eps_r=4.0, sigma=0.0)
        if with_extra:
            sim.add(Box((1e-3, 1e-3, 1e-3), (3e-3, 3e-3, 3e-3)),
                    material="pec")
        sim.add(Box((4e-3, 4e-3, 4e-3), (8e-3, 8e-3, 6e-3)), material="sub")
        sim.add(Box((4e-3, 4e-3, 6e-3), (8e-3, 8e-3, 7e-3)), material="pec")
        return sim

    base = lib.digest_fidelity(_sim(False).fidelity_report(print_report=False))
    grown = lib.digest_fidelity(_sim(True).fidelity_report(print_report=False))
    assert set(base) - set(grown) == set(), (
        "inserting one entity at the front relabelled existing keys:\n"
        f"  before: {sorted(base)}\n  after:  {sorted(grown)}")
    assert len(set(grown) - set(base)) == 1, (
        "one inserted entity must add exactly one key, got "
        f"{sorted(set(grown) - set(base))}")
    for key in base:
        assert grown[key] == base[key], (
            f"{key} changed when an unrelated entity was inserted")


def test_foreign_warnings_are_not_pinned_by_the_digest() -> None:
    """A third-party warning raised inside preflight must not enter the pin.

    ``Simulation.preflight`` records EVERY warning raised while its
    validators run and turns each into an issue row, so on jax 0.10.2 the
    snapshot for ``examples/tutorials/patch_antenna_demo.py`` picked up
    ``uncoded None: jax.experimental.shard_map is deprecated in v0.8.0`` and
    the gate went red (measured: ``1 failed, 81 passed, 1 skipped in
    54.33s``) -- for the installed jax version, not for anything about the
    example. ``digest_preflight`` drops uncoded WARNING rows for that reason;
    this test pins both halves of the filter (foreign row dropped, rfx's own
    coded rows kept).
    """
    import warnings

    from rfx import Simulation

    sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    clean = lib.digest_variant(sim)["preflight"]
    assert clean, (
        "fixture must emit at least one rfx preflight row, otherwise this "
        "test cannot tell 'filtered correctly' from 'filtered everything'")

    cls = type(sim)
    original = cls._validate_simulation_config

    def _noisy(self, *a, **kw):
        warnings.warn("jax.experimental.shard_map is deprecated in v0.8.0")
        return original(self, *a, **kw)

    cls._validate_simulation_config = _noisy
    try:
        raw = [str(i) for i in sim.preflight()]
        polluted = lib.digest_variant(sim)["preflight"]
    finally:
        cls._validate_simulation_config = original

    assert any("shard_map is deprecated" in r for r in raw), (
        "preflight() no longer folds foreign warnings into its report -- if "
        "that is now fixed upstream, digest_preflight's uncoded-warning "
        "filter can go, but do not delete this test silently")
    assert polluted == clean, (
        "a foreign warning changed the pinned digest:\n"
        f"  without: {clean}\n  with:    {polluted}")

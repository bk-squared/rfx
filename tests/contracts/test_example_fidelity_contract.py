"""No-solve contract test for the #737 example-credibility gate (P4).

Builds every committed example/validation script that CAN be built without
solving, WITHOUT solving, and compares its ``preflight()`` +
``fidelity_report()`` output against a committed snapshot
(``tests/data/example_fidelity_snapshot.json``, regenerable with
``scripts/capture_example_fidelity_snapshot.py``). It is cheap precisely
because neither call time-steps: measured 2026-09-18 on this repo's CPU
lane, ``208 passed, 20 warnings in 91.44s`` for all 62 script/builder/variant
triples with optax installed. That is a dated reading of THIS file, so a later
PR that adds a test moves it; the 62 is the load-bearing number and
``test_discovery_matches_classification_table`` is what holds it. (For history: the same
lane measured ``194 passed, 13 warnings in 91.71s`` at c6788ef7, before the
#737-item-2 builders took it from 51 triples to 62.) On a machine without optax
beam_steering_superstrate.py declares it in OPTIONAL_DEPENDENCIES and
skips rather than failing -- that is CI's configuration.

EMISSION-DRIFT PIN PLUS A ZERO-*UNEXPLAINED*-ADVISORY BAR -- STILL NOT A
PHYSICS CHECK. Nothing here time-steps: the pin covers advisory codes,
locations and multiplicities plus structured fidelity measurements with
tolerance. Message wording and descriptive report prose are omitted, so a
green run says "this example's declared geometry and its
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

HOW MANY ROWS, AND OF WHAT -- do not look for a number here. The counts
live in ``tests/data/example_fidelity_snapshot.json`` (what is pinned) and
``tests/data/example_fidelity_advisories.json`` (what is classified), and
``test_every_pinned_advisory_row_is_classified`` asserts the two agree, per
(variant, code) and in total. A count written into this docstring is a third
copy nobody updates: it read "91 rows, 90 warning + 1 info" while the
snapshot held 90 and 89, because this PR's own fix to
beam_steering_superstrate retired an ``off_lattice_design_edges`` row and
the prose stayed behind. Read the data files.

Qualitatively, of the eleven variants #737 item 2 added only two scripts emit
anything: 13_subgrid_material_validation's dielectric arms (mesh resolution)
and nonuniform_patch_demo (lossless Q, off-lattice design edges), which are
the two the tutorial's own text already explains; the other new variants emit
none. #742 (the false positives and
the never-emitted advisories that made a zero-advisory bar unworkable) is
CLOSED, and the "tighten once #742 closes" step the tracker's first comment
planned HAS been executed -- as the zero-UNEXPLAINED-advisory bar described
above, which is the form that survived #742 closing.

What is still open is the OTHER reading of that comment, a bar of zero
advisories at all. The rows that remain are true statements about the
examples' own meshes and ports, so reaching zero means changing those
examples -- separate work from this gate, which pins whatever they say.
Whether it happens is a PI decision on #737. Until it is taken this file is
both things at once: a drift pin on what every audited example emits, and a
bar on every pinned row carrying a written disposition.

Every pinned row carries a disposition. One is ``defect-open`` -- #1100 for
the taper's declared-vs-realized WR-90 guide -- and the rest are ``intended``.
The Sheen low-pass filter's seven rows (including #928's congruent-feed
parity) left with its script on 2026-09-23. Again the totals are in the two
data files, not here. The rows the eleven #737-item-2 variants
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
``lib.CLASSIFICATION``). Discovery saw 138 scripts under examples/ +
validation/, all 138 classified: audited 39, builder_fused_with_solve 8,
module_level_solve 6, no_solve 4, no_simulation 81; the snapshot held 62
variants over those 39 audited scripts.

That block is the 2026-09-16 measurement and the classification table has
moved several times since. Re-measured 2026-09-23, after the Sheen low-pass
filter's script left with its case (``Counter(v.kind for v in
lib.CLASSIFICATION.values())`` plus the snapshot's own length): 101
classified -- audited 33, no_simulation 60, builder_fused_with_solve 5,
no_solve 3 -- and 51 snapshot variants over those 33 audited scripts.

Against the 47 scripts that existed when #737 was filed (ed3484c1 -- the
tracker's own denominator, kept here because the tracker's table is stated
in it): 30 audited / 4 builder_fused_with_solve / 5 module_level_solve /
8 no_simulation. The moves since the 2026-08-27 audit's 23/10/6/8: the
Sheen low-pass filter and cv15 grew build-only entry points (23->25, 10->8),
and #737 item 2
(2026-09-16) added five more -- hello_world, boundary_spec_demo,
nu_cavity_gate_scan and 13_subgrid_material_validation out of
``builder_fused_with_solve``, and nonuniform_patch_demo out of
``module_level_solve`` (its module-scope body moved into ``main()``, so
importing it no longer solves).

* ``builder_fused_with_solve`` (build and solve share one function with
  no separable build-only path) in that 47-set, i.e. the scripts this
  gate does NOT reach: cv09, cv10. "Out of this snapshot" is not
  "unverified": cv09 is REBUILT build-only from its own constants and
  helpers in tests/crossval/test_cv09_cv10_body_contract_controls.py
  (cv10 only at the spec level there, via ``_common_spec()``) and both
  solve weekly. examples/tutorials/cad_mesh_import_demo.py was the third
  and left this bucket with #1271: it now has ``build_simulation()``, and
  ``lib.call_builder`` turns its build-time ``trimesh`` miss into a
  visible skip on a lane without the [cad] extra.
* ``module_level_solve`` (importing the module solves at module scope):
  cv01-cv05 -- also out of scope, and deliberately never imported by this
  test. cv01/cv02 have ``--replay`` re-judge subprocess tests and cv05 a
  ``RFX_CV05_BUILD_ONLY=1`` subprocess build test.
* 8 build no rfx ``Simulation`` at all (cv16/cv17/cv20/cv21, the two
  crossval comparator subdirs, and the two research RCWA scripts) -- out
  of scope BY CONSTRUCTION.

WHAT THIS DOES NOT COVER. Against #722's own list of eight scripts that
solved geometry other than what they declared (the MSL notch filter, cv20,
cv11, cv16, cv17, the Sheen low-pass filter, cv09, cv15), this gate now
reaches ONE: cv15 (it arrived with its build-only entry point; the WR-90
waveguide-port case left the table with its case on 2026-09-21, the MSL notch
filter on 2026-09-22 and the Sheen low-pass filter on 2026-09-23).
cv20/cv16/cv17 are ``no_simulation`` and cv09 is
``builder_fused_with_solve``, so a no-solve gate cannot see them as those
scripts stand today. It DOES pin
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

import jax
import re

import pytest

from tests._structured_snapshot import assert_structured_close, without_prose

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
        sim = lib.call_builder(relpath, module, builder, variant)
    except lib.MissingOptionalDependency as exc:
        # Visible SKIP, never green: this repo's exit-code convention
        # (development_methodology.md 2.7) says a missing reference is a
        # SKIP that shows, not a pass. The variant stays enumerated in
        # _FLAT_VARIANTS and pinned in the snapshot, so coverage is
        # reported as skipped rather than silently lost, and an
        # UNDECLARED missing module is still a hard error (see
        # lib.OPTIONAL_DEPENDENCIES). A declared module missing when the
        # BUILDER runs (cad_mesh_import_demo's trimesh) skips the same way.
        pytest.skip(str(exc))
    # Round-trip through JSON so tuples (live digest) compare equal to lists
    # (snapshot, loaded from JSON) -- the digest itself is unchanged either
    # way, only its Python container types are normalized for comparison.
    import json
    actual = json.loads(json.dumps(lib.digest_variant(sim)))
    expected = snapshot[key]
    assert_structured_close(without_prose(actual), without_prose(expected), key)


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


# A ``ref`` is a ``|``-separated list of pointers. Each is either a tracker URL
# or ``<path>:<spec>``, where the spec is a comma-separated list of line numbers
# and ``start-end`` ranges. Both forms are already in use and both are good
# evidence; what was NOT checked before is whether the path exists and whether
# the lines are inside the file, so ``a.py:1`` passed.
_REF_PATH_RE = re.compile(r"^(?P<path>[^\s:|]+):(?P<spec>[0-9,\-]+)$")
_REF_CHUNK_RE = re.compile(r"^(?P<start>\d+)(?:-(?P<end>\d+))?$")
_REF_URL_PREFIX = "https://github.com/bk-squared/rfx/"
# 12 words is not a style preference: a one-line "intended, see the docstring"
# is the failure this file exists to prevent, and it fits in eleven.
_MIN_REASON_WORDS = 12


def _check_ref(where: str, ref: str) -> None:
    """Every pointer in *ref* resolves: a real file, a line inside it."""
    parts = [part.strip() for part in ref.split("|")]
    assert parts and all(parts), (
        f"{where}: ref has an empty pointer -- {ref!r}")
    for part in parts:
        if part.startswith(_REF_URL_PREFIX):
            continue
        match = _REF_PATH_RE.match(part)
        assert match, (
            f"{where}: ref pointer {part!r} is neither a tracker URL under "
            f"{_REF_URL_PREFIX} nor '<path>:<lines>' (a line, a 'start-end' "
            "range, or a comma-separated list of those)")
        path = lib.REPO_ROOT / match.group("path")
        assert path.is_file(), (
            f"{where}: ref names {match.group('path')!r}, which is not a file "
            "in the tree -- a pointer nobody can follow is not evidence")
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        for chunk in match.group("spec").split(","):
            bounds = _REF_CHUNK_RE.match(chunk)
            assert bounds, f"{where}: ref line spec {chunk!r} is not N or N-M"
            start = int(bounds.group("start"))
            end = int(bounds.group("end") or start)
            assert 1 <= start <= end <= n_lines, (
                f"{where}: ref points at {match.group('path')}:{chunk}, but "
                f"that file has {n_lines} lines -- the reason cannot be read "
                "there")


def test_advisory_classifications_are_well_formed() -> None:
    """A disposition has to carry its evidence, and defect-open its issue.

    Enforced per entry: a disposition from ``lib.ADVISORY_DISPOSITIONS``, a
    ``reason`` long enough to be a reason, and a ``ref`` whose every pointer
    RESOLVES -- an existing file and a line inside it, or a tracker URL. The
    ref check used to be ``ref.strip()``, which ``a.py:1`` passes; a pointer
    nobody can follow is the same as no evidence, and the whole value of this
    file is that a reader can go and check.

    ``defect-open`` additionally needs an integer issue number, so a row
    cannot be parked as a known defect with nothing tracking it.
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
        reason = entry["reason"].strip()
        assert len(reason) >= 80 and len(reason.split()) >= _MIN_REASON_WORDS, (
            f"{where}: the reason is the point of this file -- write why this "
            "row is there, citing the example's own docstring/comment or the "
            f"physics (at least {_MIN_REASON_WORDS} words; this one has "
            f"{len(reason.split())})")
        assert entry["ref"].strip(), (
            f"{where}: ref must point at the lines (or the issue) the reason "
            "is read from")
        _check_ref(where, entry["ref"])
        if entry["disposition"] == "defect-open":
            assert isinstance(entry.get("issue"), int), (
                f"{where}: 'defect-open' requires an integer issue number -- "
                "file the issue rather than parking the row")


def test_no_reason_is_boilerplate_across_different_advisories() -> None:
    """A repeated reason is allowed only between rows of the SAME code.

    The thing worth blocking is one sentence pasted over unrelated rows, which
    is what a generic "intended, the example means it" looks like: it spans
    codes. Repetition WITHIN a code is not that. The pairs in this file that
    share a reason today are each a sibling pair emitting the same advisory --
    the off-diagonal adjudication study's two drive variants
    (sheet_effective_size), the two supraconvergence studies w4 and w4r
    (mesh_resolution), and the thru fixture's band-pulse and insitu-refplane
    variants (pec_faces_finite_pec, sheet_effective_size). Forcing artificial
    rewordings would make the file worse, not more honest; a reason crossing
    codes stays a failure.
    """
    by_reason: dict[str, set[str]] = {}
    where_by_reason: dict[str, list[str]] = {}
    for entry in lib.load_advisory_classification():
        reason = entry["reason"].strip()
        by_reason.setdefault(reason, set()).add(entry["code"])
        where_by_reason.setdefault(reason, []).append(
            f"{entry['variant']} :: {entry['code']}")

    shared_across_codes = {
        reason: sorted(codes) for reason, codes in by_reason.items() if len(codes) > 1
    }
    assert not shared_across_codes, (
        "the same reason text is used for different advisory codes: "
        + "; ".join(
            f"{codes} <- {where_by_reason[reason][:4]}"
            for reason, codes in shared_across_codes.items())
        + " -- write what THAT row is doing there. A sentence that fits two "
        "different advisories explains neither.")


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
        assert_structured_close(grown[key], base[key], key)


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


@pytest.mark.parametrize("mutation", ["wording", "remove_code", "count", "number", "number_count", "severity", "source"])
def test_snapshot_contract_mutations(monkeypatch, mutation):
    """Exercise the actual contract entry point with controlled emissions."""
    from copy import deepcopy
    from rfx.preflight._common import PreflightIssue

    key, relpath, builder, variant = _FLAT_VARIANTS[0]
    issues = [PreflightIssue("original wording 2 cells", code="mesh_resolution", loc="x")]
    expected = {"preflight": lib.digest_preflight(issues), "fidelity": {}}
    changed = deepcopy(expected)
    if mutation == "wording":
        issues = [PreflightIssue("entirely revised warning 2 cells", code="mesh_resolution", loc="x")]
        changed["preflight"] = lib.digest_preflight(issues)
        assert "message" not in changed["preflight"][0]
    elif mutation == "number":
        changed["preflight"][0]["numbers"][0] /= 2
    elif mutation == "number_count":
        changed["preflight"][0]["numbers"].append(3.0)
    elif mutation in {"severity", "source"}:
        changed["preflight"][0][mutation] = "changed"
    elif mutation == "remove_code":
        changed["preflight"].clear()
    else:
        changed["preflight"].append(deepcopy(changed["preflight"][0]))
    monkeypatch.setattr(lib, "load_module", lambda _: None)
    monkeypatch.setattr(lib, "call_builder", lambda *args: None)
    monkeypatch.setattr(lib, "digest_variant", lambda _: changed)
    monkeypatch.setitem(test_example_matches_snapshot.__globals__, "_load_snapshot",
                        lambda: {key: expected})
    if mutation == "wording":
        test_example_matches_snapshot(key, relpath, builder, variant)
    else:
        with pytest.raises(AssertionError):
            test_example_matches_snapshot(key, relpath, builder, variant)


@pytest.mark.parametrize("field", [
    "declared_um", "declared_extent_um", "realized_um", "realized_extent_um",
    "mesh_extent_um", "cell_um", "face_residual_um", "midpoint_shift_um",
    "realized_lo_um", "realized_hi_um",
])
def test_snapshot_numeric_tolerance(field):
    assert_structured_close({field: [0.5e-9, 1000.0 + 5e-9]},
                            {field: [0.0, 1000.0]})
    with pytest.raises(AssertionError):
        assert_structured_close({field: [0.0, 1000.01]},
                                {field: [0.0, 1000.0]})


def test_um_tolerance_uses_only_the_field_name():
    # An ancestor entity/path mentioning um must not relax eps_r or numbers.
    for field in ("eps_r", "numbers", "not_um_units"):
        with pytest.raises(AssertionError):
            assert_structured_close({"parent_um": {field: [5e-10]}},
                                    {"parent_um": {field: [0.0]}})


def test_message_numeric_tokens_and_tolerance():
    from tests._structured_snapshot import message_numbers, without_prose

    assert message_numbers("2 cells, -3.5 µm, +.25, 6., 1e-9, -2.5E+3") == [
        2.0, -3.5, .25, 6.0, 1e-9, -2500.0]
    assert without_prose({"message": "no quantities"}) == {"numbers": []}
    assert_structured_close({"numbers": [2.0 + 1e-9, 5e-13]},
                            {"numbers": [2.0, 0.0]})
    for numbers in ([1.0, 0.0], [2.0], [0.0, 2.0], [2.0, 5e-11]):
        with pytest.raises(AssertionError):
            assert_structured_close({"numbers": numbers}, {"numbers": [2.0, 0.0]})


@pytest.mark.parametrize("field,value", [("count", 1), ("class", "MSL port")])
def test_out_of_scope_count_and_class_are_pinned(field, value):
    from copy import deepcopy
    report = [{"entity": "NOT AUDITED by this report", "findings": [{
        "kind": "out-of-scope", "detail": "2 waveguide port(s)",
        "entities": [{"count": 2, "class": "waveguide port"}],
    }]}]
    expected = lib.digest_fidelity(report)
    changed = deepcopy(report)
    changed[0]["findings"][0]["detail"] = "rewritten description"
    assert_structured_close(lib.digest_fidelity(changed), expected)
    changed[0]["findings"][0]["entities"][0][field] = value
    with pytest.raises(AssertionError):
        assert_structured_close(lib.digest_fidelity(changed), expected)

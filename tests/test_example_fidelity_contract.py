"""No-solve contract test for the #737 example-credibility gate (P4).

Builds every committed example/validation script that CAN be built without
solving, WITHOUT solving, and compares its ``preflight()`` +
``fidelity_report()`` output against a committed snapshot
(``tests/data/example_fidelity_snapshot.json``, regenerable with
``scripts/capture_example_fidelity_snapshot.py``). It is cheap precisely
because neither call time-steps: measured 2026-08-28 on this repo's CPU
lane, ``85 passed ... in 52.52s`` for all 33 script/builder/variant triples
(``84 passed, 1 skipped in 52.69s`` without optax installed, which is CI's
configuration -- see OPTIONAL_DEPENDENCIES).

SNAPSHOT, not a zero-advisory bar. This does NOT require every example to
emit zero advisories today: #742 is open (~45% of advisory codes never fire
in this corpus, and ``absorber_budget_exceeds_axis`` false-fires on
PEC-closed axes -- see ``validation/tmtt_paper/waveguide_dielectric_taper.py``,
which declares z=Boundary(lo='pec', hi='pec') and still gets an absorber-
budget warning). What ships here PINS what each example emits TODAY, so any
DRIFT from that baseline fails CI. Tighten to a zero-advisory bar once #742
lands; until then, do not read a green run here as "this example is clean" --
read it as "this example did not change since the snapshot was taken."

COVERAGE (measured 2026-08-27, re-derive with test_discovery_matches_
classification_table and the Counter in this module's own analysis): of the
47 scripts under examples/ and validation/,

* 23 are AUDITED here (33 script/builder/variant triples -- several
  builders are called at more than one input, e.g. booleans or cell sizes,
  matching what each script's own ``main()`` drives);
* 10 are ``builder_fused_with_solve``: build and solve share one function
  with no separable build-only path (e.g. cv07/cv15's ``run_rfx()`` builds
  ``sim`` and calls a solve entrypoint in the same function). This gate
  does NOT cover them -- notably cv07, cv09, cv10, cv15, cv18,
  examples/quickstart/hello_world.py, examples/tutorials/boundary_spec_demo.py,
  examples/tutorials/cad_mesh_import_demo.py,
  validation/research/nu_cavity_gates/nu_cavity_gate_scan.py, and
  validation/research/subgrid/13_subgrid_material_validation.py are ALL
  OUT OF SCOPE until they get a builder separable from their solve;
* 6 are ``module_level_solve``: importing the module solves at module scope
  (cv01-cv05, examples/tutorials/nonuniform_patch_demo.py) -- also out of
  scope, and deliberately never imported by this test;
* 8 build no rfx ``Simulation`` at all (cv16/cv17/cv20/cv21, the two
  crossval comparator subdirs, and the two research RCWA scripts) -- out of
  scope BY CONSTRUCTION.

WHAT THIS DOES NOT COVER. Against #722's own list of eight scripts that
solved geometry other than what they declared (cv06b, cv20, cv11, cv16,
cv17, cv07, cv09, cv15), this gate reaches TWO: cv06b and cv11.
cv20/cv16/cv17 are ``no_simulation`` and cv07/cv09/cv15 are
``builder_fused_with_solve``, so a no-solve gate cannot see them as those
scripts stand today. Nor does it reach cv21's fence-post error (#739, cv21
is ``no_simulation``) or cv15's cavity (#740). It DOES pin
differentiable_s11_design's two domain widths (#738); that script's third
declared width, the port aperture, falls under fidelity_report's own
out-of-scope port row. A green run here is not evidence that the #722
campaign's class is closed -- closing it needs the remaining six scripts to
grow a builder separable from their solve.

Every one of the 47 is EXPLICITLY classified in
``tests/_example_fidelity_lib.CLASSIFICATION`` (enumerate-and-classify: a
new script with no entry fails ``test_discovery_matches_classification_
table`` instead of silently passing uncovered), and the two out-of-scope
buckets required to be machine-checkable by the 2026-08-27 review are
verified against each script's own AST in
``test_not_auditable_classifications_are_machine_checked`` below (extended,
cheaply, to all four buckets: a classification that disagrees with what the
script's source actually does is a bug in the table, not a judgement call).

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
from pathlib import Path
from typing import Any

import jax
import re

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _example_fidelity_lib as lib  # noqa: E402


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
    JAX_ENABLE_X64=1" (conftest.py; echoed at tests/test_x64_scan_carry_
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

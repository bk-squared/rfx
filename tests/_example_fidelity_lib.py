"""Shared machinery for the P4/#737 example-credibility contract.

Both ``tests/test_example_fidelity_contract.py`` (the gate) and
``scripts/capture_example_fidelity_snapshot.py`` (the regen command) import
this module, so the digest the gate compares against is BY CONSTRUCTION the
same digest the snapshot was written with — there is no second, drifting
copy of the comparison logic.

Discovery is a plain recursive glob over ``examples/`` and ``validation/``
(no directory-depth pattern, no underscore filter): a one-level
``*/*.py`` glob misses doubly-nested files
(``validation/crossval/comparators/deep/...``) and an
``name.startswith("_")`` filter hides private helpers in ANY directory —
both were measured to let a new file land with zero classification and a
green gate (2026-08-27 review). Recursive-and-unfiltered has neither hole:
every ``.py`` under either tree is discovered and must appear in
``CLASSIFICATION`` or the contract test fails.

Classification is a hand-authored table (four buckets: ``audited``,
``builder_fused_with_solve``, ``module_level_solve``, ``no_simulation``)
because "does this script have a build step separable from its solve step"
is a judgement call a machine cannot make reliably on its own — the AST
heuristic in ``_functions_building_simulation`` gets it right for all 47
scripts in this repo today (independently re-derived, matches the 2026-08-27
audit's 23/10/6/8 split exactly), so ``test_example_fidelity_contract.py``
uses it as a MACHINE CHECK on top of the hand-authored table for every
bucket, not just the two the audit required — a script whose classification
disagrees with what its own source does is a bug in the table, not a
judgement call, and should fail loudly rather than rot silently.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable

import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "tests" / "data" / "example_fidelity_snapshot.json"

# Every Simulation method that actually time-steps or otherwise solves. A
# script whose builder-candidate function calls one of these in its OWN body
# has fused build and solve -- there is no point in that function's source
# where "declared but not yet solved" holds, so it cannot be audited without
# either solving for real (forbidden) or forking the script (out of scope).
SOLVE_ATTRS = frozenset({
    "run",
    "compute_msl_s_matrix",
    "compute_waveguide_s_matrix",
    "compute_mixed_s_matrix",
    "compute_coaxial_s_matrix",
    "compute_coaxial_line_reflection",
    "compute_coaxial_two_port",
    "compute_coax_msl_transition",
    "compute_rcs",
})


def discover_scripts() -> list[str]:
    """Every committed example/validation script, as repo-relative posix paths.

    Recursive, unfiltered glob -- see module docstring for the two escape
    hatches this avoids relative to a one-level, underscore-filtered glob.
    """
    paths = sorted((REPO_ROOT / "examples").rglob("*.py")) + sorted(
        (REPO_ROOT / "validation").rglob("*.py")
    )
    return sorted(p.relative_to(REPO_ROOT).as_posix() for p in paths)


# --------------------------------------------------------------------------
# AST predicates -- these are what make each bucket machine-checkable rather
# than a trusted prose reason (2026-08-27 review, required change #4).
# --------------------------------------------------------------------------

def _parse(relpath: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relpath).read_text(), filename=relpath)


def _is_simulation_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
    return name == "Simulation"


def _is_solve_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return isinstance(f, ast.Attribute) and f.attr in SOLVE_ATTRS


def real_simulation_call_count(relpath: str) -> int:
    """AST count of real ``Simulation(...)`` calls (text/docstring mentions
    do not count -- e.g. cv21's docstring mentions "Simulation" but the
    script never constructs one)."""
    return sum(1 for n in ast.walk(_parse(relpath)) if _is_simulation_call(n))


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
            and t.left.id == "__name__" and len(t.ops) == 1
            and isinstance(t.ops[0], ast.Eq)
            and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


def has_main_guard(relpath: str) -> bool:
    """A module-scope ``if __name__ == "__main__":`` guard exists."""
    return any(_is_main_guard(node) for node in _parse(relpath).body)


def has_top_level_solve_call(relpath: str, *,
                            skip_main_guard: bool = False) -> bool:
    """A solve entrypoint is called from module scope (outside any
    function/class def).

    ``skip_main_guard=True`` additionally ignores the body of
    ``if __name__ == "__main__":`` -- that block does NOT run on import, so
    it is the right predicate for "would importing this module solve?".
    Without the flag the answer is "is there a solve call anywhere outside a
    def", which is what the ``module_level_solve`` bucket (no main guard at
    all) is classified on.
    """
    tree = _parse(relpath)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if skip_main_guard and _is_main_guard(node):
            continue
        for n in ast.walk(node):
            if _is_solve_call(n):
                return True
    return False


def functions_building_simulation(relpath: str) -> dict[str, bool]:
    """{function_name: fused} for every top-level function whose OWN body
    (not nested function defs it merely contains) constructs a Simulation.
    ``fused=True`` means that same function also calls a solve entrypoint --
    build and solve are not separable."""
    tree = _parse(relpath)
    out: dict[str, bool] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body_nodes = list(ast.walk(node))
        has_sim = any(_is_simulation_call(n) for n in body_nodes)
        if not has_sim:
            continue
        fused = any(_is_solve_call(n) for n in body_nodes)
        out[node.name] = fused
    return out


# --------------------------------------------------------------------------
# Module loading -- exec the script's top level (imports, defs, constants),
# never main(). Safe for every "audited" and "builder_fused_with_solve"
# entry: all 23 audited scripts carry a main guard AND no module-scope solve
# call, both asserted per script by the "audited" branch of
# test_not_auditable_classifications_are_machine_checked, so nothing solves
# merely by loading them. Pattern matches tests/test_crossval_example_imports.py.
# --------------------------------------------------------------------------

class MissingOptionalDependency(ImportError):
    """A script could not be loaded because a DECLARED optional dep is absent.

    Distinct from any other ``ModuleNotFoundError``, which stays a hard
    error: a script that fails to import for an UNDECLARED reason is a
    broken example, not an environment difference, and must not be able to
    quietly leave this gate's coverage.
    """

    def __init__(self, relpath: str, module: str) -> None:
        super().__init__(
            f"{relpath} needs optional dependency {module!r}, which is not "
            f"installed here (declared in OPTIONAL_DEPENDENCIES)")
        self.relpath = relpath
        self.module = module


# Optional third-party imports an audited script may legitimately lack in a
# given environment. Keyed by script, so the exemption is per-script and
# narrow: an undeclared ModuleNotFoundError is still a hard failure.
#
# Why this table exists rather than a blanket try/except on import: CI
# installs `.[dev]`, which does NOT carry optax, while a developer pod often
# does. Without the declaration a contributor adding `import optax` to any
# example would silently remove it from this gate in CI and nobody would
# see it -- the exact "visible SKIP, never green" line in
# development_methodology.md 2.7's exit-code convention.
# test_optional_dependency_declarations_are_grounded keeps the table honest
# in both directions.
OPTIONAL_DEPENDENCIES: dict[str, frozenset[str]] = {
    "validation/tmtt_paper/beam_steering_superstrate.py": frozenset({"optax"}),
    "validation/tmtt_paper/waveguide_dielectric_taper.py": frozenset({"optax"}),
}


def load_module(relpath: str) -> ModuleType:
    path = REPO_ROOT / relpath
    name = f"_example_fidelity_{path.stem}_{abs(hash(relpath))}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        declared = OPTIONAL_DEPENDENCIES.get(relpath, frozenset())
        if exc.name in declared:
            raise MissingOptionalDependency(relpath, exc.name) from exc
        raise
    finally:
        sys.modules.pop(name, None)
    return module


# --------------------------------------------------------------------------
# Classification table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    label: str
    kwargs: Callable[[ModuleType], dict]


@dataclass(frozen=True)
class Builder:
    fn: str
    result_index: int | None  # None: builder returns the Simulation directly
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Entry:
    kind: str  # audited | builder_fused_with_solve | module_level_solve | no_simulation
    reason: str
    builders: tuple[Builder, ...] = field(default_factory=tuple)


def _v(label: str, **kw) -> Variant:
    return Variant(label, lambda _m, kw=kw: dict(kw))


def _v_from(label: str, fn: Callable[[ModuleType], dict]) -> Variant:
    return Variant(label, fn)


CLASSIFICATION: dict[str, Entry] = {
    # ---- no_simulation (8): zero real Simulation() calls, AST-verified ----
    "validation/crossval/16_pec_sphere_mie_ka_sweep.py": Entry(
        "no_simulation",
        "drives the functional rfx.rcs.compute_rcs entry point directly on "
        "a hand-built Grid/MaterialArrays -- no Simulation object exists"),
    "validation/crossval/17_dielectric_sphere_mie.py": Entry(
        "no_simulation",
        "drives the functional rfx.rcs.compute_rcs entry point directly on "
        "a hand-built Grid/MaterialArrays -- no Simulation object exists"),
    "validation/crossval/20_msl_phase_referee.py": Entry(
        "no_simulation",
        "Stage A drives openEMS/ContinuousStructure and explicitly does NOT "
        "import rfx (own docstring); Stage B only reads a committed rfx-"
        "produced JSON record -- no Simulation is built by this script"),
    "validation/crossval/21_coax_two_port_referee.py": Entry(
        "no_simulation",
        "same two-stage openEMS-referee shape as cv20: no rfx Simulation is "
        "constructed by this script"),
    "validation/crossval/comparators/fdfd_hplane.py": Entry(
        "no_simulation",
        "plain numpy/scipy.sparse FDFD comparator -- no rfx import at all"),
    "validation/crossval/palace/mesh_patch.py": Entry(
        "no_simulation",
        "gmsh mesh-generation utility for the Palace comparator -- no rfx "
        "import at all"),
    "validation/research/floquet/rcwa_referee.py": Entry(
        "no_simulation",
        "plain-numpy analytic RCWA referee -- no rfx import at all"),
    "validation/research/rcwa_referee/rcwa_referee_step1.py": Entry(
        "no_simulation",
        "drives the external grcwa library directly -- no rfx import at all"),

    # ---- module_level_solve (6): solves at import time, no main guard ----
    "validation/crossval/01_waveguide_bend.py": Entry(
        "module_level_solve",
        "builds and calls .run(...) at module scope with no "
        "`if __name__ == '__main__':` guard -- importing this module solves"),
    "validation/crossval/02_ring_resonator.py": Entry(
        "module_level_solve",
        "builds and calls .run(...) at module scope with no main guard"),
    "validation/crossval/03_straight_waveguide_flux.py": Entry(
        "module_level_solve",
        "builds and calls .run(...) at module scope with no main guard"),
    "validation/crossval/04_multilayer_fresnel.py": Entry(
        "module_level_solve",
        "builds and calls .run(...) at module scope with no main guard"),
    "validation/crossval/05_patch_antenna.py": Entry(
        "module_level_solve",
        "builds and calls .run(...) at module scope with no main guard"),
    "examples/tutorials/nonuniform_patch_demo.py": Entry(
        "module_level_solve",
        "builds and calls .run(...) at module scope with no main guard"),

    # ---- builder_fused_with_solve (10): build+solve share one function ---
    "examples/quickstart/hello_world.py": Entry(
        "builder_fused_with_solve",
        "`main()` builds and calls .run(...) in the same function -- no "
        "separable build-only path"),
    "examples/tutorials/boundary_spec_demo.py": Entry(
        "builder_fused_with_solve",
        "`_run_and_report()` builds and calls .run(...) in the same "
        "function for every boundary spec under test"),
    "examples/tutorials/cad_mesh_import_demo.py": Entry(
        "builder_fused_with_solve",
        "`main()` builds and calls .run(...) in the same function"),
    "validation/crossval/07_sheen_lpf.py": Entry(
        "builder_fused_with_solve",
        "`run_rfx()` builds `sim` and calls sim.compute_msl_s_matrix(...) in "
        "the same function -- no separable build-only path"),
    "validation/crossval/09_half_symmetric_waveguide.py": Entry(
        "builder_fused_with_solve",
        "`_run_cavity()` builds and calls .run(...) in the same function"),
    "validation/crossval/10_pmc_cpml_half_symmetric.py": Entry(
        "builder_fused_with_solve",
        "`run_uniform()`/`run_nonuniform()` each build and call .run(...) "
        "in the same function"),
    "validation/crossval/15_patch_antenna_rt5880.py": Entry(
        "builder_fused_with_solve",
        "`run_rfx()` builds `sim` and calls sim.run(...) at line 268 in the "
        "same function -- no separable build-only path"),
    "validation/crossval/18_wr90_iris_modematch.py": Entry(
        "builder_fused_with_solve",
        "`run_point()` builds and calls sim.compute_waveguide_s_matrix(...) "
        "in the same function"),
    "validation/research/nu_cavity_gates/nu_cavity_gate_scan.py": Entry(
        "builder_fused_with_solve",
        "`_tm110_error()`/`_tm111_error()` each build and call .run(...) in "
        "the same function"),
    "validation/research/subgrid/13_subgrid_material_validation.py": Entry(
        "builder_fused_with_solve",
        "`run_example()` builds and calls .run(...) in the same function"),

    # ---- audited (23): builder is separable from solve ----
    "examples/inverse_design/differentiable_s11_design.py": Entry(
        "audited", "`_build_sim()` returns Simulation with no solve call",
        (Builder("_build_sim", None, (_v("default"),)),)),
    "examples/inverse_design/field_observable_shielding.py": Entry(
        "audited", "`_build_sim()` returns Simulation with no solve call",
        (Builder("_build_sim", None, (_v("default"),)),)),
    "examples/inverse_design/multilayer_ar_coating.py": Entry(
        "audited", "`_build_simulation()` returns Simulation with no solve call",
        (Builder("_build_simulation", None, (_v("default"),)),)),
    "examples/inverse_design/progressive_demo.py": Entry(
        "audited",
        "`sim_factory(dx)` returns Simulation with no solve call; main() "
        "drives it at the two progressive-refinement cell sizes",
        (Builder("sim_factory", None, (
            _v("dx=1.0mm", dx=1.0e-3), _v("dx=0.5mm", dx=0.5e-3))),)),
    "examples/tutorials/adi_solver_demo.py": Entry(
        "audited",
        "`build_cavity(solver, adi_cfl_factor)` returns Simulation with no "
        "solve call; main() drives it at the three solver/CFL settings "
        "compared in the script",
        (Builder("build_cavity", None, (
            _v("yee_cfl5", solver="yee", adi_cfl_factor=5.0),
            _v("adi_cfl2", solver="adi", adi_cfl_factor=2.0),
            _v("adi_cfl5", solver="adi", adi_cfl_factor=5.0))),)),
    "examples/tutorials/antenna_farfield_pattern.py": Entry(
        "audited", "`build_simulation()` returns Simulation with no solve call",
        (Builder("build_simulation", None, (_v("default"),)),)),
    "examples/tutorials/artifact_report_demo.py": Entry(
        "audited", "`build_demo_simulation()` returns Simulation with no solve call",
        (Builder("build_demo_simulation", None, (_v("default"),)),)),
    "examples/tutorials/materials_and_dispersion.py": Entry(
        "audited", "`make_sim()` returns Simulation with no solve call",
        (Builder("make_sim", None, (_v("default"),)),)),
    "examples/tutorials/patch_antenna_demo.py": Entry(
        "audited", "`build_simulation()` returns Simulation with no solve call",
        (Builder("build_simulation", None, (_v("default"),)),)),
    "examples/tutorials/ports_and_sparams_101.py": Entry(
        "audited",
        "four independent builders, none of which calls a solve entrypoint",
        (
            Builder("build_generic_port_demo", None, (
                _v("add_component=False", add_component=False),
                _v("add_component=True", add_component=True))),
            Builder("build_microstrip_ports", None, (_v("default"),)),
            Builder("build_waveguide_ports", None, (_v("default"),)),
            Builder("build_coaxial_port", None, (_v("default"),)),
        )),
    "examples/tutorials/rcs_scattering.py": Entry(
        "audited", "`build_preflight_model()` returns Simulation with no solve call",
        (Builder("build_preflight_model", None, (_v("default"),)),)),
    "examples/tutorials/resonance_harminv.py": Entry(
        "audited", "`build_cavity()` returns Simulation with no solve call",
        (Builder("build_cavity", None, (_v("default"),)),)),
    "examples/tutorials/run_control_and_fields.py": Entry(
        "audited", "`build_simulation()` returns Simulation with no solve call",
        (Builder("build_simulation", None, (_v("default"),)),)),
    "examples/tutorials/slab_rt_flux_monitor.py": Entry(
        "audited",
        "`build_sim(with_slab)` returns Simulation with no solve call; "
        "main() drives it at both booleans",
        (Builder("build_sim", None, (
            _v("with_slab=False", with_slab=False),
            _v("with_slab=True", with_slab=True))),)),
    "validation/crossval/06b_msl_notch_filter_uniform.py": Entry(
        "audited", "`_build_sim()` returns Simulation with no solve call",
        (Builder("_build_sim", None, (_v("default"),)),)),
    "validation/crossval/11_waveguide_port_wr90.py": Entry(
        "audited",
        "`_build_sim(freqs, ...)` returns Simulation with no solve call; "
        "main() drives it at the empty-guide and PEC-short configurations",
        (Builder("_build_sim", None, (
            _v_from("empty", lambda m: dict(freqs=m.FREQS_HZ)),
            _v_from("pec_short", lambda m: dict(
                freqs=m.FREQS_HZ, pec_short_x=m.PEC_SHORT_X)),
        )),)),
    "validation/crossval/14_rect_cavity_pozar.py": Entry(
        "audited",
        "`build_cavity(dx)` returns Simulation with no solve call; main() "
        "drives it at the main gate leg and the convergence-witness cell size",
        (Builder("build_cavity", None, (
            _v("dx=1.0mm", dx=1.0e-3), _v("dx=0.5mm", dx=0.5e-3))),)),
    "validation/crossval/19_wr90_iris_filter_aghanim.py": Entry(
        "audited",
        "`build(geo, ...)` returns (Simulation, cpml_cells, guide_length) "
        "with no solve call; main()'s gated leg calls it on "
        "rasterized_geometry(GATED_CELLS, allow_asymmetric=False)",
        (Builder("build", 0, (
            _v_from("gated", lambda m: dict(
                geo=m.rasterized_geometry(m.GATED_CELLS, allow_asymmetric=False))),
        )),)),
    "validation/research/subgrid/12_subgrid_disjoint_prototype.py": Entry(
        "audited",
        "`_build_disjoint_simulation()` returns Simulation with no solve call",
        (Builder("_build_disjoint_simulation", None, (_v("default"),)),)),
    "validation/tmtt_paper/beam_steering_superstrate.py": Entry(
        "audited",
        "`build_problem()` returns (Simulation, region, grid, plate, lam, "
        "f0) with no solve call",
        (Builder("build_problem", 0, (_v("default"),)),)),
    "validation/tmtt_paper/lumped_port_gradient_check.py": Entry(
        "audited", "`build(f0=3.0e9)` returns Simulation with no solve call",
        (Builder("build", None, (_v("default"),)),)),
    "validation/tmtt_paper/msl_stub_notch_tuning.py": Entry(
        "audited",
        "`build_sim(freqs)` returns (Simulation, y_trace, trace_y_hi, "
        "d_set, p_set) with no solve call; main() drives it at F_TARGET",
        (Builder("build_sim", 0, (
            _v_from("f_target", lambda m: dict(
                freqs=jnp.asarray([m.F_TARGET], dtype=jnp.float32))),
        )),)),
    "validation/tmtt_paper/waveguide_dielectric_taper.py": Entry(
        "audited", "`build_sim()` returns Simulation with no solve call",
        (Builder("build_sim", None, (_v("default"),)),)),
    # ---- issue #786 attribution lane (validation/research/convergence_floor)
    "validation/research/convergence_floor/fixture.py": Entry(
        "audited",
        "`build_sim(scale, dz_profile, ...)` returns the W4R P-C "
        "Simulation with no solve call; `measure()` is what runs it",
        (Builder("build_sim", None, (
            _v_from("uniform_s1.0", lambda m: dict(
                scale=1.0, dz_profile=m.pc_uniform_profile(1.0))),
            _v_from("multiband_s1.0", lambda m: dict(
                scale=1.0, dz_profile=m.pc_dz_profile_sym(1.0))),
            _v_from("no_trace_s1.0", lambda m: dict(
                scale=1.0, dz_profile=m.pc_uniform_profile(1.0),
                with_trace=False)),
        )),)),
    "validation/research/convergence_floor/d4_reference.py": Entry(
        "builder_fused_with_solve",
        "`twin_rung()` builds the exact-reference empty-box twin and calls "
        "sim.run() in the same function -- there is no build-only path",),
    "validation/research/convergence_floor/__init__.py": Entry(
        "no_simulation", "package marker"),
    "validation/research/convergence_floor/d0_reproduce.py": Entry(
        "no_simulation",
        "drives fixture.measure(); constructs no Simulation of its own"),
    "validation/research/convergence_floor/d1_geometry.py": Entry(
        "no_simulation",
        "assembles materials/masks through fixture.build_sim(); constructs "
        "no Simulation of its own"),
    "validation/research/convergence_floor/d2_edge.py": Entry(
        "no_simulation",
        "drives fixture.build_sim(); constructs no Simulation of its own"),
    "validation/research/convergence_floor/d3_port.py": Entry(
        "no_simulation",
        "drives fixture.measure(); constructs no Simulation of its own"),
    "validation/research/convergence_floor/d2_retake.py": Entry(
        "no_simulation",
        "re-applies the frozen D2 rule to the nine-rung ladder from "
        "committed JSON; no Simulation"),
    "validation/research/convergence_floor/d5_predeclare_and_run.py": Entry(
        "no_simulation",
        "drives fixture.measure(); constructs no Simulation of its own"),
    "validation/research/convergence_floor/d6_two_term_model.py": Entry(
        "no_simulation", "pure model fitting on committed JSON"),
    "validation/research/convergence_floor/estimators.py": Entry(
        "no_simulation", "signal-processing estimators; no rfx Simulation"),
    "validation/research/convergence_floor/ladder_guard.py": Entry(
        "no_simulation", "ladder-reading preconditions; no rfx Simulation"),
    "validation/research/convergence_floor/predeclare.py": Entry(
        "no_simulation", "writes the frozen window file; no Simulation"),
    "validation/research/convergence_floor/predeclare_addendum.py": Entry(
        "no_simulation", "writes the addendum window file; no Simulation"),
    "validation/research/convergence_floor/verdict.py": Entry(
        "no_simulation", "reads committed JSON only; no Simulation"),
}


def iter_audited_variants():
    """Yield (script, builder_fn, variant_label, module, sim) for every
    audited (script, builder, variant) triple -- the harness's atomic unit."""
    for relpath, entry in sorted(CLASSIFICATION.items()):
        if entry.kind != "audited":
            continue
        module = load_module(relpath)
        for builder in entry.builders:
            fn = getattr(module, builder.fn)
            for variant in builder.variants:
                kwargs = variant.kwargs(module)
                result = fn(**kwargs)
                sim = result if builder.result_index is None else result[builder.result_index]
                yield relpath, builder.fn, variant.label, sim


# --------------------------------------------------------------------------
# Digests -- keyed by entity IDENTITY (material + declared bounds) and axis
# LETTER, never by list position (2026-08-27 review, required change #2):
# inserting one new entity anywhere in a script's geometry list must not
# relabel every entity after it in the diff.
# --------------------------------------------------------------------------

def _entity_key(item: dict, seen: dict[str, int]) -> str:
    name = str(item.get("entity", ""))
    if name.startswith("domain"):
        return "domain"
    lo, hi = item.get("declared_lo"), item.get("declared_hi")
    if lo is not None and hi is not None:
        kind = "thin_conductor" if name.startswith("thin_conductor") else "geometry"
        mat = item.get("material") or {}
        tag = mat.get("name", mat.get("kind", "?"))
        lo_s = ",".join(f"{v * 1e6:.3f}" for v in lo)
        hi_s = ",".join(f"{v * 1e6:.3f}" for v in hi)
        key = f"{kind}|{tag}|{lo_s}um->{hi_s}um"
    else:
        # No analytic bounding box: falls back to the report's own name. The
        # ONE row that lands here in this corpus is fidelity_report's own
        # out-of-scope notice, ``"NOT AUDITED by this report"`` (28 of the 33
        # variants have ports/sources, so 28 of the snapshot's entity rows
        # are this key) -- a fixed string with no list index in it, so it is
        # position-independent for the same reason the keys above are. A real
        # entity with no ``bounding_box()`` would fall here too and WOULD
        # carry its index (``geometry[7] 'copper'``); nothing in this repo
        # does today, and ``test_snapshot_keys_survive_entity_insertion``
        # pins the position-independence that matters.
        key = name
    n = seen.get(key, 0)
    seen[key] = n + 1
    return key if n == 0 else f"{key}#{n + 1}"


def digest_fidelity(report: list[dict]) -> dict:
    seen: dict[str, int] = {}
    out = {}
    for item in report:
        key = _entity_key(item, seen)
        axes = {ax["axis"]: {k: v for k, v in ax.items() if k != "axis"}
                for ax in item.get("axes", [])}
        out[key] = dict(
            material=item.get("material"),
            n_cells=item.get("n_cells"),
            realization=item.get("realization"),
            findings=[dict(f) for f in item.get("findings", [])],
            axes=axes,
        )
    return out


def digest_preflight(report) -> list[dict]:
    """Preflight rows this gate pins -- rfx's OWN findings, nothing else.

    ``Simulation.preflight`` runs its validators inside
    ``warnings.catch_warnings(record=True)`` + ``simplefilter("always")``
    (rfx/api/_preflight.py) and turns EVERY warning raised in that window
    into an issue row, including warnings from third-party libraries that
    happen to warn on first use inside it. rfx's own findings always carry a
    ``code`` (they are ``PreflightWarning`` instances, or a ``ValueError``
    escalated to a coded/uncoded ERROR row); a warning-severity row with
    ``code == "uncoded"`` is therefore foreign, and pinning it makes the
    snapshot a function of the installed dependency versions rather than of
    the example.

    Measured: on jax 0.10.2 ``examples/tutorials/patch_antenna_demo.py``
    gained a 7th row, ``uncoded None: jax.experimental.shard_map is
    deprecated in v0.8.0``, and the gate failed
    (``1 failed, 81 passed, 1 skipped in 54.33s``) -- while CI stayed green
    because it resolves jax 0.6.2. It is also ORDER dependent: importing a
    module that pulls ``jax.experimental.shard_map`` earlier in the same
    process consumes the once-per-location warning and the row disappears.
    Uncoded ERROR rows are kept: those come from an rfx validator raising.
    """
    rows = [issue.to_dict() for issue in report]
    rows = [d for d in rows
            if not (d["code"] == "uncoded" and d["severity"] != "error")]
    rows.sort(key=lambda d: (d["code"], str(d["loc"]), d["message"]))
    return rows


def _round_floats(obj):
    """Round every float in the digest to 12 significant digits.

    The digest carries sums over cell-size arrays (``np.sum`` of 200+ float
    entries), whose last one or two decimal digits depend on summation order
    and therefore on the BLAS/CPU the run lands on -- the snapshot holds
    values like ``102000.00000000001`` and ``3140.000000000011``. Comparing
    those with ``==`` makes the gate fail for a reason that has nothing to do
    with the example (this repo has been bitten by cross-machine float drift
    before -- PR #119's slow-suite lane). Twelve significant digits is ~3
    orders below float64's ~15-16 and ~6 orders above any physically
    meaningful difference here (1e-12 relative on a 100 mm domain is 1e-13 m),
    so it removes the noise without hiding a real geometry change.
    """
    if isinstance(obj, float):
        return float(f"{obj:.12g}")
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def digest_variant(sim) -> dict:
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        preflight = sim.preflight()
    fidelity = sim.fidelity_report(print_report=False)
    return _round_floats(dict(
        preflight=digest_preflight(preflight),
        fidelity=digest_fidelity(fidelity),
    ))

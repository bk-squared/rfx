"""Shared machinery for the P4/#737 example-credibility contract.

Both ``tests/contracts/test_example_fidelity_contract.py`` (the gate) and
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

Classification is a hand-authored table (five buckets: ``audited``,
``builder_fused_with_solve``, ``module_level_solve``, ``no_simulation``,
``no_solve``)
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


def has_any_solve_call(relpath: str) -> bool:
    """A solve entrypoint is called ANYWHERE in the module (module scope,
    inside a function, inside the main guard). The predicate the ``no_solve``
    bucket needs: those scripts build a Simulation only to read its grid and
    must never gain a solve without reclassification."""
    return any(_is_solve_call(n) for n in ast.walk(_parse(relpath)))


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
# merely by loading them. Pattern matches tests/contracts/test_crossval_example_imports.py.
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
    kind: str  # audited | builder_fused_with_solve | module_level_solve
    #            | no_simulation | no_solve
    reason: str
    builders: tuple[Builder, ...] = field(default_factory=tuple)


def _v(label: str, **kw) -> Variant:
    return Variant(label, lambda _m, kw=kw: dict(kw))


def _v_from(label: str, fn: Callable[[ModuleType], dict]) -> Variant:
    return Variant(label, fn)


CLASSIFICATION: dict[str, Entry] = {
    # ---- no_simulation (19): zero real Simulation() calls, AST-verified --
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
    "validation/crossval/comparators/nu_cavity_gates.py": Entry(
        "no_simulation",
        "pure-numpy Pozar spectrum, exact-lattice prediction, allowance, "
        "windows and gates for cv24 -- no rfx import at all"),
    "validation/crossval/22_dispersive_slab_fresnel.py": Entry(
        "no_simulation",
        "cv22 dispersive-slab case: drives cv04's low-level rig (Grid, "
        "init_tfsf, update_e_debye/lorentz) directly under a main guard -- "
        "no Simulation() call; the documented add_material path is stated "
        "as NOT exercised in its manifest entry"),
    "validation/crossval/comparators/cv22_dispersive_gates.py": Entry(
        "no_simulation",
        "pure-numpy windows, falsifiers and TMM/ADE evaluation for cv22 -- "
        "no rfx Simulation"),
    "validation/crossval/comparators/dispersive_eps.py": Entry(
        "no_simulation",
        "pure-numpy Debye/Lorentz/Drude eps(f) and the rfx->Meep material "
        "mapping (unit-tested to 1e-9 before any FDTD) -- no rfx import"),
    "validation/crossval/23_lossy_slab_fresnel.py": Entry(
        "no_solve",
        "cv23 lossy-slab case: two arms build a Simulation through the "
        "documented add_material(sigma=) path and assemble its material "
        "arrays (asserted bit-identical to the direct construction) but time "
        "stepping is the low-level rig's -- Simulation.run() is never called"),
    "validation/crossval/comparators/cv23_lossy_gates.py": Entry(
        "no_simulation",
        "pure-numpy windows, falsifiers and TMM evaluation for cv23 (R, T and "
        "absorption A) -- no rfx Simulation"),
    "validation/crossval/comparators/slab_rig.py": Entry(
        "no_simulation",
        "shared quasi-1-D TFSF slab rig helpers (record-length derivation, "
        "tail witness, envelope fit) factored out of cv22 -- no Simulation()"),
    "validation/crossval/comparators/ring_mode_judge.py": Entry(
        "no_simulation",
        "plain numpy/scipy mode-list comparator for cv02 (#812) -- compares "
        "two lists of extracted modes, no rfx import at all"),
    "validation/crossval/comparators/spectral_features.py": Entry(
        "no_simulation",
        "pure-numpy sub-bin spectral-feature estimators shared by cv06b/cv07 "
        "and the Palace referee producers (#812 P3) -- no rfx import at all"),
    "validation/crossval/comparators/patch_mode_identification.py": Entry(
        "no_simulation",
        "pure-math patch cavity mode identification (#812) -- closed-form "
        "TM_mn0 spectrum plus a frequency-list assignment; no rfx import at "
        "all"),
    "validation/crossval/comparators/fringe_gate.py": Entry(
        "no_simulation",
        "pure numpy/scipy fringe-extremum comparator for cv04's etalon R(f) "
        "(issue #812) -- no rfx import at all"),
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
    "validation/research/portgrid/__init__.py": Entry(
        "no_simulation",
        "package docstring pointing at the SPEC-02 predeclaration -- no "
        "code, no rfx import at all"),
    "validation/research/portgrid/certificate.py": Entry(
        "no_simulation",
        "M0 dissipativity-certificate calculator over hand-assembled "
        "descriptor-system matrices -- no rfx import at all"),
    "validation/research/portgrid/fig9_extract.py": Entry(
        "no_simulation",
        "vector-data extraction of the arXiv:1606.08761 Fig. 9 curves from "
        "the PDF's own path data -- no rfx import at all"),
    "validation/research/portgrid/m1_energy_audit.py": Entry(
        "no_simulation",
        "drives the portgrid.sim2d prototype's own lax.scan stepper "
        "directly -- no rfx Simulation is constructed by this script"),
    "validation/research/portgrid/m1_reflection.py": Entry(
        "no_simulation",
        "drives the portgrid.sim2d prototype's own lax.scan stepper "
        "directly -- no rfx Simulation is constructed by this script"),
    "validation/research/portgrid/m1b_retry.py": Entry(
        "no_simulation",
        "drives the portgrid.sim2d prototype's own lax.scan stepper "
        "directly -- no rfx Simulation is constructed by this script"),
    "validation/research/portgrid/operators.py": Entry(
        "no_simulation",
        "M0 interpolation/restriction operator library over plain arrays -- "
        "no rfx import at all"),
    "validation/research/portgrid/sim2d.py": Entry(
        "no_simulation",
        "hand-rolled two-region 2-D TEz subgridding prototype (its own "
        "lax.scan stepper) -- no rfx Simulation is constructed by this "
        "script; the whole point of this research lane is to NOT go "
        "through rfx's own Yee update"),
    "validation/research/portgrid/test_portgrid_m0.py": Entry(
        "no_simulation",
        "pytest battery over the portgrid.certificate/operators modules -- "
        "no rfx Simulation is constructed by this script"),
    "validation/research/portgrid/test_portgrid_m1.py": Entry(
        "no_simulation",
        "pytest battery over the portgrid.sim2d prototype -- no rfx "
        "Simulation is constructed by this script"),
    "validation/research/portgrid/test_portgrid_m1b_retry.py": Entry(
        "no_simulation",
        "pytest battery over the portgrid.m1b_retry prototype -- no rfx "
        "Simulation is constructed by this script"),

    # SPEC-01 multiband-NU witness package (#780). These are a witness
    # LIBRARY plus per-witness measurement drivers, not demo examples; the
    # thirteen below never construct a `Simulation` (AST-verified). Ten of
    # them drive the non-uniform kernels through
    # `rfx.nonuniform.make_nonuniform_grid` / `run_nonuniform` directly
    # (the design note requires explicit profile vectors, bypassing the
    # `auto_config` builders so a solver property is never confounded with
    # a builder defect), and three are pure numpy/analysis.
    "validation/research/multiband_nu/__init__.py": Entry(
        "no_simulation", "package marker -- empty file"),
    "validation/research/multiband_nu/analytic_dispersion.py": Entry(
        "no_simulation",
        "exact discrete leapfrog eigenfrequency of an empty PEC box on the "
        "rfx 1-D operators, solved in plain numpy (the W4R3 design and "
        "fixture-validity model) -- no rfx import at all"),
    "validation/research/multiband_nu/chain_model.py": Entry(
        "no_simulation",
        "exact discrete 1-D scattering chain solved in plain numpy (the "
        "F-S2/F-S3 window model) -- no rfx import at all"),
    "validation/research/multiband_nu/fixtures.py": Entry(
        "no_simulation",
        "explicit dz/dx profile-vector builders (numpy) plus the P-C "
        "geometry constants -- constructs no Simulation"),
    "validation/research/multiband_nu/harness.py": Entry(
        "no_simulation",
        "builds NonUniformGrid/MaterialArrays through "
        "rfx.nonuniform.make_nonuniform_grid and steps the kernels "
        "directly -- no Simulation object exists"),
    "validation/research/multiband_nu/predeclare_windows.py": Entry(
        "no_simulation",
        "freezes the F-S2/F-S3 windows from chain_model into "
        "results/predeclared_windows.json -- functional grid path only"),
    "validation/research/multiband_nu/remis_energy.py": Entry(
        "no_simulation",
        "the Remis-class dual-cell energy functional and its SBP "
        "adjointness check, over a NonUniformGrid -- no Simulation"),
    "validation/research/multiband_nu/revert_proof.py": Entry(
        "no_simulation",
        "gate-2 defect-injection proof on the functional grid/kernel "
        "path -- no Simulation"),
    "validation/research/multiband_nu/w1_energy_drift.py": Entry(
        "no_simulation",
        "F-S1 energy-audit driver on the functional grid/kernel path "
        "-- no Simulation"),
    "validation/research/multiband_nu/w2_w3_reflection.py": Entry(
        "no_simulation",
        "F-S2/F-S3 two-run-differencing driver on the functional "
        "grid/kernel path -- no Simulation"),
    "validation/research/multiband_nu/w4r2_analytic_cavity.py": Entry(
        "no_simulation",
        "F-S4 analytic-cavity ladder (empty PEC box, no geometry to "
        "rasterize) driven through the functional grid/kernel path "
        "-- no Simulation"),
    "validation/research/multiband_nu/w4r3_zdominant_cavity.py": Entry(
        "no_simulation",
        "F-S4 z-dominant analytic-cavity ladder and its grading-side "
        "revert-proof (empty PEC box, no geometry to rasterize) driven "
        "through the functional grid/kernel path -- no Simulation"),
    "validation/research/multiband_nu/w5_ad_consistency.py": Entry(
        "no_simulation",
        "F-S5 jax.grad-vs-FD check over an explicit profile vector on "
        "the functional grid/kernel path -- no Simulation"),

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
        "audited",
        "`build_rfx_sim(do_gain=, two_plane=)` returns (sim, patch_shape, "
        "geom) with no solve call (separated from run_rfx() for the #740 "
        "review so the wall-plane tests exercise the production toggle); "
        "run_rfx() consumes it and solves",
        (Builder("build_rfx_sim", 0, (_v("default", do_gain=False),)),)),
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

    "validation/research/cpml_pole_pad/localize_636.py": Entry(
        "no_simulation",
        "issue #636 mode-localization analysis: reads committed field dumps "
        "and does linear algebra -- constructs no Simulation (AST-verified)"),
    "validation/research/issue683_flip_acceptance.py": Entry(
        "no_simulation",
        "issue #683 flip-acceptance harness: imports `build`/`run_pre` from "
        "issue683_sampling_order_decision.py and drives them -- the script's "
        "own body constructs no Simulation (AST-verified)"),
    "validation/research/cpml_pole_pad/eigen_scan_636.py": Entry(
        "no_solve",
        "issue #636 frozen-coefficient von Neumann scan: builds a Simulation "
        "in `fixture_dt()` ONLY to read the lock-test grid's dt, then works "
        "on hand-assembled update matrices -- no solve entrypoint anywhere"),
    "validation/research/cpml_pole_pad/finite_op_636.py": Entry(
        "no_solve",
        "issue #636 finite-operator spectral radius: same pattern -- "
        "`fixture_dt()` builds a Simulation only for the grid's dt, the "
        "operator is assembled by hand and never solved"),
    "validation/research/cpml_pole_pad/factorial_636.py": Entry(
        "builder_fused_with_solve",
        "issue #636 CFS-alpha factorial: `vacuum_floor()` (and `run_cell()` "
        "via `build_sim`) construct and call .run(...) for the same cell"),
    # ---- audited (23): builder is separable from solve ----
    "validation/research/issue683_sampling_order_decision.py": Entry(
        "audited",
        "issue #683 sampling-order decision harness: `build(nu, r_load, "
        "boundary)` returns Simulation with no solve call (the solve lives "
        "in `run_pre()`/`run_post()`); the flip-acceptance script "
        "(issue683_flip_acceptance.py) imports this same builder",
        (Builder("build", None, (_v("nu-matched", nu=True, r_load=50.0),)),)),
    "validation/research/issue764_wireport_norm_falsifiers.py": Entry(
        "audited",
        "issue #764 falsifier battery: `build_fix_a()` returns Simulation "
        "with no solve call (the solve lives in `run_nu()`); the machine "
        "check rejected the earlier builder_fused_with_solve label for "
        "exactly this separability",
        (Builder("build_fix_a", None, (_v("short", load="short"),)),)),
    "validation/research/issue770_offdiag_adjudication.py": Entry(
        "audited",
        "`build_fix_t(nu, drive)` returns the canonical-THRU Simulation "
        "with no solve call; the adjudication arms run it separately "
        "(pre-declared falsifiers live in the same file).",
        (Builder("build_fix_t", None, (
            _v("uniform-both", nu=False, drive=None),
            _v("nu-drive0", nu=True, drive=0))),)),
    "validation/research/thru_feedpost_deembed.py": Entry(
        "audited",
        "thru-deembed attempt-1 harness: `build_thru(pulse)` returns the "
        "battery-verbatim THRU Simulation with no solve call (the solve "
        "lives in `run_thru`). Classification added in the attempt-3 "
        "branch (agent/thru-deembed-r3); the attempt-1/2 branches omitted "
        "their harnesses from this table (the same base-branch omission "
        "class the #770 lane recorded for the #683 harnesses).",
        (Builder("build_thru", None, (
            _v_from("band-pulse",
                    lambda m: dict(pulse=m.GaussianPulse(**m.BAND_PULSE))),
            _v_from("insitu-refplane",
                    lambda m: dict(pulse=m.GaussianPulse(**m.EXTRACT_PULSE),
                                   reference_plane_cells=10)))),)),
    "validation/research/thru_feedpost_joint_extraction.py": Entry(
        "no_simulation",
        "thru-deembed attempt-2 harness: constructs no Simulation of its "
        "own -- it imports `run_thru`/`build_thru` from "
        "thru_feedpost_deembed (whose builder IS audited above). "
        "Classification added in the attempt-3 branch."),
    "validation/research/thru_feedpost_twoseg_extraction.py": Entry(
        "audited",
        "thru-deembed attempt-3 harness: `build_singlepost(pulse, "
        "reference_plane_cells)` returns the single-post 1-port fixture "
        "Simulation with no solve call; the measurement arms drive it "
        "separately (pre-declared falsifiers in "
        "docs/design_notes/thru_feedpost_twoseg_predeclaration.md).",
        (Builder("build_singlepost", None, (
            _v_from("refplane-n10",
                    lambda m: dict(pulse=m.GaussianPulse(**m.EXTRACT_PULSE),
                                   reference_plane_cells=10)),)),)),
    "validation/research/thru_feedpost_junction_windows.py": Entry(
        "no_simulation",
        "thru-deembed attempt-4 harness (closing lane): constructs no "
        "Simulation of its own -- it imports `build_thru`/"
        "`build_singlepost`/`run_thru` from the audited attempt-1/3 "
        "harnesses and changes only the pre-declared window constants "
        "(docs/design_notes/"
        "thru_feedpost_junction_windows_predeclaration.md)."),
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
    "validation/crossval/24_nu_rect_cavity_pozar.py": Entry(
        "audited",
        "`build_cavity(lane, dxy, dz_profile)` returns Simulation with no "
        "solve call; main() drives it once per arm (uniform / graded / "
        "uniform-fine) through run_arm()",
        (Builder("build_cavity", None, (
            _v_from("uniform", lambda m: dict(
                lane="uniform", dxy=m.G.DX_COARSE, dz_profile=m.G.PROFILES["uniform"])),
            _v_from("single_band", lambda m: dict(
                lane="nonuniform", dxy=m.G.DX_COARSE, dz_profile=m.G.PROFILES["single_band"])),
            _v_from("multi_band", lambda m: dict(
                lane="nonuniform", dxy=m.G.DX_COARSE, dz_profile=m.G.PROFILES["multi_band"])),
            _v_from("uniform_fine", lambda m: dict(
                lane="uniform", dxy=m.G.DZ_FINE, dz_profile=m.G.PROFILES["uniform_fine"])),
        )),)),
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
    "validation/research/convergence_floor/d2_triples.py": Entry(
        "no_simulation",
        "successive-triple order arithmetic on committed JSON; no "
        "Simulation"),
    "validation/research/convergence_floor/d2_retake.py": Entry(
        "no_simulation",
        "re-applies the frozen D2 rule to the nine-rung ladder from "
        "committed JSON; no Simulation"),
    "validation/research/convergence_floor/d5_predeclare_and_run.py": Entry(
        "no_simulation",
        "drives fixture.measure(); constructs no Simulation of its own"),
    "validation/research/convergence_floor/d5_instrument_check.py": Entry(
        "no_simulation",
        "drives fixture.measure() and re-estimates the stored records; "
        "constructs no Simulation of its own"),
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

    # SPEC-01 multiband-NU W4 fixtures (#780) -- the only two scripts in
    # that package that construct a `Simulation` (P-C, the microstrip-class
    # multi-band resonator; preflight is deliberately ON there). Both
    # expose `build_sim(scale, dz_profile)` with no solve call, so the gate
    # pins exactly what this lane most needs pinned: declared-vs-realized
    # geometry on a multi-band graded mesh. Audited at the COARSEST declared
    # ladder scale of each script (cheapest build, same declared geometry at
    # every scale by construction -- the alignment invariant in the design
    # note's section 1), multiband profile, which is the arm under test.
    "validation/research/multiband_nu/w4_supraconvergence.py": Entry(
        "audited",
        "`build_sim(scale, dz_profile)` returns Simulation with no solve "
        "call (phase-1 W4 fixture; its ladder verdict was INCONCLUSIVE and "
        "the script is kept as the recorded diagnostic)",
        (Builder("build_sim", None, (
            _v_from("s1.5_multiband", lambda m: dict(
                scale=1.5, dz_profile=m.fx.pc_dz_profile_sym(1.5))),
        )),)),
    "validation/research/multiband_nu/w4r_port_supraconvergence.py": Entry(
        "audited",
        "`build_sim(scale, dz_profile, antisym=True)` returns Simulation "
        "with no solve call (W4R redesign: mode-selective anti-symmetric "
        "port pair, knife-edge-free PEC drawing)",
        (Builder("build_sim", None, (
            _v_from("s1.5_multiband", lambda m: dict(
                scale=1.5, dz_profile=m.fx.pc_dz_profile_sym(1.5))),
        )),)),
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

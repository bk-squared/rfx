"""Every Simulation attribute has a cell on every time-stepping path.

The table is ``tests/contracts/path_disposition.py``; its docstring says what
the cells mean. This file checks only that the table is complete: a new
attribute, a new lane of ``_dispatch_plan``, a new function that reaches a
time-stepping kernel, or a new function that steps fields, fails here until
someone decides and records how each path treats it. The cells' truth is
checked by ``tests/unit/runners/test_path_disposition_cells.py``.

It also checks that lane admission (``rfx/runners/_admission.py``) enforces
this table: every input row has a detector, and a lane admits exactly the
rows whose cell on it is ``carries`` or ``ignorable``.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

from rfx import Simulation
from rfx.runners import _admission as A
from tests.contracts import path_disposition as T

ROOT = Path(__file__).resolve().parents[2]

# Functions that reach a kernel without _dispatch_plan, or are part of a lane.
# A value is a column of the table, "internal to <columns>" or
# "grid-level: <why>" for a function that takes a Grid, not a Simulation.
KERNEL_CALLERS = {
    "rfx/api/_execute.py:_ExecuteMixin.run":
        "internal to run_uniform, run_nonuniform, run_subgridded, run_adi, run_distributed, run_distributed_nu",
    "rfx/api/_execute.py:_ExecuteMixin.forward":
        "internal to fwd_uniform, fwd_nonuniform, fwd_distributed_nu",
    "rfx/api/_execute.py:_ExecuteMixin._forward_from_materials":
        "internal to fwd_uniform, s_matrix_scan, mixed_s_matrix, topology_optimize",
    "rfx/api/_execute.py:_ExecuteMixin._forward_nonuniform_from_materials": "internal to fwd_nonuniform",
    "rfx/api/_execute.py:_ExecuteMixin._forward_distributed_nonuniform_from_materials":
        "internal to fwd_distributed_nu",
    "rfx/api/_execute.py:_ExecuteMixin._execute_distributed_nonuniform_from_materials":
        "internal to run_distributed_nu, fwd_distributed_nu",
    "rfx/api/_execute.py:_ExecuteMixin._run_adi_from_materials": "internal to run_adi, fwd_adi",
    "rfx/api/_execute.py:_ExecuteMixin._run_nonuniform": "internal to run_nonuniform",
    "rfx/api/_execute.py:_ExecuteMixin._run_subgridded": "internal to run_subgridded",
    "rfx/runners/nonuniform.py:run_nonuniform_path":
        "internal to run_nonuniform, fwd_nonuniform, waveguide_s_matrix",
    "rfx/runners/subgridded.py:_run_subgridded_once": "internal to run_subgridded",
    "rfx/runners/uniform.py:run_uniform": "internal to run_uniform",
    "rfx/probes/sparam_driver.py:compute_lumped_wire_s_matrix_via_scan": "s_matrix_scan",
    "rfx/probes/sparam_driver.py:_distributed_lumped_accumulators": "internal to run_distributed",
    "rfx/sparams/mixed.py:compute_mixed_s_matrix": "mixed_s_matrix",
    "rfx/topology.py:topology_optimize": "topology_optimize",
    "rfx/sparams/waveguide.py:_compute_waveguide_s_matrix_nu": "waveguide_s_matrix",
    **{f"rfx/sources/waveguide_port.py:{name}": "internal to waveguide_s_matrix" for name in (
        "extract_waveguide_s_matrix", "extract_waveguide_s_matrix_flux",
        "extract_waveguide_s_params_normalized", "extract_multimode_s_matrix",
        "extract_multimode_s_matrix_flux")},
    "rfx/sparams/coax.py:compute_coaxial_line_reflection": "coaxial_line_reflection",
    "rfx/sparams/coax.py:compute_coaxial_two_port": "coaxial_two_port",
    "rfx/sparams/coax.py:compute_coax_msl_transition": "coax_msl_transition",
    "rfx/differentiable_material_fit.py:differentiable_material_fit": "material_fit",
    "rfx/gpu.py:benchmark": "grid-level: times rfx.run on a Grid",
    "rfx/rcs.py:compute_rcs": "grid-level: takes a Grid and a TFSF/NTFF setup",
}

# Top-level functions that call a field update themselves: a new one is a new
# kernel, and its paths need cells.
FIELD_STEPPERS = {
    "rfx/adi.py:run_adi_2d": "internal to run_adi",
    "rfx/adi.py:run_adi_3d": "internal to run_adi",
    "rfx/nonuniform.py:_build_nu_scan": "internal to run_nonuniform, fwd_nonuniform",
    "rfx/runners/_distributed_common.py:update_h_nu_shmap": "internal to run_distributed_nu, fwd_distributed_nu",
    "rfx/runners/_distributed_common.py:update_e_nu_shmap": "internal to run_distributed_nu, fwd_distributed_nu",
    "rfx/runners/_distributed_common.py:_update_e_local_with_dispersion":
        "internal to run_distributed, run_distributed_nu, fwd_distributed_nu",
    "rfx/runners/distributed_v2.py:run_distributed": "internal to run_distributed",
    "rfx/simulation.py:_update_e_with_optional_dispersion":
        "internal to run_uniform, fwd_uniform, s_matrix_scan, mixed_s_matrix, topology_optimize, "
        "waveguide_s_matrix, coaxial_line_reflection, coaxial_two_port, coax_msl_transition, material_fit",
    # The one uniform-mesh step (S3-2): the scan step, the material sweep and the
    # grid-level eager lumped and wire extractors all build their step here.
    "rfx/stepping/uniform.py:make_uniform_step":
        "internal to run_uniform, fwd_uniform, s_matrix_scan, mixed_s_matrix, topology_optimize, "
        "waveguide_s_matrix, coaxial_line_reflection, coaxial_two_port, coax_msl_transition, material_fit, vmap_sweep_batched",
    "rfx/subgridding/disjoint_3d.py:_yee_step": "internal to run_subgridded",
    "rfx/subgridding/disjoint_3d.py:_yee_h_step": "internal to run_subgridded",
    "rfx/subgridding/disjoint_3d.py:_yee_e_step": "internal to run_subgridded",
    "rfx/subgridding/disjoint_3d.py:_apply_faces_maxwell_sat_from_snapshot": "internal to run_subgridded",
    "rfx/subgridding/jit_runner.py:_make_step_fn": "internal to run_subgridded",
    "rfx/subgridding/runner.py:run_subgridded": "grid-level: the non-jitted subgrid runner; rfx does not call it",
    "rfx/subgridding/sbp_sat_3d.py:_update_h_only": "internal to run_subgridded",
    "rfx/subgridding/sbp_sat_3d.py:_update_e_only": "internal to run_subgridded",
}

_KERNELS = {
    "_forward_from_materials", "_forward_nonuniform_from_materials",
    "_forward_distributed_nonuniform_from_materials", "_run_adi_from_materials",
    "_execute_distributed_nonuniform_from_materials",
    "_run_nonuniform", "_run_subgridded", "run_uniform", "run_nonuniform_path",
    "run_subgridded_path", "run_disjoint_stage2_path", "run_distributed",
    "run_nonuniform_distributed_pec", "run_subgridded_jit", "run_adi_2d",
    "run_adi_3d", "run_nonuniform",
}
_SIMULATION_KERNELS = {"run", "run_until_decay"}  # rfx.simulation, under any alias
_FIELD_UPDATES = {
    "update_e", "update_h", "update_e_nu", "update_h_nu", "update_he_fast",
    "update_e_fast", "update_h_fast", "update_e_aniso", "update_e_aniso_inv",
    "update_e_nu_aniso", "_update_e_local", "_update_h_local",
    "_update_e_local_nu", "_update_h_local_nu", "adi_step_2d", "adi_step_3d",
}


def _functions(tree):
    """Top-level functions and methods, with their qualified names."""
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield top.name, top
        elif isinstance(top, ast.ClassDef):
            for node in top.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{top.name}.{node.name}", node


def _aliases(tree):
    """Local names an import binds to a watched function ({alias: name}; an
    rfx.simulation kernel is named ``rfx.simulation.<name>``), and the names
    bound to the rfx.simulation module itself."""
    bound, modules = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if node.module == "rfx.simulation" and a.name in _SIMULATION_KERNELS:
                    bound[a.asname or a.name] = f"rfx.simulation.{a.name}"
                elif a.name in _KERNELS | _FIELD_UPDATES:
                    bound[a.asname or a.name] = a.name
                elif node.module == "rfx" and a.name == "simulation":
                    modules.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            modules |= {a.asname for a in node.names if a.name == "rfx.simulation" and a.asname}
    return bound, modules


def _scan(predicate):
    """``module:function`` of every function in rfx/ with a reference ``predicate`` accepts."""
    found = set()
    for path in sorted((ROOT / "rfx").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text())
        aliases = _aliases(tree)
        for name, fn in _functions(tree):
            if any(predicate(rel, node, aliases) for node in ast.walk(fn)
                   if isinstance(node, (ast.Name, ast.Attribute))
                   and isinstance(node.ctx, ast.Load)):
                found.add(f"{rel}:{name}")
    return found


def _referenced(node, aliases):
    """The watched name a reference resolves to, through import aliases."""
    bound, modules = aliases
    if isinstance(node, ast.Name):
        return bound.get(node.id, node.id)
    if (node.attr in _SIMULATION_KERNELS and isinstance(node.value, ast.Name)
            and node.value.id in modules):
        return f"rfx.simulation.{node.attr}"
    return node.attr


def _references_a_kernel(rel, node, aliases):
    name = _referenced(node, aliases)
    if name.startswith("rfx.simulation."):
        return rel != "rfx/simulation.py"
    return name in _KERNELS


def _calls_a_field_update(rel, node, aliases):
    return _referenced(node, aliases) in _FIELD_UPDATES


def _fresh_attributes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=1e-3)
    return set(vars(sim))


def test_every_simulation_attribute_has_a_row():
    new = sorted(_fresh_attributes() - set(T.TABLE))
    # A removed attribute cannot be dropped; a stale row is harmless.
    assert not new, (
        f"Simulation gained {new}: add a row to tests/contracts/path_disposition.py "
        "with a cell on every lane (carries, refuses, falls back, not reachable or "
        "ignorable) and a ROW_CLASS entry")
    assert set(T.ROW_CLASS) == set(T.TABLE), sorted(set(T.ROW_CLASS) ^ set(T.TABLE))


def test_every_row_has_a_cell_on_every_path():
    for attr, features in T.TABLE.items():
        assert features, attr
        for feature, row in features.items():
            if "*" in row:
                assert set(row) == {"*"}, (attr, feature)
                assert row["*"].kind == T.IGNORABLE, (
                    f"{attr}: only an ignorable cell may stand for every path")
                assert T.ROW_CLASS[attr] == T.BOOKKEEPING, attr
                continue
            assert set(row) == set(T.PATHS), (attr, feature, sorted(set(row) ^ set(T.PATHS)))
            for path in T.PATHS:
                assert T.cell(attr, feature, path).kind in T.KINDS


def test_every_cell_says_what_it_needs_to():
    for attr, features in T.TABLE.items():
        for feature, row in features.items():
            for path, c in row.items():
                where = f"{attr}/{feature}/{path}"
                assert c.kind in T.KINDS and c.kind != T.UNCLASSIFIED, where
                if c.kind in (T.NOT_REACHABLE, T.IGNORABLE):
                    assert c.note, f"{where}: say why"
                if c.kind == T.FALLS_BACK:
                    assert c.to in T.LANES, f"{where}: name the lane it falls back to"
                if c.wrong:
                    assert re.fullmatch(r"#\d+", c.wrong), where
                    assert c.kind in (T.CARRIES, T.REFUSES), where
                    assert c.note, f"{where}: say what happens today"
                if T.ROW_CLASS[attr] == T.PHYSICS:
                    assert c.kind != T.IGNORABLE, f"{where}: a physics input is never ignorable"
                if T.ROW_CLASS[attr] == T.OBSERVER and c.kind == T.IGNORABLE:
                    assert path.startswith("fwd_") or path in T.CALCULATORS, (
                        f"{where}: an observer is ignorable only on a forward() lane "
                        "or a calculator retaining its existing observer handling")
                if c.declared:
                    assert c.kind == T.REFUSES, where
                if T.executable(attr) and c.kind == T.REFUSES and not c.wrong:
                    assert c.raises, f"{where}: name a fragment of the refusal's message"


def test_lane_columns_are_the_lanes_dispatch_plan_selects():
    tree = ast.parse((ROOT / "rfx/api/_execute.py").read_text())
    lanes = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_DispatchPlan"):
            continue
        lane = [kw.value for kw in node.keywords if kw.arg == "lane"]
        assert not node.args and len(lane) == 1 and isinstance(lane[0], (ast.Constant, ast.IfExp)), (
            f"rfx/api/_execute.py:{node.lineno}: a _DispatchPlan whose lane is not a keyword "
            "constant or conditional constants; this test reads those lane keywords")
        values = (lane[0].body, lane[0].orelse) if isinstance(lane[0], ast.IfExp) else lane
        assert all(isinstance(value, ast.Constant) for value in values)
        lanes.update(value.value for value in values)
    routes = set(T.KERNEL_ROUTES)
    assert lanes == set(T.LANES) - routes, (
        f"_dispatch_plan lanes {sorted(lanes)} != table lanes {sorted(set(T.LANES) - routes)}: "
        "a new lane needs a column with a cell on every row")
    assert {token for token, _ in T.KERNEL_ROUTES.values()} <= lanes, T.KERNEL_ROUTES


def _check_registry(found, registry, what):
    assert found == set(registry), (
        f"unregistered {what}: {sorted(found - set(registry))}; registered but gone: "
        f"{sorted(set(registry) - found)}. Decide which column of "
        "tests/contracts/path_disposition.py each belongs to")
    for where, owner in registry.items():
        if owner.startswith("grid-level: "):
            continue
        columns = owner.removeprefix("internal to ").split(", ")
        assert set(columns) <= set(T.PATHS), (where, owner)


def test_every_function_that_reaches_a_kernel_is_registered():
    _check_registry(_scan(_references_a_kernel), KERNEL_CALLERS,
                    "functions that reach a time-stepping kernel")


def test_every_function_that_steps_fields_is_registered():
    _check_registry(_scan(_calls_a_field_update), FIELD_STEPPERS,
                    "functions that call a field update")


def _input_rows():
    return {(attr, feature) for attr, features in T.TABLE.items()
            if T.ROW_CLASS[attr] != T.BOOKKEEPING for feature in features}


def test_every_input_row_has_a_detector():
    """A row admission cannot see is an input every lane would drop again."""
    rows = _input_rows()
    assert set(A.DETECTORS) == rows, (
        f"rows without a detector in rfx/runners/_admission.py: {sorted(rows - set(A.DETECTORS))}; "
        f"detectors without a row here: {sorted(set(A.DETECTORS) - rows)}")
    assert set(A.ROW_WORDS) == rows, sorted(set(A.ROW_WORDS) ^ rows)
    assert set(A.CALL_GATES) <= rows, sorted(set(A.CALL_GATES) - rows)


def test_admission_admits_exactly_the_carried_cells():
    """One list of what a lane admits: ADMITS and the cell kinds agree."""
    assert A.LANES == T.LANES
    assert A.CALCULATORS == T.CALCULATORS
    disagree = []
    for attr, features in T.TABLE.items():
        for feature, row in features.items():
            key = (attr, feature)
            for lane in T.PATHS:
                if T.ROW_CLASS[attr] == T.BOOKKEEPING:
                    admitted = False   # bookkeeping is no input: no detector, no admission
                else:
                    admitted = row[lane].kind in (T.CARRIES, T.IGNORABLE)
                if (key in A.ADMITS[lane]) != admitted:
                    disagree.append(f"{attr}/{feature}/{lane}: table {T.cell(attr, feature, lane).kind}, "
                                    f"ADMITS {'has' if key in A.ADMITS[lane] else 'lacks'} it")
    assert not disagree, "\n".join(disagree)


def test_lane_gates_are_the_listed_ones():
    """A lane gate admits a row the table marks refused for some declarations
    only. Each one is listed in the table module, on a cell that is not
    carried there: no gate the table does not name, and no named one missing."""
    gated = {(lane, row) for lane, gates in A.LANE_GATES.items() for row in gates}
    assert gated == set(T.LANE_GATES), sorted(gated ^ set(T.LANE_GATES))
    for lane, (attr, feature) in gated:
        assert T.cell(attr, feature, lane).kind == T.REFUSES, (lane, attr, feature)


def test_subgridded_production_disposition_requires_explicit_opt_in():
    """#1465: a carried feature never grants production admission."""
    row = ("_refinement", "slab")
    assert T.cell(*row, "run_subgridded").kind == T.REFUSES
    assert row not in A.ADMITS["run_subgridded"]
    assert ("run_subgridded", row) in T.LANE_GATES
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .016), dx=.002, boundary="pec")
    sim.add_refinement((0., .012), ratio=2)
    assert row in A.refused(sim, "run_subgridded")
    for mode in ("research", "off"):
        sim._refinement["validation"] = mode
        assert row not in A.refused(sim, "run_subgridded")

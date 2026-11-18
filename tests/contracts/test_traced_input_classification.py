"""Independent public-surface coverage, evidence and pre-body refusal cells."""

import ast
import builtins
from collections import Counter
from dataclasses import fields, is_dataclass
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import types
import typing

import jax
import jax.numpy as jnp
import pytest

import rfx
from rfx import Simulation
from rfx._traced_input_table import DEFAULT_PARAMETERS, ROWS, state_counts
from rfx._traced_inputs import TracedStaticInput
from rfx.errors import RfxError

pytestmark = pytest.mark.source_scan

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = json.loads((ROOT / "docs/guides/api_symbol_inventory.json").read_text())

VALUE_CLASSES = {
    name: getattr(rfx, name) for name in (
        "Box", "OrientedBox", "Cylinder", "Sphere", "PolylineWire", "CurvedPatch",
        "MeshShape", "Via", "GaussianPulse", "ModulatedGaussian", "CWSource",
        "CustomWaveform", "DebyePole", "LorentzPole", "LorentzPoleMap",
        "RelativeStrength", "SnapshotSpec", "Grid", "NonUniformGrid",
    )
}
for name, module in (
    ("RingdownSpec", "rfx.ringdown"),
    ("NominalProfile", "rfx.model.graded_admission"),
    ("BoundarySpec", "rfx.boundaries.spec"),
    ("Boundary", "rfx.boundaries.spec"),
):
    VALUE_CLASSES[name] = getattr(importlib.import_module(module), name)

EXPORTED_FUNCTIONS = (
    "build_runtime_report", "build_scene_artifact", "export_geometry_sdf", "fidelity_report",
    "gradient_check", "optimize", "plan_simulation_mesh", "plot_geometry_3d",
    "progressive_optimize", "quick_convergence", "save_screenshot", "topology_optimize",
    "vmap_material_sweep",
)


def _parameters(fn, drop_first=False):
    parameters = list(inspect.signature(fn).parameters.values())
    return parameters[1:] if drop_first else parameters


def _annotation(fn, parameter):
    original = inspect.unwrap(fn)
    try:
        value = typing.get_type_hints(original).get(parameter.name, inspect.Parameter.empty)
    except (NameError, TypeError):
        value = parameter.annotation
    module = sys.modules.get(fn.__module__)
    if module is None:
        class_name = fn.__qualname__.split(".")[0]
        cls = VALUE_CLASSES.get(class_name)
        if cls is None:
            cls = getattr(rfx, class_name, None)
        module = sys.modules.get(getattr(cls, "__module__", None))
    namespace = dict(vars(module) if module is not None else {}, Simulation=Simulation, __builtins__=vars(builtins))
    while isinstance(value, (typing.ForwardRef, str)):
        if isinstance(value, str):
            value = typing.ForwardRef(value)
        try:
            value = eval(value.__forward_arg__, namespace)
        except NameError:
            return value
    return value


def _admits_number(annotation):
    if annotation in (typing.Any, object, inspect.Parameter.empty, int, float, complex, bool):
        return True
    if annotation in (str, bytes, type(None), Path):
        return False
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Literal:
        return any(isinstance(arg, (int, float, complex)) for arg in args)
    if origin in (typing.Union, types.UnionType):
        return any(_admits_number(arg) for arg in args)
    if args:
        return any(_admits_number(arg) for arg in args if arg is not Ellipsis)
    return True


def test_t1_every_parameter_and_value_field_has_exactly_one_state():
    keys = [(owner, parameter) for owner, parameter, _, _ in ROWS]
    assert len(keys) == len(set(keys)), "duplicate classifications"
    rows = {(o, p): (s, e) for o, p, s, e in ROWS}
    covered = set()
    annotation_classes = set()

    def visit(owner, fn, parameters):
        for parameter in parameters:
            key = (owner, parameter.name)
            covered.add(key)
            assert key in rows or (owner, "*") in rows, key
            state, evidence = rows.get(key, rows.get((owner, "*"), (None, None)))
            if state == "no_number":
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    assert evidence.startswith("VAR_KEYWORD exception:"), key
                else:
                    assert not _admits_number(_annotation(fn, parameter)), key
            if owner in EXPORTED_FUNCTIONS:
                assert state in {"traceable", "undecided", "no_number"}, key
            try:
                annotation = _annotation(fn, parameter)
            except NameError:
                continue
            pending = [annotation]
            while pending:
                member = pending.pop()
                pending.extend(typing.get_args(member))
                if isinstance(member, type) and member.__name__ in INVENTORY["rfx_exports"]:
                    if member is not Simulation and not member.__name__.endswith(("Result", "Report", "Certificate")):
                        annotation_classes.add(member.__name__)

    for name in ("__init__", *INVENTORY["simulation_methods"]):
        fn = getattr(Simulation, name)
        visit("Simulation." + name, fn, _parameters(fn, drop_first=True))
    assert annotation_classes <= set(VALUE_CLASSES), annotation_classes - set(VALUE_CLASSES)
    for name in EXPORTED_FUNCTIONS:
        fn = getattr(rfx, name)
        visit(name, fn, _parameters(fn))
    for name, cls in VALUE_CLASSES.items():
        constructor = cls.__new__ if issubclass(cls, tuple) and hasattr(cls, "_fields") else cls.__init__
        if is_dataclass(cls):
            class_fields = fields(cls)
            annotations = typing.get_type_hints(cls)
            for field in class_fields:
                key = (name, field.name)
                covered.add(key)
                assert key in rows, key
                if rows[key][0] == "no_number":
                    assert not _admits_number(annotations[field.name]), key
        else:
            visit(name, constructor, _parameters(constructor, drop_first=True))
    assert all(p == "*" or (o, p) in covered for o, p in keys), "stale table row"
    for owner, names in DEFAULT_PARAMETERS.items():
        fn = getattr(Simulation, owner.split(".")[1])
        assert names == tuple(p.name for p in _parameters(fn, drop_first=True))
    counts = Counter(rows.get((o, p), rows.get((o, "*")))[0].split(":", 1)[0] for o, p in covered)
    assert len(covered) == 634
    assert counts["static"] == 59
    assert counts["undecided"] == 486
    assert state_counts() == {state: counts[state] for state in (
        "traceable", "value_only", "static", "undecided", "no_number", "forwards")}


def test_t1_forwarding_rows_match_real_delegates():
    rows = {(o, p): s for o, p, s, _ in ROWS}
    text = inspect.getsource(inspect.unwrap(Simulation.compute_s_matrix))
    tree = ast.parse(text)
    delegates = {
        "Simulation." + node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
        and any(k.arg is None and isinstance(k.value, ast.Name) and k.value.id == "kwargs" for k in node.keywords)
    }
    assert set(rows["Simulation.compute_s_matrix", "kwargs"].removeprefix("forwards:").split(",")) == delegates
    assert "Simulation.compute_msl_s_matrix" in delegates
    assert rows["Simulation.artifact_report", "kwargs"] == "forwards:Simulation.mesh_intelligence_report,Simulation.preflight"
    for owner, parameter, state, _ in ROWS:
        if state.startswith("forwards:"):
            assert parameter in inspect.signature(getattr(Simulation, owner.split(".")[1])).parameters
            for target in state.removeprefix("forwards:").split(","):
                assert target.removeprefix("Simulation.") in INVENTORY["simulation_methods"]


def test_t2_evidence_exists():
    for owner, parameter, state, evidence in ROWS:
        if state == "static":
            assert re.match(r"S[123]:", evidence), (owner, parameter)
            assert re.search(r"(?:MESH|SHAPE|MATERIAL|WAVE)-\d+", evidence), (owner, parameter)
        elif state == "undecided":
            assert evidence.startswith("step 2 decides") or evidence == "static in the follow-up PR (A3-7)", (owner, parameter)
        elif state in {"traceable", "value_only"}:
            pins = re.findall(r"(tests/[\w/]+\.py)::(test_\w+)", evidence)
            assert pins, (owner, parameter)
            if state == "value_only":
                assert len(pins) >= 2
            for path, name in pins:
                tree = ast.parse((ROOT / path).read_text())
                assert any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree)), (path, name)


def test_t7_dispatch_reaches_delegate_admission(monkeypatch):
    test_t1_forwarding_rows_match_real_delegates()
    sim = object.__new__(Simulation)
    sim._precision = "float32"
    monkeypatch.setattr(sim, "s_matrix_lane", lambda **kw: "compute_msl_s_matrix")
    with pytest.raises(TracedStaticInput, match=r"Simulation.compute_msl_s_matrix\(n_steps="):
        jax.make_jaxpr(lambda t: sim.compute_s_matrix(method="compute_msl_s_matrix", n_steps=t))(1.0)


STATIC_CASES = (
    ('Simulation.add_tfsf_source', 'margin', 'scalar', 'Simulation.add_tfsf_source(margin='),
    ('Simulation.add_tfsf_source', 'direction_tol_deg', 'scalar', 'Simulation.add_tfsf_source(direction_tol_deg='),
    ('Simulation.add_tfsf_source', 'max_component', 'scalar', 'Simulation.add_tfsf_source(max_component='),
    ('Simulation.compute_coax_msl_transition', 'n_steps', 'scalar', 'Simulation.compute_coax_msl_transition(n_steps='),
    ('Simulation.compute_coax_msl_transition', 'num_periods', 'scalar', 'Simulation.compute_coax_msl_transition(num_periods='),
    ('Simulation.compute_coax_msl_transition', 'n_freqs', 'scalar', 'Simulation.compute_coax_msl_transition(n_freqs='),
    ('Simulation.compute_coax_msl_transition', 'probe_count', 'scalar', 'Simulation.compute_coax_msl_transition(probe_count='),
    ('Simulation.compute_coax_msl_transition', 'probe_start_cells', 'scalar', 'Simulation.compute_coax_msl_transition(probe_start_cells='),
    ('Simulation.compute_coax_msl_transition', 'probe_spacing_cells', 'scalar', 'Simulation.compute_coax_msl_transition(probe_spacing_cells='),
    ('Simulation.compute_coax_msl_transition', 'msl_probe_count', 'scalar', 'Simulation.compute_coax_msl_transition(msl_probe_count='),
    ('Simulation.compute_coax_msl_transition', 'msl_probe_start_cells', 'scalar', 'Simulation.compute_coax_msl_transition(msl_probe_start_cells='),
    ('Simulation.compute_coax_msl_transition', 'msl_probe_spacing_cells', 'scalar', 'Simulation.compute_coax_msl_transition(msl_probe_spacing_cells='),
    ('Simulation.compute_coax_msl_transition', 'cond_warn', 'scalar', 'Simulation.compute_coax_msl_transition(cond_warn='),
    ('Simulation.compute_coaxial_line_reflection', 'n_steps', 'scalar', 'Simulation.compute_coaxial_line_reflection(n_steps='),
    ('Simulation.compute_coaxial_line_reflection', 'n_freqs', 'scalar', 'Simulation.compute_coaxial_line_reflection(n_freqs='),
    ('Simulation.compute_coaxial_line_reflection', 'dut_offset_cells', 'scalar', 'Simulation.compute_coaxial_line_reflection(dut_offset_cells='),
    ('Simulation.compute_coaxial_line_reflection', 'probe_start_cells', 'scalar', 'Simulation.compute_coaxial_line_reflection(probe_start_cells='),
    ('Simulation.compute_coaxial_line_reflection', 'probe_spacing_cells', 'scalar', 'Simulation.compute_coaxial_line_reflection(probe_spacing_cells='),
    ('Simulation.compute_coaxial_two_port', 'n_steps', 'scalar', 'Simulation.compute_coaxial_two_port(n_steps='),
    ('Simulation.compute_coaxial_two_port', 'n_freqs', 'scalar', 'Simulation.compute_coaxial_two_port(n_freqs='),
    ('Simulation.compute_coaxial_two_port', 'probe_start_cells', 'scalar', 'Simulation.compute_coaxial_two_port(probe_start_cells='),
    ('Simulation.compute_coaxial_two_port', 'probe_spacing_cells', 'scalar', 'Simulation.compute_coaxial_two_port(probe_spacing_cells='),
    ('Simulation.compute_coaxial_two_port', 'cond_warn', 'scalar', 'Simulation.compute_coaxial_two_port(cond_warn='),
    ('Simulation.compute_mixed_s_matrix', 'n_steps', 'scalar', 'Simulation.compute_mixed_s_matrix(n_steps='),
    ('Simulation.compute_mixed_s_matrix', 'num_periods', 'scalar', 'Simulation.compute_mixed_s_matrix(num_periods='),
    ('Simulation.compute_mixed_s_matrix', 'n_freqs', 'scalar', 'Simulation.compute_mixed_s_matrix(n_freqs='),
    ('Simulation.compute_mixed_s_matrix', 'reciprocity_tol', 'scalar', 'Simulation.compute_mixed_s_matrix(reciprocity_tol='),
    ('Simulation.compute_msl_s_matrix', 'n_steps', 'scalar', 'Simulation.compute_msl_s_matrix(n_steps='),
    ('Simulation.compute_msl_s_matrix', 'num_periods', 'scalar', 'Simulation.compute_msl_s_matrix(num_periods='),
    ('Simulation.compute_msl_s_matrix', 'n_freqs', 'scalar', 'Simulation.compute_msl_s_matrix(n_freqs='),
    ('Simulation.compute_msl_s_matrix', 'checkpoint_segments', 'scalar', 'Simulation.compute_msl_s_matrix(checkpoint_segments='),
    ('Simulation.compute_waveguide_s_matrix', 'n_steps', 'scalar', 'Simulation.compute_waveguide_s_matrix(n_steps='),
    ('Simulation.compute_waveguide_s_matrix', 'num_periods', 'scalar', 'Simulation.compute_waveguide_s_matrix(num_periods='),
    ('Simulation.compute_waveguide_s_matrix', 'checkpoint_segments', 'scalar', 'Simulation.compute_waveguide_s_matrix(checkpoint_segments='),
    ('Simulation.forward', 'design_box', 'corners', 'Simulation.forward(design_box='),
    ('Simulation.forward', 'n_steps', 'scalar', 'Simulation.forward(n_steps='),
    ('Simulation.forward', 'num_periods', 'scalar', 'Simulation.forward(num_periods='),
    ('Simulation.forward', 'checkpoint_segments', 'scalar', 'Simulation.forward(checkpoint_segments='),
    ('Simulation.forward', 'n_warmup', 'scalar', 'Simulation.forward(n_warmup='),
    ('Simulation.run', 'n_steps', 'scalar', 'Simulation.run(n_steps='),
    ('Simulation.run', 'num_periods', 'scalar', 'Simulation.run(num_periods='),
    ('Simulation.run', 'until_decay', 'scalar', 'Simulation.run(until_decay='),
    ('Simulation.run', 'decay_check_interval', 'scalar', 'Simulation.run(decay_check_interval='),
    ('Simulation.run', 'decay_min_steps', 'scalar', 'Simulation.run(decay_min_steps='),
    ('Simulation.run', 'decay_max_steps', 'scalar', 'Simulation.run(decay_max_steps='),
    ('Simulation.run', 'decay_energy_consecutive', 'scalar', 'Simulation.run(decay_energy_consecutive='),
    ('Simulation.run', 'radiated_flux_box', 'corners', 'Simulation.run(radiated_flux_box='),
    ('Simulation.run', 'flux_env_checks', 'scalar', 'Simulation.run(flux_env_checks='),
    ('Simulation.run', 's_param_n_steps', 'scalar', 'Simulation.run(s_param_n_steps='),
    ('Simulation.run', 'conformal_min_weight', 'scalar', 'Simulation.run(conformal_min_weight='),
    ('RingdownSpec', 'window_start', 'scalar', 'RingdownSpec(window_start='),
    ('RingdownSpec', 'guard', 'scalar', 'RingdownSpec(guard='),
    ('RingdownSpec', 'witness_tol', 'scalar', 'RingdownSpec(witness_tol='),
    ('RingdownSpec', 'freq_max', 'scalar', 'RingdownSpec(freq_max='),
    ('RingdownSpec', 'split', 'scalar', 'RingdownSpec(split='),
    ('RingdownSpec', 'sv_rel', 'scalar', 'RingdownSpec(sv_rel='),
    ('RingdownSpec', 'unit_tol', 'scalar', 'RingdownSpec(unit_tol='),
    ('RingdownSpec', 'passivity_tol', 'scalar', 'RingdownSpec(passivity_tol='),
    ('RingdownSpec', 'source_off_tol', 'scalar', 'RingdownSpec(source_off_tol='),
)


def test_t3_cells_cover_every_static_row():
    assert {(o, p) for o, p, _, _ in STATIC_CASES} == {(o, p) for o, p, s, _ in ROWS if s == "static"}


def _sentinel_arguments(callable_, parameter, value):
    kwargs = {p.name: object() for p in inspect.signature(callable_).parameters.values()
              if p.default is inspect.Parameter.empty and p.kind not in (
                  inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
    kwargs[parameter] = value
    return kwargs


@pytest.mark.parametrize("owner,parameter,shape,message", STATIC_CASES)
@pytest.mark.parametrize("transform", ["make_jaxpr", "vmap", "grad", "jvp"])
def test_t3_static_refuses_before_body(owner, parameter, shape, message, transform):
    sim = object.__new__(Simulation)
    declarations = ("_geometry", "_ports", "_probes", "_thin_conductors", "_pinned_sheets",
                    "_waveguide_ports", "_msl_ports", "_coaxial_ports", "_floquet_ports",
                    "_flux_monitors", "_dft_planes", "_lumped_rlc", "_wall_losses")
    for name in declarations:
        setattr(sim, name, [])
    sim._materials = {}
    original_lists = {name: getattr(sim, name) for name in declarations}
    target = getattr(sim, owner.split(".")[1]) if owner.startswith("Simulation.") else VALUE_CLASSES[owner]

    def call(t):
        value = ((t, 0.0, 0.0), (1.0, 1.0, 1.0)) if shape == "corners" else (t, 0.0, 0.0) if shape == "tuple" else t
        target(**_sentinel_arguments(target, parameter, value))
        return t * 0

    with pytest.raises(TracedStaticInput) as caught:
        if transform == "make_jaxpr":
            jax.make_jaxpr(call)(1.0)
        elif transform == "vmap":
            jax.vmap(call)(jnp.ones(2))
        elif transform == "grad":
            jax.grad(call)(1.0)
        else:
            jax.jvp(call, (1.0,), (1.0,))
    assert isinstance(caught.value, RfxError)
    assert message in str(caught.value)
    assert "pass a Python number" in str(caught.value)
    for name in declarations:
        assert getattr(sim, name) is original_lists[name]
        assert getattr(sim, name) == []
    assert sim._materials == {}
    assert not hasattr(sim, "_traced_inputs")


def test_t3_value_field_inside_a_method_argument():
    from rfx.ringdown import RingdownSpec
    spec = object.__new__(RingdownSpec)
    sim = object.__new__(Simulation)

    def call(t):
        object.__setattr__(spec, "window_start", t)
        return sim.forward(n_steps=2, ringdown=spec)

    with pytest.raises(TracedStaticInput, match=r"Simulation.forward\(ringdown.window_start="):
        jax.make_jaxpr(call)(1.0)


def test_named_wire_radius_refusal_is_preserved():
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.001, boundary="pec")

    def declare(t):
        sim.add_port((.01, .01, .01), extent=.003, radius=t)
        return t

    with pytest.raises(NotImplementedError, match="wire-port radius must be a static declaration") as caught:
        jax.make_jaxpr(declare)(.0001)
    assert not isinstance(caught.value, TracedStaticInput)


@pytest.mark.parametrize("parameter,message", [
    ("x_position", "x_position must be a finite scalar"),
    ("reference_plane", "reference_plane must be a finite scalar"),
    ("probe_plane", "probe_plane must be a finite scalar"),
    ("y_range", "y_range must contain finite scalars"),
    ("z_range", "z_range must contain finite scalars"),
])
def test_named_waveguide_plane_refusals_are_preserved(parameter, message):
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.001, boundary="cpml", cpml_layers=2)

    def declare(t):
        kwargs = {"x_position": .005, parameter: (t, .015) if parameter.endswith("range") else t}
        sim.add_waveguide_port(**kwargs)
        return t

    with pytest.raises(ValueError, match=message) as caught:
        jax.make_jaxpr(declare)(.005)
    assert not isinstance(caught.value, TracedStaticInput)


@pytest.mark.parametrize("parameter,message", [
    ("n_steps", "Simulation._forward_from_materials(n_steps="),
    ("checkpoint_segments", "Simulation._forward_from_materials(checkpoint_segments="),
    ("conformal_min_weight", "Simulation._forward_from_materials(conformal_min_weight="),
])
@pytest.mark.parametrize("transform", ["make_jaxpr", "vmap", "grad", "jvp"])
def test_t3_private_materials_entry_refuses_before_staging(parameter, message, transform):
    sim = object.__new__(Simulation)
    target = sim._forward_from_materials

    def call(t):
        target(**_sentinel_arguments(target, parameter, t))
        return t * 0

    with pytest.raises(TracedStaticInput) as caught:
        if transform == "make_jaxpr":
            jax.make_jaxpr(call)(1.0)
        elif transform == "vmap":
            jax.vmap(call)(jnp.ones(2))
        elif transform == "grad":
            jax.grad(call)(1.0)
        else:
            jax.jvp(call, (1.0,), (1.0,))
    assert message in str(caught.value)
    assert isinstance(caught.value, RfxError)

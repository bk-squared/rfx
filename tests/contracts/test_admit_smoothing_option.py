"""Every admitted entry forwards its smoothing option to shared admission.

Direct scan and uniform-run refusals pin unsupported full-tensor inputs, while
supported direct scans retain finite results.
Subgridded and distributed entries refuse smoothing before shared admission.
"""

import ast
from collections import Counter, defaultdict
from pathlib import Path

import jax
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation, Sphere
from rfx.errors import RfxUnsupported
from rfx.model.smoothing_admission import FULL_TENSOR_MESSAGE
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
from rfx.runners import run_uniform


ADMIT_CALLS = {
    ("rfx/probes/sparam_driver.py", "compute_lumped_wire_s_matrix_via_scan"): ("passes", 1),
    ("rfx/runners/_admission.py", "admit_run_s_matrix"): ("passes", 1),
    ("rfx/runners/uniform.py", "run_uniform"): ("passes", 1),
    ("rfx/runners/uniform_forward.py", "_forward_from_materials"): ("passes", 1),
    ("rfx/runners/nonuniform.py", "run_nonuniform_path"): ("passes", 1),
    ("rfx/sparams/waveguide.py", "compute_waveguide_s_matrix"): ("passes", 2),
    ("rfx/runners/subgridded.py", "_run_subgridded_once"): ("no option", 1),
    ("rfx/subgridding/_notice.py", "require_experimental"): ("no option", 1),
    ("rfx/runners/distributed_v2.py", "run_distributed"): ("no option", 1),
    ("rfx/model/distributed_ports.py", "assemble"): ("no option", 1),
    ("rfx/sparams/coax.py", "compute_coaxial_line_reflection"): ("no option", 1),
    ("rfx/sparams/coax.py", "compute_coaxial_two_port"): ("no option", 1),
    ("rfx/sparams/coax.py", "compute_coax_msl_transition"): ("no option", 1),
    ("rfx/sparams/mixed.py", "compute_mixed_s_matrix"): ("no option", 1),
    ("rfx/topology.py", "topology_optimize"): ("no option", 1),
    ("rfx/vmap_sweep.py", "vmap_material_sweep"): ("no option", 1),
    ("rfx/differentiable_material_fit.py", "_require_supported_fixture"): ("no option", 1),
}
FREQS = [6e9, 8e9]
N_STEPS = 20


def _calls(tree, name, enclosing="<module>"):
    if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
        enclosing = tree.name
    if isinstance(tree, ast.Call):
        func = tree.func
        if ((isinstance(func, ast.Name) and func.id == name)
                or (isinstance(func, ast.Attribute) and func.attr == name)):
            yield enclosing, tree
    for child in ast.iter_child_nodes(tree):
        yield from _calls(child, name, enclosing)


def test_admit_call_census_and_smoothing_option():
    root = Path(__file__).resolve().parents[2]
    calls = defaultdict(list)
    scan_calls = defaultdict(list)
    for path in sorted((root / "rfx").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(root).as_posix()
        for function, call in _calls(tree, "admit"):
            calls[relative, function].append(call)
        for function, call in _calls(tree, "admit_run_s_matrix"):
            scan_calls[relative, function].append(call)
    assert Counter({key: len(value) for key, value in calls.items()}) == Counter(
        {key: count for key, (_, count) in ADMIT_CALLS.items()})
    for key, (disposition, _) in ADMIT_CALLS.items():
        if disposition != "passes":
            continue
        for call in calls[key]:
            run_args = next((kw.value for kw in call.keywords if kw.arg == "run_args"), None)
            assert isinstance(run_args, ast.Dict), (key, call.lineno)
            assert any(isinstance(k, ast.Constant) and k.value == "subpixel_smoothing"
                       for k in run_args.keys), (key, call.lineno)
    uniform_calls = scan_calls["rfx/runners/uniform.py", "run_uniform"]
    assert len(uniform_calls) == 1
    assert any(kw.arg == "subpixel_smoothing" for kw in uniform_calls[0].keywords)


def _scene(boundary, eps_r=4.0):
    sim = Simulation(freq_max=12e9, domain=(13e-3, 12e-3, 10e-3), dx=1e-3,
                     boundary=boundary, cpml_layers=6)
    for n, x in enumerate((4e-3, 8e-3)):
        sim.add_port(position=(x, 6e-3, 4e-3), component="ez", impedance=50.,
                     excite=n == 0, waveform=GaussianPulse(f0=6e9, bandwidth=0.8))
    sim.add_material("d", eps_r=eps_r)
    sim.add(Sphere(center=(6.07e-3, 6.13e-3, 4.21e-3), radius=1.55e-3), material="d")
    return sim


def _fail_before_step(*args, **kwargs):
    pytest.fail("unsupported smoothing reached _forward_from_materials")


def test_direct_scan_upml_full_tensor_refuses_before_step(monkeypatch):
    monkeypatch.setattr(Simulation, "_forward_from_materials", _fail_before_step)
    with pytest.raises(NotImplementedError) as error:
        compute_lumped_wire_s_matrix_via_scan(
            _scene("upml"), FREQS, n_steps=N_STEPS, subpixel_smoothing="full_tensor")
    assert str(error.value) == FULL_TENSOR_MESSAGE


def test_direct_scan_traced_full_tensor_refuses_before_step(monkeypatch):
    monkeypatch.setattr(Simulation, "_forward_from_materials", _fail_before_step)

    def scan(eps_r):
        return compute_lumped_wire_s_matrix_via_scan(
            _scene("cpml", eps_r), FREQS, n_steps=N_STEPS,
            subpixel_smoothing="full_tensor")[0]

    with pytest.raises(RfxUnsupported, match="with a traced permittivity"):
        jax.jit(scan)(4.0)


def test_direct_run_uniform_upml_full_tensor_refuses():
    sim = _scene("upml")
    grid = sim._build_grid()
    materials, debye, lorentz, pec, shapes, _, kerr = sim._assemble_materials(grid)
    with pytest.raises(NotImplementedError) as error:
        run_uniform(sim, n_steps=N_STEPS, grid=grid, base_materials=materials,
                    debye_spec=debye, lorentz_spec=lorentz, pec_mask=pec,
                    pec_shapes=shapes, kerr_chi3=kerr, compute_s_params=False,
                    subpixel_smoothing="full_tensor")
    assert str(error.value) == FULL_TENSOR_MESSAGE


@pytest.mark.parametrize("boundary,smoothing", [("cpml", "full_tensor"), ("upml", True)])
def test_supported_direct_scan_is_finite(boundary, smoothing):
    s_matrix, freqs = compute_lumped_wire_s_matrix_via_scan(
        _scene(boundary), FREQS, n_steps=N_STEPS, subpixel_smoothing=smoothing)
    assert np.asarray(s_matrix).shape == (2, 2, 2)
    assert np.isfinite(s_matrix).all()
    np.testing.assert_array_equal(freqs, FREQS)

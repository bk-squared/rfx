"""Small concrete runner-plan falsifiers; no FDTD field advancement."""
import copy
import importlib.util
import json
from pathlib import Path
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import SheetSpec, realized_pec_edge_masks
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import MaterialArrays
from rfx.grid import Grid
from rfx.probes.probes import DFTPlaneProbe
import rfx.simulation as engine

PATH = Path(__file__).resolve().parents[2] / "docs/research_notes/issue953/consumed_plan.py"
spec = importlib.util.spec_from_file_location("cv06b_consumed_plan", PATH)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)
BOX = ((4, 6), (4, 6), (3, 3))


def _call():
    grid = Grid(1e8, (6., 6., 6.), dx=1., cpml_layers=1, pec_faces={"z_lo"})
    trace = np.zeros(grid.shape, dtype=bool)
    trace[2:8, 2:5, 3] = True
    stub = np.zeros_like(trace)
    stub[4:7, 4:7, 3] = True
    sheets = (SheetSpec(2, 3, trace), SheetSpec(2, 3, stub))
    mats = MaterialArrays(np.full(grid.shape, 2.25, dtype=np.float32),
                          np.zeros(grid.shape, dtype=np.float32), np.ones(grid.shape, dtype=np.float32))
    mats.sigma[2, 3, 1] = .2
    sources = [engine.SourceSpec(2, 3, 1, "ez", np.array([.1, .2, 0., 0.], dtype=np.float32))]
    dft = DFTPlaneProbe(np.zeros((2, 3, 4), dtype=np.complex64), np.array([1e7, 2e7], dtype=np.float32),
                        "ez", 0, 3, 4, "rect", .25, (2, 5, 0, 4))
    return (grid, mats, 4), dict(boundary="cpml", pec_axes="", periodic=(False, False, False),
        sources=sources, probes=[engine.ProbeSpec(3, 3, 1, "ez")], dft_planes=[dft],
        pec_sheets=sheets, pec_edge_masks=tuple(np.asarray(m) for m in realized_pec_edge_masks(None, sheets=sheets)),
        field_dtype=jnp.float32)


def _fingerprint(args, kwargs):
    return plan.fingerprint_run_call(engine.run, args, kwargs, allowed_stub_box=BOX)


def test_signature_is_json_native_and_ignores_only_legitimate_stub_edges():
    args, kwargs = _call()
    baseline = _fingerprint(args, kwargs)
    variant = copy.deepcopy(kwargs)
    trace, stub = variant["pec_sheets"]
    narrowed = np.zeros_like(stub.footprint)
    narrowed[5:6, 4:7, 3] = True
    variant["pec_sheets"] = (trace, SheetSpec(2, 3, narrowed))
    variant["pec_edge_masks"] = tuple(np.asarray(m) for m in realized_pec_edge_masks(None, sheets=variant["pec_sheets"]))
    assert kwargs["pec_edge_masks"][1][6, 4, 3]
    assert not variant["pec_edge_masks"][1][6, 4, 3]  # junction Ey edge may change
    candidate = _fingerprint(args, variant)
    assert baseline["observed"]["pec_full"] != candidate["observed"]["pec_full"]
    plan.compare_run_plans(json.loads(json.dumps(baseline)), candidate)


@pytest.mark.parametrize("defect, expected", [
    ("waveform", "sources"), ("crop", "dft_planes"), ("frequency", "dft_planes"),
    ("material", "materials_after_load"), ("kappa", "grid.kappa_max"),
    ("outside_pec", "pec_outside_stub"), ("normal_pec", "pec_outside_stub"),
    ("outside_sheet", "sheets_outside_stub"),
])
def test_actual_consumed_changes_are_rejected(defect, expected):
    args, kwargs = _call()
    baseline = _fingerprint(args, kwargs)
    args, kwargs = copy.deepcopy((args, kwargs))
    if defect == "waveform":
        kwargs["sources"][0] = kwargs["sources"][0]._replace(waveform=kwargs["sources"][0].waveform * 2)
    elif defect == "crop":
        kwargs["dft_planes"][0] = kwargs["dft_planes"][0]._replace(region=(3, 6, 0, 4))
    elif defect == "frequency":
        kwargs["dft_planes"][0] = kwargs["dft_planes"][0]._replace(freqs=np.array([1e7, 3e7], dtype=np.float32))
    elif defect == "material":
        args[1].eps_r[1, 1, 1] *= 1.1
    elif defect == "kappa":
        args[0].kappa_max = 3.
    elif defect in ("outside_pec", "normal_pec"):
        component, cell = (0, (1, 1, 1)) if defect == "outside_pec" else (2, (5, 5, 3))
        kwargs["pec_edge_masks"][component][cell] = True
    else:
        # Bound final masks are still unchanged: an extra declared outside
        # footprint is nevertheless preserved by the provenance check.
        trace, stub = kwargs["pec_sheets"]
        damaged = np.array(stub.footprint, copy=True)
        damaged[1, 1, 3] = True
        kwargs["pec_sheets"] = (trace, SheetSpec(2, 3, damaged))
    with pytest.raises(ValueError, match=expected):
        plan.compare_run_plans(baseline, _fingerprint(args, kwargs))


@pytest.mark.parametrize("channel", ["mag_sources", "kerr_chi3", "sheet_impedance"])
def test_unknown_physics_cannot_be_silently_omitted(channel):
    args, kwargs = _call()
    kwargs[channel] = object()
    with pytest.raises(ValueError, match=channel):
        _fingerprint(args, kwargs)


def test_actual_uniform_runner_entry_is_fingerprinted_without_stepping(monkeypatch):
    dx = .001
    sim = Simulation(freq_max=1e9, domain=(24*dx, 12*dx, 8*dx), dx=dx, cpml_layers=1,
                     boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("sub", eps_r=3.)
    sim.add(Box((0., 0., 0.), (24*dx, 12*dx, 3*dx)), material="sub")
    sim.add(Box((0., 4*dx, 3*dx), (24*dx, 6*dx, 3*dx)), material="pec")
    sim.add(Box((11*dx, 6*dx, 3*dx), (13*dx, 9*dx, 3*dx)), material="pec")
    for x, direction in ((3*dx, "+x"), (21*dx, "-x")):
        sim.add_msl_port(position=(x, 5*dx, 0.), width=2*dx, height=3*dx, direction=direction,
                         n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    original = engine.run
    captured = []

    class StopBeforeFields(BaseException):
        pass

    def capture(*args, **kwargs):
        captured.append(plan.fingerprint_run_call(original, args, kwargs,
                        allowed_stub_box=((12, 14), (7, 10), (3, 3))))
        raise StopBeforeFields

    monkeypatch.setattr(engine, "run", capture)
    with warnings.catch_warnings(), pytest.raises(StopBeforeFields):
        warnings.simplefilter("ignore")
        sim.compute_msl_s_matrix(freqs=np.array([.8e9, 1e9]), n_steps=1, enforce_passivity=False)
    assert len(captured) == 1
    signature = captured[0]["invariants"]
    assert signature["sources"] and signature["point_probes"] and signature["dft_planes"]
    assert signature["materials_after_load"]["sigma"]["dtype"] == np.dtype("float32").str
    assert signature["n_steps"] == 1

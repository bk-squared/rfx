"""Preflight decides PEC the way the assembly does (#1961, cause A).

The assembler routes a geometry entry to PEC when its RESOLVED material has
``sigma >= sim._PEC_SIGMA_THRESHOLD`` (1e6 S/m), whatever the material is
called. Preflight checks used to key on the literal name "pec", so a
conductor registered as "copper" (5.8e7 S/m) was PEC in the solve and
absent in the checks: the trace far-end refusal stayed silent and the port
end ran undriven. Every check now reads ``rfx.model.materials.assembled_as_pec``.

Rows: "pec" and "copper" are PEC in the assembly, so each check gives the
same outcome for both; "lossy" (1e3 S/m) is not PEC and each check treats
it as a dielectric or as no conductor at all. Expected outcomes are written
out per check, not computed with the predicate.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

MATERIALS = ("pec", "copper", "lossy")


def _register(sim, *, eps_r=1.0):
    sim.add_material("copper", eps_r=eps_r, sigma=5.8e7)
    sim.add_material("lossy", eps_r=eps_r, sigma=1e3)


def _codes(sim, code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight(strict=False)
    return [str(i) for i in report if getattr(i, "code", None) == code]


def _trace_sim(material):
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.001, boundary="pec")
    _register(sim)
    sim.add(Box((.003, .007, .003), (.0117, .011, .003)), material=material)
    sim.add_port((.0117, .009, .002), component="ez")
    return sim


def test_anchor_copper_sheet_is_assembled_as_the_pec_sheet():
    """Physical anchor, independent of the predicate: the assembly realizes
    the "copper" trace as the same PEC sheet, edge for edge, as "pec"."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    edges = {}
    for material in ("pec", "copper"):
        sim = _trace_sim(material)
        grid = sim._build_grid()
        sheets: list = []
        pec_mask = sim._assemble_materials(grid, pec_sheets=sheets)[3]
        assert len(sheets) == 1, material
        edges[material] = [np.asarray(m) for m in realized_pec_edge_masks(pec_mask, sheets=sheets)]
    assert sum(int(m.sum()) for m in edges["pec"]) > 0
    for a, b in zip(edges["pec"], edges["copper"], strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("material,expected", [("pec", 1), ("copper", 1), ("lossy", 0)])
def test_trace_far_end_refusal(material, expected):
    from rfx.preflight.ports import trace_far_end_findings
    assert len(trace_far_end_findings(_trace_sim(material))) == expected


def _floating_sim():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.006), dx=0.5e-3,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("substrate", eps_r=4.4)
    return sim


@pytest.mark.parametrize("material,expected", [("pec", 0), ("copper", 0), ("lossy", 1)])
def test_floating_port_adjacent_pin(material, expected):
    """A pin one cell under the port along its component axis is a PEC pin
    for "pec" and "copper"; a lossy pin leaves the port floating."""
    sim = _floating_sim()
    _register(sim)
    sim.add(Box((0.004, 0.004, 0.001), (0.016, 0.016, 0.004)), material="substrate")
    sim.add(Box((0.00973, 0.00961, 0.00113), (0.01047, 0.01029, 0.00207)), material=material)
    sim.add_port(position=(0.0101, 0.0099, 0.0025), component="ez", impedance=50.0)
    assert len(_codes(sim, "floating_port")) == expected


@pytest.mark.parametrize("material,expected", [("pec", 0), ("copper", 0), ("lossy", 1)])
def test_floating_port_enclosing_body_skips_conductors(material, expected):
    """A conductor enclosing the port is not the enclosing dielectric, even
    with eps_r 3 declared; a lossy eps_r 3 body is a dielectric."""
    sim = _floating_sim()
    _register(sim, eps_r=3.0)
    sim.add(Box((0.00413, 0.00427, 0.00113), (0.01571, 0.01583, 0.00387)), material=material)
    sim.add_port(position=(0.0101, 0.0099, 0.0025), component="ez", impedance=50.0)
    assert len(_codes(sim, "floating_port")) == expected


@pytest.mark.parametrize("material,expected", [("pec", 1), ("copper", 1), ("lossy", 0)])
def test_pec_faces_with_finite_pec(material, expected):
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.001,
                     boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
                     cpml_layers=4)
    _register(sim)
    sim.add(Box((.0063, .0071, .0072), (.0131, .0122, .0094)), material=material)
    assert len(_codes(sim, "pec_faces_finite_pec")) == expected


@pytest.mark.parametrize("material,expected", [("pec", 1), ("copper", 1), ("lossy", 0)])
def test_mesh_pec_gap(material, expected):
    """Two bodies 1.6 cells apart along x: a gap between PEC structures for
    "pec" and "copper", no PEC gap for "lossy"."""
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.001, boundary="pec")
    _register(sim)
    sim.add(Box((.0043, .0061, .0072), (.0081, .0122, .0094)), material=material)
    sim.add(Box((.0097, .0061, .0072), (.0139, .0122, .0094)), material=material)
    gaps = [m for m in _codes(sim, "mesh_resolution") if "Gap between PEC structures" in m]
    assert len(gaps) == expected


def _reflector(material, *, resolver):
    from rfx.preflight.msl_reflector import msl_nearest_downstream_reflector
    geometry = [SimpleNamespace(shape=Box((.0503, .0402, .0041), (.0611, .0571, .0052)),
                                material_name=material)]
    kw = {}
    if resolver:
        sim = Simulation(freq_max=10e9, domain=(.12, .10, .02), dx=.001, boundary="pec")
        _register(sim)
        kw = dict(resolve_material=sim._resolve_material,
                  pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD)
    distance, _, _ = msl_nearest_downstream_reflector(
        geometry, x_probe=.024, x_feed=.024, y_feed=.0483, w_trace=.010,
        dx=.001, domain_y=.100, direction="+x", **kw)
    return distance


@pytest.mark.parametrize("resolver", [True, False], ids=["resolver", "library"])
@pytest.mark.parametrize("material,expected", [("pec", .0263), ("copper", .0263), ("lossy", np.inf)])
def test_msl_reflector_conductor(material, expected, resolver):
    assert _reflector(material, resolver=resolver) == pytest.approx(expected)


@pytest.mark.parametrize("material,expected", [("pec", 1), ("copper", 1), ("lossy", 0)])
def test_ntff_small_ground_plane_census(material, expected):
    """A 60 x 55 mm plate (0.5 lambda at 2.4 GHz) under a source is a small
    PEC ground plane for "pec" and "copper"; a lossy plate is not one."""
    sim = Simulation(freq_max=3e9, domain=(.14, .135, .05), boundary="cpml",
                     cpml_layers=4, dx=2.5e-3)
    _register(sim)
    sim.add(Box((.0400, .0400, .0100), (.1000, .0950, .01435)), material=material)
    sim.add_port((.0613, .0671, .0156), "ez")
    sim.add_ntff_box(corner_lo=(.012, .012, .004), corner_hi=(.128, .123, .038),
                     freqs=(2.4e9,))
    assert len(_codes(sim, "ntff_small_ground_plane")) == expected


def test_no_name_keyed_pec_test_left_in_preflight():
    root = Path(__file__).resolve().parents[2] / "rfx" / "preflight"
    pattern = re.compile(
        r"""(material_name|mat_name)\)?(\.lower\(\))?\s*(==|!=)\s*["']pec["']""")
    hits = [f"{p.name}:{n}" for p in sorted(root.rglob("*.py"))
            for n, line in enumerate(p.read_text().splitlines(), 1)
            if pattern.search(line)]
    assert hits == []

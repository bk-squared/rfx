"""A port declared on a wire or sheet at a half node, and where it lands (#1295, #1342).

A coordinate exactly midway between two nodes is equally near both. A
PolylineWire filament vertex or a PEC sheet plane there goes to the LOWER node
(#931), and on the non-uniform lane so does a port, source or probe (#1295):
the model below lands its feed on its conductor. The shared lower-node tie
rule now holds on both uniform and non-uniform lanes (#1342).
On either lane, a port end at
float32(3.5 mm) against a sheet at 3.5 mm straddles the half node and is
refused the same way.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation

DX = 1e-3
DOM = (20e-3, 19e-3, 17e-3)      # 19 cells in y: 9.5 mm is a half node, 9 odd


def _sim(lane, dom=DOM):
    kw = dict(freq_max=10e9, domain=dom, dx=DX, boundary="pec")
    if lane == "nu":
        kw.update(**{f"d{a}_profile": np.full(int(round(d / DX)), DX)
                     for a, d in zip("xyz", dom)})
    return Simulation(**kw)


def _dipole(lane, y):
    """Two legacy filament arms along z at (10 mm, y), fed by a wire port in
    the 1 mm gap between them at the same (x, y)."""
    sim = _sim(lane)
    x = 10e-3
    sim.add(PolylineWire(((x, y, 3e-3), (x, y, 8e-3)), radius=0.),
            material="pec")
    sim.add(PolylineWire(((x, y, 9e-3), (x, y, 14e-3)), radius=0.),
            material="pec")
    sim.add_port((x, y, 8e-3), component="ez", extent=1e-3)
    return sim


def _patch(lane, h):
    """A PEC sheet at height h over the PEC floor, fed by a wire port from
    the floor up to the sheet."""
    sim = _sim(lane)
    sim.add(Box((5e-3, 5e-3, h), (15e-3, 14e-3, h)), material="pec")
    sim.add_port((10e-3, 9e-3, 0.0), component="ez", extent=h)
    return sim


def _probe_on_sheet(lane, h):
    """A probe declared on a sheet plane."""
    sim = _sim(lane)
    sim.add(Box((5e-3, 5e-3, h), (15e-3, 14e-3, h)), material="pec")
    sim.add_source((10e-3, 9e-3, 2e-3), component="ez")
    sim.add_probe((12e-3, 9e-3, h), component="ex")
    return sim


HALF = {
    "dipole": lambda lane: _dipole(lane, 9.5e-3),
    "patch": lambda lane: _patch(lane, 3.5e-3),
    "probe_on_sheet": lambda lane: _probe_on_sheet(lane, 3.5e-3),
}
ON_NODE = {
    "dipole": lambda lane: _dipole(lane, 9e-3),
    "patch": lambda lane: _patch(lane, 3e-3),
    "probe_on_sheet": lambda lane: _probe_on_sheet(lane, 3e-3),
}


def _split_findings(sim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight(strict=False)
    return [i for i in report if getattr(i, "code", None) == "half_node_split"]


def _run(sim, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.run(n_steps=2, **kw)


def _refused(sim):
    found = _split_findings(sim)
    assert found and all(i.severity == "error" for i in found)
    text = str(found[0])
    assert "#1342" in text and "lower node" in text
    with pytest.raises(ValueError, match="half_node_split|#1342"):
        _run(sim)
    with pytest.raises(ValueError, match="#1342"):
        _run(sim, skip_preflight=True)


def _nodes(sim, lane):
    """(point feature node, conductor node) along the axis the case is
    about, read off what the run assembles."""
    from rfx.nonuniform import position_to_index
    grid = sim._build_nonuniform_grid() if lane == "nu" else sim._build_grid()
    sheets, wires = [], []
    assemble = (sim._assemble_materials_nu if lane == "nu"
                else sim._assemble_materials)
    assemble(grid, sheet_specs=[], pec_sheets=sheets, pec_wires=wires)

    def lookup(pos):
        return (position_to_index(grid, pos) if lane == "nu"
                else grid.position_to_index(pos))
    if wires:                               # dipole: y column of the arms
        arms = {int(j) for w in wires
                for j in np.nonzero(np.asarray(w.edges[2]))[1]}
        return lookup(sim._ports[0].position)[1], arms
    plane = {int(s.plane) for s in sheets}
    if sim._probes:
        return lookup(sim._probes[0].position)[2], plane
    pe = sim._ports[0]
    top = (pe.position[0], pe.position[1], pe.position[2] + pe.extent)
    return lookup(top)[2], plane


@pytest.mark.parametrize("case", sorted(HALF))
def test_on_the_nu_lane_a_feature_at_a_half_node_lands_on_its_conductor(case):
    sim = HALF[case]("nu")
    point, conductor = _nodes(sim, "nu")
    assert conductor == {point}
    assert _split_findings(sim) == []
    assert np.all(np.isfinite(np.asarray(_run(sim, skip_preflight=True).time_series)))


@pytest.mark.parametrize("case", sorted(HALF))
def test_on_the_uniform_lane_a_feature_at_a_half_node_lands_on_its_conductor(case):
    sim = HALF[case]("uniform")
    point, conductor = _nodes(sim, "uniform")
    assert conductor == {point}
    assert _split_findings(sim) == []
    assert np.all(np.isfinite(np.asarray(_run(sim, skip_preflight=True).time_series)))


@pytest.mark.parametrize("lane", ["uniform", "nu"])
@pytest.mark.parametrize("case", sorted(ON_NODE))
def test_the_same_model_on_a_node_runs(case, lane):
    sim = ON_NODE[case](lane)
    assert _split_findings(sim) == []
    assert np.all(np.isfinite(np.asarray(_run(sim, skip_preflight=True).time_series)))


@pytest.mark.parametrize("h", [3.5e-3, 50.5e-3])
@pytest.mark.parametrize("lane", ["uniform", "nu"])
def test_a_float32_port_end_straddling_the_half_node_of_a_sheet_is_refused(
        lane, h):
    """The port's far end is float32(h), half a float32 step or less above
    the sheet at h: nearest node above the half node against the sheet's
    lower node. At 3.5 mm that is 1.08e-10 m, 1e-7 of a cell; at 50.5 mm it is
    1.76e-6 of a cell, outside a window of 1e-6 of a cell alone."""
    h32 = float(np.float32(h))
    assert 0 < h32 - h <= 2.0 ** -24 * h
    n = int(round(h / DX + 0.5))                 # the node above the tie
    sim = _sim(lane, dom=(20e-3, 19e-3, (n + 9) * DX))
    sim.add(Box((5e-3, 5e-3, h), (15e-3, 14e-3, h)), material="pec")
    sim.add_port((10e-3, 9e-3, 0.0), component="ez", extent=h32)
    point, conductor = _nodes(sim, lane)
    assert (point, conductor) == (n, {n - 1})
    _refused(sim)


def test_the_same_declared_coordinate_has_no_split_message():
    assert _split_findings(_dipole("uniform", 9.5e-3)) == []


@pytest.mark.parametrize("lane", ["uniform", "nu"])
def test_the_message_names_both_features_and_both_nodes(lane):
    from dataclasses import replace
    # Keep the original dipole/message witness, with the conductor's
    # float32 coordinate just above the tie so the refusal still fires.
    wire_y = float(np.nextafter(np.float32(9.5e-3), np.float32(np.inf)))
    sim = _dipole(lane, wire_y)
    sim._ports[0] = replace(sim._ports[0], position=(10e-3, 9.5e-3, 8e-3))
    text = str(_split_findings(sim)[0])
    assert "add_port at (0.01, 0.0095, 0.008)" in text
    assert "PolylineWire 'pec'" in text
    assert "y = 9.5 mm" in text
    assert "y node 9 (9 mm)" in text and "node 10 (10 mm)" in text

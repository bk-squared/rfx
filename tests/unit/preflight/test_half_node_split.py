"""A port declared on a wire or sheet at a half node lands one cell off it (#1295, #1342).

A coordinate exactly midway between two nodes is equally near both. A port,
source or probe there goes to the EVEN node, on both lanes (``round(x/dx)`` on
the uniform grid, ``nearest_node_index`` on the non-uniform one, #1295). A
PolylineWire filament vertex or a PEC sheet plane there goes to the LOWER
node (#931). Where the lower node is odd the two land one cell apart. On
1 mm cells, a gap-fed dipole whose arms and feed port are all declared at
y = 9.5 mm has its arms on node 9 and its port on node 10: the port drives the
Ez edge beside the gap, and the gap field drops 23 dB (review of PR #1341). A
wire port from the ground up to a patch sheet at z = 3.5 mm drives one Ez edge
above the patch.

Until #1342 gives every feature one tie rule, such a model is refused with
the reason, by preflight (error ``half_node_split``) and at run time, where
``skip_preflight=True`` does not bypass it. The same models declared on a node
run.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation

DX = 1e-3
DOM = (20e-3, 19e-3, 17e-3)      # 19 cells in y: 9.5 mm is a half node, 9 odd


def _sim(lane):
    kw = dict(freq_max=10e9, domain=DOM, dx=DX, boundary="pec")
    if lane == "nu":
        kw.update(dx_profile=np.full(20, DX), dy_profile=np.full(19, DX),
                  dz_profile=np.full(17, DX))
    return Simulation(**kw)


def _dipole(lane, y):
    """Two sub-cell wire arms along z at (10 mm, y), fed by a wire port in
    the 1 mm gap between them at the same (x, y)."""
    sim = _sim(lane)
    x = 10e-3
    sim.add(PolylineWire(((x, y, 3e-3), (x, y, 8e-3)), radius=0.1e-3),
            material="pec")
    sim.add(PolylineWire(((x, y, 9e-3), (x, y, 14e-3)), radius=0.1e-3),
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


@pytest.mark.parametrize("lane", ["uniform", "nu"])
@pytest.mark.parametrize("case", sorted(HALF))
def test_a_feature_one_cell_off_its_conductor_is_refused(case, lane):
    sim = HALF[case](lane)
    found = _split_findings(sim)
    assert found and all(i.severity == "error" for i in found)
    text = str(found[0])
    assert "#1342" in text and "even node" in text and "lower one" in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="half_node_split|#1342"):
            sim.run(n_steps=2)
        with pytest.raises(ValueError, match="#1342"):
            sim.run(n_steps=2, skip_preflight=True)


@pytest.mark.parametrize("lane", ["uniform", "nu"])
@pytest.mark.parametrize("case", sorted(ON_NODE))
def test_the_same_model_on_a_node_runs(case, lane):
    sim = ON_NODE[case](lane)
    assert _split_findings(sim) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=2, skip_preflight=True)
    assert np.all(np.isfinite(np.asarray(result.time_series)))


def test_the_message_names_both_features_and_both_nodes():
    text = str(_split_findings(_dipole("nu", 9.5e-3))[0])
    assert "add_port at (0.01, 0.0095, 0.008)" in text
    assert "PolylineWire 'pec'" in text
    assert "y = 9.5 mm" in text
    assert "y node 10 (10 mm)" in text and "node 9 (9 mm)" in text

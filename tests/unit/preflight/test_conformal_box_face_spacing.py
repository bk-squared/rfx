"""A PEC Box face off a node plane is judged at its drawn plane on a conformal run (#1651).

The cut-cell conformal operands place a flat wall at its drawn plane to 1e-2 cell (rfx-archive
records/20261029-1651-sc-fix, K1: a TE101 box with its +x wall 0.25 cell past a node line). The cavity-spacing preflight
must then not refuse that wall; on a staircase run the face is still solved on the node plane, 1.4 % short, and refused.
"""

import pytest

from rfx.api import Simulation
from rfx.geometry.csg import Box


def _box_with_off_node_wall():
    sim = Simulation(freq_max=24e9, domain=(0.020, 0.014, 0.010), dx=1e-3, boundary="pec")
    sim.add(Box((0.01825, -0.01, -0.01), (0.03, 0.024, 0.020)), material="pec")
    sim.add_source((0.009, 0.007, 0.005), "ey", amplitude_kind="current")
    sim.add_probe((0.008, 0.007, 0.004), "ey")
    return sim


def test_staircase_run_refuses_the_off_node_cavity_face():
    with pytest.raises(ValueError, match="parallel PEC cavity faces"):
        _box_with_off_node_wall().run(n_steps=2)


def test_conformal_run_admits_the_face_at_its_drawn_plane():
    result = _box_with_off_node_wall().run(n_steps=2, conformal_pec=True)
    assert result.state is not None

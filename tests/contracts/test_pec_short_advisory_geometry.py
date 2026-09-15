"""Reject disconnected port measurements before interpreting column power."""

import numpy as np
import pytest

from rfx.sources.waveguide_port import waveguide_plane_positions
from tests._pec_short_advisory_fixture import SHORT_FACES, build
from tests._realized_geometry import assert_wall_planes, node_index


@pytest.mark.parametrize("dx,cpml", [(0.002, 8), (0.001, 10)])
def test_short_faces_and_actual_port_planes_describe_one_fixed_network(dx, cpml):
    frequencies = np.linspace(4e9, 6e9, 6)
    sim = build(frequencies, dx, cpml)
    grid = sim._build_grid()
    faces = [node_index(grid, 0, x) for x in SHORT_FACES]
    assert_wall_planes(sim, 0, expected_planes=list(range(faces[0], faces[1] + 1)))
    assert sim.freeze_mesh().dx == dx
    expected = [(0.010, 0.016, 0.030), (0.110, 0.104, 0.090)]
    for port, want in zip(sim._waveguide_ports, expected):
        cfg = sim._build_waveguide_port_config(port, grid, frequencies, 1)
        planes = waveguide_plane_positions(cfg)
        actual = [planes[k] for k in ("source", "reference", "probe")]
        np.testing.assert_allclose(actual, want, atol=1e-12, rtol=0)
        assert (max(actual) < SHORT_FACES[0] or min(actual) > SHORT_FACES[1]), (
            "A port source and both measurements must occupy the same connected "
            "vacuum region, strictly clear of both PEC faces"
        )

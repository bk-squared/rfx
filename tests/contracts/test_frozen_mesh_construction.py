"""#931: grid-based placement must survive adding the geometry it placed."""

import numpy as np
import pytest

from rfx import Box, Simulation
from tests._realized_geometry import assert_wall_planes, node_index, realized


@pytest.mark.parametrize("dx", [None, .002])
def test_short_fixture_faces_stay_on_final_nodes(dx):
    from tests.oracle.test_waveguide_port_validation_battery import _build_sim, SHORT_CELLS

    sim = _build_sim(np.linspace(5e9, 7e9, 6), dx=dx, pec_short_x=.085)
    grid = realized(sim).grid
    lo, hi = sim._pec_short_faces_m
    for face in (lo, hi):
        assert face / grid.dx == pytest.approx(round(face / grid.dx), abs=1e-10)
    faces = [node_index(grid, 0, x) for x in (lo, hi)]
    assert faces[1] - faces[0] == SHORT_CELLS
    assert (hi - lo) / grid.dx == pytest.approx(SHORT_CELLS)
    assert_wall_planes(sim, 0, expected_planes=list(range(faces[0], faces[1] + 1)))
    assert sim.freeze_mesh().dx == grid.dx


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_freeze_before_first_geometry_survives_execution(entry):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), boundary="pec")
    declared = sim._declared_mesh
    grid = sim.freeze_mesh()
    d = float(grid.dx)
    sim.add(Box((2*d, d, d), (3*d, 4*d, 4*d)), material="pec")
    # Adding this first body used to engage the feature planner and move nodes.
    result = getattr(sim, entry)(n_steps=1, skip_preflight=True)
    assert result.grid.dx == grid.dx
    assert result.grid.shape == grid.shape
    assert sim._declared_mesh == declared
    assert_wall_planes(sim, 0, expected_planes=[2, 3], ij=(2, 2))


def test_freezing_does_not_legalize_unresolved_volume():
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), boundary="pec")
    grid = sim.freeze_mesh()
    d = grid.dx
    sim.add(Box((2*d, d, d), (2.2*d, 4*d, 4*d)), material="pec")
    with pytest.raises(ValueError, match="volume|sub.cell|thickness"):
        realized(sim)
    assert sim._build_realized_grid().dx == d


def test_freeze_keeps_automatic_profiles_and_extent_after_geometry_edit():
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), boundary="pec")
    sim.add(Box((0, 0, 0), (.008, .008, .0016)), material="fr4")
    declared = sim._declared_mesh
    grid = sim.freeze_mesh()
    assert sim._uses_nonuniform_mesh
    extent = sim._domain
    sim.add(Box((0, 0, .02), (.008, .008, .0208)), material="fr4")
    later = sim._build_realized_grid()
    assert later.shape == grid.shape
    assert later.dx == grid.dx
    np.testing.assert_array_equal(later.dz, grid.dz)
    assert sim._domain == extent
    assert sim._declared_mesh == declared


def test_freeze_copies_caller_profile_and_refuses_mesh_reassignment():
    profile = np.full(8, .001)
    domain = [.008, .008, .008]
    sim = Simulation(freq_max=10e9, domain=domain,
                     dx=.001, dz_profile=profile, boundary="pec")
    grid = sim.freeze_mesh()
    profile[:] = .002
    domain[0] = .016
    later = sim._build_realized_grid()
    assert later.shape == grid.shape
    np.testing.assert_array_equal(later.dz, grid.dz)
    with pytest.raises(ValueError, match="Mesh is frozen"):
        sim._dx = .0005

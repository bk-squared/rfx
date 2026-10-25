"""Automatic wall preservation is bounded; cell-based declarations use the grid."""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import EPS_0
from rfx.grid import C0, Grid
from rfx.probes.msl_wave_decomp import register_msl_wave_probes
from rfx.ris import RISUnitCell
from tests.unit.boundaries.test_periodic_anchor import largest_period_fit


def wall_model(*, geometry=False, domain=(.0103, .008, .006), dx=None, mode='3d'):
    sim = Simulation(C0 / .020, domain, dx=dx, cpml_layers=0, mode=mode,
                     boundary=BoundarySpec(x='periodic', y='pec', z='pec'))
    if geometry:
        sim.add_material('body', eps_r=2.25)
        sim.add(Sphere((.0051, .0036, .0028), .002), material='body')
    return sim


@pytest.mark.parametrize('geometry', [False, True])
def test_automatic_periodic_mesh_reports_costly_exact_wall_spacing(geometry):
    sim = wall_model(geometry=geometry)
    requested = .0005 if geometry else .001
    expected = largest_period_fit(.0103, (.008, .006), requested)
    assert expected == .0103 / 26
    # Period anchors find the 1 %-fitting spacing missed by exact wall anchors.
    if geometry:
        with pytest.warns(UserWarning, match='snapped'):
            grid = sim._build_grid()
        assert .75 * requested <= grid.dx <= requested
        for count, length in zip((grid.ny - 1, grid.nz - 1), (.008, .006)):
            assert abs(count * grid.dx / length - 1) <= .01
        assert grid.dx == expected and grid.shape == (26, 21, 16)
    else:
        assert expected < .75 * requested
        with pytest.raises(ValueError, match=f"largest fitting dx={expected:.12g} m") as caught:
            sim._build_grid()
        assert "0.75 * requested dx" in str(caught.value)


@pytest.mark.parametrize('domain,expected_dx,shape', [
    ((.0096, .0072, .0056), .0008, (12, 10, 8)),
    ((.00975, .00825, .006), .00075, (13, 12, 9)),
])
def test_common_wall_spacing_inside_the_cost_bound_is_adopted(domain, expected_dx, shape):
    sim = wall_model(domain=domain)
    with pytest.warns(UserWarning, match='snapped'):
        grid = sim._build_grid()
        axes = sim.fidelity_report(print_report=False)[0]['axes']
    # The largest common spacings are 0.8 and exactly 0.75 of the 1 mm request.
    assert grid.dx == pytest.approx(expected_dx, rel=2e-14, abs=0)
    assert grid.shape == shape
    assert [a['realized_extent_um'] for a in axes] == pytest.approx(
        np.array(domain) * 1e6, rel=2e-14)


def test_closed_walls_constrain_an_already_dividing_period():
    # Mixed PEC/PMC faces also close an axis. This checks lattice length;
    # it makes no claim about the separate PMC half-cell wall convention.
    grid = Grid(C0 / .020, (.008, .0088, .0072), periodic_axes='x', cpml_layers=0,
                pec_faces={'y_lo', 'z_lo', 'z_hi'}, pmc_faces={'y_hi'})
    assert grid.dx == pytest.approx(.0008, rel=2e-14)
    assert grid.shape == (10, 12, 10)


@pytest.mark.parametrize('mode', ['2d_tmz', '2d_tez'])
def test_automatic_wall_snap_ignores_the_invariant_axis(mode):
    grid = wall_model(domain=(.0096, .0072, .006157), mode=mode)._build_grid()
    assert grid.dx == pytest.approx(.0008, rel=2e-14)
    assert grid.shape == (12, 10, 1)


@pytest.mark.parametrize('length', [.0103, .010])
def test_bounded_snap_refuses_and_reports_largest_fitting_spacing(length):
    sim = wall_model(domain=(length, .008123, .006157))
    expected = largest_period_fit(length, (.008123, .006157), .001)
    assert expected == length / (32 if length == .0103 else 21)
    assert expected < .75 * .001
    # Period anchors find advice where no exact wall anchor divided the period.
    with pytest.raises(ValueError, match=f"largest fitting dx={expected:.12g} m") as caught:
        sim._build_grid()
    assert "requested dx/16" in str(caught.value)
    assert "0.75 * requested dx" in str(caught.value)


def test_explicit_periodic_spacing_and_nonperiodic_auto_spacing_are_unchanged():
    explicit = .0103 / 11
    with warnings.catch_warnings(record=True) as caught:
        grid = wall_model(dx=explicit, domain=(.0103, 9 * explicit, 7 * explicit))._build_grid()
    assert grid.dx == explicit and grid.shape == (11, 10, 8)
    assert not caught
    with pytest.raises(ValueError, match='common dx dividing each L'):
        wall_model(dx=.001)._build_grid()
    nonperiodic = Simulation(C0 / .020, (.010, .008, .006),
                             boundary='pec', cpml_layers=0)._build_grid()
    assert nonperiodic.dx == .001 and nonperiodic.shape == (11, 9, 7)


def test_an_absorber_face_excludes_its_axis_from_the_wall_snap():
    sim = Simulation(C0 / .020, (.010, .008123, .006), cpml_layers=8,
                     boundary=BoundarySpec(x='periodic',
                         y=Boundary(lo='pec', hi='cpml', hi_thickness=3), z='pec'))
    with warnings.catch_warnings(record=True) as caught:
        grid = sim._build_grid()
    assert grid.dx == .001 and not caught
    assert (grid.pad_y_lo, grid.pad_y_hi) == (0, 3)
    # The pre-existing rounded interior ends at 9 mm; three pad nodes follow.
    assert grid.node_of('y', grid.interior[1].stop - 1) == pytest.approx(.009)
    assert grid.node_of('y', grid.ny - 1) == pytest.approx(.012)


def test_msl_point_probe_ladder_uses_the_realized_periodic_cell():
    sim = wall_model(geometry=True, domain=(.0103, 17 * .0103 / 21, 13 * .0103 / 21))
    probes = register_msl_wave_probes(
        sim, feed_x=.0023, direction='+x', y_centre=.0037,
        z_ez=.0026, z_hy=.0031, n_offset_cells=5, n_spacing_cells=3)
    grid = sim._build_grid()
    assert float(sim._dx) != grid.dx  # the mesh planner's value remains a declaration
    assert probes.delta == pytest.approx(3 * grid.cells(0)[0], rel=2e-14)
    expected = .0023 + np.array([5, 8, 11, 5]) * grid.cells(0)[0]
    np.testing.assert_allclose([p.position[0] for p in sim._probes], expected, rtol=0, atol=1e-17)


@pytest.mark.parametrize('capacitance', [1e-15, 1e-13])
def test_ris_capacitance_uses_the_final_periodic_cell(capacitance):
    cell = RISUnitCell(cell_size=(.0103, .0206), substrate_thickness=.004,
                       freq_range=(C0 / .040, C0 / .020), cpml_layers=2,
                       dx=.0103 / 26)
    cell.add_varactor((.0031, .0067), capacitance_range=(1e-15, 1e-13))
    sim = cell._build_sim(capacitance_override=capacitance)
    grid = sim._build_grid()
    assert grid.dx == .0103 / 26
    # Recover the declared capacitance from the material the solver receives.
    added = float(sim._materials['substrate'].eps_r) - 4.4
    assert added * EPS_0 * grid.cells(0)[0] == pytest.approx(capacitance, rel=2e-13, abs=0)
    assert sim.freeze_mesh().shape == grid.shape


def test_ris_without_varactor_keeps_the_automatic_uniform_mesh():
    cell = RISUnitCell(cell_size=(.0103, .0206), substrate_thickness=.004,
                       freq_range=(C0 / .040, C0 / .020), cpml_layers=2)
    sim = cell._build_sim()
    sim.add_probe((.0031, .0067, .0102), component='ex')
    grid = sim._build_grid()
    assert grid.shape == (26, 52, 66)
    assert grid.dx == .0103 / 26
    assert sim._declared_mesh['_dx'] is None
    assert '_frozen_mesh' not in sim.__dict__
    trace = np.asarray(sim.run(n_steps=96, compute_s_params=False).time_series)
    assert trace.shape == (96, 2)
    assert np.isfinite(trace).all() and np.any(trace != 0)


def test_fidelity_counts_a_pec_volume_in_the_last_periodic_cell():
    sim = wall_model(domain=(.010, .006, .004), dx=.001)
    sim.add(Box((.009, .001, .001), (.010, .003, .003)), material='pec')
    body = sim.fidelity_report(print_report=False)[1]
    # One x cell across the periodic seam, two y cells and two z cells.
    assert body['n_cells'] == 1 * 2 * 2

"""DC footprints occupy the cells whose centres lie inside the drawing."""
import warnings

import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.boundaries.spec import BoundarySpec


H = .001
DOMAIN = (.016, .018, .004)
CENTRE = (.00637, .00761, .002)
RADIUS = .0033


def _realized(shape, **kwargs):
    sim = Simulation(10e9, DOMAIN, dx=H, boundary=kwargs.pop('boundary', 'pec'),
                     snap='declared', **kwargs)
    sim.add_thin_conductor(shape, sigma_bulk=100., thickness=35e-6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        record = sim.realized_geometry()
    entity = record.entities[0]
    assert entity.occupancy_role == 'solved'
    return record, np.asarray(entity.mask).any(axis=2)


def _centres(record, footprint):
    # The public record includes the far face of each stored cell.
    nodes = [np.asarray(record.nodes[a]) for a in (0, 1)]
    widths = [np.diff(n)[:footprint.shape[a]] for a, n in enumerate(nodes)]
    centres = [n[:footprint.shape[a]] + widths[a] / 2 for a, n in enumerate(nodes)]
    return centres, widths


@pytest.fixture(scope='module')
def discs():
    original = _realized(Cylinder(CENTRE, RADIUS, 0., axis='z'))
    mirrors = []
    for axis in (0, 1):
        centre = list(CENTRE)
        centre[axis] = DOMAIN[axis] - centre[axis]
        mirrors.append(_realized(Cylinder(tuple(centre), RADIUS, 0., axis='z')))
    centre = (DOMAIN[0] - CENTRE[0], DOMAIN[1] - CENTRE[1], CENTRE[2])
    mirrors.append(_realized(Cylinder(centre, RADIUS, 0., axis='z')))
    return original, mirrors


@pytest.mark.parametrize('axis', (0, 1), ids=('x', 'y'))
def test_disc_mirror_indices(discs, axis):
    (record, original), mirrors = discs
    occupied = np.argwhere(original)
    expected = occupied.copy()
    # Cell j reflects to N - 1 - j, where N counts physical domain cells.
    n_cells = np.count_nonzero(np.asarray(record.nodes[axis]) < DOMAIN[axis])
    expected[:, axis] = n_cells - 1 - expected[:, axis]
    assert set(map(tuple, np.argwhere(mirrors[axis][1]))) == set(map(tuple, expected))


@pytest.mark.parametrize('axis', (0, 1), ids=('x', 'y'))
def test_disc_centroid(discs, axis):
    (record, footprint), _ = discs
    centres, widths = _centres(record, footprint)
    area = footprint * widths[0][:, None] * widths[1][None, :]
    line = centres[axis][:, None] if axis == 0 else centres[axis][None, :]
    centroid = np.sum(area * line) / np.sum(area)
    assert abs(centroid - CENTRE[axis]) < .25 * H


def test_disc_drawn_footprint(discs):
    (record, footprint), mirrors = discs
    centres, _ = _centres(record, footprint)
    expected = ((centres[0][:, None] - CENTRE[0])**2
                + (centres[1][None, :] - CENTRE[1])**2 <= RADIUS**2)
    np.testing.assert_array_equal(footprint, expected)
    reflected = np.argwhere(expected)
    for axis in (0, 1):
        n_cells = np.count_nonzero(np.asarray(record.nodes[axis]) < DOMAIN[axis])
        reflected[:, axis] = n_cells - 1 - reflected[:, axis]
    assert set(map(tuple, np.argwhere(mirrors[2][1]))) == set(map(tuple, reflected))


@pytest.mark.parametrize('lo,hi', ((.0023, .0076), (.002, .008)),
                         ids=('off_node', 'node_aligned'))
def test_box_cells(lo, hi):
    record, footprint = _realized(Box((lo, .004, .002), (hi, .011, .002)))
    centres, _ = _centres(record, footprint)
    expected = ((centres[0][:, None] >= lo) & (centres[0][:, None] < hi)
                & (centres[1][None, :] >= .004) & (centres[1][None, :] < .011))
    np.testing.assert_array_equal(footprint, expected)
    np.testing.assert_array_equal(np.flatnonzero(footprint.any(axis=1)), np.arange(2, 8))
    if lo == .002:
        x = np.asarray(record.nodes[0])[:footprint.shape[0]]
        y = np.asarray(record.nodes[1])[:footprint.shape[1]]
        old_node_mask = ((x[:, None] >= lo) & (x[:, None] < hi)
                         & (y[None, :] >= .004) & (y[None, :] < .011))
        np.testing.assert_array_equal(footprint, old_node_mask)


@pytest.mark.parametrize('shape_kind', ('disc', 'box'))
@pytest.mark.parametrize('graded', (False, True))
def test_periodic_seam_centres(shape_kind, graded):
    centre = (.00037, .00761, .002)
    shape = (Cylinder(centre, RADIUS, 0., axis='z') if shape_kind == 'disc' else
             Box((-.0023, .004, .002), (.0026, .011, .002)))
    # The public NU lane supports periodic axes with constant profiles;
    # grade the other tangent to exercise its per-cell widths as well.
    profiles = {'dy_profile': H * np.r_[1., np.tile([.9, 1.1], 8), 1.]} if graded else {}
    record, footprint = _realized(
        shape, boundary=BoundarySpec(x='periodic', y='pec', z='pec'), **profiles)
    centres, _ = _centres(record, footprint)
    assert centres[0][-1] < DOMAIN[0]
    expected = np.zeros_like(footprint)
    for image in (-1, 0, 1):
        x = centres[0][:, None] + image * DOMAIN[0]
        y = centres[1][None, :]
        expected |= (((x - centre[0])**2 + (y - centre[1])**2 <= RADIUS**2)
                     if shape_kind == 'disc' else
                     ((x >= -.0023) & (x < .0026) & (y >= .004) & (y < .011)))
    np.testing.assert_array_equal(footprint, expected)
    assert footprint[-1].any()

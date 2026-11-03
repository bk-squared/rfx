"""DESIGN-2385: an internal same-material face has bulk outside media."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.geometry.smoothing import (
    compute_inv_eps_tensor_diag, compute_smoothed_eps, compute_smoothed_eps_nonuniform,
)


@pytest.mark.parametrize('path', ['uniform', 'graded', 'kottke_pec', 'nodes', 'graded_nodes'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
@pytest.mark.parametrize('reverse', [False, True], ids=['AB', 'BA'])
@pytest.mark.parametrize('offset', [0., .0002], ids=['node_plane', 'off_grid'])
def test_same_permittivity_face_operands(path, traced, reverse, offset):
    graded = path.startswith('graded')
    profiles = dict(dx_profile=np.full(8, .0005), dy_profile=np.full(8, .0005),
                    dz_profile=np.r_[np.full(4, .0004), np.full(4, .0006)]) if graded else {}
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0, **profiles)
    grid = sim.freeze_mesh()
    face = np.asarray(sim.realized_geometry().nodes[2])[4] + offset

    def evaluate(z):
        pairs = [(Box((-.1, -.1, -.1), (.1, .1, z)), 4.),
                 (Box((-.1, -.1, z), (.1, .1, .1)), 4.)]
        if reverse:
            pairs.reverse()
        if path.endswith('nodes'):
            return tuple(1/a for a in compute_node_tensor_eps(grid, pairs).nodes.xi[:3])
        if path == 'kottke_pec':
            return tuple(1/a for a in compute_inv_eps_tensor_diag(grid, dielectric_shapes=pairs))
        return (compute_smoothed_eps_nonuniform(grid, pairs) if graded
                else compute_smoothed_eps(grid, pairs))

    coordinate = jnp.float32(face) if traced else np.float64(face)
    rtol = 2e-5 if coordinate.dtype == np.dtype('float32') else 1e-6
    result = jax.jit(evaluate)(coordinate) if traced else evaluate(coordinate)
    for values in result:
        np.testing.assert_allclose(np.asarray(values)[4, 4, 4], 4., rtol=rtol)


@pytest.mark.parametrize('reverse', [False, True], ids=['AB', 'BA'])
def test_split_cavity_operands_equal_unsplit(reverse):
    sim = Simulation(domain=(.020, .013, .007), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    grid = sim.freeze_mesh()
    lower = (Box((0., 0., 0.), (.020, .013, .0029)), 3.38)
    unsplit = [lower, (Box((0., 0., .0029), (.020, .013, .007)), 10.2)]
    split = [lower, (Box((0., 0., .0029), (.020, .013, .00495)), 10.2),
             (Box((0., 0., .00495), (.020, .013, .007)), 10.2)]
    if reverse:
        split[1:] = reversed(split[1:])
    expected = compute_smoothed_eps(grid, unsplit)
    actual = compute_smoothed_eps(grid, split)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)


def test_subcell_sphere_keeps_main_operand():
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    pairs = [(Sphere((.002, .002, .00225), .00015), 4.)]
    result = compute_smoothed_eps(sim.freeze_mesh(), pairs)
    # Ez is at the sphere centre: normal is zero and f=.8. The selected
    # normal-source sphere stays excluded, giving arithmetic .8*4+.2*1=3.4.
    # Literal below is the measured float32 operand on main e07ba5dff.
    np.testing.assert_allclose(np.asarray(result[2])[4, 4, 4],
                               3.3999998569488525, rtol=1e-12)


def test_resolved_gap_keeps_main_operands():
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    pairs = [(Box((-.1, -.1, -.1), (.1, .1, .0021)), 4.),
             (Box((-.1, -.1, .00285), (.1, .1, .1)), 4.)]
    result = compute_smoothed_eps(sim.freeze_mesh(), pairs)
    # Ez at 2.25 mm spans 2--2.5 mm: .2 dielectric; Ez at 2.75 mm
    # spans 2.5--3 mm: .3 dielectric. The .75 mm air gap is resolved.
    expected = (np.float32(1/(.2/4+.8)), np.float32(1/(.3/4+.7)))
    for k, value in zip((4, 5), expected):
        np.testing.assert_allclose(np.asarray(result[2])[4, 4, k], value, rtol=1e-12)


@pytest.mark.parametrize('path', ['uniform', 'nodes', 'kottke_pec', 'graded', 'graded_nodes'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_periodic_unit_cell_reads_bulk(path, traced):
    """A Box spanning one 4 mm period touches its own image at x=0."""
    graded = path.startswith('graded')
    profiles = dict(dx_profile=np.full(8, .0005), dy_profile=np.full(8, .0005),
                    dz_profile=np.r_[np.full(4, .0004), np.full(4, .0006)]) if graded else {}
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary={'x': 'periodic', 'y': 'pec', 'z': 'pec'},
                     cpml_layers=0, **profiles)
    grid = sim.freeze_mesh()

    def evaluate(end):
        pairs = [(Box((0., -.1, -.1), (end, .1, .1)), 4.)]
        if path.endswith('nodes'):
            return tuple(1/a for a in compute_node_tensor_eps(grid, pairs).nodes.xi[:3])
        if path == 'kottke_pec':
            return tuple(1/a for a in compute_inv_eps_tensor_diag(grid, dielectric_shapes=pairs))
        return (compute_smoothed_eps_nonuniform(grid, pairs) if graded
                else compute_smoothed_eps(grid, pairs))

    end = jnp.float32(.004) if traced else np.float64(.004)
    result = jax.jit(evaluate)(end) if traced else evaluate(end)
    for values in result:
        # The first Yee plane includes Ex's half-cell stagger; Ey/Ez and
        # all node entries lie on x=0. Check the complete transverse plane.
        np.testing.assert_allclose(np.asarray(values)[0], 4., rtol=2e-5 if traced else 1e-6)


@pytest.mark.parametrize('reverse', [False, True], ids=['AB', 'BA'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_same_permittivity_gap_reads_nearest_box_alone(reverse, traced):
    """The 0.45 mm gap contains q, half a cell beyond either face."""
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    grid = sim.freeze_mesh()
    nodes = np.asarray(sim.realized_geometry().nodes[2])[:grid.shape[2]]

    def evaluate(faces):
        lower = (Box((-.1, -.1, -.1), (.1, .1, faces[0])), 4.)
        upper = (Box((-.1, -.1, faces[1]), (.1, .1, .1)), 4.)
        both = [upper, lower] if reverse else [lower, upper]
        return tuple(compute_smoothed_eps(grid, pairs) for pairs in (both, [lower], [upper]))

    faces = (jnp.asarray([.00202, .00247], dtype=jnp.float32) if traced
             else np.asarray([.00202, .00247], dtype=np.float64))
    combined, lower, upper = jax.jit(evaluate)(faces) if traced else evaluate(faces)
    lo, hi = np.asarray(faces)
    rtol = 2e-5 if traced else 1e-6
    for axis in range(3):
        positions = nodes + (grid.dx/2 if axis == 2 else 0.)
        near_lower = np.abs(positions-lo) < np.abs(positions-hi)
        distance = np.where(near_lower, positions-lo, hi-positions)
        q = positions + (grid.dx/2-distance)*np.where(near_lower, 1., -1.)
        selected = (q > lo) & (q < hi)
        assert np.all(selected)
        expected = np.where(near_lower[None, None, :], np.asarray(lower[axis]),
                            np.asarray(upper[axis]))[..., selected]
        np.testing.assert_allclose(np.asarray(combined[axis])[..., selected], expected, rtol=rtol)
    # Hand fractions: .54/.56 dielectric at tangential z=2/2.5 mm;
    # .06 dielectric at normal z=2.25 mm, whose nearest face is 2.47 mm.
    for axis in (0, 1):
        np.testing.assert_allclose(np.asarray(combined[axis])[4, 4, 4:6],
                                   [2.62, 2.68], rtol=rtol)
    np.testing.assert_allclose(np.asarray(combined[2])[4, 4, 4],
                               1/(.06/4+.94), rtol=rtol)

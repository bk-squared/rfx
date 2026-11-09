"""#2453: declared paint order and distant same-medium union contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Sphere
from rfx.geometry import smoothing
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.grid import Grid
from rfx.model.graded_grid import make_nonuniform_grid


MODES = (True, 'kottke_pec', 'full_tensor')


def grid_for(path='uniform', domain=.012):
    if path == 'uniform':
        return Grid(freq_max=20e9, domain=(domain,) * 3, dx=.001, cpml_layers=0)
    # Nonconstant physical widths on all three axes; node 6 remains at 6 mm.
    profile = np.array([.001] * 4 + [.0008, .0012] * 2 + [.001] * 4)
    return make_nonuniform_grid((domain, domain), profile, .001, cpml_layers=0,
                               cpml_axes='', dx_profile=profile, dy_profile=profile)


def b1_shapes(sphere=False, with_a=True):
    a = (Box((.0013,) * 3, (.0031,) * 3), 4.)
    b = (Box((.0033,) * 3, (.0097,) * 3), 10.2)
    c = (Sphere((.0065,) * 3, .0013) if sphere else
         Box((.0052,) * 3, (.0078,) * 3), 4.)
    return ([a] if with_a else []) + [b, c]


def b3_shapes(sphere=False):
    if sphere:
        a = (Box((.002,) * 3, (.0053, .008, .008)), 4.)
        c = (Sphere((.0068, .005, .005), .0015), 4.)
        d = (Box((.0095,) * 3, (.0105,) * 3), 10.2)
    else:
        a = (Box((.002,) * 3, (.005, .008, .008)), 4.)
        c = (Box((.005, .002, .002), (.0063, .008, .008)), 4.)
        d = (Box((.0084, .002, .002), (.0094, .008, .008)), 10.2)
    return a, c, d


def arrays_for(grid, pairs, mode):
    if mode == 'full_tensor':
        result = compute_node_tensor_eps(grid, pairs)
        return {'diagonal': np.asarray(result.diagonal), 'xi': np.asarray(result.nodes.xi),
                'triplets': np.asarray(result.nodes.triplets)}
    if mode == 'kottke_pec':
        return {'inverse_diagonal': np.asarray(smoothing.compute_inv_eps_tensor_diag(grid, pairs, []))}
    build = (smoothing.compute_smoothed_eps_nonuniform if hasattr(grid, 'dx_arr')
             else smoothing.compute_smoothed_eps)
    return {'diagonal': np.asarray(build(grid, pairs))}


@pytest.mark.parametrize('path,mode', [('uniform', m) for m in MODES] +
                         [('graded', True), ('graded', 'full_tensor')])
@pytest.mark.parametrize('with_a', [False, True])
def test_b1_box_values(path, mode, with_a):
    arrays = arrays_for(grid_for(path), b1_shapes(with_a=with_a), mode)
    for name, array in arrays.items():
        if name == 'triplets':
            continue  # Revision 1 pins xi and diagonal here; all triplets below.
        actual = array[:, 6, 6, 6]
        expected = ([.25, .25, .25, 0., 0., 0.] if name == 'xi' else
                    [.25, .25, .25] if name == 'inverse_diagonal' else [4., 4., 4.])
        np.testing.assert_array_max_ulp(actual, np.asarray(expected, dtype=np.float32), maxulp=1)


@pytest.mark.parametrize('path,mode', [('uniform', m) for m in MODES] +
                         [('graded', True), ('graded', 'full_tensor')])
def test_b1_sphere_remote_arrays(path, mode):
    grid = grid_for(path)
    with_a = arrays_for(grid, b1_shapes(sphere=True), mode)
    without_a = arrays_for(grid, b1_shapes(sphere=True, with_a=False), mode)
    # Physical node coordinates, independently spelled from this test's mesh.
    line = (np.arange(13) * .001 if path == 'uniform' else
            np.array([0., .001, .002, .003, .004, .005, .006, .007, .008,
                      .0088, .010, .0108, .012]))
    reach = .002 if path == 'uniform' else .0024
    far = ((line[:, None, None] > .0031 + reach) |
           (line[None, :, None] > .0031 + reach) |
           (line[None, None, :] > .0031 + reach))
    for name in with_a:
        # Leading dimension is component (3/6) or triplet (all eight).
        np.testing.assert_array_equal(with_a[name][:, far], without_a[name][:, far])


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('sphere', [False, True])
def test_b3_distant_intervening_medium(mode, sphere):
    grid = grid_for(domain=.012 if sphere else .010)
    a, c, d = b3_shapes(sphere)
    adc = arrays_for(grid, [a, d, c], mode)
    acd = arrays_for(grid, [a, c, d], mode)
    for name in adc:
        # Expected difference is exactly zero for every array and all triplets.
        np.testing.assert_array_equal(adc[name] - acd[name], 0.)


def test_join_uses_latest_member_and_unknown_bounds():
    grid = grid_for()
    a, c, d = b3_shapes()
    unknown = (object(), 2.)
    shapes = [a, d, c, unknown, a]
    groups = smoothing._medium_groups(shapes, grid)
    assert list(groups) == [(0, 4.), (1, 10.2), (2, 2.), (3, 4.)]
    assert [[shapes.index((s, eps)) for s in group] for eps, group in groups.values()] == [
        [0, 2], [1], [3], [0]]


def test_periodic_images_and_axis_lengths():
    grid = Grid(freq_max=20e9, domain=(.012, .020, .030), dx=.001,
                cpml_layers=0, periodic_axes='y')
    a = (Box((.002, .005, .002), (.004, .006, .004)), 4.)
    d = (Box((.002, .0185, .002), (.004, .0195, .004)), 10.2)
    c = (Box((.002, .0002, .002), (.004, .0012, .004)), 4.)
    assert list(smoothing._medium_groups([a, d, c], grid)) == [(0, 4.), (1, 10.2), (2, 4.)]
    # A whole-period translation keeps the same decision in absolute coordinates.
    shifted = (Box((.002, .0402, .002), (.004, .0412, .004)), 4.)
    assert list(smoothing._medium_groups([a, d, shifted], grid)) == [(0, 4.), (1, 10.2), (2, 4.)]


def test_realized_graded_maximum_and_traced_bounds():
    a, c, d = b3_shapes()
    # The 2.1 mm gap joins on 1 mm cells, splits when any axis reaches 1.2 mm.
    assert len(smoothing._medium_groups([a, d, c], grid_for())) == 2
    assert len(smoothing._medium_groups([a, d, c], grid_for('graded'))) == 3

    def count(x):
        traced = (Box((x, .002, .002), (.0063, .008, .008)), 4.)
        return jnp.asarray(len(smoothing._medium_groups([a, d, traced], grid_for())))

    assert int(jax.jit(count)(.005)) == 3

"""Singleton transverse axes retain their boundary-face semantics."""
import builtins

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.sources._waveguide_modes import _cell_centred_gradient, _shift_profile_to_dual
from rfx.sources._laplace import _solve_laplace_2d
from rfx.sources.waveguide_port import (
    _discrete_te_mode_profiles, _discrete_tm_mode_profiles,
)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('bc', ['neumann', 'dirichlet'])
@pytest.mark.parametrize('n', [1, 2])
def test_gradient_boundary_face_average(axis, bc, n):
    widths = np.array([0.7, 1.3])[:n]
    field = np.array([[2., -3., 5.], [7., 4., -1.]])[:n]
    faces = np.zeros((n + 1, 3))
    if bc == 'dirichlet':
        faces[0] = 2 * field[0] / widths[0]
        faces[-1] = -2 * field[-1] / widths[-1]
    if n == 2:
        faces[1] = (field[1] - field[0]) / widths.mean()
    expected = (faces[:-1] + faces[1:]) / 2
    actual = _cell_centred_gradient(np.moveaxis(field, 0, axis), widths, axis, bc=bc)
    np.testing.assert_allclose(actual, np.moveaxis(expected, 0, axis))


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('kind', ['TE', 'TM'])
def test_discrete_profiles_singleton_each_axis(axis, kind):
    widths = [np.full(7, .001), np.array([.001])]
    if axis == 0:
        widths.reverse()
    mode = ((0, 1) if axis == 0 else (1, 0)) if kind == 'TE' else (1, 1)
    solver = _discrete_te_mode_profiles if kind == 'TE' else _discrete_tm_mode_profiles
    ey, ez, hy, hz, kc = solver(*(w.sum() for w in widths), *mode, *widths,
                               h_offset=(.5, .5))
    dA = widths[0][:, None] * widths[1][None, :]
    assert kc > 0
    np.testing.assert_allclose(np.sum((ey**2 + ez**2) * dA), 1.)
    np.testing.assert_allclose(np.sum((ey*hz - ez*hy) * dA), 1.)
    field = np.arange(7.).reshape(ey.shape)
    offset = (.5, 0.) if axis == 0 else (0., .5)
    np.testing.assert_array_equal(_shift_profile_to_dual(field, offset), field)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('fallback', [False, True])
def test_laplace_singleton_is_linear_with_neumann_sides(axis, fallback, monkeypatch):
    if fallback:
        original = builtins.__import__
        def without_scipy(name, *args, **kwargs):
            if name.startswith('scipy.sparse'):
                raise ImportError('exercise iterative fallback')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', without_scipy)
    shape = (1, 7) if axis == 0 else (7, 1)
    trace = np.zeros(shape, bool)
    ground = np.zeros(shape, bool)
    trace.flat[-1] = True
    ground.flat[0] = True
    phi = _solve_laplace_2d(np.ones(shape), trace, ground, .002, .003)
    np.testing.assert_allclose(phi.ravel(), np.linspace(0, 1, 7), atol=1e-8)


@pytest.mark.parametrize('shape', [(1, 7), (7, 1), (1, 1)])
def test_floquet_singleton_plane(shape):
    from rfx.core.yee import EPS_0, MU_0
    from rfx.floquet import init_floquet_dft, extract_floquet_modes
    acc = init_floquet_dft(3, shape)
    e = jnp.ones((3,) + shape, dtype=jnp.complex64)
    acc = acc._replace(e_tang1_dft=e, h_tang2_dft=e / np.sqrt(MU_0 / EPS_0))
    out = extract_floquet_modes(acc, .001, .001*shape[0], .001*shape[1],
                                jnp.array([8e9, 10e9, 12e9]))
    np.testing.assert_allclose(out['forward_amplitude'], 1., atol=1e-6)
    np.testing.assert_allclose(out['S'], 0., atol=1e-6)


@pytest.mark.parametrize('axis', [0, 1])
def test_coax_cartesian_interpolation_refuses_single_sample(axis):
    from rfx.sources.coaxial_port import coaxial_tem_reference_plane_vi_from_cartesian_plane
    coords = [np.arange(5.), np.arange(5.)]
    coords[axis] = np.array([0.])
    field = np.zeros(tuple(len(c) for c in coords))
    with pytest.raises(ValueError, match='at least two samples'):
        coaxial_tem_reference_plane_vi_from_cartesian_plane(
            *coords, field, field, field, field, center_u_m=0., center_v_m=0.,
            inner_radius=.1, outer_radius=.2)


@pytest.mark.parametrize('direction', ['+x', '-x', '+y', '-y'])
@pytest.mark.parametrize('thin_axis', ['width', 'normal'])
def test_msl_one_cell_profile(direction, thin_axis):
    from rfx.grid import Grid
    from rfx.sources.msl_port import MSLPort, compute_msl_mode_profile
    dx = .001
    grid = Grid(freq_max=10e9, domain=(.01, .01, .006), dx=dx, cpml_layers=0)
    port = MSLPort(feed_x=.005, y_lo=.004, y_hi=.0044 if thin_axis == 'width' else .007,
                   z_lo=0., z_hi=dx if thin_axis == 'normal' else 2*dx,
                   direction=direction, impedance=50., excitation=None)
    result = compute_msl_mode_profile(grid, port, 2.2, refine=1,
                                      pad_y_cells=0, pad_z_cells=0)
    profile = result['ez_profile']
    assert profile.shape[0 if thin_axis == 'width' else 1] == 1
    assert np.isfinite(profile).all() and np.any(profile)
    assert np.isfinite(result['z0_static']) and result['z0_static'] > 0


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('kind', ['TE', 'TM'])
def test_scalar_eigenmode_vector_conversion_singleton(axis, kind):
    from rfx.eigenmode import (
        _scalar_eigenmodes_to_vector, _build_laplacian_2d_neumann,
        _build_laplacian_2d_dirichlet,
    )
    shape = (1, 7) if axis == 0 else (7, 1)
    builder = _build_laplacian_2d_neumann if kind == 'TE' else _build_laplacian_2d_dirichlet
    matrix = builder(*shape, .002, .003).toarray()
    values, vectors = np.linalg.eigh(matrix)
    index = 1 if kind == 'TE' else 0
    profiles = _scalar_eigenmodes_to_vector(values[index:index+1], vectors[:, index:index+1],
                                           *shape, .002, .003, kind, .014, .021)
    ey, ez, hy, hz, kc2 = profiles[0]
    assert kc2 > 0
    np.testing.assert_allclose(np.sum(ey**2 + ez**2)*.002*.003, 1.)
    zero = ey if (axis == 1 and kind == 'TE') or (axis == 0 and kind == 'TM') else ez
    np.testing.assert_array_equal(zero, 0.)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('kind', ['TE', 'TM'])
def test_analytic_profiles_singleton_spacing_fallback(axis, kind):
    from rfx.sources.waveguide_port import _te_mode_profiles, _tm_mode_profiles
    widths = [np.full(7, .001), np.array([.001])]
    if axis == 0:
        widths.reverse()
    coords = [np.cumsum(w) - w/2 for w in widths]
    mode = ((0, 1) if axis == 0 else (1, 0)) if kind == 'TE' else (1, 1)
    solver = _te_mode_profiles if kind == 'TE' else _tm_mode_profiles
    ey, ez, hy, hz = solver(*(w.sum() for w in widths), *mode, *coords)
    assert np.isfinite([ey, ez, hy, hz]).all()
    np.testing.assert_allclose(np.sum(ey**2 + ez**2)*1e-6, 1.)


@pytest.mark.parametrize('axis', [0, 1])
def test_coax_edge_profile_singleton_axis(axis):
    from types import SimpleNamespace
    from rfx.sources.coaxial_port import _coaxial_tem_edge_profile
    shape = (1, 9, 2) if axis == 0 else (9, 1, 2)
    center = (0, 4, 0) if axis == 0 else (4, 0, 0)
    grid = SimpleNamespace(shape=shape, dx=.001, position_to_index=lambda _: center)
    port = SimpleNamespace(position=(0., 0., 0.))
    metal = np.zeros(shape, bool)
    if axis == 0:
        metal[:, [0, 4, 8], :] = True
    else:
        metal[[0, 4, 8], :, :] = True
    ex, ey = _coaxial_tem_edge_profile(grid, port, 0, .004, pec_cell_mask=metal)
    assert np.isfinite([ex, ey]).all()
    np.testing.assert_array_equal(ex if axis == 0 else ey, 0.)
    assert np.any(ey if axis == 0 else ex)

"""A vacuum pulse sees 4/8 mm y absorbers, or unequal pads and a PEC z wall.
The built domain must keep every requested depth. On constant spacing the
same source and off-plane probe must give the uniform-grid waveform.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.nonuniform import make_nonuniform_grid
from tests._absorber_witness import plane_wave_reflection

FACES = tuple(f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi'))
LANES = ('uniform', 'nu_run', 'nu_forward',
         pytest.param('nu_distributed', marks=pytest.mark.distributed))


def _build(case, lane, scale=1):
    if case == 'y4_8':
        domain = (.025, .024, .024)
        depths = (8, 8, 4, 8, 8, 8)
        bc = BoundarySpec(x='cpml', y=Boundary(lo='cpml', hi='cpml',
                          lo_thickness=4*scale, hi_thickness=8*scale), z='cpml')
    else:
        domain = (.0253, .0237, .0226)
        depths = (3, 7, 5, 8, 0, 6)
        bc = BoundarySpec(
            x=Boundary(lo='cpml', hi='cpml', lo_thickness=3, hi_thickness=7),
            y=Boundary(lo='cpml', hi='cpml', lo_thickness=5, hi_thickness=8),
            z=Boundary(lo='pec', hi='cpml', hi_thickness=6))
    sim = Simulation(freq_max=15e9, domain=domain, dx=1e-3,
                     cpml_layers=8*scale, boundary=bc,
                     **({'dx_profile': np.full(round(domain[0]/1e-3), 1e-3)}
                        if lane != 'uniform' else {}))
    sim.add_source((.007, .009, .010), 'ez', amplitude_kind='field')
    sim.add_probe((.011, .006, .013), 'ez')
    return sim, dict(zip(FACES, (d*scale for d in depths)))


def _run(sim, lane, steps):
    if lane in ('uniform', 'nu_run'):
        return sim.run(n_steps=steps)
    if lane == 'nu_distributed':
        devices = jax.devices('cpu')[:2]
        if len(devices) < 2:
            pytest.skip('requires two host devices')
        return sim.forward(n_steps=steps, distributed=True, devices=devices)
    return sim.forward(n_steps=steps)


def _assert_depths(grid, expected):
    # Inspect padding and node coordinates from the grid returned by the solve;
    # do not derive the expected depths from the production resolver.
    actual = {f: int(getattr(grid, f'pad_{f}')) for f in FACES}
    assert actual == expected, f'requested {expected}; realized {actual}'
    for a in 'xyz':
        assert grid.node_of(a, 0) == pytest.approx(-expected[f'{a}_lo'] * 1e-3)
        assert grid.node_of(a, expected[f'{a}_lo']) == pytest.approx(0., abs=1e-12)
    if hasattr(grid, 'dx_arr'):
        assert grid.face_layers == expected


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('case', ('y4_8', 'two_axes_pec'))
def test_realized_face_depths(case, lane, record_property):
    sim, expected = _build(case, lane)
    result = _run(sim, lane, 2)
    _assert_depths(result.grid, expected)
    record_property('realized_shape', result.grid.shape)


@pytest.mark.parametrize('lane,scale', (
    ('nu_run', 1), ('nu_forward', 1), ('nu_run', 2), ('nu_forward', 2),
    pytest.param('nu_distributed', 1, marks=pytest.mark.distributed),
    pytest.param('nu_distributed', 2, marks=pytest.mark.distributed),
))
def test_uniform_spacing_fields_match(lane, scale, record_property):
    # Doubling the absorber explicitly covers depth sensitivity; this pins
    # cross-path parity, not a depth-independent absolute field value.
    uniform, _ = _build('y4_8', 'uniform', scale)
    nu, _ = _build('y4_8', lane, scale)
    ref = np.asarray(_run(uniform, 'uniform', 240).time_series)
    got = np.asarray(_run(nu, lane, 240).time_series)
    assert np.isfinite(got).all() and np.max(abs(ref)) > 1e-4
    residual = float(np.max(abs(got-ref)) / np.max(abs(ref)))
    record_property('relative_field_residual', residual)
    # Repeated float32 curl/CPML additions on different compiled kernels:
    # allow about ten unit roundoffs per step over this 240-step record.
    # No waveform fitting, amplitude rescaling, or time shifting is allowed.
    assert residual < 240 * 10 * np.finfo(np.float32).eps, residual


@pytest.mark.parametrize('depths', ((4, 8), (8, 4), (8, 16), (16, 8)))
@pytest.mark.parametrize('face', ('lo', 'hi'))
@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
def test_plane_wave_reflection_matches_uniform(depths, face, axis, record_property):
    freqs = (2e9, 10e9, 30e9)
    args = (max(depths), 1, face, freqs)
    uniform = plane_wave_reflection(*args, face_layers=depths, axis=axis)
    nu = plane_wave_reflection(*args, face_layers=depths, axis=axis, nonuniform=True)
    record_property('uniform_R_dB', uniform.tolist())
    record_property('nu_R_dB', nu.tolist())
    # Compare linear reflection amplitudes: 32 float32 unit roundoffs of
    # incident amplitude cover the two reference subtractions at weak R.
    # The relative term covers propagation roundoff on resolved returns.
    np.testing.assert_allclose(10**(nu/20), 10**(uniform/20),
                               rtol=1e-3, atol=32*np.finfo(np.float32).eps)


@pytest.mark.parametrize('depth', (-1, 2.5, 9))
def test_unrepresentable_depth_names_face_and_budget(depth):
    with pytest.raises(ValueError, match=r"y_lo.*cpml_layers=8"):
        make_nonuniform_grid((.025, .024), np.full(24, .001), .001,
                             cpml_layers=8, face_layers={'y_lo': depth})


def test_traced_mesh_keeps_static_face_depths():
    # Move one interior cell only; the boundary spacings and pad depths are
    # fixed. The derivative of the high wall position is exactly one metre
    # per metre of that cell's width, independent of the asymmetric pads.
    def high_wall(width):
        dz = jnp.full(11, .001).at[4].set(width)
        grid = make_nonuniform_grid((.013, .017), dz, .001,
                                   cpml_layers=8, face_layers={'z_lo': 3, 'z_hi': 7})
        assert (grid.pad_z_lo, grid.pad_z_hi) == (3, 7)
        return grid.node_of('z', grid.nz-1)
    value, derivative = jax.jit(jax.value_and_grad(high_wall))(jnp.float32(.0007))
    assert float(value) == pytest.approx(.0177, abs=5e-9)
    assert float(derivative) == pytest.approx(1., abs=1e-7)


@pytest.mark.distributed
@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
def test_thin_faces_fit_below_the_scalar_budget(axis, record_property):
    # One/two active layers around a 2 mm interior give just six nodes,
    # fewer than the budget of eight; the budget must not widen the box or
    # dictate the shape of every face's distributed correction window.
    domain = [.025, .024, .024]
    source, probe = [.007, .009, .010], [.011, .006, .013]
    index = 'xyz'.index(axis)
    domain[index] = .002
    source[index] = probe[index] = .001
    bounds = dict(x='cpml', y='cpml', z='cpml')
    bounds[axis] = Boundary(lo='cpml', hi='cpml', lo_thickness=1, hi_thickness=2)
    sim = Simulation(freq_max=15e9, domain=tuple(domain), dx=.001,
                     dx_profile=np.full(round(domain[0]/.001), .001),
                     cpml_layers=8, boundary=BoundarySpec(**bounds))
    sim.add_source(tuple(source), 'ey' if axis == 'z' else 'ez', amplitude_kind='field')
    sim.add_probe(tuple(probe), 'ey' if axis == 'z' else 'ez')
    single = _run(sim, 'nu_forward', 100)
    multi = _run(sim, 'nu_distributed', 100)
    assert single.grid.shape[index] == multi.grid.shape[index] == 6
    a, b = np.asarray(single.time_series), np.asarray(multi.time_series)
    assert np.isfinite(b).all() and abs(a).max() > 1e-4
    residual = float(abs(a-b).max()/abs(a).max())
    record_property('relative_field_residual', residual)
    assert residual < 1e-4, residual

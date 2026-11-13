"""B-grid traced values and analytic gradients, rfx#2613.

T-jit enters make_nonuniform_grid as test_grad_wrt_dx_profile_flows does.
Both profiles are differentiated; y reverses G-x's declared profile. No field
steps run here. Bars and sign conventions live in float64_grid_reference.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.cpml import init_cpml
from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
from rfx.model import NominalProfile
from rfx.model.graded_admission import host_cells
from rfx.nonuniform import make_nonuniform_grid
from tests._x64_compat import enable_x64
from tests.contracts.float64_grid_reference import (
    EPS0, MU0, FACES, QUANTITIES, padded, inverse_reference,
    assert_scaled, assert_spacings, check_profile, gradient_reference,
)

X = np.r_[.5, np.linspace(.25, .75, 78), .5]*1e-3
Y = X[::-1].copy()
Z = np.full(52, .5e-3)
PAD = 8


def traced_grid(x, y, *, dtype):
    return make_nonuniform_grid((.04, .04), Z, .5e-3, PAD,
                               dx_profile=x, dy_profile=y, dtype=dtype)


def jit_arm(x, y, *, dtype):
    grid = traced_grid(x, y, dtype=dtype)
    profiles, _ = init_cpml(grid, field_dtype=dtype)
    coords = coords_from_nonuniform_grid(grid)
    return dict(widths=(grid.dx_arr, grid.dy_arr),
                metrics=((grid.inv_dx, grid.inv_dx_h), (grid.inv_dy, grid.inv_dy_h)),
                nodes=(coords.x, coords.y), dt=grid.dt, profiles=profiles)


def weights(dtype=np.float64):
    rng = np.random.default_rng(2613)
    return tuple(rng.uniform(.5, 1.5, (2, len(padded(p)))).astype(dtype) for p in (X, Y))


def grad_arm(x, y, *, dtype):
    pairs = tuple(jnp.asarray(w, dtype=dtype) for w in weights())

    def loss(x, y):
        grid = traced_grid(x, y, dtype=dtype)
        return (jnp.sum(pairs[0][0]*grid.inv_dx)+jnp.sum(pairs[0][1]*grid.inv_dx_h)
                +jnp.sum(pairs[1][0]*grid.inv_dy)+jnp.sum(pairs[1][1]*grid.inv_dy_h))

    return jax.grad(loss, argnums=(0, 1))(x, y)


@pytest.fixture(scope='module')
def traced_values():
    print(f'B_GRID_DEVICES count={len(jax.devices())} devices={jax.devices()}')
    with enable_x64():
        inputs = tuple(jnp.asarray(p, dtype=jnp.float64) for p in (X, Y))
        values = jax.jit(lambda x, y: jit_arm(x, y, dtype=jnp.float64))(*inputs)
        gradients = jax.jit(lambda x, y: grad_arm(x, y, dtype=jnp.float64))(*inputs)
        jax.block_until_ready((values, gradients))
    yield values, gradients
    jax.clear_caches()


@pytest.mark.parametrize('axis', [0, 1], ids=['x', 'y'])
def test_t_jit_widths(traced_values, axis):
    actual = np.asarray(traced_values[0]['widths'][axis])
    assert actual.dtype == np.dtype('float64')
    np.testing.assert_array_equal(actual, padded((X, Y)[axis]))


@pytest.mark.parametrize('axis', [0, 1], ids=['x', 'y'])
@pytest.mark.parametrize('kind', [0, 1], ids=['E', 'H'])
def test_t_jit_metrics(traced_values, axis, kind):
    expected = inverse_reference(padded((X, Y)[axis]))[kind]
    assert_spacings(traced_values[0]['metrics'][axis][kind], expected, 2)


@pytest.mark.parametrize('axis', [0, 1], ids=['x', 'y'])
def test_t_jit_coordinates(traced_values, axis):
    w = padded((X, Y)[axis])
    cumulative = np.r_[0., np.cumsum(w)]
    assert_scaled(traced_values[0]['nodes'][axis], cumulative[:-1]-cumulative[PAD], len(w))


def test_t_jit_dt(traced_values):
    c0 = 1/np.sqrt(EPS0*MU0)
    expected = .99/(c0*np.sqrt(sum(1/p.min()**2 for p in (X, Y, Z))))
    assert_scaled(traced_values[0]['dt'], expected, 8)


@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('magnetic', [False, True], ids=['E', 'H'])
@pytest.mark.parametrize('quantity', QUANTITIES)
def test_t_jit_cpml(traced_values, face, magnetic, quantity):
    values = traced_values[0]
    profiles = values['profiles'].magnetic if magnetic else values['profiles']
    check_profile(getattr(profiles, face), quantity, dt=values['dt'],
                  width=.5e-3, n=PAD, high=face.endswith('hi'), magnetic=magnetic)


@pytest.mark.parametrize('axis', [0, 1], ids=['x', 'y'])
def test_t_grad(traced_values, axis):
    a, b = weights()[axis]
    expected = gradient_reference((X, Y)[axis], a, b)
    assert_scaled(traced_values[1][axis], expected, max(32, 3*(PAD+1)+4))


def test_t_window():
    nominal = np.array([1., 1.25, .75, 1.])

    def arm(x):
        grid = make_nonuniform_grid(
            (4., 3.), np.ones(3), 1., 2, dx_profile=x, dtype=jnp.float64,
            face_layers={'x_lo': 1, 'x_hi': 2},
            nominal_profiles={'x': NominalProfile(nominal, (1, 3))})
        return host_cells(grid, 'x', store=True)

    with enable_x64():
        actual = np.asarray(jax.jit(arm)(jnp.asarray(nominal)))
    assert actual.dtype == np.dtype('float64')
    np.testing.assert_array_equal(actual, padded(nominal, 1, 2))


def test_noop_cpml():
    # G-x, with concrete y widths as in the invariant declaration.
    with enable_x64():
        grid = make_nonuniform_grid((.04, .03), Z, .5e-3, PAD,
                                   dx_profile=X, dtype=jnp.float64)
        profiles, _ = init_cpml(grid, pec_faces={'x_lo'}, field_dtype=jnp.float64)
    for family in (profiles, profiles.magnetic):
        for face in FACES:
            for q in QUANTITIES:
                actual = np.asarray(getattr(getattr(family, face), q))
                assert actual.dtype == np.dtype('float64'), (face, q, actual.dtype)
                if face == 'x_lo':
                    np.testing.assert_array_equal(actual, np.full(PAD, float(q in ('kappa', 'b'))))


@pytest.mark.parametrize('direction', ['+x', '-x', '+y', '-y'])
def test_msl_owner_float64(direction):
    from tests.contracts.test_measurement_plan import test_msl_calculator_setup_and_projector
    with enable_x64():
        test_msl_calculator_setup_and_projector('graded', direction, precision='float64')


def oblique_bloch_cpml_run(*, n_steps=2):
    """Read the realized profiles and complex state on the public run path."""
    import sys
    from rfx import Simulation

    observed = []

    def observe(frame, event, value):
        if (event == 'return' and frame.f_code.co_name == 'init_cpml'
                and frame.f_globals.get('__name__') == 'rfx.boundaries.cpml'):
            observed.append((frame.f_locals['field_dtype'], value))

    with jax.enable_x64(False):
        sim = Simulation(freq_max=5e9, domain=(.12, .12, .006), dx=.004,
                         boundary='cpml', cpml_layers=4, precision='float32')
        sim.add_tfsf_source(f0=5e9, bandwidth=.2, margin=3,
                            polarization='ez', angle_deg=30.)
        sim.add_probe((.04, .06, .003), 'ez')
        previous = sys.getprofile()
        try:
            sys.setprofile(observe)
            result = sim.run(n_steps=n_steps, compute_s_params=False)
            jax.block_until_ready(result)
        finally:
            sys.setprofile(previous)
    assert observed
    return observed, result


def test_float32_oblique_bloch_real_cpml_profiles():
    observed, _ = oblique_bloch_cpml_run()
    for field_dtype, (profiles, psi) in observed:
        assert jnp.dtype(field_dtype) == jnp.dtype('complex64'), 'precondition'
        for array in jax.tree_util.tree_leaves(psi):
            assert array.dtype == jnp.complex64
        for family in (profiles, profiles.magnetic):
            for face in FACES:
                for quantity in QUANTITIES:
                    array = getattr(getattr(family, face), quantity)
                    assert array.dtype == jnp.float32, (face, quantity, array.dtype)

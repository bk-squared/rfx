"""PMC high ghosts are held by coefficients, with identical CPU field records."""
import os
import platform

os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=2')

import jax
import numpy as np
import pytest

from rfx import Simulation, Box, GaussianPulse, SnapshotSpec
from rfx.boundaries.spec import Boundary, BoundarySpec

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


def scene(*, axis='x', cpml=False, graded=False, absorber='cpml'):
    boundary = dict.fromkeys('xyz', absorber if cpml else 'pec')
    boundary[axis] = Boundary('pec', 'pmc')
    # Boundary cells retain dx; the extra declared length belongs inside.
    profiles = {'dx_profile': np.array([.001] * 8 + [.0013] + [.001] * 8)} if graded else {}
    sim = Simulation(freq_max=12e9, domain=(.0173, .0081, .0072), dx=.001,
                     boundary=BoundarySpec(**boundary), snap="declared", cpml_layers=2 if cpml else 0,
                     **profiles)
    sim.add_material('slab', eps_r=4., sigma=.01, mu_r=1.5)
    sim.add(Box((.012, .001, .001), (.017, .007, .006)), material='slab')
    position = [.003, .003, .002]
    position['xyz'.index(axis)] = {'x': .016, 'y': .007, 'z': .006}[axis]
    sim.add_source(tuple(position), 'ey' if axis == 'z' else 'ez', amplitude_kind='current',
                   waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    sim.add_probe((.014, .005, .003), 'ez')
    return sim


def record(sim, devices):
    kwargs = dict(devices=jax.devices('cpu')[:devices], compute_s_params=False, skip_preflight=True)
    if sim._uses_nonuniform_mesh:
        # Graded run() exposes Result.state, not SnapshotSpec. Observe each step.
        rows = {c: [] for c in FIELDS}
        for steps in range(1, 41):
            result = sim.run(n_steps=steps, **kwargs)
            for c in FIELDS:
                rows[c].append(np.asarray(getattr(result.state, c)))
        return {c: np.stack(rows[c]) for c in FIELDS}
    result = sim.run(n_steps=80, snapshot=SnapshotSpec(interval=1, components=FIELDS), **kwargs)
    return {c: np.asarray(result.snapshots[c]) for c in FIELDS}


def check_ghosts(records, axis, label):
    for c in 'xyz':
        if c != axis:
            ghost = np.take(records['h' + c], -1, axis='xyz'.index(axis) + 1)
            assert np.count_nonzero(ghost) == 0, f'{label}: h{c} {axis}_hi ghost must be exactly zero at every step'


@pytest.fixture(scope='module')
def pec_box_records():
    assert len(jax.devices('cpu')) >= 2, 'always-on PMC pin requires two CPU host devices'
    records = record(scene(), 1), record(scene(), 2)
    jax.clear_caches()
    return records


def test_pmc_high_ghost_zero_on_one_and_two_devices(pec_box_records):
    one, slabs = pec_box_records
    check_ghosts(one, 'x', 'one device')
    check_ghosts(slabs, 'x', 'slabs')


@pytest.mark.xfail(platform.machine() in ('x86_64', 'AMD64'), strict=True,
                   reason='#2086 uniform multi-device PMC face: 1- and 2-device fields differ on x86 CPU')
def test_pmc_high_ghost_bit_identity(pec_box_records):
    one, slabs = pec_box_records
    for c in FIELDS:
        assert one[c].tobytes() == slabs[c].tobytes(), f'{c}: every snapshot must be bit-identical on 1 and 2 devices'


@pytest.mark.parametrize('axis', tuple('xyz'))
@pytest.mark.parametrize('graded', (False, True))
@pytest.mark.parametrize('devices', (1, 2))
def test_pmc_high_ghost_with_transverse_cpml(axis, graded, devices):
    check_ghosts(record(scene(axis=axis, cpml=True, graded=graded), devices), axis,
                 f'cpml graded={graded} devices={devices}')
    jax.clear_caches()


def tfsf_scene(*, axis='x', graded=False):
    # The uniform TF/SF lane solves its side faces periodic and refuses declared ones (#2127).
    boundary = dict.fromkeys('xyz', 'cpml') if graded else {'x': 'cpml', 'y': 'periodic', 'z': 'periodic'}
    # The uniform +x plane wave needs its absorber on x_lo (a closed x axis leaves it none).
    boundary[axis] = Boundary('pec' if graded or axis != 'x' else 'cpml', 'pmc')
    profiles = {'dx_profile': np.array([.001] * 8 + [.0013] + [.001] * 8)} if graded else {}
    # A vacuum scene meets TF/SF's transverse and injection-plane conditions.
    # Periodic y and z need whole cells: 0.9 mm divides 8.1 and 7.2 mm (x stays a fractional count).
    return Simulation(freq_max=12e9, domain=(.0173, .0081, .0072), dx=.001 if graded else .0009,
                      boundary=BoundarySpec(**boundary), snap='declared', cpml_layers=2,
                      **profiles)


@pytest.mark.parametrize('graded', (False, True))
def test_pmc_high_ghost_with_tfsf(graded):
    sim = tfsf_scene(graded=graded)
    sim.add_tfsf_source(f0=5e9, bandwidth=.8, polarization='ez', direction='+x', margin=2)
    check_ghosts(record(sim, 1), 'x', f'tfsf graded={graded}')
    jax.clear_caches()


@pytest.mark.parametrize('graded', (False, True))
@pytest.mark.parametrize('axis', ('y', 'z'))
def test_tfsf_refuses_mixed_transverse_pmc_pair(axis, graded):
    reason = f'TF/SF {axis}_lo, {axis}_hi must be a pair' if graded else 'TF/SF on the uniform mesh solves axes'
    with pytest.raises(ValueError, match=reason):
        sim = tfsf_scene(axis=axis, graded=graded)
        sim.add_tfsf_source(f0=5e9, bandwidth=.8, polarization='ez', margin=2)
        record(sim, 1)


@pytest.mark.parametrize('graded', (False, True))
def test_closed_box_tfsf_refuses_pmc(graded):
    sim = tfsf_scene(graded=graded)
    reason = 'require a uniform, 3D second-order Yee' if graded else 'requires CPML on all six domain faces'
    with pytest.raises(NotImplementedError, match=reason):
        sim.add_tfsf_source(f0=5e9, bandwidth=.8, closed_box=True, margin=2)


@pytest.mark.parametrize('axis', tuple('xyz'))
def test_pmc_high_ghost_with_transverse_upml(axis):
    check_ghosts(record(scene(axis=axis, cpml=True, absorber='upml'), 1), axis, 'upml')
    jax.clear_caches()


def seeded_run(monkeypatch, graded, devices):
    """One step from whole arrays seeded to hy = +7, hz = -7 and ez = 1."""
    import jax.numpy as jnp

    original_scan = jax.lax.scan
    seeded = []

    def scan(function, initial, *args, **kwargs):
        def is_state(value):
            return all(hasattr(value, c) for c in FIELDS)

        def seed(value):
            if is_state(value):
                seeded.append(value.hy.shape)
                # Seed whole H arrays so this covers physical ghosts even in
                # packed slabs, without using production slab/index helpers.
                # Ez = 1 gives the high ghost Hy a nonzero x curl, so only its
                # update coefficient or inverse spacing can hold it.
                return value._replace(hy=jnp.full_like(value.hy, 7.),
                                      hz=jnp.full_like(value.hz, -7.),
                                      ez=jnp.ones_like(value.ez))
            return value

        initial = jax.tree_util.tree_map(seed, initial, is_leaf=is_state)
        return original_scan(function, initial, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(jax.lax, 'scan', scan)
        result = scene(graded=graded).run(n_steps=1, devices=jax.devices('cpu')[:devices],
                                        compute_s_params=False, skip_preflight=True)
    assert seeded, 'initial H carry was not injected'
    jax.clear_caches()
    return {c: np.asarray(getattr(result.state, c)) for c in FIELDS}


@pytest.mark.parametrize('devices', (1, 2))
def test_nonzero_initial_pmc_high_ghost_is_positive_zero(monkeypatch, devices):
    """Uniform: the realized row must clear any carry, including a negative ghost."""
    state = seeded_run(monkeypatch, False, devices)
    for component in ('hy', 'hz'):
        ghost = state[component][-1]
        assert np.count_nonzero(ghost.view(np.uint32)) == 0, (
            f'{component}: nonzero initial high ghost must become exactly +0.0')


@pytest.mark.parametrize('devices', (1, 2))
def test_graded_pmc_high_ghost_is_held(monkeypatch, devices):
    """Graded: the zero high inverse H spacing holds the ghost at its value.

    Physical runs start it at zero and keep it there (check_ghosts above); the
    image read cancels a zero ghost exactly.
    """
    state = seeded_run(monkeypatch, True, devices)
    for component, value in (('hy', 7.), ('hz', -7.)):
        ghost = state[component][-1]
        assert np.array_equal(ghost.view(np.uint32), np.full_like(ghost, value).view(np.uint32)), (
            f'{component}: graded high ghost must keep its initial value exactly')


def test_pmc_high_factors_match_independent_reference():
    from rfx.boundaries.pmc import magnetic_update_factors

    shape = (5, 4, 3)
    for axis, letter in enumerate('xyz'):
        factors = magnetic_update_factors({f'{letter}_hi'}, shape)
        for index, factor in enumerate(factors):
            if index == axis:
                expected = np.array([1.] * (shape[axis] - 1) + [0.], np.float32)
                np.testing.assert_array_equal(factor.view(np.uint32), expected.view(np.uint32))
            else:
                assert factor is None
    assert magnetic_update_factors({'x_lo', 'y_lo', 'z_lo'}, shape) == (None, None, None)

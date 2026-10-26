"""PMC high ghosts are held by coefficients, with identical CPU field records."""
import os

os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=2')

import jax
import numpy as np
import pytest

from rfx import Simulation, Box, GaussianPulse, SnapshotSpec
from rfx.boundaries.spec import Boundary, BoundarySpec

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


def scene(*, axis='x', cpml=False, tfsf=False, graded=False):
    boundary = dict.fromkeys('xyz', 'cpml' if cpml else 'pec')
    boundary[axis] = Boundary('pec', 'pmc')
    profiles = {'dx_profile': np.array([.001] * 16 + [.0013])} if graded else {}
    sim = Simulation(freq_max=12e9, domain=(.0173, .0081, .0072), dx=.001,
                     boundary=BoundarySpec(**boundary), snap="declared", cpml_layers=2 if cpml else 0,
                     **profiles)
    sim.add_material('slab', eps_r=4., sigma=.01, mu_r=1.5)
    sim.add(Box((.012, .001, .001), (.017, .007, .006)), material='slab')
    position = [.003, .003, .002]
    position['xyz'.index(axis)] = {'x': .016, 'y': .007, 'z': .006}[axis]
    sim.add_source(tuple(position), 'ey' if axis == 'z' else 'ez', amplitude_kind='current',
                   waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    if tfsf:
        sim.add_tfsf_source(f0=5e9, bandwidth=.8, polarization='ez', direction='+x')
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


def test_pmc_high_ghost_bit_identity():
    assert len(jax.devices('cpu')) >= 2, 'always-on PMC pin requires two CPU host devices'
    one, slabs = record(scene(), 1), record(scene(), 2)
    check_ghosts(one, 'x', 'one device')
    check_ghosts(slabs, 'x', 'slabs')
    for c in FIELDS:
        assert one[c].tobytes() == slabs[c].tobytes(), f'{c}: every snapshot must be bit-identical on 1 and 2 devices'
    jax.clear_caches()


@pytest.mark.parametrize('axis', tuple('xyz'))
@pytest.mark.parametrize('graded', (False, True))
@pytest.mark.parametrize('devices', (1, 2))
def test_pmc_high_ghost_with_transverse_cpml(axis, graded, devices):
    check_ghosts(record(scene(axis=axis, cpml=True, graded=graded), devices), axis,
                 f'cpml graded={graded} devices={devices}')
    jax.clear_caches()


@pytest.mark.parametrize('axis', tuple('xyz'))
def test_pmc_high_ghost_with_tfsf(axis):
    check_ghosts(record(scene(axis=axis, cpml=True, tfsf=True), 1), axis, 'tfsf')
    jax.clear_caches()

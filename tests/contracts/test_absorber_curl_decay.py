"""Judge 1: the absorber's own curl coefficient closes every admitted lane."""
import json

import jax
import numpy as np
import pytest

from tests.contracts._absorber_curl_scene import solve

OWNERS = ('plain', 'dielectric', 'debye', 'lorentz', 'mixed')
FACES = tuple(f'{axis}_{side}' for axis in 'xyz' for side in ('lo', 'hi'))
LANES = ('uniform_run', 'uniform_forward', 'graded_run', 'graded_forward',
         'distributed_run', 'distributed_graded_run', 'distributed_forward', 'probe_loop')
ALWAYS_ON_B = ('uniform_run', 'uniform_forward', 'graded_run',
               'distributed_run', 'distributed_graded_run')
CASES = [
    pytest.param('a', 'plain', face, lane,
                 marks=() if lane == 'uniform_run' else pytest.mark.slow)
    for face in FACES for lane in LANES + ('vmap',)
] + [
    pytest.param('b', owner, 'all', lane,
                 marks=() if lane in ALWAYS_ON_B else pytest.mark.slow)
    for owner in OWNERS
    for lane in LANES + (('vmap',) if owner in ('plain', 'dielectric') else ())
]


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize('scene,owner,face,lane', CASES)
def test_pad_slab_decays(scene, owner, face, lane, record_property):
    trace = solve(owner, face, lane, scene=scene)
    eighths = [float(np.max(np.abs(part))) for part in np.array_split(trace, 8)]
    e2, e8 = eighths[1], eighths[7]
    values = dict(scene=scene, owner=owner, face=face, lane=lane, e_2=e2, e_8=e8,
                  eighths=eighths, finite=bool(np.isfinite(trace).all()),
                  identically_zero=bool(np.all(trace == 0)))
    record_property('judge1', json.dumps(values))
    print(json.dumps(values))
    assert np.isfinite(trace).all(), f'{owner}/{face}/{lane}: nonfinite record'
    assert e8 < e2, f'{owner}/{face}/{lane}: e_8={e8} >= e_2={e2}'

"""Judge 1: the absorber's own curl coefficient closes every admitted lane."""
import functools
import json

import jax
import numpy as np
import pytest

from tests.contracts._absorber_curl_scene import solve
from tests.contracts.path_equivalence.comparison import compare

OWNERS = ('plain', 'dielectric', 'debye', 'lorentz', 'mixed')
FACES = tuple(f'{axis}_{side}' for axis in 'xyz' for side in ('lo', 'hi'))
LANES = ('uniform_run', 'uniform_forward', 'graded_run', 'graded_forward',
         'distributed_run', 'distributed_graded_run', 'distributed_forward', 'probe_loop')


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@functools.lru_cache(maxsize=30)
def reference(owner, face):
    return solve(owner, face, 'uniform_run')


@pytest.mark.parametrize('owner,face,lane', [
    (o, f, lane) for o in OWNERS for f in FACES
    for lane in LANES + (('vmap',) if o in ('plain', 'dielectric') else ())
])
def test_pad_slab_decays(owner, face, lane, record_property):
    trace = reference(owner, face) if lane == 'uniform_run' else solve(owner, face, lane)
    e2 = float(np.max(np.abs(trace[750:1500])))
    e8 = float(np.max(np.abs(trace[5250:6000])))
    values = dict(owner=owner, face=face, lane=lane, e_2=e2, e_8=e8)
    record_property('judge1', json.dumps(values))
    print(json.dumps(values))
    assert np.isfinite(trace).all(), f'{owner}/{face}/{lane}: nonfinite record'
    assert e8 < e2, f'{owner}/{face}/{lane}: e_8={e8} >= e_2={e2}'
    if lane not in ('uniform_run', 'probe_loop'):
        # S0's committed accumulated bar; vmap's existing contract is exact.
        measurements = []
        compare(trace, reference(owner, face), record='probe trace',
                kind='exact' if lane == 'vmap' else 'accumulated',
                measurements=measurements)
        record_property('path_equivalence', json.dumps(measurements))

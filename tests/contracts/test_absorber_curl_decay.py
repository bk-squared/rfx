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
# Always on: the six-face rows (scene a, uniform run) and one row per coefficient owner that grows on the base
# tree (scene b, uniform run). Every other path x owner combination is a sweep and runs weekly; the per-path
# invariant is pinned always-on by the coefficient comparison (test_absorber_curl_coefficients*.py).
ALWAYS_ON_B_OWNERS = ('plain', 'debye', 'lorentz', 'mixed')
CASES = [
    pytest.param('a', 'plain', face, lane,
                 marks=() if lane == 'uniform_run' else pytest.mark.slow)
    for face in FACES for lane in LANES + ('vmap',)
] + [
    pytest.param('b', owner, 'all', lane,
                 marks=() if (lane == 'uniform_run' and owner in ALWAYS_ON_B_OWNERS) else pytest.mark.slow)
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
    assert np.isfinite(trace).all(), f'{owner}/{face}/{lane}: nonfinite record'
    assert e8 < e2, f'{owner}/{face}/{lane}: e_8={e8} >= e_2={e2}'

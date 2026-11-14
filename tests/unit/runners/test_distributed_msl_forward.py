"""F-a: public one-device forward is the independent reference.

F-a: max_t |E_n-E_1| / max_t |E_1(2100)| per probe; bar 9e-6.
Above 9 ULP of that float32 peak is reported as a merge hold.
G-c moved to the hostmem file after its first whole case took 181.68 s.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._multi_device_scene import host_devices
from tests.unit.runners.test_distributed_msl_ports import observed_scene
from tests.unit.runners._msl_forward_bars import whole_case, emit, parameter_blocks, eps_of

pytestmark = pytest.mark.distributed


def scene_for(name, lane, mode, n):
    sim, _ = observed_scene(name, lane, mode, n, excite_all=name in ('T1A', 'Y1A'))
    if name in ('T1A', 'Y1A'):
        sim._msl_ports = [replace(p, excite=True) for p in sim._msl_ports]
    return sim


CASES = []
for lane, names in [('uniform', ('T1', 'T2', 'Y1', 'T1L', 'T1A')),
                    ('graded', ('T1', 'Y1', 'T1A'))]:
    for name in names:
        for mode in ('laplace', 'uniform'):
            for n in (2, 3):
                for length in (1050, 1400, 2100):
                    for override in (('none', 'eps', 'auto', 'sigma') if lane == 'graded' and name == 'T1A' and mode == 'laplace' else ('none',)):
                        fast = name == 'T1A' and n == 2 and length == 1050 and mode == 'laplace' and override == 'none'
                        CASES.append(pytest.param(name, lane, mode, n, length, override,
                            id=f'{name}-{lane}-{mode}-n{n}-L{length}-{override}',
                            marks=[] if fast else [pytest.mark.slow]))


@pytest.mark.parametrize('name,lane,mode,n,length,override', CASES)
def test_forward_fields(name, lane, mode, n, length, override):
    devices = host_devices(n)
    sim = scene_for(name, lane, mode, n)
    options = {}
    if override != 'none':
        base, masks = parameter_blocks(sim, name)
        if override == 'auto':
            sim._msl_ports = [replace(p, eps_r_sub=None) for p in sim._msl_ports]
            options['eps_override'] = base + .5 * masks[3]
        elif override == 'eps':
            options['eps_override'] = base + .5 * (masks[2] + masks[3])
        else:
            options['sigma_override'] = .01 * masks[2]
    longest = np.asarray(sim.forward(n_steps=2100, **options).time_series)
    expected = longest if length == 2100 else np.asarray(sim.forward(n_steps=length, **options).time_series)
    actual = np.asarray(sim.forward(n_steps=length, distributed=True, devices=devices, **options).time_series)
    peak = np.max(np.abs(longest), axis=0)
    delta = np.max(np.abs(actual - expected), axis=0)
    assert np.all(peak > 0)
    ulp = delta / np.spacing(peak.astype(np.float32))
    emit('F-a', scene=name, lane=lane, mode=mode, n=n, length=length, override=override,
         delta=delta, peak=peak, relative=delta/peak, ulp=ulp, hold=bool(np.any(ulp > 9)))
    assert np.isfinite(actual).all()
    assert np.all(delta <= 9e-6 * peak), (delta/peak, ulp)


@pytest.mark.slow
@pytest.mark.parametrize('n', [2, 3])
def test_microstrip_vmap(n):
    # The graded lumped/current-source lane carries vmap (records_vmap and
    # nu_drive_sees_override); microstrip has the same carried row.
    devices = host_devices(n)
    sim = scene_for('T1A', 'graded', 'laplace', n)
    base, masks = parameter_blocks(sim, 'T1A')
    # Match the lumped-port override row's supported batch/device layout.
    ps = jnp.linspace(0., .5, n)[:, None] * jnp.array([0., 0., 1., 1.], jnp.float32)
    def solve(p, options):
        return sim.forward(eps_override=eps_of(base, masks, p), n_steps=1050, **options).time_series
    expected = np.stack([np.asarray(solve(p, {})) for p in ps])
    actual = np.asarray(jax.vmap(lambda p: solve(p, dict(distributed=True, devices=devices)))(ps))
    peak = np.max(np.abs(expected), axis=1)
    delta = np.max(np.abs(actual-expected), axis=1)
    emit('vmap', n=n, relative=delta/peak)
    assert np.isfinite(actual).all()
    assert np.all(delta <= 9e-6*peak), delta/peak

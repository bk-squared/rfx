"""Host-memory G-a/G-b, selected by the weekly cpu-16 hostmem lane.

J[f,k] = d |S21(f)|²/dp[k], with four explicit unit-cotangent pullbacks.
R_f=max(max_k |J_1[f,k]|, 1e-3 max_f,k |J_1|); max_k |ΔJ|/R_f <= 1e-4.
The longer-record witness normalizes by each short-record bin's own peak.
It must be judged and <= 1e-2 before G-a can pass.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.ringdown import gradient_witness
from tests._multi_device_scene import host_devices
from tests.unit.runners._msl_multidevice_scene import build_scene, FREQS, report_realized
from tests.unit.runners._msl_forward_bars import (
    whole_case, emit, parameter_blocks, forward_s, jacobian, eps_of,
)

from tests.unit.runners.test_distributed_msl_forward import scene_for

pytestmark = [pytest.mark.slow, pytest.mark.distributed,
              pytest.mark.skipif(os.environ.get('RFX_HOST_MEMORY_WITNESS') != '1',
                                 reason='requires the 53 GiB weekly hostmem lane')]


@pytest.mark.parametrize('name', ['T1A_graded', 'Y1A_graded'])
@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('length', [1050, 1400])
def test_s_gradient_and_longer_record(name, n, length):
    if (n != 2 or length != 1050) and os.environ.get('RFX_MSL_FULL_MATRIX') != '1':
        pytest.skip('full matrix measured once; weekly hostmem selects n2/L1050')
    devices = host_devices(n)
    sim = build_scene(name, excite_all=True).sim
    realized = report_realized(build_scene(name, excite_all=True), n)
    assert not realized['ownership_flags'], realized
    if name.startswith('Y'):
        for port in realized['ports']:
            assert port['feed']['x_indices'] == list(range(23, 33))
            assert port['feed']['counts_per_rank'] == ([24, 16] if n == 2 else [0, 40, 0])
    else:
        assert [port['feed']['x_indices'] for port in realized['ports']] == [[18], [58]]
        assert [port['feed']['x_owners'] for port in realized['ports']] == [[0], [n - 1]]
    base, masks = parameter_blocks(sim, name)
    # Independent public call: pins assembly extraction and the removal of
    # per-step witness selection bitwise, before any derivative comparison.
    public = np.asarray(sim.compute_msl_s_matrix(eps_override=base, n_steps=length, freqs=FREQS).S)
    one_s = np.asarray(forward_s(sim, base, length))
    np.testing.assert_array_equal(one_s, public, strict=True)
    calls = []
    def spy(variant, options):
        assert options.get('distributed') is True
        assert options.get('devices') is devices
        calls.append(options)
    multi_s = np.asarray(forward_s(sim, base, length, devices, spy))
    assert len(calls) == len(sim._msl_ports)
    emit('S-sanity', scene=name, n=n, length=length, bitwise_public=True,
         max_delta=float(np.max(np.abs(multi_s-one_s))),
         S_real=multi_s.real, S_imag=multi_s.imag)
    jax.clear_caches()
    one = jacobian(sim, base, masks, length)
    multi = jacobian(sim, base, masks, length, devices)
    long_one = jacobian(sim, base, masks, 2100)
    long_multi = jacobian(sim, base, masks, 2100, devices)
    w1 = gradient_witness(one, long_one, against='longer_record', bin_axis=0)
    wn = gradient_witness(multi, long_multi, against='longer_record', bin_axis=0)
    scale = np.maximum(np.max(abs(one), axis=1), 1e-3*np.max(abs(one)))
    ratio = np.max(abs(multi-one), axis=1)/scale
    long_scale = np.maximum(np.max(abs(long_one), axis=1), 1e-3*np.max(abs(long_one)))
    long_ratio = np.max(abs(long_multi-long_one), axis=1)/long_scale
    emit('G-a/G-b', scene=name, n=n, length=length, R_f=scale,
         reference=one, actual=multi, relative=ratio,
         reference_2100=long_one, actual_2100=long_multi, relative_2100=long_ratio,
         witness_one=dict(value=float(w1.value), judged=bool(w1.judged), ok=bool(w1.ok)),
         witness_multi=dict(value=float(wn.value), judged=bool(wn.judged), ok=bool(wn.ok)))
    assert w1.judged and w1.ok and wn.judged and wn.ok, ('cannot judge G-a', w1, wn)
    assert np.all(ratio <= 1e-4), ratio
    assert np.all(long_ratio <= 1e-4), long_ratio


@pytest.mark.parametrize('n', [2, 3])
def test_field_gradient_and_jvp(n):
    if n == 3 and os.environ.get('RFX_MSL_FULL_MATRIX') != '1':
        pytest.skip('full matrix measured once; weekly hostmem selects n2')
    devices = host_devices(n)
    sim = scene_for('T1A', 'graded', 'laplace', n)
    base, masks = parameter_blocks(sim, 'T1A')
    p = jnp.zeros(4, jnp.float32)
    def objective(p, distributed):
        options = dict(distributed=True, devices=devices) if distributed else {}
        ts = sim.forward(eps_override=eps_of(base, masks, p), n_steps=1050, **options).time_series
        return jnp.log(jnp.mean(ts**2))
    one = np.asarray(jax.grad(lambda p: objective(p, False))(p))
    jax.clear_caches()
    multi = np.asarray(jax.grad(lambda p: objective(p, True))(p))
    _, tangent = jax.jvp(lambda p: objective(p, True), (p,), (jnp.array([0., 0., 1., 0.]),))
    peak = np.max(np.abs(one))
    delta = np.abs(multi-one)
    emit('G-c', n=n, reference=one, actual=multi, relative=delta/peak,
         jvp=float(tangent), jvp_relative=float(abs(tangent-one[2])/peak))
    assert np.isfinite(one).all() and np.isfinite(multi).all()
    assert np.all(one != 0)
    assert np.all(delta <= 1e-4*peak), delta/peak
    assert abs(tangent-one[2]) <= 1e-4*peak

"""A pre-injection read must preserve every distributed field and S bit."""
import sys
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx.runners import distributed_v2
from tests.contracts.path_equivalence.builders import FREQS, build


def _scene(kind, boundary):
    if boundary == 'pec':
        with patch('tests.contracts.path_equivalence.builders.boundary', return_value='pec'):
            return build(('_ports', kind + '_port'), 'run_uniform')
    return build(('_ports', kind + '_port'), 'run_uniform')


def _signature(result):
    return [(r.id, r.kind, r.channels, r.samples.shape) for r in result.sparam_time_records]


@pytest.mark.parametrize('kind', ['lumped', 'wire'])
@pytest.mark.parametrize('boundary', ['cpml', 'pec'])
def test_reference_read_preserves_fields_and_s(kind, boundary):
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('Requires XLA_FLAGS=--xla_force_host_platform_device_count=2')
    devices = devices[:2]
    runner = distributed_v2.run_distributed
    fields = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')

    def run(sample):
        states = []

        def capture(*args, **kwargs):
            result = runner(*args, **kwargs, _record_vref=sample)
            # Expected arrays come from the actual unsampled run, including
            # each isolated S drive; no record/DFT helper constructs them.
            states.append({name: np.array(getattr(result.state, name)) for name in fields})
            return result

        with patch.object(distributed_v2, 'run_distributed', capture):
            result = _scene(kind, boundary).run(
                n_steps=12, devices=devices, s_param_freqs=FREQS, skip_preflight=True)
        return result, states

    expected, baseline = run(False)
    actual, sampled = run(True)
    assert len(baseline) == len(sampled) == 2  # main run and S driver
    differences = []
    for drive, (old, new) in enumerate(zip(baseline, sampled, strict=True)):
        for name in fields:
            if not np.array_equal(old[name], new[name]):
                differences.append(f'drive={drive} {name}: max_abs={np.max(np.abs(old[name] - new[name]))}')
    for name in fields:
        if not np.array_equal(getattr(expected.state, name), getattr(actual.state, name)):
            differences.append(f'public final state: {name}')
    assert np.isfinite(np.asarray(actual.s_params)).all()
    if not np.array_equal(expected.s_params, actual.s_params):
        differences.append(f'S: max_abs={np.max(np.abs(expected.s_params - actual.s_params))}')
    for difference in differences:
        print(f'{kind}/{boundary}: {difference}', file=sys.stderr)
    assert not differences, differences

    one = _scene(kind, boundary).run(n_steps=12, s_param_freqs=FREQS, skip_preflight=True)
    assert _signature(actual) == _signature(one)
    for record in expected.sparam_time_records:
        assert not np.any(np.asarray(record['V_ref']))
    for record in actual.sparam_time_records:
        # Initially every field is zero. The first source injection is
        # nonzero: sampling after it cannot satisfy this independent oracle.
        assert record['V_ref'][0] == 0
        assert record['V'][0] != 0
        assert np.any(np.asarray(record['V_ref'][1:]) != 0)
    if kind == 'wire':
        assert len(actual.wire_port_sparams) == len(one.wire_port_sparams)
        for (meta, accs), (one_meta, one_accs) in zip(
                actual.wire_port_sparams, one.wire_port_sparams, strict=True):
            assert meta._fields == one_meta._fields
            for name in meta._fields:
                assert np.array_equal(getattr(meta, name), getattr(one_meta, name)), name
            assert [a.shape for a in accs] == [a.shape for a in one_accs]
    print(f'{kind}/{boundary}: main and S-drive six fields and S array_equal; '
          f'records={_signature(actual)}', file=sys.stderr)


@pytest.mark.parametrize('kind', ['lumped', 'wire'])
def test_records_without_s_ignore_s_options(kind):
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('Requires XLA_FLAGS=--xla_force_host_platform_device_count=2')
    options = dict(n_steps=12, compute_s_params=False, skip_preflight=True,
                   s_param_freqs=object(), s_param_n_steps=36)
    one = _scene(kind, 'cpml').run(**options)
    multi = _scene(kind, 'cpml').run(devices=devices[:2], **options)
    assert _signature(multi) == _signature(one)
    assert multi.s_params is None and multi.freqs is None
    assert multi.wire_port_sparams is None
    assert multi.time_series.shape == one.time_series.shape == (12, 2)


@pytest.mark.parametrize('kind', ['lumped', 'wire'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_records_inside_jit_match_direct_call(kind, entry):
    """Cross-trace: jitted port records within 9 float32 ULP of each channel's peak."""
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('Requires XLA_FLAGS=--xla_force_host_platform_device_count=2')
    devices = devices[:2]
    sim = _scene(kind, 'cpml')

    def call():
        if entry == 'run':
            result = sim.run(n_steps=12, devices=devices, compute_s_params=False,
                             skip_preflight=True)
        else:
            result = sim.forward(n_steps=12, distributed=True, devices=devices,
                                 skip_preflight=True)
        return tuple(record.samples for record in result.sparam_time_records)

    # Expected values: the direct (untraced) public call.
    direct = [np.asarray(samples) for samples in call()]
    jitted = [np.asarray(samples) for samples in jax.jit(call)()]
    assert direct and len(jitted) == len(direct)
    for expected, actual in zip(direct, jitted, strict=True):
        assert actual.shape == expected.shape
        for channel in range(expected.shape[1]):
            peak = np.float32(np.max(np.abs(expected[:, channel])))
            assert peak > 0
            ulp = float(np.spacing(peak))
            error = float(np.max(np.abs(actual[:, channel].astype(np.float64)
                                        - expected[:, channel])))
            assert error <= 9 * ulp, (entry, kind, channel, error / ulp)

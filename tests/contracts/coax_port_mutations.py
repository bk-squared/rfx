"""Execute S1's requested check-off and restored-defect mutations in isolation."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mutation,expected', [('a', 0), ('b', 1), ('bprime', 1)])
def test_s1_mutation(mutation, expected):
    common = '''
import pytest
import tests.contracts.test_coax_port_model_and_drive as contract
'''
    changes = {
        'a': '''
contract.check_bore = lambda *args: None
selection = 'test_bore_clears_plate'
''',
        'b': '''
from dataclasses import replace
import rfx.model.coax_port as builder
original = builder.clear_coax_bore
def skip_clearing(root, *args, **kwargs):
    report = original(root, *args, **kwargs)
    return replace(report, pec_cells=root.pec_cells, pec_edges=root.pec_edges)
builder.clear_coax_bore = skip_clearing
selection = 'test_bore_clears_plate'
''',
        'bprime': '''
from unittest.mock import patch
from rfx import Simulation
import rfx.simulation as simulation
original = Simulation._forward_from_materials
def drop_magnetic(self, *args, **kwargs):
    runner = simulation.run
    def without_magnetic(*args, **kwargs):
        kwargs['mag_sources'] = []
        return runner(*args, **kwargs)
    with patch.object(simulation, 'run', without_magnetic):
        return original(self, *args, **kwargs)
Simulation._forward_from_materials = drop_magnetic
selection = 'test_forward_against_run'
''',
    }
    command = common + changes[mutation] + '''
raise SystemExit(pytest.main(['tests/contracts/test_coax_port_model_and_drive.py',
                             '-q', '-o', 'addopts=', '-s', '-k', selection]))
'''
    result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    print(f'S1 mutation {mutation}: exit={result.returncode}\n{result.stdout}\n{result.stderr}',
          file=sys.stderr)
    assert result.returncode == expected
    if expected:
        assert ('AssertionError' in result.stdout
                or (mutation == 'b' and 'coax voltage ray has no realized outer conductor' in result.stdout))
        assert ('test_forward_against_run' if mutation == 'bprime' else 'test_bore_clears_plate') in result.stdout


@pytest.mark.parametrize('mutation,expected', [('a', 0), ('b', 1)])
def test_admission_mutation(mutation, expected):
    common = '''
import pytest
import tests.contracts.test_coax_port_refused_by_admission as contract
'''
    changes = {
        'a': '''
contract.check_refusal = lambda *args: None
''',
        'b': '''
from rfx.runners._admission import LANE_GATES
for lane in ('run_uniform', 'fwd_uniform'):
    LANE_GATES[lane][('_coaxial_ports', 'coax_port')].pop("precision='float32'")
''',
    }
    command = common + changes[mutation] + '''
raise SystemExit(pytest.main(['tests/contracts/test_coax_port_refused_by_admission.py',
                             '-q', '-o', 'addopts=']))
'''
    result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    print(f'admission mutation {mutation}: exit={result.returncode}\n{result.stdout}\n{result.stderr}',
          file=sys.stderr)
    assert result.returncode == expected
    if expected:
        assert 'DID NOT RAISE' in result.stdout
        assert 'precision-float32-run' in result.stdout
        assert 'precision-float32-forward' in result.stdout


@pytest.mark.parametrize('mutation,expected', [('a', 0), ('b', 1), ('bprime', 1),
                                             ('wave_a', 0), ('wave_b', 1)])
def test_measurement_mutation(mutation, expected):
    common = '''
import pytest
import tests.contracts.test_coax_port_s as contract
'''
    changes = {
        'a': '''
contract.check_native = lambda *args: None
selection = 'test_open_native_and_renormalization'
''',
        'b': '''
import rfx.measurement.coax_port as measurement
from rfx.measurement.plan import Owner
measurement.port_owner = lambda *args: Owner('coax_port:0:mode:0', 'coax', ())
selection = 'test_open_native_and_renormalization'
''',
        'bprime': '''
import jax.numpy as jnp
import rfx.measurement.coax_port as measurement
original = measurement.port_result
def from_stopped_records(sim, product, result, **kwargs):
    owner, = (o for o in product._prepared_solve.measurement.owners if o.kind == 'coax')
    samples = next(r.samples for r in result.sparam_time_records if r.kind == 'coax')
    times = jnp.arange(samples.shape[0]) * result.dt
    phases = jnp.exp(-2j * jnp.pi * times[:, None] * jnp.asarray(owner.frequencies)[None, :])
    result = result._replace(coax_sparam_accs=samples.T @ phases)
    return original(sim, product, result, **kwargs)
measurement.port_result = from_stopped_records
selection = 'test_occupancy_gradient'
''',
    }
    if mutation.startswith('wave_'):
        changes[mutation] = '''
import rfx.measurement.coax_port as measurement
original = measurement.port_result
def raw_voltage(*args, **kwargs):
    s11, frequencies, metadata = original(*args, **kwargs)
    for data in metadata.values():
        data['forward_voltage'] = data['plane_voltages'][0]
    return s11, frequencies, metadata
measurement.port_result = raw_voltage
selection = 'test_wave_metadata'
'''
        if mutation == 'wave_a':
            changes[mutation] += '''
original_assert = contract.np.testing.assert_allclose
def check_off(actual, desired, **kwargs):
    if contract.np.shape(actual) != (2, 3):
        original_assert(actual, desired, **kwargs)
contract.np.testing.assert_allclose = check_off
'''
    script = common + changes[mutation] + '''
raise SystemExit(pytest.main(['tests/contracts/test_coax_port_s.py', '-q', '-o', 'addopts=', '-s', '-k', selection]))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    print(f'measurement mutation {mutation}: exit={result.returncode}', file=sys.stderr)
    print(result.stdout + result.stderr, file=sys.stderr)
    assert result.returncode == expected
    if expected:
        assert ('AssertionError' if mutation in ('bprime', 'wave_b') else 'NoneType') in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s", "-o", "addopts="]))

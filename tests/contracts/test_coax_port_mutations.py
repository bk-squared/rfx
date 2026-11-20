"""Execute S1's requested check-off and restored-defect mutations in isolation."""
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
                             '-q', '-s', '-k', selection]))
'''
    result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True)
    print(f'S1 mutation {mutation}: exit={result.returncode}\n{result.stdout}\n{result.stderr}',
          file=sys.stderr)
    assert result.returncode == expected
    if expected:
        assert 'AssertionError' in result.stdout
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
    result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True)
    print(f'admission mutation {mutation}: exit={result.returncode}\n{result.stdout}\n{result.stderr}',
          file=sys.stderr)
    assert result.returncode == expected
    if expected:
        assert 'DID NOT RAISE' in result.stdout
        assert 'precision-float32-run' in result.stdout
        assert 'precision-float32-forward' in result.stdout

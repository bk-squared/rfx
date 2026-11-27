"""Exercise vector-padding pins and structural contracts with isolated mutations."""
import os
import subprocess
import sys

import pytest


PIN_FILE = 'tests/unit/runners/test_vector_pad_paths.py'
CONTRACT_FILE = 'tests/contracts/test_vector_pad_carry_rules.py'
HIGH_END_VIEW = '''
from pathlib import Path
import rfx.stepping.uniform as uniform
def high_view(state, plan):
    if plan is None:
        return state
    slices = tuple(slice(p, p+n) for n, p in zip(plan.shape, plan.pad))
    return state._replace(**{c: getattr(state, c)[slices]
                             for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')})
uniform._mutant_high_view = high_view
source = Path(uniform.__file__).read_text()
old = 'update_coax(frame.carry, view(frame.st),'
assert source.count(old) == 1
source = source.replace(old, 'update_coax(frame.carry, _mutant_high_view(frame.st, pad_plan),')
exec(compile(source, uniform.__file__, 'exec'), uniform.__dict__)
'''

VECTOR_PAD_MUTATIONS = (
    ('run-no-spectrum-rule', PIN_FILE, 'test_coax_port_run_padded', '''
from rfx.stepping import vector_pad
vector_pad._CARRY_PASS = vector_pad._CARRY_PASS - {'coax_sparam_accs'}
''', "carry entry 'coax_sparam_accs' has no pad rule", 3),
    ('run-high-end-view', PIN_FILE, 'test_coax_port_run_padded',
     HIGH_END_VIEW, 'not bitwise', 3),
    ('forward-no-frequency-rule', PIN_FILE, 'test_coax_port_forward_value_jvp_grad_padded', '''
from rfx.stepping import vector_pad
vector_pad._INVARIANT_PASS = vector_pad._INVARIANT_PASS - {'coax_freqs'}
''', "step invariant 'coax_freqs' has no pad rule", 3),
    ('forward-high-end-view', PIN_FILE, 'test_coax_port_forward_value_jvp_grad_padded',
     HIGH_END_VIEW, 'not bitwise', 3),
    ('carry-no-record-rule', CONTRACT_FILE, 'test_vector_pad_carry_rules', '''
from rfx.stepping import vector_pad
vector_pad._CARRY_PASS = vector_pad._CARRY_PASS - {'coax_time_records'}
''', 'AssertionError: coax_time_records', 1),
    ('carry-new-entry', CONTRACT_FILE, 'test_vector_pad_carry_rules', '''
from pathlib import Path
original = Path.read_text
def extra_key(path, *args, **kwargs):
    source = original(path, *args, **kwargs)
    if str(path).endswith('/rfx/measurement/coax_port.py'):
        old = "    dtype = carry['fdtd'].ex.dtype"
        assert source.count(old) == 1
        source = source.replace(old, "    carry['coax_extra'] = 0\\n" + old)
    return source
Path.read_text = extra_key
''', 'coax_extra', 1),
    ('invariant-no-table-rule', CONTRACT_FILE, 'test_vector_pad_invariant_rules', '''
from rfx.stepping import vector_pad
vector_pad._INVARIANT_PASS = vector_pad._INVARIANT_PASS - {'coax_table'}
''', 'AssertionError: coax_table', 1),
    ('invariant-new-entry', CONTRACT_FILE, 'test_vector_pad_invariant_rules', '''
from pathlib import Path
original = Path.read_text
def extra_key(path, *args, **kwargs):
    source = original(path, *args, **kwargs)
    if str(path).endswith('/rfx/stepping/uniform_setup.py'):
        old = "invariants.update(coax_table=tables['coax_table'], coax_freqs=tables['coax_freqs'])"
        assert source.count(old) == 1
        source = source.replace(old, old[:-1] + ', coax_extra=0)')
    return source
Path.read_text = extra_key
''', 'coax_extra', 1),
    ('run-before-arrival', PIN_FILE, 'test_coax_port_run_padded', '''
from tests.unit.runners import test_vector_pad_paths
test_vector_pad_paths.COAX_STEPS = 60
''', 'coax arrival P/P60=1.0 < 1e6', 3),
    ('forward-before-arrival', PIN_FILE, 'test_coax_port_forward_value_jvp_grad_padded', '''
from tests.unit.runners import test_vector_pad_paths
test_vector_pad_paths.COAX_STEPS = 60
''', 'coax arrival P/P60=1.0 < 1e6', 3),
    ('until-decay-no-spectrum-rule', PIN_FILE, 'test_coax_port_until_decay_padded', '''
from rfx.stepping import vector_pad
vector_pad._CARRY_PASS = vector_pad._CARRY_PASS - {'coax_sparam_accs'}
''', "carry entry 'coax_sparam_accs' has no pad rule", 3),
    ('until-decay-high-end-view', PIN_FILE, 'test_coax_port_until_decay_padded',
     HIGH_END_VIEW, 'not bitwise', 3),
)


@pytest.mark.parametrize('mutation', VECTOR_PAD_MUTATIONS, ids=lambda row: row[0])
def test_vector_pad_mutation(mutation):
    name, path, test, changes, message, failures = mutation
    node = f'{path}::{test}'
    command = changes + f'''
import pytest
raise SystemExit(pytest.main([{node!r}, '-q', '-s', '-o', 'addopts=']))
'''
    result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    print(f'VECTOR PAD MUTATION {name}: node={node} exit={result.returncode}', file=sys.stderr)
    print(result.stdout + result.stderr, file=sys.stderr)
    assert result.returncode == 1, name
    assert message in result.stdout, name
    assert f'{failures} failed' in result.stdout, name
    print(f'VECTOR PAD MUTATION MATCH {name}: message={message!r} '
          f'failures={failures} exit={result.returncode}', file=sys.stderr)
    if path == PIN_FILE:
        for target in ('aligned', 'odd', 'aligned_slow'):
            assert f'{test}[{target}]' in result.stdout, (name, target)
        if name == 'run-high-end-view':
            assert 'coax_port_metadata' in result.stdout, name
            assert 'sparam_time_records' in result.stdout, name


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-s', '-o', 'addopts=']))

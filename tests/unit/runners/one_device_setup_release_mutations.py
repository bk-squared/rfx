"""Release witness run explicitly with an in-memory mutation in a child."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('target', [
    'tests/unit/runners/test_one_device_setup_release.py::test_released_kernel_product_rebuilds_conductors',
    'tests/contracts/test_coax_shielded_end.py',
], ids=['pin', 'shielded_end'])
def test_dense_release_mutation_m8(target):
    script = '''
import inspect
import textwrap
import pytest
import rfx.model.realization_report as reporter

def rewrite(owner, name, old, new):
    fn = getattr(owner, name)
    source = textwrap.dedent(inspect.getsource(fn))
    assert source.count(old) == 1, (name, old)
    scope = dict(fn.__globals__)
    exec(source.replace(old, new), scope)
    setattr(owner, name, scope[name])

rewrite(reporter, 'rebuild_view',
        'and not getattr(solved.conductors, "_dense_released", False)', '')
'''
    script += f"raise SystemExit(pytest.main([{target!r}, '-q', '-o', 'addopts=', '--tb=short']))\n"
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''), timeout=480)
    output = result.stdout + result.stderr
    print(f'MUTATION m8 {target} EXIT {result.returncode}\n{output}', file=sys.stderr)
    assert result.returncode == 1, output
    assert any(line.startswith('FAILED ' + target) for line in output.splitlines()), output

"""Requested coaxial frequency bins with altered band comparisons and defaults."""
import os
import subprocess
import sys

import pytest


CASES = {
    'r1': ('a', 'tests/contracts/test_coax_requested_bins.py::test_below_band', "rewrite(coax_port, 'refuse_bins_outside_band', 'bins < np.asarray(freq_min, bins.dtype)', 'np.zeros(bins.shape, dtype=bool)')"),
    'r2': ('a', 'tests/contracts/test_coax_requested_bins.py::test_above_band', "rewrite(coax_port, 'refuse_bins_outside_band', 'bins > np.asarray(freq_max, bins.dtype)', 'np.zeros(bins.shape, dtype=bool)')"),
    'r3a': ('b', 'tests/contracts/test_coax_requested_bins.py::test_float32_lower_bound', "rewrite(coax_port, 'refuse_bins_outside_band', 'bins < np.asarray(freq_min, bins.dtype)', 'bins.astype(np.float64) < freq_min')"),
    'r3b': ('b', 'tests/contracts/test_coax_requested_bins.py::test_float32_upper_bound', "rewrite(coax_port, 'refuse_bins_outside_band', 'bins > np.asarray(freq_max, bins.dtype)', 'bins.astype(np.float64) > freq_max')"),
    'r4': ('b', 'tests/contracts/test_coax_requested_bins.py::test_default_band', "rewrite(uniform, 'run_uniform', 'jnp.linspace(sim._coaxial_ports[0].freq_min, sim._freq_max, 50, dtype=jnp.float32)', 'jnp.linspace(sim._freq_max / 10, sim._freq_max, 50, dtype=jnp.float32)')"),
}


@pytest.mark.parametrize('case', CASES)
def test_requested_bins_mutation(case, capsys):
    mode, witness, defect = CASES[case]
    script = '''
import inspect
import pytest
from rfx.measurement import coax_port
from rfx.runners import uniform

def rewrite(owner, name, old, new):
    """Keep altered coaxial functions in their own globals."""
    fn = getattr(owner, name)
    source = inspect.getsource(fn)
    assert source.count(old) == 1, (name, old, source.count(old))
    scope = inspect.unwrap(fn).__globals__
    exec(source.replace(old, new), scope)
    setattr(owner, name, scope[name])
'''
    script += defect + '\n'
    script += f"raise SystemExit(pytest.main([{witness!r}, '-q', '-o', 'addopts=', '--tb=short']))\n"
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    with capsys.disabled():
        print(f'{case} ({mode}): exit={result.returncode}\n{result.stdout}\n{result.stderr}', file=sys.stderr)
    assert result.returncode == 1
    assert 'FAILED ' + witness in result.stdout


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-s', '-o', 'addopts=']))

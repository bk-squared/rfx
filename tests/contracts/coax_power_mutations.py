"""Manual V2 check-off and restored half-edge trapezoid mutations."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mutation,expected', [('a', 0), ('b', 1)])
def test_power_mutation(mutation, expected):
    script = '''
import numpy as np
import pytest
import tests.contracts.test_coax_power as contract
import rfx.measurement.coax_voltage as voltage
def old_weights(grid, *, center_xy, **kwargs):
    u = (np.arange(grid.nx) - grid.pad_x_lo) * grid.dx
    v = (np.arange(grid.ny) - grid.pad_y_lo) * grid.dx
    radial = u - center_xy[0]
    radial = radial[(radial >= .000635) & (radial <= .002055)]
    widths = np.diff(radial)
    lengths = np.r_[widths[0]/2, (widths[:-1]+widths[1:])/2, widths[-1]/2]
    weights = {}
    for radius, length in zip(radial, lengths):
        i = int(np.clip(np.searchsorted(u, center_xy[0]+radius)-1, 0, len(u)-2))
        j = int(np.clip(np.searchsorted(v, center_xy[1])-1, 0, len(v)-2))
        tx = (center_xy[0]+radius-u[i])/(u[i+1]-u[i])
        ty = (center_xy[1]-v[j])/(v[j+1]-v[j])
        for di in (0, 1):
            for dj in (0, 1):
                key = (i+di, j+dj)
                weights[key] = weights.get(key, 0.) + length*(tx if di else 1-tx)*(ty if dj else 1-ty)
    return tuple((i, j, w) for (i, j), w in sorted(weights.items()))
voltage.coax_voltage_weights = old_weights
'''
    if mutation == 'a':
        script += 'contract.check_power = lambda ratio: None\n'
    script += '''
raise SystemExit(pytest.main(['tests/contracts/test_coax_power.py', '-q', '-s',
    '-o', 'addopts=', '-k', 'accepted_power']))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, env=dict(os.environ, PYTEST_ADDOPTS=''))
    print(f'V2 mutation {mutation}: exit={result.returncode}\n{result.stdout}\n{result.stderr}', file=sys.stderr)
    assert result.returncode == expected
    if expected:
        assert 'AssertionError' in result.stdout
        assert 'test_accepted_power_equals_line_flux' in result.stdout


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-s', '-o', 'addopts=']))

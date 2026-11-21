"""On-demand blocked-fit mutations, isolated in fresh CPU subprocesses.

Run as a module, or load explicitly as a pytest plugin with this file named
as the target. No mutation row is collected by the ordinary test suite.
"""

import importlib
import inspect
import os
import subprocess
import sys


ROWS = (
    ('projection_dropped', 'tests/unit/sparams/test_ringdown_blocked.py::test_blocked_matches_dense[3-400-x64]', 'spectra_jvp'),
    ('partial_skipped', 'tests/unit/sparams/test_ringdown_blocked.py::test_blocked_matches_dense[3-400-x64]', 'residues'),
    ('padded_tb_rows_dropped', 'tests/unit/sparams/test_ringdown_blocked.py::test_padded_512_is_finite[x64]', 'non-finite value or derivative'),
    ('partial_skipped', 'tests/unit/sparams/test_ringdown_blocked.py::test_padded_512_is_finite[x64]', 'spectra'),
    ('host_partial_skipped', 'tests/unit/sparams/test_ringdown_blocked.py::test_host_stage2_minimum_norm[True-True]', 'host residue error'),
    ('host_end_reference_dropped', 'tests/unit/sparams/test_ringdown_blocked.py::test_host_stage2_minimum_norm[True-True]', 'host residue error'),
    ('static_stage2_dropped', 'tests/contracts/test_ringdown_blocked_memory.py', 'field lane reached before static stage-2 refusal'),
    ('static_capacity_capped', 'tests/contracts/test_ringdown_blocked_memory.py', 'field lane reached before static stage-2 refusal'),
    ('pole_tangent_doubled', 'tests/contracts/test_ringdown_second_derivative.py::test_staged_first_derivative_of_the_completed_spectrum_is_the_plain_step_s[jacfwd]', 'assert same_bits'),
)



def pytest_configure(config):
    mutation = os.environ.get('RFX_BLOCKED_MUTATION')
    if not mutation:
        import pytest

        main()
        pytest.exit('all blocked-fit mutations failed in their witnesses', returncode=0)
    from rfx.measurement import ringdown_blocked as blocked
    planning = importlib.import_module('rfx.ringdown._pencil')
    from rfx.measurement import matrix_pencil as pencil

    if mutation in ('projection_dropped', 'partial_skipped', 'padded_tb_rows_dropped'):
        module = blocked
        source = inspect.getsource(module)
        if mutation == 'projection_dropped':
            old = '        ds = ringdown_jax._pole_step(q2, ghat, weights, mask)'
            new = '''        from rfx.ringdown._jax import _basis
        b, t = _basis(frozen_s, y.shape[0], dt)
        d = (t[:, None] * b * mask[None, :])[:, None, :] * (c0.T * weights[:, None])[None, :, :]
        ds = ringdown_jax._pole_step(y, d.reshape(-1, s0.size), weights, mask)'''
            source = source.replace(old, new)
        elif mutation == 'partial_skipped':
            source = source.replace('(length + size - 1) // size', 'length // size')
        else:
            source = source.replace('initial = initial.at[k:2*k, k:2*k].set(unit)', 'initial = initial')
    elif mutation in ('host_partial_skipped', 'host_end_reference_dropped'):
        module = pencil
        source = inspect.getsource(module)
        if mutation == 'host_partial_skipped':
            source = source.replace('range(0, length, size)', 'range(0, length - length % size, size)')
        else:
            source = source.replace('(t - end)[:, None] * s[None, growing]', 't[:, None] * s[None, growing]')
    elif mutation == 'pole_tangent_doubled':
        module = importlib.import_module('rfx.ringdown._jax')
        source = inspect.getsource(module)
        source = source.replace('return ds, ds_dot', 'return ds, 2 * ds_dot')
    else:
        module = planning
        original = source = inspect.getsource(module)
        if mutation == 'static_stage2_dropped':
            source = source.replace('r=capacity,', 'r=None,')
        elif mutation == 'static_capacity_capped':
            source = source.replace('capacity = rd.static_pole_bound(',
                                    'capacity = rd.HARMINV_MAX_MODES or rd.static_pole_bound(')
        else:
            raise ValueError(mutation)
        assert source != original, ('mutation text not found', mutation)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    if module is planning:
        from rfx import ringdown as rd
        rd._check_run_identification_memory = planning._check_run_identification_memory


def main():
    """Print each named witness's assertion failure and reject other outcomes."""
    for mutation, node, assertion in ROWS:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', node, '-q', '-s', '--tb=short',
             '-p', 'tests.contracts.ringdown_blocked_mutations'],
            env={**os.environ, 'RFX_BLOCKED_MUTATION': mutation, 'JAX_PLATFORMS': 'cpu'},
            text=True, capture_output=True, timeout=360)
        output = result.stdout + result.stderr
        print(f'MUTATION {mutation} witness={node} exit={result.returncode}', flush=True)
        print(output, flush=True)
        assert result.returncode == 1, (mutation, result.returncode, output)
        assert assertion in output, (mutation, assertion, output)
        assert 'E   assert ' in output or 'Failed: field lane reached' in output, output
        assert f'FAILED {node}' in output, output
        assert 'ERROR collecting' not in output and 'ERROR at setup' not in output, output


if __name__ == '__main__':
    main()

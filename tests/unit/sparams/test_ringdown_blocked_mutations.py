"""Two revived defects per new test must fail that test in a fresh CPU process."""

import os
import subprocess
import sys

import pytest


ROWS = (
    ('projection_dropped', 'tests/unit/sparams/test_ringdown_blocked.py::test_blocked_matches_dense[3-400-x64]', 'spectra_jvp'),
    ('partial_skipped', 'tests/unit/sparams/test_ringdown_blocked.py::test_blocked_matches_dense[3-400-x64]', 'spectra'),
    ('padded_tb_rows_dropped', 'tests/unit/sparams/test_ringdown_blocked.py::test_padded_512_is_finite[x64]', 'non-finite value or derivative'),
    ('partial_skipped', 'tests/unit/sparams/test_ringdown_blocked.py::test_padded_512_is_finite[x64]', 'spectra'),
    ('host_partial_skipped', 'tests/unit/sparams/test_ringdown_blocked.py::test_host_stage2_minimum_norm[True-True]', 'host residue error'),
    ('host_end_reference_dropped', 'tests/unit/sparams/test_ringdown_blocked.py::test_host_stage2_minimum_norm[True-True]', 'host residue error'),
    ('static_stage2_dropped', 'tests/contracts/test_ringdown_blocked_memory.py', 'field lane reached before static stage-2 refusal'),
    ('static_capacity_capped', 'tests/contracts/test_ringdown_blocked_memory.py', 'field lane reached before static stage-2 refusal'),
)


@pytest.mark.parametrize('mutation,node,assertion', ROWS, ids=[f'{i+1}-{row[0]}' for i, row in enumerate(ROWS)])
def test_mutation_turns_its_witness_red(mutation, node, assertion):
    """Require an assertion failure from the intended witness, never a collection or runtime error."""
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', node, '-q', '-s', '--tb=short', '-p', 'tests.ringdown_blocked_mutations'],
        env={**os.environ, 'RFX_BLOCKED_MUTATION': mutation, 'JAX_PLATFORMS': 'cpu'},
        text=True, capture_output=True, timeout=360)
    output = result.stdout + result.stderr
    print(f'\nMUTATION {mutation}\n{output}')
    assert result.returncode == 1, (mutation, result.returncode, output)
    assert assertion in output, (mutation, assertion, output)

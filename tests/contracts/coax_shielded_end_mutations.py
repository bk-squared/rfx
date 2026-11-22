"""Non-collected closure and gap mutations, run explicitly through labrun."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mutation,check,message', [
    ('cap', 'test_closure', 'flood escaped beyond cap'),
    ('pin', 'test_gap', 'pin cell left in gap'),
])
def test_mutation(mutation, check, message):
    script = """
import numpy as np
import pytest
import rfx.model.coax_shield as shield
from dataclasses import replace
original = shield.shield_coax_end
def broken(*args, **kwargs):
    materials, cells, entities = original(*args, **kwargs)
    cap = next(e for e in entities if '/cap/' in e.entity_id)
    if MUTATION == 'cap':
        shell = next(e for e in entities if e.entity_id.endswith('/shell'))
        cells = (cells & ~cap.cells) | shell.cells
        entities = [e for e in entities if e is not cap]
    else:
        pin = next(e for e in entities if e.entity_id.endswith('/pin'))
        cell = tuple(np.argwhere(cap.cells)[len(np.argwhere(cap.cells)) // 2])
        i, j = args[0].position_to_index((*kwargs['center_xy'], 0.))[:2]
        k = cell[2] - (1 if kwargs['end'].port.face == 'top' else -1)
        mask = np.asarray(pin.cells).copy()
        mask[i,j,k] = True
        cells = cells.copy()
        cells[i,j,k] = True
        entities = [replace(e, cells=mask) if e is pin else e for e in entities]
    return materials, cells, entities
shield.shield_coax_end = broken
raise SystemExit(pytest.main(['tests/contracts/test_coax_shielded_end.py', '-q', '-s', '-o', 'addopts=', '-k', CHECK]))
"""
    script = 'MUTATION = '+repr(mutation)+'\nCHECK = '+repr(check)+'\n'+script
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    print(f'S7 mutation {mutation}: exit={result.returncode}\n{result.stdout}\n{result.stderr}')
    assert result.returncode == 1
    assert message in result.stdout


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-s', '-o', 'addopts=']))

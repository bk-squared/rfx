"""Non-collected B2 chain mutations, patched only in isolated subprocesses.

M3 is green: on a shorted DUT, b/a does not depend on the source side.
"""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mutation,check,message', [
    ('M1', 'test_open_native_and_renormalization', 'A1 identity'),
    ('M2', 'test_open_native_and_renormalization', 'A1 S'),
    ('M3', 'test_shorted_two_link_chain', 'A2 S'),
    ("M3'", 'test_shorted_two_link_chain', 'A2 closure capped'),
    ("M4'", 'test_shorted_two_link_chain', 'A2 identity'),
])
def test_mutation(mutation, check, message):
    common = '''
from dataclasses import replace
import numpy as np
import pytest
import tests.contracts.test_coax_port_s as contract
'''
    changes = {
        'M1': '''
import rfx.model.coax_line as line
import rfx.model.coax_port as port_model
original = contract.open_end
def keep_setback(patch):
    layout = line.coax_port_layout
    original(patch)
    patch.setattr(line, 'coax_port_layout', layout)
    patch.setattr(port_model, 'coax_port_layout', layout)
contract.open_end = keep_setback
''',
        'M2': '''
import rfx.measurement.coax_port as measurement
original = measurement.port_owner
def shifted_reference(sim, grid, *args, **kwargs):
    owner = original(sim, grid, *args, **kwargs)
    direction = 1 if sim._coaxial_ports[0].face == 'top' else -1
    return replace(owner, reference_planes=tuple(
        replace(p, index=p.index-direction, coordinate_m=p.coordinate_m-direction*grid.dx)
        if p.name == 'deembedding' else p for p in owner.reference_planes))
measurement.port_owner = shifted_reference
''',
        'M3': '''
import rfx.model.coax_line as line
original = line.build_coax_line
def source_at_plane(*args, **kwargs):
    """Place the source after layout's length guard, bypassing that guard."""
    kwargs['ends'] = tuple(replace(end, source_index=end.plane_indices[-1])
                          if end.shielded else end for end in kwargs['ends'])
    return original(*args, **kwargs)
line.build_coax_line = source_at_plane
''',
    }
    if mutation in ("M3'", "M4'"):
        entity_kind = 'shell' if mutation == "M3'" else 'pin'
        changes[mutation] = f'''
import jax.numpy as jnp
import rfx.model.coax_shield as shield
from rfx.model.coax_stamp import PTFE_EPS_R
original = shield.shield_coax_end
def remove_slice(*args, **kwargs):
    materials, cells, entities = original(*args, **kwargs)
    end = kwargs['end']
    entity = next(e for e in entities if e.entity_id.endswith('/{entity_kind}'))
    z_index = (end.plane_indices[0] + (-2 if end.port.face == 'top' else 2)
               if {entity_kind!r} == 'shell' else
               end.feed_index + (-6 if end.port.face == 'top' else 6))
    removed = np.zeros_like(entity.cells, dtype=bool)
    removed[:, :, z_index] = np.asarray(entity.cells)[:, :, z_index]
    materials = materials._replace(
        eps_r=jnp.asarray(np.where(removed, PTFE_EPS_R, np.asarray(materials.eps_r))),
        sigma=jnp.asarray(np.where(removed, 0., np.asarray(materials.sigma))))
    entities = [replace(e, cells=np.asarray(e.cells) & ~removed) if e is entity else e
                for e in entities]
    return materials, np.asarray(cells) & ~removed, entities
shield.shield_coax_end = remove_slice
'''
    script = common + changes[mutation] + f'''
raise SystemExit(pytest.main(['tests/contracts/test_coax_port_s.py',
    '-q', '-s', '-o', 'addopts=', '--durations=0', '-k', {check!r}]))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    print(f'B2 mutation {mutation}: exit={result.returncode}\n{result.stdout}\n{result.stderr}',
          file=sys.stderr)
    if mutation == 'M3':
        assert result.returncode == 0, 'M3: expected A2 S to pass with source on last plane'
        return
    assert result.returncode == 1, f'{mutation}: expected failure on {message}'
    assert f'AssertionError: {message}' in result.stdout, (
        f'{mutation}: did not fail on its own assertion {message}')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-s', '-o', 'addopts=']))

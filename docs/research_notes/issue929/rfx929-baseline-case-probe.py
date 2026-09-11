"""Build-only falsifiers against the pre-fix case source in Git."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from dataclasses import replace
from unittest.mock import patch

from rfx import Simulation

BASE = '81467c81907548c49b126ffa4c0a457c2c5f2698'
ROOT = Path.cwd()
CV = ROOT/'validation/crossval'
sys.path.insert(0, str(CV))

def source(name):
    path = 'validation/crossval/'+name
    return subprocess.check_output(['git','show',BASE+':'+path], text=True)

def forbid(*args, **kwargs):
    raise AssertionError('baseline geometry probe reached a field solve')

out = {'case_source_commit': BASE, 'field_solves': 0}
with patch.object(Simulation, 'run', forbid), contextlib.redirect_stdout(io.StringIO()):
    name = '15_patch_antenna_rt5880.py'
    mod = types.ModuleType('_cv15_baseline929')
    mod.__file__ = str(CV/name)
    sys.modules[mod.__name__] = mod
    exec(compile(source(name), mod.__file__, 'exec'), mod.__dict__)
    sim, _, geom = mod.build_rfx_sim(do_gain=False)
    grid = sim._build_grid()
    port = sim._ports[0]
    pos = list(port.position)
    pos[2] += 2*mod.DX
    sim._ports[0] = replace(port, position=tuple(pos), extent=port.extent-2*mod.DX)
    result = mod.assert_galvanic_feed(sim, grid, geom)
    assert result['galvanic'] is True
    out['cv15'] = {'registration_gap_cells': 2, 'stale_geom_accepted': True,
                   'reported_port_z0':result['port_z0'], 'actual_port_z0':pos[2]}

    original = Simulation.add_port
    seen = []
    def shorten(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        port = self._ports[-1]
        self._ports[-1] = replace(port, extent=port.extent/2)
        seen.append({'declared_extent':port.extent, 'actual_extent':port.extent/2})
        return result
    name = '05_patch_antenna.py'
    namespace = {'__file__':str(CV/name), '__name__':'_cv05_baseline929'}
    env = dict(os.environ)
    env['RFX_CV05_BUILD_ONLY'] = '1'
    for key in ('RFX_CV05_REALIZED_JSON','RFX_CV05_SHEET_PLANE_DELTA','RFX_CV05_PATCH_L_MM'):
        env.pop(key, None)
    with patch.dict(os.environ, env, clear=True), patch.object(Simulation, 'add_port', shorten):
        try:
            exec(compile(source(name), str(CV/name), 'exec'), namespace)
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError('cv05 did not exit through build-only mode')
    assert len(seen)==1
    out['cv05'] = {'shortened_registration_accepted':True, **seen[0]}
print(json.dumps(out, indent=2))

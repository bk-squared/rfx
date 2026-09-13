from pathlib import Path
import hashlib
import json
import platform
import subprocess
from unittest.mock import patch
import jax
import numpy as np
import pytest
import scipy
from tests.oracle import test_leontovich_alpha_oracle as oracle
from rfx.geometry.rasterize_grid import coords_from_uniform_grid

out_dir = Path('/root/rfx-codex/.git/issue947')
real_run = oracle.Simulation.run
captured = []
def capture_run(sim, *args, **kwargs):
    result = real_run(sim, *args, **kwargs)
    captured.append((sim, result))
    return result

oracle._cache.clear()
with patch.object(oracle.Simulation, 'run', capture_run):
    exit_code = pytest.main([
        'tests/oracle/test_leontovich_alpha_oracle.py', '-q', '-s',
        '-o', 'addopts=', '-p', 'no:cacheprovider',
        '-k', 'alpha_envelope_regression_lock or o3_model_fits or alpha_oracle_o3',
    ])
assert len(captured) == 1, len(captured)
sim, result = captured[0]
base = oracle._cache['base']
grid = result.grid
coords = coords_from_uniform_grid(grid)
arrays = {
    'ez_midplane': np.asarray(result.dft_planes['midplane'].accumulator),
    'hy_yplane': np.asarray(result.dft_planes['yhy'].accumulator),
    'time_series': np.asarray(result.time_series),
    'fit_xs': np.asarray(base['xs']),
    'fit_zs_hy': np.asarray(base['z_nodes']),
    'fit_hy_plane': np.asarray(base['hy_plane']),
    'fit_ez_magnitude': np.asarray(base['profile']),
    'x_nodes': np.asarray(coords.x), 'y_nodes': np.asarray(coords.y),
    'z_nodes': np.asarray(coords.z),
}
for component in ('ex','ey','ez','hx','hy','hz'):
    arrays['final_' + component] = np.asarray(getattr(result.state, component))
np.savez_compressed(out_dir / 'current_raw.npz', **arrays)
summary = {
    'head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    'python': platform.python_version(), 'jax': jax.__version__, 'numpy': np.__version__,
    'scipy': scipy.__version__, 'backend': jax.default_backend(),
    'x64': bool(jax.config.jax_enable_x64), 'pytest_exit_code': int(exit_code),
    'field_runs': len(captured), 'grid_shape': list(grid.shape),
    'dx': grid.dx, 'dt': grid.dt, 'lower_pads': list(grid.axis_pads),
    'n_steps': oracle.N_STEPS, 'frequencies': list(oracle.O3_FREQS),
    'alpha_ez': list(base['alpha']), 'fit_ln_rms_ez': list(base['resid']),
    'settle_db': base['settle_db'], 'warnings': base['warnings'],
    'source_positions': [list(entry.position) for entry in sim._ports],
    'plane_registrations': [{'name':p.name,'axis':p.axis,'coordinate':p.coordinate,'component':p.component} for p in sim._dft_planes],
    'model_fits': [],
}
for fit in oracle._cache.get('model_fits', []):
    summary['model_fits'].append({key: [[complex(v).real,complex(v).imag] for v in value] if key in ('modes','amps') else value for key,value in fit.items()})
summary['source_sha256'] = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('tests/oracle/test_leontovich_alpha_oracle.py','tests/_transverse_resonance_o3.py','rfx/materials/thin_conductor.py','rfx/simulation.py')}
summary['raw_sha256'] = hashlib.sha256((out_dir/'current_raw.npz').read_bytes()).hexdigest()
(out_dir/'current_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Saved one-run raw complex profiles and input/model receipts:',out_dir)
print('Recorded pytest exit code:', int(exit_code))

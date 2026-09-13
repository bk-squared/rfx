"""Build-only exact-bound and historical Box mask replay; no FDTD."""
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
import jax
import numpy as np

repo = Path('/root/rfx-947')
receipts = Path('/root/rfx-codex/.git/issue947')
outdir = receipts / 'root_history'
raw = np.load(receipts / 'current_raw.npz')
summary = json.loads((receipts / 'current_summary.json').read_text())
dx = summary['dx']
shape = tuple(summary['grid_shape'])
pads = tuple(summary['lower_pads'])
grid = types.SimpleNamespace(shape=shape, dx=dx, axis_pads=pads)
x, y, z = (raw[k + '_nodes'] for k in ('x', 'y', 'z'))
start = round(.130 / dx)
start_array = start + pads[0]
expected = np.array([2. * ((i + .5) / 120)**2 for i in range(120)], np.float32)

def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()

refs = {'green_o3_comparator_commit': 'bb43d95f',
        'before_exact_coordinates_834': '635ab2e3^',
        'exact_coordinates_834': '635ab2e3',
        'first_parent_before_931': '485a9b98^',
        'current': 'HEAD'}
result = {'no_FDTD_runs': True, 'jax': jax.__version__,
          'jax_x64': bool(jax.config.x64_enabled), 'dx': dx,
          'shape': shape, 'pads': pads, 'revisions': {}}
arrays = {'expected': expected}
for label, ref in refs.items():
    commit = git('rev-parse', ref)
    code = subprocess.check_output(['git', 'show', f'{commit}:rfx/geometry/csg.py'], cwd=repo, text=True)
    module_name = '_issue947_history_' + label
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(code, f'{commit}:rfx/geometry/csg.py', 'exec'), module.__dict__)
    coordinates = tuple(np.asarray(a) for a in module._grid_coords(grid))
    assigned = np.zeros(shape[0], np.float32)
    selected = []
    for i in range(120):
        lo = .130 + i * dx
        hi = lo + dx
        mask = np.asarray(module.Box((lo, 0., 0.), (hi, .002, .006)).mask(grid))[:, 2, 6]
        selected.append((np.flatnonzero(mask) - start_array).tolist())
        assigned[mask] = expected[i]
    profile = assigned[start_array:start_array + 120]
    mismatched = np.flatnonzero(profile != expected)
    result['revisions'][label] = {
        'commit': commit, 'csg_source_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'csg_git_blob': git('rev-parse', f'{commit}:rfx/geometry/csg.py'),
        'coordinate_dtypes': [str(a.dtype) for a in coordinates],
        'coordinate_x_sha256': hashlib.sha256(coordinates[0].tobytes()).hexdigest(),
        'profile_sha256': hashlib.sha256(profile.tobytes()).hexdigest(),
        'mismatch_count': int(len(mismatched)), 'mismatched_local_cells': mismatched.tolist(),
        'zero_local_cells': np.flatnonzero(profile == 0).tolist(),
        'selected_local_nodes_by_layer': selected,
    }
    arrays[label] = profile
    print(label, commit, 'coords', coordinates[0].dtype, 'mismatches', mismatched.tolist())
    if label == 'current':
        current_module = module

# Keep the exact current rasterizer but express BOTH x bounds on its
# canonical integer-node arithmetic route. Do not form hi as lo+dx.
fixed = np.zeros(shape[0], np.float32)
traces = []
for i in range(120):
    lo = .130 + i * dx
    hi = lo + dx
    vol = np.flatnonzero((x >= lo) & (x < hi))
    nearest = int(np.argmin(np.abs(x - (lo + hi) / 2)))
    fixed_lo = (start + i) * dx
    fixed_hi = (start + i + 1) * dx
    mask = np.asarray(current_module.Box((fixed_lo, 0., 0.), (fixed_hi, .002, .006)).mask(grid))[:, 2, 6]
    fixed_nodes = np.flatnonzero(mask)
    assert fixed_nodes.tolist() == [start_array + i]
    fixed[mask] = expected[i]
    if 26 <= i <= 38:
        traces.append({'layer': i, 'lo_hex': lo.hex(), 'hi_hex': hi.hex(),
                       'canonical_lo_hex': float(x[start_array + i]).hex(),
                       'canonical_hi_hex': float(x[start_array + i + 1]).hex(),
                       'volume_selected_local': (vol - start_array).tolist(),
                       'nearest_midpoint_local': nearest - start_array,
                       'actual_selected_local': result['revisions']['current']['selected_local_nodes_by_layer'][i],
                       'fixed_selected_local': (fixed_nodes - start_array).tolist()})
fixed = fixed[start_array:start_array + 120]
assert np.array_equal(fixed, expected)
result['canonical_bounds_repair'] = {'start_node': start, 'end_node': start + 120,
                                     'lo_formula': '(start+i)*dx', 'hi_formula': '(start+i+1)*dx',
                                     'mismatch_count': 0, 'single_node_each_layer': True,
                                     'profile_sha256': hashlib.sha256(fixed.tobytes()).hexdigest()}
result['exact_bound_trace'] = traces
arrays['current_canonical_bounds'] = fixed
print('current canonical bounds: all120 layers select intended singleton; zero mismatches')
(outdir / 'absorber_history.json').write_text(json.dumps(result, indent=2) + '\n')
np.savez_compressed(outdir / 'absorber_history_profiles.npz', **arrays)

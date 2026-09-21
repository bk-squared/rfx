from __future__ import annotations
import argparse
import ast
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/root/workspace/bk-workspace/.801-measure')
OUT = BASE / 'msl_inset'
LO_SHIFT = 0.0
HI_SHIFT = 0.0
SELECTION = None
SRC = BASE / 'src-main'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['XDG_CACHE_HOME'] = str(OUT / 'cache')
os.environ['MPLCONFIGDIR'] = str(OUT / 'mpl_config')
sys.path.insert(0, str(SRC))

def mutation_guard(event, args):
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, int) or not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return
        p = Path(os.fsdecode(path)).resolve()
        if not p.is_relative_to(OUT) or p.exists():
            raise PermissionError(f'NEW_FILES_ONLY: {event} {p}')
    elif event == 'os.mkdir':
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(OUT) or p.exists():
            raise PermissionError(f'NEW_DIRECTORIES_ONLY: {event} {p}')
    elif event in {'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink', 'os.chmod', 'os.chown', 'os.truncate'}:
        raise PermissionError(f'NO_MUTATION: {event} {args}')

sys.addaudithook(mutation_guard)
import numpy as np
import jax
from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.sources.msl_port import msl_port_from_entry, msl_probe_x_coords_n
from rfx import simulation as lowlevel

PATHS = {
    'cv06b': 'validation/crossval/06b_msl_notch_filter_uniform.py',
    'cv20': 'scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py',
}
SETTINGS = {
    'cv06b': dict(n_freqs=100, num_periods=20.0),
    'cv20': dict(n_freqs=30, num_periods=12),
}

def serial(v):
    if dataclasses.is_dataclass(v):
        return {f.name: serial(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if hasattr(v, '_asdict'):
        return serial(v._asdict())
    if isinstance(v, dict):
        return {str(k): serial(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [serial(x) for x in v]
    if isinstance(v, (np.ndarray, jax.Array)):
        return serial(np.asarray(v).tolist())
    if isinstance(v, np.generic):
        return serial(v.item())
    if isinstance(v, complex):
        return {'real': v.real, 'imag': v.imag}
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (str, float, int, bool)) or v is None:
        return v
    return repr(v)

def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(serial(value), f, indent=2)
        f.write('\n')

def save_npz(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_module(rel):
    name = '_msl_' + hashlib.sha256(rel.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, SRC / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def builder_namespace(fixture):
    # Execute only the original numerical constants and builder definition.
    # The original main functions write into the exported source tree.
    names = {'EPS_R', 'H_SUB', 'W_TRACE', 'STUB_LEN', 'W_STUB', 'L_LINE',
             'PORT_MARGIN', 'F_MAX', 'DX', 'LX', 'LY', 'LZ', 'N_FREQS', 'NUM_PERIODS'}
    source = (SRC / PATHS[fixture]).read_text()
    nodes = [n for n in ast.parse(source).body
             if (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in names for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name == '_build_sim')]
    copied = ast.unparse(ast.Module(body=nodes, type_ignores=[])) + '\n'
    namespace = dict(Simulation=Simulation, Box=Box, Boundary=Boundary,
                     BoundarySpec=BoundarySpec, np=np)
    exec(compile(copied, str(OUT / 'measure.py') + '::' + fixture + '_builder_copy', 'exec'), namespace)
    return namespace, copied

def build_pair(fixture, variant):
    ns, copied = builder_namespace(fixture)
    baseline = ns['_build_sim']()
    if variant == 'baseline':
        return baseline, [], copied
    grid = baseline._build_grid()
    trace = baseline._geometry[1]
    lo, hi = [list(map(float, b)) for b in trace.shape.bounding_box()]
    newlo, newhi = lo.copy(), hi.copy()
    newlo[0] = (-(grid.pad_x_lo - 1) + LO_SHIFT) * float(grid.dx)
    newhi[0] = float(baseline._domain[0]) + (grid.pad_x_hi - 1 + HI_SHIFT) * float(grid.dx)
    calls = []
    def continued_box(*args, **kwargs):
        original = Box(*args, **kwargs)
        bounds = [list(map(float, b)) for b in original.bounding_box()]
        if bounds == [lo, hi]:
            calls.append(bounds)
            return Box(tuple(newlo), tuple(newhi))
        return original
    ns['Box'] = continued_box
    continued = ns['_build_sim']()
    assert len(calls) == 1, calls
    assert len(baseline._geometry) == len(continued._geometry)
    for index, (b, c) in enumerate(zip(baseline._geometry, continued._geometry)):
        if index == 1:
            assert serial(dataclasses.replace(c, shape=b.shape)) == serial(b)
        else:
            assert serial(b) == serial(c), index
    bdict, cdict = vars(baseline), vars(continued)
    assert bdict.keys() == cdict.keys()
    checked = []
    for key in bdict:
        if key != '_geometry':
            assert serial(bdict[key]) == serial(cdict[key]), key
            checked.append(key)
    return continued, [dict(geometry_index=1, old_bounds_m=[lo, hi],
                           new_bounds_m=[newlo, newhi], other_sim_fields_equal=checked)], copied

def port_positions(sim, grid):
    result = []
    for pe in sim._resolve_msl_probe_entries(grid):
        mp = msl_port_from_entry(pe)
        xs = msl_probe_x_coords_n(grid, mp, n_probes=pe.n_probes,
                                 n_offset_cells=pe.n_probe_offset,
                                 n_spacing_cells=pe.n_probe_spacing)
        pos = tuple(map(float, pe.position))
        indices = [grid.position_to_index((float(x), pos[1], pos[2]))[0] for x in xs]
        result.append(dict(name=pe.name, direction=pe.direction, source_x_m=pos[0],
                           source_x_index=grid.position_to_index(pos)[0],
                           reference_x_m=float(xs[0]), reference_x_index=indices[0],
                           probe_x_m=list(map(float, xs)), probe_x_indices=indices,
                           n_probe_offset=int(pe.n_probe_offset),
                           n_probe_spacing=int(pe.n_probe_spacing), n_probes=int(pe.n_probes)))
    return result

def arrays_for(grid, materials, kwargs):
    arrays = {k: np.asarray(getattr(materials, k)) for k in ('eps_r', 'mu_r', 'sigma')}
    pm = kwargs.get('pec_mask')
    arrays['pec_mask'] = np.zeros(grid.shape, bool) if pm is None else np.asarray(pm)
    edges = kwargs.get('pec_edge_masks')
    for i, axis in enumerate('xyz'):
        arrays['pec_edge_' + axis] = np.zeros(grid.shape, bool) if edges is None else np.asarray(edges[i])
    return arrays

def edge_record(sim, grid, arrays, positions):
    lo, hi = [list(map(float, b)) for b in sim._geometry[1].shape.bounding_box()]
    coords = coords_from_uniform_grid(grid)
    xs, ys, zs = (np.asarray(getattr(coords, a)) for a in 'xyz')
    j = int(np.argmin(abs(ys - (lo[1] + hi[1]) / 2)))
    k = int(np.argmin(abs(zs - lo[2])))
    mid = int(np.argmin(abs(xs - float(sim._domain[0]) / 2)))
    planes = np.flatnonzero(arrays['pec_edge_x'][mid, j, :] | arrays['pec_edge_y'][mid, j, :])
    planes = planes[zs[planes] > 0]
    p, q = int(grid.pad_x_lo), int(grid.pad_x_hi)
    windows = {'first': list(range(p + 4)), 'last': list(range(grid.nx - q - 4, grid.nx))}
    plane_records = []
    for kk in sorted(set([k] + planes.tolist())):
        rows = np.flatnonzero(arrays['pec_edge_x'][mid, :, kk])
        transverse = (int(rows.min()), int(rows.max()) + 1) if len(rows) else (j, j + 1)
        rr = {'z_index': kk, 'z_m': float(zs[kk]), 'y_index': j, 'y_m': float(ys[j]),
              'trace_y_index_slice': transverse, 'eps_below_z_index': kk - 1,
              'eps_below_z_m': float(zs[kk - 1]), 'windows': {}, 'edge_counts': {}}
        for name, inds in windows.items():
            rr['windows'][name] = dict(x_indices=inds, x_m=xs[inds].tolist(),
                pec_edge_x=arrays['pec_edge_x'][inds, j, kk].astype(int).tolist(),
                pec_edge_y=arrays['pec_edge_y'][inds, j, kk].astype(int).tolist(),
                pec_mask=arrays['pec_mask'][inds, j, kk].astype(int).tolist(),
                eps_r_below=arrays['eps_r'][inds, j, kk - 1].tolist())
        for axis in 'xy':
            a = arrays['pec_edge_' + axis]
            row = a[:, j, kk]
            active = np.flatnonzero(row)
            rr['edge_counts'][axis] = dict(first_x_index=int(active[0]) if len(active) else None,
                last_x_index=int(active[-1]) if len(active) else None, centre_row_total=int(row.sum()),
                centre_row_x_lo=int(row[:p].sum()), centre_row_x_hi=int(row[-q:].sum()),
                trace_width_x_lo=int(a[:p, transverse[0]:transverse[1], kk].sum()),
                trace_width_x_hi=int(a[-q:, transverse[0]:transverse[1], kk].sum()))
        plane_records.append(rr)
    return dict(grid_shape=list(grid.shape), dx_m=float(grid.dx), dt_s=float(grid.dt),
                domain_m=list(map(float, sim._domain)), trace_bounds_m=[lo, hi],
                face_pads={a+'_'+s:int(getattr(grid,'pad_'+a+'_'+s)) for a in 'xyz' for s in ('lo','hi')},
                declared_z_nearest_index=k, realized_trace_z_indices=planes.tolist(),
                planes=plane_records, ports=positions,
                totals={name:int(value.sum()) for name,value in arrays.items() if name.startswith('pec_')},
                sha256={name:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for name,a in arrays.items()})

class Captured(BaseException):
    pass

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for f in self.streams:
            f.write(s)
            f.flush()
        return len(s)
    def flush(self):
        for f in self.streams:
            f.flush()

def result_dict(result):
    if dataclasses.is_dataclass(result):
        return {f.name:getattr(result,f.name) for f in dataclasses.fields(result)}
    if hasattr(result, '_asdict'):
        return result._asdict()
    return vars(result)

def execute(fixture, variant, out, dry):
    t0 = time.perf_counter()
    status = dict(fixture=fixture, variant=variant, dry_run=int(dry), completed=0,
                  solve_calls=0, timestepping_calls=0)
    original = lowlevel.run
    try:
        sim, changes, copied = build_pair(fixture, variant)
        with (out / 'builder_copy.py').open('x') as f:
            f.write(copied)
        grid = sim._build_grid()
        positions = port_positions(sim, grid)
        write_json(out / 'settings.json', dict(source=PATHS[fixture], builder='_build_sim',
                   method='compute_msl_s_matrix', kwargs=SETTINGS[fixture],
                   x64=int(jax.config.jax_enable_x64), changes=changes,
                   geometry=sim._geometry, ports=sim._msl_ports, boundary=sim._boundary_spec))
        with (out / 'preflight.txt').open('x') as preflight:
            with contextlib.redirect_stdout(Tee(sys.stdout, preflight)), contextlib.redirect_stderr(Tee(sys.stderr, preflight)):
                preflight_issues = sim.preflight(strict=False)
        write_json(out / 'preflight.json', preflight_issues)

        def observe(run_grid, materials, n_steps, *args, **kwargs):
            number = status['solve_calls']
            status['solve_calls'] += 1
            arrays = arrays_for(run_grid, materials, kwargs)
            record = edge_record(sim, run_grid, arrays, positions)
            record.update(call=number, n_steps=int(n_steps),
                          pec_mask_present=int(kwargs.get('pec_mask') is not None),
                          pec_edge_masks_present=int(kwargs.get('pec_edge_masks') is not None),
                          pec_sheets=serial(kwargs.get('pec_sheets', ())),
                          pec_wires=serial(kwargs.get('pec_wires', ())),
                          source_specs=[{k:serial(v) for k,v in s._asdict().items() if k != 'waveform'} for s in kwargs.get('sources', [])],
                          probe_specs=serial(kwargs.get('probes', [])))
            save_npz(out / f'assembly_received_{number:02d}.npz', **arrays)
            write_json(out / f'assembly_received_{number:02d}.json', record)
            print(json.dumps(dict(fixture=fixture,variant=variant,call=number,n_steps=int(n_steps),grid=record['grid_shape'],
                                  edge_counts=record['planes'][0]['edge_counts'])), flush=True)
            checks = []
            for plane in record['planes']:
                j, k = plane['y_index'], plane['z_index']
                ex, ey = (arrays['pec_edge_' + a][:, j, k] for a in 'xy')
                union = ex | ey
                p, q = int(run_grid.pad_x_lo), int(run_grid.pad_x_hi)
                expected_indices = list(range(1, p)) + list(range(run_grid.nx-q, run_grid.nx-1))
                checks.append(dict(z_index=k,
                    outermost_flags_xy=[[int(ex[i]), int(ey[i])] for i in (0, run_grid.nx-1)],
                    adjacent_flags_xy=[[int(ex[i]), int(ey[i])] for i in (1, run_grid.nx-2)],
                    A1=int(not union[0] and not union[-1] and np.all(union[expected_indices])),
                    both_components_at_adjacent=int(ex[1] and ey[1] and ex[-2] and ey[-2])))
            status['A1'] = int(bool(checks) and all(c['A1'] for c in checks))
            write_json(out / f'A1_received_{number:02d}.json', checks)
            print(json.dumps(dict(A1=status['A1'], checks=checks)), flush=True)
            if not dry:
                if not status['A1']:
                    raise RuntimeError('A1=0 before time stepping')
                expected = json.loads((OUT / SELECTION['record_file']).read_text())
                if record['sha256'] != expected['sha256']:
                    raise RuntimeError('GPU/dry assembly SHA256 mismatch before time stepping')
            if dry:
                raise Captured()
            status['timestepping_calls'] += 1
            solve_t0 = time.perf_counter()
            result = original(run_grid, materials, n_steps, *args, **kwargs)
            rd = result_dict(result)
            witness = {k:np.asarray(v) for k,v in rd.items() if v is not None and k in ('time_series','energy_history','energy','max_field_history')}
            if witness:
                save_npz(out / f'witness_series_{number:02d}.npz', **witness)
            write_json(out / f'solve_return_{number:02d}.json', dict(wall_s=time.perf_counter()-solve_t0,
                       result_fields=list(rd), witness_fields=list(witness)))
            return result

        lowlevel.run = observe
        s0 = time.perf_counter()
        try:
            result = sim.compute_msl_s_matrix(**SETTINGS[fixture])
        except Captured:
            status.update(completed=1, dry_readback_completed=1)
            return status
        finally:
            status['public_call_wall_s'] = time.perf_counter() - s0
            lowlevel.run = original
        rd = result_dict(result)
        save_npz(out / 's.npz', S=np.asarray(result.S), freqs=np.asarray(result.freqs))
        numeric = {k:np.asarray(v) for k,v in rd.items() if isinstance(v,(np.ndarray,jax.Array,np.generic,float,int,bool,complex))}
        save_npz(out / 'diagnostics.npz', **numeric)
        write_json(out / 'diagnostics.json', rd)
        status.update(completed=1, frequencies=len(result.freqs),
                      max_column_power=float(np.max(np.sum(abs(np.asarray(result.S))**2,axis=0))))
        return status
    except BaseException as exc:
        status.update(exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
        print(status['traceback'], flush=True)
        return status
    finally:
        lowlevel.run = original
        status['wall_s'] = time.perf_counter() - t0
        write_json(out / 'status.json', status)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixture', choices=list(PATHS))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--require-gpu', action='store_true')
    ap.add_argument('--attempt', type=int, default=0)
    ap.add_argument('--lo-shift', type=float, default=0.0)
    ap.add_argument('--hi-shift', type=float, default=0.0)
    args = ap.parse_args()
    global LO_SHIFT, HI_SHIFT, SELECTION
    LO_SHIFT, HI_SHIFT = args.lo_shift, args.hi_shift
    if not args.dry_run:
        SELECTION = json.loads((OUT / 'selection.json').read_text())[args.fixture]
        assert SELECTION['A1'] == 1
        LO_SHIFT, HI_SHIFT = SELECTION['lo_shift_cells'], SELECTION['hi_shift_cells']
    jax.config.update('jax_enable_x64', args.fixture == 'cv20')
    if args.require_gpu:
        assert all(d.platform == 'gpu' for d in jax.devices()), jax.devices()
    dest = OUT / (args.fixture + f'_dry_{args.attempt:02d}' if args.dry_run else args.fixture)
    dest.mkdir()
    write_json(dest / 'provenance.json', dict(devices=[str(d) for d in jax.devices()],
               x64=int(jax.config.jax_enable_x64), source_tree=str(SRC),
               source_record=(BASE / 'PROVENANCE.txt').read_text(),
               sha256={str(p.relative_to(BASE)):sha(p) for p in (Path(__file__), SRC / PATHS[args.fixture],
               SRC / 'rfx/sparams/msl.py', SRC / 'rfx/simulation.py', SRC / 'rfx/boundaries/pec.py',
               SRC / 'rfx/boundaries/cpml.py')}))
    variants = ('inset1',)
    statuses = []
    for variant in variants:
        out = dest / variant
        out.mkdir()
        with (out / 'run.log').open('x') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
                status = execute(args.fixture, variant, out, args.dry_run)
        statuses.append(status)
        print(json.dumps(status), flush=True)
    write_json(dest / 'summary.json', statuses)

if __name__ == '__main__':
    main()

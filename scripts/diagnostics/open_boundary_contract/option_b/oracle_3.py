"""Addendum 2: execute the exported oracle method, with recorded parameter edits."""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import sweep_driver as m

ROOT = m.ROOT
ORACLE = ROOT / 'src/tests/oracle/test_pml_reflectivity.py'


def program(f0, layers, n_steps, ref_side, patch=False):
    tree = ast.parse(ORACLE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef))
    fn.decorator_list = []
    fn.name = 'run_oracle'
    edits = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ('f0', 'n_steps'):
                value = {'f0': f0, 'n_steps': n_steps}[target.id]
                if ast.literal_eval(node.value) != value:
                    edits.append([target.id, ast.unparse(node.value), repr(value)])
                    node.value = ast.Constant(value)
            if isinstance(target, ast.Name) and target.id in ('grid_ref', 'grid_cpml'):
                for kw in node.value.keywords:
                    value = None
                    if target.id == 'grid_ref' and kw.arg == 'domain':
                        value = (ref_side,) * 3
                    if target.id == 'grid_cpml' and kw.arg == 'cpml_layers':
                        value = layers
                    if value is not None and ast.literal_eval(kw.value) != value:
                        edits.append([target.id + '.' + kw.arg, ast.unparse(kw.value), repr(value)])
                        kw.value = ast.parse(repr(value), mode='eval').body
        if patch and isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == 'at':
            # The only .at indices in the committed method are its two source injections.
            ijk = node.slice.elts
            assert len(ijk) == 3
            for axis in (1, 2):
                centre = ijk[axis]
                ijk[axis] = ast.Slice(ast.BinOp(centre, ast.Sub(), ast.Constant(2)),
                                      ast.BinOp(centre, ast.Add(), ast.Constant(3)))
    if patch:
        edits.append(['source', 'single Ez node', '5 by 5 Ez nodes; all other settings unchanged'])
    at = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assert))
    fn.body.insert(at, ast.parse('_capture(locals())').body[0])
    ast.fix_missing_locations(fn)
    return ast.Module(body=[fn], type_ignores=[]), edits


def geometry(grid, source, probe):
    rec = m.grid_record(grid)
    rec.update(source_ijk=source, probe_ijk=probe)
    # Image-source distances, using the actual outer PEC node planes.
    paths = []
    for axis in range(3):
        for wall in (0, grid.shape[axis] - 1):
            image = list(source)
            image[axis] = 2 * wall - source[axis]
            paths.append(math.sqrt(sum((image[k] - probe[k])**2 for k in range(3))) * grid.dx)
    rec['earliest_geometric_wall_echo_s'] = min(paths) / 299792458.0
    rec['shortest_stencil_wall_roundtrip_steps'] = min(
        sum(abs((2 * wall - source[k] if k == axis else source[k]) - probe[k]) for k in range(3))
        for axis in range(3) for wall in (0, grid.shape[axis] - 1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['exact', 'frequency', 'patch_source'])
    ap.add_argument('--scale', type=float, default=1)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--f0-ghz', type=float, default=2)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()
    m.jax.config.update('jax_enable_x64', False)
    f0 = args.f0_ghz * 1e9
    steps = 400 if args.mode != 'frequency' else math.ceil(400 * 2e9 / f0)
    ref_side = .20
    if args.mode == 'frequency':
        dx = 299792458.0 / 5e9 / 20
        dt = dx / (299792458.0 * math.sqrt(3)) * .99
        # Eight cells beyond the light round-trip length, including the +3 probe offset.
        ref_side = 2 * math.ceil((299792458.0 * (steps - 1) * dt / dx + 8) / 2) * dx
    tree, edits = program(f0, args.layers, steps, ref_side, args.mode == 'patch_source')
    stem = f'{args.mode}_{args.f0_ghz:g}GHz_{args.scale:g}_{args.layers}'
    out = ROOT / ('dry_3' if args.dry else 'raw_3') / stem
    out.mkdir(parents=True)
    (out / 'executed_method.py').write_text(ast.unparse(tree) + '\n')
    oracle = m.load_module('tests/oracle/test_pml_reflectivity.py')
    row = dict(mode=args.mode, f0_Hz=f0, scale=args.scale, layers=args.layers,
               alpha_max_S_per_m=.05 * args.scale,
               f_alpha_formula_Hz=.05 * args.scale / (2 * math.pi * m.EPS0),
               record_steps=steps, reference_declared_side_m=ref_side,
               cpml_declared_side_m=.06, freq_max_Hz=5e9, edits=edits,
               oracle_sha256=hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
               commit=(m.SRC / 'PROVENANCE.txt').read_text().strip(),
               preflight_text='Not invoked by committed low-level oracle.',
               dtype='float32', method='20*log10(max(abs(ts_cpml-ts_ref))/max(abs(ts_ref)))')
    assert row['commit'] == m.EXPECTED_SHA
    if args.dry:
        for label, layers, size in [('reference', 0, ref_side), ('cpml', args.layers, .06)]:
            grid = oracle.Grid(freq_max=5e9, domain=(size,) * 3, cpml_layers=layers)
            centre = tuple(n // 2 for n in grid.shape)
            probe = (centre[0] + 3, centre[1], centre[2])
            row[label] = geometry(grid, centre, probe)
        row.update(status='build-only', window_s=[0, (steps - 1) * grid.dt])
        m.write_json(out / 'result.json', row)
        print('BUILD_ONLY', json.dumps(m.serial(row)), flush=True)
        return 0
    assert all(d.platform == 'gpu' for d in m.jax.devices())
    row['devices'] = list(map(str, m.jax.devices()))
    row['run_id'] = (ROOT / 'jobs_3' / ('oracle' if args.mode != 'frequency' else f'frequency_{args.f0_ghz:g}') / 'run_id.txt').read_text().strip()
    m.cpml._cpml_profile = m.scaled_profile(args.scale)

    def capture(v):
        for label, suffix in [('reference', 'r'), ('cpml', 'c')]:
            grid = v['grid_ref' if suffix == 'r' else 'grid_cpml']
            source = tuple(v[f'c{a}_{suffix}'] for a in 'xyz')
            probe = v['probe_ref' if suffix == 'r' else 'probe_cpml']
            row[label] = geometry(grid, source, probe)
        row.update(reflectivity_db=float(v['reflectivity_db']), peak_reference=float(v['peak_ref']),
                   peak_difference=float(v['peak_diff']), window_s=[0, (steps-1)*v['dt_r']],
                   actual_field_dtype=str(v['state_cpml'].ez.dtype))
        params = v['cpml_params']
        m.np.savez_compressed(out / 'cpml_profiles.npz', **{
            face + '_' + field: m.np.asarray(getattr(getattr(params, face), field))
            for face in m.FACES for field in m.cpml.CPMLParams._fields})
        m.np.savez_compressed(out / 'traces.npz', reference=v['ts_ref'], cpml=v['ts_cpml'], dt_s=v['dt_r'])
        # Additional reading of the same traces; this does not replace the oracle number.
        phase = m.np.exp(-2j * m.np.pi * f0 * m.np.arange(steps) * v['dt_r'])
        inc = phase @ v['ts_ref']
        dif = phase @ (v['ts_cpml'] - v['ts_ref'])
        row['same_trace_DFT_at_f0_db'] = float(20*m.np.log10(max(abs(dif / inc), 1e-30)))

    ns = dict(vars(oracle), _capture=capture)
    exec(compile(tree, str(out / 'executed_method.py'), 'exec'), ns)
    start = time.monotonic()
    with (out / 'run.log').open('w') as log:
        with contextlib.redirect_stdout(m.Tee(sys.stdout, log)), contextlib.redirect_stderr(m.Tee(sys.stderr, log)):
            try:
                ns['run_oracle'](None)
                row.update(status='complete', committed_minus40_assertion='passed')
            except AssertionError as exc:
                if 'reflectivity_db' not in row:
                    raise
                row.update(status='complete', committed_minus40_assertion='failed', assertion_text=str(exc))
            except Exception as exc:
                row.update(status='STOP', exception=repr(exc), traceback=traceback.format_exc())
                print(row['traceback'], flush=True)
            row['wall_s'] = time.monotonic()-start
            m.write_json(out / 'result.json', row)
            m.write_json(ROOT / 'results_3' / (stem+'.json'), row)
            print('ORACLE_FINAL', json.dumps(m.serial(row)), flush=True)
    return 0 if row['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())

from __future__ import annotations
import argparse
import ast
import contextlib
import inspect
import json
from pathlib import Path
import time
import traceback

import measure_base as m
from rfx.boundaries import cpml

np, jax = m.np, m.jax
ORIGINAL_PROFILE = cpml._cpml_profile
ORIGINAL_CORE = m.lowlevel.make_core_step
FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')


def scaled_profile(factor):
    tree = ast.parse(inspect.getsource(ORIGINAL_PROFILE))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'alpha' for t in node.targets):
            assert ast.unparse(node.value) == '0.05 * (1.0 - rho)'
            node.value = ast.BinOp(left=node.value, op=ast.Mult(), right=ast.Name(id='_alpha_factor', ctx=ast.Load()))
            changed += 1
    assert changed == 1
    ast.fix_missing_locations(tree)
    namespace = dict(ORIGINAL_PROFILE.__globals__, _alpha_factor=factor)
    exec(compile(tree, str(m.OUT / 'driver.py') + '::scaled_profile', 'exec'), namespace)
    fn = namespace['_cpml_profile']
    def wrapped(*args, **kwargs):
        with np.errstate(divide='raise', invalid='raise'):
            return fn(*args, **kwargs)
    return wrapped


def capture_core(out, dry, factor, label, lane):
    counter = [0]
    def observe(ctx):
        number = counter[0]
        counter[0] += 1
        params = ctx.cpml_params
        assert isinstance(params, cpml.CPMLAxisParams)
        arrays = {face + '_' + field: np.asarray(getattr(getattr(params, face), field))
                  for face in FACES for field in cpml.CPMLParams._fields}
        record = dict(call=number, factor=factor, x64=int(jax.config.jax_enable_x64),
                      context='rfx.simulation.make_core_step(ctx).cpml_params',
                      dtypes={k:str(v.dtype) for k,v in arrays.items()},
                      finite={k:int(np.isfinite(v).all()) for k,v in arrays.items()},
                      sha256={k:m.hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()})
        if not dry:
            with np.load(m.OUT / f'{lane}_dry_{label}' / 'cpml_received_00.npz') as z:
                record['dry_bit_identity'] = {k:int(v.dtype == z[k].dtype and v.shape == z[k].shape and v.tobytes() == z[k].tobytes()) for k,v in arrays.items()}
            assert all(record['dry_bit_identity'].values()), record
        m.save_npz(out / f'cpml_received_{number:02d}.npz', **arrays)
        m.write_json(out / f'cpml_received_{number:02d}.json', record)
        assert all(record['finite'].values()), record
        if dry:
            raise m.Captured()
        return ORIGINAL_CORE(ctx)
    m.lowlevel.make_core_step = observe
    return counter


def patch_run(out, dry):
    oracle = m.load_module('tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py')
    sim = oracle._build(n=2, pad_h=10, cpml=4)
    grid = sim._build_grid()
    row = dict(n=2, pad_h=10, cpml_layers=4, num_periods=float(oracle.NUM_PERIODS),
               grid_shape=list(grid.shape), dx_m=float(grid.dx), dt_s=float(grid.dt),
               x64=int(jax.config.jax_enable_x64), completed=0, timestepping_calls=0)
    assert oracle.NUM_PERIODS == 150
    try:
        result = sim.run(num_periods=oracle.NUM_PERIODS, skip_preflight=True)
        series = np.asarray(result.time_series)
        rates = oracle._late_time_log_rate_per_step(series)
        row.update(completed=1, timestepping_calls=1, steps=len(series),
                   settling_db=float(oracle._settling_db(series)),
                   rates_per_step=list(map(float, rates)), worst_rate_per_step=float(max(rates)))
        m.save_npz(out / 'time_series.npz', time_series=series, dt_s=np.asarray(grid.dt))
    except m.Captured:
        row.update(completed=1, dry_readback_completed=1)
    except BaseException as exc:
        row.update(exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
    m.write_json(out / 'result.json', row)
    return row


def cv20_dry(out):
    row = dict(completed=0, timestepping_calls=0)
    try:
        sim, changes, copied = m.build_pair('cv20', 'continued')
        grid = sim._build_grid()
        row.update(grid_shape=list(grid.shape), dx_m=float(grid.dx), dt_s=float(grid.dt),
                   changes=changes, settings=m.SETTINGS['cv20'])
        sim.compute_msl_s_matrix(**m.SETTINGS['cv20'])
        raise RuntimeError('dry readback did not intercept make_core_step')
    except m.Captured:
        row.update(completed=1, dry_readback_completed=1)
    except BaseException as exc:
        row.update(exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
    m.write_json(out / 'status.json', row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lane', choices=['cv20', 'patch'])
    ap.add_argument('label')
    ap.add_argument('--factor', required=True)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--require-gpu', action='store_true')
    args = ap.parse_args()
    jax.config.update('jax_enable_x64', args.lane == 'cv20')
    if args.require_gpu:
        assert all(d.platform == 'gpu' for d in jax.devices()), jax.devices()
    factor = None if args.factor == 'unpatched' else float(args.factor)
    out = m.OUT / f'{args.lane}_{"dry_" if args.dry else ""}{args.label}'
    out.mkdir()
    sources = [Path(__file__), m.OUT / 'measure_base.py', m.SRC / 'rfx/boundaries/cpml.py',
               m.SRC / 'rfx/simulation.py', m.SRC / m.PATHS['cv20'],
               m.SRC / 'tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py']
    m.write_json(out / 'provenance.json', dict(devices=list(map(str,jax.devices())),
                 x64=int(jax.config.jax_enable_x64), factor=factor,
                 source_record=(m.BASE/'PROVENANCE.txt').read_text(),
                 sha256={str(p.relative_to(m.BASE)):m.sha(p) for p in sources}))
    if factor is not None:
        cpml._cpml_profile = scaled_profile(factor)
    counter = capture_core(out, args.dry, factor, args.label, args.lane)
    start = time.perf_counter()
    with (out/'run.log').open('x') as log:
        with contextlib.redirect_stdout(m.Tee(m.sys.stdout, log)), contextlib.redirect_stderr(m.Tee(m.sys.stderr, log)):
            if args.lane == 'patch':
                row = patch_run(out, args.dry)
            elif args.dry:
                row = cv20_dry(out)
            else:
                row = m.execute('cv20', 'continued', out, False)
            print(json.dumps(m.serial(row)), flush=True)
    m.write_json(out/'summary.json', dict(result=row, core_calls=counter[0], wall_s=time.perf_counter()-start))


if __name__ == '__main__':
    main()

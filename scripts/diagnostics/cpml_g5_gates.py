"""G5 independent reference, physics controls, and frozen sensitivity curves."""
import argparse
import json
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np

from cpml_g5_reference import FIELDS, evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'validation/research/nu_cost/g5'
M = runpy.run_path(str(ROOT / 'tests/unit/boundaries/test_cpml_localization.py'))


def save(name, value):
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def trajectory(name, implementation, initial=0.0, null=False):
    grid = (M['Grid'](freq_max=10e9, domain=(0.012,) * 3, dx=1e-3, cpml_layers=0)
            if null else M['fixture'](name))
    shape = (grid.nx, grid.ny, grid.nz)
    params, psi = (None, None) if null else M['old'].init_cpml(grid)
    state = M['init_state'](shape)
    materials = M['init_materials'](shape)
    axes = getattr(grid, 'cpml_axes', 'xyz')
    periodic = tuple(ax not in axes for ax in 'xyz')
    pmc = getattr(grid, 'pmc_faces', set())
    center = tuple(n // 2 for n in shape)
    state = state._replace(ez=state.ez.at[center].set(jnp.float32(initial)))

    def step(carry, i):
        st, ps = carry
        if name == 'graded8':
            st = M['update_h_nu'](st, materials, grid.dt, grid.inv_dx_h, grid.inv_dy_h, grid.inv_dz_h)
        else:
            st = M['update_h'](st, materials, grid.dt, grid.dx, periodic=periodic)
        if not null:
            st, ps = implementation.apply_cpml_h(st, params, ps, grid, axes, materials)
        st = M['apply_pmc_faces'](st, pmc)
        if name == 'graded8':
            st = M['update_e_nu'](st, materials, grid.dt, grid.inv_dx, grid.inv_dy, grid.inv_dz)
        else:
            st = M['update_e'](st, materials, grid.dt, grid.dx, periodic=periodic)
        if not null:
            st, ps = implementation.apply_cpml_e(st, params, ps, grid, axes, materials)
        st = M['apply_pec'](st, axes=axes)
        pulse = jnp.exp(-((i - 20.0) / 6.0) ** 2)
        st = st._replace(ez=st.ez.at[center].add(pulse))
        return (st, ps), tuple(getattr(st, key) for key in FIELDS)

    return jax.lax.scan(step, (state, psi), jnp.arange(200))[1]


def witness():
    results = {}
    for name in M['FIXTURES']:
        arrays = []
        for impl, initial in ((M['old'], 1.0), (M['cpml'], 1.0),
                              (M['old'], np.nextafter(np.float32(1), np.float32(np.inf)))):
            result = jax.jit(lambda: trajectory(name, impl, initial))()
            arrays.append(tuple(np.asarray(x) for x in result))
        rows = {}
        for key, a, b, p in zip(FIELDS, *arrays):
            # Subtract in float64 so max|diff| represents the exact distance
            # between stored float32 values rather than another rounded op.
            candidate = np.max(np.abs(b.astype(np.float64) - a), axis=(1, 2, 3))
            sensitivity = np.max(np.abs(p.astype(np.float64) - a), axis=(1, 2, 3))
            bad = np.flatnonzero(candidate > sensitivity)
            rows[key] = {'candidate_curve': candidate.tolist(), 'sensitivity_curve': sensitivity.tolist(),
                         'violating_steps': (bad + 1).tolist(),
                         'first_violation': None if len(bad) == 0 else int(bad[0] + 1),
                         'max_excess': float(np.max(candidate - sensitivity))}
            if len(bad):
                i = bad[0]
                print('G5-5', name, key, 'FAIL', 'first_step', i+1,
                      'candidate', candidate[i], 'sensitivity', sensitivity[i], flush=True)
            else:
                print('G5-5', name, key, 'PASS', flush=True)
        results[name] = rows
        save('g5_5', results)
        del arrays, result
        jax.clear_caches()
        if any(row['violating_steps'] for row in rows.values()):
            print('STOP G5-5; remaining witness fixtures not run', flush=True)
            break


def identity():
    import os
    assert '--xla_disable_hlo_passes=fusion' in os.environ.get('XLA_FLAGS', '')
    results = {}
    for name in M['FIXTURES']:
        a = jax.jit(lambda: M['runner'](name, M['old'])[0])()
        b = jax.jit(lambda: M['runner'](name, M['cpml'])[0])()
        rows = M['differences'](a, b)
        print(name, '\n'.join(rows), flush=True)
        results[name] = {'pass': all('equal=True' in row for row in rows), 'arrays': rows}
        save('g5_2', results)
        if not results[name]['pass']:
            print('STOP G5-2', flush=True)
            break


def reference():
    results = {}
    baseline = np.load(OUT / 'unflagged.npz')
    for name in M['FIXTURES']:
        grid = M['fixture'](name)
        params, psi = M['old'].init_cpml(grid)
        # Match profile initialization inside the original jitted G4 runner.
        # Grid geometry/scalars retain their source values; arrays retain the
        # float32 values that actually enter the scan.
        compiled_params, _ = jax.jit(lambda: M['old'].init_cpml(M['fixture'](name)))()
        params = params._replace(**{face: getattr(compiled_params, face)
                                   for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')})
        ref = evaluate(grid, params, psi)
        result = jax.jit(lambda: M['runner'](name, M['cpml'])[0])()
        candidate = {key: np.asarray(getattr(group, key))
                     for group in result for key in group._fields if key != 'step'}
        rows = {}
        for key, r in ref.items():
            a = baseline[name + '/' + key].astype(np.float64)
            b = candidate[key].astype(np.float64)
            ae, be = np.abs(a - r), np.abs(b - r)
            bad = be > 1.05 * ae
            rows[key] = {'violating_elements': int(np.count_nonzero(bad)),
                         'elements': int(r.size), 'baseline_max_error': float(ae.max()),
                         'candidate_max_error': float(be.max()),
                         'max_norm_pass': bool(be.max() <= 1.05 * ae.max()),
                         'max_elementwise_excess': float(np.max(be - 1.05 * ae))}
            print('G5-3', name, key, json.dumps(rows[key]), flush=True)
        results[name] = rows
        save('g5_3', results)
        # G5-3 is additional, non-decisive after an effective passing G5-2;
        # finish all seven requested reference reports without retuning.
        jax.clear_caches()


def physics():
    results = {}
    oracle = runpy.run_path(str(ROOT / 'tests/unit/boundaries/test_cpml.py'))
    oracle_globals = oracle['_reflection_db_vs_clean_reference'].__globals__
    for operator in ('init_cpml', 'apply_cpml_h', 'apply_cpml_e'):
        oracle_globals[operator] = getattr(M['cpml'], operator)
    db = float(oracle['_reflection_db_vs_clean_reference'](2e9, 5e9, 8, 250))
    expected = -68.26476397028848
    results['reflection'] = {'candidate_db': db, 'baseline_median_db': expected,
                             'window_db': 0.0, 'difference_db': abs(db - expected),
                             'pass': db == expected}
    print('G5-4a', results['reflection'], flush=True)
    a = tuple(np.asarray(x) for x in jax.jit(lambda: trajectory('uniform8', M['old'], null=True))())
    b = tuple(np.asarray(x) for x in jax.jit(lambda: trajectory('uniform8', M['cpml'], null=True))())
    counts = {key: int(np.count_nonzero(x != y)) for key, x, y in zip(FIELDS, a, b)}
    def energy(fields):
        return sum(0.5 * (8.8541878128e-12 if i < 3 else 1.25663706212e-6)
                   * np.sum(x.astype(np.float64)**2, axis=(1, 2, 3)) for i, x in enumerate(fields))
    ea, eb = energy(a), energy(b)
    results['null'] = {'differing_elements': counts, 'energy_equal': bool(np.array_equal(ea, eb)),
                       'baseline_energy_step200': float(ea[-1]), 'candidate_energy_step200': float(eb[-1]),
                       'pass': not any(counts.values()) and bool(np.array_equal(ea, eb))}
    print('G5-4b', results['null'], flush=True)
    rows = {}
    for label, impl in (('baseline', M['old']), ('candidate', M['cpml'])):
        _, psi = jax.jit(lambda: M['runner']('mixed8', impl)[0])()
        rows[label] = {key: int(np.count_nonzero(np.asarray(getattr(psi, key))))
                       for key in psi._fields if key.endswith(('_xlo', '_yhi'))}
    results['nonabsorbing_psi'] = rows
    print('G5-4c', json.dumps(rows), flush=True)
    save('g5_4', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gate', choices=('identity', 'reference', 'physics', 'witness'))
    args = parser.parse_args()
    globals()[args.gate]()

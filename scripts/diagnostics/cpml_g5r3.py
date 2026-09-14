"""CPML localization r3 gates: reroll-bounded reference distance, divergence and AD parity.

Frozen rules: docs/design_notes/20260915_cpml_localization_r3_predeclaration.md.
The comparator for every gate is the contraction-suppressed baseline (a known
harmless recompilation of the SAME arithmetic), so "no worse than a reroll" is
what is tested. Baseline = validation/research/nu_cost/g4/cpml_baseline.py;
candidate = .../cpml_candidate.py selected by RFX_G4_REJECTED_CANDIDATE=1 in
tests/unit/boundaries/test_cpml_localization.py. rfx/ is never imported as the
candidate here.

Stages (each one attempt, in order): controls -> candidate -> ad.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'validation/research/nu_cost/g5r3'
SCRATCH = Path(os.environ.get('G5R3_SCRATCH', '/tmp/g5r3_scratch'))
FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
FLAGS = '--xla_disable_hlo_passes=fusion --xla_cpu_enable_fast_math=false'
LEGACY_FLAGS = '--xla_cpu_use_fusion_emitters=false --xla_cpu_enable_fast_math=false'
ULP = float(np.nextafter(np.float32(1), np.float32(np.inf)))
STEPS = 200
FIXTURES = ['uniform8', 'graded8', 'mixed8', 'uniform4', 'uniform16', 'periodic8', 'kappa8']


def load_M():
    return runpy.run_path(str(ROOT / 'tests/unit/boundaries/test_cpml_localization.py'))


def trajectory(M, name, impl, seed, history):
    """r2's seeded trajectory (controls.py) generalised: seed=None reproduces
    runner() (zeros + pulse, the float64-reference form); seed=1.0 / ULP is the
    divergence-witness form. Returns per-step fields (history) or final fields."""
    import jax
    import jax.numpy as jnp
    grid = M['fixture'](name)
    shape = (grid.nx, grid.ny, grid.nz)
    params, psi = M['old'].init_cpml(grid)
    state = M['init_state'](shape)
    materials = M['init_materials'](shape)
    axes = getattr(grid, 'cpml_axes', 'xyz')
    periodic = tuple(ax not in axes for ax in 'xyz')
    pmc = getattr(grid, 'pmc_faces', set())
    center = tuple(n // 2 for n in shape)
    if seed is not None:
        state = state._replace(ez=state.ez.at[center].set(jnp.float32(seed)))
    nu = name == 'graded8'

    def step(carry, i):
        st, ps = carry
        if nu:
            st = M['update_h_nu'](st, materials, grid.dt, grid.inv_dx_h, grid.inv_dy_h, grid.inv_dz_h)
        else:
            st = M['update_h'](st, materials, grid.dt, grid.dx, periodic=periodic)
        st, ps = impl.apply_cpml_h(st, params, ps, grid, axes, materials)
        st = M['apply_pmc_faces'](st, pmc)
        if nu:
            st = M['update_e_nu'](st, materials, grid.dt, grid.inv_dx, grid.inv_dy, grid.inv_dz)
        else:
            st = M['update_e'](st, materials, grid.dt, grid.dx, periodic=periodic)
        st, ps = impl.apply_cpml_e(st, params, ps, grid, axes, materials)
        st = M['apply_pec'](st, axes=axes)
        pulse = jnp.exp(-((i - 20.0) / 6.0) ** 2)
        st = st._replace(ez=st.ez.at[center].add(pulse))
        out = tuple(getattr(st, k) for k in FIELDS) if history else None
        return (st, ps), out

    (final, _), hist = jax.jit(lambda: jax.lax.scan(step, (state, psi), jnp.arange(STEPS)))()
    if history:
        return {k: np.asarray(h) for k, h in zip(FIELDS, hist)}
    return {k: np.asarray(getattr(final, k)) for k in FIELDS}


def reference64(M, name):
    """r2's float64 reference, with the compiled-parameter match r2 used."""
    import jax
    sys.path.insert(0, str(ROOT / 'scripts/diagnostics'))
    from cpml_g5_reference import evaluate
    grid = M['fixture'](name)
    params, psi = M['old'].init_cpml(grid)
    compiled, _ = jax.jit(lambda: M['old'].init_cpml(M['fixture'](name)))()
    params = params._replace(**{f: getattr(compiled, f) for f in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')})
    return {k: np.asarray(v, np.float64) for k, v in evaluate(grid, params, psi).items() if k in FIELDS}


def rms(a):
    return float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))


def worker(label, impl_name, fixtures):
    """Subprocess body: runs under whatever XLA_FLAGS the parent set."""
    M = load_M()
    impl = M['old'] if impl_name == 'old' else M['cpml']
    expected = os.environ.get('G5R3_EXPECT_FLAGS', FLAGS if label.startswith('flag') else '')
    assert os.environ.get('XLA_FLAGS', '') == expected, os.environ.get('XLA_FLAGS')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for name in fixtures:
        hist = trajectory(M, name, impl, 1.0, True)
        final = trajectory(M, name, impl, None, False)
        np.savez(SCRATCH / f'{label}_{name}.npz', **{f'hist_{k}': v for k, v in hist.items()},
                 **{f'final_{k}': v for k, v in final.items()})
        print(label, name, 'saved', flush=True)


def run_worker(label, impl_name, fixtures, flags):
    env = dict(os.environ, XLA_FLAGS=flags, PYTHONDONTWRITEBYTECODE='1', G5R3_EXPECT_FLAGS=flags)
    if impl_name == 'cand':
        env['RFX_G4_REJECTED_CANDIDATE'] = '1'
    log = OUT / f'{label}.log'
    with log.open('w') as stream:
        p = subprocess.run([sys.executable, __file__, 'worker', label, impl_name, ','.join(fixtures)],
                           env=env, stdout=stream, stderr=subprocess.STDOUT)
    assert p.returncode == 0, f'worker {label} failed, see {log}'


def provenance(M):
    import jax
    import rfx
    return {'rfx_file': rfx.__file__, 'jax': jax.__version__,
            'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--short', '--', 'rfx', 'scripts', 'tests', 'validation'], cwd=ROOT, text=True).strip()),
            'candidate_env': os.environ.get('RFX_G4_REJECTED_CANDIDATE', ''),
            'baseline_is_live_cpml': Path(ROOT / 'rfx/boundaries/cpml.py').read_bytes() == Path(ROOT / 'validation/research/nu_cost/g4/cpml_baseline.py').read_bytes()}


def controls():
    assert os.environ.get('RFX_G4_REJECTED_CANDIDATE', '') != '1', 'controls stage must not see the candidate'
    OUT.mkdir(parents=True, exist_ok=True)
    M = load_M()
    out = {'stage': 'controls', 'provenance': provenance(M), 'flags': FLAGS, 'ulp_seed': ULP, 'fixtures': {}}
    run_worker('flag', 'old', FIXTURES, FLAGS)                     # contraction-suppressed baseline
    ratios = []
    for name in FIXTURES:
        base_h = trajectory(M, name, M['old'], 1.0, True)
        ulp_h = trajectory(M, name, M['old'], ULP, True)
        base_f = trajectory(M, name, M['old'], None, False)
        ref = reference64(M, name)
        z = np.load(SCRATCH / f'flag_{name}.npz')
        row = {}
        for k in FIELDS:
            d_flag = np.max(np.abs(z[f'hist_{k}'].astype(np.float64) - base_h[k]), axis=(1, 2, 3))
            d_ulp = np.max(np.abs(ulp_h[k].astype(np.float64) - base_h[k]), axis=(1, 2, 3))
            rb, rf = rms(base_f[k] - ref[k]), rms(z[f'final_{k}'].astype(np.float64) - ref[k])
            farther = np.abs(z[f'final_{k}'] - ref[k]) > np.abs(base_f[k] - ref[k])
            differ = z[f'final_{k}'] != base_f[k]
            ratios.append(abs(rf / rb - 1.0) if rb > 0 else 0.0)
            row[k] = {'rms_base': rb, 'rms_flag': rf, 'rms_ratio_flag': rf / rb if rb else None,
                      'flag_farther_fraction': float(farther[differ].mean()) if differ.any() else None,
                      'flag_differing_elements': int(differ.sum()),
                      'd_flag_curve': d_flag.tolist(), 'd_ulp_curve': d_ulp.tolist()}
        out['fixtures'][name] = row
        np.savez(SCRATCH / f'base_{name}.npz', **{f'hist_{k}': v for k, v in base_h.items()},
                 **{f'final_{k}': v for k, v in base_f.items()}, **{f'ref_{k}': v for k, v in ref.items()})
        print('controls', name, {k: round(row[k]['rms_ratio_flag'], 4) for k in FIELDS}, flush=True)
    out['m'] = float(max(ratios))
    (OUT / 'controls.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print('DERIVED m =', out['m'], flush=True)


def candidate():
    assert os.environ.get('RFX_G4_REJECTED_CANDIDATE') == '1'
    M = load_M()
    ctrl = json.loads((OUT / 'controls.json').read_text())
    m = ctrl['m']
    out = {'stage': 'candidate', 'provenance': provenance(M), 'm': m, 'fixtures': {}, 'g53_fired': [], 'g55_fired': []}
    for name in FIXTURES:
        cand_h = trajectory(M, name, M['cpml'], 1.0, True)
        cand_f = trajectory(M, name, M['cpml'], None, False)
        b = np.load(SCRATCH / f'base_{name}.npz')
        row = {}
        for k in FIELDS:
            base_h, base_f, ref = b[f'hist_{k}'], b[f'final_{k}'], b[f'ref_{k}']
            d_c = np.max(np.abs(cand_h[k].astype(np.float64) - base_h), axis=(1, 2, 3))
            d_f = np.asarray(ctrl['fixtures'][name][k]['d_flag_curve'])
            viol = np.flatnonzero(d_c > d_f)
            rb = ctrl['fixtures'][name][k]['rms_base']
            rc = rms(cand_f[k].astype(np.float64) - ref)
            farther = np.abs(cand_f[k] - ref) > np.abs(base_f - ref)
            differ = cand_f[k] != base_f
            g53 = rc <= rb * (1 + m)
            g55 = len(viol) == 0
            row[k] = {'rms_cand': rc, 'rms_base': rb, 'rms_ratio_cand': rc / rb if rb else None,
                      'window_ratio': 1 + m, 'g53_pass': bool(g53),
                      'cand_farther_fraction': float(farther[differ].mean()) if differ.any() else None,
                      'cand_differing_elements': int(differ.sum()),
                      'maxnorm_ratio_r2_style': float(np.max(np.abs(cand_f[k] - ref)) / np.max(np.abs(base_f - ref))) if np.max(np.abs(base_f - ref)) else None,
                      'd_cand_curve': d_c.tolist(), 'g55_pass': bool(g55),
                      'g55_first_violation': None if g55 else int(viol[0] + 1),
                      'g55_max_ratio_cand_over_flag': float(np.max(np.where(d_f > 0, d_c / np.where(d_f > 0, d_f, 1), 0.0)))}
            if not g53:
                out['g53_fired'].append(f'{name}/{k}')
            if not g55:
                out['g55_fired'].append(f'{name}/{k}')
        out['fixtures'][name] = row
        print('candidate', name, 'G5-3p', all(r['g53_pass'] for r in row.values()), 'G5-5p', all(r['g55_pass'] for r in row.values()), flush=True)
    out['g53_verdict'] = 'FIRED' if out['g53_fired'] else 'HELD'
    out['g55_verdict'] = 'FIRED' if out['g55_fired'] else 'HELD'
    (OUT / 'candidate.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print('G5-3prime', out['g53_verdict'], out['g53_fired'], '| G5-5prime', out['g55_verdict'], out['g55_fired'], flush=True)


def reroll():
    """Diagnostic D1 (declared in the note before running): a second harmless
    reroll of the baseline (legacy emitter) judged by the G5-5' per-step
    predicate against the first (no-fusion). No candidate involved."""
    assert os.environ.get('RFX_G4_REJECTED_CANDIDATE', '') != '1'
    M = load_M()
    ctrl = json.loads((OUT / 'controls.json').read_text())
    run_worker('legacy', 'old', FIXTURES, LEGACY_FLAGS)
    out = {'stage': 'reroll_diagnostic', 'provenance': provenance(M), 'legacy_flags': LEGACY_FLAGS, 'fixtures': {}}
    for name in FIXTURES:
        b = np.load(SCRATCH / f'base_{name}.npz'); z = np.load(SCRATCH / f'legacy_{name}.npz')
        row = {}
        for k in FIELDS:
            d_l = np.max(np.abs(z[f'hist_{k}'].astype(np.float64) - b[f'hist_{k}']), axis=(1, 2, 3))
            d_f = np.asarray(ctrl['fixtures'][name][k]['d_flag_curve'])
            v1 = np.flatnonzero(d_l > d_f); v2 = np.flatnonzero(d_f > d_l)
            row[k] = {'legacy_differs_from_base': bool(np.any(d_l > 0)),
                      'steps_legacy_over_flag': int(len(v1)), 'steps_flag_over_legacy': int(len(v2)),
                      'max_ratio_legacy_over_flag': float(np.max(np.where(d_f > 0, d_l / np.where(d_f > 0, d_f, 1), 0))),
                      'max_ratio_flag_over_legacy': float(np.max(np.where(d_l > 0, d_f / np.where(d_l > 0, d_l, 1), 0))),
                      'd_legacy_curve': d_l.tolist()}
        out['fixtures'][name] = row
        print('D1', name, {k: (row[k]['steps_legacy_over_flag'], row[k]['steps_flag_over_legacy'], round(row[k]['max_ratio_legacy_over_flag'], 2)) for k in FIELDS}, flush=True)
    (OUT / 'reroll_diagnostic.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')


def grads(M, impl):
    import jax
    import jax.numpy as jnp
    dz0 = jnp.asarray(np.linspace(0.5e-3, 1e-3, 12), jnp.float32)

    def loss(dz, eps):
        (st, _), _ = M['runner']('graded8', impl, dz=dz, eps=eps, steps=STEPS)
        return jnp.sum(st.ez ** 2)
    g_dz = jax.grad(loss, 0)(dz0, jnp.float32(1.0))
    g_eps = jax.grad(loss, 1)(dz0, jnp.float32(1.0))
    return np.asarray(g_dz, np.float64), float(g_eps), float(loss(dz0, jnp.float32(1.0)))


def ad_worker(label):
    M = load_M()
    g_dz, g_eps, l0 = grads(M, M['old'])
    np.savez(SCRATCH / f'ad_{label}.npz', g_dz=g_dz, g_eps=g_eps, l0=l0)
    print(label, 'grad saved', flush=True)


def ad():
    assert os.environ.get('RFX_G4_REJECTED_CANDIDATE') == '1'
    M = load_M()
    env = dict(os.environ, XLA_FLAGS=FLAGS, PYTHONDONTWRITEBYTECODE='1'); env.pop('RFX_G4_REJECTED_CANDIDATE')
    with (OUT / 'ad_flag.log').open('w') as stream:
        p = subprocess.run([sys.executable, __file__, 'ad_worker', 'flag'], env=env, stdout=stream, stderr=subprocess.STDOUT)
    assert p.returncode == 0
    gb_dz, gb_eps, lb = grads(M, M['old'])
    gc_dz, gc_eps, lc = grads(M, M['cpml'])
    z = np.load(SCRATCH / 'ad_flag.npz'); gf_dz, gf_eps = z['g_dz'], float(z['g_eps'])
    dz_ok = float(np.linalg.norm(gc_dz - gb_dz)) <= float(np.linalg.norm(gf_dz - gb_dz))
    eps_ok = abs(gc_eps - gb_eps) <= abs(gf_eps - gb_eps)
    out = {'stage': 'ad', 'provenance': provenance(M),
           'loss_base': lb, 'loss_cand': lc, 'g_dz_base': gb_dz.tolist(), 'g_dz_cand': gc_dz.tolist(), 'g_dz_flag': gf_dz.tolist(),
           'g_eps_base': gb_eps, 'g_eps_cand': gc_eps, 'g_eps_flag': gf_eps,
           'dz_norm_cand_minus_base': float(np.linalg.norm(gc_dz - gb_dz)), 'dz_norm_flag_minus_base': float(np.linalg.norm(gf_dz - gb_dz)),
           'dz_max_rel_cand_vs_base': float(np.max(np.abs(gc_dz - gb_dz) / np.maximum(np.abs(gb_dz), 1e-300))),
           'all_finite': bool(np.isfinite(gc_dz).all() and np.isfinite(gc_eps)),
           'g_ad_dz': 'HELD' if dz_ok else 'FIRED', 'g_ad_eps': 'HELD' if eps_ok else 'FIRED'}
    (OUT / 'ad.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print('G5-AD dz', out['g_ad_dz'], out['dz_norm_cand_minus_base'], '<=', out['dz_norm_flag_minus_base'],
          '| eps', out['g_ad_eps'], abs(gc_eps - gb_eps), '<=', abs(gf_eps - gb_eps), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('controls', 'candidate', 'ad', 'reroll', 'worker', 'ad_worker'))
    p.add_argument('rest', nargs='*')
    a = p.parse_args()
    if a.stage == 'worker':
        worker(a.rest[0], a.rest[1], a.rest[2].split(','))
    elif a.stage == 'ad_worker':
        ad_worker(a.rest[0])
    else:
        globals()[a.stage]()

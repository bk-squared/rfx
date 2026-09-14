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
FLAGS2 = '--xla_disable_hlo_passes=fusion,algsimp --xla_cpu_enable_fast_math=false'   # r3b: algsimp reassociates whole-array vs slab adds
EXTRA_FIXTURES = ['thin232', 'overlap', 'kappa5_pmc']
ULP = float(np.nextafter(np.float32(1), np.float32(np.inf)))
STEPS = 200
FIXTURES = ['uniform8', 'graded8', 'mixed8', 'uniform4', 'uniform16', 'periodic8', 'kappa8']


def _guard_tree():
    import rfx
    got = Path(rfx.__file__).resolve()
    assert str(got).startswith(str(ROOT) + '/'), f'rfx imported from {got}, not from {ROOT} (fix PYTHONPATH, absolute)'


def load_M():
    _guard_tree()
    return runpy.run_path(str(ROOT / 'tests/unit/boundaries/test_cpml_localization.py'))


def extra_fixture(M, name):
    """r3b fixtures from the adversarial review (thin232 is the reviewer's, verbatim in shape/params)."""
    import jax.numpy as jnp
    Grid = M['Grid']
    if name == 'thin232':
        return Grid(freq_max=1e10, domain=(.001, .002, .001), dx=.001, cpml_layers=2, cpml_axes='', kappa_max=5.), 3, True
    if name == 'overlap':
        return Grid(freq_max=1e10, domain=(.006,) * 3, dx=.001, cpml_layers=2), 50, False
    if name == 'kappa5_pmc':
        return Grid(freq_max=1e10, domain=(.006,) * 3, dx=.001, cpml_layers=2, kappa_max=5., pmc_faces={'y_hi'}), 50, False
    raise KeyError(name)


def run_extra(M, name, impl):
    """Random-field/psi/material seeded run (reviewer's thin232 protocol) for the extra fixtures; final state."""
    import jax
    import jax.numpy as jnp
    grid, steps, randomize = extra_fixture(M, name)
    sh = (grid.nx, grid.ny, grid.nz)
    params, psi = M['old'].init_cpml(grid)   # baseline parameters for both, as in every gate
    st = M['init_state'](sh); mat = M['init_materials'](sh)
    axes = getattr(grid, 'cpml_axes', 'xyz'); pmc = getattr(grid, 'pmc_faces', set())
    rng = np.random.default_rng(913)
    if randomize:
        st = st._replace(**{k: jnp.asarray(rng.normal(size=sh).astype('float32')) for k in FIELDS})
        psi = psi._replace(**{k: jnp.asarray(rng.normal(size=getattr(psi, k).shape).astype('float32')) for k in psi._fields})
        mat = mat._replace(eps_r=jnp.asarray(rng.uniform(1, 5, sh).astype('float32')), mu_r=jnp.asarray(rng.uniform(1, 3, sh).astype('float32')))
    else:
        st = st._replace(ez=st.ez.at[tuple(n // 2 for n in sh)].set(1.0))
    periodic = tuple(ax not in axes for ax in 'xyz')

    def step(c, i):
        s_, p_ = c
        if not randomize:
            s_ = M['update_h'](s_, mat, grid.dt, grid.dx, periodic=periodic)
        s_, p_ = impl.apply_cpml_h(s_, params, p_, grid, axes, mat)
        s_ = M['apply_pmc_faces'](s_, pmc)
        if not randomize:
            s_ = M['update_e'](s_, mat, grid.dt, grid.dx, periodic=periodic)
        s_, p_ = impl.apply_cpml_e(s_, params, p_, grid, axes, mat)
        if not randomize:
            s_ = M['apply_pec'](s_, axes=axes)
        return (s_, p_), None
    (fs, fp), _ = jax.jit(lambda: jax.lax.scan(step, (st, psi), jnp.arange(steps)))()
    out = {k: np.asarray(getattr(fs, k)) for k in FIELDS}
    out.update({'psi_' + k: np.asarray(getattr(fp, k)) for k in fp._fields})
    return out


def identity2_worker(label):
    """Subprocess under FLAGS2 (or none for the effectiveness control): saves final fields+psi of the
    frozen baseline and, when RFX_G4_REJECTED_CANDIDATE=1, of the candidate too, on all fixtures."""
    M = load_M()
    want = os.environ.get('G5R3_EXPECT_FLAGS', '')
    assert os.environ.get('XLA_FLAGS', '') == want, (os.environ.get('XLA_FLAGS'), want)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    impls = {'base': M['old']}
    if os.environ.get('RFX_G4_REJECTED_CANDIDATE') == '1':
        impls['cand'] = M['cpml']
    for name in FIXTURES + EXTRA_FIXTURES:
        for tag, impl in impls.items():
            if name in EXTRA_FIXTURES:
                arrs = run_extra(M, name, impl)
            else:
                (fs, fp), _ = M['runner'](name, impl)
                arrs = {k: np.asarray(getattr(fs, k)) for k in FIELDS}
                arrs.update({'psi_' + k: np.asarray(getattr(fp, k)) for k in fp._fields})
            np.savez(SCRATCH / f'id2_{label}_{tag}_{name}.npz', **arrs)
        print(label, name, 'saved', flush=True)


def identity2():
    """G5-2b: bit-identity under FLAGS2 on 7 + 3 fixtures, with an effectiveness control."""
    M = load_M()
    out = {'stage': 'identity2', 'provenance': provenance(M), 'flags': FLAGS2, 'fixtures': {}, 'fired': []}
    for label, flags, cand in (('id2flag', FLAGS2, True), ('id2plain', '', False)):
        env = dict(os.environ, XLA_FLAGS=flags, PYTHONDONTWRITEBYTECODE='1', G5R3_EXPECT_FLAGS=flags)
        if cand:
            env['RFX_G4_REJECTED_CANDIDATE'] = '1'
        else:
            env.pop('RFX_G4_REJECTED_CANDIDATE', None)
        with (OUT / f'{label}.log').open('w') as stream:
            p = subprocess.run([sys.executable, __file__, 'identity2_worker', label], env=env, stdout=stream, stderr=subprocess.STDOUT)
        assert p.returncode == 0, f'{label} failed, see {OUT / (label + ".log")}'
    eff_changed = 0
    for name in FIXTURES + EXTRA_FIXTURES:
        b = np.load(SCRATCH / f'id2_id2flag_base_{name}.npz'); c = np.load(SCRATCH / f'id2_id2flag_cand_{name}.npz'); u = np.load(SCRATCH / f'id2_id2plain_base_{name}.npz')
        rows = {}
        for k in b.files:
            eq = bool(np.array_equal(b[k], c[k])); nd = int(np.count_nonzero(b[k] != c[k]))
            eff = int(np.count_nonzero(b[k] != u[k])); eff_changed += eff
            rows[k] = {'equal': eq, 'differing': nd, 'max_abs': float(np.max(np.abs(b[k].astype(np.float64) - c[k]))) if nd else 0.0, 'flag_vs_plain_differing': eff}
            if not eq:
                out['fired'].append(f'{name}/{k}')
        out['fixtures'][name] = rows
        print('identity2', name, 'ALL EQUAL' if all(r['equal'] for r in rows.values()) else 'DIFF ' + str([k for k, r in rows.items() if not r['equal']]), flush=True)
    out['effectiveness_flag_vs_plain_changed_elements'] = eff_changed
    out['verdict'] = 'INCONCLUSIVE' if eff_changed == 0 else ('FIRED' if out['fired'] else 'HELD')
    (OUT / 'identity2.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print('G5-2b', out['verdict'], 'fired:', out['fired'], '| effectiveness changed elements:', eff_changed, flush=True)


def s_stat():
    """G5-5''' (reviewer's trajectory statistic) on the STORED curves; no FDTD."""
    ctl = json.loads((OUT / 'controls.json').read_text()); c = json.loads((OUT / 'candidate.json').read_text())
    rows = {}; S = 0.0; fired = []
    for name in FIXTURES:
        rows[name] = {}
        for k in FIELDS:
            dc = np.asarray(c['fixtures'][name][k]['d_cand_curve']); df = np.asarray(ctl['fixtures'][name][k]['d_flag_curve'])
            rc, rf = float(np.sqrt(np.mean(dc ** 2))), float(np.sqrt(np.mean(df ** 2)))
            ratio = (rc / rf) if rf > 0 else (0.0 if rc == 0 else float('inf'))
            rows[name][k] = {'rms_t_cand': rc, 'rms_t_flag': rf, 'ratio': ratio}
            S = max(S, ratio)
            if ratio > 1.0:
                fired.append(f'{name}/{k}')
    out = {'stage': 's_stat', 'definition': 'S = max_(fixture,field) RMS_t(d_cand)/RMS_t(d_flag); gate S <= 1; zero denominator requires d_cand == 0',
           'S': S, 'verdict': 'FIRED' if fired else 'HELD', 'fired': fired, 'rows': rows}
    (OUT / 's_stat.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print("G5-5''' S =", S, out['verdict'], fired, flush=True)


def sparam_worker(label):
    """Arm A of the WR-90 dz-graded falsifier with the CPML implementation selected by label:
    base = frozen baseline swapped into rfx.boundaries.cpml (production binds these names at call time);
    live = the localized kernel in rfx/. Runs under whatever XLA_FLAGS the parent set."""
    _guard_tree()
    import importlib.util
    import rfx.boundaries.cpml as live
    if label.startswith('base'):
        spec = importlib.util.spec_from_file_location('cpml_frozen', ROOT / 'validation/research/nu_cost/g4/cpml_baseline.py')
        frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
        frozen.CPMLAxisParams = live.CPMLAxisParams; frozen.CPMLState = live.CPMLState
        for fn in ('init_cpml', 'apply_cpml_h', 'apply_cpml_e'):
            setattr(live, fn, getattr(frozen, fn))
    sys.path.insert(0, str(ROOT / 'scripts/diagnostics'))
    import wr90_dz_dispatch_falsifier as W
    rec, s_ = W._run_arm('A')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(SCRATCH / f'sparam_{label}.npz', S=np.asarray(s_), freqs=np.asarray(rec.get('freqs_hz', [])))
    (OUT / f'sparam_{label}_rec.json').write_text(json.dumps({k: v for k, v in rec.items() if isinstance(v, (int, float, str, list, bool)) or v is None}, indent=1, default=str) + '\n')
    print(label, 'S shape', np.asarray(s_).shape, 'wall', rec.get('wallclock_s'), flush=True)


def sparam():
    """G5-S: matched graded-mesh complex S-parameter, reroll-bounded."""
    M = load_M()
    for label, flags in (('base', ''), ('live', ''), ('baseflag', FLAGS2)):
        env = dict(os.environ, XLA_FLAGS=flags, PYTHONDONTWRITEBYTECODE='1'); env.pop('RFX_G4_REJECTED_CANDIDATE', None)
        with (OUT / f'sparam_{label}.log').open('w') as stream:
            p = subprocess.run([sys.executable, __file__, 'sparam_worker', label], env=env, stdout=stream, stderr=subprocess.STDOUT)
        assert p.returncode == 0, f'sparam {label} failed, see {OUT / ("sparam_" + label + ".log")}'
    Sb = np.load(SCRATCH / 'sparam_base.npz')['S']; Sl = np.load(SCRATCH / 'sparam_live.npz')['S']; Sf = np.load(SCRATCH / 'sparam_baseflag.npz')['S']
    d_live, d_flag = float(np.max(np.abs(Sl - Sb))), float(np.max(np.abs(Sf - Sb)))
    out = {'stage': 'sparam', 'provenance': provenance(M), 'arm': 'A (WR-90 dz-graded, flux-normalized, 9 bins)',
           'max_abs_dS_live_vs_base': d_live, 'max_abs_dS_flag_vs_base': d_flag,
           'all_finite': bool(np.isfinite(Sl).all() and np.isfinite(Sb).all() and np.isfinite(Sf).all()),
           'S11_left_abs_base': np.abs(Sb[0, 0, :]).tolist(), 'S11_left_abs_live': np.abs(Sl[0, 0, :]).tolist(),
           'verdict': 'HELD' if (d_live <= d_flag and np.isfinite(Sl).all()) else 'FIRED'}
    (OUT / 'sparam.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print('G5-S', out['verdict'], 'max|dS| live', d_live, '<= flag', d_flag, flush=True)



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
    p.add_argument('stage', choices=('controls', 'candidate', 'ad', 'reroll', 'identity2', 'identity2_worker', 's_stat', 'sparam', 'sparam_worker', 'worker', 'ad_worker'))
    p.add_argument('rest', nargs='*')
    a = p.parse_args()
    if a.stage == 'worker':
        worker(a.rest[0], a.rest[1], a.rest[2].split(','))
    elif a.stage == 'ad_worker':
        ad_worker(a.rest[0])
    elif a.stage == 'identity2_worker':
        identity2_worker(a.rest[0])
    elif a.stage == 'sparam_worker':
        sparam_worker(a.rest[0])
    else:
        globals()[a.stage]()

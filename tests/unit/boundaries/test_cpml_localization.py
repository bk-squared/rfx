"""Frozen G4 comparison against c30d3020; see the pre-declaration note."""
import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries import cpml
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.pmc import apply_pmc_faces
from rfx.core.yee import init_state, init_materials, update_h, update_e, update_h_nu, update_e_nu
from rfx.grid import Grid
from rfx.nonuniform import make_nonuniform_grid

_BASE = Path(__file__).resolve().parents[3] / 'validation/research/nu_cost/g4/cpml_baseline.py'
_spec = importlib.util.spec_from_file_location('cpml_g4_baseline', _BASE)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
# The frozen functions receive the existing production parameter/carry types.
old.CPMLAxisParams = cpml.CPMLAxisParams
old.CPMLState = cpml.CPMLState

# Opt-in reproduction of the rejected candidate; ordinary unit tests compare
# the production implementation with the frozen baseline.
if os.environ.get('RFX_G4_REJECTED_CANDIDATE') == '1':
    _candidate_spec = importlib.util.spec_from_file_location('cpml_g4_candidate', _BASE.with_name('cpml_candidate.py'))
    _candidate = importlib.util.module_from_spec(_candidate_spec)
    _candidate_spec.loader.exec_module(_candidate)
    _candidate.CPMLAxisParams = cpml.CPMLAxisParams
    _candidate.CPMLState = cpml.CPMLState
    cpml = _candidate

FIXTURES = ['uniform8', 'graded8', 'mixed8', 'uniform4', 'uniform16', 'periodic8', 'kappa8']


def fixture(name, dz=None):
    layers = 16 if name == 'uniform16' else 4 if name == 'uniform4' else 8
    if name == 'graded8':
        if dz is None:
            dz = np.linspace(0.5e-3, 1e-3, 12)
        return make_nonuniform_grid((0.012, 0.012), dz, 1e-3, layers)
    kwargs = {}
    if name == 'mixed8':
        kwargs.update(pec_faces={'x_lo'}, pmc_faces={'y_hi'},
                      face_layers={'x_hi': 4, 'y_lo': 6, 'z_lo': 4, 'z_hi': 8})
    if name == 'periodic8':
        kwargs['cpml_axes'] = 'yz'
    if name == 'kappa8':
        kwargs['kappa_max'] = 3.0
    return Grid(freq_max=10e9, domain=(0.012,) * 3, dx=1e-3,
                cpml_layers=layers, **kwargs)


def runner(name, implementation, dz=None, eps=None, steps=200, history=False):
    grid = fixture(name, dz)
    shape = (grid.nx, grid.ny, grid.nz)
    params, psi = cpml.init_cpml(grid)
    state = init_state(shape)
    materials = init_materials(shape)
    if eps is not None:
        materials = materials._replace(eps_r=jnp.broadcast_to(eps, shape))
    axes = getattr(grid, 'cpml_axes', 'xyz')
    periodic = tuple(ax not in axes for ax in 'xyz')
    pmc = getattr(grid, 'pmc_faces', set())
    center = tuple(n // 2 for n in shape)
    nu = name == 'graded8'

    def step(carry, i):
        st, ps = carry
        if nu:
            st = update_h_nu(st, materials, grid.dt, grid.inv_dx_h, grid.inv_dy_h, grid.inv_dz_h)
        else:
            st = update_h(st, materials, grid.dt, grid.dx, periodic=periodic)
        st, ps = implementation.apply_cpml_h(st, params, ps, grid, axes, materials)
        st = apply_pmc_faces(st, pmc)
        if nu:
            st = update_e_nu(st, materials, grid.dt, grid.inv_dx, grid.inv_dy, grid.inv_dz)
        else:
            st = update_e(st, materials, grid.dt, grid.dx, periodic=periodic)
        st, ps = implementation.apply_cpml_e(st, params, ps, grid, axes, materials)
        st = apply_pec(st, axes=axes)
        pulse = jnp.exp(-((i - 20.0) / 6.0) ** 2)
        st = st._replace(ez=st.ez.at[center].add(pulse))
        return (st, ps), (st, ps) if history else None

    return jax.lax.scan(step, (state, psi), jnp.arange(steps))


def differences(reference, candidate):
    records = []
    for group, a, b in zip(('field', 'psi'), reference, candidate):
        for name in a._fields:
            if group == 'field' and name not in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
                continue
            x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
            assert x.dtype == y.dtype == np.float32
            assert np.isfinite(x).all() and np.isfinite(y).all()
            count = np.count_nonzero(x != y)
            records.append(f'{name}: equal={np.array_equal(x, y)}, differing={count}, max_abs={np.max(np.abs(x-y)):.17g}')
    return records


# r3b contract (docs/design_notes/20260915_cpml_localization_r3_predeclaration.md):
# the localized kernel is the same arithmetic compiled differently, so a
# bit-identity assertion under DEFAULT flags is a statement about XLA's
# contraction/reassociation choices, not about the kernel. The arithmetic
# identity is asserted where it is testable -- with fusion and algsimp
# suppressed -- and under default flags the divergence from the frozen
# baseline is bounded by a harmless recompilation of the baseline itself.
IDENTITY_FLAGS = '--xla_disable_hlo_passes=fusion,algsimp --xla_cpu_enable_fast_math=false'
_HELPER = Path(__file__).resolve().parents[3] / 'scripts/diagnostics/cpml_g5r3.py'


def _subprocess(stage, label, flags, candidate, *extra):
    import subprocess
    import sys
    env = dict(os.environ, XLA_FLAGS=flags, G5R3_EXPECT_FLAGS=flags, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    env.pop('RFX_G4_REJECTED_CANDIDATE', None)
    if candidate:
        env['RFX_G4_REJECTED_CANDIDATE'] = '1'
    p = subprocess.run([sys.executable, str(_HELPER), stage, label, *extra], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]


@pytest.mark.parametrize('name', FIXTURES + ['thin232'])
def test_cpml_localization_identity(name, tmp_path):
    """Bit-identity against the frozen baseline with contraction AND algebraic
    reassociation suppressed (G5-2b). One subprocess pair per test file run."""
    scratch = tmp_path / 'id2'
    os.environ['G5R3_SCRATCH'] = str(scratch)
    _subprocess('identity2_worker', 'id2flag', IDENTITY_FLAGS, True)
    b = np.load(scratch / f'id2_id2flag_base_{name}.npz')
    c = np.load(scratch / f'id2_id2flag_cand_{name}.npz')
    bad = [k for k in b.files if not np.array_equal(b[k], c[k])]
    assert not bad, f'{name}: not bit-identical under suppressed contraction on {bad}'


@pytest.mark.parametrize('name', ['uniform8', 'kappa8'])
def test_cpml_localization_reroll_bounded(name, tmp_path):
    """Default flags: RMS over 200 steps of the candidate-vs-baseline divergence
    must not exceed that of a contraction-suppressed recompilation of the
    baseline (the reroll control the test produces itself)."""
    import runpy
    scratch = tmp_path / 'rb'
    os.environ['G5R3_SCRATCH'] = str(scratch)
    _subprocess('worker', 'flag', IDENTITY_FLAGS, False, 'old', name)
    G = runpy.run_path(str(_HELPER))
    base = G['trajectory'](G['load_M'](), name, old, 1.0, True)
    cand = G['trajectory'](G['load_M'](), name, cpml, 1.0, True)
    z = np.load(scratch / f'flag_{name}.npz')
    for k in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        d_c = np.max(np.abs(cand[k].astype(np.float64) - base[k]), axis=(1, 2, 3))
        d_f = np.max(np.abs(z[f'hist_{k}'].astype(np.float64) - base[k]), axis=(1, 2, 3))
        rc, rf = np.sqrt(np.mean(d_c ** 2)), np.sqrt(np.mean(d_f ** 2))
        assert (rc == 0 if rf == 0 else rc <= rf), (name, k, rc, rf)


@pytest.mark.parametrize('variable', ['dz_profile', 'eps_r'])
def test_cpml_localization_ad(variable):
    dz = jnp.linspace(0.5e-3, 1e-3, 12)
    # Fixed mesh stays concrete when only eps is differentiated. Indexing a
    # closed-over JAX array inside jit would trace the host grid builder.
    dz_host = np.asarray(dz)
    shape = (fixture('graded8').nx, fixture('graded8').ny, fixture('graded8').nz)
    eps = jnp.full(shape, 1.5, dtype=jnp.float32)

    def objective(value, implementation):
        st = runner('graded8', implementation,
                    dz=value if variable == 'dz_profile' else dz_host,
                    eps=value if variable == 'eps_r' else eps)[0][0]
        return sum(jnp.sum(getattr(st, field) ** 2) for field in ('ex', 'ey', 'ez'))

    value = dz if variable == 'dz_profile' else eps
    a = np.asarray(jax.jit(jax.grad(lambda v: objective(v, old)))(value))
    b = np.asarray(jax.jit(jax.grad(lambda v: objective(v, cpml)))(value))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.max(np.abs(a)) > 0 and np.max(np.abs(b)) > 0
    error = np.max(np.abs(a-b)) / max(np.max(np.abs(a)), 1e-30)
    print(f'{variable}: relative_max={error:.17g}, ref_max={np.max(np.abs(a)):.17g}, max_abs={np.max(np.abs(a-b)):.17g}')
    assert error <= 1e-6

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


@pytest.mark.parametrize('name', FIXTURES)
def test_cpml_localization_identity(name):
    reference = jax.jit(lambda: runner(name, old)[0])()
    candidate = jax.jit(lambda: runner(name, cpml)[0])()
    records = differences(reference, candidate)
    print('\n' + name + '\n' + '\n'.join(records))
    assert all('equal=True' in record for record in records), '\n'.join(records)


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

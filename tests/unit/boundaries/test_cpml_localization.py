"""Eager arithmetic identity and compiled pulse agreement with frozen CPML."""
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
from rfx.model.materials import electric_cell_sizes
from tests.contracts.path_equivalence.comparison import (
    compare, component_peaks, wave_impedance_range,
)

_product = cpml

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

FIXTURES = ['uniform8', 'graded8', 'mixed8', 'uniform4', 'uniform16', 'periodic8', 'kappa8', 'asymmetric']
FIELD_NAMES = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')

# First difference of a finite Gaussian; endpoints zero, zero sum to rounding.
_gaussian = np.exp(-((np.arange(-1, 81) - 20.0) / 6.0) ** 2).astype(np.float32)
_gaussian[0] = _gaussian[-1] = 0
ZERO_MEAN_PULSE = np.diff(_gaussian)


def psi_layout(state, *, to_old):
    """Convert xyz storage to/from derivative, component, remaining-axis order."""
    converted = {}
    for name in state._fields:
        _, component, face = name.split('_')
        order = ["xyz".index(face[0]), "xyz".index(component[1])]
        order += [axis for axis in range(3) if axis not in order]
        converted[name] = getattr(state, name).transpose(
            order if to_old else tuple(np.argsort(order)))
    return state._replace(**converted)


def fixture(name, dz=None):
    if name == 'asymmetric':
        grid = Grid(freq_max=10e9, domain=(0.012, 0.016, 0.018), dx=1e-3,
                    cpml_layers=8, pec_faces={'z_hi'},
                    face_layers=dict(zip(
                        ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi'),
                        (3, 5, 4, 6, 0, 7))))
        assert grid.face_pads == (3, 5, 4, 6, 0, 0)
        assert len(set(grid.shape)) == 3
        assert 'z_lo' not in grid.pec_faces and grid.pad_z_lo == 0
        return grid
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


def runner(name, implementation, dz=None, eps=None, steps=200, history=False,
           zero_mean=False, peak_history=False, capture_steps=(), checkpoint_interval=0):
    grid = fixture(name, dz)
    shape = (grid.nx, grid.ny, grid.nz)
    params, psi = cpml.init_cpml(grid)
    if cpml is _product and implementation is not _product:
        psi = psi_layout(psi, to_old=True)
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
            st = update_e_nu(st, materials, grid.dt, grid.inv_dx, grid.inv_dy, grid.inv_dz,
                             cell_sizes=electric_cell_sizes(grid))
        else:
            st = update_e(st, materials, grid.dt, grid.dx, periodic=periodic)
        st, ps = implementation.apply_cpml_e(st, params, ps, grid, axes, materials)
        st = apply_pec(st, axes=axes)
        if zero_mean:
            samples = jnp.asarray(ZERO_MEAN_PULSE)
            pulse = jnp.where(i < samples.size, samples[jnp.minimum(i, samples.size - 1)], 0.0)
        else:
            pulse = jnp.exp(-((i - 20.0) / 6.0) ** 2)
        st = st._replace(ez=st.ez.at[center].add(pulse))
        recorded = (st, ps) if history else None
        if peak_history:
            recorded = {key: jnp.max(jnp.abs(value), initial=0)
                        for key, value in st._asdict().items() if key in FIELD_NAMES}
        return (st, ps), recorded

    if capture_steps:
        assert jax.config.jax_disable_jit
        carry, snapshots = (state, psi), []
        for i in range(max(capture_steps)):
            carry, _ = step(carry, jnp.asarray(i, dtype=jnp.int32))
            if i + 1 in capture_steps:
                snapshots.append(carry)
        return snapshots, None
    if checkpoint_interval:
        def block(carry, indices):
            carry, peaks = jax.lax.scan(step, carry, indices)
            return carry, (carry, peaks)
        return jax.lax.scan(block, (state, psi),
                            jnp.arange(steps).reshape(-1, checkpoint_interval))
    return jax.lax.scan(step, (state, psi), jnp.arange(steps))


def _arrays(state, *, reference=False):
    fields, psi = state
    if reference or cpml is not _product:
        psi = psi_layout(psi, to_old=False)
    return {**{key: np.asarray(getattr(fields, key)) for key in FIELD_NAMES},
            **{key: np.asarray(getattr(psi, key)) for key in psi._fields}}


def _assert_identity(reference, candidate, *, peaks=None):
    a, b = _arrays(reference, reference=True), _arrays(candidate)
    assert len(a) == len(b) == 30
    # Jitted null-by-symmetry psi has no scale; family-scale 27-62 ULP stays flat at 200/400/800, fields 0.28.
    # Psi stays bitwise-judged with jit off; rfx #952 reviewer decision 2026-10-08.
    for key in a if peaks is None else FIELD_NAMES:
        assert a[key].dtype == b[key].dtype == np.float32
        assert a[key].shape == b[key].shape
        assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all()
        if peaks is None:
            np.testing.assert_array_equal(a[key].view(np.uint32), b[key].view(np.uint32),
                                          err_msg=key)
        else:
            compare(a[key], b[key], record=key, kind='step', measurements=[],
                    peak=peaks[key])


# Static scene, base vs base: H 2,160 / 5,070 record-peak ULP at 200 / 400 steps.
# E 4.5 / 4 ULP; rfx-archive record 20261007-acc-952 (scan vs jitted steps).
@pytest.mark.parametrize('name', FIXTURES)
def test_cpml_localization_identity_eager(name):
    with jax.disable_jit():
        counts = (1, 10, 50) if name in ('uniform8', 'graded8', 'asymmetric') else (1, 10)
        references = runner(name, old, capture_steps=counts)[0]
        candidates = runner(name, cpml, capture_steps=counts)[0]
        for reference, candidate in zip(references, candidates):
            _assert_identity(reference, candidate)


@pytest.mark.parametrize('name', FIXTURES)
def test_cpml_localization_identity_pulse(name):
    _, (references, a) = jax.jit(lambda: runner(
        name, old, steps=800, zero_mean=True, peak_history=True, checkpoint_interval=200))()
    _, (candidates, b) = jax.jit(lambda: runner(
        name, cpml, steps=800, zero_mean=True, peak_history=True, checkpoint_interval=200))()
    failures = []
    for steps in (200, 400, 800):
        blocks = steps // 200
        reference = jax.tree.map(lambda value: value[blocks - 1], references)
        candidate = jax.tree.map(lambda value: value[blocks - 1], candidates)
        ah = {key: value[:blocks] for key, value in a.items()}
        bh = {key: value[:blocks] for key, value in b.items()}
        # All eight scenes are vacuum, nonmagnetic; every cell and step contributes.
        peaks = component_peaks(ah, bh, paired_impedance=wave_impedance_range(1.0))
        try:
            _assert_identity(reference, candidate, peaks=peaks)
        except AssertionError as error:
            failures.append(f'{steps} steps: {error}')
    assert not failures, '\n'.join(failures)


@pytest.mark.parametrize('compiled_bar', (False, True))
def test_identity_comparison_rejects_each_changed_array(compiled_bar):
    grid = fixture('asymmetric')
    _, psi = cpml.init_cpml(grid)
    state = init_state(grid.shape)
    reference = (state, psi_layout(psi, to_old=True))
    peaks = dict.fromkeys(FIELD_NAMES, 1.0) if compiled_bar else None
    groups = ((0, FIELD_NAMES),) if compiled_bar else ((0, FIELD_NAMES), (1, psi._fields))
    for group, names in groups:
        for name in names:
            candidate = [state, psi]
            value = getattr(candidate[group], name)
            candidate[group] = candidate[group]._replace(**{name: value.at[0, 0, 0].add(1)})
            with pytest.raises(AssertionError):
                _assert_identity(reference, tuple(candidate), peaks=peaks)


@pytest.mark.parametrize('faces', [None, ('x_lo', 'y_hi')], ids=['all', 'subset'])
def test_jitted_cpml_leaves_unapplied_psi_bitwise_untouched(faces):
    grid = fixture('asymmetric')
    params, psi = cpml.init_cpml(grid)
    rng = np.random.default_rng(952)
    state = init_state(grid.shape)
    state = state._replace(**{key: jnp.asarray(rng.standard_normal(grid.shape), jnp.float32)
                              for key in FIELD_NAMES})
    # Nonzero sentinels expose writes even where the field correction is zero.
    psi = psi._replace(**{key: jnp.asarray(rng.standard_normal(value.shape), jnp.float32)
                          for key, value in psi._asdict().items()})
    requested = set(cpml.ALL_FACES if faces is None else faces)
    applied = {face for face, depth in zip(cpml.ALL_FACES, grid.face_pads)
               if depth > 0 and face in requested}
    untouched = {key for key in psi._fields
                 if f'{key[-3]}_{key[-2:]}' not in applied}
    assert {'psi_ex_zlo', 'psi_ex_zhi'} <= untouched
    if faces is not None:
        assert {'psi_ey_xhi', 'psi_ez_ylo'} <= untouched
    kwargs = {} if faces is None else {'faces': faces}

    def update(st, ps):
        st, ps = cpml.apply_cpml_h(st, params, ps, grid, **kwargs)
        return cpml.apply_cpml_e(st, params, ps, grid, **kwargs)

    _, updated = jax.jit(update)(state, psi)
    for key in untouched:
        np.testing.assert_array_equal(np.asarray(getattr(updated, key)).view(np.uint32),
                                      np.asarray(getattr(psi, key)).view(np.uint32),
                                      err_msg=key)


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
    # Frozen program vs itself under another trace: up to 5.5e-6.
    # Per-leaf gradient peak; rfx-archive record 20261007-acc-952.
    assert error <= 1e-4

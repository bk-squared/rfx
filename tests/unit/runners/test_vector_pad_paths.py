"""Vector padding on the adjoint scan, the vmap material sweep and the memory planners.

Each path pads by the stepping layer (``rfx.stepping.vector_pad``) and must
return the unpadded run's bytes on the CPU, or, for a case listed in
``NOT_BITWISE`` (measured not bitwise), stay within bar (ii) of
``tests/unit/runners/test_vector_pad.py`` at N, 2N and 4N steps, each output
judged on its own largest magnitude (a probe series per column). The memory
planners estimate from the extents the step allocates.
"""

import dataclasses
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.stepping import vector_pad
from tests.unit.runners.test_vector_pad import (
    FREQS, N_STEPS, OBSERVED, RECORD_BAR, TARGETS, _ratio, _walls, assert_bars,
    assert_bitwise, assert_padded, observed, run_record,
)

ADJOINT_STEPS = 160
SWEEP_STEPS = 60
NOT_BITWISE = {
    ('design', 'adjoint', 'aligned'), ('design', 'adjoint', 'odd'),
    ('design_h', 'adjoint', 'aligned'), ('design_h', 'adjoint', 'odd'),
    ('design', 'autodiff', 'aligned'), ('design', 'autodiff', 'odd'),
    ('design face', 'autodiff', 'aligned'), ('design face', 'autodiff', 'odd'),
    ('vmap', 'pec', 'odd'),
    ('coax', 'forward', 'odd'),
}


COAX_TARGETS = ('aligned', 'odd', 'aligned_slow')
COAX_STEPS = 300


def _coax_scene():
    dx = .0005
    sim = Simulation(freq_max=10e9, domain=(.0115, .0115, .052), dx=dx,
                     boundary='cpml', cpml_layers=4, snap='declared')
    disc = Cylinder(center=(.006, .006, .002 - dx / 2), radius=.0035, height=dx, axis='z')
    sim.add(disc, material='pec')
    sim.add_coaxial_port((.006, .006, .002), face='top', pin_length=dx, terminates=disc)
    assert tuple(sim._build_grid().shape) == (32, 32, 113)
    return sim


def _coax_observed(result):
    out = observed(result)
    names = ([f.name for f in dataclasses.fields(result)] if dataclasses.is_dataclass(result)
             else result._fields)
    for name in names:
        if name in OBSERVED or not (name.startswith('coax') or name == 'sparam_time_records'):
            continue
        for path, leaf in jax.tree_util.tree_leaves_with_path(getattr(result, name)):
            if isinstance(leaf, (np.ndarray, jax.Array)) and np.issubdtype(np.asarray(leaf).dtype, np.number):
                out[name + jax.tree_util.keystr(path)] = np.asarray(leaf)
    return out


def _coax_plans(plans, target):
    assert_padded(plans, f'coax/{target}')
    expected = (33, 33, 113) if target == 'odd' else (32, 32, 116)
    assert all(p.shape == (32, 32, 113) and p.padded == expected for p in plans), plans
    print(f'COAX PLAN {target}: {[(p.shape, p.padded) for p in plans]}', file=sys.stderr)


def _coax_arrival(result, label):
    samples = [np.asarray(r.samples) for r in result.sparam_time_records or ()
               if r.kind == 'coax']
    assert samples, f'{label}: no coaxial time records'
    peak = max(float(np.abs(a).max()) for a in samples)
    first60 = max(float(np.abs(a[:60]).max()) for a in samples)
    ratio = _ratio(peak, first60)
    print(f'COAX ARRIVAL {label}: P={peak!r} P60={first60!r} P/P60={ratio!r}',
          file=sys.stderr)
    assert ratio >= 1e6, f'{label}: coax arrival P/P60={ratio!r} < 1e6'


def _coax_run_record(target, n_steps, **extra):
    from tests.unit.runners import test_vector_pad
    results = []

    def collect(result):
        results.append(result)
        assert np.isfinite(np.asarray(result.s_params)).all(), 'coax s_params not finite'
        return _coax_observed(result)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(test_vector_pad, 'observed', collect)
        out, plans = run_record(_coax_scene(), target, n_steps=n_steps,
                                compute_s_params=True, s_param_freqs=FREQS, **extra)
    return out, plans, results[0]


@pytest.mark.parametrize('target', COAX_TARGETS)
def test_coax_port_run_padded(target):

    def records(n):
        ref, off, result = _coax_run_record('off', n)
        _coax_arrival(result, f'run/{target} N={n} source=run().sparam_time_records')
        got, plans, _ = _coax_run_record(target, n)
        assert off and not any(p.active for p in off)
        _coax_plans(plans, target)
        assert any(k.startswith('coax_port_metadata') for k in ref)
        assert any(k.startswith('sparam_time_records') for k in ref)
        return ref, got

    judge(('coax', 'run', target), f'coax run/{target}', records, COAX_STEPS)
    jax.clear_caches()


@pytest.mark.parametrize('target', COAX_TARGETS)
def test_coax_port_until_decay_padded(target, monkeypatch):
    import inspect
    calls = []
    real = vector_pad.pad_carry

    def pad_carry(*args, **kwargs):
        caller = inspect.currentframe().f_back
        calls.append((caller.f_code.co_filename, caller.f_lineno))
        return real(*args, **kwargs)

    monkeypatch.setattr(vector_pad, 'pad_carry', pad_carry)

    def records(n):
        kwargs = dict(until_decay=1e-3, decay_check_interval=50,
                      decay_min_steps=200, decay_max_steps=600)
        ref, off, result = _coax_run_record('off', n, **kwargs)
        _coax_arrival(result, f'until_decay/{target} source=run().sparam_time_records')
        got, plans, padded = _coax_run_record(target, n, **kwargs)
        assert off and not any(p.active for p in off)
        _coax_plans(plans, target)
        steps, padded_steps = result.time_series.shape[0], padded.time_series.shape[0]
        stop, = (d for d in result.diagnostics if d.code == 'until_decay.stop')
        print(f'COAX DECAY {target}: steps={steps} padded_steps={padded_steps} '
              f'criterion={stop.values["criterion"]} stop_step={stop.values["stop_step"]} '
              f'pad_carry={calls} snapshot={result.snapshots is not None}', file=sys.stderr)
        assert steps == padded_steps
        return ref, got

    judge(('coax', 'until_decay', target), f'coax until_decay/{target}', records, 600)
    jax.clear_caches()


def _coax_forward_records(target, n_steps):
    sim = _coax_scene()
    shape = sim._build_grid().shape

    def objective(value):
        rho = jnp.zeros(shape).at[18, 16, 7].set(value)
        result = sim.forward(n_steps=n_steps, port_s11_freqs=FREQS,
                             pec_occupancy_override=rho, skip_preflight=True)
        return jnp.sum(jnp.abs(result.s_params) ** 2), result

    value = jnp.asarray(.25)
    with vector_pad.force_pad(target), vector_pad.record_plans() as plans:
        loss, result = objective(value)
        _, tangent = jax.jvp(lambda v: objective(v)[0], (value,), (jnp.ones_like(value),))
        gradient = jax.grad(lambda v: objective(v)[0])(value)
    return dict(_coax_observed(result), value=np.asarray(loss),
                jvp=np.asarray(tangent), grad=np.asarray(gradient)), plans, result


@pytest.mark.parametrize('target', COAX_TARGETS)
def test_coax_port_forward_value_jvp_grad_padded(target):
    def records(n):
        ref, off, result = _coax_forward_records('off', n)
        source = 'forward().sparam_time_records'
        if not any(r.kind == 'coax' for r in result.sparam_time_records or ()):
            _, _, result = _coax_run_record('off', n)
            source = 'run().sparam_time_records'
        _coax_arrival(result, f'forward/{target} N={n} source={source}')
        for key in ('value', 'jvp', 'grad'):
            print(f'COAX DERIVATIVE {target} N={n} off {key}={ref[key]!r}', file=sys.stderr)
            assert np.isfinite(ref[key]).all() and np.all(ref[key] != 0), (target, n, 'off', key)
        got, plans, _ = _coax_forward_records(target, n)
        for key in ('value', 'jvp', 'grad'):
            print(f'COAX DERIVATIVE {target} N={n} padded {key}={got[key]!r}', file=sys.stderr)
            assert np.isfinite(got[key]).all() and np.all(got[key] != 0), (target, n, 'padded', key)
        assert off and not any(p.active for p in off)
        _coax_plans(plans, target)
        return ref, got

    judge(('coax', 'forward', target), f'coax forward/{target}', records, COAX_STEPS)
    jax.clear_caches()


def own_scale_ratios(ref, got, peaks=None):
    """difference / bar (ii): a probe series per column on its family's peak over the
    unpadded record when ``peaks`` gives it ({'e', 'h', 'families'}), else on its own
    largest magnitude; every other output on its own largest magnitude."""
    out = {}
    for key, a in ref.items():
        a = np.asarray(a, np.complex128)
        d = np.abs(np.asarray(got[key], np.complex128) - a)
        if key == 'time_series':
            for column in range(a.shape[-1]):
                out[f'time_series[{column}]'] = _ratio(d[..., column].max(),
                                                       RECORD_BAR * (np.abs(a[..., column]).max() if peaks is None
                                                                     else peaks[peaks['families'][column]]))
        else:
            out[key] = _ratio(d.max(), RECORD_BAR * np.abs(a).max())
    return out


def judge(case, label, records, n_steps):
    """Bitwise at n_steps, or, for a case in NOT_BITWISE, bar (ii) at N, 2N, 4N.

    A case not listed that is not bitwise prints its bars as evidence, then fails.
    """
    lengths = (n_steps, 2 * n_steps, 4 * n_steps)
    if case in NOT_BITWISE:
        assert_bars(label, {n: own_scale_ratios(*records(n)) for n in lengths})
        return
    first = records(n_steps)
    ref, got = first[:2]
    if not all(np.array_equal(ref[k], got[k]) for k in ref):
        by_length = {n_steps: own_scale_ratios(*first)}
        by_length.update({n: own_scale_ratios(*records(n)) for n in lengths[1:]})
        try:
            assert_bars(f'{label} (evidence: not bitwise, not listed)', by_length)
        except AssertionError:
            pass
    assert_bitwise(ref, got, label)


DESIGN_DOMAIN = (22e-3, 20e-3, 14e-3)
BOX_LO, BOX_HI = (10e-3, 8e-3, 6e-3), (14e-3, 12e-3, 10e-3)
FACE_DOMAIN = (22e-3, 20e-3, 18e-3)
FACE_LO, FACE_HI = (10e-3, 8e-3, 12e-3), (14e-3, 12e-3, 18e-3)


def _design_scene(face=False, component='ez'):
    domain, box = (FACE_DOMAIN, (FACE_LO, FACE_HI)) if face else (DESIGN_DOMAIN, (BOX_LO, BOX_HI))
    boundary = BoundarySpec(x='cpml', y='cpml', z=Boundary('pec', 'pmc')) if face else 'cpml'
    sim = Simulation(freq_max=16e9, domain=domain, dx=2e-3, boundary=boundary, cpml_layers=5)
    sim.add_source((4e-3, 10e-3, 14e-3 if face else 8e-3), 'ez', amplitude_kind='field',
                   waveform=GaussianPulse(f0=8e9, bandwidth=.8, cutoff=6))
    grid = sim._build_grid()
    ix = grid.position_to_index((18e-3, 12e-3, 14e-3 if face else 6e-3))
    sim.add_dft_plane_probe(axis='x', coordinate=18e-3, component='ey' if face else component,
                            freqs=jnp.asarray([6.4e9, 8.4e9]), name='out',
                            region=(ix[1], ix[1] + 2, ix[2], ix[2] + 3))
    lo, hi = grid.position_to_index(box[0]), grid.position_to_index(box[1])
    if face:
        assert hi[2] == grid.shape[2] - 1, (hi, tuple(grid.shape))
    shape = tuple(hi[d] - lo[d] + 1 for d in range(3))
    eps = jnp.asarray(2.0 + 2.0 * np.random.default_rng(1).random(shape), jnp.float32)
    return sim, eps, box


def _design_records(mode, gradient, n_steps, face=False, component='ez'):
    sim, eps, box = _design_scene(face, component)

    def loss(e):
        result = sim.forward(design_box=box, design_eps_override=e, n_steps=n_steps,
                             checkpoint_segments=8, gradient=gradient, skip_preflight=True)
        spectrum = result.dft_planes['out'].accumulator / result.dt
        weights = jnp.arange(spectrum.size).reshape(spectrum.shape) + 1
        return jnp.sum(weights * jnp.abs(spectrum) ** 2)

    with vector_pad.force_pad(mode), vector_pad.record_plans() as plans:
        value, grad = jax.value_and_grad(loss)(eps)
    return {'value': np.asarray(value), 'grad_eps': np.asarray(grad)}, plans


@pytest.mark.parametrize('gradient', ('adjoint', 'autodiff'))
@pytest.mark.parametrize('target', TARGETS)
def test_design_box_objective_and_gradient_padded(target, gradient):
    """The adjoint scan, and autodiff through the same design box."""
    label = f'design box {gradient}/{target}'

    def records(n):
        ref, ref_plans = _design_records('off', gradient, n)
        got, plans = _design_records(target, gradient, n)
        assert ref_plans and not any(p.active for p in ref_plans)
        assert_padded(plans, f'{label} N={n}')
        return ref, got

    judge(('design', gradient, target), label, records, ADJOINT_STEPS)
    jax.clear_caches()


@pytest.mark.parametrize('target', TARGETS)
def test_design_box_h_plane_objective_adjoint_padded(target):
    """The adjoint scan with a magnetic DFT plane: the backward run's AFTER_H hook writes H in the padded state."""
    gradient = 'adjoint'
    label = f'design box H plane {gradient}/{target}'

    def records(n):
        ref, ref_plans = _design_records('off', gradient, n, component='hy')
        got, plans = _design_records(target, gradient, n, component='hy')
        assert ref_plans and not any(p.active for p in ref_plans)
        assert_padded(plans, f'{label} N={n}')
        return ref, got

    judge(('design_h', gradient, target), label, records, ADJOINT_STEPS)
    jax.clear_caches()


@pytest.mark.parametrize('target', TARGETS)
def test_design_box_against_the_pmc_face_padded(target):
    """The design box on the last physical cell of the padded z axis, against a PMC face.

    Autodiff only: gradient='adjoint' refuses PMC faces, and the curl frame is
    read only at a PMC face's magnetic image.
    """
    gradient = 'autodiff'
    label = f'design box at the PMC face {gradient}/{target}'

    def records(n):
        ref, ref_plans = _design_records('off', gradient, n, face=True)
        got, plans = _design_records(target, gradient, n, face=True)
        assert ref_plans and not any(p.active for p in ref_plans)
        assert_padded(plans, f'{label} N={n}')
        assert all(p.pad[2] > 0 for p in plans if p.active), plans
        return ref, got

    judge(('design face', gradient, target), label, records, ADJOINT_STEPS)
    jax.clear_caches()


SWEEP_EPS = (1.5, 2.2, 3.0)


def _sweep_scene(boundary, eps_r=2.2):
    sim = Simulation(freq_max=10e9, domain=(.0123, .0071, .0052), dx=.001,
                     boundary=boundary if boundary == 'cpml' else _walls(),
                     cpml_layers=4 if boundary == 'cpml' else 0, snap='declared')
    sim.add_material('brick', eps_r=eps_r)
    sim.add(Box((.004, .002, .001), (.008, .005, .004)), material='brick')
    sim.add_source((.003, .004, .003), 'ez', waveform=GaussianPulse(f0=6e9, bandwidth=.8))
    sim.add_probe((.009, .005, .002), 'ez')
    sim.add_probe((.002, .002, .004), 'hy')
    sim.add_dft_plane_probe(axis='z', coordinate=.003, component='ez',
                            freqs=jnp.array([4e9, 6e9, 8e9]), name='plane')
    return sim


@pytest.mark.parametrize('boundary', ('pec', 'cpml'))
@pytest.mark.parametrize('target', TARGETS)
def test_vmap_material_sweep_padded(target, boundary):
    from rfx.vmap_sweep import vmap_material_sweep
    label = f'vmap sweep {boundary}/{target}'

    def records(n):
        out = {}
        for mode in ('off', target):
            with vector_pad.force_pad(mode), vector_pad.record_plans() as plans:
                result = vmap_material_sweep(_sweep_scene(boundary), 'brick.eps_r',
                                             jnp.array(SWEEP_EPS), n_steps=n)
            out[mode] = ({'time_series': np.asarray(result.time_series),
                          'plane': np.asarray(result.dft_planes['plane'])}, plans)
        assert out['off'][1] and not any(p.active for p in out['off'][1])
        assert_padded(out[target][1], f'{label} N={n}')
        return out['off'][0], out[target][0], _sweep_family_peaks(boundary, n)

    judge(('vmap', boundary, target), label, records, SWEEP_STEPS)
    jax.clear_caches()


def _planner_scene():
    sim = Simulation(freq_max=10e9, domain=(.0113, .0071, .0052), dx=.001, boundary=_walls(),
                     cpml_layers=0, snap='declared')
    sim.add_source((.003, .004, .003), 'ez', waveform=GaussianPulse(f0=6e9, bandwidth=.8))
    sim.add_probe((.009, .005, .002), 'ez')
    return sim


def _expected_padded(shape, target):
    """The padded extents, from the target's rule written out here."""
    nx, ny, nz = shape
    if target == 'aligned':
        return nx, ny, 4 * math.ceil(nz / 4)
    return tuple(n + (n % 2 == 0) for n in shape)


@pytest.mark.parametrize('target', TARGETS)
def test_memory_planner_estimates_padded_extents(target):
    """The six float32 field arrays of the scan carry take the padded extents; the reported grid stays physical."""
    sim = _planner_scene()
    shape = tuple(sim._build_grid().shape)
    padded = _expected_padded(shape, target)
    assert padded != shape
    unpadded = sim.estimate_ad_memory(n_steps=40)
    with vector_pad.force_pad(target):
        estimate = sim.estimate_ad_memory(n_steps=40)
        accounting = sim._ad_memory_static_accounting()
    rest = unpadded.carry_bytes - 6 * math.prod(shape) * 4
    assert 0 <= rest < 64, (unpadded.carry_bytes, shape)
    assert estimate.carry_bytes == 6 * math.prod(padded) * 4 + rest, (estimate.carry_bytes, shape, padded)
    assert accounting['cells'] == math.prod(padded)
    assert accounting['field_bytes'] == 6 * math.prod(padded) * 4
    assert (accounting['nx'], accounting['ny'], accounting['nz']) == shape
    jax.clear_caches()


def test_memory_planner_auto_follows_the_gate_table(monkeypatch):
    sim = _planner_scene()
    shape = tuple(sim._build_grid().shape)
    assert sim._ad_memory_static_accounting()['cells'] == math.prod(shape)
    monkeypatch.setattr(vector_pad, '_MEASURED_GAIN', {vector_pad._device_kind(): 'aligned'})
    assert sim._ad_memory_static_accounting()['cells'] == math.prod(_expected_padded(shape, 'aligned'))
    with vector_pad.record_plans() as plans:
        sim.run(n_steps=4, skip_preflight=True, compute_s_params=False)
    assert plans and all(p.padded == _expected_padded(shape, 'aligned') for p in plans)


def test_auto_config_estimates_padded_extents():
    from rfx.auto_config import SimConfig, _auto_memory_mb
    config = SimConfig(dx=.001, domain=(.0113, .0071, .0052), cpml_layers=4, n_steps=10,
                       freq_range=(1e9, 10e9), margin=0., dt=1e-12, accuracy='standard')
    shape = tuple(config.grid_shape)
    with vector_pad.force_pad('aligned'):
        assert config.estimated_memory_mb == _auto_memory_mb(
            math.prod(_expected_padded(shape, 'aligned')), 4)
    assert config.estimated_memory_mb == _auto_memory_mb(math.prod(shape), 4)


def _sweep_family_peaks(boundary, n_steps):
    """Per family, the largest |field| over the unpadded run() of each swept member,
    driven cells left out (the scale of the probe series under bar (ii))."""
    from rfx import SnapshotSpec
    from rfx.stepping import uniform_setup
    from tests.unit.runners.test_vector_pad import FIELDS, _driven, family_peaks
    real, got = uniform_setup.core_step_invariants, {}

    def invariants(*args, **kwargs):
        out = real(*args, **kwargs)
        got['ctx'] = out['ctx']
        return out

    peaks = {'e': 0., 'h': 0.}
    uniform_setup.core_step_invariants = invariants
    try:
        for eps_r in SWEEP_EPS:
            with vector_pad.force_pad('off'):
                result = _sweep_scene(boundary, eps_r).run(
                    n_steps=n_steps, skip_preflight=True, compute_s_params=False,
                    snapshot=SnapshotSpec(interval=1, components=FIELDS))
            member = family_peaks({c: result.snapshots[c] for c in FIELDS}, _driven(got['ctx']))
            peaks = {f: max(peaks[f], member[f]) for f in 'eh'}
    finally:
        uniform_setup.core_step_invariants = real
    return dict(peaks, families=('e', 'h'))

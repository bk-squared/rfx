"""#2449 A1/A3: predeclared U1/G1 step and slab, independent NumPy anchors."""

from functools import lru_cache
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.measurement.settling import settling_db_from_port_records

A, B, LX, H, L = .02286, .01016, .12192, .00127, .02032
FREQS = np.arange(70, 80, dtype=float) * 1e8
C = 299792458.
# PREDECLARE dry-run U1 dt; G1 uses U1's bulk Yee estimate as declared.
YEE_DT = 2.4213500843043776e-12


def scene(mesh, kind='STEP', amplitudes=(1., 0.)):
    profile = {} if mesh == 'U1' else {
        'dx_profile': np.r_[np.full(32, H), np.full(40, .8 * H), np.full(32, H)]}
    sim = Simulation(freq_max=7.9e9, domain=(LX, A, B), dx=H,
                     boundary=dict(x='cpml', y='pec', z='pec'), cpml_layers=8, **profile)
    sim.add_material('dielectric', eps_r=2.2)
    lo, hi = (.06096, LX) if kind == 'STEP' else (.04572, .06604)
    sim.add(Box((lo, 0., 0.), (hi, A, B)), material='dielectric')
    for name, x, direction, amplitude in zip(('p1', 'p2'), (.02032, .10160), ('+x', '-x'), amplitudes):
        sim.add_waveguide_port(x, direction=direction, amplitude=amplitude, name=name,
                               f0=7.45e9, freqs=FREQS, ref_offset=2, probe_offset=3)
    return sim


def grid_for(sim, mesh):
    return sim._build_grid() if mesh == 'U1' else sim._build_nonuniform_grid()


def closed_form(kind, eps=2.2, *, yee=False):
    """Continuum and the predeclared bulk Yee estimate; no rfx S helpers."""
    omega = 2 * np.pi * FREQS

    def beta(e):
        if not yee:
            return np.sqrt(omega**2 * e / C**2 - (np.pi / A)**2)
        q = (np.sin(omega * YEE_DT / 2) / (C / np.sqrt(e) * YEE_DT))**2
        q -= (np.sin(np.pi * H / (2 * A)) / H)**2
        return 2 / H * np.arcsin(H * np.sqrt(q))

    b1, b2 = beta(1.), beta(eps)
    gamma = (b1 - b2) / (b1 + b2)
    if kind == 'STEP':
        return np.stack((np.abs(gamma), np.sqrt(1 - np.abs(gamma)**2)))
    p = np.exp(-1j * b2 * L)
    return np.stack((np.abs(gamma * (1 - p**2) / (1 - gamma**2 * p**2)),
                     np.abs((1 - gamma**2) * p / (1 - gamma**2 * p**2))))


def derivative(*, yee=False):
    return (closed_form('SLAB', 2.2 + 1e-4, yee=yee)[1]
            - closed_form('SLAB', 2.2 - 1e-4, yee=yee)[1]) / 2e-4


def emit(label, **values):
    def serial(value):
        return value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
    print('ONE_DRIVEN ' + json.dumps(dict(label=label, **values), default=serial), flush=True)


def witness(result):
    _, detail = settling_db_from_port_records(
        tuple(result.waveguide_ports.values()), freq_max=7.9e9, return_detail=True)
    shares = np.asarray(detail['share_per_bin'])
    passed = detail['status'] == 'pass' and shares.shape == (10,) and np.isfinite(shares).all() and (shares <= .01).all()
    return bool(passed), detail


@lru_cache(maxsize=1)
def select_record_time():
    """First passing candidate for each scene/mesh, then their largest time."""
    chosen = []
    for kind in ('STEP', 'SLAB'):
        for mesh in ('U1', 'G1'):
            sim = scene(mesh, kind)
            dt = float(grid_for(sim, mesh).dt)
            for k in range(9):
                duration = 5e-9 * 1.25**k
                steps = math.ceil(duration / dt)
                result = sim.run(n_steps=steps, skip_preflight=True)
                passed, detail = witness(result)
                emit('record-candidate', scene=kind, mesh=mesh, k=k, steps=steps,
                     time_s=duration, dt_s=dt, passed=passed, status=detail['status'],
                     reason=detail['reason'], share_per_bin=detail['share_per_bin'])
                if passed:
                    chosen.append(duration)
                    break
            else:
                pytest.fail(f'{kind}/{mesh}: cannot judge; no predeclared record candidate passed')
    duration = max(chosen)
    emit('record-selected', time_s=duration, first_pass_times_s=chosen)
    return duration


@pytest.fixture
def record_time(capfd):
    with capfd.disabled():
        return select_record_time()


@pytest.mark.parametrize('mesh', ['U1', 'G1'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_a1_step_closed_form(mesh, entry, record_time, capfd):
    started = time.monotonic()
    sim = scene(mesh)
    steps = math.ceil(record_time / float(grid_for(sim, mesh).dt))
    options = {'checkpoint': False} if entry == 'forward' else {}
    result = getattr(sim, entry)(n_steps=steps, skip_preflight=True, **options)
    actual = np.abs(np.stack([result.waveguide_sparams[name].s_driven for name in ('p1', 'p2')]))
    expected = closed_form('STEP')
    bar = .01 + np.abs(closed_form('STEP', yee=True) - expected)
    power = np.sum(actual**2, axis=0)
    power_bar = 2 * np.sum(expected, axis=0) * .01 + 2e-4
    passed, detail = witness(result)
    with capfd.disabled():
        emit('A1', mesh=mesh, entry=entry, steps=steps, rfx=actual, closed_form=expected,
             difference=actual - expected, bar=bar, power=power, power_bar=power_bar,
             witness_status=detail['status'], witness_share=detail['share_per_bin'],
             wall_s=time.monotonic() - started)
    assert np.isfinite(actual).all(), 'A1 finite column'
    assert passed, f'A1 cannot judge: {detail["reason"]}'
    assert (np.abs(actual - expected) <= bar).all(), 'A1 closed-form magnitude bar'
    assert (np.abs(power - 1) <= power_bar).all(), 'A1 power-sum bar'


@pytest.mark.parametrize('mesh', ['U1', 'G1'])
def test_a3_slab_gradient(mesh, record_time, capfd):
    from rfx.preflight.realization import _CampaignStaticsContext

    started = time.monotonic()
    sim = scene(mesh, 'SLAB')
    context = _CampaignStaticsContext(sim)
    base = jnp.asarray(context.realized(strict=True).materials.eps_r)
    mask = base > 1.1
    dt = float(context.grid.dt)
    expected = derivative()
    beta2 = np.sqrt(2.2 * (2 * np.pi * FREQS / C)**2 - (np.pi / A)**2)
    bar = .01 * L * (2 * np.pi * FREQS / C)**2 / beta2 + np.abs(derivative(yee=True) - expected)
    gradients = []
    for factor in (1., 1.5, 2.):
        steps = math.ceil(factor * record_time / dt)

        def transmission(p):
            result = sim.forward(eps_override=base + p * mask, n_steps=steps,
                                 skip_preflight=True, checkpoint=False)
            return jnp.abs(result.waveguide_sparams['p2'].s_driven)

        # One scalar material parameter: forward-mode AD evaluates all bins
        # in one tangent scan without retaining a reverse-mode field history.
        gradient = np.asarray(jax.jacfwd(transmission)(jnp.asarray(0., dtype=base.dtype)))
        gradients.append(gradient)
        with capfd.disabled():
            emit('A3', mesh=mesh, factor=factor, steps=steps, rfx=gradient,
                 closed_form=expected, difference=gradient - expected, bar=bar,
                 wall_s=time.monotonic() - started)
        assert np.isfinite(gradient).all(), 'A3 finite gradient'
        if factor == 1.:
            assert (np.abs(gradient - expected) <= bar).all(), 'A3 closed-form derivative value bar'
    witness_bar = .01 * np.max(np.abs(gradients[-1]))
    with capfd.disabled():
        emit('A3-witness', mesh=mesh, difference=gradients[0] - gradients[-1],
             bar=witness_bar, wall_s=time.monotonic() - started)
    assert (np.abs(gradients[0] - gradients[-1]) <= witness_bar).all(), 'A3 longer-record gradient witness'

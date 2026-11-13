"""A4: traced-map differentiation, record convergence, and Richardson oracle."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, LorentzPole, LorentzPoleMap, RelativeStrength, Simulation, gradient_record_length_witness
from tests._x64_compat import enable_x64
from tests.unit.materials.test_pole_map_runs import capture_scan, assert_float64_scan, record

BINS = np.array([3.5e9, 4.5e9])
KAPPA = (2 * np.pi * 3e9)**2
GRADED_X = np.r_[.0005, np.linspace(.00025, .00075, 58), .0005]


def parameters():
    return (jnp.log(jnp.asarray(KAPPA)), jnp.asarray(.4))


def scene(params, graded, precision='float32', *, declaration='si'):
    log_kappa, slope = params
    sim = Simulation(freq_max=6e9, domain=(.030, .016, .016), dx=.0005,
                     cpml_layers=8, precision=precision,
                     **(dict(dx_profile=GRADED_X) if graded else {}))
    pole = LorentzPoleMap(0., 2e9,
        lambda x, y, z: jnp.exp(log_kappa) * (1 + slope * (x - .015) / .0093))
    if declaration == 'relative':
        pole = LorentzPoleMap(0., 2e9, RelativeStrength(float(KAPPA),
            lambda x, y, z: jnp.exp(log_kappa - jnp.log(KAPPA)) * (1 + slope * (x - .015) / .0093)))
    elif declaration == 'scalar':
        pole = LorentzPole(0., 2e9, jnp.exp(log_kappa))
    sim.add_material('plasma', eps_r=1., lorentz_poles=[pole])
    sim.add(Box((.01035, .00435, .00435), (.01965, .01165, .01165)), material='plasma')
    sim.add_source((.00585, .00835, .00815), 'ez', waveform=GaussianPulse(f0=4e9))
    sim.add_probe((.02435, .00785, .00835), 'ez')
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    return sim, grid


def observable(params, n_steps, graded, precision='float32', *, with_trace=False, declaration='si'):
    sim, grid = scene(params, graded, precision, declaration=declaration)
    result = sim.forward(n_steps=n_steps, checkpoint=True, skip_preflight=True)
    trace = jnp.asarray(result.time_series).reshape(n_steps, -1)[:, 0]
    kernel = jnp.exp(-2j * jnp.pi * jnp.asarray(BINS)[:, None]
                     * jnp.arange(n_steps)[None, :] * grid.dt) * grid.dt
    value = jnp.abs(kernel @ trace)**2
    return (value, trace) if with_trace else value


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_a4_first_gradient(graded):
    for declaration in ('relative', 'scalar'):
        objective = lambda p: jnp.sum(observable(p, 200, graded, declaration=declaration))
        params = parameters()
        reverse = np.asarray(jax.grad(objective)(params), float)
        forward = np.asarray([jax.jvp(objective, (params,),
            (tuple(jnp.asarray(float(i == j)) for j in range(2)),))[1] for i in range(2)], float)
        active = slice(None) if declaration == 'relative' else slice(0, 1)
        relative = abs(reverse[active] - forward[active]) / abs(forward[active])
        record(f'B6.1-{declaration}-{graded}', dict(reverse=reverse.tolist(), forward=forward.tolist(),
               relative=relative.tolist()))
        assert np.isfinite(reverse).all() and np.isfinite(forward).all()
        assert np.all(reverse[active] != 0) and np.all(forward[active] != 0)
        assert np.all(relative <= 1e-4), relative


def precondition(params, steps, graded, monkeypatch, *, declaration='si'):
    sim, _ = scene(params, graded, declaration=declaration)
    body, initial, xs = capture_scan(sim, steps, monkeypatch)
    def step(carry, x):
        carry, probes = body(carry, x)
        maximum = jnp.max(jnp.stack([jnp.max(abs(v)) for v in carry['fdtd'][:3]]))
        return carry, (maximum, probes)
    maxima, probes = jax.jit(lambda: jax.lax.scan(step, initial, xs)[1])()
    maxima = np.asarray(maxima, float)
    trace = np.asarray(probes[0] if isinstance(probes, tuple) else probes).reshape(steps, -1)[:, 0]
    return dict(field_ratio=float(maxima[-1] / maxima.max()),
                settling_db=float(20 * np.log10(np.max(abs(trace[-steps // 10:])) / np.max(abs(trace)))))


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('RFX_HOST_MEMORY_WITNESS') != '1', reason=(
    'host peak 31-45 GiB; out of device memory on the 48 GB A6000 lane: run by hand on remi cpu-16 at each release '
    'with RFX_HOST_MEMORY_WITNESS=1 (docs/guides/public_docs_maintenance.md, Release checklist)'))
@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('declaration', ['scalar', 'relative'], ids=['scalar', 'map'])
def test_p7_record_witness(monkeypatch, graded, declaration):
    pinned = parameters()
    params = pinned[:1] if declaration == 'scalar' else pinned
    names = ['log_kappa0'] if declaration == 'scalar' else ['log_kappa0', 's']
    def scene_params(p):
        return (p[0], pinned[1]) if declaration == 'scalar' else p
    steps = 13500 if graded else 9000
    condition = precondition(scene_params(params), steps, graded, monkeypatch, declaration=declaration)
    record(f'P7-precondition-{declaration}-{graded}', condition)
    assert condition['field_ratio'] <= 1e-3, f'cannot judge P7: {condition}'
    witness = gradient_record_length_witness(
        lambda p, n: observable(scene_params(p), n, graded, declaration=declaration),
        params, steps, tol=1e-2, factor=1.5)
    short = np.asarray([witness.grad[f'[{i}]'] for i in range(len(params))], float)
    long = np.asarray([witness.grad_long[f'[{i}]'] for i in range(len(params))], float)
    with np.errstate(invalid='ignore', divide='ignore'):
        relative = abs(short - long) / np.maximum(abs(short), abs(long))
    record(f'P7-{declaration}-{graded}', dict(**condition, steps=steps, steps_long=int(1.5 * steps),
           parameters=names, bins=BINS.tolist(), grad=short.tolist(), grad_long=long.tolist(),
           relative=relative.tolist(), worst=float(witness.worst), passed=bool(witness.passed),
           cosine=np.asarray(witness.cosine_by_bin).tolist()))
    assert np.isfinite(short).all() and np.all(short != 0), short
    assert np.isfinite(long).all() and np.all(long != 0), long
    assert witness.passed, witness
    assert np.all(relative <= 1e-2), relative


@pytest.mark.parametrize('graded', [
    pytest.param(False, id='uniform'),
    pytest.param(True, id='graded'),
])
def test_p9_richardson(monkeypatch, graded):
    with enable_x64():
        params = parameters()
        sim, _ = scene(params, graded, 'float64')
        try:
            captured = capture_scan(sim, 2000, monkeypatch, forward=True)
        except NotImplementedError as error:
            assert graded and 'precision' in str(error), str(error)
            record('P9-graded-float64-refusal', dict(message=str(error)))
            raise
        assert_float64_scan(captured)
        objective = lambda p: observable(p, 2000, graded, 'float64')
        values, trace = observable(params, 2000, graded, 'float64', with_trace=True)
        ad = np.asarray([jax.grad(lambda p: objective(p)[k])(params)
                         for k in range(len(BINS))], float).T
        fd = []
        for h in (1e-3, 5e-4):
            derivatives = []
            for i in range(2):
                plus = tuple(v + (h if j == i else 0.) for j, v in enumerate(params))
                minus = tuple(v - (h if j == i else 0.) for j, v in enumerate(params))
                derivatives.append(np.asarray((objective(plus) - objective(minus)) / (2*h), float))
            fd.append(np.asarray(derivatives))
        richardson = (4 * fd[1] - fd[0]) / 3
        control = abs(fd[0] - fd[1]) / abs(richardson)
        error = abs(ad - richardson) / abs(richardson)
        rounding = abs(richardson) * 1e-3 / abs(np.asarray(values))[None, :]
        rounding_ad = abs(ad) * 1e-3 / abs(np.asarray(values))[None, :]
        peak = int(np.argmax(abs(np.asarray(trace)))) + 1
        record(f'P9-{graded}', dict(observable=np.asarray(values).tolist(), ad=ad.tolist(),
               fd_h=fd[0].tolist(), fd_half=fd[1].tolist(), fd_r=richardson.tolist(),
               control=control.tolist(), relative=error.tolist(), rounding=rounding.tolist(),
               rounding_ad=rounding_ad.tolist(),
               probe_peak_step=peak))
        assert 0 < peak < 2000
        assert np.all(rounding >= 1e-5), f'cannot judge P9 rounding: {rounding}'
        assert np.all(control <= 1e-4), f'cannot judge P9 FD control: {control}'
        assert np.all(error <= 1e-6), error


def traced_map_coefficients(monkeypatch, graded, declaration):
    import rfx.materials.lorentz as lorentz
    original = lorentz.lorentz_e_coeffs
    captured = []
    def observe(*args, **kwargs):
        coeffs = original(*args, **kwargs)
        if coeffs.c[0].ndim == 4:
            captured.append(coeffs)
        return coeffs
    def measure(params):
        arms = []
        for arm in ('scalar', declaration):
            captured.clear()
            sim, _ = scene(params, graded, declaration=arm)
            capture_scan(sim, 200, monkeypatch, forward=True)
            assert captured, arm
            coeffs = captured[-1]
            assert not isinstance(coeffs.damping, tuple), arm
            assert not isinstance(coeffs.restoring, tuple), arm
            reach = jnp.any(jnp.stack([v[0] != 0 for v in coeffs.c]), axis=0)
            arms.append((coeffs.damping[0], coeffs.restoring[0], reach))
        return tuple(arms)
    with monkeypatch.context() as patch:
        patch.setattr(lorentz, 'lorentz_e_coeffs', observe)
        patch.setattr('rfx.model.dispersive_poles.lorentz_e_coeffs', observe)
        return jax.device_get(jax.jit(measure)(parameters()))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('declaration', ['relative', 'si'])
def test_constant_delta_map_damping_is_scalar(monkeypatch, graded, declaration):
    """C7.4: a traced constant-delta map forms the scalar pole's damping and restoring bit for bit."""
    scalar, mapped = traced_map_coefficients(monkeypatch, graded, declaration)
    reach = np.asarray(mapped[2])
    assert reach.any() and np.array_equal(np.asarray(scalar[2]), reach)
    values = {}
    for i, name in enumerate(('damping', 'restoring')):
        got, reference = np.asarray(mapped[i])[reach], np.asarray(scalar[i])[reach]
        assert got.dtype == reference.dtype == np.float32
        values[name] = dict(bit_equal=got.tobytes() == reference.tobytes(), reached=int(reach.sum()),
                            max_difference=float(np.max(abs(got.astype(float) - reference.astype(float)))))
    record(f'C7.4-damping-{declaration}-{graded}', values)
    for name, value in values.items():
        assert value['bit_equal'], (name, value)


@pytest.mark.parametrize('declaration', ['relative', 'si'])
def test_constant_delta_map_gradient_memory(declaration):
    """C7.4: a constant-delta map's reverse pass stores no more per step than a scalar pole's."""
    def temp(arm, steps):
        objective = lambda p: jnp.sum(observable(p, steps, False, declaration=arm))
        analysis = jax.jit(jax.grad(objective)).lower(parameters()).compile().memory_analysis()
        assert analysis is not None, 'memory_analysis() returned None'
        return int(analysis.temp_size_in_bytes)
    temps = {arm: (temp(arm, 200), temp(arm, 400)) for arm in ('scalar', declaration)}
    jax.clear_caches()
    slopes = {arm: (b - a) / 200 for arm, (a, b) in temps.items()}
    ratio = slopes[declaration] / slopes['scalar']
    record(f'C7.4-memory-{declaration}', dict(temp=temps, slope=slopes, ratio=ratio))
    assert slopes['scalar'] > 0, slopes
    assert ratio <= 1.10, (ratio, slopes)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="known defect rfx#2650: a traced callable that is "
                   "not finite outside the material gives a NaN reverse-mode gradient")
@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_traced_profile_non_finite_outside_support(graded):
    """Known defect rfx#2650: a profile that is NaN outside its Box, sqrt(1 - ((x - 6 mm)/3 mm)**2), should keep a finite gradient."""
    x0, R = .006, .003   # the Box spans x .004-.008, inside |x - x0| < R

    def objective(s, clamped):
        def kappa(x, y, z):
            u = 1 - ((x - x0) / R) ** 2
            return s * KAPPA * jnp.sqrt(jnp.maximum(u, 0.) if clamped else u)
        sim = Simulation(freq_max=6e9, domain=(.012,) * 3, dx=.0005, cpml_layers=8,
                         **(dict(dx_profile=np.r_[np.full(8, .0005), np.linspace(.0003, .0007, 8),
                                                  np.full(8, .0005)]) if graded else {}))
        sim.add_material('m', eps_r=1., lorentz_poles=[LorentzPoleMap(0., 1e9, kappa)])
        sim.add(Box((.004, .004, .004), (.008, .008, .008)), material='m')
        sim.add_source((.00285, .00535, .00485), 'ez', waveform=GaussianPulse(f0=3e9))
        sim.add_probe((.00935, .00685, .00585), 'ez')
        return jnp.sum(jnp.asarray(sim.forward(n_steps=60, skip_preflight=True).time_series) ** 2)

    value, grad = jax.value_and_grad(objective)(jnp.float32(1.), False)
    control_value, control = jax.value_and_grad(objective)(jnp.float32(1.), True)
    relative = abs(float(grad) - float(control)) / abs(float(control))
    record(f'C9-nan-outside-{graded}', dict(value=float(value), grad=float(grad), control_value=float(control_value),
                                           control_grad=float(control), relative=relative))
    assert np.isfinite(value) and np.isfinite(grad), (float(value), float(grad))
    assert np.isfinite(control) and control != 0, float(control)
    assert relative <= 1e-4, (float(grad), float(control), relative)

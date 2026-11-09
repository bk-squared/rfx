"""A4: traced-map differentiation, record convergence, and Richardson oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, LorentzPoleMap, Simulation, gradient_record_length_witness
from tests._x64_compat import enable_x64
from tests.unit.materials.test_pole_map_runs import capture_scan, assert_float64_scan, record

BINS = np.array([3.5e9, 4.5e9])
KAPPA = (2 * np.pi * 3e9)**2
GRADED_X = np.r_[.0005, np.linspace(.00025, .00075, 58), .0005]


def parameters():
    return (jnp.log(jnp.asarray(KAPPA)), jnp.asarray(.4))


def scene(params, graded, precision='float32'):
    log_kappa, slope = params
    sim = Simulation(freq_max=6e9, domain=(.030, .016, .016), dx=.0005,
                     cpml_layers=8, precision=precision,
                     **(dict(dx_profile=GRADED_X) if graded else {}))
    pole = LorentzPoleMap(0., 2e9,
        lambda x, y, z: jnp.exp(log_kappa) * (1 + slope * (x - .015) / .0093))
    sim.add_material('plasma', eps_r=1., lorentz_poles=[pole])
    sim.add(Box((.01035, .00435, .00435), (.01965, .01165, .01165)), material='plasma')
    sim.add_source((.00585, .00835, .00815), 'ez', waveform=GaussianPulse(f0=4e9))
    sim.add_probe((.02435, .00785, .00835), 'ez')
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    return sim, grid


def observable(params, n_steps, graded, precision='float32', *, with_trace=False):
    sim, grid = scene(params, graded, precision)
    result = sim.forward(n_steps=n_steps, checkpoint=True, skip_preflight=True)
    trace = jnp.asarray(result.time_series).reshape(n_steps, -1)[:, 0]
    kernel = jnp.exp(-2j * jnp.pi * jnp.asarray(BINS)[:, None]
                     * jnp.arange(n_steps)[None, :] * grid.dt) * grid.dt
    value = jnp.abs(kernel @ trace)**2
    return (value, trace) if with_trace else value


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_a4_first_gradient(graded):
    gradient = jax.grad(lambda p: jnp.sum(observable(p, 200, graded)))(parameters())
    values = np.asarray(gradient, float)
    record(f'A4-first-{graded}', dict(gradient=values.tolist()))
    assert np.isfinite(values).all() and np.all(values != 0), values


def precondition(params, steps, graded, monkeypatch):
    sim, _ = scene(params, graded)
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


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_p7_record_witness(monkeypatch, graded):
    params = parameters()
    steps = 4000
    condition = precondition(params, steps, graded, monkeypatch)
    record(f'P7-precondition-{graded}', condition)
    assert condition['field_ratio'] <= 1e-3, f'cannot judge P7: {condition}'
    witness = gradient_record_length_witness(lambda p, n: observable(p, n, graded),
                                             params, steps, tol=1e-2, factor=1.5)
    short = np.asarray(list(witness.grad.values()), float)
    long = np.asarray(list(witness.grad_long.values()), float)
    relative = abs(short - long) / np.maximum(abs(short), abs(long))
    record(f'P7-{graded}', dict(**condition, grad=short.tolist(), grad_long=long.tolist(),
           relative=relative.tolist(), worst=float(witness.worst), passed=bool(witness.passed),
           cosine=np.asarray(witness.cosine_by_bin).tolist()))
    assert np.isfinite(short[1]).all() and np.all(short[1] != 0), short
    assert np.isfinite(long[1]).all() and np.all(long[1] != 0), long
    assert witness.passed, witness
    assert np.all(relative <= 1e-2), relative


@pytest.mark.parametrize('graded', [
    pytest.param(False, id='uniform'),
    pytest.param(True, id='graded', marks=pytest.mark.xfail(strict=True,
        raises=NotImplementedError, reason='graded float64 needs rfx#2538')),
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

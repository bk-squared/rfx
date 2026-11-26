"""Compensated DFT contracts P1-P5 on the predeclared record R."""
import json
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.measurement import dft, probes

FREQS = np.array((
    450000000.0, 551249984.0, 652499968.0, 753750016.0, 854999936.0,
    956249984.0, 1057499904.0, 1158749952.0, 1260000000.0, 1361249920.0,
    1462499968.0, 1563749888.0, 1664999936.0, 1766249856.0, 1867499904.0,
    1968749824.0, 2069999872.0, 2171249920.0, 2272499712.0, 2373749760.0,
    2474999808.0, 2576249856.0, 2677499904.0, 2778749696.0, 2879999744.0,
    2981249792.0, 3082499584.0, 3183749888.0, 3284999936.0, 3386249728.0,
    3487499776.0, 3588749824.0, 3689999616.0, 3791249664.0, 3892499712.0,
    3993749760.0, 4094999552.0, 4196249600.0, 4297499648.0, 4398749696.0,
    4499999744.0,
), dtype=np.float64)
DT = 1.9065748695310057e-12
N = 146000
U = 2.0 ** -24
LAST_ROWS = {}


def _record_r():
    t = np.arange(N, dtype=np.float64) * DT
    sigma = 1.687e-10
    t0 = 6 * sigma
    tau = 146000 * DT / np.log(1000)
    pulse = np.sin(2 * np.pi * 2.475e9 * t) * np.exp(
        -(t - t0) ** 2 / (2 * sigma ** 2))
    ringing = np.sin(
        2 * np.pi * t[:, None, None] * FREQS[[10, 25, 39]][None, :, None]
        + np.pi * np.arange(4)[None, None, :] / 2,
    ).sum(axis=1)
    return (pulse[:, None] + 0.1 * np.exp(-t[:, None] / tau) * ringing).astype(np.float32)


def _scene(keep):
    sim = Simulation(freq_max=4.5e9, domain=(0.10, 0.08, 0.06), dx=0.005, boundary='pec')
    sim.add_source((0.023, 0.031, 0.021), 'ez', amplitude_kind='field',
                   waveform=GaussianPulse(f0=2.475e9, bandwidth=1.0))
    grid = sim._build_grid()
    i0, j0, _ = grid.position_to_index((0.065, 0.045, 0.035))
    sim.add_dft_plane_probe(
        axis='z', coordinate=0.035, component='ez', freqs=FREQS, name='p2',
        region=(i0, i0 + 2, j0, j0 + 2),
    )
    if keep:
        sim._time_recorded_dft_planes = {'p2'}
    return sim, float(grid.dt)


def _reference(records, dt, *, point=False):
    samples = np.asarray(records).reshape(len(records), 4)
    total = np.zeros((len(FREQS), 4), dtype=np.complex128)
    for start in range(0, len(samples), 4096):
        end = min(start + 4096, len(samples))
        weights = dft.phase(jnp.arange(start, end), FREQS, dt, 'E', dtype=jnp.complex64)
        if point:
            terms = jnp.asarray(samples[start:end])[:, None, :] * (weights / dt)[:, :, None] * dt
        else:
            terms = jnp.asarray(samples[start:end], dtype=jnp.complex64)[:, None, :] * weights[:, :, None]
        total += np.sum(np.asarray(terms).astype(np.complex128), axis=0, dtype=np.complex128)
    return total


def _row(pin, n, result, reference, bar):
    actual = np.asarray(result).reshape(reference.shape).astype(np.complex128)
    scale = np.max(np.abs(reference), axis=0)
    errors = np.abs(actual - reference) / scale[None, :]
    index = np.unravel_index(np.argmax(errors), errors.shape)
    value = float(errors[index] / U)
    row = dict(pin=pin, N=n, e_over_u=value, S=scale.tolist(),
               bin=int(index[0]), channel=int(index[1]), bar=bar)
    LAST_ROWS[pin] = row
    print('U3ROW ' + json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
    assert value <= bar, row
    return row


def _scan(samples):
    zero = jnp.zeros((len(FREQS), 4), jnp.complex64)
    def body(pair, item):
        n, sample = item
        return dft.accumulate(pair, sample, n, FREQS, DT, 'E'), None
    return jax.lax.scan(body, (zero, zero), (jnp.arange(len(samples)), samples))[0]


def check_p1():
    samples = _record_r()
    compute = jax.jit(lambda x: _scan(x)[0])
    return _row('P1', N, compute(samples), _reference(samples, DT), 16)


def check_p2():
    sim, dt = _scene(True)
    result = sim.run(n_steps=36500, skip_preflight=True)
    records = result.dft_time_records
    samples = records['p2'] if isinstance(records, dict) else records[0]
    return _row('P2', 36500, result.dft_planes['p2'].accumulator,
                _reference(samples, dt), 16)


def check_p3():
    samples = _record_r()
    def single(values):
        zero = jnp.zeros(len(FREQS), jnp.complex64)
        template = probes.DFTProbe(zero, FREQS, (0, 0, 0), 'ez', N, 'rect', .25, zero)
        def body(pair, item):
            n, sample = item
            state = SimpleNamespace(ez=sample.reshape(1, 1, 1), step=n + 1)
            updated = probes.update_dft_probe(template._replace(accumulator=pair[0], compensation=pair[1]), state, DT)
            return (updated.accumulator, updated.compensation), None
        return jax.lax.scan(body, (zero, zero), (jnp.arange(len(values)), values))[0][0]
    compute = jax.jit(jax.vmap(single, in_axes=1, out_axes=1))
    return _row('P3', N, compute(samples), _reference(samples, DT, point=True), 16)


def check_p4():
    samples = _record_r()
    compute = jax.jit(lambda x: dft.transform(x, FREQS, DT, 'E'))
    return _row('P4', N, compute(samples), _reference(samples, DT), 4)


def _p5(part):
    samples = _record_r()
    rng = np.random.default_rng(0)
    weights = jnp.asarray((rng.standard_normal((41, 4))
                          + 1j * rng.standard_normal((41, 4))).astype(np.complex64))
    def plain(x):
        zero = jnp.zeros((len(FREQS), 4), jnp.complex64)
        def body(total, item):
            n, sample = item
            term = (sample[None] * dft.phase(n, FREQS, DT, 'E', dtype=jnp.complex64)[:, None]).astype(jnp.complex64)
            return total + term, None
        return jax.lax.scan(body, zero, (jnp.arange(len(x)), x))[0]
    def loss(x):
        return jnp.real(jnp.sum(jnp.conj(weights) * _scan(x)[part]))
    actual = np.asarray(jax.jit(jax.grad(loss))(samples))
    expected = (np.asarray(jax.jit(jax.grad(lambda x: jnp.real(jnp.sum(jnp.conj(weights) * plain(x)))))(samples))
                if part == 0 else np.zeros_like(actual))
    if part == 1:
        return _zero_row(actual)
    scale = float(np.max(np.abs(expected)))
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    index = np.unravel_index(np.argmax(np.abs(actual - expected)), actual.shape)
    row = dict(pin='P5a', N=N, e_over_u=error / scale / U, S=scale, bin=None,
               sample=int(index[0]), channel=int(index[1]), bar=0)
    LAST_ROWS['P5a'] = row
    print('U3ROW ' + json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
    assert np.array_equal(actual, expected), row
    return row


def _zero_row(actual):
    peak = float(np.max(np.abs(actual)))
    index = np.unravel_index(np.argmax(np.abs(actual)), actual.shape)
    row = dict(pin='P5b', N=N, e_over_u=peak / U, S=1., bin=None,
               sample=int(index[0]), channel=int(index[1]), bar=0)
    LAST_ROWS['P5b'] = row
    print('U3ROW ' + json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
    assert peak == 0., row
    return row


def check_p5a():
    return _p5(0)


def check_p5b():
    return _p5(1)


@pytest.mark.parametrize('check', [check_p1, check_p2, check_p3, check_p4, check_p5a, check_p5b])
def test_compensated_dft(check):
    assert not jax.config.x64_enabled
    with jax.default_device(jax.devices('cpu')[0]):
        check()

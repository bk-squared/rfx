"""S2 M2: independent physical clock, phase precision, and replay judges."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement.dft import accumulate, phase, transform


@pytest.mark.parametrize('kind,offset', [('E', 1.), ('H', .5)])
def test_split_phase_at_100000_steps(kind, offset):
    """§6.2: host float64 oracle, float32 scan; five representative f*dt."""
    with jax.enable_x64(False):
        dt = 1.86e-12
        bins = np.asarray([1e9, 5e9, 10e9, 50e9, 100e9], dtype=np.float32)
        actual = jax.jit(lambda n: phase(n, bins, dt, kind))(100000)
        reference = np.exp(-2j * np.pi * bins.astype(np.float64) * dt * (100000+offset))
        error = np.abs(np.angle(np.asarray(actual).astype(np.complex128) / dt / reference))
        assert np.max(error) <= 1e-6, error


@pytest.mark.parametrize('kind,offset', [('E', 1.), ('H', .5)])
@pytest.mark.parametrize('length', [37, 513])
@pytest.mark.parametrize('window', ['rect', 'hann', 'tukey'])
def test_j7_stream_and_replay(kind, offset, length, window):
    rng = np.random.default_rng(752)
    samples = rng.normal(size=(length, 2)).astype(np.float32)
    dt, freqs = .002, np.array([2.3, 17.1, 63.7])
    initial = jnp.zeros((len(freqs), 2), dtype=jnp.complex64)
    def body(acc, item):
        n, sample = item
        return accumulate(acc, sample, n, freqs, dt, kind,
                          total_steps=length, window=window, alpha=.3), None
    live = jax.jit(lambda: jax.lax.scan(body, initial, (jnp.arange(length), samples))[0])()
    replay = transform(samples, freqs, dt, kind, window=window, alpha=.3)
    np.testing.assert_array_equal(live, replay)
    # Independent host window and Fourier integral: sharing a bug cannot pass.
    from scipy.signal.windows import tukey
    weights = {'rect': np.ones(length), 'hann': np.hanning(length),
               'tukey': tukey(length, .3)}[window]
    reference = (np.exp(-2j*np.pi*freqs[:, None]*(np.arange(length)+offset)*dt)
                 @ (samples*weights[:, None])) * dt
    assert np.max(np.abs(live-reference)) <= 1e-4*np.max(np.abs(reference))


@pytest.mark.parametrize('kind', ['E', 'H'])
def test_phase_shift_keeps_magnitudes_and_channel_ratios(kind):
    records = np.random.default_rng(8).normal(size=(101, 2)).astype(np.float32)
    freqs, dt = np.asarray([3., 9.]), .002
    physical = np.asarray(transform(records, freqs, dt, kind))
    earlier = np.asarray(transform(records, freqs, dt, kind, time_shift=-1))
    factor = np.exp(-2j*np.pi*freqs*dt)
    np.testing.assert_allclose(physical, earlier*factor[:, None], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(abs(physical), abs(earlier), rtol=1e-5)
    np.testing.assert_allclose(physical[:, 0]/physical[:, 1], earlier[:, 0]/earlier[:, 1], rtol=1e-5)


@pytest.mark.parametrize('mutation,row,record', [
    ('half_step', ('_dft_planes', 'dft_plane'), 'dft_planes.ez_x_0.accumulator'),
    ('drop_h', ('_flux_monitors', 'flux'), 'flux_monitors.flux_x_0.h1_dft'),
])
def test_s0_clock_mutations_turn_the_cell_red(monkeypatch, mutation, row, record):
    """Use the real S0 runner and judge, mutating only the NU phase argument."""
    from unittest.mock import patch
    from rfx.measurement import dft
    from tests.contracts.path_equivalence import execution
    from tests.contracts.path_equivalence.builders import BUILDERS
    from tests.contracts.path_equivalence.generation import generate
    cell = next(c for c in generate(BUILDERS) if c.row == row and c.a == 'run_uniform'
                and c.b == 'run_nonuniform' and not c.graded and c.steps == 12)
    before = execution.execute(cell)
    assert not any(f.startswith(record+':') for f in before['failures']), before
    original_phase, original_solve = dft.phase, execution.solve

    def changed_phase(step, freqs, dt, kind='E', **kwargs):
        if mutation == 'half_step':
            kwargs['time_shift'] = kwargs.get('time_shift', 0.) + .5
        elif getattr(kind, 'kind', kind) == 'H':
            kind = 'E'
        return original_phase(step, freqs, dt, kind, **kwargs)

    def changed_solve(row, lane, *args):
        if lane == 'run_nonuniform':
            with patch.object(dft, 'phase', changed_phase):
                return original_solve.__wrapped__(row, lane, *args)
        return original_solve(row, lane, *args)

    monkeypatch.setattr(execution, 'solve', changed_solve)
    after = execution.execute(cell)
    from tests.contracts.path_equivalence.reporting import fingerprint
    assert any(fingerprint(f) == record+': numeric' for f in after['failures']), after


@pytest.mark.parametrize('kind,offset', [('E', 1.), ('H', .5)])
def test_initial_state_clock_and_float64_storage(kind, offset):
    """Low-level step=0 samples must not wrap to the final phase-table row."""
    with jax.enable_x64(True):
        freqs, dt = np.array([.07, .19]), .23
        weight = phase(-1, freqs, dt, kind)
        expected = np.exp(-2j*np.pi*freqs*dt*(-1+offset))*dt
        assert weight.dtype == jnp.complex128
        np.testing.assert_allclose(weight, expected, rtol=1e-14, atol=1e-15)
        records = np.array([1., -.3, .1], dtype=np.float64)
        result = transform(records, freqs, dt, kind)
        assert result.dtype == jnp.complex128
        reference = (np.exp(-2j*np.pi*freqs[:, None]*(np.arange(3)+offset)*dt) @ records)*dt
        np.testing.assert_allclose(result, reference, rtol=1e-14, atol=1e-15)


def test_plan_channel_drives_the_clock():
    from rfx.measurement.plan import Channel
    freqs = np.array([.12, .32])
    channel = Channel('projected current', 'H', ())
    np.testing.assert_array_equal(phase(3, freqs, .1, channel), phase(3, freqs, .1, 'H'))


def test_replay_retains_the_streaming_window_index():
    from rfx.measurement.dft import dft_window_weight
    records = np.arange(37, dtype=np.float32)/37
    freqs, dt = np.array([.7, 1.3]), .02
    acc = jnp.zeros(2, dtype=jnp.complex64)
    for n, value in enumerate(records):
        acc = accumulate(acc, value, n, freqs, dt, 'H', total_steps=37,
                         window='hann', window_step=n+1)
    replay = transform(records, freqs, dt, 'H', window='hann', window_step_offset=1)
    np.testing.assert_allclose(replay, acc, rtol=1e-6, atol=1e-7)
    assert float(dft_window_weight(1, 37, 'hann', .5)) > 0


@pytest.mark.parametrize('kind,offset', [('E', 1.), ('H', .5)])
def test_float32_error_stays_inside_the_rounding_bound_over_60_db(kind, offset):
    """The float32 kernel against a float64 host sum, bin by bin, on a record
    that rings for its whole length (5 GHz, decaying to e^-3) and whose
    spectrum falls 68 dB between its line and 120 GHz.

    Bound, absolute and the same at every bin: eps32 * sqrt(N) * A with
    A = sum |x(n)| dt, the largest any partial sum can be. Each of the N
    additions rounds by at most eps32/2 of the partial sum, the weight and the
    product by as much again, and independent roundings add in quadrature. It
    is a statistical bound, not a worst case (that one is N eps32 A). Relative
    to a bin it is bound / |X(f)|: 1.5e-7 * sqrt(N) at the line and 4.4e-2 at
    the weakest bin here, which is why a bin driven far below the peak cannot
    be held to the 1e-4 cross-trace bar. Measured 2026-10-07: 0.04 (E) and
    0.18 (H) of the bound; 1.2e-5 of the bin's own magnitude at worst.
    """
    with jax.enable_x64(False):
        length, dt = 12000, 1.9e-12
        n = np.arange(length)
        samples = (np.exp(-n/(length/3.)) * np.cos(2*np.pi*5e9*n*dt)).astype(np.float32)
        freqs = np.concatenate([[5e9], np.geomspace(6e9, 120e9, 40)])
        result = np.asarray(transform(samples, freqs, dt, kind))
        assert result.dtype == np.complex64
    reference = (np.exp(-2j*np.pi*freqs[:, None]*(n+offset)*dt)
                 @ samples.astype(np.float64)) * dt
    magnitude = np.abs(reference)
    assert magnitude.min() <= 1e-3 * magnitude.max()  # the 60 dB premise
    bound = np.finfo(np.float32).eps * np.sqrt(length) * np.abs(samples.astype(np.float64)).sum() * dt
    error = np.abs(result.astype(np.complex128) - reference)
    assert np.all(error <= bound), (error/bound).max()

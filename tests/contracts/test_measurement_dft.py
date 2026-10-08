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
@pytest.mark.parametrize('turns', [.7, 1.3, 2.3, 7.9])
def test_stamp_offset_multiplies_the_whole_product(kind, offset, turns):
    """f dt above one: the H half step carries half of floor(f dt) turns too."""
    from fractions import Fraction
    with jax.enable_x64(False):
        dt = 1e-12
        f = turns / dt
        for n in (0, 7, 100000):
            actual = complex(np.asarray(phase(n, [f], dt, kind))[0]) / np.float32(dt)
            angle = float(Fraction(f * dt) * (Fraction(n) + Fraction(offset)) % 1)
            assert abs(actual - np.exp(-2j * np.pi * angle)) <= 1e-6


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
    # Streaming adds one product per step; replay adds blocks of 256 steps,
    # one matrix product each. Same weights, another order of additions: the
    # bar is J7-1's, 1e-4 of the peak (measured 2e-7 .. 4e-7 here).
    assert np.max(np.abs(np.asarray(live) - np.asarray(replay))) <= 1e-4 * np.max(np.abs(live))
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
    ('uniform_h_plane_as_e', ('_dft_planes', 'dft_plane'), 'dft_planes.hy_x_1.accumulator'),
])
def test_s0_clock_mutations_turn_the_cell_red(monkeypatch, mutation, row, record):
    """Use the real S0 runner and judge, mutating only one path's phase
    argument: the graded path's, or (third case) the uniform path's H stamp."""
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

    mutated_lane = 'run_uniform' if mutation.startswith('uniform') else 'run_nonuniform'

    def changed_solve(row, lane, *args):
        if lane == mutated_lane:
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


def test_the_plan_lists_the_sums_that_are_not_on_the_kernel():
    from tests.contracts.path_equivalence.builders import build
    from rfx.measurement.plan import measurement_plan
    plan = measurement_plan(build(('_dft_planes', 'dft_plane'), 'run_uniform'), n_steps=12,
                            path='run_uniform')
    listed = ' | '.join(plan.known_differences)
    for name in ('rfx/floquet.py', 'subgridded runner', 'rfx/adjoint.py', 'traced dt or frequencies'):
        assert name in listed, name


@pytest.mark.parametrize('kind,offset', [('E', 1.), ('H', .5)])
@pytest.mark.parametrize('length', [300, 513])
def test_n_valid_masks_tail_across_replay_blocks(kind, offset, length):
    """Physical stamps: E=(n+1)dt, H=(n+1/2)dt; integral uses exp(-jwt)."""
    with jax.enable_x64(False):
        records = np.random.default_rng(1527).normal(size=(length, 2)).astype(np.float32)
        freqs, dt = np.array([2.3, 17.1, 63.7]), .002
        traces = []
        from rfx.measurement import dft as kernel
        transform(records, freqs, dt, kind, n_valid=1)      # compile once for this shape
        compiled = kernel._replay._cache_size()

        @jax.jit
        def replay(n_valid):
            # Python executes only when tracing, never for a cached executable.
            traces.append(None)
            return transform(records, freqs, dt, kind, n_valid=n_valid)

        for k in (1, 255, 256, 257, length - 1, length):
            stamps = (np.arange(k, dtype=np.float64) + offset) * dt
            reference = (np.exp(-2j * np.pi * freqs[:, None] * stamps)
                         @ records[:k].astype(np.float64)) * dt
            eager = np.asarray(transform(records, freqs, dt, kind, n_valid=k))
            traced = np.asarray(replay(jnp.asarray(k, dtype=jnp.int32)))
            for actual in (eager, traced):
                assert np.max(np.abs(actual - reference)) <= 1e-5 * np.max(np.abs(reference))
            np.testing.assert_allclose(traced, eager, rtol=1e-6, atol=1e-9)
            assert len(traces) == 1, 'n_valid values must reuse one trace'
            assert replay._cache_size() == 1
        # The kernel's own replay compiled once for this record shape, not per n_valid
        # (the outer jit above would not retrace for an int32 scalar either way).
        assert kernel._replay._cache_size() <= compiled + 1


def test_high_product_extreme_words_and_random():
    from rfx.measurement.dft import _high_product

    rng = np.random.default_rng(1527)
    words = np.concatenate([np.array([0, 1, 0xFFFFFFFF, 0x80000000,
                                     0xFFFF0000, 0x0000FFFF], dtype=np.uint32),
                            rng.integers(0, 2**32, size=64, dtype=np.uint32)])
    # All pairs include carries from both cross products, not just squares.
    a, b = np.meshgrid(words, words, indexing='ij')
    expected = np.array([(int(x) * int(y)) >> 32 for x, y in zip(a.flat, b.flat)],
                        dtype=np.uint32).reshape(a.shape)
    with jax.enable_x64(False):
        actual = jax.jit(_high_product)(jnp.asarray(a), jnp.asarray(b))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('kind,offset', [('E', 1), ('H', .5)])
@pytest.mark.parametrize('x64', [False, True])
def test_negative_phase_zero_low_word(kind, offset, x64):
    from fractions import Fraction

    # Binary-exact f*dt=k/2**32 gives a zero low word in the 64-bit turns.
    numerators = [1, 0x0000FFFF, 0x80000000, 0xFFFF0000, 0xFFFFFFFF]
    dt = .5
    freqs = np.array(numerators, dtype=np.float64) / 2**31
    steps = np.array([-1, -2, -255, -65536, -100000, -2**31], dtype=np.int32)
    expected = np.array([
        [np.exp(-2j * np.pi * float((Fraction(k, 2**32)
                                    * (int(n) + Fraction(offset))) % 1))
         for k in numerators] for n in steps])
    with jax.enable_x64(x64):
        actual = np.asarray(jax.jit(lambda n: phase(n, freqs, dt, kind))(steps)) / dt
    error = np.abs(np.angle(actual.astype(np.complex128) / expected))
    assert np.max(error) <= 1e-6
    if x64:
        # A missing negation carry is only 2*pi/2**32 (~1.46e-9) rad:
        # the float32 bar alone cannot distinguish it.
        assert np.max(error) <= 1e-12


@pytest.mark.parametrize('kind,offset', [('E', 1), ('H', .5)])
def test_negative_phase_low_word_carry(kind, offset):
    """f dt = K / 2**64 with a low word that carries when one turn step is added."""
    from fractions import Fraction

    numerators = [2**33 - 1, (0x1234 << 32) + 0xFFFFFFFF, (0x80000 << 32) + 0xFFFF0001]
    dt = .5
    freqs = np.array([float(Fraction(k, 2**63)) for k in numerators])    # exact: f*dt = K/2**64
    assert [Fraction(f) * Fraction(dt) for f in freqs] == [Fraction(k, 2**64) for k in numerators]
    steps = np.array([-2, -3, -100000], dtype=np.int32)
    expected = np.array([
        [np.exp(-2j * np.pi * float((Fraction(k, 2**64) * (int(n) + Fraction(offset))) % 1))
         for k in numerators] for n in steps])
    with jax.enable_x64(True):
        actual = np.asarray(jax.jit(lambda n: phase(n, freqs, dt, kind))(steps)) / dt
    error = np.abs(np.angle(actual.astype(np.complex128) / expected))
    # Without the carry from the low word the phase is 2*pi/2**32 (1.46e-9 rad) off.
    assert np.max(error) <= 1e-12, error

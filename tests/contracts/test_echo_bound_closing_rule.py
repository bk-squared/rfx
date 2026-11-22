"""The shorted TEM line pins the unseen-echo identification limit (#2967)."""

from functools import lru_cache
import sys
import warnings

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.measurement.record_end import echo_delay_bound, echo_delay_bound_on_grid, with_echo_bound
from rfx.ringdown import RingdownSpec


RULE = "the record ends less than 2.05 longest round trips"
C0 = 299792458.0


def shorted_line():
    """A 1 mm grid resolves a port at 10 mm and the short at 320 mm."""
    sim = Simulation(freq_max=14e9, domain=(.340, .003, .002), dx=1e-3,
                     boundary=dict(x="cpml", y="pmc", z="pec"), cpml_layers=8)
    sim.add(Box((.320, 0, 0), (.340, .003, .002)), material="pec")
    sim.add_port(position=(.010, .001, 0), component="ez", impedance=50.,
                 extent=.002, waveform=GaussianPulse(f0=8e9, bandwidth=1.0))
    return sim


@lru_cache(maxsize=None)
def run_record(n):
    """Retain each measured main record for its forward completion's bins."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return shorted_line().run(n_steps=n, skip_preflight=True)


def test_bound_and_round_trip_are_read_from_realized_cells(capsys):
    """The domain bound is 2.2684 ns and covers the 2.0681 ns physical return."""
    sim = shorted_line()
    grid = sim._build_realized_grid()
    bound = echo_delay_bound(grid, sim=sim)
    lengths = []
    for axis in "xyz":
        cells = np.asarray(grid.cells(axis), dtype=np.float64)
        lo, hi = getattr(grid, f"pad_{axis}_lo"), getattr(grid, f"pad_{axis}_hi")
        lengths.append(np.sum(cells[lo:len(cells) - hi - 1]))
    assert bound['D'] == pytest.approx(np.linalg.norm(lengths), rel=1e-12)
    assert bound['n_max'] == 1. and bound['g'] == 1.
    assert bound['tau_max'] == pytest.approx(2.2684e-9, rel=1e-4)
    port_index = grid.index_of('x', .010)
    short_index = grid.index_of('x', .320)
    distance = grid.node_of('x', short_index) - grid.node_of('x', port_index)
    physical_return = 2 * distance / C0
    assert physical_return == pytest.approx(2.0681e-9, rel=1e-4)
    assert bound['tau_max'] >= physical_return
    with capsys.disabled():
        print(f"echo bound: {bound}; 2L/c={physical_return * 1e9:.9g} ns", flush=True, file=sys.stderr)


def test_realized_bound_reuses_declared_index_and_group_factor():
    """New host widths replace only the extent and delay of a plan's bound."""
    sim = shorted_line()
    grid = sim._build_realized_grid()
    realized = echo_delay_bound(grid, sim=sim)
    plan_bound = dict(tau_max=2 * .020 * 2. / (C0 * .5), D=.020, n_max=2., g=.5)
    original = dict(plan_bound)
    rebound = echo_delay_bound_on_grid(grid, plan_bound)
    assert rebound['D'] == realized['D']
    assert rebound['n_max'] == 2. and rebound['g'] == .5
    assert rebound['tau_max'] == 4 * realized['tau_max']
    assert plan_bound == original


def test_realized_bound_preserves_a_plan_refusal():
    """A declaration refusal survives even when the realized grid is unavailable."""
    refused = dict(tau_max=None, D=None, n_max=None, g=None,
                   reason='cannot judge: a waveguide read bin is at or below cutoff')
    assert echo_delay_bound_on_grid(None, refused) is refused


def test_echo_bound_refusal_reason_is_independent_of_trace_argument_names():
    """Two traces of one model keep material and width refusals identical."""
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001,
                     boundary='pec', dx_profile=np.full(12, .001))
    grid = sim._build_realized_grid()
    bounds = []

    def model(scale):
        live = grid._replace(dx_arr=jnp.asarray(grid.dx_arr) * scale, dx_arr_f64=None)
        material = SimpleNamespace(eps_r=scale, mu_r=1.)
        bounds.append((echo_delay_bound(grid, materials=material),
                       echo_delay_bound(live, sim=sim)))
        return scale

    jax.jit(model)(jnp.float32(1.))
    jax.jit(lambda value: model(value))(jnp.float32(1.))
    assert len(bounds) == 2
    for original, fresh in zip(*bounds):
        assert original['reason'] == fresh['reason']
        assert original['reason'] == (
            "cannot judge: echo delay bound unavailable: traced operands; pass echo_bound=")


def test_traced_step_refuses_only_until_a_realized_host_step_is_available():
    """The closing call supplies its host step without changing the stored plan."""
    import copy
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    from rfx.ringdown import RingdownForward
    from rfx.ringdown._echo import close_completion
    from rfx.ringdown._forward_context import report_context

    sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001, boundary='pec')
    grid = sim._build_realized_grid()
    realized_bound = echo_delay_bound(grid, sim=sim)

    def complete(step):
        traced_grid = copy.copy(grid)
        traced_grid.dt = step
        plan = RingdownForward(sim, RingdownSpec(), lane='uniform', n_steps=600,
                               grid=traced_grid, bins=[10e9])
        assert plan.echo_dt is None and not plan.echo_bound.get('reason')
        output = jnp.ones(1)
        refused, _, failure = close_completion(plan, output)
        assert failure.message == 'cannot judge: echo delay bound solver step is traced'
        plan.echo_bound = dict(plan.echo_bound, D=1., tau_max=2. / C0)
        result = SimpleNamespace(grid=grid, dt=grid.dt, wire_port_sparams=())
        report_context(plan, result, [10e9])
        assert plan.echo_dt is None and plan.echo_bound['D'] == 1.
        judged, _, failure = close_completion(plan, output, grid=grid, dt=float(grid.dt))
        assert failure is None
        assert echo_delay_bound_on_grid(grid, plan.echo_bound) == realized_bound
        plan.n_steps, plan.n_start = 2, 1
        short, _, failure = close_completion(plan, output, grid=grid, dt=float(grid.dt))
        assert RULE in failure.message
        return refused, judged, short

    refused, judged, short = jax.jit(complete)(jnp.float32(grid.dt))
    assert np.isnan(refused).all() and np.isnan(short).all()
    np.testing.assert_array_equal(judged, np.ones(1))


def test_traced_profile_closes_at_the_smallest_accepted_record():
    """Declared box arithmetic pins both sides of the realized-width boundary."""
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    from rfx.ringdown import RingdownForward
    from rfx.ringdown._echo import forward_closing_factor
    from rfx.ringdown._forward_context import report_context

    box = (.012, .011, .005)
    sim = Simulation(freq_max=20e9, domain=box, dx=.001, boundary='pec',
                     dx_profile=np.full(12, .001))
    sim.add_material('fill', eps_r=4.)
    sim.add(Box((0., 0., 0.), box), material='fill')
    grid = sim._build_realized_grid()
    dt = float(grid.dt)
    required = 2.05 * 2 * np.sqrt(sum(length**2 for length in box)) * 2. / C0
    window_steps = int(np.ceil(required / dt))
    accepted = next(n for n in range(2 * window_steps - 2, 2 * window_steps + 2)
                    if (n - round(.5 * n)) * dt >= required)
    assert (accepted - 1 - round(.5 * (accepted - 1))) * dt < required

    def complete(delta):
        widths = jnp.asarray(grid.dx_arr).at[3].add(delta).at[4].add(-delta)
        live = grid._replace(dx_arr=widths, dx_arr_f64=None, dt=jnp.asarray(dt))
        outputs = []
        for n in (accepted, accepted - 1):
            plan = RingdownForward(sim, RingdownSpec(), lane='graded', n_steps=n,
                                   grid=live, bins=[10e9])
            assert plan.echo_bound['D'] is None and plan.echo_bound.get('reason')
            assert plan.echo_bound['n_max'] == 2. and plan.echo_bound['g'] == 1.
            result = SimpleNamespace(grid=live, dt=live.dt, wire_port_sparams=())
            ctx = report_context(plan, result, [10e9])
            cells = tuple(getattr(live, store) for _, store in ctx.traced_axes)
            factor = forward_closing_factor(plan, ctx, dt, cells, dtype=jnp.float32)
            outputs.append((1. + delta) * factor)
        return jnp.stack(outputs)

    values, derivatives = jax.jit(lambda d: jax.jvp(
        complete, (d,), (jnp.float32(1.),)))(jnp.float32(.000125))
    assert np.isfinite(values[0]) and float(derivatives[0]) == 1.
    assert np.isnan(values[1]) and np.isnan(derivatives[1])


def test_drude_interior_does_not_raise_the_echo_index():
    """A conductor contributes its background index, excluding its Drude pole."""
    from rfx.materials.lorentz import drude_pole

    eps_inf, mu_r = 2.25, 1.44
    sim = Simulation(freq_max=8e9, domain=(.018, .012, .012), dx=.003, boundary='pec')
    sim.add_material('conductor', eps_r=eps_inf, mu_r=mu_r,
                     lorentz_poles=[drude_pole(2 * np.pi * 3e9, 2.300874e8)])
    sim.add(Box((0., 0., 0.), (.018, .012, .012)), material='conductor')
    bound = echo_delay_bound(sim._build_realized_grid(), sim=sim)
    assert not bound.get('reason'), bound
    assert np.isfinite(bound['tau_max']) and np.isfinite(bound['n_max'])
    assert bound['n_max'] == np.sqrt(eps_inf * mu_r)


def test_undamped_lorentz_resonance_in_the_read_band_refuses_the_bound():
    """A lossless resonance has no finite index even between the read bins."""
    from rfx.materials.lorentz import lorentz_pole

    sim = Simulation(freq_max=8e9, domain=(.018, .012, .012), dx=.003, boundary='pec')
    sim.add_material('resonant', eps_r=2.,
                     lorentz_poles=[lorentz_pole(1., 2 * np.pi * 3e9, 0.)])
    sim.add(Box((0., 0., 0.), (.018, .012, .012)), material='resonant')
    bound = echo_delay_bound(sim._build_realized_grid(), sim=sim, freqs=[2e9, 4e9])
    assert bound['tau_max'] is None
    assert 'undamped Lorentz resonance lies inside the read band' in bound['reason']


@pytest.mark.parametrize('n', [4232, 3200])
def test_unseen_echo_pass_is_refused(n, capsys):
    """Both wrong-pass records become undetermined with the closing reason and NaN dB."""
    result = run_record(n)
    row = result.settling_witness
    with capsys.disabled():
        print(f"settling n={n}: {row['status']}: {row['reason']}", flush=True, file=sys.stderr)
    assert row['status'] == 'undetermined', row
    assert RULE in row['reason'] and 'cannot judge: ' in row['reason'], row
    assert np.isnan(result.settling_db) and np.isnan(row['db'])
    assert row['echo_bound']['tau_max'] == pytest.approx(2.2684e-9, rel=1e-4)
    assert row['status_per_entry'] and set(row['status_per_entry'].values()) == {'undetermined'}
    assert all(group['status'] == 'undetermined' and RULE in group['reason']
               for group in row.get('runs', [row]))


def test_record_above_bound_keeps_the_tail_judgement(capsys):
    """At 4900 steps the tail check supplies the verdict, without the closing reason."""
    row = run_record(4900).settling_witness
    with capsys.disabled():
        print(f"settling n=4900: {row['status']}: {row['reason']}", flush=True, file=sys.stderr)
    assert row['status'] in {'pass', 'fail', 'undetermined'}
    assert RULE not in row['reason']
    assert not row.get('echo_bound_refused')


@pytest.mark.parametrize('n', [4232, 4900])
def test_forward_completion_uses_the_same_window_bound(n, capsys):
    """4232 steps refuse completion with NaN spectra; 4900 steps reach its own check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = shorted_line().forward(n_steps=n, ringdown=RingdownSpec(),
                                        port_s11_freqs=run_record(n).freqs, skip_preflight=True)
    completion = result.ringdown
    report = completion.report
    reason = report.failure or ''
    with capsys.disabled():
        print(f"completion n={n}: completed={report.completed}: {reason}", flush=True, file=sys.stderr)
    if n == 4232:
        assert not report.completed and 'cannot judge: ' in reason and RULE in reason
        assert np.isnan(np.asarray(completion.s_params)).all()
        assert np.isnan(np.asarray(completion.s_params_long)).all()
    else:
        assert RULE not in reason
        assert not any(w.name == 'echo_bound' for w in report.witnesses)


def test_synthetic_record_keeps_its_verdict_without_geometry():
    """A direct synthetic-record call explicitly reports an unchecked echo bound."""
    from rfx.measurement.settling import settling_db_from_named_records
    from tests.contracts.test_settling_witness_per_bin_tail import DT, BINS, SOURCE_END, modes

    db, row = settling_db_from_named_records(
        [('modes', modes(6000))], dt=DT, freqs=BINS, freq_max=1.,
        source_end_index=SOURCE_END, return_detail=True)
    assert row['status'] == 'pass' and np.isfinite(db)
    assert row['echo_bound'] is None


def test_each_coverage_demoted_row_uses_its_own_window():
    """Closing a short covered pass preserves its reason and a longer entry's pass."""
    from rfx.measurement.coverage import AccumulatorCoverage, with_coverage

    short = dict(status='pass', db=-60., reason='', window_starts={10: (5, 2)},
                 status_per_entry={'short': 'pass'})
    long = dict(status='pass', db=-60., reason='', window_starts={100: (50, 25)},
                status_per_entry={'long': 'pass'})
    combined = dict(short, runs=[short, long], groups=[short, long])
    coverage = AccumulatorCoverage((('box', 12, 10, 'cannot judge: uncovered box'),))
    db, combined = with_coverage(-60., combined, coverage)
    db, row = with_echo_bound(db, combined, dict(tau_max=4., D=1., n_max=1., g=1.), 1.)
    assert row['status'] == 'undetermined' and np.isnan(db)
    assert row['reason'].startswith('cannot judge: uncovered box; ')
    assert RULE in row['reason']
    assert row['status_per_entry'] == {'short': 'undetermined', 'long': 'pass'}
    assert row['groups'][0]['status'] == 'undetermined'
    assert row['groups'][1]['status'] == 'pass'

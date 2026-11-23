"""Live material and mesh bounds, propagating bins, and realized stop windows."""
import sys
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.errors import RfxRefusal
from rfx.measurement.record_end import echo_delay_bound, with_echo_bound
from rfx.ringdown import RingdownSpec

C = 299792458.0
BINS = np.linspace(10e9, 16e9, 7)


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    jax.clear_caches()


def box(eps, **mesh):
    sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001,
                     boundary='pec', **mesh)
    sim.add_material('fill', eps_r=eps, sigma=.3)
    sim.add(Box((0, 0, 0), (.012, .011, .005)), material='fill')
    sim.add_port(position=(.003, .003, 0), component='ez', impedance=50.,
                 extent=.001, waveform=GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5))
    sim.add_probe(position=(.008, .007, .002), component='ez')
    return sim


def forward(eps):
    return box(eps).forward(n_steps=1600, ringdown=RingdownSpec(),
                            port_s11_freqs=BINS, skip_preflight=True)


def index(result):
    return result.settling_witness['echo_bound']['n_max']


def test_live_declared_material_completes_and_transports_exact_index(capsys):
    eps = jnp.float32(2.2)
    expected_index = np.nextafter(np.sqrt(np.float32(2.2)), np.float32(np.inf))

    def loss(value):
        result = forward(value)
        return jnp.sum(jnp.abs(result.ringdown.s_params)**2), result

    eager = forward(eps)
    compiled = jax.jit(forward)(eps)
    (value, auxiliary), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(eps)
    eager_gradient, eager_auxiliary = jax.grad(loss, has_aux=True)(eps)
    batch = jax.vmap(forward)(jnp.asarray([2.2], dtype=jnp.float32))
    member = jax.tree_util.tree_map(lambda array: array[0], batch)
    expected = np.asarray(eager.ringdown.s_params)
    assert eager.ringdown.report.completed
    assert np.isfinite(expected).all() and np.isfinite(value)
    for result in (compiled, auxiliary, eager_auxiliary, member):
        assert result.ringdown.report.completed, result.ringdown.report.failure
        actual = np.asarray(result.ringdown.s_params)
        assert np.isfinite(actual).all()
        assert np.max(np.abs(actual - expected)) <= 1e-4 * np.max(np.abs(expected))
        assert result.settling_witness['status'] == eager.settling_witness['status']
    assert np.isfinite(gradient) and np.isfinite(eager_gradient)
    assert abs(float(gradient - eager_gradient)) <= 1e-4 * abs(float(eager_gradient))
    assert abs(float(eager_gradient) - (-0.5165157)) <= 1e-4 * abs(-0.5165157)
    assert abs(float(gradient) - (-0.5165154)) <= 1e-4 * abs(-0.5165154)
    for result in (eager, compiled, auxiliary, eager_auxiliary, member):
        assert np.float64(index(result)).tobytes() == np.float64(expected_index).tobytes()
    with capsys.disabled():
        print(f'12q.1: n_max={expected_index:.17g}; status={eager.settling_witness["status"]}; '
              f'db={eager.settling_db}; gradient={float(gradient):.12g}', file=sys.stderr)


def test_live_width_settling_reads_the_returned_grid(capsys):
    from rfx.model import NominalProfile
    widths = np.full(12, .001)

    def run(scale):
        live = jnp.asarray(widths, dtype=jnp.float32).at[6].multiply(scale)
        nominal = ({'x': NominalProfile(widths, traced_cells=(6, 7))}
                   if isinstance(live, jax.core.Tracer) else None)
        return box(2.2, dx_profile=live, nominal_profiles=nominal,
                   dt=1.5e-12, dt_min_cell=.0009).forward(
                       n_steps=1600, port_s11_freqs=BINS, skip_preflight=True)

    eager, compiled = run(jnp.float32(1.05)), jax.jit(run)(jnp.float32(1.05))
    expected_d = np.linalg.norm([np.sum(np.asarray(
        jnp.asarray(widths, dtype=jnp.float32).at[6].multiply(jnp.float32(1.05))),
                                dtype=np.float64), .011, .005])
    bound = compiled.settling_witness['echo_bound']
    unit = np.finfo(np.float32).eps / 2
    gamma = 5 * unit + (12 + 11 + 5 + 2) * unit**2
    upper = np.nextafter(np.float32(expected_d * (1 + gamma) / (1 - gamma)), np.float32(np.inf))
    assert expected_d <= float(bound['D']) <= upper
    assert 800 * float(compiled.dt) >= 2.05 * 2 * expected_d * np.sqrt(2.2) / C
    assert compiled.settling_witness['status'] == eager.settling_witness['status']
    assert not bound.get('reason')
    with capsys.disabled():
        print(f'12q.2: status={compiled.settling_witness["status"]}; '
              f'D={bound["D"]:.17g}', file=sys.stderr)


def test_default_guide_band_stops_after_echo_ready(capsys):
    sim = Simulation(freq_max=20e9, domain=(.024, .020, .016), dx=.001,
                     boundary=dict(x='cpml', y='pec', z='pec'), cpml_layers=2)
    sim.add_waveguide_port(.006, f0=10e9, probe_offset=3, ref_offset=2)
    result = sim.run(until_decay=1e-3)
    diagnostic, = (row for row in result.diagnostics if row.code == 'until_decay.stop')
    read = np.asarray(jnp.linspace(2e9, 20e9, 50), dtype=np.float64)
    cutoff = C * np.sin(np.pi / (2 * 20)) / (np.pi * .001)
    lowest = min(read[read > cutoff])
    group = np.sqrt(1 - (cutoff / lowest)**2)
    expected_tau = 2 * np.linalg.norm([.024, .020, .016]) / (C * group)
    assert expected_tau <= diagnostic.values['echo_tau_max_s'] <= expected_tau * (1 + 10 * np.finfo(np.float32).eps)
    assert diagnostic.values['stop_step'] >= diagnostic.values['echo_ready_step']
    assert diagnostic.values['stop_step'] < 50000
    assert diagnostic.values['criterion'] == 'threshold'
    with capsys.disabled():
        print(f'12q.3: g={group:.17g}; lowest={lowest:.17g}; tau={expected_tau:.17g}; '
              f'window_steps={2.05*expected_tau/float(result.dt):.17g}; '
              f'stop={diagnostic.values["stop_step"]}; '
              f'echo_ready={diagnostic.values["echo_ready_step"]}', file=sys.stderr)


def test_coax_two_port_bound_reads_the_stamped_fill(monkeypatch, capsys):
    from tests.unit.sparams.test_coax_two_port_smatrix import _sim, BAND
    import rfx.measurement.record_end as record_end
    original = record_end.echo_delay_bound
    bounds = []

    def observe(*args, **kwargs):
        bound = original(*args, **kwargs)
        bounds.append(bound)
        return bound

    monkeypatch.setattr(record_end, 'echo_delay_bound', observe)
    result = _sim().compute_coaxial_two_port(n_steps=6000, freqs=BAND)
    assert bounds and np.isfinite(result.s_params).all()
    expected = np.nextafter(np.sqrt(np.float32(2.1)), np.float32(np.inf))
    assert all(bound['n_max'] == expected for bound in bounds)
    assert 0 <= expected - np.sqrt(2.1) <= 2 * np.spacing(np.float32(np.sqrt(2.1)))
    with capsys.disabled():
        print(f'12q.4: stamped n_max={bounds[0]["n_max"]:.17g}; '
              f'sqrt(2.1)={np.sqrt(2.1):.17g}', file=sys.stderr)


def test_order_four_decay_window_counts_realized_solver_steps(capsys):
    sim = Simulation(freq_max=5e9, domain=(.030, .030, .030), dx=.003,
                     boundary='pec', stencil_order=4)
    sim.add_material('fill', eps_r=4., sigma=.3)
    sim.add(Box((0, 0, 0), (.030, .030, .030)), material='fill')
    sim.add_source(position=(.012, .012, .012), component='ez',
                   waveform=GaussianPulse(f0=3e9, bandwidth=1.))
    sim.add_probe(position=(.018, .018, .018), component='ez')
    result = sim.run(until_decay=1e-3, decay_max_steps=10000, skip_preflight=True)
    diagnostic, = (row for row in result.diagnostics if row.code == 'until_decay.stop')
    values = diagnostic.values
    n = int(values['stop_step'])
    tau = 2 * np.linalg.norm([.030, .030, .030]) * np.sqrt(4.) / C
    start = max(int(values['drive_end_step']), round(.5 * n))
    assert values['criterion'] == 'threshold'
    assert (n - start) * float(result.dt) >= 2.05 * tau
    assert float(result.dt) == pytest.approx(.857 * float(result.grid.dt), rel=1e-7)
    with capsys.disabled():
        print(f'12q.5: stop={n}; window={(n-start)*float(result.dt):.17g}; '
              f'required={2.05*tau:.17g}', file=sys.stderr)


def test_short_passing_entry_preserves_a_failing_row_and_prior_completion_failure():
    row = dict(status='fail', db=-12., reason='entry A failed',
               status_per_entry={'A': 'fail', 'B': 'pass'}, window_starts={10: (5, 2)})
    bound = dict(n_max=1., g=1., D=1., tau_max=4.)
    db, actual = with_echo_bound(-12., row, bound, 1.)
    assert db == -12. and actual['db'] == -12. and actual['status'] == 'fail'
    assert actual['status_per_entry'] == {'A': 'fail', 'B': 'undetermined'}
    assert actual['reason'].startswith('entry A failed; cannot judge:')
    from rfx.ringdown._echo import close_completion
    from rfx.ringdown import _NotCompleted
    failure = _NotCompleted('W0', 1., 0., 'original witness reason')
    report = SimpleNamespace(failure='W0: original witness reason')
    output = np.ones(1)
    got, same_report, same_failure = close_completion(
        SimpleNamespace(echo_bound=bound), output, report, failure)
    assert got is output and same_report is report and same_failure is failure


def test_until_identified_waits_for_the_echo_and_refuses_unavailable_at_start(monkeypatch, capsys):
    from tests.contracts.test_echo_bound_closing_rule import shorted_line
    result = shorted_line().run(n_steps=12000, ringdown=RingdownSpec(),
                                until_identified=True, skip_preflight=True)
    n = result.ringdown.stop.n_stop
    tau = 2 * np.linalg.norm([.340, .003, .002]) / C
    assert result.ringdown.stop.fired
    assert (n - round(.5 * n)) * float(result.dt) >= 2.05 * tau
    assert result.ringdown.report.completed, result.ringdown.report.failure
    with capsys.disabled():
        print(f'12q.7 head: stop={n}; completed={result.ringdown.report.completed}; '
              f'required_window_steps={2.05*tau/float(result.dt):.17g}', file=sys.stderr)
    sim = shorted_line()
    monkeypatch.setattr('rfx.measurement.echo_bound.runner_echo_bound',
                        lambda *args, **kwargs: dict(D=np.array(1.), g=np.array(1.),
                            n_max=jnp.float32(np.inf), tau_max=jnp.float32(np.inf)))
    with pytest.raises(RfxRefusal, match='until_identified: cannot judge:'):
        sim.run(n_steps=12000, ringdown=RingdownSpec(), until_identified=True,
                skip_preflight=True)

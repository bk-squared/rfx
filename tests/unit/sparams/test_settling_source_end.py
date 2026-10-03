"""Post-source timing and unavailable-witness edge cases (#1426)."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api._sparams import settling_verdict
from rfx.grid import C0
from rfx.probes.settling import source_end_step, simulation_source_end_step
from rfx.sources.waveguide_port import settling_db_from_named_records


def test_source_end_uses_last_lobe_own_peak_and_transit():
    first = np.array([0., 2., 0., 1., 0., 0., 0., 0., 0., 0.])
    second = np.array([0., 1e-12, 0., 0., 0., 0., 0., 0., 0., 0.])
    assert source_end_step([(first, C0 * .21), (second, 0.)], 10, .1) == 7
    assert source_end_step([(np.ones(10), 0.)], 10, .1) is None
    assert source_end_step([(np.zeros(10), 0.)], 10, .1) == 0


def test_callable_sampling_is_runner_float32_arithmetic():
    def waveform(t):
        return jnp.sin(t * 7.) * jnp.exp(-t * t)
    dt = .100000003
    samples = np.abs(np.asarray(jax.vmap(waveform)(jnp.arange(100, dtype=jnp.float32) * dt), dtype=np.float64))
    end, rows = source_end_step([(waveform, 0.)], 100, dt, return_detail=True)
    np.testing.assert_array_equal(rows[0][0], samples)
    assert end == int(np.flatnonzero(samples > 1e-6 * samples.max())[-1]) + 1


@pytest.mark.parametrize('end', [None, 90, 91, 100])
def test_missing_source_end_or_tail_is_never_a_pass(end):
    value, detail = settling_db_from_named_records(
        [('p', np.exp(-np.arange(100)/5))], source_end_index=end, return_detail=True,
        dt=1., freqs=[.1], freq_max=.1)
    assert np.isnan(value)
    assert settling_verdict(value) == 'absent'
    assert detail['reason']


def test_unidentifiable_post_source_record_is_not_a_pass():
    record = np.full(100, 1e-100)
    record[0] = 1e200
    value, detail = settling_db_from_named_records(
        [('p', record)], source_end_index=1, dt=1., freqs=[.1], freq_max=.1, return_detail=True)
    assert np.isnan(value)
    assert detail['status'] == 'undetermined'
    assert detail['reason']


def test_registered_drives_include_farthest_record_and_skip_passive_ports():
    def waveform(t):
        return jnp.where(t < 2., 1., 0.)
    sim = SimpleNamespace(_ports=[
        SimpleNamespace(impedance=0., waveform=waveform, position=(0.,0.,0.)),
        SimpleNamespace(impedance=50., excite=False, waveform=lambda t: 1., position=(0.,0.,0.))],
        _msl_ports=[], _waveguide_ports=[], _tfsf=None)
    probes = [SimpleNamespace(position=(0.,0.,0.)), SimpleNamespace(position=(2.5*C0,0.,0.))]
    assert simulation_source_end_step(sim, 20, 1., probes) == 5


def test_waveguide_uses_incident_record_and_farthest_plane():
    from rfx.sources.waveguide_port import settling_db_from_port_records

    record = np.exp(-np.arange(100)/20.)
    incident = np.zeros(100)
    incident[:5] = 1.
    cfg = SimpleNamespace(
        src_amp=1., v_inc_t=incident, dt=1., freqs=np.array([.1]), source_x_m=0.,
        reference_x_m=0., probe_x_m=2.5*C0,
        **{name: record for name in ('v_ref_t', 'i_ref_t', 'v_probe_t', 'i_probe_t')})
    actual = settling_db_from_port_records([cfg])
    expected = settling_db_from_named_records(
        [("p", record)], dt=1., freqs=[.1], freq_max=.1, source_end_index=8)
    assert actual == pytest.approx(expected)


def test_forward_result_preserves_source_end_through_jit():
    from rfx.api._spec import ForwardResult

    @jax.jit
    def recorded(x):
        return ForwardResult(time_series=x, settling_source_end_index=10, dt=1., freqs=jnp.array([.1]))

    x = jnp.exp(-jnp.arange(100)/10.)
    result = recorded(x)
    expected = settling_db_from_named_records([('p', x)], source_end_index=10,
                                            dt=1., freqs=np.array(result.freqs), freq_max=float(result.freqs[0]))
    assert result.settling_db == pytest.approx(expected)


def test_tfsf_uses_its_waveform_and_box_diagonal():
    from rfx.grid import Grid
    from rfx.probes.settling import tfsf_source_drive
    from rfx.sources.tfsf import init_tfsf

    grid = Grid(freq_max=10e9, domain=(.03, .03, .03), dx=.003, cpml_layers=4)
    cfg, _ = init_tfsf(grid.nx, grid.dx, grid.dt, cpml_layers=4,
                      ny=grid.ny, nz=grid.nz, closed_box=True)
    waveform, distance = tfsf_source_drive(cfg, grid)
    expected_distance = np.linalg.norm([
        (getattr(cfg, a+'_hi')-getattr(cfg, a+'_lo'))*grid.dx for a in 'xyz'])
    assert distance == pytest.approx(expected_distance)
    time = jnp.arange(300, dtype=jnp.float32)*grid.dt
    arg = (time-cfg.src_t0)/cfg.src_tau
    expected = np.asarray(cfg.src_amp*(-2*arg)*jnp.exp(-arg**2))
    np.testing.assert_array_equal(jax.vmap(waveform)(time), expected)


def test_traced_waveguide_metadata_reports_no_witness():
    from rfx.sources.waveguide_port import settling_db_from_port_records

    @jax.jit
    def diagnostic(x):
        cfg = SimpleNamespace(
            src_amp=x[0], v_inc_t=x, dt=1., source_x_m=0.,
            reference_x_m=0., probe_x_m=1.,
            **{name: x for name in ('v_ref_t', 'i_ref_t', 'v_probe_t', 'i_probe_t')})
        return settling_db_from_port_records([cfg])

    assert np.isnan(diagnostic(jnp.ones(100)))


def test_recorded_wire_endpoint_contributes_to_transit():
    drive = SimpleNamespace(impedance=0., waveform=lambda t: jnp.where(t < 2., 1., 0.),
                            position=(0., 0., 0.))
    sim = SimpleNamespace(_ports=[drive], _msl_ports=[], _waveguide_ports=[], _tfsf=None)
    record = SimpleNamespace(position=(0., 0., 0.), component='ez', extent=3*C0)
    assert simulation_source_end_step(sim, 20, 1., [record]) == 5

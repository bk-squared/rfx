"""Every single-device port retains samples independently of S readout."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


def model(graded=False, wire=False, boundary='pec'):
    sim = Simulation(freq_max=10e9, domain=(.006, .006, .002), dx=.001,
                     boundary=boundary, cpml_layers=2,
                     **({'dz_profile': np.array([.0008, .0012])} if graded else {}))
    sim.add_port((.002, .003, 0.), 'ez', impedance=50., extent=.002 if wire else None,
                 waveform=lambda t: jnp.cos(t * 2e10))
    sim.add_probe((.003, .002, .001), 'ez')
    return sim


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('wire', [False, True])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_every_port_records_without_s_request(graded, wire, entry):
    sim = model(graded, wire)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = getattr(sim, entry)(n_steps=16, skip_preflight=True,
                                    **({'compute_s_params': False} if entry == 'run' else {}))
    assert len(result.sparam_time_records) == 1
    record = result.sparam_time_records[0]
    assert record.id == 'port:0:mode:0'
    assert record.samples.shape[0] == 16
    assert record.kind == ('wire' if wire else 'lumped')
    expected = ('V', 'I', 'V_port') if wire else ('V', 'I')
    assert record.channels == expected + ('V_ref',)
    assert 'V_ref' in record.channels
    # Graded wire forward historically computes default-bin S without an explicit request.
    if not (graded and wire and entry == 'forward'):
        assert result.s_params is None
        assert result.settling_witness['scored_port_record_ids'] == []
        assert result.settling_witness['skipped_port_records'] == [
            {'id': 'port:0:mode:0', 'reason': 'S not read'}]
    if entry == 'run':
        # Independent completed-field oracle, including the Ampere circulation.
        grid, state = result.grid, result.state
        k = 1 if wire else 0
        width = float(np.asarray(grid.cells(2))[k])
        expected_v = -np.asarray(state.ez)[2, 3, k] * width
        expected_i = ((np.asarray(state.hy)[2, 3, k] - np.asarray(state.hy)[1, 3, k]) * .001
                      - (np.asarray(state.hx)[2, 3, k] - np.asarray(state.hx)[2, 2, k]) * .001)
        np.testing.assert_allclose(record['V'][-1], expected_v, rtol=2e-7, atol=0)
        np.testing.assert_allclose(record['I'][-1], expected_i, rtol=2e-7, atol=0)
        assert not np.isclose(expected_v, expected_i, rtol=1e-3, atol=0)


def test_graded_run_lumped_s_refuses():
    with pytest.raises(NotImplementedError, match='Single-cell lumped-port S-parameters'):
        model(True).run(n_steps=16, skip_preflight=True, compute_s_params=True,
                        s_param_freqs=np.array([2e9]))


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('wire', [False, True])
def test_until_decay_keeps_unread_records(graded, wire):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = model(graded, wire, boundary='cpml').run(
            until_decay=0., decay_min_steps=16, decay_max_steps=16,
            decay_check_interval=4, compute_s_params=False, skip_preflight=True)
    assert len(result.sparam_time_records) == 1
    record = result.sparam_time_records[0]
    expected = ('V', 'I', 'V_port') if wire else ('V', 'I')
    assert record.channels == expected + ('V_ref',)
    assert record.samples.shape == (16, len(record.channels))
    assert result.settling_witness['scored_port_record_ids'] == []
    assert result.settling_witness['skipped_port_records'] == [
        {'id': 'port:0:mode:0', 'reason': 'S not read'}]


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('wire', [False, True])
def test_until_decay_records_are_sized_by_decay_max_steps(graded, wire):
    # The decay loop runs past n_steps: a record sized by n_steps (8) would
    # end at step 8 or repeat its last row, not match the fixed 24-step run.
    def solve(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return model(graded, wire, boundary='cpml').run(
                compute_s_params=False, skip_preflight=True, **kw)
    decay = solve(n_steps=8, until_decay=0., decay_min_steps=24, decay_max_steps=24,
                  decay_check_interval=4)
    fixed = solve(n_steps=24)
    got, ref = decay.sparam_time_records[0].samples, fixed.sparam_time_records[0].samples
    assert got.shape == ref.shape == (24, len(decay.sparam_time_records[0].channels))
    peak = float(np.max(np.abs(ref)))
    assert peak > 0 and np.max(np.abs(np.asarray(got)[8:])) > 0.1 * peak
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4 * peak)


@pytest.mark.parametrize('graded', [False, True])
def test_forward_records_cross_grad_and_vmap(graded):
    sim = model(graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    def solve(scale):
        result = sim.forward(n_steps=16, skip_preflight=True,
                             eps_override=jnp.full(grid.shape, scale))
        return jnp.sum(result.time_series**2), result.sparam_time_records
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        (_, records), grad = jax.value_and_grad(solve, has_aux=True)(jnp.array(2.))
        _, batch = jax.vmap(solve)(jnp.array([2., 3.]))
    assert np.isfinite(grad)
    assert records[0].samples.shape[0] == 16
    assert batch[0].samples.shape[:2] == (2, 16)
    assert batch[0].id == 'port:0:mode:0'


def test_uniform_lumped_main_records_are_not_scored_for_driver_s():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = model().run(n_steps=16, skip_preflight=True, compute_s_params=True,
                             s_param_freqs=np.array([2e9]))
    assert len(result.sparam_time_records) == 1
    assert result.settling_witness['scored_port_record_ids'] == []
    assert result.settling_witness['skipped_port_records'] == [
        {'id': 'port:0:mode:0', 'reason': 'S not read'}]
    # The driver still scores the record of its own driven scan.
    assert result.settling_witness['runs'][0]['per_drive'][0]['scored_port_record_ids'] == ['port:0:mode:0']

"""Read-bin and source-off metadata through run, forward, and JIT."""
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._spec import ForwardResult
from rfx.probes.settling import probe_record_settling_witness
from rfx.sparams._tail_witness import result_read_bins


@pytest.mark.parametrize("missing", ["traced", "none", "empty", "unselected"])
def test_absent_probe_witness_has_canonical_per_bin_fields(missing):
    from rfx.sources.waveguide_port import settling_db_from_named_records

    freqs = np.array([1., 2.])
    _, canonical = settling_db_from_named_records((), freqs=freqs, return_detail=True)

    def check(series, selection):
        db, detail = probe_record_settling_witness(series, selection, freqs=freqs)
        assert db is None
        assert detail["status"] == "absent"
        assert canonical.keys() <= detail.keys()
        # An absent witness carries no NaN (#885): no number rests on it.
        assert detail["db"] is None
        assert detail["worst_freq_hz"] is None
        for key in ("share_per_bin", "error_per_bin"):
            np.testing.assert_array_equal(detail[key], np.zeros(2))
        return series

    if missing == "traced":
        jax.jit(lambda series: check(series, ((0, 2, 0),)))(jnp.ones((20, 1)))
    else:
        series = {"none": None, "empty": np.empty((20, 0)),
                  "unselected": np.ones((20, 1))}[missing]
        check(series, () if missing == "unselected" else None)


def _sim(*, probe=True, dft=False):
    sim = Simulation(freq_max=10e9, domain=(.004,)*3, dx=.001,
                     boundary='cpml', cpml_layers=4)
    sim.add_source((.002,)*3, 'ez', amplitude_kind='field')
    if probe:
        sim.add_probe((.002, .002, .003), 'ez')
    if dft:
        sim.add_dft_plane_probe(axis='z', coordinate=.002, component='ez', n_freqs=3)
    return sim


def synthetic(**kwargs):
    defaults = dict(time_series=jnp.exp(-jnp.arange(200)/20.)[:, None],
                    dt=1., freqs=jnp.array([.1]), settling_source_end_index=0)
    defaults.update(kwargs)
    return ForwardResult(**defaults)


def test_read_bins_union_includes_all_spectral_fields():
    result = SimpleNamespace(freqs=[1, 2], ntff_box=SimpleNamespace(freqs=[2, 3]),
                             dft_planes={'e': SimpleNamespace(freqs=[4])},
                             flux_monitors={'p': SimpleNamespace(freqs=[5])},
                             current_moment_monitor=SimpleNamespace(freqs=[6]),
                             waveguide_ports={'w': SimpleNamespace(freqs=[7])},
                             lumped_port_sparams=[(SimpleNamespace(freqs=[8]), None)])
    np.testing.assert_array_equal(result_read_bins(result), np.arange(1, 9))


def test_full_forward_result_crosses_jit_then_scores_concrete_arrays():
    result = jax.jit(lambda x: synthetic(time_series=x))(synthetic().time_series)
    value, detail = probe_record_settling_witness(
        result.time_series, dt=1., freqs=np.asarray(result.freqs),
        freq_max=float(result.freqs[0]), source_end_index=0)
    assert result.settling_db == value
    assert result.settling_witness['status'] == detail['status'] == 'pass'
    assert np.isfinite(value)


def test_repeated_lazy_reads_emit_no_warnings():
    result = synthetic()
    with warnings.catch_warnings(record=True) as caught:
        first = result.settling_db
        assert result.settling_db == first
        assert result.settling_witness['status'] == 'pass'
    assert not caught


@pytest.mark.parametrize('missing', ['source', 'bins', 'records'])
def test_missing_metadata_is_never_pass(missing):
    kwargs = {'source': dict(settling_source_end_index=None),
              'bins': dict(freqs=None), 'records': dict(time_series=None)}[missing]
    result = synthetic(**kwargs)
    assert result.settling_witness['status'] in {'absent', 'undetermined'}
    assert result.settling_witness['reason']
    assert result.settling_db is None or np.isnan(result.settling_db)


@pytest.mark.parametrize('bad', [np.nan, np.inf, 0.])
@pytest.mark.parametrize('reverse', [False, True])
def test_invalid_selected_channel_invalidates_companion(bad, reverse):
    y = np.column_stack([np.exp(-np.arange(200)/20.), np.full(200, bad)])
    result = synthetic(time_series=y[:, ::-1] if reverse else y)
    if bad == 0:
        assert result.settling_witness['status'] == 'pass'
        assert result.settling_db <= -40
        zero = 0 if reverse else 1
        assert result.settling_witness['per_record_db'][f'probe{zero}(?)'] == -np.inf
    else:
        assert result.settling_witness['status'] == 'undetermined'
        assert result.settling_witness['reason']
        assert np.isnan(result.settling_db)


def test_empty_selection_excludes_backend_fallback_record():
    result = synthetic(settling_probe_info=())
    assert result.settling_witness['status'] == 'absent'
    assert result.settling_db is None


def test_unselected_nonfinite_record_does_not_change_selected_channel():
    y = np.column_stack([np.exp(-np.arange(200)/20.), np.full(200, np.nan)])
    result = synthetic(time_series=y, settling_probe_info=((0, 2, 0),))
    assert result.settling_witness['status'] == 'pass'
    assert result.settling_witness['worst_record'] == 'probe0(ez)'


def test_actual_run_and_forward_refuse_source_active_window():
    sim = _sim(dft=True)
    for call in (sim.run, sim.forward):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = call(n_steps=60, skip_preflight=True)
        assert result.settling_witness['status'] == 'undetermined'
        assert 'source end' in result.settling_witness['reason']
        messages = [str(w.message) for w in caught if 'no ring-down settling witness' in str(w.message)]
        assert len(messages) == 1


def test_actual_public_forward_returns_numeric_carrier_through_outer_jit():
    sim = _sim(dft=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = jax.jit(lambda: sim.forward(n_steps=60, skip_preflight=True)._replace(
            grid=None, dft_planes=None, freqs=jnp.linspace(1e9, 10e9, 3)))()
    assert result.settling_witness['status'] == 'undetermined'
    assert result.settling_witness['reason']


@pytest.mark.parametrize('extent', [None, .001])
def test_recorded_s_channels_reproduce_accumulators(extent):
    from rfx.ringdown import plain_dft

    sim = Simulation(freq_max=5e9, domain=(.006,)*3, dx=.001,
                     boundary='pec', cpml_layers=0)
    sim.add_port((.003,)*3, 'ez', impedance=50., extent=extent)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = sim.forward(n_steps=120, port_s11_freqs=np.array([2e9, 3e9]), skip_preflight=True)
    assert result.sparam_time_records[0].shape == (120, 3 if extent is None else 4)
    raw = result.lumped_port_sparams if extent is None else result.wire_port_sparams
    meta, accs = raw[0]
    spectra = plain_dft(result.sparam_time_records[0], result.dt, meta.freqs)
    for col, slot in ((0, 0), (1, 1), (2, 2)) if extent is None else ((0, 0), (1, 1), (2, 3), (3, 4)):
        got = spectra[:, col]
        if col == 1:
            got = got * np.exp(1j*np.pi*np.asarray(meta.freqs)*result.dt)
        np.testing.assert_allclose(got, accs[slot], rtol=2e-5, atol=1e-20)
    assert result.settling_witness['route'] == 's_channels'


def test_reference_plane_drive_channels_and_source_selection(monkeypatch):
    from rfx.probes import settling
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    from rfx.sparams import _tail_witness
    from tests.unit.materials.test_conductor_mask_accessor import _refplane_thru, _RP_FREQS

    sim = _refplane_thru("pec")
    selected, shapes = [], []
    original_end = settling.simulation_source_end_step
    original_witness = _tail_witness.port_record_witness

    def source_end(driven, n_steps, dt, records, **kwargs):
        selected.append(([port.excite for port in driven._ports], len(records)))
        return original_end(driven, n_steps, dt, records, **kwargs)

    def witness(records, *args, **kwargs):
        shapes.append([record.shape for record in records])
        return original_witness(records, *args, **kwargs)

    monkeypatch.setattr(settling, 'simulation_source_end_step', source_end)
    monkeypatch.setattr(_tail_witness, 'port_record_witness', witness)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        matrix, bins, detail = compute_lumped_wire_s_matrix_via_scan(
            sim, _RP_FREQS, n_steps=12, return_settling=True)
    assert matrix.shape == (2, 2, len(bins))
    assert selected == [([True, False], 6), ([False, True], 6)]
    assert shapes == [[(12, 4)]*2 + [(12, 3)]*4]*2
    assert len(detail['per_drive']) == 2
    assert detail['status'] == 'undetermined'


def test_s_only_undetermined_warning_names_only_worst_bin_and_share(monkeypatch):
    from rfx.api._spec import Result

    sim = _sim(probe=False)
    detail = dict(status='undetermined', reason='identification error',
                  worst_freq_hz=2e9, share_per_bin=np.array([.02]))
    monkeypatch.setattr(sim, '_run_settling_witness', lambda result: (np.nan, detail))
    result = Result(state=None, time_series=None, s_params=np.zeros((1, 1, 1)),
                    freqs=np.array([2e9]))
    with pytest.warns(UserWarning, match=r'worst bin 2e\+09 Hz, share 0.02.*undetermined') as caught:
        attached = sim._attach_run_settling_witness(result, n_steps=20)
    assert attached.settling_witness['status'] == 'undetermined'
    assert attached.settling_witness['reason'] == 'identification error'
    assert all('identification error' not in str(row.message) for row in caught)


def test_probe_source_end_uses_probe_count_when_port_channels_are_present(monkeypatch):
    import rfx.probes.settling as settling

    sim = _sim()
    sim.add_probe((.003, .003, .003), 'ez')
    selected = []

    def source_end(simulation, n_steps, dt, records, **kwargs):
        selected.extend(records)
        return 10

    monkeypatch.setattr(settling, 'simulation_source_end_step', source_end)
    result = SimpleNamespace(time_series=np.ones((200, 2)), dt=1.,
                             sparam_time_records=(np.ones((200, 1)),))
    assert sim._settling_source_end(result) == 10
    assert selected == sim._probes

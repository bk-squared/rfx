"""Same-quantity rounding records are named and excluded before identification."""
import warnings
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.measurement.settling import settling_db_from_named_records, settling_db_from_port_records
from rfx.sparams import _tail_witness as witness
from tests._x64_compat import enable_x64


# Measured on 1cd4a04e8 with these records and no null records. The weak
# values were measured alone on main, before adding quantity-aware skipping.
MAIN_DB = {"float32": -69.48420364272383, "float64": -69.48420282431248}
MAIN_WEAK_DB = {"float32": -69.48420216761977, "float64": -69.48420282431276}
MAIN_RUN_DB = -57.23354351101009
OPTIONS = dict(dt=1., freqs=[.07], freq_max=.15, source_end_index=0, return_detail=True)


def _records(dtype):
    t = np.arange(400)
    return [(f'decay{k}', (np.exp(-rate*t)*np.cos(2*np.pi*.07*t)).astype(dtype))
            for k, rate in enumerate((.02, .03, .04))]


@pytest.mark.parametrize('dtype,amplitude', [(np.float32, 1e-2), (np.float64, 1e-6)])
def test_weak_physical_record_still_scored(dtype, amplitude):
    with enable_x64(dtype == np.float64):
        records = _records(dtype)
        weak = (records[0][1]*amplitude).astype(dtype)
        records.append(('weak', weak))
        _, detail = settling_db_from_named_records(
            records, kinds=dict.fromkeys(dict(records), 'H'), **OPTIONS)
        alone, _ = settling_db_from_named_records([('weak', weak)], **OPTIONS)
    assert detail['skipped_records'] == []
    assert detail['n_witnessed'] == 4
    assert np.isfinite(detail['per_record_db']['weak'])
    tolerance = 1e-5 if dtype == np.float32 else 1e-8
    assert alone == pytest.approx(MAIN_WEAK_DB[np.dtype(dtype).name], abs=tolerance)
    assert detail['per_record_db']['weak'] == pytest.approx(
        MAIN_WEAK_DB[np.dtype(dtype).name], abs=tolerance)


@pytest.mark.parametrize('dtype,amplitude', [(np.float32, 1e-5), (np.float64, 1e-14)])
def test_null_record_is_named_and_result_matches_main_without_it(dtype, amplitude):
    records = _records(dtype)
    noise = np.random.default_rng(1909).normal(size=400)
    noise = (noise/np.max(np.abs(noise))*amplitude).astype(dtype)
    records.insert(0, ('a_null', noise))
    with enable_x64(dtype == np.float64):
        db, detail = settling_db_from_named_records(
            records, kinds=dict.fromkeys(dict(records), 'H'), **OPTIONS)
    assert db == pytest.approx(MAIN_DB[np.dtype(dtype).name], abs=1e-9)
    row, = detail['skipped_records']
    assert row['name'] == 'a_null'
    assert row['kind'] == 'H'
    assert row['peak_ratio'] == pytest.approx(amplitude)
    assert row['threshold'] == 1e4*np.finfo(dtype).eps
    assert row['reason'] == (
        f"peak {row['peak_ratio']:.3g} of the largest H record, below "
        f"10000 x eps({np.dtype(dtype)}): rounding only")
    assert np.isnan(detail['per_record_db']['a_null'])
    assert detail['n_witnessed'] == 3
    assert detail['worst_record'] == 'decay0'
    assert 'floor_amplitude' not in detail


@pytest.mark.parametrize('ratio', [1/377, 1e-5])
def test_kinds_keep_the_only_weak_h_record(ratio):
    y = _records(np.float32)[0][1]
    records = [('electric', y), ('magnetic', y*np.float32(ratio))]
    _, detail = settling_db_from_named_records(
        records, kinds={'electric': 'E', 'magnetic': 'H'}, **OPTIONS)
    assert detail['skipped_records'] == []
    assert detail['n_witnessed'] == 2
    assert np.isfinite(detail['per_record_db']['magnetic'])


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('zero', [False, True])
def test_strict_margin_keeps_largest_and_all_zero_kind(dtype, zero):
    records = [('large', np.full((20, 2), 0 if zero else 1, dtype=dtype)),
               ('small', np.zeros(20, dtype=dtype))]
    skipped = witness._rounding_only_records(records, dict.fromkeys(dict(records), 'V'))
    assert [row['name'] for row in skipped] == ([] if zero else ['small'])


def test_skip_uses_complex_storage_peak_over_all_columns():
    peak = np.zeros((20, 2), dtype=np.complex64)
    peak[2, 1] = 2j
    null = np.full(20, 1e-5j, dtype=np.complex64)
    skipped = witness._rounding_only_records(
        [('peak', peak), ('null', null)], {'peak': 'V', 'null': 'V'})
    row, = skipped
    assert row['peak_ratio'] == pytest.approx(5e-6)
    assert row['threshold'] == 1e4*np.finfo(np.float32).eps


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_peak_exactly_at_margin_is_kept(dtype):
    records = [('large', np.ones(20, dtype=dtype)),
               ('edge', np.full(20, 1e4*np.finfo(dtype).eps, dtype=dtype))]
    assert witness._rounding_only_records(records, dict.fromkeys(dict(records), 'H')) == ()


def test_untyped_integer_empty_and_nonfinite_records_are_not_skipped():
    records = [('large', np.ones(20)), ('unknown', np.zeros(20)),
               ('integer', np.zeros(20, dtype=np.int32)), ('empty', np.array([])),
               ('nan', np.full(20, np.nan)), ('inf', np.full(20, np.inf))]
    kinds = {name: 'E' for name, _ in records if name != 'unknown'}
    assert witness._rounding_only_records(records, kinds) == ()


def test_null_skip_preserves_nonfinite_record_failure():
    records = _records(np.float32) + [('null', np.full(400, 1e-5, dtype=np.float32)),
                                     ('invalid', np.full(400, np.nan, dtype=np.float32))]
    db, detail = settling_db_from_named_records(
        records, kinds=dict.fromkeys(dict(records), 'H'), **OPTIONS)
    assert np.isnan(db)
    assert detail['status'] == 'undetermined'
    assert 'invalid: empty or non-finite record' in detail['reason']
    assert [row['name'] for row in detail['skipped_records']] == ['null']


@pytest.mark.parametrize('missing', ['bins', 'source', 'dt', 'invalid_dt', 'traced_bins'])
def test_skipped_names_survive_unavailable_returns(missing):
    records = [('large', np.ones(20, dtype=np.float32)),
               ('null', np.full(20, 1e-5, dtype=np.float32))]
    kwargs = dict(dt=1., source_end=0, freqs=[.07], freq_max=.15,
                  kinds={'large': 'H', 'null': 'H'})

    def check(freqs):
        kwargs['freqs'] = freqs
        result = witness.tail_share_witness(records, **kwargs)
        assert [row['name'] for row in result.skipped_records] == ['null']
        assert result.status in {'absent', 'undetermined'}
        return freqs

    if missing == 'traced_bins':
        jax.jit(check)(np.array([.07], dtype=np.float32))
    else:
        kwargs.update({'bins': dict(freqs=[]), 'source': dict(source_end=None),
                       'dt': dict(dt=None), 'invalid_dt': dict(dt=-1.)}[missing])
        check(kwargs['freqs'])


def test_every_record_skipped_is_undetermined(monkeypatch):
    # Unreachable with the strict rule (the largest record is never below its
    # own peak), so the skip list is forced.
    monkeypatch.setattr(witness, '_rounding_only_records', lambda records, kinds: tuple(
        dict(name=name, kind=kinds[name], peak_ratio=1., threshold=1., reason='forced')
        for name, _ in records))
    result = witness.tail_share_witness(
        [('only', np.ones(20, dtype=np.float32))], 1., 0, [.07], freq_max=.15,
        kinds={'only': 'H'})
    assert result.status == 'undetermined'
    assert np.isnan(result.db)
    assert result.reason == 'every record is rounding only: only'
    assert [row['name'] for row in result.skipped_records] == ['only']


def test_float16_records_are_scored_as_before():
    # precision='mixed' stores float16 records: 1e4 x eps(float16) is 9.8, a
    # margin above every peak, so nothing is skipped (main's behaviour).
    records = [('large', np.ones(20, dtype=np.float16)),
               ('small', np.full(20, 1e-3, dtype=np.float16))]
    assert witness._rounding_only_records(records, dict.fromkeys(dict(records), 'H')) == ()


def test_port_records_supply_separate_voltage_and_current_kinds():
    y = _records(np.float32)[0][1]
    cfg = SimpleNamespace(v_probe_t=y, v_ref_t=y*1e-5, i_probe_t=y/377,
                          i_ref_t=y/377*1e-5, dt=1., freqs=np.array([.07]))
    _, detail = settling_db_from_port_records([cfg], return_detail=True)
    assert {(row['name'], row['kind']) for row in detail['skipped_records']} == {
        ('port0/v_ref_t', 'V'), ('port0/i_ref_t', 'I')}


def _box(include_null=True):
    sim = Simulation(freq_max=10e9, domain=(.012, .014, .010), dx=.001,
                     boundary=BoundarySpec(x='cpml', y='cpml', z=Boundary(lo='pec', hi='cpml')),
                     cpml_layers=4, precision='float32')
    sim.add_source((.0063, .0067, .0032), 'ez', amplitude_kind='current')
    a, b = (.0097, .0092, .0063), (.0038, .0109, .0081)
    if include_null:
        sim.add_probe(a, 'hz')
        sim.add_probe(b, 'hz')
    sim.add_probe(a, 'hx')
    sim.add_probe(b, 'hy')
    sim.add_dft_plane_probe(axis='z', coordinate=.009, component='ez',
                            freqs=np.linspace(4e9, 8e9, 5))
    return sim


def test_run_and_forward_skip_the_two_hz_rounding_records():
    results = []
    for include_null, entry in ((False, 'run'), (True, 'run'), (True, 'forward')):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = getattr(_box(include_null), entry)(n_steps=600, skip_preflight=True)
        detail = result.settling_witness
        assert np.asarray(result.time_series).dtype == np.float32
        assert detail['status'] == 'pass', detail['reason']
        assert result.settling_db == pytest.approx(MAIN_RUN_DB, abs=1e-9)
        assert not any('cut transient' in str(row.message) for row in caught)
        expected = {('probe0(hz)', 'H'), ('probe1(hz)', 'H')} if include_null else set()
        assert {(row['name'], row['kind']) for row in detail['skipped_records']} == expected
        for row in detail['skipped_records']:
            assert 0 < row['peak_ratio'] < row['threshold']
            assert row['threshold'] == 1e4*np.finfo(np.float32).eps
        assert detail['n_witnessed'] == 2
        results.append(result.settling_db)
    assert max(results) - min(results) <= 1e-9

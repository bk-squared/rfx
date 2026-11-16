"""A voltage null is below the same-quantity -40 dB judging scale."""

import numpy as np
from dataclasses import replace

from rfx.measurement.settling import settling_db_from_named_records
from rfx.ringdown import judge_growing_poles
from tests.unit.sparams.test_ringdown_oracles import _model


def test_sixty_db_weaker_channel_is_reported_and_not_judged():
    # Proportional records have an exact 60 dB separation at every read bin,
    # even though the judge normalizes both time-domain peaks to one.
    t = np.arange(200)
    strong = np.exp(-t / 20.) * np.cos(2 * np.pi * .07 * t)
    records = [('plane0/V', strong), ('plane1/V', 1e-3 * strong)]
    options = dict(dt=1., freqs=[.07], freq_max=.15,
                   source_end_index=0, return_detail=True)
    _, own = settling_db_from_named_records(records, **options)
    db, detail = settling_db_from_named_records(
        records, scale_groups={name: 'port0/V' for name, _ in records}, **options)
    assert detail['dropped_per_bin'] == [
        dict(freq_hz=.07, count=1, channels=['plane1/V'])]
    assert np.isnan(detail['per_record_db']['plane1/V'])
    assert detail['per_record_db']['plane0/V'] == own['per_record_db']['plane0/V']
    assert db == detail['per_record_db']['plane0/V']
    assert detail['status'] == 'pass'


def test_column_keys_and_labels_survive_the_variation_filter():
    t = np.arange(200)
    signal = np.exp(-t / 20.) * np.cos(2 * np.pi * .07 * t)
    # Column 0 is removed by the existing variation filter. The current is
    # much weaker than either voltage, but is a different quantity.
    samples = np.column_stack([np.zeros(200), signal, 1e-3 * signal, 1e-6 * signal])
    _, detail = settling_db_from_named_records(
        [('port/V_I', samples)], dt=1., freqs=[.07], freq_max=.15,
        source_end_index=0, return_detail=True,
        scale_groups={'port/V_I': [None, 'V', 'V', 'I']},
        _channel_labels={'port/V_I': ['static', 'V0', 'V1', 'I']})
    assert detail['dropped_per_bin'] == [
        dict(freq_hz=.07, count=1, channels=['port/V_I[V1]'])]
    assert set(detail['per_record_db']) == {'port/V_I'}
    assert detail['n_witnessed'] == 1
    assert detail['worst_record'] == 'port/V_I'


def test_same_key_does_not_compare_different_record_lengths():
    records = [('long', np.exp(-np.arange(220) / 20.)),
               ('short', 1e-3 * np.exp(-np.arange(200) / 20.))]
    options = dict(dt=1., freqs=[.07], freq_max=.15,
                   source_end_index=0, return_detail=True)
    db, own = settling_db_from_named_records(records, **options)
    grouped_db, grouped = settling_db_from_named_records(
        records, scale_groups={'long': 'V', 'short': 'V'}, **options)
    assert grouped['dropped_per_bin'] == []
    assert grouped_db == db
    assert grouped['per_record_db'] == own['per_record_db']


def test_dropped_entry_does_not_change_horizon_or_kept_bin():
    # The weak channel's resolved decay at .7 sets H for the identified
    # model, including the other channel's growing pole at the kept .9 bin.
    s = np.array([-.001 + 2j*np.pi*.7, 1e-8 + 2j*np.pi*.9])
    c = np.array([[0., 1e-6], [1e-8, 0.]], complex)
    model = replace(_model(s, c, 100, .1), n_window=100)
    bins = np.array([.7, .9])
    denominator = np.array([[1., 1e-3], [1., 1.]])
    masked = denominator.copy()
    masked[0, 1] = np.inf
    before = judge_growing_poles(model, 200, bins, denominator)
    after = judge_growing_poles(model, 200, bins, masked,
                               horizon_denominator=denominator)
    assert after.share_per_bin[1] == before.share_per_bin[1]
    assert after.growing_share_per_bin[1] == before.growing_share_per_bin[1]
    assert after.horizon_steps == before.horizon_steps

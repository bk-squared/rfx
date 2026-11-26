"""Finite-window pole contributions at the record's read bins."""
from dataclasses import replace

import numpy as np

from rfx.measurement.ringdown_tail import plain_dft
from rfx.sparams._tail_witness import _pole_contributions
from tests.unit.sparams.test_ringdown_oracles import _model


def test_kept_growing_uses_tail_bound_and_other_rules_use_window_dft():
    dt, start, count = .1, 100, 80
    bins = np.array([.7, .9, 1.3])
    poles = np.array([.03+2j*np.pi*.7, 2j*np.pi*1.3, .5+2j*np.pi*.9])
    coefficients = np.array([.2+.1j, .4-.1j, .01+.02j])
    t = np.arange(count)*dt
    terms = np.exp(np.outer(t, poles))*coefficients
    record = np.ones((200, 1), dtype=complex)
    record[start:start+count, 0] = terms.sum(axis=1)
    model = replace(_model(poles[:2], coefficients[:2, None], start, dt),
                    n_window=count, s_growing=poles[2:])
    denominator = np.abs(plain_dft(record, dt, bins))
    actual = _pole_contributions(model, record, bins, denominator, len(record)*dt)
    kernel = np.exp(-2j*np.pi*bins[:, None]*(start*dt+t)[None, :])
    H = len(record)
    lam = np.exp(poles[0]*dt)
    expected = ((1+abs(lam)**(2*H))*dt*abs(coefficients[0])
                * abs(lam)**(len(record)-start)
                / abs(1-lam*np.exp(-2j*np.pi*bins*dt)) / denominator[:, 0])
    np.testing.assert_allclose(actual['kept non-static growing'], expected, rtol=1e-12)
    for rule, k in [('non-decaying sinusoidal', 1), ('discarded growing', 2)]:
        expected = np.abs(kernel @ terms[:, k])*dt/denominator[:, 0]
        np.testing.assert_allclose(actual[rule], expected, rtol=1e-12, atol=1e-13)


def test_static_pole_is_exempt_from_all_three_checks():
    record = np.ones((200, 1))
    model = replace(_model(np.array([0j]), np.array([[1.]]), 100, .1),
                    n_window=100, s_growing=np.array([], dtype=complex))
    values = _pole_contributions(model, record, np.array([.123]), np.ones((1, 1)), 20.)
    assert all(np.array_equal(value, [0.]) for value in values.values())


def test_synthetic_growing_and_decaying_records_expose_poles():
    from rfx.sparams._tail_witness import tail_share_witness

    dt, frequency, growth = .765e-12, 4.76e9, 3.28e-4
    steps = np.arange(1600)
    decay = np.exp((-.01 + 2j*np.pi*frequency*dt) * steps)
    growing = .03 * np.exp((growth + 2j*np.pi*frequency*dt) * steps)
    witness = tail_share_witness([('record', decay + growing)], dt, 100,
                                  [frequency], freq_max=10e9)
    assert witness.status == 'undetermined', witness.reason
    assert len(witness.growing_poles) == 1
    pole = witness.growing_poles[0]
    assert abs(pole['frequency_hz'] / frequency - 1) < .01
    assert abs(pole['growth_per_step'] / growth - 1) < .05
    assert abs(pole['e_folding_steps'] / 3048.780487804878 - 1) < .05
    assert pole['rule'] == 'discarded growing'
    kernel = np.exp(-2j*np.pi*frequency*dt*steps)
    expected_share = max(abs(np.sum(growing[start:] * kernel[start:])
                             / np.sum((decay + growing) * kernel))
                         for start in (400, 800))
    np.testing.assert_allclose(pole['max_share'], expected_share, rtol=.01)
    decaying = decay + .03 * np.exp((-.002 + 2j*np.pi*frequency*dt) * steps)
    control = tail_share_witness([('record', decaying)], dt, 100,
                                  [frequency], freq_max=10e9)
    assert control.growing_poles == ()


def test_kept_poles_inside_the_resolution_edge_are_not_named_as_growing():
    from rfx.sparams._tail_witness import tail_share_witness

    dt, n = 1e-12, 400
    poles = np.array([-.02/dt + 2j*np.pi*8e9,
                      1e-8/dt + 2j*np.pi*10e9,
                      2e-8/dt + 2j*np.pi*12e9])
    coefficients = np.array([[1.], [.1], [1e-8]])
    model = replace(_model(poles, coefficients, 200, dt), n_window=200,
                    s_growing=np.array([], dtype=complex))
    bins = np.array([10e9, 11e9])
    denominator = np.ones((2, 1)) * dt
    records = []
    _pole_contributions(model, np.zeros((n, 1)), bins, denominator, n*dt,
                        growing_poles=records)
    assert records == []
    empty = tail_share_witness([], dt, 0, bins, freq_max=12e9)
    assert empty.growing_poles == ()


def test_combined_witness_keeps_growth_from_every_run():
    from rfx.sparams._tail_witness import combine_witness_details

    pole = dict(frequency_hz=4.76e9, growth_per_step=3.28e-4,
                e_folding_steps=3048.780487804878, max_share=1.,
                rule='discarded growing')
    missing = dict(status='undetermined', reason='source end unavailable', db=np.nan)
    growing = dict(status='undetermined', reason='growing', db=np.nan,
                   growing_poles=(pole,))
    larger = {**growing, 'growing_poles': ({**pole, 'max_share': 2.},)}
    combined = combine_witness_details(missing, growing, larger)
    assert combined['growing_poles'] == ({**pole, 'max_share': 2.},)
    other = {**growing, 'growing_poles': ({**pole, 'frequency_hz': 4.7601e9,
                                            'growth_per_step': 3.29e-4, 'max_share': .5},)}
    assert combine_witness_details(growing, other)['growing_poles'] == (
        {**pole, 'growth_per_step': 3.29e-4, 'e_folding_steps': 1 / 3.29e-4},)


def test_real_growing_term_in_two_channels_has_one_positive_frequency_pole():
    from rfx.sparams._tail_witness import tail_share_witness

    dt, frequency, growth = .765e-12, 4.76e9, 3.28e-4
    steps = np.arange(1600)
    signal = np.exp((growth + 2j*np.pi*frequency*dt) * steps).real
    channels = np.column_stack((signal, .5 * signal))
    witness = tail_share_witness([('record', channels)], dt, 100,
                                  [frequency, -frequency], freq_max=10e9)
    assert witness.status == 'undetermined', witness.reason
    assert len(witness.growing_poles) == 1, witness.growing_poles
    pole = witness.growing_poles[0]
    assert pole['frequency_hz'] > 0
    assert abs(pole['frequency_hz'] / 4.76e9 - 1) < .01
    assert abs(pole['growth_per_step'] / 3.28e-4 - 1) < .05
    assert abs(pole['e_folding_steps'] / 3048.780487804878 - 1) < .05
    assert pole['rule'] == 'discarded growing'


def test_physical_pole_group_uses_largest_share_frequency_and_largest_growth():
    from rfx.measurement.record_end import _group_growing_poles

    entries = (
        dict(frequency_hz=4.76e9, growth_per_step=4e-4,
             e_folding_steps=2500., max_share=1., rule='kept non-static growing'),
        dict(frequency_hz=-4.7603e9, growth_per_step=4.01e-4,
             e_folding_steps=1 / 4.01e-4, max_share=3., rule='kept non-static growing'),
        dict(frequency_hz=4.7601e9, growth_per_step=4.02e-4,
             e_folding_steps=1 / 4.02e-4, max_share=2., rule='discarded growing'),
        dict(frequency_hz=-4.762e9, growth_per_step=4e-4,
             e_folding_steps=2500., max_share=4., rule='kept non-static growing'),
        dict(frequency_hz=4.76e9, growth_per_step=4.2e-4,
             e_folding_steps=1 / 4.2e-4, max_share=5., rule='discarded growing'),
    )
    expected = (
        dict(frequency_hz=4.76e9, growth_per_step=4.2e-4,
             e_folding_steps=1 / 4.2e-4, max_share=5., rule='discarded growing'),
        dict(frequency_hz=4.7603e9, growth_per_step=4.02e-4,
             e_folding_steps=1 / 4.02e-4, max_share=3., rule='discarded growing'),
        dict(frequency_hz=4.762e9, growth_per_step=4e-4,
             e_folding_steps=2500., max_share=4., rule='kept non-static growing'),
    )
    assert _group_growing_poles(entries) == expected
    assert _group_growing_poles(()) == ()

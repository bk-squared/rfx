"""Finite-window pole contributions at the record's read bins."""
from dataclasses import replace

import numpy as np

from rfx.ringdown import plain_dft
from rfx.sparams._tail_witness import _pole_contributions
from tests.unit.sparams.test_ringdown_oracles import _model


def test_each_pole_rule_uses_the_fitted_window_dft():
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
    for rule, k in [('kept non-static growing', 0),
                    ('non-decaying sinusoidal', 1), ('discarded growing', 2)]:
        expected = np.abs(kernel @ terms[:, k])*dt/denominator[:, 0]
        np.testing.assert_allclose(actual[rule], expected, rtol=1e-12, atol=1e-13)


def test_static_pole_is_exempt_from_all_three_checks():
    record = np.ones((200, 1))
    model = replace(_model(np.array([0j]), np.array([[1.]]), 100, .1),
                    n_window=100, s_growing=np.array([], dtype=complex))
    values = _pole_contributions(model, record, np.array([.123]), np.ones((1, 1)), 20.)
    assert all(np.array_equal(value, [0.]) for value in values.values())

"""Unresolved pole rates cannot set a horizon or justify extrapolated tails."""

from dataclasses import replace

import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.sparams._tail_witness import _pole_contributions, tail_share_witness

DT, N, D = 1.2e-12, 15252, 15
BINS = np.linspace(8e9, 12e9, 41)
REF = N - 900


def window_sum(s, c, lo, hi, ref):
    """Closed-form finite geometric DFT, independent of production helpers."""
    z = (s[None, :] - 2j*np.pi*BINS[:, None])*DT
    ratio = np.empty_like(z)
    exact = z == 0
    ratio[exact] = hi-lo
    ratio[~exact] = np.expm1((hi-lo)*z[~exact])/np.expm1(z[~exact])
    return DT * ((np.exp(z*lo-s[None, :]*DT*ref)*ratio) @ c)


def infinite_tail(s, c, ref):
    z = (s[None, :] - 2j*np.pi*BINS[:, None])*DT
    return DT * ((np.exp(z*N-s[None, :]*DT*ref)/(-np.expm1(z))) @ c)


def growth_bound(s, c, denominator, horizon):
    rate = s.real*DT
    return ((1+np.exp(2*horizon*rate))*DT*np.exp((N-REF)*rate)*np.abs(c)
            / np.abs(-np.expm1((s-2j*np.pi*BINS)*DT))[:, None] / denominator)


def case(rate, residue, *, growing=False):
    # A decayed pulse provides the whole-record DFT. All data come from these
    # poles/residues; no fitted model or production DFT builds an expectation.
    rates, frequencies = [-.001, rate], [10e9, 10e9]
    residues = [np.exp(-.001*REF), residue]
    if growing:
        rates.append(1e-6)
        frequencies.append(11e9)
        residues.append(1e-12)
    s = np.asarray(rates)/DT + 2j*np.pi*np.asarray(frequencies)
    c = np.asarray(residues, complex)[:, None]
    model = rd.RingdownModel(
        s=s, c=c, n_ref=REF, dt=DT, freq_max=13e9, guard=None,
        factors=(), D=D, rank=len(s), n_window=N-REF, n_decimated=60,
        n_growing_discarded=0, n_invalid=0, n_guard_dropped=0,
        singular_values=np.ones(len(s)), s_growing=np.array([], complex))
    series = np.exp((np.arange(N)[:, None]-REF)*DT*s) @ c
    denominator = np.abs(window_sum(s, c, 0, N, REF))
    return model, series, denominator


@pytest.mark.parametrize('label,rate,residue,growing,expected', [
    ('T1', -6e-11, 1e-6, True, 'pass'),
    ('T2', 7.4e-11, 1e-6, True, 'pass'),
    ('T3-negative', -6e-11, .05, False, 'cannot judge'),
    ('T3-positive', 7.4e-11, .05, False, 'cannot judge'),
    ('T4', 1e-6, .001, False, 'refused'),
    ('T5', -5e-7, .001, True, 'cannot judge'),
    ('T6', -1e-5, .001, False, 'pass'),
])
def test_pole_resolution_verdicts(label, rate, residue, growing, expected):
    model, series, denominator = case(rate, residue, growing=growing)
    own_window = np.abs(window_sum(model.s[1:2], model.c[1:2], REF, N, REF))/denominator
    with np.errstate(over='raise', invalid='raise'):
        verdict = rd.judge_growing_poles(model, N, BINS, denominator)
    assert verdict.status == expected
    if label in ('T1', 'T2'):
        assert own_window.max() < 1e-2
        assert verdict.horizon_steps == N
        assert np.max(growth_bound(model.s[-1], model.c[-1], denominator, N)) < 1e-2
        if label == 'T2':
            assert np.max(growth_bound(model.s[1], model.c[1], denominator, N)) > 1e-2
        # Both tail-witness call paths use the same in-window number.
        for growth in (None, verdict):
            contributions = _pole_contributions(model, series, BINS, denominator, N*DT, growth=growth)
            np.testing.assert_allclose(contributions['non-decaying sinusoidal'], own_window[:, 0], rtol=1e-8)
    elif label.startswith('T3'):
        assert own_window.max() > 1e-2
        assert '10000000000 Hz' in verdict.reason
        assert 'in-window share' in verdict.reason and 'decay per step=' in verdict.reason
    elif label == 'T4':
        bound = growth_bound(model.s[1], model.c[1], denominator, N)
        assert bound.max() > 1e-2
        np.testing.assert_allclose(verdict.share_per_bin, bound[:, 0], rtol=1e-8)
    else:
        own_tail = np.abs(infinite_tail(model.s[1:2], model.c[1:2], REF))/denominator
        assert own_tail.max() >= 1e-4
        assert verdict.horizon_steps == int(np.ceil(np.log(100)/-rate))
        if label == 'T5':
            assert np.isfinite(verdict.band_factor) and verdict.band_factor > 2
            assert '10000000000 Hz' in verdict.reason
            assert 'decay per step=5e-07' in verdict.reason
            assert 'inf' not in verdict.reason and 'nan' not in verdict.reason
        else:
            assert own_tail.max() > 1e-2


@pytest.mark.parametrize('rate,expected', [(-6e-11, 'pass'), (7.4e-11, 'pass'),
                                          (-1e-5, 'fail'), (-6e-11, 'undetermined')])
def test_settling_uses_only_resolved_aggregate_tails(monkeypatch, rate, expected):
    residue = .05 if expected == 'undetermined' else (1e-4 if expected == 'pass' else .001)
    model, series, denominator = case(rate, residue)
    # Isolate the caller from identification roundoff. Return exact residues at
    # each requested window's reference and under the caller's normalization.
    scale = np.max(np.maximum(abs(series.real), abs(series.imag)), axis=0)

    def identify(y, dt, lo, hi, **kwargs):
        return replace(model, n_ref=lo, n_window=hi-lo,
                       c=model.c*np.exp(model.s[:, None]*DT*(lo-REF))/scale)

    monkeypatch.setattr(rd, 'identify', identify)
    witness = tail_share_witness([('synthetic', series)], DT, 0, BINS, freq_max=13e9)
    assert witness.status == expected, witness.reason
    if expected == 'fail':
        exact = np.abs(infinite_tail(model.s, model.c, REF))/denominator
        np.testing.assert_allclose(witness.share_per_bin, exact[:, 0], rtol=1e-8)
        assert exact.max() > 1e-2
    elif expected == 'pass':
        # The unresolved pole has an enormous extrapolated tail; it is excluded.
        assert np.max(np.abs(infinite_tail(model.s[1:2], model.c[1:2], REF))/denominator) > 1e-2
        exact = np.abs(infinite_tail(model.s[:1], model.c[:1], REF))/denominator
        np.testing.assert_allclose(witness.share_per_bin, exact[:, 0], rtol=1e-8)


def test_t3_early_stop_gets_the_sinusoid_rule(monkeypatch):
    model, series, _ = case(7.4e-11, .05)
    monkeypatch.setattr(rd, 'identify', lambda *a, **kw: model)
    fields = rd._judge_record(series, DT, BINS, N, REF, rd.RingdownSpec(),
                              ref_hz=13e9, observable=lambda spectra: spectra.T)
    assert fields['growing_status'] == 'cannot judge'
    assert fields['growing_ok'] is False
    assert 'in-window share' in fields['growing_reason']


def test_overflowing_growth_is_refused_by_finite_log_bound():
    model, _, denominator = case(1e-6, .001)
    model = replace(model, s=np.array([model.s[0], 1/DT+2j*np.pi*10e9]))
    verdict = rd.judge_growing_poles(model, N, BINS, denominator)
    assert verdict.status == 'refused'
    assert np.isfinite(verdict.share_per_bin).all()
    assert 'non-finite' not in verdict.reason


def test_band_beyond_float_range_names_the_horizon_setter():
    model, _, denominator = case(-5e-7, .001, growing=True)
    # A long requested record sets H; the factor must be printed as exp(...).
    verdict = rd.judge_growing_poles(model, 10**10, BINS, denominator)
    assert verdict.status == 'cannot judge'
    assert np.isfinite(verdict.band_factor)
    assert 'band_factor=exp(' in verdict.reason and 'H set by record length' in verdict.reason

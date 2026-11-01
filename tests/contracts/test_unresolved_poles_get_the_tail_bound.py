"""PREDECLARE-FIX-2: literal verdicts and independent geometric-sum oracles."""
from dataclasses import replace

import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.sparams._tail_witness import tail_share_witness

DT, N, D = 1.2e-12, 15252, 15
BINS = np.linspace(8e9, 12e9, 41)
REF = N - 900


def window_sum(s, c, lo, hi, ref):
    z = (s[None, :] - 2j*np.pi*BINS[:, None])*DT
    ratio = np.empty_like(z)
    exact = z == 0
    ratio[exact] = hi-lo
    ratio[~exact] = np.expm1((hi-lo)*z[~exact])/np.expm1(z[~exact])
    return DT * ((np.exp(z*lo-s[None, :]*DT*ref)*ratio) @ c)


def infinite_tail(s, c, ref):
    z = (s[None, :] - 2j*np.pi*BINS[:, None])*DT
    return DT * ((np.exp(z*N-s[None, :]*DT*ref)/(-np.expm1(z))) @ c)


def bounds(model, denominator, selected, horizon):
    s, c = model.s[selected], model.c[selected]
    rate = s.real*DT
    z = (s[None, :] - 2j*np.pi*BINS[:, None])*DT
    return ((1+np.exp(2*horizon*rate))[None, :, None]*DT
            * np.exp((N-model.n_ref)*rate)[None, :, None]*np.abs(c)[None, :, :]
            / np.abs(-np.expm1(z))[:, :, None]/denominator[:, None, :])


def case(rates, frequencies, residues, *, decimation=D):
    s = np.asarray([-.001, *rates])/DT + 2j*np.pi*np.asarray([10e9, *frequencies])
    c = np.asarray([np.exp(-.001*REF), *residues], complex)[:, None]
    model = rd.RingdownModel(
        s=s, c=c, n_ref=REF, dt=DT, freq_max=13e9, guard=None,
        factors=(), D=decimation, rank=len(s), n_window=N-REF, n_decimated=60,
        n_growing_discarded=0, n_invalid=0, n_guard_dropped=0,
        singular_values=np.ones(len(s)), s_growing=np.array([], complex))
    series = np.exp((np.arange(N)[:, None]-REF)*DT*s) @ c
    return model, series, np.abs(window_sum(s, c, 0, N, REF))


@pytest.mark.parametrize('label,rates,frequencies,residues,decimation,expected,horizon', [
    ('T1', [-6e-11, 1e-9], [10.05e9, 11.05e9], [1e-4, 1e-9], 15, 'pass', N),
    ('T2a', [7.4e-11], [10e9], [1e-6], 15, 'refused', N),
    ('T2b', [-7.4e-11], [10e9], [1e-6], 15, 'refused', N),
    ('T2c-positive', [7.4e-11], [10.05e9], [1e-6], 15, 'pass', N),
    ('T2c-negative', [-7.4e-11], [10.05e9], [1e-6], 15, 'pass', N),
    ('T3', [-5e-7, 3e-8], [10e9, 11e9], [1e-3, 1e-9], 16, 'cannot judge', 9210341),
    ('T4', [-5e-7, -3e-8], [10e9, 11e9], [1e-3, 1e-9], 16, 'cannot judge', 9210341),
    ('T5', [-1e-5, 3e-8], [10e9, 11e9], [1e-3, 1e-9], 15, 'pass', 460518),
    ('T7', [1e-6], [10e9], [1e-3], 15, 'refused', N),
])
def test_literal_verdict(label, rates, frequencies, residues, decimation, expected, horizon):
    model, _, denominator = case(rates, frequencies, residues, decimation=decimation)
    judged = np.flatnonzero(model.s.real*DT > -np.log1p(1e-6)/decimation)
    expected_share = bounds(model, denominator, judged, horizon).max(axis=(1, 2))
    result = rd.judge_growing_poles(model, N, BINS, denominator)
    assert result.status == expected, result.reason
    assert result.horizon_steps == horizon
    np.testing.assert_allclose(result.share_per_bin, expected_share, rtol=1e-10)
    assert result.band_factor == pytest.approx(np.exp(2*horizon/decimation*np.log1p(1e-6)))
    assert np.isfinite(result.band_factor)
    if expected == 'cannot judge':
        assert result.band_factor == pytest.approx(3.162276, rel=1e-5)
        assert 'band_factor=' in result.reason and 'inf' not in result.reason
    else:
        assert (expected_share.max() > 1e-2) == (expected == 'refused')


def test_t3_reviewers_identified_probe():
    # The reviewer's saved test_real_identify construction, including RNG seed.
    n = np.arange(N)
    y = (np.exp(-5e-7*n)*np.cos(2*np.pi*10e9*n*DT)
         + 0.5*np.exp(-2e-4*n)*np.cos(2*np.pi*9e9*n*DT)
         + 1e-4*np.exp(3e-8*n)*np.cos(2*np.pi*10.5e9*n*DT + .3))
    rng = np.random.default_rng(0)
    y = (y + 1e-9*rng.standard_normal(N))[:, None]
    model = rd.identify(y, DT, N//2, N, freq_max=13e9)
    # Direct finite sum of the supplied noisy series, independent of plain_dft.
    denominator = np.abs(DT*np.exp(-2j*np.pi*BINS[:, None]*DT*n) @ y)
    result = rd.judge_growing_poles(model, N, BINS, denominator)
    assert model.D == 16
    assert result.status == 'cannot judge', result.reason
    assert result.band_factor == pytest.approx(3.162276, rel=1e-4)
    assert np.isfinite(result.band_factor) and 'inf' not in result.reason
    selected = np.flatnonzero(model.s.real*DT > -np.log1p(1e-6)/model.D)
    horizons = [N]
    for k, s in enumerate(model.s):
        if s.real*DT <= -np.log1p(1e-6)/model.D:
            own = np.abs(infinite_tail(model.s[k:k+1], model.c[k:k+1], model.n_ref))/denominator
            if own.max() >= 1e-4:
                horizons.append(int(np.ceil(np.log(100)/(-s.real*DT))))
    horizon = max(horizons)
    assert result.horizon_steps == horizon
    expected = bounds(model, denominator, selected, horizon).max(axis=(1, 2))
    np.testing.assert_allclose(result.share_per_bin, expected, rtol=1e-10)


def fit_exact(monkeypatch, model, series, *, normalize=False):
    scale = np.max(np.maximum(abs(series.real), abs(series.imag)), axis=0) if normalize else 1

    def identify(y, dt, lo, hi, **kwargs):
        return replace(model, n_ref=lo, n_window=hi-lo,
                       c=model.c*np.exp(model.s[:, None]*DT*(lo-REF))/scale)

    monkeypatch.setattr(rd, 'identify', identify)


def test_t6_resolved_decay_keeps_settling_tail_rule(monkeypatch):
    model, series, denominator = case([-1e-5], [10e9], [1e-3])
    fit_exact(monkeypatch, model, series, normalize=True)
    witness = tail_share_witness([('synthetic', series)], DT, 0, BINS, freq_max=13e9)
    expected = np.abs(infinite_tail(model.s, model.c, REF))/denominator
    assert witness.status == 'fail', witness.reason
    assert expected.max() > 1e-2
    np.testing.assert_allclose(witness.share_per_bin, expected[:, 0], rtol=1e-8)


@pytest.mark.parametrize('rate', [-7.4e-11, 7.4e-11])
def test_m3_completion_omits_unresolved_tail(monkeypatch, rate):
    model, series, _ = case([rate], [10.05e9], [1e-6])
    fit_exact(monkeypatch, model, series)
    expected = window_sum(model.s, model.c, 0, N, REF) + infinite_tail(model.s[:1], model.c[:1], REF)
    full = expected + infinite_tail(model.s[1:], model.c[1:], REF)
    assert np.max(abs(full-expected)) > 1e-18
    np.testing.assert_allclose(rd.complete_spectra(series, DT, BINS, model, N), expected, rtol=1e-10)
    result = rd.two_window_witness(series, DT, BINS, N, N//2, freq_max=13e9)
    np.testing.assert_allclose(result.spectra, expected, rtol=1e-10)
    np.testing.assert_allclose(result.spectra_short, expected, rtol=1e-10)

    # The traced forward completion retains both signs in its residue fit,
    # while masking only their infinite-sum tails. Use real conjugate pairs.
    import jax.numpy as jnp
    from rfx.ringdown import _jax as rj
    from rfx.measurement.ringdown_resolution import completion_poles
    from tests._x64_compat import enable_x64

    real_model = replace(model, s=np.concatenate([model.s, model.s.conj()]),
                         c=np.concatenate([model.c, model.c.conj()])/2)
    real_series = series.real
    expected = (window_sum(real_model.s, real_model.c, 0, N, REF)
                + infinite_tail(real_model.s[[0, 2]], real_model.c[[0, 2]], REF))
    with enable_x64():
        y = jnp.asarray(real_series)
        s, mask, status, arg, count = rj.host_poles(
            lambda _: completion_poles(real_model, N), y[REF:], 4, DT, BINS)
        actual = rj.completion(y, DT, BINS, REF, s, mask, arg)
        assert int(status) == rj.STATUS_OK and int(count) == 4
        np.testing.assert_array_equal(mask, [1, 2, 1, 2])
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

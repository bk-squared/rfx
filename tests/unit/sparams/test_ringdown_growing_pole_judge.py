"""Closed-form and record oracles for the kept growing-pole judge (#2158)."""
from dataclasses import replace

import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.sparams._tail_witness import tail_share_witness

DT = 1.43e-12
BINS = np.linspace(8e9, 12e9, 21)
UNIT_TOL = 1e-6


def _model(s, c, n_ref=321, D=1):
    return rd.RingdownModel(
        s=np.asarray(s, dtype=complex), c=np.asarray(c, dtype=complex),
        n_ref=n_ref, dt=DT, freq_max=13e9, guard=None, factors=(), D=D,
        rank=len(s), n_window=0, n_decimated=0, n_growing_discarded=0,
        n_invalid=0, n_guard_dropped=0, singular_values=np.zeros(0))


def _bound(s, c, n, n_ref, H, denominator):
    rate = s.real * DT
    divisor = np.abs(1 - np.exp(s * DT) * np.exp(-2j*np.pi*BINS*DT))
    return ((1 + np.exp(2*H*rate)) * DT * np.abs(c)[None, :]
            * np.exp((n-n_ref)*rate) / divisor[:, None] / denominator)


def _finite_tail(s, c, n, n_ref, h, bins=BINS):
    # Sum the actual samples, in blocks so c5 does not need a huge array.
    total = np.zeros(len(bins), dtype=complex)
    for lo in range(n, n+h, 16384):
        m = np.arange(lo, min(lo+16384, n+h))
        total += np.sum(np.exp((s-2j*np.pi*np.asarray(bins)[:, None])*DT*m
                              - s*DT*n_ref) * c, axis=1) * DT
    return total


def _c2():
    n, n_ref = 1024, 321
    decay = (-.001 + 2j*np.pi*10e9*DT) / DT
    growing = (np.log1p(.5*UNIT_TOL) + 2j*np.pi*10e9*DT) / DT
    H = int(np.ceil(np.log(100)/.001))
    denom = np.tile([1e-10, 2e-10], (len(BINS), 1))
    residue = .03 / _bound(growing, np.array([1., .4]), n, n_ref, H, denom)[10, 0]
    model = _model([decay, growing], [[1., .5], [residue, .4*residue]], n_ref)
    return model, n, H, denom


def test_c2_significant_kept_growing_tail_is_refused():
    model, n, H, denom = _c2()
    expected = _bound(model.s[1], model.c[1], n, model.n_ref, H, denom)
    judge = rd.judge_growing_poles(model, n, BINS, denom)
    assert judge.status == 'refused' and not judge.ok
    assert judge.n_kept_growing == 1 and judge.n_exempt == 0
    assert judge.horizon_steps == H
    np.testing.assert_allclose(judge.share_per_bin, expected.max(axis=1), rtol=1e-9)
    assert judge.worst == pytest.approx((10e9, .5*UNIT_TOL, .03), rel=1e-9)
    assert '10000000000 Hz' in judge.reason and 'per step' in judge.reason
    assert 'bin' in judge.reason


def test_c2_bound_covers_every_requested_finite_horizon():
    model, n, H, denom = _c2()
    judge = rd.judge_growing_poles(model, n, BINS, denom)
    for h in (1, n, H, 2*H):
        exact = np.abs(_finite_tail(model.s[1], model.c[1, 0], n, model.n_ref, h)) / denom[:, 0]
        assert np.all(judge.share_per_bin >= exact)


def test_c4_rounding_pole_at_60_ghz_passes_despite_count():
    decay = (-.001 + 2j*np.pi*10e9*DT) / DT
    growing = (np.log1p(.5*UNIT_TOL) + 2j*np.pi*60e9*DT) / DT
    model = _model([decay, growing], [[1.], [1e-8]])
    n, H = 1024, int(np.ceil(np.log(100)/.001))
    denom = np.full((len(BINS), 1), 1e-10)
    judge = rd.judge_growing_poles(model, n, BINS, denom)
    grow, _ = rd.growing_poles(model, n*DT, UNIT_TOL)
    assert len(grow) == judge.n_kept_growing == 1
    assert judge.status == 'pass' and judge.ok
    np.testing.assert_allclose(judge.share_per_bin,
                               _bound(growing, model.c[1], n, model.n_ref, H, denom)[:, 0])


def test_c5_long_decay_horizon_cannot_resolve_growth():
    n, H = 1024, 3_400_000
    decay_rate = np.log(100) / (H-.25)
    growth_rate = np.log(29) / H
    assert np.expm1(growth_rate) < UNIT_TOL
    decay = (-decay_rate + 2j*np.pi*10e9*DT) / DT
    growing = (growth_rate + 2j*np.pi*10e9*DT) / DT
    denom = np.full((len(BINS), 1), 1e-10)
    residue = 1e-3 * denom[10, 0] / (DT*np.exp(n*growth_rate)
                                         * np.expm1(H*growth_rate)/np.expm1(growth_rate))
    model = _model([decay, growing], [[1.], [residue]], n_ref=0)
    at_H = abs(_finite_tail(growing, residue, n, 0, H, BINS[10:11])[0]) / denom[10, 0]
    at_2H = abs(_finite_tail(growing, residue, n, 0, 2*H, BINS[10:11])[0]) / denom[10, 0]
    assert at_H == pytest.approx(1e-3, rel=1e-10)
    assert at_2H == pytest.approx(3e-2, rel=1e-10)
    judge = rd.judge_growing_poles(model, n, BINS, denom)
    assert judge.status == 'cannot judge' and not judge.ok
    assert judge.horizon_steps == H
    assert judge.decay_horizon_steps == pytest.approx(H-.25)
    assert judge.band_factor == pytest.approx((1+UNIT_TOL)**(2*H), rel=1e-9)
    assert judge.share_per_bin[10] >= at_2H > 1e-2
    assert f'H={H}' in judge.reason and 'D=1' in judge.reason and 'band_factor=' in judge.reason
    decimated = rd.judge_growing_poles(replace(model, D=20), n, BINS, denom)
    assert decimated.band_factor < 2 and decimated.status == 'refused'


@pytest.mark.parametrize('residue, expected_H', [(1e-20, 1024), (1., 4606)])
def test_horizon_uses_only_significant_decaying_poles(residue, expected_H):
    model = _model([(-.001+2j*np.pi*10e9*DT)/DT], [[residue]])
    judge = rd.judge_growing_poles(model, 1024, BINS, np.full((len(BINS), 1), 1e-10))
    assert judge.horizon_steps == expected_H
    assert judge.decay_horizon_steps == pytest.approx(0 if residue < 1e-4 else np.log(100)/.001)
    assert judge.status == 'pass' and judge.ok
    assert np.isnan(judge.band_factor) and judge.worst is None
    np.testing.assert_array_equal(judge.share_per_bin, np.zeros(len(BINS)))


def test_static_exemption_and_inactive_channel():
    model = _model([np.log1p(.5*UNIT_TOL)/DT,
                    (np.log1p(.5*UNIT_TOL)+2j*np.pi*60e9*DT)/DT],
                   [[1., 1.], [1e-8, 1e8]])
    denom = np.tile([1e-10, np.inf], (len(BINS), 1))
    judge = rd.judge_growing_poles(model, 1024, BINS, denom)
    assert judge.ok and judge.n_exempt == 1 and judge.n_kept_growing == 1
    assert judge.horizon_steps == 1024 and judge.decay_horizon_steps == 0


def test_no_growing_pole_passes_even_where_a_channel_dft_is_zero():
    # run(ringdown=) passes the raw |plain|: a zero there must not refuse a record nothing grows in.
    model = _model([(-.001+2j*np.pi*10e9*DT)/DT], [[1.]])
    denom = np.full((len(BINS), 1), 1e-10)
    denom[3, 0] = 0.
    judge = rd.judge_growing_poles(model, 1024, BINS, denom)
    assert judge.status == 'pass' and judge.n_kept_growing == 0


def test_any_positive_rate_is_judged_and_an_exact_sinusoid_is_not():
    # A growth of 1e-13 per step is below the in-window rule's sinusoid test but still a
    # growing pole: on a read bin its tail bound is large, so it is refused.
    slow = (1e-13 + 2j*np.pi*10e9*DT) / DT
    exact = (2j*np.pi*10e9*DT) / DT
    denom = np.full((len(BINS), 1), 1e-10)
    slow_judge = rd.judge_growing_poles(_model([slow], [[1e-3]]), 1024, BINS, denom)
    expected = _bound(slow, np.array([1e-3]), 1024, 321, 1024, denom)[:, 0]
    assert slow_judge.status == 'refused'
    # On its own bin 1 - lambda z is ~1e-13 and the product form above loses digits there.
    off = np.arange(len(BINS)) != 10
    np.testing.assert_allclose(slow_judge.share_per_bin[off], expected[off], rtol=1e-9)
    assert slow_judge.share_per_bin[10] > 1e6
    exact_judge = rd.judge_growing_poles(_model([exact], [[1e-3]]), 1024, BINS, denom)
    assert exact_judge.status == 'pass'
    np.testing.assert_array_equal(exact_judge.share_per_bin, np.zeros(len(BINS)))


@pytest.mark.parametrize('bad', ['residue', 'denominator', 'overflow'])
def test_nonfinite_values_cannot_judge(bad):
    model, n, _, denom = _c2()
    if bad == 'residue':
        model.c[1, 0] = np.nan
    elif bad == 'denominator':
        denom[0, 0] = np.nan
    else:
        model.s[1] = 1/DT + 2j*np.pi*10e9
    judge = rd.judge_growing_poles(model, n, BINS, denom)
    assert judge.status == 'cannot judge' and not judge.ok
    assert 'non-finite' in judge.reason


def test_c1_true_decaying_tail_still_fails():
    n = 1600
    rate = -np.log(1+1/.03)/n
    s = rate/DT + 2j*np.pi*10e9
    record = np.exp(s*DT*np.arange(n)).astype(np.complex64)
    z = np.exp(-2j*np.pi*BINS*DT)
    lam = np.exp(s*DT)
    whole = DT*(1-(lam*z)**n)/(1-lam*z)
    remaining = DT*(lam*z)**n/(1-lam*z)
    assert abs(remaining[10]/whole[10]) == pytest.approx(.03, rel=1e-10)
    witness = tail_share_witness([('decaying', record)], DT, 0, BINS, freq_max=13e9)
    assert witness.status == 'fail', witness.reason
    assert witness.share_per_bin[10] == pytest.approx(.03, rel=1e-4)


def test_c3_resolved_growing_window_term_still_undetermined():
    n = 1600
    m = np.arange(n)
    growth = np.log1p(1e-3)
    decay = -.01
    decaying_sum = np.exp(decay*m).sum()
    growing_sum = np.exp(growth*m).sum()
    window_sum = np.exp(growth*m[n//2:]).sum()
    residue = .03*decaying_sum/(window_sum-.03*growing_sum)
    record = ((np.exp(decay*m)+residue*np.exp(growth*m))
              * np.exp(2j*np.pi*10e9*DT*m)).astype(np.complex64)
    exact = residue*window_sum/(decaying_sum+residue*growing_sum)
    assert exact == pytest.approx(.03)
    witness = tail_share_witness([('resolved', record)], DT, 0, BINS, freq_max=13e9)
    assert witness.status == 'undetermined'
    assert 'discarded growing' in witness.reason
    assert witness.pole_contributions['discarded growing'][10] >= .03*(1-1e-4)


def test_issue_2158_seed_1_rounding_record_passes():
    n = 3514
    rng = np.random.default_rng(1)
    t = np.arange(n)*DT
    modes = [(9e9, 8), (10.4e9, 12), (11.7e9, 6), (9.8e9, 4)]
    cont = [(f, .6+2*rng.random()) for f in rng.uniform(1e9, 40e9, 40)]
    y = np.zeros((n, 6))
    for f, q in modes+cont:
        amp = rng.normal(size=6)*(1 if q > 5 else 3)
        phase = rng.uniform(0, 2*np.pi, size=6)
        y += amp*np.exp(-np.pi*f/q*t)[:, None]*np.cos(2*np.pi*f*t[:, None]+phase)
    y = y/np.max(np.abs(y), axis=0)+6e-8*rng.normal(size=y.shape)
    y = y.astype(np.float32).astype(np.float64)
    model = rd.identify(y, DT, round(.4*n), n, freq_max=13e9)
    # How many rounding poles land in the kept band depends on the platform's
    # rounding (2 on an arm64 Mac); test_c4 pins "counted, yet passes" exactly.
    judge = rd.judge_growing_poles(model, n, BINS, np.abs(rd.plain_dft(y, DT, BINS)))
    assert judge.status == 'pass' and judge.ok
    witness = tail_share_witness([('rounding', y)], DT, 0, BINS, freq_max=13e9)
    assert witness.status == 'pass', witness.reason


def test_run_witness_uses_tail_share_and_keeps_completed_s():
    from tests.unit.sparams.test_ringdown_run import _box, _run_captured, N_SHORT
    run, cap = _run_captured(_box('uniform'), N_SHORT['uniform'], ringdown=rd.RingdownSpec())
    assert run.ringdown.report.completed
    Y, dt, bins, n, start = cap['args']
    kw = cap['kw']
    model = rd.identify(Y, dt, start, n, freq_max=kw['freq_max'], guard=kw['guard'],
                        sv_rel=kw['sv_rel'], unit_tol=kw['unit_tol'])
    judge = rd.judge_growing_poles(model, n, bins, np.abs(kw['plain']), unit_tol=kw['unit_tol'])
    witness = run.ringdown.report.witness('growing_poles')
    assert witness.value == np.max(judge.share_per_bin)
    assert witness.bar == 1e-2 and witness.ok == judge.ok
    assert judge.status in witness.note

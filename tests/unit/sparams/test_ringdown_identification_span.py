"""The identification span bounds cost without changing the short-record path."""
import numpy as np
import pytest

from rfx import ringdown as rd
from rfx._ringdown_pencil import IDENTIFICATION_SAMPLE_CAP, identification_span
from rfx.sparams._tail_witness import tail_share_witness


@pytest.mark.parametrize('n', [199, 200, 1000, 32832])
def test_identification_span_is_the_whole_record_below_cap(n):
    dt, freq_max = 9.68540033721751e-13, 15e9
    assert rd.decimation_plan((n + 1)//2, dt, freq_max)[1] <= 1024
    assert identification_span(n, dt, freq_max) == n


def test_four_n2_record_identifies_at_cap_and_after_source_end(monkeypatch):
    n, dt, freq_max = 4 * 32832, 9.68540033721751e-13, 15e9
    factors, old_count = rd.decimation_plan(n//2, dt, freq_max)
    assert factors == (8, 2) and old_count > 1024
    # Each FIR stage retains ceil(W/q) - 20 samples; invert by hand.
    half = ((1024 + 20) * 2 + 20) * 8
    expected_span = 2 * half
    assert rd.decimation_plan(half + 1, dt, freq_max)[1] == 1025
    source_end = n - expected_span + 123
    models = []
    identify = rd.identify

    def capture(*args, **kwargs):
        model = identify(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(rd, 'identify', capture)
    y = np.exp((-1e6 + 2j*np.pi*10e9) * np.arange(n) * dt)
    tail_share_witness([('synthetic', y)], dt, source_end, [10e9], freq_max=freq_max)
    assert len(models) == 2
    assert models[0].n_decimated <= IDENTIFICATION_SAMPLE_CAP, 'main window exceeds N_cap'
    assert models[0].n_window == half
    assert all(model.n_ref >= source_end for model in models), 'identification precedes source end'
    assert models[1].n_window == round(.9 * half)


def test_ringdown_w2_and_early_start_use_the_same_span():
    n, dt, freq_max = 8192, 1., .2
    # No decimation: m=2048, main=1024, early=1536, split=90% of m.
    y = np.exp((-.0001 + 2j*np.pi*.1) * np.arange(n))
    w2 = rd.two_window_witness(y, dt, [.1], n, n//2, freq_max=freq_max)
    we = rd.early_start_witness(y, dt, [.1], n, .5, freq_max=freq_max)
    assert w2.model.n_ref == n - 1024
    assert w2.model_short.n_ref == n - 1024
    assert w2.model_short.n_window == round(.9 * 2048) - 1024
    assert we.model_long.n_ref == n - 1536


@pytest.mark.parametrize('start_today', [4096, 8100])
def test_until_identified_main_start_uses_span_and_todays_floor(monkeypatch, start_today):
    seen = []

    def capture(series, dt, start, stop, **kw):
        seen.append((start, stop))
        raise ValueError('window captured before pencil')

    monkeypatch.setattr(rd, 'identify', capture)
    rd._judge_record(np.ones((8192, 1)), 1., [.1], 8192, start_today,
                     rd.RingdownSpec(), ref_hz=.2, observable=lambda x: x)
    assert seen == [(max(start_today, 8192 - 1024), 8192)]

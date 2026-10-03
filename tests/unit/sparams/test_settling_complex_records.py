"""Complex per-bin tail ratios and amplitude scaling."""
import numpy as np
import pytest

from rfx.sparams._tail_witness import tail_share_witness


def score(y):
    return tail_share_witness([('record', y)], 1., 0, [.07, .12], freq_max=.15)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_global_phase_preserves_each_read_bin(dtype):
    t = np.arange(400)
    y = (np.exp((-.02+2j*np.pi*.07)*t)
         + .03*np.exp((-.005+2j*np.pi*.12)*t)).astype(dtype)
    first, rotated = score(y), score(1j*y)
    assert first.status == rotated.status == 'fail'
    np.testing.assert_allclose(first.share_per_bin, rotated.share_per_bin, rtol=1e-6)


@pytest.mark.parametrize('scale', [1., 1e200, 1e-200, 1e-38])
def test_amplitude_units_preserve_tail_share(scale):
    y = np.exp((-.02 + 2j*np.pi*.07)*np.arange(200))
    first, changed = score(y), score(y*scale)
    assert first.status == changed.status == 'fail'
    np.testing.assert_allclose(first.share_per_bin, changed.share_per_bin, rtol=1e-7)


def test_tail_ratio_matches_geometric_sum():
    n = 200
    pole = np.exp(-.02 + 2j*np.pi*.07)
    y = pole**np.arange(n)
    result = score(y)
    z = np.exp(-2j*np.pi*np.array([.07, .12]))
    expected = np.abs((pole*z)**n / (1-(pole*z)**n))
    np.testing.assert_allclose(result.share_per_bin, expected, rtol=1e-6)


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_invalid_channel_is_not_hidden(bad):
    y = np.exp((-.05 + 2j*np.pi*.07)*np.arange(200))
    result = score(np.column_stack([y, np.full(200, bad)]))
    assert result.status == 'undetermined'
    assert result.reason
    assert np.isnan(result.db)


@pytest.mark.parametrize('source_end', [0, 40])
def test_zero_post_source_channels_have_zero_share(monkeypatch, source_end):
    import rfx.ringdown as ringdown

    original = ringdown.identify
    shapes = []

    def observed(series, *args, **kwargs):
        shapes.append(series.shape)
        return original(series, *args, **kwargs)

    y = np.exp((-.02 + 2j*np.pi*.07)*np.arange(200))
    zero_tail = np.zeros_like(y)
    zero_tail[:source_end] = 1.
    expected = tail_share_witness([('live', y)], 1., source_end,
                                  [.07, .12], freq_max=.15)
    monkeypatch.setattr(ringdown, 'identify', observed)
    per_record = {}
    result = tail_share_witness(
        [('live', np.column_stack([zero_tail, y])), ('zero', zero_tail)],
        1., source_end, [.07, .12], freq_max=.15, _record_results=per_record)
    assert result.status == expected.status == 'fail'
    assert result.reason == ''
    assert shapes == [(200, 1), (200, 1)]
    np.testing.assert_allclose(result.share_per_bin, expected.share_per_bin)
    np.testing.assert_allclose(result.error_per_bin, expected.error_per_bin)
    assert per_record['live'] == pytest.approx(expected.db)
    assert 10**(per_record['zero']/20) == 0.


@pytest.mark.parametrize('source_end', [0, 40])
def test_all_zero_post_source_group_passes(monkeypatch, source_end):
    import rfx.ringdown as ringdown

    def unexpected(*args, **kwargs):
        pytest.fail('an all-zero group reached identification')

    monkeypatch.setattr(ringdown, 'identify', unexpected)
    y = np.zeros((200, 2))
    y[:source_end] = 1.
    result = tail_share_witness([('zero', y)], 1., source_end,
                                [.07, .12], freq_max=.15)
    assert result.status == 'pass'
    assert result.reason == 'zero: no post-source variation'
    assert result.db <= -40
    np.testing.assert_array_equal(result.share_per_bin, [0., 0.])


def test_named_records_share_one_multichannel_identification(monkeypatch):
    import rfx.ringdown as ringdown

    original = ringdown.identify
    calls = []

    def observed(series, dt, start, stop, **kwargs):
        calls.append((series.shape, start, stop))
        return original(series, dt, start, stop, **kwargs)

    monkeypatch.setattr(ringdown, 'identify', observed)
    y = np.exp((-.02 + 2j*np.pi*.07)*np.arange(200))
    result = tail_share_witness([('v', y), ('i', np.column_stack([2*y, 3j*y]))],
                                1., 0, [.07], freq_max=.15)
    assert result.status == 'fail'
    assert calls == [((200, 3), 100, 200), ((200, 3), 50, 200)]


def test_late_source_end_uses_remaining_window(monkeypatch):
    import rfx.ringdown as ringdown

    original = ringdown.identify
    windows = []

    def observed(series, dt, start, stop, **kwargs):
        windows.append((start, stop))
        return original(series, dt, start, stop, **kwargs)

    monkeypatch.setattr(ringdown, 'identify', observed)
    n = 700
    y = np.exp((-.002 + 2j*np.pi*.07)*np.arange(n))
    result = tail_share_witness([('record', y)], 1., round(.95*n),
                                [.07], freq_max=.15)
    assert result.status == 'fail', result.reason
    assert windows == [(665, 700), (665, 697)]
    np.testing.assert_allclose(result.share_per_bin, [np.exp(-.002*n)/(1-np.exp(-.002*n))],
                               rtol=1e-6)


@pytest.mark.parametrize('source_end', [190, 191, 199])
def test_short_identification_check_is_undetermined(monkeypatch, source_end):
    import rfx.ringdown as ringdown

    def unexpected(*args, **kwargs):
        pytest.fail('a short identification window reached the pencil')

    monkeypatch.setattr(ringdown, 'identify', unexpected)
    y = np.exp((-.002 + 2j*np.pi*.07)*np.arange(200))
    result = tail_share_witness([('record', y)], 1., source_end, [.07], freq_max=.15)
    assert result.status == 'undetermined'
    assert result.reason == 'record: post-source window too short for the identification check'

"""F-d: a direct sum, independent of the drive-tail implementation."""

import numpy as np

from rfx.measurement.record_end import drive_tail_share


def test_drive_tail_share_direct_sum():
    dt, n_record = .025, 320
    n = np.arange(2048, dtype=np.float64)
    drives = [np.exp(-n / 90) * np.cos(2 * np.pi * .7 * n * dt)
              + .3 * (1 + n / 40) ** -1.5 * np.cos(2 * np.pi * f * n * dt)
              for f in (.31, .55)]
    freqs = np.array([.31, .55, .7, .93])
    expected = np.zeros(len(freqs))
    for drive in drives:
        values = []
        for f in freqs:
            phase = np.exp(-2j * np.pi * f * n * dt)
            values.append(abs(np.sum(drive[n_record:] * phase[n_record:]))
                          / abs(np.sum(drive[:n_record] * phase[:n_record])))
        expected = np.maximum(expected, values)
    actual = drive_tail_share(iter(drives), dt, n_record, freqs)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-15,
                               err_msg="F-d: known tail / in-record DFT must match direct sums")


def test_no_samples_beyond_record():
    np.testing.assert_array_equal(drive_tail_share([np.zeros(10)], 1., 10, [.1]), [0.])


def test_witness_adds_drive_to_pole_budget():
    from rfx.sparams._tail_witness import tail_share_witness

    n = np.arange(800)
    records = [('mode', np.exp(-n / 80) * np.cos(2 * np.pi * .07 * n))]
    drive = np.exp(-np.arange(1600) / 200) * np.cos(2 * np.pi * .07 * np.arange(1600))
    kw = dict(freq_max=.15)
    poles = tail_share_witness(records, 1., 0, [.07], **kw)
    total = tail_share_witness(records, 1., 0, [.07], known_drives=[drive], n_record=800, **kw)
    np.testing.assert_allclose(total.share_per_bin,
                               poles.share_per_bin + total.drive_share_per_bin)
    assert total.drive_share_per_bin[0] > 1e-2
    assert poles.status == 'pass' and total.status == 'fail'

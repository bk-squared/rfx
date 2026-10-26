"""The known drive beyond a record: one input of the record-end judgement (#2030).

A drive whose samples are known past the record (a waveguide port's H companion
near cutoff decays as t^-3/2 and outlives any practical record) adds, to every
recorded quantity, the response to its own beyond-record part. By causality that
response lies wholly after the record, and its share of a record's in-record DFT
per bin is ``|DFT d[N:K]| / |DFT d[:N]|``: the structure's transfer cancels, so
it is a property of the drive alone. The record check adds it to the pole tail
in one per-bin budget (rfx.sparams._tail_witness); until_decay waits for it
(rfx.measurement.decay).

Limits (separate-instance review, #2022): the max over a port's E and H drives
bounds the combined share where their responses add, downstream of a one-way
source (there it is about 2x high); behind the source they cancel and it is not
a bound. Every scored plane of a waveguide port is downstream of its source.
The power-sum allowance 2s + s^2 built on this share is an estimate (e = s per
S entry), adopted after a check against settled records, not a proven bound.
"""
from __future__ import annotations

import numpy as np


def drive_tail_share(drives, dt, n_record, freqs) -> np.ndarray:
    """Known drive beyond a record / inside it, per bin, maximized over drives.

    Host-only float64 plain DFTs; the tail's time-origin phase cancels in abs.
    A drive ending at the record contributes zero, including an all-zero drive.
    """
    from rfx.measurement.ringdown_tail import plain_dft

    bins = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if int(n_record) != n_record or n_record < 0:
        raise ValueError("n_record must be a nonnegative integer")
    n_record = int(n_record)
    share = np.zeros(bins.size, dtype=np.float64)
    for raw in drives:
        drive = np.asarray(raw, dtype=np.float64)
        if drive.ndim != 1 or len(drive) < n_record or not np.isfinite(drive).all():
            raise ValueError("known drives must be finite 1-D samples of length >= n_record")
        if len(drive) == n_record:
            continue
        denominator = np.abs(plain_dft(drive[:n_record], dt, bins))
        numerator = np.abs(plain_dft(drive[n_record:], dt, bins))
        ratio = np.zeros(bins.size, dtype=np.float64)
        np.divide(numerator, denominator, out=ratio, where=denominator != 0)
        ratio[(denominator == 0) & (numerator != 0)] = np.inf
        share = np.maximum(share, ratio)
    return share

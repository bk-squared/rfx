"""A decay far shorter than its record keeps its Q through ``Result.find_resonances`` (#1626).

The sheet perturbation-Q oracle (``tests/oracle/test_sheet_perturbation_q.py``) reads a Q of about 14 at 25.4 GHz
from a 12.25 ns record, and its mode is gone after about 1 ns. A Hann-windowed half-power width of such a record reads
about 1.9x the Q: the window's rise multiplies the whole decay by about t^2. This pins the extractor the oracle uses,
on the oracle's own sampling and source onset.
"""

from __future__ import annotations

import numpy as np

from rfx import GaussianPulse, Result

_DT = 1.0209367720161057e-13  # the sheet oracle's realized dt
_N = 120000                    # its record length
_FREQ = 25.40e9
_Q = 13.9
_WAVEFORM = GaussianPulse(f0=29e9, bandwidth=0.6)


def test_short_decay_in_a_long_record_keeps_its_q():
    t = np.arange(_N) * _DT - _WAVEFORM.t0
    decay = np.exp(-np.pi * _FREQ * np.maximum(t, 0.0) / _Q) * np.cos(2 * np.pi * _FREQ * t)
    trace = np.where(t >= 0.0, decay, 0.0).astype(np.float32)
    result = Result(state=None, time_series=trace[:, None], s_params=None, freqs=None, dt=_DT)

    modes = result.find_resonances(freq_range=(20e9, 27e9), waveform=_WAVEFORM)

    mode = min(modes, key=lambda m: abs(m.freq - _FREQ))
    assert abs(mode.freq - _FREQ) / _FREQ < 1e-3, mode
    assert abs(mode.Q - _Q) / _Q < 1e-2, mode

"""A result read in time or frequency must use the step the solver took.

``stencil_order=4`` advances the fields by ``0.857 * grid.dt`` (the (2,4)
stability bound), but ``Result.dt`` used to report ``grid.dt``. Everything a
user reads through ``Result.dt`` then came out scaled by 0.857: on a
20 x 20 x 10 mm PEC cavity the TM110 mode (10.599 GHz) was reported at
9.085 GHz by ``find_resonances`` and by an FFT of the probe record, -14.3 %.

The physical witness here is a closed PEC box whose TM110 frequency is known
in closed form, c0/2 * sqrt(1/a^2 + 1/b^2); both stencil orders must put it
within 1 %. It does not share a helper with the code under test: the only
rfx inputs are the declared box and the probe record.
"""
import math

import numpy as np
import pytest

from rfx import Simulation

_A, _B, _C = 0.020, 0.020, 0.005
_F_TM110 = 299792458.0 / 2 * math.sqrt(1 / _A ** 2 + 1 / _B ** 2)   # 10.5993 GHz


def _cavity(stencil_order):
    sim = Simulation(freq_max=15e9, domain=(_A, _B, _C), dx=1e-3,
                     boundary="pec", stencil_order=stencil_order)
    sim.add_source((0.006, 0.007, 0.002), component="ez")
    sim.add_probe((0.014, 0.012, 0.002), component="ez")
    return sim


@pytest.fixture(scope="module")
def cavity_runs():
    return {order: _cavity(order).run(n_steps=2000, skip_preflight=True)
            for order in (2, 4)}


def test_result_dt_is_the_step_the_scan_took(cavity_runs):
    r2, r4 = cavity_runs[2], cavity_runs[4]
    assert r2.dt == r2.grid.dt
    # (2,4) derating, documented on Simulation(stencil_order=...) as ~0.857x.
    assert r4.dt == pytest.approx(0.857 * r4.grid.dt, rel=1e-12, abs=0.0), (
        f"stencil_order=4 Result.dt={r4.dt:.6e} s, grid.dt={r4.grid.dt:.6e} s: "
        f"the Result must report the derated step the scan advanced by")


@pytest.mark.parametrize("order", [2, 4])
def test_cavity_resonance_is_read_at_the_right_frequency(cavity_runs, order):
    res = cavity_runs[order]
    modes = res.find_resonances(freq_range=(5e9, 13e9))
    assert modes, f"order {order}: Harminv found no mode in 5-13 GHz"
    strongest = max(modes, key=lambda m: m.amplitude)
    err = (strongest.freq - _F_TM110) / _F_TM110
    assert abs(err) < 0.01, (
        f"order {order}: TM110 read at {strongest.freq / 1e9:.4f} GHz, "
        f"analytic {_F_TM110 / 1e9:.4f} GHz ({100 * err:+.2f} %)")
    # The same record by FFT with Result.dt, as a user would read it.
    ts = np.asarray(res.time_series)[:, 0].astype(np.float64)
    n = len(ts)
    spec = np.abs(np.fft.rfft(ts * np.hanning(n), 16 * n))
    freqs = np.fft.rfftfreq(16 * n, res.dt)
    band = (freqs > 5e9) & (freqs < 13e9)
    f_fft = freqs[band][np.argmax(spec[band])]
    assert abs(f_fft - _F_TM110) / _F_TM110 < 0.01, (
        f"order {order}: FFT peak {f_fft / 1e9:.4f} GHz vs analytic "
        f"{_F_TM110 / 1e9:.4f} GHz")


def test_until_decay_result_dt_is_the_step_the_loop_took():
    res = _cavity(4).run(until_decay=1e-3, decay_check_interval=20,
                         decay_min_steps=40, decay_max_steps=120,
                         skip_preflight=True)
    assert res.dt == pytest.approx(0.857 * res.grid.dt, rel=1e-12, abs=0.0)

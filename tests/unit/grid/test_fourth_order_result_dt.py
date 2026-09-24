"""A result read in time or frequency must use the step the solver took.

``stencil_order=4`` advances the fields by ``0.857 * grid.dt`` (the (2,4)
stability bound), but ``Result.dt`` used to report ``grid.dt``. Everything a
user reads through ``Result.dt`` then came out scaled by 0.857: on a
20 x 20 x 10 mm PEC cavity the TM110 mode (10.599 GHz) was reported at
9.085 GHz by ``find_resonances`` and by an FFT of the probe record, -14.3 %.
The same held for ``forward()``, whose only step was ``grid.dt``, and for the
``dt`` that ``save_simulation_dataset`` writes beside the time series.

The physical witness here is a closed PEC box whose TM110 frequency is known
in closed form, c0/2 * sqrt(1/a^2 + 1/b^2); both stencil orders must put it
within 1 % through every route a user reads a frequency by. It does not share
a helper with the code under test: the only rfx inputs are the declared box
and the probe record.
"""
import math

import numpy as np
import pytest

from rfx import Simulation, SnapshotSpec
from rfx.io import save_simulation_dataset

_A, _B, _C = 0.020, 0.020, 0.005
_F_TM110 = 299792458.0 / 2 * math.sqrt(1 / _A ** 2 + 1 / _B ** 2)   # 10.5993 GHz
_N_STEPS = 2000


def _cavity(stencil_order):
    sim = Simulation(freq_max=15e9, domain=(_A, _B, _C), dx=1e-3,
                     boundary="pec", stencil_order=stencil_order)
    sim.add_source((0.006, 0.007, 0.002), component="ez")
    sim.add_probe((0.014, 0.012, 0.002), component="ez")
    return sim


def _snapshot(sim):
    kz = sim._build_grid().position_to_index((0.0, 0.0, 0.002))[2]
    return SnapshotSpec(interval=400, components=("ez", "hx"), slice_axis=2,
                        slice_index=kz)


def _fft_peak(time_series, dt):
    ts = np.asarray(time_series)[:, 0].astype(np.float64)
    n = len(ts)
    spec = np.abs(np.fft.rfft(ts * np.hanning(n), 16 * n))
    freqs = np.fft.rfftfreq(16 * n, dt)
    band = (freqs > 5e9) & (freqs < 13e9)
    return float(freqs[band][np.argmax(spec[band])])


def _assert_tm110(f, what):
    err = (f - _F_TM110) / _F_TM110
    assert abs(err) < 0.01, (
        f"{what}: TM110 read at {f / 1e9:.4f} GHz, analytic "
        f"{_F_TM110 / 1e9:.4f} GHz ({100 * err:+.2f} %)")


@pytest.fixture(scope="module")
def cavity_runs():
    out = {}
    for order in (2, 4):
        sim = _cavity(order)
        out[order] = (sim, sim.run(n_steps=_N_STEPS, skip_preflight=True,
                                   snapshot=_snapshot(sim)))
    return out


def test_result_dt_is_the_step_the_scan_took(cavity_runs):
    r2, r4 = cavity_runs[2][1], cavity_runs[4][1]
    assert r2.dt == r2.grid.dt
    # (2,4) derating, documented on Simulation(stencil_order=...) as ~0.857x.
    assert r4.dt == pytest.approx(0.857 * r4.grid.dt, rel=1e-12, abs=0.0), (
        f"stencil_order=4 Result.dt={r4.dt:.6e} s, grid.dt={r4.grid.dt:.6e} s: "
        f"the Result must report the derated step the scan advanced by")


@pytest.mark.parametrize("order", [2, 4])
def test_snapshot_times_use_the_step_the_scan_took(cavity_runs, order):
    res = cavity_runs[order][1]
    for comp, half in (("ez", 0.0), ("hx", 0.5)):
        axes = res.snapshot_axes[comp]
        assert axes.dt == res.dt, (
            f"order {order}: snapshot_axes[{comp!r}].dt={axes.dt:.6e} s, "
            f"Result.dt={res.dt:.6e} s")
        assert axes.dt == pytest.approx(
            (0.857 if order == 4 else 1.0) * res.grid.dt, rel=1e-12, abs=0.0)
        np.testing.assert_array_equal(axes.times_s,
                                      (axes.steps - half) * res.dt)


@pytest.mark.parametrize("order", [2, 4])
def test_cavity_resonance_is_read_at_the_right_frequency(cavity_runs, order):
    res = cavity_runs[order][1]
    modes = res.find_resonances(freq_range=(5e9, 13e9))
    assert modes, f"order {order}: Harminv found no mode in 5-13 GHz"
    strongest = max(modes, key=lambda m: m.amplitude)
    _assert_tm110(strongest.freq, f"order {order} find_resonances")
    # The same record by FFT with Result.dt, as a user would read it.
    _assert_tm110(_fft_peak(res.time_series, res.dt), f"order {order} FFT")


@pytest.mark.parametrize("order", [2, 4])
def test_forward_result_carries_the_step_the_scan_took(order):
    fr = _cavity(order).forward(n_steps=_N_STEPS, skip_preflight=True)
    assert fr.dt == pytest.approx(
        (0.857 if order == 4 else 1.0) * fr.grid.dt, rel=1e-12, abs=0.0), (
        f"order {order}: ForwardResult.dt={fr.dt}, grid.dt={fr.grid.dt:.6e} s")
    _assert_tm110(_fft_peak(fr.time_series, fr.dt), f"order {order} forward()")


@pytest.mark.parametrize("order", [2, 4])
def test_saved_dataset_dt_is_the_step_of_its_time_series(cavity_runs, order,
                                                         tmp_path):
    h5py = pytest.importorskip("h5py")
    sim, res = cavity_runs[order]
    path = tmp_path / "dataset.h5"
    save_simulation_dataset(path, sim, res)
    with h5py.File(path, "r") as f:
        dt_file = float(f["input"].attrs["dt"])
        ts_file = f["output"]["time_series"][:]
    assert dt_file == float(res.dt)
    _assert_tm110(_fft_peak(ts_file, dt_file), f"order {order} saved dataset")


def test_until_decay_result_dt_is_the_step_the_loop_took():
    sim = _cavity(4)
    res = sim.run(until_decay=1e-3, decay_check_interval=20,
                  decay_min_steps=40, decay_max_steps=800,
                  skip_preflight=True, snapshot=_snapshot(sim))
    assert res.dt == pytest.approx(0.857 * res.grid.dt, rel=1e-12, abs=0.0)
    assert res.snapshot_axes["ez"].dt == res.dt

"""B2/B2': scene P frequency and growth against an independent 0-D map."""
import json
import resource
import sys
import time

import jax
import numpy as np

from rfx import ringdown
from rfx.measurement.record_end import source_end_step
from tests.oracle._drude_plasma_scene import (
    FREQ_MAX, lorentz_scene, references, scene, waveform,
)


def test_drude_plasma_resonance():
    start = time.monotonic()
    nu = 1.063e7
    pulse = waveform()
    sim, grid, n_steps = scene(nu, "float32", 8, pulse)
    dt = float(grid.dt)
    # references() builds the 0-D map in NumPy, without rfx coefficient helpers.
    ref = references(nu, dt)
    assert ref["r"] == 2**-24
    off = source_end_step([(pulse, 0.)], n_steps, dt, tolerance=1e-6)
    assert off is not None and off < n_steps
    result = sim.run(n_steps=n_steps, compute_s_params=False)
    series = np.asarray(jax.block_until_ready(result.time_series))
    model = ringdown.identify(series, dt, off, n_steps, freq_max=FREQ_MAX)
    poles = [dict(frequency=float(s.imag / (2 * np.pi)), decay=float(-s.real))
             for s in model.s if 0 <= s.imag / (2 * np.pi) <= .9 * FREQ_MAX]
    assert poles, "B2: no in-band pole identified"
    nearest = min(poles, key=lambda row: abs(row["frequency"] - ref["f_cf"]))
    error = abs(nearest["frequency"] / ref["f_cf"] - 1)
    period = 1 / (ref["f_cf"] * dt)
    first = abs(series[off:int(np.ceil(off + period))]).max(axis=0)
    last = abs(series[int(np.ceil(n_steps - period)):]).max(axis=0)
    ratios = last / first
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(json.dumps(dict(bar="B2/B2'", nu=nu, **ref, relative_error=error,
                         poles=poles, rank=int(model.rank), ratios=ratios.tolist(),
                         wall_s=time.monotonic()-start, peak_rss_bytes=peak)), flush=True)
    assert error <= 1e-3
    assert np.all(ratios <= 1.01), ratios


def test_lorentz_resonance():
    start = time.monotonic()
    nu = 1.183e7
    # lorentz_scene() builds the reference in NumPy, without rfx coefficient helpers.
    sim, grid, n_steps, pulse, ref = lorentz_scene(nu)
    dt = float(grid.dt)
    assert abs(ref["r"]) >= 2**-24
    off = source_end_step([(pulse, 0.)], n_steps, dt, tolerance=1e-6)
    assert off is not None and off < n_steps
    result = sim.run(n_steps=n_steps, compute_s_params=False)
    series = np.asarray(jax.block_until_ready(result.time_series))
    model = ringdown.identify(series, dt, off, n_steps, freq_max=16e9)
    poles = [dict(frequency=float(s.imag / (2 * np.pi)), decay=float(-s.real))
             for s in model.s if 0 <= s.imag / (2 * np.pi) <= .9 * 16e9]
    assert poles, "B2: no in-band pole identified"
    nearest = min(poles, key=lambda row: abs(row["frequency"] - ref["f_cf"]))
    error = abs(nearest["frequency"] / ref["f_cf"] - 1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(json.dumps(dict(bar="B2", nu=nu, **ref, relative_error=error,
                         poles=poles, rank=int(model.rank),
                         wall_s=time.monotonic()-start, peak_rss_bytes=peak)), flush=True)
    assert error <= 1e-3

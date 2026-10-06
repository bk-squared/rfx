"""The waveguide port's modal current is stamped half a step before its voltage.

Two fast checks of the H stamp at the two places the waveguide port turns a
current record into a spectrum (S2 M2 review P2): the wave split of the
recorded modal V/I, and the overlap extractor's running sums. Each compares
the spectrum with a NumPy float64 DFT of the same samples, E at (n+1) dt and
H at (n+1/2) dt. Stamping the current as E moves it by omega dt / 2
(0.06 .. 0.19 rad at these bins), far outside the 1e-5 bar.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from rfx.sources import waveguide_port

DT = 2.0e-12
FREQS = np.array([10e9, 20e9, 30e9])
N = 50


def _host(samples, offset):
    n = np.arange(len(samples), dtype=np.float64)
    return (np.exp(-2j * np.pi * FREQS[:, None] * (n + offset) * DT)
            @ np.asarray(samples, dtype=np.float64)) * DT


def _close(got, want):
    np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-5 * np.max(np.abs(want)))


def test_the_wave_split_stamps_the_recorded_current_as_h(monkeypatch):
    rng = np.random.default_rng(11)
    voltage, current = (rng.normal(size=N).astype(np.float32) for _ in range(2))
    seen = {}
    monkeypatch.setattr(waveguide_port, "_extract_global_waves",
                        lambda cfg, v, i: seen.update(v=v, i=i) or (v, i))
    cfg = SimpleNamespace(freqs=jnp.asarray(FREQS, jnp.float32), dt=DT, n_steps_recorded=N)
    waveguide_port._extract_global_waves_from_time_series(
        cfg, jnp.asarray(voltage), jnp.asarray(current))
    freqs32 = np.asarray(cfg.freqs, dtype=np.float64)
    assert np.allclose(freqs32, FREQS, rtol=1e-6)
    _close(seen["v"], 2 * _host(voltage, 1.0))   # single-sided spectrum: 2 x
    _close(seen["i"], 2 * _host(current, 0.5))


def test_the_overlap_sums_stamp_the_h_channel_as_h(monkeypatch):
    rng = np.random.default_rng(12)
    p1, p2 = rng.normal(size=N), rng.normal(size=N)
    monkeypatch.setattr(
        waveguide_port, "_overlap_cross_products",
        lambda state, cfg, x, dx: (jnp.float32(p1[state.step - 1] * x),
                                   jnp.float32(p2[state.step - 1] * x)))
    cfg = SimpleNamespace(freqs=jnp.asarray(FREQS, jnp.float32), ref_x=1, probe_x=2)
    zero = jnp.zeros(len(FREQS), jnp.complex64)
    acc = waveguide_port.OverlapDFTAccumulators(zero, zero, zero, zero)
    for step in range(1, N + 1):  # the state after step n carries step == n + 1
        acc = waveguide_port.update_overlap_dft(acc, cfg, SimpleNamespace(step=step), DT, 1e-3)
    _close(acc.p1_ref_dft, _host(p1, 1.0))
    _close(acc.p2_ref_dft, _host(p2, 0.5))
    _close(acc.p1_probe_dft, 2 * _host(p1, 1.0))
    _close(acc.p2_probe_dft, 2 * _host(p2, 0.5))

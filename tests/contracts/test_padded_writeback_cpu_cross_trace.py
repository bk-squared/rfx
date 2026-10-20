"""Pin CPU padded write-back probe agreement across plain and jax.jit calls.
Before #1621 the record read 32 ULP at 120 steps (63 at 480).
Why #1621 removed the spread is not known; this catches a regression
even though the mechanism is not known (rfx-lead, 2026-10-21).
"""

import jax
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries import _cpml_slab

MM = 1e-3


@pytest.mark.parametrize("n_steps", (120, 240))
def test_padded_writeback_cpu_plain_vs_jit(monkeypatch, n_steps, record_property):
    monkeypatch.setattr(_cpml_slab, "writeback_form", lambda: "padded")

    def record():
        sim = Simulation(freq_max=16e9, domain=(16 * MM, 12 * MM, 8 * MM),
                         dx=MM, boundary="cpml", cpml_layers=4)
        sim.add_material("sub", eps_r=3.0)
        sim.add(Box((2 * MM, 2 * MM, 1 * MM), (14 * MM, 10 * MM, 3 * MM)),
                material="sub")
        sim.add(Box((2 * MM, 2 * MM, 0), (14 * MM, 10 * MM, 1 * MM)),
                material="pec")
        sim.add_port(position=(6 * MM, 6 * MM, 2 * MM), component="ez",
                     impedance=50.0, waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
        sim.add_probe((10 * MM, 6 * MM, 2 * MM), "ez")
        return sim.forward(
            n_steps=n_steps, skip_preflight=True, checkpoint=False,
            port_s11_freqs=np.array([6e9, 8e9], np.float32)).time_series

    try:
        with jax.default_device(jax.devices("cpu")[0]):
            plain = np.asarray(record(), dtype=np.float64)
            compiled = np.asarray(jax.jit(record)(), dtype=np.float64)
        assert plain.shape == compiled.shape == (n_steps, 1)
        peak = np.max(np.abs(plain))
        assert np.isfinite(peak) and peak > 0
        ulp_at_peak = float(np.max(np.abs(plain - compiled))
                            / np.spacing(np.float32(peak)))
        record_property("ulp_at_peak", ulp_at_peak)
        assert ulp_at_peak <= 9, f"{n_steps} steps: {ulp_at_peak:.6g} ULP at peak"
    finally:
        jax.clear_caches()

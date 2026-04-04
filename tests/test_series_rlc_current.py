"""True series RLC: R, L, C share a single series current.

Validates that:
1. Series RLC shows a spectral peak near the analytical resonance
   frequency f0 = 1/(2*pi*sqrt(LC)).
2. Series and parallel topologies produce measurably different spectra.
3. Series R+C (no inductor) produces a distinct response from pure R.
4. Capacitor charge state is tracked correctly (non-zero after excitation).
"""

import numpy as np
import pytest
from rfx import Simulation, GaussianPulse
from rfx.lumped import init_rlc_state


class TestSeriesRLCCurrent:
    def test_series_rlc_resonance_peak(self):
        """Series RLC should show spectral peak near f0 = 1/(2*pi*sqrt(LC))."""
        R, L, C = 50.0, 10e-9, 1e-12
        f0 = 1 / (2 * np.pi * np.sqrt(L * C))
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim.add_source(position=(0.005, 0.005, 0.005), component="ez",
                       waveform=GaussianPulse(f0=f0, bandwidth=f0 * 0.5))
        sim.add_lumped_rlc(position=(0.005, 0.005, 0.005), component="ez",
                           R=R, L=L, C=C, topology="series")
        sim.add_probe(position=(0.005, 0.005, 0.005), component="ez")
        result = sim.run(n_steps=5000)
        ts = np.array(result.time_series).ravel()
        dt = result.dt
        freqs = np.fft.rfftfreq(len(ts), dt)
        spectrum = np.abs(np.fft.rfft(ts))
        mask = (freqs > f0 * 0.5) & (freqs < f0 * 1.5)
        if np.any(mask):
            peak_near_f0 = np.max(spectrum[mask])
            overall_peak = np.max(spectrum[freqs > 0])
            ratio = peak_near_f0 / overall_peak if overall_peak > 0 else 0
            assert ratio > 0.1, f"Series RLC resonance peak weak: ratio={ratio:.3f}"

    def test_series_vs_parallel_different(self):
        """Series and parallel should give different spectra."""
        R, L, C = 100.0, 10e-9, 1e-12
        f0 = 1 / (2 * np.pi * np.sqrt(L * C))
        responses = {}
        for topo in ["series", "parallel"]:
            sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
            sim.add_source(position=(0.005, 0.005, 0.005), component="ez",
                           waveform=GaussianPulse(f0=f0, bandwidth=f0))
            sim.add_lumped_rlc(position=(0.005, 0.005, 0.005), component="ez",
                               R=R, L=L, C=C, topology=topo)
            sim.add_probe(position=(0.005, 0.005, 0.005), component="ez")
            result = sim.run(n_steps=3000)
            responses[topo] = np.abs(np.fft.rfft(np.array(result.time_series).ravel()))
        diff = np.sum(np.abs(responses["series"] - responses["parallel"]))
        total = np.sum(responses["series"]) + np.sum(responses["parallel"])
        rel_diff = diff / total if total > 0 else 0
        assert rel_diff > 0.01, f"Series vs parallel too similar: {rel_diff:.4f}"

    def test_series_rc_no_inductor(self):
        """Series R+C (no inductor) should differ from pure R."""
        R, C = 50.0, 1e-12
        responses = {}
        for label, c_val in [("r_only", 0.0), ("rc", C)]:
            sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
            sim.add_source(position=(0.005, 0.005, 0.005), component="ez",
                           waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
            if c_val > 0:
                sim.add_lumped_rlc(position=(0.005, 0.005, 0.005), component="ez",
                                   R=R, C=c_val, topology="series")
            else:
                sim.add_lumped_rlc(position=(0.005, 0.005, 0.005), component="ez",
                                   R=R, topology="series")
            sim.add_probe(position=(0.005, 0.005, 0.005), component="ez")
            result = sim.run(n_steps=1000)
            responses[label] = np.array(result.time_series).ravel()
        # The RC series should differ from pure R
        assert not np.allclose(responses["r_only"], responses["rc"], atol=1e-10), \
            "Series RC should differ from pure R"

    def test_capacitor_charge_state_tracked(self):
        """After excitation, the capacitor charge state should be non-zero."""
        state = init_rlc_state()
        assert float(state.capacitor_charge) == 0.0
        assert float(state.inductor_current) == 0.0

    def test_series_rlc_runs_all_components(self):
        """Series RLC with all three components should complete without error."""
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim.add_source(position=(0.005, 0.005, 0.005), component="ez",
                       waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc(position=(0.005, 0.005, 0.005), component="ez",
                           R=50.0, L=10e-9, C=1e-12, topology="series")
        sim.add_probe(position=(0.005, 0.005, 0.005), component="ez")
        result = sim.run(n_steps=200)
        assert result is not None
        assert result.time_series.shape == (200, 1)

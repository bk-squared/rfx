"""run_until_decay must support lumped RLC elements."""
import numpy as np
from rfx import Simulation, GaussianPulse


class TestDecayWithRLC:
    def test_decay_with_inductor(self):
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02), boundary="pec")
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez",
                       waveform=GaussianPulse(f0=2.5e9, bandwidth=2.5e9))
        sim.add_lumped_rlc(position=(0.01, 0.01, 0.01), component="ez",
                           R=50.0, L=10e-9, C=0, topology="series")
        sim.add_probe(position=(0.01, 0.01, 0.01), component="ez")
        result = sim.run(until_decay=1e-2, decay_max_steps=500)
        assert result is not None
        ts = np.array(result.time_series)
        assert len(ts) > 0
        assert np.max(np.abs(ts)) > 1e-15

    def test_decay_with_high_r_stops_early(self):
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02), boundary="pec")
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez",
                       waveform=GaussianPulse(f0=2.5e9, bandwidth=2.5e9))
        sim.add_lumped_rlc(position=(0.01, 0.01, 0.01), component="ez",
                           R=1000.0, L=0, C=0, topology="series")
        sim.add_probe(position=(0.01, 0.01, 0.01), component="ez")
        result = sim.run(until_decay=1e-2, decay_max_steps=2000)
        ts = np.array(result.time_series)
        assert len(ts) < 2000

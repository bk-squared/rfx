"""Declared synthetic closed-box scene shared by the oracle and record harness."""
import numpy as np
from rfx import GaussianPulse, Simulation
from rfx import ringdown as rd

SIGMA = 1.44e-5
RECORD_S = 7.62e-9
BINS = np.linspace(7e9, 10e9, 41)
PROBES = (((.0183, .0041, .0157), "ey"),
          ((.0059, .0092, .0194), "ey"))


def scene(dx=.001, resistance=10000.):
    sim = Simulation(freq_max=15e9, domain=(.0265, .0127, .0241),
                     dx=dx, boundary="pec", precision="float64", snap="declared")
    sim.add_port(position=(.0071, (.0127 - 3 * dx) / 2, .0063),
                 component="ey", extent=3 * dx, impedance=resistance,
                 waveform=GaussianPulse(f0=8.4e9, bandwidth=.8))
    grid = sim._build_grid()
    spec = rd.RingdownSpec(freq_max=15e9, identification_probes=PROBES)
    return sim, grid, spec


"""Always-on uniform and graded y-pair conductor-loss cavity oracles."""
import time

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.ringdown import identify


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_y_pair_cavity_q(graded):
    started = time.monotonic()
    kwargs = {"dy_profile": np.array([.00075, .00075] + [.001] * 9 + [.00075, .00075])} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.030, .012, .024), dx=.001 if graded else .0015,
                     boundary="pec", snap="declared", **kwargs)
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=("y_lo", "y_hi"))
    sim.add_source((.0073, .0041, .0059), "ey", waveform=GaussianPulse(f0=8e9, bandwidth=.25),
                   amplitude_kind="current")
    sim.add_probe((.0217, .0083, .0151), "ey")
    dt = (sim._build_nonuniform_grid() if graded else sim._build_grid()).dt
    result = sim.run(n_steps=int(np.ceil(20e-9/dt)), compute_s_params=False)
    series = np.asarray(result.time_series)
    model = identify(series, dt, int(np.ceil(2e-9/dt)), len(series), freq_max=10e9)
    pole = model.s[np.argmin(np.abs(model.s.imag/(2*np.pi)-7.998e9))]
    # No paired baseline is supplied in this short oracle: alpha_0 = 0.
    q = pole.imag / (-2*pole.real)
    error = q / 1623.8 - 1
    print(f"Mac wall_s={time.monotonic()-started:.6f}, Q={q:.9f}, relative_error={error:.9g}, "
          f"rank={model.rank}, invalid={model.n_invalid}, growing_discarded={model.n_growing_discarded}")
    assert abs(error) <= 5e-2

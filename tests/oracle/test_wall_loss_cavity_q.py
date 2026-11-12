"""Always-on uniform and graded x/y/z-pair conductor-loss cavity oracles."""
import json
import time

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.ringdown import identify


def target_q(pair):
    # Independent SI closed form for TE101, E0=1; Rs is frozen at f0=8 GHz.
    mu0 = 1.25663706212e-6
    eps0 = 8.8541878128e-12
    a, b, d, ell = .030, .012, .024, 1
    f = 7.99837e9
    rs = np.sqrt(np.pi * 8e9 * mu0 / 5.8e5)
    eta = np.sqrt(mu0 / eps0)
    wavelength = 1 / (f * np.sqrt(mu0 * eps0))
    energy = eps0 * a * b * d / 8
    factor = {"x": b*d/a**2, "y": ell**2*a/(2*d)+d/(2*a),
              "z": ell**2*a*b/d**2}[pair]
    power = rs * wavelength**2 / (8 * eta**2) * factor
    return float(2 * np.pi * f * energy / power)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("pair", ["x", "y", "z"])
def test_wall_pair_cavity_q(pair, graded):
    started = time.monotonic()
    kwargs = {"dy_profile": np.array([.00075, .00075] + [.001] * 9 + [.00075, .00075])} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.030, .012, .024), dx=.001 if graded else .0015,
                     boundary="pec", snap="declared", **kwargs)
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=(pair + "_lo", pair + "_hi"))
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
    target = target_q(pair)
    error = q / target - 1
    print(json.dumps(dict(pair=pair, lane="graded" if graded else "uniform",
                          wall_s=time.monotonic()-started, Q=float(q), target=target,
                          relative_error=float(error), rank=int(model.rank),
                          invalid=int(model.n_invalid), growing_discarded=int(model.n_growing_discarded))))
    assert abs(error) <= 5e-2

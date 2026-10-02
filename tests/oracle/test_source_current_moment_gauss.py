"""Gauss-law charge deposited by a default soft-source current moment (#1373)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.core.yee import EPS_0
from rfx.sources.sources import CustomWaveform


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("dx", [.002, .001])
def test_default_current_moment_deposits_charge(graded, dx):
    n = round(.03 / dx)
    # A genuinely graded mesh, symmetric about the driven edge's region.
    widths = dx * (1 - .1 * np.sin(np.linspace(0, np.pi, n))**2)
    profiles = dict(dx_profile=widths, dy_profile=widths, dz_profile=widths) if graded else {}
    sim = Simulation(freq_max=1e10, domain=(.03, .03, .03), dx=dx,
                     boundary="pec", **profiles)
    position = (.015, .015, .015)
    t0, tau = .3e-9, .08e-9
    waveform = CustomWaveform(func=lambda t: jnp.exp(-((t - t0) / tau)**2))
    with pytest.warns(DeprecationWarning, match="current moment"):
        sim.add_source(position, "ez", waveform=waveform)
    sim.add_probe(position, "ez")
    grid = sim._build_realized_grid()
    result = sim.run(n_steps=600, skip_preflight=True)
    ex, ey, ez = (np.asarray(getattr(result.state, c), dtype=np.float64)
                  for c in ("ex", "ey", "ez"))
    if graded:
        from rfx.nonuniform import position_to_index
        i, j, k = position_to_index(grid, position)
    else:
        i, j, k = grid.position_to_index(position)
    ux, uy, uz = (grid.duals(a) for a in ("x", "y", "z"))

    def charge(kk):
        return EPS_0 * (
            (ex[i, j, kk] - ex[i - 1, j, kk]) * uy[j] * uz[kk]
            + (ey[i, j, kk] - ey[i, j - 1, kk]) * ux[i] * uz[kk]
            + (ez[i, j, kk] - ez[i, j, kk - 1]) * ux[i] * uy[j])

    # Closed-form integral of the complete Gaussian; omitted tails < 6e-8.
    expected = tau * math.sqrt(math.pi) / grid.cells("z")[k]
    got = np.array([charge(k), -charge(k + 1)])
    residual = np.max(np.abs(got / expected - 1))
    print(f"Gauss graded={graded} dx={dx}: q={got}, expected={expected}, residual={residual}")
    # Measured max 7.73e-7 over these four float32 traces; 2e-6 allows
    # accumulation rounding over 600 steps, with no absolute-error floor.
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0)

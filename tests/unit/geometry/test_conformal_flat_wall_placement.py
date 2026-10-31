"""A conformal PEC wall a fraction of a cell past a node line rings where it is drawn (#1651).

A 20 x 14 x 10 mm PEC box at 1 mm cells has its +x wall replaced by a PEC Box from x0 = 18.25 or 18.75 mm. The TE101
mode's frequency gives the wall position through the discrete Yee relation
sin^2(w dt/2)/(c dt)^2 = sin^2(pi dx/(2L))/dx^2 + sin^2(pi dz/(2b))/dz^2 (the continuum relation carries the grid's own
dispersion, +0.13 cell at this mesh). Measured with the cut-cell operands (rfx-archive records/20261029-1651-sc-fix,
K1): within 0.0078 cell at 18.25 mm and 0.0004 cell at 18.75 mm; the staircase puts these walls at 18 and 19 mm, and the
previous eps/w model 0.95 and 2.0 cells deep.
"""

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.geometry.csg import Box
from rfx.ringdown import identify
from rfx.sources.sources import ModulatedGaussian

C0 = 299792458.0
DX = 1e-3
B = 0.010


class _FinitePulse(ModulatedGaussian):
    def __call__(self, t):
        return jnp.where(t < self.t0 + 6 * self.tau, super().__call__(t), 0.0)


def _wall_from_te101(x0):
    f_guess = C0 / 2 * math.sqrt(1 / x0 ** 2 + 1 / B ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=24e9, domain=(0.020, 0.014, 0.010), dx=DX, boundary="pec")
        sim.add(Box((x0, -0.01, -0.01), (0.03, 0.024, 0.020)), material="pec")
        sim.add_source((0.009, 0.007, 0.005), "ey", waveform=_FinitePulse(f0=f_guess, bandwidth=0.2, cutoff=5.0),
                       amplitude_kind="current")
        sim.add_probe((0.008, 0.007, 0.004), "ey")
        dt = float(sim._build_grid().dt)
        n = int(np.ceil(5e-9 / dt))
        res = sim.run(n_steps=n, conformal_pec=True, compute_s_params=False)
    series = np.asarray(res.time_series).reshape(n, -1)[:, 0]
    model = identify(series, dt, int(np.ceil(2.5e-9 / dt)), n, freq_max=24e9)
    f = min((p.f_hz for p in model.poles() if 0.75 * f_guess < p.f_hz < 1.25 * f_guess),
            key=lambda v: abs(v - f_guess))
    s = math.sin(math.pi * f * dt) ** 2 / (C0 * dt) ** 2
    sz = math.sin(math.pi * DX / (2 * B)) ** 2 / DX ** 2
    return math.pi * DX / (2 * math.asin(DX * math.sqrt(s - sz)))


@pytest.mark.parametrize("x0", [18.25e-3, 18.75e-3])
def test_conformal_wall_rings_at_its_drawn_plane(x0):
    assert abs(_wall_from_te101(x0) - x0) / DX <= 0.05

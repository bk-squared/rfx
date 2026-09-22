"""The Kottke occupancy lane (``RFX_PEC_OCC_KOTTKE=1``) reproduces a cavity's
TM110 with a binary PEC slab in it (#1197).

A PEC slab across the bottom of a PEC cube must leave TM110 where it is:
that mode's Ez is uniform along z, so the slab height cannot move it. The
plain occupancy lane gets 8.8305 GHz on this fixture (analytic 8.8327). The
occupancy-to-tensor builder used to write the conductor one cell past the
occupancy and read 8.5266 GHz; this pins the oracle on the tensor lane.
"""
from __future__ import annotations

import os

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import GaussianPulse, Simulation
from rfx.harminv import harminv

C0 = 299792458.0
L, DX, STEPS = 24e-3, 1e-3, 2400


def _strongest(ts, dt):
    modes = [m for m in harminv(ts[STEPS // 4:], dt, 3e9, 12e9) if m.Q > 30]
    return max(modes, key=lambda m: m.amplitude).freq


def test_binary_pec_slab_leaves_tm110_at_the_analytic_frequency(monkeypatch):
    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "1")
    sim = Simulation(freq_max=16e9, domain=(L, L, L), dx=DX, boundary="pec")
    sim.add_source((5.5e-3, 7.5e-3, 15.5e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=8e9, bandwidth=0.9))
    sim.add_probe((17.5e-3, 15.5e-3, 19.5e-3), "ez")
    shape = tuple(sim.forward(n_steps=2, skip_preflight=True).grid.shape)
    occ = np.zeros(shape, np.float32)
    occ[:, :, :6] = 1.0                                   # slab from z = 0 to 6 mm, edge on a cell boundary
    r = sim.forward(pec_occupancy_override=jnp.asarray(occ), n_steps=STEPS,
                    skip_preflight=True, checkpoint=False)
    f = _strongest(np.asarray(r.time_series)[:, 0], float(r.grid.dt))
    f_an = C0 / 2 * np.sqrt(2) / L
    assert abs(f - f_an) / f_an < 2e-3, (f, f_an)

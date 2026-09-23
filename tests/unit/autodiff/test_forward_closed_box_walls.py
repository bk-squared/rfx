"""``forward()`` on a closed PEC box applies the walls that ``run()`` applies.

A ``boundary="pec"`` grid has no absorber (``cpml_layers == 0``) but still
reports ``cpml_axes == "xyz"``; ``forward()`` derived its PEC axes as "the
axes without CPML" and so handed the scan ``pec_axes == ""`` — an open box.
Measured before the fix: the 24 mm cube's strongest mode read 5.091 GHz
through ``forward()`` and 8.831 GHz through ``run()`` (analytic TM110
8.833 GHz); ``rfx.simulation.run(pec_axes="")`` reproduces the 5.091.
Pinned as an invariant: the two entry points agree on the ring-down, and the
resonance is the cavity's.

Mutation that must turn this red: pass ``pec_axes_run = ""`` again for the
closed box.
"""
from __future__ import annotations

import numpy as np

from rfx import GaussianPulse, Simulation
from rfx.harminv import harminv

C0 = 299792458.0
L, DX, STEPS = 24e-3, 1e-3, 2400


def _sim():
    sim = Simulation(freq_max=16e9, domain=(L, L, L), dx=DX, boundary="pec")
    sim.add_source((8.5e-3, 8.5e-3, 12.5e-3), "ez",
                   waveform=GaussianPulse(f0=8.8e9, bandwidth=0.8))
    sim.add_probe((16.5e-3, 16.5e-3, 12.5e-3), "ez")
    return sim


def _strongest_mode(ts, dt):
    modes = [m for m in harminv(ts[STEPS // 4:], dt, 2e9, 14e9) if m.Q > 30]
    return max(modes, key=lambda m: m.amplitude).freq


def test_forward_and_run_agree_on_a_closed_pec_box_and_hit_the_analytic_tm110():
    sim = _sim()
    r_fwd = sim.forward(n_steps=STEPS, skip_preflight=True, checkpoint=False)
    r_run = sim.run(n_steps=STEPS, compute_s_params=False, skip_preflight=True)
    ts_fwd = np.asarray(r_fwd.time_series)[:, 0]
    ts_run = np.asarray(r_run.time_series)[:, 0]
    # The two entry points normalise the source differently, so the samples
    # are not compared; the cavity's resonance is (harminv resolves < 0.05 %).
    f_fwd = _strongest_mode(ts_fwd, float(r_fwd.grid.dt))
    f_run = _strongest_mode(ts_run, float(r_run.grid.dt))
    f_analytic = C0 / 2 * np.sqrt(2) / L
    assert abs(f_fwd - f_run) / f_run < 1e-4, (f_fwd, f_run)
    assert abs(f_fwd - f_analytic) / f_analytic < 5e-3, (f_fwd, f_analytic)

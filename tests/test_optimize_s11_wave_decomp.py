"""Issue #72 follow-up: optimize() must plumb `port_s11_freqs` through to
forward() so the wave-decomposition S11 objective is usable inside the
optimisation loop.

If `port_s11_freqs` is dropped silently the objective raises ValueError
("requires forward(...) to be called with port_s11_freqs=..."), so this
test serves both as a smoke check (one optimisation step runs end-to-end)
and as a guard against regressions on the kwarg plumbing.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box, DesignRegion, GaussianPulse
from rfx.optimize import optimize
from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp


def _build_sim():
    a, b, d = 0.05, 0.05, 0.025
    sim = Simulation(
        freq_max=5e9,
        domain=(a, b, d),
        dx=2.5e-3,
        boundary="pec",
    )
    sim.add_material("dra_init", eps_r=4.0)
    sim.add(Box((0.015, 0.015, 0.005), (0.035, 0.035, 0.020)),
            material="dra_init")
    sim.add_port(
        position=(a / 2, b / 2, d / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0),
    )
    return sim


def test_optimize_passes_port_s11_freqs_to_forward():
    """One optimisation step must succeed when wave-decomp objective is used."""
    sim = _build_sim()
    region = DesignRegion(
        corner_lo=(0.015, 0.015, 0.005),
        corner_hi=(0.035, 0.035, 0.020),
        eps_range=(1.0, 12.0),
    )
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    obj = minimize_s11_at_freq_wave_decomp(target_freq=3.0e9, port_idx=0)

    res = optimize(
        sim, region, obj,
        n_iters=1,
        lr=0.01,
        n_steps=200,
        port_s11_freqs=freqs,
        skip_preflight=True,
        verbose=False,
    )

    assert len(res.loss_history) == 1
    loss = float(res.loss_history[0])
    assert np.isfinite(loss), f"loss not finite: {loss}"
    # PEC cavity → |S11| ~ 1, so |S11|^2 should land in [0.5, 1.5] range.
    assert 0.3 <= loss <= 1.5, f"unexpected |S11|^2 loss: {loss}"

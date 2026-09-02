"""Issue #64: optimize() must work on non-uniform meshes.

Before #64, optimize() called sim._forward_from_materials() directly,
bypassing forward()'s uniform/NU dispatch. Any NU sim raised ValueError
immediately ("no uniform grid available"). After #64 the differentiable
pipeline for optimize() goes through sim.forward(), so dz_profile works.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.optimize import DesignRegion, optimize


def _build_nu_sim():
    """Small graded-z cavity — no PEC/NTFF to keep preflight silent."""
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9,
        domain=(0.012, 0.012, float(np.sum(dz))),
        dx=0.5e-3,
        dz_profile=dz,
        cpml_layers=4,
    )
    sim.add_source((0.006, 0.006, 0.001), "ez")
    sim.add_probe((0.006, 0.006, 0.003), "ez")
    return sim


def test_optimize_runs_on_nu_mesh():
    """One iteration of optimize() on a NU grid must finish + produce finite loss."""
    sim = _build_nu_sim()
    region = DesignRegion(
        corner_lo=(0.004, 0.004, 0.0015),
        corner_hi=(0.008, 0.008, 0.0025),
        eps_range=(1.0, 4.0),
    )

    def obj(result):
        return jnp.sum(result.time_series[:, 0] ** 2)

    result = optimize(
        sim, region, obj,
        n_iters=1, n_steps=20, verbose=False, skip_preflight=True,
    )
    assert len(result.loss_history) == 1
    loss = result.loss_history[0]
    assert np.isfinite(loss), f"NU optimize loss is not finite: {loss}"
    assert result.eps_design.shape[0] > 0
    assert result.eps_design.shape[2] > 0

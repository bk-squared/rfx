"""Regression tests for differentiable ``Simulation.forward()`` on the
non-uniform mesh path (GitHub issue #33).

Before this change, ``forward()`` raised ``ValueError`` whenever any of
``dx_profile`` / ``dy_profile`` / ``dz_profile`` was set, making NU
grids unusable for gradient-based optimisation. The forward path now
routes NU profiles through ``_forward_nonuniform_from_materials``,
which wraps ``run_nonuniform_path`` with ``eps_override`` /
``sigma_override`` / ``pec_mask_override`` support.

Tests:

1. **Smoke**: ``sim.forward(n_steps=100)`` with ``dz_profile`` set must
   return a finite ``ForwardResult`` (no ValueError).
2. **AD vs FD**: ``jax.grad(loss)(eps_override)`` on a single eps cell
   matches a centred finite-difference estimate to <2% relative error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


def _build_sim():
    """Small graded-z cavity (20×20×9 interior, CPML=4)."""
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3,
        dz_profile=dz,
        cpml_layers=4,
    )
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    return sim


def test_forward_nonuniform_smoke():
    """``forward(n_steps=...)`` on a NU-z sim must return finite data."""
    sim = _build_sim()
    fr = sim.forward(n_steps=100)
    ts = np.asarray(fr.time_series)
    assert ts.shape[0] == 100
    assert np.all(np.isfinite(ts)), "NU forward produced NaN/Inf"
    # Sanity: source excites something within 100 steps.
    assert float(np.max(np.abs(ts))) > 0.0


def test_forward_nonuniform_grad_eps_matches_fd():
    """AD grad w.r.t. a single eps cell agrees with centred FD (<2% rel err).

    We scale ``eps_override`` by a scalar ``alpha`` and differentiate a
    squared-L2 probe loss w.r.t. ``alpha``. This reduces the grad check
    to a 1-D comparison that's robust to step-size / noise.
    """
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)

    # Target cell for the eps perturbation — well away from CPML and
    # source/probe cells.
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2
    n_steps = 60

    def loss(alpha):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        fr = sim.forward(eps_override=eps, n_steps=n_steps)
        return jnp.sum(fr.time_series ** 2)

    alpha0 = jnp.float32(2.0)
    grad_ad = float(jax.grad(loss)(alpha0))

    h = 1e-2
    lp = float(loss(alpha0 + h))
    lm = float(loss(alpha0 - h))
    grad_fd = (lp - lm) / (2.0 * h)

    rel_err = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
    assert rel_err < 0.02, (
        f"AD grad {grad_ad:.4e} vs FD grad {grad_fd:.4e} — "
        f"rel_err {rel_err:.4%} above 2% threshold"
    )


# --- Known gaps (not yet plumbed into the NU forward path) ---------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "pec_occupancy_override (soft-PEC continuous occupancy) is not "
        "yet supported on the NU forward path; run_nonuniform has no "
        "occupancy field plumbed through its scan body. Hard PEC via "
        "pec_mask_override works. Flipping this to XPASS means someone "
        "wired occupancy through the NU scan — update the API to accept."
    ),
)
def test_forward_nonuniform_pec_occupancy_unsupported():
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    occ = jnp.zeros(g.shape, dtype=jnp.float32)
    # Expected to raise ValueError in api.forward().
    sim.forward(pec_occupancy_override=occ, n_steps=20)

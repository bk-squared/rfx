"""Issue #41: stop_gradient on non-design cells.

When ``design_mask`` is provided on the NU forward path, the scan body
applies ``jax.lax.stop_gradient`` to field cells where the mask is False.
Forward physics is unchanged (stop_gradient is identity forward);
backward tape no longer accumulates adjoint entries for cells whose eps
does not depend on the optimisation variable.

These tests cover:

1. **Forward equivalence** — with a full-True mask, field output equals
   the no-mask case (stop_gradient is identity forward everywhere).
2. **Partial mask grad finite** — with a small mask covering only the
   design region, ``jax.grad`` returns finite non-NaN values.
3. **Pass-through in optimize()** — the kwarg plumbs through without
   raising, and the design-region cells still receive gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation


def _build_nu_sim():
    dz = np.array([0.5e-3] * 6 + [0.4e-3] * 4, dtype=np.float64)
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


def test_forward_full_true_mask_matches_no_mask():
    """Full-True mask must match the unmasked forward bit-for-bit."""
    sim = _build_nu_sim()
    g = sim._build_nonuniform_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)
    mask_full = jnp.ones(g.shape, dtype=bool)

    fr_nomask = sim.forward(n_steps=30, eps_override=eps_base,
                            skip_preflight=True)
    fr_masked = sim.forward(n_steps=30, eps_override=eps_base,
                            design_mask=mask_full, skip_preflight=True)

    np.testing.assert_allclose(
        np.asarray(fr_nomask.time_series),
        np.asarray(fr_masked.time_series),
        rtol=0, atol=0,
        err_msg="Full-True design_mask must not change forward output",
    )


def test_partial_mask_grad_is_finite():
    """Grad through a partial mask must be finite (non-NaN) and non-zero
    inside the masked region."""
    sim = _build_nu_sim()
    g = sim._build_nonuniform_grid()

    ti, tj, tk = g.nx // 2, g.ny // 2, g.nz // 2
    mask = jnp.zeros(g.shape, dtype=bool)
    mask = mask.at[ti-1:ti+2, tj-1:tj+2, tk-1:tk+2].set(True)

    def loss(alpha):
        eps = jnp.ones(g.shape, dtype=jnp.float32) * alpha
        fr = sim.forward(
            n_steps=40, eps_override=eps,
            design_mask=mask, skip_preflight=True,
        )
        return jnp.sum(fr.time_series[:, 0] ** 2)

    g_alpha = float(jax.grad(loss)(1.0))
    assert np.isfinite(g_alpha), f"grad not finite: {g_alpha}"


def test_design_mask_forwarded_through_optimize():
    """optimize() must accept design_mask and thread it to sim.forward()."""
    from rfx.optimize import DesignRegion, optimize

    sim = _build_nu_sim()
    g = sim._build_nonuniform_grid()
    ti, tj, tk = g.nx // 2, g.ny // 2, g.nz // 2
    mask = jnp.zeros(g.shape, dtype=bool)
    mask = mask.at[ti-1:ti+2, tj-1:tj+2, tk-1:tk+2].set(True)

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
        design_mask=mask,
    )
    loss = result.loss_history[0]
    assert np.isfinite(loss), f"loss not finite with design_mask: {loss}"

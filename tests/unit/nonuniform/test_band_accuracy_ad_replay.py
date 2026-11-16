"""Live AD1 gradient on the band-builder mesh, with the declared fast profile.

The two largest non-tied cells are compared with central finite differences
at the unchanged 15 % tolerance and 40-step budget. The declaration is in
``docs/design_notes/20260907_nu_band_accuracy_ad_predeclaration.md``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from validation.research.multiband_nu import w7_accuracy_ad as w7

LIVE_AD_N_STEPS = 40
LIVE_AD_TOL = 0.15


# =============================================================================
# JSON-free
# =============================================================================


def test_live_ad1_fast_profile_gradient():
    """AD1 on the s = 2 MB vector, N_STEPS = 40: finite gradient; the two
    largest non-tied cells agree with central FD to 15 % with sign."""
    dz0 = w7.a1_mb(2.0)
    eps_col = np.asarray(w7.f32_values(w7.AD1_EPS_PATTERN))
    loss = jax.jit(w7.ad1_loss_factory(eps_col, LIVE_AD_N_STEPS))
    grad = jax.jit(jax.grad(w7.ad1_loss_factory(eps_col, LIVE_AD_N_STEPS)))
    g_ad = np.asarray(grad(jnp.asarray(dz0)), np.float64)
    assert g_ad.shape == dz0.shape and np.all(np.isfinite(g_ad))
    d32 = w7.f32_values(dz0)
    tied = d32 == d32.min()
    assert tied.sum() == 2, "the two 1.0 mm cells of the thin band are the tied minimum"
    order = [k for k in np.argsort(-np.abs(g_ad)) if not tied[k]][:2]
    for k in order:
        h = w7.FD_REL_H * dz0[k]
        dp = dz0.copy()
        dp[k] += h
        dm = dz0.copy()
        dm[k] -= h
        g_fd = (float(loss(jnp.asarray(dp))) - float(loss(jnp.asarray(dm)))) / (2 * h)
        assert np.sign(g_fd) == np.sign(g_ad[k]), (k, g_ad[k], g_fd)
        rel = abs(g_ad[k] - g_fd) / abs(g_fd)
        assert rel <= LIVE_AD_TOL, (k, g_ad[k], g_fd, rel)

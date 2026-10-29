from __future__ import annotations

import jax.numpy as jnp


def _make_mats(shape):
    """Create vacuum MaterialArrays."""
    from rfx.core.yee import MaterialArrays
    return MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
    )


def _update_h_only(ex, ey, ez, hx, hy, hz, dt, dx, mats=None, *, high_wall):
    """H half-step only (Faraday)."""
    from rfx.core.yee import FDTDState
    from rfx.stepping import update_h
    shape = ex.shape
    state = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
                      step=jnp.array(0, dtype=jnp.int32))
    if mats is None:
        mats = _make_mats(shape)
    state = update_h(state, mats, dt, dx, high_wall=high_wall)
    return state.hx, state.hy, state.hz


def _update_e_only(ex, ey, ez, hx, hy, hz, dt, dx, mats=None,
                   pec_mask=None, boundary_pec=True, *, high_wall):
    """E full-step only (Ampere) + PEC."""
    from rfx.core.yee import FDTDState
    from rfx.stepping import update_e
    from rfx.boundaries.pec import apply_pec, apply_pec_mask
    shape = ex.shape
    state = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
                      step=jnp.array(0, dtype=jnp.int32))
    if mats is None:
        mats = _make_mats(shape)
    state = update_e(state, mats, dt, dx, high_wall=high_wall)
    if pec_mask is not None:
        # #931: ``apply_pec_mask`` IS the shared realization
        # (realized_pec_edge_masks, volume branch). This lane
        # carries VOLUMES only — run() refuses a declared sheet
        # or wire before it gets here (rfx/api/_execute.py) — and
        # has no periodic branch, so the default #689 flags are
        # the run's own.
        state = apply_pec_mask(state, pec_mask)
    if boundary_pec:
        state = apply_pec(state)
    return state.ex, state.ey, state.ez



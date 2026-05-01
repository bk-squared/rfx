"""Perfect Electric Conductor (PEC) boundary condition.

Zeros tangential E-field at boundary faces.
"""

from __future__ import annotations

import jax.numpy as jnp


def apply_pec(state, axes: str = "xyz") -> object:
    """Apply PEC (E_tan = 0) at domain boundaries.

    Parameters
    ----------
    state : FDTDState
    axes : str
        Which axes to apply PEC on. Default "xyz" = all 6 faces.
    """
    ex, ey, ez = state.ex, state.ey, state.ez

    if "x" in axes:
        # PEC at x=0 and x=end: Ey, Ez tangential → 0
        ey = ey.at[0, :, :].set(0.0)
        ey = ey.at[-1, :, :].set(0.0)
        ez = ez.at[0, :, :].set(0.0)
        ez = ez.at[-1, :, :].set(0.0)

    if "y" in axes:
        # PEC at y=0 and y=end: Ex, Ez tangential → 0
        ex = ex.at[:, 0, :].set(0.0)
        ex = ex.at[:, -1, :].set(0.0)
        ez = ez.at[:, 0, :].set(0.0)
        ez = ez.at[:, -1, :].set(0.0)

    if "z" in axes:
        # PEC at z=0 and z=end: Ex, Ey tangential → 0
        ex = ex.at[:, :, 0].set(0.0)
        ex = ex.at[:, :, -1].set(0.0)
        ey = ey.at[:, :, 0].set(0.0)
        ey = ey.at[:, :, -1].set(0.0)

    return state._replace(ex=ex, ey=ey, ez=ez)


def apply_pec_faces(state, faces: set[str]) -> object:
    """Apply PEC (E_tan = 0) on specific boundary faces.

    Parameters
    ----------
    state : FDTDState
    faces : set of str
        Which faces to enforce PEC on.  Valid names:
        ``"x_lo"``, ``"x_hi"``, ``"y_lo"``, ``"y_hi"``,
        ``"z_lo"``, ``"z_hi"``.
    """
    if not faces:
        return state
    ex, ey, ez = state.ex, state.ey, state.ez

    if "x_lo" in faces:
        ey = ey.at[0, :, :].set(0.0)
        ez = ez.at[0, :, :].set(0.0)
    if "x_hi" in faces:
        ey = ey.at[-1, :, :].set(0.0)
        ez = ez.at[-1, :, :].set(0.0)
    if "y_lo" in faces:
        ex = ex.at[:, 0, :].set(0.0)
        ez = ez.at[:, 0, :].set(0.0)
    if "y_hi" in faces:
        ex = ex.at[:, -1, :].set(0.0)
        ez = ez.at[:, -1, :].set(0.0)
    if "z_lo" in faces:
        ex = ex.at[:, :, 0].set(0.0)
        ey = ey.at[:, :, 0].set(0.0)
    if "z_hi" in faces:
        ex = ex.at[:, :, -1].set(0.0)
        ey = ey.at[:, :, -1].set(0.0)

    return state._replace(ex=ex, ey=ey, ez=ez)


def apply_pec_mask(state, pec_mask) -> object:
    """Zero tangential E-field components at PEC geometry cells.

    For thin PEC sheets (1 cell thick), only tangential E-components
    are zeroed; the normal component is preserved (represents surface
    charge). A component is tangential if the PEC extends >= 2 cells
    in that component's direction (i.e., has a PEC neighbor).

    Parameters
    ----------
    state : FDTDState
    pec_mask : (nx, ny, nz) boolean array
        True where material is PEC.
    """
    # Per-component masks: zero E only where PEC has extent in that direction
    # Ex(i,j,k) zeroed if pec(i,j,k) AND neighbor PEC in x
    # (if no x-neighbor is PEC → thin x-sheet → Ex is normal → preserve)
    mask_ex = pec_mask & (
        jnp.roll(pec_mask, 1, axis=0) | jnp.roll(pec_mask, -1, axis=0))
    mask_ey = pec_mask & (
        jnp.roll(pec_mask, 1, axis=1) | jnp.roll(pec_mask, -1, axis=1))
    mask_ez = pec_mask & (
        jnp.roll(pec_mask, 1, axis=2) | jnp.roll(pec_mask, -1, axis=2))

    return state._replace(
        ex=state.ex * (1.0 - mask_ex.astype(state.ex.dtype)),
        ey=state.ey * (1.0 - mask_ey.astype(state.ey.dtype)),
        ez=state.ez * (1.0 - mask_ez.astype(state.ez.dtype)),
    )


def apply_pec_h_mask(state, pec_mask=None, *,
                     mask_hx=None, mask_hy=None, mask_hz=None) -> object:
    """Zero H-field components inside PEC cells.

    Stage 2 unified-path companion to ``apply_pec_mask``. The Stage 2
    inverse-permittivity tensor freezes E inside PEC (inv=0 → Ca=1,
    Cb=0) but does NOT damp H, which propagates freely via 1/μ. Over
    many periods (≥30·τ at typical RF parameters), this seeds late-
    time growth and float32 NaN. Stage 1's ``sigma=1e10`` fold
    provided implicit damping for both E and H via the
    sigma-coupled curl interaction; Stage 2 needs explicit H zeroing.

    Two modes:

    1. **Single cell-center mask** (``pec_mask``): zero all three H
       components at any cell where ``pec_mask`` is True. Use this
       when the "fully PEC" set has been pre-computed (all three
       Yee-staggered E components frozen at this cell index).

    2. **Per-component mask** (``mask_hx`` / ``mask_hy`` / ``mask_hz``,
       Stage 2 step B-v2): zero each H component independently
       according to its *driver* E components in the Yee curl. ``Hx``
       at (i, j+½, k+½) is updated by ``∂Ez/∂y - ∂Ey/∂z``; if both
       Ey (at index inv_yy) and Ez (at index inv_zz) are frozen at
       this cell, Hx has no driver → safe to zero. This catches
       boundary cells where one E component (perpendicular to the
       wall) has fractional inv but the two tangential components
       are zero — the mode that the all-zero ``pec_mask`` misses.

    Pass either ``pec_mask`` or all three of ``mask_h*``; a mix is
    accepted and the masks are OR'd.

    Parameters
    ----------
    state : FDTDState
    pec_mask : (nx, ny, nz) boolean array, optional
        True where the cell-center is inside a fully-PEC region.
    mask_hx, mask_hy, mask_hz : (nx, ny, nz) boolean arrays, optional
        Per-component zero masks (Yee-stagger aware). Stage 2 step
        B-v2 derives these from ``(inv_xx==0, inv_yy==0, inv_zz==0)``
        pairwise combinations.
    """
    dtype = state.hx.dtype
    # Build per-component boolean masks (default: nothing zeroed).
    zero_hx = mask_hx
    zero_hy = mask_hy
    zero_hz = mask_hz
    if pec_mask is not None:
        zero_hx = pec_mask if zero_hx is None else (zero_hx | pec_mask)
        zero_hy = pec_mask if zero_hy is None else (zero_hy | pec_mask)
        zero_hz = pec_mask if zero_hz is None else (zero_hz | pec_mask)
    keep_hx = (1.0 - zero_hx.astype(dtype)) if zero_hx is not None else 1.0
    keep_hy = (1.0 - zero_hy.astype(dtype)) if zero_hy is not None else 1.0
    keep_hz = (1.0 - zero_hz.astype(dtype)) if zero_hz is not None else 1.0
    return state._replace(
        hx=state.hx * keep_hx,
        hy=state.hy * keep_hy,
        hz=state.hz * keep_hz,
    )


def apply_pec_occupancy(state, pec_occupancy) -> object:
    """Apply a relaxed PEC occupancy field to tangential E components.

    This is the differentiable analogue of :func:`apply_pec_mask`.
    ``pec_occupancy`` is a float field in ``[0, 1]`` where 0 means no
    conductor and 1 means full PEC occupancy. For binary occupancy it
    reproduces the hard-mask behaviour.
    """
    occ = jnp.clip(pec_occupancy.astype(state.ex.dtype), 0.0, 1.0)

    occ_ex = occ * jnp.maximum(jnp.roll(occ, 1, axis=0), jnp.roll(occ, -1, axis=0))
    occ_ey = occ * jnp.maximum(jnp.roll(occ, 1, axis=1), jnp.roll(occ, -1, axis=1))
    occ_ez = occ * jnp.maximum(jnp.roll(occ, 1, axis=2), jnp.roll(occ, -1, axis=2))

    return state._replace(
        ex=state.ex * (1.0 - occ_ex),
        ey=state.ey * (1.0 - occ_ey),
        ez=state.ez * (1.0 - occ_ez),
    )

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

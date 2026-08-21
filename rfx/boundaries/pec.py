"""Perfect Electric Conductor (PEC) boundary condition.

Zeros tangential E-field at boundary faces.
"""

from __future__ import annotations

import jax.numpy as jnp

from rfx.core.yee import _shift_bwd, _shift_fwd


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
        # Ez at k=nz-1 is a ghost cell outside the physical domain.
        ez = ez.at[:, :, -1].set(0.0)

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
        # Ez at k=nz-1 is a ghost cell at z=(nz-0.5)*dx, outside the
        # physical domain. Zero it to prevent ghost-layer accumulation.
        ez = ez.at[:, :, -1].set(0.0)

    return state._replace(ex=ex, ey=ey, ez=ez)


def tangential_edge_masks(cell_mask, periodic=(False, False, False)):
    """Per-component tangential E-edge masks for a boolean CELL mask.

    The single source of the thin-sheet neighbor rule (#677): a component
    is tangential to the masked body iff the body extends >= 2 cells in
    that component's direction (i.e. has a masked neighbor along it). A
    one-cell-thick sheet therefore selects only its in-plane (tangential)
    E components.

    Boundary convention (#689). The neighbour lookup used to be
    ``jnp.roll`` on every axis, which wraps: on a NON-periodic axis, a
    one-cell body on the ``0`` face and another on the ``n-1`` face saw
    each other through the domain and BOTH had their sheet-normal
    component selected. Whether a component is tangential or normal to a
    body is a property of the body, so translating the same pair one cell
    inward must not change the answer — and it did (measured, two 4x4
    plates in a (6,6,10) domain: per-component nnz [32, 32, 32] on the
    faces versus [32, 32, 0] anywhere inside). Non-periodic axes now use
    the explicit zero pad ``rfx.core.yee._shift_bwd/_shift_fwd``, which is
    also the convention the solver's own curl uses for out-of-domain H.

    The wrap is kept, deliberately, on two kinds of axis — both measured,
    neither assumed:

    * ``periodic[ax]`` — cell ``0`` and cell ``n-1`` really are
      neighbours there, so a body straddling the seam is contiguous.
      A seam-straddling body one cell either side goes from 8 selected
      edges to 0 under an unconditional zero pad. Callers on a periodic
      lane MUST pass their run's flags; the default is the non-periodic
      convention.
    * ``cell_mask.shape[ax] == 1`` — the 2-D lane. ``rfx/simulation.py``
      forces ``periodic[2] = True`` when ``grid.is_2d``, and with
      ``nz == 1`` the wrap is what makes a body self-adjacent along z.
      An unconditional zero pad selects zero Ez edges there, i.e. every
      2-D run with interior PEC silently loses that PEC (measured
      max|Ez| inside the block 0.0 -> 2.60e+06).

    Shared by :func:`apply_pec_mask` (which zeroes the selected edges) and
    the surface-impedance sheet operator
    (:func:`rfx.materials.thin_conductor.apply_sheet_impedance_e`, which
    applies a resistive update on them), so the PEC and sheet footprints
    are structurally identical — pinned by
    tests/test_sheet_impedance_operator.py (G4 footprint identity). Both
    consumers must be handed the SAME ``periodic``, or that identity is
    computed against two different neighbour rules.

    The distributed NU lane's hard-PEC kernel
    (``rfx/runners/distributed_nu.py::_apply_pec_mask_nu_shmap``) also
    CALLS this function, with ``periodic=(True, False, False)`` — its
    sharded x axis keeps the wrap because a slab's ghost rows carry the
    seam neighbour and are forced ``False`` afterwards. It used to carry
    an inlined ``roll`` copy of the rule, did not follow #689, and so
    disagreed with the single-device lane at a y or z domain face
    (measured, two 4x4 plates in a (6,6,10) domain: ``[32, 32, 32]``
    there against ``[32, 32, 0]`` here). Pinned by
    ``tests/test_distributed_nu_pec_mask_lane_parity.py``.

    Still NOT moved in lockstep, and worth knowing before the next reader
    rediscovers it: the SOFT PEC path — :func:`apply_pec_occupancy` below,
    its distributed twin ``_apply_pec_occupancy_nu_shmap``, and the
    AD-smooth dilation in ``rfx/geometry/smoothing.py`` — still spells the
    rule with ``roll``, so hard and soft PEC differ at a non-periodic
    domain face. Widening #689 into the differentiable-geometry lane was
    out of its scope; ``tests/test_topology.py`` does not discriminate
    either way (its mask spans the full y/z extent, where wrap and zero
    pad agree).

    Parameters
    ----------
    cell_mask : (nx, ny, nz) boolean array
    periodic : (bool, bool, bool)
        Per-axis periodic-boundary flags for the run. Defaults to the
        non-periodic convention.

    Returns
    -------
    (mask_ex, mask_ey, mask_ez) boolean arrays, same shape.
    """
    masks = []
    for ax in range(3):
        if cell_mask.shape[ax] == 1 or periodic[ax]:
            neighbor = (jnp.roll(cell_mask, 1, axis=ax)
                        | jnp.roll(cell_mask, -1, axis=ax))
        else:
            neighbor = _shift_bwd(cell_mask, ax) | _shift_fwd(cell_mask, ax)
        masks.append(cell_mask & neighbor)
    return tuple(masks)


def apply_pec_mask(state, pec_mask, periodic=(False, False, False)) -> object:
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
    periodic : (bool, bool, bool)
        The run's per-axis periodic flags, forwarded to
        :func:`tangential_edge_masks`. The default is the non-periodic
        convention, so a caller on a periodic lane must pass its own
        flags — see that function's boundary-convention note (#689).
        A length-1 axis is handled without it.
    """
    # Per-component masks: zero E only where PEC has extent in that direction
    # Ex(i,j,k) zeroed if pec(i,j,k) AND neighbor PEC in x
    # (if no x-neighbor is PEC → thin x-sheet → Ex is normal → preserve)
    mask_ex, mask_ey, mask_ez = tangential_edge_masks(pec_mask, periodic)

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

"""Per-step conformal PEC and resistive-sheet electric-field operators."""

import jax.numpy as jnp


def apply_conformal_pec(state, pec_mask_ex, pec_mask_ey, pec_mask_ez):
    """Apply conformal PEC: zero E inside PEC, leave partial cells to update coefficients.

    For cells with w=0 (fully inside PEC): force E=0.
    For cells with 0<w<1 (partial PEC): do NOT multiply E by w every step.
    The Dey-Mittra correction for partial cells should be applied by modifying
    the material eps_r via `conformal_eps_correction()` once at setup time.

    The boolean w == 0 masks are formed by core_step_invariants in
    uniform_setup before the compiled loop. Call AFTER update_e each timestep.
    Non-boolean masks raise TypeError: float weights given here would zero
    every open edge and leave the PEC interior; the weight-taking form is
    ``rfx.geometry.conformal.apply_conformal_pec``.
    """
    for name, mask in (("pec_mask_ex", pec_mask_ex), ("pec_mask_ey", pec_mask_ey),
                       ("pec_mask_ez", pec_mask_ez)):
        dtype = getattr(mask, "dtype", None)
        if dtype is None or not jnp.issubdtype(dtype, jnp.bool_):
            raise TypeError(
                f"{name} must be a boolean w == 0 mask, got dtype {dtype}; "
                "rfx.geometry.conformal.apply_conformal_pec takes the weights")
    return state._replace(
        ex=jnp.where(pec_mask_ex, 0.0, state.ex),
        ey=jnp.where(pec_mask_ey, 0.0, state.ey),
        ez=jnp.where(pec_mask_ez, 0.0, state.ez),
    )


def apply_sheet_impedance_e(state, e_prev, curls, ctx, coeffs):
    """Apply the node-thin resistive-sheet E update at tangential edges.

    Runs AFTER ``apply_pec_mask``/``apply_pec_occupancy`` and BEFORE domain
    PEC face writes, lumped updates, S-parameter DFT sampling and source
    injection. The following face writes enforce PEC wins on overlap at
    domain walls too. At every masked
    tangential edge the standard E-update result is REPLACED by::

        E^{n+1} = A * E^n + B * curlH^{n+1/2}

    with ``curls`` computed by the SAME stencil helpers the kernels use
    (``rfx.core.yee.curl_h`` / ``curl_h_nu``) from the same H the E-update
    consumed. Everywhere off-mask the state passes through untouched; the
    sheet-normal edge is never touched (its mask is all-False for a
    one-layer sheet).

    Parameters
    ----------
    state : FDTDState after the standard E update + boundary operators.
    e_prev : (ex, ey, ez) tuple — the E fields BEFORE the E update (E^n).
    curls : (curl_x, curl_y, curl_z) from the shared curl helper.
    ctx : SheetImpedanceCtx
    coeffs : (A, B) from :func:`sheet_update_coeffs`.
    """
    a, b = coeffs
    end = getattr(coeffs, "end", None)
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)

    def _component(c, mask, end_mask, field):
        new = a * e_prev[c].astype(_cdtype) + b * curls[c]
        if end is not None and end_mask is not None:
            new = jnp.where(
                end_mask, end[0] * e_prev[c].astype(_cdtype) + end[1] * curls[c], new)
        return jnp.where(mask, new.astype(_fdtype), field)

    ex = _component(0, ctx.mask_ex, getattr(ctx, "end_ex", None), state.ex)
    ey = _component(1, ctx.mask_ey, getattr(ctx, "end_ey", None), state.ey)
    ez = _component(2, ctx.mask_ez, getattr(ctx, "end_ez", None), state.ez)
    return state._replace(ex=ex, ey=ey, ez=ez)

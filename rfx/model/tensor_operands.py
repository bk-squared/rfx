"""Form uniform-run and waveguide permittivity operands in array order."""

import jax.numpy as jnp

from rfx.core.yee import add_lumped_eps
from rfx.geometry import conformal as conformal_module


def form_tensor_operands(sim, grid, materials, *, subpixel_smoothing,
                         conformal, conformal_min_weight, pec_shapes,
                         zero_inverse_on=None):
    """Form geometry operands; finish eps after final source/material setup.

    The caller retains its warning policy and PEC-mask handling. Only the
    waveguide device supplies ``zero_inverse_on`` for interior PEC cells.
    """
    # Stage 2 unified path: subpixel_smoothing="kottke_pec" replaces
    # the Stage 1 (Kottke dielectric + Dey-Mittra weights + per-step
    # apply_conformal_pec) chain with a single inverse-permittivity
    # tensor that encodes both dielectric subpixel smoothing and PEC
    # behaviour (inv = 0 freezes the field).
    _unextendable = []
    aniso_eps = None
    aniso_inv_eps = None
    use_kottke_pec = (subpixel_smoothing == "kottke_pec")
    if use_kottke_pec:
        from rfx.geometry.smoothing import (
            compute_inv_eps_tensor_diag, smoothed_shape_pairs, warn_curved_high_contrast,
        )
        # #1043 stage B: the pairs carry the CPML/UPML pad continuation, so a
        # dielectric that touches the domain edge is solved with the material
        # in its own pad instead of a vacuum facet at the seam. PEC shapes
        # arrive continued from assembly, with the MSL entry exception.
        shape_eps_pairs, _unextendable = smoothed_shape_pairs(sim, grid)
        warn_curved_high_contrast(shape_eps_pairs, sim=sim)
        aniso_inv_eps = compute_inv_eps_tensor_diag(
            grid,
            dielectric_shapes=shape_eps_pairs,
            pec_shapes=pec_shapes or [],
            background_eps=1.0,
        )
        # Yee-stagger correction: the Kottke union reaches inv=0
        # on Yee-staggered components only when the cell-center
        # AND the offset position are both inside the PEC shape.
        # For thin PEC obstacles (e.g. a 1-cell-wide PEC short),
        # cell-center is inside but Ey/Ez Yee positions are at
        # cell-corner offsets that fall *outside* the box → inv
        # remains 1 (vacuum). That leaves the H field free to
        # propagate inside the PEC region and seeds late-time
        # exponential growth.
        #
        # Fix: where ``pec_mask`` (cell-center binary) is True,
        # force all three inv components to zero. This is the
        # cell-center analogue of Stage 1's sigma=1e10 fold,
        # without the Ca→-1 instability that the sigma fold has
        # at Yee-staggered cells where inv > 0.
        if zero_inverse_on is not None:
            inv_xx, inv_yy, inv_zz = aniso_inv_eps
            inv_xx = jnp.where(zero_inverse_on, 0.0, inv_xx)
            inv_yy = jnp.where(zero_inverse_on, 0.0, inv_yy)
            inv_zz = jnp.where(zero_inverse_on, 0.0, inv_zz)
            aniso_inv_eps = (inv_xx, inv_yy, inv_zz)
            # zero_inverse_on marks interior PEC geometry (e.g. the PEC
            # short). The reference run has no interior PEC — do NOT
            # apply zero_inverse_on to ref_aniso_inv_eps, or the reference
            # becomes identical to the device and S11 = 0.
    elif subpixel_smoothing:
        from rfx.geometry.smoothing import (
            compute_smoothed_eps, smoothed_shape_pairs, warn_curved_high_contrast,
        )
        # #1043 stage B: same pad continuation as the Stage-2 site above.
        shape_eps_pairs, _unextendable = smoothed_shape_pairs(sim, grid)
        warn_curved_high_contrast(shape_eps_pairs, sim=sim)
        if shape_eps_pairs:
            aniso_eps = compute_smoothed_eps(grid, shape_eps_pairs, background_eps=1.0)

    # Conformal PEC (Stage 1): compute weights and produce anisotropic
    # eps. Skipped when Stage 2 unified path is active (use_kottke_pec)
    # — the inv-eps tensor already encodes the conformal behaviour.
    conformal_weights = None
    if conformal and pec_shapes and not use_kottke_pec:
        from rfx.geometry.conformal import (
            compute_conformal_weights_sdf,
            clamp_conformal_weights,
            warn_conformal_model,
        )
        warn_conformal_model()
        w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, pec_shapes)
        w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, conformal_min_weight)
        conformal_weights = (w_ex, w_ey, w_ez)

        # The 1/w permittivity is formed below, once the run's final
        # periodic flags are known (#1373).

    # Edge-owned capacitors go onto the Stage-2 inverse tensor here; the
    # Stage-1 / smoothed tensor gets them below, after the conformal
    # correction, so a conformal weight never divides a lumped C.
    aniso_inv_eps = add_lumped_eps(
        aniso_inv_eps, materials.eps_r_lumped, inverse=True)

    return aniso_eps, aniso_inv_eps, conformal_weights, _unextendable


def finish_tensor_operands(materials, aniso_eps, conformal_weights):
    """Merge conformal volume eps, then add edge-owned lumped capacitors."""
    # Conformal PEC (Stage 1) permittivity, #1373: the same four-cell
    # edge mean of the volume permittivity the plain E update uses, on the
    # run's final periodic flags (after the TFSF / 2-D overrides), divided
    # by the edge's open fraction w. On an edge no PEC touches (w == 1) it
    # is the plain path's value exactly.
    if conformal_weights is not None:
        w_ex, w_ey, w_ez = conformal_weights
        eps_mean = materials.components.eps
        eps_ex_c, eps_ey_c, eps_ez_c = conformal_module.conformal_eps_correction(
            eps_mean, w_ex, w_ey, w_ez)
        if aniso_eps is not None:
            # Merge: conformal PEC overrides smoothed eps at PEC boundary cells
            s_ex, s_ey, s_ez = aniso_eps
            eps_ex_c = jnp.where(w_ex < 1.0, eps_ex_c, s_ex)
            eps_ey_c = jnp.where(w_ey < 1.0, eps_ey_c, s_ey)
            eps_ez_c = jnp.where(w_ez < 1.0, eps_ez_c, s_ez)
        aniso_eps = (eps_ex_c, eps_ey_c, eps_ez_c)

    # Complete the volume tensor before adding edge-owned capacitors:
    # conformal weights must act on the volume, never on a lumped C.
    aniso_eps = add_lumped_eps(aniso_eps, materials.eps_r_lumped)

    return aniso_eps


def form_reference_operands(grid, materials, *, subpixel_smoothing,
                            conformal_weights, boundary_pec_shapes):
    """Form the waveguide reference operands with its existing vacuum rule."""
    ref_aniso_inv_eps = None
    ref_aniso_eps = None
    if subpixel_smoothing == "kottke_pec":
        from rfx.geometry.smoothing import compute_inv_eps_tensor_diag
        # Reference run is empty guide with same boundary walls only.
        # Must NOT include interior PEC geometry (e.g. PEC short box):
        # if device and reference share the same obstacle, both DFTs
        # are identical and (device - reference) / incident = 0.
        ref_aniso_inv_eps = compute_inv_eps_tensor_diag(
            grid,
            dielectric_shapes=[],
            pec_shapes=boundary_pec_shapes,
            background_eps=1.0,
        )
    if conformal_weights is not None:
        w_ex, w_ey, w_ez = conformal_weights
        # The reference run (vacuum) shares the same boundary
        # walls, so the conformal eps correction applies equally.
        # Build it from the ref vacuum eps so the only difference
        # ref-vs-device is the obstacle in ``materials.eps_r``.
        ref_eps_base = jnp.ones_like(materials.eps_r)
        ref_ex, ref_ey, ref_ez = conformal_module.conformal_eps_correction(
            ref_eps_base, w_ex, w_ey, w_ez,
        )
        ref_aniso_eps = (ref_ex, ref_ey, ref_ez)

    return ref_aniso_eps, ref_aniso_inv_eps

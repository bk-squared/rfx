"""Form geometry permittivity operands in array order."""

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from rfx.core.yee import add_lumped_eps
from rfx.geometry import conformal as conformal_module


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TensorOperands:
    """Finished tensor operands carried by one realized material model."""

    eps: tuple | None = None
    inv_eps: tuple | None = None
    conformal_weights: tuple | None = None
    inv_eps_smooth: bool = field(default=False, metadata={"static": True})

    node_smoothing: object = field(default=None, kw_only=True)

    @classmethod
    def of(cls, eps=None, inv_eps=None, conformal_weights=None, inv_eps_smooth=False,
           *, node_smoothing=None):
        """Represent an absent tensor by None, never an empty operand set."""
        from rfx.geometry.node_tensor import SmoothedEps
        if isinstance(eps, SmoothedEps):
            node_smoothing, eps = eps.nodes, eps.diagonal
        if eps is None and inv_eps is None and conformal_weights is None and node_smoothing is None:
            return None
        return cls(eps, inv_eps, conformal_weights, inv_eps_smooth, node_smoothing=node_smoothing)


def attach_tensor(materials, operands):
    """Attach a single operand set without changing the realization's identity."""
    components = materials.components
    if components is None:
        raise ValueError("tensor operands require realized material components")
    if components.tensor is not None and components.tensor is not operands:
        raise ValueError("two tensor operand sets for one model")
    return materials._replace(components=replace(components, tensor=operands))


def full_tensor_eps(sim, grid, materials, shape_eps_pairs):
    """Build node tensors at the realized permittivity precision on either mesh."""
    from rfx.geometry.node_tensor import compute_node_tensor_eps
    return compute_node_tensor_eps(
        grid, shape_eps_pairs, background_eps=1.0,
        field_dtype=(jnp.float64 if sim._precision == 'float64' else materials.eps_r.dtype))


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
            smoothed_shape_pairs, warn_curved_high_contrast,
        )
        # #1043 stage B: same pad continuation as the Stage-2 site above.
        shape_eps_pairs, _unextendable = smoothed_shape_pairs(sim, grid)
        if subpixel_smoothing != "full_tensor":
            warn_curved_high_contrast(shape_eps_pairs, sim=sim)
        if shape_eps_pairs:
            if subpixel_smoothing == "full_tensor":
                aniso_eps = full_tensor_eps(sim, grid, materials, shape_eps_pairs)
            else:
                from rfx.geometry.smoothing import compute_smoothed_eps
                aniso_eps = compute_smoothed_eps(grid, shape_eps_pairs, background_eps=1.0)

    # Conformal PEC (Stage 1): compute weights and produce anisotropic
    # eps. Entry admission refuses this with Stage 2; keep the exclusion
    # defensive so the two PEC treatments cannot be applied together.
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
    from rfx.geometry.node_tensor import SmoothedEps
    nodes = aniso_eps.nodes if isinstance(aniso_eps, SmoothedEps) else None
    if nodes is not None:
        aniso_eps = aniso_eps.diagonal
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

    return SmoothedEps(aniso_eps, nodes) if nodes is not None else aniso_eps


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


def refuse_full_tensor_conformal(smoothing, conformal, pec_shapes):
    """Refuse tensor modes that cannot carry the requested conformal weights."""
    if smoothing == 'kottke_pec' and conformal and pec_shapes:
        raise NotImplementedError(
            "subpixel_smoothing='kottke_pec' cannot combine with "
            "conformal_pec=True (including Boundary(conformal=True)): "
            "the smoothed inverse-permittivity tensor already treats the "
            "metal edges, so the conformal weights would not be formed. "
            "Choose subpixel_smoothing='kottke_pec' alone, or "
            "conformal_pec=True with subpixel_smoothing=False or True.")
    if smoothing == 'full_tensor' and conformal and pec_shapes:
        from rfx.model.smoothing_admission import FULL_TENSOR_MESSAGE
        raise NotImplementedError(FULL_TENSOR_MESSAGE + '; conformal PEC is not supported')


def refuse_tensor_conflicts(sim, grid, *, subpixel_smoothing, conformal_pec,
                            debye_spec, lorentz_spec, pec_shapes, kerr_chi3=None):
    """Refuse updates that cannot read geometry tensors, in run entry order."""
    refuse_full_tensor_conformal(subpixel_smoothing, conformal_pec, pec_shapes)
    if subpixel_smoothing:
        from rfx.geometry.smoothing import refuse_pmc_smoothing
        refuse_pmc_smoothing(sim, grid)

    # A Debye/Lorentz E update reads neither the smoothed permittivity
    # tensor nor the Dey-Mittra eps correction
    # (``_update_e_with_optional_dispersion``); running on would drop them.
    if debye_spec is not None or lorentz_spec is not None:
        sim._refuse_unsupported_run_kwargs(
            "uniform mesh with Debye/Lorentz materials", {
                "subpixel_smoothing": subpixel_smoothing,
                "conformal_pec": bool(conformal_pec and pec_shapes),
            },
            instead="remove the Debye/Lorentz poles",
            reason_overrides={
                "subpixel_smoothing":
                    "the dispersive E update does not read the smoothed "
                    "per-component permittivity tensor, so interfaces "
                    "would get scalar eps",
                "conformal_pec":
                    "the dispersive E update does not read the Dey-Mittra "
                    "eps correction, so PEC would be staircased",
            })

    # The UPML E update is built from the Stage-1 per-component eps
    # (``init_upml(aniso_eps=...)``) and never reads the Stage-2 inverse
    # tensor ``kottke_pec`` builds, so that tensor would be dropped.
    if sim._boundary == "upml" and grid.cpml_layers > 0:
        sim._refuse_unsupported_run_kwargs(
            "uniform mesh with boundary='upml'", {
                "subpixel_smoothing": (subpixel_smoothing
                                       if subpixel_smoothing == "kottke_pec"
                                       else False),
            },
            instead="use boundary='cpml'",
            reason_overrides={"subpixel_smoothing":
                "the UPML E update never reads the Stage-2 inverse-"
                "permittivity tensor, so dielectric interfaces would get "
                "unsmoothed eps and PEC the staircase mask"},
            remedy_overrides={"subpixel_smoothing":
                "use subpixel_smoothing=True (Stage 1, which UPML reads)"})

    use_kottke_pec = (subpixel_smoothing == "kottke_pec")
    if use_kottke_pec:
        from rfx.current_moments import refuse_h_side_conductor
        refuse_h_side_conductor(sim, 'subpixel_smoothing="kottke_pec"')
    if subpixel_smoothing and kerr_chi3 is not None:
        raise NotImplementedError(
            "subpixel_smoothing cannot combine with kerr_chi3: "
            "the Kerr E update does not read the permittivity tensor.")

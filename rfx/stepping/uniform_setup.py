from __future__ import annotations

from rfx._uniform_specs import (
    DesignBoxSpec,
    DesignOccupancySpec,
    ProbeSpec,
    SimResult,
    SnapshotSpec,
    SourceSpec,
    _ORDER4_CFL_FACTOR,
    _StepContext,
    _nearest_divisor,
    resolve_periodic,
)

from rfx.measurement import init_uniform_records, uniform_records
from dataclasses import replace
from typing import Callable, NamedTuple
from rfx import _realized, _grid_metric
import jax
import jax.numpy as jnp
from rfx.core.drives import StepDrives, drive_layout
from rfx.measurement import AUTO_END, finish_decay, make_decay_stop, one_step_xs
from rfx.model.source_coefficients import drive_table
from rfx.model import with_edge_keep
import numpy as np
from rfx.boundaries.axes import drop_periodic_axes, resolve_cpml_axes
from rfx.grid import Grid
from rfx.stepping.scan import recorded_scan
from rfx.core.yee import FDTDState, MaterialArrays, init_state, curl_h, CurlBoundary, precompute_coeffs
from rfx.stepping.yee import update_e, update_e_aniso, update_e_aniso_inv
from rfx.boundaries.pec import resolve_wall_faces, kottke_fenced_edge_masks, realized_pec_edge_masks
from rfx.progress import ProgressReporter, check_not_traced, concat_chunks, scan_with_progress, validate_report_every
from rfx.snapshots import plan_snapshot_pieces, snapshot_axes, snapshot_extractor, validate_snapshot_spec
from rfx.model import design_box_bounds, resolve_design_box

class _DesignOccupancyKeep(NamedTuple):
    """Resolved write window and ``1 - M`` factors, built once outside the scan."""
    write: tuple
    keep: tuple


def _update_e_with_optional_dispersion(
    state: FDTDState,
    materials: MaterialArrays,
    dt: float,
    dx: float,
    *,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    periodic: tuple = (False, False, False),
    aniso_eps: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    stencil_order: int = 2,
    bloch: tuple | None = None,
    boundary=None,
) -> tuple[FDTDState, object | None, object | None]:
    """Update E with standard, Debye, Lorentz, or mixed dispersion.

    Parameters
    ----------
    aniso_eps : (eps_ex, eps_ey, eps_ez) or None
        Per-component relative permittivity arrays for subpixel smoothing.
        Only used when no dispersion model is active.
    aniso_inv_eps : (inv_xx, inv_yy, inv_zz) or None
        Stage 2 unified path: per-component inverse-permittivity tensor
        diagonal. When set, takes precedence over ``aniso_eps`` and
        dispatches to ``update_e_aniso_inv``. Numerically stable in the
        PEC limit (inv = 0); see derivation §5.
    stencil_order : int
        2 (default, byte-identical) or 4 ((2,4) wide stencil). Only the
        plain vacuum/dielectric branch honours order=4 — the dispersion
        (debye/lorentz) and anisotropic (aniso_eps/aniso_inv_eps) branches
        are fenced against order=4 upstream in ``_build_step_setup`` and
        keep their 2nd-order kernels regardless.
    """
    if debye is None and lorentz is None:
        if aniso_inv_eps is not None:
            inv_xx, inv_yy, inv_zz = aniso_inv_eps
            return update_e_aniso_inv(state, materials, inv_xx, inv_yy, inv_zz,
                                      dt, dx, periodic=periodic, boundary=boundary), None, None
        if aniso_eps is not None:
            eps_ex, eps_ey, eps_ez = aniso_eps
            return update_e_aniso(state, materials, eps_ex, eps_ey, eps_ez,
                                  dt, dx, periodic=periodic, boundary=boundary), None, None
        return update_e(state, materials, dt, dx, periodic=periodic,
                        stencil_order=stencil_order, bloch=bloch, boundary=boundary), None, None

    if debye is not None and lorentz is None:
        from rfx.stepping.poles import update_e_debye

        debye_coeffs, debye_state = debye
        new_state, new_debye = update_e_debye(
            state, debye_coeffs, debye_state, dt, dx, periodic=periodic, boundary=boundary)
        return new_state, new_debye, None

    if lorentz is not None and debye is None:
        from rfx.stepping.poles import update_e_lorentz

        lorentz_coeffs, lorentz_state = lorentz
        new_state, new_lorentz = update_e_lorentz(
            state, lorentz_coeffs, lorentz_state, dt, dx, periodic=periodic, boundary=boundary)
        return new_state, None, new_lorentz

    # Mixed Debye + Lorentz update.
    from rfx.materials.debye import DebyeState
    from rfx.materials.lorentz import LorentzState

    debye_coeffs, debye_state = debye
    lorentz_coeffs, lorentz_state = lorentz

    # Narrow every output back to the dtype of the carry it came from
    # (issue #656) — same policy as the single-model bodies in
    # rfx/materials/{debye,lorentz}.py, which carry the full rationale.
    _fdtype = state.ex.dtype
    _dpdtype = jnp.promote_types(debye_state.px.dtype, _fdtype)
    _lpdtype = jnp.promote_types(lorentz_state.px.dtype, _fdtype)

    hx, hy, hz = state.hx, state.hy, state.hz

    curl_x, curl_y, curl_z = curl_h(
        hx, hy, hz, dx, periodic, boundary=boundary, scale_after=True)

    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    # Explicit Lorentz polarization update first (per component, #1260).
    from rfx.materials.debye import per_component
    from rfx.materials.lorentz import (
        lorentz_p_component, mixed_e_component_coeffs,
    )
    e_old = (ex_old, ey_old, ez_old)
    curls = (curl_x, curl_y, curl_z)
    p_l = (lorentz_state.px, lorentz_state.py, lorentz_state.pz)
    p_l_prev = (lorentz_state.px_prev, lorentz_state.py_prev,
                lorentz_state.pz_prev)
    p_d = (debye_state.px, debye_state.py, debye_state.pz)
    p_l_new = tuple(
        lorentz_p_component(lorentz_coeffs, c, e_old[c], p_l[c], p_l_prev[c],
                            _lpdtype)
        for c in range(3))
    px_l_new, py_l_new, pz_l_new = p_l_new

    e_new, p_d_new = [], []
    for c in range(3):
        dp_l = jnp.sum(p_l_new[c] - p_l[c], axis=0)
        ca, cb, cc_debye, cc_lorentz = mixed_e_component_coeffs(
            debye_coeffs, lorentz_coeffs, c, dt)
        e_c = (
            ca * e_old[c]
            + cb * curls[c]
            + jnp.sum(cc_debye * p_d[c], axis=0)
            - cc_lorentz * dp_l
        ).astype(_fdtype)
        e_new.append(e_c)
        beta_c = per_component(debye_coeffs.beta, "beta")[c]
        p_d_new.append((debye_coeffs.alpha * p_d[c]
                        + beta_c * (e_c[None] + e_old[c][None])
                        ).astype(_dpdtype))
    ex_new, ey_new, ez_new = e_new

    new_fdtd = state._replace(
        ex=ex_new,
        ey=ey_new,
        ez=ez_new,
        step=state.step + 1,
    )
    new_debye = DebyeState(px=p_d_new[0], py=p_d_new[1], pz=p_d_new[2])
    new_lorentz = LorentzState(
        px=px_l_new,
        py=py_l_new,
        pz=pz_l_new,
        px_prev=lorentz_state.px,
        py_prev=lorentz_state.py,
        pz_prev=lorentz_state.pz,
    )

    return new_fdtd, new_debye, new_lorentz


class _SimSetup(NamedTuple):
    """Shared setup artefacts returned by ``_build_step_setup``.

    ``ctx_kwargs`` is a dict of ``_StepContext`` fields that are identical
    between the two drivers.  Each driver extends / overrides it with its
    own fields before passing to ``_StepContext(**ctx_kwargs, **overrides)``.
    """
    carry_init: dict
    ctx_kwargs: dict
    # metadata kept separate for carry-assembly post-loop in run_until_decay
    src_meta: list
    mag_src_meta: list
    prb_meta: list
    dft_meta: tuple
    waveguide_meta: tuple
    wire_sparam_meta: tuple
    lumped_sparam_meta: tuple
    rlc_meta: tuple
    # issue #313 opt-in reference-plane accumulators (empty when unused)
    wire_refplane_meta: tuple
    # 8-field tuple: (axis, index, freqs, comp_names, lo1, hi1, lo2, hi2)
    flux_meta_8: tuple
    # 11-field tuple adds: total_steps, window, window_alpha
    flux_meta_11: tuple
    # resolved physical constants (both drivers need these for waveform setup)
    dt: float
    dx: float
    periodic: tuple
    pec_axes: str
    pec_faces: frozenset


def _resolve_design_occupancy(
    spec: "DesignOccupancySpec",
    *,
    grid,
    periodic,
    pec_occupancy,
    field_dtype,
) -> "_DesignOccupancyKeep":
    """Check an occupancy design box against this run, then build ``1 - M`` (#1183).

    Far fewer fences than :func:`resolve_design_box`, and the reason is
    physical: the occupancy scaling is a pure multiply applied AFTER the E
    update and the absorber, and it reads no material array. Whatever the E
    update was — dispersive, anisotropic, fourth-order — the step arrives at
    this slot with the same field on both formulations, so the window redo
    reproduces the grid-wide call. What it does need is that no cell outside
    the window can see the design occupancy, which is the shift reach
    :func:`pec_occupancy_box_keep` owns (and why it refuses periodic axes).
    """
    from rfx.boundaries.pec import pec_occupancy_box_keep

    bounds, box_shape = design_box_bounds(spec.bounds, grid)
    occ = jnp.asarray(spec.occupancy)
    if tuple(occ.shape) != box_shape:
        raise ValueError(
            f"design occupancy has shape {tuple(occ.shape)} but the design "
            f"box {bounds} realizes {box_shape} cells.")
    write, keep = pec_occupancy_box_keep(
        bounds, occ, shape=tuple(grid.shape), dtype=field_dtype,
        pec_occupancy=pec_occupancy, periodic=periodic)
    return _DesignOccupancyKeep(write=write, keep=keep)


def _build_step_setup(
    grid: "Grid",
    materials: "MaterialArrays",
    *,
    boundary: str,
    cpml_axes: str,
    pec_axes: "str | None",
    periodic: "tuple | None",
    debye: "tuple | None",
    lorentz: "tuple | None",
    tfsf: "tuple | None",
    _feature_tfsf: bool = False,
    sources: list,
    probes: list,
    dft_planes: list,
    flux_monitors: list,
    waveguide_ports: list,
    ntff: object,
    aniso_eps: "tuple | None",
    aniso_inv_eps: "tuple | None",
    aniso_inv_eps_smooth: bool,
    pec_mask: object,
    pec_sheets: object,
    pec_wires: object,
    pec_edge_masks: object,
    pec_occupancy: object,
    conformal_weights: "tuple | None",
    wire_port_sparams: list,
    lumped_port_sparams: list,
    lumped_rlc: list,
    kerr_chi3: "object | None",
    field_dtype: object,
    mag_sources: list,
    stencil_order: int = 2,
    wire_refplane_sparams: "list | None" = None,
    sheet_impedance: "object | None" = None,
    design_box: "DesignBoxSpec | None" = None,
    design_occupancy: "DesignOccupancySpec | None" = None,
    current_moments: "object | None" = None,
) -> "_SimSetup":
    """Build the shared setup artefacts used by both ``run`` and ``run_until_decay``.

    Returns a ``_SimSetup`` NamedTuple. Each driver extends ``ctx_kwargs``
    with its own driver-specific ``_StepContext`` fields before constructing
    the context.
    """
    dt = grid.dt
    dx = grid.dx
    periodic = resolve_periodic(grid, periodic)
    from rfx.model.pole_stability import judge_prebuilt_poles
    from rfx.materials.lorentz import pole_sets_judged
    if lorentz is not None and not pole_sets_judged(lorentz[0]):
        judge_prebuilt_poles(grid, lorentz, debye, periodic=periodic)

    # ---- boundary configuration ----
    from rfx.model.materials import with_components
    from rfx.model.tensor_operands import TensorOperands
    materials = with_components(materials, grid, periodic=periodic, tensor=
        TensorOperands.of(aniso_eps, aniso_inv_eps, conformal_weights, aniso_inv_eps_smooth))
    tensor = materials.components.tensor
    aniso_eps, aniso_inv_eps = (tensor.eps, tensor.inv_eps) if tensor else (None, None)
    conformal_weights = tensor.conformal_weights if tensor else None
    aniso_inv_eps_smooth = tensor.inv_eps_smooth if tensor else False
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(
        materials, lane="uniform Yee with dispersion/UPML/tensor or design-box updates",
        unsupported=(debye is not None or lorentz is not None or boundary == "upml"
                     or aniso_eps is not None or aniso_inv_eps is not None
                     or stencil_order != 2 or design_box is not None
                     or design_occupancy is not None or kerr_chi3 is not None))
    cpml_axes = drop_periodic_axes(cpml_axes, periodic)  # no CPML/PEC on periodic axes
    # ---- the walls, per face, from the grid's declaration (#1164) ----
    # ``resolve_wall_faces`` is the one rule every entry point shares:
    # periodic -> none; a magnetic face is never an electric wall; a
    # declared PEC face is one; any other face is PEC-backed unless the
    # legacy ``pec_axes`` string withholds that default for its axis.
    # The scan applies these per face only; the axis-wide ``apply_pec``
    # is gone from the step (an axis wall over a PMC face shorted a port
    # on that plane through run(), and through forward() after #1194).
    _pec_faces_frozen, _pmc_faces_frozen = resolve_wall_faces(
        grid, periodic, pec_axes)
    use_pec_faces = bool(_pec_faces_frozen)
    use_pmc_faces = bool(_pmc_faces_frozen)
    # Axes with an electric wall on BOTH faces, for the GPU fast path's
    # coefficient bake (``precompute_coeffs``) and for callers reading it.
    pec_axes = "".join(
        a for a in "xyz"
        if f"{a}_lo" in _pec_faces_frozen and f"{a}_hi" in _pec_faces_frozen)

    # ---- subsystem flags (resolved at Python trace time) ----
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0
    use_upml = boundary == "upml" and grid.cpml_layers > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_tfsf = tfsf is not None
    use_ntff = ntff is not None
    use_current_moments = current_moments is not None
    use_dft_planes = len(dft_planes) > 0
    use_flux_monitors = len(flux_monitors) > 0
    use_waveguide_ports = len(waveguide_ports) > 0
    # ---- #931 §1.7: realize the PEC edge masks ONCE, here ----
    # ``pec_edge_masks`` may arrive pre-realized (and port-cleared) from a
    # lane that had to read them before the run; otherwise realize from the
    # volume cell mask plus the declared sheets/wires with THIS run's
    # periodic flags.
    pec_sheets = tuple(pec_sheets or ())
    pec_wires = tuple(pec_wires or ())
    if pec_edge_masks is None and (
            pec_mask is not None or pec_sheets or pec_wires):
        pec_edge_masks = realized_pec_edge_masks(
            pec_mask, sheets=pec_sheets, wires=pec_wires, periodic=periodic)
    use_pec_edges = pec_edge_masks is not None
    use_pec_occupancy = pec_occupancy is not None
    # The soft lane carries sheets/wires as STATIC edge masks (§1.6).
    #
    # Realized from the declarations, then INTERSECTED with the masks this
    # call was handed.  Those arrived port-cleared (§1.9: a port releases
    # the one component it drives, at its own cells), and re-realizing from
    # the declaration alone puts the conductor back over the port — a
    # 50 ohm Ex port standing on a PEC sheet reads a False Ex entry in the
    # masks it was given and a True one in the reconstruction.  Measured
    # through ``pec_occupancy_override=zeros``, which §1.6 requires to be
    # the hard path exactly: an off-sheet Ez probe moved 5.7245574 ->
    # 12.5340872 on a 12 mm board at dx = 2 mm.  The identity has to hold
    # THROUGH the override entry point, so the clearing is carried here
    # rather than recomputed.
    pec_static_edge_masks = None
    if use_pec_occupancy and (pec_sheets or pec_wires):
        pec_static_edge_masks = realized_pec_edge_masks(
            None, sheets=pec_sheets, wires=pec_wires, periodic=periodic)
        if pec_edge_masks is not None:
            pec_static_edge_masks = tuple(
                s & m for s, m in zip(pec_static_edge_masks, pec_edge_masks))
    use_conformal = conformal_weights is not None
    # Stage 2: when aniso_inv_eps is set, the inverse-permittivity tensor
    # encodes both PEC behaviour and dielectric subpixel smoothing —
    # apply_conformal_pec is redundant and SKIPPED to avoid double-zeroing.
    use_aniso_inv = aniso_inv_eps is not None
    # ---- #931 §1.8 fence: Kottke Stage-2 owns its own volume ----
    # ``subpixel_smoothing="kottke_pec"`` builds the tensor from the SAME
    # ``pec_shapes`` that produced ``pec_mask``, and a partially filled edge
    # gets a FRACTIONAL inverse permittivity.  Applying the §1.2 ownership
    # rule on top hard-zeroes those edges and throws the subpixel model away.
    # Restrict the applied set to what the tensor froze, plus the sheets and
    # wires the tensor cannot see (they own no cell).  ``aniso_inv_eps_smooth``
    # marks the occupancy-derived tensor instead, where the static declaration
    # and the traced override are two different geometries and the
    # intersection would silently delete declared metal — that lane keeps the
    # realized edges.
    if use_pec_edges and use_aniso_inv and not aniso_inv_eps_smooth:
        pec_edge_masks = kottke_fenced_edge_masks(
            pec_edge_masks, aniso_inv_eps,
            sheets=pec_sheets, wires=pec_wires, periodic=periodic)
    from rfx.model.node_pec import restrict_node_pec
    materials = restrict_node_pec(materials, pec_edge_masks, _pec_faces_frozen)
    tensor = materials.components.tensor
    use_wire_sparams = len(wire_port_sparams) > 0
    use_lumped_sparams = len(lumped_port_sparams) > 0
    wire_refplane_sparams = wire_refplane_sparams or []
    use_wire_refplanes = len(wire_refplane_sparams) > 0
    use_lumped_rlc = len(lumped_rlc) > 0
    use_kerr = kerr_chi3 is not None

    from rfx.boundaries.setup import oblique_bloch
    _oblique_bloch = oblique_bloch(tfsf, use_ntff=use_ntff,
                                 use_dft_planes=use_dft_planes, use_flux_monitors=use_flux_monitors)

    # The far-field integral is over the SCATTERED field, which it only is
    # when the Huygens box encloses the whole injected region. Plain ints,
    # evaluated here at trace time.
    from rfx.boundaries.tfsf import admit_setup
    waveguide_ports = admit_setup(grid=grid, tfsf=tfsf, materials=materials, waveguide_ports=waveguide_ports,
                ntff=ntff if use_ntff else None, periodic=periodic, feature_owned=_feature_tfsf,
                updates=dict(debye=debye, lorentz=lorentz, aniso_eps=aniso_eps,
                             aniso_inv_eps=aniso_inv_eps, pec_mask=pec_mask, pec_edge_masks=pec_edge_masks,
                             pec_sheets=pec_sheets, pec_wires=pec_wires,
                             pec_occupancy=pec_occupancy, conformal_weights=conformal_weights,
                             sheet_impedance=sheet_impedance, kerr_chi3=kerr_chi3,
                             design_box=design_box, design_occupancy=design_occupancy),
                localized=(("sources", sources), ("mag_sources", mag_sources),
                           ("waveguide_ports", waveguide_ports), ("wire_port_sparams", wire_port_sparams),
                           ("lumped_port_sparams", lumped_port_sparams), ("lumped_rlc", lumped_rlc)))

    # ---- (2,4) fourth-order-in-space stencil (PR-1b) ----
    # order=2 is the default and BYTE-IDENTICAL: dt and every kernel call are
    # untouched.  order=4 is reachable ONLY on the plain uniform-Cartesian
    # vacuum/dielectric path (pec/periodic boundary, default solver, no
    # dispersion / anisotropy / conformal-PEC / Kerr).  Any other sub-feature
    # under order=4 raises here — a silent 2nd-order result on an unsupported
    # path is a CRITICAL correctness failure.
    if stencil_order not in (2, 4):
        raise ValueError(f"stencil_order must be 2 or 4, got {stencil_order}")
    if stencil_order == 4:
        _unsupported = []
        if use_cpml:
            _unsupported.append("boundary='cpml'")
        if use_upml:
            _unsupported.append("boundary='upml'")
        if use_debye:
            _unsupported.append("debye dispersion")
        if use_lorentz:
            _unsupported.append("lorentz dispersion")
        if aniso_eps is not None:
            _unsupported.append("aniso_eps (subpixel/anisotropic eps)")
        if aniso_inv_eps is not None:
            _unsupported.append("aniso_inv_eps (Kottke inverse-eps tensor)")
        if use_conformal:
            _unsupported.append("conformal PEC")
        if use_kerr:
            _unsupported.append("Kerr nonlinearity")
        if _unsupported:
            raise NotImplementedError(
                "stencil_order=4 is only supported on the plain uniform "
                "Cartesian vacuum/dielectric path (pec/periodic boundary, "
                "default solver, no dispersion/anisotropy/conformal/Kerr). "
                "Unsupported feature(s) present: " + ", ".join(_unsupported)
                + ". Use stencil_order=2 (the default) for these."
            )
        # Derate the timestep for the (2,4) stability bound.
        dt = dt * _ORDER4_CFL_FACTOR

    # Lazily-imported helper callables (imported now so they can be passed
    # into _StepContext and survive JIT reuse across calls).
    apply_cpml_h = apply_cpml_e = None
    apply_upml_h = apply_upml_e = None
    cpml_params = None
    upml_coeffs = None
    apply_tfsf_h = apply_tfsf_e = None
    update_tfsf_1d_h = update_tfsf_1d_e = None
    update_tfsf_2d_h = update_tfsf_2d_e = None
    _tfsf_is_2d = False
    tfsf_cfg = None
    _bloch = None
    init_ntff_data_fn = accumulate_ntff_fn = None
    apply_kerr_ade = None
    update_rlc_elements = None
    debye_coeffs = lorentz_coeffs = None

    # ---- initialise states ----
    _field_dtype = field_dtype if field_dtype is not None else jnp.float32
    if _oblique_bloch:
        # Complex envelope P; overrides mixed-precision (float16) for this path.
        _field_dtype = jnp.complex64
    fdtd = init_state(grid.shape, field_dtype=_field_dtype)
    materials = with_edge_keep(materials, pec_occupancy, dtype=fdtd.ex.dtype,
        periodic=periodic, sheet_edge_masks=pec_static_edge_masks)
    carry_init: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml
        from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h
        # psi carry dtype follows the field dtype so the scan carry stays
        # dtype-consistent when fields are complex (#404 oblique path).
        cpml_params, cpml_state = init_cpml(grid, field_dtype=_field_dtype)
        carry_init["cpml"] = cpml_state
    elif use_upml:
        from rfx.boundaries.upml import init_upml
        from rfx.boundaries.upml import apply_upml_e, apply_upml_h
        upml_coeffs = init_upml(grid, materials, axes=cpml_axes,
                                aniso_eps=aniso_eps, periodic=periodic)

    if use_debye:
        debye_coeffs, debye_state = debye
        carry_init["debye"] = debye_state

    if use_lorentz:
        lorentz_coeffs, lorentz_state = lorentz
        carry_init["lorentz"] = lorentz_state

    if use_tfsf:
        from rfx.sources.tfsf import (
            update_tfsf_1d_e,
            update_tfsf_1d_h,
            apply_tfsf_e,
            apply_tfsf_h,
            is_tfsf_2d, prepare_tfsf_for_scan,
        )
        tfsf_cfg, tfsf_state = prepare_tfsf_for_scan(tfsf, dx, dt)
        carry_init["tfsf"] = tfsf_state
        _tfsf_is_2d = is_tfsf_2d(tfsf_cfg)
        if _tfsf_is_2d:
            from rfx.sources.tfsf_2d import (
                update_tfsf_2d_h, update_tfsf_2d_e, bloch_phase_tuple,
            )
            # Per-axis Bloch phase on the 3D periodic roll (transverse axis),
            # matching the 2D-aux grid's transform (#404).
            _bloch = bloch_phase_tuple(tfsf_cfg, grid.dx)

    if use_ntff:
        from rfx.farfield import init_ntff_data as _init_ntff_data
        from rfx.farfield import accumulate_ntff as _accumulate_ntff
        init_ntff_data_fn = _init_ntff_data
        accumulate_ntff_fn = _accumulate_ntff
        # NTFF accumulator dtype follows the field dtype with a float32 floor
        # (#646), mirroring the CPML psi carry above. accumulate_ntff pins its
        # phase arithmetic to whatever is allocated here, so the scan carry
        # closes under jax_enable_x64 instead of raising a dtype mismatch.
        carry_init["ntff"] = _init_ntff_data(ntff, field_dtype=_field_dtype)

    accumulate_current_moments_fn = None
    if use_current_moments:
        from rfx.current_moments import (
            accumulate_current_moments as _accumulate_cm,
            init_current_moment_data as _init_cm,
        )
        accumulate_current_moments_fn = _accumulate_cm
        # Same dtype policy as the NTFF carry (#646): the accumulator follows
        # the field dtype with a complex64 floor so the scan carry closes
        # under jax_enable_x64.
        carry_init["current_moments"] = _init_cm(
            current_moments, field_dtype=_field_dtype)

    if use_dft_planes:
        carry_init["dft_planes"] = tuple(probe.accumulator for probe in dft_planes)

    if use_waveguide_ports:
        carry_init["waveguide_port_accs"] = tuple(
            (
                cfg.v_probe_t,
                cfg.v_ref_t,
                cfg.i_probe_t,
                cfg.i_ref_t,
                cfg.v_inc_t,
                cfg.n_steps_recorded,
            )
            for cfg in waveguide_ports
        )

    # Port V/I DFT accumulator dtype (wire + lumped).  ``v * phase`` inside the
    # scan body promotes to complex128 under a scoped ``jax_enable_x64`` (dt/dx
    # and the phase factor become float64/complex128), so a hardcoded complex64
    # accumulator trips the lax.scan carry-dtype contract there.  Deriving the
    # dtype from ``result_type(complex64, float64)`` keeps it complex64 with x64
    # OFF (byte-identical to the historical pin) and promotes to complex128 with
    # x64 ON — this is what makes forward(port_s11_freqs=...) differentiable
    # under scoped x64 (WP 4-E gate + the pre-existing eps_override AD lane).
    _sparam_acc_dtype = jnp.result_type(jnp.complex64, jnp.float64)

    wire_sparam_meta: tuple = ()
    if use_wire_sparams:
        # Initialize V, I, V_inc DFT accumulators per wire port.
        carry_init["wire_sparam_accs"] = tuple(
            (
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),  # v_dft
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),  # i_dft
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),  # v_inc_dft
                # v_port_dft (issue #764): whole-port gap voltage
                # V_port = sum over LIVE cells of -E_c*dx.
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),
                # v_ref_dft (issue #683 x #764): PRE-injection drive-sample
                # reference for the #308-calibrated off-diagonal
                # decomposition (bit-identical to the historical v_dft).
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),
            )
            for wp in wire_port_sparams
        )
        wire_sparam_meta = tuple(wire_port_sparams)

    wire_refplane_meta: tuple = ()
    if use_wire_refplanes:
        # Initialize V, I(-dx/2), I(+dx/2) DFT accumulators per reference
        # plane (issue #313 opt-in; two planes per opted port).  Same
        # accumulator dtype and rect-DFT kernel as the port-cell channels.
        carry_init["wire_refplane_accs"] = tuple(
            (
                jnp.zeros(len(rp.freqs), dtype=_sparam_acc_dtype),  # v_dft
                jnp.zeros(len(rp.freqs), dtype=_sparam_acc_dtype),  # i_minus
                jnp.zeros(len(rp.freqs), dtype=_sparam_acc_dtype),  # i_plus
            )
            for rp in wire_refplane_sparams
        )
        wire_refplane_meta = tuple(wire_refplane_sparams)

    lumped_sparam_meta: tuple = ()
    if use_lumped_sparams:
        # Initialize V, I DFT accumulators per lumped port (issue #72).
        # No V_inc accumulator needed: the wave decomposition
        # ``a = (-V + Z0·I)/(2√Z0)`` is exact regardless of source pulse shape.
        # The third channel is the PRE-injection drive sample (the
        # historical #72 v_dft, bit-for-bit), kept because the #308
        # off-diagonal incident wave is calibrated against it; the
        # physical V/I are accumulated post-injection.
        carry_init["lumped_sparam_accs"] = tuple(
            (
                jnp.zeros(len(lp.freqs), dtype=_sparam_acc_dtype),  # v_dft
                jnp.zeros(len(lp.freqs), dtype=_sparam_acc_dtype),  # i_dft
                jnp.zeros(len(lp.freqs), dtype=_sparam_acc_dtype),  # v_ref_dft
            )
            for lp in lumped_port_sparams
        )
        lumped_sparam_meta = tuple(lumped_port_sparams)

    # Flux monitor carry + both meta flavours (8-field for decay, 11 for run).
    flux_meta_8: tuple = ()
    flux_meta_11: tuple = ()
    if use_flux_monitors:
        from rfx.measurement.accumulators import flux_metadata
        flux_meta_8 = flux_metadata(flux_monitors, window=False)
        flux_meta_11 = flux_metadata(flux_monitors, window=True)
        carry_init["flux_monitors"] = tuple(
            (fm.e1_dft, fm.e2_dft, fm.h1_dft, fm.h2_dft) for fm in flux_monitors
        )

    rlc_meta: tuple = ()
    if use_lumped_rlc:
        from rfx.lumped import init_rlc_state, rlc_carry_dtype
        from rfx.stepping.lumped import update_rlc_elements
        # Thread the ADE-carry dtype from the metas (WP 4-E).  Concrete run()
        # metas are all Python floats + float32 fields -> float32 (byte-identical
        # to the historical init_rlc_state() pin); a traced float64 component
        # value under scoped-x64 promotes the carry so the lax.scan carry
        # input/output dtypes agree.
        _rlc_dtype = rlc_carry_dtype(lumped_rlc, _field_dtype)
        carry_init["rlc_states"] = tuple(
            init_rlc_state(dtype=_rlc_dtype) for _ in lumped_rlc)
        rlc_meta = tuple(lumped_rlc)

    if use_kerr:
        from rfx.stepping.kerr import apply_kerr_ade

    # ---- metadata tuples ----
    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    mag_src_meta = [(s.i, s.j, s.k, s.component) for s in mag_sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]
    from rfx.measurement.accumulators import plane_metadata
    dft_meta = plane_metadata(dft_planes)
    waveguide_meta = tuple(waveguide_ports)

    # ---- #1179 design box: fence, then build its coefficients once ----
    design_box_coeffs = None
    if design_box is not None:
        design_box_coeffs = resolve_design_box(
            design_box,
            grid=grid,
            materials=materials,
            dt=dt,
            periodic=periodic,
            use_cpml=use_cpml,
            use_upml=use_upml,
            cpml_axes=cpml_axes,
            use_debye=use_debye,
            use_lorentz=use_lorentz,
            use_kerr=use_kerr,
            aniso_eps=aniso_eps,
            aniso_inv_eps=aniso_inv_eps,
            stencil_order=stencil_order,
            bloch=_bloch,
            sheet_impedance=sheet_impedance,
            cell_metas=(
                ("source", [(s.i, s.j, s.k, s.component) for s in sources]),
                ("magnetic source", [(s.i, s.j, s.k) for s in mag_sources]),
                ("lumped port", [(p.i, p.j, p.k, p.component)
                                 for p in lumped_sparam_meta]),
                ("wire port", [
                    (*cell, p.component)
                    for p in wire_sparam_meta
                    for cell in (tuple(p.live_cells)
                                 or ((p.mid_i, p.mid_j, p.mid_k),))
                ]),
                ("lumped RLC element", [(m.i, m.j, m.k) for m in rlc_meta]),
            ),
        )

    # ---- #1183 design occupancy: fence, then build its 1 - M once ----
    design_occupancy_keep = None
    if design_occupancy is not None:
        design_occupancy_keep = _resolve_design_occupancy(
            design_occupancy,
            grid=grid,
            periodic=periodic,
            pec_occupancy=pec_occupancy,
            field_dtype=_field_dtype,
        )

    # ---- shared _StepContext keyword arguments ----
    # These are identical for both drivers.  Each driver extends this dict
    # with its own overrides before calling _StepContext(**ctx_kwargs).
    ctx_kwargs: dict = dict(
        grid=grid,
        materials=materials,
        dt=dt,
        dx=dx,
        periodic=periodic,
        pec_axes=pec_axes,
        stencil_order=stencil_order,
        use_upml=use_upml,
        use_cpml=use_cpml,
        use_tfsf=use_tfsf,
        use_debye=use_debye,
        use_lorentz=use_lorentz,
        use_ntff=use_ntff,
        use_current_moments=use_current_moments,
        current_moments=current_moments,
        use_dft_planes=use_dft_planes,
        use_flux_monitors=use_flux_monitors,
        use_waveguide_ports=use_waveguide_ports,
        use_pec_faces=use_pec_faces,
        use_pmc_faces=use_pmc_faces,
        use_aniso_inv=use_aniso_inv,
        aniso_inv_eps_smooth=aniso_inv_eps_smooth,
        use_pec_edges=use_pec_edges,
        use_pec_occupancy=use_pec_occupancy,
        use_conformal=use_conformal,
        use_wire_sparams=use_wire_sparams,
        use_lumped_sparams=use_lumped_sparams,
        use_wire_refplanes=use_wire_refplanes,
        use_lumped_rlc=use_lumped_rlc,
        use_kerr=use_kerr,
        use_sheet_impedance=sheet_impedance is not None,
        sheet_impedance=sheet_impedance,
        use_design_box=design_box_coeffs is not None,
        design_box=design_box_coeffs,
        use_design_occupancy=design_occupancy_keep is not None,
        design_occupancy=design_occupancy_keep,
        cpml_params=cpml_params,
        cpml_axes=cpml_axes,
        upml_coeffs=upml_coeffs,
        tfsf_cfg=tfsf_cfg,
        tfsf_is_2d=_tfsf_is_2d,
        bloch=_bloch,
        debye_coeffs=debye_coeffs,
        lorentz_coeffs=lorentz_coeffs,
        pec_edge_masks=pec_edge_masks,
        pec_static_edge_masks=pec_static_edge_masks,
        kerr_chi3=kerr_chi3,
        ntff=ntff,
        pec_faces_frozen=_pec_faces_frozen,
        pmc_faces_frozen=_pmc_faces_frozen,
        curl_boundary=CurlBoundary(_pec_faces_frozen, _pmc_faces_frozen, periodic),
        prb_meta=tuple(prb_meta),
        dft_meta=dft_meta,
        waveguide_meta=waveguide_meta,
        wire_sparam_meta=wire_sparam_meta,
        lumped_sparam_meta=lumped_sparam_meta,
        wire_refplane_meta=wire_refplane_meta,
        rlc_meta=rlc_meta,
        apply_cpml_h=apply_cpml_h,
        apply_cpml_e=apply_cpml_e,
        apply_upml_h=apply_upml_h,
        apply_upml_e=apply_upml_e,
        apply_tfsf_h=apply_tfsf_h,
        apply_tfsf_e=apply_tfsf_e,
        update_tfsf_1d_h=update_tfsf_1d_h,
        update_tfsf_1d_e=update_tfsf_1d_e,
        update_tfsf_2d_h=update_tfsf_2d_h if _tfsf_is_2d else None,
        update_tfsf_2d_e=update_tfsf_2d_e if _tfsf_is_2d else None,
        init_ntff_data=init_ntff_data_fn,
        accumulate_ntff=accumulate_ntff_fn,
        accumulate_current_moments=accumulate_current_moments_fn,
        apply_kerr_ade=apply_kerr_ade if use_kerr else None,
        update_rlc_elements=update_rlc_elements if use_lumped_rlc else None,
    )

    return _SimSetup(
        carry_init=carry_init,
        ctx_kwargs=ctx_kwargs,
        src_meta=src_meta,
        mag_src_meta=mag_src_meta,
        prb_meta=prb_meta,
        dft_meta=dft_meta,
        waveguide_meta=waveguide_meta,
        wire_sparam_meta=wire_sparam_meta,
        lumped_sparam_meta=lumped_sparam_meta,
        rlc_meta=rlc_meta,
        wire_refplane_meta=wire_refplane_meta,
        flux_meta_8=flux_meta_8,
        flux_meta_11=flux_meta_11,
        dt=dt,
        dx=dx,
        periodic=periodic,
        pec_axes=pec_axes,
        pec_faces=_pec_faces_frozen,
    )


def make_core_step(ctx: _StepContext, invariants: dict | None = None,
                   *, design_hook=None, hooks=None):
    """Compose the uniform step; retain the runner and adjoint call signatures.

    Returns ``core_step(carry, step_idx, src_vals, mag_src_vals)`` yielding
    ``(new_carry, probe_out, extras)``. The design callback attaches immediately
    after the E update; optional hook attachments use the transient uniform frame.
    """
    from rfx.stepping.uniform import make_uniform_step
    if invariants is None:
        invariants = core_step_invariants(ctx)
    return make_uniform_step(invariants["ctx"], invariants,
                             design_hook=design_hook, hooks=hooks)


def core_step_invariants(ctx: _StepContext) -> dict:
    """What :func:`make_core_step`'s kernel reads besides its arguments.

    ``ctx`` itself plus arrays derived from it once, before the loop: the
    CPML permittivity, sheet coefficients and boolean conformal zero masks. A jitted
    caller passes the per-cell arrays in this dict as an argument
    (:func:`rfx.core.jax_utils.split_loop_invariants`) and hands the rebuilt
    dict to ``make_core_step(ctx, invariants=...)`` inside the trace, so they
    are not compiled into the program as grid-sized constants.
    """
    from rfx.model.materials import with_components
    ctx = replace(ctx, materials=with_components(ctx.materials, ctx.grid, periodic=ctx.periodic))
    materials = ctx.materials
    tensor = materials.components.tensor
    aniso_eps, aniso_inv_eps = (tensor.eps, tensor.inv_eps) if tensor else (None, None)
    # Stage 1's zero set is loop-invariant, like the update coefficients.
    # Keep it separate from pec_edge_masks: sheets use those for PEC ownership.
    conformal_zero_masks = None
    if ctx.use_conformal and not ctx.use_aniso_inv:
        conformal_zero_masks = tuple(w == 0.0 for w in tensor.conformal_weights)

    # #1043. ``apply_cpml_e`` builds its psi coefficient from a permittivity,
    # and it has to be the SAME one the Yee half of this timestep used, or the
    # two halves integrate different media and the combined update can
    # amplify (see that function's ``inv_eps_r_update`` docstring for the
    # derivation and the measured spectral radius). ``None`` on every path
    # that has no anisotropic array, which keeps those byte-identical.
    # With dispersion active the E update ignores anisotropic arrays,
    # so the CPML coefficient must also ignore them.
    #
    # Plain and dispersive updates share the realized E-edge permittivity.
    # #1260 made the DISPERSIVE update per-component as well: its ε_∞ is the
    # same ``component_e_materials`` mean, also threaded for dispersive runs.
    _aniso_is_live = not (ctx.use_debye or ctx.use_lorentz)
    if _aniso_is_live and aniso_inv_eps is not None:
        cpml_inv_eps_r = aniso_inv_eps
    elif _aniso_is_live and aniso_eps is not None:
        cpml_inv_eps_r = tuple(1.0 / e for e in aniso_eps)
    else:
        cpml_inv_eps_r = tuple(1.0 / e for e in materials.components.eps_update)

    # #677 surface-impedance sheet: Holland exponential-stepping A/B built
    # once from the FINAL run materials (background eps_r/sigma at the sheet
    # cells + the ctx's sigma_sheet), applied per step at tangential edges.
    _sheet_coeffs = None
    if ctx.use_sheet_impedance:
        from rfx.materials.thin_conductor import (
            sheet_update_coeffs as _sheet_update_coeffs,
        )
        _sheet_coeffs = _sheet_update_coeffs(
            ctx.sheet_impedance.sigma_sheet, materials, ctx.dt)
    from rfx.stepping.absorber import cpml_context
    ctx, e_loss = cpml_context(ctx)
    return {"ctx": ctx, "cpml_inv_eps_r": cpml_inv_eps_r if ctx.use_cpml else None,
            "sheet_coeffs": _sheet_coeffs, "e_loss": e_loss,
            "conformal_zero_masks": conformal_zero_masks}

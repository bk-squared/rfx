from __future__ import annotations

from rfx._internal._precision import resolve_coefficient_dtype


def _admit_waveguide_options(
    RfxUnsupported, conformal_pec, port_reference_sims, resolve_field_dtype, self, subpixel_smoothing,
):
    from rfx.model.wall_loss import refuse_wall_loss
    refuse_wall_loss(self, "waveguide S-matrix including multimode")
    if conformal_pec is not None and conformal_pec != "cut_cell_floor":
        raise RfxUnsupported(
            f"compute_waveguide_s_matrix received conformal_pec={conformal_pec!r}; "
            'conformal PEC on this entry is declared with Boundary(conformal=True); '
            'the keyword only takes "cut_cell_floor"')
    from rfx.runners._admission import admit_conformal_floor
    admit_conformal_floor(self, "waveguide_s_matrix", conformal_pec, smoothing=subpixel_smoothing)
    if conformal_pec is None:
        conformal_pec = bool(self._boundary_spec.conformal_faces())
    if conformal_pec == "cut_cell_floor" and port_reference_sims is not None:
        raise NotImplementedError('conformal_pec="cut_cell_floor" cannot carry port_reference_sims; '
                                  'use the uniform boundary-wall reference '
                                  '(https://github.com/bk-squared/rfx/issues/1373)')
    if conformal_pec and port_reference_sims is not None:
        raise RfxUnsupported('conformal PEC cannot carry port_reference_sims: reference volume '
                             'operands are unavailable (https://github.com/bk-squared/rfx/issues/1373)')
    from rfx.boundaries.features import refuse_traced_guide_mesh
    refuse_traced_guide_mesh(self, "compute_waveguide_s_matrix")
    resolve_field_dtype(self._precision)
    from rfx.model.smoothing_admission import admit_full_tensor
    admit_full_tensor(self, subpixel_smoothing, unsupported=True)
    from rfx.runners._smoothing_admission import warn_kottke_pec_volume
    warn_kottke_pec_volume(self, subpixel_smoothing)
    return conformal_pec


def _validate_waveguide_ports(
    eps_override, normalize, port_reference_sims, self, sigma_override,
):
    if self._ports or self._tfsf:
        raise ValueError(
            "compute_waveguide_s_matrix() is not supported together with a lumped or wire port, or a soft source (add_source) or TFSF"
        )
    if self._periodic_axes:
        raise ValueError(
            "compute_waveguide_s_matrix() is not supported with manual periodic-axis overrides"
        )
    if len(self._waveguide_ports) < 2:
        raise ValueError(
            "compute_waveguide_s_matrix() requires at least two waveguide ports"
        )

    entries = list(self._waveguide_ports)
    if any(entry.probe_plane is not None for entry in entries):
        raise ValueError(
            "compute_waveguide_s_matrix() does not use per-port probe_plane; use reference_plane only or leave probe_plane unset"
        )
    if any(entry.calibration_preset not in (None, "measured") for entry in entries):
        raise ValueError(
            "compute_waveguide_s_matrix() currently supports only measured/default reference planes or explicit reference_plane overrides"
        )

    if port_reference_sims is not None:
        if normalize != "flux":
            raise ValueError(
                "port_reference_sims requires normalize='flux' — the "
                "per-port reference feeds the flux P_inc normalization"
            )
        if any(entry.n_modes > 1 for entry in entries):
            raise NotImplementedError(
                "port_reference_sims is not supported with multimode ports "
                "(n_modes>1): the multimode extractor has no per-port-"
                "reference support"
            )
        if eps_override is not None or sigma_override is not None:
            raise NotImplementedError(
                "port_reference_sims combined with eps_override / "
                "sigma_override is an unvalidated combination and is not "
                "supported"
            )
        if len(port_reference_sims) != len(entries):
            raise ValueError(
                "port_reference_sims must supply one Simulation per "
                f"waveguide port ({len(entries)}), got "
                f"{len(port_reference_sims)}"
            )
    return entries


def _validate_waveguide_nu(
    checkpoint_segments, entries, eps_override, normalize, port_reference_sims, sigma_override, subpixel_smoothing,
):
    if checkpoint_segments is not None and checkpoint_segments < 1:
        raise ValueError(
            f"checkpoint_segments must be >= 1, got {checkpoint_segments}"
        )
    unsupported = []
    if normalize is not True and normalize != "flux":
        unsupported.append("normalize=True or normalize='flux' is required")
    if any(entry.n_modes > 1 for entry in entries):
        unsupported.append("multi-mode ports (n_modes>1) are not supported")
    if eps_override is not None and normalize != "flux":
        unsupported.append(
            "eps_override (differentiable AD channel) on the NU path "
            "requires normalize='flux'"
        )
    if sigma_override is not None and normalize != "flux":
        unsupported.append(
            "sigma_override (differentiable AD channel) on the NU path "
            "requires normalize='flux'"
        )
    if subpixel_smoothing:
        unsupported.append("subpixel_smoothing is not supported")
    if port_reference_sims is not None:
        unsupported.append(
            "port_reference_sims (per-port straight-guide junction "
            "references) is not supported on the non-uniform lane"
        )
    if unsupported:
        raise NotImplementedError(
            "compute_waveguide_s_matrix() on a non-uniform mesh "
            "(dx_profile / dy_profile / dz_profile) supports "
            "normalize=True or "
            "normalize='flux' and single-mode ports. "
            + "; ".join(unsupported)
            + ". Drop the dx/dy/dz profile to use the uniform lane."
        )
    return


def _validate_waveguide_uniform(
    eps_override, is_tracer, normalize, port_reference_sims, self, sigma_override,
):
    for model in (self, *(port_reference_sims or ())):
        if model._lumped_rlc:
            raise NotImplementedError(
                "compute_waveguide_s_matrix() on a uniform mesh does not "
                "implement lumped RLC elements in the device or reference "
                "simulations. Use run() or forward() for fields "
                "with these elements, or remove them from the S-matrix model.")

    self._require_no_refinement_without_a_subgrid(
        "compute_waveguide_s_matrix()")

    if normalize and normalize != "flux":
        for _ov_name, _ov_val in (
            ("eps_override", eps_override),
            ("sigma_override", sigma_override),
        ):
            if _ov_val is not None and is_tracer(_ov_val):
                raise NotImplementedError(
                    "compute_waveguide_s_matrix(normalize=True) is the "
                    "host-assembled two-run lane and is outside the "
                    f"differentiable chain: a traced {_ov_name} (under "
                    "jax.grad / jax.jit) cannot pass its np.array sites. "
                    "Use normalize=False or normalize='flux' to "
                    f"differentiate with respect to {_ov_name}."
                )
    return


def _validate_waveguide_materials(
    conformal_pec, debye_spec, eps_override, lorentz_spec, pec_shapes, self, subpixel_smoothing,
):
    from rfx.model.tensor_operands import refuse_full_tensor_conformal
    refuse_full_tensor_conformal(
        subpixel_smoothing, conformal_pec, pec_shapes)
    if eps_override is not None and subpixel_smoothing:
        self._refuse_unsupported_run_kwargs(
            "waveguide S-matrix with eps_override",
            {"subpixel_smoothing": subpixel_smoothing},
            instead="declare the permittivity as a material instead of "
                    "eps_override",
            entry="compute_waveguide_s_matrix()",
            reason_overrides={
                "subpixel_smoothing":
                    "the smoothed permittivity tensor is built from the "
                    "declared shapes and replaces the cell permittivity in "
                    "the E update, so eps_override would not reach the "
                    "solve",
            })
    if debye_spec is not None or lorentz_spec is not None:
        self._refuse_unsupported_run_kwargs(
            "waveguide S-matrix with Debye/Lorentz materials", {
                "subpixel_smoothing": subpixel_smoothing,
                "conformal_pec": conformal_pec if pec_shapes else False,
            },
            instead="remove the Debye/Lorentz poles",
            entry="compute_waveguide_s_matrix()",
            reason_overrides={
                "subpixel_smoothing":
                    "the dispersive E update does not read the smoothed "
                    "per-component permittivity tensor, so interfaces "
                    "would get scalar eps",
                "conformal_pec":
                    "Boundary(conformal=True) asks for Dey-Mittra walls, "
                    "and the dispersive E update does not read their eps "
                    "correction, so the walls would be staircased",
            },
            remedy_overrides={
                "conformal_pec": "drop Boundary(conformal=True) "
                                 "(staircase PEC)",
            })
    return


def _prepare_waveguide_conductors(
    conductors, subpixel_smoothing, conformal_weights,
):
    from rfx.model.conductors import stepped_conductors
    conductors = stepped_conductors(conductors, conformal_operands=conformal_weights)
    _wg_pec_edge_masks = conductors.pec_edges
    _wg_geometry_pec_edges = _wg_pec_edge_masks
    _wg_sheet_ctx = conductors.sheet_context(_wg_pec_edge_masks)
    if _wg_sheet_ctx is not None and subpixel_smoothing:
        raise ValueError(
            "surface-impedance (surface_impedance_f0) sheets are not "
            "supported with subpixel_smoothing / conformal on the "
            "waveguide S-matrix method: the sheet operator "
            "assumes the plain isotropic E update at its edges.")
    _use_kottke_pec_early = (subpixel_smoothing == "kottke_pec")
    if _use_kottke_pec_early:
        _wg_pec_edge_masks = None
    return _wg_pec_edge_masks, _wg_geometry_pec_edges, _wg_sheet_ctx


def _prepare_waveguide_references(
    grid, port_reference_sims,
):
    ref_materials_per_port = None
    ref_pec_edge_masks_per_port = None
    ref_pec_masks = []
    if port_reference_sims is not None:
        ref_materials_per_port = []
        ref_pec_edge_masks_per_port = []
        for _i, _ref_sim in enumerate(port_reference_sims):
            _ref_sim._require_no_refinement_without_a_subgrid(
                f"compute_waveguide_s_matrix(port_reference_sims[{_i}])")
            _ref_grid = _ref_sim._build_grid()
            if _ref_grid.shape != grid.shape or float(_ref_grid.dx) != float(grid.dx):
                raise ValueError(
                    f"port_reference_sims[{_i}] grid "
                    f"(shape={_ref_grid.shape}, dx={_ref_grid.dx}) must "
                    f"match the device grid (shape={grid.shape}, "
                    f"dx={grid.dx})"
                )
            _ref_pec_sheets: list = []
            _ref_pec_wires: list = []
            _ref_sheet_specs: list = []
            if any(tc.surface_impedance_f0 is not None for tc in _ref_sim._thin_conductors):
                raise NotImplementedError(
                    f"port_reference_sims[{_i}] declares a surface-impedance sheet "
                    "(add_thin_conductor(..., surface_impedance_f0=...)); the reference "
                    "run does not model sheet impedance and would run without it. "
                    "Remove it from the reference simulation.")
            from rfx.model.conductors import solve_conductors, assembled_materials
            from rfx.model.realization_table import finish_table
            reference = solve_conductors(_ref_sim, _ref_grid)
            _ref_base, _ref_debye, _ref_lorentz, _ref_pec_mask, _, _, _ = assembled_materials(
                reference, pec_sheets=_ref_pec_sheets,
                pec_wires=_ref_pec_wires, sheet_specs=_ref_sheet_specs)
            if _ref_sheet_specs:
                raise NotImplementedError(
                    f"port_reference_sims[{_i}] declares a surface-impedance sheet "
                    "(add_thin_conductor(..., surface_impedance_f0=...)); the reference "
                    "run does not model sheet impedance and would run without it. "
                    "Remove it from the reference simulation.")
            if _ref_debye is not None or _ref_lorentz is not None:
                raise NotImplementedError(
                    f"port_reference_sims[{_i}] declares a dispersive (Debye or "
                    "Lorentz) material; the reference run does not model "
                    "dispersion and would run it as a constant permittivity. "
                    "Use non-dispersive materials in reference simulations.")
            ref_materials_per_port.append(_ref_base)
            ref_pec_masks.append(_ref_pec_mask)
            _ref_edges_i = None
            if (_ref_pec_mask is not None or _ref_pec_sheets
                    or _ref_pec_wires):
                from rfx.boundaries.pec import realized_pec_edge_masks
                _ref_edges_i = realized_pec_edge_masks(
                    _ref_pec_mask, sheets=tuple(_ref_pec_sheets),
                    wires=tuple(_ref_pec_wires))
            finish_table(_ref_sim, reference, lane="waveguide_s_matrix_reference")
            ref_pec_edge_masks_per_port.append(_ref_edges_i)
    return ref_materials_per_port, ref_pec_edge_masks_per_port, ref_pec_masks


def _prepare_waveguide_configs(
    _reference_medium, debye, entries, freqs, grid, lorentz, materials,
    n_steps, normalize, pec_mask_wg, ref_materials_per_port, ref_pec_masks, self,
):
    has_multimode = any(entry.n_modes > 1 for entry in entries)
    raw_cfgs = [self._build_waveguide_port_config(entry, grid, freqs, n_steps) for entry in entries]
    from rfx.boundaries.features import resolve_waveguide_section_medium
    for _i, (_ref_base, _ref_pec_mask) in enumerate(zip(ref_materials_per_port or (), ref_pec_masks)):
        try:
            resolve_waveguide_section_medium(
                raw_cfgs, grid, _ref_base, dict(debye=None, lorentz=None, pec_mask=_ref_pec_mask), ())
        except (ValueError, NotImplementedError) as exc:
            raise type(exc)(f"port_reference_sims[{_i}]: {exc}") from exc
    raw_cfgs = resolve_waveguide_section_medium(
        raw_cfgs, grid, materials, dict(debye=debye, lorentz=lorentz, pec_mask=pec_mask_wg), ())
    reference_medium = _reference_medium(raw_cfgs, normalize)
    return has_multimode, raw_cfgs, reference_medium


def _extract_waveguide_multimode(
    _homogeneous_reference_materials, _wg_pec_edge_masks, aniso_eps, aniso_inv_eps, conformal_weights, debye, entries,
    extract_multimode_s_matrix, extract_multimode_s_matrix_flux, grid, lorentz, materials, n_steps, normalize,
    np, raw_cfgs, ref_aniso_eps, ref_aniso_inv_eps, ref_conformal, reference_medium, resolve_reported_planes,
    self, waveguide_plane_positions,
):
    """Returned reference planes are absolute de-embedding target coordinates."""
    settling_details_mm = []
    port_mode_cfgs: list[list] = []
    for entry, raw in zip(entries, raw_cfgs):
        if isinstance(raw, list):
            port_mode_cfgs.append(raw)
        else:
            port_mode_cfgs.append([raw])

    ref_shifts_mm = []
    desired_refs_mm = []
    for entry, mode_cfgs in zip(entries, port_mode_cfgs):
        first_cfg = mode_cfgs[0]
        planes = waveguide_plane_positions(first_cfg)
        desired_ref = resolve_reported_planes(entry, first_cfg).reference
        ref_shifts_mm.append(desired_ref - planes["reference"])
        desired_refs_mm.append(desired_ref)

    mm_pec_axes = None
    if normalize == "flux":
        from rfx.core.yee import init_materials as _init_vacuum_materials
        ref_materials = _homogeneous_reference_materials(grid.shape, reference_medium, dtype=resolve_coefficient_dtype(self._precision))
        s_params, mode_map = extract_multimode_s_matrix_flux(
            grid,
            materials,
            ref_materials,
            port_mode_cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes=mm_pec_axes,
            debye=debye,
            lorentz=lorentz,
            ref_shifts=ref_shifts_mm,
            aniso_eps=aniso_eps, ref_aniso_eps=ref_aniso_eps,
            ref_aniso_inv_eps=ref_aniso_inv_eps, ref_conformal=ref_conformal,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=_wg_pec_edge_masks,
            field_dtype=self._resolve_field_dtype(),
            settling_details=settling_details_mm,
        )
    elif normalize:
        raise ValueError(
            "compute_waveguide_s_matrix(normalize=True) is not "
            "supported with n_modes > 1 (multi-mode modal reference "
            "subtraction is not implemented). Use "
            "normalize='flux' for multi-mode S-matrices."
        )
    else:
        s_params, mode_map = extract_multimode_s_matrix(
            grid,
            materials,
            port_mode_cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes=mm_pec_axes,
            debye=debye,
            lorentz=lorentz,
            ref_shifts=ref_shifts_mm,
            aniso_eps=aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=_wg_pec_edge_masks,
            field_dtype=self._resolve_field_dtype(),
            settling_details=settling_details_mm,
        )
    reference_planes = np.array(desired_refs_mm, dtype=float)
    port_names_mm = []
    port_directions_mm = []
    for port_idx, mode_idx, mtype, m_n in mode_map:
        entry = entries[port_idx]
        port_names_mm.append(f"{entry.name}_mode{mode_idx}_{mtype}{m_n[0]}{m_n[1]}")
        port_directions_mm.append(entry.direction)
    settling_db_mm = np.asarray([row['db'] for row in settling_details_mm])
    return s_params, reference_planes, port_names_mm, port_directions_mm, settling_db_mm, settling_details_mm


def _prepare_waveguide_apertures(
    _slices_overlap, cfgs, entries, resolve_reported_planes, waveguide_plane_positions,
):
    by_direction = {}
    for entry, cfg in zip(entries, cfgs):
        by_direction.setdefault(entry.direction, []).append(cfg)

    for direction, side_cfgs in by_direction.items():
        plane_indices = {cfg.x_index for cfg in side_cfgs}
        if len(plane_indices) != 1:
            raise ValueError(
                f"waveguide ports on boundary {direction} must share one boundary plane"
            )
        for i in range(len(side_cfgs)):
            for j in range(i + 1, len(side_cfgs)):
                if _slices_overlap((side_cfgs[i].u_lo, side_cfgs[i].u_hi), (side_cfgs[j].u_lo, side_cfgs[j].u_hi)) and _slices_overlap((side_cfgs[i].v_lo, side_cfgs[i].v_hi), (side_cfgs[j].v_lo, side_cfgs[j].v_hi)):
                    raise ValueError(
                        f"waveguide ports on the same {direction} boundary must have disjoint apertures"
                    )

    ref_shifts = []
    for entry, cfg in zip(entries, cfgs):
        planes = waveguide_plane_positions(cfg)
        desired_ref = resolve_reported_planes(entry, cfg).reference
        ref_shifts.append(desired_ref - planes["reference"])
    return ref_shifts

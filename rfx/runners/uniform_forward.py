"""_forward_from_materials moved from rfx/api/_execute.py in Block E1, a pure move; the body is unchanged."""
from __future__ import annotations

from rfx._internal._precision import resolve_coefficient_dtype
from rfx.measurement.calculator_drive import calculator_drive

from rfx.errors import RfxUnsupported

from rfx.api._spec import ForwardResult
from rfx.sources.sources import GaussianPulse
from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.model import conductors as _conductors
from rfx.api._dispatch import _declaration_setup_at_trace_time
from rfx._internal import _grid_metric
from rfx.preflight.line_stub import line_stub_admission as _line_stub_admit
from rfx.model import occupancy as _occupancy
from rfx.api._refplane_checks import (
    _refplane_conductor_mask,
    _refplane_reject_msl_ladder_in_plane_zone,
    _refplane_reject_planes_off_the_uniform_line,
)
from rfx.measurement.plane_records import time_recorded_dft_selection, named_plane_records

import jax.numpy as jnp
import numpy as np
from rfx.api._forward_result import pack_uniform_forward_result
from rfx._diagnostic_transport import result_with_diagnostics


@_declaration_setup_at_trace_time
def _forward_from_materials(
    self,
    grid: Grid,
    materials: MaterialArrays,
    debye_spec: tuple | None,
    lorentz_spec: tuple | None,
    *,
    subpixel_smoothing: bool | str = False,
    pec_shapes=None,
    diagnostics=(),
    n_steps: int,
    conductors=None,
    tfsf_angle_override=None,
    checkpoint: bool = True,
    gradient: str = "autodiff",
    checkpoint_segments: int | None = None,
    pec_mask: jnp.ndarray | None = None,
    pec_sheets: object = (),
    pec_wires: object = (),
    pec_occupancy: jnp.ndarray | None = None,
    kerr_chi3: jnp.ndarray | None = None,
    port_s11_freqs: object | None = None,
    rlc_values_override: dict | None = None,
    _sparam_drive_idx: int | None = None,
    _return_raw_port_sparams: bool = False,
    sheet_impedance: object | None = None,
    design_box: object | None = None,
    design_occupancy: object | None = None,
    monitor_overrides: dict | None = None,
    lane: str | None = None,
    conformal_pec: bool | None = None,
    conformal_min_weight: float = 0.1,
) -> ForwardResult | dict:
    """Run a minimal differentiable forward path from explicit materials.

        ``lane`` is ``"fwd_uniform"`` when ``forward()`` calls: a declared
        input that lane does not carry is refused before the first step
        (``rfx.runners._admission``). The calculators that also enter here
        (``run()``'s lumped/wire S-matrix scan, ``compute_mixed_s_matrix``,
        ``topology_optimize``) admit their model at their own entry and
        leave it ``None`` here.

        Internal multi-drive S-matrix hook (item-5 Stage 1)
        --------------------------------------------------------------
        ``_sparam_drive_idx`` and ``_return_raw_port_sparams`` exist solely
        for the production-scan S-matrix driver
        (``rfx.probes.sparam_driver``).  Their defaults preserve the current
        ``forward()`` / ``run()`` behavior EXACTLY:

        * ``_sparam_drive_idx`` (default ``None``): when set to an integer,
          build the excitation source for ONLY the sparam-eligible
          lumped/wire port at that 0-based index (in registration order over
          the lumped+wire ports), ignoring every port's ``excite`` flag.  All
          other ports remain matched loads (their impedance is still folded
          into materials via ``setup_lumped_port`` / ``setup_wire_port``).
          When ``None``, excitation follows ``pe.excite`` as before.
        * ``_return_raw_port_sparams`` (default ``False``): when ``True``,
          return ``{"lumped": result.lumped_port_sparams,
          "wire": result.wire_port_sparams, "freqs": ...}`` (the all-port
          ``(v_dft, i_dft)`` accumulators) instead of the diagonal
          ``ForwardResult``.  When ``False``, the normal ``ForwardResult`` is
          returned unchanged.

        ``conformal_pec`` resolves as in ``run()``: ``None`` reads
        ``Boundary(conformal=True)``. Uniform Yee scans form the same
        default conformal operands before entering the compiled loop.
        """
    from rfx.model.wall_loss import admit_wall_loss
    admit_wall_loss(self, "fwd_uniform", smoothing=subpixel_smoothing, conformal=conformal_pec)
    from rfx.runners._admission import refuse_mixed_waveguide_sources
    refuse_mixed_waveguide_sources(self)
    if self._waveguide_ports:
        from rfx.runners._admission import require_isolated_s_drive
        require_isolated_s_drive(self)
    from rfx.preflight.waveguide import resolve_shared_waveguide_port_freqs
    wg_freqs = resolve_shared_waveguide_port_freqs(self, route="uniform forward()")
    from rfx.model.tensor_operands import (
        refuse_tensor_conflicts, form_tensor_operands, finish_tensor_operands, finish_conformal_mu)
    from rfx.runners._admission import admit_conformal_floor
    admit_conformal_floor(self, "fwd_uniform", conformal_pec, grid=grid)
    _conformal = (bool(self._boundary_spec.conformal_faces())
                  if conformal_pec is None else bool(conformal_pec))
    if (subpixel_smoothing or _conformal) and sheet_impedance is not None:
        raise ValueError(
            "surface-impedance (surface_impedance_f0) sheets are not "
            "supported with subpixel_smoothing / conformal_pec"
            ": the sheet operator assumes the plain isotropic "
            "E update at its edges (Stage-1/2 conformal and Kottke "
            "lanes do not apply the sheet operator).")
    refuse_tensor_conflicts(self, grid, subpixel_smoothing=subpixel_smoothing,
        conformal_pec=_conformal, debye_spec=debye_spec, lorentz_spec=lorentz_spec,
        pec_shapes=pec_shapes, kerr_chi3=kerr_chi3)

    self._require_no_refinement_without_a_subgrid(
        "forward()/optimize()/topology_optimize()/compute_lumped_wire_s_matrix_via_scan()",
        diagnostics=diagnostics,
    )
    _line_stub_scope = _line_stub_admit(self, port_s11_freqs)

    from rfx.simulation import (
        run as _run, resolve_periodic,
        make_probe,
        make_port_source,
        make_wire_port_sources,
    )
    from rfx.sources.sources import (
        LumpedPort,
        WirePort,
        setup_lumped_port,
        setup_wire_port,
        _wire_port_cells,
    )

    rlc_metas = None
    if self._lumped_rlc:
        from rfx.lumped import (
            setup_rlc_materials,
            build_rlc_meta,
        )
        _rlc_ov = rlc_values_override or {}
        for _rlc_idx, _rlc_spec in enumerate(self._lumped_rlc):
            _v = _rlc_ov.get(_rlc_idx, {})
            materials = setup_rlc_materials(
                grid, _rlc_spec, materials,
                r_val=_v.get("R"), c_val=_v.get("C"),
            )

    aniso_eps, aniso_inv_eps, conformal_weights, unextendable, conformal_areas = form_tensor_operands(
        self, grid, materials, subpixel_smoothing=subpixel_smoothing,
        conformal=_conformal, conformal_min_weight=conformal_min_weight, pec_shapes=pec_shapes)
    if conformal_weights is not None:
        pec_mask = None
    tensor_smoothing = subpixel_smoothing if aniso_eps is not None or aniso_inv_eps is not None else False
    if subpixel_smoothing:
        from rfx.geometry.smoothing import warn_unextendable_shapes
        warn_unextendable_shapes([u for u in unextendable if not u.conductor])

    from rfx.sources.port_drive import PortSourceQueue
    sources = PortSourceQueue()
    mag_sources: list = []
    probes = []
    periodic_bool = resolve_periodic(grid, self._periodic_flags())

    cpml_axes_run = grid.cpml_axes
    pec_axes_run = None

    tfsf_run = None
    if self._tfsf is not None:
        from rfx.sources.tfsf import tfsf_from_declaration
        tfsf_run = tfsf_from_declaration(
            self._tfsf, grid, self._freq_max, n_steps=n_steps,
        dtype=resolve_coefficient_dtype(self._precision))
        if tfsf_angle_override is not None:
            from rfx.sources.tfsf_2d import with_angle
            tfsf_run = (with_angle(tfsf_run[0], tfsf_angle_override), tfsf_run[1])
        from dataclasses import replace
        from rfx.api._ad_memory_model import face_record_pays
        if face_record_pays(tfsf_run[0], n_steps, checkpoint=checkpoint,
                            checkpoint_segments=checkpoint_segments):
            tfsf_run = (replace(tfsf_run[0], record_faces=True), tfsf_run[1])
        from rfx.boundaries.tfsf import boundary_flags as tfsf_boundary_flags
        periodic_bool, cpml_axes_run = tfsf_boundary_flags(tfsf_run[0], grid)
        pec_axes_run = None


    pec_sheets = tuple(pec_sheets or ())
    pec_wires = tuple(pec_wires or ())
    pec_mask_local = pec_mask
    pec_occupancy_local = pec_occupancy
    from rfx.model.source_admission import conductor_operator
    conductors, _drawn_materials, pec_edge_masks_local = _conductors.forward_products(
        self, grid, materials, pec_mask, pec_sheets, pec_wires, periodic_bool,
        conductors, sheet_impedance,
        operator=conductor_operator(smoothing=subpixel_smoothing, conformal=_conformal))
    if self._coaxial_ports:
        from rfx.model.coax_port import assemble_coax_port
        materials, conductors, pec_occupancy_local = assemble_coax_port(
            self, grid, materials, conductors, sources, mag_sources, n_steps=n_steps,
            occupancy=pec_occupancy_local)
        from rfx.model.coax_admission import refuse_design_box
        refuse_design_box(design_box, conductors)
        pec_mask_local = conductors.pec_cells
        pec_edge_masks_local = conductors.pec_edges
    from rfx.measurement.prepared import attach_measurement
    attach_measurement(self, conductors, n_steps=n_steps, path=lane or 'fwd_uniform',
                       frequencies=port_s11_freqs,
                       default_probes=() if self._probes else tuple(
                           (i, p.position, p.component) for i, p in enumerate(self._ports)),
                       overrides={**(monitor_overrides or {}), 'design_box': design_box,
                                  'design_occupancy': design_occupancy})
    _msl_geometry_edges = pec_edge_masks_local
    from rfx.measurement import ordered_specs, register_lumped, register_wire
    port_record_specs = []
    lumped_port_sparam_specs: list = []
    wire_port_sparam_specs: list = []
    wire_refplane_specs: list = []
    if port_s11_freqs is not None:
        _s11_freqs_arr = jnp.asarray(port_s11_freqs, dtype=jnp.float32)
    else:
        _s11_freqs_arr = None

    _sparam_port_idx = -1

    def _excite_this_port(pe) -> bool:
        if _sparam_drive_idx is None:
            return pe.excite
        return calculator_drive(_sparam_port_idx == _sparam_drive_idx, excite=pe.excite)

    frozen_drive_requests = []
    for _port_index, pe in enumerate(self._ports):
        if pe.is_soft_source:
            from rfx.sources.port_drive import defer_soft_source
            defer_soft_source(sources, grid, pe, n_steps,
                              field_dtype=self._resolve_field_dtype())
            if pe.amplitude_kind != "field":
                frozen_drive_requests.append((
                    f"Soft source _ports[{_port_index}]", pe.position, pe.component,
                    (_grid_metric.field_index(grid, pe.position, pe.component),)))
            continue

        _sparam_port_idx += 1

        _drive_this_port = _excite_this_port(pe)
        _drive_waveform = pe.waveform
        if _drive_this_port and _drive_waveform is None:
            _drive_waveform = GaussianPulse(
                f0=self._freq_max / 2, bandwidth=0.8)

        if pe.extent is not None:
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            axis = axis_map[pe.component]
            end = list(pe.position)
            end[axis] += pe.extent
            wp = WirePort(
                start=pe.position,
                end=tuple(end),
                component=pe.component,
                impedance=pe.impedance,
                excitation=_drive_waveform,
                radius=pe.radius, allow_in_absorber=pe.allow_in_absorber,
            )
            materials = setup_wire_port(
                grid, wp, materials,
                pec_edge_masks=pec_edge_masks_local,
                radius_sim=self, radius_port=f"port[{_port_index}]")
            if _drive_this_port:
                sources.defer(make_wire_port_sources,
                    grid, wp, n_steps=n_steps,
                    pec_edge_masks=pec_edge_masks_local)
                frozen_drive_requests.append((
                    f"Wire port _ports[{_port_index}]", pe.position, pe.component,
                    tuple(wp._drive_stamps)))
            wp_cells = _wire_port_cells(grid, wp)
            from rfx.sources.sources import _wire_port_live_cells
            _, wp_live_flags, _ = _wire_port_live_cells(
                grid, wp, pec_edge_masks_local)
            _wp_live_cells = [c for c, _l in zip(wp_cells, wp_live_flags)
                              if _l]
            conductors, pec_mask_local, pec_edge_masks_local = _conductors.forward_port_stage(conductors, _wp_live_cells, pe.component, f"port[{_port_index}]", port_kind="wire", release_edges=False)
            register_wire(port_record_specs, wire_port_sparam_specs, pe.component,
                          wp_cells, wp_live_flags, _s11_freqs_arr, pe.impedance, _drive_this_port, _port_index)
            if (pe.reference_plane_cells is not None
                    and _s11_freqs_arr is not None
                    and _sparam_drive_idx is not None
                    and wp_cells):
                from rfx.probes.refplane import build_wire_refplane_specs
                _axis_of = {"x": 0, "y": 1, "z": 2}
                _l_ax = _axis_of[pe.direction[1]]
                _sign = -1 if pe.direction[0] == "+" else +1
                _i_port = int(grid.position_to_index(pe.position)[_l_ax])
                _i_far = _i_port + _sign * 2 * int(pe.reference_plane_cells)
                for _other in self._ports:
                    if _other is pe or _other.is_soft_source:
                        continue
                    _oi = int(grid.position_to_index(
                        _other.position)[_l_ax])
                    _in_zone = (_i_port < _oi <= _i_far) if _sign > 0 \
                        else (_i_far <= _oi < _i_port)
                    if _in_zone:
                        raise ValueError(
                            "reference_plane_cells: the reference "
                            f"planes of the port at {pe.position} "
                            f"(indices up to {_i_far} on axis {_l_ax}) "
                            "reach past another port at "
                            f"{_other.position} (index {_oi}). Reduce "
                            "N so both planes stay on the uniform "
                            "line between the ports. Note: this check "
                            "compares line-axis indices only "
                            "(conservative), so a TRANSVERSELY "
                            "separated port on a different parallel "
                            "trace also trips it — the check can be "
                            "revisited if that layout is needed.")
                if self._msl_ports:
                    _refplane_reject_msl_ladder_in_plane_zone(
                        self, grid, pe, line_axis=_l_ax,
                        outboard_sign=_sign, i_port=_i_port,
                        i_far=_i_far)
                _plane_specs = build_wire_refplane_specs(
                    grid=grid,
                    port_cells=wp_cells,
                    e_component=pe.component,
                    impedance=float(pe.impedance),
                    direction=pe.direction,
                    n_cells=int(pe.reference_plane_cells),
                    freqs=_s11_freqs_arr,
                    port_index=_sparam_port_idx,
                    pec_mask=_refplane_conductor_mask(
                        pec_mask, sheet_impedance, pec_sheets),
                )
                _refplane_reject_planes_off_the_uniform_line(
                    grid, pe, _plane_specs, line_axis=_l_ax)
                wire_refplane_specs.extend(_plane_specs)
            continue

        lp = LumpedPort(
            position=pe.position,
            component=pe.component,
            impedance=pe.impedance,
            excitation=_drive_waveform,
            allow_in_absorber=pe.allow_in_absorber,
        )
        materials = setup_lumped_port(grid, lp, materials)
        if _drive_this_port:
            sources.defer(make_port_source, grid, lp, n_steps=n_steps)
            frozen_drive_requests.append((
                f"Lumped port _ports[{_port_index}]", pe.position, pe.component,
                tuple(lp._drive_stamps)))
        idx = _grid_metric.field_index(grid, pe.position, pe.component)
        conductors, pec_mask_local, pec_edge_masks_local = _conductors.forward_lumped_port_stage(conductors, pe, f"port[{_port_index}]")
        register_lumped(port_record_specs, lumped_port_sparam_specs, pe.component,
                        idx, _s11_freqs_arr, pe.impedance, _drive_this_port, _port_index)

    if self._msl_ports:
        from rfx.sources.msl_port import (
            _msl_yz_cells,
            msl_normal_component as _msl_normal_component,
            compute_msl_mode_profile,
            make_msl_port_sources,
            msl_cell,
            msl_cross_section_span,
            msl_port_from_entry,
            setup_msl_port,
        )
        _static_eps_483 = None
        if any(pe.eps_r_sub is None and getattr(pe, "mode", "uniform") == "laplace"
               for pe in self._msl_ports):
            _static_eps_483 = _conductors.drawn_launch_eps(self, grid, _drawn_materials)
        for _msl_port_index, pe in enumerate(self._msl_ports):
            mp = msl_port_from_entry(pe)
            from rfx.sources.msl_port import validate_msl_port_geometry
            validate_msl_port_geometry(
                grid, mp, pec_edge_masks=_msl_geometry_edges,
                sheet_impedance=sheet_impedance, periodic=periodic_bool,
                pec_faces=self._boundary_spec.pec_faces(), name=pe.name)
            port_mode = getattr(pe, "mode", "uniform")
            mode_profile = None
            if port_mode == "laplace":
                span = msl_cross_section_span(grid, mp)
                k_mid = (span["n_lo"] + span["n_hi"]) // 2
                eps_cell = msl_cell(pe.direction, span["i_feed"],
                                    span["w_centre"], k_mid)
                if pe.eps_r_sub is not None:
                    eps_r_sub = float(pe.eps_r_sub)
                else:
                    eps_r_sub = float(np.asarray(
                        _static_eps_483[eps_cell]
                    ))
                mode_profile = compute_msl_mode_profile(grid, mp, eps_r_sub)
            elif port_mode == "eigenmode":
                raise NotImplementedError(
                    "MSL port mode='eigenmode' is supported on the "
                    "imperative Simulation.run() path "
                    "(`runners/uniform.py::run_uniform_path`) but not "
                    "yet on the differentiable Simulation.forward() "
                    "path — the MSL J+M Schelkunoff source pair is not "
                    "wired into its magnetic-source channel. "
                    "Use mode='laplace' "
                    "(the default) on the differentiable path."
                )
            materials = setup_msl_port(grid, mp, materials,
                                       mode_profile=mode_profile)
            if pe.excite and pe.waveform is not None:
                frozen_drive_requests.append((
                    f"msl_port[{_msl_port_index}]", pe.position, "ez",
                    tuple(mp._drive_stamps)))
                sources.defer(make_msl_port_sources,
                    grid, mp, n_steps=n_steps,
                    mode_profile=mode_profile, dtype=resolve_coefficient_dtype(self._precision),
                )
            _msl_cells = list(_msl_yz_cells(grid, mp))
            conductors, pec_mask_local, pec_edge_masks_local = _conductors.forward_port_stage(
                conductors, _msl_cells, _msl_normal_component(mp), f"msl_port[{_msl_port_index}]",
                port_kind="microstrip", skip_empty=True, released_edges=(mode_profile["cell_indices"] if mode_profile is not None else _msl_cells))

    for pe in self._probes:
        probes.append(make_probe(grid, pe.position, pe.component))

    if not probes and self._ports:
        for pe in self._ports:
            probes.append(make_probe(grid, pe.position, pe.component))

    from rfx.measurement.identification import read_ringdown_probes
    for pe in read_ringdown_probes(self._probes, "runners/uniform_forward.py"):
        probes.append(make_probe(grid, pe.position, pe.component))

    from rfx.model.materials import with_components
    materials = with_components(materials, grid, periodic=periodic_bool,
        debye_spec=debye_spec, lorentz_spec=lorentz_spec,
        field_dtype=self._resolve_field_dtype() or jnp.float32)
    materials = finish_conformal_mu(materials, conformal_areas)
    aniso_eps = finish_tensor_operands(materials, aniso_eps, conformal_weights)
    _, debye, lorentz = self._init_dispersion(
        materials, grid.dt, debye_spec, lorentz_spec,
        periodic=periodic_bool, grid=grid,
    )

    ntff_box = None
    if self._ntff is not None:
        from rfx.farfield import make_ntff_box
        corner_lo, corner_hi, freqs = self._ntff
        ntff_box = make_ntff_box(grid, corner_lo, corner_hi, freqs)

    from rfx.current_moments import monitor_for_simulation as _cm_for_sim
    current_moments_fwd = _cm_for_sim(
        self, grid, periodic_bool, conductors=conductors,
        overrides={**(monitor_overrides or {}), "design_box": design_box,
                   "design_occupancy": design_occupancy})

    flux_monitor_cfgs = []
    if self._flux_monitors:
        from rfx.runners.uniform import build_flux_monitor_cfgs
        flux_monitor_cfgs = build_flux_monitor_cfgs(self, grid, n_steps, periodic=periodic_bool)

    from rfx.measurement.setup import build_planes
    dft_planes = build_planes(self, grid, n_steps)

    waveguide_ports = []
    if self._waveguide_ports:
        for pe in self._waveguide_ports:
            waveguide_ports.append(
                self._build_waveguide_port_config(pe, grid, wg_freqs, n_steps))

    conductors, pec_occupancy_local = _conductors.prepare_port_occupancy(conductors, waveguide_ports, pec_occupancy_local, design_occupancy, sim=self, sheet_operator=sheet_impedance)

    pec_occupancy_for_run, aniso_inv_eps_run, _occupancy_tensor = _occupancy.select_occupancy_operator(
        self, grid, materials, pec_occupancy_local, periodic_bool, design_occupancy,
        debye=debye, lorentz=lorentz)
    _inv_eps_smooth = aniso_inv_eps is None and aniso_inv_eps_run is not None
    if aniso_inv_eps is not None:
        aniso_inv_eps_run = aniso_inv_eps
    from rfx.model.tensor_operands import attach_run_tensor
    materials = attach_run_tensor(materials, aniso_eps, aniso_inv_eps_run,
                                  conformal_weights, _inv_eps_smooth)
    from rfx.model.source_admission import refuse_frozen_drive_edges
    _run_tensor = getattr(materials.components, "tensor", None)
    refuse_frozen_drive_edges(_run_tensor, frozen_drive_requests)
    from rfx.model.source_coefficients import resolve_run_sources, drive_materials
    sources = resolve_run_sources(sources, materials, self, grid)

    if self._floquet_ports:
        from rfx.floquet import floquet_port_source
        for fpe in self._floquet_ports:
            sources.append(floquet_port_source(grid, fpe, self._freq_max, self._domain, n_steps, dtype=resolve_coefficient_dtype(self._precision))[1])

    if self._lumped_rlc:
        from rfx.lumped import build_rlc_meta
        _rlc_ov = rlc_values_override or {}
        rlc_metas = []
        from rfx.simulation import resolve_periodic as _resolve_periodic
        _rlc_periodic = _resolve_periodic(grid, periodic_bool)
        for _rlc_idx, _rlc_spec in enumerate(self._lumped_rlc):
            _v = _rlc_ov.get(_rlc_idx, {})
            rlc_metas.append(build_rlc_meta(
                grid, _rlc_spec, drive_materials(materials, self, grid),
                r_val=_v.get("R"), l_val=_v.get("L"), c_val=_v.get("C"),
                periodic=_rlc_periodic,
            ))
        from rfx.lumped import refuse_stacked_solved_elements
        refuse_stacked_solved_elements(self._lumped_rlc, rlc_metas)

    if self._tfsf is not None and self._tfsf.closed_box:
        from rfx.boundaries.tfsf import validate_realized_vacuum_boundary
        validate_realized_vacuum_boundary(
            self, grid, materials, tfsf_run[0], pec_edge_masks=pec_edge_masks_local,
            debye_spec=debye_spec, lorentz_spec=lorentz_spec, kerr_chi3=kerr_chi3,
            aniso_inv_eps=(_run_tensor.inv_eps if _run_tensor is not None else None),
            sheet_ctx=sheet_impedance,
            extra_masks=(pec_occupancy_local,), design_regions=(design_box, design_occupancy),
            rlc_indices=(_grid_metric.field_index(grid, spec.position, spec.component)
                         for spec in self._lumped_rlc))
    elif self._tfsf is not None:
        nonvacuum = []
        for spec in (design_box, design_occupancy):
            if spec is not None:
                b = spec.bounds
                window = tuple(slice(b[2 * d], min(b[2 * d + 1] + 1, grid.shape[d]))
                               for d in range(3))
                nonvacuum.append(jnp.zeros(grid.shape, dtype=bool).at[window].set(True))
        self._validate_tfsf_vacuum_boundary(
            materials, tfsf_run[0], nonvacuum_masks=tuple(nonvacuum))

    if lane is not None:
        from rfx.runners._admission import admit
        admit(
            self,
            lane,
            diagnostics=diagnostics,
            run_args={"subpixel_smoothing": subpixel_smoothing,
                      **({"tfsf_materials": materials} if self._tfsf is not None else {})},
        )

    from rfx.stepping import prepared as prepared_step
    if self._coaxial_ports and getattr(materials.components, 'tensor', None) is None:
        materials = materials._replace(components=None)
    conductors = prepared_step.stage(conductors, grid=grid, materials=materials)
    stepping_product = conductors
    if self._lumped_rlc:
        from rfx.model.source_admission import LumpedRLCInputs
        rlc_metas = LumpedRLCInputs(rlc_metas, self, stepping_product)
    from rfx.runners.uniform import assemble_wall_loss
    sheet_impedance = assemble_wall_loss(self, conductors, sheet_impedance, lane=lane or "fwd_uniform")
    conductors, pec_edge_masks_local, sheet_impedance = _conductors.forward_kernel_inputs(
        self, conductors, lane or "fwd_uniform", pec_edge_masks_local, sheet_impedance,
        realization_options=(subpixel_smoothing, _conformal))
    result = prepared_step.advance(stepping_product, _run,
        grid,
        materials,
        n_steps,
        boundary=self._boundary,
        cpml_axes=cpml_axes_run,
        pec_axes=pec_axes_run,
        periodic=periodic_bool,
        tfsf=tfsf_run, _feature_tfsf=True,
        debye=debye,
        lorentz=lorentz,
        sources=sources,
        mag_sources=mag_sources or None,
        probes=probes,
        waveguide_ports=waveguide_ports if waveguide_ports else None,
        ntff=ntff_box,
        current_moments=current_moments_fwd,
        checkpoint=checkpoint,
        checkpoint_segments=checkpoint_segments,
        pec_mask=pec_mask_local,
        pec_edge_masks=pec_edge_masks_local,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_occupancy=pec_occupancy_for_run,
        subpixel_smoothing=tensor_smoothing,
        port_record_specs=ordered_specs(port_record_specs),
        lumped_port_sparams=lumped_port_sparam_specs or None,
        wire_port_sparams=wire_port_sparam_specs or None,
        wire_refplane_sparams=wire_refplane_specs or None,
        lumped_rlc=rlc_metas,
        kerr_chi3=kerr_chi3,
        dft_planes=dft_planes if dft_planes else None,
        record_dft=time_recorded_dft_selection(self),
        flux_monitors=flux_monitor_cfgs if flux_monitor_cfgs else None,
        return_state=False,
        stencil_order=self._stencil_order,
        sheet_impedance=sheet_impedance,
        design_box=design_box,
        gradient=gradient,
        design_occupancy=design_occupancy,
        field_dtype=self._resolve_field_dtype(),
    )

    from rfx.measurement.echo_bound import runner_echo_bound
    from rfx.measurement.read_bins import declared_read_bins
    echo_bound = runner_echo_bound(self, grid, materials, freqs=declared_read_bins(self, _s11_freqs_arr, waveguide_ports=waveguide_ports),
        debye=debye_spec, lorentz=lorentz_spec, smoothing=subpixel_smoothing,
        waveguide_ports=waveguide_ports, design_box=design_box)

    from rfx.measurement import reconstruct_oblique_physical
    from rfx.sources.tfsf import is_tfsf_2d
    if tfsf_run is not None and is_tfsf_2d(tfsf_run[0]):
        result = reconstruct_oblique_physical(result, tfsf_run[0], grid, probes)

    if _return_raw_port_sparams:
        _raw_freqs = None
        if result.lumped_port_sparams:
            _raw_freqs = result.lumped_port_sparams[0][0].freqs
        elif result.wire_port_sparams:
            _raw_freqs = result.wire_port_sparams[0][0].freqs
        return {
            "lumped": result.lumped_port_sparams,
            "wire": result.wire_port_sparams,
            "sparam_time_records": result.sparam_time_records,
            "dft_time_records": named_plane_records(self, result.dft_time_records),
            "wire_refplane": result.wire_refplane_sparams,
            "freqs": _raw_freqs,
            "dft_planes": (
                {
                    entry.name: probe
                    for entry, probe in zip(
                        self._dft_planes,
                        getattr(result, "dft_planes", None) or (),
                    )
                }
                if self._dft_planes else None
            ),
            "time_series": getattr(result, "time_series", None),
            "settling_echo_bound": echo_bound,
            "dt": getattr(result, "dt", grid.dt),
            "flux_monitors": (
                {
                    entry.name: fmv
                    for entry, fmv in zip(
                        self._flux_monitors,
                        getattr(result, "flux_monitors", None) or (),
                    )
                }
                if self._flux_monitors else None
            ),
        }

    coax_metadata = None
    s_params_out = getattr(result, "s_params", None)
    freqs_out = getattr(result, "freqs", None)
    if self._coaxial_ports and port_s11_freqs is not None:
        from rfx.measurement.coax_port import port_result
        s_params_out, freqs_out, coax_metadata = port_result(self, stepping_product, result)
    from rfx.measurement.port_s_contract import uniform_record_pairs
    pairs = uniform_record_pairs(result)
    if pairs:
        from rfx.measurement import port_s_from_record
        metas, channels = zip(*pairs)
        s_params_out = port_s_from_record(
            channels, metas, wire_mode=pairs is result.wire_port_sparams)
        freqs_out = metas[0].freqs

    if s_params_out is not None and (
        result.lumped_port_sparams or result.wire_port_sparams
    ):
        from rfx.probes.probes import warn_if_nonpassive_lumped_s11
        from rfx.measurement import observed_port_s
        measured = observed_port_s(s_params_out, pairs)
        if measured is s_params_out:
            measured = jnp.stack([s_params_out[k, k] for k in range(len(pairs))])
        warn_if_nonpassive_lumped_s11(
            measured, freqs_out,
            extractor="forward(port_s11_freqs=...)",
        )

    from rfx.sparams._record_selection import finish_forward
    return result_with_diagnostics(finish_forward(pack_uniform_forward_result(
        self, grid, result, diagnostics=diagnostics, s_params=s_params_out,
        freqs=freqs_out, current_moment_monitor=current_moments_fwd, coax_port_metadata=coax_metadata,
        realized_geometry=stepping_product._realization_record,
        settling_echo_bound=echo_bound,
    )._replace(tfsf_angle_deg=tfsf_angle_override)), conductors.design_findings)


_forward_from_materials.__qualname__ = "_ExecuteMixin._forward_from_materials"
_forward_from_materials.__wrapped__.__qualname__ = "_ExecuteMixin._forward_from_materials"


def refuse_sheet_layout():
    raise RfxUnsupported(
        "pinned_sheet_layout requires the single-device graded lane; "
        "a uniform mesh can be passed as a graded profile")

"""_forward_from_materials moved from rfx/api/_execute.py in Block E1, a pure move; the body is unchanged."""
from __future__ import annotations

from rfx.api._spec import ForwardResult
from rfx.sources.sources import GaussianPulse
from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.model import conductors as _conductors
from rfx.api._dispatch import _declaration_setup_at_trace_time
from rfx import _grid_metric
from rfx.preflight.line_stub import line_stub_admission as _line_stub_admit
from rfx.model import occupancy as _occupancy
from rfx.api._refplane_checks import (
    _refplane_conductor_mask,
    _refplane_reject_msl_ladder_in_plane_zone,
    _refplane_reject_planes_off_the_uniform_line,
)
import jax.numpy as jnp
import numpy as np
import os
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
) -> ForwardResult | dict:
    """Run a minimal differentiable forward path from explicit materials.

        ``lane`` is ``"fwd_uniform"`` when ``forward()`` calls: a declared
        input that lane does not carry is refused before the first step
        (``rfx.runners._admission``). The calculators that also enter here
        (``run()``'s lumped/wire S-matrix scan, ``compute_mixed_s_matrix``,
        ``topology_optimize``) admit their model at their own entry and
        leave it ``None`` here. The ADI route below is additionally judged
        as ``fwd_adi``.

        Internal multi-drive S-matrix hook (item-5 Stage 1, 2026-06-22)
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
        ``Boundary(conformal=True)``. Neither route here has a Dey-Mittra
        update, so a conformal request on a model with PEC to conform is
        refused before the first step; an explicit ``False`` asks for
        staircase PEC and passes (#1299).
        """
    if subpixel_smoothing and sheet_impedance is not None:
        raise ValueError(
            "surface-impedance (surface_impedance_f0) sheets are not "
            "supported with subpixel_smoothing / conformal_pec "
            "(#677 v1): the sheet operator assumes the plain isotropic "
            "E update at its edges (Stage-1/2 conformal and Kottke "
            "lanes are untouched by #677).")
    from rfx.model.tensor_operands import (
        refuse_tensor_conflicts, form_tensor_operands, finish_tensor_operands)
    refuse_tensor_conflicts(self, grid, subpixel_smoothing=subpixel_smoothing,
        conformal_pec=False, debye_spec=debye_spec, lorentz_spec=lorentz_spec,
        pec_shapes=pec_shapes, kerr_chi3=kerr_chi3)
    _conformal = (bool(self._boundary_spec.conformal_faces())
                  if conformal_pec is None else bool(conformal_pec))
    from rfx.model.tensor_operands import refuse_full_tensor_conformal
    refuse_full_tensor_conformal(subpixel_smoothing, _conformal, pec_shapes)
    if _conformal and self._has_pec_to_conform():
        raise NotImplementedError(
            "Conformal PEC (Boundary(conformal=True) or conformal_pec=True) "
            "is not carried by the uniform forward scan that forward(), "
            "optimize(), topology_optimize(), compute_mixed_s_matrix(), "
            "compute_lumped_wire_s_matrix_via_scan() and run()'s lumped/wire "
            "S-matrix step: it has no Dey-Mittra update, so the walls and PEC "
            "shapes would be staircased. It is refused before the first time "
            "step. Pass conformal_pec=False (or drop Boundary(conformal=True)) "
            "for staircase walls, or use run(compute_s_params=False) on a "
            "uniform mesh for conformal fields.")
    if self._solver == "adi":
        from rfx.materials.thin_conductor import refuse_f0_sheets
        refuse_f0_sheets(self._thin_conductors, "ADI forward")
        from rfx.current_moments import refuse_current_moment_monitor
        refuse_current_moment_monitor(self, "ADI lane (solver='adi')")
        if pec_occupancy is not None:
            raise ValueError(
                "solver='adi' does not support pec_occupancy_override; "
                "use solver='yee' to retain the declared conductor.")
        # #931 §1.9: a sheet and a wire own no cell, so ``pec_mask``
        # alone carries neither.  ``run()``'s ADI branch threads them;
        # this one did not, and a declared PEC sheet came out of
        # ``forward()`` bit-identical to empty geometry.  The lane
        # itself realizes them correctly — ``_run_adi_from_materials``
        # calls ``realized_pec_edge_masks`` — so the whole defect was
        # the two arguments missing here.
        _line_stub_scope = _line_stub_admit(self, port_s11_freqs)
        return self._run_adi_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=n_steps,
            pec_mask=pec_mask,
            pec_sheets=tuple(pec_sheets or ()),
            pec_wires=tuple(pec_wires or ()),
            return_state=False,
            lane="fwd_adi",
            diagnostics=diagnostics,
        )

    # The uniform forward lane has no subgrid (#1240). forward() (and
    # optimize() through it), topology_optimize() and
    # compute_lumped_wire_s_matrix_via_scan() all enter here; its other
    # callers never carry a refinement (run() sends one to the subgridded
    # lane, compute_mixed_s_matrix refuses it first).
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

    # ── Lumped RLC elements — differentiable lane (WP 4-E) ──────────
    # Fold R/C into the jnp materials arrays FIRST (mirrors the concrete
    # run() ordering in runners/uniform.py:82), so subsequent port folds
    # stack on top. The traced metas are built at the very end (after all
    # material folding) just before the _run(...) call.  Component values
    # may be supplied AS tracers via ``rlc_values_override`` so
    # ``jax.grad`` flows w.r.t. R/L/C (LumpedRLCSpec stores plain floats).
    # For a sim with NO lumped RLC this block is skipped and ``rlc_metas``
    # stays None -> the _run(...) call is byte-identical to before.
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

    aniso_eps, aniso_inv_eps, _, unextendable = form_tensor_operands(
        self, grid, materials, subpixel_smoothing=subpixel_smoothing,
        conformal=False, conformal_min_weight=0.1, pec_shapes=pec_shapes)
    if subpixel_smoothing:
        from rfx.geometry.smoothing import warn_unextendable_shapes
        warn_unextendable_shapes([u for u in unextendable if not u.conductor])

    from rfx.sources.port_drive import PortSourceQueue
    sources = PortSourceQueue()
    probes = []
    # ── Effective boundary flags, resolved BEFORE the PEC realization ──
    # #931 §1.7: the realized PEC edge masks (Mx, My, Mz) are only
    # correct under the periodic flags the step function will use, and
    # the port setup below reads those masks, so the flag resolution
    # moves ahead of it.  The Floquet SOURCE injection stays where it
    # was, so the source list order is unchanged.
    periodic_bool = resolve_periodic(grid, self._periodic_flags())

    # Forward cpml_axes from the grid — when waveguide ports are
    # present the grid restricts CPML to the non-propagation axes.
    # Low-level runners also default to the grid. Keep this explicit
    # value so the TFSF subset override below shares these flags.
    cpml_axes_run = grid.cpml_axes
    pec_axes_run = None

    # Differentiable TFSF plane-wave (#404): build the 2D/1D-aux TFSF cfg and
    # force its boundary (transverse-periodic + CPML on the propagation axis),
    # mirroring rfx/runners/uniform.py. The shared scan (_build_step_setup ->
    # make_core_step) injects it and auto-activates the complex Bloch path for
    # oblique (field_dtype -> complex64, per-axis roll phase). Gradients flow
    # w.r.t. the scatterer materials through the same validated kernel as run().
    tfsf_run = None
    if self._tfsf is not None:
        from rfx.sources.tfsf import tfsf_from_declaration
        tfsf_run = tfsf_from_declaration(
            self._tfsf, grid, self._freq_max, n_steps=n_steps,
        )
        from dataclasses import replace
        from rfx.api._ad_memory_model import face_record_pays
        if face_record_pays(tfsf_run[0], n_steps, checkpoint=checkpoint,
                            checkpoint_segments=checkpoint_segments):
            tfsf_run = (replace(tfsf_run[0], record_faces=True), tfsf_run[1])
        # The slab's vacuum check uses the final materials below.
        # The closed box also checks the final realized operators below,
        # after material overrides and port setup (including AD values).
        from rfx.boundaries.tfsf import boundary_flags as tfsf_boundary_flags
        periodic_bool, cpml_axes_run = tfsf_boundary_flags(tfsf_run[0], grid)
        pec_axes_run = None


    # #931 §1.7: realize (Mx, My, Mz) ONCE, here, from the volume cells
    # plus the declared sheets/wires.  Port clearing, wire-port liveness
    # and the reference-plane footprint all read THIS object from here
    # on; a sheet owns no cell, so it exists nowhere else.
    pec_sheets = tuple(pec_sheets or ())
    pec_wires = tuple(pec_wires or ())
    pec_mask_local = pec_mask
    pec_occupancy_local = pec_occupancy
    conductors, _drawn_materials, pec_edge_masks_local = _conductors.forward_products(self, grid, materials, pec_mask, pec_sheets, pec_wires, periodic_bool, conductors, sheet_impedance)
    _msl_geometry_edges = pec_edge_masks_local  # before ANY port clearing
    from rfx.measurement import ordered_specs, register_lumped, register_wire
    port_record_specs = []
    lumped_port_sparam_specs: list = []
    wire_port_sparam_specs: list = []
    wire_refplane_specs: list = []
    if port_s11_freqs is not None:
        _s11_freqs_arr = jnp.asarray(port_s11_freqs, dtype=jnp.float32)
    else:
        _s11_freqs_arr = None

    # Multi-drive S-matrix hook (item-5 Stage 1): 0-based counter over the
    # sparam-eligible lumped/wire ports (impedance != 0) in registration
    # order. When ``_sparam_drive_idx`` is set, only the port whose counter
    # equals it is excited; all others are matched loads regardless of
    # ``pe.excite``. When ``_sparam_drive_idx`` is None this counter is
    # inert and excitation follows ``pe.excite`` (current behavior).
    _sparam_port_idx = -1

    def _excite_this_port(pe) -> bool:
        if _sparam_drive_idx is None:
            return pe.excite
        return _sparam_port_idx == _sparam_drive_idx

    for _port_index, pe in enumerate(self._ports):
        if pe.impedance == 0.0:
            from rfx.sources.port_drive import defer_soft_source
            defer_soft_source(sources, grid, pe, n_steps,
                              field_dtype=self._resolve_field_dtype())
            continue

        # Sparam-eligible lumped/wire port — advance the multi-drive index.
        _sparam_port_idx += 1

        # Effective drive decision + waveform for this pass (issue #322).
        # A passive port (excite=False) registers waveform=None BY DESIGN
        # (add_port only defaults the waveform when excite=True). When the
        # S-matrix scan driver (_sparam_drive_idx; PR #258 wire reroute /
        # #214 lumped reroute) selects such a port as the driven one,
        # synthesize the same default excitation add_port() applies for
        # excite=True ports — otherwise make_port_source /
        # make_wire_port_sources hits jax.vmap(None)
        # ("TypeError: Expected a callable value, got None").
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
                radius=pe.radius,
            )
            # Capture the pre-clearing live-cell mask for the deferred drive.
            materials = setup_wire_port(
                grid, wp, materials,
                pec_edge_masks=pec_edge_masks_local)
            if _drive_this_port:
                sources.defer(make_wire_port_sources,
                    grid, wp, n_steps=n_steps,
                    pec_edge_masks=pec_edge_masks_local)
            wp_cells = _wire_port_cells(grid, wp)
            # Clear PEC mask/occupancy at LIVE wire cells only (issue
            # #318 commit 2). Dead extent cells stay PEC — the old
            # all-cells clearing punched a one-cell in-plane
            # conductivity hole in the DUT conductor. Only live edges
            # publish incident cells for occupancy release.
            from rfx.sources.sources import _wire_port_live_cells
            _, wp_live_flags, _ = _wire_port_live_cells(
                grid, wp, pec_edge_masks_local)
            _wp_live_cells = [c for c, _l in zip(wp_cells, wp_live_flags)
                              if _l]
            # No edge clearing (#931 §1.9, corrected): a cell is LIVE
            # exactly when the port component's own edge is not PEC, so
            # releasing that component is a no-op, and releasing the two
            # tangential edges would open the conductor the port foot
            # stands on.  The CELL clearing below stays: it is the
            # volume carrier; the stage also publishes live port edges.
            conductors, pec_mask_local, pec_edge_masks_local = _conductors.forward_port_stage(conductors, _wp_live_cells, pe.component, f"port[{_port_index}]", port_kind="wire", release_edges=False)
            # Register a JIT-integrated S-param accumulator for this
            # WirePort when forward(port_s11_freqs=...) was requested
            # (issue #79 follow-up to PR #72). Mirrors the lumped
            # registration below; uses the LIVE-run midpoint cell as
            # the V/I reference plane (issue #764 — identical to the
            # all-extent midpoint with no dead cells; consistent with
            # wire_sparam_meta in the JIT scan body of
            # rfx/simulation.py) and carries the static live run for
            # the whole-port gap-voltage accumulator.
            register_wire(port_record_specs, wire_port_sparam_specs, pe.component,
                          wp_cells, wp_live_flags, _s11_freqs_arr, pe.impedance, _drive_this_port, _port_index)
            # Opt-in reference-plane V/I accumulators (issue #313).
            # Registered ONLY on the production-scan S-matrix driver
            # path (_sparam_drive_idx is not None): the plane waves
            # feed the OFF-diagonal S_ij and need the multi-drive
            # sweep; forward()'s diagonal path is untouched (its S11
            # never uses the planes), and the geometry derivation
            # below needs concrete (non-traced) pec_mask values.
            if (pe.reference_plane_cells is not None
                    and _s11_freqs_arr is not None
                    and _sparam_drive_idx is not None
                    and wp_cells):
                from rfx.probes.refplane import build_wire_refplane_specs
                # Guard: the planes must not reach past another port
                # along the same line axis (they must sit on the
                # uniform line BETWEEN the ports).
                _axis_of = {"x": 0, "y": 1, "z": 2}
                _l_ax = _axis_of[pe.direction[1]]
                _sign = -1 if pe.direction[0] == "+" else +1
                _i_port = int(grid.position_to_index(pe.position)[_l_ax])
                _i_far = _i_port + _sign * 2 * int(pe.reference_plane_cells)
                for _other in self._ports:
                    if _other is pe or _other.impedance == 0.0:
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
                # Same guard, extended to MSL ports: a microstrip
                # port's de-embedding probe ladder must stay
                # outboard of the plane zone too. The helper walks
                # the RESOLVED ladder (the one the S-matrix drivers
                # probe), not the registration ladder.
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
                    # #695: hand it the FULL conductor footprint, not
                    # the bare pec_mask. Since #677 an f0 sheet is a
                    # node-thin operator present in neither pec_mask
                    # nor materials.sigma, so a sheet-traced line
                    # reached build_wire_refplane_specs as "no metal"
                    # and raised on a healthy model. The ctx's
                    # sigma_sheet is nonzero exactly on the sheet
                    # cells, so no re-assembly is needed here. Skipped
                    # under a tracer: this whole branch already
                    # requires concrete pec_mask values (see the
                    # comment above), and a traced OR would only move
                    # the failure.
                    pec_mask=_refplane_conductor_mask(
                        pec_mask, sheet_impedance, pec_sheets),
                )
                # Both planes must lie INSIDE the declared domain
                # and clear of the absorber — a distinct message
                # class. Checked AFTER build_wire_refplane_specs so
                # its own "no conductor found" geometry error keeps
                # precedence (frozen by
                # tests/locks/test_refplane_port_waves.py::
                # test_refplane_requires_pec_trace_at_plane).
                _refplane_reject_planes_off_the_uniform_line(
                    grid, pe, _plane_specs, line_axis=_l_ax)
                wire_refplane_specs.extend(_plane_specs)
            continue

        lp = LumpedPort(
            position=pe.position,
            component=pe.component,
            impedance=pe.impedance,
            excitation=_drive_waveform,
        )
        materials = setup_lumped_port(grid, lp, materials)
        if _drive_this_port:
            sources.defer(make_port_source, grid, lp, n_steps=n_steps)
        idx = _grid_metric.field_index(grid, pe.position, pe.component)
        conductors, pec_mask_local, pec_edge_masks_local = _conductors.forward_lumped_port_stage(conductors, pe, f"port[{_port_index}]")
        # Register a JIT-integrated S-param accumulator for this
        # lumped port when the user requested forward(port_s11_freqs=...)
        # (issue #72). Skipping passive ports keeps S11 indexing
        # aligned with the active-port list.
        register_lumped(port_record_specs, lumped_port_sparam_specs, pe.component,
                        idx, _s11_freqs_arr, pe.impedance, _drive_this_port, _port_index)

    # MSL ports — full cross-section distributed feed; built like the
    # wire-port branch above but with N_y × N_z cells.  S-parameters are
    # extracted post-scan by ``compute_msl_s_matrix`` from registered
    # DFT plane probes (no JIT accumulator wiring needed here).
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
        # Issue #483: all automatic Laplace launches read the original
        # drawn substrate, already carried by the solve assembly.
        _static_eps_483 = None
        if any(pe.eps_r_sub is None and getattr(pe, "mode", "uniform") == "laplace"
               for pe in self._msl_ports):
            # Read the drawn value before design/material overrides.
            _static_eps_483 = _conductors.drawn_launch_eps(self, grid, _drawn_materials)
        for _msl_port_index, pe in enumerate(self._msl_ports):
            # Use the same physical-to-port frame as run(). In a y-fed
            # port feed_x names physical y, and y_lo/y_hi name width x.
            mp = msl_port_from_entry(pe)
            from rfx.sources.msl_port import validate_msl_port_geometry
            validate_msl_port_geometry(
                grid, mp, pec_edge_masks=_msl_geometry_edges,
                sheet_impedance=sheet_impedance, periodic=periodic_bool,
                pec_faces=self._boundary_spec.pec_faces(), name=pe.name)
            # Honour `pe.mode` so the source distribution matches
            # the imperative `run_uniform_path` (Phase 3 of gap #2/#4
            # closure, 2026-05-07).  ``laplace`` is the default for
            # ``add_msl_port`` and gives a structured Ez source from
            # the static-Laplace quasi-TEM mode shape.  ``eigenmode``
            # mode requires the J+M Schelkunoff source pair which the
            # legacy uniform forward path does not yet wire — refuse
            # explicitly so users do not silently get a wrong source.
            port_mode = getattr(pe, "mode", "uniform")
            mode_profile = None
            if port_mode == "laplace":
                span = msl_cross_section_span(grid, mp)
                # Keep the declared midpoint sample when the source edge
                # list becomes half-open (#729); it is not the midpoint
                # of the list's last E-edge index. Match run() on all axes.
                k_mid = (span["n_lo"] + span["n_hi"]) // 2
                eps_cell = msl_cell(pe.direction, span["i_feed"],
                                    span["w_centre"], k_mid)
                if pe.eps_r_sub is not None:
                    eps_r_sub = float(pe.eps_r_sub)
                else:
                    # Issue #483: the launch fixture (mode profile,
                    # added sigma, imposed force e*u) is STATIC
                    # and must derive from the REGISTERED
                    # materials — never from `materials`, which may
                    # carry a (traced or concrete) override. The actual
                    # E increment Cb(eps, sigma_total)*e*u MUST retain
                    # its material dependence on the AD tape. The old
                    # code sampled the override through stop_gradient:
                    # finite differences then re-derived the fixture at
                    # alpha±h while the AD tape saw it frozen at the
                    # linearization point, so FD and jax.grad
                    # differentiated DIFFERENT functions (measured:
                    # 61.7% raw / 13.7% converged gradient deficit;
                    # with the fixture constant on both sides the same
                    # f64 referee reads 0.01-0.09%). The registered materials are
                    # re-assembled host-side once per forward build
                    # (hoisted above the port loop) — correct
                    # for every caller (forward, topology,
                    # sparam_driver) without threading a handle.
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
                    "path — the J+M Schelkunoff source pair needs a "
                    "magnetic-source channel that `_forward_from_materials` "
                    "does not currently expose.  Use mode='laplace' "
                    "(the default) on the differentiable path."
                )
            materials = setup_msl_port(grid, mp, materials,
                                       mode_profile=mode_profile)
            if pe.excite and pe.waveform is not None:
                sources.defer(make_msl_port_sources,
                    grid, mp, n_steps=n_steps,
                    mode_profile=mode_profile,
                )
            # Clear PEC mask over the cross-section so the source/σ cells
            # are not zeroed by the PEC update.
            _msl_cells = list(_msl_yz_cells(grid, mp))
            conductors, pec_mask_local, pec_edge_masks_local = _conductors.forward_port_stage(
                conductors, _msl_cells, _msl_normal_component(mp), f"msl_port[{_msl_port_index}]",
                port_kind="microstrip", skip_empty=True, released_edges=(mode_profile["cell_indices"] if mode_profile is not None else _msl_cells))

    for pe in self._probes:
        probes.append(make_probe(grid, pe.position, pe.component))

    if not probes and self._ports:
        for pe in self._ports:
            probes.append(make_probe(grid, pe.position, pe.component))

    from rfx.model.materials import with_components
    materials = with_components(materials, grid, periodic=periodic_bool,
        debye_spec=debye_spec, lorentz_spec=lorentz_spec)
    from rfx.model.source_coefficients import resolve_run_sources
    sources = resolve_run_sources(sources, materials, self, grid,
        tensor=(pec_occupancy_local is not None and os.environ.get("RFX_PEC_OCC_KOTTKE", "0") not in ("0", "", "false", "False"))
            or aniso_eps is not None or aniso_inv_eps is not None)
    aniso_eps = finish_tensor_operands(materials, aniso_eps, None)
    _, debye, lorentz = self._init_dispersion(
        materials, grid.dt, debye_spec, lorentz_spec,
        periodic=periodic_bool,
    )

    ntff_box = None
    if self._ntff is not None:
        from rfx.farfield import make_ntff_box
        corner_lo, corner_hi, freqs = self._ntff
        ntff_box = make_ntff_box(grid, corner_lo, corner_hi, freqs)

    # In-loop block current moments. Same monitor object the run() lane
    # builds, so a forward()/value_and_grad call accumulates the same
    # numbers the forward run does.
    from rfx.current_moments import monitor_for_simulation as _cm_for_sim
    current_moments_fwd = _cm_for_sim(
        self, grid, periodic_bool, conductors=conductors,
        overrides={**(monitor_overrides or {}), "design_box": design_box,
                   "design_occupancy": design_occupancy})

    # Declared flux monitors are accumulated and returned on forward as on run (#1373 D4).
    # The DFT accumulators add to the AD tape only when the user declares a monitor.
    flux_monitor_cfgs = []
    if self._flux_monitors:
        from rfx.runners.uniform import build_flux_monitor_cfgs
        flux_monitor_cfgs = build_flux_monitor_cfgs(self, grid, n_steps, periodic=periodic_bool)

    # DFT plane probes — mirror runners/uniform.py:340-359 so the
    # JIT scan body actually accumulates plane-resolved DFT, then
    # carry the result back through ForwardResult.dft_planes for
    # plane-integrated V/I objectives (known gap #4, 2026-05-05).
    from rfx.measurement.setup import build_planes
    dft_planes = build_planes(self, grid, n_steps)

    # Waveguide ports (differentiable DFT accumulation inside scan)
    waveguide_ports = []
    if self._waveguide_ports:
        wg_freqs = None
        for pe in self._waveguide_ports:
            if pe.freqs is not None:
                wg_freqs = jnp.asarray(pe.freqs, dtype=jnp.float32)
                break
        if wg_freqs is None:
            wg_freqs = jnp.linspace(
                self._freq_max * 0.5, self._freq_max, 20, dtype=jnp.float32)
        for pe in self._waveguide_ports:
            waveguide_ports.append(
                self._build_waveguide_port_config(pe, grid, wg_freqs, n_steps))

    # Floquet ports — inject soft source, same as run_uniform.py:274-327
    if self._floquet_ports:
        axis_map_str = {"x": 0, "y": 1, "z": 2}
        for fpe in self._floquet_ports:
            axis_idx = axis_map_str[fpe.axis]
            fp_f0 = fpe.f0 if fpe.f0 is not None else self._freq_max / 2
            from rfx.sources.sources import GaussianPulse as _GP
            wf = _GP(f0=fp_f0, bandwidth=fpe.bandwidth, amplitude=fpe.amplitude)
            center = [self._domain[i] / 2.0 for i in range(3)]
            center[axis_idx] = fpe.position
            if fpe.polarization == "te":
                comp = {"z": "ex", "x": "ey", "y": "ex"}[fpe.axis]
            else:
                comp = {"z": "ey", "x": "ez", "y": "ez"}[fpe.axis]
            from rfx.simulation import make_source as _make_src
            sources.append(_make_src(grid, tuple(center), comp, wf, n_steps))
    conductors, pec_occupancy_local = _conductors.prepare_port_occupancy(conductors, waveguide_ports, pec_occupancy_local, design_occupancy, sim=self, sheet_operator=sheet_impedance)

    # Tensor lane (opt-in via ``RFX_PEC_OCC_KOTTKE=1``): when
    # ``pec_occupancy_local`` is supplied, build an ``aniso_inv_eps``
    # tensor from the occupancy with the lattice contract's edge rule
    # (#1197: ``inv_c = (1 - M_c) * baseline``) and run it through the
    # same ``update_e_aniso_inv`` path hard ``Box(material="pec")``
    # uses. Bypasses the ``apply_pec_occupancy`` E-scaling path on
    # this branch — the two would double-correct at sigmoid edges.
    pec_occupancy_for_run, aniso_inv_eps_run = _occupancy.select_occupancy_operator(
        self, grid, materials, pec_occupancy_local, periodic_bool, design_occupancy)

    if aniso_inv_eps is not None:
        aniso_inv_eps_run = aniso_inv_eps

    # Build the RLC metas now that all material folding (RLC + ports)
    # is complete — build_rlc_meta reads the final eps/sigma at each
    # element cell, mirroring the concrete runners/uniform.py:516 ordering.
    if self._lumped_rlc:
        from rfx.lumped import build_rlc_meta
        _rlc_ov = rlc_values_override or {}
        rlc_metas = []
        from rfx.simulation import resolve_periodic as _resolve_periodic
        # The run's own periodic flags: a series element's D0 is its
        # edge's E-update coefficient (#1163).
        _rlc_periodic = _resolve_periodic(grid, periodic_bool)
        for _rlc_idx, _rlc_spec in enumerate(self._lumped_rlc):
            _v = _rlc_ov.get(_rlc_idx, {})
            rlc_metas.append(build_rlc_meta(
                grid, _rlc_spec, materials,
                r_val=_v.get("R"), l_val=_v.get("L"), c_val=_v.get("C"),
                periodic=_rlc_periodic,
            ))
        # #1245: at most one element with its own solve per realized edge.
        from rfx.lumped import refuse_stacked_solved_elements
        refuse_stacked_solved_elements(self._lumped_rlc, rlc_metas)

    # Validate the operators handed to the scan, after every port fold
    # and PEC clearing. A passive load can reach the source shell even
    # when all of its conductor geometry is clear of that shell.
    if self._tfsf is not None and self._tfsf.closed_box:
        nonvacuum = list(pec_edge_masks_local or ())
        for spec in (debye_spec, lorentz_spec):
            if spec is not None:
                nonvacuum.extend(spec[1])
        if kerr_chi3 is not None:
            nonvacuum.append(kerr_chi3)
        if pec_occupancy_local is not None:
            nonvacuum.append(pec_occupancy_local)
        if aniso_inv_eps_run is not None:
            nonvacuum.extend(a != 1.0 for a in aniso_inv_eps_run)
        if sheet_impedance is not None:
            nonvacuum.extend((sheet_impedance.mask_ex, sheet_impedance.mask_ey,
                              sheet_impedance.mask_ez))
        if self._lumped_rlc:
            mask = jnp.zeros(grid.shape, dtype=bool)
            for spec in self._lumped_rlc:
                mask = mask.at[_grid_metric.field_index(grid, spec.position, spec.component)].set(True)
            nonvacuum.append(mask)
        for spec in (design_box, design_occupancy):
            if spec is not None:
                b = spec.bounds
                window = tuple(slice(b[2 * d], min(b[2 * d + 1] + 1, grid.shape[d]))
                               for d in range(3))
                nonvacuum.append(jnp.zeros(grid.shape, dtype=bool).at[window].set(True))
        self._validate_tfsf_vacuum_boundary(
            materials, tfsf_run[0], nonvacuum_masks=tuple(nonvacuum))
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

    # Every declared input this lane does not carry is refused here,
    # after the specific refusals above and before the first step.
    if lane is not None:
        from rfx.runners._admission import admit
        admit(
            self,
            lane,
            diagnostics=diagnostics,
            run_args={"tfsf_materials": materials} if self._tfsf is not None else None,
        )

    conductors, pec_edge_masks_local, sheet_impedance = _conductors.forward_kernel_inputs(self, conductors, lane or "fwd_uniform", pec_edge_masks_local, sheet_impedance)
    result = _run(
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
        aniso_eps=aniso_eps,
        aniso_inv_eps=aniso_inv_eps_run,
        # Only the occupancy operator is smooth; the Kottke tensor keeps
        # run()'s fenced PEC edges.
        aniso_inv_eps_smooth=(aniso_inv_eps is None and aniso_inv_eps_run is not None),
        port_record_specs=ordered_specs(port_record_specs),
        lumped_port_sparams=lumped_port_sparam_specs or None,
        wire_port_sparams=wire_port_sparam_specs or None,
        wire_refplane_sparams=wire_refplane_specs or None,
        lumped_rlc=rlc_metas,
        kerr_chi3=kerr_chi3,
        dft_planes=dft_planes if dft_planes else None,
        record_dft=bool(getattr(self, "_internal_probe_indices", ())),
        flux_monitors=flux_monitor_cfgs if flux_monitor_cfgs else None,
        return_state=False,
        stencil_order=self._stencil_order,
        sheet_impedance=sheet_impedance,
        design_box=design_box,
        gradient=gradient,
        design_occupancy=design_occupancy,
        field_dtype=self._resolve_field_dtype(),
    )

    # Return the physical point-probe field, as run_uniform does.
    from rfx.measurement import reconstruct_oblique_physical
    from rfx.sources.tfsf import is_tfsf_2d
    if tfsf_run is not None and is_tfsf_2d(tfsf_run[0]):
        result = reconstruct_oblique_physical(result, tfsf_run[0], grid, probes)

    # Multi-drive S-matrix hook (item-5 Stage 1): return the raw all-port
    # (v_dft, i_dft) accumulators for off-line wave decomposition by the
    # production-scan driver, bypassing the diagonal-only extraction below.
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
            "dft_time_records": {entry.name: rec for entry, rec in zip(
                self._dft_planes, result.dft_time_records or ())},
            "wire_refplane": result.wire_refplane_sparams,
            "freqs": _raw_freqs,
            # Mixed-family S-matrix hook (issue #488): the DFT plane
            # probes and probe time series recorded in the SAME run, so
            # compute_mixed_s_matrix can read MSL port planes + settling
            # witnesses alongside the lumped/wire accumulators without a
            # second pass. The name-keyed dict mirrors the run()/forward()
            # ``ForwardResult.dft_planes`` contract below. PURE ADD — the
            # lumped/wire scan driver reads only the keys above.
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
            "dt": getattr(result, "dt", grid.dt),
            # Flux monitors for the issue-#488 arch-A magnitude channel
            # (same name-keyed contract the runner uses for run()).
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

    s_params_out = getattr(result, "s_params", None)
    freqs_out = getattr(result, "freqs", None)
    from rfx.measurement.port_s_contract import uniform_record_pairs
    pairs = uniform_record_pairs(result)
    if pairs:
        from rfx.measurement import port_s_from_record
        metas, channels = zip(*pairs)
        s_params_out = port_s_from_record(
            channels, metas, wire_mode=pairs is result.wire_port_sparams)
        freqs_out = metas[0].freqs

    # Passivity self-check (tracer-safe; skipped under jax.grad/jit) — the
    # eager single-cell lumped/wire extractor can return non-physical
    # |S11|>1 where the incident wave is weak (spectral band edges), and
    # forward() previously surfaced no check (unlike compute_*_s_matrix).
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
        freqs=freqs_out, current_moment_monitor=current_moments_fwd,
    )), conductors.design_findings)


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly (as rfx/preflight/sources.py
# does). The function above was a ``def`` in ``_ExecuteMixin``'s body. Its
# decorator's wrapper copies the qualname at decoration time, and
# ``rfx/api/__init__.py`` rewrites exactly ``<mixin>.<name>`` ->
# ``Simulation.<name>`` on the wrapper at class composition and skips a bare
# name. The wrapped function keeps ``_ExecuteMixin._forward_from_materials``,
# which is what a TypeError for a bad keyword reports, as before the move.
# ``tests/contracts/test_execute_split_qualnames.py`` pins both.
# ---------------------------------------------------------------------------
_forward_from_materials.__qualname__ = "_ExecuteMixin._forward_from_materials"
_forward_from_materials.__wrapped__.__qualname__ = "_ExecuteMixin._forward_from_materials"


def refuse_sheet_layout():
    raise NotImplementedError(
        "pinned_sheet_layout requires the single-device graded lane; "
        "a uniform mesh can be passed as a graded profile")

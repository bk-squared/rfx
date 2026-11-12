from __future__ import annotations

import numpy as np

from rfx.errors import AccuracyWarning, RfxUnsupported


def _validate_coax_msl_registration(
    eps_r_sub, self,
):
    # Boundary rationale (moved from API docstring).
    # Same solver/precision/boundary contract as
    # :meth:`compute_coaxial_two_port` (float32, 3D uniform Yee,
    # ``boundary='cpml'`` with positive CPML on all six faces, absorption on
    # all three axes): the coax's own far end needs an absorbing z face
    # exactly like the coax lane, the MSL trace radiates in x/y too,
    # and the caller's own ground plane is an INTERNAL stamped/registered
    # PEC layer rather than the domain's z boundary — this is also WHY
    # this method cannot reuse a PEC ``z_lo`` domain boundary as the MSL
    # ground reference the way :meth:`compute_mixed_s_matrix`'s
    # ``magnitude_channel="flux"`` fixtures do (that would conflict with
    # the coax's own need for an absorbing z_lo face).
    #
    if self._boundary != "cpml" or self._cpml_layers <= 0:
        raise ValueError(
            "compute_coax_msl_transition() requires boundary='cpml' "
            "with cpml_layers > 0."
        )
    if any(token != "cpml" for _, _, token in self._boundary_spec.faces()):
        raise ValueError(
            "compute_coax_msl_transition() requires CPML tokens on all "
            "six boundary faces; mixed BoundarySpec faces are not "
            "supported (the coax stub needs an absorbing z_lo face and "
            "the MSL trace needs absorbing x/y faces; the ground plane "
            "is an internal registered/stamped PEC layer, not a domain "
            "boundary)."
        )
    if self._periodic_axes:
        raise ValueError(
            "compute_coax_msl_transition() does not support periodic "
            "boundary axes."
        )
    if self._mode != "3d":
        raise ValueError("compute_coax_msl_transition() requires mode='3d'.")
    if self._solver != "yee":
        raise ValueError(
            "compute_coax_msl_transition() supports solver='yee' only."
        )
    if self._precision != "float32":
        raise ValueError("compute_coax_msl_transition() requires precision='float32'.")
    if self._stencil_order != 2:
        raise ValueError(
            "compute_coax_msl_transition() requires stencil_order=2."
        )
    if self._tfsf is not None:
        raise ValueError(
            "compute_coax_msl_transition() creates its own coax TEM "
            "source and does not accept an existing TFSF source."
        )
    if (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    ):
        raise ValueError(
            "compute_coax_msl_transition() supports only a uniform "
            "Yee grid; dx_profile/dy_profile/dz_profile are not "
            "supported."
        )
    if self._refinement is not None:
        raise ValueError(
            "compute_coax_msl_transition() does not support SBP-SAT "
            "refinement."
        )
    if self._lumped_rlc:
        raise ValueError(
            "compute_coax_msl_transition() does not support registered "
            "lumped RLC elements."
        )
    if self._probes or self._dft_planes or self._flux_monitors or self._ntff:
        raise ValueError(
            "compute_coax_msl_transition() does not consume registered "
            "probes, DFT planes, flux monitors, or NTFF boxes (it "
            "builds its own)."
        )
    if len(self._coaxial_ports) != 1:
        raise ValueError(
            "compute_coax_msl_transition() is built from exactly one "
            "add_coaxial_port()."
        )
    if len(self._msl_ports) != 1:
        raise ValueError(
            "compute_coax_msl_transition() is built from exactly one "
            "add_msl_port()."
        )
    if self._ports or self._waveguide_ports or self._floquet_ports:
        raise NotImplementedError(
            "compute_coax_msl_transition() is defined only for a "
            "coax + MSL port pair; no other port families."
        )
    if not self._geometry:
        raise ValueError(
            "compute_coax_msl_transition() consumes the caller's own "
            "registered DUT geometry (substrate, trace, ground plane, "
            "clearance hole, pin-to-trace post) — none is registered. "
            "This method builds only the coax stub; register the "
            "junction geometry via sim.add(...) first."
        )
    port = self._coaxial_ports[0]
    if port.face != "bottom":
        raise NotImplementedError(
            "compute_coax_msl_transition() currently requires the "
            "registered coax port's face='bottom' (the coax stub is "
            "built from the domain's low-z CPML face up to "
            "position[2])."
        )
    msl_pe = self._msl_ports[0]
    if msl_pe.mode != "laplace":
        raise RfxUnsupported(
            f"compute_coax_msl_transition(): MSL port {msl_pe.name!r} has "
            f"unsupported mode={msl_pe.mode!r}; use mode=\"laplace\" (the default)."
        )
    # The coax stub's own top face stops AT port.position[2] (rounded to
    # the nearest grid node); the caller's own registered ground-plane
    # conductor is expected to occupy that node and MSL's substrate
    # (msl_pe.position[2], its own z_lo / substrate-bottom convention)
    # to begin at or above it -- the exact gap is the caller's own
    # ground-plane thickness (fixture-specific, not knowable here), so
    # this only guards the ORDER and catches a grossly misaligned
    # ground reference (e.g. forgetting to raise msl_z_lo at all).
    if float(msl_pe.position[2]) < float(port.position[2]) - 1.5e-9:
        raise ValueError(
            "compute_coax_msl_transition(): the registered MSL port's "
            f"substrate-bottom height ({msl_pe.position[2]:.6g} m) sits "
            f"BELOW the coax port's own junction height "
            f"({port.position[2]:.6g} m) — both must reference the SAME "
            "physical ground plane, with the MSL substrate at or above "
            "it."
        )
    eps_r_sub_resolved = (
        float(eps_r_sub) if eps_r_sub is not None
        else (float(msl_pe.eps_r_sub) if msl_pe.eps_r_sub is not None else None)
    )
    if eps_r_sub_resolved is None:
        raise ValueError(
            "compute_coax_msl_transition() needs eps_r_sub, either "
            "passed directly or set on the registered add_msl_port() "
            "(this method does not auto-detect it from geometry)."
        )
    return port, msl_pe, eps_r_sub_resolved


def _prepare_coax_msl_stub_layout(
    grid, msl_cross_section_span, msl_port_base, port, probe_count, probe_spacing_cells, probe_start_cells,
):
    from rfx.geometry.rasterize_grid import (
        _local_cell, _nearest_plane, coords_from_uniform_grid,
        cell_sizes_from_uniform_grid,
    )
    z_nodes = coords_from_uniform_grid(grid)[2]
    z_sizes = cell_sizes_from_uniform_grid(grid)[2]
    z_local = _local_cell(z_nodes, z_sizes, float(port.position[2]))
    z_junction_idx = _nearest_plane(
        z_nodes, float(port.position[2]), z_local,
        what="coax-MSL junction", axis=2)
    if z_junction_idx > msl_cross_section_span(grid, msl_port_base)["n_lo"]:
        raise ValueError(
            "compute_coax_msl_transition(): the realized coax junction is "
            "above the MSL ground plane; make both registrations refer "
            "to the same physical ground layer.")
    z_stub_lo = int(grid.pad_z_lo) + 2
    z_feed = z_stub_lo + 1
    z_src = z_stub_lo + 3
    z_stub_hi = z_junction_idx - 1
    if z_stub_hi <= z_src:
        raise ValueError(
            "compute_coax_msl_transition(): domain too short between "
            "the low-z CPML and the junction height for the coax "
            "stub's source/feed/probe layout; increase the z domain or "
            "lower position[2]."
        )
    probes_coax = sorted(
        z_src + int(probe_start_cells) + int(probe_spacing_cells) * k
        for k in range(int(probe_count))
    )
    if probes_coax[-1] >= z_stub_hi:
        raise ValueError(
            "compute_coax_msl_transition(): the coax probe array "
            f"reaches index {probes_coax[-1]}, at or past the junction "
            f"({z_stub_hi}); increase the z domain or reduce "
            "probe_count/probe_start_cells/probe_spacing_cells."
        )
    return z_junction_idx, z_stub_lo, z_feed, z_src, z_stub_hi, probes_coax


def _prepare_coax_msl_ladder(
    dz, grid, junction_x, msl_pe, msl_port_base, msl_probe_count, msl_probe_spacing_cells,
    msl_probe_start_cells, msl_probe_x_coords_n, probe_count, probe_spacing_cells, probe_start_cells, self,
):
    # The MSL-side probe ladder is independent of the coax-side one.
    # Unset MSL ladder parameters default to the coax values. The
    # microstrip ladder may need a meaningful fraction of a guided wavelength,
    # while the short coax stub cannot host that span. Separate optional
    # parameters decouple those extents; None retains the shared defaults.
    _msl_probe_count = int(probe_count if msl_probe_count is None else msl_probe_count)
    _msl_probe_start_cells = int(
        probe_start_cells if msl_probe_start_cells is None else msl_probe_start_cells
    )
    _msl_probe_spacing_cells = int(
        probe_spacing_cells if msl_probe_spacing_cells is None else msl_probe_spacing_cells
    )
    probe_xs = msl_probe_x_coords_n(
        grid, msl_port_base, n_probes=_msl_probe_count,
        n_offset_cells=_msl_probe_start_cells,
        n_spacing_cells=_msl_probe_spacing_cells,
    )
    xs_ladder = [float(x) for x in probe_xs]
    lx_dom = float(self._domain[0])
    mono = all(
        (xs_ladder[q + 1] - xs_ladder[q]) * (1 if msl_pe.direction == "+x" else -1)
        > 0.5 * dz
        for q in range(len(xs_ladder) - 1)
    )
    if (not mono) or min(xs_ladder) <= 0.0 or max(xs_ladder) >= lx_dom:
        raise ValueError(
            "compute_coax_msl_transition(): the MSL probe ladder "
            f"({', '.join(f'{x * 1e3:.2f}' for x in xs_ladder)} mm) "
            f"leaves the declared x-domain (0, {lx_dom * 1e3:.2f}) mm "
            "or was clamped at its edge. Face the port toward the "
            "junction (direction), reduce n_probe_offset/spacing, or "
            "enlarge the domain."
        )
    # coaxial_line_reflection_from_plane_voltages requires STRICTLY
    # INCREASING plane positions; a "-x"-facing port's own ladder comes
    # back decreasing in x (probe n steps AWAY from feed_x, toward the
    # junction). Sort once here and use this order everywhere below so
    # DFT-plane construction and the voltage array stay index-consistent.
    xs_sorted = sorted(xs_ladder)

    # ---- source near-field standoff on the REALIZED ladder.
    # NEAR-FIELD STANDOFF — the one constraint these
    # three are NOT free of. Every probe must clear BOTH ends the
    # ladder is referred to (the MSL port's own feed plane AND the
    # reference plane at ``junction_x``) by at least
    # ``max(3, round(5*h_sub/dx))`` cells. Both are launch
    # discontinuities, and within a few substrate thicknesses of one
    # the field is not the guided mode yet, so a matrix-pencil fit
    # that includes such a probe reports the LADDER's error as the
    # field's: on the settled attempt-3 run (VESSL 369367257533) a
    # single probe 0.4 mm = 1.33*h_sub from the feed dragged the
    # full-ladder residual from 3.7e-3 to 0.342, and this lane spent
    # three attempts attributing that to junction physics. The
    # threshold is the source near-field standoff constant —
    # see :func:`rfx.preflight._msl_probe_rules.msl_source_near_field_standoff_cells`
    # for the derivation that licenses reusing it here, and for the
    # W/h limitation it carries. Preflight cannot enforce it on this
    # lane (these are METHOD arguments and never reach the
    # registered ``_MSLPortEntry``), so this method evaluates the
    # same predicate on its own REALIZED ladder and emits a
    # ``UserWarning`` naming the offending probes. REPORT-ONLY:
    # nothing is refused. Read it together with
    # ``result.ladder_split_gamma_dev`` /
    # ``result.ladder_split_reflection_decades`` (computed when
    # ``return_ladder_voltages=True``, ``None`` otherwise), which say
    # whether the ladder actually disagrees with itself.
    #
    # ``msl_probe_count/start/spacing`` are METHOD arguments — they never
    # reach the registered ``_MSLPortEntry``, so preflight's own check 5
    # (``_check_msl_port_geometry``) is structurally blind to this ladder.
    # Evaluate the SAME predicate here, on the coordinates the extractor
    # will actually sample, at BOTH ends the ladder is referred to: the
    # port's own feed plane AND the reference plane at the junction.
    #
    # Emitted with ``warnings.warn``, deliberately NOT by constructing a
    # PreflightWarning: this is not a preflight check (this method never
    # calls preflight — it is DIAGNOSTIC_ONLY in
    # tests/unit/preflight/test_preflight_advisory_emission_contract.py's
    # EMISSION_CLASSIFICATION) and the emission-site freeze in that file
    # counts PreflightWarning/PreflightErrorWarning/PreflightIssue/
    # PreflightConfigError constructions only.
    #
    # REPORT-ONLY: no gate, no refusal, and no msl_fit_residual_max —
    # that gate is explicitly out of scope until the standoff rule and
    # the ladder self-consistency witness have both been exercised on a
    # settled run (PI sequencing).
    from rfx.preflight._msl_probe_rules import (
        msl_source_near_field_standoff_cells as _msl_standoff_cells,
    )
    _standoff_cells = _msl_standoff_cells(float(msl_pe.height), float(dz))
    _standoff_m = _standoff_cells * float(dz)
    _feed_x_msl = float(msl_pe.position[0])
    _standoff_hits = []
    for _end_label, _end_x in (
        (f"the MSL port feed plane (x = {_feed_x_msl * 1e3:.2f} mm)", _feed_x_msl),
        (f"the reference plane at the junction "
         f"(x = {float(junction_x) * 1e3:.2f} mm)", float(junction_x)),
    ):
        _bad = [x for x in xs_sorted if abs(x - _end_x) < _standoff_m - 1e-12]
        if _bad:
            _standoff_hits.append(
                f"{len(_bad)} of {len(xs_sorted)} probes sit within "
                f"{_standoff_m * 1e3:.2f} mm of {_end_label} "
                f"(nearest {min(abs(x - _end_x) for x in _bad) * 1e3:.2f} mm "
                f"= {min(abs(x - _end_x) for x in _bad) / float(msl_pe.height):.2f}"
                f"·h_sub)"
            )
    if _standoff_hits:
        import warnings as _wnf
        _wnf.warn(
            "compute_coax_msl_transition(): the MSL probe ladder "
            f"(msl_probe_count={_msl_probe_count}, "
            f"msl_probe_start_cells={_msl_probe_start_cells}, "
            f"msl_probe_spacing_cells={_msl_probe_spacing_cells}; realized "
            f"x = {', '.join(f'{x * 1e3:.2f}' for x in xs_sorted)} mm) "
            "violates the source near-field standoff of "
            f"{_standoff_cells} cells ({_standoff_m * 1e3:.2f} mm = "
            "5·h_sub, the source near-field standoff constant add_msl_port's own auto "
            "n_probe_offset already floors to): "
            + "; ".join(_standoff_hits)
            + ". Within a few substrate thicknesses of a launch "
            "discontinuity the field is not the guided mode yet, and the "
            "matrix-pencil fit reports the LADDER's error as the field's: "
            "measured on the settled attempt-3 run (VESSL 369367257533) a "
            "probe at 1.33·h_sub carried 82-128% two-wave model error and "
            "dragged the full-ladder fit_residual to 0.342/0.264/0.222, "
            "while every window excluding it fit to 1e-5..4e-3. Move the "
            "ladder inside [junction + 5·h_sub, feed - 5·h_sub] by raising "
            "msl_probe_start_cells and/or lowering msl_probe_count, and "
            "read result.ladder_split_gamma_dev / "
            "result.ladder_split_reflection_decades (computed when "
            "return_ladder_voltages=True) for whether this ladder "
            "actually disagrees with itself. REPORT-ONLY: nothing is "
            "refused.", AccuracyWarning,
            stacklevel=3,
        )
    return xs_sorted


def _prepare_coax_msl_trace(
    _cx_pec_edge_masks, _msl_cell_profile, grid, msl_cross_section_span, msl_port_base, self,
):
    span = msl_cross_section_span(grid, msl_port_base)
    k_lo_msl, k_hi_msl = span["n_lo"], span["n_hi"]
    j_centre_msl = span["w_centre"]
    i_feed_msl = span["i_feed"]
    # realized wall planes locate the trace, not cells.
    from rfx.probes.msl_wave_decomp import (
        realized_trace_planes_on_column as _trace_planes_cx,
    )
    k_trace_lo, _ = _trace_planes_cx(
        _cx_pec_edge_masks, 2, (i_feed_msl, j_centre_msl), k_hi_msl,
        periodic=self._periodic_flags())
    if k_trace_lo is None:
        raise RuntimeError(
            "compute_coax_msl_transition(): no realized PEC trace "
            "conductor found above the substrate top at the registered "
            "MSL port's own feed plane; declare the microstrip trace as "
            "a Box(material='pec') (a volume) or as a zero-thickness "
            "Box / add_thin_conductor (a sheet)."
        )
    dz_arr = _msl_cell_profile(grid, "z", grid.nz)
    return k_lo_msl, j_centre_msl, k_trace_lo, dz_arr


def _collect_coax_msl_ladder_witness(
    _ladder_split_witness, grid, junction_x, line_impedances, msl_z_lo, probes_coax, ref_coax_m,
    return_ladder_voltages, v_coax_by_drive, v_msl_by_drive, xs_sorted, y_centre, z_planes_coax_m,
):
    # Ladder dump implementation (moved from the API docstring).
    # ``return_ladder_voltages``:
    # a second read-only channel, additive in exactly the same sense. When
    # ``True``, the RAW per-probe modal voltages this method already
    # computed for both drives are attached to
    # ``result.ladder_voltages`` (``None`` otherwise) so the ladders can be
    # re-read OFFLINE — adjacent-pair phase slope (which way the dominant
    # wave travels), standing-wave ratio, subset matrix-pencil fits —
    # without a second FDTD run and without trusting any incident/outgoing
    # LABEL. The dict is documented field-by-field on
    # :class:`~rfx.api._spec.CoaxMSLTransitionResult`. It is built from
    # ``.copy()`` of arrays that are complete before, and consumed by,
    # :func:`_assemble_coax_msl_transition_from_voltages`, so every
    # S-parameter number is bit-identical with the option off or on —
    # gated by ``tests/unit/sparams/test_coax_msl_transition_ladder_dump.py::
    # test_return_ladder_voltages_does_not_perturb_s`` (byte-identity A/B,
    # the same discipline as the flux witness above; the round-trip
    # assertion there also proves the dump IS what the assembler consumed).
    # The same flag switches on the LADDER SELF-CONSISTENCY
    # WITNESS, ``result.ladder_split_gamma_dev`` /
    # ``result.ladder_split_reflection_decades`` (``None`` when the flag
    # is off): a disjoint-half refit of each ladder, i.e. a Python loop of
    # 2 drives x n_freqs x 2 matrix pencils per ladder over exactly the
    # arrays the dump exposes. It is computed AFTER the assembler from
    # those arrays and moves no S-parameter number; it is opt-in so that
    # a default call pays for no extra pencil solves. Documented
    # field-by-field on :class:`~rfx.api._spec.CoaxMSLTransitionResult`.
    # ---- ladder self-consistency witness (REPORT-ONLY) ----
    # Refit each port's ladder on two disjoint contiguous halves with the
    # same reference plane and the same extractor, and report the
    # disagreement. Computed AFTER the assembler has produced every
    # number above, from the same arrays, so it cannot move one:
    # ``fit_residual`` cannot detect that its own window is the problem,
    # this can. See _ladder_split_witness for the measured separation.
    # OPT-IN with the ladder dump: it is a Python-loop refit (2 drives x
    # n_freqs x 2 matrix pencils per ladder) over the very arrays
    # ``return_ladder_voltages`` exposes, so a default call never runs
    # it and both fields stay None.
    ladder_split_gamma_dev = None
    ladder_split_reflection_decades = None
    if return_ladder_voltages:
        _split_g_coax, _split_d_coax = _ladder_split_witness(
            z_planes_coax_m, v_coax_by_drive, ref_coax_m)
        _split_g_msl, _split_d_msl = _ladder_split_witness(
            np.asarray(xs_sorted), v_msl_by_drive, float(junction_x))
        ladder_split_gamma_dev = np.stack([_split_g_coax, _split_g_msl])
        ladder_split_reflection_decades = np.stack([_split_d_coax, _split_d_msl])

    # ladder dump (read-only). Taken AFTER the assembler consumed
    # the very same arrays, so no number above can depend on the flag.
    # ``msl_ladder_i`` is recomputed here with the same expression the
    # drive loop used (``i_x`` there is loop-local by construction).
    ladder_voltages = None
    if return_ladder_voltages:
        ladder_voltages = {
            "coax_ladder_v": v_coax_by_drive.copy(),
            "coax_ladder_z_m": np.asarray(z_planes_coax_m, dtype=np.float64).copy(),
            "coax_ladder_k": np.asarray(probes_coax, dtype=np.int64),
            "msl_ladder_v": v_msl_by_drive.copy(),
            "msl_ladder_x_m": np.asarray(xs_sorted, dtype=np.float64),
            "msl_ladder_i": np.asarray(
                [int(grid.position_to_index((x, y_centre, msl_z_lo))[0])
                 for x in xs_sorted],
                dtype=np.int64,
            ),
            "drive_order": ("coax", "msl"),
            "ref_coax_m": float(ref_coax_m),
            "ref_msl_m": float(junction_x),
            "z0_ref": line_impedances.copy(),
        }
    return ladder_split_gamma_dev, ladder_split_reflection_decades, ladder_voltages

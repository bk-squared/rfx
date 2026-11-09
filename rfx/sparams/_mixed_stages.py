from __future__ import annotations

import numpy as np

from rfx.errors import AccuracyWarning
from rfx.sources.msl_port import MSLPort


def _validate_mixed_family(
    RfxRefusal, RfxUnsupported, _diagnostics, diagnostic_refusal, self,
):
    # ---- Registration guards (v1 envelope) --------------------------
    if not self._msl_ports:
        raise diagnostic_refusal(RfxRefusal(
            "compute_mixed_s_matrix() needs at least one add_msl_port() "
            "registration (for a pure lumped/wire multiport use the "
            "production scan driver / extract_s_matrix)."
        ), _diagnostics)
    lw_entries = [pe for pe in self._ports if pe.is_port]
    if not lw_entries:
        raise diagnostic_refusal(RfxRefusal(
            "compute_mixed_s_matrix() needs at least one sparam-eligible "
            "add_port() lumped/wire port (impedance != 0). For a pure "
            "MSL multiport use compute_msl_s_matrix()."
        ), _diagnostics)
    if any(pe.is_soft_source for pe in self._ports):
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix() does not support bare sources / "
            "0-ohm ports (add_source or add_port(impedance=0)): they "
            "are not excite-gated and would fire in EVERY drive run, "
            "contaminating the single-drive S-parameter contract."
        ), _diagnostics)
    if self._waveguide_ports or self._floquet_ports:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix() v1 covers lumped/wire + MSL only; "
            "waveguide/Floquet ports are not part of the validated "
            "mixed method."
        ), _diagnostics)
    _mixed_non_x = [
        pe.name for pe in self._msl_ports
        if pe.direction not in ("+x", "-x")
    ]
    if _mixed_non_x:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix() v1 covers '+x'/'-x' MSL ports "
            f"only; {_mixed_non_x} are not x-directed. The y-directed "
            "MSL support is available for compute_msl_s_matrix(); "
            "this mixed method is itself EXPERIMENTAL with "
            "both diagonals unverified, so it is fenced rather than "
            "extended untested. Use compute_msl_s_matrix() for a pure "
            "MSL multiport."
        ), _diagnostics)
    if self._coaxial_ports:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix() v1 covers lumped/wire + MSL only; "
            "coaxial ports need a separate calibration contract."
        ), _diagnostics)
    if self._tfsf is not None:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix() is not supported together with "
            "TFSF; TFSF is a plane-wave source, not a port."
        ), _diagnostics)
    is_wire = [pe.extent is not None for pe in lw_entries]
    if any(is_wire) and not all(is_wire):
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix(): mixed lumped + wire port sets "
            "are not supported (the off-diagonal wave-decomposition "
            "conventions differ — same fence as "
            "compute_lumped_wire_s_matrix_via_scan)."
        ), _diagnostics)
    wire_mode = all(is_wire)
    if any(getattr(pe, "reference_plane_cells", None) for pe in lw_entries):
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix() v1 does not support "
            "add_port(reference_plane_cells=...); the mixed method uses "
            "the delivered-power witness for magnitude honesty instead."
        ), _diagnostics)
    if (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    ):
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix() v1 supports the uniform mesh "
            "only (non-uniform meshes are outside the validated scope"
            ")."
        ), _diagnostics)
    if self._refinement is not None:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_mixed_s_matrix() is not supported with SBP-SAT "
            "subgridding."
        ), _diagnostics)
    return lw_entries, wire_mode


def _prepare_mixed_probes(
    grid, msl_h_plane_stencil, msl_probe_x_coords_n, self,
):
    # ---- MSL geometry prep (mirrors compute_msl_s_matrix; uniform
    # lane only, so the NU grid/dz-profile machinery is not needed) ----
    entries = self._resolve_msl_probe_entries(grid)
    n_msl = len(entries)
    msl_ports: list[MSLPort] = []
    for pe in entries:
        x_feed, y_centre, z_lo = pe.position
        msl_ports.append(MSLPort(
            feed_x=float(x_feed),
            y_lo=float(y_centre - pe.width / 2),
            y_hi=float(y_centre + pe.width / 2),
            z_lo=float(z_lo),
            z_hi=float(z_lo + pe.height),
            direction=pe.direction,
            impedance=pe.impedance,
            excitation=pe.waveform,
        ))
    # Probe-0 x-coordinate per port: the S1 V*I wave split lives at
    # probe 0 (the de-embedding reference plane); the mixed lane does
    # not run the N-probe spatial fit, so only probe 0 is recorded.
    probe_xs = [
        msl_probe_x_coords_n(
            grid, mp,
            n_probes=int(pe.n_probes),
            n_offset_cells=pe.n_probe_offset,
            n_spacing_cells=pe.n_probe_spacing,
        )
        for mp, pe in zip(msl_ports, entries)
    ]
    h_stencils = [msl_h_plane_stencil(grid, mp, xs[0])
                  for mp, xs in zip(msl_ports, probe_xs)]
    # Resolved in rfx.preflight.msl, NOT through the rfx.api._preflight
    # re-export, so that this reader and preflight's own reader
    # (_check_msl_port_geometry, which lives in that module and looks the
    # name up in its globals) share ONE lookup target.
    # tests/unit/ports/test_msl_clearance_diagnostic.py asserts that one
    # monkeypatch covers both readers; importing the re-export would
    # split them into two bindings of which only one is patched.
    from rfx.preflight.msl import msl_probe_clearance_for_port
    probe_clearance = tuple(msl_probe_clearance_for_port(
        self, pe, grid, probe_coordinates=xs,
    ) for pe, xs in zip(entries, probe_xs))
    return entries, n_msl, msl_ports, probe_xs, h_stencils, probe_clearance


def _check_mixed_ladder(
    RfxRefusal, _diagnostics, diagnostic_refusal, entries, grid, probe_xs,
    self,
):
    # Probe-LADDER validation. The V*I split records probe 0 only, but
    # must not silently accept a ladder the validated MSL lane rejects.
    # msl_probe_x_coords_n clamps coordinates outside the declared domain.
    # A probe near the domain edge can sample the trace's open end
    # (a Box is rasterized
    # only inside the declared domain — the trace does NOT continue
    # into the CPML padding, so every boundary-touching trace has an
    # OPEN end there). The plane then measures the open-stub
    # standing-wave impedance -j*Z0*cot(beta*d) ~ 1/f (~1400 ohm at
    # 1 GHz vs the 48 ohm line) instead of a travelling wave.
    # Hard guard: every ladder coordinate strictly inside (0, lx)
    # and strictly monotonic (clamp shows up as duplicates).
    # Advisory: probe-0 closer than lambda_g/4 to a domain x-edge.
    from rfx.api._preflight import msl_min_probe_clearance
    _lx_dom = float(self._domain[0])
    _clear = msl_min_probe_clearance(float(self._freq_max))
    for pe, pxs in zip(entries, probe_xs):
        xs = [float(x) for x in pxs]
        mono = all(
            (xs[q + 1] - xs[q]) * (1 if pe.direction == "+x" else -1)
            > 0.5 * float(grid.dx)
            for q in range(len(xs) - 1)
        )
        if (not mono) or min(xs) <= 0.0 or max(xs) >= _lx_dom:
            raise diagnostic_refusal(RfxRefusal(
                f"compute_mixed_s_matrix: MSL port {pe.name!r} probe "
                f"ladder ({', '.join(f'{x * 1e3:.2f}' for x in xs)} mm) "
                f"leaves the declared x-domain (0, {_lx_dom * 1e3:.2f}) "
                "mm or was clamped at its edge — the equivalent "
                "registration is rejected by compute_msl_s_matrix, and "
                "a plane near the trace's open end at the domain edge "
                "measures the stub standing wave, not the line. Face "
                "the port toward the DUT (direction), reduce "
                "n_probe_offset/n_probe_spacing, or enlarge the domain."
            ), _diagnostics)
        _edge_d = min(xs[0], _lx_dom - xs[0])
        if _edge_d < _clear:
            import warnings as _w488
            _w488.warn(
                f"compute_mixed_s_matrix: MSL port {pe.name!r} probe-0 "
                f"plane is {_edge_d * 1e3:.2f} mm from a domain x-edge "
                f"(< lambda_g/4 = {_clear * 1e3:.2f} mm at freq_max). "
                "This is a layout recommendation near a possible "
                "trace discontinuity, not an accuracy bound. Check "
                "the realized termination and observation-plane "
                "sensitivity; reliable only screens relative low signal.", AccuracyWarning,
                stacklevel=3,
            )
    return


def _prepare_mixed_indices(
    _msl_cell_profile, grid, msl_cross_section_span, msl_ports,
):
    dy_arr = _msl_cell_profile(grid, "y", grid.ny)
    dz_arr = _msl_cell_profile(grid, "z", grid.nz)
    port_idx_meta = []
    for mp in msl_ports:
        span = msl_cross_section_span(grid, mp, require_contiguous_width=True)
        port_idx_meta.append(dict(
            j_lo=span["w_lo"], j_hi=span["w_hi"],
            k_lo=span["n_lo"], k_hi=span["n_hi"],
            j_centre=span["w_centre"], k_top=span["n_hi"],
        ))
    return dy_arr, dz_arr, port_idx_meta


def _prepare_mixed_anchors(
    RfxRefusal, _msl_yz_cells, _mx_pec_edge_masks, _mx_pec_edges_traced, _trace_planes_mx, entries, freqs_arr,
    grid, hammerstad_jensen_z0_eps_eff, lw_entries, materials, msl_ports, n_lw, n_msl,
    port_idx_meta, self, wire_mode,
):
    # Analytic Hammerstad-Jensen anchor per MSL port (eps precedence
    # mirrors compute_msl_s_matrix: explicit eps_r_sub > rasterised
    # eps_r at the trace-centre substrate cell).
    z0_hj_per_port: list[float] = []
    beta0_per_port: list[np.ndarray] = []
    eps_r_ref_per_port: list[float] = []
    from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
    _c0_mixed = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
    for p_idx, pe in enumerate(entries):
        meta = port_idx_meta[p_idx]
        if pe.eps_r_sub is not None:
            eps_r_ref = float(pe.eps_r_sub)
        else:
            k_mid = (meta["k_lo"] + meta["k_hi"]) // 2
            i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
            eps_r_ref = float(np.asarray(
                materials.eps_r[i_feed_p, meta["j_centre"], k_mid]
            ))
        eps_r_ref_per_port.append(eps_r_ref)
        z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(
            pe.width, pe.height, eps_r_ref
        )
        z0_hj_per_port.append(float(z0_hj))
        beta0_per_port.append(
            2.0 * np.pi * freqs_arr * float(np.sqrt(eps_eff_hj))
            / _c0_mixed
        )

    # Trace-conductor z-cell span (closed Ampere loop needs the PEC
    # trace; mirrors compute_msl_s_matrix).
    trace_k_per_port: list[tuple[int, int]] = []
    for p_idx in range(n_msl):
        meta = port_idx_meta[p_idx]
        i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
        _k_lo_tr, _k_hi_tr = _trace_planes_mx(
            _mx_pec_edge_masks, 2, (i_feed_p, meta["j_centre"]),
            meta["k_top"], periodic=self._periodic_flags())
        if _k_lo_tr is None:
            raise RuntimeError(
                "compute_mixed_s_matrix: no realized PEC trace "
                "conductor found above the substrate top for MSL port "
                f"{entries[p_idx].name!r}; the closed Ampere-loop "
                "current needs the trace. Declare the microstrip trace "
                "as a Box(material='pec') (a volume) or as a "
                "zero-thickness Box / add_thin_conductor (a sheet"
                ")."
            )
        trace_k_per_port.append((_k_lo_tr, _k_hi_tr))

    # Wire live-cell counts for the per-cell impedance normalization
    # (mirrors compute_lumped_wire_s_matrix_via_scan).
    n_live_lw = np.ones(n_lw, dtype=np.int64)
    if wire_mode:
        from rfx.sources.sources import WirePort, _wire_port_live_cells
        axis_map = {"ex": 0, "ey": 1, "ez": 2}
        for idx, pe in enumerate(lw_entries):
            if _mx_pec_edges_traced:
                raise RfxRefusal(
                    f'compute_mixed_s_matrix: wire port {idx} live cells depend on PEC '
                    'geometry that is traced, so they cannot be counted. '
                    'Make that geometry concrete or remove the port.'
                )
            end = list(pe.position)
            end[axis_map[pe.component]] += pe.extent
            wp = WirePort(
                start=pe.position, end=tuple(end),
                component=pe.component, impedance=pe.impedance,
                excitation=pe.waveform,
            )
            n_live_lw[idx] = _wire_port_live_cells(
                grid, wp, _mx_pec_edge_masks)[2]
    return z0_hj_per_port, beta0_per_port, eps_r_ref_per_port, trace_k_per_port, n_live_lw


def _check_mixed_flux(
    RfxRefusal, RfxUnsupported, _diagnostics, diagnostic_refusal, lw_entries, magnitude_channel, refuse_port_on_magnetic_face,
    self,
):
    if magnitude_channel not in ("flux", "wave"):
        raise diagnostic_refusal(RfxRefusal(
            "compute_mixed_s_matrix: magnitude_channel must be 'flux' "
            f"(default) or 'wave', got {magnitude_channel!r}."
        ), _diagnostics)
    if magnitude_channel == "flux":
        try:
            refuse_port_on_magnetic_face(self, "compute_mixed_s_matrix",
                                         reason="Flux magnitudes cannot compare these powers.")
        except ValueError as error:
            raise diagnostic_refusal(error, _diagnostics) from None
        # The per-port flux box is CLOSED by the z-lo PEC ground: it
        # has five faces and omits the bottom because flux through a
        # PEC face is identically zero. It also treats pe.extent as a
        # VERTICAL height. Both are physical preconditions, so check
        # them instead of assuming (review finding: a user with an
        # open z-lo boundary or a horizontal port would otherwise get
        # a silently wrong P_net on the DEFAULT channel).
        _pec_faces = (
            self._boundary_spec.pec_faces()
            if self._boundary_spec is not None else set()
        )
        if "z_lo" not in _pec_faces:
            raise diagnostic_refusal(RfxUnsupported(
                "compute_mixed_s_matrix(magnitude_channel='flux') "
                "requires a PEC z_lo boundary: the per-port flux box "
                "omits its bottom face because flux through a PEC "
                "ground is identically zero. Your z_lo face is "
                f"{sorted(_pec_faces) or 'not PEC'}, so the box would "
                "not be closed and P_net would be wrong. Use "
                "BoundarySpec(z=Boundary(lo='pec', ...)), or pass "
                "magnitude_channel='wave' (which carries the "
                "port-cell deflation instead)."
            ), _diagnostics)
        _bad = [pe for pe in lw_entries if pe.component != "ez"]
        if _bad:
            raise diagnostic_refusal(RfxUnsupported(
                "compute_mixed_s_matrix(magnitude_channel='flux') "
                "supports vertical (component='ez') lumped/wire ports "
                "only: the flux box is built assuming the port extent "
                "is a z height above the ground plane. Offending "
                f"component(s): {sorted({pe.component for pe in _bad})}."
            ), _diagnostics)
    return


def _prepare_mixed_observers(
    declaration_variant, entries, freqs_arr, grid, jnp, lw_entries, magnitude_channel,
    probe_xs, self,
):
    observers = declaration_variant(
        self, _internal_probe_indices=set(self._internal_probe_indices))
    # Ring-down settling witness probes at every MSL probe plane,
    # mid-substrate (worst plane wins — a single plane is
    # standing-wave-node sensitive; mirrors compute_msl_s_matrix).
    _witness_base = len(observers._probes)
    _witness_total = 0
    for pe_w, pxs_w in zip(entries, probe_xs):
        # This calculator-created midpoint is not a declaration. Keep
        # its historical half-to-even node when the substrate has an
        # odd number of cells; user port/source coordinates still use
        # the shared lower-node tie rule.
        witness_z = float(pe_w.position[2]) + 0.5 * float(pe_w.height)
        dz = float(grid.cells(2)[0])   # uniform mesh (refused graded above)
        witness_z = int(round(witness_z / dz)) * dz
        for _x_w in pxs_w:
            observers.add_probe(
                position=(
                    float(_x_w),
                    float(pe_w.position[1]),
                    witness_z,
                ),
                component="ez",
            )
            _witness_total += 1
    observers._internal_probe_indices.update(
        range(_witness_base, _witness_base + _witness_total)
    )

    # Flux surfaces for the magnitude channel (registered once —
    # geometry is drive-independent). Lumped/wire port p: a closed
    # 5-face box over the z-lo ground plane (flux through the PEC
    # ground face is identically zero); outward-signed face list.
    # MSL port p: one full-cross-section x-plane at probe 0 (its
    # de-embedding reference plane).
    flux_faces_lw: list[list[tuple[str, float]]] = []
    flux_names_msl: list[str] = []
    if magnitude_channel == "flux":
        _m = 3.0 * float(grid.dx)
        for p, pe in enumerate(lw_entries):
            x0_p, y0_p, _z0p = (float(c) for c in pe.position)
            z_top = float(pe.extent or grid.dx) + _m
            faces = []
            for ax, coord, size, center, sgn in (
                ("x", x0_p - _m, (2 * _m, z_top), (y0_p, z_top / 2), -1.0),
                ("x", x0_p + _m, (2 * _m, z_top), (y0_p, z_top / 2), +1.0),
                ("y", y0_p - _m, (2 * _m, z_top), (x0_p, z_top / 2), -1.0),
                ("y", y0_p + _m, (2 * _m, z_top), (x0_p, z_top / 2), +1.0),
                ("z", z_top, (2 * _m, 2 * _m), (x0_p, y0_p), +1.0),
            ):
                nm = f"_mixed_flux_lw{p}_{ax}{'+' if sgn > 0 else '-'}{coord:.6g}"
                observers.add_flux_monitor(
                    axis=ax, coordinate=float(coord),
                    freqs=jnp.asarray(freqs_arr),
                    size=(float(size[0]), float(size[1])),
                    center=(float(center[0]), float(center[1])),
                    name=nm,
                )
                faces.append((nm, sgn))
            flux_faces_lw.append(faces)
        for p, pxs in enumerate(probe_xs):
            nm = f"_mixed_flux_msl{p}"
            observers.add_flux_monitor(
                axis="x", coordinate=float(pxs[0]),
                freqs=jnp.asarray(freqs_arr), name=nm,
            )
            flux_names_msl.append(nm)
    return observers, _witness_base, _witness_total, flux_faces_lw, flux_names_msl


def _prepare_mixed_drive(
    GaussianPulse, _dc, _register_msl_h_planes, declaration_variant, entries, fam, freqs_arr,
    h_stencils, jnp, loc, observers, probe_xs, run_idx, self,
):
    # Excite exactly one port; every other port keeps its
    # physical termination (matched resistor cells for
    # lumped/wire, passive probe column for MSL). The driven
    # lumped/wire port's default waveform is synthesised by
    # the runner; the driven MSL port mirrors the
    # compute_msl_s_matrix default.
    run_ports = [
        _dc.replace(pe, excite=(fam == "lw" and k == loc))
        for k, pe in enumerate(self._ports)
    ]
    # Use the resolved ladder that the material/probe setup describes.
    run_msl = []
    for k, pe in enumerate(entries):
        driven = fam == "msl" and k == loc
        wf = (
            (pe.waveform if pe.waveform is not None else
             GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8))
            if driven else None
        )
        run_msl.append(_dc.replace(pe, excite=driven, waveform=wf))
    variant = declaration_variant(
        observers, _ports=run_ports, _msl_ports=run_msl,
        _internal_probe_indices=set(observers._internal_probe_indices),
        _dft_plane_regions=dict(getattr(self, "_dft_plane_regions", {})))

    # Per-run named DFT planes: Ez at probe 0 (line voltage),
    # Hy+Hz at probe 0 (closed Ampere-loop current legs).
    names = []
    h_names = []
    for p_idx, pxs in enumerate(probe_xs):
        nm = f"_mixed_run{run_idx}_p{p_idx}"
        # Full ez ladder: probe 0 feeds the V*I wave split;
        # the N-probe least-squares fit (measured line Zc for
        # the MSL diagonal — see the z0_fit block below) needs
        # every plane.
        for q_idx, x_c in enumerate(pxs):
            variant.add_dft_plane_probe(
                axis="x", coordinate=float(x_c), component="ez",
                freqs=jnp.asarray(freqs_arr),
                name=nm + f"_ez{q_idx}",
            )
        h_names.append(_register_msl_h_planes(
            variant, nm, h_stencils[p_idx], ("hy", "hz"),
            jnp.asarray(freqs_arr),
        ))
        names.append(nm)

    variant._time_recorded_dft_planes = (
        {name + f"_ez{q}" for p, name in enumerate(names) for q in range(len(probe_xs[p]))}
        | {name for pairs in h_names for pair in pairs for name in pair}
    )
    return variant, names, h_names


def _run_mixed_drive(
    _diagnostics, _mx_pec_sheets, _mx_pec_wires, box_lw, debye_spec, flux_faces_lw, flux_names_msl,
    freqs_arr, grid, i_lw, lorentz_spec, magnitude_channel, materials, merge_diagnostics,
    n_freqs_used, n_lw, n_steps, pec_mask, plane_msl, run_idx,
    v_lw, variant, vref_lw, wire_mode,
):
    raw = variant._forward_from_materials(
        grid, materials, debye_spec, lorentz_spec,
        n_steps=n_steps, checkpoint=False, pec_mask=pec_mask,
        pec_sheets=tuple(_mx_pec_sheets),
        pec_wires=tuple(_mx_pec_wires),
        port_s11_freqs=freqs_arr,
        _return_raw_port_sparams=True,
    )
    _diagnostics = merge_diagnostics(_diagnostics, raw.get("diagnostics", ()))
    accs = raw["wire"] if wire_mode else raw["lumped"]
    if accs is None or len(accs) != n_lw:
        raise RuntimeError(
            "compute_mixed_s_matrix: production scan returned "
            f"{0 if accs is None else len(accs)} "
            f"{'wire' if wire_mode else 'lumped'} accumulators "
            f"for run {run_idx}, expected {n_lw}."
        )
    for i_port in range(n_lw):
        _spec_i, vi = accs[i_port]
        v_lw[run_idx, i_port, :] = np.asarray(vi[0])
        i_lw[run_idx, i_port, :] = np.asarray(vi[1])
        if wire_mode:
            # Pre-injection drive-sample reference channel (vi[4]): the wire wave
            # algebra is calibrated against this sample; passive samples are
            # slot-invariant. A shorter tuple marks a lane without a separate v_ref channel
            # (the NU lane): it samples POST-injection and its decomposer is calibrated
            # in that frame, so vi[0] is that lane's correct drive reference.
            vref_lw[run_idx, i_port, :] = np.asarray(
                vi[4] if len(vi) > 4 else vi[0])
        else:
            # The lumped family also uses the pre-injection drive sample, vi[2]
            # (scripts/diagnostics/lumped_port_known_load_line.py). A shorter tuple has
            # no separate reference channel, and vi[0] is that sample.
            vref_lw[run_idx, i_port, :] = np.asarray(
                vi[2] if len(vi) > 2 else vi[0])

    planes = raw.get("dft_planes")
    if not planes:
        raise RuntimeError(
            "compute_mixed_s_matrix: the production scan "
            "returned no DFT planes — the raw hook "
            "is out of sync with _forward_from_materials."
        )

    if magnitude_channel == "flux":
        from rfx.probes.probes import flux_spectrum
        fmon = raw.get("flux_monitors")
        if not fmon:
            raise RuntimeError(
                "compute_mixed_s_matrix: the production scan "
                "returned no flux monitors — the "
                "flux hook is out of sync with "
                "_forward_from_materials."
            )
        for p, faces in enumerate(flux_faces_lw):
            acc = np.zeros(n_freqs_used)
            for nm, sgn in faces:
                acc = acc + sgn * np.asarray(
                    flux_spectrum(fmon[nm]), dtype=np.float64
                )
            box_lw[run_idx, p, :] = acc
        for p, nm in enumerate(flux_names_msl):
            plane_msl[run_idx, p, :] = np.asarray(
                flux_spectrum(fmon[nm]), dtype=np.float64
            )
    return _diagnostics, raw, planes, v_lw, i_lw, vref_lw, box_lw, plane_msl


def _score_mixed_record(
    _witness_base, _witness_total, dy_arr, dz_arr, freqs_arr, grid, h_names,
    h_stencils, is_tracer, msl_ports, names, port_idx_meta, probe_xs,
    raw, run_idx, self, settling_db_runs, settling_details, trace_k_per_port, variant,
):
    # Score recorded S channels and witness probes.
    _ts = raw.get("time_series")
    if _ts is not None and not is_tracer(_ts):
        _ts_np = np.asarray(
            _ts[:, _witness_base:_witness_base + _witness_total],
        )
        if (_ts_np.shape[0] >= 10
                and _ts_np.shape[1] == _witness_total):
            from rfx.measurement.record_end import injection_end
            from rfx.sources.waveguide_port import settling_db_from_named_records
            _source_end = injection_end(
                _ts_np.shape[0], raw.get("dt", grid.dt),
                records=variant._probes[_witness_base:_witness_base + _witness_total],
                grid=grid, sim=variant)
            from rfx.sparams._tail_witness import msl_time_channels
            _plane_records = raw.get("dft_time_records")
            _channels = msl_time_channels(
                _plane_records,
                [[name + f"_ez{q}" for q in range(len(probe_xs[p]))]
                 for p, name in enumerate(names)],
                h_names, h_stencils,
                [{**meta, "dy_arr": dy_arr} for meta in port_idx_meta],
                trace_k_per_port, dz_arr, [port.direction for port in msl_ports],
            ) if _plane_records else []
            _channels.extend((f"{record.id}/{channel}", record[channel])
                             for record in raw.get("sparam_time_records") or ()
                             for channel in record.channels)
            _channels.extend((f"probe{i}", _ts_np[:, i])
                             for i in range(_ts_np.shape[1]))
            from rfx.sparams._settling_groups import msl_column_groups, port_column_groups
            _groups, _labels = msl_column_groups(_channels)
            _port_groups, _ = port_column_groups(raw.get("sparam_time_records") or ())
            for record in raw.get("sparam_time_records") or ():
                _groups.update((f"{record.id}/{channel}", key)
                               for channel, key in zip(record.channels, _port_groups[record.id]))
            _groups.update((f"probe{i}", "ez") for i in range(_ts_np.shape[1]))
            settling_db_runs[run_idx], settling_details[run_idx] = settling_db_from_named_records(
                _channels, source_end_index=_source_end, dt=raw.get("dt", grid.dt),
                freqs=freqs_arr, freq_max=self._freq_max, return_detail=True,
                scale_groups=_groups, _channel_labels=_labels)

    return settling_db_runs, settling_details


def _fit_mixed_impedance(
    beta0_per_port, entries, i_msl, jax, jnp, n_freqs_used,
    n_lw, n_msl, probe_xs, v_lad, z0_hj_per_port,
):
    # ---- N-probe line Zc: DIAGNOSTIC ONLY ---------
    # This lane does NOT substitute a fitted Zc into the MSL
    # diagonal: the fit's sign is not stable here, and a substitution
    # that silently falls back to the analytic anchor can look like
    # a confirmed measurement.
    #
    # Mechanism (why a sign fix would not be enough): the model
    # V_n = alpha*exp(-j beta x_n) + gamma*exp(+j beta x_n) is
    # invariant under (beta -> -beta, alpha <-> gamma), so a beta
    # scan that lands on the wrong branch SWAPS the two wave
    # roles and flips the sign of (alpha - gamma) — and therefore
    # of z0 = (alpha - gamma)/I1. Measured on one fixture: the
    # fitted sign differs between num_periods=4 and 20.
    #
    # msl_loop_current uses a direction-dependent convention. Own-drive
    # diagonal on the committed thru fixture:
    # Re((alpha-gamma)/I1) = +57.52 ohm at the "+x" port and
    # -57.56 ohm at the "-x" port (same magnitude to 0.08%), so
    # the lane is self-consistent: |S11|=|S22| to 5 decimals,
    # reciprocity 1.27e-05, column power <= 0.99998, both reported
    # Z0 positive. The beta-branch instability above is separate.
    #
    # The measurement (VESSL 369367257265, driver
    # scripts/diagnostics/msl_passive_port_reflection.py) found: the two ports read
    # |Gamma| 0.1799 and 0.1759 on the shipped R = 50 ohm fixture.
    # The cell-aligned and +/-y-rotated variants reproduce that to 4
    # decimals. The fitted Zc here is 41.9 ohm against the
    # Hammerstad-Jensen 47.9 ohm -- a dx bias, not a
    # termination error. What remains unexplained is the passive
    # port's reflection itself: |Gamma_passive| = 0.176-0.180 at
    # R = 50 ohm with no mechanism named. Both single-reflector
    # models were falsified over a 25 / Zc / 50 / open R sweep
    # (shunt Zc/(2R+Zc) misses by up to 0.119, end |R-Zc|/(R+Zc)
    # by 0.242; implied k runs 1.91 -> 1.03, so the shape does not
    # follow 1/(2R+Zc)); reported and stopped there, no two-
    # parameter fit.
    #
    # The fit is still computed and EXPOSED (return_diagnostics)
    # because it is the only handle on the measured-vs-analytic Zc
    # gap (41.9 ohm fitted vs 47.9 ohm Hammerstad-Jensen on this
    # mesh), but it never feeds a shipped
    # number, and its magnitude is reported without a sign claim.
    from rfx.probes.msl_wave_decomp import extract_msl_nprobe
    z0_msl_fit = np.full((n_msl, n_freqs_used), np.nan)
    beta_railed_msl = np.zeros((n_msl, n_freqs_used), dtype=bool)
    for d in range(n_msl):
        run_d = n_lw + d
        n_p = len(probe_xs[d])
        res_fit = extract_msl_nprobe(
            jnp.asarray(v_lad[run_d, d, :n_p, :].T),
            jnp.asarray(np.asarray(probe_xs[d], dtype=float)),
            jnp.asarray(i_msl[run_d, d, :]),
            jnp.asarray(beta0_per_port[d]),
            z0_hj=z0_hj_per_port[d],
        )
        z0_msl_fit[d, :] = np.abs(np.asarray(
            jax.lax.stop_gradient(res_fit["z0"])
        ).astype(np.complex128))
        beta_railed_msl[d, :] = np.asarray(
            jax.lax.stop_gradient(res_fit["beta_railed"])
        ).astype(bool)

        _dev = float(np.max(
            np.abs(z0_msl_fit[d, :] - z0_hj_per_port[d])
            / z0_hj_per_port[d]
        ))
        if _dev > 0.10:
            import warnings as _wz
            _wz.warn(
                f"compute_mixed_s_matrix: the N-probe |Zc| fit for "
                f"MSL port {entries[d].name!r} deviates up to "
                f"{_dev * 100:.1f}% from analytic Hammerstad-Jensen "
                f"{z0_hj_per_port[d]:.2f} ohm. The MSL diagonal "
                "uses the analytic anchor; this fitted value is "
                "not used in S. That does not certify the shared "
                "V/I inputs. Check settling_db, the relative "
                "low-signal mask, probe_clearance and observation-"
                "plane sensitivity before interpreting S.", AccuracyWarning,
                stacklevel=3,
            )
    return z0_msl_fit, beta_railed_msl


def _shift_mixed_planes(
    S, _complex_dtype, entries, eps_r_ref_per_port, freqs_arr, grid, msl_ports,
    n_lw, n_msl, probe_xs, s_wave_full,
):
    # ---- Reference plane: MSL probe 0 -> declared feed plane -------
    # The same resolver as compute_msl_s_matrix (one expression for
    # both calculators): L_p from the realized nodes, the closed-form
    # shift beta (Hammerstad-Jensen + Kirschning-Jansen dispersion),
    # S[i, j] <- D_i S[i, j] D_j with D = exp(-j beta L) on the MSL
    # rows and columns and D = 1 on the lumped/wire ones. After the
    # flux magnitude override, which it cannot disturb (|D| = 1).
    # S_wave (the pre-override wave-channel matrix) gets the same D, so
    # every S this result reports is referred to one plane per port.
    from rfx.measurement.msl_planes import (
        resolve_msl_planes as _resolve_msl_planes,
        apply_feed_plane_shift as _apply_feed_plane_shift,
        full_port_shift as _full_port_shift,
    )
    _msl_plane = _resolve_msl_planes(
        grid, msl_ports,
        [int(pe.n_probe_offset) for pe in entries],
        [xs[0] for xs in probe_xs],
        [float(pe.width) for pe in entries],
        [float(pe.height) for pe in entries],
        eps_r_ref_per_port, freqs_arr,
    )
    plane_offsets_all, shift_beta_all = _full_port_shift(
        n_lw + n_msl, range(n_lw, n_lw + n_msl), _msl_plane,
    )
    S = _apply_feed_plane_shift(
        S, plane_offsets_all, shift_beta_all,
    ).astype(_complex_dtype)
    if s_wave_full is not None:
        s_wave_full = np.asarray(_apply_feed_plane_shift(
            s_wave_full, plane_offsets_all, shift_beta_all,
        ))
    return S, s_wave_full, plane_offsets_all, shift_beta_all

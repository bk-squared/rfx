from __future__ import annotations

from rfx.errors import AccuracyWarning


def _validate_msl_family(
    RfxRefusal, RfxUnsupported, _diagnostics, diagnostic_refusal, self,
):
    if not self._msl_ports:
        raise diagnostic_refusal(RfxRefusal("No MSL ports registered. Call add_msl_port() first."), _diagnostics)
    if self._ports or self._waveguide_ports or self._floquet_ports:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_msl_s_matrix() is defined only for add_msl_port(...) "
            "families in the current simulation. Use separate "
            "simulations for add_port(...), add_waveguide_port(...), "
            "or add_floquet_port(...) S-parameter workflows."
        ), _diagnostics)
    if self._tfsf is not None:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_msl_s_matrix() is not supported together with TFSF; "
            "TFSF is a plane-wave source, not an MSL port."
        ), _diagnostics)
    if self._coaxial_ports:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_msl_s_matrix() does not include add_coaxial_port(...); "
            "coaxial-port S-parameters need a separate validated V/I "
            "extraction and calibration contract."
        ), _diagnostics)
    is_nonuniform = (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    )
    if is_nonuniform and any(
        getattr(pe, "mode", "laplace") == "eigenmode"
        for pe in self._msl_ports
    ):
        raise diagnostic_refusal(RfxUnsupported(
            "compute_msl_s_matrix() on a non-uniform mesh supports "
            "mode='laplace'/'uniform' (Ez static-Laplace feed) only; the "
            "eigenmode J+M launch needs the magnetic-source channel that "
            "the non-uniform runner does not carry. Use mode='laplace' "
            "(the add_msl_port default) on the graded-mesh lane."
        ), _diagnostics)
    if self._refinement is not None:
        raise diagnostic_refusal(RfxUnsupported(
            "compute_msl_s_matrix() is not supported with SBP-SAT "
            "subgridding."
        ), _diagnostics)
    return is_nonuniform


def _prepare_msl_probes(
    entries, grid, msl_h_plane_stencil, msl_port_from_entry, msl_probe_x_coords_n, self,
):
    # Build MSLPort descriptors and probe coords once (geometry shared).
    # Issue #661: msl_port_from_entry projects ``position`` onto the
    # port frame for whichever in-plane axis ``direction`` names.
    msl_ports = [msl_port_from_entry(pe) for pe in entries]

    # N-probe placement (issue #80 Fix C). Probe n sits at
    # offset + n*spacing cells from the feed plane. N >= 3.
    n_probes_per_port = [int(pe.n_probes) for pe in entries]
    probe_xs = [
        msl_probe_x_coords_n(
            grid, mp,
            n_probes=n_probes,
            n_offset_cells=pe.n_probe_offset,
            n_spacing_cells=pe.n_probe_spacing,
        )
        for mp, pe, n_probes in zip(msl_ports, entries, n_probes_per_port)
    ]
    # Resolve before registering internal observations or advancing fields.
    # H[i] lies half a propagation cell beyond E[i], for either port sign.
    h_stencils = [msl_h_plane_stencil(grid, mp, xs[0])
                  for mp, xs in zip(msl_ports, probe_xs)]
    # Resolved in rfx.preflight.msl, NOT through the rfx.api._preflight
    # re-export, so that this reader and preflight's own reader
    # (_check_msl_port_geometry, which lives in that module and looks the
    # name up in its globals) share ONE lookup target. #980 Phase 3 leg 1
    # moved both; before it, a single monkeypatch on rfx.api._preflight
    # covered the two, and
    # tests/unit/ports/test_msl_clearance_diagnostic.py asserts that it
    # still does -- the re-export would have split them into two bindings
    # of which only one is ever patched.
    from rfx.preflight.msl import msl_probe_clearance_for_port
    probe_clearance = tuple(msl_probe_clearance_for_port(
        self, pe, grid, probe_coordinates=xs,
    ) for pe, xs in zip(entries, probe_xs))
    return msl_ports, n_probes_per_port, probe_xs, h_stencils, probe_clearance


def _prepare_msl_geometry(
    _msl_cell_profile, entries, eps_override, freqs_arr, grid, hammerstad_jensen_z0_eps_eff, is_nonuniform,
    jax, msl_ampere_pair, msl_cell, msl_cross_section_span, msl_ports, n_ports, np,
    self,
):
    # ``probe_xs`` are the N physical x-coordinates fed to the
    # N-probe extractor (issue #80 Fix C), which fits
    # V_n = alpha*exp(-j*beta*x_n) + gamma*exp(+j*beta*x_n). The
    # coordinates are increasing for ``+x`` ports and decreasing for
    # ``-x`` ports; the extractor anchors the model at probe 0 and
    # only uses coordinate differences, so feeding raw physical x
    # keeps alpha = the +x-travelling wave for BOTH port directions
    # — matching the legacy 3-probe sign convention the S11 sign
    # was validated against.

    # Per-axis cell-size arrays for V/I integration. Both uniform and
    # non-uniform grids are supported — NonUniformGrid exposes per-cell
    # dx_arr/dy_arr/dz (NOT *_profile); see _msl_cell_profile.
    #
    # Issue #661: ports may point along different in-plane axes, so the
    # transverse profiles are resolved PER PORT. The substrate-normal
    # profile is always z (the normal axis is welded — see
    # msl_axis_roles), which is why the modal-voltage integration below
    # needs no per-port branch.
    _axis_n = {"x": grid.nx, "y": grid.ny, "z": grid.nz}

    def _prof(ax):
        return _msl_cell_profile(grid, ax, _axis_n[ax])

    dz_arr = _prof("z")     # substrate-normal profile (V integration)

    # Fixed cross-section indices per port (same across all runs).
    # Names are the historical x-frame names; their MEANING is
    # (width, substrate-normal): ``j_*`` indexes the trace-width axis
    # and ``k_*`` the normal axis. A DFT plane normal to the
    # propagation axis is stored as [freq, width, normal] for BOTH
    # "x" and "y" plane normals (probes.py: axis 0 -> (ny, nz),
    # axis 1 -> (nx, nz)), so these indices address the recorded
    # planes directly without a per-direction branch.
    port_idx_meta = []
    for mp in msl_ports:
        span = msl_cross_section_span(grid, mp, require_contiguous_width=True)
        a_ax, b_ax = msl_ampere_pair(mp.direction)
        port_idx_meta.append(dict(
            j_lo=span["w_lo"], j_hi=span["w_hi"],
            k_lo=span["n_lo"], k_hi=span["n_hi"],
            j_centre=span["w_centre"], k_top=span["n_hi"],
            height=mp.z_hi - mp.z_lo,
            i_feed=span["i_feed"],
            prop_axis=span["prop_axis"], width_axis=span["width_axis"],
            normal_axis=span["normal_axis"], sign=span["sign"],
            prop_idx=span["prop_idx"], width_idx=span["width_idx"],
            normal_idx=span["normal_idx"],
            # Closed-Ampere-loop transverse pair (a_hat x b_hat = p_hat).
            a_axis=a_ax, b_axis=b_ax,
            a_is_width=(a_ax == span["width_axis"]),
            a_arr=_prof(a_ax), b_arr=_prof(b_ax),
            h_a=f"h{a_ax}", h_b=f"h{b_ax}",
        ))

    # Analytic Hammerstad-Jensen anchor per port (issue #80 Fix C).
    # ``beta0_per_port[p]`` is the (n_freqs,) propagation-constant
    # guess ``omega * sqrt(eps_eff) / c`` used to centre the N-probe
    # extractor's robust beta scan; ``z0_hj_per_port[p]`` is the
    # analytic Z0 used by the honesty guard.
    #
    # Substrate permittivity precedence (mirrors rfx/runners/uniform.py
    # so the beta anchor and the source see the SAME eps_r): explicit
    # add_msl_port(eps_r_sub=...) > the rasterised FDTD eps_r at the
    # trace-centre substrate cell. Reading the material array makes
    # this robust even when the user did not pass eps_r_sub — a plain
    # pe.eps_r_sub-or-1.0 fallback would anchor the scan on vacuum and
    # land beta outside the scan window for a loaded substrate.
    from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
    _C0_MSL = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
    # #679: surface_impedance_f0 sheets are supported on this lane. NO
    # ctx is built here — every device dispatch below goes through the
    # public run()/forward(), which assemble their own materials with
    # sheet_specs and build the ctx against their OWN final pec_mask
    # (run: _execute.py -> runners/uniform.py build_sheet_impedance_ctx;
    # forward: _execute.py; NU lanes: runners/nonuniform.py). The anchor
    # assembly below is deliberately sheet-free (apply_thin_conductor
    # emits nothing without a sheet_specs collector) — correct, because
    # an f0 sheet carries no eps and never enters pec_mask. This lane
    # has NO vacuum reference run, so there is no strip_sheet_impedance
    # analogue.
    _msl_pec_sheets: list = []
    _msl_pec_wires: list = []
    _msl_assembled = (
        self._assemble_materials_nu(
            grid, pec_sheets=_msl_pec_sheets, pec_wires=_msl_pec_wires)
        if is_nonuniform
        else self._assemble_materials(
            grid, pec_sheets=_msl_pec_sheets, pec_wires=_msl_pec_wires)
    )
    _msl_materials = _msl_assembled[0]
    # #931 §1.9: the trace is located by its REALIZED wall planes, not
    # by a pec_mask cell scan — a sheet-declared trace owns no cell.
    from rfx.boundaries.pec import (
        realized_pec_edge_masks as _rpem_msl,
    )
    from rfx.probes.msl_wave_decomp import (
        realized_trace_planes_on_column as _trace_planes,
    )
    _msl_pec_edge_masks = None
    if (_msl_assembled[3] is not None or _msl_pec_sheets
            or _msl_pec_wires):
        with jax.ensure_compile_time_eval():
            _msl_pec_edge_masks = _rpem_msl(
                _msl_assembled[3], sheets=tuple(_msl_pec_sheets),
                wires=tuple(_msl_pec_wires),
                periodic=self._periodic_flags())
    beta0_per_port: list[np.ndarray] = []
    z0_hj_per_port: list[float] = []
    eps_r_ref_per_port: list[float] = []
    for p_idx, pe in enumerate(entries):
        meta = port_idx_meta[p_idx]
        if pe.eps_r_sub is not None:
            eps_r_ref = float(pe.eps_r_sub)
        else:
            k_mid = (meta["k_lo"] + meta["k_hi"]) // 2
            eps_cell = msl_cell(
                pe.direction, meta["i_feed"], meta["j_centre"], k_mid
            )
            eps_r_ref = float(np.asarray(_msl_materials.eps_r[eps_cell]))
        eps_r_ref_per_port.append(eps_r_ref)
        z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(
            pe.width, pe.height, eps_r_ref
        )
        beta0_per_port.append(
            2.0 * np.pi * freqs_arr * float(np.sqrt(eps_eff_hj)) / _C0_MSL
        )
        z0_hj_per_port.append(float(z0_hj))
    # #2199: under eps_override the line between each feed plane and probe 0
    # is the solved one, so the plane shift reads its permittivity there
    # (traced; refused by name where it varies along that length).
    line_eps_per_port = None
    if eps_override is not None:
        from rfx.measurement.msl_planes import msl_line_permittivity
        line_eps_per_port = [
            msl_line_permittivity(
                eps_override, port_name=pe.name, direction=pe.direction,
                i_feed=port_idx_meta[p]["i_feed"],
                sign=int(port_idx_meta[p]["sign"]),
                n_probe_offset=int(pe.n_probe_offset),
                j_centre=port_idx_meta[p]["j_centre"],
                k_lo=port_idx_meta[p]["k_lo"], k_hi=port_idx_meta[p]["k_hi"])
            for p, pe in enumerate(entries)
        ]
    # #2014: the S reference is the declared impedance=, not the line Z0.
    declared_refs = np.asarray(
        [float(pe.impedance) for pe in entries], dtype=np.float64)

    # Trace-conductor z-cell span per port (issue #80 stage S1). The
    # closed Ampere-loop current needs the PEC trace cells; the trace
    # is the PEC run at/above the substrate top in the port's centre
    # column (the ground-plane PEC sits far below near k_lo).
    trace_k_per_port: list[tuple[int, int]] = []
    for p_idx in range(n_ports):
        meta = port_idx_meta[p_idx]
        # Walk UP the substrate-normal axis (always z) from the
        # substrate top, at the feed cell on the propagation axis and
        # the trace centre on the width axis (issue #661).
        _ij = tuple(
            meta["i_feed"] if c == meta["prop_idx"] else meta["j_centre"]
            for c in range(3) if c != meta["normal_idx"]
        )
        _k_lo_tr, _k_hi_tr = _trace_planes(
            _msl_pec_edge_masks, meta["normal_idx"], _ij, meta["k_top"],
            periodic=self._periodic_flags())
        if _k_lo_tr is None:
            raise RuntimeError(
                "compute_msl_s_matrix: no realized PEC trace conductor "
                "found above the substrate top for MSL port "
                f"{entries[p_idx].name!r}; the closed Ampere-loop "
                "current needs the trace. Declare "
                "the microstrip trace as a Box(material='pec') (a "
                "volume) or as a zero-thickness Box / add_thin_conductor "
                "(a sheet). A surface_impedance_f0 thin conductor "
                "is NOT a trace conductor here — it realizes no PEC "
                "edge, and the Ampere-loop current and V span anchor on "
                "realized PEC wall planes. Keep the trace PEC and use f0 "
                "sheets for auxiliary lossy metal only."
            )
        trace_k_per_port.append((_k_lo_tr, _k_hi_tr))
    return dz_arr, port_idx_meta, beta0_per_port, z0_hj_per_port, eps_r_ref_per_port, line_eps_per_port, declared_refs, trace_k_per_port


def _prepare_msl_witnesses(
    entries, msl_physical_point, n_ports, np, observers, port_idx_meta, probe_xs,
):
    # Point probes at every port observation plane.
    _witness_base = len(observers._probes)
    _witness_counts: list[int] = []
    for pe_w, pxs_w, meta_w in zip(entries, probe_xs, port_idx_meta):
        _w_centre_m = float(pe_w.position[meta_w["width_idx"]])
        _n_lo_m = float(pe_w.position[meta_w["normal_idx"]])
        for _x_w in pxs_w:
            # ``_x_w`` is a coordinate on the PROPAGATION axis;
            # rebuild the physical point for this port's direction.
            observers.add_probe(
                position=msl_physical_point(
                    pe_w.direction,
                    float(_x_w),
                    _w_centre_m,
                    _n_lo_m + 0.5 * float(pe_w.height),
                ),
                component="ez",
            )
        _witness_counts.append(len(pxs_w))
    _witness_total = sum(_witness_counts)
    # Mark the witness probes as library-internal so probe-placement
    # preflight advisories and the #332 tail advisory skip them
    # (issue #470: 10 self-inflicted advisories per driven run on
    # the 2-port thru buried the genuine MSL port-clearance
    # advisories, and #332 double-fired next to settling_db).
    observers._internal_probe_indices.update(
        range(_witness_base, _witness_base + _witness_total)
    )
    settling_db_runs = np.full(n_ports, np.nan)
    settling_details = [{"status": "undetermined", "reason": "traced or unavailable S channels"} for _ in range(n_ports)]
    return _witness_base, _witness_total, settling_db_runs, settling_details, observers


def _prepare_msl_drive(
    GaussianPulse, _MSLPortEntry, _register_msl_h_planes, declaration_variant, driven, entries, freqs_arr,
    h_stencils, jnp, msl_ports, n_ports, observers, port_idx_meta, probe_xs,
    self, trace_k_per_port,
):
    # Rebuild the resolved entries with exactly this drive active.
    run_entries = []
    for idx, pe in enumerate(entries):
        new_excite = (idx == driven) and pe.excite
        if new_excite:
            wf = pe.waveform if pe.waveform is not None else \
                GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
        else:
            wf = None
        # Carry the whole probe ladder: an omitted n_probes
        # silently reverted to the dataclass default (5) here,
        # so a port registered with n_probes=3 ran with a
        # five-rung bookkeeping entry (the reference-plane
        # crossing guard in _forward_from_materials walks it).
        run_entries.append(_MSLPortEntry(
            name=pe.name, position=pe.position,
            width=pe.width, height=pe.height,
            direction=pe.direction, impedance=pe.impedance,
            waveform=wf, excite=new_excite,
            n_probe_offset=pe.n_probe_offset,
            n_probe_spacing=pe.n_probe_spacing,
            n_probes=pe.n_probes,
            mode=pe.mode,
            eps_r_sub=pe.eps_r_sub,
            terminates=pe.terminates,
        ))
    variant = declaration_variant(
        observers, _msl_ports=run_entries,
        _internal_probe_indices=set(observers._internal_probe_indices),
        _dft_plane_regions=dict(getattr(self, "_dft_plane_regions", {})))

    # Register DFT plane probes for V (Ez) and I (Hy).
    ez_probe_names: list[list[str]] = [[] for _ in range(n_ports)]
    h_probe_names = []
    for p_idx, (mp, pxs) in enumerate(zip(msl_ports, probe_xs)):
        # Plane normal = this port's PROPAGATION axis; the two
        # H components are the closed-Ampere-loop pair
        # (a_hat x b_hat = p_hat), not a fixed (hy, hz).
        _meta_p = port_idx_meta[p_idx]
        _plane_axis = _meta_p["prop_axis"]
        for q_idx, x_coord in enumerate(pxs):
            nm = f"_msl_run{driven}_p{p_idx}_ez{q_idx}"
            variant.add_dft_plane_probe(
                axis=_plane_axis, coordinate=float(x_coord),
                component="ez", freqs=jnp.asarray(freqs_arr),
                name=nm,
            )
            variant._dft_plane_regions[nm] = (
                _meta_p["j_centre"],
                _meta_p["j_centre"] + 1,
                _meta_p["k_lo"],
                trace_k_per_port[p_idx][0],
            )
            ez_probe_names[p_idx].append(nm)
        _k_tr_lo, _k_tr_hi = trace_k_per_port[p_idx]
        _h_crop_region = (
            _meta_p["j_lo"] - 1,
            _meta_p["j_hi"] + 1,
            _k_tr_lo - 1,
            _k_tr_hi + 1,
        )
        h_probe_names.append(_register_msl_h_planes(
            variant, f"_msl_run{driven}_p{p_idx}", h_stencils[p_idx],
            (_meta_p["h_a"], _meta_p["h_b"]), jnp.asarray(freqs_arr),
            _h_crop_region,
        ))

    variant._time_recorded_dft_planes = (
        {name for ladder in ez_probe_names for name in ladder}
        | {name for pairs in h_probe_names for pair in pairs for name in pair}
    )
    return variant, ez_probe_names, h_probe_names


def _score_msl_record(
    _ts_result, _witness_base, _witness_total, driven, dz_arr, ez_probe_names, freqs_arr,
    grid, h_probe_names, h_stencils, is_tracer, msl_ports, np, port_idx_meta,
    self, settling_db_runs, settling_details, trace_k_per_port, variant,
):
    # Score recorded S channels and witness probes.
    _ts = getattr(_ts_result, "time_series", None)
    if _ts is not None and not is_tracer(_ts):
        _ts_np = np.asarray(
            _ts[:, _witness_base:_witness_base + _witness_total],
        )
        if _ts_np.shape[0] >= 10 and _ts_np.shape[1] == _witness_total:
            from rfx.measurement.record_end import injection_end
            from rfx.sources.waveguide_port import settling_db_from_named_records
            _source_end = injection_end(
                _ts_np.shape[0], getattr(_ts_result, "dt", None) or grid.dt,
                records=variant._probes[_witness_base:_witness_base + _witness_total],
                grid=grid, sim=variant)
            from rfx.sparams._tail_witness import msl_time_channels
            _plane_records = getattr(_ts_result, "dft_time_records", None)
            _channels = msl_time_channels(
                _plane_records, ez_probe_names, h_probe_names, h_stencils,
                port_idx_meta, trace_k_per_port, dz_arr,
                [port.direction for port in msl_ports], regions=variant._dft_plane_regions,
            ) if _plane_records else []
            _channels.extend((f"probe{i}", _ts_np[:, i])
                             for i in range(_ts_np.shape[1]))
            settling_db_runs[driven], settling_details[driven] = settling_db_from_named_records(
                _channels, source_end_index=_source_end,
                dt=getattr(_ts_result, "dt", None) or grid.dt,
                freqs=freqs_arr, freq_max=self._freq_max, return_detail=True)
    return settling_db_runs, settling_details


def _integrate_msl_port(
    _collocated_msl_h, _complex_dtype, driven, dz_arr, ez_probe_names, h_probe_names, h_stencils,
    i_first_per_port, jnp, meta, msl_loop_current, msl_modal_voltage, msl_ports, p_idx,
    planes, raw_3probe_dump_path, raw_i1_left, raw_i1_same_index, trace_k_per_port, v_per_port, variant,
):
    vs = []
    for nm in ez_probe_names[p_idx]:
        ez_plane = jnp.asarray(planes[nm].accumulator)
        _v_region = variant._dft_plane_regions.get(nm)
        # ez_plane shape: (n_freqs, ny, nz)
        # Top of the V span = the RASTERIZED trace's bottom
        # node, not round(h_sub/dx) (= meta["k_hi"]): Box
        # rasterization is half-open over node coordinates,
        # so for frac(h_sub/dx) in (0, 0.5) the trace lands
        # at ceil = k_hi + 1 and the k_hi anchor is one
        # substrate edge SHORT — V and the Ampere-loop
        # current would reference different conductor
        # planes (PR #516 review, finding F2; measured on
        # the dx=80um gate fixture: trace node 4, k_hi 3).
        # trace_k_per_port is the same PEC search the
        # current integration uses, so V and I share one
        # conductor plane by construction.
        # ez_plane is [freq, width, normal] for an x- OR a
        # y-normal plane alike (issue #661), so j_centre /
        # k_lo / k_hi address it unchanged and the dz_arr
        # here is always the substrate-normal profile.
        if (
            _v_region is not None
            and ez_plane.shape[1:] == (
                _v_region[1] - _v_region[0],
                _v_region[3] - _v_region[2],
            )
        ):
            vs.append(msl_modal_voltage(
                ez_plane,
                j_centre=0,
                k_lo=0,
                k_hi=_v_region[3] - _v_region[2],
                dz_arr=dz_arr[_v_region[2]:_v_region[3]],
                dtype=_complex_dtype,
            ))
        else:
            # Replay fixtures and third-party test doubles
            # may still return legacy full-plane arrays.
            vs.append(msl_modal_voltage(
                ez_plane, j_centre=meta["j_centre"],
                k_lo=meta["k_lo"],
                k_hi=trace_k_per_port[p_idx][0],
                dz_arr=dz_arr, dtype=_complex_dtype,
            ))
    v_per_port.append(vs)
    # G-AD-WIRE: keep on JAX tape when eps_override is
    # set. np.asarray() would concretise a JAX tracer and
    # break jax.grad. jnp.asarray() is a no-op on a real
    # jnp.ndarray and still works for numpy arrays.
    hy_plane, hz_plane = _collocated_msl_h(
        planes, h_probe_names[p_idx], h_stencils[p_idx]["weights"],
    )
    _h_region = variant._dft_plane_regions.get(
        h_probe_names[p_idx][0][1]
    )
    _h_is_cropped = (
        _h_region is not None
        and hy_plane.shape[1:] == (
            _h_region[1] - _h_region[0],
            _h_region[3] - _h_region[2],
        )
        and hz_plane.shape == hy_plane.shape
    )
    # Closed Ampere-loop current ∮H·dl around the trace
    # conductor (issue #80 stage S1). The pre-S1 inline
    # integral summed the bottom Hy leg only and undercounted
    # I by ~1.5x, inflating the de-embedded Z0 to ~74 vs the
    # ~48 ohm analytic value. msl_loop_current closes the
    # contour (bottom/top Hy legs + left/right Hz legs) and
    # carries the +x current-sign convention.
    k_tr_lo, k_tr_hi = trace_k_per_port[p_idx]
    if _h_is_cropped:
        assert _h_region is not None
        _w_lo, _w_hi, _n_lo, _n_hi = _h_region
        _j_lo = meta["j_lo"] - _w_lo
        _j_hi = meta["j_hi"] - _w_lo
        _k_lo = k_tr_lo - _n_lo
        _k_hi = k_tr_hi - _n_lo
    else:
        _w_lo, _w_hi = 0, hy_plane.shape[1]
        _n_lo, _n_hi = 0, hy_plane.shape[2]
        _j_lo, _j_hi = meta["j_lo"], meta["j_hi"]
        _k_lo, _k_hi = k_tr_lo, k_tr_hi
    # Issue #661: msl_loop_current wants the planes in the
    # right-handed transverse frame [freq, a, b] with
    # a_hat x b_hat = p_hat. The recorded planes are
    # [freq, width, normal]. For "+x"/"-x" the pair is
    # (y, z) = (width, normal) — already in frame, no
    # transpose, byte-identical to the pre-#661 call. For
    # "+y"/"-y" the pair is (z, x) = (normal, width), so
    # the two axes swap and the spans swap with them.
    # Do NOT "simplify" this into a plain x<->y rename:
    # that is a reflection, it flips the sign of I, and it
    # inverts S silently (see _MSL_CYCLIC_PAIR).
    if meta["a_is_width"]:
        _ha, _hb = hy_plane, hz_plane
        _a_lo, _a_hi = _j_lo, _j_hi
        _b_lo, _b_hi = _k_lo, _k_hi
        _a_arr = meta["a_arr"][_w_lo:_w_hi]
        _b_arr = meta["b_arr"][_n_lo:_n_hi]
    else:
        _ha = jnp.transpose(hy_plane, (0, 2, 1))
        _hb = jnp.transpose(hz_plane, (0, 2, 1))
        _a_lo, _a_hi = _k_lo, _k_hi
        _b_lo, _b_hi = _j_lo, _j_hi
        _a_arr = meta["a_arr"][_n_lo:_n_hi]
        _b_arr = meta["b_arr"][_w_lo:_w_hi]
    i_f = msl_loop_current(
        _ha, _hb,
        j_lo=_a_lo, j_hi=_a_hi,
        k_trace_lo=_b_lo, k_trace_hi=_b_hi,
        dy_arr=_a_arr, dz_arr=_b_arr,
        direction=msl_ports[p_idx].direction,
    )
    if raw_3probe_dump_path is not None:
        assert raw_i1_left is not None and raw_i1_same_index is not None
        side_currents = []
        for side in (0, 1):
            side_a = jnp.asarray(planes[h_probe_names[p_idx][0][side]].accumulator)
            side_b = jnp.asarray(planes[h_probe_names[p_idx][1][side]].accumulator)
            if not meta["a_is_width"]:
                side_a = jnp.transpose(side_a, (0, 2, 1))
                side_b = jnp.transpose(side_b, (0, 2, 1))
            side_currents.append(msl_loop_current(
                side_a, side_b, j_lo=_a_lo, j_hi=_a_hi,
                k_trace_lo=_b_lo, k_trace_hi=_b_hi,
                dy_arr=_a_arr, dz_arr=_b_arr,
                direction=msl_ports[p_idx].direction,
            ))
        raw_i1_left = raw_i1_left.at[driven, p_idx, :].set(side_currents[0])
        raw_i1_same_index = raw_i1_same_index.at[driven, p_idx, :].set(side_currents[1])
    i_first_per_port.append(i_f)
    return i_f, raw_i1_left, raw_i1_same_index, v_per_port, i_first_per_port


def _fit_msl_port(
    _complex_dtype, beta0_per_port, driven, extract_msl_nprobe, i_f, jnp, msl_axis_roles,
    msl_ports, n_probes_per_port, np, p_idx, probe_xs, raw_beta_railed, raw_i1,
    raw_q, raw_v, raw_z0, v_per_port, z0_hj_per_port,
):
    # N-probe least-squares wave decomposition (issue #80
    # Fix C). Stack the N voltage probes into (n_freqs, N),
    # anchor the beta scan on the analytic HJ guess, and
    # solve the over-determined (alpha, gamma) system by
    # SVD lstsq — this removes the 3-probe q->1 singularity.
    n_probes_p = n_probes_per_port[p_idx]
    v_stack = jnp.stack(v_per_port[p_idx], axis=-1)  # (n_freqs, N)
    raw_v = raw_v.at[driven, p_idx, :n_probes_p, :].set(jnp.asarray(v_stack.T, dtype=_complex_dtype))
    raw_i1 = raw_i1.at[driven, p_idx, :].set(jnp.asarray(i_f, dtype=_complex_dtype))
    res_p = extract_msl_nprobe(
        jnp.asarray(v_stack),
        jnp.asarray(np.asarray(probe_xs[p_idx], dtype=float)),
        jnp.asarray(i_f),
        jnp.asarray(beta0_per_port[p_idx]),
        z0_hj=z0_hj_per_port[p_idx],
    )
    # Normalize the REPORTED characteristic-impedance sign per port
    # (issue #140). msl_loop_current negates the loop current ONLY for
    # "+x" ports (rfx/sources/msl_port.py:947-948), so a "-x" port's
    # fitted z0 = (alpha - gamma)/I inherits a negative sign while a
    # physical Z0 is positive-real. Mirror that exact binary so BOTH
    # ports report a positive Z0. This touches ONLY the reported/dumped
    # Z0 (raw_z0, Z0_per_run) and the |Z0| honesty-guard; it never
    # enters S11/S21 (which use the static analytic Hammerstad-Jensen
    # z0_hj). It removes the spurious ~228% guard deviation on -x ports
    # while leaving the genuine ~20-27% 3-cell Yee-staircase Z0 warning on
    # both ports. NB: the raw current dump (raw_i1) intentionally keeps
    # its un-normalized sign; only the DERIVED Z0 is sign-normalized.
    dir_sign = float(
        msl_axis_roles(msl_ports[p_idx].direction)[3]
    )
    z0_fit = jnp.asarray(res_p["z0"], dtype=_complex_dtype) * dir_sign
    raw_z0 = raw_z0.at[driven, p_idx, :].set(z0_fit)
    raw_q = raw_q.at[driven, p_idx, :].set(jnp.asarray(res_p["q"], dtype=_complex_dtype))
    raw_beta_railed = raw_beta_railed.at[driven, p_idx, :].set(
        jnp.asarray(res_p["beta_railed"], dtype=bool)
    )
    return raw_v, raw_i1, raw_z0, raw_q, raw_beta_railed, res_p, z0_fit


def _check_msl_extraction(
    RfxRefusal, S, _diagnostics, diagnostic_refusal, entries, freqs_arr,
    jax, n_ports, np, probe_clearance, raw_z0, reliable, strict_extractor,
    z0_hj_per_port,
):
    # --- Honesty guard (issue #80 Fix A, retargeted in stage S1) ---
    # S11/S21 come from the OpenEMS-style V·I wave amplitudes, now
    # combined by the multi-drive solve (issue #507). |S11| > 1 on a
    # passive structure remains the primary red flag, but read it
    # correctly: the single-plane split itself was bounded by
    # construction, whereas the solve is not — it can exceed 1 when
    # the recorded wave system is inconsistent (a mismeasured
    # current's sign or scale, a degenerate drive matrix, or an
    # under-settled record). That is what raises under
    # strict_extractor=True, and cond(A) plus settling_db are the
    # two handles for telling those apart. The reported Z0 and beta
    # still ride on the retained N-probe fit, which can be noisy
    # per-frequency on coarse meshes, so a Z0 deviation from
    # analytic Hammerstad-Jensen is reported as a SEPARATE, softer
    # caveat. The fitted numbers do not enter S, but the V/I
    # records and analytic reference still need their own checks.
    import warnings as _w

    # Issue #726: ONE text for what probe-clearance corruption does,
    # shared with the preflight layout warning that used to contradict
    # this guard about the same condition. Imported, not copied.
    from rfx.preflight.msl import (
        MSL_PROBE_CLEARANCE_EFFECT,
        MSL_PROBE_CLEARANCE_GUIDANCE,
    )

    _S11_MAX = 1.0 + 0.05
    _Z0_TOL = 0.10
    for driven in range(n_ports):
        pe = entries[driven]
        z0_hj = z0_hj_per_port[driven]
        s11_abs = np.abs(np.asarray(jax.lax.stop_gradient(S[driven, driven, :])))
        k_s = int(np.argmax(s11_abs))
        s11_max = float(s11_abs[k_s])
        z0_dev = np.abs(np.asarray(jax.lax.stop_gradient(raw_z0[driven, driven, :])) - z0_hj) / z0_hj
        k_z = int(np.argmax(z0_dev))
        z0_dev_max = float(z0_dev[k_z])
        # Primary — V·I-split S11 boundedness (extraction soundness).
        if s11_max > _S11_MAX:
            # Cross-reference the standing-wave-null reliability mask (computed above): if
            # the peak-|S11| bin is flagged, signal strength is an
            # additional concern, not a diagnosis of the root cause.
            at_null = reliable is not None and not bool(np.asarray(reliable)[driven, k_s])
            cause = (
                "the relative low-signal mask also flags this bin; "
                "check signal uncertainty and the drive system"
                if at_null else
                "check current sign/scale, mode validity, settling and drive conditioning"
            )
            msg = (
                f"compute_msl_s_matrix: V·I-split |S11| = "
                f"{s11_max:.3f} > 1 for MSL port {pe.name!r} at "
                f"f = {freqs_arr[k_s] / 1e9:.4f} GHz — non-physical "
                f"for a passive structure. {cause}; the extracted "
                "S11/S21 at this bin are UNRELIABLE."
            )
            if strict_extractor:
                raise diagnostic_refusal(RfxRefusal(msg), _diagnostics)
            _w.warn(msg, AccuracyWarning, stacklevel=3)
        # Secondary — reported-Z0 sanity (retained N-probe fit).
        if z0_dev_max > _Z0_TOL:
            # Issue #726 review P1b: the clearance story is only ONE of
            # this guard's causes, and on a satisfied port it is the
            # WRONG advice — the common trigger is coarse-mesh staircase
            # bias (#752), where "lengthen the feed" does nothing. Read
            # the same per-port record the result carries, by the same
            # index the entries tuple uses.
            _cl = (probe_clearance[driven]
                   if probe_clearance is not None
                   and driven < len(probe_clearance) else None)
            _cl_status = getattr(_cl, "status", None)
            if _cl_status == "insufficient":
                _clearance_clause = (
                    " This port's probe clearance is INSUFFICIENT, so "
                    "that is a live cause here: "
                    + MSL_PROBE_CLEARANCE_EFFECT + " "
                    + MSL_PROBE_CLEARANCE_GUIDANCE)
            elif _cl_status == "satisfied":
                _clearance_clause = (
                    " This port's probe clearance is satisfied, so "
                    "standing-wave content at the probes is not the "
                    "cause; refine the mesh or reconcile the rasterized "
                    "board with the declared one before quoting Z0.")
            else:
                _clearance_clause = (
                    " This port's probe clearance could not be "
                    "evaluated"
                    + (f" ({_cl.note})" if getattr(_cl, "note", None)
                       else "")
                    + ", so probe-clearance corruption cannot be ruled "
                    "out as a cause; read probe_clearance and "
                    "beta_railed before interpreting Z0.")
            _w.warn(
                f"compute_msl_s_matrix: reported Z0 for MSL port "
                f"{pe.name!r} = "
                f"{float(np.asarray(jax.lax.stop_gradient(raw_z0[driven, driven, k_z])).real):.2f} ohm deviates "
                f"{z0_dev_max * 100:.1f}% from analytic Hammerstad-"
                f"Jensen {z0_hj:.2f} ohm at "
                f"f = {freqs_arr[k_z] / 1e9:.4f} GHz. Z0 rides on the "
                "retained N-probe fit (S1 transitional); this can "
                "reflect Yee-staircase bias, or that the rasterized "
                "board (h_sub/W snapped to the lattice; see "
                "sim.fidelity_report()) differs from the "
                "declared one — not necessarily an extraction "
                "fault. Fitted Z0/beta are not used in "
                "S11/S21, which use measured V/I and the analytic "
                "Z0 anchor; this does not certify those inputs."
                + _clearance_clause +
                " Check settling_db, probe_clearance, beta_railed and "
                "observation-plane sensitivity before interpreting S.", AccuracyWarning,
                stacklevel=3,
            )
    return


def _dump_msl_record(
    S, Z0_per_run, _reported, beta_first, declared_refs, entries, freqs_arr,
    grid, h_stencils, msl_assembly, msl_planes, n_ports, n_probes_per_port, n_steps,
    np, num_periods, raw_3probe_dump_path, raw_i1, raw_i1_left, raw_i1_same_index, raw_q,
    raw_v, raw_z0, s_line_refs, self,
):
    if raw_3probe_dump_path is not None:
        import json
        from pathlib import Path

        path = Path(raw_3probe_dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "rfx.msl_nprobe_dump",
            "schema_version": 5,
            "s_wave_convention": "power",
            "wave_definition": "a=(V+R*I)/(2*sqrt(R)); b=(V-R*I)/(2*sqrt(R))",
            "solver_wave_common_scale": "sqrt(reference_impedances[0]); cancels from S and cond(A)",
            "current_spatial_alignment": "linear_bracketing_H_to_E_node",
            "current_plane_stencils": h_stencils,
            "s_reference_impedances_ohm": [
                float(z) for z in _reported(s_line_refs, (n_ports,))],
            "declared_reference_impedances_ohm": [
                float(z) for z in declared_refs],
            "production_smatrix_schema": "S[receiver_port, driven_port, frequency_index]",
            "production_smatrix_stage": (
                "PRE-passivity-projection raw extraction referred to "
                "the declared feed planes, at the line references "
                "s_reference_impedances_ohm; MSLSMatrixResult.S (and "
                "S_raw when enforce_passivity=True clipped something) "
                "is this matrix referred to "
                "declared_reference_impedances_ohm by "
                "renormalize_power_waves (#2014), and is this matrix "
                "itself where the two lists are equal. raw_v/raw_i1 stay at the probe "
                "planes; undo the shift with "
                "feed_plane_shift before comparing to a probe-0 replay"
            ),
            # v3 (issue #523): production_smatrix is no longer always
            # the N-probe-fit-derived S. Record WHICH assembly made
            # it, so a replayed dump cannot be misattributed.
            #
            # NB production_smatrix is written PRE-projection, so a
            # fallback dump does still carry the >1 column power
            # (an enforce_passivity=True MSLSMatrixResult.S is
            # post-projection and does not).
            # The marker is not a substitute for that symptom — it is
            # more specific: >1 column power has several causes, only
            # one of which is the fallback.
            "production_smatrix_assembly": (
                "unknown" if msl_assembly is None else msl_assembly
            ),
            "raw_v_shape": "(n_driven, n_ports, n_probes_max, n_freqs)",
            "raw_i1_shape": "(n_driven, n_ports, n_freqs)",
            "n_probes_per_port": [int(n) for n in n_probes_per_port],
            "phase_convention": "DFT accumulator convention from add_dft_plane_probe",
            "current_convention": "native_msl_loop_current",
            "deembedding": (
                "N equally spaced voltage probes plus current at "
                "probe 0. The reported Z0/beta come from the N-probe "
                "least-squares wave-decomposition extractor (issue #80 "
                "Fix C), which fits V_n = alpha*exp(-j beta x_n) + "
                "gamma*exp(+j beta x_n) by SVD lstsq. The production "
                "S-matrix is solved from the probe-0 wave amplitudes "
                "over all drives, S = B @ inv(A) (issue #507), then "
                "referred to the declared feed planes: "
                "S_feed[i, j] = D_i S[i, j] D_j with "
                "D_p = exp(-j beta_p L_p), L_p the realized "
                "feed-to-probe-0 distance and beta_p the closed-form "
                "Hammerstad-Jensen + Kirschning-Jansen beta (see "
                "feed_plane_shift; the fitted beta does not enter S), "
                "with the modal voltage "
                "spanning ground to the rasterized trace node (#511) "
                "-- see production_smatrix_assembly for which rule "
                "actually produced this dump's S"
            ),
            "feed_plane_shift": {
                "formula": (
                    "S_feed[i, j, f] = D_i(f) S_probe0[i, j, f] D_j(f), "
                    "D_p(f) = exp(-1j * shift_beta[p, f] * "
                    "reference_plane_offsets_m[p])"),
                "reference_plane_offsets_m": [
                    float(x) for x in msl_planes.offsets_m],
                "shift_beta_rad_per_m": np.asarray(
                    msl_planes.shift_beta, dtype=np.float64).tolist(),
                "shift_beta_source": (
                    "closed form: (2 pi f / c) sqrt(eps_eff(f)), "
                    "Hammerstad-Jensen eps_eff(0) with Kirschning-"
                    "Jansen dispersion"),
            },
            "grid": {
                "dx_m": float(grid.dx),
                "dt_s": float(grid.dt),
                "nx": int(grid.nx),
                "ny": int(grid.ny),
                "nz": int(grid.nz),
            },
            "simulation": {
                "freq_max_hz": float(self._freq_max),
                "num_periods": float(num_periods),
                "n_steps": None if n_steps is None else int(n_steps),
            },
            "port_definitions": [
                {
                    "name": str(pe.name),
                    "position_m": [float(x) for x in pe.position],
                    "width_m": float(pe.width),
                    "height_m": float(pe.height),
                    "direction": pe.direction,
                    "impedance_ohm": float(pe.impedance),
                    "n_probe_offset": int(pe.n_probe_offset),
                    "n_probe_spacing": int(pe.n_probe_spacing),
                    "n_probes": int(pe.n_probes),
                    "mode": pe.mode,
                }
                for pe in entries
            ],
        }
        np.savez(
            path,
            metadata_json=np.asarray(json.dumps(metadata)),
            freqs_hz=np.asarray(freqs_arr, dtype=np.float64),
            raw_v=raw_v,
            raw_i1=raw_i1,
            raw_i1_left=raw_i1_left,
            raw_i1_same_index=raw_i1_same_index,
            raw_z0=raw_z0,
            raw_q=raw_q,
            production_smatrix=S,
            production_z0=Z0_per_run,
            production_beta=beta_first,
            port_names=np.asarray(tuple(pe.name for pe in entries), dtype=object),
            driven_port_indices=np.arange(n_ports, dtype=np.int64),
        )
    return


def _shift_msl_reference_planes(
    S, _complex_dtype, _reported, entries, eps_r_ref_per_port, freqs_arr,
    grid, jnp, line_eps_per_port, msl_ports, probe_xs, renormalize_power_waves, z0_hj_per_port,
):
    # ---- Reference plane: probe 0 -> declared feed plane ----------
    # S above is referred to each port's probe-0 plane. One resolver
    # (rfx.measurement.msl_planes, shared with the mixed calculator)
    # gives the realized feed-to-probe-0 distance L_p and the closed-
    # form shift beta per bin (Hammerstad-Jensen eps_eff(0) with
    # Kirschning-Jansen dispersion, from width, height and the eps_r
    # resolved above; never the fitted beta), and
    # S_feed[i, j] = D_i S[i, j] D_j with D = exp(-j beta L). beta is
    # real, so |S|, singular values and cond(A) are unchanged. Applied
    # before the dump and before the opt-in passivity projection, so
    # S_raw carries it too.
    from rfx.measurement.msl_planes import (
        resolve_msl_planes as _resolve_msl_planes,
        apply_feed_plane_shift as _apply_feed_plane_shift,
    )
    msl_planes = _resolve_msl_planes(
        grid, msl_ports,
        [int(pe.n_probe_offset) for pe in entries],
        [xs[0] for xs in probe_xs],
        [float(pe.width) for pe in entries],
        [float(pe.height) for pe in entries],
        eps_r_ref_per_port,
        freqs_arr,
    )
    # The waves above are at the declared line's Z0 (z0_hj). The phase-
    # only shift is exact in the reference of the line it moves along,
    # so under eps_override (#2199) S is first referred to the solved
    # line's Z0, shifted with that line's beta, and referred to
    # impedance= from there below.
    s_line_refs = z0_hj_per_port
    if line_eps_per_port is None:
        S = _apply_feed_plane_shift(
            S, msl_planes.offsets_m, msl_planes.shift_beta,
        ).astype(_complex_dtype)
    else:
        from rfx.measurement.msl_planes import (
            msl_line_beta_z0 as _line_beta_z0,
        )
        _bz = [
            _line_beta_z0(pe.width, pe.height, e, freqs_arr)
            for pe, e in zip(entries, line_eps_per_port)
        ]
        line_beta = jnp.stack([b for b, _ in _bz])
        s_line_refs = jnp.stack([z for _, z in _bz])
        S = renormalize_power_waves(S, z0_hj_per_port, s_line_refs)
        S = _apply_feed_plane_shift(
            S, msl_planes.offsets_m, line_beta,
        ).astype(_complex_dtype)
        msl_planes = type(msl_planes)(
            offsets_m=msl_planes.offsets_m,
            shift_beta=_reported(line_beta, line_beta.shape))
    return S, msl_planes, s_line_refs

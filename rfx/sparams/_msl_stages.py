from __future__ import annotations

from rfx.measurement.calculator_drive import calculator_drive

import numpy as np

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
    """The H sample at index ``i`` lies half a propagation cell beyond E node ``i`` along the positive coordinate axis."""
    msl_ports = [msl_port_from_entry(pe) for pe in entries]

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
    h_stencils = [msl_h_plane_stencil(grid, mp, xs[0])
                  for mp, xs in zip(msl_ports, probe_xs)]
    from rfx.preflight.msl import msl_probe_clearance_for_port
    probe_clearance = tuple(msl_probe_clearance_for_port(
        self, pe, grid, probe_coordinates=xs,
    ) for pe, xs in zip(entries, probe_xs))
    return msl_ports, n_probes_per_port, probe_xs, h_stencils, probe_clearance


def _prepare_msl_geometry(
    _msl_cell_profile, entries, eps_override, freqs_arr, grid, hammerstad_jensen_z0_eps_eff, is_nonuniform,
    jax, msl_ampere_pair, msl_cell, msl_cross_section_span, msl_ports, n_ports,
    self,
):

    """In port metadata, ``j_*`` indexes trace width and ``k_*`` indexes the substrate-normal z axis."""
    _axis_n = {"x": grid.nx, "y": grid.ny, "z": grid.nz}

    def _prof(ax):
        return _msl_cell_profile(grid, ax, _axis_n[ax])

    dz_arr = _prof("z")

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
            a_axis=a_ax, b_axis=b_ax,
            a_is_width=(a_ax == span["width_axis"]),
            a_arr=_prof(a_ax), b_arr=_prof(b_ax),
            h_a=f"h{a_ax}", h_b=f"h{b_ax}",
        ))

    from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
    _C0_MSL = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
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
    declared_refs = np.asarray(
        [float(pe.impedance) for pe in entries], dtype=np.float64)

    trace_k_per_port: list[tuple[int, int]] = []
    for p_idx in range(n_ports):
        meta = port_idx_meta[p_idx]
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
    entries, msl_physical_point, n_ports, observers, port_idx_meta, probe_xs,
):
    _witness_base = len(observers._probes)
    _witness_counts: list[int] = []
    for pe_w, pxs_w, meta_w in zip(entries, probe_xs, port_idx_meta):
        _w_centre_m = float(pe_w.position[meta_w["width_idx"]])
        _n_lo_m = float(pe_w.position[meta_w["normal_idx"]])
        for _x_w in pxs_w:
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
    run_entries = []
    for idx, pe in enumerate(entries):
        new_excite = calculator_drive(idx == driven, excite=pe.excite)
        if new_excite:
            wf = pe.waveform if pe.waveform is not None else \
                GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
        else:
            wf = None
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

    ez_probe_names: list[list[str]] = [[] for _ in range(n_ports)]
    h_probe_names = []
    for p_idx, (mp, pxs) in enumerate(zip(msl_ports, probe_xs)):
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
    grid, h_probe_names, h_stencils, is_tracer, msl_ports, port_idx_meta,
    self, settling_db_runs, settling_details, trace_k_per_port, variant,
):
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
            from rfx.sparams._settling_groups import msl_column_groups
            _groups, _labels = msl_column_groups(_channels)
            _groups.update((f"probe{i}", "ez") for i in range(_ts_np.shape[1]))
            settling_db_runs[driven], settling_details[driven] = settling_db_from_named_records(
                _channels, source_end_index=_source_end,
                dt=getattr(_ts_result, "dt", None) or grid.dt,
                freqs=freqs_arr, freq_max=self._freq_max, return_detail=True,
                scale_groups=_groups, _channel_labels=_labels)
    return settling_db_runs, settling_details


def _integrate_msl_port(
    _collocated_msl_h, _complex_dtype, driven, dz_arr, ez_probe_names, h_probe_names, h_stencils,
    i_first_per_port, jnp, meta, msl_loop_current, msl_modal_voltage, msl_ports, p_idx,
    planes, raw_3probe_dump_path, raw_i1_left, raw_i1_same_index, trace_k_per_port, v_per_port, variant,
):
    """The transverse Ampere frame is ordered ``(y, z)`` for x-directed ports and ``(z, x)`` for y-directed ports."""
    vs = []
    for nm in ez_probe_names[p_idx]:
        ez_plane = jnp.asarray(planes[nm].accumulator)
        _v_region = variant._dft_plane_regions.get(nm)
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
            vs.append(msl_modal_voltage(
                ez_plane, j_centre=meta["j_centre"],
                k_lo=meta["k_lo"],
                k_hi=trace_k_per_port[p_idx][0],
                dz_arr=dz_arr, dtype=_complex_dtype,
            ))
    v_per_port.append(vs)
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
    msl_ports, n_probes_per_port, p_idx, probe_xs, raw_beta_railed, raw_i1,
    raw_q, raw_v, raw_z0, v_per_port, z0_hj_per_port,
):
    n_probes_p = n_probes_per_port[p_idx]
    v_stack = jnp.stack(v_per_port[p_idx], axis=-1)
    raw_v = raw_v.at[driven, p_idx, :n_probes_p, :].set(jnp.asarray(v_stack.T, dtype=_complex_dtype))
    raw_i1 = raw_i1.at[driven, p_idx, :].set(jnp.asarray(i_f, dtype=_complex_dtype))
    res_p = extract_msl_nprobe(
        jnp.asarray(v_stack),
        jnp.asarray(np.asarray(probe_xs[p_idx], dtype=float)),
        jnp.asarray(i_f),
        jnp.asarray(beta0_per_port[p_idx]),
        z0_hj=z0_hj_per_port[p_idx],
    )
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
    jax, n_ports, probe_clearance, raw_z0, reliable, strict_extractor,
    z0_hj_per_port,
):
    import warnings as _w

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
        if s11_max > _S11_MAX:
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
        if z0_dev_max > _Z0_TOL:
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
    num_periods, raw_3probe_dump_path, raw_i1, raw_i1_left, raw_i1_same_index, raw_q,
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
    """The input S-matrix is referenced to probe 0 at each port."""
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

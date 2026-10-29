"""Production-scan lumped/wire S-matrix driver (item-5 Stage 1).

This module rebuilds the full N-port lumped/wire S-parameter extraction on
the **production JIT scan** (``Simulation._forward_from_materials`` →
``rfx.simulation.run``) instead of the hand-maintained eager Python FDTD
loops in ``rfx.probes.probes`` (``extract_s_matrix`` /
``extract_s_matrix_wire``).

It drives one sparam-eligible port at a time via the
``_sparam_drive_idx`` / ``_return_raw_port_sparams`` hook on
``_forward_from_materials``, collects the all-port ``(v_dft, i_dft)``
accumulators, and feeds them to the *shared* pure wave decomposers
(``decompose_lumped_s_matrix`` / ``decompose_wire_s_matrix`` in
``rfx.probes.probes``) — the same decomposition the eager path now uses, so
the driver and the eager extractor agree by construction.

Stage 1 is a PURE ADD: nothing here reroutes ``forward()`` / ``run()``; the
eager loops are untouched (their removal is Stage 3).
"""

from __future__ import annotations

from rfx._precision import resolve_field_dtype

from rfx._grid_metric import field_index
import numpy as np

import jax.numpy as jnp

from rfx.probes.probes import (
    PortVIReplayBundle,
    WirePortVIReplayBundle,
    decompose_lumped_s_matrix,
    decompose_wire_s_matrix,
)


def refuse_distributed_lumped_s_pmc(sim, *, entry="run"):
    """Refuse lumped S and wire loads/drives under the distributed PMC rule.

    ``entry`` is the call the user made: ``"run"`` (``devices=...``) or
    ``"forward"`` (``distributed=True``).
    """
    option = "distributed=True" if entry == "forward" else "devices=..."
    faces = sorted(sim._boundary_spec.pmc_faces())
    if faces and any(pe.impedance != 0 and pe.extent is not None for pe in sim._ports):
        raise NotImplementedError(
            f"Wire ports with magnetic (PMC) face(s) {', '.join(faces)} "
            f"are not supported with {option}; the multi-device main run "
            "zeroes tangential E on a declared PMC face, so nearby wire loads "
            "and drives would be wrong even with compute_s_params=False "
            f"(rfx #1221, B3b). Use one device (omit {option})."
        )
    if faces and any(pe.impedance != 0 for pe in sim._ports):
        raise NotImplementedError(
            f"Lumped-port S-parameters with magnetic (PMC) face(s) {', '.join(faces)} "
            f"are not supported with {option}; use one device (omit {option})."
        )


def compute_lumped_wire_s_matrix_via_scan(
    sim, freqs, *, n_steps=None, return_vi_dump=False,
    return_refplane_diagnostics=False, conformal_pec=None, devices=None,
    _main_wire_record=False,
    return_settling=False,
    _forward_port_s11=False,
    skip_preflight: bool = False,
    subpixel_smoothing=False,
):
    """Full lumped/wire N-port S-matrix via the production scan.

    Drives each sparam-eligible lumped/wire port in turn through the
    production JIT scan, then applies the shared pure decomposers.

    Parameters
    ----------
    sim : Simulation
        A built ``Simulation`` carrying lumped and/or wire ports (added via
        ``add_port``).  Ports are addressed in registration order over the
        impedance!=0 lumped/wire ports.
    freqs : array-like
        Frequencies (Hz) at which to extract the S-matrix.
    skip_preflight : bool
        Skip shared preflight checks when True. Default False.
    n_steps : int or None
        Number of FDTD steps per drive.  Defaults to
        ``grid.num_timesteps(num_periods=30)`` (matching the eager
        extractors' default).
    return_vi_dump : bool
        When True, return the SAME replay bundle type the eager extractor
        produces (:class:`~rfx.probes.probes.PortVIReplayBundle` for an
        all-lumped set, :class:`~rfx.probes.probes.WirePortVIReplayBundle`
        for an all-wire set), populated from the driver's per-drive V/I
        accumulators with the identical sign conventions, shapes
        ``(n_driven, n_ports, n_freqs)``, and field names/order as the eager
        bundle.  This lets the production-scan driver feed the existing replay
        path byte-for-byte (item-5 Stage 2, PURE ADD).  Not supported
        together with reference-plane ports (raises NotImplementedError —
        the dump schema has no plane-phasor fields).
    return_refplane_diagnostics : bool
        When True and any port opted into ``reference_plane_cells``
        (issue #313), return ``(S, freqs, diagnostics)`` where
        ``diagnostics`` carries the per-port measured Zc(f) and beta(f)
        (R5 inspection surface).  ``(S, freqs, None)`` when no port
        opted in.
    devices : list or None
        Uniform distributed scan devices. Excited lumped/wire ports without planes.
        Records the live E line and four midpoint H samples per receive port;
        host DFTs feed the same decomposer. return_vi_dump=True is refused
        because pre-injection drive reference voltages are not recorded.
    conformal_pec : bool or None
        As in ``run()``: ``None`` reads ``Boundary(conformal=True)``. The
        production scan has no conformal update, so a conformal request on a
        model with PEC to conform is refused before the first step; ``False``
        asks for staircase PEC (#1299).

    Returns
    -------
    (S, freqs) : (np.ndarray, np.ndarray)
        Default.  ``S`` has shape ``(n_ports, n_ports, n_freqs)`` where
        ``S[i, j]`` is the response at receive port *i* when driving port *j*.
    PortVIReplayBundle or WirePortVIReplayBundle
        When ``return_vi_dump=True`` — the same bundle the eager
        ``extract_s_matrix`` / ``extract_s_matrix_wire`` return.

    Notes
    -----
    Mixed lumped + wire port sets are not supported on the uniform mesh. On a
    graded mesh each drive is the graded record with only that port excited and
    its column is formed as forward() forms it; mixed sets are supported, and S
    lists wire ports before lumped ones, the graded record's (and forward()'s) order.
    """
    # A plain source fires in every port drive, so it is refused here (#1420).
    resolve_field_dtype(sim._precision)
    from rfx.runners._admission import refuse_plain_sources_s_matrix
    refuse_plain_sources_s_matrix(sim, main_record=_main_wire_record)
    if subpixel_smoothing and devices is not None:
        raise NotImplementedError(
            "subpixel_smoothing is not applied by the multi-device S drives; "
            "use one device (omit devices=) or subpixel_smoothing=False.")
    # Graded: each drive is the graded record with only that port excited, its
    # column formed as forward() forms it (port_s_from_record).
    graded = sim._uses_nonuniform_mesh
    if graded and (devices is not None or return_vi_dump or return_refplane_diagnostics
                   or _main_wire_record or n_steps is None
                   or any(getattr(pe, "reference_plane_cells", None) for pe in sim._ports)):
        raise NotImplementedError(
            "compute_lumped_wire_s_matrix_via_scan on a graded mesh drives each port on one "
            "device with an explicit n_steps; devices=, return_vi_dump, reference planes and "
            "refplane diagnostics are uniform-mesh only.")
    if devices is not None:
        if return_vi_dump and any(pe.impedance > 0 and pe.extent is not None
                                  for pe in sim._ports):
            raise NotImplementedError(
                "return_vi_dump=True is not supported with devices=... for wire "
                "ports; their pre-injection drive reference voltages are not recorded. "
                "Use one device (omit devices=...).")
        from rfx.runners.distributed_v2 import refuse_unsupported_distributed_features
        refuse_unsupported_distributed_features(sim, lane="distributed S-matrix scan")
        refuse_distributed_lumped_s_pmc(sim)
    freqs = np.asarray(freqs, dtype=np.float64)
    n_freqs = len(freqs)

    # The distributed runner owns slab staging: do not retain a second
    # whole-domain material assembly across its scans.
    if not graded:
        sim._require_uniform_mesh("compute_lumped_wire_s_matrix_via_scan")
        from rfx.runners._admission import admit
        admit(sim, "s_matrix_scan", run_args={"conformal_pec": conformal_pec})
    if not skip_preflight:
        from rfx._diagnostic_transport import report_diagnostics
        report_diagnostics(sim._auto_preflight(
            context="compute_lumped_wire_s_matrix_via_scan", check_ntff="advisory"))
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    if devices is None and not graded:
        _sheet_specs: list = []
        _pec_sheets: list = []
        _pec_wires: list = []
        from rfx.model.conductors import realized_conductors, assembled_materials
        conductors = realized_conductors(sim, grid)
        materials, debye_spec, lorentz_spec, pec_mask, _, _, _ = assembled_materials(
            conductors, sheet_specs=_sheet_specs, pec_sheets=_pec_sheets,
            pec_wires=_pec_wires)
        _pec_sheets = tuple(_pec_sheets)
        _pec_wires = tuple(_pec_wires)
        # #931 §1.7: the realized PEC edges of this model — volumes, sheets and
        # wires — read once here, under the RUN's #689 flags (preflight refuses
        # lumped/wire S-params under periodic axes (#206), so this is normally
        # the non-periodic convention — but the flags are read, not assumed).
        _pec_edge_masks = conductors.pec_edges
        _sheet_ctx = conductors.sheet_context(_pec_edge_masks)

    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=30)

    # Ordered sparam-eligible lumped/wire ports (impedance != 0), in
    # registration order — the same order the multi-drive index counts.
    eligible_ids = [i for i, pe in enumerate(sim._ports) if pe.impedance != 0.0]
    eligible = [sim._ports[i] for i in eligible_ids]
    if not eligible:
        raise ValueError(
            "compute_lumped_wire_s_matrix_via_scan: no sparam-eligible "
            "lumped/wire ports (all ports have impedance 0)."
        )

    is_wire = [pe.extent is not None for pe in eligible]
    # Graded drives form each column from one record, which mixes lumped and wire ports.
    if any(is_wire) and not all(is_wire) and not graded:
        raise NotImplementedError(
            "compute_lumped_wire_s_matrix_via_scan: mixed lumped + wire port "
            "sets are not supported in Stage 1 (the off-diagonal wave-"
            "decomposition conventions differ).  Use a homogeneous all-lumped "
            "or all-wire port set."
        )
    wire_mode = any(is_wire)
    if wire_mode and devices is not None:
        # Geometry only: release the assembled materials before any scan stages slabs.
        from rfx.model.conductors import realized_conductors
        conductors = realized_conductors(sim, grid)
        _pec_edge_masks = conductors.pec_edges
        del conductors

    n_ports = len(eligible)
    z0 = np.asarray([pe.impedance for pe in eligible], dtype=np.float64)

    # Per-port wire LIVE cell counts (only needed for the wire decomposer).
    # Issue #318: the wave-decomposition normalization is Z0c = Z0/n_live so
    # it matches the live-cell sigma fold's per-cell resistor law
    # R_cell = Z0/n_live (the #308 receive-channel selection relies on that
    # identity). The realized edge masks here are the assembled-geometry
    # state BEFORE
    # any port-cell clearing — exactly the "live" definition. With no dead
    # cells this is the historical all-cells count.
    if wire_mode and not graded:
        from rfx.sources.sources import WirePort, _wire_port_live_cells
        axis_map = {"ex": 0, "ey": 1, "ez": 2}
        port_cell_counts = np.zeros(n_ports, dtype=np.int64)
        for idx, pe in enumerate(eligible):
            if pe.extent is None:
                port_cell_counts[idx] = 1
                continue
            end = list(pe.position)
            end[axis_map[pe.component]] += pe.extent
            wp = WirePort(
                start=pe.position,
                end=tuple(end),
                component=pe.component,
                impedance=pe.impedance,
                excitation=pe.waveform,
            )
            port_cell_counts[idx] = _wire_port_live_cells(
                grid, wp, _pec_edge_masks)[2]

    if devices is not None and wire_mode:
        del _pec_edge_masks

    # FDTD-sign V/I phasors per (drive j, receive i).
    v_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    i_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    # Whole-port gap-voltage phasors (issue #764, wire mode): V_port =
    # sum over LIVE wire cells of -E_c*dx per (drive j, receive i).
    # The whole-port decomposer consumes both diagonal and off-diagonal V.
    vp_all = (np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
              if wire_mode else None)
    # PRE-injection drive-sample reference phasors (wire: issue #683 x
    # #764; lumped: the known-load decision run,
    # scripts/diagnostics/lumped_port_known_load_line.py):
    # bit-identical to the historical pre-flip v_all; only the drive
    # diagonal vref_all[j, j] is consumed (off-diagonal incident wave +
    # the legacy diagonal on pre-decision data).
    vref_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)

    # Opt-in reference-plane accumulators (issue #313): raw plane phasors
    # per (drive j, port p, plane slot).  Allocated lazily on the first
    # drive pass that returns plane data.
    plane_v = plane_im = plane_ip = None
    plane_enabled = np.zeros(n_ports, dtype=bool)
    plane_offsets = np.zeros(n_ports, dtype=np.int64)
    plane_outboard = np.zeros(n_ports, dtype=np.int64)

    settling_runs = []
    columns = []
    # A graded record lists wire ports before lumped ones; its S keeps that order.
    record_order = sorted(range(n_ports), key=lambda k: not is_wire[k])
    for j in range(n_ports):
        if graded:
            from copy import copy
            from dataclasses import replace
            from rfx.runners.nonuniform import run_nonuniform_path
            from rfx.sources.sources import GaussianPulse
            drive = copy(sim)
            drive._ports = [replace(pe, excite=(pe is eligible[j]), waveform=(
                pe.waveform if pe.waveform is not None or pe is not eligible[j]
                else GaussianPulse(f0=sim._freq_max / 2, bandwidth=0.8)))
                if pe.impedance != 0 else pe for pe in sim._ports]
            record = run_nonuniform_path(drive, n_steps=n_steps, compute_s_params=True,
                                         s_param_freqs=freqs, conformal_pec=conformal_pec,
                                         subpixel_smoothing=subpixel_smoothing, _drive_scan=True)
            columns.append((record_order.index(j), np.asarray(record.s_params)[:, record_order.index(j)]))
            raw = {"sparam_time_records": record.sparam_time_records or (), "dt": record.dt}
        elif devices is not None:
            raw = _distributed_lumped_accumulators(sim, grid, eligible, freqs,
                                                  n_steps, devices,
                                                  None if _main_wire_record else j)
        else:
            raw = sim._forward_from_materials(
                grid,
                materials,
                debye_spec,
                lorentz_spec,
                n_steps=n_steps,
                checkpoint=False,
                pec_mask=pec_mask,
                pec_sheets=_pec_sheets,
                pec_wires=_pec_wires,
                port_s11_freqs=freqs,
                _sparam_drive_idx=j,
                _return_raw_port_sparams=True,
                sheet_impedance=_sheet_ctx, conductors=conductors,
                conformal_pec=conformal_pec, subpixel_smoothing=subpixel_smoothing,
            )

        if return_settling:
            from rfx.measurement.record_end import injection_end
            from rfx.sparams._tail_witness import port_record_witness
            from copy import copy
            from dataclasses import replace
            driven_sim = copy(sim)
            driven_sim._ports = [replace(pe, excite=(pe is eligible[j]))
                                 if pe.impedance != 0 else pe for pe in sim._ports]
            receivers_by_id = {f"port:{i}:mode:0": pe for i, pe in zip(eligible_ids, eligible)}
            plane_counts = {}
            for rp, _ in raw.get("wire_refplane") or ():
                pe = eligible[rp.port_index]
                position = list(pe.position)
                if hasattr(grid, "position_to_index"):
                    i0 = grid.position_to_index(pe.position)[rp.line_axis]
                else:
                    from rfx.nonuniform import position_to_index
                    i0 = position_to_index(grid, pe.position)[rp.line_axis]
                widths = np.asarray(grid.cells(rp.line_axis), dtype=np.float64)
                n_out = int(rp.n_cells_outboard)
                span = (widths[i0:i0 + n_out] if rp.outboard_sign > 0
                        else widths[max(i0 - n_out, 0):i0])
                position[rp.line_axis] += rp.outboard_sign * float(np.sum(span))
                port_id = f"port:{eligible_ids[rp.port_index]}:mode:0"
                plane = plane_counts.get(port_id, 0)
                plane_counts[port_id] = plane + 1
                receivers_by_id[f"{port_id}/plane{plane}"] = replace(pe, position=tuple(position))
            receivers = [receivers_by_id[record.id] for record in raw.get("sparam_time_records", ())]
            end = injection_end(n_steps, grid.dt, records=receivers, grid=grid, sim=driven_sim)
            settling_runs.append(port_record_witness(
                raw.get("sparam_time_records", ()), raw.get("dt", grid.dt),
                end, freqs, freq_max=sim._freq_max))
        if graded:
            continue
        accs = raw["wire"] if wire_mode else raw["lumped"]
        if accs is None or len(accs) != n_ports:
            raise RuntimeError(
                "compute_lumped_wire_s_matrix_via_scan: production scan "
                f"returned {0 if accs is None else len(accs)} "
                f"{'wire' if wire_mode else 'lumped'} accumulators for drive "
                f"{j}, expected {n_ports}.  The port-spec registration is "
                "out of sync with the eligible-port list."
            )

        result_dtype = np.result_type(
            accs[0][1][0].dtype if sim._precision == "float64" else np.complex64,
            np.complex64)
        for i in range(n_ports):
            spec, vi = accs[i]
            # Lumped accs are (v_dft, i_dft, v_ref_dft); wire accs are
            # (v_dft, i_dft, v_inc_dft, v_port_dft, v_ref_dft) — take the
            # first two either way, the pre-injection drive reference from
            # its own slot in each family, plus the whole-port gap voltage
            # in wire mode (issue #764).
            v_dft, i_dft = vi[0], vi[1]
            v_all[j, i, :] = np.asarray(v_dft, dtype=np.complex128)
            i_all[j, i, :] = np.asarray(i_dft, dtype=np.complex128)
            if wire_mode and is_wire[i]:
                vp_all[j, i, :] = np.asarray(vi[3], dtype=np.complex128)
                vref_all[j, i, :] = np.asarray(vi[4], dtype=np.complex128)
            else:
                if wire_mode:
                    vp_all[j, i, :] = v_all[j, i, :]
                vref_all[j, i, :] = np.asarray(vi[2], dtype=np.complex128)

        rp_accs = raw.get("wire_refplane")
        if rp_accs:
            if plane_v is None:
                plane_v = np.zeros((n_ports, n_ports, 2, n_freqs),
                                   dtype=np.complex128)
                plane_im = np.zeros_like(plane_v)
                plane_ip = np.zeros_like(plane_v)
            for rp_spec, rp_vi in rp_accs:
                p = int(rp_spec.port_index)
                s = int(rp_spec.plane_slot)
                plane_v[j, p, s, :] = np.asarray(rp_vi[0],
                                                 dtype=np.complex128)
                plane_im[j, p, s, :] = np.asarray(rp_vi[1],
                                                  dtype=np.complex128)
                plane_ip[j, p, s, :] = np.asarray(rp_vi[2],
                                                  dtype=np.complex128)
                plane_enabled[p] = True
                plane_outboard[p] = int(rp_spec.outboard_sign)
                if s == 0:
                    plane_offsets[p] = int(rp_spec.n_cells_outboard)

    def _forward_columns(S):
        # forward()'s port S is run()'s matrix with each undeclared drive's
        # column NaN (#1373 decision C), as one-device forward() returns it.
        if not _forward_port_s11:
            return S
        driven = np.asarray([bool(pe.excite) for pe in eligible])
        return np.where(driven[None, :, None], S, np.asarray(np.nan + 1j * np.nan, dtype=S.dtype))

    if graded:
        S = np.stack([column for _, column in sorted(columns, key=lambda c: c[0])], axis=1)
    if wire_mode and plane_v is not None:
        # issue #313 reference-plane path: byte-frozen legacy diagonals +
        # plane-wave off-diagonals where both ports opted in.  The V/I
        # replay dump schema has no plane-phasor fields — fail loudly
        # rather than return a dump whose replay cannot reproduce S.
        if return_vi_dump:
            raise NotImplementedError(
                "return_vi_dump=True is not supported together with "
                "add_port(reference_plane_cells=...): the midpoint V/I "
                "replay bundle cannot reproduce the plane-wave "
                "off-diagonals (issue #313)."
            )
        from rfx.probes.refplane import (
            decompose_wire_s_matrix_with_reference_planes,
        )
        out = decompose_wire_s_matrix_with_reference_planes(
            v_all, i_all, z0, port_cell_counts,
            v_ref=vref_all,
            plane_v=plane_v, plane_im=plane_im, plane_ip=plane_ip,
            plane_enabled=plane_enabled,
            plane_offsets=plane_offsets,
            outboard_signs=plane_outboard,
            freqs=freqs,
            dt=float(grid.dt),
            dx=float(grid.cells('x')[0]),
            return_line_diagnostics=return_refplane_diagnostics,
        )
        if return_refplane_diagnostics:
            S, diag = out
            return np.asarray(S, dtype=result_dtype), freqs, diag
        if return_settling:
            missing = [item for item in settling_runs if not np.isfinite(item[0])]
            worst = missing[0] if missing else max(settling_runs, key=lambda item: item[0])
            return np.asarray(out, dtype=result_dtype), freqs, {
                **worst[1], "db": worst[0], "per_drive": [d for _, d in settling_runs]}
        return _forward_columns(np.asarray(out, dtype=result_dtype)), freqs

    if graded:
        pass
    elif wire_mode:
        # Issue #764 diagonal + issue #770 off-diagonal: the whole-port
        # gap-voltage channel feeds BOTH (frame-consistent whole-port
        # wave pair; the per-cell #308 frame is refuted physics kept only
        # on the legacy v_port=None path).
        if _main_wire_record:
            # Match run_uniform's one-wire fast path, including a vanishing wave.
            from rfx.probes.probes import driven_port_reflection
            S = np.asarray(driven_port_reflection(
                jnp.asarray(vp_all[0, 0]), jnp.asarray(i_all[0, 0]), z0[0]),
                dtype=result_dtype)[None, None, :]
        else:
            S = np.asarray(
                decompose_wire_s_matrix(v_all, i_all, z0, port_cell_counts,
                                        v_port=vp_all, v_ref=vref_all),
                dtype=result_dtype,
            )
        if return_vi_dump:
            # Mirror ``extract_s_matrix_wire``'s WirePortVIReplayBundle
            # field-for-field.  The wire dump stores the FDTD-sign midpoint V
            # (no negation, unlike lumped) — ``v_all``/``i_all`` already hold
            # the same ``sprobes[i].v_dft`` / ``i_dft`` the eager path stores.
            return WirePortVIReplayBundle(
                s_params=S,
                freqs=jnp.asarray(freqs),
                raw_voltages_fdt=v_all,
                raw_currents=i_all,
                port_impedances=z0,
                port_cell_counts=port_cell_counts,
                port_names=tuple(f"wire_{idx}" for idx in range(n_ports)),
                driven_port_indices=tuple(range(n_ports)),
                raw_port_voltages_fdt=vp_all,
                raw_drive_ref_voltages_fdt=vref_all,
            )
    else:
        S = np.asarray(
            decompose_lumped_s_matrix(v_all, i_all, z0, v_ref=vref_all),
            dtype=result_dtype,
        )
        if return_vi_dump:
            # Mirror ``extract_s_matrix``'s PortVIReplayBundle field-for-field.
            # The portable dump schema uses voltage positive into the DUT, so
            # store ``-V`` (the FDTD-sign V is ``v_all``); current is positive
            # into the DUT, so store ``+I`` (``i_all``) — matching the eager
            # comment exactly.
            return PortVIReplayBundle(
                s_params=S,
                freqs=jnp.asarray(freqs),
                voltages=-v_all,
                currents=i_all,
                port_impedances=z0,
                port_names=tuple(f"port_{idx}" for idx in range(n_ports)),
                driven_port_indices=tuple(range(n_ports)),
            )

    if return_refplane_diagnostics:
        return S, freqs, None
    if return_settling:
        missing = [item for item in settling_runs if not np.isfinite(item[0])]
        worst = missing[0] if missing else max(settling_runs, key=lambda item: item[0])
        return S, freqs, {**worst[1], "db": worst[0], "per_drive": [d for _, d in settling_runs]}
    return _forward_columns(S), freqs


def _lumped_recording_dfts(samples, freqs, dt, dx):
    """Host DFT of post-injection V and Yee-staggered I using scan helpers."""
    from rfx.measurement.dft import transform
    from rfx.probes.probes import _ampere_loop_values, _port_voltage_value

    fields = np.asarray(samples).reshape(len(samples), -1, 5)
    v = _port_voltage_value(fields[:, :, 0], dx)
    i = _ampere_loop_values(*(fields[:, :, k] for k in range(1, 5)), dx)
    f = np.asarray(freqs, dtype=np.float32)
    return (np.asarray(transform(v, f, dt, 'E')).T,
            np.asarray(transform(i, f, dt, 'H')).T)


def _distributed_lumped_accumulators(sim, grid, ports, freqs, n_steps, devices, drive):
    """S-driver adapter over the same in-loop main-record sampler as run()."""
    from rfx.runners.distributed_v2 import run_distributed
    result = run_distributed(sim, n_steps=n_steps, devices=devices,
                             _source_port_indices=None if drive is None else (drive,),
                             s_param_freqs=freqs, compute_s_params=True)
    return {"lumped": result.lumped_port_sparams, "wire": result.wire_port_sparams,
            "sparam_time_records": result.sparam_time_records, "dt": result.dt}

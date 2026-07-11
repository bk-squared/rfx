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

import numpy as np

import jax.numpy as jnp

from rfx.probes.probes import (
    PortVIReplayBundle,
    WirePortVIReplayBundle,
    decompose_lumped_s_matrix,
    decompose_wire_s_matrix,
)


def compute_lumped_wire_s_matrix_via_scan(
    sim, freqs, *, n_steps=None, return_vi_dump=False,
    return_refplane_diagnostics=False,
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
    Stage 1 supports a homogeneous all-lumped **or** all-wire port set (the
    eager ``extract_s_matrix`` / ``extract_s_matrix_wire`` are likewise
    called per-family).  A mixed lumped+wire set raises ``NotImplementedError``
    because the lumped and wire off-diagonal wave-decomposition conventions
    differ (per-cell impedance normalization for wire) and cross-family
    coupling is out of Stage-1 scope.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    n_freqs = len(freqs)

    # Build grid + materials exactly as the uniform forward lane does.
    grid = sim._build_grid()
    materials, debye_spec, lorentz_spec, pec_mask, _, _, _ = \
        sim._assemble_materials(grid)

    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=30)

    # Ordered sparam-eligible lumped/wire ports (impedance != 0), in
    # registration order — the same order the multi-drive index counts.
    eligible = [pe for pe in sim._ports if pe.impedance != 0.0]
    if not eligible:
        raise ValueError(
            "compute_lumped_wire_s_matrix_via_scan: no sparam-eligible "
            "lumped/wire ports (all ports have impedance 0)."
        )

    is_wire = [pe.extent is not None for pe in eligible]
    if any(is_wire) and not all(is_wire):
        raise NotImplementedError(
            "compute_lumped_wire_s_matrix_via_scan: mixed lumped + wire port "
            "sets are not supported in Stage 1 (the off-diagonal wave-"
            "decomposition conventions differ).  Use a homogeneous all-lumped "
            "or all-wire port set."
        )
    wire_mode = all(is_wire)

    n_ports = len(eligible)
    z0 = np.asarray([pe.impedance for pe in eligible], dtype=np.float64)

    # Per-port wire LIVE cell counts (only needed for the wire decomposer).
    # Issue #318: the wave-decomposition normalization is Z0c = Z0/n_live so
    # it matches the live-cell sigma fold's per-cell resistor law
    # R_cell = Z0/n_live (the #308 receive-channel selection relies on that
    # identity). ``pec_mask`` here is the assembled-geometry state BEFORE
    # any port-cell clearing — exactly the "live" definition. With no dead
    # cells this is the historical all-cells count.
    if wire_mode:
        from rfx.sources.sources import WirePort, _wire_port_live_cells
        axis_map = {"ex": 0, "ey": 1, "ez": 2}
        port_cell_counts = np.zeros(n_ports, dtype=np.int64)
        for idx, pe in enumerate(eligible):
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
                grid, wp, pec_mask)[2]

    # FDTD-sign V/I phasors per (drive j, receive i).
    v_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    i_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)

    # Opt-in reference-plane accumulators (issue #313): raw plane phasors
    # per (drive j, port p, plane slot).  Allocated lazily on the first
    # drive pass that returns plane data.
    plane_v = plane_im = plane_ip = None
    plane_enabled = np.zeros(n_ports, dtype=bool)
    plane_offsets = np.zeros(n_ports, dtype=np.int64)
    plane_outboard = np.zeros(n_ports, dtype=np.int64)

    for j in range(n_ports):
        raw = sim._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=n_steps,
            checkpoint=False,
            pec_mask=pec_mask,
            port_s11_freqs=freqs,
            _sparam_drive_idx=j,
            _return_raw_port_sparams=True,
        )

        accs = raw["wire"] if wire_mode else raw["lumped"]
        if accs is None or len(accs) != n_ports:
            raise RuntimeError(
                "compute_lumped_wire_s_matrix_via_scan: production scan "
                f"returned {0 if accs is None else len(accs)} "
                f"{'wire' if wire_mode else 'lumped'} accumulators for drive "
                f"{j}, expected {n_ports}.  The port-spec registration is "
                "out of sync with the eligible-port list."
            )

        for i in range(n_ports):
            spec, vi = accs[i]
            # Lumped accs are (v_dft, i_dft); wire accs are
            # (v_dft, i_dft, v_inc_dft) — take the first two either way.
            v_dft, i_dft = vi[0], vi[1]
            v_all[j, i, :] = np.asarray(v_dft, dtype=np.complex128)
            i_all[j, i, :] = np.asarray(i_dft, dtype=np.complex128)

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
            plane_v=plane_v, plane_im=plane_im, plane_ip=plane_ip,
            plane_enabled=plane_enabled,
            plane_offsets=plane_offsets,
            outboard_signs=plane_outboard,
            freqs=freqs,
            dt=float(grid.dt),
            dx=float(grid.dx),
            return_line_diagnostics=return_refplane_diagnostics,
        )
        if return_refplane_diagnostics:
            S, diag = out
            return np.asarray(S, dtype=np.complex64), freqs, diag
        return np.asarray(out, dtype=np.complex64), freqs

    if wire_mode:
        S = np.asarray(
            decompose_wire_s_matrix(v_all, i_all, z0, port_cell_counts),
            dtype=np.complex64,
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
            )
    else:
        S = np.asarray(
            decompose_lumped_s_matrix(v_all, i_all, z0),
            dtype=np.complex64,
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
    return S, freqs

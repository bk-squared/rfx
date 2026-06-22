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

from rfx.probes.probes import (
    decompose_lumped_s_matrix,
    decompose_wire_s_matrix,
)


def compute_lumped_wire_s_matrix_via_scan(sim, freqs, *, n_steps=None):
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

    Returns
    -------
    (S, freqs) : (np.ndarray, np.ndarray)
        ``S`` has shape ``(n_ports, n_ports, n_freqs)`` where ``S[i, j]`` is
        the response at receive port *i* when driving port *j*.

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

    # Per-port wire cell counts (only needed for the wire decomposer).
    if wire_mode:
        from rfx.sources.sources import WirePort, _wire_port_cells
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
            port_cell_counts[idx] = max(len(_wire_port_cells(grid, wp)), 1)

    # FDTD-sign V/I phasors per (drive j, receive i).
    v_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    i_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)

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

    if wire_mode:
        S = np.asarray(
            decompose_wire_s_matrix(v_all, i_all, z0, port_cell_counts),
            dtype=np.complex64,
        )
    else:
        S = np.asarray(
            decompose_lumped_s_matrix(v_all, i_all, z0),
            dtype=np.complex64,
        )

    return S, freqs

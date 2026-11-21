from __future__ import annotations

import numpy as np

from rfx.errors import AccuracyWarning, RfxUnsupported


def _validate_coax_msl_registration(
    eps_r_sub, self,
):
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
    """Return the MSL probe x positions in increasing x for either port direction, as
    ``coaxial_line_reflection_from_plane_voltages`` requires."""
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
    xs_sorted = sorted(xs_ladder)

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
    ladder_split_gamma_dev = None
    ladder_split_reflection_decades = None
    if return_ladder_voltages:
        _split_g_coax, _split_d_coax = _ladder_split_witness(
            z_planes_coax_m, v_coax_by_drive, ref_coax_m)
        _split_g_msl, _split_d_msl = _ladder_split_witness(
            np.asarray(xs_sorted), v_msl_by_drive, float(junction_x))
        ladder_split_gamma_dev = np.stack([_split_g_coax, _split_g_msl])
        ladder_split_reflection_decades = np.stack([_split_d_coax, _split_d_msl])

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

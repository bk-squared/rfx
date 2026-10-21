"""Assemble graded lumped and wire ports before source realization."""

from __future__ import annotations

from typing import NamedTuple

from rfx._grid_metric import field_index
from rfx.boundaries.pec import edge_is_pec as _edge_is_pec
from rfx.core.yee import MaterialArrays
from rfx.model.conductors import lumped_port_stage, publish_port_release
from rfx.nonuniform import (
    e_node_dual_spacing_at, port_metric, port_metric_axes, position_to_index,
)
from rfx.sources.sources import stamp_lumped_sigma as _stamp_lumped_sigma


class GradedPorts(NamedTuple):
    """Realized loads, conductors, indexed drives, and ordered V/I specs.

    Each drive is ``(port_index, (cell, component, waveform, sigma_port,
    unit_field))``. Its waveform is the declared excitation; the caller
    realizes it against the final materials when resolving its source queue.
    """

    materials: MaterialArrays
    materials_drive: MaterialArrays
    conductors: object
    drives: list
    port_specs: list


def assemble_graded_ports(sim, grid, materials, materials_drive, conductors, *,
                          pec_edge_masks, sizing_n, lane, s_param_freqs, compute_s_params=None):
    """Stamp positive-impedance ports without owning the source queue.

    Liveness precedes each port's release. Only lumped releases advance the
    masks read by subsequent ports, preserving the runner's existing order.
    ``sizing_n`` is retained in the assembly interface; drive sampling stays
    deferred in the caller until all material stamps have been applied.
    """
    drives = []
    wire_port_specs = []
    lumped_port_specs = []

    # Domain extents (for auto-detecting port direction).
    dom_lx = float(sim._domain[0])
    dom_ly = float(sim._domain[1])

    def _auto_direction(position) -> str:
        """Pick the outward-normal direction (+x/-x/+y/-y) of the port
        by finding the smallest *relative* distance to a boundary
        face. Relative (not absolute) distance avoids the trap where
        a short domain extent makes a completely-centered port appear
        "close" to its narrow-axis boundary.
        """
        x, y, _z = position
        rel = {
            "-x": float(x) / max(dom_lx, 1e-30),
            "+x": float(dom_lx - x) / max(dom_lx, 1e-30),
            "-y": float(y) / max(dom_ly, 1e-30),
            "+y": float(dom_ly - y) / max(dom_ly, 1e-30),
        }
        return min(rel, key=rel.get)

    for _port_index, pe in enumerate(sim._ports):
        if pe.impedance == 0.0:
            continue
        idx = (position_to_index(grid, pe.position) if getattr(pe, "extent", None) is not None
               else field_index(grid, pe.position, pe.component))
        if pe.extent is not None:
            # Wire port on non-uniform grid
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            axis = axis_map[pe.component]
            end_pos = list(pe.position)
            end_pos[axis] += pe.extent
            from rfx._periodic import interval_indices
            idx, idx_end = interval_indices(grid, pe.position, tuple(end_pos))
            lo_k = min(idx[axis], idx_end[axis])
            hi_k = max(idx[axis], idx_end[axis])

            # HALF-OPEN in edges, from the shared spelling: the driven
            # edges are the ones whose own location lies inside the
            # declared extent. This lane had its own endpoint-INCLUSIVE
            # copy, so the same declaration drove one more edge here than
            # on the uniform lane once that lane was corrected.
            from rfx.sources.sources import wire_port_edge_span
            _first_k, _last_k = wire_port_edge_span(
                grid, axis, lo_k, hi_k,
                float(pe.position[axis]), float(end_pos[axis]))
            wire_cells = list(range(_first_k, _last_k + 1))

            # Live-cell split (issue #318): a cell whose extent lies inside
            # PEC (assembled-geometry mask, read BEFORE this port's own
            # clearing below) is dead — it carries no port sigma and no
            # source, and drops out of the 1/n scaling. With no dead cells
            # this is bit-identical to the historical all-cells formula.
            _cells_ijk = []
            for k in wire_cells:
                cell = list(idx)
                cell[axis] = k
                _cells_ijk.append(tuple(cell))
            if pec_edge_masks is not None:
                live_flags = [
                    not _edge_is_pec(pec_edge_masks, pe.component,
                                     c[0], c[1], c[2])
                    for c in _cells_ijk]
            else:
                live_flags = [True] * len(_cells_ijk)
            n_live = sum(live_flags)
            if n_live == 0:
                raise ValueError(
                    f"WirePort at {pe.position} ({pe.component}, extent "
                    f"{pe.extent}): all {len(_cells_ijk)} extent cells land "
                    "inside PEC geometry, so the port has no live cell to "
                    "terminate or drive (issue #318). Shorten the extent or "
                    "move the port so at least one cell center sits outside "
                    "PEC."
                )

            # float64 so these are the SAME numbers the extractor reads at
            # rfx/nonuniform.py (`_dx_arr_np`/`_dy_arr_np`, both float64).
            # sigma and I must be sized on one set of metrics. Bit-identical
            # on a uniform profile. A TRACED axis stays a tracer instead —
            # ``port_metric_axes`` is the one place that decides which.
            (_dx_np, _x_host), (_dy_np, _y_host), (_dz_m, _z_host) = \
                port_metric_axes(grid)
            drive_stamps = {}
            for (ci, cj, ck), live in zip(_cells_ijk, live_flags):
                dxi = port_metric(_dx_np[ci], _x_host)
                dyj = port_metric(_dy_np[cj], _y_host)
                dual_xi = port_metric(
                    e_node_dual_spacing_at(_dx_np, ci), _x_host)
                dual_yj = port_metric(
                    e_node_dual_spacing_at(_dy_np, cj), _y_host)
                dual_zk = port_metric(
                    e_node_dual_spacing_at(_dz_m, ck), _z_host)
                # 3D wire port: σ = n_live * d_parallel / (Z0 * d_perp1 * d_perp2)
                # Each LIVE cell in the wire carries 1/n_live of total
                # impedance Z0 (issue #318 — dead cells excluded).
                #
                # #688: d_perp1/d_perp2 are the E-node DUAL spacings on the
                # two axes TRANSVERSE to the port component — the same
                # metrics `wire_port_current` weights its Ampere legs by
                # (#672). They are the sides of the dual face the conduction
                # current pierces, so V/I closes only if the termination and
                # the measurement use one metric family. d_parallel stays
                # PRIMAL: it is the length of the E edge V is taken over.
                # The old primal spelling for d_perp is right only where the
                # two transverse axes are locally uniform (measured 1.7778x
                # too small a conductance on a 2:1 step on both).
                if live:
                    if axis == 2:
                        d_cell = port_metric(_dz_m[ck], _z_host)
                        dp1, dp2 = dual_xi, dual_yj
                    elif axis == 1:
                        d_cell = dyj
                        dp1, dp2 = dual_xi, dual_zk
                    else:
                        d_cell = dxi
                        dp1, dp2 = dual_yj, dual_zk
                    sigma_port = n_live * d_cell / (pe.impedance * dp1 * dp2)
                    drive_stamps[(ci, cj, ck)] = (sigma_port, 1 / (n_live * d_cell))
                    # #1210: a port's load is a device across ONE edge, so
                    # it is recorded as an edge-owned stamp and kept out of
                    # the edge average. Stamped bare it was quartered, and a
                    # 50 ohm termination presented 200 ohm.
                    materials = _stamp_lumped_sigma(
                        materials, (ci, cj, ck), sigma_port, pe.component)
                    # #1256: and into the copy the drive is built from.
                    materials_drive = _stamp_lumped_sigma(
                        materials_drive, (ci, cj, ck), sigma_port, pe.component)
                    # No PEC clearing here (#931 §1.9, corrected): a cell
                    # is LIVE exactly when the port component's own edge is
                    # not PEC, so releasing that component is a no-op, and
                    # releasing the two tangential edges would open the
                    # conductor the port foot stands on.

            # Create per-cell sources — only when the port is excited.
            # Passive (excite=False) ports contribute just the σ
            # resistive termination above, acting as a matched load.
            #
            # Issue #764: the V/I reference cell is the midpoint of the
            # LIVE run, not of the raw extent. A dead extent cell (inside
            # PEC, #318) carries essentially no port current
            # (|I_dead|/|I_mid| = 0.003-0.03 measured on the #313 thru),
            # so an all-extent midpoint landing on a dead cell read a
            # quenched Ampere loop as the port current. With no dead
            # cells the live midpoint is bit-identical to the historical
            # all-extent midpoint.
            _live_cells = tuple(
                c for c, live in zip(_cells_ijk, live_flags) if live)
            conductors = publish_port_release(conductors, _live_cells,
                component=pe.component, entity_id=f"port[{_port_index}]", kind="wire")
            mid_cell = list(_live_cells[len(_live_cells) // 2])

            if pe.radius is not None:
                from rfx.sources.wire_radius import stamp_wire_radius
                materials = stamp_wire_radius(
                    grid, materials, pe.component, pe.radius, _live_cells)
                materials_drive = stamp_wire_radius(
                    grid, materials_drive, pe.component, pe.radius, _live_cells)

            if pe.excite:
                for cell_ijk, live in zip(_cells_ijk, live_flags):
                    # Dead extent cells get no source (issue #318).
                    if not live:
                        continue
                    drives.append((_port_index, (
                        cell_ijk, pe.component, pe.waveform,
                        drive_stamps[tuple(cell_ijk)][0],
                        drive_stamps[tuple(cell_ijk)][1])))

            # Wire port S-param spec — include excite/direction so the
            # runner scan body can record V/I and the post-processing
            # can orient the wave decomposition correctly.
            port_direction = pe.direction or _auto_direction(pe.position)
            wire_port_specs.append({
                'mid_i': mid_cell[0], 'mid_j': mid_cell[1],
                'mid_k': mid_cell[2],
                'component': pe.component,
                'impedance': pe.impedance,
                'excite': bool(pe.excite),
                'direction': port_direction,
                # Issue #764: static live-cell run for the whole-port gap
                # voltage V_port = sum_live(-E_c * d_par,c) and its count.
                'live_cells': _live_cells,
                'n_live': int(n_live),
                'kind': 'wire', 'port_id': _port_index,
            })
        else:
            # Single-cell lumped port
            i, j, k = idx
            (_dx_np, _x_host), (_dy_np, _y_host), (_dz_m, _z_host) = \
                port_metric_axes(grid)
            dxi = port_metric(_dx_np[i], _x_host)
            dyj = port_metric(_dy_np[j], _y_host)
            dual_xi = port_metric(e_node_dual_spacing_at(_dx_np, i), _x_host)
            dual_yj = port_metric(e_node_dual_spacing_at(_dy_np, j), _y_host)
            dual_zk = port_metric(e_node_dual_spacing_at(_dz_m, k), _z_host)
            # 3D lumped port: σ = d_parallel / (Z0 * d_perp1 * d_perp2)
            # This ensures correct power dissipation P = V²/Z0 in
            # anisotropic cells where dz ≠ dx.  The old formula
            # σ = 1/(Z0*d_parallel) is only valid for cubic cells.
            #
            # #688: d_perp1/d_perp2 are the E-node DUAL spacings on the two
            # axes transverse to the port component (the sides of the dual
            # face the conduction current pierces), matching the metrics
            # the wire-port Ampere loop uses (#672). d_parallel stays PRIMAL
            # — it is the E edge V is taken over. Sharpening the sentence
            # above: the primal spelling for d_perp is valid whenever the
            # two TRANSVERSE axes are locally uniform, which is strictly
            # weaker than cubic and is what actually makes the uniform lane
            # right.
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            port_axis = axis_map[pe.component]
            if port_axis == 2:
                d_parallel = port_metric(_dz_m[k], _z_host)
                d_perp1, d_perp2 = dual_xi, dual_yj
            elif port_axis == 1:
                d_parallel = dyj
                d_perp1, d_perp2 = dual_xi, dual_zk
            else:
                d_parallel = dxi
                d_perp1, d_perp2 = dual_yj, dual_zk
            sigma_port = d_parallel / (pe.impedance * d_perp1 * d_perp2)
            materials = _stamp_lumped_sigma(      # #1210, #1236
                materials, (i, j, k), sigma_port, pe.component)
            materials_drive = _stamp_lumped_sigma(    # #1256, #1236
                materials_drive, (i, j, k), sigma_port, pe.component)
            conductors = lumped_port_stage(conductors, pe, f"port[{_port_index}]")
            pec_edge_masks = conductors.pec_edges
            if pe.excite:
                drives.append((_port_index, (
                    idx, pe.component, pe.waveform, sigma_port, 1 / d_parallel)))

            # Requested bins opt lumped ports into V/I/reference accumulators
            # on both entries. An explicit run S request, or an all-passive
            # set on run (its V/I stay available with S off, as on uniform),
            # also admits default bins; otherwise the wire-only default holds.
            from rfx.measurement.port_s_contract import passive_record_diagnostic
            lumped_port_specs.append({
                'mid_i': i, 'mid_j': j, 'mid_k': k,
                'component': pe.component,
                'impedance': pe.impedance,
                'excite': bool(pe.excite),
                'direction': pe.direction or _auto_direction(pe.position),
                'live_cells': ((i, j, k),),
                'n_live': 1,
                'kind': 'lumped', 'port_id': _port_index,
                'record_only': not (s_param_freqs is not None or (lane == 'run_nonuniform' and (
                    compute_s_params is True or passive_record_diagnostic(sim)))),
            })

    # Preserve wire indices when explicit bins also include lumped ports.
    wire_port_specs.extend(lumped_port_specs)

    return GradedPorts(materials, materials_drive, conductors, drives, wire_port_specs)

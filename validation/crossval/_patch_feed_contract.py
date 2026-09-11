"""Build-only contact contract for the explicitly galvanic cv05/cv15 feeds.

This is a case requirement, not a general port policy: capacitive and open
wire ports remain valid API inputs. Read the registered port and the same
edge span used by its runner, never a builder's geometry metadata copy.
"""
from __future__ import annotations

import numpy as np


def assert_galvanic_patch_feed(sim, grid, *, ground_node, patch_node):
    """Require contact with the stack's identified planes and a live source."""
    from rfx.boundaries.pec import edge_is_pec, realized_pec_edge_masks, realized_wall_planes
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid,
    )
    from rfx.nonuniform import NonUniformGrid, position_to_index
    from rfx.sources.sources import WirePort, _wire_port_cells, wire_port_edge_span

    ports = list(sim._ports)
    if (len(ports) != 1 or ports[0].component != "ez"
            or ports[0].extent is None or not np.isfinite(ports[0].extent)
            or ports[0].extent <= 0 or not np.isfinite(ports[0].impedance)
            or ports[0].impedance <= 0):
        raise RuntimeError(
            "assert_galvanic_feed: this patch case requires one registered "
            "z-directed wire port with positive finite extent and impedance")
    pe = ports[0]
    end = list(pe.position)
    end[2] += pe.extent
    if isinstance(grid, NonUniformGrid):
        # The NU runner uses cumulative-coordinate lookup on all three axes,
        # then the SAME half-open edge-span primitive as the uniform runner.
        start_idx = position_to_index(grid, pe.position)
        end_idx = position_to_index(grid, tuple(end))
        lo, hi = sorted((start_idx[2], end_idx[2]))
        first, last = wire_port_edge_span(
            grid, 2, lo, hi, float(pe.position[2]), float(end[2]))
        ij = tuple(map(int, start_idx[:2]))
        coords = coords_from_nonuniform_grid(grid)
        assemble = sim._assemble_materials_nu
    else:
        wire = WirePort(start=pe.position, end=tuple(end), component=pe.component,
                        impedance=pe.impedance)
        cells = _wire_port_cells(grid, wire)
        first, last = cells[0][2], cells[-1][2]
        ij = tuple(map(int, cells[0][:2]))
        coords = coords_from_uniform_grid(grid)
        assemble = sim._assemble_materials

    lower, upper = int(first), int(last) + 1
    z_nodes = np.asarray(coords.z, dtype=np.float64)
    if not (0 <= lower < upper < len(z_nodes)):
        raise RuntimeError("assert_galvanic_feed: realized wire endpoints leave the grid")
    sheets, wires = [], []
    assembled = assemble(grid, pec_sheets=sheets, pec_wires=wires)
    mask = assembled[3]
    if mask is None and not sheets and not wires:
        raise RuntimeError("assert_galvanic_feed: no realized conductor for the patch feed")
    periodic = sim._periodic_flags()
    edges = realized_pec_edge_masks(mask, sheets=sheets, wires=wires, periodic=periodic)
    if all(edge_is_pec(edges, pe.component, *ij, k) for k in range(lower, upper)):
        raise RuntimeError(
            "assert_galvanic_feed: the registered wire has no live source edge; "
            "contact with wall planes inside one PEC body is not a driven gap")
    planes = realized_wall_planes(edges, 2, ij=ij, periodic=periodic)
    contacts = (lower in planes, upper in planes)
    if not all(contacts):
        raise RuntimeError(
            "assert_galvanic_feed: the registered wire's actual source endpoints "
            f"at column {ij}, nodes ({lower}, {upper}) "
            f"(z={z_nodes[lower]:.9g}, {z_nodes[upper]:.9g} m), do not both "
            f"meet realized conductor planes {list(planes)}. This case requires "
            "a galvanic ground-to-patch feed before any solve.")
    expected = (int(ground_node), int(patch_node))
    if (lower, upper) != expected:
        raise RuntimeError(
            "assert_galvanic_feed: the registered wire's actual source endpoints "
            f"({lower}, {upper}) do not match the stack ground/patch nodes "
            f"{expected}; another conductor cannot substitute for either terminal")
    return dict(
        port_z0=float(pe.position[2]), port_extent=float(pe.extent),
        z0_node_k=lower, z1_node_k=upper,
        z0_on_realized_plane=True, z1_on_realized_plane=True, galvanic=True,
    )

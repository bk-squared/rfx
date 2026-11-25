"""Realized interfaces and the declared MSL trace footprint's solved sheet edges.

A port attached to a thick conductor still supplies a declared trace plane;
this module evaluates that footprint and introduces no volume-edge model.
"""

from rfx.geometry.rasterize_grid import _box_axis_closed
from rfx.mesh_edges import solved_sheet_span
from rfx.preflight._common import local_cell, profile_boundary_cell
from rfx.preflight.msl_codes import msl_join, msl_text


def clearance_face(grid, axis_nodes, domain, dx, thick, axis, index, end, profile=None):
    """Return the interface and end cell on realized or declared coordinates.

    Without realized nodes the absorber starts at the declared domain face;
    its end cell is the declared profile's first or last cell, or scalar dx.
    Absorber thickness never enters the distance.
    """
    if axis_nodes is None:
        return (0.0 if end == "lo" else float(domain[index])), profile_boundary_cell(dx, profile, end)
    nodes = axis_nodes[index]
    pad = getattr(grid, f"pad_{axis}_{end}")
    k = int(pad) if end == "lo" else len(nodes) - int(pad) - 1
    cell = (nodes[k] - nodes[k - 1] if end == "lo" and pad
            else nodes[k + 1] - nodes[k] if end == "hi" and pad
            else nodes[1] - nodes[0] if end == "lo"
            else nodes[-1] - nodes[-2])
    return float(nodes[k]), float(cell)


def clearance_rule(realized, boundary, axis, end, height, thick, cell, end_cells):
    """Return the face's threshold and the measurement scope of its evidence."""
    scope = "" if realized else msl_text("clearance_unrealized")
    if thick <= 0:
        wall = getattr(getattr(boundary, axis), end)
        evidence = msl_text(
            "clearance_wall" if wall in ("pec", "pmc") else "clearance_inactive",
            wall=wall.upper(),
        )
        return 2.0 * height, msl_join(" ", (scope, evidence)) if scope else evidence
    threshold = max(end_cells / 3.0 * height, end_cells * cell)
    mesh_ratio = height / cell
    evidence = msl_text("clearance_measured")
    if abs(mesh_ratio - 3.0) > 1e-6 * 3.0:
        evidence = msl_join(" ", (evidence, msl_text(
            "clearance_unmeasured", mesh_height_ratio=mesh_ratio,
            threshold_end_cells=end_cells, end_cell_m=cell,
        )))
    return threshold, msl_join(" ", (scope, evidence)) if scope else evidence


def clearance_span(grid, nodes, axis, lo, hi, centre, domain_hi):
    """Apply the shared sheet-edge model to a closed declared port footprint.

    An empty footprint retains the declared bounds; the conductor-attachment
    check reports the missing conductor independently.
    """
    footprint = _box_axis_closed(nodes, lo, hi, local_cell(grid, axis, centre))
    pad_lo = getattr(grid, f"pad_{axis}_lo")
    pad_hi = getattr(grid, f"pad_{axis}_hi")
    footprint[:pad_lo] = False
    if pad_hi:
        footprint[-pad_hi:] = False
    span = solved_sheet_span(
        footprint, 0, nodes, lo, hi, domain_hi,
        periodic=axis in getattr(grid, "periodic_axes", ""),
    )
    return (span.lo, span.hi) if span is not None else (lo, hi)

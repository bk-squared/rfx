"""Admission of fixed-plane window deformations for the flux S-matrix entry."""
from contextvars import ContextVar
from functools import wraps

import numpy as np

from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer
from rfx.model.graded_admission import host_cells, traced_cell_mask

_flux_entry = ContextVar("waveguide_flux_mesh_entry", default=False)


def flux_mesh_entry(function):
    """Scope mesh admission to this calculator, restoring it even on refusal."""
    @wraps(function)
    def wrapped(sim, *args, **kwargs):
        token = _flux_entry.set(kwargs.get("normalize", False) == "flux")
        try:
            return function(sim, *args, **kwargs)
        finally:
            _flux_entry.reset(token)
    return wrapped


def admit_traced_window(sim, traced, prefix):
    """Return only for x-normal flux S on a fixed, isolated x window."""
    def refuse(reason):
        raise TracedStructureRefusal(f"{prefix}: {reason}; declare nominal_profiles= "
                                    "with a fixed-total interior x window.")

    if not _flux_entry.get():
        refuse("traced waveguide meshes require compute_waveguide_s_matrix(normalize='flux'); "
               "run()/forward() and normalize=True/False remain refused")
    if traced != ["x"] or any(e.direction[1] != "x" for e in sim._waveguide_ports):
        refuse("only port-normal x cell widths may be traced; transverse y/z and scalar dx are refused")
    grid = sim._build_nonuniform_grid()
    if is_tracer(grid.dt):
        refuse("the time step follows the traced cells; pin it with dt= and dt_min_cell=")
    mask = traced_cell_mask(grid, "x")
    if mask is None:
        refuse("the traced x axis needs a declared traced_cells window")
    interior = traced_cell_mask(grid, "x", padded=False)
    if interior[0] or interior[-1]:
        refuse("traced_cells touches an interior end copied into CPML pads")
    widths = host_cells(grid, "x")
    from rfx.geometry.rasterize_grid import _axis_node_positions
    from rfx.model.graded_grid import _nearest_on_line
    nodes = _axis_node_positions(widths, grid.pad_x_lo)
    for entry in sim._waveguide_ports:
        source = _nearest_on_line(nodes, widths, entry.x_position)
        sign = 1 if entry.direction.startswith("+") else -1
        planes = [source, source + sign * entry.ref_offset,
                  source + sign * entry.probe_offset]
        for coordinate in (entry.reference_plane, entry.probe_plane):
            if coordinate is not None:
                planes.append(_nearest_on_line(nodes, widths, coordinate))
        lo, hi = max(0, min(planes) - 1), min(len(mask), max(planes) + 2)
        if np.any(mask[lo:hi]):
            refuse(f"window overlaps source/probe/reference plane cells of waveguide port {entry.name!r}")
    from rfx.model.graded_grid import _assert_traced_window_total
    _assert_traced_window_total(sim._dx_profile, grid.nominal_cells["x"],
                                grid.traced_cells["x"], "x")


def host_index(grid, axis, position):
    """Physical node lookup for a fixed guide plane; concrete lookup unchanged."""
    if not hasattr(grid, "is_traced") or not grid.is_traced(axis):
        return grid.index_of(axis, position)
    from rfx.geometry.rasterize_grid import _axis_node_positions
    from rfx.model.graded_grid import DECLARED_SPAN_TOL_M, _nearest_on_line, interior_cells
    from rfx.model.graded_admission import refuse_window_cells
    ax = "xyz"[axis] if isinstance(axis, int) else axis
    widths = host_cells(grid, axis)
    pad_lo, pad_hi = (getattr(grid, f"pad_{ax}_{side}") for side in ("lo", "hi"))
    nodes = _axis_node_positions(widths, pad_lo)[pad_lo:len(widths) - pad_hi]
    if not nodes[0] - DECLARED_SPAN_TOL_M <= position <= nodes[-1] + DECLARED_SPAN_TOL_M:
        raise ValueError(f"waveguide ports: coordinate {position} outside nominal_profiles= {ax!r} interior")
    cells = interior_cells(widths, pad_lo, pad_hi, periodic=ax in grid.periodic_axes)
    index = _nearest_on_line(nodes, cells, position) + pad_lo
    refuse_window_cells(grid, axis, slice(max(0, index - 1), index + 1), "waveguide ports")
    return index

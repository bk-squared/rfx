"""Admission of point objects on the realized lattice, before any placement.

The builder retains declarations: automatic meshes and boundary pads can still
change after add_*(). Preflight's assembly is the first resolved lattice;
solve assembly uses the same check even when advisory preflight is disabled.
Coordinate-only geometry/plane lookups are not point-object declarations.
"""
from __future__ import annotations

import math

from rfx._internal._grid_metric import node_tie_tol
from rfx.errors import RfxRefusal


def admit_axis(position, *, axis, lo, hi, cell_lo, cell_hi, pad_lo, pad_hi,
               allow_in_absorber, label="Point"):
    """Check physical bounds before rounding; the opt-in never admits OOB."""
    pos = float(position)
    side = "lo" if pos < lo else "hi"
    depth = (lo - pos) / cell_lo if side == "lo" else (pos - hi) / cell_hi
    pad = pad_lo if side == "lo" else pad_hi
    tol_lo = node_tie_tol(pos, cell_lo)
    tol_hi = node_tie_tol(pos, cell_hi)
    outside = (not math.isfinite(pos) or
               pos < lo - pad_lo * cell_lo - tol_lo or
               pos > hi + pad_hi * cell_hi + tol_hi)
    in_pad = pos < lo - tol_lo or pos > hi + tol_hi
    if not outside and (not in_pad or allow_in_absorber):
        return
    cells = "cell" if math.isclose(depth, 1.) else "cells"
    where = f"{depth:g} {cells} into the {axis}-{side} CPML pad"
    if outside:
        where += " (outside the grid shape / physical grid bounds)"
    remedy = ("allow_in_absorber=True does not permit positions beyond the grid."
              if outside else
              "Pass allow_in_absorber=True on this object to keep this pad position; "
              "its fields are absorber-attenuated.")
    if outside and pad == 0:
        remedy += (f" Declared {axis} coordinate {position} is outside domain [{lo:g}, {hi}]. "
                   f"Boundary face {axis}_{side}: move it at least one cell inside the domain, off the wall.")
    raise RfxRefusal(f"{label} at {pos:g} m is {where}. {remedy}")


def admit_point(grid, position, *, allow_in_absorber=False, label="Point"):
    """Use the same full-grid bounds on uniform, graded and nominal meshes."""
    from rfx.model.graded_grid import NonUniformGrid, host_nominal_grid
    if isinstance(grid, NonUniformGrid):
        grid = host_nominal_grid(grid)
    for a, coordinate in enumerate(position):
        if grid.shape[a] == 1:
            continue
        axis = "xyz"[a]
        pad_lo = getattr(grid, f"pad_{axis}_lo")
        pad_hi = getattr(grid, f"pad_{axis}_hi")
        last = grid.shape[a] - pad_hi - 1
        # Periodic endpoints identify the seam; retain that domain endpoint.
        hi = (float(grid.domain[a]) if axis in getattr(grid, "periodic_axes", "")
              else float(grid.node_of(a, last)))
        cells = grid.cells(a)
        admit_axis(coordinate, axis=axis, lo=0., hi=hi,
                   cell_lo=float(cells[pad_lo]), cell_hi=float(cells[max(0, last - 1)]),
                   pad_lo=pad_lo, pad_hi=pad_hi,
                   allow_in_absorber=allow_in_absorber, label=label)


def admit_simulation_points(sim, grid):
    """Only add_probe/add_source/add_port entries, never plane objects."""
    for kind, entries in (("Probe", sim._probes), ("Source/port", sim._ports)):
        for i, entry in enumerate(entries):
            if kind == "Probe" and i in getattr(sim, "_internal_probe_indices", ()):
                continue
            options = dict(allow_in_absorber=getattr(entry, "allow_in_absorber", False),
                           label=f"{kind}[{i}]")
            admit_point(grid, entry.position, **options)
            if getattr(entry, "extent", None) is not None:
                end = list(entry.position)
                end["xyz".index(entry.component[1])] += entry.extent
                admit_point(grid, end, **options)

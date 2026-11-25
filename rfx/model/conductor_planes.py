"""Node-line comparisons shared by automatic meshing and realization admission."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ConductorMiss(NamedTuple):
    extent_percent: float
    subcell: bool
    delta: float
    length: float

    @property
    def refused(self):
        return self.subcell or abs(self.extent_percent) > 1. or self.delta > .01 * self.length


def conductor_node_miss(nodes, bounds, axis, neighbours=(), plane=None, domain_bounds=None,
                        period=None):
    """Compare a Box window or an already realized sheet plane on one axis.

    Sheet planes are supplied by sheet_spec_from_shape or dc_film_plane,
    including when they are read back from a prepared realization record.
    """
    from rfx._internal._grid_metric import half_open_volume_mask, node_tie_tol
    from rfx.geometry.rasterize_grid import _subcell_axes
    nodes = np.asarray(nodes, dtype=np.float64)
    widths = np.diff(nodes)
    lo, hi = (float(side[axis]) for side in bounds)
    if plane is None:
        mask = half_open_volume_mask((nodes[:-1] + nodes[1:]) / 2, lo, hi, widths)
        selected = np.flatnonzero(mask)
        extent = 0. if not selected.size else float(nodes[selected[-1] + 1] - nodes[selected[0]])
        subcell = any(a == axis for a, _, _ in _subcell_axes(
            *bounds, (nodes,) * 3, (widths,) * 3))
        drawn = max(0., min(hi, float(nodes[-1])) - max(lo, float(nodes[0])))
        percent = 100 * (extent - drawn) / drawn if drawn else 0.
        return ConductorMiss(percent, subcell, 0., float('inf'))
    from rfx.geometry.rasterize_grid import _local_cell
    tol = max(float(node_tie_tol(p, _local_cell(nodes, widths, p))) for p in (lo, hi))
    images = (plane,) if period is None else (plane, plane + period, plane - period)
    delta = min(max(lo - p, p - hi, 0.) for p in images)
    delta = 0. if delta <= tol else delta
    other = [a for a in range(3) if a != axis]
    planes = (list(domain_bounds if domain_bounds is not None else (nodes[0], nodes[-1]))
              if period is None else [])
    for blo, bhi in neighbours:
        if all(min(bounds[1][a], bhi[a]) > max(bounds[0][a], blo[a]) for a in other):
            planes.extend((float(blo[axis]), float(bhi[axis])))
    if period is not None:
        planes = [p + shift for p in planes for shift in (0., period, -period)] + [lo - period, hi + period]
    below = [p for p in planes if p < lo - tol]
    above = [p for p in planes if p > hi + tol]
    layers = ([lo - max(below)] if below else []) + ([min(above) - hi] if above else [])
    return ConductorMiss(0., False, delta, min(layers, default=float('inf')))



def merged_plane_remedy(label, bounds, merges):
    """Describe a recorded merge affecting this declaration, if one exists."""
    if bounds is None:
        return None
    candidates = (bounds[0][2], bounds[1][2], .5 * (bounds[0][2] + bounds[1][2]))
    match = next(((p, q) for p, q in merges
                  if any(abs(p - c) < 1e-12 for c in candidates)), None)
    if match is None:
        return None
    p, q = match
    return (f"{label}'s plane at {p:.12g} m is closer than a quarter of the local cell "
            f"to the plane at {q:.12g} m, so the automatic mesh merges them; draw it on "
            f"{q:.12g} m, or give it the thickness (add_thin_conductor), or pass dx= with a dz_profile.")

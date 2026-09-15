"""Resolve finite flux windows against the cells the monitor integrates.

Geometry-only host arithmetic shared by runners and preflight. Full-plane
``size=None`` keeps its existing runtime convention and has no finite window.
"""
from __future__ import annotations

import warnings
from math import ulp

import numpy as np


def validate_flux_region_inputs(size, center):
    """Require exactly two finite tangential coordinates/extents."""
    result = []
    for name, value in (("size", size), ("center", center)):
        if value is None:
            result.append(None)
            continue
        array = np.asarray(value)
        if array.shape != (2,) or np.iscomplexobj(array):
            raise ValueError(f"flux monitor {name} must contain exactly two real tangential values")
        try:
            array = array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"flux monitor {name} must contain exactly two real tangential values") from exc
        if not np.isfinite(array).all() or (name == "size" and np.any(array <= 0)):
            raise ValueError(f"flux monitor {name} must be finite" + (" and positive" if name == "size" else ""))
        result.append(tuple(float(x) for x in array))
    return tuple(result)


def resolve_flux_axis(edges, pad_lo, center, size, *, indices=None):
    """Nearest-edge finite CELL slice, clamped to the physical interior.

    ``indices`` preserves the uniform lane's existing round-to-even rule.
    Otherwise nearest cumulative edges are used, with the lower edge at a tie.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("flux monitor needs increasing finite interior cell edges")
    try:
        size = float(size)
        center = float((edges[0] + edges[-1]) / 2 if center is None else center)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("flux monitor requires a finite center and positive finite size") from exc
    if not np.isfinite([center, size]).all() or size <= 0:
        raise ValueError("flux monitor requires a finite center and positive finite size")
    requested = (center - size / 2, center + size / 2)
    if not np.isfinite(requested).all():
        raise ValueError("flux monitor endpoints must be finite")
    if indices is None:
        indices = [int(np.argmin(abs(edges - endpoint))) for endpoint in requested]
    lo, hi = (max(0, min(int(i), edges.size - 1)) for i in indices)
    if hi <= lo:
        raise ValueError("flux monitor resolves to a degenerate aperture with no interior cell; widen size or move center")
    # Distinguish roundoff at a boundary from physical overflow. Allow one
    # rounding of each supplied centre, size and boundary, plus the endpoint
    # addition: |delta c| + |delta s|/2 + |delta endpoint| + |delta boundary|.
    # Operand ULPs matter near zero, where subtraction can cancel. This bound
    # has no cell-size fraction and does not change the chosen cell indices.
    roundoff = [
        .5 * ulp(center) + .25 * ulp(size) + .5 * ulp(endpoint) + .5 * ulp(float(boundary))
        for endpoint, boundary in zip(requested, (edges[0], edges[-1]))
    ]
    return dict(
        cell_slice=(int(pad_lo) + lo, int(pad_lo) + hi),
        requested_bounds_m=requested,
        realized_bounds_m=(float(edges[lo]), float(edges[hi])),
        clamped_low=bool(edges[0] - requested[0] > roundoff[0]),
        clamped_high=bool(requested[1] - edges[-1] > roundoff[1]),
    )


def flux_region_message(record):
    axes = record["tangential_axes"]
    parts = []
    for axis, requested, realized in zip(axes, record["requested_bounds_m"], record["realized_bounds_m"]):
        parts.append(f"{axis}: requested [{requested[0]:.12g}, {requested[1]:.12g}] m, "
                     f"realized [{realized[0]:.12g}, {realized[1]:.12g}] m")
    suffix = " CLAMPED to interior cells; CPML cells are excluded." if record["clamped"] else " Interior cells; endpoint snapping is included in the reported bounds."
    return (f"Flux monitor {record['name']!r}, {record['axis']} plane "
            f"requested {record['requested_coordinate_m']:.12g} m, "
            f"realized {record['realized_coordinate_m']:.12g} m: " + "; ".join(parts) + suffix)


def resolve_flux_region(grid, entry, domain, *, warn=True):
    """Return a JSON-native finite-region record, or None for legacy full plane."""
    size, center = validate_flux_region_inputs(entry.size, getattr(entry, "center", None))
    if size is None:
        return None
    if entry.axis not in ("x", "y", "z") or not np.isfinite(entry.coordinate):
        raise ValueError("flux monitor requires a valid axis and finite coordinate")
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid, coords_from_uniform_grid
    from rfx.nonuniform import NonUniformGrid, position_to_index

    nonuniform = isinstance(grid, NonUniformGrid)
    axis = "xyz".index(entry.axis)
    tangential = [i for i in range(3) if i != axis]
    point = tuple(float(entry.coordinate) if i == axis else 0.0 for i in range(3))
    index = int(position_to_index(grid, point)[axis] if nonuniform else grid.position_to_index(point)[axis])
    coords = coords_from_nonuniform_grid(grid) if nonuniform else coords_from_uniform_grid(grid)
    axes = []
    for n, t in enumerate(tangential):
        letter = "xyz"[t]
        pad_lo = int(getattr(grid, f"pad_{letter}_lo"))
        pad_hi = int(getattr(grid, f"pad_{letter}_hi"))
        c = None if center is None else center[n]
        indices = None
        if nonuniform:
            # Use the canonical float64 geometry spine. Cumulative sums of
            # solver-facing float32 spacings can select a different boundary
            # and disagree with the coordinates that preflight reports.
            edges = np.asarray(getattr(coords, letter))[pad_lo:grid.shape[t] - pad_hi]
        else:
            # The collapsed z direction in 2D is one extruded integration
            # cell, not a pair of stored bounding nodes.
            cells = 1 if getattr(grid, "is_2d", False) and t == 2 else grid.shape[t] - pad_lo - pad_hi - 1
            edges = np.arange(cells + 1, dtype=float) * grid.dx
            c = float(domain[t]) / 2 if c is None else c
            indices = (round(c / grid.dx - size[n] / (2 * grid.dx)),
                       round(c / grid.dx + size[n] / (2 * grid.dx)))
        axes.append(resolve_flux_axis(edges, pad_lo, c, size[n], indices=indices))
    record = dict(
        name=entry.name, axis=entry.axis, lane="nonuniform" if nonuniform else "uniform",
        requested_coordinate_m=float(entry.coordinate),
        realized_coordinate_m=float(getattr(coords, entry.axis)[index]), normal_index=index,
        tangential_axes=["xyz"[t] for t in tangential], requested_size_m=list(size),
        requested_center_m=None if center is None else list(center),
        cell_slices=[list(a["cell_slice"]) for a in axes],
        requested_bounds_m=[list(a["requested_bounds_m"]) for a in axes],
        realized_bounds_m=[list(a["realized_bounds_m"]) for a in axes],
        clamped_ends=[[a["clamped_low"], a["clamped_high"]] for a in axes],
        clamped=any(a["clamped_low"] or a["clamped_high"] for a in axes),
    )
    if warn and record["clamped"]:
        warnings.warn(flux_region_message(record), UserWarning, stacklevel=2)
    return record

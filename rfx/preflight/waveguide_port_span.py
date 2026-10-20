"""Refuse an object face hidden between launch and reflection measurement.

Scans declared volumes, metric thin-conductor sheets and pinned PEC sheets.
Probe planes and post-processing reference_plane shifts are not this interval.
"""
from __future__ import annotations

import numpy as np

from rfx.model.waveguide_planes import entry_planes
from rfx.preflight._common import PreflightErrorWarning


def _bounds(shape):
    try:
        c1, c2 = shape.bounding_box()
        corners = np.asarray((c1, c2), dtype=float)
        return corners.min(axis=0), corners.max(axis=0)
    except (NotImplementedError, TypeError):
        return None


def _objects(sim, grid, axis):
    for i, g in enumerate(sim._geometry):
        bounds = _bounds(g.shape)
        if bounds is not None:
            yield f"{g.material_name} (geometry[{i}])", bounds
    for i, sheet in enumerate(sim._thin_conductors):
        bounds = _bounds(sheet.shape)
        if bounds is not None and abs(bounds[1][axis] - bounds[0][axis]) <= 1e-9:
            yield f"thin conductor[{i}]", bounds
    for i, sheet in enumerate(sim._pinned_sheets):
        if sheet.normal_axis != axis:
            continue
        lo, hi = np.zeros(3), np.zeros(3)
        pad = getattr(grid, f"pad_{'xyz'[axis]}_lo")
        lo[axis] = hi[axis] = grid.node_of(axis, pad + sheet.plane_index)
        transverse = [a for a in range(3) if a != axis]
        for a, indices in zip(transverse, (sheet.i_range, sheet.j_range)):
            pad = getattr(grid, f"pad_{'xyz'[a]}_lo")
            lo[a], hi[a] = (grid.node_of(a, pad + n) for n in indices)
        yield sheet.name or f"pinned PEC sheet[{i}]", (lo, hi)


def _validate_cfg_waveguide_port_span(self, _w):
    """One blocking finding per port/object, nearest hidden face first."""
    if not self._waveguide_ports or not (
            self._geometry or self._thin_conductors or self._pinned_sheets):
        return
    grid = self._build_realized_grid()
    # The mode builder places the plane from float32 cell widths, about 5e-8 m
    # from this float64 sum on a graded axis; a face that close is on the plane.
    tol = 1e-6
    for p, entry in enumerate(self._waveguide_ports):
        axis = "xyz".index(entry.direction[1])
        source, reference = entry_planes(entry, grid)
        start, end = sorted((source, reference))
        aperture = {}
        for a in range(3):
            if a != axis:
                limits = getattr(entry, f"{'xyz'[a]}_range")
                aperture[a] = limits if limits is not None else (0.0, self._domain[a])
        obstacles = []
        for name, (lo, hi) in _objects(self, grid, axis):
            if not all(hi[a] > limits[0] + tol and lo[a] < limits[1] - tol
                       for a, limits in aperture.items()):
                continue
            faces = [face for face in (lo[axis], hi[axis])
                     if start + tol < face < end - tol]
            if faces:
                face = min(faces, key=lambda f: abs(f - source))
                obstacles.append((abs(face - source), name, face))
        for _, name, face in sorted(obstacles):
            _w.warn(PreflightErrorWarning(
                f"waveguide port '{entry.name or p}' (source plane {source * 1e3:.6g} mm, "
                f"direction {entry.direction}) measures its reflection at {reference * 1e3:.6g} mm "
                f"(ref_offset={entry.ref_offset} cells); '{name}' has a face at "
                f"{face * 1e3:.6g} mm between the two, so the port cannot see that "
                "object's reflection and its S-parameters would not describe it. "
                f"Move the port or the object so the face lies beyond {reference * 1e3:.6g} mm "
                "in the launch direction, or pass a smaller ref_offset (minimum 3).",
                code="waveguide_port_obstacle_before_reference",
                loc=f"waveguide_port[{p}]",
                source="_validate_cfg_waveguide_port_span",
            ), stacklevel=3)


_validate_cfg_waveguide_port_span.__qualname__ = "_PreflightMixin._validate_cfg_waveguide_port_span"

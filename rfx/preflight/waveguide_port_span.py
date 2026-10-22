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


_VACUUM = (1.0, 0.0, 1.0, None, None, 0.0)


def _medium_at(sim, point):
    """The declared medium at a point: later geometry wins, as when assembled.

    A Box is its bounding box; any other shape reads as its own medium, so a
    face it straddles always counts (the conservative side).
    """
    from rfx.geometry.csg import Box
    medium = _VACUUM
    for i, g in enumerate(sim._geometry):
        bounds = _bounds(g.shape)
        if bounds is None or not all(bounds[0][a] <= point[a] <= bounds[1][a] for a in range(3)):
            continue
        if isinstance(g.shape, Box):
            spec = sim._resolve_material(g.material_name)
            medium = (float(spec.eps_r), float(spec.sigma), float(spec.mu_r),
                      repr(spec.debye_poles), repr(spec.lorentz_poles), float(spec.chi3))
            medium = _VACUUM if medium[:3] == _VACUUM[:3] and medium[3:] == ("None", "None", 0.0) else medium
        else:
            medium = ("shape", i)
    return medium


def _face_sample_points(grid, lo, hi, aperture):
    """Overlap centre first, then realised cell centres, at most 64 x 64."""
    from rfx.geometry.rasterize_grid import (
        centres_from_nonuniform_grid, centres_from_uniform_grid,
    )
    from rfx.nonuniform import NonUniformGrid

    centres = (centres_from_nonuniform_grid(grid) if isinstance(grid, NonUniformGrid)
               else centres_from_uniform_grid(grid))
    centre = np.zeros(3)
    axes = []
    for a, limits in aperture.items():
        lower, upper = max(lo[a], limits[0]), min(hi[a], limits[1])
        centre[a] = 0.5 * (lower + upper)
        cells = np.asarray(centres[a])
        cells = cells[(cells >= lower) & (cells <= upper)]
        if cells.size > 64:
            cells = cells[np.linspace(0, cells.size - 1, 64, dtype=int)]
        axes.append(cells)
    yield centre
    if any(cells.size == 0 for cells in axes):
        return
    transverse = list(aperture)
    samples = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    # Reserve the centre's slot at the cap by replacing its nearest sample.
    distances = np.sum((samples - centre[transverse]) ** 2, axis=1)
    omit = int(np.argmin(distances)) if len(samples) == 64 * 64 else None
    for i, sample in enumerate(samples):
        if i == omit or distances[i] == 0:
            continue
        point = centre.copy()
        point[transverse] = sample
        yield point


def _built(grid, axis, x):
    """The node a face at ``x`` is built on, placed as entry_planes places the source."""
    name = "xyz"[axis]
    index = grid.index_of(name, x)
    if hasattr(grid, "dx_arr"):
        pad = getattr(grid, f"pad_{name}_lo")
        cells = np.asarray(grid.cells(name))
        return float(np.insert(np.cumsum(cells[pad:]), 0, 0.0)[index - pad])
    return grid.node_of(name, index)


def _objects(sim, grid, axis):
    for i, g in enumerate(sim._geometry):
        bounds = _bounds(g.shape)
        if bounds is not None:
            yield f"{g.material_name} (geometry[{i}])", bounds, True
    for i, sheet in enumerate(sim._thin_conductors):
        bounds = _bounds(sheet.shape)
        if bounds is not None and abs(bounds[1][axis] - bounds[0][axis]) <= 1e-9:
            yield f"thin conductor[{i}]", bounds, False
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
        yield sheet.name or f"pinned PEC sheet[{i}]", (lo, hi), False


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
        for name, (lo, hi), volume in _objects(self, grid, axis):
            if not all(hi[a] > limits[0] + tol and lo[a] < limits[1] - tol
                       for a, limits in aperture.items()):
                continue
            # Open at the source plane, closed at the reference plane: a short
            # whose face lies on the reference plane reads |S22| 0.50 (review
            # of #1923), so a face there is an obstacle too. A PEC face is built
            # on its nearest node (drawn at 80.5 mm, built at 81 mm), so test
            # both; a face outside the domain cannot be built on the plane.
            def inside(face):
                return start + tol < face < end - tol or abs(face - reference) <= tol

            faces = [face for face in (lo[axis], hi[axis])
                     if inside(face) or (0.0 <= face <= self._domain[axis]
                                         and inside(_built(grid, axis, face)))]
            if volume and faces:
                # A face counts only where the medium changes across it: two
                # abutting boxes of one material, or a vacuum box, are a
                # uniform guide (#1913 review). Sample the realised transverse
                # cell centres and overlap centre, 10 um (or half the box) either side.
                points = tuple(_face_sample_points(grid, lo, hi, aperture))

                # A box thinner than 20 um has both 10 um samples outside it.
                step = min(1e-5, 0.5 * (hi[axis] - lo[axis]))

                def changes(face):
                    if hi[axis] - lo[axis] <= tol:  # a sheet: always a reflector
                        return True
                    for point in points:
                        before, after = point.copy(), point.copy()
                        before[axis], after[axis] = face - step, face + step
                        if _medium_at(self, before) != _medium_at(self, after):
                            return True
                    return False

                faces = [face for face in faces if changes(face)]
            if faces:
                face = min(faces, key=lambda f: abs(f - source))
                obstacles.append((abs(face - source), name, face))
        for _, name, face in sorted(obstacles):
            _w.warn(PreflightErrorWarning(
                f"waveguide port '{entry.name or p}' (source plane {source * 1e3:.6g} mm, "
                f"direction {entry.direction}) measures its reflection at {reference * 1e3:.6g} mm "
                f"(ref_offset={entry.ref_offset} cells); '{name}' has a face at "
                f"{face * 1e3:.6g} mm between the two (or at the measured plane once built on the mesh), so the port cannot see that "
                "object's reflection and its S-parameters would not describe it. "
                f"Move the port or the object so the face lies beyond {reference * 1e3:.6g} mm "
                "in the launch direction, or pass a smaller ref_offset (minimum 3).",
                code="waveguide_port_obstacle_before_reference",
                loc=f"waveguide_port[{p}]",
                source="_validate_cfg_waveguide_port_span",
            ), stacklevel=3)


_validate_cfg_waveguide_port_span.__qualname__ = "_PreflightMixin._validate_cfg_waveguide_port_span"

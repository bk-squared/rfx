"""Realize volume geometry as a closed cross-section on the invariant node.

Finite extrusion caps lie beyond the entire grid and its dual cells. Box
distance consumers form a centre from both bounds, so infinite Box bounds
would produce NaNs. These caps never intersect a sampled cell or interface.
"""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from rfx._internal._grid_metric import node_tie_tol
from rfx.core.jax_utils import is_tracer
from rfx.errors import RfxUnsupported
from rfx.geometry.csg import Box, Cylinder, OrientedBox, PolylineWire, Sphere, _grid_coords
from rfx.geometry.via import Via


def _concrete(value, shape):
    if is_tracer(value):
        raise RfxUnsupported(
            f"{type(shape).__name__}: a traced z bound or z centre needed for "
            "presence is unsupported in a 2-D cross-section")
    return float(value)


def _present(lo, hi, z0, width, shape):
    lo, hi = _concrete(lo, shape), _concrete(hi, shape)
    return lo - node_tie_tol(lo, width) <= z0 <= hi + node_tie_tol(hi, width)


def _refuse(shape, z0):
    lo, hi = shape.bounding_box()
    label = getattr(shape, "label", type(shape).__name__)
    raise RfxUnsupported(
        f"{label}: in a 2-D model the geometry is its cross-section at z = {z0:g} m; "
        f"this shape's z extent [{float(lo[2]):g}, {float(hi[2]):g}] m does not contain it. "
        f"Move the shape so its z extent contains z = {z0:g} m.")


def _bounds(shape):
    try:
        return shape.bounding_box()
    except (AttributeError, NotImplementedError):
        return None


@dataclass(frozen=True)
class _NodeSection:
    declared: object
    z0: float
    bounds: tuple

    def bounding_box(self):
        return self.bounds

    def mask_on_coords(self, x, y, z):
        plane = self.declared.mask_on_coords(x, y, np.array([self.z0]))
        return jnp.broadcast_to(plane, (len(x), len(y), len(z)))

    def mask(self, grid):
        if callable(getattr(self.declared, "mask_on_coords", None)):
            try:
                return self.mask_on_coords(*_grid_coords(grid))
            except NotImplementedError:
                pass
        return self.declared.mask(grid)


@dataclass(frozen=True)
class _BoxSections(Via):
    boxes: tuple

    def to_shapes(self):
        return [(box, "") for box in self.boxes]

    def bounding_box(self):
        return (tuple(min(b.corner_lo[a] for b in self.boxes) for a in range(3)),
                tuple(max(b.corner_hi[a] for b in self.boxes) for a in range(3)))

    def mask_on_coords(self, x, y, z):
        result = jnp.zeros((len(x), len(y), len(z)), dtype=bool)
        for box in self.boxes:
            result = result | box.mask_on_coords(x, y, z)
        return result

    def mask(self, grid):
        return self.mask_on_coords(*_grid_coords(grid))


def cross_section_2d(shape, grid):
    """Return the z extrusion at the node, refusing an absent top-level volume.

    Sheets retain their sampling rules after presence admission; wires retain
    their own invariant-axis admission. A custom
    shape may supply ``cross_section_2d(grid)``; otherwise its own mask is
    evaluated on the singleton node plane. Presence decisions are concrete;
    a radius remains differentiable when its centre is on the node.
    """
    if not getattr(grid, "is_2d", False) or isinstance(shape, PolylineWire):
        return shape
    from rfx.geometry.curved import CurvedPatch
    if is_tracer(getattr(grid, "dx", None)):
        raise RfxUnsupported("2-D cross-section: traced uniform dx is unsupported; use a concrete mesh")
    z0 = float(_grid_coords(grid)[2][0])
    width = float(grid.cells(2)[0])
    bounds = _bounds(shape)
    sheets = ([] if bounds is None else [a for a, (lo, hi) in enumerate(zip(*bounds))
              if not is_tracer(lo) and not is_tracer(hi) and lo == hi])
    if sheets:
        if 2 in sheets and not _present(
                bounds[0][2], bounds[1][2], z0, width, shape):
            _refuse(shape, z0)
        return shape
    cap = 4 * (max(abs(float(v)) for line in _grid_coords(grid) for v in line)
               + max(float(np.max(grid.cells(a))) for a in range(3)))
    bottom, top = z0 - cap, z0 + cap
    if isinstance(shape, Box):
        if not _present(shape.corner_lo[2], shape.corner_hi[2], z0, width, shape):
            _refuse(shape, z0)
        return Box((*shape.corner_lo[:2], bottom), (*shape.corner_hi[:2], top))
    if isinstance(shape, (Sphere, Cylinder)):
        zc = _concrete(shape.center[2], shape)
        dz = z0 - zc
        axial = isinstance(shape, Cylinder) and shape.axis == "z"
        reach = shape.height / 2 if axial else shape.radius
        if not (dz == 0 and is_tracer(reach)):
            if not _present(zc - reach, zc + reach, z0, width, shape):
                _refuse(shape, z0)
        if axial:
            return replace(shape, center=(*shape.center[:2], z0), height=2 * cap)
        radius = (shape.radius if dz == 0 else
                  jnp.sqrt(jnp.maximum(shape.radius**2 - dz**2, 0.))
                  if is_tracer(shape.radius) else np.sqrt(max(shape.radius**2 - dz**2, 0.)))
        if isinstance(shape, Sphere):
            return Cylinder((*shape.center[:2], z0), radius, 2 * cap, "z")
        axis = "xy".index(shape.axis)
        half = [radius, radius]
        half[axis] = shape.height / 2
        return Box(tuple(shape.center[a] - half[a] for a in range(2)) + (bottom,),
                   tuple(shape.center[a] + half[a] for a in range(2)) + (top,))
    if isinstance(shape, OrientedBox):
        rotation = np.asarray(shape.rotation)
        axes = [a for a in range(3) if np.allclose(
            np.abs(rotation[:, a]), (0., 0., 1.), rtol=0,
            atol=shape.orthogonality_tolerance)]
        if not len(axes):
            raise RfxUnsupported("OrientedBox: tilted out of the 2-D plane")
        axis = int(axes[0])
        if not _present(shape.center[2] - shape.size[axis] / 2,
                        shape.center[2] + shape.size[axis] / 2, z0, width, shape):
            _refuse(shape, z0)
        size = list(shape.size)
        size[axis] = 2 * cap
        return replace(shape, center=(*shape.center[:2], z0), size=tuple(size))
    if isinstance(shape, CurvedPatch):
        if not _present(bounds[0][2], bounds[1][2], z0, width, shape):
            _refuse(shape, z0)
        return shape
    if isinstance(shape, Via):
        if not _present(bounds[0][2], bounds[1][2], z0, width, shape):
            _refuse(shape, z0)
        boxes = tuple(b if b.corner_lo[2] == b.corner_hi[2] else
                      Box((*b.corner_lo[:2], bottom), (*b.corner_hi[:2], top))
                      for b, _ in shape.to_shapes()
                      if b.corner_lo[2] == b.corner_hi[2]
                      or _present(b.corner_lo[2], b.corner_hi[2], z0, width, b))
        return _BoxSections(boxes)
    bounds = _bounds(shape)
    if bounds is not None and not _present(bounds[0][2], bounds[1][2], z0, width, shape):
        _refuse(shape, z0)
    method = getattr(shape, "cross_section_2d", None)
    if callable(method):
        return method(grid)
    if bounds is None:
        return shape
    return _NodeSection(shape, z0,
                        ((*bounds[0][:2], bottom), (*bounds[1][:2], top)))


def realize_sections(entries, grid):
    """Transform declarations once per assembly; lend them to later consumers."""
    if not getattr(grid, "is_2d", False):
        return entries
    realized = tuple(replace(e, shape=cross_section_2d(e.shape, grid)) for e in entries)
    grid._cross_section_entries = (tuple(entries), realized)
    return realized


def section_entries(sim, grid):
    """Read cached sections, recomputing them when the declarations change."""
    saved = getattr(grid, "_cross_section_entries", None)
    if saved is None:
        return realize_sections(sim._geometry, grid)
    declared, realized = saved
    if len(declared) == len(sim._geometry) and all(a is b for a, b in zip(declared, sim._geometry)):
        return realized
    return realize_sections(sim._geometry, grid)

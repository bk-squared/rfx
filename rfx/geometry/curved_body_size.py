"""Size rule for a material Sphere or Cylinder on the primal-cell centres.

A material volume owns the primal cells whose centres it contains. A Sphere
of radius r holds a centre wherever it is placed only when r reaches half
the diagonal of the cell that contains its own centre; a Cylinder needs its
radius to reach half the diagonal of that cell's cross-section. Below that
size, whether any centre falls inside depends on where the body sits, so a
position sweep or a gradient through the position can cross from a result
to an empty body. The rule here decides from the declared radius and the
realized widths of the containing cell only, never from where the centres
happen to fall.

The rule is not applied, and the empty-mask test alone decides, when the
body's centre lies outside the grid's cells, when an axis holds a single
cell (a 2D grid), or when the centre or the cell widths are traced.
"""

from __future__ import annotations

import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx._internal._grid_metric import node_tie_tol
from rfx.errors import RfxRefusal
from rfx.geometry.csg import Cylinder, Sphere


def containing_cell_widths(point, centres, cell_sizes):
    """Primal widths of the cell holding ``point``, or None.

    Cell ``i`` spans ``[c_i - w_i/2, c_i + w_i/2)`` on each axis, which holds
    both for node-registered grids and for grids whose coordinates are the
    cell centres. None when an axis holds one cell, a value is traced, or the
    point lies outside the cells.
    """
    widths = []
    for p, c, w in zip(point, centres[:3], cell_sizes):
        if is_tracer(p) or is_tracer(c) or is_tracer(w):
            return None
        c = np.asarray(c, dtype=np.float64)
        w = np.broadcast_to(np.asarray(w, dtype=np.float64), c.shape)
        if c.size <= 1:
            return None
        k = int(np.searchsorted(c - w / 2, float(p), side="right")) - 1
        if k < 0 or (k == c.size - 1 and float(p) >= c[-1] + w[-1] / 2):
            return None
        widths.append(float(w[k]))
    return tuple(widths)


def _cross_axes(shape):
    if isinstance(shape, Sphere):
        return (0, 1, 2)
    return {"z": (0, 1), "y": (0, 2)}.get(shape.axis, (1, 2))


def body_size_threshold(shape, centres, cell_sizes):
    """``(D, tau, widths, axes)`` for a Sphere or Cylinder, or None.

    ``D`` is half the diagonal of the containing cell over the axes across
    the body (all three for a Sphere, the two cross-section axes for a
    Cylinder); ``tau`` is the node tie band at the body's centre. A radius
    below ``D + tau`` may hold no primal-cell centre at some position.
    """
    if not isinstance(shape, (Sphere, Cylinder)):
        return None
    if any(is_tracer(v) for v in shape.center):
        return None
    widths = containing_cell_widths(shape.center, centres, cell_sizes)
    if widths is None:
        return None
    axes = _cross_axes(shape)
    threshold = 0.5 * float(np.sqrt(sum(widths[a] ** 2 for a in axes)))
    tau = float(node_tie_tol(max(abs(float(v)) for v in shape.center), threshold))
    return threshold, tau, widths, axes


def is_below_size(shape, centres, cell_sizes) -> bool:
    """True when a concrete radius is below the threshold of this module."""
    rule = body_size_threshold(shape, centres, cell_sizes)
    if rule is None or is_tracer(shape.radius):
        return False
    threshold, tau, _, _ = rule
    return float(shape.radius) < threshold + tau


def refuse_small_body(shape, centres, cell_sizes, name):
    """Refuse a Sphere or Cylinder that may own no primal-cell centre.

    A traced radius with a concrete centre and widths is compared through
    the shared traced check.
    """
    rule = body_size_threshold(shape, centres, cell_sizes)
    if rule is None:
        return
    threshold, tau, widths, axes = rule
    kind = type(shape).__name__
    across = ("the cell" if kind == "Sphere" else
              f"the cell's ({'xyz'[axes[0]]}, {'xyz'[axes[1]]}) cross-section")

    def check(radius):
        radius = float(np.asarray(radius))
        if radius < threshold + tau:
            raise RfxRefusal(
                f"Material volume {name!r} ({kind}) is smaller than one cell can hold "
                f"at every position: radius {radius:.6g} m is below {threshold:.6g} m, "
                f"half the diagonal of {across} that contains its centre (cell widths "
                f"(x, y, z) = {widths} m), so whether any primal-cell centre lies inside "
                "it depends on where it is placed. Bodies just above this size are "
                "still over-polarized: with subpixel smoothing a sphere two cells "
                "across measured 1.30× and a rod two cells across 1.10× the converged "
                "value, and smaller bodies measured 1.15–6.4×. Put more cells across "
                "the body (a finer local mesh).")

    from rfx._traced_check import traced_check
    traced_check("material volume", "body size", check, shape.radius)

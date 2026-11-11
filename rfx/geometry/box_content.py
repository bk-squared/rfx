"""Exact box content on full-tensor edge supports, near axis-aligned Boxes.

The planar triplet rule gives a node one plane: the nearest face of the
union, extended without end. At a box edge or corner, or on the bisector of
one, the floating-point tie between two faces then picks the plane and the
model jumps; next to a box edge the extended plane also gives content to a
support no box reaches.

Value. For component ``a`` of a node the support is the one
:func:`rfx.geometry.triplet_tensor.fractions` uses: the edge from the node to
its neighbour along ``a`` (``h_a^+`` or ``h_a^-``) times the dual face
``[p - h^-/2, p + h^+/2]`` on each transverse axis. Its stored inverse
permittivity is the series over slices along ``a`` of the parallel average
across the dual face::

    1/eps_eff,a = sum_s (len_s / h) / (sum_r A_r(s) eps_r)

The slices and the sub-rectangles ``r`` are cut at the faces of the Boxes
that meet this support (at most :data:`MAX_BOXES_PER_SUPPORT`); each
sub-rectangle takes the permittivity of the last declared Box covering it,
or the background, as the assembled model paints them. A support of one
medium stores ``1/eps`` exactly. Off-diagonal entries are 0: the faces are
axis-aligned.

Scope, per node ``n``, from the realized cell sizes:

- the reach ``R(n)`` is 1.25 times the largest distance from ``n`` to a
  point of its supports;
- ``n`` is in the Box region when some Box's closed extent lies within
  ``R(n)``; only these nodes are touched, every other node keeps the planar
  rule bit for bit;
- only non-Box shapes block: with ``d(n)`` the distance to the nearest
  non-Box bounding box (0 for a shape without bounds) and ``h(n)`` the node's
  largest cell, ``w(n) = clip((d(n) - R(n)) / h(n), 0, 1)``, and the node's
  eight triplets become ``w * exact + (1 - w) * planar``; ``w = 0`` keeps the
  planar triplets bit for bit, ``w = 1`` writes the exact ones.

The scope is decided on concrete host coordinates in float64. With a Box in
the scene, a traced coordinate or permittivity is refused by name.
"""
from typing import NamedTuple

import numpy as np

from rfx.errors import RfxUnsupported

MAX_BOXES_PER_SUPPORT = 8
"""Boxes (periodic images counted) whose closed extent may meet one edge support."""

REACH_FACTOR = 1.25
"""Margin on the largest node-to-support distance that defines the Box region."""

TRACED_REASON = (
    "the exact box content near an axis-aligned Box is scoped on concrete coordinates and realized "
    "cell sizes; under a trace the build runs in float32, and a plain and a compiled run of one "
    "request differed by 9.7e-4 of the loss "
    "(a traced sphere radius beside a concrete eps_r 2 box)")
"""Why a traced input is refused when a Box is in the scene."""

KNOWN_LIMITATION = ("see docs/guides/known_limitations.md, "
                    "full-tensor smoothing at box edges")


def traced_coordinate_message(names):
    """The refusal text for a traced shape coordinate beside a Box."""
    return ("subpixel_smoothing='full_tensor' with a traced shape coordinate (material(s) "
            + ", ".join(repr(n) for n in names) + ") in a scene with an axis-aligned Box is "
            "refused: " + TRACED_REASON + ". Pass concrete coordinates, or use "
            "subpixel_smoothing=True or 'kottke_pec'; " + KNOWN_LIMITATION + ".")


class BoxScope(NamedTuple):
    """Host float64 scope and exact values of the Box rule.

    ``region`` and ``weight`` have the grid's shape; ``values[a][s]`` (sign 0
    positive) hold the exact inverse permittivity at the nodes with
    ``weight > 0`` inside the region and are NaN elsewhere.
    """

    region: np.ndarray
    weight: np.ndarray
    values: tuple

    @property
    def written(self):
        """Nodes whose triplets change: in the region with w > 0."""
        return self.region & (self.weight > 0)

    @property
    def replaced(self):
        """Nodes whose triplets are the exact ones (w = 1)."""
        return self.region & (self.weight == 1)


def is_exact_box(shape):
    """Only the plain axis-aligned Box carries the exact content."""
    from rfx.geometry.csg import Box
    return type(shape) is Box


def _images(grid, shape, bounds):
    """Periodic bounds within the Box scope and blocker reach ``R + h``.

    Evaluate ``_reach`` with each axis's largest cell to bound every node's
    scope radius R and the extra cell h in the blocker weight. On a cubic
    mesh, R = REACH_FACTOR * sqrt(h**2 + h**2/4 + h**2/4), so R + h is
    about 2.53 h. Both Box and blocker images use this wider padding.
    """
    from itertools import product

    from rfx._internal._periodic import _sdf_image_ranges
    largest = [np.asarray([np.max(grid.cells(a))]) for a in range(3)]
    radius, extra = _reach([(h, h) for h in largest])
    ranges = _sdf_image_ranges(grid, shape, padding=float(np.max(radius + extra)))
    lo, hi = bounds
    out = []
    for shift in product(*ranges):
        offset = [-k * float(grid.domain[a]) if k else 0. for a, k in enumerate(shift)]
        out.append((tuple(float(lo[a]) + offset[a] for a in range(3)),
                    tuple(float(hi[a]) + offset[a] for a in range(3))))
    return out


def _distance(lines, lo, hi):
    """Euclidean distance from every node to the closed box [lo, hi]."""
    total = 0.
    for a in range(3):
        gap = np.maximum(np.maximum(lo[a] - lines[a], lines[a] - hi[a]), 0.)
        shape = [1, 1, 1]
        shape[a] = -1
        total = total + gap.reshape(shape) ** 2
    return np.sqrt(total)


def _classify(grid, shapes, masks, z0):
    """Box images ``(lo, hi, eps)`` in declared order, and blocking bounds."""
    from rfx.geometry import smoothing as smooth
    from rfx.geometry.csg import declared_bounds

    boxes: list = []
    blockers: list = []  # (lo, hi) bounds, or None for a shape without bounds
    planar_2d = getattr(grid, 'is_2d', False)
    for shape, eps in shapes:
        if is_exact_box(shape) and not (
                planar_2d and not float(shape.corner_lo[2]) <= z0 <= float(shape.corner_hi[2])):
            for lo, hi in _images(grid, shape, (shape.corner_lo, shape.corner_hi)):
                if planar_2d:  # one z plane: the box is invariant along z
                    lo, hi = (lo[0], lo[1], -np.inf), (hi[0], hi[1], np.inf)
                boxes.append((lo, hi, float(eps)))
            continue
        if not smooth._has_sdf(shape) and masks.get(id(shape)) is None:
            continue
        bounds = declared_bounds(shape)
        if bounds is None:
            blockers.append(None)
            continue
        for lo, hi in _images(grid, shape, bounds):
            if planar_2d:
                lo, hi = (lo[0], lo[1], -np.inf), (hi[0], hi[1], np.inf)
            blockers.append((lo, hi))
    return boxes, blockers


def _widths(grid, coords, cell_lengths):
    """Per axis the 1-D node line and its (minus, plus) cell widths."""
    from rfx.geometry.node_tensor import uniform_cell

    lines = [np.asarray(coords[a], dtype=np.float64).reshape(-1) for a in range(3)]
    if cell_lengths is None:
        h = uniform_cell(grid)
        return lines, [(np.full(line.shape, h), np.full(line.shape, h)) for line in lines]
    widths = []
    for a in range(3):
        minus, plus = cell_lengths[a]
        widths.append(tuple(np.broadcast_to(np.asarray(w, dtype=np.float64).reshape(-1),
                                            lines[a].shape) for w in (minus, plus)))
    return lines, widths


def _reach(widths):
    """R(n) and h(n) on the grid from the realized cell sizes."""
    largest = [np.maximum(*widths[a]) for a in range(3)]
    shaped = []
    for a in range(3):
        shape = [1, 1, 1]
        shape[a] = -1
        shaped.append(largest[a].reshape(shape))
    r2 = 0.
    for a in range(3):
        b, c = [k for k in range(3) if k != a]
        r2 = np.maximum(r2, shaped[a] ** 2 + (shaped[b] / 2) ** 2 + (shaped[c] / 2) ** 2)
    h = np.maximum(np.maximum(shaped[0], shaped[1]), shaped[2])
    return REACH_FACTOR * np.sqrt(r2), h


def _breaks(start, stop, lo, hi, valid):
    """Sorted cut points of [start, stop] at the gathered boxes' faces, shape (N, 2k + 2)."""
    clip_lo = np.where(valid, np.clip(lo, start[:, None], stop[:, None]), start[:, None])
    clip_hi = np.where(valid, np.clip(hi, start[:, None], stop[:, None]), start[:, None])
    return np.sort(np.concatenate([start[:, None], stop[:, None], clip_lo, clip_hi], axis=1),
                   axis=1)


def _support_values(lines, widths, nodes, boxes, background_eps):
    """Exact inverse permittivity per ``[a][s]`` at the flat node indices ``nodes``."""
    index = np.unravel_index(nodes, tuple(len(line) for line in lines))
    point = [lines[a][index[a]] for a in range(3)]
    minus = [widths[a][0][index[a]] for a in range(3)]
    plus = [widths[a][1][index[a]] for a in range(3)]
    lo_all = np.asarray([b[0] for b in boxes], dtype=np.float64)    # (M, 3)
    hi_all = np.asarray([b[1] for b in boxes], dtype=np.float64)
    eps_all = np.asarray([b[2] for b in boxes], dtype=np.float64)
    out = []
    for a in range(3):
        b, c = [k for k in range(3) if k != a]
        per_sign = []
        for sign in (0, 1):
            start = {a: point[a] if sign == 0 else point[a] - minus[a]}
            stop = {a: point[a] + plus[a] if sign == 0 else point[a]}
            for t in (b, c):
                start[t], stop[t] = point[t] - minus[t] / 2, point[t] + plus[t] / 2
            # Gather: boxes whose closed extent meets the closed support, in declared order.
            meets = np.ones((len(nodes), len(boxes)), dtype=bool)
            for t in range(3):
                meets &= (lo_all[None, :, t] <= stop[t][:, None]) & (hi_all[None, :, t] >= start[t][:, None])
            count = meets.sum(axis=1)
            most = int(count.max()) if count.size else 0
            if most > MAX_BOXES_PER_SUPPORT:
                raise RfxUnsupported(
                    f"subpixel_smoothing='full_tensor': {most} boxes (periodic images counted) "
                    f"meet one edge support; the exact box-content rule handles at most "
                    f"{MAX_BOXES_PER_SUPPORT}. Merge touching boxes of one material, or use "
                    "subpixel_smoothing=True.")
            k = max(most, 1)
            slots = np.argsort(~meets, axis=1, kind='stable')[:, :k]
            valid = np.arange(k)[None, :] < count[:, None]
            lo = lo_all[slots]                                        # (N, k, 3)
            hi = hi_all[slots]
            eps = eps_all[slots]
            cuts = {t: _breaks(start[t], stop[t], lo[..., t], hi[..., t], valid) for t in range(3)}
            length = stop[a] - start[a]
            span = {t: stop[t] - start[t] for t in (b, c)}
            total = np.zeros(len(nodes))
            low = np.full(len(nodes), np.inf)
            high = np.full(len(nodes), -np.inf)
            for i in range(cuts[a].shape[1] - 1):
                piece = cuts[a][:, i + 1] - cuts[a][:, i]
                mid_a = (cuts[a][:, i + 1] + cuts[a][:, i]) / 2
                cover = valid & (lo[..., a] < mid_a[:, None]) & (mid_a[:, None] < hi[..., a])
                mean = np.zeros(len(nodes))
                for j in range(cuts[b].shape[1] - 1):
                    width_b = cuts[b][:, j + 1] - cuts[b][:, j]
                    mid_b = (cuts[b][:, j + 1] + cuts[b][:, j]) / 2
                    in_b = cover & (lo[..., b] < mid_b[:, None]) & (mid_b[:, None] < hi[..., b])
                    for m in range(cuts[c].shape[1] - 1):
                        width_c = cuts[c][:, m + 1] - cuts[c][:, m]
                        mid_c = (cuts[c][:, m + 1] + cuts[c][:, m]) / 2
                        inside = in_b & (lo[..., c] < mid_c[:, None]) & (mid_c[:, None] < hi[..., c])
                        medium = np.full(len(nodes), float(background_eps))
                        for slot in range(k):  # later declared wins
                            medium = np.where(inside[:, slot], eps[:, slot], medium)
                        area = (width_b / span[b]) * (width_c / span[c])
                        mean = mean + area * medium
                        present = (piece > 0) & (area > 0)
                        low = np.where(present, np.minimum(low, medium), low)
                        high = np.where(present, np.maximum(high, medium), high)
                safe = np.where(mean > 0, mean, 1.)
                total = total + np.where(piece > 0, (piece / length) / safe, 0.)
            one = low == high
            per_sign.append(np.where(one, 1 / np.where(one, low, 1.), total))
        out.append(tuple(per_sign))
    return out


def box_scope(grid, shapes, coords, cell_lengths, background_eps, masks, *, host):
    """The Box rule's scope and exact values, or None without a Box.

    ``coords`` are the three node lines; ``cell_lengths`` is None on a uniform
    mesh or the graded per-axis ``(minus, plus)`` widths. ``host`` is the
    builder's decision that every input is concrete; with a Box in the scene a
    traced input is refused by name.
    """
    if not any(is_exact_box(shape) for shape, _ in shapes):
        return None
    if not host:
        from rfx.core.jax_utils import is_tracer
        kind = ("permittivity" if any(is_tracer(e) for _, e in shapes) else
                "shape coordinate or mesh")
        raise RfxUnsupported(
            f"subpixel_smoothing='full_tensor' with a traced {kind} in a scene with an "
            "axis-aligned Box is refused: " + TRACED_REASON + "; " + KNOWN_LIMITATION + ".")
    lines, widths = _widths(grid, coords, cell_lengths)
    boxes, blockers = _classify(grid, shapes, masks, float(lines[2][0]))
    shape3 = tuple(len(line) for line in lines)
    if not boxes:
        return None
    reach, h = _reach(widths)
    reach = np.broadcast_to(reach, shape3)
    region = np.zeros(shape3, dtype=bool)
    for lo, hi, _ in boxes:
        region |= _distance(lines, lo, hi) <= reach
    gap = np.full(shape3, np.inf)
    for bounds in blockers:
        gap = np.zeros(shape3) if bounds is None else np.minimum(gap, _distance(lines, *bounds))
    weight = np.clip((gap - reach) / h, 0., 1.)
    written = region & (weight > 0)
    nodes = np.flatnonzero(written)
    values = []
    flat = _support_values(lines, widths, nodes, boxes, background_eps) if nodes.size else None
    for a in range(3):
        per_sign = []
        for s in (0, 1):
            full = np.full(shape3, np.nan)
            if flat is not None:
                full.reshape(-1)[nodes] = flat[a][s]
            per_sign.append(full)
        values.append(tuple(per_sign))
    return BoxScope(region, weight, tuple(values))


def apply_box_scope(triplets, scope):
    """Blend the exact triplets into the planar ones at the written nodes (host)."""
    from rfx.geometry.triplet_tensor import SIGNS

    if scope is None or not scope.written.any():
        return triplets
    out = np.array(triplets, dtype=np.float64, copy=True)
    where = np.nonzero(scope.written)
    w = scope.weight[where][:, None, None]
    full = (w == 1)
    for t, signs in enumerate(SIGNS):
        exact = np.zeros((len(where[0]), 3, 3))
        for a, s in enumerate(signs):
            exact[:, a, a] = scope.values[a][s][where]
        planar = out[t][where]
        out[t][where] = np.where(full, exact, w * exact + (1 - w) * planar)
    return out.astype(np.asarray(triplets).dtype, copy=False)

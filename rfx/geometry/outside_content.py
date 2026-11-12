"""The outside part of a smoothed sample's normal support.

A smoothed sample's cell, seen along the normal of its face, is the
sample's own medium up to the face and, beyond it, whatever the other
declared shapes put on that line. This module gives that outside part:
each Box, Sphere and Cylinder (and each periodic image) is intersected
exactly with the normal line, the pieces are painted in declaration order
and merged by medium identity, and a segment of the line (a Yee cell, a
node cell, a triplet edge or dual face) gets its mean permittivity, mean
inverse permittivity, the one-medium select and its medium bounds.
For a Yee sample it also gives the sample's own material on the line: the
merged runs of its own group's bodies, the fill they leave in the support
and the gaps between them, so that a planar body with both faces inside the
support is averaged with its own length and the media on both sides.
``rfx.geometry.smoothing`` (Yee samples), ``rfx.geometry.node_tensor``
(node xi) and ``rfx.geometry.triplet_tensor`` (planar triplets) read it.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import numpy as np

from rfx.model.line_streams import _merge_stream, _own_stream


class _LinePieces(NamedTuple):
    lo: Any
    hi: Any
    eps: Any
    face: Any
    own: Any = None


class _OwnRuns(NamedTuple):
    """The own group's material on a sample's normal support.

    ``thin`` marks samples where one merged run of planar own bodies has
    both ends strictly inside the support; ``fill`` is the length of all
    own runs inside the support over its length; ``gaps`` are the support's
    portions outside every own run, as (start, end) pairs.
    """
    thin: Any
    fill: Any
    gaps: Any


class _SegmentContent(NamedTuple):
    eps: Any
    inv: Any
    single: Any
    eps_min: Any
    eps_max: Any


def _slab_interval(xp, p, n, center, half, low, high, reach):
    """The interval of p + t*n between two parallel planes of one axis."""
    moving = n != 0
    a = xp.where(moving, low - p, 0) / xp.where(moving, n, 1)
    b = xp.where(moving, high - p, 0) / xp.where(moving, n, 1)
    inside = xp.abs(p - center) - half <= 0
    return (xp.where(moving, xp.minimum(a, b), xp.where(inside, -reach, reach)),
            xp.where(moving, xp.maximum(a, b), xp.where(inside, reach, -reach)))


def _axial_interval(xp, shape, points, normal, reach):
    """A Cylinder's interval between its two end caps, clipped to the window."""
    a = 'xyz'.index(shape.axis)
    center, half = shape.center[a], shape.height / 2.0
    lo, hi = _slab_interval(xp, points[a], normal[a], center, half,
                            center - half, center + half, reach)
    return xp.maximum(lo, -reach), xp.minimum(hi, reach)


def _line_interval(xp, shape, points, normal, reach):
    """Exact solid intersection with p + t*n, clipped to a finite window.

    Parallel face tests deliberately use the SDF's arithmetic, including
    its centre/half-size construction on the image-shifted coordinates.
    Unused divisions and square roots have finite primal and AD operands.
    """
    from rfx.geometry.csg import Box, Cylinder

    lo, hi = -reach, reach

    if isinstance(shape, Box):
        low, high = xp.array(shape.corner_lo), xp.array(shape.corner_hi)
        center, half = (low + high) / 2.0, (high - low) / 2.0
        for a in range(3):
            left, right = _slab_interval(xp, points[a], normal[a], center[a], half[a],
                                         low[a], high[a], reach)
            lo, hi = xp.maximum(lo, left), xp.minimum(hi, right)
    else:
        axes = list(range(3))
        if isinstance(shape, Cylinder):
            axes.remove('xyz'.index(shape.axis))
            lo, hi = _axial_interval(xp, shape, points, normal, reach)
        delta = [points[a] - shape.center[a] for a in axes]
        ns = [normal[a] for a in axes]
        aa = sum(n * n for n in ns)
        bb = sum(p * n for p, n in zip(delta, ns))
        rr = sum(p * p for p in delta)
        cc = rr - shape.radius ** 2
        discriminant = bb * bb - aa * cc
        moving = aa > 0
        positive = moving & (discriminant > 0)
        root = xp.where(positive, xp.sqrt(xp.where(positive, discriminant, 1)), 0)
        left = xp.where(moving, -bb - root, 0) / xp.where(moving, aa, 1)
        right = xp.where(moving, -bb + root, 0) / xp.where(moving, aa, 1)
        radial = xp.where(rr > 0, xp.sqrt(xp.where(rr > 0, rr, 1)), 0) - shape.radius
        inside = radial <= 0
        left = xp.where(moving, left, xp.where(inside, -reach, reach))
        right = xp.where(moving, right, xp.where(inside, reach, -reach))
        valid = ~moving | (discriminant >= 0)
        lo = xp.maximum(lo, xp.where(valid, left, reach))
        hi = xp.minimum(hi, xp.where(valid, right, -reach))
    return lo, hi


def _outside_pieces(xp, groups, current, coords, distance, normal, length,
                    background, masks, grid, *, normal_source, normal_image,
                    best_s, nominal_h_max=None, own_fill=False):
    """Paint and merge the normal line once, in original declaration order.

    Shapes are filtered only with concrete declared bounds and widths.
    Medium identities, rather than equal traced values, decide merging.
    The finite window contains every non-bulk Yee/node/triplet support.
    With ``own_fill`` the result also carries the current group's own runs
    (:func:`_own_runs`) over the support [-length/2, length/2].
    """
    from itertools import product
    import math
    from rfx.geometry.csg import Box, Sphere, Cylinder, declared_bounds
    from rfx._internal._periodic import _fill_image_range

    widths = tuple(grid.cells(a) for a in range(3))
    concrete_widths = _host_path(*widths)
    h_max = max(float(np.max(w)) for w in widths) if concrete_widths else nominal_h_max
    # With traced widths use their actual maximum for the window. The host
    # candidate filter is off; the nominal reach still controls image lists.
    reach = (h_max if concrete_widths else
             xp.maximum(xp.maximum(xp.max(widths[0]), xp.max(widths[1])), xp.max(widths[2])))
    padding = h_max if h_max is not None else max(float(grid.domain[a]) for a in range(3))
    target_bounds = [declared_bounds(s) for s in groups[current][1]]

    def candidate(shape):
        bounds = declared_bounds(shape)
        if not concrete_widths or bounds is None:
            return True
        for target in target_bounds:
            if target is None or not _host_path(*jax.tree.leaves((target, bounds))):
                return True
            if not all(math.isfinite(float(v)) for v in jax.tree.leaves((target, bounds))):
                return True
            overlaps = True
            for a, name in enumerate('xyz'):
                alo, ahi = float(target[0][a]), float(target[1][a])
                blo, bhi = float(bounds[0][a]), float(bounds[1][a])
                if name in grid.periodic_axes:
                    period = float(grid.domain[a])
                    overlaps &= math.ceil((alo - 3 * h_max - bhi) / period) <= math.floor((ahi + 3 * h_max - blo) / period)
                else:
                    overlaps &= alo - 3 * h_max <= bhi and blo - 3 * h_max <= ahi
            if overlaps:
                return True
        return False

    supported = [s for s in groups[current][1] if _has_sdf(s)]
    entries = getattr(groups, 'declared', tuple((s, eps) for eps, group in groups.values() for s in group))
    declaration_groups = getattr(groups, 'declaration_groups', None)
    intervals = []
    own_lines = []
    face = -distance
    keys: dict = {}
    source_count = 0
    for declared_index, (shape, eps) in enumerate(entries):
        if declaration_groups is not None:
            own = declaration_groups[declared_index] == current and _has_sdf(shape)
            source_index = source_count if own else -1
            source_count += int(own)
        else:
            source_index = next((i for i, s in enumerate(supported) if s is shape), -1)
        if not candidate(shape):
            continue
        key = medium_key(eps)
        code = keys.setdefault(key, len(keys) + 1)
        if not isinstance(shape, (Box, Sphere, Cylinder)):
            # Point-only shapes retain the existing mask/SDF lookup, with
            # an artificial 0/1 material to recover its membership mask.
            only = {current: (1., [shape])}
            point = _bulk_outside(
                xp, only, current, coords, distance, normal, length, 0., masks, grid,
                normal_source=xp.where(normal_source == source_index, 0, -1),
                normal_image=normal_image)
            intervals.append((xp.where(point != 0, -reach, reach), reach, eps, code))
            continue
        images = [_fill_image_range(grid, shape, a, padding=.5 * float(grid.cells(a)[0]) + padding)
                  if 'xyz'[a] in grid.periodic_axes else (0,) for a in range(3)]
        for image in product(*images):
            shifted = tuple(p + k * float(grid.domain[a]) if k else p
                            for a, (p, k) in enumerate(zip(coords, image)))
            lo, hi = _line_interval(xp, shape, shifted, normal, reach)
            if own_fill and source_index >= 0:
                # The own body's whole interval, the normal source included.
                own_lines.append((lo, hi, _planar_line(xp, shape, shifted, normal, reach, lo, hi)))
            excluded = normal_source == source_index
            for k, source in zip(image, normal_image):
                excluded = excluded & (k == source)
            # A33: use the source's own exit only when it also supplies the
            # union's signed value. Inspect the raw interval before exclusion
            # and empty-interval normalization; clipped/empty sources retain -s.
            source_end = excluded & (best_s == distance) & (hi >= lo) & (hi < reach) & (hi > -reach)
            face = xp.where(source_end, hi, face)
            lo = xp.where(excluded, reach, xp.clip(lo, -reach, reach))
            hi = xp.where(excluded, reach, xp.clip(hi, -reach, reach))
            hi = xp.maximum(lo, hi)
            intervals.append((lo, hi, eps, code))
    broadcast = xp.broadcast_arrays(distance, background, *coords, *normal)[0]
    ends = [xp.full_like(broadcast, -reach), xp.full_like(broadcast, reach)]
    ends += [xp.broadcast_to(end, broadcast.shape) for lo, hi, _, _ in intervals for end in (lo, hi)]
    ends = xp.sort(xp.stack(ends), axis=0)
    low, high = ends[:-1], ends[1:]
    mid = low + (high - low) / 2
    values = xp.broadcast_to(background, mid.shape)
    codes = xp.zeros(mid.shape, dtype=xp.int32)
    for lo, hi, eps, code in intervals:
        inside = (lo <= mid) & (mid < hi)
        merge_code = code
        if _host_path(eps, background):
            merge_code = xp.where(background == eps, 0, code)
        values = xp.where(inside, eps, values)
        codes = xp.where(inside, merge_code, codes)
    merged_lo, merged_hi, merged_eps = _merge_stream(xp, low, high, values, codes)
    own = _own_runs(xp, own_lines, length, reach, broadcast.shape) if own_lines else None
    return _LinePieces(merged_lo, merged_hi, merged_eps, face, own)


def _planar_line(xp, shape, points, normal, reach, lo, hi):
    """Whether a body's interval on this line lies between parallel planes.

    A Box's always does. A Cylinder's does only where both ends are its end
    caps, not its curved side. A Sphere's never does: a chord is not a
    laminate, so a thin Sphere keeps the one-face fill.
    """
    from rfx.geometry.csg import Box, Cylinder
    if isinstance(shape, Box):
        return True
    if isinstance(shape, Cylinder):
        cap_lo, cap_hi = _axial_interval(xp, shape, points, normal, reach)
        return (lo == cap_lo) & (hi == cap_hi)
    return False


def _own_runs(xp, lines, length, reach, shape):
    """Merge the own group's intervals into runs over the support.

    Intervals that overlap or abut form one run; an empty interval (lo >= hi)
    takes no part. A run is planar when every interval in it is planar
    (:func:`_planar_line`). A sample is thin where some planar run has both
    ends strictly inside (-length/2, length/2): the one-face fill
    ``0.5 - sdf/length`` then misses the run's far face. The fill counts every
    own run inside the support; the gaps are the rest of the support.
    """
    half = length / 2
    far = 2 * reach  # beyond the window, so an empty interval sorts last
    lo = xp.stack([xp.broadcast_to(a, shape) for a, _, _ in lines])
    hi = xp.stack([xp.broadcast_to(b, shape) for _, b, _ in lines])
    flat = xp.stack([xp.broadcast_to(xp.asarray(p), shape) for _, _, p in lines])
    valid = lo < hi
    lo, hi = xp.where(valid, lo, far), xp.where(valid, hi, far)
    order = xp.argsort(lo, axis=0)
    lo, hi, flat = (xp.take_along_axis(a, order, axis=0) for a in (lo, hi, flat))
    thin, inside, gaps = _own_stream(xp, lo, hi, flat, half, shape)
    return _OwnRuns(thin, inside / length, gaps)


def _depth_cdf(xp, depth, alpha, beta):
    """CDF of two centred uniforms, including uniform and point limits."""
    a, b = xp.maximum(alpha, beta), xp.minimum(alpha, beta)
    both = b > 0
    u = xp.clip(depth, -a - b, a + b)
    left = xp.where(both, (u + a + b) ** 2, 0) / xp.where(both, 8 * a * b, 1)
    right = 1 - xp.where(both, (a + b - u) ** 2, 0) / xp.where(both, 8 * a * b, 1)
    middle = xp.where(a > 0, u + a, 0) / xp.where(a > 0, 2 * a, 1)
    trapezoid = xp.where(u < -a + b, left, xp.where(u > a - b, right, middle))
    uniform = xp.clip(middle, 0, 1)
    return xp.where(a > 0, xp.where(both, trapezoid, uniform), (depth > 0).astype(u.dtype))


def _segment_content(xp, pieces, face, end, *, start=None, cdf=None, gaps=None):
    """Return (<eps>, <1/eps>, single, eps_min, eps_max) on a portion.

    CDF weights integrate a dual face; otherwise depth is uniform. ``gaps``
    replaces the portion [face, end] by a union of (start, end) portions,
    weighted by length. A zero-weight portion takes the right-hand medium
    at the source face. A single merged piece returns its original operand
    without a sum.
    """
    if gaps is None:
        start = face if start is None else xp.maximum(face, start)
        end = xp.maximum(start, end)
        lo = xp.maximum(pieces.lo, start)
        hi = xp.maximum(lo, xp.minimum(pieces.hi, end))
        weight = hi - lo if cdf is None else cdf(hi) - cdf(lo)
    else:
        weight = sum(xp.maximum(xp.minimum(pieces.hi, b) - xp.maximum(pieces.lo, a), 0)
                     for a, b in gaps)
    positive = weight > 0
    total = xp.sum(weight, axis=0)
    right = (pieces.lo <= face) & (pieces.hi > face)
    fallback = xp.sum(xp.where(right, pieces.eps, 0), axis=0)
    # Outside the window only discarded bulk nodes can ask for a portion.
    fallback = xp.where(xp.any(right, axis=0), fallback, pieces.eps[-1])
    single = xp.sum(positive, axis=0) <= 1
    operand = xp.where(total > 0, xp.sum(xp.where(positive, pieces.eps, 0), axis=0), fallback)
    mean = xp.where(total > 0, xp.sum(weight * pieces.eps, axis=0), 0) / xp.where(total > 0, total, 1)
    inv = xp.where(total > 0, xp.sum(weight / pieces.eps, axis=0), 0) / xp.where(total > 0, total, 1)
    mean = xp.where(single, operand, mean)
    inv = xp.where(single, 1 / operand, inv)
    minimum = xp.min(xp.where(positive, pieces.eps, xp.inf), axis=0)
    maximum = xp.max(xp.where(positive, pieces.eps, -xp.inf), axis=0)
    return _SegmentContent(mean, inv, single,
                           xp.where(total > 0, minimum, fallback),
                           xp.where(total > 0, maximum, fallback))


# rfx.geometry.smoothing imports this module part-way through its own body,
# after it defines these four names, so either import order resolves them.
from rfx.geometry.smoothing import _bulk_outside, _has_sdf, _host_path, medium_key  # noqa: E402

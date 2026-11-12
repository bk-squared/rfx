"""Host-side tensor-lattice parity for watertight triangle meshes.

Edges use exact signs (filtered float64, rational fallback) and canonical
endpoint order. Zero signs use the symbolic displacement (+epsilon, +epsilon²)
in (y, z), so adjacent triangles own a shared edge exactly once. The caller
applies the finite D1 nudge; x crossings are strictly beyond the sample.
"""
from fractions import Fraction

import numpy as np


def _orient(a, b, p):
    """Canonical 2-D determinant and its exact sign, broadcast over rows."""
    reverse = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    lo = np.where(reverse[:, None], b, a)
    hi = np.where(reverse[:, None], a, b)
    u, v = hi - lo, p - lo
    left, right = u[:, 0] * v[:, 1], u[:, 1] * v[:, 0]
    det = left - right
    sign = np.sign(det)
    uncertain = np.abs(det) <= 8 * np.finfo(float).eps * (np.abs(left) + np.abs(right))
    for i in np.flatnonzero(uncertain):
        ay, az, by, bz, py, pz = map(Fraction, (*lo[i], *hi[i], *p[i]))
        exact = (by - ay) * (pz - az) - (bz - az) * (py - ay)
        sign[i] = (exact > 0) - (exact < 0)
        det[i] = float(exact)
    factor = np.where(reverse, -1, 1)
    return det * factor, sign * factor


def _owned(a, b, sign, orientation):
    # d/dy of orient is -dz; if zero, the next symbolic term is +dy.
    dy, dz = (b - a).T
    tie = np.where(dz != 0, -np.sign(dz), np.sign(dy)) * orientation
    return (sign * orientation > 0) | ((sign == 0) & (tie > 0))


def _outward(lo, hi):
    return np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)


def _subtract(a, b):
    return _outward(a[0] - b[1], a[1] - b[0])


def _multiply(a, b):
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return _outward(np.minimum.reduce(products), np.maximum.reduce(products))


def _determinant(a, b):
    return _subtract(_multiply(a[0], b[1]), _multiply(a[1], b[0]))


def _crossing_ranks(triangles, points, x):
    """Number of sorted samples strictly below each exact plane crossing.

    Outward-rounded intervals certify insertion ranks without rounding a
    crossing onto a sample. Only intervals spanning samples need Fraction
    arithmetic. Storage remains per crossing, never samples times faces.
    """
    side = "left"  # x < x_cross; M2 changes this to include equality.
    a, b, c = triangles.transpose(1, 0, 2)
    flat = (a[:, 0] == b[:, 0]) & (a[:, 0] == c[:, 0])
    ranks = np.searchsorted(x, a[:, 0], side=side)
    ids = np.flatnonzero(~flat)
    a, b, c, p = a[ids], b[ids], c[ids], points[ids]
    u = [_subtract((b[:, k],) * 2, (a[:, k],) * 2) for k in range(3)]
    v = [_subtract((c[:, k],) * 2, (a[:, k],) * 2) for k in range(3)]
    w = [_subtract((p[:, k],) * 2, (a[:, k + 1],) * 2) for k in range(2)]
    area = _determinant(u[1:], v[1:])
    ub = _multiply(_determinant(w, v[1:]), u[0])
    vc = _multiply(_determinant(u[1:], w), v[0])
    numerator = _outward(ub[0] + vc[0], ub[1] + vc[1])
    singular = (area[0] <= 0) & (area[1] >= 0)
    denominator = tuple(np.where(singular, 1., bound) for bound in area)
    quotients = tuple(n / d for n in numerator for d in denominator)
    qlo, qhi = _outward(np.minimum.reduce(quotients), np.maximum.reduce(quotients))
    lower, upper = _outward(a[:, 0] + qlo, a[:, 0] + qhi)
    uncertain = singular | ~np.isfinite(lower) | ~np.isfinite(upper)
    lower, upper = np.where(uncertain, -np.inf, lower), np.where(uncertain, np.inf, upper)
    left = np.searchsorted(x, lower, side=side)
    right = np.searchsorted(x, upper, side=side)
    for i in np.flatnonzero(left != right):
        aa, bb, cc = [list(map(Fraction, vertex)) for vertex in (a[i], b[i], c[i])]
        yy, zz = map(Fraction, p[i])
        uu = [bb[k] - aa[k] for k in range(3)]
        vv = [cc[k] - aa[k] for k in range(3)]
        wy, wz = yy - aa[1], zz - aa[2]
        exact = aa[0] + ((wy * vv[2] - wz * vv[1]) * uu[0]
                         + (uu[1] * wz - uu[2] * wy) * vv[0]) / (uu[1] * vv[2] - uu[2] * vv[1])
        lo, hi = int(left[i]), int(right[i])
        while lo < hi:
            mid = (lo + hi) // 2
            sample = Fraction(x[mid])
            if sample < exact or (side == "right" and sample == exact):
                lo = mid + 1
            else:
                hi = mid
        left[i] = lo
    ranks[ids] = left
    return ranks


def scanline_mask(triangles, x, y, z):
    """Return occupancy on already nudged coordinates, including unsorted axes.

    Candidate storage scales with projected triangle footprints, not volume
    samples times faces. Sorting vertices also fixes interpolation arithmetic
    under face, vertex, winding and cyclic-index permutations.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = np.take_along_axis(triangles, order[:, :, None], axis=1)
    a, b, c = triangles.transpose(1, 0, 2)
    _, orientation = _orient(a[:, 1:], b[:, 1:], c[:, 1:])
    keep = orientation != 0
    triangles, orientation = triangles[keep], orientation[keep]
    sy, sz = np.argsort(y), np.argsort(z)
    ys, zs = y[sy], z[sz]
    lower, upper = triangles[:, :, 1:].min(axis=1), triangles[:, :, 1:].max(axis=1)
    yl, yh = np.searchsorted(ys, lower[:, 0]), np.searchsorted(ys, upper[:, 0], side="right")
    zl, zh = np.searchsorted(zs, lower[:, 1]), np.searchsorted(zs, upper[:, 1], side="right")
    counts = (yh - yl) * (zh - zl)
    ids = np.repeat(np.arange(len(triangles)), counts)
    offsets = np.arange(len(ids)) - np.repeat(np.cumsum(counts) - counts, counts)
    yi = yl[ids] + offsets // (zh - zl)[ids]
    zi = zl[ids] + offsets % (zh - zl)[ids]
    p = np.column_stack((ys[yi], zs[zi]))
    a, b, c = triangles[ids].transpose(1, 0, 2)
    _, sab = _orient(a[:, 1:], b[:, 1:], p)
    _, sbc = _orient(b[:, 1:], c[:, 1:], p)
    _, sca = _orient(c[:, 1:], a[:, 1:], p)
    direction = orientation[ids]
    hit = (_owned(a[:, 1:], b[:, 1:], sab, direction)
           & _owned(b[:, 1:], c[:, 1:], sbc, direction)
           & _owned(c[:, 1:], a[:, 1:], sca, direction))
    sx = np.argsort(x)
    ranks = _crossing_ranks(triangles[ids[hit]], p[hit], x[sx])
    lines = sy[yi[hit]] * len(z) + sz[zi[hit]]
    order = np.lexsort((ranks, lines))
    lines, ranks = lines[order], ranks[order]
    unique, starts, sizes = np.unique(lines, return_index=True, return_counts=True)
    out = np.zeros((len(x), len(y) * len(z)), dtype=bool)
    for line, start, size in zip(unique, starts, sizes):
        crossings = ranks[start:start + size]
        beyond = size - np.searchsorted(crossings, np.arange(len(x)), side="right")
        out[sx, line] = beyond % 2 == 1
    return out.reshape(len(x), len(y), len(z))

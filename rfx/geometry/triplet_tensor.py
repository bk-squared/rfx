"""Werner--Bauer--Cary Sec. 7, scalar materials (Eqs. 15--20).

Triplet signs are ordered +++, ++-, +-+, +--, -++, -+-, --+, ---.
Fractions analytically clip edges and dual faces against one local plane.
Concrete inputs accumulate in host float64; tracers use the field dtype.
"""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

SIGNS = tuple(product((0, 1), repeat=3))  # 0 positive, 1 negative
PAIRS = ((0, 1), (0, 2), (1, 2))
MATRIX_BATCH_SIZE = 2**20


def _linalg(xp, operation, matrix):
    """Bound traced solver batches without changing individual matrix solves."""
    solve = getattr(xp.linalg, operation)
    if xp is np:
        return solve(matrix)
    flat = matrix.reshape((-1, 3, 3))
    pieces = [solve(flat[start:start + MATRIX_BATCH_SIZE])
              for start in range(0, flat.shape[0], MATRIX_BATCH_SIZE)]
    if operation == "eigh":
        w, v = zip(*pieces)
        return (jnp.concatenate(w).reshape(matrix.shape[:-1]),
                jnp.concatenate(v).reshape(matrix.shape))
    tail = (3,) if operation == "eigvalsh" else (3, 3)
    return jnp.concatenate(pieces).reshape(matrix.shape[:-2] + tail)


def fractions(xp, grid, shapes, nodes, *, signed_distance=None, normal=None,
              cell_lengths=None):
    """Planar length/area fractions indexed [axis][sign] (decision record 6).

    The optional plane reuses the union SDF and nearest normal from the node
    builder. Direct callers construct that same plane, never sample the shape.
    """
    if signed_distance is None:
        from rfx.geometry.node_tensor import _union

        fill_length = None if cell_lengths is None else (
            (sum(cell_lengths[0]) * sum(cell_lengths[1]) * sum(cell_lengths[2]) / 8)
            ** (1.0 / 3.0))
        _, normal, signed_distance = _union(
            xp, grid, shapes, nodes, distance=True, fill_length=fill_length)
    dtype = np.float64 if xp is np else getattr(signed_distance, 'dtype', jnp.float32)
    from rfx.geometry.node_tensor import uniform_cell
    h = uniform_cell(grid) if cell_lengths is None else None
    s, *ns = xp.broadcast_arrays(xp.asarray(signed_distance, dtype=dtype),
                                 *(xp.asarray(n, dtype=dtype) for n in normal))
    corners = xp.asarray([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]], dtype=dtype)
    following = xp.roll(corners, -1, axis=0)
    lengths, areas = [], []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        ls, ars = [], []
        for sign in (1, -1):
            if cell_lengths is not None:
                h = cell_lengths[axis][1 if sign == 1 else 0]
            d1 = s + sign * h * ns[axis]
            crossing = (s < 0) != (d1 < 0)
            # Double-where keeps parallel and unused intersections finite in AD.
            t = xp.where(crossing, s, 0) / xp.where(crossing, s - d1, 1)
            ls.append(xp.where(crossing, xp.where(s < 0, t, 1 - t),
                               (s < 0).astype(dtype)))
            mid = s + sign * h * ns[axis] / 2
            if cell_lengths is None:
                d = mid[..., None] + h * (
                    ns[other[0]][..., None] * corners[:, 0]
                    + ns[other[1]][..., None] * corners[:, 1])
            else:
                b, c = other
                bm, bp = cell_lengths[b]
                cm, cp = cell_lengths[c]
                wb, wc = (bm + bp) / 2, (cm + cp) / 2
                mid = mid + ns[b] * (bp - bm) / 4 + ns[c] * (cp - cm) / 4
                d = mid[..., None] + wb[..., None] * (
                    ns[b][..., None] * corners[:, 0]
                    + ns[c][..., None] * (wc / wb)[..., None] * corners[:, 1])
            dn = xp.roll(d, -1, axis=-1)
            inside = d < 0
            cross = inside != (dn < 0)
            t = xp.where(cross, d, 0) / xp.where(cross, d - dn, 1)
            intersections = corners + t[..., None] * (following - corners)
            # One Sutherland-Hodgman half-plane pass: emit each inside corner,
            # then its outgoing intersection. Compact into at most 5 vertices.
            candidates = xp.stack((xp.broadcast_to(corners, intersections.shape),
                                   intersections), axis=-2).reshape(s.shape + (8, 2))
            valid = xp.stack((inside, cross), axis=-1).reshape(s.shape + (8,))
            slots = xp.cumsum(valid, axis=-1) - 1
            select = valid[..., :, None] & (slots[..., :, None] == xp.arange(5))
            polygon = xp.sum(xp.where(select[..., None], candidates[..., :, None, :], 0),
                             axis=-3)
            count = xp.sum(valid, axis=-1)
            successor = xp.where(xp.arange(5) + 1 < count[..., None], xp.arange(5) + 1, 0)
            q = xp.take_along_axis(polygon, successor[..., None], axis=-2)
            terms = polygon[..., 0] * q[..., 1] - polygon[..., 1] * q[..., 0]
            area = xp.abs(xp.sum(xp.where(xp.arange(5) < count[..., None], terms, 0),
                                 axis=-1)) / 2
            # Coordinates are normalized to the square: its area is exactly 1.
            ars.append(area)
        lengths.append(tuple(ls))
        areas.append(tuple(ars))
    return tuple(lengths), tuple(areas)


def accurate_tensor(xp, normal, lengths, areas, eps1, eps2):
    """Eqs. 15--19, with vector/matrix dimensions last; also return singular mask."""
    eye = xp.eye(3, dtype=lengths.dtype)
    nn = normal[..., :, None] * normal[..., None, :]
    e1, e2 = xp.asarray(eps1, dtype=lengths.dtype)[..., None, None], xp.asarray(eps2, dtype=lengths.dtype)[..., None, None]
    g1, g2 = eye + (1 / e1 - 1) * nn, eye + (1 / e2 - 1) * nn
    gamma = lengths[..., :, None] * g1 + (1 - lengths[..., :, None]) * g2
    pi = areas[..., :, None] * e1 * g1 + (1 - areas[..., :, None]) * e2 * g2
    singular = xp.linalg.det(pi) == 0
    safe_pi = xp.where(singular[..., None, None], eye, pi)
    return gamma @ _linalg(xp, "inv", safe_pi), singular


def _clip_primal(xp, tensor, lo, hi):
    w, v = _linalg(xp, "eigh", tensor)
    w = xp.clip(w, lo[..., None], hi[..., None])
    return xp.sum(v[..., :, None, :] * v[..., None, :, :]
                  * w[..., None, None, :], axis=-1)


@jax.custom_jvp
def _spectral_clip(tensor, lo, hi):
    return _clip_primal(jnp, tensor, lo, hi)


@_spectral_clip.defjvp
def _spectral_clip_jvp(primals, tangents):
    tensor, lo, hi = primals
    dt, dlo, dhi = tangents
    result = _spectral_clip(tensor, lo, hi)
    w, v = _linalg(jnp, "eigh", tensor)
    f = jnp.clip(w, lo[..., None], hi[..., None])
    inside = (w >= lo[..., None]) & (w <= hi[..., None])
    gap = w[..., :, None] - w[..., None, :]
    # One relative machine epsilon resolves ties at the eigensolve precision.
    tol = jnp.finfo(w.dtype).eps * jnp.max(jnp.abs(w), axis=-1)[..., None, None]
    separated = jnp.abs(gap) > tol
    divided = jnp.where(separated, f[..., :, None] - f[..., None, :], 0) / jnp.where(separated, gap, 1)
    L = jnp.where(separated, divided, inside[..., :, None].astype(w.dtype))
    vt = jnp.swapaxes(v, -1, -2)
    tangent = v @ (L * (vt @ dt @ v)) @ vt
    bound_tangent = jnp.where(w < lo[..., None], dlo[..., None],
                             jnp.where(w > hi[..., None], dhi[..., None], 0))
    tangent = tangent + jnp.sum(v[..., :, None, :] * v[..., None, :, :]
                               * bound_tangent[..., None, None, :], axis=-1)
    return result, tangent



def guarded_tensor(xp, normal, lengths, areas, eps1, eps2, fallback):
    """Bound Eq. 20 to the material pair; only singular Pi uses the fallback.

    The returned guard mask counts singular fallbacks, not spectral clips.
    In-range and bulk tensors retain their bits (decision record 3).
    """
    acc, singular = accurate_tensor(xp, normal, lengths, areas, eps1, eps2)
    tensor = (acc + xp.swapaxes(acc, -1, -2)) * .5
    bulk_any = xp.zeros_like(singular)
    for fraction, eps in ((0, eps2), (1, eps1)):
        bulk = xp.all(lengths == fraction, axis=-1) & xp.all(areas == fraction, axis=-1)
        tensor = xp.where(bulk[..., None, None],
                          xp.eye(3, dtype=tensor.dtype) / xp.asarray(eps, dtype=tensor.dtype)[..., None, None], tensor)
        bulk_any = bulk_any | bulk
    guard = singular & ~bulk_any
    lo = xp.asarray(1 / xp.maximum(eps1, eps2), dtype=tensor.dtype)
    hi = xp.asarray(1 / xp.minimum(eps1, eps2), dtype=tensor.dtype)
    w = _linalg(xp, "eigvalsh", tensor)
    clip = ((w[..., 0] < lo) | (w[..., -1] > hi)) & ~bulk_any & ~singular
    # Double-where: no repeated bulk/in-range eigenvalues enter the eigh
    # derivative. The mask's eigvalsh is used only for boolean selection.
    dummy = xp.diag(xp.asarray([1., 2., 3.], dtype=tensor.dtype))
    safe_tensor = xp.where(clip[..., None, None], tensor, dummy)
    clipped = (_clip_primal(xp, safe_tensor, lo, hi) if xp is np else
               _spectral_clip(safe_tensor, lo, hi))
    tensor = xp.where(clip[..., None, None], clipped, tensor)
    tensor = xp.where(guard[..., None, None], fallback, tensor)
    return tensor, guard


def matrices(xp, entries):
    xx, yy, zz, xy, xz, yz = entries
    return xp.stack([xp.stack(row, axis=-1) for row in
                     ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))], axis=-2)


def build_triplets(xp, grid, shapes, nodes, normal, eps, outside, fallback,
                   *, signed_distance=None, cell_lengths=None):
    lengths, areas = fractions(xp, grid, shapes, nodes,
                               signed_distance=signed_distance, normal=normal,
                               cell_lengths=cell_lengths)
    ns = xp.stack(xp.broadcast_arrays(*normal), axis=-1)
    tensors, guards, touched = [], [], []
    for signs in SIGNS:
        ls = xp.stack([lengths[a][s] for a, s in enumerate(signs)], axis=-1)
        ars = xp.stack([areas[a][s] for a, s in enumerate(signs)], axis=-1)
        tensor, guard = guarded_tensor(xp, ns, ls, ars, eps, outside, fallback)
        tensors.append(tensor)
        guards.append(guard)
        touched.append(xp.any(ls > 0, axis=-1) | xp.any(ars > 0, axis=-1))
    return xp.stack(tensors), xp.stack(guards), xp.stack(touched)


def triplet_weights(xp, cell_lengths):
    """Octant volume divided by the incident edge's dual volume."""
    weights = []
    for a in range(3):
        b, c = [axis for axis in range(3) if axis != a]
        weights.append(tuple(
            cell_lengths[b][1 - signs[b]] * cell_lengths[c][1 - signs[c]]
            / (2 * sum(cell_lengths[b]) * sum(cell_lengths[c])) for signs in SIGNS))
    return tuple(weights)


def coefficients(xp, triplets, weights=None):
    """Eq. 10: node-end diagonals (four triplets) and pair couplings (two)."""
    if weights is not None:
        diagonal = tuple(tuple(sum(weights[a][t] * triplets[t, ..., a, a]
                                   for t, signs in enumerate(SIGNS) if signs[a] == s)
                               for s in (0, 1)) for a in range(3))
        def cross_for(reverse):
            return tuple(xp.stack([xp.stack([
                sum(weights[b if reverse else a][t] * triplets[t, ..., a, b]
                    for t, signs in enumerate(SIGNS) if signs[a] == sa and signs[b] == sb)
                for sb in (0, 1)]) for sa in (0, 1)]) for a, b in PAIRS)
        return diagonal, cross_for(False), cross_for(True)
    diagonal = tuple(tuple(sum(triplets[t, ..., a, a]
                              for t, signs in enumerate(SIGNS) if signs[a] == s) / 8
                           for s in (0, 1)) for a in range(3))
    cross = tuple(xp.stack([xp.stack([
        sum(triplets[t, ..., a, b] for t, signs in enumerate(SIGNS)
            if signs[a] == sa and signs[b] == sb) / 8
        for sb in (0, 1)]) for sa in (0, 1)]) for a, b in PAIRS)
    return diagonal, cross

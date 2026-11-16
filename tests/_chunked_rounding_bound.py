"""A35--A36 running float32 error analysis, evaluated with float64 NumPy.

The capture function alone calls the product, in a separate eager jax.jvp.
Everything below ``_TERMS`` is an independent NumPy recomputation. No compiled
build being compared is wrapped, traced for recording, or used for capture.

E2 follows JAX 0.10.2 jax/_src/lax/lax.py:4885 (_add_jvp), 4941
(_sub_jvp), 5009 (mul), 5096 (div), 4823--4861 (integer_pow lowering),
6025 (dot_general), and 8397--8420 (reduce_sum); and
jax/_src/interpreters/ad.py:1065,1101 (linear_jvp, standard_jvp).
The custom spectral JVP follows S2, not E2. The installed-rule identity is
checked by the caller's recorded environment, not inferred from its outputs.
"""
from dataclasses import dataclass
from itertools import product
import sys

import numpy as np


def capture_operands(build, radius):
    """Return M1' operands from one eager build, with symbolic-zero metadata.

    Temporary Python hooks record guarded_tensor's arguments and node assembly
    inputs. They are removed before returning. jax.jvp runs without an outer
    jit; its returned value is the capture tree, not the assembled result.
    """
    import jax
    import jax.numpy as jnp
    from jax.interpreters import ad
    from rfx.geometry import node_tensor as node
    from rfx.geometry import triplet_tensor as triplet

    original = triplet.guarded_tensor
    previous_trace = sys.gettrace()
    operands, active, assembly = [], [], {}

    def symbolic(value):
        return hasattr(value, 'tangent') and not isinstance(value.tangent, ad.Zero)

    def record(xp, normal, lengths, areas, eps1, eps2, fallback):
        row = dict(normal=normal, lengths=lengths, areas=areas, eps1=eps1,
                   fallback=fallback)
        if hasattr(eps2, 'edge_eps'):
            row.update({name: getattr(eps2, name) for name in eps2._fields})
        else:
            row['operand'] = eps2
        operands.append(row)
        active.append({name: symbolic(value) for name, value in row.items()})
        return original(xp, normal, lengths, areas, eps1, eps2, fallback)

    def trace(frame, event, arg):
        if frame.f_code is node.compute_node_tensor_eps.__code__:
            if event == 'return':
                for key in ('midpoint', 'mid_bulk', 'pad', 'periodic'):
                    assembly[key] = frame.f_locals[key]
                assembly['weights'] = None  # the bar's grid is uniform
            return trace
        return None

    def eager(r):
        result = build(r)
        assembly['other_leaves'] = tuple(jax.tree.leaves(result))
        return tuple(operands), assembly

    try:
        triplet.guarded_tensor = record
        sys.settrace(trace)
        primal, tangent = jax.jvp(eager, (radius,), (jnp.ones_like(radius),))
    finally:
        triplet.guarded_tensor = original
        sys.settrace(previous_trace)
    assert len(operands) == 8, 'This witness covers the bar\'s one material group.'
    primal, tangent = jax.tree.map(np.asarray, (primal, tangent))
    return dict(primal=primal, tangent=tangent, active=active,
                method='separate eager jax.jvp; no outer jit')


_TERMS = ('E1', 'E4', 'S1_backward', 'S2_backward', 'S2_nonorthogonality',
          'S2_bound_sensitivity', 'M1_near', 'J')
EPS = 2.0**-23
UNIT = 2.0**-24
P_EIGH = 27
SIGNS = tuple(product((0, 1), repeat=3))
PAIRS = ((0, 1), (0, 2), (1, 2))


class B:
    """A float64 value and additive first-order error terms (E1/E3/E4/E5)."""
    def __init__(self, value, error=None):
        self.v = np.asarray(value, dtype=np.float64)
        self.e = (np.zeros((len(_TERMS),) + self.v.shape) if error is None
                  else np.asarray(error, dtype=np.float64))
        assert self.e.shape == (len(_TERMS),) + self.v.shape

    @property
    def delta(self):
        return self.e.sum(axis=0)

    def errors(self, shape):
        return np.broadcast_to(self.e.reshape((len(_TERMS),)
                               + (1,) * (len(shape) - self.v.ndim) + self.v.shape),
                               (len(_TERMS),) + shape)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return B(self.v[key], self.e[(slice(None),) + key])

    def swap(self):
        return B(self.v.swapaxes(-1, -2), self.e.swapaxes(-1, -2))

    def rounded(self, term=0):
        e = self.e.copy()
        e[term] += EPS * np.abs(self.v)
        return B(self.v, e)

    def __add__(self, other):
        b = as_b(other)
        z = self.v + b.v
        return B(z, self.errors(z.shape) + b.errors(z.shape)).rounded()

    __radd__ = __add__

    def __neg__(self):
        return B(-self.v, self.e).rounded()

    def __sub__(self, other):
        b = as_b(other)
        z = self.v - b.v
        return B(z, self.errors(z.shape) + b.errors(z.shape)).rounded()

    def __mul__(self, other):
        b = as_b(other)
        z = self.v * b.v
        e = self.errors(z.shape) * np.abs(b.v) + b.errors(z.shape) * np.abs(self.v)
        exact_half = ((self.v.ndim == 0 and self.v == .5 and not np.any(self.e))
                      or (b.v.ndim == 0 and b.v == .5 and not np.any(b.e)))
        value = B(z, e)
        return value if exact_half else value.rounded()

    __rmul__ = __mul__

    def __truediv__(self, other):
        b = as_b(other)
        z = self.v / b.v
        e = (self.errors(z.shape) / np.abs(b.v)
             + b.errors(z.shape) * np.abs(self.v / b.v**2))
        value = B(z, e)
        return value if b.v.ndim == 0 and b.v == 8 and not np.any(b.e) else value.rounded()

    def __matmul__(self, other):
        b = as_b(other)
        z = self.v @ b.v
        e = self.e @ np.abs(b.v) + np.abs(self.v) @ b.e
        e[1] += self.v.shape[-1] * EPS * (np.abs(self.v) @ np.abs(b.v))
        return B(z, e)

    def sum_axis(self, axis):
        axis %= self.v.ndim
        e = self.e.sum(axis=axis + 1)
        e[1] += self.v.shape[axis] * EPS * np.abs(self.v).sum(axis=axis)
        return B(self.v.sum(axis=axis), e)


def as_b(x):
    return x if isinstance(x, B) else B(x)


def bwhere(mask, a, b):
    a, b = as_b(a), as_b(b)
    z = np.where(mask, a.v, b.v)
    return B(z, np.where(mask, a.errors(z.shape), b.errors(z.shape)))


def bstack(values, axis=0):
    values = list(values)
    axis %= values[0].v.ndim + 1
    return B(np.stack([v.v for v in values], axis=axis),
             np.stack([v.e for v in values], axis=axis + 1))


class D:
    """A primal B and its radius tangent B, or a symbolic-zero tangent."""
    def __init__(self, value, tangent=None):
        self.p, self.d = as_b(value), None if tangent is None else as_b(tangent)

    def __getitem__(self, key):
        return D(self.p[key], None if self.d is None else self.d[key])

    def swap(self):
        return D(self.p.swap(), None if self.d is None else self.d.swap())

    def __add__(self, other):
        b = as_d(other)
        return D(self.p + b.p, tangent_add(self.d, b.d))

    __radd__ = __add__

    def __sub__(self, other):
        b = as_d(other)
        dot = (None if b.d is None else -b.d) if self.d is None else (
            self.d if b.d is None else self.d - b.d)
        return D(self.p - b.p, dot)

    def __rsub__(self, other):
        return as_d(other) - self

    def __mul__(self, other):
        b = as_d(other)
        return D(self.p * b.p, tangent_add(None if self.d is None else self.d * b.p,
                                          None if b.d is None else self.p * b.d))

    __rmul__ = __mul__

    def __truediv__(self, other):
        b = as_d(other)
        # lax.py:5096: g/y; (-g)*x*integer_pow(y,-2), then standard_jvp add.
        lhs = None if self.d is None else self.d / b.p
        rhs = None if b.d is None else (-b.d) * self.p * (B(1) / (b.p * b.p))
        return D(self.p / b.p, tangent_add(lhs, rhs))

    def __rtruediv__(self, other):
        return as_d(other) / self

    def __matmul__(self, other):
        b = as_d(other)
        return D(self.p @ b.p, tangent_add(None if self.d is None else self.d @ b.p,
                                          None if b.d is None else self.p @ b.d))

    def sum_axis(self, axis):
        return D(self.p.sum_axis(axis), None if self.d is None else self.d.sum_axis(axis))


def as_d(x):
    return x if isinstance(x, D) else D(x)


def tangent_add(a, b):
    return b if a is None else a if b is None else a + b


def tangent(value):
    return B(np.zeros_like(value.p.v)) if value.d is None else value.d


def dwhere(mask, a, b):
    a, b = as_d(a), as_d(b)
    return D(bwhere(mask, a.p, b.p), None if a.d is b.d is None
             else bwhere(mask, tangent(a), tangent(b)))


def dstack(values, axis=0):
    values = list(values)
    return D(bstack([v.p for v in values], axis), None if all(v.d is None for v in values)
             else bstack([tangent(v) for v in values], axis))


def extremum(a, b, maximum):
    """M1-only choices: captured branch, largest tangent magnitude on ties."""
    a, b = as_d(a), as_d(b)
    choose = a.p.v >= b.p.v if maximum else a.p.v <= b.p.v
    value = dwhere(choose, a, b)
    if value.d is not None:
        tied = a.p.v == b.p.v
        tied_dot = bwhere(np.abs(tangent(a).v) >= np.abs(tangent(b).v), tangent(a), tangent(b))
        value.d = bwhere(tied, tied_dot, value.d)
    return value


def axis_extremum(x, maximum):
    result = x[..., 0]
    for i in range(1, x.p.v.shape[-1]):
        result = extremum(result, x[..., i], maximum)
    return result


def _lu_abs_product(x):
    """Independent 3x3 partial-pivot LU; S1 uses |L| |U| as written."""
    u = x.copy()
    l = np.broadcast_to(np.eye(3), x.shape).copy()
    flat_u, flat_l = u.reshape(-1, 3, 3), l.reshape(-1, 3, 3)
    for a, b in zip(flat_u, flat_l):
        for k in range(2):
            pivot = k + np.argmax(np.abs(a[k:, k]))
            a[[k, pivot]] = a[[pivot, k]]
            b[[k, pivot], :k] = b[[pivot, k], :k]
            for j in range(k + 1, 3):
                b[j, k] = a[j, k] / a[k, k]
                a[j, k:] -= b[j, k] * a[k, k:]
    return np.abs(l) @ np.abs(u)


def inverse(x):
    """S1 primal solve and JAX tangent solve; no product solver is called."""
    y = np.linalg.inv(x.p.v)
    ay, ax = np.abs(y), np.abs(x.p.v)
    lu = _lu_abs_product(x.p.v)
    gamma9 = 9 * UNIT / (1 - 9 * UNIT)
    e = ay @ x.p.e @ ay
    e[2] += 2 * gamma9 * (ay @ lu @ ay)
    yp = B(y, e)
    yd = None
    if x.d is not None:
        w = x.d @ yp
        dy = -(y @ w.v)
        de = ay @ w.e + ay @ x.p.e @ np.abs(dy)
        de[2] += 2 * gamma9 * (ay @ lu @ np.abs(dy))
        yd = B(dy, de)
    kappa = np.max(np.sum(ay @ ax, axis=-1), axis=-1)
    return D(yp, yd), kappa


def first_divided(a, b, lo, hi):
    ca, cb = np.clip(a, lo, hi), np.clip(b, lo, hi)
    distinct = a != b
    return np.where(distinct, (ca - cb) / np.where(distinct, a - b, 1),
                    ((a >= lo) & (a <= hi)).astype(float))


def second_divided(a, b, c, lo, hi):
    ab, bc = first_divided(a, b, lo, hi), first_divided(b, c, lo, hi)
    ac = a != c
    unequal = a != b
    repeated = np.where(unequal, (((a >= lo) & (a <= hi)).astype(float) - ab)
                        / np.where(unequal, a - b, 1), 0)
    return np.where(ac, (ab - bc) / np.where(ac, a - c, 1), repeated)


def eigen_data(tensor):
    w, v = np.linalg.eigh(tensor.p.v)
    av, avt = np.abs(v), np.abs(v.swapaxes(-1, -2))
    pe = avt @ tensor.p.e @ av
    pe[3] += (2 * P_EIGH * UNIT * np.max(np.abs(w), axis=-1))[..., None, None]
    return w, v, pe


def spectral_clip(tensor, lo, hi, w, v, pe, *, outcome=None, cluster=None):
    """S2 including read-7 nonorthogonality and bound-sum sensitivity."""
    av, avt, vt = np.abs(v), np.abs(v.swapaxes(-1, -2)), v.swapaxes(-1, -2)
    low, high = lo.p.v[..., None], hi.p.v[..., None]
    c = np.clip(w, low, high)
    below, above = w < low, w > high
    inside = ~below & ~above
    if outcome is not None:
        state, forced_separated = outcome
        below, above = state < 0, state > 0
        inside = state == 0
        c = np.where(below, low, np.where(above, high, w))
    gap = w[..., :, None] - w[..., None, :]
    tol = EPS * np.max(np.abs(w), axis=-1)[..., None, None]
    separated = np.abs(gap) > tol
    if outcome is not None:
        separated = forced_separated
    divided = (c[..., :, None] - c[..., None, :]) / np.where(separated, gap, 1)
    L = np.where(separated, divided, inside[..., :, None])
    distinct_eigenvalues = gap != 0
    analytic_L = np.where(distinct_eigenvalues,
                          (c[..., :, None] - c[..., None, :]) / np.where(distinct_eigenvalues, gap, 1),
                          inside[..., :, None].astype(float))
    def outer_contract(value):
        return av @ value @ avt if cluster is None else cluster_contract(cluster, value)
    # Reconstruction arithmetic; sensitivity and orthogonality are added once.
    bv, bw = B(v), B(w)
    clipped = bwhere(below, lo.p[..., None], bwhere(above, hi.p[..., None], bw))
    reconstructed = (bv[..., :, None, :] * bv[..., None, :, :]
                     * clipped[..., None, None, :]).sum_axis(-1)
    e = reconstructed.e + outer_contract(np.abs(analytic_L) * pe)
    e[4] += (4 * P_EIGH * UNIT * np.max(np.abs(c), axis=-1))[..., None, None]
    # Clipped bound uncertainty already propagates through reconstructed.e.
    result = B(reconstructed.v, e)
    dt = tangent(tensor)
    # In the custom JVP, V,w,lo,hi start without propagated primal errors.
    # E1/E3/E4 roundings and tangent-input errors propagate through its code.
    bf = bwhere(below, B(lo.p.v)[..., None], bwhere(above, B(hi.p.v)[..., None], bw))
    bgap = bw[..., :, None] - bw[..., None, :]
    bd = bwhere(separated, bf[..., :, None] - bf[..., None, :], B(0)) / bwhere(separated, bgap, B(1))
    bl = bwhere(separated, bd, B(inside[..., :, None].astype(float)))
    h = B(vt) @ dt @ B(v)
    main = B(v) @ (bl * h) @ B(vt)
    b = bwhere(below, tangent(lo)[..., None], bwhere(above, tangent(hi)[..., None], B(0)))
    bound_sum = (B(v)[..., :, None, :] * B(v)[..., None, :, :]
                 * b[..., None, None, :]).sum_axis(-1)
    derivative = main + bound_sum
    h_magnitude = np.abs(h.v) if cluster is None else cluster_h(cluster, dt.v, v)
    qe = np.zeros_like(pe)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ac = w[..., i] != w[..., j]
                ab = w[..., i] != w[..., k]
                repeated = np.where(ab, (inside[..., i].astype(float) - analytic_L[..., i, k])
                                    / np.where(ab, w[..., i] - w[..., k], 1), 0)
                second = np.abs(np.where(ac, (analytic_L[..., i, k] - analytic_L[..., k, j])
                                        / np.where(ac, w[..., i] - w[..., j], 1), repeated))
                qe[..., i, j] += second * (pe[..., i, k] * h_magnitude[..., k, j]
                                          + h_magnitude[..., i, k] * pe[..., k, j])
    derivative.e += outer_contract(qe)
    norm_dt = np.linalg.norm(dt.v, axis=(-2, -1))
    norm_lh = np.linalg.norm(analytic_L * h_magnitude, axis=(-2, -1))
    magnitude = np.max(np.abs(analytic_L), axis=(-2, -1)) * norm_dt + norm_lh + np.max(np.abs(b.v), axis=-1)
    derivative.e[4] += (4 * P_EIGH * UNIT * magnitude)[..., None, None]
    distinct = gap != 0
    bdifference = (b.v[..., :, None] - b.v[..., None, :]) / np.where(distinct, gap, 1)
    sensitivity = np.where(distinct, bdifference, 0)
    derivative.e += outer_contract(np.abs(sensitivity) * pe)
    return D(result, derivative), L, inside, gap, tol


def _input(primal, dot, active, near):
    p = B(primal)
    if near:
        p.e[6] = EPS * np.abs(p.v)
    d = B(dot) if active else None
    if near and d is not None:
        d.e[6] = EPS * np.abs(d.v)
    return D(p, d)


def recompute_triplet(primal, dots, active, near=False):
    """Independent operation sequence of accurate_tensor and guarded_tensor."""
    x = {name: _input(value, dots[name], active[name], near)
         for name, value in primal.items() if name not in ('edge_single', 'face_single')}
    normal, lengths, areas, e1 = (x[k] for k in ('normal', 'lengths', 'areas', 'eps1'))
    e2 = x['operand']
    eye = D(np.eye(3))
    nn = normal[..., :, None] * normal[..., None, :]
    a1, a2 = e1[..., None, None], e2[..., None, None]
    g1, g2 = eye + (1 / a1 - 1) * nn, eye + (1 / a2 - 1) * nn
    face_map = a2 * g2
    content = 'edge_eps' in primal
    if content:
        edge, face = x['edge_eps'][..., :, None], x['face_eps'][..., :, None]
        g2 = dwhere(primal['edge_single'][..., :, None],
                    eye + (1 / edge - 1) * nn,
                    eye + (x['edge_inv'][..., :, None] - 1) * nn)
        face_map = dwhere(primal['face_single'][..., :, None],
                         face * (eye + (1 / face - 1) * nn),
                         face * (eye - nn) + nn)
    gamma = lengths[..., :, None] * g1 + (1 - lengths[..., :, None]) * g2
    if content:
        face = x['face_eps'][..., :, None]
        outside = dwhere(primal['face_single'][..., :, None],
                         (1 - areas[..., :, None]) * face * (eye + (1 / face - 1) * nn),
                         (1 - areas[..., :, None]) * face_map)
        pi = areas[..., :, None] * a1 * g1 + outside
    else:
        pi = areas[..., :, None] * a1 * g1 + (1 - areas[..., :, None]) * a2 * g2
    determinant = np.linalg.det(pi.p.v)
    singular = determinant == 0
    safe = dwhere(singular[..., None, None], eye, pi)
    inv, kappa = inverse(safe)
    acc = gamma @ inv
    tensor = (acc + acc.swap()) * .5
    bulk_any = np.zeros(singular.shape, dtype=bool)
    for fraction, eps in ((0, e2), (1, e1)):
        bulk = (np.all(lengths.p.v == fraction, axis=-1)
                & np.all(areas.p.v == fraction, axis=-1))
        tensor = dwhere(bulk[..., None, None], eye / eps[..., None, None], tensor)
        bulk_any |= bulk
    guard = singular & ~bulk_any
    maximum, minimum = extremum(e1, e2, True), extremum(e1, e2, False)
    if content:
        edge = dwhere(primal['edge_single'], x['edge_eps'], 1 / x['edge_inv'])
        maximum = extremum(e1, extremum(axis_extremum(edge, True), axis_extremum(x['face_eps'], True), True), True)
        minimum = extremum(e1, extremum(axis_extremum(edge, False), axis_extremum(x['face_eps'], False), False), False)
    lo, hi = 1 / maximum, 1 / minimum
    w, v, pe = eigen_data(tensor)
    clip = ((w[..., 0] < lo.p.v) | (w[..., -1] > hi.p.v)) & ~bulk_any & ~singular
    clipped, L, inside, gap, tol = spectral_clip(tensor, lo, hi, w, v, pe)
    final = dwhere(clip[..., None, None], clipped, tensor)
    final = dwhere(guard[..., None, None], x['fallback'], final)
    # Cofactors by 2x2 minors, including singular matrices, rather than det*inv.
    cofactor = np.empty_like(pi.p.v)
    for i in range(3):
        for j in range(3):
            rows, cols = [a for a in range(3) if a != i], [a for a in range(3) if a != j]
            cofactor[..., i, j] = ((-1)**(i+j) * (
                pi.p.v[..., rows[0], cols[0]] * pi.p.v[..., rows[1], cols[1]]
                - pi.p.v[..., rows[0], cols[1]] * pi.p.v[..., rows[1], cols[0]]))
    determinant_delta = np.sum(np.abs(cofactor) * pi.p.delta, axis=(-2, -1))
    dlambda = np.diagonal(pe.sum(axis=0), axis1=-2, axis2=-1)
    clip_near = np.any((np.abs(w - lo.p.v[..., None]) <= dlambda + lo.p.delta[..., None])
                       | (np.abs(w - hi.p.v[..., None]) <= dlambda + hi.p.delta[..., None]), axis=-1)
    divided = first_divided(w[..., :, None], w[..., None, :], lo.p.v[..., None, None], hi.p.v[..., None, None])
    tie = inside[..., :, None].astype(float)
    sep_near = np.any((np.abs(np.abs(gap) - tol) <= dlambda[..., :, None] + dlambda[..., None, :]
                       + EPS * np.max(np.abs(w), axis=-1)[..., None, None])
                      & (np.abs(divided - tie) > EPS), axis=(-2, -1)) & clip
    below, above = w < lo.p.v[..., None], w > hi.p.v[..., None]
    sides = ((inside[..., :, None] & ~inside[..., None, :])
             | (~inside[..., :, None] & inside[..., None, :])
             | (below[..., :, None] & above[..., None, :])
             | (above[..., :, None] & below[..., None, :]))
    valid_kappa = EPS * np.max(np.abs(w), axis=-1)[..., None, None] / np.where(sides, np.abs(gap), 1)
    clip_invalid = clip & np.any(sides & (valid_kappa > 1e-3), axis=(-2, -1))
    off = ~np.eye(3, dtype=bool)
    off_near = np.all((np.abs(final.p.v) <= final.p.delta) | ~off, axis=(-2, -1))
    diagonal = np.diagonal(final.p.v, axis1=-2, axis2=-1)
    diagonal_delta = np.diagonal(final.p.delta, axis1=-2, axis2=-1)
    diag_near = np.all(np.abs(diagonal - diagonal[..., :1]) <= diagonal_delta + diagonal_delta[..., :1], axis=-1)
    isotropic = np.all(final.p.v == final.p.v[..., :1, :1] * np.eye(3), axis=(-2, -1))
    # M1-only bulk gate: clip/inside choices cannot change at a bulk triplet.
    clip_near &= ~bulk_any
    cases = dict(singular=singular | (np.abs(determinant) <= determinant_delta),
                 inverse_validity=EPS * kappa > 1e-3,
                 clip_inside_near=clip_near, separated_near=sep_near,
                 clipped_validity=clip_invalid,
                 isotropy_near=~bulk_any & ((off_near & diag_near) | isotropic),
                 guarded=guard)
    details = dict(pi=pi, gamma=gamma, inverse=inv, accurate=acc, tensor=tensor, lo=lo, hi=hi,
                   final=final, kappa=kappa, w=w, v=v, P=pe, clip=clip,
                   bulk=bulk_any, determinant=determinant, determinant_delta=determinant_delta)
    touched = np.any(lengths.p.v > 0, axis=-1) | np.any(areas.p.v > 0, axis=-1)
    return final, cases, details, touched, isotropic


def next_node(value, axis, periodic):
    if periodic[axis]:
        return np.roll(value, -1, axis=axis)
    return np.take(value, np.minimum(np.arange(value.shape[axis]) + 1, value.shape[axis] - 1), axis=axis)


def bnext(value, axis, periodic):
    return B(next_node(value.v, axis, periodic),
             np.stack([next_node(e, axis, periodic) for e in value.e]))


def dnext(value, axis, periodic):
    return D(bnext(value.p, axis, periodic), None if value.d is None else bnext(value.d, axis, periodic))


def tied_clusters(w, v, near_p):
    """Connected components of the completed tau_pair relation (A36'')."""
    dlambda = np.diag(near_p)
    peak = np.max(np.abs(w))
    dpeak = np.max(dlambda[np.abs(w) == peak])
    threshold = EPS * peak + dlambda[:, None] + dlambda[None, :] + EPS * dpeak
    connected = np.abs(w[:, None] - w[None, :]) <= threshold
    for k in range(3):
        connected |= connected[:, k, None] & connected[None, k, :]
    projectors, sizes, amplitudes = [], [], []
    for a in range(3):
        members = np.flatnonzero(connected[a])
        projector = v[:, members] @ v[:, members].T
        projectors.append(projector)
        sizes.append(len(members))
        amplitudes.append(np.sqrt(np.maximum(np.diag(projector), 0)))
    # For singleton clusters this is exactly |v_a|, without a new rounding site.
    outer = np.einsum('ai,bl->abil', amplitudes, amplitudes)
    return dict(connected=connected, projectors=projectors, sizes=sizes, outer=outer,
                threshold=threshold)


def cluster_contract(cluster, value):
    return np.einsum('abil,...ab->...il', cluster['outer'], value)


def cluster_h(cluster, dt, v):
    result = np.abs(v.T @ dt @ v)
    for a in range(3):
        for b in range(3):
            if cluster['sizes'][a] > 1:
                projector = cluster['projectors'][a]
                result[a, b] = (np.linalg.norm(projector @ dt @ projector, ord=2)
                                if cluster['connected'][a, b]
                                else np.linalg.norm(projector @ dt @ v[:, b]))
            elif cluster['sizes'][b] > 1:
                result[a, b] = np.linalg.norm(cluster['projectors'][b] @ dt @ v[:, a])
    return result


def max_errors(reference, candidate):
    """Entrywise maximum of whole bounds, retaining the winning term vector."""
    return B(reference.v, np.where(candidate.delta > reference.delta, candidate.e, reference.e))


def slice_dual(value, index):
    if value.p.v.ndim >= 3 and value.p.v.shape[:3] == (6, 6, 4):
        return value[index]
    return value


def near_triplet_bounds(final, stages, near_stages, cases, fallback):
    """Enumerate every near outcome; add the completed cluster-safe J once."""
    exact_singular = stages['determinant'] == 0
    bulk = stages['bulk']
    affected = np.zeros(bulk.shape, dtype=bool)
    for name in ('clip_inside_near', 'separated_near', 'singular'):
        affected |= cases[name]
    affected |= stages['clip']
    affected &= ~bulk & ~exact_singular
    counts = np.ones(bulk.shape, dtype=int)
    jumps = np.zeros_like(final.p.v)
    djumps = np.zeros_like(final.p.v)
    final = D(B(final.p.v, final.p.e.copy()), B(tangent(final).v, tangent(final).e.copy()))
    for index_array in np.argwhere(affected):
        index = tuple(index_array)
        tensor = slice_dual(stages['tensor'], index)
        lo, hi = slice_dual(stages['lo'], index), slice_dual(stages['hi'], index)
        nlo, nhi = slice_dual(near_stages['lo'], index), slice_dual(near_stages['hi'], index)
        ntensor = slice_dual(near_stages['tensor'], index)
        w, v = stages['w'][index], stages['v'][index]
        pe = stages['P'][(slice(None),) + index]
        pnear = near_stages['P'][(slice(None),) + index].sum(axis=0)
        dlambda = np.diag(pnear)
        cluster = tied_clusters(w, v, pnear)
        near_low = np.abs(w - lo.p.v) <= dlambda + nlo.p.delta
        near_high = np.abs(w - hi.p.v) <= dlambda + nhi.p.delta
        choices = []
        for k in range(3):
            if near_low[k]:
                choices.append(('eigen', k, -1))
            if near_high[k]:
                choices.append(('eigen', k, 1))
        gap = w[:, None] - w[None, :]
        tol = EPS * np.max(np.abs(w))
        default_state = np.where(w < lo.p.v, -1, np.where(w > hi.p.v, 1, 0))
        default_sep = np.abs(gap) > tol
        exact_divided = first_divided(w[:, None], w[None, :], lo.p.v, hi.p.v)
        tie = ((w >= lo.p.v) & (w <= hi.p.v)).astype(float)
        separated_choices = []
        if cases['separated_near'][index]:
            for i, j in PAIRS:
                close = abs(abs(gap[i, j]) - tol) <= dlambda[i] + dlambda[j] + EPS*np.max(np.abs(w))
                different = max(abs(exact_divided[i, j]-tie[i]), abs(exact_divided[j, i]-tie[j])) > EPS
                if close and different:
                    separated_choices.append((i, j))
                    choices.append(('separated', i, j))
        if cases['clip_inside_near'][index]:
            choices.append(('whole', 0, 0))
        near_guard = cases['singular'][index] and not exact_singular[index]
        if near_guard:
            choices.append(('guard', 0, 0))
        best = D(final.p[index], tangent(final)[index])
        for flags in product((False, True), repeat=len(choices)):
            state, separated = default_state.copy(), default_sep.copy()
            use_clip, use_guard = bool(stages['clip'][index]), False
            # Apply lower then upper, matching clip's maximum then minimum.
            for flag, (kind, i, j) in zip(flags, choices):
                if kind == 'eigen':
                    if flag:
                        state[i] = j
                    elif state[i] == j:
                        state[i] = 0
                elif kind == 'separated':
                    separated[i, j] = separated[j, i] = flag
                elif kind == 'whole':
                    use_clip = flag
                else:
                    use_guard = flag
            candidate = tensor
            if use_clip:
                candidate = spectral_clip(tensor, lo, hi, w, v, pe,
                                          outcome=(state, separated), cluster=cluster)[0]
            if use_guard:
                candidate = slice_dual(fallback, index)
            best = D(max_errors(best.p, candidate.p), max_errors(tangent(best), tangent(candidate)))
        counts[index] = 2**len(choices)
        dt, ndt = tangent(tensor), tangent(ntensor)
        H = v.T @ dt.v @ v
        h_abs = cluster_h(cluster, dt.v, v)
        jump_p = np.zeros((3, 3))
        G = np.zeros((3, 3))
        eigen_near = near_low | near_high
        for k in range(3):
            for selected, threshold, dthreshold, ndthreshold in (
                    (near_low[k], lo.p.v, tangent(lo), tangent(nlo)),
                    (near_high[k], hi.p.v, tangent(hi), tangent(nhi))):
                if not selected:
                    continue
                margin = abs(w[k] - threshold) + dlambda[k]
                jump_p += cluster['outer'][k, k] * margin
                if cluster['sizes'][k] > 1:
                    G[k, k] += np.linalg.norm(dt.v, ord='fro') + abs(dthreshold.v)
                else:
                    delta_h = (np.abs(v.T) @ ndt.delta @ np.abs(v))[k, k] + ndthreshold.delta
                    for j in range(3):
                        if j != k:
                            delta_h += 2 * pnear[j, k] * h_abs[j, k] / abs(w[j]-w[k])
                    G[k, k] += abs(H[k, k] - dthreshold.v) + delta_h
                for j in range(3):
                    if j != k and not eigen_near[j] and not cluster['connected'][k, j]:
                        amount = margin * h_abs[k, j] / abs(w[k] - w[j])
                        G[k, j] += amount
                        G[j, k] += amount
        for i, j in PAIRS:
            if (eigen_near[i] and eigen_near[j]) or (cluster['connected'][i, j] and (eigen_near[i] or eigen_near[j])):
                # Shared pairs and cluster ties replace, rather than add, per-k terms.
                G[i, j] = G[j, i] = h_abs[i, j]
        for i, j in separated_choices:
            amount = abs(exact_divided[i, j] - tie[i]) * h_abs[i, j]
            G[i, j] += amount
            G[j, i] += amount
        jump_d = cluster_contract(cluster, G)
        if near_guard:
            fb = slice_dual(fallback, index)
            accurate = slice_dual(stages['accurate'], index)
            jump_p += np.abs(fb.p.v - accurate.p.v)
            jump_d += np.abs(tangent(fb).v - tangent(accurate).v)
        final.p.e[(slice(None),) + index] = best.p.e
        final.d.e[(slice(None),) + index] = tangent(best).e
        final.p.e[(-1,) + index] += jump_p
        final.d.e[(-1,) + index] += jump_d
        jumps[index], djumps[index] = jump_p, jump_d
    return final, dict(combinations=counts, primal_J=jumps, tangent_J=djumps)


@dataclass
class Witness:
    values: list
    bounds: list
    terms: list
    outside: list
    cases: dict
    triplets: list
    capture_method: str
    strict: list
    allowed_masks: list


def rounding_bound(capture):
    """Recompute every solver-reached leaf and A36 downstream mask."""
    operands, assembly = capture['primal']
    dots, dassembly = capture['tangent']
    assert assembly['weights'] is None, 'A37 scene is uniform.'
    finals, outside, details, isotropics, cases = [], [], [], [], {}
    strict_triplets, iso_near = [], []
    for row, dot, active in zip(operands, dots, capture['active']):
        final, _, stages, touched, isotropic = recompute_triplet(row, dot, active)
        _, near_cases, near_stages, _, _ = recompute_triplet(row, dot, active, near=True)
        fallback = _input(row['fallback'], dot['fallback'], active['fallback'], False)
        final, near_details = near_triplet_bounds(final, stages, near_stages, near_cases, fallback)
        strict_triplets.append((stages['determinant'] == 0) & touched)
        iso_near.append(near_cases['isotropy_near'] & touched)
        # Every A36 test uses the M4' near run, including M1 input roundings.
        # Exact float64 decisions are unchanged between these two runs.
        mask = np.zeros(touched.shape, dtype=bool)
        for name, selected in near_cases.items():
            cases.setdefault(name, []).append(selected & touched)
            mask |= selected & touched
        final = dwhere(touched[..., None, None], final, D(np.eye(3)))
        finals.append(final)
        outside.append(mask)
        isotropics.append(np.where(touched, isotropic, True))
        details.append(dict(stages=stages, near_stages=near_stages, cases=near_cases, near_outcomes=near_details))
    triplets = dstack(finals)
    outside = np.stack(outside)
    strict_triplets = np.stack(strict_triplets)
    iso_near = np.stack(iso_near)
    isotropics = np.stack(isotropics)
    periodic = tuple(bool(a) for a in assembly['periodic'])
    pad = assembly['pad'].astype(bool)
    ends = tuple(tuple(sum((finals[t][..., a, a] for t, signs in enumerate(SIGNS) if signs[a] == s), D(0)) / 8
                       for s in (0, 1)) for a in range(3))
    cross = tuple(dstack([dstack([
        sum((finals[t][..., a, b] for t, signs in enumerate(SIGNS) if signs[a] == sa and signs[b] == sb), D(0)) / 8
        for sb in (0, 1)]) for sa in (0, 1)]) for a, b in PAIRS)
    bulk_ends = tuple(tuple(np.all(np.stack([isotropics[t] for t, signs in enumerate(SIGNS) if signs[a] == s]), axis=0)
                           for s in (0, 1)) for a in range(3))
    edge_bulk = tuple(bulk_ends[a][0] & next_node(bulk_ends[a][1], a, periodic)
                      & assembly['mid_bulk'][a] for a in range(3))
    end_outside = tuple(tuple(np.any(np.stack([outside[t] for t, signs in enumerate(SIGNS) if signs[a] == s]), axis=0)
                             for s in (0, 1)) for a in range(3))
    diagonal_outside = [end_outside[a][0] | next_node(end_outside[a][1], a, periodic) for a in range(3)]
    cross_outside = [np.stack([np.stack([
        np.any(np.stack([outside[t] for t, signs in enumerate(SIGNS) if signs[a] == sa and signs[b] == sb]), axis=0)
        for sb in (0, 1)]) for sa in (0, 1)]) for a, b in PAIRS]
    strict_ends = tuple(tuple(np.any(np.stack([strict_triplets[t] for t, signs in enumerate(SIGNS) if signs[a] == s]), axis=0)
                             for s in (0, 1)) for a in range(3))
    strict_diagonal = [strict_ends[a][0] | next_node(strict_ends[a][1], a, periodic) for a in range(3)]
    strict_cross = [np.stack([np.stack([
        np.any(np.stack([strict_triplets[t] for t, signs in enumerate(SIGNS) if signs[a] == sa and signs[b] == sb]), axis=0)
        for sb in (0, 1)]) for sa in (0, 1)]) for a, b in PAIRS]
    can_bulk_ends = tuple(tuple(np.all(np.stack([(isotropics[t] | iso_near[t]) for t, signs in enumerate(SIGNS) if signs[a] == s]), axis=0)
                               for s in (0, 1)) for a in range(3))
    can_nonbulk_ends = tuple(tuple(np.any(np.stack([(~isotropics[t] | iso_near[t]) for t, signs in enumerate(SIGNS) if signs[a] == s]), axis=0)
                                  for s in (0, 1)) for a in range(3))
    edge_near = tuple((can_bulk_ends[a][0] & next_node(can_bulk_ends[a][1], a, periodic)
                      & (can_nonbulk_ends[a][0] | next_node(can_nonbulk_ends[a][1], a, periodic))
                      & assembly['mid_bulk'][a] & ~pad) for a in range(3))
    other = assembly['other_leaves']
    dother = dassembly['other_leaves']
    leaves = []
    for p, d in zip(other, dother):
        if p.dtype.kind == 'f':
            # R10 for xi, midpoint/pad entries and other solver-unreached values.
            bp, bd = B(p).rounded(), B(d).rounded()
            leaves.append(D(bp, bd))
        else:
            leaves.append(None)
    masks = [np.zeros(p.shape, dtype=bool) for p in other]
    strict_masks = [np.zeros(p.shape, dtype=bool) for p in other]
    allowed_masks = [np.zeros(p.shape, dtype=bool) for p in other]
    for a in range(3):
        midpoint = D(B(assembly['midpoint'][a]).rounded(), B(dassembly['midpoint'][a]).rounded())
        assembled = 1 / (ends[a][0] + dnext(ends[a][1], a, periodic))
        leaves[a] = dwhere(edge_bulk[a] | pad, midpoint, assembled)
        alternate = dwhere(edge_bulk[a] | pad, assembled, midpoint)
        near_value = D(max_errors(leaves[a].p, alternate.p), max_errors(tangent(leaves[a]), tangent(alternate)))
        near_value.p.e[-1] += np.abs(midpoint.p.v - assembled.p.v)
        near_value.d.e[-1] += np.abs(tangent(midpoint).v - tangent(assembled).v)
        leaves[a] = dwhere(edge_near[a], near_value, leaves[a])
        masks[a] = diagonal_outside[a] | edge_near[a]
        strict_masks[a] = strict_diagonal[a]
        for leaf_index in (9+a, 12+a):
            allowed_masks[leaf_index] = edge_near[a]
            strict_masks[leaf_index] = strict_diagonal[a]
    leaves[15] = triplets
    masks[15] = np.broadcast_to(outside[..., None, None], triplets.p.v.shape)
    strict_masks[15] = np.broadcast_to(strict_triplets[..., None, None], triplets.p.v.shape)
    for i, value in enumerate(cross):
        leaves[16+i], masks[16+i] = value, cross_outside[i]
        strict_masks[16+i] = strict_cross[i]
    values, bounds, terms, all_masks = [], [], [], []
    for dotted in (False, True):
        for i, value in enumerate(leaves):
            if value is None:
                values.append((dother if dotted else other)[i])
                bounds.append(np.zeros(other[i].shape))
                terms.append(np.zeros((len(_TERMS),) + other[i].shape))
            else:
                v = tangent(value) if dotted else value.p
                values.append(v.v)
                bounds.append(v.delta)
                terms.append(v.e)
            all_masks.append(masks[i])
    return Witness(values, bounds, terms, all_masks, {k: np.stack(v) for k, v in cases.items()}, details,
                   capture['method'], strict_masks + strict_masks, allowed_masks + allowed_masks)


def byte_differences(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    if a.dtype.itemsize == 0:
        return np.zeros(a.shape, dtype=bool)
    return np.any(a.reshape(-1).view(np.uint8).reshape(a.shape + (a.dtype.itemsize,))
                  != b.reshape(-1).view(np.uint8).reshape(b.shape + (b.dtype.itemsize,)), axis=-1)


def compare_builds(reference, actual, witness):
    """A37 per-entry checks and power data; does not adjust any bound."""
    import jax
    rows, violations = [], []
    for leaf, (a, b, bound, mask, terms) in enumerate(zip(jax.tree.leaves(reference), jax.tree.leaves(actual),
                                                       witness.bounds, witness.outside, witness.terms)):
        a, b = np.asarray(a), np.asarray(b)
        changed = byte_differences(a, b)
        row = dict(leaf=leaf, entries=int(a.size), outside=int(np.count_nonzero(mask)),
                   differing=int(np.count_nonzero(changed)))
        jump = terms[-1]
        strict = witness.strict[leaf]
        row['J_entries'] = int(np.count_nonzero(jump))
        row['largest_J_over_bound'] = float(np.max(np.divide(jump, bound, out=np.zeros_like(bound), where=bound != 0)))
        row['allowed_mask_differences'] = int(np.count_nonzero(changed & witness.allowed_masks[leaf] & ~strict))
        row['strict_entries'] = int(np.count_nonzero(strict))
        if a.dtype.kind != 'f':
            bad = changed & (~witness.allowed_masks[leaf] | strict)
            row.update(max_ratio=0., max_difference=0., bound_at_max=0., largest_bound_over_difference=None,
                       median_bound_over_value=None)
            diff = np.zeros(a.shape)
        else:
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            bad = (changed & strict) | (diff > bound) | ~np.isfinite(diff) | ~np.isfinite(bound)
            ratio = np.divide(diff, bound, out=np.where(diff == 0, 0., np.inf), where=bound != 0)
            index = np.unravel_index(np.argmax(diff), diff.shape)
            nonzero = diff != 0
            scale = np.abs(a.astype(np.float64))
            power = np.divide(bound, scale, out=np.where(bound == 0, 0., np.inf), where=scale != 0)
            row.update(max_ratio=float(np.max(ratio)), max_difference=float(diff[index]),
                       max_index=list(map(int, index)), bound_at_max=float(bound[index]),
                       largest_bound_over_difference=float(np.max(bound[nonzero] / diff[nonzero])) if np.any(nonzero) else None,
                       median_bound_over_value=float(np.median(power)))
        row['violations'] = int(np.count_nonzero(bad))
        rows.append(row)
        for index in np.argwhere(bad):
            ix = tuple(index)
            violations.append(dict(leaf=leaf, index=index.tolist(), outside=bool(mask[ix]), strict=bool(strict[ix]), J=float(jump[ix]),
                                   reference=float(a[ix]), actual=float(b[ix]), difference=float(diff[ix]),
                                   bound=float(bound[ix]), terms={name: float(terms[(k,) + ix]) for k, name in enumerate(_TERMS)}))
    return dict(leaves=rows, violations=violations,
                cases={name: int(np.count_nonzero(mask)) for name, mask in witness.cases.items()},
                entries=sum(row['entries'] for row in rows), outside=sum(row['outside'] for row in rows),
                differing=sum(row['differing'] for row in rows))

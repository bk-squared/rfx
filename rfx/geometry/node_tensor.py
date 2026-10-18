"""Uniform Stage-1 node tensors and the Werner--Bauer--Cary triplet mean."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rfx.geometry import smoothing as smooth


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NodeTensor:
    """Kottke fallback, triplet tensors, Eq. 10 couplings and edge masks.

    ``xi`` retains attempt 1's node tensors solely for fallback/composition.
    The update uses ``cross`` and the assembled triplet diagonal instead.
    ``guard_count`` counts replaced triplets over all material groups.
    """

    xi: tuple
    bulk: tuple
    active: tuple
    triplets: object
    cross: tuple
    guard_count: object
    pec_keep: tuple | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SmoothedEps:
    """Assembly-only iterable; TensorOperands.of separates its two operands."""

    diagonal: tuple
    nodes: NodeTensor

    def __iter__(self):
        return iter(self.diagonal)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return self.diagonal[index]


def uniform_cell(grid):
    """The one cell width of a grid whose three axes are constant (full_tensor is uniform-only)."""
    widths = {float(grid.cells(a)[0]) for a in range(3)}
    if not all(grid.is_constant(a) for a in range(3)) or len(widths) != 1:
        raise NotImplementedError("the node/triplet tensor needs equal constant cells on every axis")
    return widths.pop()


def _union(xp, grid, shapes, coords, *, distance=False):
    from rfx._periodic import periodic_sdf

    union = nearest = normal = None
    for shape in shapes:
        sdf, ns = periodic_sdf(
            grid, shape, coords, smooth._get_sdf_fn(shape), xp=xp,
            normal_fn=smooth._get_normal_fn(shape))
        if union is None:
            union, nearest, normal = sdf, xp.abs(sdf), ns
        else:
            closer = xp.abs(sdf) < nearest
            normal = tuple(xp.where(closer, n, old) for n, old in zip(ns, normal))
            nearest = xp.minimum(nearest, xp.abs(sdf))
            union = xp.minimum(union, sdf)
    result = (xp.clip(0.5 - union / uniform_cell(grid), 0.0, 1.0), normal)
    return result + (union,) if distance else result


def _next(xp, a, axis, periodic):
    if periodic[axis]:
        return xp.roll(a, -1, axis)
    return xp.take(a, xp.minimum(xp.arange(a.shape[axis]) + 1, a.shape[axis] - 1), axis)


def compute_node_tensor_eps(grid, shapes, background_eps=1.0, *, field_dtype=None):
    """Host-float64 concrete geometry, or the smoother's traced JAX path (#833).

    Same-material groups use the existing union, nearest analytic normal and
    SDF cell-length rule. Later groups compose against the prior bulk eps or
    the reciprocal mean diagonal of a prior interface tensor (decision 3).
    """
    from collections import OrderedDict
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    from rfx.geometry.triplet_tensor import SIGNS, build_triplets, coefficients, matrices

    coords = coords_from_uniform_grid(grid)
    masks = smooth._fallback_masks([s for s, _ in shapes], lambda s: s.mask(grid))
    host = smooth._host_path(*coords[:3], uniform_cell(grid), background_eps,
                            *(e for _, e in shapes),
                            *(p for s, _ in shapes for p in smooth._shape_parameters(s)),
                            *masks.values())
    xp = np if host else jnp
    field_dtype = jnp.float32 if field_dtype is None else field_dtype
    acc_dtype = np.float64 if host else field_dtype
    midpoint = smooth._smoothed_eps_uniform(xp, grid, coords, shapes, background_eps, masks)
    samples = smooth._uniform_sample_coords(xp, coords, uniform_cell(grid))
    x, y, z = (xp.asarray(a, dtype=acc_dtype) for a in coords[:3])
    nodes = (x[:, None, None], y[None, :, None], z[None, None, :])
    xi = [xp.full(grid.shape, 1.0 / background_eps, dtype=acc_dtype) for _ in range(3)]
    xi += [xp.zeros(grid.shape, dtype=acc_dtype) for _ in range(3)]
    triplets = xp.broadcast_to(matrices(xp, xi), (8,) + grid.shape + (3, 3))
    guard_count = xp.asarray(0, dtype=xp.int32)
    bulk = xp.ones(grid.shape, dtype=bool)
    mid_bulk = [bulk.copy() for _ in range(3)]
    groups = OrderedDict()
    for shape, eps in shapes:
        groups.setdefault(eps, []).append(shape)
    for eps, group in groups.items():
        for shape in group:
            if smooth._has_sdf(shape):
                continue
            mask = masks[id(shape)]
            if mask is None:
                continue
            xi = [xp.where(mask, 1.0 / eps if i < 3 else 0.0, a) for i, a in enumerate(xi)]
            triplets = xp.where(mask[None, ..., None, None], xp.eye(3) / eps, triplets)
            bulk = xp.where(mask, True, bulk)
        supported = [s for s in group if smooth._has_sdf(s)]
        if not supported:
            continue
        f, ns, signed_distance = _union(xp, grid, supported, nodes, distance=True)
        interface = (f > 0) & (f < 1)
        outside = 1.0 / xp.where(bulk, xi[0], (xi[0] + xi[1] + xi[2]) / 3.0)
        inv_perp = f / eps + (1.0 - f) / (outside + 1e-30)
        inv_par = 1.0 / (f * eps + (1.0 - f) * outside + 1e-30)
        entries = [n * n * inv_perp + (1 - n * n) * inv_par for n in ns]
        entries += [ns[i] * ns[j] * (inv_perp - inv_par) for i, j in ((0, 1), (0, 2), (1, 2))]
        xi = [xp.where(f >= 1, 1.0 / eps if i < 3 else 0.0,
                       xp.where(interface, value, old))
              for i, (value, old) in enumerate(zip(entries, xi))]
        current, guards, touched = build_triplets(
            xp, grid, supported, nodes, ns, eps, outside, matrices(xp, xi),
            signed_distance=signed_distance)
        triplets = xp.where(touched[..., None, None], current, triplets)
        guard_count = guard_count + xp.sum(guards & touched, dtype=xp.int32)
        bulk = xp.where(f > 0, ~interface, bulk)
        for axis, sample in enumerate(samples.values()):
            fm, _ = _union(xp, grid, supported, sample)
            mid_bulk[axis] = xp.where(fm > 0, fm >= 1, mid_bulk[axis])
    periodic = tuple(a in grid.periodic_axes or (a == 'z' and grid.is_2d) for a in 'xyz')
    pad = np.zeros(grid.shape, dtype=bool)
    for axis, a in enumerate('xyz'):
        for side in ('lo', 'hi'):
            count = getattr(grid, f'pad_{a}_{side}')
            if count:
                sl = [slice(None)] * 3
                sl[axis] = slice(0, count) if side == 'lo' else slice(-count, None)
                pad[tuple(sl)] = True
    # Attempt-2 decision 1: all eight incident triplets must be exactly
    # isotropic bulk tensors. A node-volume mask can delete only one side of
    # an Eq. 10 coupling and therefore break assembled symmetry.
    triplet_bulk = xp.all(triplets == triplets[..., :1, :1] * xp.eye(3), axis=(-2, -1))
    bulk_ends = tuple(tuple(xp.all(xp.stack([
        triplet_bulk[t] for t, signs in enumerate(SIGNS) if signs[a] == s]), axis=0)
        for s in (0, 1)) for a in range(3))
    edge_bulk = tuple(bulk_ends[a][0] & _next(xp, bulk_ends[a][1], a, periodic)
                      & mid_bulk[a] for a in range(3))
    ends, cross = coefficients(xp, triplets)
    diagonal = tuple(xp.where(edge_bulk[a] | pad, midpoint[a],
                             1.0 / (ends[a][0] + _next(xp, ends[a][1], a, periodic)))
                     for a in range(3))
    def cast_arrays(arrays):
        return tuple(jnp.asarray(a, dtype=field_dtype) for a in arrays)
    return SmoothedEps(cast_arrays(diagonal), NodeTensor(
        cast_arrays(xi), tuple(jnp.asarray(b) for b in edge_bulk),
        tuple(jnp.asarray(~(b | pad)) for b in edge_bulk),
        cast_arrays((triplets,))[0], cast_arrays(cross),
        jnp.asarray(guard_count)))

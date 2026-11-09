"""Realization and edge reduction of spatial Lorentz pole declarations."""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.core.yee import edge_mean_components
from rfx.materials.debye import pole_reach


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class SampledPoleMap:
    omega_0: object
    delta: object
    kappa: object
    centres: tuple
    source: object = field(metadata={"static": True})

    def __iter__(self):
        return iter((self.omega_0, self.delta, self.kappa))


def is_map(pole):
    from rfx.materials.lorentz import LorentzPoleMap
    return isinstance(pole, (LorentzPoleMap, SampledPoleMap))


def refuse_maps(poles, site):
    if any(is_map(p) for p in poles or ()):
        raise ValueError(f"LorentzPoleMap is not supported by {site}; use scalar poles")


def sample_spec(spec, grid, shape):
    """Sample each declaration once; retain host values for concrete reductions."""
    if spec is None or not any(is_map(p) for p in spec[0]):
        return spec
    if grid is None:
        raise ValueError("LorentzPoleMap requires a realized grid; grid-less realization is unsupported")
    if all(not is_map(p) or isinstance(p, SampledPoleMap) for p in spec[0]):
        return spec
    poles, masks = spec
    masks = masks if isinstance(masks, (list, tuple)) else [masks] * len(poles)
    if len(masks) != len(poles):
        raise ValueError(f"Expected {len(poles)} Lorentz masks, got {len(masks)}")
    from rfx.geometry.rasterize_grid import (
        cell_centres_from_nodes, coords_from_nonuniform_grid, coords_from_uniform_grid,
        cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
    )
    coords = (coords_from_nonuniform_grid(grid) if hasattr(grid, "dx_arr")
              else coords_from_uniform_grid(grid))
    sizes = (cell_sizes_from_nonuniform_grid(grid) if hasattr(grid, "dx_arr")
             else cell_sizes_from_uniform_grid(grid))
    centres = cell_centres_from_nodes(coords, sizes)
    axes = (centres.x, centres.y, centres.z)
    xyz = tuple(v.reshape(tuple(-1 if i == a else 1 for i in range(3)))
                for a, v in enumerate(axes))
    sampled = []
    for pole, mask in zip(poles, masks):
        if not is_map(pole) or isinstance(pole, SampledPoleMap):
            sampled.append(pole)
            continue
        values = [v(*xyz) if callable(v) else v for v in (pole.delta, pole.kappa)]
        xp = jnp if any(is_tracer(v) for v in values) or is_tracer(mask) else np
        occupied = xp.ones(shape, dtype=bool) if mask is None else xp.broadcast_to(mask, shape).astype(bool)
        values = [xp.broadcast_to(v, shape) for v in values]
        from rfx._traced_check import traced_check

        def check(active, delta, kappa):
            for name, value in (("delta", delta), ("kappa", kappa)):
                bad = np.asarray(active) & (~np.isfinite(value) | (np.asarray(value) < 0))
                if np.any(bad):
                    index = tuple(np.argwhere(bad)[0])
                    position = tuple(float(axis[i]) for axis, i in zip(axes, index))
                    raise ValueError(f"LorentzPoleMap {name} at {position} m is {np.asarray(value)[index]}; "
                                     "occupied values must be finite and non-negative")

        traced_check("LorentzPoleMap", "occupied cell values", check, occupied, *values, ordered=True)
        delta, kappa = (xp.where(occupied, v, 0.) for v in values)
        sampled.append(SampledPoleMap(pole.omega_0, delta, kappa, axes, pole))
    return sampled, masks


def map_terms(pole, mask, dt, shape, fractions, *, periodic, high_wall, cell_sizes):
    """Use the shared builder after the declared strength-weighted edge rule."""
    from rfx.materials.lorentz import LorentzPole, lorentz_pole_coeffs, pole_update_coeffs
    concrete = not any(is_tracer(v) for v in (pole.delta, pole.kappa, mask))
    xp = np if concrete else jnp
    occupied = xp.ones(shape, dtype=bool) if mask is None else xp.broadcast_to(mask, shape).astype(bool)
    kappa, delta = xp.asarray(pole.kappa), xp.asarray(pole.delta)
    if concrete:
        active_k, active_d = kappa[occupied], delta[occupied]
        ks = float(active_k.max()) if active_k.size else 0.
        ds = float(active_d[np.argmax(active_k)]) if active_k.size else 0.
        constant_delta = not active_d.size or bool(np.all(active_d == ds))
        constant_kappa = not active_k.size or bool(np.all(active_k == ks))
        if ks == 0 or (constant_delta and constant_kappa):
            a, b, c = lorentz_pole_coeffs([LorentzPole(pole.omega_0, ds, ks)], dt, shape, [fractions])
            return a[0], b[0], tuple(v[0] for v in c)
    else:
        # The normalization scale cancels between the builder and edge weights.
        # Holding it fixed avoids differentiating a squared SI-strength divisor.
        ks = jax.lax.stop_gradient(xp.max(xp.where(occupied, kappa, 0.)))
        ds = delta.reshape(-1)[xp.argmax(xp.where(occupied, kappa, 0.))]
        constant_delta = False
    k = xp.where(ks > 0, kappa / xp.where(ks > 0, ks, 1.), 0.)
    # Concrete products are rounded before entering the edge-mean program.
    edge_xp = np if concrete and not constant_delta else jnp
    mk = edge_xp.asarray(xp.where(occupied, k, 0.))
    kwargs = dict(periodic=periodic, high_wall=high_wall, cell_sizes=cell_sizes, array_module=edge_xp)
    weights = edge_mean_components(mk, **kwargs)
    if constant_delta:
        av, bv, cv = pole_update_coeffs(pole.omega_0, ds, ks, dt)
        reach = jnp.ones(shape, bool) if fractions is None else pole_reach(fractions)
        return jnp.where(reach, av, 0.), jnp.where(reach, bv, 0.), tuple(cv * w for w in weights)
    product = edge_xp.asarray(xp.where(occupied, k * (delta - ds), 0.))
    offsets = edge_mean_components(product, **kwargs)
    terms = []
    for w, offset in zip(weights, offsets):
        de = ds + edge_xp.where(w > 0, offset / edge_xp.where(w > 0, w, 1.), 0.)
        a, b, c = pole_update_coeffs(pole.omega_0, de, ks, dt)
        terms.append(tuple(jnp.asarray(v) for v in (a, b, edge_xp.where(ks > 0, c * w, 0.))))
    return tuple(tuple(t[i] for t in terms) for i in range(3))


def build_map_coeffs(poles, masks, dt, shape, weights, *, periodic, high_wall, cell_sizes):
    from rfx.materials.lorentz import lorentz_pole_coeffs
    terms = []
    for pole, mask, fractions in zip(poles, masks, weights):
        if isinstance(pole, SampledPoleMap):
            terms.append(map_terms(pole, mask, dt, shape, fractions, periodic=periodic,
                                   high_wall=high_wall, cell_sizes=cell_sizes))
        else:
            a, b, c = lorentz_pole_coeffs([pole], dt, shape, [fractions])
            terms.append((a[0], b[0], tuple(v[0] for v in c)))
    result = []
    for i in range(3):
        if any(isinstance(t[i], tuple) for t in terms):
            result.append(tuple(jnp.stack([t[i][a] if isinstance(t[i], tuple) else t[i] for t in terms])
                                for a in range(3)))
        else:
            result.append(jnp.stack([t[i] for t in terms]))
    return tuple(result)


def pattern_corner(pole, selected, xp):
    """A corner and its kappa-max position for this membership pattern only."""
    from rfx.materials.lorentz import LorentzPole
    k = xp.where(selected, pole.kappa, -xp.inf)
    index = xp.unravel_index(xp.argmax(k), selected.shape)
    position = tuple(xp.asarray(axis)[i] for axis, i in zip(pole.centres, index))
    corner = LorentzPole(pole.omega_0, xp.min(xp.where(selected, pole.delta, xp.inf)), xp.max(k))
    return corner, position


def minimum_damping(pole, sim, entry):
    """Highest-Q damping over the declared material's realized occupied cells."""
    if not is_map(pole):
        return float(pole.delta)
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid,
        centres_from_nonuniform_grid, centres_from_uniform_grid, _material_cell_mask,
    )
    graded = sim._uses_nonuniform_mesh
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    coords = (coords_from_nonuniform_grid if graded else coords_from_uniform_grid)(grid)
    centres = (centres_from_nonuniform_grid if graded else centres_from_uniform_grid)(grid)
    mask = _material_cell_mask(entry.shape, coords, centres, grid=grid, refuse_empty=False)
    sampled = sample_spec(([pole], [mask]), grid, grid.shape)[0][0]
    active = np.asarray(mask, dtype=bool)
    return float(np.min(np.asarray(sampled.delta)[active])) if np.any(active) else 0.

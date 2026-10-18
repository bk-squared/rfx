"""Admission of the explicit Lorentz update on realized cell pole patterns.

The state is (e, h, p_1, q_1, ...), with polarization in eps0 units and
q the previous polarization. Cell patterns use their lowest epsilon and
both conductivity extrema. Traced masks cannot be grouped and are skipped.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

C0 = 299792458.0
EPS_0 = 8.8541878128e-12
ROOT_TOLERANCE = 1e-6
X_SAMPLES = 129


def largest_growth(eps_inf, sigma, poles, dt, x_max):
    """Maximum amplification-matrix spectral radius over 129 wave modes."""
    dt = float(dt)
    poles = [(float(w), float(d), float(k)) for w, d, k in poles if float(k) != 0.0]
    beta = float(sigma) * dt / (2.0 * EPS_0)
    a_eps, b_eps = float(eps_inf) + beta, float(eps_inf) - beta
    size = 2 + 2 * len(poles)
    matrix = np.zeros((X_SAMPLES, size, size), dtype=np.float64)
    matrix[:, 1, 0] = -np.linspace(0.0, x_max, X_SAMPLES)
    matrix[:, 1, 1] = 1.0
    matrix[:, 0, :] = matrix[:, 1, :]
    matrix[:, 0, 0] += b_eps
    for i, (w0, delta, kappa) in enumerate(poles):
        p, q = 2 + 2 * i, 3 + 2 * i
        d = delta * dt
        matrix[:, p, 0] = kappa * dt * dt / (1.0 + d)
        matrix[:, p, p] = (2.0 - (w0 * dt) ** 2) / (1.0 + d)
        matrix[:, p, q] = -(1.0 - d) / (1.0 + d)
        matrix[:, q, p] = 1.0
        matrix[:, 0, :] -= matrix[:, p, :]
        matrix[:, 0, p] += 1.0
    matrix[:, 0, :] /= a_eps
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


class CourantBudget(float):
    """``c0^2 * sum 1/d_min^2`` of the realized cells; times ``4 dt^2`` it is the largest ``x``.

    As a float it is the whole grid's value. :meth:`local` gives the value of the cells a pole set occupies
    (and one neighbour on each side): a medium on the coarse cells of a graded mesh supports no wave shorter
    than those cells carry, so the smallest cell elsewhere does not bound it (measured: a Drude medium on
    1 mm cells beside 0.25 mm cells is bounded up to the 1 mm limit).
    """

    def __new__(cls, cells):
        self = super().__new__(cls, C0 * C0 * sum(1.0 / float(np.min(c)) ** 2 for c in cells))
        self.cells = tuple(np.asarray(c, dtype=np.float64) for c in cells)
        return self

    def local(self, selected):
        return self.local_axes(tuple(
            np.any(selected, axis=tuple(a for a in range(selected.ndim) if a != axis))
            for axis in range(len(self.cells))))

    def local_axes(self, occupied_per_axis):
        """Budget of global axis projections, including one neighbour."""
        total = 0.0
        for occupied, sizes in zip(occupied_per_axis, self.cells):
            if sizes.size == occupied.size - 1:      # cell metrics omit the terminal node
                sizes = np.append(sizes, sizes[-1])
            if sizes.size != occupied.size or not occupied.any():
                return float(self)
            near = occupied.copy()
            near[1:] |= occupied[:-1]
            near[:-1] |= occupied[1:]
            total += 1.0 / float(np.min(sizes[near])) ** 2
        return C0 * C0 * total


def courant_budget(grid):
    """The grid's :class:`CourantBudget` (two axes on a 2-D grid)."""
    from rfx.core.jax_utils import is_tracer
    axes = (0, 1) if getattr(grid, "is_2d", False) else (0, 1, 2)
    if not callable(getattr(grid, "cells", None)):
        return None   # a stand-in without per-axis cells (unit rigs of the layer functions) is not judged
    if any(is_tracer(grid.cells(a)) for a in axes):
        return None   # a traced mesh has no concrete cells to judge; refuse_unstable_cells skips it
    return CourantBudget([_cells(grid, a) for a in axes])


def _cells(grid, axis):
    return np.atleast_1d(np.asarray(grid.cells(axis), dtype=np.float64))


def largest_stable_step(eps_inf, sigma, poles, dt, budget):
    """Bisect the stable step, or return None if the lower bracket is unstable."""
    def stable(step):
        return largest_growth(eps_inf, sigma, poles, step, 4.0 * step * step * budget) <= 1.0 + ROOT_TOLERANCE
    if stable(dt):
        return dt
    lo, hi = 1e-3 * dt, dt
    if not stable(lo):
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return lo


class PoleSetExtremes(NamedTuple):
    """Exact reductions for one pole pattern, with global axis occupancy."""

    code: int
    poles: tuple
    eps_low: object
    sigma_low: object
    sigma_high: object
    count: int
    occupied: tuple


def pole_set_extremes(materials, lorentz_spec, *, view=None):
    """Reduce static pole patterns over owned cells only; never count halos."""
    import jax.numpy as jnp
    from rfx.core.jax_utils import is_tracer

    if lorentz_spec is None:
        return ()
    poles, masks = lorentz_spec
    if not poles:
        return ()
    masks = masks if isinstance(masks, (tuple, list)) else [masks] * len(poles)
    if len(masks) != len(poles):
        raise ValueError(f"Expected {len(poles)} Lorentz masks, got {len(masks)}")
    if any(is_tracer(m) for m in masks):
        return ()  # D4: traced pole masks cannot be grouped.
    if view is not None:
        from types import SimpleNamespace
        input_shape = materials.eps_r.shape
        materials = SimpleNamespace(**{name: view.own(getattr(materials, name))
                                       for name in ("eps_r", "sigma")})
        masks = [None if m is None else view.own(np.broadcast_to(m, input_shape)) for m in masks]
    shape = materials.eps_r.shape
    # One integer code per cell, accumulated pole by pole (no cells x poles array).
    wide = len(poles) > 62
    codes = np.zeros(int(np.prod(shape)), dtype=object if wide else np.int64)
    for bit, m in enumerate(masks):
        present = np.ones(shape, dtype=bool) if m is None else np.broadcast_to(np.asarray(m, dtype=bool), shape)
        codes = codes + (present.reshape(-1).astype(object if wide else np.int64) * (1 << bit))
    unique_codes, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    patterns = [[bool((int(code) >> bit) & 1) for bit in range(len(poles))] for code in unique_codes]
    values = [materials.eps_r, materials.sigma] + [v for p in poles for v in p]
    traced = any(is_tracer(v) for v in values)
    # A global array spread over processes cannot be read on the host.
    spread = not traced and any(not getattr(v, "is_fully_addressable", True) for v in values)
    xp = jnp if traced or spread else np
    records = []
    for index, (pattern, count) in enumerate(zip(patterns, counts)):
        if not np.any(pattern):
            continue
        selected = (inverse == index).reshape(shape)
        chosen = tuple(p for p, present in zip(poles, pattern) if present)
        eps_low = xp.min(xp.where(selected, materials.eps_r, xp.inf))
        sigma_low = xp.min(xp.where(selected, materials.sigma, xp.inf))
        sigma_high = xp.max(xp.where(selected, materials.sigma, -xp.inf))

        occupied = tuple(np.any(selected, axis=tuple(a for a in range(selected.ndim) if a != axis))
                         for axis in range(selected.ndim))
        if view is not None:
            global_x = np.zeros(view.nx, dtype=bool)
            global_x[view.x0:view.x1] = occupied[0]
            occupied = (global_x,) + occupied[1:]
        records.append(PoleSetExtremes(int(unique_codes[index]), chosen, eps_low,
                                      sigma_low, sigma_high, int(count), occupied))
    return tuple(records)


def combine_pole_extremes(records_from_all_slabs):
    """Combine exact reductions in pole-code order, without cell arrays."""
    import jax.numpy as jnp
    from rfx.core.jax_utils import is_tracer

    combined = {}
    for records in records_from_all_slabs:
        for record in records:
            if record.code not in combined:
                combined[record.code] = record
                continue
            previous = combined[record.code]
            values = (previous.eps_low, previous.sigma_low, previous.sigma_high,
                      record.eps_low, record.sigma_low, record.sigma_high)
            xp = jnp if any(is_tracer(v) for v in values) else np
            combined[record.code] = previous._replace(
                eps_low=xp.minimum(previous.eps_low, record.eps_low),
                sigma_low=xp.minimum(previous.sigma_low, record.sigma_low),
                sigma_high=xp.maximum(previous.sigma_high, record.sigma_high),
                count=previous.count + record.count,
                occupied=tuple(a | b for a, b in zip(previous.occupied, record.occupied)))
    return tuple(combined[code] for code in sorted(combined))


def _concrete(value):
    """The concrete value under an eager trace (jvp, grad), else the value."""
    from rfx.core.jax_utils import is_tracer

    if not is_tracer(value):
        return value
    concrete = value.to_concrete_value()
    return value if concrete is None else concrete


def judge_pole_sets(records, dt, budget, *, grid_kind):
    """Apply the existing admission rule and ordered callback to reductions."""
    from rfx._traced_check import traced_check

    if budget is None:
        return
    for record in records:
        count = record.count
        local = budget.local_axes(record.occupied) if hasattr(budget, "local_axes") else float(budget)

        def check(eps, low, high, step, *parameters, count=int(count), budget=local):
            pole_values = np.asarray(parameters, dtype=np.float64).reshape(-1, 3)
            eps, low, high, step = map(float, (eps, low, high, step))
            if not (np.all(np.isfinite([eps, low, high, step])) and np.all(np.isfinite(pole_values)) and eps > 0):
                return   # invalid cells are reported by the checks that own them
            if step * step * budget > eps * (1.0 + ROOT_TOLERANCE):
                return   # the step is above the Courant limit of these cells without any pole: not the poles' doing
            growth = max(largest_growth(eps, s, pole_values, step, 4.0 * step * step * budget)
                         for s in (low, high))
            if growth <= 1.0 + ROOT_TOLERANCE:
                return
            limits = [largest_stable_step(eps, s, pole_values, step, budget) for s in (low, high)]
            stable = None if any(v is None for v in limits) else min(limits)
            if stable is None:
                limit = "no stable step found"
                remedy = "use finer cells; or raise eps_inf"
            else:
                limit = (f"largest stable step for this set is {stable:.16g} s; "
                         f"dt/stable = {step / stable:.6g}")
                remedy = (f"pass dt={stable:.16g}" if grid_kind == "NonUniformGrid" else
                          "use finer cells, or declare a dz_profile/dx_profile/dy_profile "
                          f"so that dt={stable:.16g} can be passed") + "; or raise eps_inf"
            names = ", ".join(f"(omega_0={w:.6g}, delta={d:.6g}, kappa={k:.6g})"
                              for w, d, k in pole_values)
            raise ValueError(
                f"Lorentz/Drude poles {names} on {count} cells, eps_low={eps:.6g}, "
                f"sigma_low={low:.6g}, sigma_high={high:.6g}: unstable at this model's time step "
                f"dt = {step:.6g} s; growth per step = {growth:.9g}; {limit}. To run: {remedy}.")

        args = (record.eps_low, record.sigma_low, record.sigma_high, dt,
                *(v for p in record.poles for v in p))
        # Eager autodiff carries concrete primals: judge them on the host. A
        # callback would copy a value replicated over processes to one device.
        args = tuple(_concrete(v) for v in args)
        traced_check("Lorentz/Drude poles", "time-step stability", check, *args, ordered=True)


def refuse_unstable_cells(materials, lorentz_spec, dt, budget, *, grid_kind):
    """Judge whole-cell pole patterns, retaining the legacy entry point."""
    if lorentz_spec is None or budget is None:
        return
    judge_pole_sets(pole_set_extremes(materials, lorentz_spec), dt, budget, grid_kind=grid_kind)

"""Admission of the explicit Lorentz update on realized cell pole patterns.

The state is (e, h, p_1, q_1, ...), with polarization in eps0 units and
q the previous polarization. Cell patterns use their lowest epsilon and
both conductivity extrema, including Debye polarizations in the same cells.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

C0 = 299792458.0
EPS_0 = 8.8541878128e-12
ROOT_TOLERANCE = 1e-6
X_SAMPLES = 129


def largest_growth(eps_inf, sigma, poles, dt, x_max, *, debye=(), x_min=0.0):
    """Maximum spectral radius of the actual combined ADE update.

    State: (e, h, p_1, q_1, ..., r_1, ...), with Lorentz p=P/eps0,
    q=p at the previous step, Debye r=P_D/eps0, and h=dt curl(H)/eps0.
    Faraday gives h'=h-x e, x=4(c0 dt)^2 sum sin(k_a d_a/2)^2/d_a^2.
    ``lorentz_p_component`` gives p'=A p+B q+C e, q'=p, where
    A=(2-w0^2 dt^2)/(1+delta dt), B=-(1-delta dt)/(1+delta dt),
    C=kappa dt^2/(1+delta dt). ``debye_e_component`` and the mixed update
    in ``uniform_setup`` use a=(2 tau-dt)/(2 tau+dt), b=de dt/(2 tau+dt):
    gamma=eps_inf+sigma dt/(2 eps0)+sum b,
    gamma e'=(eps_inf-sigma dt/(2 eps0)-sum b)e+h'
             -sum(p'-p)+sum((1-a)r), and r'=a r+b(e'+e).
    These are also the Debye-only rows when there are no Lorentz poles.
    The default samples the unchanged 129 modes; x_min=x_max isolates one
    Fourier mode for the independent time-domain kernel witness.
    """
    dt = float(dt)
    poles = [(float(w), float(d), float(k)) for w, d, k in poles if float(k) != 0.0]
    beta = float(sigma) * dt / (2.0 * EPS_0)
    a_eps, b_eps = float(eps_inf) + beta, float(eps_inf) - beta
    debye = [(float(de), float(tau)) for de, tau in debye if float(de) != 0.0]
    terms = [((2 * tau - dt) / (2 * tau + dt), de * dt / (2 * tau + dt))
             for de, tau in debye]
    a_eps += sum(b for a, b in terms)
    b_eps -= sum(b for a, b in terms)
    size = 2 + 2 * len(poles) + len(terms)
    matrix = np.zeros((X_SAMPLES, size, size), dtype=np.float64)
    matrix[:, 1, 0] = -np.linspace(x_min, x_max, X_SAMPLES)
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
    start = 2 + 2 * len(poles)
    for i, (a, b) in enumerate(terms, start):
        matrix[:, 0, i] = 1 - a
    matrix[:, 0, :] /= a_eps
    for i, (a, b) in enumerate(terms, start):
        matrix[:, i, :] = b * matrix[:, 0, :]
        matrix[:, i, 0] += b
        matrix[:, i, i] += a
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
        if any(_is_abstract(v) for v in occupied_per_axis):
            return _ArrayBudget(self.cells).local_axes(occupied_per_axis)
        total = 0.0
        for occupied, sizes in zip(occupied_per_axis, self.cells):
            occupied = np.asarray(_concrete(occupied), dtype=bool)
            if sizes.size == occupied.size - 1:      # cell metrics omit the terminal node
                sizes = np.append(sizes, sizes[-1])
            if sizes.size != occupied.size or not occupied.any():
                return float(self)
            near = occupied.copy()
            near[1:] |= occupied[:-1]
            near[:-1] |= occupied[1:]
            total += 1.0 / float(np.min(sizes[near])) ** 2
        return C0 * C0 * total


class _ArrayBudget:
    """Local metric reductions when the cell spine or occupancy is abstract."""

    def __init__(self, cells):
        self.cells = cells

    def local_axes(self, occupied_per_axis):
        import jax.numpy as jnp
        total = 0.
        for occupied, sizes in zip(occupied_per_axis, self.cells):
            occupied, sizes = jnp.asarray(occupied), jnp.atleast_1d(sizes)
            if sizes.size == occupied.size - 1:
                sizes = jnp.concatenate((sizes, sizes[-1:]))
            if sizes.size != occupied.size:
                raise ValueError("pole stability: occupied axes and cell sizes do not align")
            near = occupied | jnp.concatenate((occupied[1:], jnp.array([False])))
            near |= jnp.concatenate((jnp.array([False]), occupied[:-1]))
            total += 1. / jnp.min(jnp.where(near, sizes, jnp.inf)) ** 2
        return C0 * C0 * total


def _is_abstract(value):
    from rfx.core.jax_utils import is_tracer
    return is_tracer(_concrete(value))


def courant_budget(grid):
    """The grid's :class:`CourantBudget` (two axes on a 2-D grid)."""
    axes = (0, 1) if getattr(grid, "is_2d", False) else (0, 1, 2)
    if not callable(getattr(grid, "cells", None)):
        return None   # a stand-in without per-axis cells (unit rigs of the layer functions) is not judged
    cells = tuple(_concrete(grid.cells(a)) for a in axes)
    if any(_is_abstract(c) for c in cells):
        return _ArrayBudget(cells)
    return CourantBudget([_cells(grid, a) for a in axes])


def _cells(grid, axis):
    return np.atleast_1d(np.asarray(_concrete(grid.cells(axis)), dtype=np.float64))


def largest_stable_step(eps_inf, sigma, poles, dt, budget, *, debye=()):
    """Bisect the stable step, or return None if the lower bracket is unstable."""
    def stable(step):
        return largest_growth(eps_inf, sigma, poles, step, 4.0 * step * step * budget,
                              debye=debye) <= 1.0 + ROOT_TOLERANCE
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
    count: object
    occupied: tuple
    debye: tuple = ()


def pole_set_extremes(materials, lorentz_spec, *, view=None, debye_spec=None):
    """Reduce static pole patterns over owned cells only; never count halos."""
    import jax.numpy as jnp
    from rfx.core.jax_utils import is_tracer

    if lorentz_spec is None and debye_spec is None:
        return ()
    lp, lm = lorentz_spec if lorentz_spec is not None else ((), ())
    dp, dm = debye_spec if debye_spec is not None else ((), ())
    def expand(poles, masks):
        masks = masks if isinstance(masks, (tuple, list)) else [masks] * len(poles)
        if len(masks) != len(poles):
            raise ValueError(f"Expected {len(poles)} pole masks, got {len(masks)}")
        return [_concrete(m) for m in masks]
    poles, masks = tuple(lp) + tuple(dp), expand(lp, lm) + expand(dp, dm)
    if not poles:
        return ()
    masks = masks if isinstance(masks, (tuple, list)) else [masks] * len(poles)
    if len(masks) != len(poles):
        raise ValueError(f"Expected {len(poles)} Lorentz masks, got {len(masks)}")
    abstract_masks = any(is_tracer(m) for m in masks)
    if abstract_masks and len(poles) > 8:
        raise ValueError(f"pole stability cannot judge: {len(poles)} poles give 2^{len(poles)} patterns")
    if view is not None:
        from types import SimpleNamespace
        input_shape = materials.eps_r.shape
        materials = SimpleNamespace(**{name: view.own(getattr(materials, name))
                                       for name in ("eps_r", "sigma")})
        masks = [None if m is None else view.own((jnp if abstract_masks else np).broadcast_to(m, input_shape)) for m in masks]
    shape = materials.eps_r.shape
    # One integer code per cell, accumulated pole by pole (no cells x poles array).
    wide = len(poles) > 62
    codes = np.zeros(int(np.prod(shape)), dtype=object if wide else np.int64)
    for bit, m in enumerate(masks if not abstract_masks else ()):
        present = np.ones(shape, dtype=bool) if m is None else np.broadcast_to(np.asarray(m, dtype=bool), shape)
        codes = codes + (present.reshape(-1).astype(object if wide else np.int64) * (1 << bit))
    if abstract_masks:
        # The static spec fixes pole identities; only membership is dynamic.
        # Enumerate possible sets, reducing empty ones to count=0 at runtime.
        unique_codes = range(1, 1 << len(poles))
        inverse, counts = None, [None] * len(unique_codes)
    else:
        unique_codes, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    patterns = [[bool((int(code) >> bit) & 1) for bit in range(len(poles))] for code in unique_codes]
    values = [materials.eps_r, materials.sigma] + [v for p in poles for v in p]
    traced = abstract_masks or any(is_tracer(v) for v in values)
    # A global array spread over processes cannot be read on the host.
    spread = not traced and any(not getattr(v, "is_fully_addressable", True) for v in values)
    xp = jnp if traced or spread else np
    # A concrete sigma keeps concrete extremes when only eps is traced (the passive-Debye skip reads them).
    sigma = values[1]
    xs = np if not (abstract_masks or is_tracer(sigma)
                    or not getattr(sigma, "is_fully_addressable", True)) else xp
    records = []
    for index, (pattern, count) in enumerate(zip(patterns, counts)):
        if not np.any(pattern):
            continue
        if abstract_masks:
            selected = jnp.ones(shape, dtype=bool)
            for m, present in zip(masks, pattern):
                active = jnp.ones(shape, dtype=bool) if m is None else jnp.broadcast_to(m, shape).astype(bool)
                selected &= active if present else ~active
            count = jnp.sum(selected)
        else:
            selected = (inverse == index).reshape(shape)
        chosen = tuple(p for p, present in zip(lp, pattern) if present)
        debye = tuple(p for p, present in zip(dp, pattern[len(lp):]) if present)
        eps_low = xp.min(xp.where(selected, materials.eps_r, xp.inf))
        sigma_low = xs.min(xs.where(selected, materials.sigma, xs.inf))
        sigma_high = xs.max(xs.where(selected, materials.sigma, -xs.inf))

        occupied = tuple((jnp if abstract_masks else np).any(selected, axis=tuple(a for a in range(selected.ndim) if a != axis))
                         for axis in range(selected.ndim))
        if view is not None:
            global_x = jnp.zeros(view.nx, dtype=bool).at[view.x0:view.x1].set(occupied[0]) if abstract_masks else np.zeros(view.nx, dtype=bool)
            if not abstract_masks:
                global_x[view.x0:view.x1] = occupied[0]
            occupied = (global_x,) + occupied[1:]
        records.append(PoleSetExtremes(int(unique_codes[index]), chosen, eps_low,
                                      sigma_low, sigma_high, count, occupied, debye))
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
            # Counts and occupancy follow the extremes: a jax operator on concrete
            # records under an outer jit would stage them, and the judge would refuse.
            values = (previous.eps_low, previous.sigma_low, previous.sigma_high,
                      record.eps_low, record.sigma_low, record.sigma_high,
                      previous.count, record.count, *previous.occupied, *record.occupied)
            xp = jnp if any(is_tracer(v) for v in values) else np
            combined[record.code] = previous._replace(
                eps_low=xp.minimum(previous.eps_low, record.eps_low),
                sigma_low=xp.minimum(previous.sigma_low, record.sigma_low),
                sigma_high=xp.maximum(previous.sigma_high, record.sigma_high),
                count=xp.add(previous.count, record.count),
                occupied=tuple(xp.logical_or(a, b) for a, b in zip(previous.occupied, record.occupied)))
    return tuple(combined[code] for code in sorted(combined))


def _concrete(value):
    """The concrete value under an eager trace (jvp, grad), else the value."""
    from rfx.core.jax_utils import is_tracer

    if not is_tracer(value):
        return value
    concrete = value.to_concrete_value()
    return value if concrete is None else concrete


def _refuse_traced_out_of_program():
    raise ValueError(
        "cannot judge Lorentz/Drude time-step stability: this multi-device run's material values or "
        "cell sizes are traced by an outer jax.jit (or a nested jax.vmap), and a check cannot be raised "
        "from inside a multi-device program. Call it without jax.jit (jax.grad, jax.jvp and one "
        "jax.vmap alone keep concrete values and are judged), or run on one device.")


def _eager_batch_members(args):
    """Each member's concrete arguments under one eager ``jax.vmap``, else None.

    An eager vmap carries concrete values (``BatchTracer.val``); a vmap under
    ``jit`` or a nested vmap does not, and returns None.
    """
    from jax._src.interpreters.batching import BatchTracer

    batched = [v for v in args if isinstance(v, BatchTracer) and v.batch_dim is not None]
    if not batched or any(v._trace is not batched[0]._trace for v in batched):
        return None
    if any(_is_abstract(v) and not isinstance(v, BatchTracer) for v in args) or \
            any(_is_abstract(v.val) for v in args if isinstance(v, BatchTracer)):
        return None
    size = batched[0].val.shape[batched[0].batch_dim]

    def member(v, i):
        if not isinstance(v, BatchTracer):
            return v
        value = np.asarray(_concrete(v.val))   # an eager grad/jvp around the vmap wraps the batch
        return value if v.batch_dim is None else np.take(value, i, axis=v.batch_dim)
    return [tuple(member(v, i) for v in args) for i in range(size)]


def _passive_debye(record):
    """Concrete Debye poles with delta_eps >= 0, tau > 0 and a concrete sigma_low >= 0."""
    values = [_concrete(record.sigma_low)] + [_concrete(v) for p in record.debye for v in p]
    if any(_is_abstract(v) for v in values):
        return False
    sigma, *flat = (float(v) for v in values)
    return sigma >= 0 and all(de >= 0 and tau > 0 for de, tau in zip(flat[::2], flat[1::2]))


def judge_pole_sets(records, dt, budget, *, grid_kind, in_program=True):
    """Judge concrete values; stage checks only in single-device programs.

    Out of program (multi-device) the members of an eager ``jax.vmap`` are judged
    one by one; any other traced value is refused: it cannot be judged there.
    """
    from rfx._traced_check import traced_check

    if budget is None:
        return
    for record in records:
        local = budget.local_axes(record.occupied) if hasattr(budget, "local_axes") else float(budget)

        def check(eps, low, high, step, budget, count, *parameters, n_lorentz=len(record.poles)):
            pole_values = np.asarray(parameters[:3 * n_lorentz], dtype=np.float64).reshape(-1, 3)
            debye = np.asarray(parameters[3 * n_lorentz:], dtype=np.float64).reshape(-1, 2)
            eps, low, high, step = map(float, (eps, low, high, step))
            count, budget = int(count), float(budget)
            if not count:
                return
            if not (np.all(np.isfinite([eps, low, high, step])) and np.all(np.isfinite(pole_values)) and eps > 0):
                return   # invalid cells are reported by the checks that own them
            if step * step * budget > eps * (1.0 + ROOT_TOLERANCE):
                return   # the step is above the Courant limit of these cells without any pole: not the poles' doing
            growth = max(largest_growth(eps, s, pole_values, step, 4.0 * step * step * budget, debye=debye)
                         for s in (low, high))
            if growth <= 1.0 + ROOT_TOLERANCE:
                return
            limits = [largest_stable_step(eps, s, pole_values, step, budget, debye=debye) for s in (low, high)]
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

        args = (record.eps_low, record.sigma_low, record.sigma_high, dt, local, record.count,
                *(v for p in record.poles + record.debye for v in p))
        # Eager autodiff carries concrete primals: judge them on the host. A
        # callback would copy a value replicated over processes to one device.
        args = tuple(_concrete(v) for v in args)
        if not in_program and any(_is_abstract(v) for v in args):
            if not record.poles and _passive_debye(record):
                # A passive Debye-only set (delta_eps >= 0, tau > 0, sigma >= 0) grows at most 1 inside the
                # check's Courant pre-filter (trapezoidal update), so the check cannot refuse it.
                continue
            members = _eager_batch_members(args)
            if members is None:
                _refuse_traced_out_of_program()
            for member in members:
                check(*member)
            continue
        traced_check("Lorentz/Drude poles", "time-step stability", check, *args, ordered=True)


def refuse_unstable_cells(materials, lorentz_spec, dt, budget, *, grid_kind, debye_spec=None,
                          in_program=True):
    """Judge whole-cell pole patterns, retaining the legacy entry point."""
    if (lorentz_spec is None and debye_spec is None) or budget is None:
        return
    judge_pole_sets(pole_set_extremes(materials, lorentz_spec, debye_spec=debye_spec),
                    dt, budget, grid_kind=grid_kind, in_program=in_program)


def _prebuilt_cell_error(edges, component, precision):
    transverse = np.prod([edges.shape[a] for a in range(3) if a != component])
    return 8 * precision * transverse * np.max(abs(edges))


def _check_prebuilt_cells(values, edges, component, periodic, precision, *, positive=False,
                          invalid_message=None, high_wall: tuple[bool, bool, bool]):
    """Refuse uncertain positivity or a reconstruction that changes an edge."""
    from rfx.core.yee import _edge_mean_component
    error = _prebuilt_cell_error(edges, component, precision)
    if not np.all(np.isfinite(values)) or np.any(values <= error if positive else values < -error):
        raise ValueError(invalid_message or (
            "cannot judge prebuilt Lorentz coefficients: recovered cells are not "
            + ("finite and positive" if positive else "finite and nonnegative")))
    values = np.maximum(values, 0.)
    axes = [a for a in range(3) if a != component]
    averaged = _edge_mean_component(values, *axes, periodic, None, array_module=np, high_wall=high_wall)
    if not np.all(np.isfinite(averaged)) or np.any(abs(averaged - edges) > error):
        raise ValueError(f"cannot judge prebuilt Lorentz coefficients: recovered cells do not reproduce component {component} edges within rounding")
    return values


def _unaverage_prebuilt(edges, component, periodic, precision, *, positive=False, high_wall: tuple[bool, bool, bool]):
    """Invert clipped means or the odd periodic cyclic system, then check it.

    Along a periodic axis, y[i]+y[i-1]=2*edge[i], including the seam.
    Odd lengths fix y[0] from that seam. Even lengths have an alternating
    null mode: constant rows are retained; other rows are refused after the
    re-average check (the zero-null-mode candidate also detects inconsistent
    edges). No unchecked candidate is handed to the pole judge.
    """
    if not np.all(np.isfinite(edges)):
        raise ValueError("cannot judge prebuilt Lorentz coefficients: non-finite edge values")
    values = np.array(edges, dtype=np.float64, copy=True)
    ambiguous = None
    for axis in reversed([a for a in range(3) if a != component]):
        rows = np.moveaxis(values, axis, 0)
        first = rows[0].copy()
        if periodic[axis] and np.all(rows == first):
            continue
        if periodic[axis]:
            rows[0] = 0.
        for i in range(1, len(rows)):
            rows[i] = 2. * rows[i] - rows[i - 1]
        if periodic[axis]:
            signs = np.where(np.arange(len(rows)) % 2, -1., 1.).reshape((-1, 1, 1))
            if len(rows) % 2:
                seed = first - .5 * rows[-1]
            else:
                seed = -np.mean(signs * rows, axis=0)
                ambiguous = axis
            rows += signs * seed
    ambiguity = (f"cannot judge prebuilt Lorentz coefficients: periodic edge mean on even axis {ambiguous} does not determine unique cells"
                 if ambiguous is not None else None)
    values = _check_prebuilt_cells(values, edges, component, periodic, precision,
                                  positive=positive, invalid_message=ambiguity, high_wall=high_wall)
    if ambiguity is not None:
        raise ValueError(ambiguity)
    return values


def judge_prebuilt_poles(grid, lorentz, debye=None, *, periodic=(False, False, False),
                         in_program=True):
    """Judge prebuilt component ADE terms, without assuming their source spec.

    Invert the coefficients actually used: gamma=1/cc, eps=(1+ca)gamma/2,
    sigma=(1-ca)gamma/dt; 1+delta dt=2/(1-b), w0^2 dt^2=2-a(1+delta dt),
    kappa=c(1+delta dt)/(eps0 dt^2). Debye tau and delta_eps follow from
    alpha and beta by the same inversion. Undo backward edge means with the
    run's periodic flags, check their round trip, then group pole parameters, reducing
    eps/sigma over their occupied cells and neighbours, never a global surrogate.
    """
    if lorentz is None:
        return
    from rfx._traced_check import traced_check
    from rfx.materials.debye import per_component
    from rfx.geometry.rasterize_grid import unpadded_high_walls
    high_wall = unpadded_high_walls(grid)
    cells = getattr(grid, "cells", None)
    if not callable(cells) or getattr(grid, "dt", None) is None:
        raise ValueError("prebuilt Lorentz coefficients require cell sizes and dt for pole stability admission")
    axes = (0, 1) if getattr(grid, "is_2d", False) else (0, 1, 2)
    lc = lorentz[0]
    dc = None if debye is None else debye[0]

    def check(step, widths, coeffs, debye_coeffs):
        step = float(step)
        budget = CourantBudget(widths)
        groups = {}
        for component in range(3):
            ca = np.asarray(per_component(coeffs.ca, "ca")[component])
            cc = np.asarray(per_component(coeffs.cc, "cc")[component])
            precision = max(np.finfo(ca.dtype).eps, np.finfo(cc.dtype).eps)
            ca = ca.astype(np.float64)
            a, b = np.asarray(coeffs.a), np.asarray(coeffs.b)
            c = np.asarray(per_component(coeffs.c, "c")[component])
            shape, n = ca.shape, a.shape[0]
            def unaverage(values, *, positive=False):
                return _unaverage_prebuilt(values, component, periodic, precision, positive=positive, high_wall=high_wall)

            def pole_cells(values):
                result = unaverage(values)
                scale = np.max(abs(result))
                if not scale:
                    return result
                # Multiplication by a three-quarter fraction rounds once.
                # Undo that roundoff, rather than inventing tiny extra poles
                # in empty cells. Only a recovered binary cell mask qualifies.
                binary = np.rint(result / scale)
                transverse = np.prod([shape[t] for t in range(3) if t != component])
                error = 4 * np.finfo(values.dtype).eps * transverse * scale
                if np.all(abs(result - binary * scale) <= error) and np.all((binary == 0) | (binary == 1)):
                    quarters = np.rint(4 * values / scale)
                    for q in (4, 2, 1):
                        if np.any(quarters == q):
                            scale = np.median(values[quarters == q]) * (4 / q)
                            break
                    result = binary * scale
                return _check_prebuilt_cells(result, values, component, periodic, precision, high_wall=high_wall)

            with np.errstate(divide="ignore", invalid="ignore"):
                eps = unaverage((1. + ca) / (2. * cc * EPS_0), positive=True)
                sigma = unaverage((1. - ca) / (cc * step))
            cell_c = np.stack([pole_cells(v) for v in c])
            columns = [v.reshape(-1) for terms in (a, b, cell_c) for v in terms]
            nd = 0 if debye_coeffs is None else debye_coeffs.alpha.shape[0]
            if nd:
                beta = per_component(debye_coeffs.beta, "beta")[component]
                cell_beta = np.stack([pole_cells(v) for v in beta])
                columns.extend(np.asarray(v).reshape(-1) for terms in
                               (debye_coeffs.alpha, cell_beta) for v in terms)
            rows, inverse = np.unique(np.stack(columns, axis=1), axis=0, return_inverse=True)
            for index, row in enumerate(rows):
                av, bv, cv = row[:3 * n].reshape(3, n)
                present = cv != 0.
                if not np.any(present):
                    continue
                av, bv, cv = av[present], bv[present], cv[present]
                denom = 2. / (1. - bv)
                poles = tuple(zip(np.sqrt(np.maximum(2. - av * denom, 0.)) / step,
                                  (denom - 1.) / step, cv * denom / (EPS_0 * step * step)))
                selected = (inverse == index).reshape(shape)
                dp = ()
                if nd:
                    alpha, beta = row[3 * n:].reshape(2, nd)
                    present = beta != 0.
                    dp = tuple(zip(2. * beta[present] / (EPS_0 * (1. - alpha[present])),
                                   step * (1. + alpha[present]) / (2. * (1. - alpha[present]))))
                if not np.all(np.isfinite([v for p in poles + dp for v in p])):
                    raise ValueError("cannot judge prebuilt Lorentz coefficients: non-finite recovered pole parameters")
                key = (poles, dp)
                low, sig_low, sig_high = np.min(eps[selected]), np.min(sigma[selected]), np.max(sigma[selected])
                if key in groups:
                    previous, e, lo, hi = groups[key]
                    groups[key] = (previous | selected, min(e, low), min(lo, sig_low), max(hi, sig_high))
                else:
                    groups[key] = (selected, low, sig_low, sig_high)
        records = []
        for code, ((poles, dp), (selected, eps, low, high)) in enumerate(groups.items()):
            occupied = tuple(np.any(selected, axis=tuple(a for a in range(3) if a != axis))
                             for axis in range(3))
            records.append(PoleSetExtremes(code, poles, eps, low, high,
                                          np.count_nonzero(selected), occupied, dp))
        judge_pole_sets(records, step, budget, grid_kind=type(grid).__name__, in_program=in_program)

    import jax
    operands = (grid.dt, tuple(cells(a) for a in axes), lc, dc)
    operands = jax.tree.map(_concrete, operands)
    if not in_program and any(_is_abstract(v) for v in jax.tree.leaves(operands)):
        _refuse_traced_out_of_program()
    traced_check("prebuilt Lorentz coefficients", "time-step stability", check, *operands, ordered=True)

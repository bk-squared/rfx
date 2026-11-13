"""Dispersive pole coefficients and initial state, built from the realized material.

The pole declarations stay in rfx.materials.debye and rfx.materials.lorentz; the per-step pole updates are time-step code and do not belong here.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TYPE_CHECKING
from weakref import ref
import os
import warnings

import jax
import jax.numpy as jnp

from rfx.core.yee import (
    si_value_eps_r_grad, EPS_0, MaterialArrays, ade_state_dtype,
)

from rfx.materials.relative_strength import RelativeStrength
from rfx.errors import AccuracyWarning

if TYPE_CHECKING:
    from rfx.materials.debye import DebyePole
    from rfx.materials.lorentz import LorentzPole


class DebyeCoeffs(NamedTuple):
    """Precomputed ADE update coefficients for all Debye poles.

    Per E component -- each a 3-tuple ``(x, y, z)``:

    ca : (nx, ny, nz) each — E decay coefficient
    cb : (nx, ny, nz) each — curl(H) coupling coefficient
    cc : (n_poles, nx, ny, nz) each — P^n coupling into E update
    beta : (n_poles, nx, ny, nz) each — E coupling into P update

    One array for all components:

    alpha : (n_poles, nx, ny, nz) — P decay coefficient; depends on τ_p only,
        and is the pole's value on every cell whose edges the pole reaches
        (where an edge carries no pole its ``beta`` is 0 and its P stays 0).
    """
    ca: tuple
    cb: tuple
    cc: tuple            # 3 x (n_poles, nx, ny, nz)
    alpha: jnp.ndarray   # (n_poles, nx, ny, nz)
    beta: tuple          # 3 x (n_poles, nx, ny, nz)


def per_component(field, name="coefficient"):
    """``field`` as the ``(x, y, z)`` tuple the per-component update needs.

    A bare grid array here is a coefficient built by the old cell-owned rule
    (one value per cell for all three components) -- indexing it by
    component would silently take a y-z plane and broadcast it, so it is
    refused instead.
    """
    if isinstance(field, (tuple, list)) and len(field) == 3:
        return field
    raise TypeError(
        f"dispersive {name} must be a per-E-component 3-tuple (x, y, z) "
        f"built by init_debye / init_lorentz; got "
        f"{type(field).__name__}")


def pole_edge_fractions(mask, periodic=(False, False, False), *, high_wall: tuple[bool, bool, bool]):
    """Per E component, the fraction of each edge's four cells that carry a
    pole: ``edge_mean_components`` of the pole's cell mask. ``None``
    (the pole everywhere) stays ``None``."""
    from rfx.model.materials import pole_component_weights
    return pole_component_weights(mask, periodic, high_wall=high_wall)


def pole_reach(fractions):
    """Cells at which ANY component's edge carries some of the pole."""
    return (fractions[0] > 0) | (fractions[1] > 0) | (fractions[2] > 0)


class DebyeState(NamedTuple):
    """Auxiliary polarization fields for Debye ADE.

    Each pole has 3 polarization components (px, py, pz).
    Stored as (n_poles, nx, ny, nz) for each component.
    """
    px: jnp.ndarray  # (n_poles, nx, ny, nz)
    py: jnp.ndarray
    pz: jnp.ndarray


def debye_pole_coeffs(poles, dt, shape, fractions_by_pole):
    """Material-independent Debye ADE terms from per-pole edge fractions."""
    # Per-pole coefficients
    alpha_list = []
    beta_lists = ([], [], [])
    for pole, fractions in zip(poles, fractions_by_pole):
        tau = pole.tau
        de = pole.delta_eps
        a = (2.0 * tau - dt) / (2.0 * tau + dt)
        b = EPS_0 * de * dt / (2.0 * tau + dt)

        if fractions is not None:
            alpha_list.append(jnp.where(pole_reach(fractions), a, 0.0))
            for c in range(3):
                beta_lists[c].append(
                    jnp.where(fractions[c] > 0, b * fractions[c], 0.0))
        else:
            # No dtype pin — see the matching note in
            # ``rfx.materials.lorentz.init_lorentz``: ``a``/``b``
            # are numpy scalars (``dt`` is ``grid.dt``, an np.float64), so
            # this matches the masked branch above instead of capping the
            # ADE coefficients at float32 under ``precision="float64"``.
            # With x64 off JAX clamps to float32: default lane unchanged.
            alpha_list.append(jnp.full(shape, a))
            b_arr = jnp.full(shape, b)
            for c in range(3):
                beta_lists[c].append(b_arr)

    alpha = jnp.stack(alpha_list)  # (n_poles, nx, ny, nz)
    beta = tuple(jnp.stack(bl) for bl in beta_lists)

    return alpha, beta


def _debye_e_coeffs_si(e_materials, dt, alpha, beta):
    """Build E coefficients from the component means and fixed ADE terms.

    Distributed runners form these inside the time loop so autodiff does
    not accumulate the coefficient cotangents before their reverse pass.
    """
    eps_c, sig_c = e_materials
    n_poles = alpha.shape[0]
    ca, cb, cc = [], [], []
    for c in range(3):
        eps_inf = eps_c[c] * EPS_0
        sigma = sig_c[c]
        # Sum of beta across poles
        beta_sum = jnp.sum(beta[c], axis=0)
        # Modified update coefficients
        gamma = eps_inf + beta_sum + sigma * dt / 2.0
        # Guard against zero (vacuum cells with no Debye)
        safe_gamma = jnp.maximum(gamma, EPS_0 * 1e-10)
        ca.append((eps_inf - beta_sum - sigma * dt / 2.0) / safe_gamma)
        cb.append(dt / safe_gamma)
        # Cc for each pole: (1 - alpha_p) / gamma
        cc.append(jnp.stack([(1.0 - alpha[p]) / safe_gamma
                             for p in range(n_poles)]))

    return DebyeCoeffs(ca=tuple(ca), cb=tuple(cb), cc=tuple(cc),
                         alpha=alpha, beta=beta)


def _debye_e_coeffs_eps_r(e_materials, dt, alpha, beta):
    """The Debye coefficients in relative-permittivity units."""
    eps_c, sig_c = e_materials
    n_poles = alpha.shape[0]
    ca, cb, cc = [], [], []
    for c in range(3):
        eps_inf = eps_c[c]
        sigma = sig_c[c]
        # Sum of beta across poles
        beta_sum = jnp.sum(beta[c], axis=0) / EPS_0
        # Modified update coefficients
        gamma = eps_inf + beta_sum + sigma * (dt / (2.0 * EPS_0))
        # Guard against zero (vacuum cells with no Debye)
        safe_gamma = jnp.maximum(gamma, 1e-10)
        ca.append((eps_inf - beta_sum - sigma * (dt / (2.0 * EPS_0))) / safe_gamma)
        cb.append((dt / EPS_0) / safe_gamma)
        # Cc for each pole: (1 - alpha_p) / gamma
        cc.append(jnp.stack([((1.0 - alpha[p]) / EPS_0) / safe_gamma
                             for p in range(n_poles)]))

    return DebyeCoeffs(ca=tuple(ca), cb=tuple(cb), cc=tuple(cc),
                         alpha=alpha, beta=beta)


def debye_e_coeffs(e_materials, dt, alpha, beta):
    """Build component E coefficients with SI values and eps_r derivatives.

    Shared by init_debye and the distributed in-loop coefficient builders.
    """
    return si_value_eps_r_grad(
        _debye_e_coeffs_si, _debye_e_coeffs_eps_r, e_materials, dt, alpha, beta)


def init_debye(
    poles: list[DebyePole],
    materials: MaterialArrays,
    dt: float,
    mask: jnp.ndarray | list[jnp.ndarray] | tuple[jnp.ndarray, ...] | None = None,
    *,
    field_dtype=None,
    periodic=(False, False, False),
    grid=None,
    high_wall=None,
) -> tuple[DebyeCoeffs, DebyeState]:
    """Initialize Debye ADE coefficients and auxiliary state.

    Parameters
    ----------
    poles : list of DebyePole
        Debye relaxation poles.
    materials : MaterialArrays
        Base material arrays (eps_r = ε_∞, sigma, mu_r) and the lumped-stamp
        records.
    dt : float
        Timestep in seconds.
    mask : (nx, ny, nz) bool array or per-pole mask list, optional
        Where to apply Debye dispersion. If a list/tuple is provided,
        it must align one-to-one with ``poles``.
    field_dtype : jnp dtype, optional
        Dtype of the E/H field storage this ADE state will be driven by.
        The P carry is allocated at ``ade_state_dtype(field_dtype)`` —
        ``promote_types(field_dtype, float32)``. ``None`` (the default, for
        callers that do not thread it) means the ambient default float with
        the same float32 floor. This used to be a hard ``dtype=jnp.float32``
        pin, which made ``precision="float64"`` + any pole fail the
        ``lax.scan`` carry contract.
    periodic : per-axis flags
        The run's periodic flags, as its non-dispersive update takes them: a
        periodic axis wraps the edge average, any other replicates the
        boundary cell (``rfx.core.yee._material_bwd_neighbour``).

    Returns
    -------
    coeffs : DebyeCoeffs
        ``ca``/``cb``/``cc``/``beta`` per E component (see the class).
    state : DebyeState

    The rule: per component, ε_∞ and σ are the edge mean of
    :func:`rfx.core.yee.component_e_materials`, and each pole's Δε_p the edge
    mean of the cells' Δε_p (``delta_eps`` times the fraction of the four
    cells in the pole's mask). Homogeneous regions give the bit pattern the
    one-per-cell rule gave: the four summands are equal and the fraction is
    exactly 1.
    """
    shape = materials.eps_r.shape
    n_poles = len(poles)

    from rfx.model.materials import validate_dispersion, with_components
    components = with_components(materials, grid, periodic=periodic, high_wall=high_wall,
        debye_spec=(poles, mask)).components
    eps_c, sig_c = components.eps_update, components.sigma_update
    realized = components.debye
    if realized is None or len(realized.poles) != n_poles:
        raise ValueError("realized debye poles do not match the requested poles")
    validate_dispersion(realized, poles, dt)
    alpha, beta = (realized.coefficients if realized.coefficients is not None else
                  debye_pole_coeffs(poles, dt, shape, realized.weights))
    coeffs = debye_e_coeffs((eps_c, sig_c), dt, alpha, beta)

    # Zero-initialized polarization state
    p_zeros = jnp.zeros((n_poles,) + shape, dtype=ade_state_dtype(field_dtype))
    state = DebyeState(px=p_zeros, py=p_zeros.copy(), pz=p_zeros.copy())

    return coeffs, state


class LorentzCoeffs(NamedTuple):
    """Precomputed ADE coefficients for Lorentz/Drude poles.

    Per E component -- each a 3-tuple ``(x, y, z)``:

    ca, cb : (nx, ny, nz) each — E update coefficients
    cc : (nx, ny, nz) each — P→E coupling: 1/γ
    c : (n_poles, nx, ny, nz) each — E^n coupling into the P recurrence (∝ κ)

    One array for all components, or component triples for a damping map:

    damping, restoring : (n_poles, nx, ny, nz) — increment coefficients (ω₀, δ only),
        the pole's values on every cell whose edges the pole reaches.
    """
    ca: tuple
    cb: tuple
    damping: jnp.ndarray | tuple
    restoring: jnp.ndarray | tuple
    c: tuple
    cc: tuple


_judged_pole_arrays = {}


def mark_judged(coeffs):
    """Record weak array identities without changing the coefficient pytree."""
    arrays = jax.tree.leaves(coeffs)
    key = tuple(id(a) for a in arrays)

    def remove(dead):
        recorded = _judged_pole_arrays.get(key)
        if recorded is not None and any(original is dead for original in recorded):
            _judged_pole_arrays.pop(key, None)

    _judged_pole_arrays[key] = tuple(ref(a, remove) for a in arrays)
    return coeffs


def pole_sets_judged(coeffs):
    """Only a complete set of original, already-judged array leaves qualifies."""
    arrays = jax.tree.leaves(coeffs)
    recorded = _judged_pole_arrays.get(tuple(id(a) for a in arrays))
    return recorded is not None and len(arrays) == len(recorded) and all(
        a is original() for a, original in zip(arrays, recorded))


class LorentzState(NamedTuple):
    """Auxiliary polarization and its per-step increment.

    p_{x,y,z} : (n_poles, nx, ny, nz) — P^n
    v_{x,y,z} : (n_poles, nx, ny, nz) — P^n - P^{n-1}
    """
    px: jnp.ndarray
    py: jnp.ndarray
    pz: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    vz: jnp.ndarray


def pole_underflow_guard(out, factors, *, scalar=True):
    """Identity solve whose transpose warns below the normal SI cotangent range.

    Scalar factors are (denom, dt**2, EPS_0); map normalization supplies
    (ks,) with scalar=False. Call sites opt in at trace time with
    ``RFX_POLE_UNDERFLOW_CHECK=1``: set it before the first call; a cached
    compile keeps the old setting. Off by default, float32 traced SI only.
    """
    args = tuple(jax.lax.stop_gradient(jnp.asarray(a)) for a in factors)

    def report(count):
        count = int(count)
        if count == 0:
            return
        warning: Any
        warning = AccuracyWarning(
            "The float32 reverse-mode gradient with respect to a traced pole strength "
            "in SI units fell below the smallest normal float32 value (1.2e-38): "
            "it is zero on a backend that flushes subnormal values and has lost precision "
            "on one that keeps them; declare the strength as "
            "RelativeStrength(reference, value) or divide the objective by a constant of its order.")
        warning.count = count
        warnings.warn(warning, stacklevel=2)

    def check(vecmat, ct):
        replay = ct / args[0]
        if scalar:
            replay = replay * args[1]
            replay = args[2] * replay
        flagged = (ct != 0) & (jnp.abs(replay) < jnp.finfo(jnp.float32).tiny)

        def warn(n):
            from rfx._traced_check import traced_check
            traced_check("RFX_POLE_UNDERFLOW_CHECK", "pole strength cotangent underflow", report, n)
            return jnp.int32(0)

        jax.lax.cond(jnp.any(flagged), warn, lambda n: jnp.int32(0), jnp.sum(flagged))
        return ct

    return jax.lax.custom_linear_solve(lambda x: x, out, lambda matvec, b: b,
                                       transpose_solve=check)


def pole_update_coeffs(omega_0, delta, kappa, dt, field_dtype=None):
    """Elementwise ADE increments, in the inputs' promoted arithmetic dtype."""
    relative = isinstance(kappa, RelativeStrength)
    si_tracer = isinstance(kappa, jax.core.Tracer) and kappa.dtype == jnp.float32
    if relative:
        from rfx._traced_check import traced_check
        from rfx.errors import RfxUnsupported
        import numpy as np

        if callable(kappa.value):
            raise RfxUnsupported("scalar Lorentz poles do not implement a callable RelativeStrength value")

        def check_value(value):
            if not np.all(np.isfinite(value)) or np.any(np.asarray(value) < 0):
                raise ValueError("RelativeStrength value must be finite and non-negative")

        traced_check("RelativeStrength", "pole strength", check_value, kappa.value)
        s = float(EPS_0) * float(kappa.reference) * float(dt) ** 2
        kappa = kappa.value

    dtype = jnp.result_type(omega_0, delta, kappa, dt)
    omega_0, delta, kappa, dt = jnp.broadcast_arrays(
        *(jnp.asarray(value, dtype=dtype) for value in (omega_0, delta, kappa, dt)))
    denom = 1.0 + delta * dt
    if relative:
        strength = (s / denom) * kappa
    else:
        strength = EPS_0 * kappa * dt ** 2 / denom
        if si_tracer and os.environ.get("RFX_POLE_UNDERFLOW_CHECK") == "1":
            strength = pole_underflow_guard(strength, (denom, dt ** 2, EPS_0))
    terms = (2.0 * delta * dt / denom,
             omega_0 ** 2 * dt ** 2 / denom,
             strength)
    if field_dtype is not None:
        terms = tuple(value.astype(ade_state_dtype(field_dtype)) for value in terms)
    return terms


def lorentz_pole_coeffs(poles, dt, shape, fractions_by_pole):
    """Material-independent Lorentz ADE terms from per-pole edge fractions."""
    from rfx.materials.pole_map import refuse_maps
    refuse_maps(poles, "grid-less lorentz_pole_coeffs")
    damping_list, restoring_list = [], []
    c_lists = ([], [], [])
    for pole, fractions in zip(poles, fractions_by_pole):
        damping, restoring, strength = pole_update_coeffs(
            pole.omega_0, pole.delta, pole.kappa, dt, field_dtype=None)
        if fractions is not None:
            reach = pole_reach(fractions)
            damping_list.append(jnp.where(reach, damping, 1.0))
            restoring_list.append(jnp.where(reach, restoring, 1.0))
            for comp in range(3):
                c_lists[comp].append(jnp.where(
                    fractions[comp] > 0, strength * fractions[comp], 0.0))
        else:
            damping_list.append(jnp.full(shape, damping))
            restoring_list.append(jnp.full(shape, restoring))
            for comp in range(3):
                c_lists[comp].append(jnp.full(shape, strength))
    return (jnp.stack(damping_list), jnp.stack(restoring_list),
            tuple(jnp.stack(values) for values in c_lists))


def _lorentz_e_coeffs_si(e_materials, dt, damping, restoring, c):
    """Build E coefficients from the component means and fixed ADE terms.

    Distributed runners form these inside the time loop so autodiff does
    not accumulate the coefficient cotangents before their reverse pass.
    """
    eps_c, sig_c = e_materials
    ca, cb, cc = [], [], []
    for comp in range(3):
        eps_inf = eps_c[comp] * EPS_0
        sigma = sig_c[comp]
        gamma = eps_inf + sigma * dt / 2.0
        safe_gamma = jnp.maximum(gamma, EPS_0 * 1e-10)
        ca.append((eps_inf - sigma * dt / 2.0) / safe_gamma)
        cb.append(dt / safe_gamma)
        cc.append(1.0 / safe_gamma)

    return LorentzCoeffs(ca=tuple(ca), cb=tuple(cb), damping=damping, restoring=restoring, c=c,
                           cc=tuple(cc))


def _lorentz_e_coeffs_eps_r(e_materials, dt, damping, restoring, c):
    """The Lorentz coefficients in relative-permittivity units."""
    eps_c, sig_c = e_materials
    ca, cb, cc = [], [], []
    for comp in range(3):
        eps_inf = eps_c[comp]
        sigma = sig_c[comp]
        gamma = eps_inf + sigma * (dt / (2.0 * EPS_0))
        safe_gamma = jnp.maximum(gamma, 1e-10)
        ca.append((eps_inf - sigma * (dt / (2.0 * EPS_0))) / safe_gamma)
        cb.append((dt / EPS_0) / safe_gamma)
        cc.append((1.0 / EPS_0) / safe_gamma)

    return LorentzCoeffs(ca=tuple(ca), cb=tuple(cb), damping=damping, restoring=restoring, c=c,
                           cc=tuple(cc))


def lorentz_e_coeffs(e_materials, dt, damping, restoring, c):
    """Build component E coefficients with SI values and eps_r derivatives.

    Shared by init_lorentz and the distributed in-loop coefficient builders.
    """
    return si_value_eps_r_grad(
        _lorentz_e_coeffs_si, _lorentz_e_coeffs_eps_r, e_materials, dt, damping, restoring, c)


def init_lorentz(
    poles: list[LorentzPole],
    materials,
    dt: float,
    mask: jnp.ndarray | list[jnp.ndarray] | tuple[jnp.ndarray, ...] | None = None,
    *,
    field_dtype=None,
    budget=None,
    grid_kind="Grid",
    grid=None,
    periodic=(False, False, False),
    high_wall=None,
) -> tuple[LorentzCoeffs, LorentzState]:
    """Initialize Lorentz/Drude ADE coefficients and state.

    Parameters
    ----------
    poles : list of LorentzPole
    materials : MaterialArrays (eps_r = ε_∞) with its lumped records
    dt : float
    mask : optional spatial mask
    field_dtype : jnp dtype, optional
        Dtype of the E/H field storage this ADE state will be driven by.
        The P carry is allocated at ``ade_state_dtype(field_dtype)`` —
        ``promote_types(field_dtype, float32)``. ``None`` (the default, for
        callers that do not thread it) means the ambient default float with
        the same float32 floor. This used to be a hard ``dtype=jnp.float32``
        pin, which made ``precision="float64"`` + any pole fail the
        ``lax.scan`` carry contract.
    periodic : per-axis flags
        The run's periodic flags, for the edge average (see ``init_debye``).

    Returns
    -------
    (LorentzCoeffs, LorentzState)

    The rule is ``init_debye``'s: per component, ε_∞ and σ from
    :func:`rfx.core.yee.component_e_materials`, and κ_p times the fraction of
    the edge's four cells in the pole's mask.
    """
    shape = materials.eps_r.shape
    n_poles = len(poles)

    from rfx.model.materials import validate_dispersion, with_components
    if grid is not None:
        from rfx.model.pole_stability import courant_budget
        budget, grid_kind = courant_budget(grid), type(grid).__name__
    components = with_components(materials, grid, periodic=periodic, high_wall=high_wall,
        lorentz_spec=(poles, mask), dt=dt, budget=budget, grid_kind=grid_kind).components
    eps_c, sig_c = components.eps_update, components.sigma_update
    realized = components.lorentz
    if realized is None or len(realized.poles) != n_poles:
        raise ValueError("realized lorentz poles do not match the requested poles")
    validate_dispersion(realized, poles, dt)
    damping, restoring, c = (realized.coefficients if realized.coefficients is not None else
                  lorentz_pole_coeffs(poles, dt, shape, realized.weights))
    coeffs = lorentz_e_coeffs((eps_c, sig_c), dt, damping, restoring, c)
    if components.pole_sets_judged:
        coeffs = mark_judged(coeffs)

    zeros = jnp.zeros((n_poles,) + shape, dtype=ade_state_dtype(field_dtype))
    state = LorentzState(
        px=zeros, py=zeros.copy(), pz=zeros.copy(),
        vx=zeros.copy(), vy=zeros.copy(), vz=zeros.copy(),
    )

    return coeffs, state


def mixed_e_component_coeffs(debye_coeffs, lorentz_coeffs, comp: int, dt):
    """``(ca, cb, cc_debye, cc_lorentz)`` of the combined Debye + Lorentz E
    update for component ``comp`` (each from that component's own
    coefficients). One spelling for every lane that runs both models.

    The Lorentz ΔP enters with 1/γ_total, where γ_total = γ_Lorentz + Σβ is
    the Debye γ: the Debye P^{n+1} is implicit in E^{n+1}, the Lorentz one
    explicit in E^n.
    """
    return si_value_eps_r_grad(
        _mixed_e_coeffs_si, _mixed_e_coeffs_eps_r,
        per_component(debye_coeffs.beta, "beta")[comp], debye_coeffs.alpha,
        per_component(lorentz_coeffs.ca, "ca")[comp],
        per_component(lorentz_coeffs.cc, "cc")[comp], dt)


def _mixed_e_coeffs_si(beta, alpha, lorentz_ca, lorentz_cc, dt):
    """The original mixed update, retaining its SI operation order."""
    beta_sum = jnp.sum(beta, axis=0)
    gamma_base = 1.0 / lorentz_cc
    gamma_total = jnp.maximum(gamma_base + beta_sum, EPS_0 * 1e-10)
    numer_base = lorentz_ca * gamma_base

    ca = (numer_base - beta_sum) / gamma_total
    cb = dt / gamma_total
    cc_debye = (1.0 - alpha) / gamma_total
    cc_lorentz = 1.0 / gamma_total
    return ca, cb, cc_debye, cc_lorentz


def _mixed_e_coeffs_eps_r(beta, alpha, lorentz_ca, lorentz_cc, dt):
    """Cancel EPS_0 before dividing by the mixed permittivity.

    lorentz_cc is 1/gamma in SI units. Its relative-unit reciprocal is
    1/(lorentz_cc*EPS_0); no SI-sized denominator is squared by autodiff.
    """
    beta_r = jnp.sum(beta, axis=0) / EPS_0
    gamma_base_r = 1.0 / (lorentz_cc * EPS_0)
    gamma_total_r = jnp.maximum(gamma_base_r + beta_r, 1e-10)
    numer_base_r = lorentz_ca * gamma_base_r
    ca = (numer_base_r - beta_r) / gamma_total_r
    cb = (dt / EPS_0) / gamma_total_r
    cc_debye = ((1.0 - alpha) / EPS_0) / gamma_total_r
    cc_lorentz = (1.0 / EPS_0) / gamma_total_r
    return ca, cb, cc_debye, cc_lorentz

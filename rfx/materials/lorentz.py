"""Lorentz and Drude dispersive materials via ADE.

Time convention e^{+jωt} throughout (Im ε < 0 for loss), matching the ADE
this module discretizes.

Lorentz model: ε(ω) = ε_∞ + Σ_p κ_p / (ω₀_p² - ω² + 2jδ_pω)

Drude model is Lorentz with ω₀ = 0:
    ε(ω) = ε_∞ − ω_p² / (ω² − jγω)

Each pole introduces a second-order auxiliary polarization P_p:
    d²P_p/dt² + 2δ_p dP_p/dt + ω₀_p² P_p = ε₀ κ_p E

Discretized (central difference + Crank-Nicolson damping):
    v^{n+1} = (v^n + (c_p E^n - damping_p v^n)) - restoring_p P^n
    P^{n+1} = P^n + v^{n+1}

    damping_p = 2δΔt / (1 + δΔt)
    restoring_p = ω₀²Δt² / (1 + δΔt)
    c_p = ε₀ κ_p Δt² / (1 + δΔt)

Per E component: ε_∞ and σ are the edge mean over the component's
four incident cells (:func:`rfx.core.yee.component_e_materials`), and each
pole's κ_p (∝ Δε_p) is κ_p times the fraction of those cells that carry the
pole -- the same parallel-susceptibility rule as σ, see
``rfx.materials.debye``. ``ca``/``cb``/``cc``/``c`` are 3-tuples ``(x, y,
z)``; ``damping`` and ``restoring`` depend on ω₀ and δ only and stay one array per pole.

References:
    Taflove & Hagness, Ch. 9
"""

from __future__ import annotations

from typing import NamedTuple
from weakref import ref

import jax
import jax.numpy as jnp

from rfx.core.yee import (
    si_value_eps_r_grad, EPS_0, ade_state_dtype,
)
from rfx.materials.debye import per_component, pole_reach


class LorentzPole(NamedTuple):
    """Single Lorentz oscillator pole.

    omega_0 : float
        Resonant angular frequency (rad/s). Set to 0 for Drude.
    delta : float
        Damping coefficient (rad/s).
    kappa : float
        Coupling strength = Δε · ω₀² for Lorentz, ω_p² for Drude.
    """
    omega_0: float
    delta: float
    kappa: float


def drude_pole(omega_p: float, gamma: float) -> LorentzPole:
    """Create a Drude pole from plasma frequency and collision rate.

    Parameters
    ----------
    omega_p : float
        Plasma frequency (rad/s).
    gamma : float
        Collision rate (rad/s).
    """
    return LorentzPole(omega_0=0.0, delta=gamma / 2.0, kappa=omega_p ** 2)


def lorentz_pole(delta_eps: float, omega_0: float, delta: float) -> LorentzPole:
    """Create a Lorentz pole from physical parameters.

    Parameters
    ----------
    delta_eps : float
        Permittivity contribution Δε (dimensionless).
    omega_0 : float
        Resonant angular frequency (rad/s).
    delta : float
        Damping coefficient (rad/s).
    """
    return LorentzPole(omega_0=omega_0, delta=delta, kappa=delta_eps * omega_0 ** 2)


class LorentzCoeffs(NamedTuple):
    """Precomputed ADE coefficients for Lorentz/Drude poles.

    Per E component -- each a 3-tuple ``(x, y, z)``:

    ca, cb : (nx, ny, nz) each — E update coefficients
    cc : (nx, ny, nz) each — P→E coupling: 1/γ
    c : (n_poles, nx, ny, nz) each — E^n coupling into the P recurrence (∝ κ)

    One array for all components:

    damping, restoring : (n_poles, nx, ny, nz) — increment coefficients (ω₀, δ only),
        the pole's values on every cell whose edges the pole reaches.
    """
    ca: tuple
    cb: tuple
    damping: jnp.ndarray
    restoring: jnp.ndarray
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


def pole_update_coeffs(omega_0, delta, kappa, dt, field_dtype=None):
    """Elementwise ADE increments, in the inputs' promoted arithmetic dtype."""
    dtype = jnp.result_type(omega_0, delta, kappa, dt)
    omega_0, delta, kappa, dt = jnp.broadcast_arrays(
        *(jnp.asarray(value, dtype=dtype) for value in (omega_0, delta, kappa, dt)))
    denom = 1.0 + delta * dt
    terms = (2.0 * delta * dt / denom,
             omega_0 ** 2 * dt ** 2 / denom,
             EPS_0 * kappa * dt ** 2 / denom)
    if field_dtype is not None:
        terms = tuple(value.astype(ade_state_dtype(field_dtype)) for value in terms)
    return terms


def lorentz_pole_coeffs(poles, dt, shape, fractions_by_pole):
    """Material-independent Lorentz ADE terms from per-pole edge fractions."""
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


def lorentz_p_component(coeffs: LorentzCoeffs, comp: int, e_old, p, v,
                        pdtype=None):
    """Advance P and its increment, grouping drive with damping before v."""
    c = per_component(coeffs.c, "c")[comp]
    v_new = (v + (c * e_old[None] - coeffs.damping * v)) - coeffs.restoring * p
    p_new = p + v_new
    if pdtype is not None:
        return p_new.astype(pdtype), v_new.astype(pdtype)
    return p_new, v_new


def lorentz_e_component(coeffs: LorentzCoeffs, comp: int, e_old, curl, dp,
                        fdtype=None):
    """E^{n+1} = Ca E^n + Cb curl - Cc ΔP for E component ``comp``."""
    ca = per_component(coeffs.ca, "ca")[comp]
    cb = per_component(coeffs.cb, "cb")[comp]
    cc = per_component(coeffs.cc, "cc")[comp]
    e_new = ca * e_old + cb * curl - cc * dp
    return e_new if fdtype is None else e_new.astype(fdtype)


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


# Lazy compatibility export for the time-step implementation.
_LAZY_EXPORTS = {
    "update_e_lorentz": "rfx.stepping.poles",
}


def __getattr__(name):
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _LAZY_EXPORTS, name)

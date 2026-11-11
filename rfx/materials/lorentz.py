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
z)``; ``damping`` and ``restoring`` depend on ω₀ and δ only; for a
``LorentzPoleMap`` whose δ varies they are component triples formed from
each edge's strength-weighted δ.

References:
    Taflove & Hagness, Ch. 9
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple
from dataclasses import dataclass
from weakref import ref
import os
import warnings

import jax
import jax.numpy as jnp

from rfx.core.yee import (
    si_value_eps_r_grad, EPS_0, ade_state_dtype,
)
from rfx.materials.debye import per_component, pole_reach
from rfx.materials.relative_strength import RelativeStrength
from rfx.errors import AccuracyWarning


class LorentzPole(NamedTuple):
    """Single Lorentz oscillator pole.

    omega_0 : float
        Resonant angular frequency (rad/s). Set to 0 for Drude.
    delta : float
        Damping coefficient (rad/s).
    kappa : float or RelativeStrength
        Coupling strength = Δε · ω₀² for Lorentz, ω_p² for Drude.
    """
    omega_0: float
    delta: float
    kappa: float | RelativeStrength


@dataclass(frozen=True, eq=False)
class LorentzPoleMap:
    """One oscillator with strength and damping sampled at physical cell centres.

    Callables take broadcastable X, Y, Z arrays in metres. On occupied cells,
    delta and kappa must be finite and non-negative. Maps are keyed by identity.
    """
    omega_0: float
    delta: float | Callable
    kappa: float | Callable

    def __post_init__(self):
        if callable(self.omega_0) or jnp.ndim(self.omega_0) != 0:
            raise ValueError("LorentzPoleMap omega_0 must be one scalar")
        for name in ("delta", "kappa"):
            value = getattr(self, name)
            if not callable(value) and jnp.ndim(value) != 0:
                raise ValueError(f"LorentzPoleMap {name} must be a scalar or callable")


def drude_pole_map(omega_p: float | Callable, gamma: float | Callable) -> LorentzPoleMap:
    """Declare a plasma-frequency and collision-rate profile in physical metres."""
    kappa = (lambda x, y, z: omega_p(x, y, z) ** 2) if callable(omega_p) else omega_p ** 2
    delta = (lambda x, y, z: gamma(x, y, z) / 2.) if callable(gamma) else gamma / 2.
    return LorentzPoleMap(0., delta, kappa)


def lorentz_pole_map(delta_eps: float | Callable, omega_0: float,
                     delta: float | Callable) -> LorentzPoleMap:
    """Declare spatial oscillator strength and damping at one resonance."""
    kappa = (lambda x, y, z: delta_eps(x, y, z) * omega_0 ** 2) if callable(delta_eps) else delta_eps * omega_0 ** 2
    return LorentzPoleMap(omega_0, delta, kappa)


def drude_pole(omega_p: float, gamma: float, *, density: object = None) -> LorentzPole:
    """Create a Drude pole from plasma frequency and collision rate.

    Parameters
    ----------
    omega_p : float
        Plasma frequency (rad/s).
    gamma : float
        Collision rate (rad/s).
    density : optional
        Relative density n_e / n_ref. With this keyword, omega_p is the
        concrete plasma frequency at n_ref and strength is RelativeStrength.
    """
    kappa = omega_p ** 2 if density is None else RelativeStrength(omega_p ** 2, density)
    return LorentzPole(omega_0=0.0, delta=gamma / 2.0, kappa=kappa)


def lorentz_pole(delta_eps: float, omega_0: float, delta: float, *, relative: bool = False) -> LorentzPole:
    """Create a Lorentz pole from physical parameters.

    Parameters
    ----------
    delta_eps : float
        Permittivity contribution Δε (dimensionless).
    omega_0 : float
        Resonant angular frequency (rad/s).
    delta : float
        Damping coefficient (rad/s).
    relative : bool
        Use RelativeStrength(omega_0**2, delta_eps), with concrete omega_0 > 0.
    """
    if relative and (isinstance(omega_0, jax.core.Tracer) or omega_0 <= 0):
        raise ValueError("relative lorentz_pole requires concrete omega_0 > 0; use drude_pole for omega_0=0")
    kappa = RelativeStrength(omega_0 ** 2, delta_eps) if relative else delta_eps * omega_0 ** 2
    return LorentzPole(omega_0=omega_0, delta=delta, kappa=kappa)


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


def lorentz_p_component(coeffs: LorentzCoeffs, comp: int, e_old, p, v,
                        pdtype=None):
    """Advance P and its increment, grouping drive with damping before v."""
    c = per_component(coeffs.c, "c")[comp]
    damping = per_component(coeffs.damping, "damping")[comp] if isinstance(coeffs.damping, tuple) else coeffs.damping
    restoring = per_component(coeffs.restoring, "restoring")[comp] if isinstance(coeffs.restoring, tuple) else coeffs.restoring
    v_new = (v + (c * e_old[None] - damping * v)) - restoring * p
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

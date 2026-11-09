"""Lorentz and Drude dispersive materials via ADE.

Time convention e^{+jωt} throughout (Im ε < 0 for loss), matching the ADE
this module discretizes.

Lorentz model: ε(ω) = ε_∞ + Σ_p κ_p / (ω₀_p² - ω² + 2jδ_pω)

Drude model is Lorentz with ω₀ = 0:
    ε(ω) = ε_∞ − ω_p² / (ω² − jγω)

Each pole introduces a second-order auxiliary polarization P_p:
    d²P_p/dt² + 2δ_p dP_p/dt + ω₀_p² P_p = ε₀ κ_p E

Discretized (central difference + Crank-Nicolson damping):
    P^{n+1} = a_p P^n + b_p P^{n-1} + c_p E^n

    a_p = (2 - ω₀²Δt²) / (1 + δΔt)
    b_p = -(1 - δΔt) / (1 + δΔt)
    c_p = ε₀ κ_p Δt² / (1 + δΔt)

Per E component: ε_∞ and σ are the edge mean over the component's
four incident cells (:func:`rfx.core.yee.component_e_materials`), and each
pole's κ_p (∝ Δε_p) is κ_p times the fraction of those cells that carry the
pole -- the same parallel-susceptibility rule as σ, see
``rfx.materials.debye``. ``ca``/``cb``/``cc``/``c`` are 3-tuples ``(x, y,
z)``; ``a`` and ``b`` depend on ω₀ and δ only and stay one array per pole.

References:
    Taflove & Hagness, Ch. 9
"""

from __future__ import annotations

from typing import NamedTuple
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

    One array for all components:

    a, b : (n_poles, nx, ny, nz) — P recurrence coefficients (ω₀, δ only),
        the pole's values on every cell whose edges the pole reaches.
    """
    ca: tuple
    cb: tuple
    a: jnp.ndarray
    b: jnp.ndarray
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
    """Auxiliary polarization state (current and previous step).

    p_{x,y,z} : (n_poles, nx, ny, nz) — P^n
    p_{x,y,z}_prev : (n_poles, nx, ny, nz) — P^{n-1}
    """
    px: jnp.ndarray
    py: jnp.ndarray
    pz: jnp.ndarray
    px_prev: jnp.ndarray
    py_prev: jnp.ndarray
    pz_prev: jnp.ndarray


def pole_underflow_guard(out, factors, *, scalar=True):
    """Identity solve whose transpose warns when an SI cotangent flushes.

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
        warning = AccuracyWarning(
            "The float32 reverse-mode gradient with respect to a traced pole strength "
            "in SI units underflowed to zero; declare the strength as "
            "RelativeStrength(reference, value) or divide the objective by a constant of its order.")
        warning.count = count
        warnings.warn(warning, stacklevel=2)

    def check(vecmat, ct):
        replay = ct / args[0]
        if scalar:
            replay = replay * args[1]
            replay = args[2] * replay
        flushed = (ct != 0) & (replay == 0)

        def warn(n):
            jax.debug.callback(report, n)
            return jnp.int32(0)

        jax.lax.cond(jnp.any(flushed), warn, lambda n: jnp.int32(0), jnp.sum(flushed))
        return ct

    return jax.lax.custom_linear_solve(lambda x: x, out, lambda matvec, b: b,
                                       transpose_solve=check)


def pole_update_coeffs(omega_0, delta, kappa, dt, field_dtype=None):
    """Elementwise Lorentz ADE terms, optionally cast to the field's ADE dtype."""
    w0, d, k = omega_0, delta, kappa
    denom = 1.0 + d * dt

    a_val = (2.0 - w0 ** 2 * dt ** 2) / denom
    b_val = -(1.0 - d * dt) / denom
    if isinstance(k, RelativeStrength):
        from rfx._traced_check import traced_check
        from rfx.errors import RfxUnsupported
        import numpy as np

        if callable(k.value):
            raise RfxUnsupported("scalar Lorentz poles do not implement a callable RelativeStrength value")

        def check_value(value):
            if not np.all(np.isfinite(value)) or np.any(np.asarray(value) < 0):
                raise ValueError("RelativeStrength value must be finite and non-negative")

        traced_check("RelativeStrength", "pole strength", check_value, k.value)
        s = float(EPS_0) * float(k.reference) * float(dt) ** 2
        c_val = (s / denom) * k.value
    else:
        c_val = EPS_0 * k * dt ** 2 / denom
        if (isinstance(k, jax.core.Tracer) and k.dtype == jnp.float32
                and os.environ.get("RFX_POLE_UNDERFLOW_CHECK") == "1"):
            c_val = pole_underflow_guard(c_val, (denom, dt ** 2, EPS_0))
    values = (a_val, b_val, c_val)
    if field_dtype is not None:
        return tuple(jnp.asarray(v, dtype=ade_state_dtype(field_dtype)) for v in values)
    return values


def lorentz_pole_coeffs(poles, dt, shape, fractions_by_pole):
    """Material-independent Lorentz ADE terms from per-pole edge fractions."""
    a_list, b_list = [], []
    c_lists = ([], [], [])

    for pole, fractions in zip(poles, fractions_by_pole):
        a_val, b_val, c_val = pole_update_coeffs(pole.omega_0, pole.delta, pole.kappa, dt)

        if fractions is not None:
            reach = pole_reach(fractions)
            a_list.append(jnp.where(reach, a_val, 0.0))
            b_list.append(jnp.where(reach, b_val, 0.0))
            for comp in range(3):
                c_lists[comp].append(jnp.where(
                    fractions[comp] > 0, c_val * fractions[comp], 0.0))
        else:
            # No dtype pin: ``a_val``/``b_val``/``c_val`` are numpy scalars
            # (``dt`` is ``grid.dt``, an np.float64), so this matches the
            # masked branch above, which has always produced whatever
            # ``jnp.where`` promotes to. A hard float32 here made the two
            # branches disagree and capped the ADE coefficients at float32
            # under ``precision="float64"``. With x64 off JAX
            # clamps to float32, so the default lane is unchanged.
            a_list.append(jnp.full(shape, a_val))
            b_list.append(jnp.full(shape, b_val))
            c_arr = jnp.full(shape, c_val)
            for comp in range(3):
                c_lists[comp].append(c_arr)

    a = jnp.stack(a_list)
    b = jnp.stack(b_list)
    c = tuple(jnp.stack(cl) for cl in c_lists)

    return a, b, c


def _lorentz_e_coeffs_si(e_materials, dt, a, b, c):
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

    return LorentzCoeffs(ca=tuple(ca), cb=tuple(cb), a=a, b=b, c=c,
                           cc=tuple(cc))


def _lorentz_e_coeffs_eps_r(e_materials, dt, a, b, c):
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

    return LorentzCoeffs(ca=tuple(ca), cb=tuple(cb), a=a, b=b, c=c,
                           cc=tuple(cc))


def lorentz_e_coeffs(e_materials, dt, a, b, c):
    """Build component E coefficients with SI values and eps_r derivatives.

    Shared by init_lorentz and the distributed in-loop coefficient builders.
    """
    return si_value_eps_r_grad(
        _lorentz_e_coeffs_si, _lorentz_e_coeffs_eps_r, e_materials, dt, a, b, c)


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
    a, b, c = (realized.coefficients if realized.coefficients is not None else
                  lorentz_pole_coeffs(poles, dt, shape, realized.weights))
    coeffs = lorentz_e_coeffs((eps_c, sig_c), dt, a, b, c)
    if components.pole_sets_judged:
        coeffs = mark_judged(coeffs)

    zeros = jnp.zeros((n_poles,) + shape, dtype=ade_state_dtype(field_dtype))
    state = LorentzState(
        px=zeros, py=zeros.copy(), pz=zeros.copy(),
        px_prev=zeros.copy(), py_prev=zeros.copy(), pz_prev=zeros.copy(),
    )

    return coeffs, state


def lorentz_p_component(coeffs: LorentzCoeffs, comp: int, e_old, p, p_prev,
                        pdtype=None):
    """P^{n+1} = a P^n + b P^{n-1} + c_comp E^n for E component ``comp``."""
    c = per_component(coeffs.c, "c")[comp]
    p_new = coeffs.a * p + coeffs.b * p_prev + c * e_old[None]
    return p_new if pdtype is None else p_new.astype(pdtype)


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

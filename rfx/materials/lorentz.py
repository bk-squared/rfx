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

from typing import Callable, NamedTuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from rfx.materials.debye import per_component
from rfx.materials.relative_strength import RelativeStrength
from rfx.model.dispersive_poles import (
    LorentzCoeffs,
    _judged_pole_arrays,
    mark_judged,
    pole_sets_judged,
    LorentzState,
    pole_underflow_guard,
    pole_update_coeffs,
    lorentz_pole_coeffs,
    _lorentz_e_coeffs_si,
    _lorentz_e_coeffs_eps_r,
    lorentz_e_coeffs,
    init_lorentz,
    mixed_e_component_coeffs,
    _mixed_e_coeffs_si,
    _mixed_e_coeffs_eps_r,
)


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
    delta and kappa must be finite and non-negative. Under jax.grad a callable and
    its derivative must also be finite at every cell centre outside the material;
    clamp the argument of a function that is undefined there (a NaN there makes the
    gradient NaN). Maps are keyed by identity.
    """
    omega_0: float
    delta: float | Callable
    kappa: float | Callable | RelativeStrength

    def __post_init__(self):
        if callable(self.omega_0) or jnp.ndim(self.omega_0) != 0:
            raise ValueError("LorentzPoleMap omega_0 must be one scalar")
        for name in ("delta", "kappa"):
            value = getattr(self, name)
            if name == "kappa" and isinstance(value, RelativeStrength):
                value = value.value
            if not callable(value) and jnp.ndim(value) != 0:
                raise ValueError(f"LorentzPoleMap {name} must be a scalar or callable")


def drude_pole_map(omega_p: float | Callable, gamma: float | Callable, *,
                   density: float | Callable | None = None) -> LorentzPoleMap:
    """Declare a plasma-frequency and collision-rate profile in physical metres."""
    kappa: float | Callable | RelativeStrength
    if density is not None:
        if callable(omega_p) or isinstance(omega_p, jax.core.Tracer):
            raise ValueError("drude_pole_map density requires a concrete reference omega_p")
        kappa = RelativeStrength(float(omega_p ** 2), density)
    else:
        kappa = (lambda x, y, z: omega_p(x, y, z) ** 2) if callable(omega_p) else omega_p ** 2
    delta = (lambda x, y, z: gamma(x, y, z) / 2.) if callable(gamma) else gamma / 2.
    return LorentzPoleMap(0., delta, kappa)


def lorentz_pole_map(delta_eps: float | Callable, omega_0: float,
                     delta: float | Callable, *, relative: bool = False) -> LorentzPoleMap:
    """Declare spatial oscillator strength and damping at one resonance."""
    kappa: float | Callable | RelativeStrength
    if relative and (isinstance(omega_0, jax.core.Tracer) or omega_0 <= 0):
        raise ValueError("relative lorentz_pole_map requires concrete omega_0 > 0; use drude_pole_map for omega_0=0")
    if relative:
        kappa = RelativeStrength(float(omega_0 ** 2), delta_eps)
    else:
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


# Lazy compatibility export for the time-step implementation.
_LAZY_EXPORTS = {
    "update_e_lorentz": "rfx.stepping.poles",
}


def __getattr__(name):
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _LAZY_EXPORTS, name)

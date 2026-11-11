"""Debye dispersive materials via Auxiliary Differential Equation (ADE).

Debye model: ε(ω) = ε_∞ + Σ_p Δε_p / (1 + jωτ_p)

Each Debye pole p introduces an auxiliary polarization field P_p that
satisfies τ_p · dP_p/dt + P_p = ε₀ · Δε_p · E.

Semi-implicit (Crank-Nicolson) discretization:
    P_p^{n+1} = α_p · P_p^n + β_p · (E^{n+1} + E^n)

    α_p = (2τ_p - dt) / (2τ_p + dt)
    β_p = ε₀ · Δε_p · dt / (2τ_p + dt)

The E update becomes (including conductivity σ):
    E^{n+1} = Ca · E^n + Cb · curl(H^{n+1/2}) + Σ_p Cc_p · P_p^n

    γ     = ε₀·ε_∞ + Σ_p β_p + σ·dt/2
    Ca    = (ε₀·ε_∞ - Σ_p β_p - σ·dt/2) / γ
    Cb    = dt / γ
    Cc_p  = (1 - α_p) / γ

Every quantity that depends on the material is built PER E COMPONENT:
an E component lies on an edge shared by four cells, and its constitutive
parameters are the tangential (arithmetic) mean over those cells -- ε_∞ and σ
from :func:`rfx.core.yee.component_e_materials` (the rule, lumped
stamps on their own component), and each pole's Δε_p as Δε_p times
the fraction of the four cells that carry the pole. The four susceptibilities
are in parallel along the edge exactly like the conductivities, and a pole
has one τ_p wherever it is present, so the edge is itself a Debye medium
with those averaged parameters. ``ca``/``cb``/``cc``/``beta`` are therefore
3-tuples ``(x, y, z)``; ``alpha`` depends on τ_p only and stays one array
per pole. Using one value per cell for all three components would put the whole grid on a cell-owned rule whenever any
material is dispersive.

References:
    Taflove & Hagness, "Computational Electrodynamics", 3rd ed., Ch. 9
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from rfx.model.dispersive_poles import (
    DebyeCoeffs,
    per_component,
    pole_edge_fractions,
    pole_reach,
    DebyeState,
    debye_pole_coeffs,
    _debye_e_coeffs_si,
    _debye_e_coeffs_eps_r,
    debye_e_coeffs,
    init_debye,
)


class DebyePole(NamedTuple):
    """Single Debye pole parameters.

    delta_eps : float
        Permittivity contribution Δε (dimensionless).
    tau : float
        Relaxation time in seconds.
    """
    delta_eps: float
    tau: float


def debye_e_component(coeffs: DebyeCoeffs, c: int, e_old, curl, p,
                      fdtype=None, pdtype=None):
    """One E component's Debye step: ``(E^{n+1}, P^{n+1})`` for component
    ``c`` (0, 1, 2) with its own coefficients.

        E^{n+1} = Ca_c·E^n + Cb_c·curl + Σ_p Cc_{p,c}·P_p^n
        P_p^{n+1} = α_p·P_p^n + β_{p,c}·(E^{n+1} + E^n)

    ``fdtype`` / ``pdtype``: the carry dtypes to narrow back to;
    ``None`` leaves the result as computed (the distributed slab bodies).
    """
    ca = per_component(coeffs.ca, "ca")[c]
    cb = per_component(coeffs.cb, "cb")[c]
    cc = per_component(coeffs.cc, "cc")[c]
    beta = per_component(coeffs.beta, "beta")[c]
    e_new = ca * e_old + cb * curl + jnp.sum(cc * p, axis=0)
    if fdtype is not None:
        e_new = e_new.astype(fdtype)
    p_new = coeffs.alpha * p + beta * (e_new[None] + e_old[None])
    if pdtype is not None:
        p_new = p_new.astype(pdtype)
    return e_new, p_new


# Lazy compatibility export for the time-step implementation.
_LAZY_EXPORTS = {
    "update_e_debye": "rfx.stepping.poles",
}


def __getattr__(name):
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _LAZY_EXPORTS, name)

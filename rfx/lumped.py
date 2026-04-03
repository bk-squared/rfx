"""Lumped RLC elements for FDTD via material modification + inductor ADE.

R and C are folded into the cell's sigma and eps_r (unconditionally
stable, same approach as the existing LumpedPort for R).  L requires
an Auxiliary Differential Equation (ADE) updated each timestep.

Derivation (inductor ADE)
-------------------------
At the inductor cell, Ampere's law with current density J_L = I_L/dx^2:

    eps*(E^{n+1}-E^n)/dt + sigma*(E^{n+1}+E^n)/2
        = curl_H/dx - I_L^{n+1}/dx^2

Inductor relation (leapfrog):

    I_L^{n+1} = I_L^n + (dt*dx/L) * E^{n+1}

Substituting and solving for E^{n+1}:

    E^{n+1} = [D0*E_std - I_L^n/dx^2] / (D0 + gamma)

where:
    D0    = eps/dt + sigma/2         (standard Yee denominator)
    gamma = dt/(L*dx)                (inductor contribution)
    E_std = Ca*E^n + Cb*curl_H       (standard Yee update result)

The key insight: E_std already incorporates D0 in its coefficients,
so we can write the correction as a post-update rescaling:

    E^{n+1} = (D0 * E_std - I_L^n/dx^2) / (D0 + gamma)

Then update I_L:

    I_L^{n+1} = I_L^n + (dt*dx/L) * E^{n+1}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from rfx.core.yee import EPS_0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LumpedRLCSpec:
    """Lumped RLC element specification.

    Parameters
    ----------
    R : float
        Resistance in ohms. 0 = no resistive component.
    L : float
        Inductance in henries. 0 = no inductive component.
    C : float
        Capacitance in farads. 0 = no capacitive component.
    topology : str
        "series" or "parallel".  Currently both topologies fold R/C
        into the material and use an ADE for L.  The distinction
        matters for future extensions (e.g., series current tracking).
    position : (x, y, z) in metres
    component : str
        E-field component ("ex", "ey", or "ez").
    """
    R: float = 0.0
    L: float = 0.0
    C: float = 0.0
    topology: str = "series"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    component: str = "ez"


# ---------------------------------------------------------------------------
# ADE state (only needed when L > 0)
# ---------------------------------------------------------------------------

class RLCState(NamedTuple):
    """ADE auxiliary state for one lumped RLC element."""
    inductor_current: jnp.ndarray  # I_L in amperes


def init_rlc_state() -> RLCState:
    """Create zero-initialised RLC ADE state."""
    return RLCState(inductor_current=jnp.array(0.0, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# Precomputed per-element metadata
# ---------------------------------------------------------------------------

class RLCCellMeta(NamedTuple):
    """Grid-resolved metadata for one RLC element.

    Precomputed once and captured by the scan closure.
    """
    i: int
    j: int
    k: int
    component: str
    has_inductor: bool
    gamma: float   # dt / (L * dx) — inductor ADE term (0 if L == 0)
    D0: float      # eps/dt + sigma/2 — Yee denominator at cell
    dx: float      # cell size
    dt_dx_over_L: float  # dt * dx / L — for I_L update (0 if L == 0)


def setup_rlc_materials(grid, spec: LumpedRLCSpec, materials):
    """Fold R and C into material arrays at the element cell.

    - R: adds sigma_R = 1 / (R * dx) to the cell conductivity.
    - C: adds eps_r_extra = C / (dx * EPS_0) to the cell permittivity.

    Both modifications are unconditionally stable because they enter
    through the standard Yee update coefficients Ca and Cb.

    Returns updated MaterialArrays.
    """
    idx = grid.position_to_index(spec.position)
    i, j, k = idx
    dx = grid.dx

    sigma = materials.sigma
    eps_r = materials.eps_r

    if spec.R > 0:
        sigma = sigma.at[i, j, k].add(1.0 / (spec.R * dx))

    if spec.C > 0:
        eps_r = eps_r.at[i, j, k].add(spec.C / (dx * EPS_0))

    return materials._replace(sigma=sigma, eps_r=eps_r)


def build_rlc_meta(grid, spec: LumpedRLCSpec, materials) -> RLCCellMeta:
    """Build per-element metadata from (modified) materials.

    Must be called AFTER ``setup_rlc_materials()`` so that eps_r and
    sigma at the cell reflect the R and C contributions.
    """
    idx = grid.position_to_index(spec.position)
    i, j, k = idx
    dx = grid.dx
    dt = grid.dt

    eps = float(materials.eps_r[i, j, k]) * EPS_0
    sigma = float(materials.sigma[i, j, k])

    D0 = eps / dt + sigma / 2.0

    has_inductor = spec.L > 0
    if has_inductor:
        gamma = dt / (spec.L * dx)
        dt_dx_over_L = dt * dx / spec.L
    else:
        gamma = 0.0
        dt_dx_over_L = 0.0

    return RLCCellMeta(
        i=i, j=j, k=k,
        component=spec.component,
        has_inductor=has_inductor,
        gamma=gamma,
        D0=D0,
        dx=dx,
        dt_dx_over_L=dt_dx_over_L,
    )


# ---------------------------------------------------------------------------
# Per-timestep inductor ADE update
# ---------------------------------------------------------------------------

def update_rlc_element(state, rlc_state: RLCState, meta: RLCCellMeta):
    """Update inductor ADE and correct the E-field at the element cell.

    Called AFTER the standard ``update_e()`` in the scan body.
    If ``meta.has_inductor`` is False, this is a no-op (R and C are
    already handled by the material coefficients).

    Returns (new_fdtd_state, new_rlc_state).
    """
    i, j, k = meta.i, meta.j, meta.k

    e_field = getattr(state, meta.component)
    e_std = e_field[i, j, k]

    i_L = rlc_state.inductor_current

    dx = meta.dx
    D0 = meta.D0
    gamma = meta.gamma

    # E^{n+1} = (D0 * E_std - I_L^n / dx^2) / (D0 + gamma)
    A = D0 + gamma
    e_new = jnp.where(
        meta.has_inductor,
        (D0 * e_std - i_L / (dx * dx)) / A,
        e_std,
    )

    # I_L^{n+1} = I_L^n + (dt * dx / L) * E^{n+1}
    i_L_new = jnp.where(
        meta.has_inductor,
        i_L + meta.dt_dx_over_L * e_new,
        i_L,
    )

    field_new = e_field.at[i, j, k].set(e_new)
    state_new = state._replace(**{meta.component: field_new})

    return state_new, RLCState(inductor_current=i_L_new)

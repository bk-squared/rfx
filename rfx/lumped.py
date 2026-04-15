"""Lumped RLC elements for FDTD via material modification + ADE.

Parallel topology
-----------------
R and C are folded into the cell's sigma and eps_r (unconditionally
stable, same approach as the existing LumpedPort for R).  L requires
an Auxiliary Differential Equation (ADE) updated each timestep.

Series topology  (true series RLC current tracking)
---------------------------------------------------
In series RLC, R, L, and C share a single series current I_s.
Instead of folding R and C into independent material properties,
we track the series current and capacitor charge via ADE.

For the series path, `setup_rlc_materials` does NOT fold R or C
into material arrays (only parallel topology does that).  The
series ADE handles R, L, and C together:

    V_total = E * dx = V_R + V_L + V_C
    V_R = R * I_s,  V_L = L * dI_s/dt,  V_C = Q / C

Semi-implicit update (with inductor):
    I_s^{n+1} * (1 + dt*R/(2*L)) = I_s^n * (1 - dt*R/(2*L))
                                    + (dt/L) * (E^n * dx - Q^n / C)
    Q^{n+1} = Q^n + I_s^{n+1} * dt
    E correction from series current density

Series R+C without inductor:
    dQ/dt = I,  V_R + V_C = E*dx
    R*I + Q/C = E*dx  =>  R*dQ/dt + Q/C = E*dx
    Semi-implicit:  Q^{n+1}*(R/dt + 1/C) = Q^n*R/dt + E*dx
    Current I = (Q^{n+1} - Q^n) / dt

Parallel topology (inductor ADE only)
--------------------------------------
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
        "series" or "parallel".  Series topology tracks a shared
        current through R, L, C via ADE (true series RLC).  Parallel
        topology folds R/C into material properties independently
        and uses an ADE only for L.
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
# ADE state (used for inductor current and/or capacitor charge tracking)
# ---------------------------------------------------------------------------

class RLCState(NamedTuple):
    """ADE auxiliary state for one lumped RLC element."""
    inductor_current: jnp.ndarray  # I_L in amperes
    capacitor_charge: jnp.ndarray  # Q_C in coulombs


def init_rlc_state() -> RLCState:
    """Create zero-initialised RLC ADE state."""
    return RLCState(
        inductor_current=jnp.array(0.0, dtype=jnp.float32),
        capacitor_charge=jnp.array(0.0, dtype=jnp.float32),
    )


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
    has_capacitor: bool     # True when C > 0
    gamma: float   # dt / (L * dx) — inductor ADE term (0 if L == 0)
    D0: float      # eps/dt + sigma/2 — Yee denominator at cell
    dx: float      # cell size
    dt_dx_over_L: float  # dt * dx / L — for I_L update (0 if L == 0)
    dt_over_C_dx: float  # dt / (C * dx) — for capacitor ADE (0 if C == 0)
    R: float               # resistance in ohms
    dt: float              # timestep in seconds
    is_series: bool        # True for series topology


def _series_needs_ade(spec: LumpedRLCSpec) -> bool:
    """Return True if this series element requires the ADE path.

    The series ADE is needed when multiple components share a current
    (e.g., R+C, R+L, L+C, R+L+C).  A standalone component (pure R,
    pure C, or pure L) can be handled by material folding / the
    standard inductor ADE without the full series current tracker.
    """
    n_components = (spec.R > 0) + (spec.L > 0) + (spec.C > 0)
    return n_components >= 2


def _resolve_position_to_index(grid, position):
    """Resolve a physical (x, y, z) to grid indices.

    Duck-types over uniform ``Grid`` (method) and ``NonUniformGrid``
    (module-level helper in ``rfx.nonuniform``).
    """
    if hasattr(grid, "position_to_index"):
        return grid.position_to_index(position)
    from rfx.nonuniform import position_to_index as _nu_p2i
    return _nu_p2i(grid, position)


def setup_rlc_materials(grid, spec: LumpedRLCSpec, materials):
    """Fold R and C into material arrays at the element cell.

    For **parallel** topology:
    - R: adds sigma_R = 1 / (R * dx) to the cell conductivity.
    - C: adds eps_r_extra = C / (dx * EPS_0) to the cell permittivity.

    For **series** topology with multiple components (R+C, R+L+C, etc.):
    - R and C are handled by the series ADE (shared current), so they
      are NOT folded into material arrays here.

    For **series** topology with a single component (pure R or pure C):
    - The component is folded into material arrays as in the parallel
      case, since a single component does not need current sharing.

    Returns updated MaterialArrays.
    """
    idx = _resolve_position_to_index(grid, spec.position)
    i, j, k = idx

    sigma = materials.sigma
    eps_r = materials.eps_r

    # Series topology with multiple components: ADE handles R and C
    if spec.topology == "series" and _series_needs_ade(spec):
        return materials

    # Parallel topology, or series with a single component: fold into material
    # Use axis-aware formula: σ_R = d_parallel / (R · d_perp1 · d_perp2)
    from rfx.sources.sources import port_sigma as _port_sigma, port_d_parallel as _d_par
    if spec.R > 0:
        sigma = sigma.at[i, j, k].add(
            _port_sigma(grid, (i, j, k), spec.component, spec.R))

    if spec.C > 0:
        d_par = _d_par(grid, (i, j, k), spec.component)
        eps_r = eps_r.at[i, j, k].add(spec.C / (d_par * EPS_0))

    return materials._replace(sigma=sigma, eps_r=eps_r)


def build_rlc_meta(grid, spec: LumpedRLCSpec, materials) -> RLCCellMeta:
    """Build per-element metadata from (modified) materials.

    Must be called AFTER ``setup_rlc_materials()`` so that eps_r and
    sigma at the cell reflect the R and C contributions.
    """
    from rfx.sources.sources import port_d_parallel as _d_par
    idx = _resolve_position_to_index(grid, spec.position)
    i, j, k = idx
    d_par = _d_par(grid, idx, spec.component)
    dt = grid.dt

    eps = float(materials.eps_r[i, j, k]) * EPS_0
    sigma = float(materials.sigma[i, j, k])

    D0 = eps / dt + sigma / 2.0

    has_inductor = spec.L > 0
    has_capacitor = spec.C > 0
    is_series = spec.topology == "series" and _series_needs_ade(spec)

    if has_inductor:
        gamma = dt / (spec.L * d_par)
        dt_dx_over_L = dt * d_par / spec.L
    else:
        gamma = 0.0
        dt_dx_over_L = 0.0

    if has_capacitor:
        dt_over_C_dx = dt / (spec.C * d_par)
    else:
        dt_over_C_dx = 0.0

    return RLCCellMeta(
        i=i, j=j, k=k,
        component=spec.component,
        has_inductor=has_inductor,
        has_capacitor=has_capacitor,
        gamma=gamma,
        D0=D0,
        dx=d_par,
        dt_dx_over_L=dt_dx_over_L,
        dt_over_C_dx=dt_over_C_dx,
        R=spec.R,
        dt=dt,
        is_series=is_series,
    )


# ---------------------------------------------------------------------------
# Per-timestep ADE update
# ---------------------------------------------------------------------------

def _update_parallel(state, rlc_state: RLCState, meta: RLCCellMeta):
    """Parallel topology: inductor ADE only; R/C are in material arrays.

    If ``meta.has_inductor`` is False, this is a no-op.
    """
    i, j, k = meta.i, meta.j, meta.k

    e_field = getattr(state, meta.component)
    e_std = e_field[i, j, k]

    i_L = rlc_state.inductor_current
    Q = rlc_state.capacitor_charge

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

    return state_new, RLCState(inductor_current=i_L_new, capacitor_charge=Q)


def _update_series(state, rlc_state: RLCState, meta: RLCCellMeta):
    """Series topology: R, L, C share a single series current.

    The series current I_s flows through all components.  The E-field
    at the cell drives the total voltage V = E * dx, which is split
    across R, L, and C in series.

    Case 1: has_inductor (L > 0), possibly with R and C
        Semi-implicit leapfrog on I_s:
            V_cell = E_std * dx   (from standard Yee update)
            (1 + dt*R/(2*L)) * I_s^{n+1} = (1 - dt*R/(2*L)) * I_s^n
                                            + (dt/L) * (V_cell - Q^n/C)
            Q^{n+1} = Q^n + dt * I_s^{n+1}
            J_s = I_s^{n+1} / dx^2  (current density correction)

        E-field correction: the standard Yee update did not account for
        the series current sink.  We correct by subtracting J_s from
        the update:
            E^{n+1} = (D0 * E_std - I_s^{n+1} / dx^2) / (D0 + gamma)

        where gamma = dt/(L*dx) couples the inductor.

    Case 2: no inductor (L == 0), has R and/or C
        RC series: R * dQ/dt + Q/C = E*dx
        Semi-implicit:
            Q^{n+1} * (R/dt + 1/C) = Q^n * R/dt + E_std * dx
        Current I = (Q^{n+1} - Q^n) / dt
        E correction from current density.
    """
    i, j, k = meta.i, meta.j, meta.k

    e_field = getattr(state, meta.component)
    e_std = e_field[i, j, k]

    i_L = rlc_state.inductor_current  # series current
    Q = rlc_state.capacitor_charge

    dx = meta.dx
    dt = meta.dt
    D0 = meta.D0
    R = meta.R

    # Voltage at cell from standard Yee update
    V_cell = e_std * dx

    # --- Case 1: has inductor ---
    # Semi-implicit: (1 + dt*R/(2L)) * I_new = (1 - dt*R/(2L)) * I_old
    #                + (dt/L) * (V_cell - Q/C)
    # where dt/L = gamma * dx  and  dt*R/(2L) = gamma*dx*R / (2*dx) ... let's
    # use direct expressions for clarity.

    # Precomputed: dt_dx_over_L = dt*dx/L, gamma = dt/(L*dx)
    # dt/L = dt_dx_over_L / dx  (when L > 0)
    # dt*R/(2*L) = (dt/L) * R/2

    dt_over_L = jnp.where(meta.has_inductor, meta.dt_dx_over_L / dx, 0.0)
    half_dtR_over_L = dt_over_L * R / 2.0

    # Capacitor voltage contribution
    inv_C = jnp.where(meta.has_capacitor, meta.dt_over_C_dx * dx / dt, 0.0)
    # inv_C = 1/C when C > 0, else 0
    V_cap = Q * inv_C  # Q/C

    # Semi-implicit inductor update
    denom_L = 1.0 + half_dtR_over_L
    numer_L = (1.0 - half_dtR_over_L) * i_L + dt_over_L * (V_cell - V_cap)
    i_L_inductor = numer_L / jnp.maximum(denom_L, 1e-30)

    # --- Case 2: no inductor, RC series ---
    # Q^{n+1} * (R/dt + 1/C) = Q^n * R/dt + E_std * dx
    R_eff = jnp.where(R > 0, R, 1e-10)  # avoid division by zero for pure C
    R_over_dt = R_eff / dt
    denom_RC = R_over_dt + inv_C
    Q_rc = (Q * R_over_dt + V_cell) / jnp.maximum(denom_RC, 1e-30)
    i_rc = (Q_rc - Q) / dt  # current from charge change

    # Select path based on has_inductor
    i_new = jnp.where(meta.has_inductor, i_L_inductor, i_rc)
    Q_new = jnp.where(
        meta.has_capacitor,
        jnp.where(meta.has_inductor, Q + dt * i_L_inductor, Q_rc),
        Q,
    )

    # E-field correction: subtract series current density from standard update.
    # For the inductor path, use the coupled correction:
    #   E^{n+1} = (D0 * E_std - I_new / dx^2) / (D0 + gamma)
    # For the RC-only path (no inductor), gamma=0 so:
    #   E^{n+1} = E_std - I_new / (D0 * dx^2)
    # Both simplify to the same formula since gamma=0 when L=0.
    gamma = meta.gamma  # 0 when L == 0
    A = D0 + gamma
    e_new = (D0 * e_std - i_new / (dx * dx)) / jnp.maximum(A, 1e-30)

    # For elements with no active ADE (shouldn't happen for series with
    # at least one nonzero component, but guard anyway)
    needs_update = meta.has_inductor | meta.has_capacitor | (R > 0)
    e_final = jnp.where(needs_update, e_new, e_std)

    field_new = e_field.at[i, j, k].set(e_final)
    state_new = state._replace(**{meta.component: field_new})

    return state_new, RLCState(inductor_current=i_new, capacitor_charge=Q_new)


def update_rlc_element(state, rlc_state: RLCState, meta: RLCCellMeta):
    """Update ADE and correct the E-field at the element cell.

    Dispatches between series and parallel topology at Python trace
    time (``meta.is_series`` is a static bool), so only the needed
    code path is compiled into the XLA graph.

    Called AFTER the standard ``update_e()`` in the scan body.

    Returns (new_fdtd_state, new_rlc_state).
    """
    if meta.is_series:
        return _update_series(state, rlc_state, meta)
    return _update_parallel(state, rlc_state, meta)

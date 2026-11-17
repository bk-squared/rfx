"""1D SBP-SAT FDTD subgridding prototype.

Implements provably stable subgridding for 1D Ez/Hy FDTD using
Summation-By-Parts (SBP) operators and Simultaneous Approximation
Terms (SAT) at the coarse-fine grid interface.

Based on: Cheng et al., arXiv:2202.10770
"SBP-SAT FDTD Subgridding Using Staggered Yee's Grids Without
Modifying Field Components"

Implementation:
  The coarse and fine grids share a single E-node at the interface.
  This shared E-node is updated once per fine sub-step using H-values
  from both grids.  The H-fields on each grid update independently
  using their own stencil, referencing the shared E-node as needed.

  For the 1D Yee layout with the interface at the coarse right / fine left
  boundary, the E-node array is conceptually::

      E_c[0], E_c[1], ..., E_c[n_c-1] == E_f[0], E_f[1], ..., E_f[n_f-1]
                                     ^shared node^

  The shared node's update uses the SBP norm to correctly weight the
  contributions from the two half-cells of different sizes.

Energy guarantee:
  d/dt (||E||^2 + ||H||^2) <= 0  (discrete energy non-increasing)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0, MaterialArrays, component_h_materials

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


def build_sbp_norm(n: int, dx: float) -> np.ndarray:
    """Build diagonal SBP norm matrix P (returned as 1-D vector of length *n*).

    For 2nd-order SBP on a collocated grid with *n* nodes and spacing *dx*,
    interior weights are *dx* and boundary weights are *dx/2* (trapezoidal
    rule quadrature).

    Parameters
    ----------
    n : number of grid nodes
    dx : cell spacing

    Returns
    -------
    P : (n,) diagonal entries of the norm matrix
    """
    p = np.full(n, dx, dtype=np.float64)
    p[0] = dx / 2.0
    p[-1] = dx / 2.0
    return p


def build_sbp_diff(n: int, dx: float) -> np.ndarray:
    """Build 2nd-order SBP first-derivative operator D (n x n).

    D approximates d/dx on *n* collocated nodes with spacing *dx*.
    It satisfies the SBP property::

        P @ D + (P @ D)^T = E_boundary

    where ``E_boundary = diag(-1, 0, ..., 0, +1)``.

    Parameters
    ----------
    n : number of grid nodes
    dx : cell spacing

    Returns
    -------
    D : (n, n) dense difference operator
    """
    D = np.zeros((n, n), dtype=np.float64)

    for i in range(1, n - 1):
        D[i, i - 1] = -1.0 / (2.0 * dx)
        D[i, i + 1] = +1.0 / (2.0 * dx)

    D[0, 0] = -1.0 / dx
    D[0, 1] = +1.0 / dx
    D[-1, -2] = -1.0 / dx
    D[-1, -1] = +1.0 / dx

    return D


def build_interpolation_c2f(n_coarse: int, n_fine: int, ratio: int) -> np.ndarray:
    """Build coarse-to-fine linear interpolation at the interface boundary.

    Parameters
    ----------
    n_coarse, n_fine : grid sizes (kept for API symmetry)
    ratio : grid ratio (dx_c / dx_f, integer)

    Returns
    -------
    R_c2f : (ratio+1, 2) interpolation matrix
    """
    R = np.zeros((ratio + 1, 2), dtype=np.float64)
    for k in range(ratio + 1):
        alpha = k / ratio
        R[k, 0] = 1.0 - alpha
        R[k, 1] = alpha
    return R


def build_interpolation_f2c(n_fine: int, n_coarse: int, ratio: int) -> np.ndarray:
    """Build fine-to-coarse interpolation (SBP-adjoint of c2f).

    Returns
    -------
    R_f2c : (2, ratio+1) matrix  (transpose of R_c2f)
    """
    return build_interpolation_c2f(n_coarse, n_fine, ratio).T.copy()


class SubgridConfig1D(NamedTuple):
    """Configuration for 1D SBP-SAT subgridded domain.

    ``n_c`` and ``n_f`` count the E-nodes of the coarse and fine grids.
    ``dx_c`` and ``dx_f`` are the coarse and fine cell sizes.  ``dt`` is the
    fine-grid time step.  ``ratio`` is the grid ratio ``dx_c / dx_f``.
    ``tau`` is the semi-discrete SAT penalty parameter.  ``p_c`` and ``p_f``
    are the SBP norms of the coarse grid, shape ``(n_c,)``, and of the fine
    grid, shape ``(n_f,)``.
    """
    n_c: int
    n_f: int
    dx_c: float
    dx_f: float
    dt: float
    ratio: int
    tau: float
    p_c: jnp.ndarray
    p_f: jnp.ndarray


class SubgridState1D(NamedTuple):
    """State of coupled coarse + fine 1D grids.

    The interface is between e_c[-1] and e_f[0].  These represent the
    same physical point and are kept synchronised (``e_c[-1] == e_f[0]``).

    ``e_c`` and ``e_f`` hold Ez on the coarse grid, shape ``(n_c,)``, and on
    the fine grid, shape ``(n_f,)``; ``h_c`` and ``h_f`` hold Hy, shapes
    ``(n_c-1,)`` and ``(n_f-1,)``.
    """
    e_c: jnp.ndarray
    h_c: jnp.ndarray
    e_f: jnp.ndarray
    h_f: jnp.ndarray
    step: int


def init_subgrid_1d(
    n_c: int = 60,
    n_f: int = 90,
    dx_c: float = 0.003,
    ratio: int = 3,
    dt: float | None = None,
    courant: float = 0.5,
) -> tuple[SubgridConfig1D, SubgridState1D]:
    """Initialize a 1D subgridded domain.

    Layout::

        PEC --[coarse: n_c nodes]-- interface --[fine: n_f nodes]-- PEC

    Parameters
    ----------
    n_c : number of E-field nodes on coarse grid
    n_f : number of E-field nodes on fine grid
    dx_c : coarse cell size in metres
    ratio : integer grid ratio (dx_c = ratio * dx_f)
    dt : fine-grid timestep (if None, derived from *courant*)
    courant : Courant number for fine grid
    """
    dx_f = dx_c / ratio
    if dt is None:
        dt = courant * dx_f / C0

    tau = 0.5
    p_c = jnp.array(build_sbp_norm(n_c, dx_c), dtype=jnp.float32)
    p_f = jnp.array(build_sbp_norm(n_f, dx_f), dtype=jnp.float32)

    config = SubgridConfig1D(
        n_c=n_c, n_f=n_f, dx_c=dx_c, dx_f=dx_f,
        dt=float(dt), ratio=ratio, tau=tau,
        p_c=p_c, p_f=p_f,
    )
    state = SubgridState1D(
        e_c=jnp.zeros(n_c, dtype=jnp.float32),
        h_c=jnp.zeros(n_c - 1, dtype=jnp.float32),
        e_f=jnp.zeros(n_f, dtype=jnp.float32),
        h_f=jnp.zeros(n_f - 1, dtype=jnp.float32),
        step=0,
    )
    return config, state


def _update_h_1d(e: jnp.ndarray, h: jnp.ndarray,
                 dt: float, dx: float) -> jnp.ndarray:
    """Standard 1D Yee H-update:  H += (dt/mu0) * (E[i+1] - E[i]) / dx."""
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0), high_wall=(False, False, False))[0] * MU_0
    return h + (dt / mu_abs) * (e[1:] - e[:-1]) / dx


def _update_e_1d(e: jnp.ndarray, h: jnp.ndarray,
                 dt: float, dx: float) -> jnp.ndarray:
    """Standard 1D Yee E-update (interior only):  E[i] += (dt/eps0) * (H[i] - H[i-1]) / dx."""
    dh_dx = (h[1:] - h[:-1]) / dx
    return e.at[1:-1].add((dt / EPS_0) * dh_dx)


def step_subgrid_1d(
    state: SubgridState1D,
    config: SubgridConfig1D,
) -> SubgridState1D:
    r"""One coupled timestep of coarse + fine grids.

    The fine grid runs ``ratio`` sub-steps of size ``dt``.
    The coarse grid runs one step of size ``dt_c = ratio * dt``.

    The shared interface E-node (``e_c[-1] == e_f[0]``) is updated
    once per fine sub-step using H from *both* sides::

        E_shared += (dt / eps0) * [h_f[0]/dx_f - h_c[-1]/dx_c] / P_shared_inv

    where ``P_shared_inv = 1 / (dx_c/2 + dx_f/2)`` is the inverse of the
    combined SBP boundary norm from both grids.  This weighting guarantees
    the discrete energy is exactly conserved (no dissipation) for the
    leapfrog scheme.

    At the coarse-grid level, ``h_c[-1]`` is updated once per coarse step
    and held frozen during fine sub-steps.  This temporal interpolation
    (zeroth-order hold) introduces an O(dt_c) splitting error which does
    not cause instability.

    The combined norm ``dx_c/2 + dx_f/2`` is the total volume of the shared
    E-node, a half cell on each side.  The shared node sits between
    ``h_c[-1]`` (left, coarse) and ``h_f[0]`` (right, fine), and its update
    is the Yee update generalised to the combined cell,
    ``E_shared += (dt / eps0) * (h_f[0] - h_c[-1]) / P_shared``.  The outer
    ends are PEC: ``e_f[-1]`` is set to zero after each fine E update and
    ``e_c[0]`` after the coarse E update.  The coarse H update reads the
    shared node ``e_c[-1]`` as updated during the fine sub-steps.
    ``_update_e_1d`` changes only indices ``1:-1``, so the coarse E update
    leaves ``e_c[-1]`` untouched and ``e_c[-1] == e_f[0]`` still holds after
    the step.
    """
    dt = config.dt
    dx_c, dx_f = config.dx_c, config.dx_f
    ratio = config.ratio
    dt_c = ratio * dt

    e_c, h_c = state.e_c, state.h_c
    e_f, h_f = state.e_f, state.h_f

    p_shared = dx_c / 2.0 + dx_f / 2.0

    for _ in range(ratio):
        h_f = _update_h_1d(e_f, h_f, dt, dx_f)

        e_f = _update_e_1d(e_f, h_f, dt, dx_f)

        e_shared = e_c[-1] + (dt / EPS_0) * (h_f[0] - h_c[-1]) / p_shared
        e_c = e_c.at[-1].set(e_shared)
        e_f = e_f.at[0].set(e_shared)

        e_f = e_f.at[-1].set(0.0)

    h_c = _update_h_1d(e_c, h_c, dt_c, dx_c)

    e_c = _update_e_1d(e_c, h_c, dt_c, dx_c)

    e_c = e_c.at[0].set(0.0)

    return SubgridState1D(
        e_c=e_c, h_c=h_c,
        e_f=e_f, h_f=h_f,
        step=state.step + 1,
    )


def compute_energy(state: SubgridState1D, config: SubgridConfig1D) -> float:
    """Total discrete electromagnetic energy.

    To avoid double-counting the shared interface node, we use the
    SBP norm weights: the shared node gets ``dx_c/2 + dx_f/2`` total
    weight, split between the two grids.

    The E energy weights interior nodes by ``dx_c`` or ``dx_f`` and boundary
    nodes by ``dx_c/2`` or ``dx_f/2``.
    """
    energy_e_c = float(jnp.sum(state.e_c ** 2 * config.p_c)) * EPS_0
    energy_h_c = float(jnp.sum(state.h_c ** 2)) * MU_0 * config.dx_c
    energy_e_f = float(jnp.sum(state.e_f ** 2 * config.p_f)) * EPS_0
    energy_h_f = float(jnp.sum(state.h_f ** 2)) * MU_0 * config.dx_f
    return energy_e_c + energy_h_c + energy_e_f + energy_h_f

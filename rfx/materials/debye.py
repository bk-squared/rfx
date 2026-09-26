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

Every quantity that depends on the material is built PER E COMPONENT (#1260):
an E component lies on an edge shared by four cells, and its constitutive
parameters are the tangential (arithmetic) mean over those cells -- ε_∞ and σ
from :func:`rfx.core.yee.component_e_materials` (the #1210 rule, lumped
stamps on their own component per #1236), and each pole's Δε_p as Δε_p times
the fraction of the four cells that carry the pole. The four susceptibilities
are in parallel along the edge exactly like the conductivities, and a pole
has one τ_p wherever it is present, so the edge is itself a Debye medium
with those averaged parameters. ``ca``/``cb``/``cc``/``beta`` are therefore
3-tuples ``(x, y, z)``; ``alpha`` depends on τ_p only and stays one array
per pole. Until #1260 all of them were one value per cell for all three
components, so a dispersive material anywhere in a model put the whole grid
back on the cell-owned rule.

References:
    Taflove & Hagness, "Computational Electrodynamics", 3rd ed., Ch. 9
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from rfx.core.yee import (
    EPS_0, FDTDState, MaterialArrays, _shift_bwd, ade_state_dtype,
    component_e_materials, edge_mean_components,
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


class DebyeCoeffs(NamedTuple):
    """Precomputed ADE update coefficients for all Debye poles.

    Per E component (#1260) -- each a 3-tuple ``(x, y, z)``:

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
    (one value per cell for all three components, #1260) -- indexing it by
    component would silently take a y-z plane and broadcast it, so it is
    refused instead.
    """
    if isinstance(field, (tuple, list)) and len(field) == 3:
        return field
    raise TypeError(
        f"dispersive {name} must be a per-E-component 3-tuple (x, y, z) "
        f"built by init_debye / init_lorentz (#1260); got "
        f"{type(field).__name__}")


def pole_edge_fractions(mask, periodic=(False, False, False)):
    """Per E component, the fraction of each edge's four cells that carry a
    pole: ``edge_mean_components`` of the pole's cell mask (#1260). ``None``
    (the pole everywhere) stays ``None``."""
    if mask is None:
        return None
    m = jnp.asarray(mask, dtype=bool).astype(jnp.float32)
    return edge_mean_components(m, periodic)


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


def init_debye(
    poles: list[DebyePole],
    materials: MaterialArrays,
    dt: float,
    mask: jnp.ndarray | list[jnp.ndarray] | tuple[jnp.ndarray, ...] | None = None,
    *,
    field_dtype=None,
    periodic=(False, False, False),
) -> tuple[DebyeCoeffs, DebyeState]:
    """Initialize Debye ADE coefficients and auxiliary state.

    Parameters
    ----------
    poles : list of DebyePole
        Debye relaxation poles.
    materials : MaterialArrays
        Base material arrays (eps_r = ε_∞, sigma, mu_r) and the lumped-stamp
        records (#1236).
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
        ``lax.scan`` carry contract (issue #656).
    periodic : per-axis flags
        The run's periodic flags, as its non-dispersive update takes them: a
        periodic axis wraps the edge average, any other replicates the
        boundary cell (``rfx.core.yee._material_bwd_neighbour``).

    Returns
    -------
    coeffs : DebyeCoeffs
        ``ca``/``cb``/``cc``/``beta`` per E component (see the class).
    state : DebyeState

    The rule (#1260): per component, ε_∞ and σ are the edge mean of
    :func:`rfx.core.yee.component_e_materials`, and each pole's Δε_p the edge
    mean of the cells' Δε_p (``delta_eps`` times the fraction of the four
    cells in the pole's mask). Homogeneous regions give the bit pattern the
    one-per-cell rule gave: the four summands are equal and the fraction is
    exactly 1.
    """
    shape = materials.eps_r.shape
    n_poles = len(poles)

    if isinstance(mask, (list, tuple)):
        if len(mask) != n_poles:
            raise ValueError(
                f"Expected {n_poles} Debye masks, got {len(mask)}"
            )
        pole_masks = [jnp.asarray(mask_i, dtype=bool) for mask_i in mask]
    else:
        shared_mask = None if mask is None else jnp.asarray(mask, dtype=bool)
        pole_masks = [shared_mask] * n_poles

    # Per-component ε_∞ and σ: the #1210 edge mean, lumped stamps removed
    # before it and added back to their own component (#1236).
    eps_c, sig_c = component_e_materials(materials, periodic)

    # Per-pole coefficients
    alpha_list = []
    beta_lists = ([], [], [])
    for pole, pole_mask in zip(poles, pole_masks):
        tau = pole.tau
        de = pole.delta_eps
        a = (2.0 * tau - dt) / (2.0 * tau + dt)
        b = EPS_0 * de * dt / (2.0 * tau + dt)

        fractions = pole_edge_fractions(pole_mask, periodic)
        if fractions is not None:
            alpha_list.append(jnp.where(pole_reach(fractions), a, 0.0))
            for c in range(3):
                beta_lists[c].append(
                    jnp.where(fractions[c] > 0, b * fractions[c], 0.0))
        else:
            # No dtype pin — see the matching note in
            # ``rfx.materials.lorentz.init_lorentz`` (issue #656): ``a``/``b``
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

    coeffs = DebyeCoeffs(ca=tuple(ca), cb=tuple(cb), cc=tuple(cc),
                         alpha=alpha, beta=beta)

    # Zero-initialized polarization state
    p_zeros = jnp.zeros((n_poles,) + shape, dtype=ade_state_dtype(field_dtype))
    state = DebyeState(px=p_zeros, py=p_zeros.copy(), pz=p_zeros.copy())

    return coeffs, state


def debye_e_component(coeffs: DebyeCoeffs, c: int, e_old, curl, p,
                      fdtype=None, pdtype=None):
    """One E component's Debye step: ``(E^{n+1}, P^{n+1})`` for component
    ``c`` (0, 1, 2) with its own coefficients (#1260).

        E^{n+1} = Ca_c·E^n + Cb_c·curl + Σ_p Cc_{p,c}·P_p^n
        P_p^{n+1} = α_p·P_p^n + β_{p,c}·(E^{n+1} + E^n)

    ``fdtype`` / ``pdtype``: the carry dtypes to narrow back to (#656);
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


def update_e_debye(
    state: FDTDState,
    coeffs: DebyeCoeffs,
    debye_state: DebyeState,
    dt: float,
    dx: float,
    periodic: tuple = (False, False, False),
) -> tuple[FDTDState, DebyeState]:
    """E-field update with Debye ADE dispersion.

    Replaces update_e() when Debye materials are present.
    Handles conductivity (via ca/cb) and Debye polarization simultaneously.

    Update order:
    1. Compute curl(H) (backward differences)
    2. E^{n+1} = Ca·E^n + Cb·curl(H) + Σ_p Cc_p·P_p^n
    3. P_p^{n+1} = α_p·P_p^n + β_p·(E^{n+1} + E^n)

    Dtype note (issue #656)
    -----------------------
    Both outputs are narrowed back to the dtype of the carry they came from.
    See the fuller note on ``rfx.materials.lorentz.update_e_lorentz``: the
    coefficients are built at setup time from ``dt`` (``grid.dt``, a numpy
    float64 scalar, STRONGLY typed in JAX), so allocation-site promotion
    alone does not close this scan.
    """
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    _fdtype = state.ex.dtype
    _pdtype = jnp.promote_types(debye_state.px.dtype, _fdtype)

    hx, hy, hz = state.hx, state.hy, state.hz

    # curl(H) via backward differences
    curl_x = ((hz - bwd(hz, 1)) - (hy - bwd(hy, 2))) / dx
    curl_y = ((hx - bwd(hx, 2)) - (hz - bwd(hz, 0))) / dx
    curl_z = ((hy - bwd(hy, 0)) - (hx - bwd(hx, 1))) / dx

    # E^{n+1} = Ca_c·E^n + Cb_c·curl(H) + Σ_p Cc_{p,c}·P_p^n, then
    # P_p^{n+1} = α_p·P_p^n + β_{p,c}·(E^{n+1} + E^n), per component (#1260)
    ex_new, px_new = debye_e_component(coeffs, 0, state.ex, curl_x,
                                       debye_state.px, _fdtype, _pdtype)
    ey_new, py_new = debye_e_component(coeffs, 1, state.ey, curl_y,
                                       debye_state.py, _fdtype, _pdtype)
    ez_new, pz_new = debye_e_component(coeffs, 2, state.ez, curl_z,
                                       debye_state.pz, _fdtype, _pdtype)

    new_fdtd = state._replace(
        ex=ex_new, ey=ey_new, ez=ez_new,
        step=state.step + 1,
    )
    new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)

    return new_fdtd, new_debye

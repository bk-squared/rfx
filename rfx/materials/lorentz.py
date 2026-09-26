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

Per E component (#1260): ε_∞ and σ are the edge mean over the component's
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

import jax.numpy as jnp

from rfx.core.yee import (
    EPS_0, FDTDState, _shift_bwd, ade_state_dtype, component_e_materials,
)
from rfx.materials.debye import per_component, pole_edge_fractions, pole_reach


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

    Per E component (#1260) -- each a 3-tuple ``(x, y, z)``:

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


def init_lorentz(
    poles: list[LorentzPole],
    materials,
    dt: float,
    mask: jnp.ndarray | list[jnp.ndarray] | tuple[jnp.ndarray, ...] | None = None,
    *,
    field_dtype=None,
    periodic=(False, False, False),
) -> tuple[LorentzCoeffs, LorentzState]:
    """Initialize Lorentz/Drude ADE coefficients and state.

    Parameters
    ----------
    poles : list of LorentzPole
    materials : MaterialArrays (eps_r = ε_∞) with its lumped records (#1236)
    dt : float
    mask : optional spatial mask
    field_dtype : jnp dtype, optional
        Dtype of the E/H field storage this ADE state will be driven by.
        The P carry is allocated at ``ade_state_dtype(field_dtype)`` —
        ``promote_types(field_dtype, float32)``. ``None`` (the default, for
        callers that do not thread it) means the ambient default float with
        the same float32 floor. This used to be a hard ``dtype=jnp.float32``
        pin, which made ``precision="float64"`` + any pole fail the
        ``lax.scan`` carry contract (issue #656).
    periodic : per-axis flags
        The run's periodic flags, for the edge average (see ``init_debye``).

    Returns
    -------
    (LorentzCoeffs, LorentzState)

    The rule is ``init_debye``'s (#1260): per component, ε_∞ and σ from
    :func:`rfx.core.yee.component_e_materials`, and κ_p times the fraction of
    the edge's four cells in the pole's mask.
    """
    shape = materials.eps_r.shape
    n_poles = len(poles)

    if isinstance(mask, (list, tuple)):
        if len(mask) != n_poles:
            raise ValueError(
                f"Expected {n_poles} Lorentz masks, got {len(mask)}"
            )
        pole_masks = [jnp.asarray(mask_i, dtype=bool) for mask_i in mask]
    else:
        shared_mask = None if mask is None else jnp.asarray(mask, dtype=bool)
        pole_masks = [shared_mask] * n_poles

    eps_c, sig_c = component_e_materials(materials, periodic)

    a_list, b_list = [], []
    c_lists = ([], [], [])

    for pole, pole_mask in zip(poles, pole_masks):
        w0, d, k = pole.omega_0, pole.delta, pole.kappa
        denom = 1.0 + d * dt

        a_val = (2.0 - w0 ** 2 * dt ** 2) / denom
        b_val = -(1.0 - d * dt) / denom
        c_val = EPS_0 * k * dt ** 2 / denom

        fractions = pole_edge_fractions(pole_mask, periodic)
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
            # under ``precision="float64"`` (issue #656). With x64 off JAX
            # clamps to float32, so the default lane is unchanged.
            a_list.append(jnp.full(shape, a_val))
            b_list.append(jnp.full(shape, b_val))
            c_arr = jnp.full(shape, c_val)
            for comp in range(3):
                c_lists[comp].append(c_arr)

    a = jnp.stack(a_list)
    b = jnp.stack(b_list)
    c = tuple(jnp.stack(cl) for cl in c_lists)

    ca, cb, cc = [], [], []
    for comp in range(3):
        eps_inf = eps_c[comp] * EPS_0
        sigma = sig_c[comp]
        gamma = eps_inf + sigma * dt / 2.0
        safe_gamma = jnp.maximum(gamma, EPS_0 * 1e-10)
        ca.append((eps_inf - sigma * dt / 2.0) / safe_gamma)
        cb.append(dt / safe_gamma)
        cc.append(1.0 / safe_gamma)

    coeffs = LorentzCoeffs(ca=tuple(ca), cb=tuple(cb), a=a, b=b, c=c,
                           cc=tuple(cc))

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
    update for component ``comp`` (#1260: each from that component's own
    coefficients). One spelling for every lane that runs both models.

    The Lorentz ΔP enters with 1/γ_total, where γ_total = γ_Lorentz + Σβ is
    the Debye γ: the Debye P^{n+1} is implicit in E^{n+1}, the Lorentz one
    explicit in E^n.
    """
    beta_sum = jnp.sum(per_component(debye_coeffs.beta, "beta")[comp], axis=0)
    gamma_base = 1.0 / per_component(lorentz_coeffs.cc, "cc")[comp]
    gamma_total = jnp.maximum(gamma_base + beta_sum, EPS_0 * 1e-10)
    numer_base = per_component(lorentz_coeffs.ca, "ca")[comp] * gamma_base

    ca = (numer_base - beta_sum) / gamma_total
    cb = dt / gamma_total
    cc_debye = (1.0 - debye_coeffs.alpha) / gamma_total
    cc_lorentz = 1.0 / gamma_total
    return ca, cb, cc_debye, cc_lorentz


def update_e_lorentz(
    state: FDTDState,
    coeffs: LorentzCoeffs,
    lor_state: LorentzState,
    dt: float,
    dx: float,
    periodic: tuple = (False, False, False),
) -> tuple[FDTDState, LorentzState]:
    """E-field update with Lorentz/Drude ADE dispersion.

    Update order:
    1. P^{n+1} = a P^n + b P^{n-1} + c E^n  (explicit)
    2. E^{n+1} = Ca E^n + Cb curl(H) - Cc Σ(P^{n+1} - P^n)

    Dtype note (issue #656)
    -----------------------
    Both outputs are narrowed back to the dtype of the carry they came from,
    the ``rfx.core.yee.update_e`` idiom (#630) plus #646's "let the body
    derive its dtype from the carry". Allocation-site promotion alone does
    NOT close this scan: ``ca``/``cb``/``cc`` and (masked) ``a``/``b``/``c``
    are built at setup time from ``dt``, which is ``grid.dt``, a **numpy
    float64 scalar** — and numpy scalars are STRONGLY typed in JAX where
    Python floats are weak. So under x64 the coefficients are float64 and
    this body would return float64 for a float32 field/P carry.

    ``_pdtype`` promotes rather than pinning to ``lor_state.px.dtype`` on
    purpose: with a real P carry and a COMPLEX field (oblique Bloch, #404) it
    stays complex, so the carry mismatch is still raised. Pinning would cast
    the imaginary part away and close the scan on wrong physics — the
    dispersive branches ignore the Bloch phase entirely.
    """
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    _fdtype = state.ex.dtype
    _pdtype = jnp.promote_types(lor_state.px.dtype, _fdtype)

    hx, hy, hz = state.hx, state.hy, state.hz

    # curl(H)
    curl_x = ((hz - bwd(hz, 1)) - (hy - bwd(hy, 2))) / dx
    curl_y = ((hx - bwd(hx, 2)) - (hz - bwd(hz, 0))) / dx
    curl_z = ((hy - bwd(hy, 0)) - (hx - bwd(hx, 1))) / dx

    # P^{n+1} = a P^n + b P^{n-1} + c_comp E^n (per pole, per component)
    px_new = lorentz_p_component(coeffs, 0, state.ex, lor_state.px,
                                 lor_state.px_prev, _pdtype)
    py_new = lorentz_p_component(coeffs, 1, state.ey, lor_state.py,
                                 lor_state.py_prev, _pdtype)
    pz_new = lorentz_p_component(coeffs, 2, state.ez, lor_state.pz,
                                 lor_state.pz_prev, _pdtype)

    # ΔP = P^{n+1} - P^n, summed over poles
    dpx = jnp.sum(px_new - lor_state.px, axis=0)
    dpy = jnp.sum(py_new - lor_state.py, axis=0)
    dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

    # E^{n+1} = Ca E^n + Cb curl(H) - Cc ΔP, per component (#1260)
    ex_new = lorentz_e_component(coeffs, 0, state.ex, curl_x, dpx, _fdtype)
    ey_new = lorentz_e_component(coeffs, 1, state.ey, curl_y, dpy, _fdtype)
    ez_new = lorentz_e_component(coeffs, 2, state.ez, curl_z, dpz, _fdtype)

    new_fdtd = state._replace(
        ex=ex_new, ey=ey_new, ez=ez_new,
        step=state.step + 1,
    )
    new_lor = LorentzState(
        px=px_new, py=py_new, pz=pz_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )

    return new_fdtd, new_lor

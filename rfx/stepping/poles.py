"""Uniform E-field updates for Debye and Lorentz/Drude pole media."""

from __future__ import annotations

import jax.numpy as jnp

from rfx.core.yee import FDTDState, curl_h
from rfx.materials.debye import DebyeCoeffs, DebyeState, debye_e_component
from rfx.materials.lorentz import (
    LorentzCoeffs, LorentzState, lorentz_e_component, lorentz_p_component,
)


def update_e_debye(
    state: FDTDState,
    coeffs: DebyeCoeffs,
    debye_state: DebyeState,
    dt: float,
    dx: float,
    periodic: tuple = (False, False, False),
    boundary=None,
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
    _fdtype = state.ex.dtype
    _pdtype = jnp.promote_types(debye_state.px.dtype, _fdtype)

    hx, hy, hz = state.hx, state.hy, state.hz

    # curl(H) via backward differences
    curl_x, curl_y, curl_z = curl_h(
        hx, hy, hz, dx, periodic, boundary=boundary, scale_after=True)

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


def update_e_lorentz(
    state: FDTDState,
    coeffs: LorentzCoeffs,
    lor_state: LorentzState,
    dt: float,
    dx: float,
    periodic: tuple = (False, False, False),
    boundary=None,
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
    _fdtype = state.ex.dtype
    _pdtype = jnp.promote_types(lor_state.px.dtype, _fdtype)

    hx, hy, hz = state.hx, state.hy, state.hz

    # curl(H)
    curl_x, curl_y, curl_z = curl_h(
        hx, hy, hz, dx, periodic, boundary=boundary, scale_after=True)

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

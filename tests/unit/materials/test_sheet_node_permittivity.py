"""f0 sheet coefficients are eps-independent at any real metal.

The rest of this file pinned the #702 sheet-node resample
(``resample_sheet_node_materials``), deleted by the lattice ownership contract
(#931): a sheet owns no cell, so there is no "own cell" whose statics could be
re-sampled. The one physical case it served — a stack-up drawn with a slot for
the foil — is a preflight WARNING (``sheet_slot_vacuum``) under the contract.
"""
import numpy as np
import pytest
import jax.numpy as jnp

D_STACK = 31.43e-6    # stack cell


def test_f0_sheet_coefficients_are_in_the_resistive_limit():
    """An f0 sheet's update coefficients do not see the node's eps_r at any
    real metal, and this test pins the reason rather than the claim.

    ``sheet_update_coeffs`` reads ``materials.eps_r`` at the sheet nodes.
    ``x2 = sigma_tot*dt/(eps0*eps_r)`` is thousands for a metal sheet, and
    both coefficients saturate there: ``A -> 0`` and ``B -> 1/sigma_tot``,
    which is the resistive-sheet limit ``E_tan = Rs*Js`` and carries no eps
    at all. The eps-independence is that limit, not a general property —
    the second half shows the coefficients separating cleanly at a sheet
    three orders of magnitude more resistive than copper foil. (Kept from
    the #702 resample suite, which the #931 contract deleted: a sheet owns
    no cell, so there is no "own cell" to re-sample.)
    """
    from rfx.core.yee import MaterialArrays
    from rfx.materials.thin_conductor import sheet_update_coeffs

    c0, mu0, eps0 = 299792458.0, 4e-7 * np.pi, 8.8541878128e-12
    dt = 0.99 * D_STACK / (c0 * np.sqrt(3.0))          # 5.992e-14 s
    sigma_cu, f0 = 5.8e7, 28e9
    skin = np.sqrt(1.0 / (np.pi * f0 * mu0 * sigma_cu))
    sigma_sheet = 1.0 / ((1.0 / (sigma_cu * skin)) * D_STACK)
    assert sigma_sheet == pytest.approx(7.288e5, rel=1e-3)

    def coeffs(eps_r, ss):
        mats = MaterialArrays(eps_r=jnp.asarray(eps_r), sigma=jnp.asarray(0.0),
                              mu_r=jnp.asarray(1.0))
        a, b = sheet_update_coeffs(jnp.asarray(ss), mats, dt)
        return float(a), float(b)

    a1, b1 = coeffs(1.0, sigma_sheet)       # vacuum at the sheet node
    a2, b2 = coeffs(3.38, sigma_sheet)      # dielectric at the sheet node
    assert (a1, b1) == (a2, b2), (
        f"copper at {f0/1e9:.0f} GHz: eps_r 1.0 -> (A={a1:.6e}, B={b1:.6e}) "
        f"but eps_r 3.38 -> (A={a2:.6e}, B={b2:.6e})")
    assert a1 == 0.0
    assert b1 == pytest.approx(1.0 / sigma_sheet, rel=1e-6)
    assert b1 == pytest.approx(1.372111e-06, rel=1e-4)
    for eps_r in (1.0, 3.38):
        assert sigma_sheet * dt / (eps0 * eps_r) > 1e3, "not in the limit"

    # the converse, so the limit is what carries the claim
    a_lo, b_lo = coeffs(1.0, 1e3)
    a_hi, b_hi = coeffs(3.38, 1e3)
    assert a_lo != a_hi and b_lo != b_hi, (
        "at sigma_sheet = 1e3 S/m the coefficients must still see eps_r")

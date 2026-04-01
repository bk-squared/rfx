"""Kerr nonlinear material for FDTD.

The Kerr effect: eps_r depends on E-field intensity:
  eps_r(E) = eps_r_linear + chi3 * |E|^2

Implemented as an auxiliary update applied after each E-field update.
All operations use JAX for differentiability.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class KerrMaterial(NamedTuple):
    """Kerr nonlinear material specification.

    Parameters
    ----------
    eps_r_linear : float
        Linear (low-field) relative permittivity.
    chi3 : float
        Third-order susceptibility in m^2/V^2.
    """
    eps_r_linear: float
    chi3: float


def apply_kerr_update(materials, state, kerr_regions):
    """Update eps_r based on instantaneous E-field intensity.

    Parameters
    ----------
    materials : MaterialArrays
    state : FDTDState
    kerr_regions : list of (mask_array, KerrMaterial)
        Each entry is a boolean mask and Kerr material spec.

    Returns
    -------
    Updated MaterialArrays with modified eps_r.
    """
    eps_r = materials.eps_r
    e_sq = state.ex ** 2 + state.ey ** 2 + state.ez ** 2

    for mask, kerr in kerr_regions:
        eps_nonlinear = kerr.eps_r_linear + kerr.chi3 * e_sq
        eps_r = jnp.where(mask, eps_nonlinear, eps_r)

    return materials._replace(eps_r=eps_r)

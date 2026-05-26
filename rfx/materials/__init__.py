"""Material definitions and property arrays."""

from rfx.core.yee import MaterialArrays, init_materials  # noqa: F401
from rfx.materials.debye import (
    DebyePole, DebyeCoeffs, DebyeState,  # noqa: F401
    init_debye, update_e_debye,  # noqa: F401
)

import jax.numpy as jnp


def set_material(materials: MaterialArrays, mask, *,
                 eps_r=None, sigma=None, mu_r=None) -> MaterialArrays:
    """Set material properties where mask is True.

    Parameters
    ----------
    materials : MaterialArrays
    mask : boolean array of shape (nx, ny, nz)
    eps_r : relative permittivity (optional)
    sigma : conductivity in S/m (optional)
    mu_r : relative permeability (optional)

    Returns
    -------
    MaterialArrays with updated properties.
    """
    if eps_r is not None:
        materials = materials._replace(
            eps_r=jnp.where(mask, eps_r, materials.eps_r)
        )
    if sigma is not None:
        materials = materials._replace(
            sigma=jnp.where(mask, sigma, materials.sigma)
        )
    if mu_r is not None:
        materials = materials._replace(
            mu_r=jnp.where(mask, mu_r, materials.mu_r)
        )
    return materials

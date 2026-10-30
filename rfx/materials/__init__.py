"""Material definitions and property arrays."""

from rfx.core.yee import MaterialArrays as _MaterialArrays

import jax.numpy as jnp


def set_material(materials: _MaterialArrays, mask, *,
                 eps_r=None, sigma=None, mu_r=None) -> _MaterialArrays:
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


# Compatibility imports warn and resolve to the owning module on every access.
_MOVED_NAMES = {
    "MaterialArrays": "rfx.core.yee",
    "init_materials": "rfx.core.yee",
    "DebyePole": "rfx.materials.debye",
    "DebyeCoeffs": "rfx.materials.debye",
    "DebyeState": "rfx.materials.debye",
    "init_debye": "rfx.materials.debye",
    "update_e_debye": "rfx.stepping.poles",
}


def __getattr__(name):
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _MOVED_NAMES, name)

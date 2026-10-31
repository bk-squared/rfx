"""Material definitions and property arrays."""

from rfx.core.yee import MaterialArrays, init_materials  # noqa: F401
from rfx.materials.debye import DebyePole, DebyeCoeffs, DebyeState, init_debye

import jax.numpy as jnp

# update_e_debye moved to the stepping layer; an eager import here would load
# rfx.stepping whenever the model layer loads this package (rfx#1818).
_LAZY_EXPORTS = {
    "update_e_debye": "rfx.stepping.poles",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value


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

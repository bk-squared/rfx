"""The electric absorber uses the curl owner's loss or dispersive coefficient."""
import jax.numpy as jnp

from rfx.core.yee import e_coeffs_eps_r_units
from rfx.materials.debye import per_component
from rfx.materials.lorentz import mixed_e_component_coeffs


def loss_terms(e_materials, dt, *, inverse=False):
    """Use the Yee coefficient function's loss, component by component."""
    return tuple(e_coeffs_eps_r_units(e, s, dt, inverse=inverse)[0]
                 for e, s in zip(*e_materials))


def material_loss_operands(materials, *, eps=None, inv=None):
    """Keep the E operands until the step forms its loss coefficient."""
    # The step's realized per-component operands (the model layer formed them).
    e, sigma = materials.components.eps_update, materials.components.sigma_update
    if inv is not None:
        return inv, sigma, True
    return e if eps is None else eps, sigma, False


def in_loop_loss(operands, dt):
    """Form loss in the same compiled step as the E coefficient owner."""
    if operands is None:
        return None
    eps, sigma, inverse = operands
    return loss_terms((eps, sigma), dt, inverse=inverse)


def dispersive_curl(debye, lorentz, dt):
    """Debye's actual curl coefficient; Lorentz alone follows the plain rule."""
    if debye is None:
        return None
    if lorentz is None:
        return per_component(debye.cb, "cb")
    return tuple(mixed_e_component_coeffs(debye, lorentz, c, dt)[1]
                 for c in range(3))


def corrected_coefficient(today, loss=None, curl=None):
    """Preserve lossless bits; a dispersive owner overrides both operands."""
    if curl is not None:
        return curl
    if loss is None:
        return today
    # Inverse subpixel tensors may be wider than the material dtype. Keep
    # the historical absorber precision, including an exact x / 1 at zero loss.
    return (today / (1.0 + loss)).astype(jnp.asarray(today).dtype)

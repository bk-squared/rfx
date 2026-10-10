"""The electric absorber uses the curl owner's loss or dispersive coefficient."""
import jax.numpy as jnp

from rfx.core.yee import component_e_materials, e_coeffs_eps_r_units
from rfx.materials.debye import per_component
from rfx.materials.lorentz import mixed_e_component_coeffs


def loss_terms(e_materials, dt, *, inverse=False):
    """Use the Yee coefficient function's loss, component by component."""
    return tuple(e_coeffs_eps_r_units(e, s, dt, inverse=inverse)[0]
                 for e, s in zip(*e_materials))


def material_loss_operands(materials, *, eps=None, inv=None):
    """Keep the E operands until the step forms its loss coefficient."""
    components = materials.components
    e, sigma = ((components.eps_update, components.sigma_update)
                if components is not None else component_e_materials(materials))
    if inv is not None:
        return inv, sigma, True
    return e if eps is None else eps, sigma, False


def in_loop_loss(operands, dt):
    """Form loss in the same compiled step as the E coefficient owner."""
    if operands is None:
        return None
    eps, sigma, inverse = operands
    return loss_terms((eps, sigma), dt, inverse=inverse)


def material_loss(materials, dt, *, eps=None, inv=None):
    """Loss of the exact component operands selected for this E update."""
    return in_loop_loss(material_loss_operands(materials, eps=eps, inv=inv), dt)


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


def face_coefficient(eps, dt, component, index, e_loss=None, e_curl_coeff=None, e_sigma=None):
    """Distributed face-sized coefficient, retaining its historical SI bits."""
    from rfx.boundaries.cpml import _ce_si, _ce_eps_r
    from rfx.core.yee import si_value_eps_r_grad
    from rfx.runners._distributed_common import cpml_coeff_e_vacuum
    today = (cpml_coeff_e_vacuum(dt) if eps is None else
             si_value_eps_r_grad(_ce_si, _ce_eps_r, eps[index], dt))
    loss = None if e_loss is None else e_loss[component][index]
    if e_loss is None and e_curl_coeff is None and e_sigma is not None:
        # Slice first: the uniform CPML shard holds face-sized loss terms,
        # formed by the same Yee function from its existing component means.
        loss = loss_terms(((eps[index],), (e_sigma[component][index],)), dt)[0]
    curl = None if e_curl_coeff is None else e_curl_coeff[component][index]
    return corrected_coefficient(today, loss, curl)


def slab_face_coefficients(eps_r, dt, psi, ghost, pad_x, *,
                           e_loss=None, e_curl_coeff=None, e_sigma=None):
    """The twelve face/component coefficients of either distributed copy.

    e_sigma : tuple of three arrays or None
        Component conductivity means, sliced before forming face-sized loss.
        Plain and Lorentz uniform runs use these existing CPML operands.
    e_loss : tuple of three arrays or None
        Loss from the E update's component epsilon and conductivity.
        Each array has the field slab's layout, including ghost rows.
        Face slices divide the historical coefficient by ``1 + loss``.
        The caller forms loss inside the time loop when E does so.
        A zero loss retains the historical lossless coefficient bits.
    e_curl_coeff : tuple of three arrays or None
        The Debye or mixed update's actual in-loop curl coefficient.
        Overrides epsilon and loss on every component and face.
        The mixed model supplies its combined coefficient, not Lorentz.cb.
        The arrays use the same slab and ghost layout as ``e_loss``.
        Lorentz-only models use conductivity loss because poles do not enter Cb.
        Both inputs default to None for callers representing lossless media.
    """
    if eps_r is None and e_loss is None and e_curl_coeff is None and e_sigma is None:
        from rfx.runners._distributed_common import cpml_coeff_e_vacuum
        return (cpml_coeff_e_vacuum(dt),) * 12
    eps = tuple(eps_r) if isinstance(eps_r, (tuple, list)) else (eps_r,) * 3
    nxlo, nxhi = psi.psi_ey_xlo.shape[0], psi.psi_ey_xhi.shape[0]
    nylo, nyhi = psi.psi_ex_ylo.shape[0], psi.psi_ex_yhi.shape[0]
    nzlo, nzhi = psi.psi_ex_zlo.shape[0], psi.psi_ex_zhi.shape[0]
    edge = ghost + pad_x
    xlo = slice(ghost, ghost + nxlo)
    xhi = slice(-(edge + nxhi), -edge) if edge else slice(-nxhi, None)
    all_ = slice(None)
    pairs = ((1, xlo), (1, xhi), (2, xlo), (2, xhi),
             (0, (all_, slice(None, nylo))), (0, (all_, slice(-nyhi, None))),
             (2, (all_, slice(None, nylo))), (2, (all_, slice(-nyhi, None))),
             (0, (all_, all_, slice(None, nzlo))), (0, (all_, all_, slice(-nzhi, None))),
             (1, (all_, all_, slice(None, nzlo))), (1, (all_, all_, slice(-nzhi, None))))
    return tuple(face_coefficient(eps[c], dt, c, index, e_loss, e_curl_coeff, e_sigma)
                 for c, index in pairs)

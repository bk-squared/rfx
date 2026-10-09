"""Thread the E update's coefficient operands into its absorber correction."""
from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h
from rfx.boundaries.electric_coefficient import (
    material_loss, material_loss_operands, in_loop_loss, dispersive_curl,
)


def cpml_context(ctx):
    """Retain operands before the kernel view drops assembly-only arrays."""
    from rfx.model.materials import kernel_context
    loss = None
    if ctx.use_cpml and not ctx.use_debye:
        plain = not ctx.use_lorentz
        loss = material_loss_operands(ctx.materials,
                             eps=ctx.aniso_eps if plain else None,
                             inv=ctx.aniso_inv_eps if plain else None)
    return kernel_context(ctx), loss


def apply_cpml_e_step(*args, debye=None, lorentz=None, **kwargs):
    """Take mixed coefficients inside the same step as the mixed E update."""
    grid = args[3]
    kwargs["e_loss"] = in_loop_loss(kwargs.get("e_loss"), grid.dt)
    kwargs["e_curl_coeff"] = dispersive_curl(debye, lorentz, grid.dt)
    return apply_cpml_e(*args, **kwargs)


def local_e_with_absorber(state, materials, dt, dx, *, debye, lorentz,
                          slab, e_materials, rank):
    """Build local ADE coefficients once, inside the distributed E step."""
    from rfx.runners._distributed_common import (
        _update_e_local_with_dispersion, slab_e_component_materials,
        slab_dispersion_coeffs,
    )
    from rfx.boundaries.electric_coefficient import loss_terms
    means = (slab_e_component_materials(materials, *slab, rank=rank)
             if e_materials is None else e_materials)
    if debye is not None:
        debye = (slab_dispersion_coeffs(debye[0], means, dt, *slab, rank=rank), debye[1])
    if lorentz is not None:
        lorentz = (slab_dispersion_coeffs(lorentz[0], means, dt, *slab, rank=rank), lorentz[1])
    curl = dispersive_curl(None if debye is None else debye[0],
                          None if lorentz is None else lorentz[0], dt)
    loss = loss_terms(means, dt) if debye is None else None
    result = _update_e_local_with_dispersion(
        state, materials, dt, dx, debye=debye, lorentz=lorentz,
        slab=slab, rank=rank)
    return (*result, (loss, curl))


def coarse_cpml_loss(materials, dt):
    """Match coarse update_e's edge loss while retaining its CPML epsilon route."""
    return material_loss(materials, dt)


def coarse_e_boundary(st, materials, dt, dx, cpml, params, psi, grid, axes, enabled):
    """Coarse electric step, retaining the caller's CPML observation point."""
    from rfx.core.yee import update_e
    st = update_e(st, materials, dt, dx)
    if enabled:
        st, psi = cpml(st, params, psi, grid, axes, materials=materials,
                       e_loss=coarse_cpml_loss(materials, dt))
    return st, psi

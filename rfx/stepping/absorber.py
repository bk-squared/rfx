"""Thread the E update's coefficient operands into its absorber correction."""
from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h
from rfx.boundaries.electric_coefficient import (
    material_loss_operands, in_loop_loss, dispersive_curl,
)


def cpml_context(ctx):
    """Retain operands before the kernel view drops assembly-only arrays."""
    from rfx.model.materials import kernel_context
    loss = None
    if ctx.use_cpml and not ctx.use_debye:
        tensor = ctx.materials.components.tensor
        plain = not ctx.use_lorentz and tensor is not None
        loss = material_loss_operands(ctx.materials,
                             eps=tensor.eps if plain else None,
                             inv=tensor.inv_eps if plain else None)
    return kernel_context(ctx), loss


def apply_cpml_e_step(*args, debye=None, lorentz=None, **kwargs):
    """Take mixed coefficients inside the same step as the mixed E update."""
    grid = args[3]
    kwargs["e_loss"] = in_loop_loss(kwargs.get("e_loss"), grid.dt)
    kwargs["e_curl_coeff"] = dispersive_curl(debye, lorentz, grid.dt)
    from rfx.boundaries import cpml
    return cpml.apply_cpml_e(*args, **kwargs)


def local_e_with_absorber(state, materials, dt, dx, *, debye, lorentz,
                          slab, e_materials, rank, high_wall, boundary=None, frame=None, owned=None):
    """Build local ADE coefficients once, inside the distributed E step."""
    from rfx.stepping.slab_update import (
        _update_e_local_with_dispersion, slab_e_component_materials,
        slab_dispersion_coeffs,
    )
    means = e_materials
    if (debye is not None or lorentz is not None) and means is None:
        means = slab_e_component_materials(materials, *slab, rank=rank, high_wall=high_wall)
    if debye is not None:
        debye = (slab_dispersion_coeffs(debye[0], means, dt, *slab, rank=rank), debye[1])
    if lorentz is not None:
        lorentz = (slab_dispersion_coeffs(lorentz[0], means, dt, *slab, rank=rank), lorentz[1])
    curl = dispersive_curl(None if debye is None else debye[0],
                          None if lorentz is None else lorentz[0], dt)
    result = _update_e_local_with_dispersion(
        state, materials, dt, dx, debye=debye, lorentz=lorentz,
        slab=slab, e_materials=means if debye is None and lorentz is None else None,
        rank=rank, high_wall=high_wall, boundary=boundary, frame=frame, owned=owned)
    return (*result, curl)


def graded_cpml_operands(materials, aniso_eps, *, use_cpml, use_debye, use_lorentz):
    """Select the graded update's pad permittivity and loss operands together."""
    if not (use_debye or use_lorentz) and aniso_eps is not None:
        inv_eps_r = tuple(1.0 / e for e in aniso_eps)
    else:
        inv_eps_r = tuple(1.0 / e for e in materials.components.eps_update)

    loss = (material_loss_operands(materials, eps=aniso_eps if not use_lorentz else None)
            if use_cpml and not use_debye else None)
    return inv_eps_r if use_cpml else None, loss

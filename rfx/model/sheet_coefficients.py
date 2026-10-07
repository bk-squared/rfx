"""Component background operands for the graded surface-impedance operator."""
from typing import NamedTuple


class ComponentSheetCoeffs(NamedTuple):
    ex: object
    ey: object
    ez: object


def sheet_update_coeffs(sigma_sheet, materials, grid):
    """Keep equal-cell bits; graded sheets read the realized E background."""
    from rfx.materials.thin_conductor import sheet_update_coeffs as build
    if all(grid.is_constant(axis) for axis in range(3)):
        return build(sigma_sheet, materials, grid.dt)
    components = materials.components
    return ComponentSheetCoeffs(*(build(
        sigma_sheet, materials._replace(eps_r=eps, sigma=sigma), grid.dt)
        for eps, sigma in zip(components.eps_update, components.sigma_update)))


def apply_sheet_impedance_e(state, e_prev, curls, ctx, coeffs):
    """Apply each component's coefficients with the existing sheet masks."""
    from rfx.materials.thin_conductor import apply_sheet_impedance_e as apply
    if not isinstance(coeffs, ComponentSheetCoeffs):
        return apply(state, e_prev, curls, ctx, coeffs)
    # Only the selected component survives tracing; XLA eliminates the other
    # two results of each call. Mask and end-row semantics stay in one owner.
    fields = {name: getattr(apply(state, e_prev, curls, ctx, c), name)
              for name, c in zip(('ex', 'ey', 'ez'), coeffs)}
    return state._replace(**fields)

"""Cell-value overrides, before component realization and drive construction.

Whole-grid inputs replace drawn cells before ports/RLC/wire-radius stages.
Design windows replace volume values in already-stamped cells, retaining the
edge-owned records for the material realization to remove and re-add. The
window's held edges and explicitly per-edge conductivity are resolved by
``materials._design_box_edge_coeffs`` after this cell stage.
"""
import jax.numpy as jnp

from rfx.core.yee import MaterialArrays, lumped_total, map_lumped


def apply_material_overrides(materials, *, eps_override=None,
                             sigma_override=None, mu_r_override=None,
                             design_box=None, window=None, box_local=None,
                             keep_lumped=False, judge=None, in_program=True):
    """Apply cell inputs without realizing or gathering any component arrays.

    ``window`` selects a design box's computation window, and ``box_local``
    its cell slice within it. A sigma tuple is already per-edge and leaves
    the cell conductivity alone. Without a window, ``design_box`` only
    promotes the fixed background to the design precision; the window is
    installed later after its admission checks and the ordinary stamps.

    Whole-grid callers own no stamps yet. Their replaced cell total loses
    its matching lumped record; ``keep_lumped`` preserves the waveguide
    extractor's existing record policy. No copy or arithmetic is performed
    on a whole-grid override, including a traced or x-sharded input.

    ``judge=(lorentz_spec, grid[, debye_spec])`` judges pole stability on the
    overridden cells: assembly judged the declared cells, and an override can
    lower eps_inf below what the poles need at this time step. A path that
    realizes components once judges there (``realize_components``); a path
    that stages cells without realizing them passes ``judge``.
    Multi-device callers pass ``in_program=False``: concrete values are judged,
    but outer-transformed values cannot stage a raising host callback there.
    """
    if window is not None:
        def cell_value(value, record):
            total = lumped_total(record)
            if total is None:
                return value
            return value + jnp.asarray(total)[window][box_local]

        eps_box = jnp.asarray(eps_override)
        eps = jnp.asarray(materials.eps_r)[window]
        # Promote the background, never quantize a traced design value.
        eps = eps.astype(jnp.promote_types(eps.dtype, eps_box.dtype))
        eps = eps.at[box_local].set(cell_value(
            eps_box, getattr(materials, "eps_r_lumped", None)))
        sig = jnp.asarray(materials.sigma)[window]
        if not isinstance(sigma_override, (tuple, list)) and sigma_override is not None:
            sig_box = jnp.asarray(sigma_override)
            sig = sig.astype(jnp.promote_types(sig.dtype, sig_box.dtype))
            sig = sig.at[box_local].set(cell_value(
                sig_box, getattr(materials, "sigma_lumped", None)))
        return MaterialArrays(
            eps_r=eps, sigma=sig, mu_r=None,
            sigma_film=map_lumped(materials.sigma_film, lambda a: jnp.asarray(a)[window]),
            eps_r_lumped=map_lumped(
                getattr(materials, "eps_r_lumped", None),
                lambda a: jnp.asarray(a)[window].astype(eps.dtype)),
            sigma_lumped=map_lumped(
                getattr(materials, "sigma_lumped", None),
                lambda a: jnp.asarray(a)[window].astype(sig.dtype)))

    if design_box is not None:
        dtype = jnp.promote_types(
            materials.eps_r.dtype, jnp.result_type(design_box.eps_r))
        return (materials if dtype == materials.eps_r.dtype else
                materials._replace(eps_r=materials.eps_r.astype(dtype)))

    if eps_override is None and sigma_override is None and mu_r_override is None:
        return materials
    out = materials._replace(
        eps_r=eps_override if eps_override is not None else materials.eps_r,
        sigma=sigma_override if sigma_override is not None else materials.sigma,
        mu_r=mu_r_override if mu_r_override is not None else materials.mu_r,
        eps_r_lumped=(None if eps_override is not None and not keep_lumped
                      else materials.eps_r_lumped),
        sigma_lumped=(None if sigma_override is not None and not keep_lumped
                      else materials.sigma_lumped))

    if judge is not None:
        from rfx.model.pole_stability import courant_budget, refuse_unstable_cells
        lorentz_spec, grid, *debye = judge
        # A distributed forward override (shard_distributed_override) holds
        # the grid's x rows first and padding rows after; judge real cells only.
        nx = grid.shape[0]
        judged = out._replace(eps_r=out.eps_r[:nx], sigma=out.sigma[:nx])
        refuse_unstable_cells(judged, lorentz_spec, grid.dt, courant_budget(grid),
                              grid_kind=type(grid).__name__, debye_spec=debye[0] if debye else None,
                              in_program=in_program)
    return out

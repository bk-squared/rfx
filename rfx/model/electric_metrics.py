"""One-dimensional electric material metrics for distributed slabs."""
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple


class SlabElectricMetrics(NamedTuple):
    """Primal operands passed explicitly into the distributed compiled program."""
    e_cell_sizes: tuple
    nx: int
    nx_per_rank: int
    nx_local: int


def slab_cell_sizes(grid, rank):
    """Select primal widths of a slab and its existing halo, never a 3-D array.

    Physical boundary ghosts replicate the first/last primal width, matching
    the material selection. Constancy is decided on the whole axis, so a
    constant sub-slab of a graded axis uses the same arithmetic as single-device.
    """
    if grid is None:
        return None
    axes = getattr(grid, "e_cell_sizes", None)
    exact = axes is not None
    if axes is None:
        axes = (grid.dx_padded, grid.dy_cells, grid.dz_cells)
    out = []
    for axis, widths in enumerate(axes):
        if widths is None and exact:
            out.append(None)
            continue
        if widths is None:
            raise ValueError("graded electric materials require primal cell widths")
        if not exact and np.all(np.asarray(widths) == np.asarray(widths)[0]):
            out.append(None)
        elif axis == 0:
            indices = rank * grid.nx_per_rank - 1 + jnp.arange(grid.nx_local)
            out.append(jnp.asarray(widths)[jnp.clip(indices, 0, grid.nx - 1)])
        else:
            out.append(jnp.asarray(widths))
    return tuple(out)


def slab_metric_kwargs(grid, rank):
    """Keep the historical equal-cell call signature for callback adapters."""
    widths = slab_cell_sizes(grid, rank)
    return {} if widths is None or all(w is None for w in widths) else dict(cell_sizes=widths)


def material_drive_scales(eps_r, sigma, mesh, drives, dt, *, ranks, grid=None,
                          reduce_devices=True, sigma_film=None, realized=False, eager=False):
    """``Cb`` of each material-driven current source, read from the slabs
    the E update receives (#1279). Called inside the runner's jitted program.

    A current source enters the field as ``E += Cb * I(t) / dV``, and ``Cb =
    (dt/eps)/(1 + sigma*dt/(2*eps))`` has to be taken from the permittivity
    the field is stepped with -- under an override, the override. The host
    cannot read it when it is an x-sharded override whose slabs live in
    other processes, or when it is traced, so each device reads it from its
    own staged slab here. The per-edge rule is the one ``make_current_source``
    and the single-device E update use: the mean over the four cells incident
    to the edge (#1210), which this lane's own E update takes too since
    #1303 (:func:`rfx.runners._distributed_common.slab_e_component_materials`).

    ``eps_r`` / ``sigma`` : the staged ``(n_devices * nx_local, ny, nz)``
        arrays on ``P("x")`` -- ghost rows filled by the staging with the
        neighbour's real cells for every override form
        (``stage_forward_array_x_slab``, ``stage_sharded_forward_override``,
        ``stage_concrete_forward_array``).
    ``drives`` : static tuple of ``(dev_id, row0, cell, component, dV, pole)``,
        one per material-driven source. The edge's owner reads the four
        cells :func:`rfx.core.yee.cell_component_e_materials` names for
        ``cell`` in its local slab from row ``row0`` on. ``row0 = 0`` keeps
        the left ghost row, so an Ey/Ez edge on a rank's first real cell
        takes its i-1 cells from the left neighbour. On the domain's x-lo
        face (global ``i == 0``) that row is vacuum padding, and ``row0``
        is the ghost width: the helper then replicates the boundary cell,
        the single-device rule. ``dV`` is the E node's control volume
        (:func:`rfx.nonuniform.current_source_volume`).

    With ``reduce_devices=False``, returns ``(n_devices, len(drives))``
    on the slab axis without a collective; non-owner entries are zero.
    The default returns a replicated ``(len(drives),)`` float32 vector: every device
    computes its own four-cell mean, the owner's is kept by the mask and
    ``psum`` hands it to all. The arithmetic is
    :func:`rfx.nonuniform.current_source_cb`'s eps_r-units formula, the one the
    single-device lane uses on every trace, so a traced (or
    sharded) permittivity stays on the tape and the gradient flows back to
    the design's owning slabs through the staging transpose.
    """
    if eager and realized:
        from rfx.stepping.slab_setup import eager_drive_scales
        return eager_drive_scales(eps_r, sigma, mesh, drives, dt, ranks, reduce_devices)
    from functools import partial
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from rfx.core.yee import MaterialArrays, map_lumped
    from rfx.model.materials import e_update_material_at
    from rfx.runners._distributed_common import rank_shard_map
    from rfx.nonuniform import current_source_cb

    @partial(rank_shard_map, mesh=mesh, in_specs=(P("x"), P("x"), P("x")),
             out_specs=P() if reduce_devices else P("x"), check_rep=False)
    def _scales(eps_local, sigma_local, film_local, *, rank):
        device = rank
        out = []
        for dev_id, row0, cell, component, dV, pole in drives:
            if realized:
                axis = ('ex', 'ey', 'ez').index(component)
                eps_c, sigma_c = eps_local[axis][cell], sigma_local[axis][cell]
            else:
                view = MaterialArrays(eps_r=eps_local[row0:],
                                      sigma=sigma_local[row0:], mu_r=None,
                                      sigma_film=map_lumped(film_local, lambda a: a[row0:]))
                widths = slab_cell_sizes(grid, rank)
                if widths is not None:
                    widths = (None if widths[0] is None else widths[0][row0:], *widths[1:])
                eps_c, sigma_c = e_update_material_at(view, cell, component,
                                                           cell_sizes=widths)
            owner = device == dev_id
            # A non-owner read someone else's cells; keep its masked branch
            # finite so the cotangent through the mask stays zero, not NaN.
            cb = current_source_cb(jnp.where(owner, eps_c, 1.0),
                                   jnp.where(owner, sigma_c + pole, 0.0), dt)
            out.append(jnp.where(owner, cb, 0.0))
        values = jnp.stack(out)
        return lax.psum(values, "x") if reduce_devices else values[None, :]

    return _scales(ranks, eps_r, sigma, sigma_film)


def stage_forward_dispersion_x_slab(materials, dt, spec, sharded_grid, mesh, kind):
    """Stage canonical ADE operands from local material and pole-mask rows."""
    from rfx.stepping.slab import Slab
    from rfx.stepping.slab_wrap import select_step
    if spec is None:
        return None
    step_kind = sharded_grid.step_kind
    if step_kind is None:
        step_kind = select_step(cpml=bool(sharded_grid.cpml_layers), **{kind: True})
    if step_kind == 'legacy':
        import jax
        from rfx.stepping.slab import cut
        from rfx.runners._distributed_common import stage_slab_pole_coeffs
        poles, masks = spec
        masks = jax.tree.map(lambda mask: cut(mask, Slab.from_grid(sharded_grid),
                                             'pole_mask', mesh=mesh), masks)
        return stage_slab_pole_coeffs(poles, masks, dt, kind, mesh,
            sharded_grid.nx_per_rank, sharded_grid.nx, materials.eps_r.shape, grid=sharded_grid)
    from rfx.stepping.slab_setup import stage_dispersion
    return stage_dispersion(materials, dt, spec, sharded_grid.global_grid,
                            Slab.from_grid(sharded_grid), mesh, kind)


def stage_forward_materials(materials, layout, mesh, *, uniform=False):
    """Release each whole-grid field after its graded forward slab staging."""
    from rfx.core.yee import MaterialArrays, map_lumped
    from rfx.stepping.slab import cut
    if uniform:
        import jax
        from rfx.stepping.slab_view import uniform_slab_rows, stack_and_place
        rows = uniform_slab_rows(layout)
        return jax.tree.map(lambda a: stack_and_place(a, rows, mesh)
            if a is not None and getattr(a, 'ndim', 0) else a,
            materials._replace(components=None, edge_keep=None))
    staged = {}
    for name, kind in (("eps_r", "eps_r"), ("sigma", "sigma"),
                       ("mu_r", "mu_r"), ("sigma_film", "lumped")):
        staged[name] = cut(getattr(materials, name), layout, kind, mesh=mesh)
        materials = materials._replace(**{name: None})
    return MaterialArrays(**staged)


def _slab_x_lo_view(arr, rank):
    return arr.at[0].set(jnp.where(rank == 0, arr[1], arr[0]))


def _slab_model_rows(nx_local, nx_per, nx, rank):
    local = jnp.arange(nx_local)
    rows = rank * nx_per - 1 + local
    return ((local >= 1) & (local < nx_local - 1) & (rows < nx))[:, None, None]

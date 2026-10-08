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


def material_drive_scales(eps_r, sigma, mesh, drives, dt, *, ranks, grid=None):
    """``Cb/dV`` of each material-driven current source, read from the slabs
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

    Returns a replicated ``(len(drives),)`` float32 vector: every device
    computes its own four-cell mean, the owner's is kept by the mask and
    ``psum`` hands it to all. The arithmetic is
    :func:`rfx.nonuniform.current_source_cb`'s traced branch, the one the
    single-device lane uses for a traced override, so a traced (or
    sharded) permittivity stays on the tape and the gradient flows back to
    the design's owning slabs through the staging transpose.
    """
    from functools import partial
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from rfx.core.yee import MaterialArrays
    from rfx.model.materials import e_update_material_at
    from rfx.runners._distributed_common import rank_shard_map
    from rfx.nonuniform import current_source_cb

    @partial(rank_shard_map, mesh=mesh, in_specs=(P("x"), P("x")),
             out_specs=P(), check_rep=False)
    def _scales(eps_local, sigma_local, *, rank):
        device = rank
        out = []
        for dev_id, row0, cell, component, dV, pole in drives:
            view = MaterialArrays(eps_r=eps_local[row0:],
                                  sigma=sigma_local[row0:], mu_r=None)
            widths = slab_cell_sizes(grid, rank)
            if widths is not None:
                widths = (None if widths[0] is None else widths[0][row0:], *widths[1:])
            eps_c, sigma_c = e_update_material_at(view, cell, component,
                                                       cell_sizes=widths)
            owner = device == dev_id
            # A non-owner read someone else's cells; keep its masked branch
            # finite so the cotangent through the mask stays zero, not NaN.
            cb = current_source_cb(jnp.where(owner, eps_c, 1.0),
                                   jnp.where(owner, sigma_c + pole, 0.0), dt,
                                   traced=True)
            out.append(jnp.where(owner, cb / dV, 0.0))
        return lax.psum(jnp.stack(out), "x")

    return _scales(ranks, eps_r, sigma)


def stage_forward_dispersion_x_slab(materials, dt, spec, sharded_grid, mesh, kind):
    """Stage fixed pole terms; epsilon/sigma E coefficients wait for the loop.

    Pole masks take the same slab halo and boundary replication as the shared
    material means. No differentiable coefficients pass through eager
    shard_map setup, and no whole-domain ADE arrays are allocated.
    """
    import jax
    from rfx.runners._distributed_common import stage_slab_pole_coeffs
    from rfx.stepping.slab import Slab, cut

    if spec is None:
        return None
    poles, masks = spec
    masks = jax.tree.map(
        lambda mask: cut(mask, Slab.from_grid(sharded_grid), "pole_mask", mesh=mesh),
        masks)
    return stage_slab_pole_coeffs(
        poles, masks, dt, kind, mesh, sharded_grid.nx_per_rank,
        sharded_grid.nx, materials.eps_r.shape, grid=sharded_grid)

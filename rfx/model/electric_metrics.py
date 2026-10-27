"""One-dimensional electric material metrics for distributed slabs."""
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple

from rfx import _realized
from rfx.core.yee import MaterialArrays, cell_owned_component_materials, component_e_materials, map_lumped
from rfx.model.graded_admission import concrete_distributed_x_cells


class SlabElectricMetrics(NamedTuple):
    """Primal operands passed explicitly into the distributed compiled program."""
    e_cell_sizes: tuple
    nx: int
    nx_per_rank: int
    nx_local: int
    high_wall: tuple[bool, bool, bool]


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
    """Global boundary ownership and metrics of a cropped slab."""
    if grid is None:
        raise TypeError("slab material lookup requires global boundary metadata")
    widths = slab_cell_sizes(grid, rank)
    return dict(high_wall=grid.high_wall, cell_sizes=widths)


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
    to the edge (#1210), which this lane's own E update takes too since #1303
    (:func:`slab_e_component_materials`).

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
    from rfx.model.source_coefficients import current_source_cb

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
                if grid is None:
                    raise TypeError("slab material lookup requires global boundary metadata")
                eps_volume = slab_wall_view(eps_local, grid.nx_per_rank, grid.nx,
                                            rank, high_wall=grid.high_wall)
                sigma_volume = slab_wall_view(sigma_local, grid.nx_per_rank, grid.nx,
                                              rank, high_wall=grid.high_wall)
                view = MaterialArrays(eps_r=eps_volume[row0:],
                                      sigma=sigma_volume[row0:], mu_r=None,
                                      sigma_film=map_lumped(film_local, lambda a: a[row0:]))
                widths = slab_cell_sizes(grid, rank)
                if widths is not None:
                    wx = slab_wall_view(widths[0], grid.nx_per_rank, grid.nx,
                                        rank, high_wall=grid.high_wall)
                    widths = (None if wx is None else wx[row0:], *widths[1:])
                eps_c, sigma_c = e_update_material_at(view, cell, component,
                    high_wall=(False, *grid.high_wall[1:]), cell_sizes=widths)
            owner = device == dev_id
            # A non-owner read someone else's cells; keep its masked branch
            # finite so the cotangent through the mask stays zero, not NaN.
            cb = current_source_cb(jnp.where(owner, eps_c, 1.0),
                                   jnp.where(owner, sigma_c + pole, 0.0), dt)
            out.append(jnp.where(owner, cb, 0.0))
        values = jnp.stack(out)
        return lax.psum(values, "x") if reduce_devices else values[None, :]

    return _scales(ranks, eps_r, sigma, sigma_film)


def stage_forward_dispersion_x_slab(materials, dt, spec, sharded_grid, mesh, kind, *, debye_spec=None):
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
            sharded_grid.nx_per_rank, sharded_grid.nx, materials.eps_r.shape,
            high_wall=sharded_grid.high_wall, grid=sharded_grid)
    from rfx.stepping.slab_setup import stage_dispersion
    return stage_dispersion(materials, dt, spec, sharded_grid.global_grid,
                            Slab.from_grid(sharded_grid), mesh, kind, debye_spec=debye_spec)


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


def slab_wall_view(arr, nx_per, nx, rank, *, high_wall, ghost=1):
    """Read the inside cell at the global wall, including alignment ghosts."""
    if arr is None or jnp.ndim(arr) == 0:
        return arr
    local = jnp.arange(arr.shape[0])
    global_rows = rank * nx_per - ghost + local
    owns_wall = (rank * nx_per <= nx - 1) & ((rank + 1) * nx_per > nx - 1)
    inside = jnp.clip(nx - 2 - rank * nx_per + ghost, 0, arr.shape[0] - 1)
    indices = jnp.where(high_wall[0] & owns_wall & (global_rows >= nx - 1), inside, local)
    indices = jnp.where((rank == 0) & (local < ghost), ghost, indices)
    return jnp.take(arr, indices, axis=0)


def _slab_model_rows(nx_local, nx_per, nx, rank):
    local = jnp.arange(nx_local)
    rows = rank * nx_per - 1 + local
    return ((local >= 1) & (local < nx_local - 1) & (rows < nx))[:, None, None]


def slab_e_component_materials(materials, nx_per, nx, rank=None, *, high_wall: tuple[bool, bool, bool], cell_sizes=None):
    """Per-E-component ``(eps_r, sigma)`` of one x slab, by the single-device
    rule (#1303).

    A Yee E component lies on an edge shared by four cells, and every lane
    gives it the mean of their eps and sigma: THE rule is
    :func:`rfx.core.yee.component_e_materials` (#1210, lumped stamps on their
    own edge #1236), and this function only hands it the right cells. The
    mean along x reads the row BEFORE the edge (Ey: i-1 and k-1, Ez: i-1 and
    j-1; Ex never reads x), so a slab needs one row below its first real row:

    * at a seam, the neighbour's last real cell. The staged material halo
      already holds it in the left ghost row; nothing is exchanged per step.
    * at the domain's x-lo face (rank 0), the boundary cell itself: outside
      the lattice the single-device rule edge-replicates. The staged ghost
      row there is vacuum padding, so the mean reads the first real row in
      its place. Only the mean does; the staged ghost keeps its value for
      every other reader (the #1317 drive reads it the same way).

    Rows that are not cells of the model -- the ghost rows at both slab ends
    and the high-x alignment pad -- keep the value of the cell they hold
    (:func:`rfx.core.yee.cell_owned_component_materials`), as they had before
    this lane was converted: their fields are replaced by the ghost exchange
    or stand in for the outside of the domain, and what the domain faces do
    is left as it was.

    ``materials`` : the slab's :class:`MaterialArrays`, ``(nx_local, ny, nz)``
        with one ghost row at each x end (lumped records optional).
    ``nx_per`` : real rows per slab (``nx_padded // n_devices``).
    ``nx`` : the unpadded global x cell count.
    ``rank`` : the slab's mesh position, supplied as data by the shard-map adapter.

    Returns ``((eps_x, eps_y, eps_z), (sig_x, sig_y, sig_z))``, slab-shaped.
    y and z are whole on every slab and non-periodic on these lanes (periodic
    axes are refused), so they take the helper's own edge replication.
    """
    if rank is None:
        raise ValueError("slab rank must be supplied as data")
    nx_local = materials.eps_r.shape[0]

    from rfx.model.electric_metrics import slab_wall_view
    from rfx.core.yee import permittivity_without_lumped, lumped_total, lumped_components, add_lumped_eps, add_film
    def read(arr):
        return slab_wall_view(arr, nx_per, nx, rank, high_wall=high_wall)
    stored_sigma = sigma = materials.sigma
    stamp = lumped_total(materials.sigma_lumped)
    if stamp is not None:
        sigma = sigma - stamp
    view = MaterialArrays(eps_r=read(permittivity_without_lumped(materials)),
                          sigma=read(sigma), mu_r=materials.mu_r, sigma_film=None)
    if cell_sizes is not None:
        cell_sizes = (read(cell_sizes[0]), *cell_sizes[1:])
    eps_edge, sig_edge = component_e_materials(view, (False,) * 3,
        high_wall=(False, *high_wall[1:]), cell_sizes=cell_sizes)
    eps_edge = add_lumped_eps(eps_edge, materials.eps_r_lumped)
    sig_edge = tuple(s if p is None else s + p for s, p in
                     zip(sig_edge, lumped_components(materials.sigma_lumped)))
    sig_edge = add_film(sig_edge, materials)
    eps_cell, sig_cell = cell_owned_component_materials(materials)
    model_cell = _slab_model_rows(nx_local, nx_per, nx, rank)

    def per_row(edge, cell):
        return tuple(jnp.where(model_cell, e, c) for e, c in zip(edge, cell))

    eps, sigma = per_row(eps_edge, eps_cell), per_row(sig_edge, sig_cell)
    if _realized.ACTIVE is not None:
        # Both coefficient paths consume these operands, including the
        # checkpointed shard_map. Record global ownership for the
        # full-grid reference.
        site = ("distributed_nu.E" if _realized.ACTIVE.lane == "fwd_distributed_nu"
                else "distributed.E")
        def cells(arr):
            return slab_wall_view(arr, nx_per, nx, rank, high_wall=(False,) * 3)
        # The record holds the stored cells with their lumped and film
        # records, as the one-device site records them; ``view`` is the
        # volume-only temporary of the mean.
        recorded = MaterialArrays(
            eps_r=cells(materials.eps_r), sigma=cells(stored_sigma), mu_r=materials.mu_r,
            sigma_film=map_lumped(materials.sigma_film, cells),
            sigma_lumped=map_lumped(materials.sigma_lumped, cells),
            eps_r_lumped=map_lumped(materials.eps_r_lumped, cells))
        eps, sigma = _realized.electric(
            recorded, eps, sigma, site, owned_start=rank * nx_per,
            owned_count=jnp.minimum(nx_per, nx - rank * nx_per))
    return eps, sigma


def _build_sharded_inv_dx_arrays(grid, n_devices, pad_x=0):
    """Build per-device x-axis inverse-spacing slabs for the shard_map runner.

    The caller has already decided to pad the global x-extent by
    ``pad_x`` cells (to align ``nx`` on ``n_devices``).  Unlike the
    pre-#622 version, this does NOT re-derive ``inv_dx``/``inv_dx_h``
    from a padded cell-size profile — it reads ``grid.inv_dx`` /
    ``grid.inv_dx_h`` directly (the single-device reference arrays,
    which already carry the correct trailing H zero at the REAL
    boundary node ``nx - 1``, see ``nonuniform.py:219``) and appends
    ``pad_x`` zero coefficients.  The pad region is therefore
    STRUCTURALLY INERT: with every E and H coefficient zero there, no
    field evolves in the pad cells regardless of what the PEC-mask
    padding does, and the wall stays exactly where the single-device
    grid put it (#622 — the old re-derivation moved the trailing H
    zero from real node ``nx-1`` to padded index ``nx_padded-1``,
    leaving the real boundary cell's H term live).

    Parameters
    ----------
    grid : NonUniformGrid
    n_devices : int
    pad_x : int
        Number of inert cells appended to the high-x end of the domain
        so that ``(nx + pad_x) % n_devices == 0``.

    Returns
    -------
    inv_dx_e_global : (nx_padded,) np.ndarray
        E-update inverse spacing — mean ``2/(d[k-1]+d[k])`` (leading
        ``1/d[0]``), zero in the pad region. Consumed by
        ``_update_e_local_nu``.
    inv_dx_h_global : (nx_padded,) np.ndarray
        H-update inverse spacing — local ``1/d[k]`` (zero at real
        boundary node ``nx-1`` AND throughout the pad region).
        Consumed by ``_update_h_local_nu``.
    dx_padded : (nx_padded,) np.ndarray
        The cell-size profile (float32), pad region filled with the
        boundary cell size for diagnostics (e.g. CPML dx_boundary) —
        this is geometry bookkeeping only; it does NOT feed the E/H
        inverse-spacing arrays above.
    """
    dx_arr = concrete_distributed_x_cells(grid)
    inv_dx_e = np.asarray(grid.inv_dx, dtype=np.float64)
    inv_dx_h = np.asarray(grid.inv_dx_h, dtype=np.float64)
    if pad_x > 0:
        dx_arr = np.concatenate(
            [dx_arr, np.full(pad_x, float(dx_arr[-1]))]
        )
        inv_dx_e = np.concatenate([inv_dx_e, np.zeros(pad_x, dtype=np.float64)])
        inv_dx_h = np.concatenate([inv_dx_h, np.zeros(pad_x, dtype=np.float64)])
    nx = inv_dx_e.shape[0]
    if nx % n_devices != 0:
        raise ValueError(
            f"After padding nx={nx} is not divisible by n_devices={n_devices}"
        )

    return (
        inv_dx_e.astype(np.float32),
        inv_dx_h.astype(np.float32),
        dx_arr.astype(np.float32),
    )

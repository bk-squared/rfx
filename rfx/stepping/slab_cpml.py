"""Place native CPML operands; the boundary owner retains all arithmetic."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp


def apply(kernel, ctx, invariants, state, psi, **kwargs):
    """Call the canonical boundary kernel in the legacy slab argument form."""
    from rfx.runners._distributed_common import cpml_slab_frame, cpml_absorbing_faces

    params = jax.tree.map(lambda a: a[0], invariants['cpml_x'])
    electric, magnetic = ctx.cpml_spacings
    params = params._replace(**electric)
    if magnetic:
        params = params._replace(magnetic=params.magnetic._replace(**magnetic))
    ghost, end = ctx.x_face_frame['x']
    pad_x = state.ex.shape[0] - ghost - end
    return kernel(state, params, psi, ctx.cpml_grid,
                  frame=cpml_slab_frame(state.ex.shape, ghost, pad_x),
                  faces=cpml_absorbing_faces(ctx.cpml_grid),
                  owned=invariants['owns_x'], **kwargs)


def export_state(psi, grid, layout, *, native=False):
    """Return psi in the runner's ``init_cpml_slabs`` layout (#1745).

    Field order with the rank merged into x: y/z faces keep this rank's x rows;
    x faces hold the one-device depth, real only on the owning end rank (rank 0
    for x_lo, the last rank for x_hi) and zero elsewhere.
    """
    if native:
        return psi
    from rfx.boundaries.cpml import _axis_buffer_depths
    full = _axis_buffer_depths(grid, grid.cpml_layers)[0]
    fields = {}
    for name, a in zip(psi._fields, psi):
        side = name.split('_')[-1][1:]
        if name.split('_')[-1][0] != 'x':
            fields[name] = a
            continue
        rows = a.reshape((layout.n_devices, layout.nx_local) + a.shape[1:])
        depth = max(1, getattr(grid, 'pad_x_' + side))
        end = layout.ghost_width + layout.nx_per_rank - layout.pad_x
        owner = 0 if side == 'lo' else layout.n_devices - 1
        real = (rows[owner, layout.ghost_width:layout.ghost_width + depth] if side == 'lo'
                else rows[owner, end - depth:end])
        real = jnp.pad(real, [(0, full - depth) if side == 'lo' else (full - depth, 0), (0, 0), (0, 0)])
        out = jnp.zeros((layout.n_devices, full) + a.shape[1:], a.dtype).at[owner].set(real)
        fields[name] = out.reshape((layout.n_devices * full,) + a.shape[1:])
    return type(psi)(**fields)


def import_state(psi, canonical, layout, *, native=False):
    """Recover one-device psi from the runner's per-rank input.

    ``init_cpml_slabs`` layout (#1745): field order on every rank, x faces
    (s_x, Ny, Nz) at the one-device depth, y/z faces over the rank's local x
    rows; the rank axis is leading or merged into x.
    """
    if native:
        import jax
        return jax.tree.map(lambda a: a.reshape((-1,) + a.shape[-2:]), psi)
    from rfx.runners._distributed_common import unstack_and_gather
    fields = {}
    for name, a in zip(psi._fields, psi):
        _, component, face = name.split('_')
        axis, side = face[0], face[1:]
        rows = a if a.ndim == 4 else a.reshape((layout.n_devices, a.shape[0] // layout.n_devices) + a.shape[1:])
        depth = rows.shape[1 + 'xyz'.index(axis)]
        if axis == 'x':
            whole = rows[0 if side == 'lo' else -1]
        else:
            merged = rows.reshape((layout.n_devices * layout.nx_local,) + rows.shape[2:])
            whole = unstack_and_gather(merged, layout.n_devices, layout.nx_local,
                                      layout.ghost_width, layout.pad_x, layout.nx)
        pads = [(0, 0)] * 3
        extra = getattr(canonical, name).shape['xyz'.index(axis)] - depth
        pads['xyz'.index(axis)] = (0, extra) if side == 'lo' else (extra, 0)
        fields[name] = jnp.pad(whole, pads)
    return type(psi)(**fields)


def publish(setup, layout, mesh):
    """Publish dynamic whole profiles and rank-local ``init_cpml_slabs`` psi.

    X faces retain one-device depth on each rank; y/z faces use local x rows.
    The step restores the original widths without compact slicing or scatter.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from rfx.boundaries._cpml_slab import cpml_grid_metadata
    from rfx.stepping.slab import cut, place

    ctx = setup.step_fn_inv.context
    grid = ctx.cpml_grid
    spacing_names = ('dx_x_lo', 'dx_x_hi', 'dx_y_lo', 'dx_y_hi', 'dz_lo', 'dz_hi')
    # Capture the original objects before stripping, exactly as the legacy
    # runner does. No dtype conversion or grid-metadata reconstruction.
    electric = {name: getattr(ctx.cpml_params, name) for name in spacing_names}
    magnetic = ({name: getattr(ctx.cpml_params.magnetic, name) for name in spacing_names}
                if ctx.cpml_params.magnetic is not None else {})

    def profiles(params):
        if params is None:
            return None
        return params._replace(**dict.fromkeys(spacing_names),
                               magnetic=profiles(params.magnetic))

    params = profiles(ctx.cpml_params)
    # A leading rank dimension makes every profile a dynamic shard input.
    # apply removes that singleton rank dimension inside the local step.
    params = jax.tree.map(lambda a: place(jnp.broadcast_to(a, (layout.n_devices,) + a.shape),
                             NamedSharding(mesh, P('x'))), params)
    fields = {}
    psi = setup.carry_init['cpml']
    for name, value in zip(psi._fields, psi):
        face = name.split('_')[-1]
        if face[0] == 'x':
            owner = 0 if face.endswith('lo') else layout.n_devices - 1
            rows = jnp.zeros((layout.n_devices,) + value.shape, value.dtype)
            rows = rows.at[owner].set(value)
            value = rows.reshape((-1,) + value.shape[1:])
            fields[name] = place(value, NamedSharding(mesh, P('x')))
        else:
            fields[name] = cut(value, layout, 'cpml_psi', mesh=mesh)
    context = replace(ctx, cpml_params=None, cpml_grid=cpml_grid_metadata(grid),
                      cpml_spacings=(electric, magnetic))
    return context, {**setup.invariants, 'cpml_x': params}, {
        **setup.carry_init, 'cpml': type(psi)(**fields)}


from functools import partial
from jax.sharding import PartitionSpec as P
from rfx.boundaries.cpml import apply_cpml_e
from rfx.boundaries.cpml import apply_cpml_h
# Both updates round psi to its carry dtype in _cpml_slab.store_psi before
# partial-window writeback; casting after these calls would be too late.
from rfx.stepping.rank import rank_shard_map
from rfx.stepping.slab_update import cpml_absorbing_faces
from rfx.stepping.slab_update import cpml_h_materials
from rfx.stepping.slab_update import cpml_slab_frame
from rfx.stepping.slab_update import slab_e_component_materials


def _apply_cpml_e_shmap(state, cpml_params, cpml_state, n_cpml, dt, dx,
                         mesh, n_devices, ghost=1, materials=None, pad_x=0,
                         e_slab=None, e_materials=None,
                         *, ranks, grid, high_wall, e_curl_coeff=None, boundary=None):
    """Apply CPML E-field correction using shard_map.

    ``materials`` (the x-sharded :class:`MaterialArrays`) is a READ-ONLY
    input: it is threaded into the inner shard so the CPML correction is
    material-aware (#205). Its arrays are NOT returned, so they get extra
    ``P("x")`` entries in ``in_specs`` only. ``None`` reproduces the vacuum
    (pre-#205) coefficient bit-identically.

    The psi coefficient takes the permittivity the E update of the same step
    used (#1043): with ``e_slab = (nx_per, nx)`` each component's four-cell
    edge mean, built on each slab by
    :func:`rfx.model.electric_metrics.slab_e_component_materials`
    (#1303), including on dispersive models (#1260). ``e_materials`` can
    supply the same means already formed before the time loop.

    ``pad_x`` (#623, same class as #622): forwarded to
    :func:`rfx.runners.distributed._apply_cpml_e_distributed` so the x-hi
    CPML window on the last rank targets the real physical face (global
    node ``nx - 1``) instead of the padded slab end.
    """

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(
            P("x"),   # ex
            P("x"),   # ey
            P("x"),   # ez
            P("x"),   # hx
            P("x"),   # hy
            P("x"),   # hz
            # cpml psi arrays – all x-sharded
            P("x"), P("x"), P("x"), P("x"),  # ey_xlo/xhi, ez_xlo/xhi
            P("x"), P("x"), P("x"), P("x"),  # ex_ylo/yhi, ez_ylo/yhi
            P("x"), P("x"), P("x"), P("x"),  # ex_zlo/zhi, ey_zlo/zhi
            P("x"),  # materials (read-only, material-aware coefficient)
            P("x"),  # precomputed component means, if supplied
            P("x"),  # in-loop dispersive curl coefficient; no slab-sized loss
        ),
        out_specs=(
            P("x"), P("x"), P("x"),           # ex, ey, ez
            P("x"), P("x"), P("x"), P("x"),   # psi ey/ez x-faces
            P("x"), P("x"), P("x"), P("x"),   # psi ex/ez y-faces
            P("x"), P("x"), P("x"), P("x"),   # psi ex/ey z-faces
        ),
        check_rep=False,
    )
    def _cpml_e(ex, ey, ez, hx, hy, hz,
                psi_ey_xlo, psi_ey_xhi, psi_ez_xlo, psi_ez_xhi,
                psi_ex_ylo, psi_ex_yhi, psi_ez_ylo, psi_ez_yhi,
                psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi,
                mat_slab, means, curl, *, rank):
        # Reconstruct minimal state and cpml_state objects
        from rfx.core.yee import FDTDState as _FS
        _st = _FS(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
        if mat_slab is not None and means is None:
            means = slab_e_component_materials(mat_slab, *e_slab, rank=rank, high_wall=high_wall)
        eps_r_slab = None if means is None else means[0]
        _cs = cpml_state._replace(
            psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
            psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
            psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
            psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
            psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
            psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
        )
        from rfx.boundaries.electric_coefficient import loss_terms
        loss = loss_terms(means, dt) if means is not None and curl is None else None
        new_st, new_cs = apply_cpml_e(
            _st, cpml_params, _cs, grid, materials=mat_slab,
            inv_eps_r_update=None if eps_r_slab is None else tuple(1 / e for e in eps_r_slab),
            e_loss=loss, e_curl_coeff=curl, faces=cpml_absorbing_faces(grid), boundary=boundary,
            frame=cpml_slab_frame(ex.shape, ghost, pad_x),
            owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})
        return (
            new_st.ex, new_st.ey, new_st.ez,
            new_cs.psi_ey_xlo, new_cs.psi_ey_xhi,
            new_cs.psi_ez_xlo, new_cs.psi_ez_xhi,
            new_cs.psi_ex_ylo, new_cs.psi_ex_yhi,
            new_cs.psi_ez_ylo, new_cs.psi_ez_yhi,
            new_cs.psi_ex_zlo, new_cs.psi_ex_zhi,
            new_cs.psi_ey_zlo, new_cs.psi_ey_zhi,
        )

    (ex, ey, ez,
     psi_ey_xlo, psi_ey_xhi, psi_ez_xlo, psi_ez_xhi,
     psi_ex_ylo, psi_ex_yhi, psi_ez_ylo, psi_ez_yhi,
     psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi) = _cpml_e(
        ranks, state.ex, state.ey, state.ez,
        state.hx, state.hy, state.hz,
        cpml_state.psi_ey_xlo, cpml_state.psi_ey_xhi,
        cpml_state.psi_ez_xlo, cpml_state.psi_ez_xhi,
        cpml_state.psi_ex_ylo, cpml_state.psi_ex_yhi,
        cpml_state.psi_ez_ylo, cpml_state.psi_ez_yhi,
        cpml_state.psi_ex_zlo, cpml_state.psi_ex_zhi,
        cpml_state.psi_ey_zlo, cpml_state.psi_ey_zhi,
        materials, e_materials, e_curl_coeff,
    )
    new_state = state._replace(ex=ex, ey=ey, ez=ez)
    new_cpml = cpml_state._replace(
        psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
        psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
        psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
        psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
        psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
        psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
    )
    return new_state, new_cpml


def _apply_cpml_h_shmap(state, cpml_params, cpml_state, n_cpml, dt, dx,
                         mesh, n_devices, ghost=1, mu_r=None, pad_x=0,
                         *, ranks, grid, high_wall, slab):
    """Apply CPML H-field correction using shard_map.

    ``mu_r`` (the x-sharded per-cell relative permeability slab) is a new
    READ-ONLY input threaded into the inner shard for material-aware CPML
    (#205).  Like ``eps_r`` in the E-variant it gets one extra ``P("x")``
    in ``in_specs`` only and is not returned.  ``None`` reproduces the
    vacuum (pre-#205) coefficient bit-identically.

    ``pad_x`` (#623, same class as #622): forwarded to
    :func:`rfx.runners.distributed._apply_cpml_h_distributed` so the x-hi
    CPML window on the last rank targets the real physical face (global
    node ``nx - 1``) instead of the padded slab end.
    """

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(
            P("x"),   # ex
            P("x"),   # ey
            P("x"),   # ez
            P("x"),   # hx
            P("x"),   # hy
            P("x"),   # hz
            P("x"), P("x"), P("x"), P("x"),  # hy_xlo/xhi, hz_xlo/xhi
            P("x"), P("x"), P("x"), P("x"),  # hx_ylo/yhi, hz_ylo/yhi
            P("x"), P("x"), P("x"), P("x"),  # hx_zlo/zhi, hy_zlo/zhi
            P("x"),  # mu_r (read-only, material-aware coefficient)
        ),
        out_specs=(
            P("x"), P("x"), P("x"),           # hx, hy, hz
            P("x"), P("x"), P("x"), P("x"),   # psi hy/hz x-faces
            P("x"), P("x"), P("x"), P("x"),   # psi hx/hz y-faces
            P("x"), P("x"), P("x"), P("x"),   # psi hx/hy z-faces
        ),
        check_rep=False,
    )
    def _cpml_h(ex, ey, ez, hx, hy, hz,
                psi_hy_xlo, psi_hy_xhi, psi_hz_xlo, psi_hz_xhi,
                psi_hx_ylo, psi_hx_yhi, psi_hz_ylo, psi_hz_yhi,
                psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi,
                mu_r_slab, *, rank):
        from rfx.core.yee import FDTDState as _FS
        _st = _FS(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
        _cs = cpml_state._replace(
            psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
            psi_hz_xlo=psi_hz_xlo, psi_hz_xhi=psi_hz_xhi,
            psi_hx_ylo=psi_hx_ylo, psi_hx_yhi=psi_hx_yhi,
            psi_hz_ylo=psi_hz_ylo, psi_hz_yhi=psi_hz_yhi,
            psi_hx_zlo=psi_hx_zlo, psi_hx_zhi=psi_hx_zhi,
            psi_hy_zlo=psi_hy_zlo, psi_hy_zhi=psi_hy_zhi,
        )
        new_st, new_cs = apply_cpml_h(
            _st, cpml_params, _cs, grid,
            materials=cpml_h_materials(mu_r_slab, rank, ghost, nx_per=slab[0], nx=slab[1], high_wall=high_wall),
            faces=cpml_absorbing_faces(grid), frame=cpml_slab_frame(ex.shape, ghost, pad_x),
            owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})
        return (
            new_st.hx, new_st.hy, new_st.hz,
            new_cs.psi_hy_xlo, new_cs.psi_hy_xhi,
            new_cs.psi_hz_xlo, new_cs.psi_hz_xhi,
            new_cs.psi_hx_ylo, new_cs.psi_hx_yhi,
            new_cs.psi_hz_ylo, new_cs.psi_hz_yhi,
            new_cs.psi_hx_zlo, new_cs.psi_hx_zhi,
            new_cs.psi_hy_zlo, new_cs.psi_hy_zhi,
        )

    (hx, hy, hz,
     psi_hy_xlo, psi_hy_xhi, psi_hz_xlo, psi_hz_xhi,
     psi_hx_ylo, psi_hx_yhi, psi_hz_ylo, psi_hz_yhi,
     psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi) = _cpml_h(
        ranks, state.ex, state.ey, state.ez,
        state.hx, state.hy, state.hz,
        cpml_state.psi_hy_xlo, cpml_state.psi_hy_xhi,
        cpml_state.psi_hz_xlo, cpml_state.psi_hz_xhi,
        cpml_state.psi_hx_ylo, cpml_state.psi_hx_yhi,
        cpml_state.psi_hz_ylo, cpml_state.psi_hz_yhi,
        cpml_state.psi_hx_zlo, cpml_state.psi_hx_zhi,
        cpml_state.psi_hy_zlo, cpml_state.psi_hy_zhi,
        mu_r,
    )
    new_state = state._replace(hx=hx, hy=hy, hz=hz)
    new_cpml = cpml_state._replace(
        psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
        psi_hz_xlo=psi_hz_xlo, psi_hz_xhi=psi_hz_xhi,
        psi_hx_ylo=psi_hx_ylo, psi_hx_yhi=psi_hx_yhi,
        psi_hz_ylo=psi_hz_ylo, psi_hz_yhi=psi_hz_yhi,
        psi_hx_zlo=psi_hx_zlo, psi_hx_zhi=psi_hx_zhi,
        psi_hy_zlo=psi_hy_zlo, psi_hy_zhi=psi_hy_zhi,
    )
    return new_state, new_cpml

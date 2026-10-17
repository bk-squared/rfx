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


"""Place canonical CPML x operands; the boundary owner retains all arithmetic."""
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp

from rfx.boundaries.cpml import _axis_buffer_depths, _clip_lo, _clip_hi


def publish(setup):
    """Expand compact physical faces to global x rows before the common cut."""
    ctx = setup.step_fn_inv.context
    grid, params = ctx.cpml_grid, ctx.cpml_params
    nx = grid.nx
    depth = _axis_buffer_depths(grid, grid.cpml_layers)[0]
    payload = {}

    def expand(a, side, fill=0):
        pads = [(0, nx - a.shape[0]) if side == 'lo' else (nx - a.shape[0], 0)]
        return jnp.pad(a, pads + [(0, 0)] * (a.ndim - 1), constant_values=fill)

    def profiles(p, prefix=''):
        if p is None:
            return None
        for side in ('lo', 'hi'):
            profile = getattr(p, 'x_' + side)
            clip = _clip_lo if side == 'lo' else _clip_hi
            for name in profile._fields:
                a = clip(getattr(profile, name), depth, grid.cpml_layers)
                payload[prefix + 'x_' + side + '.' + name] = expand(a, side, int(name in ('b', 'kappa')))
            payload[prefix + 'dx_x_' + side] = jnp.full((nx,), getattr(p, 'dx_x_' + side))
        return p._replace(x_lo=None, x_hi=None, dx_x_lo=None, dx_x_hi=None,
                          magnetic=profiles(p.magnetic, 'magnetic.'))

    static = profiles(params)
    psi = setup.carry_init['cpml']
    psi = type(psi)(*(expand(a, name[-2:]) if name.split('_')[-1][0] == 'x' else a
                      for name, a in zip(psi._fields, psi)))
    # Static geometry retains y/z data and face depths, but no x vector.
    view = SimpleNamespace(**grid._asdict())
    for axis in 'xyz':
        for side in ('lo', 'hi'):
            setattr(view, f'pad_{axis}_{side}', getattr(grid, f'pad_{axis}_{side}'))
    view.cpml_axes = getattr(grid, 'cpml_axes', 'xyz')
    view.dx_arr = None
    view.inv_dx = view.inv_dx_h = None
    context = replace(ctx, cpml_params=static, cpml_grid=view)
    return context, {**setup.invariants, 'cpml_x': payload}, {**setup.carry_init, 'cpml': psi}


def operands(ctx, invariants, psi):
    """Recover compact face views from this rank's cut x rows."""
    from rfx.boundaries.cpml import CPMLParams
    grid = SimpleNamespace(**vars(ctx.cpml_grid))
    grid.dx_arr = invariants['dx_arr']
    grid.nx = grid.dx_arr.shape[0]
    depths = {side: max(1, getattr(grid, 'pad_x_' + side)) for side in ('lo', 'hi')}
    lo, hi = ctx.x_face_frame['x']
    windows = {'lo': slice(lo, lo + depths['lo']), 'hi': slice(hi - depths['hi'], hi)}
    payload = invariants['cpml_x']

    def profiles(p, prefix=''):
        if p is None:
            return None
        values = {}
        for side, window in windows.items():
            def profile_row(name):
                row = payload[prefix + 'x_' + side + '.' + name][window]
                pad = grid.cpml_layers - depths[side]
                return jnp.pad(row, (0, pad) if side == 'lo' else (pad, 0),
                               constant_values=int(name in ('b', 'kappa')))
            values['x_' + side] = CPMLParams(*(profile_row(name) for name in CPMLParams._fields))
            values['dx_x_' + side] = payload[prefix + 'dx_x_' + side][window, None, None]
        return p._replace(**values, magnetic=profiles(p.magnetic, 'magnetic.'))

    compact = type(psi)(*(a[windows[name[-2:]]] if name.split('_')[-1][0] == 'x' else a
                          for name, a in zip(psi._fields, psi)))
    return profiles(ctx.cpml_params), grid, compact, windows


def apply(kernel, ctx, invariants, state, psi, **kwargs):
    """Call the canonical boundary kernel and restore its cut psi layout."""
    params, grid, compact, windows = operands(ctx, invariants, psi)
    # The kernel keeps x psi at the one-device depth and reads each framed
    # face's realized rows (lo: first rows, hi: last rows); pad to that depth.
    full = grid.cpml_layers

    def pad(a, side):
        extra = full - a.shape[0]
        return jnp.pad(a, [(0, extra) if side == 'lo' else (extra, 0)] + [(0, 0)] * (a.ndim - 1))

    def active(a, side, depth):
        return a[:depth] if side == 'lo' else a[full - depth:]

    x_psi = lambda name: name.split('_')[-1][0] == 'x'
    padded = type(psi)(*(pad(a, name[-2:]) if x_psi(name) else a for name, a in zip(psi._fields, compact)))
    state, updated = kernel(state, params, padded, grid, ctx.cpml_axes_eff,
                           frame=ctx.x_face_frame, owned=invariants['owns_x'], **kwargs)
    updated = type(psi)(*(a.at[windows[name[-2:]]].set(
                              active(b, name[-2:], windows[name[-2:]].stop - windows[name[-2:]].start))
                          if x_psi(name) else b
                          for name, a, b in zip(psi._fields, psi, updated)))
    return state, updated


def export_state(psi, grid, layout):
    """Return psi in the runner's ``init_cpml_slabs`` layout (#1745).

    Field order with the rank merged into x: y/z faces keep this rank's x rows;
    x faces hold the one-device depth, real only on the owning end rank (rank 0
    for x_lo, the last rank for x_hi) and zero elsewhere.
    """
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


def import_state(psi, canonical, layout):
    """Recover one-device psi from the runner's per-rank input.

    ``init_cpml_slabs`` layout (#1745): field order on every rank, x faces
    (s_x, Ny, Nz) at the one-device depth, y/z faces over the rank's local x
    rows; the rank axis is leading or merged into x.
    """
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

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
    state, updated = kernel(state, params, compact, grid, ctx.cpml_axes_eff,
                           frame=ctx.x_face_frame, owned=invariants['owns_x'],
                           x_depths=tuple(w.stop - w.start for w in windows.values()), **kwargs)
    updated = type(psi)(*(a.at[windows[name[-2:]]].set(b) if name.split('_')[-1][0] == 'x' else b
                          for name, a, b in zip(psi._fields, psi, updated)))
    return state, updated


def export_state(psi, grid, layout):
    """Retain the runner's public derivative/component/remaining-axis packing."""
    fields = {}
    for name, a in zip(psi._fields, psi):
        _, component, face = name.split('_')
        axis, side = face[0], face[1:]
        depth = max(1, getattr(grid, 'pad_' + axis + '_' + side))
        rows = a.reshape((layout.n_devices, layout.nx_local) + a.shape[1:])
        window = [slice(None)] * 4
        if axis == 'x':
            end = layout.ghost_width + layout.nx_per_rank - layout.pad_x
            window[1] = (slice(layout.ghost_width, layout.ghost_width + depth) if side == 'lo'
                         else slice(end - depth, end))
        else:
            window[1 + 'xyz'.index(axis)] = slice(None, depth) if side == 'lo' else slice(-depth, None)
        rows = rows[tuple(window)]
        order = (axis, component[1], next(c for c in 'xyz' if c not in (axis, component[1])))
        rows = rows.transpose((0,) + tuple(1 + 'xyz'.index(c) for c in order))
        fields[name] = rows.reshape((layout.n_devices * depth,) + rows.shape[2:])
    return type(psi)(**fields)


def import_state(psi, canonical, layout):
    """Recover physical psi rows from the runner's existing compact input."""
    from rfx.runners._distributed_common import unstack_and_gather
    fields = {}
    for name, a in zip(psi._fields, psi):
        _, component, face = name.split('_')
        axis, side = face[0], face[1:]
        depth = a.shape[0] // layout.n_devices
        rows = a.reshape((layout.n_devices, depth) + a.shape[1:])
        order = (axis, component[1], next(c for c in 'xyz' if c not in (axis, component[1])))
        rows = rows.transpose((0,) + tuple(1 + order.index(c) for c in 'xyz'))
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

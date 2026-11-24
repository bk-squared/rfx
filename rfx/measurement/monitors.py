"""Frozen NTFF face stencils and block-current moment contributions.

Current moments combine H curl, post-update E and saved pre-update E. Channels
with the same block/component/weight prefix sum to one accumulated moment;
E_prev reads the saved state, not the current field. Geometry weights are the
realized monitor's volume/displacement weights derived from grid.cells.
"""

from rfx.model.graded_admission import host_cells
from itertools import product
from functools import lru_cache
from contextlib import nullcontext
from copy import copy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from .plan import Channel, NodeArrays, Owner, _COMPONENTS, _phase_values, _traced


def default_frequency_grid(freq_max, n_freqs, *, host=False):
    """The default DFT frequency grid: ``n_freqs`` points from ``freq_max/10`` to ``freq_max``.

    ``host=True`` gives the float64 numpy grid the graded run's S default has
    always used; it differs from the float32 device grid by up to 2 ULP.
    """
    if host:
        return np.linspace(freq_max / 10, freq_max, n_freqs)
    return jnp.asarray(_eager_frequency_grid(float(freq_max), int(n_freqs), jax.config.x64_enabled))


@lru_cache(maxsize=128)
def _eager_frequency_grid(freq_max, n_freqs, x64):
    """Host constant with eager linspace's arithmetic, independent of outer traces.

    Reset the outer trace and its constant-folding switch while evaluating
    linspace, then publish the same concrete bins in every caller.
    Cache by precision as well as bounds/count; never cache a tracer.
    Jit is forced on: under ``jax.disable_jit()`` linspace runs op by op and
    its float32 bins differ by up to 1 ULP, which the cache would then serve
    to every later call in the process.
    """
    from rfx.core.jax_utils import compile_time_switch
    with jax.core.eval_context(), jax.disable_jit(False):
        switch = compile_time_switch()
        with switch(False) if switch is not None else nullcontext():
            start, stop = freq_max / 10, freq_max
            result = np.asarray(jnp.linspace(start, stop, n_freqs))
    result.setflags(write=False)
    return result


def monitor_owners(sim, grid, *, conductors=None, periodic=None, overrides=None):
    from rfx.current_moments import monitor_for_simulation
    from rfx.farfield import make_ntff_box
    owners = []
    if sim._ntff is not None:
        lo, hi, freqs = sim._ntff
        owners.append(ntff_owner(grid, make_ntff_box(grid, lo, hi, freqs)))
    if sim._current_moments is not None:
        freqs = sim._current_moments[3]
        declaration = sim
        if _traced(freqs):
            declaration = copy(sim)
            spec = list(sim._current_moments)
            spec[3] = np.zeros(freqs.shape, dtype=np.float32)
            declaration._current_moments = tuple(spec)
        try:
            monitor = monitor_for_simulation(declaration, grid, sim._periodic_flags() if periodic is None else periodic,
                                             conductors=conductors, overrides=overrides)
        except NotImplementedError as exc:
            owners.append(Owner('current_moment:0', 'current_moment', (),
                                _phase_values(sim._current_moments[3]),
                                availability=f'unavailable: {exc}'))
        else:
            owner = current_moment_owner(grid, monitor)
            owners.append(replace(owner, frequencies=freqs) if _traced(freqs) else owner)
    return owners


def ntff_owner(grid, box, *, channels_only=False):
    """Freeze the normal interpolation and tangential midpoint averages."""
    channels = ntff_channels(grid, box)
    if channels_only:
        return channels
    return Owner('ntff_box:0', 'ntff', tuple(channels), _phase_values(box.freqs), source=box)


def ntff_channels(grid, box):
    """Rebuild at most one face channel at a time for validation."""
    bounds = ((box.i_lo, box.i_hi), (box.j_lo, box.j_hi), (box.k_lo, box.k_hi))
    from rfx.model.graded_admission import refuse_window_cells
    for a in range(3):
        refuse_window_cells(grid, a, slice(None), "waveguide ports NTFF owner")
    cells = [np.asarray(host_cells(grid, a)) for a in range(3)]
    for axis, side in product(range(3), range(2)):
        face = 'xyz'[axis] + ('_lo', '_hi')[side]
        plane = bounds[axis][side]
        tangents = [a for a in range(3) if a != axis]
        lower = getattr(box, 'w_'+face)
        u, v = np.meshgrid(*(np.arange(*bounds[a], dtype=np.int32) for a in tangents), indexing='ij')
        u, v = u.ravel(), v.ravel()
        areas = cells[tangents[0]][u] * cells[tangents[1]][v]
        for kind, component_axis in product(('E', 'H'), tangents):
            component = kind.lower()+'xyz'[component_axis]
            shift_axis = next(a for a in tangents if a != component_axis) if kind == 'E' else component_axis
            normals = ((plane-1, lower), (plane, 1-lower)) if kind == 'H' and box.face_centre else ((plane, 1.),)
            shifts = ((0, .5), (1, .5)) if box.face_centre else ((0, 1.),)
            stencil = tuple(product(normals, shifts))
            repeats = len(stencil)
            indices = [None] * 3
            indices[axis] = np.tile(np.asarray([p for (p, _), _ in stencil], np.int32), len(u))
            indices[tangents[0]] = np.repeat(u, repeats)
            indices[tangents[1]] = np.repeat(v, repeats)
            indices[shift_axis] += np.tile([shift for _, (shift, _) in stencil], len(u))
            i, j, k = indices
            nodes = NodeArrays(component=np.full(len(u)*repeats, _COMPONENTS.index(component), np.int8),
                               i=i, j=j, k=k,
                               weight=np.tile([float(normal*transverse)
                                               for (_, normal), (_, transverse) in stencil], len(u)),
                               area=np.repeat(areas, repeats))
            yield Channel(face+':'+component, kind, nodes, e_time_offset=0., h_time_offset=0.)
            del nodes


def current_moment_owner(grid, monitor, *, channels_only=False):
    """Freeze slab curls, sharing geometry across weights and E/E_prev."""
    from rfx.core.yee import EPS_0
    dtype = np.asarray(monitor.w_ex).dtype
    coef = dtype.type(EPS_0)/(jnp.asarray(grid.dt, dtype=dtype) if _traced(grid.dt)
                              else dtype.type(grid.dt))
    channels = current_moment_channels(monitor, coef)
    if channels_only:
        return channels
    return Owner('current_moment:0', 'current_moment', tuple(channels),
                 _phase_values(monitor.freqs), source=monitor)


def current_moment_channels(m, coef):
    """Yield channels in component/block/weight order without cell objects.

    Within each block, cells keep np.ndindex's row-major order.
    """
    inverse = tuple(np.asarray(a) for a in (m.inv_dx_e, m.inv_dy_e, m.inv_dz_e))
    starts = (m.i_lo, m.j_lo, m.k_lo)
    curls = (((5, 1), (4, 2)), ((3, 2), (5, 0)), ((4, 0), (3, 1)))
    segments = np.asarray(m.seg).ravel()
    order = np.argsort(segments, kind='stable')
    boundaries = np.searchsorted(segments[order], np.arange(m.n_blocks+1))
    for c, weights in enumerate((m.w_ex, m.w_ey, m.w_ez)):
        weights = np.asarray(weights)
        _, ni, nj, nk = weights.shape
        for block in range(m.n_blocks):
            xy = order[boundaries[block]:boundaries[block+1]]
            local = (np.repeat(xy // nj, nk).astype(np.int32),
                     np.repeat(xy % nj, nk).astype(np.int32),
                     np.tile(np.arange(nk, dtype=np.int32), len(xy)))
            indices = tuple(a+start for a, start in zip(local, starts))
            count = len(local[0])
            components = np.full(count, c, np.int8)
            h_components = np.tile(np.repeat([term[0] for term in curls[c]], 2).astype(np.int8), count)
            h_indices = [np.repeat(a, 4).reshape(count, 4) for a in indices]
            for term, (_, axis) in enumerate(curls[c]):
                h_indices[axis][:, 2*term+1] -= 1
            hi, hj, hk = (a.ravel() for a in h_indices)
            i, j, k = indices
            for w in range(m.n_weights):
                value = weights[w][local]
                prefix = f'block:{block}:component:{c}:weight:{w}:'
                electric = NodeArrays(component=components, i=i, j=j, k=k, weight=-value*coef)
                yield Channel(prefix+'E', 'H', electric,
                              h_time_offset=float(m.half_step)-.5)
                yield Channel(prefix+'E_prev', 'H', replace(electric, weight=-electric.weight),
                              sample_stage='pre-update', h_time_offset=float(m.half_step)-.5)
                del electric
                magnetic = np.empty((count, 4), dtype=weights.dtype)
                for term, (_, axis) in enumerate(curls[c]):
                    weight = value*inverse[axis][local[axis]]*m.curl_signs[2*c+term]
                    magnetic[:, 2*term], magnetic[:, 2*term+1] = weight, -weight
                yield Channel(prefix+'H', 'H', NodeArrays(component=h_components, i=hi, j=hj, k=hk,
                                                       weight=magnetic.ravel()),
                              h_time_offset=float(m.half_step)-.5)
                del magnetic, weight

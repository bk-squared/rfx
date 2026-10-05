"""Host adapters for existing modal and calculator measurement setups."""
from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np
import jax.numpy as jnp

from .plan import Channel, Node, Owner, ReferencePlane, _frequencies


def waveguide_owners(sim, grid, n_steps, nu):
    from rfx.runners.nonuniform import _build_waveguide_port_config_nu
    result = []
    for i, entry in enumerate(sim._waveguide_ports):
        freqs = jnp.asarray(_frequencies(sim, entry))
        cfgs = (_build_waveguide_port_config_nu(sim, entry, grid, freqs, n_steps) if nu
                else sim._build_waveguide_port_config(entry, grid, freqs, n_steps))
        if hasattr(cfgs, 'normal_axis'):
            cfgs = [cfgs]
        for mode, cfg in enumerate(cfgs):
            result.append(waveguide_owner(grid, cfg, port_id=i, mode_id=mode))
    return result


def waveguide_owner(grid, cfg, *, port_id=0, mode_id=0):
    """Copy the config's mode reference and expand its H co-location stencil."""
    axis = 'xyz'.index(cfg.normal_axis)
    axes = [a for a in range(3) if a != axis]
    area = np.asarray(cfg.aperture_dA)
    if not area.size:
        area = np.asarray(cfg.u_widths)[:, None] * np.asarray(cfg.v_widths)[None, :]
    channels = []
    for label, plane in (('probe', cfg.probe_x), ('ref', cfg.ref_x)):
        for kind, components, profiles in (
                ('E', (cfg.e_u_component, cfg.e_v_component), (cfg.ey_profile, cfg.ez_profile)),
                ('H', (cfg.h_u_component, cfg.h_v_component), (cfg.hy_profile, cfg.hz_profile))):
            nodes = []
            for component, profile in zip(components, profiles):
                weighted = np.asarray(profile) * area
                # Adjoint of the actual edge-clamped transverse H average.
                if kind == 'H':
                    for a, offset in reversed(tuple(enumerate(cfg.h_offset))):
                        if offset == 0:
                            continue
                        if offset != .5:
                            raise ValueError('unsupported waveguide H offset')
                        bar = .25 * weighted
                        expanded = 2.0 * bar
                        # Keep the sampler's transpose addition order, including
                        # the two clamped endpoints, in its profile dtype.
                        low, high = [slice(None)]*2, [slice(None)]*2
                        low[a], high[a] = 0, -1
                        expanded[tuple(high)] += bar[tuple(high)]
                        left, right = [slice(None)]*2, [slice(None)]*2
                        left[a], right[a] = slice(None, -1), slice(1, None)
                        expanded[tuple(right)] += bar[tuple(left)]
                        expanded[tuple(left)] += bar[tuple(right)]
                        expanded[tuple(low)] += bar[tuple(low)]
                        weighted = expanded
                for u, v in np.ndindex(weighted.shape):
                    for p, factor in (((plane, 1.0),) if kind == 'E' else
                                      ((plane-1, .5), (plane, .5))):
                        idx = [0, 0, 0]
                        idx[axis] = p % grid.shape[axis]
                        idx[axes[0]], idx[axes[1]] = cfg.u_lo+u, cfg.v_lo+v
                        nodes.append(Node(component, *idx, float(weighted[u, v])*factor,
                                          float(area[u, v])))
            channels.append(Channel(f'{"V" if kind == "E" else "I"}_{label}', kind,
                                    tuple(nodes), slot_offset=1,
                                    e_time_offset=0.0, h_time_offset=0.0))
    refs = tuple(ReferencePlane(name, axis, int(index), float(value)) for name, index, value in (
        ('source', cfg.x_index, cfg.source_x_m), ('reference', cfg.ref_x, cfg.reference_x_m),
        ('probe', cfg.probe_x, cfg.probe_x_m)))
    return Owner(f'waveguide_port:{port_id}:mode:{mode_id}', 'waveguide', tuple(channels),
                 tuple(float(f) for f in np.asarray(cfg.freqs)), refs, tuple(cfg.mode_indices),
                 f'{cfg.mode_type}{tuple(cfg.mode_indices)}: config transverse profiles times aperture_dA',
                 ('waveguide writes step k to slot k+1',))


def msl_owner(grid, *, port_id, direction, voltage_indices, span, trace_planes,
              h_stencil, frequencies, reference_index=None, nonuniform=False):
    """Describe the calculator's resolved V ladder and closed I contour.

    Arguments are the calculator setup's span, PEC trace planes and H stencil,
    not feed-profile normalization. Indices use the same propagation/width/z
    frame as msl_cross_section_span. No measurement convention is changed.
    """
    prop, width = 'xyz'.index(direction[1]), 1 if direction[1] == 'x' else 0
    cells = [np.asarray(grid.cells(a)).astype(np.float32).astype(float) if nonuniform
             else np.asarray(grid.cells(a)) for a in range(3)]
    offset = -1.0 if nonuniform else 0.0
    channels = []
    for number, p in enumerate(voltage_indices):
        nodes = []
        for k in range(span['n_lo'], trace_planes[0]):
            idx = [0, 0, k]
            idx[prop], idx[width] = int(p), span['w_centre']
            nodes.append(Node('ez', *idx, float(cells[2][k])))
        channels.append(Channel(f'V:{number}', 'E', tuple(nodes),
                                e_time_offset=offset, h_time_offset=offset))
    # Current uses the right-handed transverse frame (a,b), including for y ports.
    a, b = ((1, 2) if prop == 0 else (2, 0))
    alo, ahi, blo, bhi = ((span['w_lo'], span['w_hi'], *trace_planes) if prop == 0
                          else (*trace_planes, span['w_lo'], span['w_hi']))
    sign = -1.0 if direction[0] == '+' else 1.0
    nodes = []
    for p, interpolation in zip(h_stencil['h_indices'], h_stencil['weights']):
        for comp, fixed_axis, fixed, varying_axis, start, stop, s in (
                (a, b, blo-1, a, alo, ahi, 1), (a, b, bhi, a, alo, ahi, -1),
                (b, a, ahi, b, blo, bhi, 1), (b, a, alo-1, b, blo, bhi, -1)):
            for q in range(start, stop+1):
                idx = [0, 0, 0]
                idx[prop], idx[fixed_axis], idx[varying_axis] = int(p), fixed, q
                nodes.append(Node('h'+'xyz'[comp], *idx,
                                  float(sign*s*interpolation*cells[comp][q])))
    channels.append(Channel('I', 'H', tuple(nodes), e_time_offset=offset,
                            h_time_offset=offset))
    refs = [ReferencePlane(f'probe:{i}', prop, int(p), float(grid.node_of(prop, int(p))))
            for i, p in enumerate(voltage_indices)]
    if reference_index is not None:
        refs.append(ReferencePlane('deembedding', prop, int(reference_index),
                                   float(grid.node_of(prop, int(reference_index)))))
    return Owner(f'msl_port:{port_id}:mode:0', 'msl', tuple(channels),
                 tuple(float(f) for f in frequencies), tuple(refs),
                 mode_profile_reference='V: primal ladder; I: PEC trace planes and H stencil',
                 known_differences=('MSL calculator projector requires PEC trace',
                                    'uniform plane stamp n+1; NU plane stamp n'))


def coax_owner(grid, *, port_id, plane_indices, center_xy, pin_radius, outer_radius,
               frequencies, reference_index=None):
    """Describe the voltage ladders of the existing coax line calculators.

    These calculators fit plane voltages; they do not sample an H current
    contour. Keep that absence explicit instead of inventing a coax I channel.
    The plane indices must come from the selected calculator's setup because
    reflection, two-port and transition calculators choose different planes.
    """
    from rfx.nonuniform import NonUniformGrid
    if isinstance(grid, NonUniformGrid) or any(not grid.is_constant(a) for a in range(3)):
        raise ValueError('coax calculators require a uniform grid')
    coords = [np.asarray([grid.node_of(a, i) for i in range(grid.shape[a])]) for a in (0, 1)]
    u, v = coords
    cu, cv = center_xy
    radial = u-cu
    radial = radial[(radial >= pin_radius) & (radial <= outer_radius)]
    if radial.size < 2:
        radial = np.linspace(pin_radius, outer_radius, 33)
    widths = np.diff(radial)
    quadrature = np.r_[widths[0]/2, (widths[:-1]+widths[1:])/2, widths[-1]/2]
    weights = defaultdict(float)
    for r, length in zip(radial, quadrature):
        i = int(np.clip(np.searchsorted(u, cu+r)-1, 0, u.size-2))
        j = int(np.clip(np.searchsorted(v, cv)-1, 0, v.size-2))
        tu, tv = (cu+r-u[i])/(u[i+1]-u[i]), (cv-v[j])/(v[j+1]-v[j])
        for di, dj in product((0, 1), repeat=2):
            weights[i+di, j+dj] += float(length*(tu if di else 1-tu)*(tv if dj else 1-tv))
    channels = tuple(Channel(f'V:{n}', 'E', tuple(Node('ex', i, j, int(p), w)
                                                 for (i, j), w in sorted(weights.items())),
                             e_time_offset=0.0, h_time_offset=0.0)
                     for n, p in enumerate(plane_indices))
    refs = [ReferencePlane(f'probe:{i}', 2, int(p), float(grid.node_of(2, int(p))))
            for i, p in enumerate(plane_indices)]
    if reference_index is not None:
        refs.append(ReferencePlane('deembedding', 2, int(reference_index),
                                   float(grid.node_of(2, int(reference_index)))))
    return Owner(f'coax_port:{port_id}:mode:0', 'coax', channels,
                 tuple(float(f) for f in frequencies), tuple(refs),
                 mode_profile_reference='positive-x radial ray, bilinear interpolation, trapezoid',
                 known_differences=('coax calculator only; NU refused; I inferred from voltage fit',))


def build_msl_owners(sim, grid, *, frequencies):
    """Build calculator observations from its resolved declarations and grid.

    Uses the calculator's plane-placement and realized-trace setup APIs, with
    no registration or run. An invalid calculator layout raises its setup
    error, even if the same declaration can be used as a run-only feed.
    """
    from rfx.sources.msl_port import (
        msl_port_from_entry, msl_cross_section_span, msl_probe_x_coords_n,
        msl_h_plane_stencil, msl_physical_point, _msl_position_to_index,
    )
    from rfx.probes.msl_wave_decomp import realized_trace_planes_on_column
    from rfx.nonuniform import NonUniformGrid
    from .plan import _pec_edges
    edges = _pec_edges(sim, grid)
    result = []
    for i, entry in enumerate(sim._resolve_msl_probe_entries(grid)):
        port = msl_port_from_entry(entry)
        span = msl_cross_section_span(grid, port, require_contiguous_width=True)
        positions = msl_probe_x_coords_n(grid, port, n_probes=int(entry.n_probes),
                                        n_offset_cells=entry.n_probe_offset,
                                        n_spacing_cells=entry.n_probe_spacing)
        stencil = msl_h_plane_stencil(grid, port, positions[0])
        column = tuple(span['i_feed'] if a == span['prop_idx'] else span['w_centre']
                       for a in range(3) if a != span['normal_idx'])
        trace = realized_trace_planes_on_column(edges, span['normal_idx'], column,
                                                span['n_hi'], periodic=sim._periodic_flags())
        if trace[0] is None:
            raise ValueError(f'MSL port {i} has no realized PEC trace for measurement')
        indices = tuple(int(_msl_position_to_index(grid, msl_physical_point(
            port.direction, p, port.y_lo, port.z_lo))[span['prop_idx']]) for p in positions)
        result.append(msl_owner(grid, port_id=i, direction=port.direction,
                                voltage_indices=indices, span=span, trace_planes=trace,
                                h_stencil=stencil, frequencies=frequencies,
                                nonuniform=isinstance(grid, NonUniformGrid)))
    return tuple(result)

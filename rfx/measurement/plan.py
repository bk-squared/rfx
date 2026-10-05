"""Immutable, host-only snapshot of today's measurement conventions.

M1 is observational: no runner imports this module. Call the builder once with
its realized grid and retain the returned snapshot. There are no buffers,
waveforms, material masks, or mutable array leaves in the result. Sampling
weights and quadrature areas are separate: a raw DFT plane does not integrate
its field, whereas a modal channel does.

TimeBase is physical leapfrog: scan step n has E=(n+1)*dt, H=(n+.5)*dt.
Channel time offsets, in units of dt, describe TODAY's stamps relative to
that clock (including raw H planes stamped as E). They do not change stepping.
slot_offset describes storage only: scan step n writes record[n+slot_offset].
It is independent of the timestamp; waveguide writes slot n+1 with zero time
offset. Raw point probes have no Fourier stamp and use their physical
sample time. Current-moment E contributions are stamped at the half-step current
time; their pre-update/post-injection stages distinguish the two E states.
Calculator-only observations require their actual setup (MSL/coax calculators
choose planes independently of run()). Missing observations remain owners with
an availability explanation, never invented run() channels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Node:
    component: str
    i: int
    j: int
    k: int
    weight: float
    area: float | None = None


@dataclass(frozen=True)
class Channel:
    name: str
    kind: str
    nodes: tuple[Node, ...]
    sample_stage: str = 'post-injection'
    slot_offset: int = 0
    e_time_offset: float = -1.0
    h_time_offset: float = -1.0


@dataclass(frozen=True)
class ReferencePlane:
    name: str
    axis: int
    index: int
    coordinate_m: float


@dataclass(frozen=True)
class Owner:
    id: str
    kind: str
    channels: tuple[Channel, ...]
    frequencies: tuple[float, ...] = ()
    reference_planes: tuple[ReferencePlane, ...] = ()
    mode: tuple[int, int] | None = None
    mode_profile_reference: str | None = None
    known_differences: tuple[str, ...] = ()
    availability: str = 'sampled'


@dataclass(frozen=True)
class TimeBase:
    dt: float
    record_length: int
    e_offset: float = 1.0
    h_offset: float = 0.5
    host_step_dtype: str = 'int64'
    scan_step_dtype: str = 'int32'
    host_time_dtype: str = 'float64'

    def time(self, n: int, kind: str) -> float:
        """Compute from the integer clock, never repeated addition."""
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise TypeError('measurement step must be an integer')
        if kind not in ('E', 'H'):
            raise ValueError('channel kind must be E or H')
        return float((np.float64(n) + (self.e_offset if kind == 'E' else self.h_offset))
                     * np.float64(self.dt))


@dataclass(frozen=True)
class MeasurementPlan:
    path: str
    time_base: TimeBase
    owners: tuple[Owner, ...]
    schema_version: int = 1
    known_differences: tuple[str, ...] = ()


def _frequencies(sim, entry=None, freqs=None):
    if freqs is None and entry is not None:
        freqs = getattr(entry, 'freqs', None)
    if freqs is None:
        freqs = jnp.linspace(sim._freq_max / 10, sim._freq_max,
                             getattr(entry, 'n_freqs', 50))
    return tuple(float(v) for v in np.asarray(freqs))


def _index(grid, position):
    from rfx.nonuniform import NonUniformGrid, position_to_index
    indices = position_to_index(grid, position) if isinstance(grid, NonUniformGrid) else grid.position_to_index(position)
    return tuple(int(i) for i in indices)


def _plane_nodes(grid, axis, index, component, region=None, weights=None):
    axes = [a for a in range(3) if a != axis]
    sizes = tuple(len(grid.cells(a)) for a in axes)
    lo, hi, low, high = region or (0, sizes[0], 0, sizes[1])
    cells = [np.asarray(grid.cells(a)) for a in axes]
    nodes = []
    for u, v in product(range(lo, hi), range(low, high)):
        idx = [0, 0, 0]
        idx[axis], idx[axes[0]], idx[axes[1]] = index, u, v
        w = 1.0 if weights is None else float(weights[u-lo, v-low])
        nodes.append(Node(component, *idx, w, float(cells[0][u] * cells[1][v])))
    return tuple(nodes)


def _pec_edges(sim, grid):
    from rfx.boundaries.pec import realized_pec_edge_masks
    sheets, wires = [], []
    assemble = sim._assemble_materials_nu if sim._uses_nonuniform_mesh else sim._assemble_materials
    result = assemble(grid, pec_sheets=sheets, pec_wires=wires)
    if result[3] is None and not sheets and not wires:
        return None
    return realized_pec_edge_masks(result[3], sheets=tuple(sheets), wires=tuple(wires),
                                   periodic=sim._periodic_flags())


def _port_cells(grid, axis, nu):
    # Today's NU setup reads solver stores: f32 widened on x/y, f32 on z.
    # Derive from cells, retaining that rounding instead of moving a weight.
    cells = np.asarray(grid.cells(axis))
    if nu:
        cells = cells.astype(np.float32)
        if axis != 2:
            cells = cells.astype(np.float64)
    return cells


def _port_channels(sim, grid, entry, live, nu):
    mid = live[len(live)//2]
    axis = 'xyz'.index(entry.component[1])
    widths = _port_cells(grid, axis, nu)
    voltage = tuple(Node(entry.component, *p, -float(widths[p[axis]])) for p in live)
    v = (Node(entry.component, *mid, -float(widths[mid[axis]])),)
    # Ampere circulation: each H term is weighted by its own dual edge.
    pairs = {'ez': (('hy', 0, 1), ('hx', 1, -1)),
             'ex': (('hz', 1, 1), ('hy', 2, -1)),
             'ey': (('hx', 2, 1), ('hz', 0, -1))}
    current = []
    for component, back_axis, sign in pairs[entry.component]:
        length_axis = 'xyz'.index(component[1])
        c = _port_cells(grid, length_axis, nu)
        p = mid[length_axis]
        width = float(c[p] if not nu or p == 0 else (c[p-1]+c[p])*0.5)
        from rfx.boundaries.pmc import magnetic_image_faces
        faces = magnetic_image_faces(sim._boundary_spec.pmc_faces(), grid.shape, sim._periodic_flags())
        wraps = sim._periodic_flags()[back_axis] or len(grid.cells(back_axis)) == 1
        factor = 1
        if not wraps and mid[back_axis] == 0 and f'{"xyz"[back_axis]}_lo' in faces:
            factor = 2
        if not wraps and mid[back_axis] == grid.shape[back_axis]-1 and f'{"xyz"[back_axis]}_hi' in faces:
            factor = 0
        current.append(Node(component, *mid, sign*width*factor))
        back = list(mid)
        back[back_axis] -= 1
        if back[back_axis] < 0:
            if sim._periodic_flags()[back_axis] or len(grid.cells(back_axis)) == 1:
                back[back_axis] %= len(grid.cells(back_axis))
            else:
                # h_neighbor's zero exterior contributes no sampled node.
                continue
        current.append(Node(component, *back, -sign*width*(2 if factor == 0 else 1)))
    channels = [Channel('V', 'E', v), Channel('I', 'H', tuple(current))]
    if entry.extent is not None or nu:
        channels.append(Channel('V_port', 'E', voltage))
    if not nu:
        channels.append(Channel('V_ref', 'E', v, sample_stage='pre-injection'))
    return tuple(channels)


def build_measurement_plan(sim, grid, *, n_steps, path=None, frequencies=None,
                           calculator_owners=()):
    """Build a frozen snapshot from a Simulation and the run's realized grid.

    ``path`` selects run/forward availability on the selected mesh. Optional
    ``calculator_owners`` are frozen observations made by the calculator setup
    adapters; calculator planes cannot be inferred from a run declaration.
    ``frequencies`` is the run's explicit port frequency request, if any.
    """
    from rfx.nonuniform import NonUniformGrid
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError('n_steps must be a nonnegative integer')
    nu = isinstance(grid, NonUniformGrid)
    path = path or ('run_nonuniform' if nu else 'run_uniform')
    allowed = ('run_nonuniform', 'fwd_nonuniform') if nu else ('run_uniform', 'fwd_uniform')
    if path == ('msl_nonuniform' if nu else 'msl_uniform'):
        from .ports import build_msl_owners
        if calculator_owners:
            raise ValueError('msl path builds its own calculator owners')
        bins = frequencies if frequencies is not None else jnp.linspace(sim._freq_max/10, sim._freq_max, 100)
        calculator_owners = build_msl_owners(sim, grid, frequencies=bins)
    elif path not in allowed:
        raise ValueError(f'unsupported measurement path {path!r} for this grid; expected {allowed}')
    owners = []
    for i, entry in enumerate(sim._probes):
        owners.append(Owner(f'probe:{i}', 'probe',
                            (Channel(entry.component, entry.component[0].upper(),
                                     (Node(entry.component, *_index(grid, entry.position), 1.0),),
                                     e_time_offset=0., h_time_offset=0.),)))
    for i, entry in enumerate(sim._dft_planes):
        axis = 'xyz'.index(entry.axis)
        point = [0.0, 0.0, 0.0]
        point[axis] = entry.coordinate
        index = _index(grid, tuple(point))[axis]
        region = getattr(sim, '_dft_plane_regions', {}).get(entry.name)
        offset = 0.0 if nu else 1.0
        # Today's DFT plane gives H the E stamp; the MSL calculator corrects it later.
        channel = Channel(entry.component, entry.component[0].upper(),
                          _plane_nodes(grid, axis, index, entry.component, region),
                          e_time_offset=offset-1, h_time_offset=offset-.5)
        owners.append(Owner(f'dft_plane:{i}', 'dft_plane', (channel,),
                            _frequencies(sim, entry), known_differences=(
                                'uniform plane E stamp n+1; NU plane E stamp n',
                                'raw H plane uses E stamp; consumer owns half-step correction')))
    from rfx.probes.flux_region import resolve_flux_region
    from rfx.probes.probes import init_flux_monitor, _FLUX_COMPONENTS
    from rfx.boundaries.pec import resolve_wall_faces
    from rfx.boundaries.pmc import magnetic_image_faces
    for i, entry in enumerate(sim._flux_monitors):
        axis = 'xyz'.index(entry.axis)
        point = [0.0, 0.0, 0.0]
        point[axis] = entry.coordinate
        index = _index(grid, tuple(point))[axis]
        tangential = [a for a in range(3) if a != axis]
        region = resolve_flux_region(grid, entry, sim._domain)
        bounds = (0, len(grid.cells(tangential[0])), 0, len(grid.cells(tangential[1])))
        if region is not None:
            index = region['normal_index']
            bounds = tuple(v for span in region['cell_slices'] for v in span)
        cells = [np.asarray(grid.cells(a)) for a in tangential]
        cfg = init_flux_monitor(axis, index, jnp.asarray(_frequencies(sim, entry)), grid.shape,
                                cells[0] if nu else cells[0][0],
                                cells[1] if nu else cells[1][0],
                                lo1=bounds[0], hi1=bounds[1], lo2=bounds[2], hi2=bounds[3],
                                pmc_faces=(magnetic_image_faces(sim._boundary_spec.pmc_faces(), grid.shape)
                                           if nu else resolve_wall_faces(grid, sim._periodic_flags())[1]),
                                staggered_area=nu)
        shape = (bounds[1]-bounds[0], bounds[3]-bounds[2])
        areas = (np.broadcast_to(np.asarray(cfg.dA), shape),
                 np.broadcast_to(np.asarray(cfg.dA if cfg.dA2 is None else cfg.dA2), shape))
        channels = []
        for c, component in enumerate(_FLUX_COMPONENTS[axis]):
            # Weight belongs to the E/H product, not to each linear DFT.
            area = areas[(0, 1, 1, 0)[c]]
            planes = ((index, 1.0),) if c < 2 else ((max(index-1, 0), .5), (index, .5))
            nodes = tuple(Node(n.component, n.i, n.j, n.k, weight, float(a))
                          for plane, weight in planes
                          for n, a in zip(_plane_nodes(grid, axis, plane, component, bounds), area.flat))
            offset = 0.0 if nu else 1.0
            channels.append(Channel(component, component[0].upper(), nodes,
                                    e_time_offset=offset-1, h_time_offset=offset-1))
        owners.append(Owner(f'flux_plane:{i}', 'flux', tuple(channels), _frequencies(sim, entry),
                            known_differences=('uniform flux stamp n+1; NU flux stamp n',
                                               'uniform dA broadcast; NU staggered dA and dA2'),
                            availability='result missing' if path.startswith('fwd') else 'sampled'))
    edges = _pec_edges(sim, grid) if any(p.impedance and p.extent is not None for p in sim._ports) else None
    from rfx.sources.sources import WirePort, _wire_port_live_cells
    for i, entry in enumerate(sim._ports):
        if not entry.impedance:
            continue  # a soft source has no measurement owner
        live = (_index(grid, entry.position),)
        if entry.extent is not None:
            end = list(entry.position)
            end['xyz'.index(entry.component[1])] += entry.extent
            port = WirePort(entry.position, tuple(end), entry.component, entry.impedance,
                            entry.waveform, radius=entry.radius)
            cells, flags, _ = _wire_port_live_cells(grid, port, edges)
            live = tuple(tuple(int(v) for v in p) for p, flag in zip(cells, flags) if flag)
        if not live:
            raise ValueError(f'port:{i} has no live measurement edges')
        references = ()
        if getattr(entry, 'reference_plane_cells', None) is not None:
            line_axis = 'xyz'.index(entry.direction[1])
            outboard = -1 if entry.direction[0] == '+' else 1
            target = live[len(live)//2][line_axis]
            indices = (target, target+outboard*entry.reference_plane_cells,
                       target+outboard*2*entry.reference_plane_cells)
            references = tuple(ReferencePlane(name, line_axis, int(p),
                                              float(grid.node_of(line_axis, int(p))))
                               for name, p in zip(('port', 'reference', 'impedance'), indices))
        owners.append(Owner(f'port:{i}:mode:0', 'wire' if entry.extent is not None else 'lumped',
                            _port_channels(sim, grid, entry, live, nu),
                            _frequencies(sim, freqs=(np.asarray(frequencies, dtype=np.float32)
                                                    if frequencies is not None else None)),
                            reference_planes=references,
                            known_differences=(('NU missing pre-injection V_ref',
                                                'NU port weights retain solver-store float32 rounding') if nu else ())
                            + (('wire reference planes belong to multi-drive scan, not diagonal forward',)
                               if references else ()),
                            availability=(('V/I unavailable in main scan; sampled in second scan when compute_s_params is on'
                                           if not nu else 'V/I unavailable in run without wire extent')
                                          if entry.extent is None and path.startswith('run') else 'sampled')))
        if entry.extent is None and edges is not None:
            from rfx.boundaries.pec import clear_edges
            edges = clear_edges(edges, list(live), component=entry.component)
    from .ports import waveguide_owners
    owners.extend(waveguide_owners(sim, grid, n_steps, nu))
    supplied = {owner.id: owner for owner in calculator_owners}
    for attr, kind in (('_msl_ports', 'msl'), ('_coaxial_ports', 'coax')):
        for i, _ in enumerate(getattr(sim, attr)):
            key = f'{kind}_port:{i}:mode:0'
            owners.append(supplied.pop(key) if key in supplied else Owner(
                key, kind, (), availability='calculator setup required; run() has no V/I projector',
                known_differences=(('coax calculator refuses NU; no run() measurement' if kind == 'coax'
                                    else 'MSL calculator owns V ladder and PEC-anchored I stencil'),)))
    owners.extend(supplied.values())
    from .monitors import monitor_owners
    owners.extend(monitor_owners(sim, grid))
    owners = [with_clock_differences(owner) for owner in owners]
    ids = [o.id for o in owners]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate measurement owner id')
    return MeasurementPlan(path, TimeBase(float(grid.dt), int(n_steps)), tuple(owners))


def measurement_plan(sim, *, n_steps, path=None, frequencies=None, calculator_owners=()):
    """Frozen measurement plan on the grid this configuration builds (read-only; no path consumes it yet)."""
    return build_measurement_plan(
        sim, sim._build_realized_grid(), n_steps=n_steps, path=path,
        frequencies=frequencies, calculator_owners=calculator_owners)


def with_clock_differences(owner):
    """Enumerate every channel whose present stamp differs from the time base."""
    flags = tuple(f'{c.name}: {c.kind} stamp offset {offset:+g} dt from physical clock'
                  for c in owner.channels
                  for offset in (c.e_time_offset if c.kind == 'E' else c.h_time_offset,)
                  if offset != 0)
    return replace(owner, known_differences=tuple(dict.fromkeys((*owner.known_differences, *flags))))

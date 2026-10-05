"""Frozen NTFF face stencils and block-current moment contributions.

Current moments combine H curl, post-update E and saved pre-update E. Channels
with the same block/component/weight prefix sum to one accumulated moment;
E_prev reads the saved state, not the current field. Geometry weights are the
realized monitor's volume/displacement weights derived from grid.cells.
"""
from itertools import product

import numpy as np

from .plan import Channel, Node, Owner


def monitor_owners(sim, grid):
    from rfx.current_moments import monitor_for_simulation
    from rfx.farfield import make_ntff_box
    owners = []
    if sim._ntff is not None:
        lo, hi, freqs = sim._ntff
        owners.append(ntff_owner(grid, make_ntff_box(grid, lo, hi, freqs)))
    if sim._current_moments is not None:
        try:
            monitor = monitor_for_simulation(sim, grid, sim._periodic_flags())
        except NotImplementedError as exc:
            owners.append(Owner('current_moment:0', 'current_moment', (),
                                tuple(float(f) for f in sim._current_moments[3]),
                                availability=f'unavailable: {exc}'))
        else:
            owners.append(current_moment_owner(grid, monitor))
    return owners


def ntff_owner(grid, box):
    """Expand the normal interpolation and tangential midpoint averages."""
    bounds = ((box.i_lo, box.i_hi), (box.j_lo, box.j_hi), (box.k_lo, box.k_hi))
    channels = []
    cells = [np.asarray(grid.cells(a)) for a in range(3)]
    for axis, side in product(range(3), range(2)):
        face = 'xyz'[axis] + ('_lo', '_hi')[side]
        plane = bounds[axis][side]
        tangents = [a for a in range(3) if a != axis]
        lower = getattr(box, 'w_'+face)
        for kind, component_axis in product(('E', 'H'), tangents):
            component = kind.lower()+'xyz'[component_axis]
            shift_axis = next(a for a in tangents if a != component_axis) if kind == 'E' else component_axis
            normals = ((plane-1, lower), (plane, 1-lower)) if kind == 'H' and box.face_centre else ((plane, 1.),)
            shifts = ((0, .5), (1, .5)) if box.face_centre else ((0, 1.),)
            nodes = []
            for u, v in product(range(*bounds[tangents[0]]), range(*bounds[tangents[1]])):
                area = float(cells[tangents[0]][u]*cells[tangents[1]][v])
                for (p, normal), (shift, transverse) in product(normals, shifts):
                    idx = [0, 0, 0]
                    idx[axis], idx[tangents[0]], idx[tangents[1]] = p, u, v
                    idx[shift_axis] += shift
                    nodes.append(Node(component, *idx, float(normal*transverse), area))
            channels.append(Channel(face+':'+component, kind, tuple(nodes),
                                    e_time_offset=0., h_time_offset=0.))
    return Owner('ntff_box:0', 'ntff', tuple(channels), tuple(float(f) for f in box.freqs))


def current_moment_owner(grid, monitor):
    """Expand the existing slab curl and volume-weighted block reductions."""
    from rfx.core.yee import EPS_0
    m = monitor
    dtype = np.asarray(m.w_ex).dtype
    coef = dtype.type(EPS_0)/dtype.type(grid.dt)
    inverse = tuple(np.asarray(a) for a in (m.inv_dx_e, m.inv_dy_e, m.inv_dz_e))
    starts = (m.i_lo, m.j_lo, m.k_lo)
    curls = ((('hz', 1), ('hy', 2)), (('hx', 2), ('hz', 0)), (('hy', 0), ('hx', 1)))
    channels = []
    for c, weights in enumerate((m.w_ex, m.w_ey, m.w_ez)):
        weights = np.asarray(weights)
        segments = np.asarray(m.seg).reshape(weights.shape[1:3])
        for block, w in product(range(m.n_blocks), range(m.n_weights)):
            electric, magnetic = [], []
            for index in np.ndindex(weights.shape[1:]):
                if segments[index[:2]] != block:
                    continue
                value = weights[(w, *index)]
                idx = tuple(p+q for p, q in zip(starts, index))
                electric.append(Node('e'+'xyz'[c], *idx, float(-value*coef)))
                for term, (component, axis) in enumerate(curls[c]):
                    weight = float(value*inverse[axis][index[axis]]*m.curl_signs[2*c+term])
                    back = list(idx)
                    back[axis] -= 1
                    magnetic.extend((Node(component, *idx, weight), Node(component, *back, -weight)))
            prefix = f'block:{block}:component:{c}:weight:{w}:'
            previous = tuple(Node(n.component, n.i, n.j, n.k, -n.weight) for n in electric)
            for name, kind, nodes, stage in (('E', 'E', tuple(electric), 'post-injection'),
                                             ('E_prev', 'E', previous, 'pre-update'),
                                             ('H', 'H', tuple(magnetic), 'post-injection')):
                channels.append(Channel(prefix+name, kind, nodes, sample_stage=stage,
                                        e_time_offset=float(m.half_step)-1,
                                        h_time_offset=float(m.half_step)-.5))
    return Owner('current_moment:0', 'current_moment', tuple(channels),
                 tuple(float(f) for f in m.freqs))

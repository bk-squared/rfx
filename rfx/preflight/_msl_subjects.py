"""Owners of the material column and conductor faces inspected by an MSL check."""
import numpy as np


def conductor_owners(sim, pe, assembled):
    if assembled is None:
        return ()
    from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry
    grid = assembled[0]
    try:
        span = msl_cross_section_span(grid, msl_port_from_entry(pe))
    except (ValueError, NotImplementedError):
        return ()
    normal, prop, width = (span[key] for key in ('normal_idx', 'prop_idx', 'width_idx'))
    tangent = [a for a in range(3) if a != normal]
    owners = []
    for entry in sim._campaign_ctx().entry_realizations():
        if not entry.is_pec:
            continue
        for k in (span['n_lo'], span['n_hi']):
            foot = entry.footprint_on_plane(normal, k, assembled[-1].periodic, grid.shape)
            index = [slice(None), slice(None)]
            index[tangent.index(prop)] = span['i_feed']
            index[tangent.index(width)] = list(span['width_nodes'])
            if np.any(foot[tuple(index)]):
                owners.append(entry)
                break
    faces = sim._boundary_spec.pec_faces()
    for side, k in (('lo', span['n_lo']), ('hi', span['n_hi'])):
        pad = getattr(grid, f'pad_{"xyz"[normal]}_{side}')
        face_k = pad if side == 'lo' else grid.shape[normal] - pad - 1
        if f'{"xyz"[normal]}_{side}' in faces and k == face_k:
            owners.append('domain')
    return tuple(owners)


def material_owners(sim, realized, index, normal, start, count):
    selected = list(index)
    selected[normal] = slice(start, start + count)
    masks = dict(realized.geometry_masks)
    return tuple(entry for entry in sim._geometry
                 if id(entry) in masks and np.any(np.asarray(masks[id(entry)])[tuple(selected)]))

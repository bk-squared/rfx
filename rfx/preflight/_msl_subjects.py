"""Owners of the material column and conductor faces inspected by an MSL check."""
import numpy as np


def conductor_owners(sim, pe, assembled):
    if assembled is None:
        return ()
    from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry
    from rfx.preflight.realization import context_from_conductors
    grid = assembled[0]
    try:
        span = msl_cross_section_span(grid, msl_port_from_entry(pe))
    except (ValueError, NotImplementedError):
        return ()
    normal, prop, width = (span[key] for key in ('normal_idx', 'prop_idx', 'width_idx'))
    tangent = [a for a in range(3) if a != normal]
    owners = []
    for entry in context_from_conductors(sim, assembled[-1]).entry_realizations():
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


def _msl_realized_substrate(self, pe, inr, assembled=None):
    """Same-permittivity material-column extent under a port, or None.

    This is NOT the conductor-plane gap. The material walk can include
    sample slots also owned by the trace PEC volume; the lattice ownership contract uses different
    samplers for dielectric material and PEC volume ownership. Keep its
    count/extent separate from the validated normal source interval.

    On a graded mesh, derive realized substrate thickness from the local profile. Multiplying ``n_cells`` by the
    uniform scalar ``dx`` can report 300 um (+18%) for a three-cell substrate that ``fidelity_report()`` measures as
    254.00 um on the same simulation.

    So this reads the substrate the way ``rfx.fidelity`` does: build
    the grid the run will use (non-uniform when any profile is set),
    assemble materials on it, walk the permittivity column under the
    port along the substrate-normal axis from the ground plane, and
    sum the ACTUAL cell sizes of the cells that carry the substrate's
    permittivity. Returns a dict with

      n          same-permittivity sample-slot count under the port,
      h_real     their summed material-column extent (m),
      frac       where the DECLARED top face sits inside the cell that
                 contains it, as a fraction of that cell (0 = on a node),
      declared_n_above  upper face-bracket count relative to the port's
                 canonical ground node (independent of material n),
      d_iface    that cell's size (m),
      nonuniform whether a mesh profile was in force,

    or None when the material column cannot be derived. Declared-face
    geometry is available independently through _msl_declared_face_geometry;
    the compatibility frac/bracket fields here delegate to that helper.

    ``assembled`` is the per-check cache from :meth:`_msl_assemble_once`
    (grid, materials, per-axis cell sizes, nonuniform flag) so N MSL
    ports cost one rasterization, not N.

    The walk starts at the PORT'S OWN ground plane -- ``pe.position``'s
    substrate-normal coordinate -- not at the domain floor. The first
    version started at ``pad_lo`` (domain z = 0), which is only right
    when the ground plane sits on the floor; on a stripline-like or
    multi-layer stack it walked whatever dielectric lies BELOW the
    ground and reported that as "the substrate" (a
    ground plane at 800 um over an eps_r 9 filler read back n=10,
    h_real=800 um, never seeing the 254 um / eps_r 3.66 substrate above
    it). The class, reintroduced in a new form -- fixed here.

    Lattice ownership contract: the walk starts at the
    first cell ABOVE the realized ground. A sheet ground at the port
    plane owns no cell, so that is the port's own cell; a VOLUME
    ground drawn upward from the port plane (``Box(z_g, z_g + t)``)
    occupies the cells ``k0 .. k0+n-1`` and its top face is a wall at
    ``k0+n`` — the substrate begins there, and the returned dict says
    how many conductor cells were skipped (``ground_cells``) so the
    caller can name a port that sits UNDER its own ground.
    """
    try:
        if assembled is None:
            assembled = self._msl_assemble_once()
        if assembled is None:
            return None
        grid, mats, sizes, nonuniform, realized = assembled
        pos = tuple(float(v) for v in pe.position)
        if nonuniform:
            from rfx.nonuniform import position_to_index as _nu_p2i
            idx = list(_nu_p2i(grid, pos))
        else:
            idx = list(grid.position_to_index(pos))
        # Keep propagation/width lookup unchanged, but use the source's
        # lower-tie normal plane. Generic uniform banker's rounding
        # can start a half-cell-offset ground in the next dielectric layer.
        from rfx.sources.msl_port import _msl_normal_bounds, msl_port_from_entry
        idx[inr] = _msl_normal_bounds(grid, msl_port_from_entry(pe))[0]
        eps = np.asarray(mats.eps_r, dtype=float)
        k0 = int(idx[inr])  # the port's own ground plane, NOT pads[inr]
        ground_cells = 0
        if realized.pec_mask is not None:
            sl = [int(idx[0]), int(idx[1]), int(idx[2])]
            sl[inr] = slice(k0, None)
            occ = np.asarray(realized.pec_mask[tuple(sl)], dtype=bool).ravel()
            while ground_cells < occ.size and occ[ground_cells]:
                ground_cells += 1
            k0 += ground_cells
        sl = [int(idx[0]), int(idx[1]), int(idx[2])]
        sl[inr] = slice(k0, None)
        col = np.asarray(eps[tuple(sl)], dtype=float).ravel()
        if col.size == 0 or col[0] <= 1.0 + 1e-6:
            return None
        eps_sub = float(col[0])
        n = 0
        while n < col.size and abs(col[n] - eps_sub) <= 1e-3 * eps_sub:
            n += 1
        ax_sizes = sizes[inr][k0:]
        h_real = float(np.sum(ax_sizes[:n]))
        try:
            face = self._msl_declared_face_geometry(pe, grid)
        except (ValueError, TypeError, NotImplementedError):
            face = {}
        return dict(n=int(n), h_real=h_real, frac=face.get("frac"),
                    declared_n_above=face.get("declared_n_above"),
                    d_iface=face.get("d_iface"), nonuniform=bool(nonuniform),
                    ground_cells=int(ground_cells),
                    owners=material_owners(self, realized, idx, inr, k0, n))
    except (ValueError, NotImplementedError):
        return None

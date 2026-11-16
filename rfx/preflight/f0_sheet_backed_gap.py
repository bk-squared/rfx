"""Missing wall loss behind realized surface-impedance films."""

import jax
import numpy as np

from rfx.boundaries.pec import resolve_wall_faces, wall_edge_masks
from rfx.model.conductors import SheetConductors
from rfx.preflight._common import PreflightWarning
from rfx.preflight._subjects import finding_subjects
from rfx.preflight.realization import _shift_back_np


def _conductor_nodes(root, edges, axis):
    """PEC cells or incident tangential PEC edges, plus other f0 planes."""
    nodes = np.zeros(root.grid.shape, dtype=bool)
    if root.pec_mask is not None:
        nodes |= np.asarray(root.pec_mask, dtype=bool)
    for component in range(3):
        if component != axis:
            nodes |= edges[component] | _shift_back_np(
                edges[component], component, root.periodic[component])
    for spec in root.sheet_impedance:
        nodes |= np.asarray(spec.mask, dtype=bool)
    return nodes


def _gap_distances(footprint, conductors, plane, axis, grid, nodes):
    """First conductor in each direction on the SAME footprint column.

    The starting sheet node is field space. Adjacent conductor planes are
    one cell away; no periodic wrap or absorber node is visited.
    """
    lo = getattr(grid, f"pad_{'xyz'[axis]}_lo")
    hi = grid.shape[axis] - getattr(grid, f"pad_{'xyz'[axis]}_hi") - 1
    if not lo <= plane <= hi:
        return None
    column = np.take(footprint, plane, axis=axis)
    distances = []
    for step, end in ((-1, lo - 1), (1, hi + 1)):
        first = np.full(column.shape, -1, dtype=int)
        for k in range(plane + step, end, step):
            hit = column & (first < 0) & np.take(conductors, k, axis=axis)
            first[hit] = k
        distances.append(first)
    both = column & (distances[0] >= 0) & (distances[1] >= 0)
    if not np.any(both):
        return None
    index = tuple(np.argwhere(both)[0])
    return tuple(abs(float(nodes[first[index]] - nodes[plane])) for first in distances)


def _validate_cfg_f0_sheet_backed_gap(self, _w):
    """Report a film's missing wall loss using the production realization."""
    sheets = [(i, tc) for i, tc in enumerate(self._thin_conductors)
              if tc.surface_impedance_f0 is not None]
    if not sheets:
        return
    ctx = self._campaign_ctx()
    if ctx.error is not None:
        return  # Includes traced meshes: no concrete footprint or distances.
    root = ctx.realized()
    if root is None:
        return  # The assembly check reports the refusal.
    faces = wall_edge_masks(root.grid.shape, resolve_wall_faces(root.grid, root.periodic)[0])
    edges = tuple(edge | face for edge, face in zip(root.edges, faces, strict=True))
    walls = {}
    for (index, tc), spec in zip(sheets, root.sheet_impedance, strict=True):
        axis = int(spec.normal_axis)
        footprint = np.asarray(spec.mask, dtype=bool)
        # Reuse the operator's edge rule, including PEC exclusion. Conductance
        # may be traced: only the concrete geometric masks are inspected here.
        with jax.ensure_compile_time_eval():
            selected = SheetConductors((spec,), root.periodic)
            bare = selected.sheet_context(None)
            live = selected.sheet_context(root.edges)
        owned = 0
        for a in "xyz":
            coincident = (np.asarray(getattr(bare, f"mask_e{a}")) &
                          ~np.asarray(getattr(live, f"mask_e{a}")))
            # A rim touching a perpendicular PEC wall has zero tangential E
            # in the continuum too. Read end rows before PEC removes them.
            boundary_rows = getattr(bare, f"end_e{a}")
            if boundary_rows is not None:
                coincident &= ~np.asarray(boundary_rows)
            owned += np.count_nonzero(coincident)
        if axis not in walls:
            walls[axis] = _conductor_nodes(root, edges, axis)
        # PEC-owned starting columns are coincident, not a field gap.
        field = footprint.copy()
        for dimension, name in enumerate("xyz"):
            for start, stop in ((0, getattr(root.grid, f"pad_{name}_lo")),
                                (root.grid.shape[dimension] - getattr(root.grid, f"pad_{name}_hi"),
                                 root.grid.shape[dimension])):
                section = [slice(None)] * 3
                section[dimension] = slice(start, stop)
                field[tuple(section)] = False
        for component in range(3):
            if component != axis:
                field &= ~(edges[component] | _shift_back_np(
                    edges[component], component, root.periodic[component]))
        cases = []
        planes = np.flatnonzero(np.any(footprint, axis=tuple(a for a in range(3) if a != axis)))
        for plane in planes:
            gap = _gap_distances(field, walls[axis], plane, axis, root.grid, ctx.nodes[axis])
            if gap is not None:
                cases.append(f"case G: conductors at -{'xyz'[axis]} distance {gap[0]:.9g} m "
                             f"and +{'xyz'[axis]} distance {gap[1]:.9g} m; conductor loss "
                             "of fields with E normal to the sheet is missing there")
                break
        if owned:
            cases.append(f"case C: {owned} PEC-owned tangential edges; the sheet's loss "
                         "is absent where PEC owns its edges")
        if cases:
            label = getattr(tc, "name", None) or f"thin_conductor[{index}]"
            _w.warn(PreflightWarning(
                f"f0 sheet {label}: {'; '.join(cases)}. For a lossy wall use "
                "add_wall_loss(faces=...) or add_wall_loss(shape=...).",
                code="f0_sheet_backed_gap", severity="warning",
                source="_validate_cfg_f0_sheet_backed_gap",
                **finding_subjects(self, objects=(tc,)),
            ), stacklevel=3)

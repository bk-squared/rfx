"""Admit a rectangular guide mode only on a realized conducting aperture."""

from rfx.errors import RfxRefusal
import jax
import numpy as np

from rfx.boundaries.depths import Kind, grid_face_depths
from rfx.model.operands import realized_operands


_DEFAULT = object()


def refuse_bare_waveguide_edges(sim, grid, *, lane, conformal_weights=_DEFAULT):
    """Read production PEC operands before dispatch hands off to one device.

    An aperture edge is a line of tangential electric edges on the source
    plane. A PEC domain face at that node plane, realized PEC edges, or a
    conformal wall must cover the entire line. A conformal edge with weight
    w < 1 has a PEC surface within half a cell of its midpoint, so
    the wall lies in the cell that holds the snapped edge. Dispatch judges
    the weights run() forms by default; run_uniform judges the ones it
    formed. Distance names the nearest parallel conductor plane covering
    that same span, not a conductor touching only its corner.
    """
    from rfx.model.conductors import realized_conductors

    with jax.ensure_compile_time_eval():
        root = realized_conductors(sim, grid, nonuniform=sim._uses_nonuniform_mesh)
        if conformal_weights is _DEFAULT:
            conformal_weights = default_conformal_weights(sim, grid, root)
        operands = realized_operands(grid, None, {
            "pec_mask": root.pec_cells, "pec_edge_masks": root.pec_edges,
            "conformal_weights": conformal_weights}, ())
        masks = {}
        for record in operands:
            if isinstance(record.array, jax.core.Tracer):
                raise RfxRefusal(
                    f"{lane}: waveguide aperture conductor check cannot judge traced "
                    f"PEC operand {record.name}; provide static PEC walls.")
            if record.name.startswith("pec_edge_masks."):
                edges = np.asarray(record.array, dtype=bool)
            elif record.name.startswith("conformal_weights."):
                edges = np.asarray(record.array) < 1.0
            else:
                continue
            masks[record.component] = masks.get(record.component, False) | edges
        _check_edges(sim, grid, masks, lane)


def default_conformal_weights(sim, grid, root):
    """Weights run() forms when ``conformal_pec`` follows Boundary(conformal=True)."""
    if sim._uses_nonuniform_mesh or not sim._boundary_spec.conformal_faces():
        return None
    pec_shapes = root.assembly[4]
    if not pec_shapes:
        return None
    from rfx.geometry.conformal import clamp_conformal_weights, compute_conformal_weights_sdf
    return clamp_conformal_weights(*compute_conformal_weights_sdf(grid, pec_shapes), 0.1)


def _check_edges(sim, grid, masks, lane):
    from rfx.geometry.rasterize_grid import _axis_node_positions
    from rfx.model.graded_grid import position_to_index
    from rfx.runners.nonuniform import _range_to_slice_nu

    # Uniform grids retain the face record; graded grids thread its canonical
    # declaration separately from their allocated pads, as their setup does.
    faces = {face.name: face for face in grid_face_depths(grid, spec=sim._boundary_spec)}
    nodes = {axis: np.asarray(_axis_node_positions(
        grid.cells(axis), getattr(grid, f"pad_{axis}_lo"))) for axis in "xyz"}
    for number, entry in enumerate(sim._waveguide_ports):
        normal = "xyz".index(entry.direction[1])
        transverse = [a for a in range(3) if a != normal]
        point = [0., 0., 0.]
        point[normal] = entry.x_position
        source = (position_to_index(grid, tuple(point)) if sim._uses_nonuniform_mesh
                  else grid.position_to_index(tuple(point)))[normal]
        spans = {}
        for a in transverse:
            axis = "xyz"[a]
            requested = getattr(entry, f"{axis}_range")
            if sim._uses_nonuniform_mesh:
                spans[a], _ = _range_to_slice_nu(grid, axis, requested)
            else:
                spans[a], _ = sim._range_to_slice(
                    requested, sim._domain[a], float(grid.cells(axis)[0]), grid.shape[a],
                    getattr(grid, f"pad_{axis}_lo"), getattr(grid, f"pad_{axis}_hi"))
        bare = []
        for a in transverse:
            axis = "xyz"[a]
            tangent = next(b for b in transverse if b != a)
            cut = [slice(None)] * 3
            cut[normal] = source
            cut[tangent] = slice(spans[tangent][0], spans[tangent][1] - 1)
            # The remaining two dimensions retain their original order.
            mask = masks.get("xyz"[tangent])
            conductor = np.zeros(grid.shape[a], dtype=bool)
            if mask is not None:
                line = mask[tuple(cut)]
                conductor = np.all(line, axis=0 if tangent < a else 1)
            for side, index in (("lo", 0), ("hi", grid.shape[a] - 1)):
                face = faces[f"{axis}_{side}"]
                if face.kind == Kind.PEC and getattr(grid, f"pad_{axis}_{side}") == 0:
                    conductor[index] = True
            for side, plane in (("lo", spans[a][0]), ("hi", spans[a][1] - 1)):
                if conductor[plane]:
                    continue
                locations = nodes[axis][conductor]
                distance = (f"{np.min(np.abs(locations - nodes[axis][plane])) * 1e3:.9g} mm"
                            if locations.size else "none (no conductor covers this span)")
                bare.append(f"{axis}_{side} node plane {nodes[axis][plane] * 1e3:.9g} mm, "
                            f"nearest conductor distance {distance}")
        if bare:
            raise ValueError(
                f"{lane}: Waveguide port {entry.name or number!r} has bare aperture edges: "
                + "; ".join(bare)
                + ". Draw a PEC wall on each aperture edge, or make the domain cross-section "
                "equal to the aperture and declare its transverse faces PEC. The TE mode "
                "requires conductors at its aperture edges on both meshes.")

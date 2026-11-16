"""Lumped and wire port load stamping into the material arrays.

The LumpedPort and WirePort declarations and their per-step drive stay in rfx.sources.sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rfx._internal._grid_metric import field_index

import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import lumped_axis, lumped_components

if TYPE_CHECKING:
    from rfx.sources.sources import LumpedPort


_PORT_AXIS = {"ex": 0, "ey": 1, "ez": 2}


def _nu_cell_arrays(grid):
    """Per-cell (dx, dy, dz) arrays for a ``NonUniformGrid``, else ``None``.

    Read as float64 — the same dtype the already-corrected termination stamp
    in ``rfx/runners/nonuniform.py`` uses — so the sigma a port stamps and the
    metrics an extractor weights its Ampere legs by are one set of numbers.

    A TRACED mesh (mesh-as-design-variable) raises. That is INTENTIONAL and it
    is a deliberate behaviour change — see the ``NotImplementedError`` below.
    """
    dx_arr = getattr(grid, "dx_arr", None)
    if dx_arr is None:
        return None
    import numpy as _np
    import jax
    try:
        return (
            _np.asarray(dx_arr, dtype=_np.float64),
            _np.asarray(grid.dy_arr, dtype=_np.float64),
            _np.asarray(grid.dz, dtype=_np.float64),
        )
    except jax.errors.TracerArrayConversionError as exc:
        # The port metric needs concrete per-cell spacings. Falling back to SCALAR
        # ``grid.dx`` / ``grid.dy`` for the transverse axes would size the port from
        # the BOUNDARY cell instead of the node's cell. Those scalars are plain
        # floats even when the profile is traced: a traced ``dx_profile`` with
        # concrete ``dz`` would run and return a gradient of exactly 0.0 (measured
        # ``jax.grad`` and grad-norm). An optimiser would read that silently zero
        # gradient as an insensitive design variable, so refuse the traced metric
        # instead of falling back to scalars.
        raise NotImplementedError(
            "port metrics (port_sigma / "
            "port_dual_transverse) need CONCRETE per-cell mesh spacings: the "
            "realized conductance of a stamped sigma is fixed by the E node's "
            "discrete Ampere control volume, which is built from the cell "
            "sizes AT the port node. This grid carries a TRACED profile "
            "(mesh-as-design-variable), which is not supported on this path. "
            "Differentiate with respect to the port/material values on a "
            "concrete mesh instead.\n\n"
            "Before this check, the path did not raise: it silently used the "
            "BOUNDARY cell size for the two transverse axes and returned a "
            "gradient of exactly 0.0 w.r.t. the mesh. Raising is intentional."
        ) from exc


def _axis_cell_sizes(grid, position_ijk: tuple[int, int, int]
                     ) -> tuple[float, float, float]:
    """PRIMAL (dx, dy, dz) cell sizes AT ``position_ijk``.

    Uniform ``Grid`` is cubic and carries a single scalar, so all three are
    ``grid.dx``.

    ``NonUniformGrid`` carries per-cell ``dx_arr`` / ``dy_arr`` / ``dz``.
    This used to read the SCALAR ``grid.dx`` / ``grid.dy`` for the x and y
    sizes, which on that grid are the BOUNDARY cell sizes (see the
    ``NonUniformGrid`` field comments) — the wrong cell everywhere the mesh
    is graded, not merely at a grading step. Measured on a
    0.5/1.0 mm doubly-graded fixture, a nominal 50 ohm parallel-R RLC element
    realized 12.5 ohm at a LOCALLY UNIFORM node in the coarse region.
    """
    arrs = _nu_cell_arrays(grid)
    if arrs is None:
        dx = float(grid.dx)
        dy = float(getattr(grid, "dy", dx))
        dz_arr = getattr(grid, "dz", None)
        dz = float(dz_arr[position_ijk[2]]) if dz_arr is not None else dx
        return dx, dy, dz
    dx_arr, dy_arr, dz_arr = arrs
    return (float(dx_arr[position_ijk[0]]),
            float(dy_arr[position_ijk[1]]),
            float(dz_arr[position_ijk[2]]))


def _axis_dual_sizes(grid, position_ijk: tuple[int, int, int]
                     ) -> tuple[float, float, float]:
    """E-node DUAL spacings ``(dual_x, dual_y, dual_z)`` at ``position_ijk``.

    ``dual[k] = (d[k-1] + d[k]) / 2`` (``dual[0] = d[0]``) — the control
    volume the non-uniform E update actually divides its curl by. Delegates
    to :func:`rfx.nonuniform.e_node_dual_spacing_at` so there is one spelling
    of that rule. On a uniform mesh dual == primal bit-exactly, so a uniform
    ``Grid`` returns the same three numbers :func:`_axis_cell_sizes` does.
    """
    arrs = _nu_cell_arrays(grid)
    if arrs is None:
        return _axis_cell_sizes(grid, position_ijk)
    from rfx.model.graded_grid import e_node_dual_spacing_at
    return tuple(
        float(e_node_dual_spacing_at(arrs[ax], int(position_ijk[ax])))
        for ax in range(3)
    )


def port_dual_transverse(grid, position_ijk: tuple[int, int, int],
                         component: str) -> tuple[float, float]:
    """The two DUAL spacings transverse to ``component`` at the node.

    They are the sides of the dual face the node's conduction / displacement
    current pierces, i.e. the area factor in the discrete Ampere law the E
    update integrates. Anything folded into ``sigma`` or ``eps_r`` at an E
    node realizes its lumped value through THIS area — never through the
    primal cell widths (the measured current uses the same metric).
    """
    duals = _axis_dual_sizes(grid, position_ijk)
    axis = _PORT_AXIS[component]
    return tuple(duals[i] for i in range(3) if i != axis)


def port_sigma(grid, position_ijk: tuple[int, int, int],
               component: str, impedance: float) -> float:
    """Compute lumped port equivalent conductivity for anisotropic cells.

    σ = d_parallel_primal / (Z0 · dual_perp1 · dual_perp2)

    ``d_parallel`` is PRIMAL — it is the length of the E edge the port
    voltage ``V = E·d_parallel`` is taken over. The two transverse factors
    are DUAL: the realized conductance of a stamped σ is
    ``G = σ·dual_b·dual_c/d_parallel``, fixed by the E node's discrete
    Ampere / Joule control volume. This ensures
    P_dissipated = V²/Z0 regardless of cell aspect ratio; for cubic cells
    (dx=dy=dz=d) it reduces to 1/(Z0·d).
    """
    axis = _PORT_AXIS[component]
    d_par = _axis_cell_sizes(grid, position_ijk)[axis]
    d_perp = port_dual_transverse(grid, position_ijk, component)
    return d_par / (impedance * d_perp[0] * d_perp[1])


def _stamp_lumped(materials, cell, value, component, total_name, record_name):
    i, j, k = (int(c) for c in cell)
    axis = lumped_axis(component)
    total = getattr(materials, total_name)
    parts = list(lumped_components(getattr(materials, record_name, None)))
    own = parts[axis]
    if own is None:
        own = jnp.zeros_like(total)
    parts[axis] = own.at[i, j, k].add(value)
    return materials._replace(**{
        total_name: total.at[i, j, k].add(value),
        record_name: tuple(parts)})


def stamp_lumped_sigma(materials, cell, value, component):
    """Add a lumped conductance on ONE E edge, recorded as edge-owned.

    ``component`` (``"ex"``, ``"ey"`` or ``"ez"``) is the edge the element
    sits on: the edge that starts at ``cell``'s node along that axis.

    The stamp goes into ``materials.sigma`` as before -- a reader that wants
    the total conductivity at the cell still finds it there -- and into
    component ``component`` of ``materials.sigma_lumped``, which the E update
    removes before averaging over the edge's four cells and adds back at this
    cell, to that component only. A port's load is a device across one edge,
    not a property of a cell volume: averaged, a 50 ohm wire port read
    |S11| = 0.20 against the closed-form 1/3, and a lumped port inside a CPML
    dielectric read a non-physical |S11| = 1.86. Added back to all
    three components, as the record did before it carried one, a
    50 ohm port on Ez was also a 50 ohm resistor on the Ex and Ey edges
    leaving its node.
    """
    return _stamp_lumped(materials, cell, value, component,
                         "sigma", "sigma_lumped")


def stamp_lumped_eps(materials, cell, value, component):
    """Add a lumped permittivity (a capacitor's fold) on ONE E edge.

    Same argument as :func:`stamp_lumped_sigma`, for the ``C`` fold of an RLC
    element: a capacitor sits across one edge, ``component``'s.
    """
    return _stamp_lumped(materials, cell, value, component,
                         "eps_r", "eps_r_lumped")


def port_d_parallel(grid, position_ijk: tuple[int, int, int],
                    component: str) -> float:
    """Return the port edge's PRIMAL length, retaining a traced profile.

    Only the parallel axis is needed. In particular, a traced transverse
    axis must not force a host read when building a voltage-port drive.
    Concrete metrics retain the stored-array value used historically.
    """
    axis = _PORT_AXIS[component]
    if getattr(grid, "dx_arr", None) is not None:
        from rfx.core.jax_utils import is_tracer

        widths = getattr(grid, ("dx_arr", "dy_arr", "dz")[axis])
        value = widths[position_ijk[axis]]
        return value if is_tracer(value) else float(value)
    return _axis_cell_sizes(grid, position_ijk)[axis]


def setup_lumped_port(grid: Grid, port: LumpedPort, materials) -> object:
    """Fold port impedance into material conductivity at the port cell.

    Uses the 3D formula σ = d_parallel / (Z0 · d_perp1 · d_perp2)
    which reduces to 1/(Z0·dx) for cubic cells.
    """
    idx = field_index(grid, port.position, port.component,
                      allow_in_absorber=port.allow_in_absorber)
    sp = port_sigma(grid, idx, port.component, port.impedance)
    port._drive_stamps.clear()
    port._drive_stamps[tuple(idx)] = (sp, 1 / port_d_parallel(grid, idx, port.component))
    return stamp_lumped_sigma(materials, idx, sp, port.component)


def _wire_port_cells(grid, port):
    """The E edges the wire port drives — the extent, HALF-OPEN in edges.

    The port's declared extent is the node interval ``[start, end)`` along
    its axis, and the edges it drives are the ones whose OWN location lies
    inside that interval. The port's component is staggered half a cell
    along the axis, so edge ``a`` sits between node ``a`` and node
    ``a + 1``: it lies inside ``[start, end)`` exactly for
    ``lo <= a <= hi - 1``, where ``lo``/``hi`` are the endpoints' node
    indices. An extent of n cells therefore drives n edges.

    This used to be endpoint-INCLUSIVE (``range(lo, hi + 1)``), which
    drove n + 1 edges — the last one spanning one cell ABOVE the declared
    end. On the THRU that surplus Ez edge sits above the
    microstrip trace: while the trace was drawn as a one-cell PEC Box the
    edge was shorted and ``_wire_port_live_cells`` reported the right
    ``n_live`` by accident, so the off-by-one was invisible. Under the
    lattice ownership contract the trace is a sheet, which does not short
    the edge above it, and the port drove one cell too many
    (``n_live`` 2 -> 3, i.e. ``Z0_cell = Z0 / n_live`` off by 3/2).

    An extent shorter than the cell it lands in snaps to a single node
    (``lo == hi``) and would give NO edge. That port still drives exactly
    one edge — the one its extent lies in — chosen on the side of the node
    the extent occupies; it is a sub-cell declaration, not an empty one.
    """
    import numpy as np
    from rfx.model.point_placement import admit_point
    admit_point(grid, port.start, allow_in_absorber=port.allow_in_absorber)
    admit_point(grid, port.end, allow_in_absorber=port.allow_in_absorber)
    s = np.array(port.start)
    e = np.array(port.end)
    diff = e - s
    nonzero = np.abs(diff) > 1e-10
    if np.sum(nonzero) != 1:
        raise ValueError("WirePort start and end must be axis-aligned")
    axis = int(np.argmax(nonzero))

    if port.radius is not None and port.component != ("ex", "ey", "ez")[axis]:
        raise ValueError("a radius wire port's component must follow its start/end axis")

    from rfx._internal._periodic import interval_indices
    idx_s, idx_e = interval_indices(grid, tuple(s), tuple(e))

    lo = min(idx_s[axis], idx_e[axis])
    hi = max(idx_s[axis], idx_e[axis])
    first, last = wire_port_edge_span(
        grid, axis, lo, hi, float(s[axis]), float(e[axis]))

    cells = []
    for a in range(first, last + 1):
        cell = list(idx_s)
        cell[axis] = a
        cells.append(tuple(cell))
    return cells


def wire_port_edge_span(grid, axis: int, lo: int, hi: int,
                        start_pos: float, end_pos: float) -> tuple[int, int]:
    """``(first, last)`` E-edge index of a wire port extent — ONE spelling.

    ``lo``/``hi`` are the extent endpoints' node indices (ordered). The
    span is HALF-OPEN in edges: ``lo .. hi - 1``. Both the uniform lane
    (:func:`_wire_port_cells`) and the non-uniform runner call this, so
    the same declaration cannot rasterize to a different number of driven
    edges on the two lanes.

    ``start_pos``/``end_pos`` are the declared endpoint coordinates on
    ``axis`` and are used only for the sub-cell case below.
    """
    if hi > lo:
        return lo, hi - 1
    # Sub-cell extent: both ends snapped to node ``lo``, so the half-open
    # span is empty. The port still drives exactly one edge — the one its
    # extent lies in — chosen on the side of the node the extent occupies.
    # A tie (an extent centred on the node) takes the upper edge, which is
    # what the half-open rule would give for a one-cell extent.
    mid = 0.5 * (float(start_pos) + float(end_pos))
    node_pos = _axis_node_position(grid, axis, lo)
    k = lo if mid >= node_pos else max(lo - 1, 0)
    return k, k


def _axis_node_position(grid, axis: int, index: int) -> float:
    """Physical position of node ``index`` on ``axis``, in DOMAIN metres.

    Read from the library's own node-line producers so this does not
    become a second spelling of where a node is (the class).
    """
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid)
    from rfx.model.graded_grid import NonUniformGrid
    import numpy as np

    if 'xyz'[axis] in getattr(grid, 'periodic_axes', '') and index == grid.shape[axis]:
        return float(grid.domain[axis])
    coords = (coords_from_nonuniform_grid(grid)
              if isinstance(grid, NonUniformGrid)
              else coords_from_uniform_grid(grid))
    line = np.asarray((coords.x, coords.y, coords.z)[axis], dtype=np.float64)
    return float(line[int(np.clip(index, 0, line.size - 1))])


def _wire_port_live_cells(grid, port, pec_edge_masks=None):
    """Split the wire cells into (cells, live_flags, n_live).

    A cell is *live* when the port's OWN E edge at that index is not a
    realized PEC edge (``edge_is_pec(edge_masks, component, i,
    j, k)``, read BEFORE the runner's port-cell clearing).  A dead cell —
    one whose edge is shorted by a conductor — carries essentially no port
    current (measured on the thru: |I_dead|/|I_mid| =
    0.003-0.03), so its per-cell port resistor is bypassed and it must not
    be counted in the series impedance distribution or the
    drive/normalization cell counts.  Reading the component's own edge, not
    a cell mask, is what makes a sheet trace (which owns no cell) visible
    here.

    ``pec_edge_masks=None`` treats every cell as live, so n_live == n.

    The masks must be concrete (host-readable): the live split is a static
    geometry decision, made once at setup time, mirroring the
    reference-plane spec builder's concreteness requirement.

    Raises
    ------
    ValueError
        When every extent cell is inside PEC (n_live == 0): such a port
        has no live cell to terminate or drive.
    """
    from rfx.boundaries.pec import edges_are_pec

    cells = _wire_port_cells(grid, port)
    if pec_edge_masks is None:
        return cells, [True] * len(cells), max(len(cells), 1)

    # One host transfer for the whole extent (``edge_is_pec`` pulls the
    # full component mask per call, and the eager S-param loops read this
    # three times per port per step).
    live_flags = [not d for d in
                  edges_are_pec(pec_edge_masks, port.component, cells)]
    n_live = sum(live_flags)
    if n_live == 0:
        raise ValueError(
            f"WirePort {port.start} -> {port.end} ({port.component}): all "
            f"{len(cells)} extent cells have their {port.component} edge "
            "shorted by realized PEC, so the "
            "port has no live cell to terminate or drive. "
            "Shorten the extent or move the port so at least one of its "
            "rasterized cells is not PEC (per the assembled geometry -- "
            "not a cell-center guess)."
        )
    return cells, live_flags, n_live


def setup_wire_port(grid, port, materials, pec_edge_masks=None):
    """Distribute port impedance across the LIVE wire cells.

    For n_live live cells in series with total impedance Z0:
      σ_cell = n_live · d_parallel / (Z0 · d_perp1 · d_perp2)
    On uniform grids this reduces to n_live / (Z0 · dx).

    Cells whose extent lies inside PEC (dead cells) get NO port sigma:
    their resistor is bypassed by the surrounding conductor, so counting
    them made the physical series termination Z0·(n_live/n) instead of Z0
    (measured 33.3 ohm on the thru, n=3 with one
    dead cell).  With ``pec_edge_masks=None`` — or no dead cells — this is
    bit-identical to the historical all-cells formula.
    """
    cells, live_flags, n_live = _wire_port_live_cells(grid, port, pec_edge_masks)
    port._drive_stamps.clear()

    if port.radius is not None:
        from rfx.sources.wire_radius import stamp_wire_radius
        materials = stamp_wire_radius(
            grid, materials, port.component, port.radius,
            [c for c, live in zip(cells, live_flags) if live])
    for cell, live in zip(cells, live_flags):
        if not live:
            continue
        sp = port_sigma(grid, cell, port.component, port.impedance) * n_live
        port._drive_stamps[tuple(cell)] = (
            sp, 1 / (n_live * port_d_parallel(grid, cell, port.component)))
        materials = stamp_lumped_sigma(materials, cell, sp, port.component)
    return materials

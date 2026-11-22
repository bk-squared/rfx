"""Local self-field model for a declared thin-probe radius.

For a square transverse Yee lattice, its static Green function gives the
filament radius c*d, c = exp(-EulerGamma)/sqrt(8). Four equal circulating
H edges store inductance per length L' = mu/4. Replacing their permeability by

    mu * (1 + (2/pi)*log(c*d/a))

therefore adds exactly (mu/2pi)*log(c*d/a) of inductance per length.
This is an energy correction to the COMPLETE curl on each affected H edge,
not a rescaling of just its wire-E term (which would break reciprocity).

The port observes the current through its dual-face Ampere contour. The
electric storage inside that contour is on the driven edge, on the source
side of this current measurement. Represent its exterior contribution by
equal quarter-area electric storage on the four neighbouring parallel E
edges. This nearest-neighbour quadrature is a low-frequency approximation;
the parallel-plate Hankel oracle, not a claimed exact contour integration,
sets its tested accuracy. No S-parameter postprocessing is changed.

The magnetic increments have a shared per-H-component material owner; the
electric increments use the existing per-E-component lumped record.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.errors import RfxUnsupported
from rfx.core.yee import lumped_components
from rfx.model.materials import e_update_material_at


LATTICE_RADIUS = math.exp(-np.euler_gamma) / math.sqrt(8.0)
# Long-record sweeps bracket a loss of stability near 0.29 at the default step.
# At the conservative bound below mu_eff/mu = 0.9952 > 0.99**2: it also
# retains the unmodified grid's 0.99 vacuum-CFL margin (E stamps are positive).
# No negative magnetic energy / negative series inductor is admitted.
MAX_RADIUS_RATIO = 0.2


def validate_radius(radius):
    if radius is None:
        return
    if is_tracer(radius):
        raise RfxUnsupported("wire-port radius must be a static declaration")
    if not np.isscalar(radius) or not math.isfinite(float(radius)) or radius <= 0:
        raise ValueError("wire-port radius must be a finite positive length in metres")


RADIUS_SUPPORTED_PATHS = frozenset({
    "run_uniform", "run_nonuniform", "fwd_uniform", "fwd_nonuniform",
    "s_matrix_scan", "mixed_s_matrix", "topology_optimize",
    "run_distributed", "run_distributed_nu", "fwd_distributed_nu",
})


def require_radius_support(sim, lane):
    """Admission for declarations; never inspect a traced material array."""
    if not any(getattr(p, "radius", None) is not None for p in sim._ports):
        return
    reason = None
    if lane not in RADIUS_SUPPORTED_PATHS:
        reason = lane
    elif sim._solver != "yee" or sim._mode != "3d" or sim._stencil_order != 2:
        reason = "only the 3-D second-order Yee solver carries this radius model"
    elif sim._boundary == "upml":
        reason = "UPML does not read the wire H-component material record"
    elif sim._refinement is not None:
        reason = "subgridding does not carry the radius stamps"
    if reason is not None:
        raise NotImplementedError(
            f"wire-port radius is not implemented for {reason}. "
            "Use a supported 3-D Yee run()/forward(), or "
            "resolve the pin geometrically (for example a coax feed or a volume wire).")


def require_radius_pole_clearance(sim, grid, component, cells, *, port):
    """Refuse declared pole cells in the live radius edges' cell support."""
    entries = [entry for entry in sim._geometry
               if (sim._resolve_material(entry.material_name).debye_poles
                   or sim._resolve_material(entry.material_name).lorentz_poles)]
    if not entries:
        return
    from rfx.geometry.smoothing import _shape_parameters
    from rfx.geometry.rasterize_grid import (
        _material_cell_mask, coords_from_uniform_grid, coords_from_nonuniform_grid,
        centres_from_uniform_grid, centres_from_nonuniform_grid,
        cell_sizes_from_uniform_grid, cell_sizes_from_nonuniform_grid,
        volume_domain_faces,
    )
    from rfx.model.graded_grid import host_nominal_grid

    axis = ("ex", "ey", "ez").index(component)
    transverse = [a for a in range(3) if a != axis]
    edges = set(map(tuple, cells))
    for cell in cells:
        for a in transverse:
            for offset in (-1, 1):
                edge = list(cell)
                edge[a] += offset
                edges.add(tuple(edge))
    support = set()
    for edge in edges:
        for u in (-1, 0):
            for v in (-1, 0):
                cell = list(edge)
                cell[transverse[0]] += u
                cell[transverse[1]] += v
                support.add(tuple(cell))
    support = sorted(c for c in support if all(0 <= i < n for i, n in zip(c, grid.shape)))
    with jax.ensure_compile_time_eval():
        graded = hasattr(grid, "dx_arr")
        nominal = host_nominal_grid(grid) if graded else grid
        coords = (coords_from_nonuniform_grid(nominal) if graded
                  else coords_from_uniform_grid(nominal))
        centres = (centres_from_nonuniform_grid(nominal, coords) if graded
                   else centres_from_uniform_grid(nominal))
        sizes = (cell_sizes_from_nonuniform_grid(nominal) if graded
                 else cell_sizes_from_uniform_grid(nominal))
        for entry in entries:
            if any(is_tracer(value) for value in _shape_parameters(entry.shape)):
                raise NotImplementedError(
                    f"wire-port radius at {port} cannot check traced Debye/Lorentz "
                    f"geometry {entry.material_name!r}; declare concrete pole geometry")
            mask = _material_cell_mask(
                entry.shape, coords, centres, grid=nominal if not graded or nominal.periodic_axes else None,
                cell_sizes=sizes, domain_faces=volume_domain_faces(nominal),
                name=entry.material_name, refuse_empty=False)
            if is_tracer(mask):
                raise NotImplementedError(
                    f"wire-port radius at {port} cannot check traced Debye/Lorentz "
                    f"geometry {entry.material_name!r}; declare concrete pole geometry")
            declared = np.asarray(mask)
            hit = next((cell for cell in support if declared[cell]), None)
            if hit is not None:
                raise NotImplementedError(
                    f"wire-port radius at {port}: Debye/Lorentz material "
                    f"{entry.material_name!r} reaches radius stencil cell {hit}; "
                    "the electric stamp is ε∞/4, while the storage it represents "
                    "scales with the medium's ε(ω); per-pole quarter stamps are "
                    "not implemented. Move the pole geometry outside the stencil "
                    "or resolve the pin geometrically.")


@jax.tree_util.register_pytree_node_class
class _RadiusContours(tuple):
    """Carry stamped E-edge indices and declaration clearance with H arrays."""

    def __new__(cls, parts, edges=(), cleared=False):
        result = super().__new__(cls, parts)
        result.edges = edges
        result.cleared = cleared
        return result

    def tree_flatten(self):
        return tuple(self), (self.edges, self.cleared)

    @classmethod
    def tree_unflatten(cls, metadata, parts):
        return cls(parts, *metadata)


def require_radius_update(materials, *, lane, unsupported=False, debye=None, lorentz=None):
    """Defense for low-level runners taking already-stamped materials."""
    if getattr(materials, "mu_r_wire", None) is not None and unsupported:
        raise NotImplementedError(
            f"wire-port radius material records are not implemented on {lane}; "
            "resolve the pin geometrically (for example a coax feed or a volume wire).")
    contours = getattr(materials, "mu_r_wire", None)
    if contours is None or (debye is None and lorentz is None):
        return
    terms = tuple(poles[0].beta if kind == "Debye" else poles[0].c
                  for kind, poles in (("Debye", debye), ("Lorentz", lorentz))
                  if poles is not None)
    if any(is_tracer(value) for value in jax.tree.leaves(terms)):
        if getattr(contours, "cleared", False):
            return
        raise NotImplementedError(
            "wire-port radius cannot check traced Debye/Lorentz coefficients; "
            "declare concrete pole geometry through Simulation")
    edges = getattr(contours, "edges", None)
    if edges is None:
        raise NotImplementedError(
            "wire-port radius with Debye/Lorentz coefficients needs its stamped "
            "edge indices; rebuild the port with setup_wire_port")
    for component, cell in edges:
        if any(np.any(np.asarray(term[component])[(slice(None),) + cell] != 0)
               for term in terms):
            raise NotImplementedError(
                f"wire-port radius: Debye/Lorentz pole reaches stamped "
                f"{('ex', 'ey', 'ez')[component]} edge {cell}; the electric stamp "
                "is ε∞/4, while the storage it represents scales with the "
                "medium's ε(ω); per-pole quarter stamps are not implemented")


def stamp_wire_radius(grid, materials, component, radius, cells, *, pole_clearance=False):
    """Stamp live probe edges once, before sources/coefficients are built.

    The transverse mesh must be locally square and uniform; spacing along
    the live port must also be uniform. Axial grading did not retain the
    plate oracle's 1% resistance bound in the measured profiles. Grading
    away from this stencil is allowed, including declared traced ranges
    separated from it by a one-cell margin. The stamp reads only widths
    and indices, and the pin index is resolved on the nominal profile:
    a moving range elsewhere may shift node positions without
    changing the stamp.
    Because the stamp uses nominal widths, a whole-profile gradient has no
    stamp term for the stencil's own cells; those cells are declared fixed
    (the run stops if they move), so their gradient entries are not meaningful.
    Material differentiation stays in JAX throughout the stamping itself.
    """
    validate_radius(radius)
    from rfx.sources.sources import stamp_lumped_eps

    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    transverse = [t for t in range(3) if t != axis]
    contours = materials.mu_r_wire
    edges = set(getattr(contours, "edges", ()))
    cleared = pole_clearance and (contours is None or getattr(contours, "cleared", False))
    if hasattr(grid, "dx_arr"):
        from rfx.model.graded_ports import port_metric_axes
        spine = (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)
        if all(v is not None for v in spine):
            # A fixed mesh built inside jit has traced solver arrays but
            # concrete geometry spines. Validate geometry on those spines.
            widths = [np.asarray(v) for v in spine]
        else:
            metrics = port_metric_axes(grid)
            from rfx.model.graded_admission import nominal_cells, traced_cell_mask
            widths = []
            for a, (v, host) in enumerate(metrics):
                if host:
                    widths.append(np.asarray(v))
                    continue
                if spine[a] is not None:
                    widths.append(np.asarray(spine[a]))
                    continue
                w = nominal_cells(grid, a, padded=True)
                m = traced_cell_mask(grid, a, padded=True)
                if w is None or m is None:
                    raise NotImplementedError(
                        "wire-port radius does not support traced mesh metrics; "
                        "a nominal profile with traced_cells is needed")
                widths.append(np.asarray(w, dtype=np.float64))
                for cell in cells:
                    i = cell[a]
                    lo, hi = (i, i + 1) if a == axis else (i - 2, i + 2)
                    if np.any(m[max(0, lo - 1):hi + 1]):
                        raise NotImplementedError(
                            f"wire-port radius at port cell {cell}, axis {'xyz'[a]} "
                            f"reads stencil cells {lo}..{hi - 1} (padded); "
                            f"declared traced range {grid.traced_cells['xyz'[a]]} "
                            "(half-open interior) overlaps the stencil or its margin. "
                            "A pin inside or next to a moving range needs the "
                            "off-square Hankel bound and AD-vs-FD work, "
                            "which is not implemented.")
    else:
        widths = [grid.cells(t) for t in range(3)]

    # The measured model excludes absorber intersections. Keep every
    # modified H edge outside its pads, including along the port axis.
    pad_lo = getattr(grid, f"pad_{'xyz'[axis]}_lo", 0)
    pad_hi = getattr(grid, f"pad_{'xyz'[axis]}_hi", 0)
    duplicate = int('xyz'[axis] not in getattr(grid, "periodic_axes", ""))
    if any(c[axis] < pad_lo or c[axis] >= grid.shape[axis]-pad_hi-duplicate for c in cells):
        raise ValueError("wire-port radius must occupy live axial cells outside axial CPML")
    axial = np.asarray([widths[axis][c[axis]] for c in cells])
    if len(axial) and not np.allclose(axial, axial[0], rtol=2e-6, atol=0):
        raise NotImplementedError(
            "wire-port radius requires uniform spacing along the port; "
            "axial grading is not implemented for this model")

    h_parts = list(materials.mu_r_wire or (None, None, None))
    # Read the background before any of this port's E quadrature stamps.
    own_eps = lumped_components(materials.eps_r_lumped)[axis]
    eps = [e_update_material_at(materials, c, component,
            tuple(a in getattr(grid, "periodic_axes", "") for a in "xyz"), grid=grid)[0]
           - (0 if own_eps is None else own_eps[c]) for c in cells]
    for cell, eps_c in zip(cells, eps):
        edges.add((axis, tuple(cell)))
        local = []
        for t in transverse:
            i = cell[t]
            pad_lo = getattr(grid, f"pad_{'xyz'[t]}_lo", 0)
            pad_hi = getattr(grid, f"pad_{'xyz'[t]}_hi", 0)
            if i < pad_lo + 2 or i >= grid.shape[t] - pad_hi - 2:
                raise ValueError("wire-port radius stencil must be at least two cells from transverse boundaries/CPML")
            local.extend(widths[t][i-2:i+2])
        d = float(local[0])
        if not np.allclose(local, d, rtol=2e-6, atol=0):
            raise NotImplementedError("wire-port radius requires a locally uniform square transverse mesh")
        if radius / d > MAX_RADIUS_RATIO * (1 + 1e-7):
            raise ValueError(
                f"wire-port radius/d = {radius/d:g} exceeds the supported "
                f"bound {MAX_RADIUS_RATIO:g}; resolve the pin geometrically "
                "(for example a coax feed or a volume wire).")
        delta = (2 / math.pi) * math.log(LATTICE_RADIUS * d / radius)
        for t in transverse:
            h_axis = 3 - axis - t
            if h_parts[h_axis] is None:
                h_parts[h_axis] = jnp.zeros_like(materials.mu_r)
            for offset in (-1, 0):
                h_cell = list(cell)
                h_cell[t] += offset
                h_cell = tuple(h_cell)
                h_parts[h_axis] = h_parts[h_axis].at[h_cell].add(
                    materials.mu_r[h_cell] * delta)
            for offset in (-1, 1):
                e_cell = list(cell)
                e_cell[t] += offset
                edges.add((axis, tuple(e_cell)))
                materials = stamp_lumped_eps(materials, tuple(e_cell), eps_c / 4, component)
    return materials._replace(mu_r_wire=_RadiusContours(h_parts, tuple(sorted(edges)), cleared))

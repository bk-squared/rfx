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

import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.core.yee import cell_component_e_materials, lumped_components


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
        raise NotImplementedError("wire-port radius must be a static declaration")
    if not np.isscalar(radius) or not math.isfinite(float(radius)) or radius <= 0:
        raise ValueError("wire-port radius must be a finite positive length in metres")


RADIUS_SUPPORTED_PATHS = frozenset({
    "run_uniform", "run_nonuniform", "fwd_uniform", "fwd_nonuniform",
    "s_matrix_scan", "mixed_s_matrix", "topology_optimize",
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
    else:
        for entry in sim._geometry:
            material = sim._resolve_material(entry.material_name)
            if material.debye_poles or material.lorentz_poles:
                reason = "Debye/Lorentz materials with a radius port are not implemented"
                break
    if reason is not None:
        raise NotImplementedError(
            f"wire-port radius is not implemented for {reason}. "
            "Use a single-device, nondispersive Yee run()/forward(), or "
            "resolve the pin geometrically (for example a coax feed or a volume wire).")


def require_radius_update(materials, *, lane, unsupported=False):
    """Defense for low-level runners taking already-stamped materials."""
    if getattr(materials, "mu_r_wire", None) is not None and unsupported:
        raise NotImplementedError(
            f"wire-port radius material records are not implemented on {lane}; "
            "resolve the pin geometrically (for example a coax feed or a volume wire).")


def stamp_wire_radius(grid, materials, component, radius, cells):
    """Stamp live probe edges once, before sources/coefficients are built.

    The transverse mesh must be locally square and uniform; spacing along
    the live port must also be uniform. Axial grading did not retain the
    plate oracle's 1% resistance bound in the measured profiles. Grading
    away from this stencil is allowed. Traced mesh metrics are refused:
    the local geometry/stability envelope cannot be checked there.
    Material differentiation stays in JAX throughout the stamping itself.
    """
    validate_radius(radius)
    from rfx.sources.sources import stamp_lumped_eps

    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    transverse = [t for t in range(3) if t != axis]
    if hasattr(grid, "dx_arr"):
        from rfx.nonuniform import port_metric_axes
        spine = (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)
        if all(v is not None for v in spine):
            # A fixed mesh built inside jit has traced solver arrays but
            # concrete geometry spines. Validate geometry on those spines.
            widths = [np.asarray(v) for v in spine]
        else:
            metrics = port_metric_axes(grid)
            if not all(host for _, host in metrics):
                raise NotImplementedError("wire-port radius does not support traced mesh metrics")
            widths = [np.asarray(v) for v, _ in metrics]
    else:
        widths = [grid.cells(t) for t in range(3)]

    # The measured model excludes absorber intersections. Keep every
    # modified H edge outside its pads, including along the port axis.
    pad_lo = getattr(grid, f"pad_{'xyz'[axis]}_lo", 0)
    pad_hi = getattr(grid, f"pad_{'xyz'[axis]}_hi", 0)
    if any(c[axis] < pad_lo or c[axis] >= grid.shape[axis]-pad_hi-1 for c in cells):
        raise ValueError("wire-port radius must occupy live axial cells outside axial CPML")
    axial = np.asarray([widths[axis][c[axis]] for c in cells])
    if len(axial) and not np.allclose(axial, axial[0], rtol=2e-6, atol=0):
        raise NotImplementedError(
            "wire-port radius requires uniform spacing along the port; "
            "axial grading is not implemented for this model")

    h_parts = list(materials.mu_r_wire or (None, None, None))
    # Read the background before any of this port's E quadrature stamps.
    own_eps = lumped_components(materials.eps_r_lumped)[axis]
    eps = [cell_component_e_materials(materials, c, component)[0]
           - (0 if own_eps is None else own_eps[c]) for c in cells]
    for cell, eps_c in zip(cells, eps):
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
                materials = stamp_lumped_eps(materials, tuple(e_cell), eps_c / 4, component)
    return materials._replace(mu_r_wire=tuple(h_parts))

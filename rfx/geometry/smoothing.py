"""Kottke tensor-averaged subpixel smoothing at dielectric interfaces.

Implements the Kottke–Farjadpour–Johnson scheme (PRE 77, 036612, 2008):
at every Yee voxel that straddles a dielectric boundary the effective
inverse-permittivity *tensor* is

    (ε_eff)⁻¹ = Pₙ [f/ε₁ + (1−f)/ε₂]  +  Pₜ / [f·ε₁ + (1−f)·ε₂]

where  Pₙ = n̂·n̂ᵀ  is the projection onto the interface normal and
Pₜ = I − Pₙ  is the tangential projection.  The diagonal elements of
ε_eff are then extracted for each E-field component on the Yee grid.

Three design improvements over the original rfx linear-SDF scheme:

1. **Same-material union corner fix** — voxels fully inside the union
   of same-material shapes get bulk ε directly; no gradient needed.

2. **Analytic normals per shape** — Box (nearest face), Sphere
   (radial), Cylinder (radial + axial); eliminates the central-
   difference artefact at SDF union seams.

3. **Full Kottke tensor** — replaces the scalar cos²/sin² mixing
   formula with the proper 3×3 projection, giving second-order
   convergence for arbitrarily oriented interfaces.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.geometry.csg import Shape


# ---------------------------------------------------------------------------
# Signed distance functions for supported shapes
# ---------------------------------------------------------------------------
#
# Every SDF and normal below takes an array backend ``xp`` -- ``jax.numpy``
# (the default, and the only choice for traced inputs) or ``numpy``. The
# smoothing functions evaluate them with ``xp=numpy`` on host float64 when
# every input is concrete (#833): the same formulas, only the arithmetic
# precision differs, and the result is cast to the active JAX dtype once at
# the end. ``rfx.geometry.conformal`` calls them with the default.

def _sdf_sphere(x, y, z, shape, xp=jnp):
    """Signed distance: negative inside, positive outside."""
    cx, cy, cz = shape.center
    r = xp.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
    return r - shape.radius


def _sdf_box(x, y, z, shape, xp=jnp):
    """Signed distance for axis-aligned box."""
    lo = xp.array(shape.corner_lo)
    hi = xp.array(shape.corner_hi)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0

    dx = xp.abs(x - center[0]) - half[0]
    dy = xp.abs(y - center[1]) - half[1]
    dz = xp.abs(z - center[2]) - half[2]

    ox = xp.maximum(dx, 0.0)
    oy = xp.maximum(dy, 0.0)
    oz = xp.maximum(dz, 0.0)
    outside = xp.sqrt(ox**2 + oy**2 + oz**2)
    inside = xp.minimum(xp.maximum(xp.maximum(dx, dy), dz), 0.0)
    return outside + inside


def _sdf_cylinder(x, y, z, shape, xp=jnp):
    """Signed distance for a cylinder along a given axis."""
    cx, cy, cz = shape.center

    if shape.axis == "z":
        r = xp.sqrt((x - cx)**2 + (y - cy)**2) - shape.radius
        h = xp.abs(z - cz) - shape.height / 2.0
    elif shape.axis == "y":
        r = xp.sqrt((x - cx)**2 + (z - cz)**2) - shape.radius
        h = xp.abs(y - cy) - shape.height / 2.0
    else:
        r = xp.sqrt((y - cy)**2 + (z - cz)**2) - shape.radius
        h = xp.abs(x - cx) - shape.height / 2.0

    outside = xp.sqrt(xp.maximum(r, 0.0)**2 + xp.maximum(h, 0.0)**2)
    inside = xp.minimum(xp.maximum(r, h), 0.0)
    return outside + inside


def _get_sdf_fn(shape: Shape):
    """Return the SDF function for a known shape type, or None."""
    from rfx.geometry.csg import Box, Sphere, Cylinder
    if isinstance(shape, Sphere):
        return _sdf_sphere
    elif isinstance(shape, Box):
        return _sdf_box
    elif isinstance(shape, Cylinder):
        return _sdf_cylinder
    return None


# ---------------------------------------------------------------------------
# Analytic normals per shape (outward-pointing)
# ---------------------------------------------------------------------------

def _normal_sphere(x, y, z, shape, xp=jnp):
    """Analytic outward normal for a sphere: radial direction."""
    cx, cy, cz = shape.center
    dx = x - cx
    dy = y - cy
    dz = z - cz
    r = xp.sqrt(dx**2 + dy**2 + dz**2 + 1e-30)
    return dx / r, dy / r, dz / r


def _normal_box(x, y, z, shape, xp=jnp):
    """Analytic outward normal for an axis-aligned box.

    At each point the normal is determined by which face is nearest
    (the axis with the smallest penetration distance).
    """
    lo = xp.array(shape.corner_lo)
    hi = xp.array(shape.corner_hi)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0

    # Signed distance to each face pair (positive = outside that pair)
    rx = xp.abs(x - center[0]) - half[0]
    ry = xp.abs(y - center[1]) - half[1]
    rz = xp.abs(z - center[2]) - half[2]

    # The nearest face corresponds to the axis with the largest
    # (least-negative inside, or least-positive outside) component.

    # For interior points: nearest face = axis with *max* signed dist
    # For exterior points: same logic (largest component dominates SDF)

    # Use soft selection: pick the axis closest to the surface.
    # For box interior, max_r < 0 and we want the axis where r is
    # closest to zero (largest / least negative).
    is_x = (rx >= ry) & (rx >= rz)
    is_y = (ry > rx) & (ry >= rz)
    # is_z = everything else

    sign_x = xp.sign(x - center[0])
    sign_y = xp.sign(y - center[1])
    sign_z = xp.sign(z - center[2])

    # Default to z-face normal
    nx = xp.where(is_x, sign_x, 0.0)
    ny = xp.where(is_y, sign_y, xp.where(is_x, 0.0, 0.0))
    nz = xp.where(is_x | is_y, 0.0, sign_z)

    # Ensure unit length (should already be 1 for pure axis normals,
    # but guard against degenerate zero-sign cases)
    mag = xp.sqrt(nx**2 + ny**2 + nz**2 + 1e-30)
    return nx / mag, ny / mag, nz / mag


def _normal_cylinder(x, y, z, shape, xp=jnp):
    """Analytic outward normal for a cylinder.

    The radial component dominates on the curved surface; the axial
    component dominates on the caps.
    """
    cx, cy, cz = shape.center
    R = shape.radius
    H2 = shape.height / 2.0

    if shape.axis == "z":
        dx, dy = x - cx, y - cy
        r = xp.sqrt(dx**2 + dy**2 + 1e-30)
        dh = xp.abs(z - cz) - H2
        dr = r - R
        on_cap = dh > dr
        nx = xp.where(on_cap, 0.0, dx / r)
        ny = xp.where(on_cap, 0.0, dy / r)
        nz = xp.where(on_cap, xp.sign(z - cz), 0.0)
    elif shape.axis == "y":
        dx, dz = x - cx, z - cz
        r = xp.sqrt(dx**2 + dz**2 + 1e-30)
        dh = xp.abs(y - cy) - H2
        dr = r - R
        on_cap = dh > dr
        nx = xp.where(on_cap, 0.0, dx / r)
        ny = xp.where(on_cap, xp.sign(y - cy), 0.0)
        nz = xp.where(on_cap, 0.0, dz / r)
    else:  # axis == "x"
        dy, dz = y - cy, z - cz
        r = xp.sqrt(dy**2 + dz**2 + 1e-30)
        dh = xp.abs(x - cx) - H2
        dr = r - R
        on_cap = dh > dr
        nx = xp.where(on_cap, xp.sign(x - cx), 0.0)
        ny = xp.where(on_cap, 0.0, dy / r)
        nz = xp.where(on_cap, 0.0, dz / r)

    mag = xp.sqrt(nx**2 + ny**2 + nz**2 + 1e-30)
    return nx / mag, ny / mag, nz / mag


def _get_normal_fn(shape: Shape):
    """Return the analytic normal function for a known shape type."""
    from rfx.geometry.csg import Box, Sphere, Cylinder
    if isinstance(shape, Sphere):
        return _normal_sphere
    elif isinstance(shape, Box):
        return _normal_box
    elif isinstance(shape, Cylinder):
        return _normal_cylinder
    return None


# ---------------------------------------------------------------------------
# CPML pad continuation for the SMOOTHED lane (#1043 stage B)
# ---------------------------------------------------------------------------
#
# ``_assemble_materials`` continues a boundary-touching structure into the
# absorber pad by replicating the interior-edge slice of the eps_r/sigma/mu_r
# ARRAYS outward (``extend_cpml_pad_materials``), "as if the geometry continued
# beyond the domain". The smoothed lane rebuilds its update permittivity from
# ``sim._geometry`` and never saw that step, so under ``subpixel_smoothing`` a
# guide that touches the domain edge was solved with VACUUM in the pad and a
# Kottke half-cell at the seam -- a facet the guided mode reflects off (#831:
# ``|B/A|`` 0.53 at 20 CPML cells, and WORSE as the absorber deepens, because
# the first pad cell's conductivity falls as N^-3 and loads the facet less).
#
# The fix cannot reuse ``extend_cpml_pad_materials`` on the smoothing OUTPUT.
# Measured, with no FDTD, on cv01's committed build (guide centre row, x-lo
# seam, ``eps_r = 12``; driver
# ``scripts/diagnostics/cv03_seam_facet/pad_replication_shape.py``):
#
#   variant                                  pad      row across the seam
#   committed (no replication)                1.000    1.0,  1.0,  6.5, 12.0
#   naive extend_cpml_pad_materials on it     6.500    6.5,  6.5,  6.5, 12.0
#   sourced one column inward                12.000   12.0, 12.0,  6.5, 12.0
#   reference: _assemble_materials           12.000   12.0, 12.0, 12.0, 12.0
#
# The smoothed array's interior-edge column IS the Kottke half-cell, so
# replicating it fills the pad with a medium that is neither the guide nor
# vacuum; sourcing one column inward gets the pad right but strands that
# half-cell INSIDE the absorber -- the one-cell film #655 already had to repair
# on the staircase lane. Only continuing the GEOMETRY and smoothing THAT
# reproduces the reference row, which is what this helper does: the reached
# face is moved out past the array, so the smoothing sees no interface there at
# all and the pad and seam cells come out at bulk eps through the ordinary
# ``f >= 1`` branch.


#: A face counts as REACHING a padded boundary when the shape's bounding box
#: gets to the outermost interior node plane. The rule is exactly that -- "the
#: declared face reaches the boundary" -- and this constant is only the
#: numerical slack that makes the comparison decidable, in local cells: a
#: corner spelled ``a - n*dx`` can land an f64 ulp off the algebraically
#: identical ``m*dx`` (see ``Box``'s docstring on knife-edge corners), and one
#: ulp must not decide whether a structure is continued. 1e-6 cells is ~1e-13 m
#: at rfx's finer meshes -- eight orders above ulp noise and eight below any
#: geometry anyone draws on purpose.
#:
#: Deliberately NOT half a cell. A face drawn 0.3 cells inside the boundary is
#: 0.3 cells inside it, and resolving that is what subpixel smoothing is FOR;
#: continuing it would move the structure to the boundary and throw away the
#: resolution the lane exists to provide.
#:
#: The consequence, stated rather than left to be found: on the hi face the two
#: lanes disagree over a one-cell window. The staircase lane's rule is a
#: cell-centre test, so ``extend_cpml_pad_materials``' #627a fallback continues
#: any box with ``corner_hi > interior_hi - dx`` (the half-open convention drops
#: the last node, and the fallback promotes from one column inward); this lane
#: continues only ``corner_hi >= interior_hi``. No tolerance makes them agree
#: everywhere, because one rule is a cell-centre test and the other is sub-cell.
#: A structure meant to reach the boundary should be drawn to it, and then both
#: lanes continue it.
_PAD_REACH_TOL_CELLS = 1e-6

#: How far past the array the continued face is pushed, in local cells. The
#: outermost pad sample must sit at least half a cell inside the continued
#: shape for ``f = clip(0.5 - sdf/dx, 0, 1)`` to reach 1 and take the bulk
#: branch; two cells clears that for every Yee component offset.
_PAD_CONTINUE_CELLS = 2.0


class UnextendableShape(NamedTuple):
    """One (shape, axis, side) a pad continuation could not express.

    ``axis`` is 0/1/2 and ``side`` is ``"lo"`` / ``"hi"``. It carries the
    shape's declared ``eps_r`` so a caller can say what the pad will hold
    instead of what was drawn, without re-resolving the material.

    ``entry_index`` and ``material_name`` identify the geometry entry, and
    they are FIELDS rather than something a caller recovers by identity.
    Round-1 verification found why: ``extend_shapes_into_cpml_pad`` rewrites
    the shape as it continues each axis, so a shape continued on x and
    unextendable on y reports the CONTINUED object, and an ``id()`` lookup
    against ``sim._geometry`` misses it -- the advisory printed
    ``Material '?' (geometry entry #-1, Cylinder)``. ``smoothed_shape_pairs``
    fills both in from the per-entry loop, where the answer is not in doubt,
    and restores ``shape`` to the DECLARED object.

    Two callers surface these, and they are the SAME list from the SAME
    predicate, not two rules that happen to agree today (the #627
    duplication class): :func:`warn_unextendable_shapes` at run time, and
    preflight's ``_validate_cfg_dielectric_at_absorber_seam``, which calls
    :func:`smoothed_shape_pairs` itself rather than re-deriving "reaches a
    padded face" from the declared domain.
    """

    shape: object
    axis: int
    side: str
    reason: str
    eps_r: float = 1.0
    entry_index: int = -1
    material_name: str = "?"


def warn_unextendable_shapes(unextendable, *, stacklevel: int = 3) -> None:
    """Emit ONE run-time warning for shapes left carrying a seam facet.

    Preflight says this before the run when it can (it builds the grid
    itself); this says it from inside the runner, where the realized pads
    are no longer in question. A shape that reaches a padded face and is
    not continued is solved with vacuum in its own absorber -- the #831
    facet -- and nothing downstream distinguishes that from a structure
    the user meant to end there.
    """
    if not unextendable:
        return
    import warnings as _w
    faces = ", ".join(
        f"{type(u.shape).__name__} at {'xyz'[u.axis]}-{u.side} "
        f"(eps_r {u.eps_r:g}; {u.reason})"
        for u in unextendable)
    _w.warn(
        "subpixel smoothing: "
        f"{len(unextendable)} declared face(s) reach a CPML/UPML pad and "
        f"were NOT continued into it -- {faces}. Those pads are solved at "
        "eps_r = 1.0, so each structure is terminated by an end facet at the "
        "interior/pad seam and the guided/standing field sees a reflector "
        "there (issue #1043). Draw the structure as a Box or an "
        "axis-aligned Cylinder, or move it clear of the face.",
        stacklevel=stacklevel,
    )


def _axis_cells(nodes) -> tuple[float, float]:
    """Local cell width at the lo and the hi end of a 1-D node array."""
    n = len(nodes)
    if n < 2:
        return (0.0, 0.0)
    return (float(nodes[1] - nodes[0]), float(nodes[n - 1] - nodes[n - 2]))


def _continue_box(shape, axis: int, side: str, target: float):
    from rfx.geometry.csg import Box
    lo = list(shape.corner_lo)
    hi = list(shape.corner_hi)
    if side == "lo":
        lo[axis] = min(lo[axis], target)
    else:
        hi[axis] = max(hi[axis], target)
    return Box(tuple(lo), tuple(hi))


def _continue_cylinder(shape, axis: int, side: str, target: float):
    """Continue a cylinder along ITS OWN axis; ``None`` across it."""
    from rfx.geometry.csg import Cylinder
    if {"x": 0, "y": 1, "z": 2}[shape.axis] != axis:
        return None
    centre_a = float(shape.center[axis])
    half = float(shape.height) / 2.0
    lo, hi = centre_a - half, centre_a + half
    if side == "lo":
        lo = min(lo, target)
    else:
        hi = max(hi, target)
    centre = list(shape.center)
    centre[axis] = (lo + hi) / 2.0
    return Cylinder(tuple(centre), shape.radius, hi - lo, shape.axis)


def extend_shapes_into_cpml_pad(
    shapes: list[tuple[Shape, float]],
    node_coords,
    pads,
) -> tuple[list[tuple[Shape, float]], list["UnextendableShape"]]:
    """Continue boundary-touching shapes out through the absorber pads.

    The smoothed lane's counterpart to
    :func:`rfx.geometry.rasterize_grid.extend_cpml_pad_materials`, and
    deliberately a GEOMETRY transform rather than an array one -- see this
    section's header comment for the measured reason.

    Parameters
    ----------
    shapes : list of (shape, eps_r)
        The pairs the smoothing is about to be handed. Shapes whose material
        carries a dispersion pole must be filtered out by the caller: pole
        extension into a pad diverges (#627b) and promoting a pole-carrying
        column's statics moved a committed Debye recovery past its gate
        (#808).
    node_coords : (x, y, z)
        1-D E-NODE coordinate arrays over the FULL array, pads included --
        ``coords_from_uniform_grid`` / ``coords_from_nonuniform_grid``.
    pads : ((plx, phx), (ply, phy), (plz, phz))
        Allocated absorbing cells per face. A face with ``0`` is not a pad
        (reflector / periodic / no absorber) and is never continued, which is
        how per-face ``BoundarySpec`` allocation reaches this function without
        being re-derived here.

    Returns
    -------
    extended : list of (shape, eps_r)
        Same length and order as ``shapes``. An entry reaching no padded face
        is the SAME object, so a geometry that touches nothing makes this
        function an identity by construction -- that is what keeps every
        non-touching configuration bit-identical.
    unextendable : list of UnextendableShape
        Faces reached by a shape whose type has no continuation (a sphere's
        tangency, a cylinder reached across its axis, an imported mesh).
        Those shapes are returned unchanged and still carry the facet; the
        caller surfaces them.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.geometry.csg import Box, Cylinder

    out: list[tuple[Shape, float]] = []
    unextendable: list[UnextendableShape] = []
    for shape, eps_r in shapes:
        try:
            bbox_lo, bbox_hi = shape.bounding_box()
        except (NotImplementedError, AttributeError):
            out.append((shape, eps_r))
            continue
        current = shape
        for axis in range(3):
            nodes = node_coords[axis]
            # PER AXIS, not per run. A mesh-as-design-variable profile makes
            # ONE axis' node positions tracers (a traced dz leaves x and y
            # concrete), and skipping all three then left the concrete axes'
            # facets in place -- an optimizer would still be descending
            # against a reflector on x. The traced axis is skipped because
            # its reach test has no concrete answer, and forcing one would
            # make the pad depend on a value the tape differentiates through.
            if is_tracer(nodes):
                continue
            n = len(nodes)
            cell_lo, cell_hi = _axis_cells(nodes)
            for side, pad, cell in (("lo", int(pads[axis][0]), cell_lo),
                                    ("hi", int(pads[axis][1]), cell_hi)):
                if pad <= 0 or cell <= 0.0 or n < 2:
                    continue
                if side == "lo":
                    edge = float(nodes[pad])
                    reaches = (float(bbox_lo[axis])
                               <= edge + _PAD_REACH_TOL_CELLS * cell)
                    target = float(nodes[0]) - _PAD_CONTINUE_CELLS * cell
                else:
                    edge = float(nodes[n - 1 - pad])
                    reaches = (float(bbox_hi[axis])
                               >= edge - _PAD_REACH_TOL_CELLS * cell)
                    target = float(nodes[n - 1]) + _PAD_CONTINUE_CELLS * cell
                if not reaches:
                    continue
                if isinstance(current, Box):
                    current = _continue_box(current, axis, side, target)
                elif isinstance(current, Cylinder):
                    grown = _continue_cylinder(current, axis, side, target)
                    if grown is None:
                        unextendable.append(UnextendableShape(
                            current, axis, side,
                            "a cylinder has no continuation across its axis",
                            float(eps_r)))
                    else:
                        current = grown
                else:
                    unextendable.append(UnextendableShape(
                        current, axis, side,
                        f"{type(current).__name__} has no pad continuation",
                        float(eps_r)))
        out.append((current, eps_r))
    return out, unextendable


def smoothed_shape_pairs(sim, grid):
    """Build the (shape, eps_r) pairs the smoothed lane solves, pad included.

    ONE implementation for the three sites that rebuild the update
    permittivity from ``sim._geometry`` -- Stage-2 ``kottke_pec`` and Stage-1
    in ``rfx/runners/uniform.py`` and the NU mirror in
    ``rfx/runners/nonuniform.py``. #627 exists because the array-side
    replication was hand-duplicated across two assemblers and drifted; this
    one is not duplicated a second time.

    The continuation is applied when the run has absorbing pads at all
    (``sim._boundary`` in ``cpml``/``upml`` with ``_cpml_layers > 0``) --
    the same gate ``_assemble_materials`` uses for
    ``extend_cpml_pad_materials``, MINUS its ``include_cpml_pad_extension``
    keyword, which is a private ``_assemble_materials`` argument with one
    in-tree caller (``rfx/vmap_sweep.py``) and does not reach a runner. That
    caller passes ``False`` and runs no subpixel lane, so the two gates cannot
    disagree today; a future caller that wants the flag honoured here has to
    thread it -- and only to shapes whose material carries
    no dispersion pole (#627b, #808: a pole in a pad diverges, and a
    pole-carrying column's promoted statics are a material no declared model
    has).

    Returns ``(pairs, unextendable)``; ``unextendable`` is empty whenever
    nothing reaches a padded face.
    """
    pairs = [(entry.shape, sim._resolve_material(entry.material_name).eps_r)
             for entry in sim._geometry]
    if not pairs:
        return pairs, []
    if (getattr(sim, "_boundary", None) not in ("cpml", "upml")
            or int(getattr(sim, "_cpml_layers", 0)) <= 0):
        return pairs, []

    if hasattr(grid, "dx_arr"):
        from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
        coords = coords_from_nonuniform_grid(grid)
    else:
        from rfx.geometry.rasterize_grid import coords_from_uniform_grid
        coords = coords_from_uniform_grid(grid)
    node_coords = (coords.x, coords.y, coords.z)
    # A traced mesh (mesh-as-design-variable) makes node positions tracers on
    # the traced axis, and the reach test reads them as Python floats. That is
    # handled PER AXIS inside extend_shapes_into_cpml_pad rather than here:
    # skipping the whole run when any one axis was traced left the concrete
    # axes carrying their facets, so a run optimizing a dz profile still
    # descended against an x-face reflector. Box corners themselves are always
    # concrete (``Box._axis_mask`` relies on it: ``extent = float(hi - lo)``),
    # so only the grid side needs the guard at all.
    pads = ((grid.pad_x_lo, grid.pad_x_hi),
            (grid.pad_y_lo, grid.pad_y_hi),
            (grid.pad_z_lo, grid.pad_z_hi))

    # Per entry, in the DECLARED order: `compute_smoothed_eps` applies groups
    # in insertion order and later shapes overwrite earlier ones, so the list
    # is rebuilt position for position rather than partitioned and rejoined.
    out = []
    unextendable = []
    # 1e6, the value every other reader of this threshold defaults to
    # (rfx/surrogate.py, rfx/fidelity.py, rfx/pcb.py). An ``inf``
    # default fails OPEN: a sim without the attribute would classify a
    # PEC material as a dielectric and continue metal into the pad,
    # which is the one thing both lanes agree never to do.
    pec_sigma = float(getattr(sim, "_PEC_SIGMA_THRESHOLD", 1e6))
    for idx, (entry, (shape, eps_r)) in enumerate(zip(sim._geometry, pairs)):
        mat = sim._resolve_material(entry.material_name)
        # PEC volumes are not continued on EITHER lane: ``pec_mask`` is not in
        # ``extend_cpml_pad_materials``' signature, so the staircase lane ends
        # a PEC structure at the seam too, and the two lanes have to agree
        # about what stands in a pad.
        if float(getattr(mat, "sigma", 0.0)) >= pec_sigma:
            out.append((shape, eps_r))
            continue
        if (getattr(mat, "debye_poles", None)
                or getattr(mat, "lorentz_poles", None)):
            out.append((shape, eps_r))
            continue
        one, unext = extend_shapes_into_cpml_pad(
            [(shape, eps_r)], node_coords, pads)
        out.extend(one)
        # Stamp the entry's identity HERE, in the loop that knows it, and put
        # the DECLARED shape back. The builder reports whatever object it held
        # when the face was reached, which is already a continued copy once a
        # lower axis was rewritten -- so a caller matching on identity misses
        # exactly the multi-face cases it most needs to name.
        unextendable.extend(
            u._replace(shape=shape, entry_index=idx,
                       material_name=entry.material_name)
            for u in unext)
    return out, unextendable


# ---------------------------------------------------------------------------
# Coordinate helpers for Yee-offset positions
# ---------------------------------------------------------------------------

def _yee_coords(grid: Grid):
    """Return E-NODE coordinates (x, y, z) as 1-D arrays.

    On the Yee grid:
    - Ex lives at (i+0.5, j, k)
    - Ey lives at (i, j+0.5, k)
    - Ez lives at (i, j, k+0.5)

    We return the integer-grid positions (nodes); the caller ADDS +0.5*dx to
    reach a centre. That direction is the contract: the NU sibling
    ``compute_smoothed_eps_nonuniform`` subtracted instead, and when #562 made
    the NU coordinates node-based the subtraction put every smoothed voxel half
    a cell off (review F1). Derive centres FROM nodes, never the reverse.
    """
    # Keep the coordinate construction in the shared host-float64 node
    # builder.  Constructing ``jnp.arange`` here made the Kottke SDF sample
    # positions depend on ``jax_enable_x64`` (the subtraction and multiply
    # rounded in different precisions), even though the binary rasterizer
    # used the exact node spine.  The returned arrays are converted only
    # after the exact node line has been formed; this is a constant geometry
    # input, not a traced design variable.
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    coords = coords_from_uniform_grid(grid)
    return tuple(jnp.asarray(axis) for axis in (coords.x, coords.y, coords.z))


# ---------------------------------------------------------------------------
# Kottke tensor averaging
# ---------------------------------------------------------------------------

def _kottke_tensor_eps(
    f: jnp.ndarray,
    eps_inside: float,
    eps_outside: jnp.ndarray,
    nx: jnp.ndarray,
    ny: jnp.ndarray,
    nz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the Kottke tensor-averaged ε and return its diagonals.

    The effective inverse-permittivity tensor is:

        (ε_eff)⁻¹ = Pₙ · [f/ε₁ + (1−f)/ε₂]  +  Pₜ · 1/[f·ε₁ + (1−f)·ε₂]

    where  Pₙ = n̂ n̂ᵀ,  Pₜ = I − Pₙ.

    We invert element-wise to get ε_eff and extract the diagonal for
    each E-field component.  Off-diagonal terms are discarded (standard
    practice for Yee FDTD; Meep does the same).

    Returns (eps_xx, eps_yy, eps_zz).
    """
    # Harmonic (perpendicular) inverse-eps
    inv_perp = f / eps_inside + (1.0 - f) / (eps_outside + 1e-30)

    # Arithmetic (parallel) eps, then invert
    eps_par = f * eps_inside + (1.0 - f) * eps_outside
    inv_par = 1.0 / (eps_par + 1e-30)

    # Projection tensor Pₙ = n̂ n̂ᵀ   (only need diagonal + cross terms)
    nxnx = nx * nx
    nyny = ny * ny
    nznz = nz * nz

    # (ε_eff)⁻¹ diagonal elements:
    #   inv_xx = nxnx * inv_perp + (1 - nxnx) * inv_par
    inv_xx = nxnx * inv_perp + (1.0 - nxnx) * inv_par
    inv_yy = nyny * inv_perp + (1.0 - nyny) * inv_par
    inv_zz = nznz * inv_perp + (1.0 - nznz) * inv_par

    # Invert to get ε_eff diagonals
    eps_xx = 1.0 / (inv_xx + 1e-30)
    eps_yy = 1.0 / (inv_yy + 1e-30)
    eps_zz = 1.0 / (inv_zz + 1e-30)

    return eps_xx, eps_yy, eps_zz


# ---------------------------------------------------------------------------
# Host-float64 evaluation for concrete inputs (#833)
# ---------------------------------------------------------------------------
#
# The smoothed permittivity is a constant geometry input, not a traced design
# variable, in every in-tree caller. Evaluating it through ``jax.numpy`` tied
# the result to ``jax_enable_x64``: with the flag off, the SDF, fill fraction,
# normals and Kottke averaging all rounded in float32, and every interface
# voxel came out up to 15 f32 ulps away from the x64=1 value (measured on the
# boundary-voxel fixture in tests/unit/geometry/test_smoothing_coordinate_
# contract.py). Coordinate routing alone (ab280a38, #1088) could not close
# that: the coordinates were already the correctly rounded float32 of the
# exact spine, and the rest was arithmetic.
#
# So when every input is concrete -- host coordinate arrays, Python-float
# shape parameters and permittivities -- the whole chain runs in numpy
# float64 and the result is cast to the active JAX dtype ONCE at the end.
# Then eps(x64=0) is exactly float32(eps(x64=1)) by construction. Any traced
# input (a Sphere radius under ``jax.grad``, a mesh-as-design-variable axis)
# takes the same body through ``jax.numpy`` unchanged: one formula, two
# array backends, chosen by :func:`_host_path`.

def _shape_parameters(shape: Shape) -> list:
    """The numbers an SDF / normal reads off ``shape`` (empty for shapes
    without an SDF -- their staircase mask is evaluated, not traced)."""
    from rfx.geometry.csg import Box, Sphere, Cylinder
    if isinstance(shape, Box):
        return [*shape.corner_lo, *shape.corner_hi]
    if isinstance(shape, Sphere):
        return [*shape.center, shape.radius]
    if isinstance(shape, Cylinder):
        return [*shape.center, shape.radius, shape.height]
    return []


def _host_path(*values) -> bool:
    """True when no value is a JAX tracer, so the smoothing may run in host
    float64. Arrays and Python scalars are both accepted."""
    from rfx.core.jax_utils import is_tracer
    return not any(is_tracer(v) for v in values)


def _has_sdf(shape: Shape) -> bool:
    return _get_sdf_fn(shape) is not None and _get_normal_fn(shape) is not None


def _to_jax(arrays, *, host: bool, promoted: bool):
    """Cast host results to the dtype the ``jax.numpy`` path returns.

    The JAX path yields the active default float dtype (float32 at x64=0,
    float64 at x64=1) whenever SDF arithmetic on the coordinate arrays
    reached the output (``promoted``), and float32 otherwise (an all-
    staircase or empty shape list never leaves the float32 accumulators).
    The host path reproduces that so callers see the same dtypes they did
    before; this is the ONE cast on the host path.
    """
    if not host:
        return tuple(arrays)
    dtype = jax.dtypes.canonicalize_dtype(np.float64) if promoted else jnp.float32
    return tuple(jnp.asarray(a, dtype=dtype) for a in arrays)


def _kottke_smooth(xp, shapes, sample_coords, cell_len, eps, fallback_mask):
    """Kottke-average ``shapes`` into the per-component permittivity arrays.

    This is the ONE smoothing body; the uniform and non-uniform public
    functions differ only in how they build the sample coordinates and the
    local cell length, and in what a staircase fallback does.

    Parameters
    ----------
    xp : numpy or jax.numpy
    shapes : list of (shape, eps_r)
        Applied in order; later shapes overwrite earlier ones. Shapes sharing
        an ``eps_r`` form one group and are smoothed against their union SDF
        with the nearest shape's analytic normal, so overlapping same-material
        bodies are not double-smoothed.
    sample_coords : {"ex" | "ey" | "ez": (X, Y, Z)}
        Broadcastable coordinate arrays at each component's Yee position.
    cell_len : scalar or broadcastable array
        Local cell length normalising SDF -> fill fraction,
        ``f = clip(0.5 - sdf / cell_len, 0, 1)``.
    eps : (eps_ex, eps_ey, eps_ez)
        Starting (background) arrays; returned updated.
    fallback_mask : callable(shape) -> bool mask or None
        Staircase mask for a shape without an SDF; ``None`` skips it.
    """
    eps = dict(zip(("ex", "ey", "ez"), eps))
    from collections import OrderedDict
    groups: OrderedDict[float, list] = OrderedDict()
    for shape, eps_r in shapes:
        groups.setdefault(eps_r, []).append(shape)

    for eps_r, group_shapes in groups.items():
        sdf_shapes = [s for s in group_shapes if _has_sdf(s)]

        # Fallback shapes: staircased mask
        for shape in group_shapes:
            if _has_sdf(shape):
                continue
            mask = fallback_mask(shape)
            if mask is None:
                continue
            for comp in eps:
                eps[comp] = xp.where(mask, eps_r, eps[comp])

        if not sdf_shapes:
            continue

        # --- For each E-component position, compute union SDF and
        #     select the best analytic normal from the nearest shape ---
        for comp, (Xc, Yc, Zc) in sample_coords.items():
            # Union SDF + per-voxel nearest-shape tracking: for analytic
            # normals we pick the shape whose SDF is closest to zero (the
            # one whose boundary is nearest).
            sdf_union = None
            best_abs_sdf = None
            best_nx = best_ny = best_nz = None

            for shape in sdf_shapes:
                s = _get_sdf_fn(shape)(Xc, Yc, Zc, shape, xp=xp)
                n_x, n_y, n_z = _get_normal_fn(shape)(Xc, Yc, Zc, shape, xp=xp)

                if sdf_union is None:
                    sdf_union = s
                    best_abs_sdf = xp.abs(s)
                    best_nx, best_ny, best_nz = n_x, n_y, n_z
                else:
                    sdf_union = xp.minimum(sdf_union, s)
                    # Update normal where this shape's surface is closer
                    closer = xp.abs(s) < best_abs_sdf
                    best_abs_sdf = xp.where(closer, xp.abs(s), best_abs_sdf)
                    best_nx = xp.where(closer, n_x, best_nx)
                    best_ny = xp.where(closer, n_y, best_ny)
                    best_nz = xp.where(closer, n_z, best_nz)

            # Fill fraction from union SDF
            f = xp.clip(0.5 - sdf_union / cell_len, 0.0, 1.0)

            # Fix #1: fully-inside voxels get bulk eps, no smoothing
            inside = f >= 1.0
            # Boundary voxels
            bnd = (f > 0.0) & (f < 1.0)

            # The "outside" eps is whatever was there before.
            # Fix #3: full Kottke tensor averaging
            kt = _kottke_tensor_eps(f, eps_r, eps[comp], best_nx, best_ny, best_nz)
            smooth = kt[("ex", "ey", "ez").index(comp)]

            # Apply: interior -> bulk, boundary -> Kottke, exterior -> unchanged
            eps[comp] = xp.where(inside, eps_r, xp.where(bnd, smooth, eps[comp]))

    return eps["ex"], eps["ey"], eps["ez"]


def _uniform_sample_coords(xp, coords, dx):
    """Yee-position coordinates on a uniform grid from the shared node
    spine: Ex at (i+1/2, j, k), Ey at (i, j+1/2, k), Ez at (i, j, k+1/2).
    Centres are derived FROM nodes by adding half a cell -- see
    :func:`_yee_coords` for why the direction is the contract."""
    x, y, z = (xp.asarray(a) for a in (coords.x, coords.y, coords.z))
    X, Y, Z = x[:, None, None], y[None, :, None], z[None, None, :]
    half = dx * 0.5
    return {"ex": (X + half, Y, Z), "ey": (X, Y + half, Z), "ez": (X, Y, Z + half)}


def _smoothed_eps_uniform(xp, grid, coords, shapes, background_eps):
    """``compute_smoothed_eps`` body on backend ``xp``; returns ``xp`` arrays."""
    acc_dtype = np.float64 if xp is np else jnp.float32
    eps = tuple(xp.full(grid.shape, background_eps, dtype=acc_dtype) for _ in range(3))
    return _kottke_smooth(
        xp, shapes, _uniform_sample_coords(xp, coords, grid.dx), grid.dx, eps,
        fallback_mask=lambda shape: shape.mask(grid),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_smoothed_eps_nonuniform(
    nu_grid,
    shapes: list[tuple[Shape, float]],
    background_eps: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Kottke tensor-averaged ε on a NonUniformGrid (per-component).

    Same semantics as :func:`compute_smoothed_eps` but uses per-axis
    cell-size arrays from ``nu_grid`` for both coordinate placement
    (per-axis half-cell offsets) and SDF fill-fraction normalisation.

    Fill-fraction normalisation uses the geometric mean of the three
    local cell sizes — first-order accurate for non-cubic Yee cells, in
    line with the SDF-to-fill approximation in the uniform path.

    Concrete inputs are evaluated in host float64 and cast once (#833);
    a traced axis (mesh as design variable) keeps traced arithmetic.
    """
    from rfx.geometry.rasterize_grid import (
        cell_sizes_from_nonuniform_grid, centres_from_nonuniform_grid,
        coords_from_nonuniform_grid,
    )

    coords = coords_from_nonuniform_grid(nu_grid)
    # `coords` are E-NODE positions (cell edges) since #562 unified the NU
    # convention with the uniform builder's. This function needs BOTH the node
    # and the cell centre per axis, so the centre is derived FROM the node
    # rather than the other way round. Getting that direction wrong is not a
    # cosmetic slip: it displaces every smoothed voxel by half a cell, which is
    # exactly the defect #562 removed elsewhere, and the committed
    # interface-value bound (1 < eps < 4) passes under either sign — see
    # test_compute_smoothed_eps_nonuniform_reduces_to_uniform, which is the
    # assertion that catches it.
    #
    # Cell centres — centre[i] = node[i] + d[i]/2 from the shared producer
    # rasterize_grid.centres_from_nonuniform_grid: d comes from the float64
    # cell-size spine (not the float32 store) and the sum is formed in host
    # float64 (#833). ``node + f32(store)/2`` sat 3.7e-12 m off the exact
    # spine at x64=1 and 1.1e-9 m at x64=0 on the graded WR-90 fixture.
    centres = centres_from_nonuniform_grid(nu_grid, coords)
    # Per-cell sizes from the same float64 spine, for the fill-fraction
    # normalisation (a traced axis comes back as its tracer).
    cell_sizes = cell_sizes_from_nonuniform_grid(nu_grid)

    host = _host_path(*coords[:3], *centres[:3], *cell_sizes, background_eps,
                      *(e for _, e in shapes),
                      *(p for s, _ in shapes for p in _shape_parameters(s)))
    xp = np if host else jnp

    node_x, node_y, node_z = (xp.asarray(a) for a in coords[:3])
    centre_x, centre_y, centre_z = (xp.asarray(a) for a in centres[:3])
    dx_arr, dy_arr, dz_arr = (xp.asarray(d) for d in cell_sizes)

    # Local-cell characteristic length (geometric mean of three cell
    # widths) — used to normalise SDF → fill fraction. Anisotropic cell
    # geometry: this is a first-order approximation, mirroring the
    # uniform-path single-dx normalisation.
    dx_loc = (dx_arr[:, None, None]
              * dy_arr[None, :, None]
              * dz_arr[None, None, :]) ** (1.0 / 3.0)

    # Per-component half-cell offsets along the COMPONENT axis only:
    # Ex sits at (centre_x, node_y, node_z); Ey at (node_x, centre_y,
    # node_z); Ez at (node_x, node_y, centre_z).
    sample = {
        "ex": (centre_x[:, None, None], node_y[None, :, None], node_z[None, None, :]),
        "ey": (node_x[:, None, None], centre_y[None, :, None], node_z[None, None, :]),
        "ez": (node_x[:, None, None], node_y[None, :, None], centre_z[None, None, :]),
    }

    def fallback_mask(shape):
        # Staircase via the shape's cell mask if available; otherwise skip
        # (NU grid has no Grid-style mask adapter for arbitrary shapes today).
        if not hasattr(shape, "mask"):
            return None
        try:
            return shape.mask(nu_grid)
        except Exception as exc:
            # A shape whose .mask() cannot handle a NonUniformGrid is
            # skipped — but make that visible. The pre-fix bare
            # ``except: pass`` silently dropped the shape's geometry AND
            # would have masked a real bug (NaN, typo, unexpected error)
            # just as quietly.
            warnings.warn(
                f"smoothing: {type(shape).__name__}.mask() failed "
                f"on the non-uniform grid ({exc!r}); this shape is "
                f"SKIPPED and its geometry is NOT applied.",
                stacklevel=3,
            )
            return None

    acc_dtype = np.float64 if host else jnp.float32
    eps = tuple(xp.full(tuple(nu_grid.shape), background_eps, dtype=acc_dtype)
                for _ in range(3))
    eps = _kottke_smooth(xp, shapes, sample, dx_loc, eps, fallback_mask)
    return _to_jax(eps, host=host, promoted=any(_has_sdf(s) for s, _ in shapes))


def compute_smoothed_eps(
    grid: Grid,
    shapes: list[tuple[Shape, float]],
    background_eps: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Kottke tensor-averaged permittivity for each E-field component.

    At interface voxels, uses the full Kottke inverse-permittivity
    tensor with analytic interface normals.  Interior voxels get the
    bulk permittivity.

    Concrete inputs (host node spine, Python-float shape parameters and
    permittivities) are evaluated in host float64 and cast to the active
    JAX dtype once, so the result does not depend on ``jax_enable_x64``
    (#833); a traced shape parameter keeps traced ``jax.numpy`` arithmetic.

    Parameters
    ----------
    grid : Grid
    shapes : list of (shape, eps_r) pairs
        Applied in order; later shapes overwrite earlier ones.
    background_eps : float
        Background relative permittivity.

    Returns
    -------
    eps_ex, eps_ey, eps_ez : jnp.ndarray
        Per-component permittivity arrays, each of shape grid.shape.
    """
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    coords = coords_from_uniform_grid(grid)
    host = _host_path(*coords[:3], grid.dx, background_eps,
                      *(e for _, e in shapes),
                      *(p for s, _ in shapes for p in _shape_parameters(s)))
    eps = _smoothed_eps_uniform(np if host else jnp, grid, coords, shapes, background_eps)
    return _to_jax(eps, host=host, promoted=any(_has_sdf(s) for s, _ in shapes))


# ---------------------------------------------------------------------------
# Stage 2 — unified inverse-permittivity tensor with PEC limit
# ---------------------------------------------------------------------------
#
# Returns the diagonal of the lab-frame ε̄⁻¹ tensor directly, without
# the round-trip inversion that ``_kottke_tensor_eps`` does at the end.
# The PEC branch (ε_inside → ∞) is handled explicitly so the limit
# inv_perp = (1−f)/ε_outside, inv_par = 0 is reached without the
# `1e-30` epsilon trap that would otherwise produce ε ≈ 1e30 (huge but
# finite) for ε_inside = ∞.
#
# Reference: subpixel-averaging C_a/C_b derivation.


def _kottke_inv_eps_diag(
    f,
    eps_inside,
    eps_outside,
    n_x,
    n_y,
    n_z,
    *,
    is_pec: bool = False,
    xp=jnp,
):
    """Diagonal of the Kottke (ε̄⁻¹)_lab tensor at one point.

    Parameters
    ----------
    f : scalar or array
        Fill fraction of the inside material (0 ≤ f ≤ 1).
    eps_inside : scalar or array
        Permittivity inside the shape. Pass ``jnp.inf`` (or any large
        sentinel) when ``is_pec=True``; the value is then ignored.
    eps_outside : scalar or array
        Background permittivity (where the cell is *not* inside the shape).
    n_x, n_y, n_z : scalar or array
        Components of the outward unit normal at the interface.
    is_pec : bool
        Switches to the σ→∞ / ε→∞ limit branch (Farjadpour 2006 §VI
        plus the limit derived in stage2_ca_cb_derivation.md §4).
    xp : numpy or jax.numpy
        Array backend; ``numpy`` on the concrete host-float64 path (#833).

    Returns
    -------
    (inv_xx, inv_yy, inv_zz)
        Diagonal entries of the lab-frame ε̄⁻¹ tensor.
    """
    if is_pec:
        # PEC limit (ε_inside → ∞):
        # ⟨ε⁻¹⟩ = f/∞ + (1−f)/ε_out → (1−f)/ε_out (finite for any f ≥ 0)
        # ⟨ε⟩    = f·∞ + (1−f)·ε_out → ∞ (for any f > 0); = ε_out for f=0
        # ⟨ε⟩⁻¹ → 0 (for any f > 0); = 1/ε_out for f=0
        # The discontinuity at f=0 is physically correct — any amount
        # of PEC freezes the parallel direction.
        inv_perp = (1.0 - f) / eps_outside
        inv_par = xp.where(
            f > 0.0,
            xp.zeros_like(inv_perp),
            1.0 / eps_outside,
        )
    else:
        # Standard isotropic-on-both-sides Kottke (Farjadpour 2006 Eq. 1).
        # No 1e-30 guard needed — for finite eps > 0 inputs, the
        # arithmetic is well-defined.
        inv_perp = f / eps_inside + (1.0 - f) / eps_outside
        eps_par = f * eps_inside + (1.0 - f) * eps_outside
        inv_par = 1.0 / eps_par

    nxnx = n_x * n_x
    nyny = n_y * n_y
    nznz = n_z * n_z
    inv_xx = nxnx * inv_perp + (1.0 - nxnx) * inv_par
    inv_yy = nyny * inv_perp + (1.0 - nyny) * inv_par
    inv_zz = nznz * inv_perp + (1.0 - nznz) * inv_par
    return inv_xx, inv_yy, inv_zz


def compute_inv_eps_tensor_diag(
    grid: Grid,
    dielectric_shapes: list[tuple[Shape, float]] | None = None,
    pec_shapes: list[Shape] | None = None,
    background_eps: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Stage 2 unified subpixel ε⁻¹ tensor: dielectric Kottke + PEC limit.

    Computes the diagonal of the lab-frame inverse-permittivity tensor
    at every Yee component position. PEC shapes are handled via the
    σ→∞ limit (zero parallel-direction inverse-eps; (1−f)/ε_outside on
    the perpendicular direction). Dielectric shapes route through the
    existing Kottke isotropic Eq. (1) form.

    Parameters
    ----------
    grid : Grid
    dielectric_shapes : list of (Shape, eps_r), optional
        Same as ``compute_smoothed_eps`` — applied first.
    pec_shapes : list of Shape, optional
        PEC objects. Each is folded in via elementwise minimum on the
        per-component inverse-eps tensor.

        **Precondition (Stage 2 v1)**: ``pec_shapes`` must be
        **non-overlapping** in space. Two overlapping PEC bodies with
        different surface normals at the same Yee point would produce
        incorrect tensor entries under the current min-merge — each
        body contributes a Kottke result based on its own normal, and
        ``min`` over independently-projected diagonals can collapse to
        zero on both axes even when the geometric union of fills is
        less than 1. The proper resolution is the union-SDF +
        nearest-shape-normal pattern that ``compute_smoothed_eps``
        already uses (see ``OrderedDict`` grouping, line 474+) —
        deferred to Stage 2 v2 once a multi-PEC use case appears in
        the acceptance ladder. For axis-aligned WR-90, mitred bend,
        and Vivaldi (Tier 0-2 use cases), non-overlapping PEC is the
        natural representation and this precondition holds trivially.
    background_eps : float
        Vacuum / background ε. Default 1.0.

    Returns
    -------
    (inv_xx, inv_yy, inv_zz)
        Per-component inverse-permittivity diagonal arrays, each of
        shape ``grid.shape``, dtype ``float32``. Feed directly into
        ``update_e_aniso_inv`` (Stage 2 Step 2).
    """
    if dielectric_shapes is None:
        dielectric_shapes = []
    if pec_shapes is None:
        pec_shapes = []
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    coords = coords_from_uniform_grid(grid)
    host = _host_path(*coords[:3], grid.dx, background_eps,
                      *(e for _, e in dielectric_shapes),
                      *(p for s, _ in dielectric_shapes for p in _shape_parameters(s)),
                      *(p for s in pec_shapes for p in _shape_parameters(s)))
    xp = np if host else jnp

    # Step 1: dielectric subpixel smoothing (existing Kottke path).
    if dielectric_shapes:
        eps = _smoothed_eps_uniform(xp, grid, coords, dielectric_shapes, background_eps)
    else:
        acc_dtype = np.float64 if host else jnp.float32
        eps = tuple(xp.full(tuple(grid.shape), background_eps, dtype=acc_dtype)
                    for _ in range(3))

    # The dielectric inverse is a float32 quantity by this function's
    # contract (downstream ``update_e_aniso_inv`` expects float32, and
    # ``eps_r`` may arrive as a Python float), so it is rounded here on both
    # backends and the PEC limit below is folded in on top of that value. The
    # host path widens the rounded value back to float64 for the PEC
    # arithmetic: that keeps x64=1 bit-identical to the pre-#833 path and
    # makes x64=0 its float32 image.
    inv = [(1.0 / e).astype(xp.float32) for e in eps]
    if host:
        inv = [a.astype(np.float64) for a in inv]

    if not pec_shapes:
        return _to_jax(inv, host=host, promoted=False)

    # Step 2: apply each PEC shape via Kottke PEC limit, taking the
    # elementwise minimum (union of PEC effects — the most-restrictive
    # contribution wins per cell). Each Yee E-component position carries
    # its own half-cell offset.
    sample = _uniform_sample_coords(xp, coords, grid.dx)
    promoted = False
    for pec_shape in pec_shapes:
        if not _has_sdf(pec_shape):
            # Fallback: staircase mask. Cells inside the shape get
            # full-PEC (inv = 0); cells outside are unchanged.
            mask = pec_shape.mask(grid)
            inv = [xp.where(mask, 0.0, a) for a in inv]
            continue
        promoted = True
        for i, comp in enumerate(("ex", "ey", "ez")):
            Xc, Yc, Zc = sample[comp]
            sdf = _get_sdf_fn(pec_shape)(Xc, Yc, Zc, pec_shape, xp=xp)
            n_x, n_y, n_z = _get_normal_fn(pec_shape)(Xc, Yc, Zc, pec_shape, xp=xp)

            f = xp.clip(0.5 - sdf / grid.dx, 0.0, 1.0)
            inv_c = _kottke_inv_eps_diag(
                f, xp.inf, eps[i], n_x, n_y, n_z, is_pec=True, xp=xp,
            )[i]
            # Kottke correctly zeros parallel (tangential) components for
            # f > 0, but assigns inv_perp = (1−f)/ε > 0 when the
            # E-position is inside the PEC body (SDF < 0, 0 < f < 1).
            # That fractional inv for a point already inside the PEC is
            # physically wrong (E = 0 inside a conductor) and creates a
            # CPML resonance at boundary cells near domain walls.
            # Override: wherever the E-component Yee position is inside
            # the PEC shape (SDF ≤ 0), force its inv component to 0.
            e_inside = (sdf <= 0.0)
            inv_c = xp.where(e_inside, 0.0, inv_c)
            inv[i] = xp.minimum(inv[i], inv_c)

    return _to_jax(inv, host=host, promoted=promoted)


def kottke_inv_eps_from_occupancy(
    grid: Grid,
    pec_occupancy: jnp.ndarray,
    *,
    aniso_inv_eps_baseline: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    background_eps: float = 1.0,
    grad_eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Stage 2 Kottke inv-eps tensor from a continuous-fill PEC occupancy.

    The AD-traceable analogue of the ``pec_shapes`` branch in
    :func:`compute_inv_eps_tensor_diag`.  ``pec_occupancy`` is a per-cell
    fill fraction in ``[0, 1]``; ``f = occ`` plays the role of Kottke's
    fill, and the interface normal is derived from ``∇occ / |∇occ|``.

    For a sigmoid mask, ``∇occ`` is large only at the boundary cells
    (occ ≈ 0.5 transition zone); interior PEC cells (occ ≈ 1) and
    exterior vacuum cells (occ ≈ 0) have small ``|∇occ|`` but
    Kottke's diagonal output is independent of the normal there
    (``inv_perp = inv_par = (1−f)/ε`` reduces to a scalar at f → 0
    or f → 1 in the PEC limit), so the small-norm regime is benign.

    Yee staggering is approximated by applying the cell-centred Kottke
    output to all three E-component positions of the cell.  The error
    is O(dx) at the sigmoid edge — second-order to the staircase
    error that the override path replaces.

    Parameters
    ----------
    grid : Grid
    pec_occupancy : (nx, ny, nz) float array
        Continuous-fill PEC field; values clipped to ``[0, 1]``.
    aniso_inv_eps_baseline : (inv_xx, inv_yy, inv_zz) tuple, optional
        Pre-computed inv-eps tensor with dielectric subpixel smoothing
        already applied.  When supplied, the Kottke PEC limit is taken
        as the elementwise minimum against this baseline (union of
        PEC effects on top of dielectrics).  When ``None``, the
        background uses ``background_eps``.
    background_eps : float
        Background permittivity used when ``aniso_inv_eps_baseline``
        is ``None``.  Default 1.0 (vacuum).
    grad_eps : float
        Numerical guard for ``|∇occ|`` normalisation.  Cells with
        ``|∇occ| < grad_eps`` get a default normal direction (x̂);
        the Kottke output is independent of the normal in those cells
        (see docstring above).

    Returns
    -------
    (inv_xx, inv_yy, inv_zz)
        Per-component inverse-permittivity diagonal arrays, each of
        shape ``grid.shape``, dtype ``float32``.  Feed directly into
        ``simulation.run(aniso_inv_eps=...)`` or
        ``update_e_aniso_inv``.
    """
    f = jnp.clip(pec_occupancy.astype(jnp.float32), 0.0, 1.0)
    # Clamp the sigmoid tail to exactly zero so the strict-Kottke
    # `f > 0` selector does not trip on `1e-30` etc.  AD cells in the
    # tail (occ < 1e-3) contribute nothing to the cost — their
    # gradient is genuinely zero at this scale — so the hard clamp
    # is functionally smooth.  Cells above the threshold flow
    # gradients normally.
    f = jnp.where(f < 1e-3, jnp.zeros_like(f), f)

    # Central-difference gradient of the occupancy.  ``jnp.roll`` gives
    # periodic boundary semantics; on rfx's CPML/PEC domain boundaries
    # the occupancy is vacuum (0) by construction (PEC stub geometry
    # never extends to the absorbing boundary).  Periodic boundary
    # therefore evaluates to "0 next to 0" at the domain edge, giving
    # a vanishing gradient — physically correct: no interface there.
    dx = float(grid.dx)
    dy = float(getattr(grid, "dy", grid.dx))
    dz = float(getattr(grid, "dz", grid.dx))
    grad_x = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    grad_y = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    grad_z = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2.0 * dz)
    norm = jnp.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + grad_eps ** 2)
    n_x = grad_x / norm
    n_y = grad_y / norm
    n_z = grad_z / norm

    # Kottke PEC limit at every cell.  ``eps_outside`` is the dielectric
    # background — for the sigmoid stub on ro4350b, this is the substrate
    # ε at substrate cells and vacuum elsewhere.  When the caller passes
    # an ``aniso_inv_eps_baseline``, we use the baseline's reciprocal
    # as the local background ε (one per axis); otherwise the scalar
    # ``background_eps`` applies everywhere.
    if aniso_inv_eps_baseline is not None:
        inv_xx_b, inv_yy_b, inv_zz_b = aniso_inv_eps_baseline
        # The PEC Kottke needs a scalar eps_outside per cell — the
        # baseline already encodes the dielectric subpixel smoothing
        # per axis.  We invert each component locally.
        eps_outside_x = 1.0 / (inv_xx_b + 1e-30)
        eps_outside_y = 1.0 / (inv_yy_b + 1e-30)
        eps_outside_z = 1.0 / (inv_zz_b + 1e-30)
    else:
        bg = jnp.asarray(background_eps, dtype=jnp.float32)
        eps_outside_x = bg
        eps_outside_y = bg
        eps_outside_z = bg

    # Strict-Kottke PEC limit (`is_pec=True`): `inv_par = 0` for any
    # f > 0 (frozen parallel-to-interface E), `inv_perp = (1-f)/eps`.
    # The f-tail clamp above ensures only "real" PEC cells (occ ≥
    # 1e-3) trip the `f > 0` selector — sigmoid-floor values like
    # 1e-30 are clamped to 0 and correctly behave as vacuum.
    inv_xx_c, _, _ = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside_x, n_x, n_y, n_z, is_pec=True,
    )
    _, inv_yy_c, _ = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside_y, n_x, n_y, n_z, is_pec=True,
    )
    _, _, inv_zz_c = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside_z, n_x, n_y, n_z, is_pec=True,
    )

    # Force-zero interior cells AND dilate the PEC by 1 cell so the
    # boundary cell acts as a hard mirror.  The reference imperative
    # path (``compute_inv_eps_tensor_diag``, line ~779) sets
    # `inv = where(e_inside, 0, inv)` — interior cells are exactly 0;
    # only the *single* boundary cell at the SDF crossing keeps the
    # Kottke output.  For the occupancy path we don't have an SDF;
    # we replicate this by applying a sigmoid Heaviside projection to
    # the *neighbor-max* occupancy.  Cells whose own occ OR any 6-neighbor
    # occ exceeds 0.5 get full PEC (interior + 1-cell dilation); cells
    # with all neighbors at ≤ 0.5 keep their Kottke output.  This
    # dilation is the AD-smooth analogue of the binary `apply_pec_mask`
    # `pec_mask & (roll | roll)` rule (which is what the legacy
    # imperative cross-solver uses for hard `Box(material="pec")`).
    occ_dilated = f
    for _axis in range(3):
        occ_dilated = jnp.maximum(occ_dilated, jnp.roll(f, 1, axis=_axis))
        occ_dilated = jnp.maximum(occ_dilated, jnp.roll(f, -1, axis=_axis))
    smooth_width = 0.05
    interior_mask = jax.nn.sigmoid((occ_dilated - 0.5) / smooth_width)
    inv_xx_c = (1.0 - interior_mask) * inv_xx_c
    inv_yy_c = (1.0 - interior_mask) * inv_yy_c
    inv_zz_c = (1.0 - interior_mask) * inv_zz_c

    if aniso_inv_eps_baseline is not None:
        inv_xx_b, inv_yy_b, inv_zz_b = aniso_inv_eps_baseline
        inv_xx = jnp.minimum(inv_xx_b, inv_xx_c).astype(jnp.float32)
        inv_yy = jnp.minimum(inv_yy_b, inv_yy_c).astype(jnp.float32)
        inv_zz = jnp.minimum(inv_zz_b, inv_zz_c).astype(jnp.float32)
    else:
        inv_xx = inv_xx_c.astype(jnp.float32)
        inv_yy = inv_yy_c.astype(jnp.float32)
        inv_zz = inv_zz_c.astype(jnp.float32)

    return inv_xx, inv_yy, inv_zz

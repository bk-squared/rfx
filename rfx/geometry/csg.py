"""Constructive Solid Geometry (CSG) primitives.

Defines geometric shapes and boolean operations that produce material arrays
from geometric descriptions. All shapes operate on grid coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.jax_utils import is_tracer

# PROTOTYPE (#802b): a declared face that lies within this fraction of the
# LOCAL cell size of a node is treated as lying exactly ON that node. The
# half-open test ``lo <= x_j < hi`` is then evaluated on the snapped face, so
# an algebraically node-aligned face (``k*dx`` typed as a decimal, or
# ``a - n*dx`` computed in float64) rasterizes identically however it was
# computed and whatever ``jax_enable_x64`` is. 1e-6 of a cell is ~1e4x the
# float64 rounding of any realistic coordinate and ~1e-3 of the smallest
# offset a user could mean (a 1e-9 m offset at dx=100 um is 1e-5 dx and is
# NOT snapped). Applies to concrete coordinates only; the traced lane keeps
# the plain float32 comparison (see Box._axis_mask).
NODE_SNAP_REL = 1e-6


class Shape(Protocol):
    """Protocol for CSG shapes — must implement mask and mask_on_coords."""

    def mask(self, grid: Grid) -> jnp.ndarray:
        """Return boolean mask (True inside shape) on the given grid."""
        ...

    def mask_on_coords(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate shape occupancy on explicit 1D coordinate arrays.

        Coordinates are **node** positions, and each shape defines its own
        boundary rule; :class:`Box` is half-open ``[lo, hi)``, which makes a
        drawn extent realize one cell shorter at the ``hi`` face. See the
        :class:`Box` docstring before drawing a PEC obstacle to a nominal
        physical dimension.

        Parameters
        ----------
        x, y, z : 1D arrays of physical coordinates (metres)

        Returns
        -------
        (Nx, Ny, Nz) boolean array — True inside the shape.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement mask_on_coords(). "
            f"This shape cannot be used on nonuniform or subgridded meshes."
        )

    def bounding_box(self) -> tuple[tuple[float, float, float],
                                     tuple[float, float, float]]:
        """Return (corner_lo, corner_hi) axis-aligned bounding box."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement bounding_box()."
        )


def _grid_coords(grid: Grid):
    """Extract 1D physical coordinate arrays from a uniform Grid.

    PROTOTYPE (#802): concrete float64 numpy node coordinates
    ``(i - pad) * dx``, independent of ``jax_enable_x64``. Under the JAX
    default dtype the float32 product ``i*dx`` lands ~1e-10 m off the exact
    node value and the half-open ``[lo, hi)`` test in ``Box._axis_mask``
    flips at node-aligned faces (hi face gained, lo face lost). The uniform
    grid's nodes are never traced, so nothing downstream needs them as JAX
    arrays; shapes evaluated on these coordinates keep their comparisons in
    float64 (see ``Box.mask_on_coords``).
    """
    nx, ny, nz = grid.shape
    dx = float(grid.dx)
    pad_x, pad_y, pad_z = grid.axis_pads
    x = (np.arange(nx, dtype=np.float64) - pad_x) * dx
    y = (np.arange(ny, dtype=np.float64) - pad_y) * dx
    z = (np.arange(nz, dtype=np.float64) - pad_z) * dx
    return x, y, z


def _snap_tol(*coord_arrays) -> float:
    """PROTOTYPE (#802b): ``NODE_SNAP_REL`` times the smallest positive node
    spacing over the given concrete axes; ``0.0`` when any axis is traced (the
    traced lane keeps the plain comparison) or no spacing is defined."""
    spacings = []
    for c in coord_arrays:
        if is_tracer(c):
            return 0.0
        a = np.asarray(c, dtype=np.float64).ravel()
        if a.size > 1:
            d = np.diff(a)
            d = d[d > 0]
            if d.size:
                spacings.append(float(d.min()))
    return NODE_SNAP_REL * min(spacings) if spacings else 0.0


@dataclass(frozen=True)
class Box:
    """Axis-aligned box defined by two corners (meters).

    **Rasterization convention (read before drawing a PEC obstacle).**
    On each axis the volume branch is **half-open** ``[lo, hi)`` over
    **node** coordinates: node ``j`` at ``y_j = j * dx`` belongs to the box
    iff ``lo <= y_j < hi``. The convention is deliberate and several paths
    depend on it (see ``_axis_mask`` for the issue history), but it has two
    consequences that bite when a box is drawn to a nominal physical size:

    1. **The ``hi`` face contributes no cell.** A box whose corners both
       land on node planes, ``lo = i*dx`` and ``hi = k*dx``, occupies nodes
       ``i .. k-1``. Its realized extent between the first and last occupied
       node plane is therefore ``(hi - lo) - dx``, one cell short of the
       drawn extent, and the shortfall is entirely at the ``hi`` face, so a
       **single box** is also displaced by ``dx/2`` toward ``lo``. Note the
       two faces are NOT interchangeable: ``lo`` is inclusive and ``hi`` is
       exclusive. Whether that per-box displacement survives in a
       multi-object structure depends on which face of each object is the
       load-bearing one — see the facing-pair discussion below, where it
       cancels in one case and not the other.

    2. **A corner exactly on a node plane is a knife edge.** What you can
       observe: *the same drawing recipe, on the same grid, gives a
       one-cell-too-wide opening at one aperture and a two-cell-too-wide
       opening at another* — WR-90 fins drawn to the nominal opening give
       ``d + dx`` at ``d`` = 7.620 and 18.288 mm but ``d + 2*dx`` at
       12.192 mm, at both a/30 and a/60. Nothing about the nominal
       dimensions predicts which.

       Why: masks are evaluated in the default float32 precision, and the
       node coordinates are themselves double-rounded.
       :func:`_grid_coords` computes ``f32(f32(i) * f32(dx))`` while a
       caller's corner is computed in float64 and cast once, so
       **algebraically equal values land on opposite sides of the
       comparison**. On a real WR-90 grid an f64 reconstruction of the nodes
       disagrees with production on 30 of 31 nodes by up to 1.1e-9 m (1e-6
       of a cell), which is enough to move the occupied-node count by a
       whole cell. A corner computed as ``a - n*dx`` may thus rasterize
       differently from the algebraically identical ``m*dx``.

    For a **PEC obstacle** the occupied nodes are where the tangential ``E``
    is zeroed, so consequence (1) is an *electrical* dimension error, not a
    sub-cell cosmetic one. **Transverse to the propagation direction** — an
    aperture width, a guide width, anything that sets a cutoff — the
    electrical dimension is the span between the innermost ZEROED node
    planes, ``(n_open + 1) * dx``, the same measure that reproduces the
    guide's own width ``a = cells * dx`` exactly (counting open nodes alone
    would call WR-90 22.098 mm instead of 22.86 at a/30). An independent
    refit of 16 committed WR-90 single-iris configurations across two meshes
    pins the realized transverse aperture offset to within 1/20 of a cell of
    that identity (measured during the stage-S3 / issue #499 review; no
    committed record carries the refit yet, so like the longitudinal caution
    below it is a recorded observation, not a number to build on).

    **Do not carry that identity into the propagation direction.** The
    electrical *thickness* of an obstacle (an iris's extent along the guide
    axis) is not the zeroed-plane span: it is set by how the fields interact
    with the discontinuity, not by a cutoff condition. Measured on a
    single-iris geometry at four drawn thicknesses, the effective thickness
    lands consistently **between** ``t_cells * dx`` and
    ``(t_cells - 1) * dx``, so *neither* integer rule is right and the
    residual is a fixed per-face offset rather than a discretization error
    that shrinks with ``dx``. The longitudinal convention is **not settled**;
    treat it as an unknown of order half a cell, and if a comparator needs a
    thickness, make the sensitivity to that half cell part of the reported
    envelope rather than picking a rule. (Measured during the stage-S3 /
    issue #499 review; no committed record carries it yet, so it is stated
    here as a caution, not as a number to build on.) This measured
    *effective* thickness is a different quantity from a cascade
    comparator's electrical-length bookkeeping — issue #499's comparator
    deliberately draws ``t_c = round(t/dx) + 1`` so that ``(t_c - 1)*dx``
    conserves the cascade's total electrical length; that bookkeeping choice
    answers a different question and is not contradicted by this caution.

    **A facing pair is not symmetric under this rule, because its two
    interior faces are different kinds of corner.** For fins drawn from each
    wall inward, the lo fin's interior face is a ``hi`` corner, which
    half-openness **always** drops, so that fin always retreats one cell.
    The hi fin's interior face is a ``lo`` corner, which ``coords >= lo``
    normally **keeps** — it retreats only when (2) puts the node just below
    the corner. Hence the realized opening is:

    * ``d + dx`` when only the lo fin retreated. The opening is then
      **asymmetric**: its centre sits ``dx/2`` below the guide centre.
    * ``d + 2*dx`` when rounding made the hi fin retreat as well. The two
      retreats cancel and the opening is **symmetric**.

    Which one you get is **not predictable from the nominal dimensions**.
    Measured on WR-90 at both a/30 and a/60: ``d`` = 7.620 and 18.288 mm
    give ``d + dx`` (centre offset -0.5 cell), while ``d`` = 12.192 mm gives
    ``d + 2*dx`` (centred). Over ~99k (guide, mesh, aperture) combinations
    the split is 82% / 9% / 8% across ``d + dx`` / ``d + 2*dx`` / ``d`` at
    even parity, shifting one cell up at odd parity (see the recipe).
    Shifting each interior face half a cell the *wrong* way (toward the
    metal) retreats both faces by construction rather than by luck, giving
    ``d + 2*dx`` deterministically at every aperture — that is the drawing
    case 18's blocked revision used, and it is why re-comparing that
    revision against ``oracle(d + 2*dx)`` collapsed every row.

    In PR #480's WR-90 single-iris lane this inflated the ``|S11|`` error
    against an analytic mode-matching oracle by 4-6x, and because it scales
    with ``dx`` it is easy to misread as first-order convergence. For a
    **resonant** structure (multi-iris filter, cavity) it shifts the
    passband rather than widening a magnitude tolerance, so it will not look
    like a discretization error at all.

    **Recipe**, two conditions, both needed:

    * Put interior corners on **cell midpoints**, ``(j + 0.5) * dx``, to make
      node ``j`` the innermost occupied plane. Every corner value in a
      one-cell-wide interval selects the same footprint (``(j*dx, (j+1)*dx]``
      for a ``hi`` corner, ``[j*dx, (j+1)*dx)`` for a ``lo`` corner), and the
      midpoint is that interval's **centre** — not merely a safe nudge away
      from the node. That is precisely why it is immune to the float32
      effects in (2), which only ever perturb a corner by a fraction of a
      cell.
    * Keep the metal depth an exact number of cells, i.e. for a symmetric
      obstacle keep ``(cells - d_cells)`` **even**. When it is odd,
      ``fin_depth = (cells - d_cells)//2`` truncates and a symmetric opening
      of that width is simply not representable on the grid: the opening is
      one cell **wider** however it is drawn. That is a representability
      limit, not a rasterization defect.

    Under both conditions the realized opening equals the nominal one
    exactly (measured, 100% of ~50k even-parity combinations). Then still
    assert the realized footprint (count the occupied node planes) against
    the intended one — see ``run_point`` in
    ``validation/crossval/18_wr90_iris_modematch.py`` for the pattern.

    **Odd parity is a fork, not a dead end.** Symmetric fins can only realize
    apertures whose cell count has the parity of ``cells``, so when the
    aperture you want has the wrong parity you must choose, and both options
    cost something:

    * change ``dx`` (or the aperture) so the parity works, which moves every
      other dimension on that axis; or
    * place the fins **asymmetrically** on purpose, accepting a known
      half-cell offset of the opening rather than rounding the aperture to
      the wrong parity — which would change the aperture itself, the quantity
      that sets the cutoff.

    This docstring does **not** recommend one over the other: how much the
    half-cell offset costs has not been measured here, so treat it as an
    open trade rather than a cheap escape. What is *not* optional is that the
    offset be recorded and **representable by whatever you compare against** —
    an off-centre aperture only stays a known quantity if the oracle can model
    it. If your comparator assumes a centred obstacle, an intentional offset
    silently becomes comparator error, which is the failure mode this whole
    docstring exists to prevent.

    A box thinner than one local cell takes a separate **thin-sheet**
    branch (single nearest-centre node); the notes above apply to the
    volume branch only.
    """

    corner_lo: tuple[float, float, float]
    corner_hi: tuple[float, float, float]

    def bounding_box(self):
        return (self.corner_lo, self.corner_hi)

    def mask_on_coords(self, x, y, z):
        """Occupancy on explicit node coordinates.

        Volume branch is half-open ``[lo, hi)`` per axis, so the ``hi`` face
        contributes no node and a drawn extent realizes as ``extent - dx``
        between the outermost occupied planes. See the class docstring for
        the PEC-obstacle consequences and the half-cell-offset recipe.
        """
        def _axis_mask(coords, lo, hi):
            # Use LOCAL cell size at the geometry's midpoint — critical for
            # thin objects on a non-uniform axis. Using the first-cell dc
            # (as the legacy implementation did) causes a 0.25 mm PEC
            # sheet inside a 1 mm-dz region to be snapped onto two or
            # three cells (issue #48 / deep dive).
            #
            # Issue #75: thin-sheet snaps to the single argmin-nearest
            # cell; the volume path uses half-open ``coords < hi`` so a
            # ``hi`` landing on a cell centre does not admit an extra cell.
            #
            # GEO Tier-2: this stays fully JAX-traceable — ``np.asarray``
            # on ``coords`` raised TracerArrayConversionError on the
            # differentiable-mesh path. Both branches are computed and
            # selected with ``jnp.where``; the output is byte-identical
            # to the pre-refactor np-based path on concrete coordinates.
            # PROTOTYPE (#802): concrete numpy coordinates (the uniform
            # grid from ``_grid_coords``) are compared in float64 with
            # numpy so the realized cells do not depend on
            # ``jax_enable_x64``; ``jnp.asarray`` would DOWNCAST them to
            # float32 under the default config. Traced / JAX coordinates
            # (differentiable mesh, NU lane) take the jnp path unchanged.
            # PROTOTYPE (#802b): any CONCRETE coordinate array (numpy from
            # ``_grid_coords`` / ``coords_from_nonuniform_grid``, or a
            # concrete jnp array a caller built itself) is compared in
            # float64 with numpy; only a tracer takes the jnp lane.
            concrete = not is_tracer(coords)
            xp = np if concrete else jnp
            coords = (np.asarray(coords, dtype=np.float64) if concrete
                      else jnp.asarray(coords))
            mid = (lo + hi) * 0.5
            extent = float(hi - lo)          # lo/hi are concrete Box corners
            if coords.size <= 1:             # static (shape, not values)
                dc_local = 1e-3
            else:
                # Local cell WIDTH at the midpoint cell — the smaller of the two
                # neighbouring centre-to-centre spacings, NOT the single forward
                # spacing coords[k+1]-coords[k] the legacy code used (#374). At a
                # fine/coarse grading transition the forward spacing is the mean
                # of the two adjacent widths and over-counts, mis-classifying a
                # shape that spans >1 fine cell as thin (which then collapses it
                # onto a single argmin cell, losing extent). ``min`` is exact for
                # interior cells and the fine side of a transition, and errs on
                # the safe side (toward the volume branch, preserving extent) on
                # the coarse side. Exact cell widths are not recoverable from
                # centres alone; a fully exact classification would need the
                # grid's per-cell width array threaded through mask_on_coords.
                # On a uniform axis s_left==s_right==dx so this is bit-identical
                # to the legacy centre-spacing.
                k_mid = xp.clip(
                    xp.searchsorted(coords, mid) - 1, 0, coords.size - 2)
                im1 = xp.clip(k_mid - 1, 0, coords.size - 1)
                ip1 = xp.clip(k_mid + 1, 0, coords.size - 1)
                s_left = coords[k_mid] - coords[im1]    # 0 at the lo end
                s_right = coords[ip1] - coords[k_mid]   # 0 at the hi end
                dc_local = xp.where(
                    (s_left > 0) & (s_right > 0),
                    xp.minimum(s_left, s_right),
                    xp.maximum(s_left, s_right))
            # Thin sheet: the single cell whose centre is nearest ``mid``.
            # (#371) On the collocated scheme, apply_pec_mask zeros tangential
            # Ex/Ey at this cell's CENTRE, so nearest-centre = minimum realized-
            # plane placement error. A matching-thickness (sub-cell) box takes
            # this same thin branch and agrees; only a >=1-cell VOLUME box
            # (different object) selects a different layer on a graded axis.
            if concrete:
                # PROTOTYPE (#802b): snap tolerance relative to the LOCAL
                # cell; a face within ``tol`` of a node is ON that node.
                tol = NODE_SNAP_REL * float(dc_local)
                dist = np.abs(coords - mid)
                # Nearest-centre node; a TIE (two nodes equidistant from
                # ``mid`` within ``tol`` -- every 1-cell box drawn between
                # two node planes) goes to the LOWER node, i.e. the same
                # lo-inclusive side the volume branch keeps. Before #802
                # float32 noise decided this per axis.
                nearest_idx = int(np.argmax(dist <= dist.min() + tol))
                thin_mask = np.zeros(coords.shape, dtype=bool)
                thin_mask[nearest_idx] = True
                # Volume: half-open [lo, hi) on the snapped faces.
                volume_mask = (coords >= lo - tol) & (coords < hi - tol)
            else:
                nearest_idx = jnp.argmin(jnp.abs(coords - mid))
                thin_mask = jnp.zeros(coords.shape, dtype=bool).at[
                    nearest_idx].set(True)
                # Volume: half-open [lo, hi).
                volume_mask = (coords >= lo) & (coords < hi)
            # Thin sheet when the extent is within one local cell.
            is_thin = extent <= dc_local * 1.01
            return xp.where(is_thin, thin_mask, volume_mask)

        mx = _axis_mask(x, self.corner_lo[0], self.corner_hi[0])
        my = _axis_mask(y, self.corner_lo[1], self.corner_hi[1])
        mz = _axis_mask(z, self.corner_lo[2], self.corner_hi[2])
        # Boolean result is dtype-safe under any x64 setting; return a JAX
        # array so callers see the same type as before the prototype.
        return jnp.asarray(mx[:, None, None] & my[None, :, None] & mz[None, None, :])

    def mask(self, grid: Grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)


@dataclass(frozen=True)
class Cylinder:
    """Cylinder along a given axis."""

    center: tuple[float, float, float]
    radius: float
    height: float
    axis: str = "z"  # "x", "y", or "z"

    def bounding_box(self):
        r = self.radius
        h = self.height / 2
        cx, cy, cz = self.center
        if self.axis == "z":
            return ((cx - r, cy - r, cz - h), (cx + r, cy + r, cz + h))
        elif self.axis == "y":
            return ((cx - r, cy - h, cz - r), (cx + r, cy + h, cz + r))
        else:
            return ((cx - h, cy - r, cz - r), (cx + h, cy + r, cz + r))

    def mask_on_coords(self, x, y, z):
        xc = x - self.center[0]
        yc = y - self.center[1]
        zc = z - self.center[2]

        x3 = xc[:, None, None]
        y3 = yc[None, :, None]
        z3 = zc[None, None, :]

        if self.axis == "z":
            r2 = x3**2 + y3**2
            h = z3
        elif self.axis == "y":
            r2 = x3**2 + z3**2
            h = y3
        else:
            r2 = y3**2 + z3**2
            h = x3

        # PROTOTYPE (#802): ``abs`` dispatches to numpy on concrete float64
        # coordinates and to jnp on tracers; ``jnp.abs`` would downcast the
        # float64 nodes to float32 under the default config.
        # PROTOTYPE (#802b): closed tests ``r <= R`` and ``|h| <= H/2`` get
        # the same node-snap tolerance as Box on concrete coordinates.
        tol = _snap_tol(x, y, z)
        r_ok = r2 <= (self.radius + tol) ** 2
        h_ok = abs(h) <= self.height / 2 + tol
        return jnp.asarray(r_ok & h_ok)

    def mask(self, grid: Grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)


@dataclass(frozen=True)
class Sphere:
    """Sphere defined by center and radius."""

    center: tuple[float, float, float]
    radius: float

    def bounding_box(self):
        r = self.radius
        cx, cy, cz = self.center
        return ((cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r))

    def mask_on_coords(self, x, y, z):
        xc = x - self.center[0]
        yc = y - self.center[1]
        zc = z - self.center[2]
        r2 = xc[:, None, None]**2 + yc[None, :, None]**2 + zc[None, None, :]**2
        # PROTOTYPE (#802b): same node-snap tolerance as Box / Cylinder.
        return jnp.asarray(r2 <= (self.radius + _snap_tol(x, y, z)) ** 2)

    def mask(self, grid: Grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)


@dataclass(frozen=True)
class PolylineWire:
    """Wire defined by a polyline path with constant circular cross-section.

    Voxelizes by computing the distance from each grid point to the nearest
    line segment of the polyline.  Grid points within ``radius`` of any
    segment are marked as inside.

    Parameters
    ----------
    points : tuple of tuple[float, float, float]
        Ordered vertices in metres, e.g. ((x0,y0,z0), (x1,y1,z1), ...).
    radius : float
        Wire radius in metres.
    """

    points: tuple[tuple[float, float, float], ...]
    radius: float

    def bounding_box(self):
        pts = np.array(self.points)
        lo = tuple(float(v) for v in pts.min(axis=0) - self.radius)
        hi = tuple(float(v) for v in pts.max(axis=0) + self.radius)
        return (lo, hi)

    def mask_on_coords(self, x, y, z):
        r2_thresh = self.radius ** 2
        pts = np.array(self.points, dtype=np.float64)

        # Filter out degenerate (zero-length) segments
        A = pts[:-1]  # (n_seg, 3)
        B = pts[1:]   # (n_seg, 3)
        D = B - A     # direction vectors
        seg_len2 = np.sum(D ** 2, axis=1)  # (n_seg,)
        valid = seg_len2 > 1e-30
        A = A[valid]
        D = D[valid]
        seg_len2 = seg_len2[valid]

        if len(A) == 0:
            return jnp.zeros((len(x), len(y), len(z)), dtype=jnp.bool_)

        # Pre-compute segment data as JAX arrays: (n_valid, 3) and (n_valid,)
        A_j = jnp.array(A, dtype=jnp.float32)
        D_j = jnp.array(D, dtype=jnp.float32)
        seg_len2_j = jnp.array(seg_len2, dtype=jnp.float32)

        # 3D coordinate grids (Nx, Ny, Nz)
        X = x[:, None, None]
        Y = y[None, :, None]
        Z = z[None, None, :]

        # Use lax.scan to accumulate boolean mask without materializing
        # all n_seg intermediate distance arrays simultaneously.
        def _scan_step(mask_acc, seg_idx):
            ax, ay, az = A_j[seg_idx, 0], A_j[seg_idx, 1], A_j[seg_idx, 2]
            dx_s, dy_s, dz_s = D_j[seg_idx, 0], D_j[seg_idx, 1], D_j[seg_idx, 2]
            sl2 = seg_len2_j[seg_idx]

            t = ((X - ax) * dx_s + (Y - ay) * dy_s + (Z - az) * dz_s) / sl2
            t = jnp.clip(t, 0.0, 1.0)

            cx = ax + t * dx_s
            cy = ay + t * dy_s
            cz = az + t * dz_s

            dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
            return mask_acc | (dist2 <= r2_thresh), None

        init_mask = jnp.zeros((len(x), len(y), len(z)), dtype=jnp.bool_)
        mask, _ = jax.lax.scan(
            _scan_step, init_mask, jnp.arange(len(A_j)))

        return mask

    def mask(self, grid: Grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)


def union(a: Shape, b: Shape, grid: Grid) -> jnp.ndarray:
    return a.mask(grid) | b.mask(grid)


def difference(a: Shape, b: Shape, grid: Grid) -> jnp.ndarray:
    return a.mask(grid) & ~b.mask(grid)


def intersection(a: Shape, b: Shape, grid: Grid) -> jnp.ndarray:
    return a.mask(grid) & b.mask(grid)


def rasterize(
    grid: Grid,
    shapes: list[tuple[Shape, float, float]],
    background_eps: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rasterize shapes onto grid, producing (eps_r, sigma) arrays.

    Shapes are sampled on **node** coordinates. :class:`Box` uses a
    half-open ``[lo, hi)`` volume rule, so a box drawn between two node
    planes realizes one cell short of its drawn extent, asymmetrically at
    the ``hi`` face. For a PEC obstacle that is an electrical dimension
    error — a nominal opening ``d`` between two facing boxes rasterizes to
    ``d + dx``. Read the :class:`Box` docstring before drawing an obstacle
    to a nominal physical size, and assert the realized footprint (this
    function's ``sigma`` output is what the raster asserts in
    ``validation/crossval/18_wr90_iris_modematch.py`` inspect).

    Parameters
    ----------
    grid : Grid
    shapes : list of (Shape, eps_r, sigma) tuples
        Applied in order; later shapes overwrite earlier ones.
    background_eps : float
        Background relative permittivity.

    Returns
    -------
    eps_r, sigma : jnp.ndarray
    """
    eps_r = jnp.full(grid.shape, background_eps, dtype=jnp.float32)
    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)

    for shape, er, sig in shapes:
        m = shape.mask(grid)
        eps_r = jnp.where(m, er, eps_r)
        sigma = jnp.where(m, sig, sigma)

    return eps_r, sigma

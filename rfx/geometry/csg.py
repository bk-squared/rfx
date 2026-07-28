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
    """Extract 1D physical coordinate arrays from a uniform Grid."""
    nx, ny, nz = grid.shape
    dx = grid.dx
    pad_x, pad_y, pad_z = grid.axis_pads
    x = (jnp.arange(nx) - pad_x) * dx
    y = (jnp.arange(ny) - pad_y) * dx
    z = (jnp.arange(nz) - pad_z) * dx
    return x, y, z


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
       drawn extent, and the shortfall is entirely at the ``hi`` face — the
       footprint is **asymmetric**, so the object is also displaced by
       ``dx/2`` toward ``lo``.

    2. **A corner exactly on a node plane is a knife edge.** Masks are
       evaluated in the default float32 precision, where one ULP of the
       corner value is ~5e-10 m at ``dx`` = 0.762 mm (6e-7 of a cell).
       Perturbing ``hi`` by that single ULP moves the occupied-node count by
       a whole cell, so a corner computed as ``a - n*dx`` may rasterize
       differently from the algebraically identical ``m*dx``.

    For a **PEC obstacle** the occupied nodes are where the tangential ``E``
    is zeroed, so consequence (1) is an *electrical* dimension error, not a
    sub-cell cosmetic one. Two facing fins drawn to leave a nominal opening
    ``d`` leave an electrical opening of ``d + dx`` (measured, WR-90 iris at
    a/30 and a/60); shifting each interior face half a cell the *wrong* way
    — a natural attempt at (2) — gives ``d + 2*dx``. In PR #480's WR-90
    single-iris lane that inflated the ``|S11|`` error against an analytic
    mode-matching oracle by 4-6x, and because it scales with ``dx`` it is
    easy to misread as first-order convergence. For a **resonant** structure
    (multi-iris filter, cavity) it shifts the passband rather than widening
    a magnitude tolerance, so it will not look like a discretization error
    at all.

    **Recipe.** To make node ``j`` the innermost occupied plane, put the
    corner at the cell midpoint ``(j + 0.5) * dx`` rather than on a node
    plane: any value in ``(j*dx, (j+1)*dx]`` selects the same footprint, and
    the midpoint is the farthest point from both knife edges. Then assert
    the realized footprint (count the occupied node planes) against the
    intended one — see ``run_point`` in
    ``validation/crossval/18_wr90_iris_modematch.py`` for the pattern.

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
            coords = jnp.asarray(coords)
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
                k_mid = jnp.clip(
                    jnp.searchsorted(coords, mid) - 1, 0, coords.size - 2)
                im1 = jnp.clip(k_mid - 1, 0, coords.size - 1)
                ip1 = jnp.clip(k_mid + 1, 0, coords.size - 1)
                s_left = coords[k_mid] - coords[im1]    # 0 at the lo end
                s_right = coords[ip1] - coords[k_mid]   # 0 at the hi end
                dc_local = jnp.where(
                    (s_left > 0) & (s_right > 0),
                    jnp.minimum(s_left, s_right),
                    jnp.maximum(s_left, s_right))
            # Thin sheet: the single cell whose centre is nearest ``mid``.
            # (#371) On the collocated scheme, apply_pec_mask zeros tangential
            # Ex/Ey at this cell's CENTRE, so nearest-centre = minimum realized-
            # plane placement error. A matching-thickness (sub-cell) box takes
            # this same thin branch and agrees; only a >=1-cell VOLUME box
            # (different object) selects a different layer on a graded axis.
            nearest_idx = jnp.argmin(jnp.abs(coords - mid))
            thin_mask = jnp.zeros(coords.shape, dtype=bool).at[
                nearest_idx].set(True)
            # Volume: half-open [lo, hi).
            volume_mask = (coords >= lo) & (coords < hi)
            # Thin sheet when the extent is within one local cell.
            is_thin = extent <= dc_local * 1.01
            return jnp.where(is_thin, thin_mask, volume_mask)

        mx = _axis_mask(x, self.corner_lo[0], self.corner_hi[0])
        my = _axis_mask(y, self.corner_lo[1], self.corner_hi[1])
        mz = _axis_mask(z, self.corner_lo[2], self.corner_hi[2])
        return mx[:, None, None] & my[None, :, None] & mz[None, None, :]

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

        return (r2 <= self.radius**2) & (jnp.abs(h) <= self.height / 2)

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
        return r2 <= self.radius**2

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

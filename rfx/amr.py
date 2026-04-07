"""Static pre-AMR error indicator for adaptive mesh refinement.

Provides utilities to identify regions of high field gradient from a
coarse simulation result, suggest refinement regions as ``Box`` objects,
and automatically add refinement to a ``Simulation`` for a fine re-run.

Workflow::

    result = sim.run(n_steps=500)
    error  = compute_error_indicator(result, component="ez")
    boxes  = suggest_refinement_regions(error, grid=sim._build_grid(), threshold=0.5)
    # boxes are Box objects suitable for geometry-level reasoning

The error indicator is *static* (post-hoc on a completed run) rather than
inline, so it works with any backend (PEC, CPML, subgridding).
"""

from __future__ import annotations

import numpy as np

from rfx.geometry.csg import Box


# ---------------------------------------------------------------------------
# Error indicator
# ---------------------------------------------------------------------------

def compute_error_indicator(result, *, component: str = "ez") -> np.ndarray:
    """Compute spatial error indicator from a simulation result.

    Returns a 3-D array where high values indicate regions that would
    benefit from mesh refinement.  The proxy is the magnitude of the
    spatial gradient of the final-state field component, normalized to
    [0, 1].

    Parameters
    ----------
    result : Result or SimResult
        A completed simulation result whose ``.state`` carries the final
        field arrays (``ex``, ``ey``, ``ez``, ``hx``, ``hy``, ``hz``).
    component : str
        Field component to analyze (default ``"ez"``).

    Returns
    -------
    error : ndarray, shape (Nx, Ny, Nz)
        Normalized error indicator in [0, 1].  Values near 1 mark cells
        with the steepest field gradients.
    """
    state = result.state
    field = np.asarray(getattr(state, component))

    # Compute |grad(field)| via central differences along each axis.
    # For axes of length 1 (2-D mode) the gradient is zero.
    grad_sq = np.zeros_like(field, dtype=np.float64)

    for axis in range(field.ndim):
        if field.shape[axis] > 1:
            g = np.gradient(field, axis=axis)
            grad_sq += g ** 2

    grad_mag = np.sqrt(grad_sq)

    # Normalize to [0, 1]
    vmax = float(np.max(grad_mag))
    if vmax > 0:
        grad_mag = grad_mag / vmax

    return grad_mag.astype(np.float64)


# ---------------------------------------------------------------------------
# Region suggestion
# ---------------------------------------------------------------------------

def suggest_refinement_regions(
    error_map: np.ndarray,
    *,
    grid=None,
    threshold: float = 0.5,
    min_region_size: int = 4,
) -> list[Box]:
    """Suggest ``Box`` regions for SBP-SAT refinement based on error map.

    Identifies connected regions where the error indicator exceeds
    *threshold* and returns axis-aligned bounding boxes for each region
    that is at least *min_region_size* cells in every used dimension.

    Parameters
    ----------
    error_map : ndarray, shape (Nx, Ny, Nz)
        Normalized error indicator (output of ``compute_error_indicator``).
    grid : Grid or None
        If provided, box corners are in physical (metre) coordinates.
        Otherwise they are in cell-index coordinates.
    threshold : float
        Cells with ``error_map > threshold`` are flagged (default 0.5).
    min_region_size : int
        Minimum bounding-box extent in cells per axis to keep a region
        (default 4).  Smaller regions are discarded as noise.

    Returns
    -------
    list of Box
        Suggested refinement bounding boxes.
    """
    mask = error_map > threshold

    if not np.any(mask):
        return []

    # Label connected components via simple flood-fill using scipy if
    # available, otherwise fall back to single bounding-box over all
    # flagged cells.
    try:
        from scipy.ndimage import label as _label
        labeled, n_regions = _label(mask)
    except ImportError:
        # Fallback: treat all flagged cells as one region
        labeled = mask.astype(np.int32)
        n_regions = 1

    boxes: list[Box] = []
    for region_id in range(1, n_regions + 1):
        coords = np.argwhere(labeled == region_id)
        if len(coords) == 0:
            continue

        lo = coords.min(axis=0)
        hi = coords.max(axis=0)

        # Extent check — skip tiny regions (noise)
        extents = hi - lo + 1
        # Only check axes that are not singleton (2-D safe)
        used_axes = [i for i in range(error_map.ndim) if error_map.shape[i] > 1]
        if any(extents[ax] < min_region_size for ax in used_axes):
            continue

        if grid is not None:
            dx = grid.dx
            pad_x, pad_y, pad_z = grid.axis_pads
            corner_lo = (
                float((lo[0] - pad_x) * dx),
                float((lo[1] - pad_y) * dx),
                float((lo[2] - pad_z) * dx),
            )
            corner_hi = (
                float((hi[0] - pad_x) * dx),
                float((hi[1] - pad_y) * dx),
                float((hi[2] - pad_z) * dx),
            )
        else:
            corner_lo = (float(lo[0]), float(lo[1]), float(lo[2]))
            corner_hi = (float(hi[0]), float(hi[1]), float(hi[2]))

        boxes.append(Box(corner_lo=corner_lo, corner_hi=corner_hi))

    return boxes


# ---------------------------------------------------------------------------
# Auto-refine helper
# ---------------------------------------------------------------------------

def auto_refine(sim, result, *, threshold: float = 0.5, component: str = "ez",
                min_region_size: int = 2, ratio: int = 2):
    """Automatically add a refinement region based on the error indicator.

    Computes the error indicator from *result*, finds the dominant high-
    error region, and calls ``sim.add_refinement()`` with the
    corresponding z-range.

    Parameters
    ----------
    sim : Simulation
        The simulation to add refinement to (mutated in-place).
    result : Result
        Completed coarse simulation result.
    threshold : float
        Error threshold for region suggestion (default 0.5).
    component : str
        Field component for the error indicator (default ``"ez"``).
    min_region_size : int
        Minimum cells per axis for a suggested region (default 2).
    ratio : int
        Subgrid refinement ratio passed to ``add_refinement()`` (default 2).

    Returns
    -------
    sim : Simulation
        The same simulation object (for chaining).
    """
    error_map = compute_error_indicator(result, component=component)
    grid = result.grid if hasattr(result, "grid") and result.grid is not None else sim._build_grid()
    boxes = suggest_refinement_regions(
        error_map, grid=grid, threshold=threshold, min_region_size=min_region_size,
    )

    if not boxes:
        return sim

    # Pick the largest region (by volume) and extract its z-range.
    def _box_volume(b):
        return (
            abs(b.corner_hi[0] - b.corner_lo[0])
            * abs(b.corner_hi[1] - b.corner_lo[1])
            * abs(b.corner_hi[2] - b.corner_lo[2])
        )

    best = max(boxes, key=_box_volume)
    z_lo = best.corner_lo[2]
    z_hi = best.corner_hi[2]

    # Guard: ensure z_range has nonzero extent
    if z_hi <= z_lo:
        z_hi = z_lo + grid.dx

    sim.add_refinement((z_lo, z_hi), ratio=ratio)
    return sim

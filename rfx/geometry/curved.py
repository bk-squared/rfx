"""Curved surface primitives for conformal antenna geometries.

Uses staircase approximation: the curved surface is decomposed
into thin rectangular slices along the curvature axis.
"""

from __future__ import annotations

import numpy as np

from rfx.geometry.csg import Box


class CurvedPatch:
    """A rectangular patch curved along one axis with a given radius.

    The patch is approximated as a series of flat Box segments
    (staircase approximation). The number of segments depends on
    the grid resolution dx.

    Parameters
    ----------
    center : tuple (x, y, z)
        Center of the patch in 3D space.
    length : float
        Patch length along the curvature axis.
    width : float
        Patch width perpendicular to curvature.
    radius : float
        Curvature radius (larger = flatter).
    axis : str
        Axis along which the patch is curved ("x" or "y").
    """

    def __init__(self, *, center, length, width, radius, axis="x"):
        self.center = center
        self.length = length
        self.width = width
        self.radius = radius
        self.axis = axis

    def to_staircase(self, dx: float) -> list[Box]:
        """Decompose curved patch into staircase boxes.

        Parameters
        ----------
        dx : float
            Grid cell size. Each box is approximately one cell wide
            along the curvature axis.

        Returns
        -------
        list of Box
            Flat box segments approximating the curved surface.
        """
        n_segments = max(2, int(np.ceil(self.length / dx)))
        boxes = []
        cx, cy, cz = self.center
        half_len = self.length / 2
        seg_width = self.length / n_segments

        for i in range(n_segments):
            # Position along curvature axis relative to center
            s = -half_len + (i + 0.5) * seg_width
            # Arc height: z_offset = R - sqrt(R^2 - s^2)
            s_clamped = min(abs(s), self.radius * 0.99)  # Prevent sqrt of negative
            z_offset = self.radius - np.sqrt(self.radius**2 - s_clamped**2)

            if self.axis == "x":
                box_center_x = cx + s
                box_center_y = cy
                box_center_z = cz + z_offset
                corner_lo = (
                    box_center_x - seg_width / 2,
                    box_center_y - self.width / 2,
                    box_center_z,
                )
                corner_hi = (
                    box_center_x + seg_width / 2,
                    box_center_y + self.width / 2,
                    box_center_z,
                )
            elif self.axis == "y":
                box_center_x = cx
                box_center_y = cy + s
                box_center_z = cz + z_offset
                corner_lo = (
                    box_center_x - self.width / 2,
                    box_center_y - seg_width / 2,
                    box_center_z,
                )
                corner_hi = (
                    box_center_x + self.width / 2,
                    box_center_y + seg_width / 2,
                    box_center_z,
                )
            else:
                raise ValueError(f"axis must be 'x' or 'y', got '{self.axis}'")

            boxes.append(Box(corner_lo=corner_lo, corner_hi=corner_hi))

        return boxes

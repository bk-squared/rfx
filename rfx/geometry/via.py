"""Via / through-hole geometry helper.

Composes a vertical conductor and copper pads on each layer boundary
using existing Box primitives.  Intended for multi-layer PCB simulations
where via structures connect copper layers.

In FDTD the distinction between a circular drill and a square cell-aligned
box is negligible when the drill diameter is on the order of a few cells,
so this helper uses axis-aligned Box shapes throughout.
"""

from __future__ import annotations

import jax.numpy as jnp

from rfx.geometry.csg import Box


class Via:
    """Via through-hole for multi-layer PCB structures.

    Parameters
    ----------
    center : tuple (x, y)
        Position in the xy-plane (meters).
    drill_radius : float
        Drill hole radius (meters).  Determines the width of the
        vertical conductor.
    pad_radius : float
        Copper pad radius (meters) on each layer boundary.
    layers : list of (z_bottom, z_top)
        PCB layer boundaries from bottom to top (meters).
        Each tuple defines one dielectric layer.
    material : str, optional
        Material name assigned to all generated shapes.
        Defaults to ``"pec"``.
    """

    def __init__(
        self,
        *,
        center: tuple[float, float],
        drill_radius: float,
        pad_radius: float,
        layers: list[tuple[float, float]],
        material: str = "pec",
    ):
        if drill_radius <= 0:
            raise ValueError(f"drill_radius must be positive, got {drill_radius}")
        if pad_radius < drill_radius:
            raise ValueError(
                f"pad_radius ({pad_radius}) must be >= drill_radius ({drill_radius})"
            )
        if not layers:
            raise ValueError("layers must be non-empty")

        self.center = center
        self.drill_radius = drill_radius
        self.pad_radius = pad_radius
        self.layers = layers
        self.material = material

    def bounding_box(self):
        x, y = self.center
        r = self.pad_radius
        z_min = min(z for z, _ in self.layers)
        z_max = max(z for _, z in self.layers)
        return ((x - r, y - r, z_min), (x + r, y + r, z_max))

    def mask_on_coords(self, x, y, z):
        """Evaluate via occupancy — union of decomposed Box shapes."""
        result = jnp.zeros((len(x), len(y), len(z)), dtype=jnp.bool_)
        for box, _ in self.to_shapes():
            result = result | box.mask_on_coords(x, y, z)
        return result

    def mask(self, grid):
        from rfx.geometry.csg import _grid_coords
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)

    def to_shapes(self) -> list[tuple[Box, str]]:
        """Return a list of ``(Box, material_name)`` tuples.

        Creates:
        - One vertical conductor (Box) spanning from the lowest layer
          bottom to the highest layer top.
        - One pad (thin Box) at each unique layer boundary z-coordinate.

        Pads at shared boundaries (e.g. z_top of layer 0 == z_bot of
        layer 1) are emitted only once.
        """
        x, y = self.center
        z_min = min(z for z, _ in self.layers)
        z_max = max(z for _, z in self.layers)

        shapes: list[tuple[Box, str]] = []

        # --- vertical conductor through all layers ---
        half_drill = self.drill_radius
        shapes.append((
            Box(
                corner_lo=(x - half_drill, y - half_drill, z_min),
                corner_hi=(x + half_drill, y + half_drill, z_max),
            ),
            self.material,
        ))

        # --- pads at each unique layer boundary ---
        boundary_zs: set[float] = set()
        for z_bot, z_top in self.layers:
            boundary_zs.add(z_bot)
            boundary_zs.add(z_top)

        half_pad = self.pad_radius
        for z in sorted(boundary_zs):
            shapes.append((
                Box(
                    corner_lo=(x - half_pad, y - half_pad, z),
                    corner_hi=(x + half_pad, y + half_pad, z),
                ),
                self.material,
            ))

        return shapes

    def to_simulation_items(self) -> list[tuple[Box, str]]:
        """Alias for :meth:`to_shapes` -- ready for ``Simulation.add()``."""
        return self.to_shapes()

    def __repr__(self) -> str:
        return (
            f"Via(center={self.center}, drill_radius={self.drill_radius}, "
            f"pad_radius={self.pad_radius}, layers={self.layers}, "
            f"material={self.material!r})"
        )

"""PCB stackup builder for multi-layer board simulation.

Constructs layer-by-layer PCB geometry as a sequence of :class:`Box` shapes
with associated material names, ready for ``Simulation.add()``.

Example
-------
>>> stackup = Stackup([
...     PCBLayer(thickness=0.035e-3, material="copper", name="top"),
...     PCBLayer(thickness=0.2e-3, material="prepreg"),
...     PCBLayer(thickness=0.035e-3, material="copper", name="inner1"),
...     PCBLayer(thickness=1.0e-3, material="fr4", name="core"),
...     PCBLayer(thickness=0.035e-3, material="copper", name="inner2"),
...     PCBLayer(thickness=0.2e-3, material="prepreg"),
...     PCBLayer(thickness=0.035e-3, material="copper", name="bottom"),
... ])
>>>
>>> shapes = stackup.to_shapes(center_xy=(0, 0), size_xy=(0.02, 0.02))
>>> for shape, material in shapes:
...     sim.add(shape, material=material)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rfx.geometry.csg import Box


# ---------------------------------------------------------------------------
# Material aliases for common PCB dielectrics not in MATERIAL_LIBRARY
# ---------------------------------------------------------------------------

PCB_MATERIAL_ALIASES: dict[str, str] = {
    # "prepreg" is typically similar to FR-4 in FDTD terms.  Users who need
    # precise Dk/Df should register a custom material on the Simulation
    # object; the alias simply lets the stackup builder pass a valid
    # library name downstream.
    "prepreg": "fr4",
}


def resolve_pcb_material(material: str) -> str:
    """Return the MATERIAL_LIBRARY-compatible name for a PCB material.

    If *material* is a known alias (e.g. ``"prepreg"``), the canonical
    library name is returned.  Otherwise *material* is returned as-is,
    assuming it is either a library material or will be registered by the
    user before simulation.
    """
    return PCB_MATERIAL_ALIASES.get(material, material)


# ---------------------------------------------------------------------------
# PCBLayer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PCBLayer:
    """Single PCB layer (copper or dielectric).

    Parameters
    ----------
    thickness : float
        Layer thickness in metres.
    material : str
        Material identifier — must be a key in ``MATERIAL_LIBRARY``
        (e.g. ``"copper"``, ``"fr4"``, ``"rogers4003c"``) or a known
        alias (``"prepreg"``).  Users may also register custom materials
        on the :class:`Simulation` object.
    name : str or None
        Optional human-readable label (e.g. ``"top"``, ``"core"``).
        Named layers can be looked up with :meth:`Stackup.get_layer_z`.
    """

    thickness: float
    material: str = "copper"
    name: str | None = None


# ---------------------------------------------------------------------------
# Stackup
# ---------------------------------------------------------------------------

@dataclass
class Stackup:
    """PCB stackup builder.

    Layers are specified bottom-to-top (index 0 is the lowest z layer).
    The stackup is centred at z = 0 so that
    ``z_bottom = -total_thickness / 2`` and
    ``z_top = +total_thickness / 2``.

    Parameters
    ----------
    layers : sequence of :class:`PCBLayer`
        Ordered from bottom to top.
    """

    layers: list[PCBLayer] = field(default_factory=list)

    def __init__(self, layers: Sequence[PCBLayer]):
        self.layers = list(layers)

    # -- properties ---------------------------------------------------------

    @property
    def total_thickness(self) -> float:
        """Sum of all layer thicknesses (metres)."""
        return sum(layer.thickness for layer in self.layers)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    # -- query helpers ------------------------------------------------------

    def get_layer_z(self, name: str) -> tuple[float, float]:
        """Return ``(z_lo, z_hi)`` of the named layer.

        The stackup is centred at z = 0.

        Raises
        ------
        KeyError
            If no layer with *name* exists.
        """
        z = -self.total_thickness / 2.0
        for layer in self.layers:
            z_lo = z
            z_hi = z + layer.thickness
            if layer.name == name:
                return (z_lo, z_hi)
            z = z_hi
        raise KeyError(f"No layer named {name!r} in stackup")

    # -- geometry generation ------------------------------------------------

    def to_shapes(
        self,
        center_xy: tuple[float, float] = (0.0, 0.0),
        size_xy: tuple[float, float] = (0.02, 0.02),
    ) -> list[tuple[Box, str]]:
        """Generate :class:`Box` shapes for each layer.

        Parameters
        ----------
        center_xy : (cx, cy)
            Centre of the board in the xy-plane (metres).
        size_xy : (sx, sy)
            Full extent of the board in x and y (metres).

        Returns
        -------
        list of (Box, material_name) tuples
            Each tuple contains the layer geometry and the resolved
            material name suitable for ``sim.add(shape, material=...)``.
        """
        cx, cy = center_xy
        sx, sy = size_xy
        x_lo = cx - sx / 2.0
        x_hi = cx + sx / 2.0
        y_lo = cy - sy / 2.0
        y_hi = cy + sy / 2.0

        shapes: list[tuple[Box, str]] = []
        z = -self.total_thickness / 2.0

        for layer in self.layers:
            z_lo = z
            z_hi = z + layer.thickness
            box = Box(
                corner_lo=(x_lo, y_lo, z_lo),
                corner_hi=(x_hi, y_hi, z_hi),
            )
            mat = resolve_pcb_material(layer.material)
            shapes.append((box, mat))
            z = z_hi

        return shapes

    # -- convenience constructors -------------------------------------------

    @classmethod
    def standard_2layer(
        cls,
        substrate_thickness: float = 1.6e-3,
        substrate_material: str = "fr4",
        copper_thickness: float = 0.035e-3,
    ) -> Stackup:
        """Standard 2-layer PCB (top copper / substrate / bottom copper).

        Parameters
        ----------
        substrate_thickness : float
            Dielectric core thickness (default 1.6 mm).
        substrate_material : str
            Dielectric material name (default ``"fr4"``).
        copper_thickness : float
            Copper foil thickness (default 35 um / 1 oz).
        """
        return cls([
            PCBLayer(thickness=copper_thickness, material="copper", name="bottom"),
            PCBLayer(thickness=substrate_thickness, material=substrate_material, name="substrate"),
            PCBLayer(thickness=copper_thickness, material="copper", name="top"),
        ])

    @classmethod
    def standard_4layer(
        cls,
        core_thickness: float = 1.0e-3,
        prepreg_thickness: float = 0.2e-3,
        copper_thickness: float = 0.035e-3,
    ) -> Stackup:
        """Standard 4-layer PCB stackup.

        Layer order (bottom to top)::

            bottom copper | prepreg | inner2 copper | core (FR-4) |
            inner1 copper | prepreg | top copper

        Parameters
        ----------
        core_thickness : float
            Central dielectric core thickness (default 1.0 mm).
        prepreg_thickness : float
            Pre-impregnated dielectric thickness (default 0.2 mm).
        copper_thickness : float
            Copper foil thickness for all layers (default 35 um / 1 oz).
        """
        return cls([
            PCBLayer(thickness=copper_thickness, material="copper", name="bottom"),
            PCBLayer(thickness=prepreg_thickness, material="prepreg", name="prepreg_bottom"),
            PCBLayer(thickness=copper_thickness, material="copper", name="inner2"),
            PCBLayer(thickness=core_thickness, material="fr4", name="core"),
            PCBLayer(thickness=copper_thickness, material="copper", name="inner1"),
            PCBLayer(thickness=prepreg_thickness, material="prepreg", name="prepreg_top"),
            PCBLayer(thickness=copper_thickness, material="copper", name="top"),
        ])

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [f"Stackup({self.num_layers} layers, total={self.total_thickness*1e3:.3f} mm)"]
        z = -self.total_thickness / 2.0
        for layer in self.layers:
            label = layer.name or layer.material
            lines.append(
                f"  z={z*1e3:+8.3f} mm | {label:12s} | "
                f"{layer.thickness*1e3:.3f} mm ({layer.material})"
            )
            z += layer.thickness
        return "\n".join(lines)

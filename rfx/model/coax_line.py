"""Build one stamped coax section with independently driven line ends."""

from dataclasses import dataclass
from typing import Any, NamedTuple

from rfx.model.coax_stamp import (
    build_coaxial_tem_plane_source_specs,
    stamp_coaxial_annular_resistor,
    stamp_coaxial_line,
)


@dataclass(frozen=True)
class CoaxLineEnd:
    """A calculator's realized feed, source descriptor and voltage ladder.

    source_index is the actual TEM plane, including the descriptor's historical
    half-cell rounding. The descriptor carries the face and waveform without
    requiring a dependency on the calculator or source descriptor class.
    """

    feed_index: int
    source_index: int
    plane_indices: tuple[int, ...]
    feed_impedance: float
    port: Any


class CoaxLineEndResult(NamedTuple):
    """One end's source, measurement indices and realized TEM impedance."""

    source_spec: Any
    plane_indices: tuple[int, ...]
    z_tem_ohm: float


class CoaxLineResult(NamedTuple):
    """Stamped section and conductor entities before kernel realization."""

    materials: Any
    pec_cells: Any
    conductor_entities: list
    shell_inner_radius: float
    ends: tuple[CoaxLineEndResult, ...]


def build_coax_line(
    grid, materials, *, center_xy, pin_radius, outer_radius,
    z_lo_index, z_hi_index, ends, n_steps, field_scale,
    finish_line=None, stamp_line=stamp_coaxial_line,
    stamp_resistor=stamp_coaxial_annular_resistor,
    build_source=build_coaxial_tem_plane_source_specs,
):
    """Stamp a common line, each feed, then each end's TEM source.

    finish_line(materials, shell_inner_radius, pec_cells) returns materials
    and PEC cells after calculator-owned DUT termination or junction splicing.
    It runs after all feed stamps and before source construction, preserving
    the source's view of the complete conductor mask. Optional operations
    retain the calculators' established instrumentation seams; model callers
    use the model-layer defaults. Axial placement remains caller-owned.
    """
    ends = tuple(ends)
    entities = []
    materials, shell_inner, pec_cells = stamp_line(
        grid, materials, center_xy=center_xy, z_lo_index=z_lo_index,
        z_hi_index=z_hi_index, pin_radius=pin_radius, outer_radius=outer_radius,
        realized_entities=entities,
    )
    for end in ends:
        materials = stamp_resistor(
            grid, materials, center_xy=center_xy, z_index=end.feed_index,
            pin_radius=pin_radius, outer_radius=outer_radius,
            target_impedance=end.feed_impedance, shell_inner_radius=shell_inner,
            pec_cell_mask=pec_cells,
        )
    if finish_line is not None:
        materials, pec_cells = finish_line(materials, shell_inner, pec_cells)
    built_ends = []
    for end in ends:
        spec = build_source(
            grid=grid, port=end.port, n_steps=int(n_steps),
            plane_axial_index=end.source_index, field_scale=float(field_scale),
            magnetic_ratio=1.0, shell_inner_radius=shell_inner,
            pec_cell_mask=pec_cells,
        )
        built_ends.append(CoaxLineEndResult(spec, end.plane_indices, spec.z_tem_ohm))
    return CoaxLineResult(materials, pec_cells, entities, shell_inner, tuple(built_ends))

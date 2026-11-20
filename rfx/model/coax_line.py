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


@dataclass(frozen=True)
class CoaxPortLayout:
    """Inclusive conductor cells and realized node indices for one port."""

    z_lo_index: int
    z_hi_index: int
    feed_index: int
    source_index: int
    plane_indices: tuple[int, ...]
    reference_index: int


def coax_port_layout(grid, port, *, absorber_pads, ntff=None):
    """Lay out the uniform port behind its declared reference node.

    Pads are the six per-face cell counts in x-lo, x-hi, y-lo, y-hi,
    z-lo, z-hi order. A bottom line is a node mirror, including its
    cell ownership, of the corresponding top line.
    """
    import numpy as np
    from rfx.errors import RfxUnsupported
    from rfx.model.coax_stamp import SHELL_THICKNESS_M
    dx = float(grid.dx)
    room = []
    for axis, name in enumerate('xy'):
        low, high = absorber_pads[2 * axis:2 * axis + 2]
        span = (grid.shape[axis] - 1 - low - high) * dx
        room.extend(((f'{name}-lo', port.position[axis]), (f'{name}-hi', span - port.position[axis])))
    face, clearance = min(room, key=lambda item: item[1])
    available = clearance - port.outer_radius - dx
    if available < dx:
        raise RfxUnsupported(
            f'add_coaxial_port(): {face} absorber leaves {available / dx:.2f} wall cells '
            'after keeping 1 cell of vacuum; needs at least 1 wall cell')
    radius = port.outer_radius + min(SHELL_THICKNESS_M, available)
    coords = [(np.arange(grid.shape[a]) - absorber_pads[2 * a]) * dx - port.position[a] for a in (0, 1)]
    radial = coords[0][:, None] ** 2 + coords[1][None, :] ** 2
    shell = (radial <= radius ** 2) & (radial > port.outer_radius ** 2)
    indices = np.nonzero(shell)
    for axis, name in enumerate('xy'):
        low, high = absorber_pads[2 * axis:2 * axis + 2]
        if not indices[axis].size:
            raise RfxUnsupported('add_coaxial_port(): shell realizes 0 cells; needs at least 1 wall cell')
        first, last = int(indices[axis].min()), int(indices[axis].max()) + 1
        for face, gap in ((f'{name}-lo', first - low), (f'{name}-hi', grid.shape[axis] - 1 - high - last)):
            if gap < 1:
                raise RfxUnsupported(f'add_coaxial_port(): shell is {gap} cells from the {face} absorber; needs at least 1 cell of vacuum')
        if ntff is not None and ((first - low) * dx < ntff[0][axis] or (last - low) * dx > ntff[1][axis]):
            raise RfxUnsupported('add_coaxial_port(): NTFF box must contain the realized shell')
    ref = int(grid.position_to_index(port.position)[2])
    dz = float(grid.dx)
    direction = 1 if port.face == 'top' else -1
    far = (grid.shape[2] - int(absorber_pads[5]) - 2 if direction == 1
           else int(absorber_pads[4]) + 1)
    if ntff is not None:
        lo, hi = ntff[:2]
        bound = int(grid.position_to_index(hi if direction == 1 else lo)[2]) - 2 * direction
        far = min(far, bound) if direction == 1 else max(far, bound)
        for axis in (0, 1):
            if not (lo[axis] < port.position[axis] - port.outer_radius
                    and hi[axis] > port.position[axis] + port.outer_radius):
                raise RfxUnsupported('add_coaxial_port(): NTFF box must contain the shell')
        tip = port.position[2] - direction * port.pin_length
        if not lo[2] <= tip <= hi[2] or not lo[2] <= port.position[2] <= hi[2]:
            raise RfxUnsupported('add_coaxial_port(): NTFF box must contain the reference and pin')
    distance = direction * (far - ref)
    nominal = ref + distance - 3
    source = int(round(((nominal - grid.pad_z_lo) * dz - dz / 2)
                       / float(np.asarray(grid.cells(2))[0]))) + grid.pad_z_lo
    source = ref + direction * (source - ref)
    planes = tuple(ref + direction * (8 + 4 * k) for k in range(12))
    if distance < 60 or direction * (source - planes[-1]) < 4:
        raise RfxUnsupported(
            'add_coaxial_port(): domain too short; needs at least 61 line cells '
            '(12 planes, start 8, spacing 4, source clearance 4, feed setback 3)')
    return CoaxPortLayout(ref if direction == 1 else far - 1,
                          far if direction == 1 else ref - 1,
                          far - direction, source, planes, ref)


def build_coax_port_line(grid, materials, port, *, absorber_pads, ntff=None,
                         n_steps, finish_line=None):
    """Build the single uniform port with the calculator's feed and TEM drive."""
    from rfx.sources.coaxial_port import coaxial_tem_characteristic_impedance
    layout = coax_port_layout(grid, port, absorber_pads=absorber_pads, ntff=ntff)
    end = CoaxLineEnd(layout.feed_index, layout.source_index, layout.plane_indices,
                      float(coaxial_tem_characteristic_impedance(
                          port.pin_radius, port.outer_radius)), port)
    try:
        line = build_coax_line(grid, materials, center_xy=port.position[:2],
            pin_radius=port.pin_radius, outer_radius=port.outer_radius,
            z_lo_index=layout.z_lo_index, z_hi_index=layout.z_hi_index,
            ends=(end,), n_steps=n_steps, field_scale=1.0e4, finish_line=finish_line)
    except ValueError as error:
        if 'absorber' not in str(error):
            raise
        from rfx.errors import RfxUnsupported
        raise RfxUnsupported(f'add_coaxial_port(): {error}') from error
    if ntff is not None:
        import numpy as np
        from rfx.errors import RfxUnsupported
        indices = np.nonzero(line.pec_cells)
        for axis, name in enumerate('xyz'):
            pad = getattr(grid, f'pad_{name}_lo')
            lo = (indices[axis].min() - pad) * grid.dx
            hi = (indices[axis].max() + 1 - pad) * grid.dx
            if lo < ntff[0][axis] or hi > ntff[1][axis]:
                raise RfxUnsupported('add_coaxial_port(): NTFF box must contain the realized shell')
    return layout, line

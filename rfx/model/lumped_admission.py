"""Disjoint realized edges required by the batched RLC update."""

from rfx._grid_metric import field_index
from rfx.lumped import _describe_spec
from rfx.sources.sources import WirePort, _wire_port_cells


def refuse_rlc_overlap(specs, metas, ports, grid):
    """Name both declarations when RLC elements or ports share an edge."""
    occupied = {}
    for index, (spec, meta) in enumerate(zip(specs, metas, strict=True)):
        edge = (meta.component, (int(meta.i), int(meta.j), int(meta.k)))
        label = _describe_spec(index, spec)
        if edge in occupied:
            _refuse(occupied[edge], label, edge)
        occupied[edge] = label
    for index, port in enumerate(ports):
        if port.impedance == 0:
            continue  # A point source has no port termination.
        if port.extent is None:
            cells = (field_index(grid, port.position, port.component),)
            kind = "lumped port"
        else:
            end = list(port.position)
            end[("ex", "ey", "ez").index(port.component)] += port.extent
            cells = _wire_port_cells(grid, WirePort(
                start=port.position, end=tuple(end), component=port.component,
                impedance=port.impedance, excitation=port.waveform, radius=port.radius))
            kind = "wire port"
        for cell in cells:
            edge = (port.component, tuple(map(int, cell)))
            if edge in occupied:
                _refuse(occupied[edge], f"add_port #{index} ({kind})", edge)


def _refuse(first, second, edge):
    raise NotImplementedError(
        f"{first} and {second} both land on the {edge[0]} edge of cell {edge[1]}. "
        "The batched RLC update requires disjoint element and port edges; "
        "combine the elements in ONE add_lumped_rlc with the combined value "
        "or place them on different edges.")

"""Run-specific source coefficient selection and absorber admission."""
from dataclasses import replace


def edge_in_absorber_pad(grid, cell):
    """The exact index spans populated by UPML's axis sigma builder."""
    axes = getattr(grid, "cpml_axes", "xyz")
    return any(axis in axes and (
        cell[a] < getattr(grid, f"pad_{axis}_lo", 0) or
        cell[a] >= grid.shape[a] - getattr(grid, f"pad_{axis}_hi", 0))
        for a, axis in enumerate("xyz"))


def _refuse_pad_edge(grid, cell, label):
    if edge_in_absorber_pad(grid, cell):
        raise ValueError(
            f"{label} edge {tuple(cell)} lies inside a UPML absorber pad; "
            "its source coefficient does not include the absorber operator. "
            "Move the source/port edge into the interior.")


def resolve_run_sources(queue, materials, sim, grid, *, tensor=False):
    """Resolve after all stamps; preserve tensor branches pending their audit."""
    upml = sim._boundary == "upml" and grid.cpml_layers > 0
    if upml:
        from rfx.sources.sources import _wire_port_cells, wire_port_from_entry
        from rfx.sources.msl_port import _msl_yz_cells, msl_port_from_entry
        # Include passive ports and every declared wire edge, not only the
        # excited source specs. Modal fringe source edges are checked below.
        for n, port in enumerate(sim._ports):
            cells = (_wire_port_cells(grid, wire_port_from_entry(port))
                     if port.extent is not None else
                     [grid.position_to_index(port.position)])
            for cell in cells:
                _refuse_pad_edge(grid, cell, f"source/port[{n}]")
        for n, port in enumerate(sim._msl_ports):
            for cell in _msl_yz_cells(grid, msl_port_from_entry(port)):
                _refuse_pad_edge(grid, cell, f"msl_port[{n}]")
        for n, port in enumerate(sim._floquet_ports):
            position = [length / 2 for length in sim._domain]
            position["xyz".index(port.axis)] = port.position
            _refuse_pad_edge(grid, grid.position_to_index(tuple(position)),
                             f"floquet_port[{n}]")
        materials = materials._replace(components=replace(
            materials.components, source_upml=not tensor))
    sources = queue.resolve(materials)
    if upml:
        for source in sources:
            cell = (source.i, source.j, source.k)
            _refuse_pad_edge(grid, cell, f"{source.component} source")
    return sources


def tensor_replaced_edges(materials, *, aniso_eps=None, aniso_inv_eps=None, upml=False):
    """Audit predicate: edges whose tensor epsilon differs from the plain edge.

    This observes finished kernel operands, including the same final stamps.
    It does not refuse sources or change smoothing. Compare inverse values
    for Stage 2 so a frozen PEC edge (inverse=0) remains representable.
    """
    baseline = materials.components.upml_eps if upml else materials.components.eps_update
    if aniso_inv_eps is not None:
        return tuple(tensor != 1.0 / plain for tensor, plain in
                     zip(aniso_inv_eps, baseline))
    if aniso_eps is not None:
        return tuple(tensor != plain for tensor, plain in
                     zip(aniso_eps, baseline))
    return None

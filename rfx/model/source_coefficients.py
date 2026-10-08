"""Run-specific source coefficient selection and absorber admission."""
from dataclasses import replace


def edge_in_absorber_pad(grid, cell, component=None):
    """Whether the UPML update on this edge carries an absorber conductivity.

    The UPML E update of a component uses only the conductivities of the
    axes perpendicular to it, so an edge inside the pad of its own axis
    alone keeps the interior coefficient. ``component=None`` asks about any
    pad (a launch that drives more than one component).
    """
    axes = getattr(grid, "cpml_axes", "xyz")
    own = component[-1] if component else None
    return any(axis in axes and axis != own and (
        cell[a] < getattr(grid, f"pad_{axis}_lo", 0) or
        cell[a] >= grid.shape[a] - getattr(grid, f"pad_{axis}_hi", 0))
        for a, axis in enumerate("xyz"))


def _refuse_pad_edge(grid, cell, label, component=None):
    if edge_in_absorber_pad(grid, cell, component):
        raise ValueError(
            f"{label} edge {tuple(cell)} lies inside a UPML absorber pad; "
            "its source coefficient does not include the absorber operator. "
            "Move the source/port edge into the interior.")


def resolve_run_sources(queue, materials, sim, grid, *, tensor=False):
    """Resolve after all stamps; preserve tensor branches pending their audit."""
    upml = sim._boundary == "upml" and grid.cpml_layers > 0
    undriven = set()
    if upml:
        from rfx.sources.sources import _wire_port_cells, wire_port_from_entry
        from rfx.sources.msl_port import _msl_yz_cells, msl_port_from_entry
        # Only a current drive multiplies the update coefficient. A source
        # that prescribes the field increment and a port that is a load only
        # inject nothing through it and are admitted anywhere.
        for n, port in enumerate(sim._ports):
            cells = (_wire_port_cells(grid, wire_port_from_entry(port))
                     if port.extent is not None else
                     [grid.position_to_index(port.position)])
            prescribed = (port.impedance == 0.0
                          and port.amplitude_kind == "field")
            if prescribed or not getattr(port, "excite", True):
                undriven.update((tuple(int(c) for c in cell), port.component)
                                for cell in cells)
                continue
            for cell in cells:
                _refuse_pad_edge(grid, cell, f"source/port[{n}]", port.component)
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
            cell = (int(source.i), int(source.j), int(source.k))
            if (cell, source.component) not in undriven:
                _refuse_pad_edge(grid, cell, f"{source.component} source",
                                 source.component)
    return sources


def dispersive_drive_model(materials, debye_spec, lorentz_spec, grid=None):
    """Materials a drive reads its coefficient from on a lane that keeps no
    realization of its own (the distributed lanes). Nothing whole-domain is
    built: a dispersive model carries its pole specs, read edge by edge."""
    if debye_spec is None and lorentz_spec is None and grid is None:
        return materials
    from rfx.model.materials import EdgePoles
    return materials._replace(components=EdgePoles(debye_spec, lorentz_spec, grid))


def debye_pole_term(drive_model, cell, component, dt):
    """The Debye poles' part of one edge's ADE coefficient, as the
    conductivity that adds the same term: ``Cb = dt/(eps + sum(beta) +
    sigma*dt/2)``, so ``sum(beta)`` equals a conductivity ``2*sum(beta)/dt``.
    It does not depend on eps_inf or sigma, so it stays a host constant
    under a traced override."""
    from rfx.model.materials import EdgePoles
    c = getattr(drive_model, "components", None)
    debye = c.at(cell, dt)[0] if isinstance(c, EdgePoles) else None
    if debye is None:
        return 0.0
    axis = {"ex": 0, "ey": 1, "ez": 2}[str(component).lower()]
    return 2.0 * float(debye[1][axis].sum()) / float(dt)


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

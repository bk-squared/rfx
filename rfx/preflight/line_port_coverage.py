"""Grid-local drawing allowance for line-port conductor coverage."""


def local_port_cell(sim, position, axis="x"):
    """Cell size at the same node used by the port, on either grid lane."""
    from rfx.sources.msl_port import _axis_cell_size, _msl_position_to_index

    grid = sim._build_realized_grid()
    index = _msl_position_to_index(grid, position)["xyz".index(axis)]
    return _axis_cell_size(grid, axis, index)


def port_node_coordinate(sim, position, axis="x"):
    """Physical coordinate of the node used by this line port on this grid."""
    from rfx.sources.msl_port import _msl_grid_geometry, _msl_position_to_index

    grid = sim._build_realized_grid()
    a = "xyz".index(axis)
    nodes, _ = _msl_grid_geometry(grid)
    return float(nodes[a][_msl_position_to_index(grid, position)[a]])

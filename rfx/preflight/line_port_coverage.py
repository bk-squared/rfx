"""Grid-local drawing allowance for line-port conductor coverage."""


def local_port_cell(sim, position, axis="x"):
    """Cell size at the same node used by the port, on either grid lane."""
    from rfx.sources.msl_port import _axis_cell_size, _msl_position_to_index

    grid = sim._build_realized_grid()
    index = _msl_position_to_index(grid, position)["xyz".index(axis)]
    return _axis_cell_size(grid, axis, index)

"""Refuse flux windows that can see a calculator's open coax feed end."""
import numpy as np

from rfx.errors import RfxRefusal


def refuse_open_feed_flux(entries, grid, domain, name, *, low, high=None):
    """Judge every axial endpoint, including tangential finite windows."""
    for entry in entries or ():
        if entry.axis == 'z':
            lo = hi = int(grid.position_to_index((0., 0., entry.coordinate))[2])
        elif entry.size is None:
            lo, hi = grid.pad_z_lo, grid.shape[2] - 1 - grid.pad_z_hi
        else:
            center = domain[2] / 2 if entry.center is None else entry.center[1]
            extent = entry.size[1] / 2
            lo = int(np.floor((center - extent) / grid.dx)) + grid.pad_z_lo
            hi = int(np.ceil((center + extent) / grid.dx)) + grid.pad_z_lo
        side = 'bottom' if lo <= low else 'top' if high is not None and hi >= high else None
        if side is not None:
            raise RfxRefusal(f'{name}(): extra flux monitor {entry.name!r} reaches the open {side} feed end '
                             '(at or beyond its feed sheet); the run()/forward() port has a shielded end')


def two_port_feed_layout(grid):
    """Return the two calculators' historical mirrored extremity/feed/source indices."""
    nz = grid.shape[2]
    z_hi_coax_top = nz - int(grid.pad_z_hi) - 2
    z_feed_top = z_hi_coax_top - 1
    z_src_top = z_hi_coax_top - 3
    z_lo_coax_bot = int(grid.pad_z_lo) + 2
    z_feed_bot = z_lo_coax_bot + 1
    z_src_bot = z_lo_coax_bot + 3
    return z_hi_coax_top, z_feed_top, z_src_top, z_lo_coax_bot, z_feed_bot, z_src_bot

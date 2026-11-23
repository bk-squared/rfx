"""Test-only calculator stand-in: run() with its source end laid out and built
as compute_coaxial_line_reflection builds it, open and with no setback.
"""
from dataclasses import replace

import numpy as np


def open_end(patch):
    import rfx.model.coax_line as line
    import rfx.model.coax_port as port_model
    original_layout = line.coax_port_layout
    original_builder = line.build_coax_line

    def layout(grid, port, **kwargs):
        result = original_layout(grid, port, **kwargs)
        setback = max(2, int(np.ceil(float(port.outer_radius) / float(grid.dx)))) + 1
        shift = setback if port.face == 'top' else -setback
        return replace(result, feed_index=result.feed_index + shift,
                       source_index=result.source_index + shift)

    def builder(*args, **kwargs):
        kwargs['ends'] = tuple(replace(end, shielded=False) for end in kwargs['ends'])
        return original_builder(*args, **kwargs)

    patch.setattr(line, 'coax_port_layout', layout)
    patch.setattr(port_model, 'coax_port_layout', layout)
    patch.setattr(line, 'build_coax_line', builder)

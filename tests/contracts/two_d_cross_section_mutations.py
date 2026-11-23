"""Opt-in mutations of the public two-dimensional realization contract.

Run each command with the normal test environment; each must exit nonzero::

    RFX_CROSS_SECTION_MUTATION=kottke python -m pytest tests/contracts/test_two_d_cross_section.py -q -k 'box_extent_table and run-2d_tez-kottke-z-0.3_2'
    RFX_CROSS_SECTION_MUTATION=presence python -m pytest tests/contracts/test_two_d_cross_section.py -q -k 'box_extent_table and run-2d_tez-off-z0.1_2'
    RFX_CROSS_SECTION_MUTATION=sphere python -m pytest tests/contracts/test_two_d_cross_section.py -q -k 'curved_footprint and sphere-run-2d_tez'
"""

from dataclasses import replace

import numpy as np


def install(name, monkeypatch):
    """Install exactly one deliberately incorrect rule for a named contract row."""
    from rfx.geometry import cross_section
    from rfx.geometry.csg import Sphere

    if name == "kottke":
        monkeypatch.setattr(cross_section, "section_entries", lambda sim, grid: sim._geometry)
    elif name == "presence":
        monkeypatch.setattr(cross_section, "_present", lambda *args: True)
    elif name == "sphere":
        original = cross_section.cross_section_2d

        def at_cell_centre(shape, grid):
            result = original(shape, grid)
            if isinstance(shape, Sphere) and grid.is_2d:
                dz = float(grid.cells(2)[0]) / 2 - shape.center[2]
                result = replace(result, radius=np.sqrt(shape.radius**2 - dz**2))
            return result

        monkeypatch.setattr(cross_section, "cross_section_2d", at_cell_centre)
    else:
        raise ValueError(f"Unknown cross-section mutation: {name}")

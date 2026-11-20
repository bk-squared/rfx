"""A modal section judges thin elements on both declared guide wall nodes."""
import re

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.errors import RfxRefusal
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from tests.oracle.test_waveguide_port_validation_battery import (
    DOMAIN, PORT_LEFT_X, PORT_RIGHT_X,
)


def _guide(*, volume_walls=False):
    """Use the battery's two-port TE10 declaration and read realized nodes."""
    freqs = np.array([4.5e9, 8e9])
    f0 = float(freqs.mean())
    sim = Simulation(freq_max=float(freqs[-1]), domain=DOMAIN, dx=0.001,
                     cpml_layers=10, boundary={"x": "cpml", "y": "pec", "z": "pec"})
    grid = sim.freeze_mesh()
    coords = coords_from_uniform_grid(grid)
    nodes = tuple(np.asarray(line, dtype=float) for line in (coords.x, coords.y, coords.z))
    walls = [(span.start, span.stop - 1) for span in grid.interior]
    if volume_walls:
        outer = list(walls)
        walls[1] = (outer[1][0] + 2, outer[1][1] - 2)
        walls[2] = (outer[2][0] + 1, outer[2][1] - 1)
        for axis in (1, 2):
            for side in (0, 1):
                lo = [nodes[a][outer[a][0]] for a in range(3)]
                hi = [nodes[a][outer[a][1]] for a in range(3)]
                if side == 0:
                    hi[axis] = nodes[axis][walls[axis][0]]
                else:
                    lo[axis] = nodes[axis][walls[axis][1]]
                sim.add(Box(tuple(lo), tuple(hi)), material="pec")
    aperture = ({f"{axis}_range": tuple(nodes[a][i] for i in walls[a])
                 for a, axis in ((1, "y"), (2, "z"))} if volume_walls else {})
    for position, direction, name in ((PORT_LEFT_X, "+x", "left"),
                                      (PORT_RIGHT_X, "-x", "right")):
        sim.add_waveguide_port(
            position, direction=direction, mode=(1, 0), mode_type="TE", freqs=freqs,
            f0=f0, bandwidth=(freqs[-1] - freqs[0]) / f0,
            waveform="modulated_gaussian", n_modes=1, name=name, **aperture)
    return sim, nodes, walls


@pytest.mark.parametrize("axis", [1, 2], ids=["y", "z"])
@pytest.mark.parametrize("row", ["hi", "lo", "PEC"])
def test_waveguide_section_wall_nodes(row, axis):
    """Hi/lo loss refuses with the declared cell; both kinds of PEC wall pass."""
    if row == "PEC":
        for volume_walls in (False, True):
            sim, _, _ = _guide(volume_walls=volume_walls)
            sim.forward(n_steps=2, skip_preflight=True)
        return
    sim, nodes, walls = _guide()
    source = int(np.argmin(np.abs(nodes[0] - PORT_LEFT_X)))
    along = 3 - axis
    start, stop = walls[along][0] + 2, walls[along][1] - 4
    assert start - walls[along][0] != walls[along][1] - stop
    wall = walls[axis][int(row == "hi")]
    lo = [source + 1, 0, 0]
    hi = [source + 2, 0, 0]
    lo[axis] = hi[axis] = wall
    lo[along], hi[along] = start, stop
    sim.add_thin_conductor(
        Box(tuple(nodes[a][i] for a, i in enumerate(lo)),
            tuple(nodes[a][i] for a, i in enumerate(hi))),
        sigma_bulk=1e3, thickness=1e-4, surface_impedance_f0=6.25e9)
    expected = (
        "surface-impedance sheet (sheet_impedance.mask_ex) "
        f"at cell {tuple(lo)}; the port has no model for a thin element in its section; "
        "draw it beyond the reference plane.")
    with pytest.raises(RfxRefusal, match=re.escape(expected)):
        sim.forward(n_steps=2, skip_preflight=True)

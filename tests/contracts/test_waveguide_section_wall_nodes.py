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
            waveform="modulated_gaussian", n_modes=1, name=name,
            amplitude=1.0 if name == "left" else 0.0, **aperture)
    return sim, nodes, walls


@pytest.mark.parametrize("axis", [1, 2], ids=["y", "z"])
@pytest.mark.parametrize("row", ["hi", "lo", "PEC"])
def test_waveguide_section_wall_nodes(row, axis, capsys):
    """Hi/lo wall loss refuses at its wall node; both PEC wall kinds pass."""
    if row == "PEC":
        for volume_walls in (False, True):
            sim, _, _ = _guide(volume_walls=volume_walls)
            sim.forward(n_steps=2, skip_preflight=True)
        return
    sim, nodes, _ = _guide(volume_walls=False)
    face = f"{'xyz'[axis]}_{row}"
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=6.25e9, faces=(face,))
    with pytest.raises(
            RfxRefusal,
            match="the port has no model for a thin element in its section") as refusal:
        sim.forward(n_steps=2, skip_preflight=True)
    message = str(refusal.value)
    assert "sheet_impedance.mask_e" in message
    match = re.search(r"at cell \((\d+), (\d+), (\d+)\)", message)
    assert match is not None, message
    cell = tuple(int(index) for index in match.groups())
    span = sim.freeze_mesh().interior[axis]
    assert cell[axis] == (span.stop - 1 if row == "hi" else span.start)
    port_nodes = tuple(int(np.argmin(np.abs(nodes[0] - position)))
                       for position in (PORT_LEFT_X, PORT_RIGHT_X))
    assert min(port_nodes) <= cell[0] <= max(port_nodes)
    with capsys.disabled():
        print(f"{row}-{'xyz'[axis]}: {message}")

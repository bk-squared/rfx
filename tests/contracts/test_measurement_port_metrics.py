"""Port projectors read primal voltage lengths and transverse dual contours."""
from types import SimpleNamespace

import numpy as np
import pytest

from rfx.probes.probes import _metric_ampere_loop


@pytest.mark.parametrize('component,axes', [('ex', (1, 2)), ('ey', (2, 0)), ('ez', (0, 1))])
def test_ampere_contour_uses_both_transverse_duals(component, axes):
    # Independent linear H fields make each backward difference exactly one.
    indices = np.indices((4, 4, 4))
    state = SimpleNamespace(hx=indices[2]+indices[1],
                            hy=indices[0]+indices[2], hz=indices[1]+indices[0])
    lengths = (2., 3., 5.)
    grid = SimpleNamespace(duals=lambda axis: np.full(4, lengths[axis]))
    got = _metric_ampere_loop(state, (2, 2, 2), component, grid, (False,)*3)
    assert got == lengths[axes[1]] - lengths[axes[0]]


def test_wire_port_lengths_are_the_realized_cells_of_a_graded_grid():
    """On a real graded grid (no mock) the wire port's gap voltage reads the
    cells along its own axis and its Ampere loop the two transverse dual
    lengths at its node. PEC walls, so the realized cells are the declared
    profiles: port node x = 3.3 mm (cells 0.9 | 1.2 mm, dual 1.05 mm),
    y = 3.8 mm (0.6 | 0.8 mm, dual 0.7 mm), gap z 0 .. 2.8 mm (cells 0.8,
    0.6, 1.4 mm). Before S2 M2 these functions multiplied by ``grid.dx``."""
    from rfx.probes.probes import (wire_port_current, wire_port_gap_voltage,
                                   wire_port_voltage)
    from rfx.sources.sources import wire_port_from_entry
    from tests.unit.sparams.test_ringdown_run import DXP, DYP, DZP, _box
    sim = _box('graded', cells=3)
    grid = sim._build_nonuniform_grid()
    np.testing.assert_allclose(np.asarray(grid.cells('z'))[:len(DZP)], DZP, rtol=1e-6)
    port = wire_port_from_entry(sim._ports[0])
    shape = tuple(grid.shape)
    indices = np.indices(shape).astype(np.float32)
    # Unit backward differences in H, and a distinct E value per z cell.
    state = SimpleNamespace(hx=indices[2]+indices[1], hy=indices[0]+indices[2],
                            hz=indices[1]+indices[0],
                            ez=1.0 + indices[2], ex=indices[0]*0, ey=indices[0]*0)
    dual_x, dual_y = (DXP[3] + DXP[4]) / 2, (DYP[3] + DYP[4]) / 2
    np.testing.assert_allclose(float(wire_port_current(state, grid, port)),
                               dual_y - dual_x, rtol=1e-5)
    np.testing.assert_allclose(float(wire_port_gap_voltage(state, grid, port)),
                               -(1 * DZP[0] + 2 * DZP[1] + 3 * DZP[2]), rtol=1e-5)
    np.testing.assert_allclose(float(wire_port_voltage(state, grid, port)),
                               -2 * DZP[1], rtol=1e-5)

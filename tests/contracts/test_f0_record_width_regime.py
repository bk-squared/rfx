"""D:c1d171: only PEC-like f0 sheets claim the measured free-edge span."""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.core.yee import EPS_0, MU_0


@pytest.mark.parametrize('ratio', [0.999, 1.001], ids=['below', 'above'])
def test_f0_record_width_regime(ratio):
    dx, f0 = 1e-3, 9e9
    resistance = ratio * 1e-2 * np.sqrt(MU_0 / EPS_0)
    sim = Simulation(freq_max=10e9, domain=(10 * dx,) * 3, dx=dx,
                     boundary='pec', snap='declared')
    sim.add_thin_conductor(Box((2 * dx, 3 * dx, 5 * dx), (7 * dx, 8 * dx, 5 * dx)),
                           sigma_bulk=np.pi * f0 * MU_0 / resistance**2,
                           thickness=35e-6, surface_impedance_f0=f0)
    grid = sim._build_grid()
    # Expected endpoints come from the grid nodes, independently of report helpers.
    nodes = np.arange(grid.ny, dtype=float) * float(grid.dx)
    offset = 0.35 * float(grid.dx) if ratio < 1 else 0.0
    expected = (nodes[3] - offset, nodes[8] + offset)
    record = sim.realized_geometry()
    entity = record.entities[0]
    assert entity.axes[1].bounds_m == pytest.approx(expected, rel=1e-9)
    assert entity.axes[1].extent_m == pytest.approx(expected[1] - expected[0], rel=1e-9)
    note = ('edge measured at σ 5.8e7 S/m; PEC-like regime up to 1e-2·η0 by rule, not measured between'
            if ratio < 1 else 'edge not measured')
    assert entity.edge_note == note
    row = record.realization_table.row('thin_conductor[0]')
    assert row.edge_note == note
    assert row.model == ('solved sheet edges' if ratio < 1 else 'node span')

"""G1: production conductor products, persistent stages, and consumption replay."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rfx.conductors import clear_conductor_edges, realized_conductors
from .builders import build


GEOMETRY_ROWS = [('_geometry', 'pec_volume'), ('_geometry', 'pec_sheet'),
                 ('_geometry', 'pec_wire'), ('_thin_conductors', 'pec_sheet'),
                 ('_pinned_sheets', 'pec_sheet')]


@pytest.mark.parametrize('row', GEOMETRY_ROWS)
def test_conductors_constant_grid_equivalence(row):
    a = build(row, 'run_uniform')
    ga = a._build_grid()
    b = build(row, 'run_nonuniform', dt=ga.dt)
    gb = b._build_nonuniform_grid()
    ca = realized_conductors(a, ga)
    cb = realized_conductors(b, gb, nonuniform=True)
    for x, y in zip(ca.edges, cb.edges, strict=True):
        np.testing.assert_array_equal(x, y)
    assert (ca.pec_cells is None) == (cb.pec_cells is None)
    if ca.pec_cells is not None:
        np.testing.assert_array_equal(ca.pec_cells, cb.pec_cells)
    assert len(ca.sheets) == len(cb.sheets)
    assert len(ca.wires) == len(cb.wires)


def test_port_clearing_is_persistent():
    sim = build(('_geometry', 'pec_volume'), 'run_uniform')
    c = realized_conductors(sim, sim._build_grid())
    cell = tuple(int(k) for k in np.argwhere(c.edges[2])[0])
    before = tuple(np.array(m) for m in c.edges)
    after = clear_conductor_edges(c, [cell], component='ez', entity_id='port[0]')
    assert after is not c
    assert not after.edges[2][cell]
    for old, original in zip(c.edges, before, strict=True):
        np.testing.assert_array_equal(old, original)
    assert after.provenance[-1].entity_ids == ('port[0]',)
    assert after.provenance[-1].stage == 'port-edge-clearing'
    with pytest.raises(FrozenInstanceError):
        c.pec_edges = after.pec_edges

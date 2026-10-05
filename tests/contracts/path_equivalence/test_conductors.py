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


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform', 'fwd_uniform', 'fwd_nonuniform'])
def test_preflight_and_run_share_one_assembly(monkeypatch, lane):
    sim = build(('_geometry', 'pec_sheet'), lane)
    method = '_assemble_materials_nu' if lane.endswith('nonuniform') else '_assemble_materials'
    original = getattr(sim, method)
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sim, method, count)
    sim.preflight()
    initial = sim._campaign_ctx().realized()
    assert len(initial.sheets) == 1
    assert any(edge.any() for edge in initial.edges)
    runner = sim.forward if lane.startswith('fwd_') else sim.run
    result = runner(n_steps=2, skip_preflight=True)
    final = (sim._campaign_ctx().realized() if lane.startswith('fwd_')
             else result.realized_geometry.conductors)
    assert calls == [1]
    assert final is initial
    assert sim._campaign_ctx().realized() is final
    assert sim.realized_geometry().conductors is final


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform'])
def test_record_reads_kernel_after_port_clearing(monkeypatch, lane):
    from rfx import Box
    from .builders import point
    sim = build(('_ports', 'lumped_port'), lane)
    sim.add(Box(point(4, 2, 2), point(6, 4, 4)), material='pec')
    grid = sim._build_nonuniform_grid() if lane == 'run_nonuniform' else sim._build_grid()
    original = realized_conductors(sim, grid, nonuniform=lane == 'run_nonuniform')
    idx = tuple(grid.index_of(a, x) for a, x in enumerate(point(5, 3, 3)))
    assert original.edges[2][idx]
    result = sim.run(n_steps=2, skip_preflight=True)
    final = result.realized_geometry.conductors
    assert not final.edges[2][idx]
    assert final.provenance[-1].stage == 'port-edge-clearing'
    assert sim._campaign_ctx().realized() is final
    np.testing.assert_array_equal(sim.realized_geometry().edge_masks[2], final.edges[2])


@pytest.mark.parametrize('graded', [False, True])
def test_two_device_conductors_are_the_same_object_fields(graded):
    import jax
    if len(jax.devices()) < 2:
        pytest.skip('two CPU devices required')
    lane = 'run_nonuniform' if graded else 'run_uniform'
    a = build(('_geometry', 'pec_volume'), lane, graded=graded)
    b = build(('_geometry', 'pec_volume'), lane, graded=graded)
    ca = a.run(n_steps=2, skip_preflight=True).realized_geometry.conductors
    cb = b.run(n_steps=2, skip_preflight=True,
               devices=jax.devices()[:2]).realized_geometry.conductors
    for x, y in zip(ca.pec_edges, cb.pec_edges, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(ca.pec_cells, cb.pec_cells)
    assert ca.provenance == cb.provenance
    assert b._campaign_ctx().realized() is cb


@pytest.mark.parametrize('lane,distributed', [('run_uniform', False),
    ('run_nonuniform', False), ('run_uniform', True), ('run_nonuniform', True)])
def test_dump_replay_moves_fields_and_identity_is_exact(lane, distributed):
    import jax
    import jax.numpy as jnp
    from rfx import _realized
    from .builders import point
    if distributed and len(jax.devices()) < 2:
        pytest.skip('two CPU devices required')
    sim = build(('_geometry', 'pec_volume'), lane, graded=lane == 'run_nonuniform')
    grid = sim._build_nonuniform_grid() if lane == 'run_nonuniform' else sim._build_grid()
    index = tuple(grid.index_of(a, x) for a, x in enumerate(point(2.1, 2.3, 2.2)))
    kwargs = dict(n_steps=12, skip_preflight=True, compute_s_params=False)
    if distributed:
        kwargs['devices'] = jax.devices()[:2]
    with _realized.capture() as dumped:
        baseline = sim.run(**kwargs)
    rows = [r for r in dumped.records if r['site'] == 'conductors']
    assert len(rows) == 1
    saved = tuple(jnp.asarray(m) for m in rows[0]['pec_edges'])
    assert not saved[2][index]

    def replay(changed):
        def transform(site, quantity, values):
            if site != 'conductors' or quantity != 'pec_edges':
                return values
            return saved[:2] + (saved[2].at[index].set(True),) if changed else saved
        with _realized.capture(transform=transform):
            return sim.run(**kwargs)

    same = replay(False)
    changed = replay(True)
    for field in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        np.testing.assert_array_equal(getattr(baseline.state, field), getattr(same.state, field))
    np.testing.assert_array_equal(baseline.time_series, same.time_series)
    assert not np.array_equal(baseline.time_series, changed.time_series)
    assert changed.realized_geometry.conductors.edges[2][index]


def test_kernel_rejects_port_edits_outside_object():
    from rfx.conductors import at_kernel
    sim = build(('_geometry', 'pec_volume'), 'run_uniform')
    c = realized_conductors(sim, sim._build_grid())
    arrays = tuple(np.array(edge) for edge in c.pec_edges)
    arrays[2][0, 0, 0] = True
    with pytest.raises(ValueError, match='outside a conductor stage'):
        at_kernel(sim, c, lane='run_uniform', pec_edges=arrays)


def test_sheet_spans_are_read_from_conductors(monkeypatch):
    import rfx.mesh_edges as shared
    sim = build(('_geometry', 'pec_sheet'), 'run_uniform')
    c = sim._campaign_ctx().realized()
    assert any(span is not None for span in c.sheets[0].solved_spans)

    def unexpected(*args, **kwargs):
        raise AssertionError('a report recomputed the solved sheet span')

    monkeypatch.setattr(shared, 'solved_sheet_span', unexpected)
    record = sim.realized_geometry()
    sim.preflight()
    assert record.conductors is c

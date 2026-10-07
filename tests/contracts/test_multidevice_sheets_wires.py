"""G3: owned conductor edges, visible fields, and sheet-contact wire S11."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rfx import Box, PolylineWire, Simulation
from rfx.core.yee import init_state
from rfx.model import conductors as C
from rfx.runners import _distributed_common as D
from rfx.runners._distributed_common import apply_pec_mask_shmap
from rfx.runners._rank import mesh_ranks
from rfx.sources.sources import _wire_port_live_cells, wire_port_from_entry

GATE = 1e-3
STEPS = 300
KINDS = ('sheet_x_seam', 'sheet_z_cross', 'sheet_z_edge', 'wire_x_cross', 'wire_z_seam')
# Measured float32 max_t |multi-single| / max_t |single|, per probe.
MEASURED = {
    ('sheet_x_seam', False, 'pec'): (0, 0),
    ('sheet_z_cross', False, 'pec'): (0, 0),
    ('sheet_z_edge', False, 'pec'): (0, 0),
    ('wire_x_cross', False, 'pec'): (0, 0),
    ('wire_z_seam', False, 'pec'): (0, 0),
    ('sheet_x_seam', True, 'pec'): (0, 0),
    ('sheet_z_cross', True, 'pec'): (0, 0),
    ('sheet_z_edge', True, 'pec'): (0, 0),
    ('wire_x_cross', True, 'pec'): (0, 0),
    ('wire_z_seam', True, 'pec'): (0, 0),
    ('sheet_x_seam', False, 'cpml'): (8.858390174e-06, 4.120632002e-05),
    ('sheet_z_cross', False, 'cpml'): (2.135175237e-06, 4.732910384e-05),
    ('sheet_z_edge', False, 'cpml'): (2.570882543e-06, 3.363429641e-05),
    ('wire_x_cross', False, 'cpml'): (1.009433163e-06, 4.949859886e-06),
    ('wire_z_seam', False, 'cpml'): (9.534178389e-06, 3.207571353e-05),
    ('sheet_x_seam', True, 'cpml'): (9.636387404e-06, 3.420851863e-05),
    ('sheet_z_cross', True, 'cpml'): (2.496851494e-06, 3.244226536e-05),
    ('sheet_z_edge', True, 'cpml'): (2.645492259e-06, 4.210601401e-05),
    ('wire_x_cross', True, 'cpml'): (8.705042092e-07, 6.864503121e-06),
    ('wire_z_seam', True, 'cpml'): (7.589248526e-06, 2.719858458e-05),
}


def _devices(count=2):
    if len(jax.devices()) < count:
        pytest.skip(f'need {count} CPU devices; 3-slab evidence uses a separate process')
    return jax.devices()[:count]


def _build(kind, graded, boundary, conductor=True, count=2):
    profile = np.array([.001] + [.0009, .0011]*6 + [.001, .001])
    sim = Simulation(freq_max=15e9, domain=(.031, .015, .013), dx=.001,
                     boundary=boundary, cpml_layers=4 if boundary == 'cpml' else 0,
                     snap='declared', **({'dy_profile': profile} if graded else {}))
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    per = (grid.shape[0] + count - 1)//count
    seam = grid.node_of(0, per)
    # A 3-slab crossing sheet spans the whole middle slab.
    left, right = (seam-.005, seam+.005) if count == 2 else (
        grid.node_of(0, per-2), grid.node_of(0, 2*per+2))
    shapes = {
        'sheet_x_seam': Box((seam, .003, .002), (seam, .011, .010)),
        'sheet_z_cross': Box((left, .003, .006), (right, .011, .006)),
        'sheet_z_edge': Box((seam-.006, .003, .006), (seam, .011, .006)),
        'wire_x_cross': PolylineWire(((seam-.005, .006, .005), (seam+.004, .006, .005)), radius=0),
        'wire_z_seam': PolylineWire(((seam, .006, .002), (seam, .006, .010)), radius=0),
    }
    if conductor:
        sim.add(shapes[kind], material='pec')
    component = 'ez' if kind in ('sheet_x_seam', 'wire_z_seam') else 'ex'
    sim.add_source((seam-.003, .005, .004), component, amplitude_kind='field')
    probes = ((seam-.001, .006, .005), (seam+.003, .009, .008))
    if kind == 'wire_x_cross':
        probes = ((seam-.001, .005, .005), (seam+.002, .007, .006))
    for pos in probes:
        sim.add_probe(pos, component)
    return sim, grid, shapes[kind]


def _reference(kind, graded, boundary, count=2):
    sim, grid, drawn = _build(kind, graded, boundary, count=count)
    root = C.solve_conductors(sim, grid, nonuniform=graded)
    assert root.pec_edges is not None and sum(np.count_nonzero(e) for e in root.pec_edges)
    if kind.startswith('sheet'):
        assert len(root.sheets) == 1
        sheet = root.sheets[0]
        axis = sheet.normal_axis
        assert grid.node_of(axis, sheet.plane) == pytest.approx(drawn.corner_lo[axis], abs=1e-12)
        assert drawn.corner_lo[axis] == drawn.corner_hi[axis]
        assert np.count_nonzero(sheet.footprint)
    return root


def _owned_masks(monkeypatch, kind, graded, boundary, count=2):
    """Capture the actual public runner's staged edges, then apply its kernel."""
    devices = _devices(count)
    expected = _reference(kind, graded, boundary, count)
    staged = []
    original = C.stage_distributed_edges if graded else D.shard_x_slabs

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if graded:
            staged.extend(result)
        elif args[0].dtype == jnp.bool_:
            staged.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(C if graded else D, 'stage_distributed_edges' if graded else 'shard_x_slabs', capture)
        sim, grid, _ = _build(kind, graded, boundary, count=count)
        sim.run(n_steps=2, devices=devices, compute_s_params=False, skip_preflight=True)
    assert len(staged) == 3
    mesh = Mesh(np.array(devices), ('x',))
    shape = staged[0].shape
    local = shape[0]//count
    per = local-2
    ones = jax.device_put(jnp.ones(shape, dtype=jnp.float32), NamedSharding(mesh, P('x')))
    state = init_state(shape)._replace(ex=ones, ey=ones, ez=ones)
    masked = apply_pec_mask_shmap(state, tuple(staged), mesh, count, local, ranks=mesh_ranks(mesh))
    # Independent ownership: physical node positions, high side at equality.
    nodes = np.array([grid.node_of(0, i) for i in range(grid.shape[0])])
    seams = np.array([grid.node_of(0, r*per) for r in range(1, count)])
    owners = np.searchsorted(seams, nodes, side='right')
    for component, reference in zip(('ex', 'ey', 'ez'), expected.pec_edges):
        applied = (np.asarray(getattr(masked, component)) == 0).reshape(count, local, *shape[1:])
        assert not applied[:, [0, -1]].any(), 'a slab directly masked a ghost edge'
        owned = applied[:, 1:-1].reshape(count*per, *shape[1:])[:grid.shape[0]]
        np.testing.assert_array_equal(owned, np.asarray(reference))
        applications = np.zeros(grid.shape, dtype=np.int32)
        for rank in range(count):
            for li in range(local):
                gi = rank*per+li-1
                if 0 <= gi < grid.shape[0]:
                    applications[gi] += applied[rank, li]
                    if applied[rank, li].any():
                        assert owners[gi] == rank, 'seam belongs to the high-side slab'
        assert np.count_nonzero(applications > 1) == 0
        np.testing.assert_array_equal(applications, np.asarray(reference).astype(np.int32))
    print(f'B1 {kind} graded={graded} boundary={boundary} slabs={count}: exact; duplicates=0 ghosts=0')


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_b1_owned_masks(monkeypatch, kind, graded, boundary):
    _owned_masks(monkeypatch, kind, graded, boundary)


@pytest.mark.parametrize('graded', [False, True])
def test_b1_three_slabs(monkeypatch, graded):
    _owned_masks(monkeypatch, 'sheet_z_cross', graded, 'pec', count=3)


def _traces(kind, graded, boundary, conductor=True, distributed=False, count=2):
    sim, grid, _ = _build(kind, graded, boundary, conductor=conductor, count=count)
    result = sim.run(n_steps=STEPS, devices=_devices(count) if distributed else None,
                     compute_s_params=False, skip_preflight=True)
    assert result.realized_geometry.lane == (
        "run_distributed" if distributed else "run_nonuniform" if graded else "run_uniform")
    return np.asarray(result.time_series), grid.dt


def _field_parity(kind, graded, boundary, count=2):
    _reference(kind, graded, boundary, count)
    single, dt = _traces(kind, graded, boundary, count=count)
    multi, multi_dt = _traces(kind, graded, boundary, distributed=True, count=count)
    empty, empty_dt = _traces(kind, graded, boundary, conductor=False, count=count)
    assert dt == multi_dt == empty_dt
    assert single.dtype == multi.dtype == np.float32
    peaks = np.max(np.abs(single), axis=0)
    assert np.all(peaks > 0)
    rel = np.max(np.abs(multi-single), axis=0)/peaks
    visibility = np.max(np.abs(single-empty), axis=0)/np.max(np.abs(empty), axis=0)
    print(f'B2 {kind} graded={graded} boundary={boundary} slabs={count}: rel={rel.tolist()} visibility={visibility.tolist()}')
    assert np.all(rel <= GATE), rel
    assert np.max(visibility) > 100*GATE, visibility


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_b2_visible_fields(kind, graded, boundary):
    _field_parity(kind, graded, boundary)


@pytest.mark.parametrize('graded', [False, True])
def test_b2_three_slabs(graded):
    _devices(3)
    _field_parity('sheet_z_cross', graded, 'pec', count=3)


def _port_model():
    from tests.unit.runners.test_distributed_wire_ports import _model
    sim = _model('ez')
    # The first Ez port lands on this x-normal sheet; part of its line is dead.
    sim.add(Box((.005, .006, .004), (.005, .010, .005)), material='pec')
    sim._snap = 'declared'
    return sim


def test_b3_sheet_contact_wire_port(monkeypatch):
    from rfx.probes import sparam_driver
    from tests.unit.runners.test_distributed_wire_ports import FREQS
    expected_sim = _port_model()
    grid = expected_sim._build_grid()
    root = C.solve_conductors(expected_sim, grid)
    # Count expected live edges directly from the independent single-device object.
    expected_counts = []
    for pe in expected_sim._ports:
        start = grid.position_to_index(pe.position)
        stop = grid.index_of(2, pe.position[2]+pe.extent)
        expected_counts.append(sum(not bool(root.pec_edges[2][start[0], start[1], k])
                                   for k in range(start[2], stop)))
    assert 0 < expected_counts[0] < 4
    kwargs = dict(n_steps=300, s_param_n_steps=320, s_param_freqs=FREQS,
                  compute_s_params=True, skip_preflight=True)
    single = expected_sim.run(**kwargs)
    counts = []
    reader_counts = []
    from rfx.sources import sources as port_sources
    original_live = port_sources._wire_port_live_cells

    def capture_live(g, port, pec_edge_masks=None):
        result = original_live(g, port, pec_edge_masks)
        reader_counts.append((tuple(port.start), result[2]))
        return result

    monkeypatch.setattr(port_sources, '_wire_port_live_cells', capture_live)
    original = sparam_driver._distributed_port_cells

    def capture(g, ports, pec_edge_masks=None):
        for pe in ports:
            counts.append(_wire_port_live_cells(g, wire_port_from_entry(pe), pec_edge_masks)[2])
        return original(g, ports, pec_edge_masks)

    monkeypatch.setattr(sparam_driver, '_distributed_port_cells', capture)
    multi = _port_model().run(devices=_devices(), **kwargs)
    assert counts == expected_counts
    by_position = {tuple(pe.position): n for pe, n in zip(expected_sim._ports, expected_counts)}
    assert reader_counts
    for position, live_count in reader_counts:
        assert live_count == by_position[position], ('load/drive live count', position, live_count)
    delta = np.abs(multi.s_params[0, 0]-single.s_params[0, 0])
    print(f'B3 frequencies={FREQS.tolist()} live={counts} abs_S11_difference={delta.tolist()}')
    assert np.all(delta <= 1e-3), delta


# Decision record 1: recorded values are observations, not hard-coded controls.
MEASURED_JVP = {
    'volume': (0.002868985990062356, 0.0010550093138590455),
    'empty': (0.0024162980262190104, 0.0025435416027903557),
    'sheet': (0.004277050960808992, 0.0012863829033449292),
}


def _graded_forward_control(kind, amplitude, distributed):
    sim = Simulation(freq_max=15e9, domain=(.015, .011, .009), dx=.001,
                     boundary='pec', dy_profile=np.array([.001]+[.0009, .0011]*4+[.001, .001]),
                     snap='declared')
    if kind != 'empty':
        sim.add(Box((.008, .002, .002),
                    (.009 if kind == 'volume' else .008, .009, .007)), material='pec')
    sim.add_source((.003, .004, .003), 'ez', amplitude_kind='field')
    for pos in ((.006, .005, .004), (.011, .007, .006)):
        sim.add_probe(pos, 'ez')
    grid = sim._build_nonuniform_grid()
    return sim.forward(n_steps=180, eps_override=jnp.ones(grid.shape)*amplitude,
                       distributed=distributed, devices=_devices() if distributed else None,
                       checkpoint=False, skip_preflight=True).time_series


def test_graded_forward_jvp_against_controls():
    differences = {}
    for kind in ('volume', 'empty', 'sheet'):
        single = jax.jvp(lambda a: _graded_forward_control(kind, a, False),
                         (jnp.float32(1.1),), (jnp.float32(1),))
        multi = jax.jvp(lambda a: _graded_forward_control(kind, a, True),
                        (jnp.float32(1.1),), (jnp.float32(1),))
        for name, reference, actual in zip(('primal', 'jvp'), single, multi):
            reference, actual = np.asarray(reference), np.asarray(actual)
            assert reference.shape == actual.shape == (180, 2)
            assert reference.dtype == actual.dtype == np.float32
            peak = np.max(np.abs(reference), axis=0)
            assert np.all(peak > 0)
            relative = np.max(np.abs(actual-reference), axis=0)/peak
            assert np.all(np.isfinite(relative))
            print(f'graded forward {kind} {name}: {relative.tolist()}')
            if name == 'jvp':
                differences[kind] = relative
    limit = 2*np.maximum(differences['volume'], differences['empty'])
    assert np.all(differences['sheet'] <= limit), (differences, limit)


@pytest.fixture(autouse=True)
def _quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield

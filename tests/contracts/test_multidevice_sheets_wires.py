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
# PEC walls: multi and single device differ only by float32 rounding, which a
# lossless box carries forward; one rounding per step bounds it at
# n_steps * eps(float32) of the peak. Measured 0 on arm64 and up to 2e-6 on
# x86-64 (same code, 180 steps), so bit-equality is not a portable bar.
def _rounding_bar(n_steps):
    return n_steps*float(np.finfo(np.float32).eps)

STEPS = 300
KINDS = ('sheet_x_seam', 'sheet_z_cross', 'sheet_z_edge', 'wire_x_cross', 'wire_z_seam')
# Measured float32 max_t |multi-single| / max_t |single|, per probe.
MEASURED = {
    ('sheet_x_seam', False, 'pec'): {'conductor': (0, 0)},
    ('sheet_z_cross', False, 'pec'): {'conductor': (0, 0)},
    ('sheet_z_edge', False, 'pec'): {'conductor': (0, 0)},
    ('wire_x_cross', False, 'pec'): {'conductor': (0, 0)},
    ('wire_z_seam', False, 'pec'): {'conductor': (0, 0)},
    ('sheet_x_seam', True, 'pec'): {'conductor': (0, 0)},
    ('sheet_z_cross', True, 'pec'): {'conductor': (0, 0)},
    ('sheet_z_edge', True, 'pec'): {'conductor': (0, 0)},
    ('wire_x_cross', True, 'pec'): {'conductor': (0, 0)},
    ('wire_z_seam', True, 'pec'): {'conductor': (0, 0)},
    ('sheet_x_seam', False, 'cpml'): {'conductor': (8.858390174e-06, 4.120632002e-05), 'empty': (4.503315267356811e-06, 5.9312609664630145e-05)},
    ('sheet_z_cross', False, 'cpml'): {'conductor': (2.135175237e-06, 4.732910384e-05), 'empty': (1.8594737412058748e-06, 2.1051429939689115e-05)},
    ('sheet_z_edge', False, 'cpml'): {'conductor': (2.570882543e-06, 3.363429641e-05), 'empty': (1.8594737412058748e-06, 2.1051429939689115e-05)},
    ('wire_x_cross', False, 'cpml'): {'conductor': (1.009433163e-06, 4.949859886e-06), 'empty': (7.053798753986484e-07, 5.264490482659312e-06)},
    ('wire_z_seam', False, 'cpml'): {'conductor': (9.534178389e-06, 3.207571353e-05), 'empty': (4.503315267356811e-06, 5.9312609664630145e-05)},
    ('sheet_x_seam', True, 'cpml'): {'conductor': (9.636387404e-06, 3.420851863e-05), 'empty': (4.891567186859902e-06, 4.07846127927769e-05)},
    ('sheet_z_cross', True, 'cpml'): {'conductor': (2.496851494e-06, 3.244226536e-05), 'empty': (1.936479748110287e-06, 1.9140967197017744e-05)},
    ('sheet_z_edge', True, 'cpml'): {'conductor': (2.645492259e-06, 4.210601401e-05), 'empty': (1.936479748110287e-06, 1.9140967197017744e-05)},
    ('wire_x_cross', True, 'cpml'): {'conductor': (8.705042092e-07, 6.864503121e-06), 'empty': (1.0786160373754683e-06, 5.5382238315360155e-06)},
    ('wire_z_seam', True, 'cpml'): {'conductor': (7.589248526e-06, 2.719858458e-05), 'empty': (4.891567186859902e-06, 4.07846127927769e-05)},
}


def _devices(count=2):
    if len(jax.devices()) < count:
        pytest.skip(f'need {count} CPU devices; 3-slab evidence uses a separate process')
    return jax.devices()[:count]


def _build(kind, graded, boundary, conductor=True, count=2):
    profile = np.array([.001] + [.0009, .0011]*6 + [.001, .001])
    sim = Simulation(freq_max=15e9, domain=(.030 if kind == 'sheet_z_high_padded' else .031, .015, .013), dx=.001,
                     boundary=boundary, cpml_layers=4 if boundary == 'cpml' else 0,
                     snap='declared', **({'dy_profile': profile} if graded else {}))
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    per = (grid.shape[0] + count - 1)//count
    seam = grid.node_of(0, per)
    # A 3-slab crossing sheet spans the whole middle slab.
    left, right = (seam-.005, seam+.005) if count == 2 else (
        grid.node_of(0, per-2), grid.node_of(0, 2*per+2))
    shapes = {
        'sheet_z_high_padded': Box((.020, .003, .006), (grid.node_of(0, grid.shape[0]-1), .011, .006)),
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


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_b1_owned_masks(monkeypatch, kind, graded, boundary):
    _owned_masks(monkeypatch, kind, graded, boundary)


@pytest.mark.parametrize('graded', [False, True])
def test_b1_three_slabs(monkeypatch, graded):
    _owned_masks(monkeypatch, 'sheet_z_cross', graded, 'pec', count=3)


@pytest.mark.parametrize('graded', [False, True])
def test_b1_high_x_padding(monkeypatch, graded):
    kind = 'sheet_z_high_padded'
    _, grid, _ = _build(kind, graded, 'pec')
    assert grid.shape[0] % 2 != 0
    root = _reference(kind, graded, 'pec')
    assert any(np.asarray(edge)[-1].any() for edge in root.pec_edges)
    _owned_masks(monkeypatch, kind, graded, 'pec')


def _traces(kind, graded, boundary, conductor=True, distributed=False, count=2):
    sim, grid, _ = _build(kind, graded, boundary, conductor=conductor, count=count)
    result = sim.run(n_steps=STEPS, devices=_devices(count) if distributed else None,
                     compute_s_params=False, skip_preflight=True)
    assert result.realized_geometry.lane == (
        ("run_distributed_nu" if graded else "run_distributed") if distributed
        else "run_nonuniform" if graded else "run_uniform")
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
    if boundary == 'pec':
        assert np.all(rel <= _rounding_bar(STEPS)), rel
    else:
        assert np.all(rel <= GATE), rel
        empty_multi, empty_multi_dt = _traces(kind, graded, boundary, conductor=False,
                                              distributed=True, count=count)
        assert empty_multi_dt == empty_dt
        empty_rel = np.max(np.abs(empty_multi-empty), axis=0)/np.max(np.abs(empty), axis=0)
        # The absorber alone accounts for a difference of this size (MEASURED).
        assert np.all(empty_rel <= GATE), empty_rel
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
    assert np.all(delta <= 1e-3), delta


# Decision records 2-3: primal and block JVP agree to accumulated float32 rounding.
def _graded_forward_sheet(amplitude, distributed):
    sim = Simulation(freq_max=15e9, domain=(.015, .011, .009), dx=.001,
                     boundary='pec', cpml_layers=0,
                     dy_profile=np.array([.001]+[.0009, .0011]*4+[.001, .001]),
                     snap='declared')
    sim.add(Box((.008, .002, .002), (.008, .009, .007)), material='pec')
    sim.add_source((.003, .004, .003), 'ez')
    for pos in ((.006, .005, .004), (.011, .007, .006)):
        sim.add_probe(pos, 'ez')
    grid = sim._build_nonuniform_grid()
    lo = [grid.index_of(i, v) for i, v in enumerate((.006, .003, .003))]
    hi = [grid.index_of(i, v) for i, v in enumerate((.010, .008, .006))]
    block = np.zeros(grid.shape, np.float32)
    block[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    eps = 1+(amplitude-1)*jnp.asarray(block)
    return sim.forward(n_steps=180, eps_override=eps,
                       distributed=distributed, devices=_devices() if distributed else None,
                       checkpoint=False, skip_preflight=True).time_series


def test_graded_forward_default_source_block_jvp():
    single = jax.jvp(lambda a: _graded_forward_sheet(a, False),
                     (jnp.float32(2),), (jnp.float32(1),))
    multi = jax.jvp(lambda a: _graded_forward_sheet(a, True),
                    (jnp.float32(2),), (jnp.float32(1),))
    for name, reference, actual in zip(('primal', 'jvp'), single, multi):
        reference, actual = np.asarray(reference), np.asarray(actual)
        assert reference.shape == actual.shape == (180, 2)
        assert reference.dtype == actual.dtype == np.float32
        peak = np.max(np.abs(reference), axis=0)
        assert np.all(peak > 0)
        relative = np.max(np.abs(actual-reference), axis=0)/peak
        assert np.all(relative <= _rounding_bar(180)), relative


def _lumped_model(kind, boundary):
    from rfx.sources.sources import GaussianPulse
    sim = Simulation(freq_max=15e9, domain=(.031, .015, .013), dx=.001,
                     boundary=boundary, cpml_layers=4 if boundary == 'cpml' else 0,
                     snap='declared')
    shape = {
        'volume': Box((.016, .007, .003), (.017, .008, .010)),
        'sheet': Box((.016, .003, .003), (.016, .011, .010)),
        'wire': PolylineWire(((.016, .007, .003), (.016, .007, .010)), radius=0),
    }[kind]
    sim.add(shape, material='pec')
    sim.add_port((.016, .007, .005), 'ez', impedance=50,
                 waveform=GaussianPulse(f0=7e9, bandwidth=1.2))
    for pos, component in (((.015, .007, .005), 'ez'), ((.019, .009, .008), 'ez'),
                           ((.014, .006, .004), 'ex'), ((.027, .008, .007), 'ez'),
                           ((.016, .007, .005), 'ez')):
        sim.add_probe(pos, component)
    return sim


@pytest.mark.parametrize('kind', ['volume', 'sheet', 'wire'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_lumped_port_on_conductor(monkeypatch, kind, boundary):
    single_sim = _lumped_model(kind, boundary)
    grid = single_sim._build_grid()
    root = C.solve_conductors(single_sim, grid)
    # Locate the driven edge from physical nodes, independently of the stage.
    index = tuple(int(np.flatnonzero(np.isclose(
        [grid.node_of(a, i) for i in range(grid.shape[a])], position,
        rtol=0, atol=1e-12))[0]) for a, position in enumerate((.016, .007, .005)))
    expected = [np.asarray(edge).copy() for edge in root.pec_edges]
    assert expected[2][index]
    expected[2][index] = False
    freqs = np.array([1e9, 2.5e9, 5e9, 7.5e9, 10e9])
    kwargs = dict(n_steps=STEPS, s_param_n_steps=320, s_param_freqs=freqs,
                  compute_s_params=True, skip_preflight=True)
    single = single_sim.run(**kwargs)
    staged = []
    original = D.shard_x_slabs

    def capture(*args, **kw):
        result = original(*args, **kw)
        if args[0].dtype == jnp.bool_:
            staged.append(np.asarray(result))
        return result

    monkeypatch.setattr(D, 'shard_x_slabs', capture)
    multi_sim = _lumped_model(kind, boundary)
    assert multi_sim._build_grid().dt == grid.dt
    multi = multi_sim.run(devices=_devices(), **kwargs)
    assert len(staged) >= 3 and len(staged) % 3 == 0
    # Main run and S-matrix drive scans must both release exactly this edge.
    for offset in range(0, len(staged), 3):
        for component, reference in enumerate(expected):
            slabs = staged[offset+component].reshape(2, -1, *grid.shape[1:])
            owned = slabs[:, 1:-1].reshape(-1, *grid.shape[1:])[:grid.shape[0]]
            np.testing.assert_array_equal(owned, reference)
    reference, actual = np.asarray(single.time_series), np.asarray(multi.time_series)
    assert reference.dtype == actual.dtype == np.float32
    peaks = np.max(np.abs(reference), axis=0)
    assert np.all(peaks > 0)
    relative = np.max(np.abs(actual-reference), axis=0)/peaks
    if boundary == 'pec':
        assert np.all(relative <= _rounding_bar(STEPS)), relative
    else:
        assert np.all(relative <= GATE), relative
    delta = np.abs(multi.s_params[0, 0]-single.s_params[0, 0])
    assert np.all(delta <= 1e-3), delta


@pytest.fixture(autouse=True)
def _quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield

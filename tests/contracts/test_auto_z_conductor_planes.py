"""Declared conductor planes and refusals before any time step."""
import re

import numpy as np
import pytest

from rfx import Box, GaussianPulse, RfxRefusal, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.mesh_edges import edge_aware_profiles


DOMAIN = (.05, .05, .02)
BOARD_LO, BOARD_HI = .005, .0066


def _box(lo, hi):
    return Box((.005, .005, lo), (.045, .045, hi))


def _model(lo=None, hi=None, material='pec', board=True, **kwargs):
    sim = Simulation(freq_max=10e9, domain=DOMAIN, boundary='cpml', **kwargs)
    if board:
        sim.add(_box(BOARD_LO, BOARD_HI), material='fr4')
    if lo is not None:
        sim.add(_box(lo, hi), material=material)
    return sim


def _interior_nodes(record, axis=2):
    d = record.domain[axis]
    return np.asarray(record.nodes[axis], dtype=np.float64)[
        d.padding[0]:d.padding[0] + d.n_cells + 1]


def _conductor_findings(report, label):
    return [i for i in report if i.code == 'realization' and i.loc == label]


@pytest.mark.parametrize('lo,hi,material,board', [
    pytest.param(.004, .005, 'pec', True, id='plate-4.0'),
    pytest.param(.0045, .005, 'pec', True, id='plate-4.5'),
    pytest.param(.004, .005, 'copper', True, id='copper'),
    pytest.param(.0055, .007, 'pec', True, id='buried-volume'),
    pytest.param(.0075, .0075, 'pec', True, id='sheet-air'),
    pytest.param(.0055, .0055, 'pec', True, id='sheet-board'),
    pytest.param(.0041, .005, 'pec', False, id='volume-alone'),
    pytest.param(.0041, .0041, 'pec', False, id='sheet-alone'),
])
def test_b1_declared_conductor_planes(lo, hi, material, board):
    sim = _model(lo, hi, material, board)
    report = sim.preflight()
    record = sim.realized_geometry()
    nodes = _interior_nodes(record)
    cells = np.diff(nodes)
    for plane in {lo, hi}:
        k = int(np.argmin(abs(nodes-plane)))
        touching = cells[max(0, k-1):min(len(cells), k+1)]
        assert abs(nodes[k]-plane) <= 1e-9 * min(touching)
    label = f'geometry[{int(board)}]'
    assert not _conductor_findings(report, label)
    assert not [i for i in report.errors if label in str(i) and i.code.startswith('pec_')]
    assert abs(sum(cells) - DOMAIN[2]) <= 1e-12


def test_b2_existing_block_node_keeps_profile_bit_identical():
    alone = _model()
    with_sheet = _model(BOARD_LO + (BOARD_HI-BOARD_LO)/4,
                        BOARD_LO + (BOARD_HI-BOARD_LO)/4)
    alone.preflight()
    report = with_sheet.preflight()
    a = _interior_nodes(alone.realized_geometry())
    b = _interior_nodes(with_sheet.realized_geometry())
    assert a.tobytes() == b.tobytes()
    ra, rb = alone.realized_geometry(), with_sheet.realized_geometry()
    da, db = ra.domain[2], rb.domain[2]
    ca = np.asarray(ra.cell_sizes[2])[da.padding[0]:da.padding[0] + da.n_cells]
    cb = np.asarray(rb.cell_sizes[2])[db.padding[0]:db.padding[0] + db.n_cells]
    assert ca.tobytes() == cb.tobytes()
    assert not _conductor_findings(report, 'geometry[1]')


@pytest.mark.parametrize('plane', [.0076, .0057], ids=['air', 'board'])
def test_b3_user_uniform_sheet_off_plane_refuses(plane):
    sim = _model(plane, plane, dx=.0005)
    issues = _conductor_findings(sim.preflight(), 'geometry[1]')
    assert any(i.severity == 'error' and 'sheet plane miss' in str(i) for i in issues)
    with pytest.raises(RfxRefusal, match=r'geometry\[1\].*sheet plane miss'):
        sim.run(n_steps=1)


@pytest.mark.parametrize('plane', [.0075, .0055], ids=['air', 'board'])
def test_b3c_user_uniform_sheet_on_plane_is_admitted(plane):
    sim = _model(plane, plane, dx=.0005)
    assert not [i for i in _conductor_findings(sim.preflight(), 'geometry[1]')
                if i.severity == 'error']


def test_b4_literal_automatic_xy_remedy():
    board = _box(BOARD_LO, BOARD_HI)
    plate = Box((.0203, .005, .0046), (.024, .045, .005))
    automatic = _model()
    automatic.add(plate, material='pec')
    report = automatic.preflight()
    issues = _conductor_findings(report, 'geometry[1]')
    assert any(i.severity == 'error' and 'axes="xyz"' in str(i) for i in issues)
    record = automatic.realized_geometry()
    dx = float(np.diff(_interior_nodes(record, 0))[0])
    assert any(f'dx={dx:.12g} m' in str(i) for i in issues)
    with pytest.raises(RfxRefusal, match=re.escape(f'dx={dx:.12g} m') +
                       r'[\s\S]*edge_aware_profiles\('):
        automatic.run(n_steps=1)
    profiles = edge_aware_profiles(DOMAIN, dx, solids=[board, plate], axes='xyz')
    sim = Simulation(freq_max=10e9, domain=DOMAIN, boundary='cpml', dx=dx, **profiles)
    sim.add(board, material='fr4')
    sim.add(plate, material='pec')
    assert not [i for i in sim.preflight().errors if i.code == 'realization']


def test_b5_refinement_conflict_at_mesh_resolution():
    sim = _model(.0041, .005, board=False)
    sim.add_refinement((.003, .007), ratio=2, validation='research')
    with pytest.raises(RfxRefusal) as error:
        sim.realized_geometry()
    text = str(error.value)
    for expected in ('geometry[0]', '4.1 mm', 'grading', 'add_refinement',
                     'multiple of dx=', 'pass dx=', 'remove the refinement'):
        assert expected in text


def test_b11_merged_sheet_names_declared_and_solved_planes():
    sim = _model(.00505, .00505)
    issues = _conductor_findings(sim.preflight(), 'geometry[1]')
    assert any(i.severity == 'error' and '0.00505 m' in str(i)
               and '0.005 m' in str(i) and 'merges' in str(i) for i in issues)


def _tmz_model(lo, hi, **kwargs):
    sim = Simulation(freq_max=10e9, domain=(.05, .05, .004),
                     boundary='cpml', mode='2d_tmz', **kwargs)
    sim.add(_box(lo, hi), material='pec')
    sim.add_source((.002, .025, 0), 'ez', waveform=GaussianPulse(f0=5e9),
                   amplitude_kind='field')
    sim.add_probe((.048, .025, 0), 'ez')
    return sim


def test_b12a_two_d_keeps_uniform_resolution():
    sim = _tmz_model(-.0007, .0013)
    report = sim.preflight()
    assert sim._resolve_mesh()['_dz_profile'] is None
    assert not [i for i in report.errors if i.code in ('realization', 'assembly_fault')]


def test_b12b_wall_top_keeps_uniform_resolution():
    sim = Simulation(freq_max=10e9, domain=(.05, .05, .02015), boundary='pec')
    sim.add(_box(.018, .02015), material='pec')
    report = sim.preflight()
    assert sim._resolve_mesh()['_dz_profile'] is None
    assert not report.errors


PERIOD = 20.6 * .00125
PERIODIC_DX = PERIOD / 21


def _periodic_sheet(plane):
    sim = Simulation(freq_max=10e9, domain=(PERIOD, PERIOD, .02),
                     boundary=BoundarySpec(x='periodic', y='periodic', z='cpml'))
    sim.add(Box((0., 0., plane), (PERIOD, PERIOD, plane)), material='pec')
    return sim


@pytest.mark.parametrize('refined', [False, True], ids=['plain', 'refinement'])
def test_b12c_realized_periodic_lattice_keeps_uniform_resolution(refined):
    sim = _periodic_sheet(3 * PERIODIC_DX)
    if refined:
        sim.add_refinement((.009, .015), ratio=2, validation='research')
    report = sim.preflight()
    assert sim._resolve_mesh()['_dz_profile'] is None
    assert not report.errors


@pytest.mark.parametrize('dx', [None, .001], ids=['automatic', 'explicit'])
def test_b13_two_d_box_judge_completes(dx):
    sim = _tmz_model(0., .004, dx=dx)
    report = sim.preflight()
    assert not [i for i in report.errors if i.code == 'assembly_fault']


@pytest.mark.parametrize('dx', [.0001, None], ids=['explicit', 'automatic'])
def test_b14_touching_face_roundoff_is_a_tie(dx):
    sim = _model(board=False, dx=dx)
    sim.add(_box(.0005, .0005 + .0016), material='fr4')
    sim.add(_box(.0021, .0021), material='pec')
    assert not [i for i in _conductor_findings(sim.preflight(), 'geometry[1]')
                if i.severity == 'error']


@pytest.mark.parametrize('plane', [.0075, .0055], ids=['air', 'board'])
def test_b15_thin_conductor_interval_contains_realized_node(plane):
    sim = _model()
    lo, hi = plane - 35e-6 / 2, plane + 35e-6 / 2
    sim.add_thin_conductor(_box(lo, hi), sigma_bulk=1e7, thickness=35e-6)
    report = sim.preflight()
    nodes = _interior_nodes(sim.realized_geometry())
    assert np.any((nodes >= lo - 1e-12) & (nodes <= hi + 1e-12))
    assert not _conductor_findings(report, 'thin_conductor[0]')


def test_b16_sheet_neighbour_sets_acceptance_length():
    sim = _model(.00653, .00653, dx=.0005)
    issues = _conductor_findings(sim.preflight(), 'geometry[1]')
    assert any(i.severity == 'error' and 'sheet plane miss' in str(i)
               and 'L=7e-05 m' in str(i) for i in issues)


def test_b17_split_block_keeps_quarter_cell_floor():
    sim = _model(.005104, .005104)
    sim.preflight()
    nodes = _interior_nodes(sim.realized_geometry())
    assert np.min(abs(nodes - .005104)) <= 1e-12
    assert np.min(np.diff(nodes)) >= .0001 - 1e-12


def test_b18_periodic_rebuild_keeps_realized_xy_lattice():
    plane = 3 * .00125
    sim = _periodic_sheet(plane)
    report = sim.preflight()
    record = sim.realized_geometry()
    assert np.min(abs(_interior_nodes(record) - plane)) <= 1e-12
    for axis in (0, 1):
        d = record.domain[axis]
        cells = np.asarray(record.cell_sizes[axis])[d.padding[0]:d.padding[0] + d.n_cells]
        np.testing.assert_allclose(cells, PERIODIC_DX, rtol=1e-12, atol=0.)
    assert not [i for i in _conductor_findings(report, 'geometry[0]') if i.severity == 'error']


def test_b19_zero_node_sheet_is_reported_by_preflight():
    sim = _model()
    sim.add(Box((.0203, .0203, .0075), (.0204, .0204, .0075)), material='pec')
    report = sim.preflight()
    assert any(i.code == 'pec_zero_cells' and i.severity == 'error' for i in report)


@pytest.mark.parametrize('resolve_first', [True, False], ids=['resolved-first', 'port-first'])
def test_b20_floquet_declaration_updates_realized_lattice(resolve_first):
    sim = Simulation(freq_max=10e9, domain=(PERIOD, PERIOD, .02), boundary='cpml')
    plane = 3 * .0012262
    sim.add(Box((0., 0., plane), (PERIOD, PERIOD, plane)), material='pec')
    if resolve_first:
        sim._resolve_mesh()
    sim.add_floquet_port(.015, axis='z')
    assert sim._resolve_mesh()['_dz_profile'] is None
    assert not sim.preflight().errors


@pytest.mark.parametrize('plane', [.010, .0098], ids=['b21a-period', 'b21b-off-plane'])
def test_b21_periodic_lossy_sheet_plane(plane):
    sim = Simulation(20e9, (.010, .006, .004), dx=.001, cpml_layers=0, snap='strict',
                     boundary=BoundarySpec(x='periodic', y='pec', z='pec'))
    sim.add_thin_conductor(Box((plane, .001, .001), (plane, .005, .003)),
                           thickness=35e-6, sigma_bulk=1e4)
    issues = [i for i in sim.preflight() if i.severity == 'error' and i.code == 'realization']
    if plane == .010:
        assert not issues
    else:
        assert any('thin_conductor[0]' in str(i) and 'δ=0.0002 m' in str(i)
                   and 'L=0.01 m' in str(i) for i in issues)


@pytest.mark.parametrize('plane', [.010, .0047], ids=['b22a-period', 'b22b-off-plane'])
def test_b22_periodic_z_keeps_uniform_resolution(plane):
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .010),
                     boundary=BoundarySpec(x='pec', y='pec', z='periodic'))
    sim.add(Box((0., 0., plane), (.02, .02, plane)), material='pec')
    assert sim._resolve_mesh()['_dz_profile'] is None
    issues = [i for i in sim.preflight() if i.severity == 'error' and i.code == 'realization']
    if plane == .010:
        assert not issues
    else:
        assert any('geometry[0]' in str(i) and 'z sheet plane miss' in str(i)
                   for i in issues)

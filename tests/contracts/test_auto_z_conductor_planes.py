"""Declared conductor planes and refusals before any time step."""
import numpy as np
import pytest

from rfx import Box, RfxRefusal, Simulation
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

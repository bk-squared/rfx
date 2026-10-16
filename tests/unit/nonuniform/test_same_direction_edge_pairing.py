"""Close sheet tops: D - r = 0 gives r = D, with a 10% grading margin."""
import re

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx import mesh_edges
from rfx.mesh_edges import EDGE_OFFSET, SheetEdge, edge_aware_profile, edge_aware_profiles

DOMAIN = (.030, .030, .006)
DX = .001


def _sheets(top):
    return [Box((.004, .005, .003), (.014, .015, .003)),
            Box((.015, .005, .003), (.025, top, .003))]


def _disable_pairing(monkeypatch):
    original = mesh_edges._chain_pairs

    def without(*args, **kwargs):
        kwargs['same_direction'] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(mesh_edges, '_chain_pairs', without)


@pytest.mark.parametrize('top, minimum', [(.0155, .00045), (.01525, .000225)])
def test_close_tops(top, minimum):
    sheets = _sheets(top)
    profiles = edge_aware_profiles(DOMAIN, DX, sheets=sheets, axes='xyz')
    cells = profiles['dy_profile']
    assert cells.min() >= minimum
    nodes = np.r_[0., np.cumsum(cells)]
    for edge in (.015, top):
        i = np.searchsorted(nodes, edge, side='right') - 1
        assert nodes[i] + EDGE_OFFSET * (nodes[i + 1] - nodes[i]) == pytest.approx(edge, abs=1e-12)
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary='pec',
                     snap='strict', **profiles)
    for sheet in sheets:
        sim.add(sheet, material='pec')
    issues = [issue.to_dict() for issue in sim.preflight()]
    assert not [i for i in issues if i['code'] == 'sheet_effective_size' and i['severity'] == 'error']
    for issue in issues:
        if issue['code'] == 'off_lattice_design_edges':
            assert not re.search(r"geometry\[[01]\] '[^']*' \(sheet\) y:", issue['message'])


@pytest.mark.parametrize('sheet', [0, 1])
@pytest.mark.parametrize('top', [.016, .0155, .01525])
def test_single_sheet_unchanged(monkeypatch, sheet, top):
    kwargs = dict(sheets=[_sheets(top)[sheet]], axes='xyz')
    actual = edge_aware_profiles(DOMAIN, DX, **kwargs)
    _disable_pairing(monkeypatch)
    expected = edge_aware_profiles(DOMAIN, DX, **kwargs)
    for axis in actual:
        assert actual[axis].tobytes() == expected[axis].tobytes()


def test_strip_gap_unchanged(monkeypatch):
    # The existing chain fixture: two 16.35 mm strips, 1.15 mm gap,
    # 10 mm margin at both ends; all adjacent directions alternate.
    positions = np.array([10., 26.35, 27.5, 43.85]) * .001
    edges = [SheetEdge(p, 1 if i % 2 == 0 else -1) for i, p in enumerate(positions)]
    actual = edge_aware_profile(0., .05385, DX, sheet_edges=edges)
    _disable_pairing(monkeypatch)
    expected = edge_aware_profile(0., .05385, DX, sheet_edges=edges)
    for a, b in zip(actual, expected):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def _edges(*pairs):
    return [SheetEdge(p * .001, s) for p, s in pairs]


def test_paired_profile_not_used_when_it_is_worse(monkeypatch):
    # Sheets on y 5..10 and 8.75..11.75 mm: pairing both steps gives a
    # 0.5625 mm smallest cell, below the 0.735 mm of the unpaired profile.
    edges = _edges((5., 1), (8.75, 1), (10., -1), (11.75, -1))
    actual = edge_aware_profile(0., .030, DX, sheet_edges=edges)
    _disable_pairing(monkeypatch)
    expected = edge_aware_profile(0., .030, DX, sheet_edges=edges)
    for a, b in zip(actual, expected):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_paired_build_failure_keeps_unpaired_profile(monkeypatch):
    edges = _edges((5., 1), (15., -1), (15.5, -1))
    paired = edge_aware_profile(0., .030, DX, sheet_edges=edges)
    _disable_pairing(monkeypatch)
    expected = edge_aware_profile(0., .030, DX, sheet_edges=edges)
    monkeypatch.undo()
    assert paired.cells.min() > expected.cells.min()  # the drawing reaches the pairing
    original = mesh_edges._build

    def failing(*args, same_direction=False, **kwargs):
        if same_direction:
            raise RuntimeError('band fit did not converge')
        return original(*args, same_direction=same_direction, **kwargs)

    monkeypatch.setattr(mesh_edges, '_build', failing)
    actual = edge_aware_profile(0., .030, DX, sheet_edges=edges)
    for a, b in zip(actual, expected):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_paired_profile_not_used_when_it_adds_cells(monkeypatch):
    # Pairing the two steps joins every strip into one chain sharing one
    # edge cell: the smallest cell rises by under 1 % (0.0379 -> 0.0382 mm)
    # but the axis gets 318 cells instead of 110, so the run would cost more.
    edges = _edges((9.4, 1), (10.7, 1), (15.5, -1), (18.1, -1), (18.15, 1), (20.3, -1))
    kwargs = dict(sheet_edges=edges, boundary_cell=DX)
    actual = edge_aware_profile(0., .0257, DX, **kwargs)
    _disable_pairing(monkeypatch)
    expected = edge_aware_profile(0., .0257, DX, **kwargs)
    for a, b in zip(actual, expected):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_equal_smallest_cell_keeps_the_unpaired_profile(monkeypatch):
    # Both profiles have 41 cells and the same smallest cell, but different
    # nodes: a tie keeps the profile without pairing, bit for bit.
    edges = _edges((11.680410395, 1), (15.250706531, 1), (19.287643817, -1), (23.738570959, -1))
    kwargs = dict(sheet_edges=edges, faces=(.030272982073, .035701041445))
    actual = edge_aware_profile(0., .03711581947841022, DX, **kwargs)
    _disable_pairing(monkeypatch)
    expected = edge_aware_profile(0., .03711581947841022, DX, **kwargs)
    for a, b in zip(actual, expected):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_paired_profile_with_a_smaller_cell_is_not_used(monkeypatch):
    # Here the paired profile has one cell fewer per smallest cell but a
    # smaller smallest cell (0.5207 vs 0.5267 of a 2.5 mm cell): the time
    # step would shrink, so it is not used.
    edges = [SheetEdge(.024281894731, 1), SheetEdge(.02762771539, 1),
             SheetEdge(.030269626333, -1), SheetEdge(.038522909468, -1)]
    kwargs = dict(sheet_edges=edges, faces=(.012444978249126602, .03274230881937341),
                  edge_offset=.3)
    actual = edge_aware_profile(0., .09001548760046064, .0025, **kwargs)
    _disable_pairing(monkeypatch)
    expected = edge_aware_profile(0., .09001548760046064, .0025, **kwargs)
    for a, b in zip(actual, expected):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()

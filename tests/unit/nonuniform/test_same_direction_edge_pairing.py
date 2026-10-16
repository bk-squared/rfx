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

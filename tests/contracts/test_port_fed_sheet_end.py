"""P1–P3: port-fed sheet end rows, with coordinate-owned expectations."""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import SheetSpec
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.mesh_edges import SolvedSheetSpan, solved_sheet_span
from rfx.model import conductors as C


def guide_board(width=.0015, *, graded=False, behind=False):
    profile = np.full(32, .00025)
    profile[2:4] = [.0002, .0003]
    sim = Simulation(freq_max=10e9, domain=(.013, .008, .003), dx=.00025,
        cpml_layers=6, boundary=BoundarySpec(x='cpml', y='cpml',
        z=Boundary(lo='pec', hi='cpml')),
        **({'dy_profile': profile} if graded else {}))
    sim.add_material('sub', eps_r=3.66)
    sim.add(Box((0, 0, 0), (.013, .008, .00075)), material='sub')
    sim.add_thin_conductor(Box((.0025 if behind else .0035, .004-width/2, .00075),
                              (.0095, .004+width/2, .00075)))
    for x, direction, name in ((.0035, '+x', 'p1'), (.0095, '-x', 'p2')):
        sim.add_msl_port((x, .004, 0), width=width, height=.00075,
                         direction=direction, eps_r_sub=3.66, name=name)
    return sim


def realize(sim):
    graded = sim._uses_nonuniform_mesh
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    return C.realized_conductors(sim, grid, nonuniform=graded)


@pytest.mark.parametrize('width', [.001675, .0015])
def test_c1_guide_verdict(width):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = guide_board(width)
        report = sim.preflight()
    rows = report.by_code('sheet_effective_size')
    print('\nC1', width, '\n' + '\n'.join(map(str, rows)))
    if width == .001675:
        assert not rows
    else:
        row, = rows
        assert str(row).startswith('1 conductor sheet dimension(s)')
        assert ' y: drawn 1.5mm' in str(row)
        assert 'solved as 1.675mm (+11.67%)' in str(row)
    span = sim._campaign_ctx().realized().sheets[0].solved_spans[0]
    assert (span.free_lo, span.free_hi) == (False, False)
    assert span.hi-span.lo == pytest.approx(.006)


def test_c3_end_behind_feed_is_free():
    root = realize(guide_board(behind=True))
    span = root.sheets[0].solved_spans[0]
    assert (span.free_lo, span.free_hi) == (True, False)
    assert span.lo == pytest.approx(.0025 - .35*.00025)
    assert span.hi == pytest.approx(.0095)


@pytest.mark.parametrize('x', [.004, .006])
def test_c4_one_lumped_node_on_six_node_end_row(x):
    sim = Simulation(freq_max=5e9, domain=(.014, .014, .008), dx=.001, boundary='pec')
    sim.add_thin_conductor(Box((.004, .004, .004), (.010, .009, .004)))
    sim.add_port((x, .006, .003), 'ez', impedance=50.)
    span = realize(sim).sheets[0].solved_spans[0]
    assert (span.free_lo, span.free_hi) == (True, True)
    assert span.lo == pytest.approx(.00365)
    assert span.hi == pytest.approx(.01035)


def test_c4_one_node_wide_end_is_fed():
    # A sheet sampled at one y node (drawn width smaller than one cell).
    sim = Simulation(freq_max=5e9, domain=(.014, .014, .008), dx=.001, boundary='pec')
    sim.add_thin_conductor(Box((.004, .0059, .004), (.010, .0061, .004)))
    sim.add_port((.004, .006, .003), 'ez', impedance=50.)
    root = realize(sim)
    assert np.flatnonzero(np.asarray(root.sheets[0].footprint).any(axis=(0, 2))).size == 1
    span = root.sheets[0].solved_spans[0]
    assert (span.free_lo, span.free_hi) == (False, True)
    assert span.lo == pytest.approx(.004)
    assert span.hi == pytest.approx(.01035)


def test_p1_does_not_preview_or_solve_laplace(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('fed nodes must not preview ports or solve a Laplace profile')
    monkeypatch.setattr(C, 'preview_port_stages', forbidden)
    monkeypatch.setattr('rfx.sources.msl_port.compute_msl_mode_profile', forbidden)
    root = realize(guide_board(graded=True))
    assert root.sheets[0].solved_spans[0].free_lo is False


def test_c5_graded_fringe_is_not_fed():
    sim = guide_board(graded=True)
    sim.add_thin_conductor(Box((.0035, .005, .00075), (.007, .0055, .00075)))
    root = realize(sim)
    preview = C.preview_port_stages(sim, root)
    # Independent physical coordinates: trace y nodes 3.25..4.75 mm;
    # adjacent sheet y nodes 5..5.5 mm, in the Laplace fringe support.
    from rfx.realized_geometry import _node_arrays
    _, nodes = _node_arrays(sim, root.grid, True)
    fringe = [edge for _, component, edge in preview.released_port_edges
              if component == 'ez' and .005-1e-12 <= nodes[1][edge[1]] <= .0055+1e-12]
    assert fringe, 'fixture must actually intersect published Laplace fringe edges'
    fed = C.port_fed_nodes(sim, root)
    assert not any(fed[edge] for edge in fringe)
    span = root.sheets[1].solved_spans[0]
    assert (span.free_lo, span.free_hi) == (True, True)
    assert span.lo == pytest.approx(.0035 - .35*.00025)
    assert span.hi == pytest.approx(.007 + .35*.00025)


@pytest.mark.parametrize('periodic,shape', [(False, (12, 12, 8)),
                                           (True, (12, 12, 8)),
                                           (False, (12, 12, 1))])
def test_c7_index_declared_no_port_bit_equal(periodic, shape):
    sheet = SheetSpec.from_node_ranges(shape, normal_axis=2,
        plane=0 if shape[2] == 1 else 4, in_plane_ranges=((3, 8), (3, 8)))
    nodes = np.arange(12)*.001
    before = solved_sheet_span(sheet.footprint, 0, nodes, .003, .008, .012,
                               periodic=periodic)
    after = solved_sheet_span(sheet.footprint, 0, nodes, .003, .008, .012,
                              periodic=periodic, fed=None)
    assert after == before == SolvedSheetSpan(3, 8, .003-.35*.001,
                                             .008+.35*.001, True, True)
    assert np.asarray(after, dtype=np.float64).tobytes() == np.asarray(before, dtype=np.float64).tobytes()


def test_c7_pinned_no_port_bit_equal(monkeypatch):
    sim = Simulation(freq_max=5e9, domain=(.012, .012, .008), dx=.001, boundary='pec')
    sim.add_pinned_sheet(plane_index=4, i_range=(3, 8), j_range=(3, 8))
    actual = realize(sim).sheets[0].solved_spans
    assert C.port_fed_nodes(sim, realize(sim)) is None
    monkeypatch.setattr(C, 'port_fed_nodes', lambda *args: None)
    assert realize(sim).sheets[0].solved_spans == actual
    assert actual[0] == SolvedSheetSpan(3, 8, .003-.35*.001, .008+.35*.001, True, True)

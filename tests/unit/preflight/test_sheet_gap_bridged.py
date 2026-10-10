"""Drawn sheet gaps are judged against the production PEC edge union (#1373)."""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.boundaries.spec import BoundarySpec


CODE = "sheet_gap_bridged"


def model(intervals=((.005, .010), (.011, .015)), **kwargs):
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=1e-3,
                     boundary=kwargs.pop("boundary", "pec"), **kwargs)
    for y0, y1 in intervals:
        sheet(sim, y0, y1)
    return sim


def sheet(sim, y0, y1, *, x=.010, sigma_bulk=5.8e7, **kwargs):
    sim.add_thin_conductor(Box((x, y0, .005), (x, y1, .015)),
                           sigma_bulk=sigma_bulk, thickness=35e-6, **kwargs)


def test_t1_one_cell_gap():
    finding, = model().preflight().by_code(CODE)
    assert finding.severity == "error"
    assert finding.source == "_validate_cfg_sheet_gap_bridged"
    message = str(finding)
    for expected in ("thin_conductor[0] and thin_conductor[1] lie on one plane",
                     "drawn 1mm apart", "11 Ey edge(s)", "ONE conductor", "tracker 1373"):
        assert expected in message


def test_t2_half_cell_gap():
    assert len(model(((.005, .010), (.0105, .015))).preflight().by_code(CODE)) == 1


@pytest.mark.parametrize("join", [.0095, .0094, .0096])
def test_t3_off_mesh_abutting(join):
    assert not model(((.005, join), (join, .015))).preflight().by_code(CODE)


@pytest.mark.parametrize("end", [.010, .011])
def test_t4_abutting_or_overlapping(end):
    assert not model(((.005, end), (.010, .015))).preflight().by_code(CODE)


@pytest.mark.parametrize("start", [.0115, .012])
def test_t5_open_gaps(start):
    assert not model(((.005, .010), (start, .015))).preflight().by_code(CODE)


@pytest.mark.parametrize("normal", ["same", "different"])
def test_t6_different_planes_or_normals(normal):
    sim = model(((.005, .010),))
    if normal == "same":
        sheet(sim, .011, .015, x=.012)
    else:
        sim.add_thin_conductor(Box((.005, .011, .005), (.015, .011, .015)),
                               sigma_bulk=5.8e7, thickness=35e-6)
    assert not sim.preflight().by_code(CODE)


def test_t7_three_sheets():
    findings = model(((.003, .006), (.007, .010), (.011, .014))).preflight().by_code(CODE)
    assert len(findings) == 2
    assert "thin_conductor[0] and thin_conductor[1]" in str(findings[0])
    assert "thin_conductor[1] and thin_conductor[2]" in str(findings[1])
    assert all("thin_conductor[0] and thin_conductor[2]" not in str(row) for row in findings)


def test_t8_graded_local_cell_gap():
    sim = model((), dy_profile=np.array([.001] + [.0009, .0011] * 9 + [.001]))
    nodes = sim._campaign_ctx().nodes[1]
    # Nodes 5, 10, 11, 15 are 5, 9.9, 11, 15 mm: gap is the 1.1 mm cell.
    np.testing.assert_allclose(nodes[[5, 10, 11, 15]], [.005, .0099, .011, .015], atol=1e-9)
    sheet(sim, float(nodes[5]), float(nodes[10]))
    sheet(sim, float(nodes[11]), float(nodes[15]))
    finding, = sim.preflight().by_code(CODE)
    assert "11 Ey edge(s)" in str(finding)
    assert "local cell 1.1mm" in str(finding)


def test_t9_run_gate_and_skip_and_strict():
    sim = model()
    sim.add_source((.002, .002, .002), component="ez")
    sim.add_probe((.003, .002, .002), component="ez")
    with pytest.raises(ValueError, match="sheet_gap_bridged|solved as ONE conductor"):
        sim.run(n_steps=2, compute_s_params=False)
    result = sim.run(n_steps=2, compute_s_params=False, skip_preflight=True)
    assert result is not None
    with pytest.raises(ValueError, match="sheet_gap_bridged|solved as ONE conductor"):
        sim.preflight(strict=True)


def test_t10_periodic_seam_matches_realized_masks():
    sim = model(((.015, .020), (.001, .005)),
                boundary=BoundarySpec(x="pec", y="periodic", z="pec"))
    ctx = sim._campaign_ctx()
    sheets = [entry.sheet for entry in ctx.entry_realizations() if entry.kind == "sheet"]
    pair = realized_pec_edge_masks(None, sheets, periodic=ctx.periodic)
    a = realized_pec_edge_masks(None, sheets[:1], periodic=ctx.periodic)
    b = realized_pec_edge_masks(None, sheets[1:], periodic=ctx.periodic)
    counts = [int(np.count_nonzero(np.asarray(pair[t]) & ~(np.asarray(a[t]) | np.asarray(b[t]))))
              for t in (1, 2)]
    findings = sim.preflight().by_code(CODE)
    assert bool(findings) == bool(sum(counts))
    assert counts == [0, 0]  # Measured seam convention: no bridging edges.


@pytest.mark.parametrize("kwargs", [{"sigma_bulk": 1e3}, {"surface_impedance_f0": 10e9}])
def test_t11_lossy_and_f0_excluded(kwargs):
    sim = model(())
    sheet(sim, .005, .010, **kwargs)
    sheet(sim, .011, .015, **kwargs)
    assert not sim.preflight().by_code(CODE)


@pytest.mark.parametrize("first_route,labels", [
    ("thin", "geometry[0] and thin_conductor[0]"),
    ("add", "geometry[0] and geometry[1]"),
])
def test_n1_both_declaration_routes(first_route, labels):
    sim = model(())
    if first_route == "thin":
        sheet(sim, .005, .010)
    else:
        sim.add(Box((.01, .005, .005), (.01, .010, .015)), material="pec")
    sim.add(Box((.01, .011, .005), (.01, .015, .015)), material="pec")
    finding, = sim.preflight().by_code(CODE)
    # The classifier lists geometry declarations before thin conductors.
    assert str(finding).startswith(labels + " lie on one plane")
    assert "11 Ey edge(s)" in str(finding)


@pytest.mark.parametrize("cover", [(.009, .012), (.010, .011), (.004, .016)])
def test_n2_gap_covered_by_one_sheet(cover):
    sim = model()
    sheet(sim, *cover)
    assert not sim.preflight().by_code(CODE)


def test_n2_partial_cover_counts_only_unowned_edges():
    sim = model()
    sim.add_thin_conductor(Box((.01, .009, .005), (.01, .012, .010)),
                           sigma_bulk=5.8e7, thickness=35e-6)
    # A-B has 11 Ey edges at z nodes 5..15. C owns nodes 5..10 inclusive,
    # hence 6 of these edges: 11 - 6 = 5 remain at z nodes 11..15.
    ctx = sim._campaign_ctx()
    sheets = [entry.sheet for entry in ctx.entry_realizations() if entry.kind == "sheet"]
    pair = realized_pec_edge_masks(None, sheets[:2], periodic=ctx.periodic)
    own = np.zeros(ctx.grid.shape, dtype=bool)
    for sp in sheets:
        own |= np.asarray(realized_pec_edge_masks(None, [sp], periodic=ctx.periodic)[1])
    assert np.count_nonzero(np.asarray(pair[1]) & ~own) == 5
    finding, = sim.preflight().by_code(CODE)
    assert str(finding).startswith("thin_conductor[0] and thin_conductor[1] lie on one plane")
    assert "5 Ey edge(s)" in str(finding)


@pytest.mark.parametrize("boxes,component,normal", [
    ((((.005, .01, .005), (.015, .01, .010)),
      ((.005, .01, .011), (.015, .01, .015))), "Ez", "y"),
    ((((.005, .005, .01), (.010, .015, .01)),
      ((.011, .005, .01), (.015, .015, .01))), "Ex", "z"),
])
def test_n3_other_normal_and_gap_axes(boxes, component, normal):
    sim = model(())
    for corners in boxes:
        sim.add_thin_conductor(Box(*corners), sigma_bulk=5.8e7, thickness=35e-6)
    finding, = sim.preflight().by_code(CODE)
    assert f"one plane ({normal} = 10mm)" in str(finding)
    assert "drawn 1mm apart" in str(finding)
    assert f"11 {component} edge(s)" in str(finding)


@pytest.mark.parametrize("intervals", [((.005, .010), (.011, .015)),
                                       ((.011, .015), (.005, .010))])
def test_n4_gap_format_in_both_declaration_orders(intervals):
    finding, = model(intervals).preflight().by_code(CODE)
    assert "drawn 1mm apart" in str(finding)


def test_n5_swapped_box_corners():
    sim = model(((.005, .010),))
    sim.add_thin_conductor(Box((.01, .015, .015), (.01, .011, .005)),
                           sigma_bulk=5.8e7, thickness=35e-6)
    report = sim.preflight()
    assert not report.by_code(CODE)
    refusal, = report.by_code("pec_zero_cells")
    assert "thin_conductor[1]" in str(refusal)
    assert "footprint rasterizes to ZERO nodes" in str(refusal)


def test_a_gap_that_contains_a_node_line_is_open():
    """[5, 9.9] + [10.1, 15] mm: node 10 lies inside the 0.2-cell gap."""
    sim = model(((.005, .0099), (.0101, .015)))
    assert not sim.preflight().by_code(CODE)

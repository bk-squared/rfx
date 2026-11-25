"""Microstrip port clearance to the absorber, read from the solved strip edge: preflight-only pins."""

import math

import numpy as np
import pytest

from rfx import Boundary, BoundarySpec, Box, Simulation


U = 2.0**-13
EDGE_OFFSET = 0.35


def _findings(sim, diagnostic_id):
    return [finding for finding in sim.preflight(strict=False)
            if finding.code == "msl_port_geometry"
            and finding.diagnostic.code == diagnostic_id]


def _board(cells, *, height=3 * U, feed=8 * U, reverse=False, wall=None):
    """A sheet on a declared mesh; reverse mirrors the graded width profile."""
    cells = np.asarray(cells, dtype=float)
    low = float(sum(cells[:2]))
    high = float(sum(cells[:6]))
    extent = float(sum(cells))
    if reverse:
        low, high = extent - high, extent - low
        cells = cells[::-1].copy()
    sim = Simulation(
        snap="declared", freq_max=5e9, domain=(32 * U, extent, 12 * U),
        dx=U, dy_profile=cells, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y=wall or "cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=3.66)
    sim.add(Box((0, 0, 0), (32 * U, extent, height)), material="substrate")
    sim.add(Box((0, low, height), (32 * U, high, height)), material="pec")
    sim.add_msl_port(
        (feed, (low + high) / 2, 0), width=high - low, height=height,
        direction="+x", name="pin", n_probe_offset=5, n_probe_spacing=2,
    )
    return sim


def test_pin1_thru_line_32_layers_has_no_clearance_advisory():
    """The _run_msl_thru(0.008) sheet board, without a field solve."""
    height, width, margin, length = 254e-6, 600e-6, 2e-3, 8e-3
    cell = height / 3
    ly = width + 2 * (2 * height + 8 * cell)
    lx = length + 2 * margin
    sim = Simulation(
        snap="declared", freq_max=5e9, domain=(lx, ly, height + 1.5e-3),
        dx=cell, cpml_layers=32,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, height)), material="ro4350b")
    low, high = (ly - width) / 2, (ly + width) / 2
    sim.add(Box((0, low, height), (lx, high, height)), material="pec")
    for feed, direction in ((margin, "+x"), (margin + length, "-x")):
        sim.add_msl_port((feed, ly / 2, 0), width=width, height=height,
                         direction=direction, impedance=50)
    lateral_low = math.ceil(low / cell) * cell - EDGE_OFFSET * cell
    lateral_high = (math.ceil(ly / cell) * cell
                    - math.floor(high / cell) * cell - EDGE_OFFSET * cell)
    feed_low = round(margin / cell) * cell
    feed_high = math.ceil(lx / cell) * cell - round((margin + length) / cell) * cell
    assert min(lateral_low, lateral_high) > max((2.2 / 3) * height, 2.2 * cell)
    assert min(feed_low, feed_high) > max((4 / 3) * height, 4 * cell)
    findings = sim.preflight(strict=False)
    assert not [finding for finding in findings
                if finding.code == "msl_port_geometry" and finding.diagnostic.code in {
                    "msl.lateral_clearance", "msl.source_absorber_clearance"}]


def test_pin2_below_lateral_and_feed_thresholds_warns():
    cells = [U] * 16
    sim = _board(cells, feed=3 * U)
    lateral = _findings(sim, "msl.lateral_clearance")
    feed = _findings(sim, "msl.source_absorber_clearance")
    expected_distance = sum(cells[:2]) - EDGE_OFFSET * cells[1]
    expected_lateral = max((2.2 / 3) * (3 * U), 2.2 * cells[0])
    expected_feed = max((4 / 3) * (3 * U), 4 * U)
    assert len(lateral) == len(feed) == 1
    assert lateral[0].diagnostic.values["side"] == "−y"
    assert lateral[0].diagnostic.values["lateral_clearance_m"] == pytest.approx(expected_distance)
    assert lateral[0].diagnostic.values["recommended_clearance_m"] == pytest.approx(expected_lateral)
    assert feed[0].diagnostic.values["source_clearance_m"] == pytest.approx(3 * U)
    assert feed[0].diagnostic.values["recommended_clearance_m"] == pytest.approx(expected_feed)


@pytest.mark.parametrize("reverse", [False, True], ids=["lo", "hi"])
def test_pin3_solved_edge_warns_when_last_node_clears_threshold(reverse):
    cells = [U, 1.25 * U] + [1.25 * U] * 14
    sim = _board(cells, reverse=reverse)
    node_distance = sum(cells[:2])
    solved_distance = node_distance - EDGE_OFFSET * cells[1]
    threshold = max((2.2 / 3) * (3 * U), 2.2 * cells[0])
    assert node_distance == 2.25 * U
    assert solved_distance == 1.8125 * U
    assert solved_distance < threshold <= node_distance
    hits = _findings(sim, "msl.lateral_clearance")
    assert len(hits) == 1
    assert hits[0].diagnostic.values["side"] == ("+y" if reverse else "−y")
    assert hits[0].diagnostic.values["lateral_clearance_m"] == pytest.approx(solved_distance)
    assert hits[0].diagnostic.values["recommended_clearance_m"] == pytest.approx(threshold)


def test_pin4_unmeasured_mesh_uses_end_cell_threshold_and_says_so():
    cells = [U] * 16
    height = 2 * U
    lateral = _findings(_board(cells, height=height, feed=3 * U), "msl.lateral_clearance")
    feed = _findings(_board(cells, height=height, feed=3 * U), "msl.source_absorber_clearance")
    expected_distance = sum(cells[:2]) - EDGE_OFFSET * cells[1]
    expected_lateral = max((2.2 / 3) * height, 2.2 * cells[0])
    expected_feed = max((4 / 3) * height, 4 * U)
    assert len(lateral) == len(feed) == 1
    for finding, threshold in ((lateral[0], expected_lateral), (feed[0], expected_feed)):
        values = finding.diagnostic.values
        assert values["recommended_clearance_m"] == pytest.approx(threshold)
        assert values["mesh_height_ratio"] == height / cells[0]
        assert "measured at 3 cells per h_sub" in str(finding)
        assert "This mesh (2 cells per h_sub) was not measured" in str(finding)
    assert lateral[0].diagnostic.values["lateral_clearance_m"] == pytest.approx(expected_distance)
    assert lateral[0].diagnostic.values["threshold_end_cells"] == 2.2
    assert feed[0].diagnostic.values["threshold_end_cells"] == 4


@pytest.mark.parametrize("wall", ["pec", "pmc"])
def test_wall_retains_two_heights_from_solved_edge(wall):
    cells = [U] * 16
    hits = _findings(_board(cells, wall=wall), "msl.lateral_clearance")
    assert len(hits) == 1
    assert hits[0].diagnostic.values["recommended_clearance_m"] == 2 * (3 * U)
    assert hits[0].diagnostic.values["wall"] == wall.upper()


def test_feed_at_threshold_is_not_below_it():
    assert not _findings(_board([U] * 16, feed=4 * U), "msl.source_absorber_clearance")

"""Wire terminal separation is measured on actual conductor planes."""
import pytest

from rfx import Box, Simulation

DX = 1e-3


def _model(axis, side, gap, *, reverse=False):
    sim = Simulation(freq_max=5e9, domain=(12*DX,)*3, dx=DX,
                     boundary="pec", cpml_layers=0)
    for lower, upper in ((2, 3), (9, 10)):
        lo, hi = [0., 0., 0.], [12*DX]*3
        lo[axis], hi[axis] = lower*DX, upper*DX
        sim.add(Box(tuple(lo), tuple(hi)), material="pec")
    start = 3+gap if side < 0 else 3
    end = 9-gap if side > 0 else 9
    position = [6*DX]*3
    position[axis] = (end if reverse else start)*DX
    extent = (end-start)*DX*(-1 if reverse else 1)
    sim.add_port(position=tuple(position), component="e"+"xyz"[axis],
                 impedance=50., extent=extent)
    return sim


@pytest.mark.parametrize("axis", [0, 1, 2], ids=list("xyz"))
@pytest.mark.parametrize("side", [-1, 1], ids=["lower", "upper"])
@pytest.mark.parametrize("gap", [0, 1, 2, 3])
def test_every_resolved_gap_is_reported_on_its_actual_side(axis, side, gap):
    report = _model(axis, side, gap).preflight(check_ntff=False)
    hits = report.by_code("wire_port_end_gap_to_conductor")
    assert len(hits) == (0 if gap == 0 else 1)
    if gap:
        assert hits[0].severity == "warning"
        text = str(hits[0])
        assert ("+" if side > 0 else "-")+"xyz"[axis]+"-side end node" in text
        assert f"{gap} cell(s)" in text
        assert f"gap = {gap*DX:g} m" in text
        assert "nearest" in text
        assert "capacitive only" not in text  # geometry cannot establish this


@pytest.mark.parametrize("side", [-1, 1])
def test_negative_extent_preserves_the_same_physical_gap(side):
    positive = _model(2, side, 2).preflight(check_ntff=False)
    negative = _model(2, side, 2, reverse=True).preflight(check_ntff=False)
    for report in (positive, negative):
        (hit,) = report.by_code("wire_port_end_gap_to_conductor")
        assert "gap = 0.002 m" in str(hit)


def test_the_nearest_wall_wins_over_a_further_plane():
    sim = _model(2, -1, 3)
    # The source starts at z=6dx. A sheet at 4dx is closer than the
    # existing volume's upper face at 3dx, and owns no volume cell.
    sim.add(Box((0, 0, 4*DX), (12*DX, 12*DX, 4*DX)), material="pec")
    (hit,) = sim.preflight(check_ntff=False).by_code("wire_port_end_gap_to_conductor")
    assert "2 cell(s)" in str(hit)
    assert "gap = 0.002 m" in str(hit)


def test_an_off_column_conductor_does_not_create_a_gap_finding():
    sim = Simulation(freq_max=5e9, domain=(12*DX,)*3, dx=DX, boundary="pec")
    sim.add(Box((0, 0, 2*DX), (2*DX, 2*DX, 3*DX)), material="pec")
    sim.add_port(position=(6*DX, 6*DX, 5*DX), component="ez",
                 impedance=50., extent=2*DX)
    assert not sim.preflight(check_ntff=False).by_code("wire_port_end_gap_to_conductor")


@pytest.mark.parametrize("gap", [0, 2, 3])
@pytest.mark.parametrize("side", [-1, 1])
def test_axial_filament_contact_does_not_need_a_tangential_wall(gap, side):
    from rfx.geometry.csg import PolylineWire
    from tests._realized_geometry import realized
    sim = Simulation(freq_max=5e9, domain=(12*DX,)*3, dx=DX, boundary="pec")
    wire_lo, wire_hi = (1, 3) if side < 0 else (9, 11)
    target_edge, target_node = (2, 3) if side < 0 else (9, 9)
    sim.add(PolylineWire(points=((6*DX, 6*DX, wire_lo*DX), (6*DX, 6*DX, wire_hi*DX)),
                         radius=.1*DX), material="pec")
    port_start = 3+gap if side < 0 else 6-gap
    sim.add_port(position=(6*DX, 6*DX, port_start*DX), component="ez",
                 impedance=50., extent=3*DX)
    rz = realized(sim)
    assert rz.edge_masks[2][6, 6, target_edge]
    assert not rz.edge_masks[0][6, 6, target_node] and not rz.edge_masks[1][6, 6, target_node]
    hits = sim.preflight(check_ntff=False).by_code("wire_port_end_gap_to_conductor")
    assert len(hits) == (0 if gap == 0 else 1)
    if gap:
        assert f"gap = {gap*DX:g} m" in str(hits[0])
        assert "ez-edge endpoint" in str(hits[0])

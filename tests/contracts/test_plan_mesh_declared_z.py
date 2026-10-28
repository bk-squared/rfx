"""A plan places both margins; automatic meshes solve the declared column."""
import numpy as np
import pytest

from rfx import Box, Simulation, auto_configure, plan_mesh
from rfx.boundaries.spec import Boundary, BoundarySpec


@pytest.mark.parametrize("z0", [0.0, 0.05])
def test_plan_places_both_margins(z0):
    plan = plan_mesh(
        [(Box((0, 0, z0), (.02, .02, z0 + .0016)), "board")],
        (1.5e9, 3.5e9), materials={"board": {"eps_r": 4.4}},
        accuracy="standard")
    margin = 0.25 * 299792458.0 / 1.5e9
    assert plan.offset[2] == pytest.approx(margin - z0, rel=1e-9)
    profile = plan.to_sim_kwargs()["dz_profile"]
    nodes = np.r_[0.0, np.cumsum(profile)]
    assert nodes[-1] == pytest.approx((2 * margin + .0016), rel=1e-9)
    assert nodes[-1] == pytest.approx(plan.domain[2], rel=1e-9)
    lower = nodes[np.argmin(abs(nodes - margin))]
    upper = nodes[np.argmin(abs(nodes - (margin + .0016)))]
    assert lower == pytest.approx(margin, rel=1e-9)
    assert upper == pytest.approx((margin + .0016), rel=1e-9)
    assert lower == pytest.approx(plan.margin, rel=1e-9)
    assert nodes[-1] - upper == pytest.approx(margin, rel=1e-9)


def test_auto_mesh_keeps_declared_z_and_drawn_faces():
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .01), boundary="pec")
    sim.add_material("board", eps_r=4.4)
    sim.add(Box((0, 0, .01 / 3), (.02, .02, .01 / 3 + .0016)), material="board")
    grid = sim._build_realized_grid()
    cells = grid.cells("z")[grid.pad_z_lo:grid.nz - grid.pad_z_hi - 1]
    nodes = np.r_[0.0, np.cumsum(cells)]
    assert nodes[-1] == pytest.approx(.01, rel=1e-9)
    assert nodes[np.argmin(abs(nodes - (.01 / 3)))] == pytest.approx(
        (.01 / 3), rel=1e-9)
    assert nodes[np.argmin(abs(nodes - (.01 / 3 + .0016)))] == pytest.approx(
        (.01 / 3 + .0016), rel=1e-9)


def test_clipped_features_leave_an_exact_empty_column():
    config = auto_configure(
        [(Box((0, 0, .02), (.02, .02, .0216)), "board")],
        (1.5e9, 3.5e9), materials={"board": {"eps_r": 4.4}},
        domain=(.02, .02, .0093))
    assert config.offset == (0.0, 0.0, 0.0)
    assert np.sum(config.dz_profile) == pytest.approx(.0093, rel=1e-9)


def test_board_flush_on_z_absorber_warns_once():
    sim = Simulation(freq_max=3.5e9, domain=(.02, .02, .01), dx=.0004,
                     boundary="cpml", cpml_layers=2)
    sim.add_material("board", eps_r=4.4)
    sim.add(Box((0, 0, 0), (.02, .02, .0016)), material="board")
    issues = [i for i in sim.preflight() if i.code == "body_flush_on_absorber"]
    assert len(issues) == 1
    assert "z-lo" in issues[0].loc
    assert "1 geometry entries" in issues[0]
    assert "half-space" in issues[0]


def test_msl_substrate_through_x_absorber_is_silent():
    sim = Simulation(freq_max=3.5e9, domain=(.02, .02, .01), dx=.0004,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")),
                     cpml_layers=2)
    sim.add_material("board", eps_r=4.4)
    sim.add(Box((0, 0, 0), (.02, .02, .0016)), material="board")
    sim.add(Box((.002, .008, .0016), (.018, .012, .0016)), material="pec")
    assert not [i for i in sim.preflight() if i.code == "body_flush_on_absorber"]


@pytest.mark.parametrize("family", ["debye", "lorentz"])
def test_a_pole_carrying_board_on_the_absorber_is_left_to_the_dispersive_advisory(family):
    # Its hi pad stays background (#808) and its lo pad drops the poles, so
    # "continues this body's material" would misstate the pad; the dispersive
    # advisory reports that face with the real mechanism.
    from rfx.materials.debye import DebyePole
    from rfx.materials.lorentz import LorentzPole
    poles = (dict(debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)]) if family == "debye"
             else dict(lorentz_poles=[LorentzPole(omega_0=2e10, delta=1e9, kappa=2e10 ** 2)]))
    sim = Simulation(freq_max=3.5e9, domain=(.02, .02, .01), dx=.0004,
                     boundary="cpml", cpml_layers=2)
    sim.add_material("board", eps_r=4.4, **poles)
    sim.add(Box((0, 0, 0), (.02, .02, .0016)), material="board")
    codes = [i.code for i in sim.preflight()]
    assert "dispersive_pole_at_absorber_face" in codes
    assert "body_flush_on_absorber" not in codes


@pytest.mark.parametrize("face, z_lo, fires", [
    # ends 0.7 cell short of the z-hi face: the node one cell in is the board's,
    # and the hi-face promotion (#638/#655) continues it into the pad
    ("z-hi", .01 - .0016 - .7 * .0004, True),
    # starts 0.3 cell above the z-lo face: the face node is vacuum, so is the pad
    ("z-lo", .3 * .0004, False),
])
def test_a_box_face_window_follows_its_node_sampling(face, z_lo, fires):
    sim = Simulation(freq_max=3.5e9, domain=(.02, .02, .01), dx=.0004,
                     boundary="cpml", cpml_layers=2)
    sim.add_material("board", eps_r=4.4)
    sim.add(Box((0, 0, z_lo), (.02, .02, z_lo + .0016)), material="board")
    locs = [i.loc for i in sim.preflight() if i.code == "body_flush_on_absorber"]
    assert any(face in loc for loc in locs) is fires

"""cv05's two conductors land where the script declares them (#931, X-A).

Nothing in this repo asserted WHERE cv05's ground and patch actually were.
The realized planes existed only as prose inside a committed VESSL run log
(``_05_patch_results/cv05_run_openems_369367257743.log``), and a log line is
not a gate: the case ran for months with the two conductors on 12.0000 and
13.8161 mm — a 1.8161 mm cavity against a 1.5 mm laminate, its top 455 um
air rather than FR4 — and nothing went red.

This is the cheapest check that can catch that class: build cv05's two
Simulations, realize the PEC edge set through the single owner
(``realized_pec_edge_masks`` / ``realized_wall_planes``), and compare the
planes and footprints against the declaration. No FDTD.

The script itself carries the assertion (``assert_realized_sheets``, run on
both builds before any solve), so this file drives the script rather than
re-deriving the geometry — a second copy of the declaration here would be a
second rule to keep in sync, which is the defect #931 exists to remove. The
script is executed as a subprocess in ``RFX_CV05_BUILD_ONLY=1`` mode and
writes its measured stack as JSON; the expectations below are the DECLARED
board (1.5 mm FR4, six 250 um cells, a 60 x 55 mm ground), not numbers
copied out of a run.

Runtime: about 25 s per arm on CPU (two builds and two rasterizations, no
time stepping).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CV05 = REPO_ROOT / "validation" / "crossval" / "05_patch_antenna.py"

# The DECLARED board, from the script's own parameter block. Restated here so
# the gate fails when the realization stops matching the declaration, whichever
# of the two moved.
H_SUB_MM = 1.5
N_SUB = 6
DZ_SUB_MM = 0.25
GROUND_MM = (60.0, 55.0)
PATCH_DECLARED_MM = (29.5, 38.0)
# The patch's four in-plane faces sit half a 1 mm cell off the lattice, so the
# closed footprint rule (#931 §1.3) realizes 28 x 37 edge-lengths from that
# declaration. That is mesh resolution on a 1 mm grid, not a realization rule,
# and it is pinned here so a change to either is visible.
PATCH_REALIZED_MM = (28.0, 37.0)


def _run(tmp_path: Path, delta: int | None = None) -> dict:
    out = tmp_path / "realized.json"
    env = dict(os.environ)
    env["RFX_CV05_BUILD_ONLY"] = "1"
    env["RFX_CV05_REALIZED_JSON"] = str(out)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if delta is not None:
        env["RFX_CV05_SHEET_PLANE_DELTA"] = str(delta)
    proc = subprocess.run([sys.executable, str(CV05)], env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise AssertionError(
            f"cv05 build-only exited {proc.returncode}\n"
            f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-4000:]}")
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def realized(tmp_path_factory) -> dict:
    return _run(tmp_path_factory.mktemp("cv05_realized"))


@pytest.mark.parametrize("leg", ["no_port", "with_port"])
def test_sheets_are_one_plane_each(realized, leg):
    """Each conductor is realized on exactly ONE node plane.

    A sheet owns no cell and is realized at one plane (#931 §1.3). Two planes
    for one conductor means a volume was declared; that is how the pre-#931
    ``two_plane`` flag and the one-cell PEC Box differ from a foil.
    """
    st = realized[leg]
    assert st["substrate_cells_between"] == N_SUB
    assert st["ground"]["k"] + N_SUB == st["patch"]["k"]


@pytest.mark.parametrize("leg", ["no_port", "with_port"])
def test_cavity_is_the_declared_laminate(realized, leg):
    """The two walls stand exactly h_sub apart with FR4 in between.

    This is the +21.1 % error stated as a gate. Before the migration the walls
    were 1.8161 mm apart on a 1.5 mm substrate and the top 455 um of that
    cavity was vacuum, because both conductors were sub-cell PEC Boxes placed
    by a nearest-node argmin over a graded mesh.
    """
    st = realized[leg]
    assert st["cavity_physical_mm"] == pytest.approx(H_SUB_MM, abs=1e-9)
    assert st["cavity_node_to_node_mm"] == pytest.approx(H_SUB_MM, abs=1e-6), (
        "realized cavity is not the declared laminate thickness")
    eps = st["cavity_eps_r"]
    assert len(eps) == N_SUB + 1
    # Tolerance is float32 storage resolution, not an accuracy allowance: the
    # question is "FR4 or vacuum", a factor of 4.3 apart.
    assert all(e == pytest.approx(4.3, abs=1e-5) for e in eps[:-1]), (
        f"a cavity node carries the wrong medium: {eps}")
    assert eps[-1] == pytest.approx(1.0, abs=1e-5), (
        "the node at the patch plane is not air; the substrate leaks above the patch")


def test_port_does_not_open_a_hole_in_either_sheet(realized):
    """Adding the probe feed removes no wall edge from either conductor.

    §1.9 as amended: a port releases the ONE component it drives, at its own
    cells. Releasing the two tangential edges at the port foot would punch a
    hole in the ground plane the port never asked for. Same planes, same
    footprints, with and without the port is the observable form of that.
    """
    a, b = realized["no_port"], realized["with_port"]
    for name in ("ground", "patch"):
        for key in ("k", "x_edge_cells", "y_edge_cells",
                    "x_index_range", "y_index_range",
                    "realized_x_lo_mm", "realized_x_hi_mm",
                    "realized_y_lo_mm", "realized_y_hi_mm"):
            assert a[name][key] == b[name][key], (
                f"the port changed {name}.{key}: {a[name][key]} -> {b[name][key]}")


def _external_geometry_module():
    import importlib.util
    path = CV05.with_name("_patch_external_geometry.py")
    spec = importlib.util.spec_from_file_location("cv05_external_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_sheet_edges_retain_the_complete_realized_extent(realized):
    board = realized["external_board_mm"]
    assert board["ground"]["lo"] == pytest.approx([50, 50, 25])
    assert board["ground"]["hi"] == pytest.approx([110, 105, 25])
    assert board["patch"]["lo"] == pytest.approx([66, 59, 26.5])
    assert board["patch"]["hi"] == pytest.approx([94, 96, 26.5])
    assert board["substrate"]["lo"] == pytest.approx([50, 50, 25])
    assert board["substrate"]["hi"] == pytest.approx([110, 105, 26.5])
    for name in ("ground", "patch"):
        for axis, key in enumerate(("realized_x_mm", "realized_y_mm")):
            assert board[name]["hi"][axis] - board[name]["lo"][axis] == pytest.approx(
                realized["with_port"][name][key])
    # Relative feed-to-edge distances must survive the solver-frame change.
    for terminal, source in (("lo", "realized_start_m"), ("hi", "realized_end_m")):
        assert board["feed"][terminal] == pytest.approx([
            v * 1000 + shift for v, shift in zip(
                realized["feed_check"][source], board["translation_mm"])])


def test_external_air_margin_translates_every_object_together(realized):
    module = _external_geometry_module()
    moved = module.external_patch_board(
        realized["with_port"], realized["feed_check"], margin_mm=75.0,
        ground_z_mm=25.)
    for name in ("ground", "patch", "substrate", "feed"):
        for corner in ("lo", "hi"):
            old = realized["external_board_mm"][name][corner]
            assert moved[name][corner] == pytest.approx([old[0]+25, old[1]+25, old[2]])


def _assert_mesh_contains_every_board_feature(record):
    board = record["external_board_mm"]
    lines = record["external_mesh_lines_mm"]
    for name in ("ground", "patch", "substrate", "feed"):
        for corner in ("lo", "hi"):
            for axis, label in enumerate("xyz"):
                assert board[name][corner][axis] in lines[label], (name, corner, label)


def test_external_mesh_preserves_faces_and_terminals(realized):
    _assert_mesh_contains_every_board_feature(realized)
    module = _external_geometry_module()
    lines = {k: list(v) for k, v in realized["external_mesh_lines_mm"].items()}
    module.assert_external_patch_mesh(realized["external_board_mm"], lines)
    lines["x"].remove(realized["external_board_mm"]["patch"]["hi"][0])
    with pytest.raises(ValueError, match="mesh lost patch.hi.x"):
        module.assert_external_patch_mesh(realized["external_board_mm"], lines)


def test_external_runs_cannot_reuse_legacy_or_previous_board_files(tmp_path):
    module = _external_geometry_module()
    for name in ("port_ut_1", "port_it_1", "et", "ht"):
        (tmp_path / name).write_text("legacy board")
    called = []

    class Solver:
        def Run(self, path, **kwargs):
            called.append(path)
            assert not (Path(path) / "port_ut_1").exists()
            (Path(path) / "port_ut_1").write_text("new board")

    first = module.run_openems_reference(Solver(), tmp_path)
    second = module.run_openems_reference(Solver(), tmp_path)
    assert called == [first, second]
    assert first != second
    assert (tmp_path / "port_ut_1").read_text() == "legacy board"
    assert (Path(first) / "port_ut_1").read_text() == "new board"
    assert (Path(second) / "port_ut_1").read_text() == "new board"


def test_external_ground_at_the_domain_floor_is_rejected(realized):
    import numpy as np
    module = _external_geometry_module()
    board = realized["external_board_mm"]
    # Max-res regular lines are the conservative PML-thickness limit.
    lines = dict(x=np.linspace(0, 160, 65), y=np.linspace(0, 155, 63),
                 z=np.linspace(0, 66.5, 28))
    module.assert_external_absorber_clearance(board, lines, pml_layers=8)
    old = module.external_patch_board(realized["with_port"], realized["feed_check"],
                                      margin_mm=50., ground_z_mm=0.)
    with pytest.raises(ValueError, match="ground intersects z PML"):
        module.assert_external_absorber_clearance(old, lines, pml_layers=8)


def _check_native_xml(path, board):
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    for property_name, geometry_name in (
        ("FR4", "substrate"), ("ground", "ground"), ("patch", "patch"),
        ("port_resist_1", "feed"), ("port_excite_1", "feed"), ("port_ut_1", "feed"),
    ):
        boxes = tree.findall(f".//*[@Name='{property_name}']/Primitives/Box")
        assert len(boxes) == 1
        for xml_corner, corner in (("P1", "lo"), ("P2", "hi")):
            point = boxes[0].find(xml_corner)
            assert [float(point.attrib[axis]) for axis in "XYZ"] == pytest.approx(
                board[geometry_name][corner], abs=1e-10, rel=0)


def test_retained_native_board_and_feed_match_the_current_realization(realized):
    path = REPO_ROOT / "tests/fixtures/patch_sheet_board_native_xml/board_delta0.xml"
    _check_native_xml(path, realized["external_board_mm"])


@pytest.mark.parametrize("sheet_delta", [0, 1])
def test_native_openems_keeps_the_transferred_board(tmp_path, realized, monkeypatch, sheet_delta):
    """Optional native constructor/XML check; no electromagnetic time stepping."""
    import numpy as np

    pytest.importorskip("CSXCAD")
    pytest.importorskip("openEMS")
    from CSXCAD.CSXCAD import ContinuousStructure
    from CSXCAD.SmoothMeshLines import SmoothMeshLines
    from openEMS.openEMS import openEMS

    for name, value in (("float", float), ("int", int), ("complex", complex)):
        monkeypatch.setattr(np, name, value, raising=False)
    record = realized if sheet_delta == 0 else _run(tmp_path, delta=sheet_delta)
    module = _external_geometry_module()
    board = record["external_board_mm"]
    csx = ContinuousStructure()
    fdtd = openEMS(NrTS=25000, EndCriteria=1e-7)
    fdtd.SetGaussExcite(2.4e9, 1.0e9)
    fdtd.SetBoundaryCond(["PML_8"]*6)
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    for axis in "xyz":
        mesh.SetLines(axis, SmoothMeshLines(
            np.asarray(record["external_mesh_lines_mm"][axis]), 2.5, ratio=1.4))
    native_lines = {axis: np.asarray(mesh.GetLines(axis)) for axis in "xyz"}
    module.assert_external_patch_mesh(board, native_lines)
    module.assert_external_absorber_clearance(board, native_lines, pml_layers=8)
    module.add_openems_patch_board(csx, fdtd, board, eps_r=4.3)
    out_dir = Path(os.environ.get("RFX_CV05_NATIVE_DIR", str(tmp_path)))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cv05-delta{sheet_delta}.xml"
    csx.Write2XML(str(path))
    _check_native_xml(path, board)


def test_openems_receives_the_inspected_board_and_rejects_a_detached_feed(realized):
    from copy import deepcopy
    module = _external_geometry_module()
    boxes, properties, ports = {}, {}, []

    class Property:
        def __init__(self, name):
            self.name = name

        def AddBox(self, lo, hi, **kwargs):
            boxes[self.name] = {"lo": lo, "hi": hi}

        def SetMaterialProperty(self, **kwargs):
            properties[self.name] = kwargs

    class CSX:
        AddMaterial = staticmethod(Property)
        AddMetal = staticmethod(Property)

    class FDTD:
        @staticmethod
        def AddLumpedPort(**kwargs):
            ports.append(kwargs)
            return kwargs

    board = realized["external_board_mm"]
    module.add_openems_patch_board(CSX(), FDTD(), board, eps_r=4.3)
    assert boxes == {"FR4": board["substrate"], "ground": board["ground"],
                     "patch": board["patch"]}
    assert properties == {"FR4": {"epsilon": 4.3}}
    assert ports[0]["start"] == board["feed"]["lo"]
    assert ports[0]["stop"] == board["feed"]["hi"]
    detached = deepcopy(board)
    detached["feed"]["lo"][0] -= 40
    with pytest.raises(ValueError, match="feed misses ground"):
        module.add_openems_patch_board(CSX(), FDTD(), detached, eps_r=4.3)


def test_registered_port_reaches_the_measured_sheets(realized):
    stack, feed = realized["with_port"], realized["feed_check"]
    assert feed["galvanic"]
    assert feed["z0_node_k"] == stack["ground"]["k"]
    assert feed["z1_node_k"] == stack["patch"]["k"]


@pytest.mark.parametrize("leg", ["no_port", "with_port"])
def test_ground_footprint_is_the_drawn_rectangle(realized, leg):
    """The 60 x 55 mm ground realizes 60 x 55 mm.

    Its corners are on the lattice, so the closed footprint rule realizes the
    drawn rectangle exactly. It used to realize 58 x 54: the hi row was lost to
    half-open node sampling, and in x a second cell went because
    ``(dom_x - gx) / 2`` evaluated one f64 route-ulp ABOVE the 10 mm node while
    the identical expression in y landed one ulp below. Same declaration, two
    realizations, on two axes of one box.
    """
    g = realized[leg]["ground"]
    assert (g["realized_x_mm"], g["realized_y_mm"]) == pytest.approx(GROUND_MM)
    assert (g["declared_x_mm"], g["declared_y_mm"]) == pytest.approx(GROUND_MM)


@pytest.mark.parametrize("leg", ["no_port", "with_port"])
def test_patch_footprint_is_the_half_cell_debt_and_says_so(realized, leg):
    """The patch realizes 28 x 37 mm from a 29.5 x 38.0 mm declaration.

    Not a defect the contract fixes: L and W are the antenna's design
    dimensions; openEMS now consumes the measured footprint. The declared
    design remains in the record, separately from the antenna actually solved.
    """
    p = realized[leg]["patch"]
    assert (p["declared_x_mm"], p["declared_y_mm"]) == pytest.approx(PATCH_DECLARED_MM)
    assert (p["realized_x_mm"], p["realized_y_mm"]) == pytest.approx(PATCH_REALIZED_MM)


def test_sheet_plane_falsifier_moves_the_cavity(tmp_path):
    """The +/-1 node arm reproduces the defect class the contract removes.

    cv04 ships ``thickness_plus_cell`` / ``thickness_minus_cell``; cv05 shipped
    only a mis-realized-LENGTH arm, which perturbs a design dimension rather
    than the realization rule. ``RFX_CV05_SHEET_PLANE_DELTA=1`` puts the ground
    one plane below the laminate floor and the patch one plane above its top:
    the walls are no longer the laminate faces and vacuum sits in series with
    the board. The measurement, not a label, is what makes this a falsifier -
    so the arm must MOVE the cavity, and the build assertion must still pass
    (the geometry is legal, just not the declared board).
    """
    record = _run(tmp_path, delta=1)
    _assert_mesh_contains_every_board_feature(record)
    _check_native_xml(
        REPO_ROOT / "tests/fixtures/patch_sheet_board_native_xml/board_delta1.xml",
        record["external_board_mm"])
    st = record["no_port"]
    assert st["substrate_cells_between"] == N_SUB + 2
    assert st["cavity_node_to_node_mm"] > H_SUB_MM + DZ_SUB_MM, (
        "the sheet-plane falsifier did not move the cavity; it cannot bound "
        "how much the post-contract resonance is allowed to move")
    assert st["cavity_eps_r"][0] == pytest.approx(1.0, abs=1e-5), (
        "the arm was supposed to put a vacuum cell inside the cavity")
    assert record["feed_check"]["z0_node_k"] == st["ground"]["k"]
    assert record["feed_check"]["z1_node_k"] == st["patch"]["k"]


def test_the_old_cv05_declaration_is_refused_and_names_the_sheet_api():
    """cv05's pre-#931 conductor declaration cannot come back silently.

    Both conductors were 250 um PEC Boxes. On cv05's mesh that is 0.25 of a
    cell at the ground and 0.55 at the patch — neither a volume nor a sheet,
    and the old code resolved it with a third rule (csg's thin branch: the
    single cell whose centre is nearest the mid-plane), which is what put the
    ground on a node OUTSIDE its own drawn window and the patch 316 um above
    the laminate.

    Under §1.5 that Box is refused, and the refusal must NAME the sheet
    declaration — an error that only says "too thin" leaves the author to
    guess, and guessing is what produced the compensations this branch
    deletes. Cheap and self-contained: a five-cell box on a 1 mm grid, no
    solve, no cv05 build.
    """
    from rfx import Box, Simulation
    from rfx.boundaries.spec import BoundarySpec

    sim = Simulation(freq_max=4e9, domain=(20e-3, 20e-3, 20e-3), dx=1.0e-3,
                     boundary=BoundarySpec.uniform("cpml"), cpml_layers=4)
    sim.add(Box((5e-3, 5e-3, 10e-3), (15e-3, 15e-3, 10.25e-3)), material="pec")
    with pytest.raises(ValueError) as exc:
        sim.conductor_mask()
    msg = str(exc.value)
    assert "thinner than one cell" in msg
    assert "add_thin_conductor" in msg, (
        f"the refusal does not name the sheet API: {msg}")

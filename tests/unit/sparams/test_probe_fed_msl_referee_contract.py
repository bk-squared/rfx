"""Contract tests for the #498 openEMS referee lane.

openEMS is NOT installed on this pod, so everything here is either pure
arithmetic or a subprocess of the script's own openEMS-free paths. The
tests that matter:

  * the STAGE CONTRACT -- Stage 2 refuses to run unless Stage 1 ran and
    passed BOTH reproduce legs (fail-before-fix: with
    ``assert_stage1_gate_passed`` neutered, ``test_stage2_refuses_*`` and
    ``test_cli_stage_2_without_stage1_record_exits_4`` fail; see the PR
    body for the quoted red run);
  * the DE-EMBEDDING arithmetic, on planted data with a known answer;
  * the MESH/GEOMETRY self-check, pure numpy;
  * that the module imports and ``--dry-run``s with no openEMS present.

Nothing here asserts a physics number. No lumped/wire diagonal value is
pinned by any test in this file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "diagnostics" / "probe_fed_msl_openems_referee.py"
_YAML = _REPO_ROOT / "scripts" / "vessl_probe_fed_msl_referee.yaml"
_PREDECLARATION = (
    _REPO_ROOT / "docs" / "design_notes" / "mixed_refplane_predeclaration.md"
)


_SMOOTHER_CACHE = []


def _smoother():
    """The referee's own vendored CSXCAD smoother, loaded once."""
    if not _SMOOTHER_CACHE:
        _SMOOTHER_CACHE.append(_load_referee()._csxcad_smooth_mesh_lines)
    return _SMOOTHER_CACHE[0]


def _load_referee():
    spec = importlib.util.spec_from_file_location("_probe_fed_msl_referee_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref():
    return _load_referee()


# ---------------------------------------------------------------------------
# Import-guarded structure
# ---------------------------------------------------------------------------
def test_module_imports_without_openems(ref):
    """The referee must be importable (and therefore testable) on a pod
    with no openEMS: the solver import is deferred into _import_openems."""
    assert "openEMS" not in sys.modules
    assert "CSXCAD" not in sys.modules
    assert callable(ref._import_openems)


def test_dry_run_prints_stage_plan_and_geometry_without_openems():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--dry-run", "--stage", "both"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    # the stage plan
    assert "STAGE 1: reproduce-gate" in out
    assert "MSL_NotchFilter.py" in out
    assert "Simple_Patch_Antenna.py" in out
    assert "STAGE 2: the DUT (REFUSED unless Stage 1 ran and passed)" in out
    # the geometry it WOULD build
    assert "GEOMETRY IT WOULD BUILD" in out
    assert "BOTH ENDS OPEN" in out
    assert "MSL meas plane   x = 4.72 mm" in out
    assert "WHAT CANNOT BE COMPARED" in out
    assert "no openEMS was imported, no geometry was built" in out


def test_self_check_cli_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--self-check"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "self-check passed: True" in proc.stdout
    assert "FAIL" not in proc.stdout


# ---------------------------------------------------------------------------
# THE STAGE CONTRACT (fail-before-fix target)
# ---------------------------------------------------------------------------
def _passing_leg():
    return {"status": "RUN", "passed": True}


def _passing_stage1():
    return {"a1": _passing_leg(), "a2": _passing_leg(), "passed": True}


def test_stage2_refuses_when_stage1_absent(ref):
    with pytest.raises(ref.Stage1GateError, match="no Stage 1 reproduce-gate record"):
        ref.assert_stage1_gate_passed(None)


def test_stage2_refuses_when_a_leg_is_only_recorded_not_run(ref):
    """A leg carrying the historical numbers but status='RECORDED' is NOT a
    pass -- the gate must demand an actual run."""
    s1 = _passing_stage1()
    s1["a2"] = {"status": "RECORDED", "passed": True}
    with pytest.raises(ref.Stage1GateError, match=r"\['a2'\]"):
        ref.assert_stage1_gate_passed(s1)


def test_stage2_refuses_when_one_leg_fails_its_gate(ref):
    s1 = _passing_stage1()
    s1["a1"] = {"status": "RUN", "passed": False}
    with pytest.raises(ref.Stage1GateError, match=r"\['a1'\]"):
        ref.assert_stage1_gate_passed(s1)


def test_stage2_refuses_when_both_legs_missing(ref):
    with pytest.raises(ref.Stage1GateError, match=r"\['a1', 'a2'\]"):
        ref.assert_stage1_gate_passed({"passed": True})


def test_stage2_refuses_when_overall_passed_is_false(ref):
    s1 = _passing_stage1()
    s1["passed"] = False
    with pytest.raises(ref.Stage1GateError, match="overall 'passed'"):
        ref.assert_stage1_gate_passed(s1)


def test_stage2_accepts_only_when_both_legs_ran_and_passed(ref):
    ref.assert_stage1_gate_passed(_passing_stage1())  # must not raise


def test_stage1_leg_passed_predicate(ref):
    assert ref.stage1_leg_passed({"status": "RUN", "passed": True})
    assert not ref.stage1_leg_passed({"status": "RUN", "passed": False})
    assert not ref.stage1_leg_passed({"status": "RECORDED", "passed": True})
    assert not ref.stage1_leg_passed(None)
    assert not ref.stage1_leg_passed("RUN")


def test_cli_stage_2_without_stage1_record_exits_4(tmp_path):
    """End-to-end: --stage 2 with no Stage-1 PASS record must exit 4, build
    NO geometry, and write an artifact whose stage2 is null."""
    out = tmp_path / "referee.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--stage", "2", "--output", str(out)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert proc.returncode == 4, (proc.returncode, proc.stdout, proc.stderr)
    assert "STAGE 2 REFUSED" in proc.stderr
    artifact = json.loads(out.read_text())
    assert artifact["stage2"] is None
    assert "no Stage 1 reproduce-gate record" in artifact["stage2_refusal"]
    # the reproduce-gate record travels with every artifact
    assert artifact["reproduce_gate_record"]["a1"]["example"].endswith(
        "MSL_NotchFilter.py")
    assert artifact["cannot_compare"]


def test_cli_stage_2_with_a_failing_stage1_json_exits_4(tmp_path):
    s1 = tmp_path / "stage1.json"
    s1.write_text(json.dumps({"stage1": {"a1": {"status": "RUN", "passed": True},
                                         "a2": {"status": "RUN", "passed": False},
                                         "passed": False}}))
    out = tmp_path / "referee.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--stage", "2", "--stage1-json", str(s1),
         "--output", str(out)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert proc.returncode == 4, (proc.returncode, proc.stdout, proc.stderr)
    assert "Stage A FAILED its reproduce-gate" in proc.stderr


def test_cli_stage_1_without_openems_exits_2_and_writes_no_number(tmp_path):
    """On this pod openEMS is absent: Stage 1 must exit 2 with the
    VESSL-only message and must NOT write an artifact carrying a
    reproduced number it never measured."""
    out = tmp_path / "referee.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--stage", "1", "--output", str(out)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "openEMS Python bindings not importable" in proc.stderr
    assert "vessl_probe_fed_msl_referee.yaml" in proc.stderr
    assert not out.exists()


def test_cli_stage_both_without_openems_never_reaches_stage_2(tmp_path):
    out = tmp_path / "referee.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--stage", "both", "--output", str(out)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert proc.returncode == 2
    assert "STAGE 2" not in proc.stdout.split(
        "--- STAGE 1: reproduce-gate (openEMS's own canonical examples) ---")[-1]


# ---------------------------------------------------------------------------
# DE-EMBEDDING ARITHMETIC on planted data
# ---------------------------------------------------------------------------
def test_refer_plane_recovers_a_planted_load(ref):
    """Plant a known load behind a known length of lossless line; the
    de-embedding must return it exactly."""
    beta = np.asarray([40.0, 60.0, 80.0])          # rad/m
    d = 1.92e-3                                     # 4.72 mm -> 2.80 mm
    gamma_true = np.asarray([0.43 + 0.0j, 0.1 - 0.2j, -0.05 + 0.3j])
    s_meas = gamma_true * np.exp(-1j * beta * d * 2.0)   # reflection: twice
    recovered = ref.refer_plane(s_meas, beta, d, 2)
    np.testing.assert_allclose(recovered, gamma_true, atol=1e-12)


def test_refer_plane_is_magnitude_preserving_on_a_lossless_line(ref):
    beta = np.asarray([40.0, 60.0, 80.0])
    s = np.asarray([0.3 + 0.1j, -0.2 + 0.4j, 0.05 - 0.5j])
    out = ref.refer_plane(s, beta, 1.92e-3, 2)
    np.testing.assert_allclose(np.abs(out), np.abs(s), atol=1e-12)


def test_deembed_two_port_recovers_a_planted_thru(ref):
    """An ideal matched thru of length L, measured L1 in front of port 1
    and L2 in front of port 2, must de-embed to S21 = 1, S11 = S22 = 0."""
    beta = np.asarray([50.0, 75.0, 100.0])
    l1, l2 = 0.80e-3, 1.20e-3
    s11 = np.zeros(3, dtype=complex)
    s22 = np.zeros(3, dtype=complex)
    s21 = np.exp(-1j * beta * (l1 + l2))
    s12 = s21.copy()
    got = ref.deembed_two_port(s11, s21, s22, s12, beta1=beta, d1_m=l1,
                               beta2=beta, d2_m=l2)
    np.testing.assert_allclose(got["s21"], np.ones(3), atol=1e-12)
    np.testing.assert_allclose(got["s12"], np.ones(3), atol=1e-12)


def test_deembed_two_port_recovers_a_planted_mismatched_load(ref):
    beta = np.asarray([50.0, 75.0, 100.0])
    l1, l2 = 0.80e-3, 1.20e-3
    g1 = np.asarray([0.2 + 0.0j, 0.2 + 0.0j, 0.2 + 0.0j])
    g2 = np.asarray([0.43 + 0.0j, 0.43 + 0.0j, 0.43 + 0.0j])
    s11 = g1 * np.exp(-2j * beta * l1)
    s22 = g2 * np.exp(-2j * beta * l2)
    got = ref.deembed_two_port(s11, np.zeros(3, complex), s22,
                               np.zeros(3, complex), beta1=beta, d1_m=l1,
                               beta2=beta, d2_m=l2)
    np.testing.assert_allclose(got["s11"], g1, atol=1e-12)
    np.testing.assert_allclose(got["s22"], g2, atol=1e-12)


def test_snap_shift_to_mesh_on_grid_is_exact_and_off_grid_rounds(ref):
    # the fixture's own 0.80 mm shift at the 50 um comparator mesh: 16
    # cells exactly -> a measured no-op, reported not skipped
    assert ref.snap_shift_to_mesh(0.80e-3, 50e-6) == pytest.approx(0.80e-3, abs=1e-15)
    assert ref.snap_shift_to_mesh(0.80e-3, 80e-6) == pytest.approx(0.80e-3, abs=1e-15)
    # off-grid rounds to the nearest line, and the residual is what the
    # effective CalcPort shift then has to carry
    assert ref.snap_shift_to_mesh(0.82e-3, 50e-6) == pytest.approx(0.80e-3, abs=1e-15)
    assert ref.snap_shift_to_mesh(0.84e-3, 50e-6) == pytest.approx(0.85e-3, abs=1e-15)


def test_implied_plane_error_recovers_a_planted_offset(ref):
    beta = np.asarray([50.0, 100.0, 200.0])
    planted_d = 4.0 * 50e-6          # a 4-cell plane-position error
    dphi = beta * planted_d
    np.testing.assert_allclose(ref.implied_plane_error_m(dphi, beta),
                               np.full(3, planted_d), atol=1e-15)


def test_phase_self_consistency_is_zero_on_an_ideal_line_and_finds_a_planted_error(ref):
    beta = np.asarray([50.0, 75.0, 100.0])
    length = 2.72e-3               # 2.00 mm -> 4.72 mm
    s21_ideal = 0.9 * np.exp(-1j * beta * length)
    np.testing.assert_allclose(
        ref.phase_self_consistency_deg(s21_ideal, beta, length),
        np.zeros(3), atol=1e-9)
    planted = 4.0 * 50e-6
    s21_off = 0.9 * np.exp(-1j * beta * (length + planted))
    dev = ref.phase_self_consistency_deg(s21_off, beta, length)
    np.testing.assert_allclose(dev, -np.degrees(beta * planted), atol=1e-9)


# ---------------------------------------------------------------------------
# MESH / GEOMETRY self-check (pure numpy)
# ---------------------------------------------------------------------------
def test_geometry_self_check_passes(ref):
    res = ref.geometry_self_check(verbose=False)
    assert res["passed"], res["failures"]
    assert res["checks"]


def test_realized_board_is_the_one_stage2_builds(ref):
    """The #723 lesson in code: Stage 2's constants are rfx's REALIZED
    board, not its declared one."""
    real = ref.RFX_REALIZED_RECORD["realized"]
    decl = ref.RFX_REALIZED_RECORD["declared"]
    assert ref.B_H_SUB_M == real["h_sub_m"] == pytest.approx(320e-6)
    assert decl["h_sub_m"] == pytest.approx(254e-6)
    assert ref.B_W_TRACE_M == real["w_trace_node_span_m"] == pytest.approx(480e-6)
    assert ref.B_Y_C_M == pytest.approx(1.52e-3)
    # both open ends, and the open stub past the MSL feed plane
    assert ref.B_TRACE_X_LO_M == 0.0 and ref.B_TRACE_X_HI_M == pytest.approx(8.0e-3)
    assert real["conductor_extends_into_pad"] is False
    assert real["dielectric_extends_into_pad"] is True


def test_openems_mesh_plan_reproduces_the_open_ends_and_the_planes(ref):
    plan = ref.openems_mesh_plan(ref.B_DX_COMPARATOR_M)
    # blocker B4: the metal ends at the absorber's inner face, 8 cells from
    # the PML, with the pad OUTSIDE the declared domain -- not a matched
    # termination running into the PML.
    assert plan["trace_x_span_m"] == [0.0, pytest.approx(8.0e-3)]
    assert plan["pml_cells"] == 8
    assert plan["domain_with_pad_m"]["x"][0] == pytest.approx(-8 * 50e-6)
    assert plan["domain_with_pad_m"]["x"][1] == pytest.approx(8.0e-3 + 8 * 50e-6)
    assert plan["open_stub_beyond_msl_feed_m"] == pytest.approx(2.48e-3)
    # the comparator mesh must clear the do_not_repeat 3.175-cell trap
    assert plan["n_substrate_cells"] >= 5
    # every port/plane sits exactly on a mesh line
    for x in (0.0, 2.0e-3, 4.72e-3, 5.52e-3, 8.0e-3):
        assert float(np.min(np.abs(plan["x_lines_m"] - x))) < 1e-12
    assert plan["measplane_shift_target_m"] == pytest.approx(0.80e-3)
    assert abs(plan["effective_calcport_shift_predicted_m"]) < 1e-12


def test_openems_mesh_plan_has_no_ulp_duplicated_lines(ref):
    """A plane of record must appear ONCE, not twice.

    ``openems_mesh_plan`` unions an ``np.arange`` background grid with the
    literal plane-of-record coordinates.  ``np.arange`` reproduces a plane
    that IS on its own grid only to within a ULP, so a plain
    ``np.unique`` keeps BOTH copies and leaves a cell of ~1e-19 m between
    them.  openEMS then builds an operator with a zero-width cell -- every
    field it writes is NaN, which is the ``uf_inc=nan`` VESSL run
    369367257610 died on.  Guarded here for EVERY axis of BOTH mesh legs,
    because the dx=80 um leg has the same pathology on y as well as x.
    """
    for dx_m in (ref.B_DX_COMPARATOR_M, ref.B_DX_REPORTED_ONLY_M):
        plan = ref.openems_mesh_plan(dx_m)
        for axis in ("x", "y", "z"):
            lines = plan[f"{axis}_lines_m"]
            cells = np.diff(lines)
            # Any cell below this is not a mesh feature, it is round-off:
            # the smallest DELIBERATE spacing in this plan is the
            # thirds-rule y line at dx/12, seven orders of magnitude above.
            floor_m = 1e-6 * dx_m
            worst = float(np.min(cells))
            bad = [(int(i), float(lines[i]), float(lines[i + 1]), float(cells[i]))
                   for i in np.where(cells < floor_m)[0]]
            assert not bad, (
                f"dx={dx_m*1e6:.0f} um, {axis} axis: {len(bad)} degenerate "
                f"cell(s) below {floor_m:.3e} m (min cell {worst:.6e} m): "
                f"{bad}"
            )


def test_lumped_port_feed_plane_is_a_single_mesh_line(ref):
    """The failing leg's own port plane, isolated.

    x = 2.00 mm is the lumped probe's feed plane.  ``x_line_present``
    (the self-check that passed) only asks whether SOME line is within
    0 nm of it -- which a DUPLICATED line satisfies.  This asks the
    question that matters instead: how many.
    """
    for dx_m in (ref.B_DX_COMPARATOR_M, ref.B_DX_REPORTED_ONLY_M):
        lines = ref.openems_mesh_plan(dx_m)["x_lines_m"]
        n = int(np.sum(np.abs(lines - ref.B_FEED_X_M) < 1e-6 * dx_m))
        assert n == 1, (
            f"dx={dx_m*1e6:.0f} um: the lumped feed plane x="
            f"{ref.B_FEED_X_M*1e3:.2f} mm is carried by {n} mesh lines, "
            f"not 1 -- a zero-width cell at the port itself"
        )


def test_reported_only_dx80_leg_is_labelled_and_not_the_comparator(ref):
    assert ref.B_DX_COMPARATOR_M == pytest.approx(50e-6)
    assert ref.B_DX_REPORTED_ONLY_M == pytest.approx(80e-6)
    assert "dx=50 um" in ref.REPRODUCE_GATE_RECORD["do_not_repeat"]
    assert "REPORTED ONLY" in ref.REPRODUCE_GATE_RECORD["do_not_repeat"]


def test_rfx_node_index_matches_the_measured_grid(ref):
    # measured on the real rfx grid, quoted verbatim in RFX_REALIZED_RECORD
    assert ref.rfx_node_index(0.0) == 8
    assert ref.rfx_node_index(1.44e-3) == 26
    assert ref.rfx_node_index(2.00e-3) == 33
    assert ref.rfx_node_index(2.80e-3) == 43
    assert ref.rfx_node_index(3.60e-3) == 53
    assert ref.rfx_node_index(4.72e-3) == 67
    assert ref.rfx_node_index(5.50e-3) == 77   # snaps to 5.52 mm
    assert ref.rfx_node_index(8.00e-3 - 1e-9) == 108


# ---------------------------------------------------------------------------
# Stage-2 BUILDER plumbing, against a recording double (openEMS is absent
# on this pod, so the geometry it WOULD build is verified structurally)
# ---------------------------------------------------------------------------
class _FakeProp:
    def __init__(self, name, sink):
        self.name, self._sink = name, sink

    def AddBox(self, start, stop, priority=0, **kw):
        self._sink.append({"prop": self.name, "start": list(start),
                           "stop": list(stop), "priority": priority})


class _FakeMesh:
    def __init__(self):
        self.unit = None
        self.lines = {}
        self.smoothed = []

    def SetDeltaUnit(self, u):
        self.unit = u

    def AddLine(self, ax, lines):
        self.lines.setdefault(ax, []).extend(np.atleast_1d(lines).tolist())

    def SmoothMeshLines(self, ax, res, ratio):
        self.smoothed.append((ax, res, ratio))
        # Smooth with the SAME routine the plan predicts with, so the builder's
        # own prediction-vs-real cross-check is genuinely exercised rather than
        # trivially satisfied by a crude stand-in.
        cur = np.asarray(sorted(set(self.lines.get(ax, []))), dtype=float)
        if cur.size >= 2 and res > 0:
            self.lines[ax] = sorted(
                _smoother()(cur, res, ratio).tolist())

    def GetLines(self, ax):
        return np.asarray(sorted(set(self.lines.get(ax, []))), dtype=float)


class _FakeCSX:
    def __init__(self):
        self.boxes = []
        self._mesh = _FakeMesh()
        self.props = []

    def GetGrid(self):
        return self._mesh

    def AddMaterial(self, name, **kw):
        self.props.append(("material", name, kw))
        return _FakeProp(name, self.boxes)

    def AddMetal(self, name):
        self.props.append(("metal", name, {}))
        return _FakeProp(name, self.boxes)


class _FakeLumpedPort:
    def __init__(self, rec):
        self.rec = rec


class _FakeFDTD:
    def __init__(self, NrTS=None, EndCriteria=None):
        self.nrts, self.end_criteria = NrTS, EndCriteria
        self.boundary = None
        self.gauss = None
        self.csx = None
        self.lumped = None

    def SetGaussExcite(self, f0, fc):
        self.gauss = (f0, fc)

    def SetBoundaryCond(self, bc):
        self.boundary = list(bc)

    def SetCSX(self, csx):
        self.csx = csx

    def AddLumpedPort(self, nr, R, start, stop, p_dir, excite, priority=0):
        self.lumped = {"nr": nr, "R": R, "start": list(start),
                       "stop": list(stop), "dir": p_dir, "excite": excite,
                       "priority": priority}
        return _FakeLumpedPort(self.lumped)


class _FakeMSLPort:
    instances: list = []

    def __init__(self, csx, port_nr=None, metal_prop=None, start=None,
                 stop=None, prop_dir=None, exc_dir=None, excite=None,
                 Feed_R=None, MeasPlaneShift=None, priority=0):
        self.start = list(start)
        self.stop = list(stop)
        self.prop_dir, self.exc_dir = prop_dir, exc_dir
        self.excite, self.Feed_R = excite, Feed_R
        self.measplane_shift = MeasPlaneShift
        self.port_nr = port_nr
        _FakeMSLPort.instances.append(self)


@pytest.mark.parametrize("drive", ["lumped", "msl"])
def test_stage2_builder_plumbing_against_a_recording_double(ref, drive):
    _FakeMSLPort.instances = []
    fdtd, lumped, msl, plan, port_info = ref._build_stage2(
        _FakeCSX, _FakeFDTD, _FakeMSLPort, dx_m=ref.B_DX_COMPARATOR_M,
        drive=drive, nrts=1000, end_criteria=1e-4)

    # rfx's boundary topology, cell for cell: PEC at z_lo, PML_8 elsewhere
    assert fdtd.boundary == ["PML_8", "PML_8", "PML_8", "PML_8", "PEC", "PML_8"]
    boxes = {b["prop"]: b for b in fdtd.csx.boxes}

    # BLOCKER B4: the conductor spans EXACTLY 0..8.00 mm and stops at the
    # absorber's inner face -- it does NOT run into the PML.
    trace = boxes["trace"]
    assert trace["start"][0] == pytest.approx(0.0)
    assert trace["stop"][0] == pytest.approx(8.0)          # mm
    pad_mm = ref.B_PML_CELLS * ref.B_DX_COMPARATOR_M * 1e3
    assert trace["start"][0] > plan["domain_with_pad_m"]["x"][0] * 1e3
    assert trace["stop"][0] < plan["domain_with_pad_m"]["x"][1] * 1e3
    assert trace["stop"][0] + pad_mm == pytest.approx(
        plan["domain_with_pad_m"]["x"][1] * 1e3)

    # ... while the DIELECTRIC *is* edge-replicated through the pad, which
    # is what rfx measurably does.
    sub = boxes["ro4350b"]
    assert sub["start"][0] == pytest.approx(plan["domain_with_pad_m"]["x"][0] * 1e3)
    assert sub["stop"][0] == pytest.approx(plan["domain_with_pad_m"]["x"][1] * 1e3)
    assert sub["stop"][2] == pytest.approx(ref.B_H_SUB_M * 1e3)   # realized 0.32 mm

    # the trace sits ON the substrate, one cell thick
    assert trace["start"][2] == pytest.approx(ref.B_H_SUB_M * 1e3)
    assert trace["stop"][2] - trace["start"][2] == pytest.approx(
        ref.B_DX_COMPARATOR_M * 1e3)

    # (i) the lumped probe at rfx's own lw port-cell x, ground -> trace
    assert fdtd.lumped["start"] == [pytest.approx(2.0), pytest.approx(1.52),
                                    pytest.approx(0.0)]
    assert fdtd.lumped["stop"] == [pytest.approx(2.0), pytest.approx(1.52),
                                   pytest.approx(ref.B_H_SUB_M * 1e3)]
    assert fdtd.lumped["R"] == pytest.approx(50.0)
    assert fdtd.lumped["excite"] == (1.0 if drive == "lumped" else 0.0)

    # (ii) the MSL port: start at the 5.52 mm feed plane, prop_dir INTO
    # the line (stop < start on x), stencil at rfx's probe-0 plane
    assert msl.start[0] == pytest.approx(5.52)
    assert msl.stop[0] < msl.start[0]
    assert msl.measplane_shift == pytest.approx(0.80)   # mm -> 4.72 mm
    assert msl.excite == (1.0 if drive == "msl" else 0.0)
    assert port_info["msl_meas_plane_x_m"] == pytest.approx(4.72e-3)
    assert abs(port_info["effective_calcport_shift_real_m"]) < 1e-12
    assert port_info["ref_plane_shift_lumped_m"] == 0.0

    # the x mesh stays uniform and start-aligned (the snap prediction's
    # own assumption); only y is smoothed
    assert fdtd.csx._mesh.smoothed == [("y", ref.B_DX_COMPARATOR_M * 1e3 / 4.0, 1.4)]


def test_stage2_lumped_port_is_a_well_formed_openems_port(ref):
    """Structural contract on the port openEMS is actually handed.

    Checked WITHOUT openEMS, against the mesh and the primitives the
    builder itself records, and phrased as the four things
    ``openEMS.ports.LumpedPort`` + CSXCAD need to be true:

      (a) the box is NON-DEGENERATE on its excitation axis -- LumpedPort
          raises otherwise ("start and stop may not be identical in
          excitation direction");
      (b) the excitation axis is consistent with start/stop -- LumpedPort
          takes ``direction = sign(stop[exc_ny] - start[exc_ny])`` and
          builds ``exc_vec[exc_ny] = -direction*excite`` from it, so a
          transverse axis would launch nothing;
      (c) every box coordinate lands on EXACTLY ONE mesh line.  "At least
          one" is not enough: two ULP-separated lines put the port's own
          cells in a zero-width cell, and openEMS's operator there is
          singular;
      (d) no HIGHER-priority primitive claims the port's cells.  The port
          is priority 5; the substrate below it is priority 0 (the port
          wins, as in the patch precedent) and the trace above it is
          priority 10, which must therefore not reach down into the
          port's z span.
    """
    _FakeMSLPort.instances = []
    fdtd, _lumped, _msl, plan, _pi = ref._build_stage2(
        _FakeCSX, _FakeFDTD, _FakeMSLPort, dx_m=ref.B_DX_COMPARATOR_M,
        drive="lumped", nrts=1000, end_criteria=1e-4)
    port = fdtd.lumped
    start, stop = port["start"], port["stop"]
    axis = {"x": 0, "y": 1, "z": 2}[port["dir"]]

    # (a) + (b)
    assert start[axis] != stop[axis], (
        "lumped port is degenerate on its own excitation axis "
        f"{port['dir']}: start={start} stop={stop}")
    for other in (a for a in range(3) if a != axis):
        assert start[other] == pytest.approx(stop[other]), (
            "the lumped probe must be a line along its excitation axis "
            "only, as the patch precedent's feed is")
    assert port["excite"] != 0.0
    assert start[axis] == pytest.approx(0.0)                    # ground
    assert stop[axis] == pytest.approx(ref.B_H_SUB_M * 1e3)     # trace

    # (c) exactly one mesh line under each coordinate, on the mesh the
    #     builder actually fed the grid.
    tol_mm = 1e-6 * ref.B_DX_COMPARATOR_M * 1e3
    for name, coord in (("x", 0), ("y", 1), ("z_start", 2), ("z_stop", 2)):
        val = start[coord] if name != "z_stop" else stop[coord]
        ax = {"x": "x", "y": "y", "z_start": "z", "z_stop": "z"}[name]
        lines = np.asarray(fdtd.csx._mesh.lines[ax])
        n = int(np.sum(np.abs(lines - val) < tol_mm))
        assert n == 1, (
            f"lumped port {name}={val} mm is carried by {n} mesh lines on "
            f"the {ax} axis, not 1 -- a zero-width cell at the port")

    # (d) nothing of higher priority reaches into the port's z span
    for box in fdtd.csx.boxes:
        if box["priority"] <= port["priority"]:
            continue
        lo = [min(a, b) for a, b in zip(box["start"], box["stop"])]
        hi = [max(a, b) for a, b in zip(box["start"], box["stop"])]
        covers_xy = all(lo[k] <= start[k] <= hi[k] for k in (0, 1))
        z_lo, z_hi = min(start[2], stop[2]), max(start[2], stop[2])
        overlap = min(hi[2], z_hi) - max(lo[2], z_lo)
        assert not (covers_xy and overlap > 0), (
            f"primitive {box['prop']!r} (priority {box['priority']}) "
            f"overlaps the lumped port's z span by {overlap} mm and would "
            f"claim its cells away from the port (priority "
            f"{port['priority']})")


def test_stage2_builder_rejects_an_unknown_drive(ref):
    with pytest.raises(ref.ConfigError, match="drive must be"):
        ref._build_stage2(_FakeCSX, _FakeFDTD, _FakeMSLPort,
                          dx_m=ref.B_DX_COMPARATOR_M, drive="both",
                          nrts=10, end_criteria=1e-4)


# ---------------------------------------------------------------------------
# rfx artifact contract (data dependency, never a python import of rfx)
# ---------------------------------------------------------------------------
def test_rfx_artifact_missing_file_is_a_config_error(ref, tmp_path):
    with pytest.raises(ref.ConfigError, match="not found"):
        ref.load_rfx_artifact(str(tmp_path / "nope.json"))


def test_rfx_artifact_missing_key_is_a_config_error(ref, tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"freqs_hz": [1e9]}))
    with pytest.raises(ref.ConfigError, match="missing required key 's_raw'"):
        ref.load_rfx_artifact(str(p))


def test_rfx_artifact_wrong_shape_is_a_config_error(ref, tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"freqs_hz": [1e9], "s_raw": [[1.0, 2.0]]}))
    with pytest.raises(ref.ConfigError, match="expected"):
        ref.load_rfx_artifact(str(p))


def _planted_rfx_artifact(tmp_path, *, s21=0.5, s22=0.02, s11=0.38):
    freqs = [1.0e9, 1.75e9]
    s = [[[[s11, 0.0], [s11, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
         [[[s21, 0.0], [s21, 0.0]], [[s22, 0.0], [s22, 0.0]]]]
    p = tmp_path / "rfx.json"
    p.write_text(json.dumps({"freqs_hz": freqs, "s_raw": s}))
    return p


def test_comparison_arithmetic_on_planted_data(ref, tmp_path):
    """Plant an rfx |S21| exactly sqrt(n_live)=2 above the openEMS one: the
    comparison must REPORT the ratio 2.0 and must NOT draw a verdict."""
    p = _planted_rfx_artifact(tmp_path, s21=0.5, s22=0.02)
    rfx = ref.load_rfx_artifact(str(p))
    oe = {"freqs_hz": np.asarray([1.0e9, 1.75e9]),
          "s11": [0.1 + 0j, 0.1 + 0j],
          "s21": [0.25 + 0j, 0.25 + 0j],
          "s22": [0.05 + 0j, 0.05 + 0j]}
    cmp_ = ref.compare_against_rfx(oe, rfx)
    assert cmp_["abs_s21_ratio_band"] == [pytest.approx(2.0), pytest.approx(2.0)]
    assert cmp_["abs_s22_max_diff"] == pytest.approx(0.03)
    assert "sqrt(n_live)" in cmp_["frame_note"]
    assert "computed FROM THE rfx RUN" in cmp_["budget_note"]
    # nothing in the comparison is a verdict or a gate
    assert "passed" not in cmp_ and "verdict" not in cmp_


def test_comparison_refuses_a_bin_the_openems_grid_does_not_have(ref, tmp_path):
    p = _planted_rfx_artifact(tmp_path)
    rfx = ref.load_rfx_artifact(str(p))
    oe = {"freqs_hz": np.asarray([3.0e9]), "s11": [0j], "s21": [0j], "s22": [0j]}
    with pytest.raises(ref.ConfigError, match="no bin at"):
        ref.compare_against_rfx(oe, rfx)


# ---------------------------------------------------------------------------
# Declared limits travel with every artifact (frozen by CONTENT, not line)
# ---------------------------------------------------------------------------
def test_cannot_compare_names_the_frame_the_projection_and_the_anchor(ref):
    joined = " ".join(ref.CANNOT_COMPARE)
    assert "PRE-INJECTION" in joined and "12.5 ohm" in joined
    assert "enforce_passivity" in joined and "S_raw" in joined
    assert "sqrt(n_live) = 2.0" in joined
    assert "47.89479996289313" in joined
    assert "MESH CONVERGENCE" in joined
    assert "WHICH OF rfx's OWN DIAGONALS IS LYING" in joined


def test_no_lumped_wire_diagonal_number_is_pinned_by_this_lane(ref):
    """The do-not-pin list's first entry, enforced where it can be: the
    referee reports |S11| and never gates on it, and no shipped lw-diagonal
    value appears as a threshold anywhere in the script's gates."""
    text = _SCRIPT.read_text()
    for pinned in ("0.3814", "0.3863", "0.3922", "0.3980", "0.4027", "0.4324"):
        assert pinned not in text, f"{pinned} must not appear in the referee"
    assert "REPORTED only" in text or "REPORTED, not adjudicated" in text


def test_reproduce_gate_record_is_serialized_into_every_artifact(ref, tmp_path):
    """Prose living only in a docstring never reaches the artifact -- the
    record must be a module-level dict that main() writes out."""
    rec = ref.REPRODUCE_GATE_RECORD
    for leg in ("a1", "a2"):
        assert rec[leg]["example"].startswith("openEMS python/Tutorials/")
        assert rec[leg]["recorded_reproduction"]["log_path"]
        assert "log_present_in_tree" in rec[leg]["recorded_reproduction"]
    # A2's log is local-only and NOT in the tree -- the caveat the
    # predeclaration dropped is restored and must stay
    assert rec["a2"]["recorded_reproduction"]["log_present_in_tree"] is False
    assert "LOCAL-ONLY" in rec["a2"]["recorded_reproduction"]["log_caveat"]
    # A1's log IS tracked -- assert it really is
    a1_log = _REPO_ROOT / rec["a1"]["recorded_reproduction"]["log_path"]
    assert a1_log.exists(), f"A1's cited log is missing: {a1_log}"


def test_a2_gate_separates_upstream_from_repo_internal_provenance(ref):
    """The reviewer's non-blocking note: '~7 dBi broadside' is the only
    upstream number; the 2.43 GHz half is a regression lock on this repo's
    own run and must be labelled as such."""
    a2 = ref.REPRODUCE_GATE_RECORD["a2"]
    assert "~7 dBi" in a2["gate_upstream_anchored"]
    assert "NOT a reproduction of an upstream number" in a2["gate_repo_internal_lock"]


# ---------------------------------------------------------------------------
# The VESSL lane must name its own prerequisites instead of dying anonymously
# ---------------------------------------------------------------------------
def _yaml_run_block() -> str:
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(_YAML.read_text())
    return doc["run"]


def _run_clone_prelude() -> str:
    """The run block from ``ROOT=`` through ``cd "$ROOT"``, inclusive.

    Frozen by CONTENT (the two symbols), never by line number.
    """
    lines = _yaml_run_block().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("ROOT="))
    stop = next(i for i, ln in enumerate(lines) if ln.strip() == 'cd "$ROOT"')
    assert start < stop, "the ROOT assignment must precede the cd into it"
    return "set -eu\n" + "\n".join(lines[start : stop + 1]) + "\necho REACHED_CD\n"


def _sh(fragment: str, run_clone: str):
    env = dict(os.environ)
    env["RUN_CLONE"] = run_clone
    return subprocess.run(
        ["sh", "-c", fragment], capture_output=True, text=True, env=env
    )


def test_vessl_lane_refuses_a_missing_run_clone_before_it_cds_into_it(tmp_path):
    """FAIL-BEFORE-FIX: without the guard the job dies at the bare ``cd``
    under ``set -eux`` with an anonymous shell error, BEFORE any EXPECT or
    content guard and before the verdict ``case`` can speak.  With it the
    failure names itself and exits 3 (config error), like every other
    prerequisite in this lane."""
    frag = _run_clone_prelude()
    missing = tmp_path / "no-such-run-clone"
    assert not missing.exists()
    proc = _sh(frag, str(missing))
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "RUN CLONE MISSING" in proc.stdout
    assert str(missing) in proc.stdout
    assert "REACHED_CD" not in proc.stdout


def test_vessl_lane_refuses_a_run_clone_that_is_not_a_git_clone(tmp_path):
    """A directory with no ``.git`` cannot establish provenance, so the
    EXPECT/ancestor guards downstream would be meaningless."""
    not_a_clone = tmp_path / "not-a-clone"
    not_a_clone.mkdir()
    proc = _sh(_run_clone_prelude(), str(not_a_clone))
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "NOT A GIT CLONE" in proc.stdout
    assert "REACHED_CD" not in proc.stdout


def test_vessl_lane_proceeds_when_the_run_clone_exists(tmp_path):
    """The positive control: a real clone directory reaches the ``cd``."""
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    proc = _sh(_run_clone_prelude(), str(clone))
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "REACHED_CD" in proc.stdout


def test_vessl_lane_requires_run_clone_by_name():
    """The yaml carries NO default run-clone path (it is per-workspace, and
    a public repo must not ship a private absolute path): an empty
    RUN_CLONE fails by NAME with exit 3, before the ``cd``."""
    run = _yaml_run_block()
    assert 'ROOT="${RUN_CLONE:-}"' in run
    assert "/root/" not in run and "~/" not in run
    proc = _sh(_run_clone_prelude(), "")
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "RUN CLONE UNSET" in proc.stdout
    assert "REACHED_CD" not in proc.stdout


# ---------------------------------------------------------------------------
# The predeclaration and the code must describe the SAME board
# ---------------------------------------------------------------------------
def _predeclaration_section_7_2() -> str:
    text = _PREDECLARATION.read_text()
    assert "### 7.2" in text and "### 7.3" in text
    return text.split("### 7.2", 1)[1].split("### 7.3", 1)[0]


def test_predeclaration_7_2_describes_the_board_stage_2_actually_builds(ref):
    """#723 in documentation form: a Stage-2 number quoted against a §7.2
    that describes a different model is the same failure as measuring the
    wrong board.  The section must carry the realized board's dimensions,
    taken from the code's own record, and it must not still assert the
    superseded declared geometry as the model."""
    sec = _predeclaration_section_7_2()
    realized = ref.RFX_REALIZED_RECORD["realized"]
    h_um = round(realized["h_sub_m"] * 1e6)
    w_um = round(realized["w_trace_node_span_m"] * 1e6)
    assert h_um == 320 and w_um == 480
    assert f"{h_um}" in sec, "§7.2 does not carry the realized substrate height"
    assert f"{w_um}" in sec, "§7.2 does not carry the realized trace width"
    # the superseded bullet must not survive verbatim
    assert (
        "`h_sub = 254 µm`, `W = 600 µm`, trace from x = 0 to 8 mm" not in sec
    ), "§7.2 still asserts the declared geometry as the model"


def test_predeclaration_7_2_carries_an_explicit_supersession_note():
    """Not merely edited: the amendment must NAME what it supersedes --
    review blocker B4 (both open ends) and the realized board -- so the
    change is auditable against the review that required it."""
    sec = _predeclaration_section_7_2()
    assert "SUPERSESSION" in sec
    assert "B4" in sec
    assert "REALIZED" in sec.upper()
    # and the top-of-document amendment log must point at it
    head = _PREDECLARATION.read_text().split("## 1.", 1)[0]
    assert "Amendment log" in head and "7.2" in head



# ---------------------------------------------------------------------------
# The plan must not claim a mesh the builder does not build, and a self-check
# that returns a literal is not a check (VESSL 369367257610 follow-up).
# ---------------------------------------------------------------------------
def test_x_uniformity_is_computed_and_reports_the_off_lattice_planes(ref):
    """The old row was the literal True. x is NOT uniform and never was."""
    plan = ref.openems_mesh_plan(50e-6)
    assert "x_mesh_is_uniform_and_start_aligned" not in plan, (
        "the hardcoded uniformity literal must be gone: a self-check that "
        "returns a constant is not a check")
    rep = plan["x_mesh_uniformity"]
    assert rep["uniform"] is False, (
        "the MSL measurement plane (4.72 mm) and port start (5.52 mm) are off "
        "the 50 um lattice, so x cannot be uniform; a True here would mean the "
        "report was computed against the wrong lines")
    assert rep["n_cells_off_nominal"] > 0
    assert rep["min_cell_m"] < rep["nominal_dx_m"]
    # and it must still be a MEASURED report, not an assertion that fails the run
    assert rep["max_cell_m"] <= rep["nominal_dx_m"] * 1.001


def test_plan_states_the_y_smoothing_the_builder_will_apply(ref):
    """The plan's y lines are pre-smoothing; it must say what happens next.

    Superseded the earlier BOUND on the line count with the computed mesh --
    see test_plan_computes_the_post_smoothing_mesh_the_builder_hands_openems.
    """
    plan = ref.openems_mesh_plan(50e-6)
    assert plan["y_lines_are_pre_smoothing"] is True
    built = plan["mesh_as_planned_built"]
    assert built["max_cell_target_m"] == pytest.approx(50e-6 / 4.0)
    assert built["n_lines"]["y"] > plan["n_lines"]["y"], (
        "smoothing to dx/4 must ADD y lines")
    # the y PML depth follows the smoothed cells, not the pad it was sized with
    assert built["pml_depth_m"]["y_lo"] < ref.B_PML_CELLS * 50e-6
    assert built["max_cell_m"]["y"] <= 50e-6 / 4.0 * (1 + 1e-9)


def test_plan_computes_the_post_smoothing_mesh_the_builder_hands_openems(ref):
    """A bound is not the built mesh; the record has to carry the real one.

    The reference numbers are CSXCAD's OWN ``SmoothMeshLines`` (fetched from
    thliebig/CSXCAD and run against this plan's y lines offline), so this is
    a lock on the actual smoother, not on a paraphrase of it.
    """
    expect = {
        50e-6: {"y_lines": 401, "cells": 1_708_800,
                "y_min_um": 3.6364, "y_max_um": 12.5,
                "pml_y_lo_um": 67.708, "pml_y_hi_um": 70.0},
        80e-6: {"y_lines": 301, "cells": 626_400,
                "y_min_um": 6.6667, "y_max_um": 20.0,
                "pml_y_lo_um": 92.826, "pml_y_hi_um": 105.746},
    }
    for dx_m, want in expect.items():
        plan = ref.openems_mesh_plan(dx_m)
        built = plan["mesh_as_planned_built"]
        assert built["n_lines"]["y"] == want["y_lines"]
        assert built["n_lines"]["x"] == plan["n_lines"]["x"]   # x is not smoothed
        assert built["n_lines"]["z"] == plan["n_lines"]["z"]
        assert built["n_cells_total"] == want["cells"]
        assert built["min_cell_m"]["y"] * 1e6 == pytest.approx(want["y_min_um"], rel=1e-4)
        assert built["max_cell_m"]["y"] * 1e6 == pytest.approx(want["y_max_um"], rel=1e-6)
        # PML depth per FACE -- a cell COUNT, so smoothing shrinks it on y only
        pml = built["pml_depth_m"]
        assert pml["y_lo"] * 1e6 == pytest.approx(want["pml_y_lo_um"], rel=1e-4)
        assert pml["y_hi"] * 1e6 == pytest.approx(want["pml_y_hi_um"], rel=1e-4)
        assert pml["x_lo"] == pytest.approx(ref.B_PML_CELLS * dx_m)
        assert pml["x_hi"] == pytest.approx(ref.B_PML_CELLS * dx_m)
        assert pml["z_hi"] == pytest.approx(ref.B_PML_CELLS * dx_m)
        assert "z_lo" not in pml, "z_lo is PEC, not PML -- it has no depth"
        # and the honest headline: the built mesh is several times the plan
        assert built["n_cells_total"] > 4 * plan["n_cells_total"]


def test_planned_total_cells_reaches_the_artifact_instead_of_zero(ref):
    """mesh_as_built compared the built count against plan['total_cells'],
    a key the plan does not define, so every artifact recorded 0."""
    _FakeMSLPort.instances = []
    fdtd, _l, _m, plan, _pi = ref._build_stage2(
        _FakeCSX, _FakeFDTD, _FakeMSLPort, dx_m=ref.B_DX_COMPARATOR_M,
        drive="lumped", nrts=10, end_criteria=1e-4)
    as_built = plan["mesh_as_built"]
    assert as_built["planned_total_cells"] == plan["n_cells_total"]
    assert as_built["planned_total_cells"] > 0, (
        "the artifact recorded a planned cell count of 0 -- the record was "
        "wrong about itself, which is the whole defect class here")


def test_build_refuses_when_the_real_smoother_disagrees_with_the_plan(ref):
    """The plan predicts the built mesh with a PORT of CSXCAD's smoother.

    If the installed CSXCAD ever smooths differently, the artifact's mesh
    record would be wrong. That must fail LOUD at build time, in seconds,
    not silently inside a recorded number.
    """
    class _DriftingMesh(_FakeMesh):
        def SmoothMeshLines(self, ax, res, ratio):
            # one line coarser than the plan predicts
            super().SmoothMeshLines(ax, res * 1.05, ratio)

    class _DriftingCSX(_FakeCSX):
        def __init__(self):
            super().__init__()
            self._mesh = _DriftingMesh()

    _FakeMSLPort.instances = []
    with pytest.raises(ref.ConfigError, match="has drifted from the installed"):
        ref._build_stage2(_DriftingCSX, _FakeFDTD, _FakeMSLPort,
                          dx_m=ref.B_DX_COMPARATOR_M, drive="lumped",
                          nrts=10, end_criteria=1e-4)


def test_excitation_check_separates_a_nan_field_from_a_starved_port(ref):
    """nan and 0.0 have different causes and must not share a sentence."""
    class _P:
        def __init__(self, v):
            self.uf_inc = np.asarray(v, dtype=np.complex128)

    with pytest.raises(RuntimeError, match="FIELD"):
        ref._check_excitation(_P([np.nan + 0j]), "nanleg")
    with pytest.raises(RuntimeError, match="NO wave energy"):
        ref._check_excitation(_P([0j]), "zeroleg")
    # a healthy port still returns its peak
    assert ref._check_excitation(_P([0.25 + 0j]), "ok") == pytest.approx(0.25)


def test_nan_message_does_not_blame_the_port(ref):
    """The old message sent the next reader after the port; the source says
    a lumped port cannot produce nan from a zero field."""
    class _P:
        uf_inc = np.asarray([np.nan + 0j])

    with pytest.raises(RuntimeError) as exc:
        ref._check_excitation(_P(), "leg")
    msg = str(exc.value)
    assert "not the suspect" in msg.lower() or "NOT the suspect" in msg
    assert "excitation did not couple" not in msg

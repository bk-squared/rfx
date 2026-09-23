"""The graded-mesh notch filter builds the board it declares -- no time step.

Two arms of the band-graded MSL notch filter (the coarsest rung, node ON the
metal edge and node 0.35 of a cell INSIDE it) are built and read back from the
lattice.  The three refusals the pre-declaration puts before any solve are
asserted through the instrument's own functions, so a refusal that stops
passing here would have stopped an arm on the GPU:

* R1 -- every declared metal-edge node, the substrate top and every band edge
  is where the profile's own cumulative sum puts it; every adjacent cell ratio
  obeys its axis cap; nine cells of exactly 127 um stand against each in-plane
  absorber face;
* R2 -- preflight draws no absorber-runway line, no geometry-in-absorber line
  and no buried-sheet line;
* R3 -- one sheet plane on the substrate top with air above it, no PEC volume
  cell, n+1 node rows under the line and n+1 node columns under the stub, the
  stub joined to the line and 12 mm long, and the outermost realized node of
  every free in-plane edge exactly where the arm places it.

Three deliberate defects are built and must be refused.  (a) is the check
itself: ``_check_realized_fields`` is handed a measurement with one field
moved, so an emptied body fails here rather than silently passing every arm.
(b) draws the stub one coarse cell short with every helper call left in place.
(c) draws the offset arm's sheets to their nodes instead of to their true
600 um, which is the arm the note calls "on-node" built on the offset mesh --
the mesh says one thing and the metal another.

No solve: the whole file is build-time work on a ~1 M cell grid.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from validation.research.multiband_nu import msl_notch_graded as ins
from tests.crossval.msl_notch_filter import test_msl_notch_filter as case


ARMS_BUILT = ("A_on", "A_off")


@pytest.fixture(scope="module")
def built():
    """Both coarsest-rung arms, built once: assembly dominates this file."""
    out = {}
    for key in ARMS_BUILT:
        arm = ins.ARMS[key]
        sim, prof, board = ins.build_graded(arm.rung, arm.placement,
                                            arm.arm_length_m)
        out[key] = (sim, prof, board)
    return out


# ----------------------------------------------------------------------- R1
@pytest.mark.parametrize("key", ARMS_BUILT)
def test_r1_nodes_ratios_and_runways(built, key):
    """Every intended node is where the board declares it, and the absorber's
    runway is nine cells of exactly the coarse cell."""
    _sim, prof, board = built[key]
    report = ins.assert_profiles_graded(prof, board)

    x, y, z = prof
    assert np.all(x[:ins.RUNWAY_CELLS] == ins.C_COARSE_M)
    assert np.all(x[-ins.RUNWAY_CELLS:] == ins.C_COARSE_M)
    assert np.all(y[:ins.RUNWAY_CELLS] == ins.C_COARSE_M)
    assert np.all(y[-ins.RUNWAY_CELLS:] == ins.C_COARSE_M)
    assert len(set(z[-ins.RUNWAY_CELLS:].tolist())) == 1
    assert z[-1] <= ins.C_COARSE_M

    assert report["x"]["worst_ratio"] <= ins.CAP_XY + 1e-9
    assert report["y"]["worst_ratio"] <= ins.CAP_XY + 1e-9
    assert report["z"]["worst_ratio"] <= ins.CAP_Z + 1e-9
    assert max(report["intended_node_miss_m"].values()) <= ins.NODE_TOL_M

    # The profile sums ARE the domain the arm solves.
    for axis, p in zip(range(3), prof):
        assert float(np.sum(p)) == pytest.approx(board["domain_m"][axis],
                                                 abs=1e-12)


@pytest.mark.parametrize("key", ARMS_BUILT)
def test_r1_refuses_a_runway_that_is_not_the_coarse_cell(built, key):
    """A profile whose first cell is not C is refused -- the check is on the
    cells, not on a flag."""
    _sim, (x, y, z), board = built[key]
    bad = x.copy()
    bad[0] *= 0.999
    bad[1] += x[0] - bad[0]
    with pytest.raises(AssertionError, match="bit-exactly"):
        ins.assert_profiles_graded((bad, y, z), board)


# ----------------------------------------------------------------------- R2
@pytest.mark.parametrize("key", ARMS_BUILT)
def test_r2_preflight_draws_none_of_the_refused_lines(built, key):
    sim, _prof, _board = built[key]
    report = ins.assert_preflight_graded(sim)
    assert report, "preflight returned nothing at all"
    for line in report:
        for needle in ins.R2_FORBIDDEN:
            assert needle not in line


def test_r2_refuses_a_report_that_carries_a_forbidden_line():
    """The R2 reader refuses on the text it is given, not on a build flag."""
    class _Fake:
        def preflight(self):
            return ["dy_profile is not uniform within the 8 interior cells "
                    "adjacent to the y_lo absorber face"]

    with pytest.raises(AssertionError, match="not uniform within the"):
        ins.assert_preflight_graded(_Fake())


# ----------------------------------------------------------------------- R3
@pytest.mark.parametrize("key", ARMS_BUILT)
def test_r3_the_lattice_built_the_declared_board(built, key):
    sim, _prof, board = built[key]
    g = ins.assert_realized_graded(sim, board)
    n = board["n_across_metal"]
    f = board["fine_cell_m"]

    assert g["n_sheet_planes"] == 1
    assert g["n_volume_cells"] == 0
    assert g["eps_below_sheet"] == pytest.approx(case.EPS_R_SUBSTRATE)
    assert g["eps_above_sheet"] == 1.0
    assert g["sheet_plane_z_m"] == pytest.approx(case.SUBSTRATE_THICKNESS_M,
                                                 abs=ins.NODE_TOL_M)
    assert g["n_trace_rows"] == g["n_stub_cols"] == n + 1
    assert g["trace_width_node_span_m"] == pytest.approx(n * f,
                                                         abs=ins.NODE_TOL_M)
    assert g["stub_width_node_span_m"] == pytest.approx(n * f,
                                                        abs=ins.NODE_TOL_M)
    assert g["stub_attached"] and g["stub_contiguous"]
    assert g["stub_length_m"] == pytest.approx(case.STUB_LENGTH_M,
                                               abs=ins.NODE_TOL_M)
    assert g["n_ports"] == 2


@pytest.mark.parametrize("key", ARMS_BUILT)
def test_the_metal_is_solved_at_its_drawn_size(built, key):
    """The point of the two arms, read off the lattice.

    On the offset arm the metal's outermost node stands 0.35 of a fine cell
    inside the drawn edge, so its SOLVED edge lands on the drawn 600 um.  On
    the on-node arm the node is the drawn edge and the solved metal is
    0.7 cell wider.
    """
    sim, _prof, board = built[key]
    g = ins.realized_graded(sim)
    f = board["fine_cell_m"]
    solved_width = g["trace_width_node_span_m"] + 2.0 * ins.EDGE_OFFSET * f
    if board["edge_offset_m"] == 0.0:
        assert solved_width > case.TRACE_WIDTH_M * 1.05
    else:
        assert solved_width == pytest.approx(case.TRACE_WIDTH_M, abs=1e-12)


@pytest.mark.parametrize("key", ARMS_BUILT)
def test_every_cell_the_probe_array_spans_is_the_coarse_cell(built, key):
    """The extractor counts its probe spacing in cells, so every cell from the
    feed plane to the deepest probe has to be that one size."""
    sim, _prof, _board = built[key]
    planes = ins.probe_planes(sim)
    runway = ins.assert_probes_on_the_uniform_runway(sim, ins.C_COARSE_M)
    assert len(planes) == 2 and len(runway["ports"]) == 2
    for xs, port in zip(planes, runway["ports"]):
        assert len(xs) == ins.N_PROBES
        assert port["n_cells_off_coarse"] == 0
        assert port["n_cells_spanned"] == (
            ins.N_PROBE_OFFSET + (ins.N_PROBES - 1) * ins.N_PROBE_SPACING)


@pytest.mark.parametrize("key", ARMS_BUILT)
def test_the_port_feed_footprint_lies_inside_the_fine_band(built, key):
    """The port pads its Ez source about one substrate thickness each side of
    the trace and resolves that pad in cells, so the pad must be inside the
    band that makes those cells one size."""
    sim, _prof, board = built[key]
    got = ins.assert_port_footprint_in_fine_band(sim, board)
    lo, hi = board["band_y_trace_m"]
    for port in got["ports"]:
        assert lo <= port["y_lo_m"] and port["y_hi_m"] <= hi
        assert port["pad_cells_lo"] >= 1 and port["pad_cells_hi"] >= 1
        assert port["eps_r_sub"] == pytest.approx(case.EPS_R_SUBSTRATE)


def test_mutation_a2_a_probe_array_that_reaches_the_ramp_is_refused():
    """The ramp is where the cell size changes fastest, and the first version
    of this refusal checked only the band.  The mesh is untouched: only the
    probe offset moves, so nothing but the probe check can see it."""
    arm = ins.ARMS["A_off"]
    sim, prof, board = ins.build_graded(
        arm.rung, arm.placement, arm.arm_length_m, _mutate_probe_offset=47)
    ins.assert_profiles_graded(prof, board)
    ins.assert_realized_graded(sim, board)
    ins.assert_port_footprint_in_fine_band(sim, board)   # still inside
    with pytest.raises(AssertionError, match="not the coarse cell"):
        ins.assert_probes_on_the_uniform_runway(sim, ins.C_COARSE_M)


def test_mutation_b2_a_band_too_narrow_for_the_feed_is_refused():
    """A band of two fine cells beyond each metal edge still carries the metal
    and still passes R1 and R3; what it no longer carries is the port's own
    lateral pad."""
    arm = ins.ARMS["A_off"]
    sim, prof, board = ins.build_graded(
        arm.rung, arm.placement, arm.arm_length_m, _mutate_margin_cells=2)
    ins.assert_profiles_graded(prof, board)
    ins.assert_realized_graded(sim, board)
    ins.assert_probes_on_the_uniform_runway(sim, ins.C_COARSE_M)   # still coarse
    with pytest.raises(AssertionError, match="outside the fine y band"):
        ins.assert_port_footprint_in_fine_band(sim, board)


def test_the_uniform_reference_arms_build_on_the_cases_own_board():
    """A.1: U_h2, U_h4 and U_h6 are the case's build at the case's LADDER_M.

    The line does not move for them -- that is the whole point of a reference
    ladder -- so this pins the one thing that would silently invalidate the
    cost ratios and W3's target: the uniform arms running on the graded arms'
    board instead of the case's.
    """
    assert ins.UNIFORM_LADDER == ("U_h2", "U_h4", "U_h6")
    assert [ins.UNIFORM_ARMS[k] for k in ins.UNIFORM_LADDER] == list(case.LADDER_M)
    assert ins.COST_REFERENCE_ARM == "U_h6"
    sim = case.build(ins.UNIFORM_ARMS["U_h2"])
    g = case.assert_realized(sim, ins.UNIFORM_ARMS["U_h2"])
    assert g["n_ports"] == 2
    # The case's board, not this note's: the line sits where the case puts it.
    assert case.LATERAL_CLEARANCE_M < ins.LATERAL_CLEARANCE_M
    ys = [e.position[1] for e in sim._msl_ports]
    assert ys == [case.TRACE_CENTRE_Y_M, case.TRACE_CENTRE_Y_M]


def test_grid_cells_counts_cells_and_not_nodes():
    """Both sides of every cost ratio count the cells the solver steps."""
    rec = {"grid_shape": [584, 462, 51]}
    assert ins.grid_cells(rec) == 583 * 461 * 50 == 13_438_150
    rec["realized"] = {"n_grid_cells": 13_438_150}
    assert ins.grid_cells(rec) == 13_438_150
    rec["realized"]["n_grid_cells"] = 13_760_208          # the node product
    with pytest.raises(AssertionError, match="n_grid_cells"):
        ins.grid_cells(rec)


# ------------------------------------------------------- mutation (a) the check
@pytest.mark.parametrize("field,value,needle", [
    ("n_trace_rows", 99, "node rows"),
    ("n_volume_cells", 7, "PEC VOLUME"),
    ("stub_attached", False, "not joined"),
    ("stub_length_m", case.STUB_LENGTH_M - 1e-6, "realized stub length"),
    ("trace_width_node_span_m", 1e-3, "node span"),
    ("eps_above_sheet", case.EPS_R_SUBSTRATE, "not air"),
])
def test_mutation_a_the_r3_check_refuses_each_field_it_claims_to_check(
        built, field, value, needle):
    """Mutation (a): the R3 body itself.

    ``_check_realized_fields`` is given a REAL measurement of A_off with one
    field moved.  A no-op check passes every arm and every mutant alike, so
    this parametrization is what makes an emptied body red.
    """
    sim, _prof, board = built["A_off"]
    g = dict(ins.realized_graded(sim))
    ins._check_realized_fields(g, board)          # the untouched measurement
    g[field] = value
    with pytest.raises(AssertionError, match=needle):
        ins._check_realized_fields(g, board)


# ---------------------------------------------- mutation (b) a shortened stub
def test_mutation_b_a_stub_one_coarse_cell_short_is_refused():
    """Mutation (b): the declared board, drawn with the stub 127 um short.

    Every helper the good arm calls is called here -- ``profiles``,
    ``build_graded``, ``realized_graded``, ``_check_realized_fields`` -- so the
    refusal can only come from the measurement, not from a different code path.
    """
    arm = ins.ARMS["A_off"]
    sim, prof, board = ins.build_graded(
        arm.rung, arm.placement, arm.arm_length_m,
        _mutate_stub_short_m=ins.C_COARSE_M)
    ins.assert_profiles_graded(prof, board)       # the MESH is untouched
    with pytest.raises(AssertionError, match="realized stub length"):
        ins.assert_realized_graded(sim, board)


# ------------------------------------------- mutation (c) the node on the edge
def test_mutation_c_the_offset_arm_drawn_to_its_nodes_is_refused():
    """Mutation (c): the offset mesh with the sheets drawn to their nodes.

    The metal is then 566.93 um of drawn copper on a mesh built for 600 um, and
    its solved edge lands 0.35 of a cell outside the drawn one instead of on
    it.  The realized node count, the stub length and the joint are all still
    right, so only the edge-placement check can see it.
    """
    arm = ins.ARMS["A_off"]
    sim, prof, board = ins.build_graded(
        arm.rung, arm.placement, arm.arm_length_m, _mutate_draw_offset=0.0)
    ins.assert_profiles_graded(prof, board)
    g = ins.realized_graded(sim)
    assert g["n_trace_rows"] == board["n_across_metal"] + 1
    assert g["stub_length_m"] == pytest.approx(case.STUB_LENGTH_M,
                                               abs=ins.NODE_TOL_M)
    with pytest.raises(AssertionError, match="outermost realized node"):
        ins.assert_realized_graded(sim, board)


# ------------------------------------------------------------ the arms declared
def test_every_declared_arm_has_a_profile_and_a_rung():
    """The five arms of the note, and no sixth."""
    assert sorted(ins.ARMS) == ["A_off", "A_off_longarms", "A_on", "B_off",
                                "C_off"]
    assert ins.LADDER == ("A_off", "B_off", "C_off")
    for key, arm in ins.ARMS.items():
        f = ins.fine_cell(arm.rung, arm.placement)
        n = ins.RUNGS[arm.rung][1]
        if arm.placement == "offset":
            assert (n + 2 * ins.EDGE_OFFSET) * f == pytest.approx(
                case.TRACE_WIDTH_M, abs=1e-15)
        else:
            assert n * f == pytest.approx(case.TRACE_WIDTH_M, abs=1e-15)
    # The ladder refines the substrate 6 -> 8 -> 12 cells and the metal with it.
    assert [ins.RUNGS[ins.ARMS[k].rung] for k in ins.LADDER] == [
        (6, 12), (8, 16), (12, 24)]


def test_the_board_records_where_it_departs_from_the_case():
    """The two numbers this mesh moves, and the arithmetic that forced them.

    A change nobody can see in the record is a change nobody can judge.
    """
    assert ins.LATERAL_CLEARANCE_M > case.LATERAL_CLEARANCE_M
    assert ins.LATERAL_CLEARANCE_M == pytest.approx(17.0 * ins.C_COARSE_M)
    # Why the case's 1.55 mm cannot carry the runway: the declared band starts
    # 500 um below the metal and nine coarse cells are longer than what is left.
    band_start = case.LATERAL_CLEARANCE_M - ins.BAND_MARGIN_M
    assert band_start < ins.RUNWAY_CELLS * ins.C_COARSE_M
    # Everything else is the case's.
    for key in ARMS_BUILT:
        arm = ins.ARMS[key]
        b = ins.board(arm.rung, arm.placement, arm.arm_length_m)
        assert b["stub_open_y_m"] - b["trace_y_hi_m"] == pytest.approx(
            case.STUB_LENGTH_M)
        assert b["trace_y_hi_m"] - b["trace_y_lo_m"] == pytest.approx(
            case.TRACE_WIDTH_M)
        assert b["domain_y_m"] - b["stub_open_y_m"] == pytest.approx(
            case.DOMAIN_Y_M - (case.TRACE_CENTRE_Y_M + case.TRACE_WIDTH_M / 2
                               + case.STUB_LENGTH_M))
        assert b["port_x_m"] == (case.PORT_MARGIN_M,
                                 case.PORT_MARGIN_M + 2.0 * arm.arm_length_m)


def test_the_crossval_case_is_imported_not_copied():
    """Every threshold and every board constant comes from the case module."""
    assert ins.C_COARSE_M is case.DX_COARSEST_M
    assert ins._H is case.SUBSTRATE_THICKNESS_M
    assert ins._W is case.TRACE_WIDTH_M
    # No numeric LITERAL in the instrument's code may equal a board constant
    # the case already owns.  Read from the syntax tree, so the module's own
    # prose (which does quote the board in words) is not what is checked.
    import ast

    owned = {
        case.EPS_R_SUBSTRATE, case.SUBSTRATE_THICKNESS_M, case.TRACE_WIDTH_M,
        case.STUB_LENGTH_M, case.ARM_LENGTH_M, case.WITNESS_ARM_LENGTH_M,
        case.PORT_MARGIN_M, case.DOMAIN_X_M, case.DOMAIN_Y_M, case.DOMAIN_Z_M,
        case.TRACE_CENTRE_Y_M, case.LATERAL_CLEARANCE_M, case.FREQ_MAX_HZ,
        case.PORT_IMPEDANCE_OHM, case.N_FREQS, case.NUM_PERIODS,
        case.DEEP_NULL_DB, case.SETTLING_DB,
    }
    # 2.0 dB and 0.5 dB are too ordinary a number to scan for, so the two bars
    # that carry them are checked by name instead.
    tree = ast.parse(open(ins.__file__).read())
    found = {node.value for node in ast.walk(tree)
             if isinstance(node, ast.Constant)
             and isinstance(node.value, (int, float))
             and not isinstance(node.value, bool)
             and node.value in owned}
    assert not found, (
        f"{sorted(found)} are case constants re-typed as literals; import them")
    src = open(ins.__file__).read()
    for name in ("MAG_BAR_DB", "ARM_WITNESS_BAR_DB", "FREQ_BAR",
                 "LADDER_AGREEMENT", "PASSIVITY_EXCESS_BAR"):
        assert f"case.{name}" in src, f"the {name} bar must be read from the case"


# -------------------------------------------------- the recorded arms' meshes
RECORD = (Path(__file__).resolve().parents[3] / "validation" / "research"
          / "multiband_nu" / "results" / "msl_notch_graded.json")
RECORDED_GRADED_ARMS = ("A_on", "A_off", "A_off_longarms", "B_off", "C_off")

#: How far a solved ramp cell may sit from the one the record holds, in units
#: of the last bit of a float64.  Defined by the instrument, not here, so the
#: gate and the note's F.0 quote one number; ``ins.rebuild_report()`` measures
#: what the distance actually is, and F.0 prints that measurement.
#:
#: Not a tolerance on the mesh: the five graded arms were solved inside the
#: GPU job's container (numpy on python 3.10) and this gate runs in the
#: project venv (numpy 2.4 on python 3.11), and the two disagree in the last
#: bit or two of ``np.sum`` and of the power ufunc, which is what the
#: geometric ramp's ratio is bisected on.  Measured 2026-09-22: at most 4
#: cells of one profile and 7 of one arm moved, by at most 6 ulp; 8 leaves two
#: bits of headroom.  Everything the mesh DECLARES is compared with ``==``
#: below and is bit-identical.
RECORD_ULP_FLOOR = ins.RECORD_ULP_FLOOR

#: Fields the record holds that ``profiles`` or ``build_graded`` produces and
#: that are exact float arithmetic on the case's constants -- no reduction, no
#: bisection -- so the only honest bar is bit equality.
EXACT_SCALARS = (
    "rung", "placement", "arm_length_m", "fine_cell_m", "substrate_cell_m",
    "coarse_cell_m", "z_tail_cell_m", "n_across_metal", "n_substrate_cells",
    "margin_cells", "edge_offset_m", "lateral_clearance_m",
    "drawn_stub_open_y_m", "drawn_sheet_z_m", "n_interior_cells",
)
EXACT_MAPPINGS = ("declared", "node_index", "segment_cells")
#: Cumulative sums over the ramp cells: they inherit the ramp's last bit.
NEAR_SEQUENCES = ("domain_m", "band_x_m", "band_y_trace_m",
                  "band_y_stub_end_m", "band_z_m", "drawn_trace_y_m",
                  "drawn_stub_x_m")


def _ulps(a: float, b: float) -> float:
    """Distance between two float64 in units of the last bit of the larger."""
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("inf")
    return abs(a - b) / math.ulp(max(abs(a), abs(b)))


@pytest.fixture(scope="module")
def recorded_arms():
    assert RECORD.is_file(), f"the record is missing: {RECORD}"
    with RECORD.open() as fh:
        return json.load(fh)["arms"]


@pytest.mark.parametrize("key", RECORDED_GRADED_ARMS)
def test_the_recorded_arms_meshes_are_rebuilt_from_the_record(recorded_arms,
                                                              key):
    """Every graded arm already in the record rebuilds to the mesh it recorded.

    This is the regression the FZ pre-declaration requires before the rung
    spec is generalized.  The five arms were solved on GPUs and their meshes
    written into the record; a rung table, band arithmetic or runway solver
    that moves any of them makes the record's notch frequencies unattributable
    to a mesh, so it has to be red here.

    The comparison is against STORED data, never against the same expression
    evaluated twice.  It is split by what each number is: everything the mesh
    declares is compared with ``==``, and only the cells whose size comes out
    of a bisection on a numpy reduction are allowed the float64 last bit
    (``RECORD_ULP_FLOOR``, whose comment carries the measurement).  A rung
    changed by one cell moves these numbers by about 1e13 ulp.
    """
    _assert_rebuilds_to(recorded_arms[key], ins.ARMS[key], key)


def _assert_rebuilds_to(rec: dict, arm, key: str) -> None:
    """The rebuild comparison itself, against one stored arm record.

    Shared by the first record's arms and by the third note's arms, which are
    compared against their namesakes in the second record: one comparison, so
    the two gates cannot drift apart in what they call identical.
    """
    x, y, z, board = ins.profiles(arm.rung, arm.placement, arm.arm_length_m)
    prof_report = ins.assert_profiles_graded((x, y, z), board)
    sim, _prof, built = ins.build_graded(arm.rung, arm.placement,
                                         arm.arm_length_m)
    assert sim is not None
    got = dict(built, rung=arm.rung, placement=arm.placement,
               n_interior_cells=int(x.size * y.size * z.size),
               lateral_clearance_m=ins.LATERAL_CLEARANCE_M)

    for field in EXACT_SCALARS:
        assert got[field] == rec[field], f"{key}: {field} moved"
    for field in EXACT_MAPPINGS:
        assert dict(got[field]) == dict(rec[field]), f"{key}: {field} moved"
    assert [int(v) for v in rec["interior_shape"]] == [x.size, y.size, z.size]

    worst = 0.0
    for axis, p in zip("xyz", (x, y, z)):
        stored = np.asarray(rec["profiles"][axis], dtype=float)
        assert stored.size == p.size, f"{key}: d{axis}_profile changed length"
        # Every cell the mesh DECLARES -- the fine band's own cell and the
        # coarse cell -- bit-for-bit, and in the same places.
        fine = board["substrate_cell_m"] if axis == "z" else board["fine_cell_m"]
        for size, what in ((fine, "fine"), (ins.C_COARSE_M, "coarse")):
            assert np.array_equal(p == size, stored == size), (
                f"{key}: the {what} cells of d{axis}_profile are not where the "
                f"record puts them")
        ramp = ~((p == fine) | (p == ins.C_COARSE_M))
        for a, b in zip(p[ramp].tolist(), stored[ramp].tolist()):
            worst = max(worst, _ulps(a, b))
        # The profile sum IS the domain the arm solved, and it closes exactly.
        assert float(np.sum(p)) == float(np.sum(stored)), (key, axis)

    for field in NEAR_SEQUENCES:
        for a, b in zip(got[field], rec[field]):
            worst = max(worst, _ulps(a, b))
    for label, value in got["node"].items():
        worst = max(worst, _ulps(value, rec["node"][label]))
    for axis in "xyz":
        for label, value in prof_report[axis].items():
            if isinstance(value, float):
                worst = max(worst,
                            _ulps(value, rec["profile_summary"][axis][label]))
            else:
                assert value == rec["profile_summary"][axis][label], (key, axis,
                                                                     label)
    assert worst <= RECORD_ULP_FLOOR, (
        f"{key}: a rebuilt mesh number is {worst:.1f} ulp from the record, "
        f"past the {RECORD_ULP_FLOOR:.0f} ulp this machine's numpy explains")

    # R1's own residues are round-off either way, and both sides are ten orders
    # of magnitude inside the 1 nm the refusal allows.
    for label, miss in prof_report["intended_node_miss_m"].items():
        assert miss <= 1e-16 and rec["intended_node_miss_m"][label] <= 1e-16, (
            key, label)
        assert miss <= ins.NODE_TOL_M


# ------------------------------- the third note's arms against their namesakes
FZ_RECORD = RECORD.parent / "msl_notch_graded_fz.json"


@pytest.fixture(scope="module")
def recorded_fz_arms():
    assert FZ_RECORD.is_file(), f"the record is missing: {FZ_RECORD}"
    with FZ_RECORD.open() as fh:
        return json.load(fh)["arms"]


@pytest.mark.parametrize("key", sorted(ins.AFTER_1213_ARMS))
def test_the_after_1213_arms_rebuild_to_their_r2_namesakes_meshes(
        recorded_fz_arms, key):
    """Each of the third note's arms builds the mesh its R2 namesake solved.

    ``docs/design_notes/20260923_msl_notch_fz_after_1213_predeclaration.md``
    section 2: the only intended difference between an arm and its namesake
    is the ``rfx/`` tree, and this is shown before anything is submitted.
    The arm IS the namesake's Arm object, and its rebuild on this tree is
    compared with the profiles and fields the namesake's GPU job stored, by
    the same comparison the first record's arms go through.  Run on the
    revert branch, the ``Z6r`` row is the same statement for that tree.
    """
    namesake = ins.AFTER_1213_NAMESAKES[key]
    arm = ins.AFTER_1213_ARMS[key]
    assert arm is ins.FZ_ARMS[namesake]
    _assert_rebuilds_to(recorded_fz_arms[namesake], arm,
                        f"{key} against {namesake}")


def test_the_rebuild_comparison_refuses_a_mesh_one_ramp_cell_off(
        recorded_fz_arms):
    """The comparison above, handed a stored mesh it must not accept.

    One ramp cell of Z6's stored y profile moved by 1e-12 of itself -- about
    4500 ulp, a thousand times the floor and nothing like a rung change -- and
    the comparison has to refuse.  An emptied comparison passes every arm and
    this record alike, so this is the case that makes it red.
    """
    rec = json.loads(json.dumps(recorded_fz_arms["Z6"]))
    arm = ins.AFTER_1213_ARMS["Z6m"]
    y = rec["profiles"]["y"]
    board = ins.profiles(arm.rung, arm.placement, arm.arm_length_m)[3]
    ramp = [i for i, c in enumerate(y)
            if c not in (board["fine_cell_m"], ins.C_COARSE_M)]
    y[ramp[0]] *= 1.0 + 1e-12
    with pytest.raises(AssertionError):
        _assert_rebuilds_to(rec, arm, "Z6m against a doctored Z6")


def test_the_after_1213_arms_are_their_own_record_kind():
    """Which file each of the third note's arms lands in, and no overlap."""
    assert sorted(ins.AFTER_1213_ARMS) == ["Cm", "Z16m", "Z6m", "Z6r", "Z8m"]
    assert ins.AFTER_1213_NAMESAKES == {"Z6m": "Z6", "Z8m": "Z8",
                                        "Cm": "C_off_re", "Z16m": "Z16",
                                        "Z6r": "Z6"}
    assert set(ins.AFTER_1213_ARMS) & (set(ins.ARMS) | set(ins.FZ_ARMS)
                                       | set(ins.UNIFORM_ARMS)) == set()
    for key in ins.AFTER_1213_ARMS:
        assert ins.record_kind(key) == "after_1213"
        assert ins.default_out(key) == ins.DEFAULT_AFTER_1213_OUT
        assert ins.ARM_TABLE[key] is ins.AFTER_1213_ARMS[key]
    assert ins.DEFAULT_AFTER_1213_OUT.name == "msl_notch_graded_fz_after_1213.json"


# ------------------------------------------------------- the FZ ladder's rungs
#: The five arms of
#: ``docs/design_notes/20260922_msl_notch_fz_ladder_predeclaration.md``.  Every
#: one is built and put through R1 below, which is pure arithmetic on the
#: profiles and costs nothing.
FZ_ARMS_BUILT = ("Z6", "Z8", "Z16", "F16Z6", "ON13")
#: The two the lattice-level refusals and the mutations run on.  R2, R3, the
#: probe check and the port footprint each assemble the whole grid, and on
#: these meshes that is 10 to 270 seconds an arm, so this file runs them where
#: they can find something the other arms cannot: ON13 is the only arm with an
#: odd number of cells across the metal (its feed sits on no node) and Z6 is a
#: substrate cell and an in-plane cell the rung table never paired before.
#: Every arm gets the full set anyway before it is submitted -- that is what
#: ``--dry-run`` does, and the record carries the result per arm.
FZ_LATTICE_ARMS = ("ON13", "Z6")


@pytest.fixture(scope="module")
def fz_built():
    """The two FZ arms the lattice tests use, built and measured once."""
    out = {}
    for key in FZ_LATTICE_ARMS:
        arm = ins.FZ_ARMS[key]
        sim, prof, board = ins.build_graded(arm.rung, arm.placement,
                                            arm.arm_length_m)
        out[key] = (sim, prof, board, ins.realized_graded(sim))
    return out


@pytest.mark.parametrize("key", FZ_ARMS_BUILT)
def test_r1_holds_on_every_new_rung(built, key):
    """R1 on all five new arms: nodes, ratios and runways.

    The rung spec is now a ``(n_z, n)`` pair with the two independent, so
    these meshes pair a substrate cell and an in-plane cell the first note
    never put together.  Nothing about R1 changes for them.
    """
    arm = ins.FZ_ARMS[key]
    x, y, z, board = ins.profiles(arm.rung, arm.placement, arm.arm_length_m)
    report = ins.assert_profiles_graded((x, y, z), board)

    assert np.all(x[:ins.RUNWAY_CELLS] == ins.C_COARSE_M)
    assert np.all(x[-ins.RUNWAY_CELLS:] == ins.C_COARSE_M)
    assert np.all(y[:ins.RUNWAY_CELLS] == ins.C_COARSE_M)
    assert np.all(y[-ins.RUNWAY_CELLS:] == ins.C_COARSE_M)
    assert len(set(z[-ins.RUNWAY_CELLS:].tolist())) == 1
    assert z[-1] <= ins.C_COARSE_M
    assert report["x"]["worst_ratio"] <= ins.CAP_XY + 1e-9
    assert report["y"]["worst_ratio"] <= ins.CAP_XY + 1e-9
    assert report["z"]["worst_ratio"] <= ins.CAP_Z + 1e-9
    assert max(report["intended_node_miss_m"].values()) <= ins.NODE_TOL_M

    n_z, n = ins.rung_spec(arm.rung)
    assert board["n_substrate_cells"] == n_z
    assert board["n_across_metal"] == n
    # The substrate band is 2*n_z cells of exactly FZ from the ground plane up.
    assert np.all(z[:2 * n_z] == board["substrate_cell_m"])
    assert float(np.sum(z[:n_z])) == pytest.approx(
        case.SUBSTRATE_THICKNESS_M, abs=ins.NODE_TOL_M)
    # And the in-plane band is one size across the metal and its margin.
    m = board["margin_cells"]
    assert np.sum(x == board["fine_cell_m"]) >= n + 2 * m
    assert np.sum(y == board["fine_cell_m"]) >= n + 2 * m


def test_r1_refuses_a_straddled_centre_that_is_not_symmetric():
    """The odd-n arm's own R1 statement, refused when it is false.

    Thirteen cells across the metal put no node on its centre line, so R1
    asks instead that the two rows either side be one fine cell apart with the
    centre exactly halfway.  Moving one of those rows by a tenth of a cell has
    to be red: without this the odd-n path would assert nothing at all.
    """
    arm = ins.FZ_ARMS["ON13"]
    x, y, z, board = ins.profiles(arm.rung, arm.placement, arm.arm_length_m)
    assert board["centre_is_a_node"] is False
    ins.assert_profiles_graded((x, y, z), board)          # the real mesh

    f = board["fine_cell_m"]
    for label, needle in (("trace_y_centre", "straddle"),
                          ("stub_x_centre", "straddle")):
        st = board["centre_straddle"][label]
        lo, hi = st["nodes_m"]
        bad = dict(board, centre_straddle=dict(
            board["centre_straddle"],
            **{label: dict(st, nodes_m=(lo - 0.1 * f, hi))}))
        with pytest.raises(AssertionError, match=needle):
            ins.assert_profiles_graded((x, y, z), bad)
    # A pair that is symmetric but a whole cell wide apart is refused too.
    st = board["centre_straddle"]["trace_y_centre"]
    lo, hi = st["nodes_m"]
    bad = dict(board, centre_straddle=dict(
        board["centre_straddle"],
        trace_y_centre=dict(st, nodes_m=(lo - f, hi + f))))
    with pytest.raises(AssertionError, match="apart and this arm"):
        ins.assert_profiles_graded((x, y, z), bad)

    # And the check cannot be switched off by dropping the key.  Reading it
    # with .get() made exactly that possible: a board without the entry ran an
    # empty loop and passed, so the odd-n refusal was absent rather than met.
    dropped = {k: v for k, v in board.items() if k != "centre_straddle"}
    with pytest.raises(AssertionError, match="no 'centre_straddle' entry"):
        ins.assert_profiles_graded((x, y, z), dropped)


def test_an_even_rung_carries_an_empty_straddle_rather_than_no_key():
    """The key is always set, so its absence can mean one thing only.

    An even rung has a node on the metal's centre and nothing to straddle, and
    it says so with an empty mapping.  If it left the key out instead, the
    refusal above could not tell "nothing to check" from "this board did not
    come from profiles()".
    """
    arm = ins.FZ_ARMS["Z6"]
    x, y, z, board = ins.profiles(arm.rung, arm.placement, arm.arm_length_m)
    assert board["centre_is_a_node"] is True
    assert board["centre_straddle"] == {}
    assert "centre_straddle" in board
    ins.assert_profiles_graded((x, y, z), board)
    dropped = {k: v for k, v in board.items() if k != "centre_straddle"}
    with pytest.raises(AssertionError, match="no 'centre_straddle' entry"):
        ins.assert_profiles_graded((x, y, z), dropped)


@pytest.mark.parametrize("key", FZ_LATTICE_ARMS)
def test_r3_the_lattice_built_the_declared_board_on_a_new_rung(fz_built, key):
    _sim, _prof, board, g = fz_built[key]
    n = board["n_across_metal"]
    f = board["fine_cell_m"]
    ins._check_realized_fields(g, board)

    assert g["n_sheet_planes"] == 1
    assert g["n_volume_cells"] == 0
    assert g["eps_below_sheet"] == pytest.approx(case.EPS_R_SUBSTRATE)
    assert g["eps_above_sheet"] == 1.0
    assert g["sheet_plane_z_m"] == pytest.approx(case.SUBSTRATE_THICKNESS_M,
                                                 abs=ins.NODE_TOL_M)
    assert g["n_trace_rows"] == g["n_stub_cols"] == n + 1
    assert g["trace_width_node_span_m"] == pytest.approx(n * f,
                                                         abs=ins.NODE_TOL_M)
    assert g["stub_attached"] and g["stub_contiguous"]
    assert g["stub_length_m"] == pytest.approx(case.STUB_LENGTH_M,
                                               abs=ins.NODE_TOL_M)
    assert g["n_ports"] == 2
    # The metal is solved at its drawn size on the offset arm and 0.7 of a
    # cell wider on the on-node one -- the same statement as the first note's
    # two arms, on a rung that did not exist then.
    solved = g["trace_width_node_span_m"] + 2.0 * ins.EDGE_OFFSET * f
    if board["edge_offset_m"] == 0.0:
        assert solved > case.TRACE_WIDTH_M * 1.05
    else:
        assert solved == pytest.approx(case.TRACE_WIDTH_M, abs=1e-12)


def test_the_feed_of_the_odd_rung_sits_on_the_metals_centre(fz_built):
    """ON13's port goes where the metal's centre is, not on the nearest node.

    The MSL port lays its width span at the coordinate it is given plus and
    minus half the trace width, so a node half a fine cell off centre would
    drive the strip asymmetrically.  The other arms are unchanged: their
    centre IS a node and that node is what they pass.
    """
    _sim, _prof, board, _g = fz_built["ON13"]
    f = board["fine_cell_m"]
    assert board["centre_is_a_node"] is False
    assert board["port_centre_y_m"] == board["trace_centre_y_m"]
    lo, hi = board["centre_straddle"]["trace_y_centre"]["nodes_m"]
    assert board["port_centre_y_m"] - lo == pytest.approx(f / 2.0, abs=1e-12)
    assert hi - board["port_centre_y_m"] == pytest.approx(f / 2.0, abs=1e-12)

    _sim2, _p2, even, _g2 = fz_built["Z6"]
    assert even["centre_is_a_node"] is True
    assert even["port_centre_y_m"] == even["node"]["trace_y_centre_node"]


def test_r2_and_the_port_checks_hold_on_a_new_rung(fz_built):
    """R2, the probe runway and the feed footprint on the odd-n arm."""
    sim, _prof, board, _g = fz_built["ON13"]
    report = ins.assert_preflight_graded(sim)
    assert report, "preflight returned nothing at all"
    for line in report:
        for needle in ins.R2_FORBIDDEN:
            assert needle not in line

    planes = ins.probe_planes(sim)
    runway = ins.assert_probes_on_the_uniform_runway(sim, ins.C_COARSE_M)
    assert len(planes) == 2 and len(runway["ports"]) == 2
    for port in runway["ports"]:
        assert port["n_cells_off_coarse"] == 0

    got = ins.assert_port_footprint_in_fine_band(sim, board)
    lo, hi = board["band_y_trace_m"]
    for port in got["ports"]:
        assert lo <= port["y_lo_m"] and port["y_hi_m"] <= hi
        assert port["pad_cells_lo"] >= 1 and port["pad_cells_hi"] >= 1
        assert port["eps_r_sub"] == pytest.approx(case.EPS_R_SUBSTRATE)


# --------------------------------------- the three mutations, on the new rungs
@pytest.mark.parametrize("key", FZ_LATTICE_ARMS)
@pytest.mark.parametrize("field,value,needle", [
    ("n_trace_rows", 99, "node rows"),
    ("n_volume_cells", 7, "PEC VOLUME"),
    ("stub_attached", False, "not joined"),
    ("stub_length_m", case.STUB_LENGTH_M - 1e-6, "realized stub length"),
    ("trace_width_node_span_m", 1e-3, "node span"),
    ("eps_above_sheet", case.EPS_R_SUBSTRATE, "not air"),
])
def test_mutation_a_on_a_new_rung(fz_built, key, field, value, needle):
    """Mutation (a) on the FZ rungs: the R3 body itself.

    A real measurement of the arm with one field moved.  An emptied check body
    passes every arm and every mutant alike, and a rung table that grew
    without this would carry that hole into five new arms.
    """
    _sim, _prof, board, measured = fz_built[key]
    g = dict(measured)
    ins._check_realized_fields(g, board)          # the untouched measurement
    g[field] = value
    with pytest.raises(AssertionError, match=needle):
        ins._check_realized_fields(g, board)


def test_mutation_b_on_a_new_rung_a_stub_one_coarse_cell_short():
    """Mutation (b) on ON13: the declared board, stub 127 um short.

    Every helper the good arm calls is called here, so the refusal can only
    come from the measurement.
    """
    arm = ins.FZ_ARMS["ON13"]
    sim, prof, board = ins.build_graded(
        arm.rung, arm.placement, arm.arm_length_m,
        _mutate_stub_short_m=ins.C_COARSE_M)
    ins.assert_profiles_graded(prof, board)       # the MESH is untouched
    with pytest.raises(AssertionError, match="realized stub length"):
        ins.assert_realized_graded(sim, board)


def test_mutation_c_on_a_new_rung_the_offset_arm_drawn_to_its_nodes():
    """Mutation (c) on Z6: the offset mesh with the sheets drawn to their nodes.

    The realized node count, the stub length and the joint are all still
    right, so only the edge-placement check can see it.  ON13 cannot carry
    this mutation: its declared offset is already zero, so drawing to the
    nodes IS the arm.
    """
    arm = ins.FZ_ARMS["Z6"]
    assert arm.placement == "offset"
    sim, prof, board = ins.build_graded(
        arm.rung, arm.placement, arm.arm_length_m, _mutate_draw_offset=0.0)
    ins.assert_profiles_graded(prof, board)
    g = ins.realized_graded(sim)
    assert g["n_trace_rows"] == board["n_across_metal"] + 1
    assert g["stub_length_m"] == pytest.approx(case.STUB_LENGTH_M,
                                               abs=ins.NODE_TOL_M)
    with pytest.raises(AssertionError, match="outermost realized node"):
        ins.assert_realized_graded(sim, board)


def test_the_two_records_never_hold_the_same_arm():
    """Which file an arm belongs in, and that no arm belongs in both."""
    assert set(ins.ARMS) & set(ins.FZ_ARMS) == set()
    assert set(ins.UNIFORM_ARMS) & set(ins.FZ_ARMS) == set()
    for key in ins.ARMS:
        assert ins.record_kind(key) == "graded"
        assert ins.default_out(key) == ins.DEFAULT_OUT
    for key in ins.UNIFORM_ARMS:
        assert ins.record_kind(key) == "graded"
    for key in ins.FZ_ARMS:
        assert ins.record_kind(key) == "fz"
        assert ins.default_out(key) == ins.DEFAULT_FZ_OUT
    assert ins.default_out(None) == ins.DEFAULT_OUT
    with pytest.raises(SystemExit, match="no record holds"):
        ins.record_kind("D_off")
    # The first record's arm list is what its own replay pins, and it did not
    # grow: the FZ arms are not in it.
    assert sorted(ins.ALL_ARMS) == ["A_off", "A_off_longarms", "A_on", "B_off",
                                    "C_off", "U_h2", "U_h4", "U_h6"]


# ------------------------------------------- meshes costed but not solved
def test_a_costed_mesh_is_built_and_refused_like_an_arm():
    """F.4b: a cost quoted for a mesh nobody built is a guess.

    The row goes through the same ``build_graded``, R1 and R3 an arm does, so
    a mesh that could not be solved cannot be costed.  Nothing is stepped.
    """
    rows = ins.build_only_rows()
    assert len(rows) == len(ins.BUILD_ONLY_MESHES)
    for r, ((n_z, n), placement, _note) in zip(rows, ins.BUILD_ONLY_MESHES):
        assert r["solved"] is False
        assert r["n_substrate_cells"] == n_z and r["n_across_metal"] == n
        assert r["placement"] == placement
        assert r["substrate_cell_m"] == pytest.approx(
            case.SUBSTRATE_THICKNESS_M / n_z)
        assert r["fine_cell_m"] == pytest.approx(
            ins.fine_cell((n_z, n), placement))
        assert r["worst_ratio"]["x"] <= ins.CAP_XY + 1e-9
        assert r["worst_ratio"]["y"] <= ins.CAP_XY + 1e-9
        assert r["worst_ratio"]["z"] <= ins.CAP_Z + 1e-9
        assert r["n_grid_cells"] == (r["grid_shape"][0] - 1) * (
            r["grid_shape"][1] - 1) * (r["grid_shape"][2] - 1)
        # The step count is the one a run would derive from this dt.
        assert r["n_time_steps_derived"] == int(np.ceil(
            case.NUM_PERIODS / case.FREQ_MAX_HZ / r["dt_s"]))
        assert r["num_periods"] == case.NUM_PERIODS


def test_the_costed_mesh_borrows_both_cells_from_recorded_arms(
        recorded_arms):
    """The cheap path is A_off's in-plane mesh with Z16's substrate mesh.

    Both halves are cells this record measured, so the row is a cost for a
    combination of two solved meshes rather than for a new one.
    """
    arms = recorded_arms
    row = ins.build_only_rows()[0]
    assert row["fine_cell_m"] == pytest.approx(arms["A_off"]["fine_cell_m"])
    assert row["n_across_metal"] == arms["A_off"]["n_across_metal"]
    fz = ins.load_fz_arms()
    assert row["substrate_cell_m"] == pytest.approx(
        fz["Z16"]["substrate_cell_m"])
    assert row["n_substrate_cells"] == fz["Z16"]["n_substrate_cells"]
    # Cheaper than the ladder's finest rung, which is the point of quoting it.
    assert row["n_grid_cells"] < ins.grid_cells(fz["Z16"])
    assert row["dt_s"] > fz["Z16"]["dt_s"]

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
#: of the last bit of a float64 (``math.ulp``).  Not a tolerance on the mesh:
#: the five graded arms were solved inside the GPU job's container (numpy on
#: python 3.10) and this gate runs in the project venv (numpy 2.4 on python
#: 3.11), and the two disagree in the last bit or two of ``np.sum`` and of the
#: power ufunc, which is what the geometric ramp's ratio is bisected on.
#: Measured across the five arms on 2026-09-22: at most 6 ulp, on at most 4 of
#: a profile's 8-27 ramp cells; 8 leaves two bits of headroom.  Everything the
#: mesh DECLARES -- the fine cell, the substrate cell, the solved z tail cell,
#: every cell of every fine band, every cell of every coarse run, every
#: declared coordinate, every node index and every segment length -- is
#: compared with ``==`` below and is bit-identical.  In metres the worst cell
#: disagreement is 8e-16 of its own size, on cells that sit in the coarse
#: transition away from the metal; no cell that touches the line, the stub or
#: the substrate is among them.
RECORD_ULP_FLOOR = 8.0

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
    rec = recorded_arms[key]
    arm = ins.ARMS[key]
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

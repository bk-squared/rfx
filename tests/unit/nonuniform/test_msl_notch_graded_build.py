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
def test_every_probe_plane_is_in_the_uniform_coarse_region(built, key):
    sim, _prof, board = built[key]
    planes = ins.probe_planes(sim)
    ins.assert_probes_in_uniform_region(planes, board)
    assert len(planes) == 2
    for xs in planes:
        assert len(xs) == ins.N_PROBES


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

"""Replay of the graded-mesh notch arms from their record -- no FDTD.

The five graded arms of
``docs/design_notes/20260922_msl_notch_graded_mesh_predeclaration.md`` and the
three uniform reference arms its section A.1 declares were solved on GPUs and
their curves committed to
``validation/research/multiband_nu/results/msl_notch_graded.json``.  This file
re-derives, from those curves and with the instrument's own functions, every
notch frequency, the four frozen windows' arithmetic, the uniform ladder's own
mesh statement and the cost ratios, and pins what the record says each one was.

Both sides of every cost ratio, and both limits W3 compares, come from arms in
this file.  Nothing here is divided by a number typed from a ledger.

Pinned means pinned: a window that FIRED is pinned fired.  The pre-declaration
allows one attempt per arm, so the alternative to pinning a fired window is
losing the fact that it fired.

Nothing here builds a ``Simulation`` or steps a field -- the guard below
replaces the instrument's ``Simulation`` with one that raises, so a future edit
that makes a window function solve is caught rather than merely slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.research.multiband_nu import msl_notch_graded as ins
from tests.crossval.msl_notch_filter import test_msl_notch_filter as case

RESULTS = (Path(__file__).resolve().parents[3] / "validation" / "research"
           / "multiband_nu" / "results" / "msl_notch_graded.json")

#: What the record holds, typed from the committed file.  Frequencies in GHz.
RECORDED_NOTCH_GHZ = {
    "A_off": 3.747389,
    "A_off_longarms": 3.747310,
    "A_on": 3.732318,
    "B_off": 3.732214,
    "C_off": 3.716635,
    "U_h2": 3.875690,
    "U_h4": 3.794409,
    "U_h6": 3.748393,
}
RECORDED_VERDICTS = {"W1": "HELD", "W2": "FIRED", "W3": "FIRED", "W4": "HELD"}
RECORDED_W1_LAST_TWO_PCT = 0.4174
RECORDED_W2_NOTCH_PCT = 1.1507
RECORDED_W2_MAX_DB = 2.7874
RECORDED_W3_ORDER = 0.9225
RECORDED_W3_LIMIT_GHZ = 3.68229
RECORDED_W3_DISTANCE_PCT = 97.6400
RECORDED_W4_S21_DB = 0.1346
RECORDED_EDGE_OFFSET_PCT = -0.4022
RECORDED_UNIFORM_ORDER = 0.0595
RECORDED_UNIFORM_LIMIT_GHZ = 1.86313
RECORDED_UNIFORM_LAST_TWO_PCT = 1.2127
RECORDED_UNIFORM_MESH_STATEMENT = "FIRED"
RECORDED_U_H6_CELLS = 13438150
RECORDED_CELLS_RATIO = {
    "A_off": 0.11502,
    "A_off_longarms": 0.15502,
    "A_on": 0.11184,
    "B_off": 0.15132,
    "C_off": 0.24494,
    "U_h2": 0.05538,
    "U_h4": 0.33037,
    "U_h6": 1.00000,
}
RECORDED_WALL_S = {
    "A_off": 106.1,
    "A_off_longarms": 129.5,
    "A_on": 101.4,
    "B_off": 333.1,
    "C_off": 1007.7,
    "U_h2": 19.2,
    "U_h4": 135.8,
    "U_h6": 672.0,
}


@pytest.fixture(scope="module")
def arms():
    assert RESULTS.is_file(), f"the record is missing: {RESULTS}"
    with RESULTS.open() as fh:
        return json.load(fh)["arms"]


@pytest.fixture(autouse=True)
def _no_solver(monkeypatch):
    """This file replays a record; it may not build or step anything."""
    class _Refused:
        def __init__(self, *a, **k):
            raise AssertionError(
                "the replay built a Simulation: every number here comes from "
                "the committed curves, never from a new solve")

    monkeypatch.setattr(ins, "Simulation", _Refused)


def test_the_record_holds_every_declared_arm(arms):
    assert sorted(arms) == sorted(ins.ALL_ARMS)
    for key, rec in sorted(arms.items()):
        assert rec["arm"] == key
        assert rec["smoke"] is False, f"{key} was recorded from a capped run"
        assert len(rec["freqs_hz"]) == case.N_FREQS
        assert rec["num_periods"] == case.NUM_PERIODS
        for field in ("s11_re", "s11_im", "s21_re", "s21_im"):
            assert len(rec[field]) == case.N_FREQS, (key, field)


def test_every_arm_names_the_run_and_the_commit_that_made_it(arms):
    """Provenance, read rather than assumed.

    The first A_off attempt (VESSL 369367263251) recorded no commit at all: the
    job exports the pinned tree with ``git archive`` into scratch, so git had
    nothing to read inside it.  The instrument now refuses an arm it cannot
    name a commit for, and this is where that refusal is worth something.
    """
    runs = set()
    for key, rec in sorted(arms.items()):
        p = rec["provenance"]
        assert ins._SHA_RE.match(p["git_sha"]), (key, p["git_sha"])
        assert p["run_id"] and str(p["run_id"]).isdigit(), (key, p["run_id"])
        assert p["jax_backend"] == "gpu", (key, p["jax_backend"])
        assert p["git_sha_source"], key
        assert p["run_id"] not in runs, f"{key} reuses run {p['run_id']}"
        runs.add(p["run_id"])
    # One commit per ladder: the graded arms ran before the uniform reference
    # arms were declared (note A.1), so two commits, not one.
    graded = {arms[k]["provenance"]["git_sha"] for k in ins.ARMS}
    uniform = {arms[k]["provenance"]["git_sha"] for k in ins.UNIFORM_LADDER}
    assert len(graded) == 1, f"the graded arms span {len(graded)} commits"
    assert len(uniform) == 1, f"the uniform arms span {len(uniform)} commits"
    # The GPU model is recorded wherever it was measured anew.
    for key in ins.UNIFORM_LADDER:
        assert arms[key]["provenance"]["gpu_device_kind"], key


def test_the_uniform_arms_ran_on_the_cases_own_board(arms):
    """A.1's load-bearing premise: the reference ladder is the case's board.

    If these arms had run on the graded arms' board the cost ratios would
    compare two different structures and W3's target would not be the uniform
    ladder at all.
    """
    for key in ins.UNIFORM_LADDER:
        rec = arms[key]
        assert rec["kind"] == "uniform"
        assert rec["lateral_clearance_m"] == pytest.approx(
            case.LATERAL_CLEARANCE_M)
        assert rec["lateral_clearance_m"] < ins.LATERAL_CLEARANCE_M
        assert rec["domain_m"] == pytest.approx(
            [case.DOMAIN_X_M, case.DOMAIN_Y_M, case.DOMAIN_Z_M])
        assert rec["dx_m"] == rec["fine_cell_m"] == rec["substrate_cell_m"]
    assert [arms[k]["dx_m"] for k in ins.UNIFORM_LADDER] == list(case.LADDER_M)
    for key in ins.ARMS:
        assert arms[key].get("kind") != "uniform"
        assert arms[key]["lateral_clearance_m"] == pytest.approx(
            ins.LATERAL_CLEARANCE_M)


def test_the_witnesses_are_recorded_for_every_arm(arms):
    """R4 on every arm, graded or uniform: the two the case defines."""
    for key, rec in sorted(arms.items()):
        w = rec["witnesses"]
        assert w["worst_settling_db"] <= case.SETTLING_DB, key
        assert w["worst_sigma_excess"] <= case.PASSIVITY_EXCESS_BAR, key
        g = rec["realized"]
        assert g["n_sheet_planes"] == 1 and g["n_volume_cells"] == 0, key
        assert abs(g["stub_length_m"] - case.STUB_LENGTH_M) <= max(
            ins.NODE_TOL_M, rec["fine_cell_m"]), key


def test_the_graded_refusals_are_recorded_for_every_graded_arm(arms):
    """R1, R2 and R3 evidence, as each graded arm recorded it before solving."""
    for key in sorted(ins.ARMS):
        rec = arms[key]
        assert max(rec["intended_node_miss_m"].values()) <= ins.NODE_TOL_M, key
        assert rec["profile_summary"]["x"]["worst_ratio"] <= ins.CAP_XY + 1e-9
        assert rec["profile_summary"]["y"]["worst_ratio"] <= ins.CAP_XY + 1e-9
        assert rec["profile_summary"]["z"]["worst_ratio"] <= ins.CAP_Z + 1e-9
        for line in rec["preflight"]:
            for needle in ins.R2_FORBIDDEN:
                assert needle not in line, (key, needle)
        g = rec["realized"]
        assert g["eps_above_sheet"] == 1.0, key
        n = rec["n_across_metal"]
        assert g["n_trace_rows"] == g["n_stub_cols"] == n + 1, key
        assert abs(g["stub_length_m"] - case.STUB_LENGTH_M) <= ins.NODE_TOL_M
        # Three fields the five graded arms predate: the edge residual and
        # the two refusals added after the first review.  Those arms are left
        # as they were measured rather than re-run for a field, so the check
        # is on whatever a record carries -- and every arm recorded from here
        # on carries all three, which is what the build test holds.
        res = g.get("edge_offset_residual_m")
        if res is not None:
            assert res <= 1e-16, key
        if "probe_runway" in rec:
            for port in rec["probe_runway"]["ports"]:
                assert port["n_cells_off_coarse"] == 0, (key, port["name"])
        if "port_footprint" in rec:
            lo, hi = rec["band_y_trace_m"]
            for port in rec["port_footprint"]["ports"]:
                assert lo <= port["y_lo_m"] and port["y_hi_m"] <= hi, key


@pytest.mark.parametrize("key", sorted(RECORDED_NOTCH_GHZ))
def test_each_notch_is_re_derived_from_the_curve(arms, key):
    """The shared estimator on the recorded |S21|, against the pinned value.

    W3 fired at 97.6 % and it is pinned fired: the uniform ladder's three
    notches differ by a ratio of 1.7664 where an order approaching zero already
    gives 1.7095, so the order that fits them is 0.059 and the limit it
    extrapolates to is 1.863 GHz.  Whether that says anything about the two
    lanes converging to one board is the leader's sentence, not this file's;
    what this file does is stop the number from moving unnoticed.
    """
    got = ins.notch_of(arms[key])
    assert got["f"] == pytest.approx(arms[key]["notch"]["f"], abs=1.0)
    assert got["f"] / 1e9 == pytest.approx(RECORDED_NOTCH_GHZ[key], abs=5e-6)
    assert got["depth_db"] == pytest.approx(arms[key]["notch"]["depth_db"],
                                            abs=1e-9)


def test_w1_mesh_statement(arms):
    r = ins.w1_mesh_statement(arms)
    assert r["last_two_pct"] == pytest.approx(RECORDED_W1_LAST_TWO_PCT, abs=1e-4)
    assert r["bar_pct"] == case.LADDER_AGREEMENT * 100.0
    assert r["verdict"] == RECORDED_VERDICTS["W1"]


def test_w2_comparison_against_the_openems_tutorial(arms):
    r = ins.w2_comparison(arms)
    assert r["notch_pct"] == pytest.approx(RECORDED_W2_NOTCH_PCT, abs=1e-4)
    assert r["max_abs_delta_db"] == pytest.approx(RECORDED_W2_MAX_DB, abs=1e-3)
    assert r["notch_bar_pct"] == case.FREQ_BAR * 100.0
    assert r["mag_bar_db"] == case.MAG_BAR_DB
    assert r["verdict"] == RECORDED_VERDICTS["W2"]


def test_w3_cross_ladder_consistency(arms):
    r = ins.w3_cross_ladder(arms)
    assert r["fitted_order"] == pytest.approx(RECORDED_W3_ORDER, abs=1e-3)
    assert r["limit_ghz"] == pytest.approx(RECORDED_W3_LIMIT_GHZ, abs=1e-4)
    assert r["distance_pct"] == pytest.approx(RECORDED_W3_DISTANCE_PCT, abs=1e-3)
    assert r["verdict"] == RECORDED_VERDICTS["W3"]
    # The target is the uniform ladder in this same file, fitted the same way.
    assert r["uniform_limit_ghz"] == pytest.approx(
        RECORDED_UNIFORM_LIMIT_GHZ, abs=1e-4)
    assert r["uniform_limit_ghz"] == pytest.approx(
        ins.uniform_ladder_statement(arms)["limit_ghz"], abs=1e-12)
    assert r["uniform_fitted_order"] == pytest.approx(
        RECORDED_UNIFORM_ORDER, abs=1e-3)
    assert not hasattr(ins, "UNIFORM_LADDER_LIMIT_GHZ")


def test_the_uniform_ladders_own_mesh_statement(arms):
    """Reported beside W1, on the case's own LADDER_AGREEMENT rule."""
    r = ins.uniform_ladder_statement(arms)
    assert r["notches_ghz"] == pytest.approx(
        [RECORDED_NOTCH_GHZ[k] for k in ins.UNIFORM_LADDER], abs=5e-6)
    assert r["last_two_pct"] == pytest.approx(
        RECORDED_UNIFORM_LAST_TWO_PCT, abs=1e-4)
    assert r["bar_pct"] == case.LADDER_AGREEMENT * 100.0
    assert r["mesh_statement"] == RECORDED_UNIFORM_MESH_STATEMENT
    assert r["fitted_order"] == pytest.approx(RECORDED_UNIFORM_ORDER, abs=1e-3)
    assert r["limit_ghz"] == pytest.approx(RECORDED_UNIFORM_LIMIT_GHZ, abs=1e-4)


def test_w4_arm_length_witness(arms):
    r = ins.w4_arm_length_witness(arms)
    assert r["max_abs_delta_s21_db"] == pytest.approx(RECORDED_W4_S21_DB,
                                                      abs=1e-3)
    assert r["bar_db"] == case.ARM_WITNESS_BAR_DB
    assert r["verdict"] == RECORDED_VERDICTS["W4"]


def test_what_the_edge_offset_is_worth(arms):
    """Reported, no window: A_on against A_off at the same rung."""
    v = ins.verdicts(arms)["edge_offset_worth"]
    assert v["delta_pct"] == pytest.approx(RECORDED_EDGE_OFFSET_PCT, abs=1e-4)


def test_the_cost_ratios(arms):
    """Both sides measured, and cells counted as cells."""
    out = ins.verdicts(arms)
    v, ref = out["cost"], out["cost_reference"]
    for key, ratio in sorted(RECORDED_CELLS_RATIO.items()):
        assert v[key]["cells_ratio"] == pytest.approx(ratio, abs=1e-5), key
        assert v[key]["wall_s"] == pytest.approx(RECORDED_WALL_S[key], abs=0.05)
        assert v[key]["n_grid_cells"] == ins.grid_cells(arms[key]), key
    assert ref["arm"] == ins.COST_REFERENCE_ARM == "U_h6"
    assert ref["n_grid_cells"] == RECORDED_U_H6_CELLS
    assert ref["wall_s"] == pytest.approx(RECORDED_WALL_S["U_h6"], abs=0.05)
    assert v["U_h6"]["cells_ratio"] == 1.0 and v["U_h6"]["wall_ratio"] == 1.0
    # The denominator is a measurement in this file, not a typed constant.
    assert not hasattr(ins, "UNIFORM_FINEST_CELLS")
    assert not hasattr(ins, "UNIFORM_FINEST_WALL_S")
    # A node count is not a cell count: the node product of U_h6's grid is
    # 2.4 % larger, which is the error the review caught.
    shape = arms["U_h6"]["grid_shape"]
    nodes = shape[0] * shape[1] * shape[2]
    assert nodes > ref["n_grid_cells"]
    assert nodes / ref["n_grid_cells"] == pytest.approx(1.024, abs=5e-3)


def test_the_ladder_is_read_in_the_declared_order(arms):
    """W1 and W3 walk A_off, B_off, C_off, and the mesh really does refine."""
    cells = [arms[k]["substrate_cell_m"] for k in ins.LADDER]
    fine = [arms[k]["fine_cell_m"] for k in ins.LADDER]
    assert cells == sorted(cells, reverse=True)
    assert fine == sorted(fine, reverse=True)
    assert [arms[k]["n_substrate_cells"] for k in ins.LADDER] == [6, 8, 12]


def test_the_two_arm_lengths_share_one_frequency_grid(arms):
    """W4 subtracts two curves; a different grid would hide a shift in it."""
    a = np.asarray(arms["A_off"]["freqs_hz"], float)
    b = np.asarray(arms["A_off_longarms"]["freqs_hz"], float)
    assert np.allclose(a, b)
    assert arms["A_off_longarms"]["arm_length_m"] == case.WITNESS_ARM_LENGTH_M
    assert arms["A_off"]["arm_length_m"] == case.ARM_LENGTH_M

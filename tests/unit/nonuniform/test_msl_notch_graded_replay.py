"""Replay of the graded-mesh notch arms from their record -- no FDTD.

The five arms of ``docs/design_notes/20260922_msl_notch_graded_mesh_predeclaration.md``
were solved on GPUs and their curves committed to
``validation/research/multiband_nu/results/msl_notch_graded.json``.  This file
re-derives, from those curves and with the instrument's own functions, every
notch frequency, the four frozen windows' arithmetic and the cost ratios, and
pins what the record says each one was.

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
    "A_on": 3.732318,
    "A_off": 3.747389,
    "B_off": 3.732214,
    "C_off": 3.716635,
    "A_off_longarms": 3.747310,
}
RECORDED_VERDICTS = {"W1": "HELD", "W2": "FIRED", "W3": "HELD", "W4": "HELD"}
RECORDED_W1_LAST_TWO_PCT = 0.4174
RECORDED_W2_NOTCH_PCT = 1.1507
RECORDED_W2_MAX_DB = 2.7874
RECORDED_W3_ORDER = 0.9225
RECORDED_W3_LIMIT_GHZ = 3.68229
RECORDED_W3_DISTANCE_PCT = 0.3349
RECORDED_W4_S21_DB = 0.1346
RECORDED_EDGE_OFFSET_PCT = -0.4022
RECORDED_CELLS_RATIO = {
    "A_on": 0.11335,
    "A_off": 0.11655,
    "B_off": 0.15263,
    "C_off": 0.24552,
    "A_off_longarms": 0.15692,
}
RECORDED_WALL_S = {
    "A_on": 101.4,
    "A_off": 106.1,
    "B_off": 333.1,
    "C_off": 1007.7,
    "A_off_longarms": 129.5,
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


def test_the_record_holds_the_five_declared_arms(arms):
    assert sorted(arms) == sorted(ins.ARMS)
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
    shas = set()
    for key, rec in sorted(arms.items()):
        p = rec["provenance"]
        assert ins._SHA_RE.match(p["git_sha"]), (key, p["git_sha"])
        assert p["run_id"] and str(p["run_id"]).isdigit(), (key, p["run_id"])
        assert p["jax_backend"] == "gpu", (key, p["jax_backend"])
        assert p["git_sha_source"], key
        shas.add(p["git_sha"])
    assert len(shas) == 1, f"the five arms were solved at {len(shas)} commits: {shas}"


def test_the_refusals_are_recorded_for_every_arm(arms):
    """R1, R2 and R3 evidence, as each arm recorded it before it solved."""
    for key, rec in sorted(arms.items()):
        assert max(rec["intended_node_miss_m"].values()) <= ins.NODE_TOL_M, key
        assert rec["profile_summary"]["x"]["worst_ratio"] <= ins.CAP_XY + 1e-9
        assert rec["profile_summary"]["y"]["worst_ratio"] <= ins.CAP_XY + 1e-9
        assert rec["profile_summary"]["z"]["worst_ratio"] <= ins.CAP_Z + 1e-9
        for line in rec["preflight"]:
            for needle in ins.R2_FORBIDDEN:
                assert needle not in line, (key, needle)
        g = rec["realized"]
        assert g["n_sheet_planes"] == 1 and g["n_volume_cells"] == 0, key
        assert g["eps_above_sheet"] == 1.0, key
        n = rec["n_across_metal"]
        assert g["n_trace_rows"] == g["n_stub_cols"] == n + 1, key
        assert abs(g["stub_length_m"] - case.STUB_LENGTH_M) <= ins.NODE_TOL_M
        w = rec["witnesses"]
        assert w["worst_settling_db"] <= case.SETTLING_DB, key
        assert w["worst_sigma_excess"] <= case.PASSIVITY_EXCESS_BAR, key


@pytest.mark.parametrize("key", sorted(RECORDED_NOTCH_GHZ))
def test_each_notch_is_re_derived_from_the_curve(arms, key):
    """The shared estimator on the recorded |S21|, against the pinned value."""
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
    assert r["uniform_limit_ghz"] == ins.UNIFORM_LADDER_LIMIT_GHZ
    assert r["verdict"] == RECORDED_VERDICTS["W3"]


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
    v = ins.verdicts(arms)["cost"]
    for key, ratio in sorted(RECORDED_CELLS_RATIO.items()):
        assert v[key]["cells_ratio"] == pytest.approx(ratio, abs=1e-5), key
        assert v[key]["wall_s"] == pytest.approx(RECORDED_WALL_S[key], abs=0.05)
    assert ins.UNIFORM_FINEST_CELLS == 13_800_000
    assert ins.UNIFORM_FINEST_WALL_S == 661.0


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

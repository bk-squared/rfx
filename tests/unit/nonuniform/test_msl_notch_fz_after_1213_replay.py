"""The substrate-cell ladder after #1213, re-derived -- no FDTD.

``docs/design_notes/20260923_msl_notch_fz_after_1213_predeclaration.md`` solves
the microstrip stub notch's substrate-cell ladder again on a tree that contains
#1213 (a tangential E component multiplies by the mean permittivity of the four
cells sharing its edge), and one rung again on that tree with #1213's ``rfx/``
change reverted.  Its four windows are arithmetic on nine measured notches --
five arms solved for the note and R2's four rungs reused -- and this file
re-derives all of it with the instrument's own functions and pins what the
records say each window was.

Four kinds of test live here.

* The windows on SYNTHETIC ladders whose answer is known, one per branch of
  each verdict.  These need no record.
* The windows on the committed records.  Pinned means pinned: a window that
  fired is pinned fired.
* The second note's Results section, byte for byte against what its
  ``--tables`` prints today, because the third note adds functions to the
  same instrument and may not change any of the second note's output.
* This note's own Results against its ``--tables``, its sections 0-7 against
  the pre-declaration commit, and the two earlier records and R2's replay
  against main's bytes.

Nothing here builds a ``Simulation`` or steps a field -- the guard below
replaces the instrument's ``Simulation`` with one that raises.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from validation.research.multiband_nu import msl_notch_graded as ins
from tests.crossval.msl_notch_filter import test_msl_notch_filter as case

_ROOT = Path(__file__).resolve().parents[3]
_RESULTS = _ROOT / "validation" / "research" / "multiband_nu" / "results"
RECORD = _RESULTS / "msl_notch_graded_fz_after_1213.json"
R2_RECORD = _RESULTS / "msl_notch_graded_fz.json"
BASE_RECORD = _RESULTS / "msl_notch_graded.json"


@pytest.fixture(autouse=True)
def _no_solver(monkeypatch):
    """This file replays records; it may not build or step anything."""
    class _Refused:
        def __init__(self, *a, **k):
            raise AssertionError(
                "the replay built a Simulation: every number here comes from "
                "the committed curves, never from a new solve")

    monkeypatch.setattr(ins, "Simulation", _Refused)


# ------------------------------------------------- what the note declares
def test_the_five_arms_are_section_2s_table():
    """Section 2, as cells rather than prose.

    Every arm IS its R2 namesake's Arm object, so the rung entry, the
    placement and the arm length are the ones already solved; the in-plane
    and substrate cells below are the note's micrometres recomputed from the
    case's own trace width and substrate thickness.
    """
    assert ins.AFTER_1213_NAMESAKES == {"Z6m": "Z6", "Z8m": "Z8",
                                        "Cm": "C_off_re", "Z16m": "Z16",
                                        "Z6r": "Z6"}
    assert ins.AFTER_1213_LADDER == ("Z6m", "Z8m", "Cm", "Z16m")
    assert ins.R2_LADDER == ins.FZ_LADDER_ONE_COMMIT == (
        "Z6", "Z8", "C_off_re", "Z16")
    assert ins.AFTER_1213_REVERT_ARM == "Z6r"
    want = {"Z6m": (24, 6, 42.333), "Z8m": (24, 8, 31.750),
            "Cm": (24, 12, 21.167), "Z16m": (24, 16, 15.875),
            "Z6r": (24, 6, 42.333)}
    for key, (n, n_z, fz_um) in want.items():
        arm = ins.AFTER_1213_ARMS[key]
        assert arm is ins.FZ_ARMS[ins.AFTER_1213_NAMESAKES[key]], key
        assert ins.rung_spec(arm.rung) == (n_z, n), key
        assert arm.placement == "offset" and arm.arm_length_m == (
            case.ARM_LENGTH_M), key
        f = ins.fine_cell(arm.rung, arm.placement)
        # Section 2 prints three decimals, 24.292; the cell is 24.29150 um.
        assert f * 1e6 == pytest.approx(24.2915, abs=5e-5), key
        assert case.SUBSTRATE_THICKNESS_M / n_z * 1e6 == pytest.approx(
            fz_um, abs=5e-4), key


def test_the_windows_constants_are_section_4s():
    """The frozen numbers, each once, where the windows read them."""
    assert ins.AFTER_1213_PRIMARY == "power"
    assert ins.W9_MOVED_BAR_HZ == 1.0e6
    assert ins.W9_ATTRIBUTION_BAR_HZ == 0.05e6
    assert ins.W10_INTERFACE_SPAN_FRACTION == 0.5
    assert ins.W10_INTERFACE_MIN_ORDER == 1.5
    assert ins.W10_NOT_INTERFACE_SPAN_FRACTION == 0.8
    assert ins.W11_BAR_PCT == 0.20
    rec = ins._empty_file("after_1213")
    assert rec["schema"] == "msl_notch_graded_fz_after_1213/1"
    assert rec["note"] == ins.AFTER_1213_NOTE_PATH
    assert (_ROOT / rec["note"]).is_file()
    assert rec["reused_arms"] == list(ins.R2_LADDER)
    w = rec["windows"]
    assert w["primary_transform"] == "power"
    assert w["W12_freq_bar_pct"] == case.FREQ_BAR * 100.0


# ------------------------------------------------------ synthetic records
_R2_FZ_UM = {"Z6": 42.33333333333333, "Z8": 31.75,
             "C_off_re": 21.166666666666664, "Z16": 15.875}
_FINE_M = 24.29149797570851e-6
#: Where the synthetic zero is: B and the floor of a lossy zero, on an evenly
#: spaced 400-bin grid across the reference band.
_B_HZ, _FLOOR = 0.4e9, 1e-5


def _curve(f0_hz: float) -> dict:
    """A |S21| whose |S21|^2 is ``((f - f0) / B)^2 + floor`` exactly.

    That is a parabola in f, so the primary estimator returns ``f0`` wherever
    it falls in its bin on an evenly spaced grid.
    """
    lo, hi = case.REFERENCE_BAND_HZ
    f = np.linspace(lo, hi, case.N_FREQS)
    mag = np.sqrt(((f - f0_hz) / _B_HZ) ** 2 + _FLOOR)
    return dict(freqs_hz=f.tolist(), s21_re=mag.tolist(),
                s21_im=np.zeros_like(mag).tolist())


def test_the_synthetic_curve_carries_the_notch_it_claims():
    for f0 in (3.6601e9, 3.7123e9, 3.75e9):
        got = ins.notch_of(_curve(f0), "power")["f"]
        assert got == pytest.approx(f0, abs=1.0), f0


def _arm(key: str, f0_hz: float, fz_um: float, tree: str = "a") -> dict:
    return dict(_curve(f0_hz), arm=key, fine_cell_m=_FINE_M,
                substrate_cell_m=fz_um * 1e-6,
                n_substrate_cells=int(round(
                    case.SUBSTRATE_THICKNESS_M / (fz_um * 1e-6))),
                profiles={"x": [1.0], "y": [1.0], "z": [fz_um]},
                realized={"n_sheet_planes": 1}, wall_s=1.0,
                grid_shape=[2, 2, 2],
                provenance=dict(git_sha=tree * 40, run_id="1",
                                rfx_tree=tree * 40))


#: R2's four |S21|^2 notches, the note's section 1.
_R2_GHZ = {"Z6": 3.752466, "Z8": 3.736288, "C_off_re": 3.718408,
           "Z16": 3.708800}


def _ladder(new_ghz: dict, z6r_ghz: float | None = None) -> dict:
    """R2's four rungs as the note quotes them, and a new ladder."""
    arms = {k: _arm(k, v * 1e9, _R2_FZ_UM[k], "d") for k, v in _R2_GHZ.items()}
    for key in ins.AFTER_1213_LADDER:
        old = ins.AFTER_1213_NAMESAKES[key]
        arms[key] = _arm(key, new_ghz[key] * 1e9, _R2_FZ_UM[old], "b")
    if z6r_ghz is not None:
        arms["Z6r"] = _arm("Z6r", z6r_ghz * 1e9, _R2_FZ_UM["Z6"], "e")
    return arms


def _power_ladder(f_inf_ghz: float, span_mhz: float, p: float) -> dict:
    """A new ladder that IS ``f_inf + A FZ^p`` with the given span."""
    h = [_R2_FZ_UM[ins.AFTER_1213_NAMESAKES[k]] for k in ins.AFTER_1213_LADDER]
    a = span_mhz * 1e-3 / (h[0] ** p - h[-1] ** p)
    return {k: f_inf_ghz + a * v ** p
            for k, v in zip(ins.AFTER_1213_LADDER, h)}


@pytest.mark.parametrize("delta_mhz,verdict", [
    (0.99, "did not move"), (1.01, "moved"), (-1.01, "moved"),
    (-0.99, "did not move")])
def test_w9_moved_is_read_off_z6_alone(delta_mhz, verdict):
    """|Z6m - Z6| against 1.0 MHz under |S21|^2; the other rungs say nothing."""
    new = {k: _R2_GHZ[ins.AFTER_1213_NAMESAKES[k]]
           for k in ins.AFTER_1213_LADDER}
    new["Z6m"] += delta_mhz * 1e-3
    new["Z16m"] -= 30.0e-3          # a big move elsewhere changes nothing
    r = ins.w9_board_moved(_ladder(new))
    assert r["delta_z6_hz"] == pytest.approx(delta_mhz * 1e6, abs=2.0)
    assert r["verdict"] == verdict
    assert r["transform"] == "power"
    assert [row["arm"] for row in r["rows"]] == list(ins.AFTER_1213_LADDER)
    assert [row["namesake"] for row in r["rows"]] == list(ins.R2_LADDER)
    assert r["revert_present"] is False
    assert r["attribution"] == "no attribution"


@pytest.mark.parametrize("rest_mhz,attribution", [
    (0.04, "#1213 alone"), (-0.04, "#1213 alone"),
    (0.06, "not #1213 alone"), (-0.06, "not #1213 alone")])
def test_w9_attribution_splits_z6_on_one_mesh(rest_mhz, attribution):
    """Z6r within 0.05 MHz of R2's Z6, or not; the two parts add to the whole."""
    new = {k: _R2_GHZ[ins.AFTER_1213_NAMESAKES[k]] - 5e-3
           for k in ins.AFTER_1213_LADDER}
    arms = _ladder(new, z6r_ghz=_R2_GHZ["Z6"] + rest_mhz * 1e-3)
    r = ins.w9_board_moved(arms)
    assert r["revert_present"] is True
    assert r["attribution"] == attribution
    for t in ins.TRANSFORMS:
        assert r["rest_of_diff_hz"][t] + r["change_1213_hz"][t] == (
            pytest.approx(r["rows"][0]["delta_hz"][t], abs=1e-3))
    assert r["rest_of_diff_hz"]["power"] == pytest.approx(rest_mhz * 1e6,
                                                          abs=2.0)


@pytest.mark.parametrize("span_mhz,p,verdict", [
    (15.0, 2.0, "the interface was the substrate term"),
    (21.5, 1.6, "the interface was the substrate term"),
    (15.0, 1.2, "partly"),                  # span small, order too low
    (22.5, 2.0, "partly"),                  # order high, span above half
    (30.0, 0.9, "partly"),
    (35.5, 0.8, "not the interface"),
    (43.0, 2.0, "not the interface"),       # a high order does not rescue it
    # A ladder that RISES as the cell shrinks, as the recorded one does: the
    # bars read the span's magnitude, so each branch is reached from below.
    (-15.0, 2.0, "the interface was the substrate term"),
    (-15.0, 1.2, "partly"),
    (-43.0, 2.0, "not the interface"),
])
def test_w10_reads_both_the_span_and_the_order(span_mhz, p, verdict):
    """Each branch of W10 on a ladder that is an exact power law.

    The bars are fractions of R2's own span, 43.666 MHz, so half is 21.83
    and 0.8 of it is 34.93: the spans above sit clear of both edges.  A
    negative span is a ladder that rises toward its limit.
    """
    arms = _ladder(_power_ladder(3.668, span_mhz, p))
    r = ins.w10_new_ladder(arms)
    assert r["span_r2_hz"] / 1e6 == pytest.approx(43.666, abs=2e-3)
    assert r["span_hz"] / 1e6 == pytest.approx(span_mhz, abs=1e-3)
    assert r["declared_order"] == pytest.approx(p, rel=1e-4)
    assert r["interface_span_bar_hz"] == pytest.approx(
        0.5 * r["span_r2_hz"], rel=1e-12)
    assert r["not_interface_span_bar_hz"] == pytest.approx(
        0.8 * r["span_r2_hz"], rel=1e-12)
    assert r["verdict"] == verdict
    # The limit of an exact power law is its f_inf, under the primary reading.
    assert r["new"]["power"]["declared"]["limit_hz"] / 1e9 == pytest.approx(
        3.668, abs=2e-6)


def test_w10_reads_r2s_ladder_as_r2s_f6_did():
    """R2's side of W10 is R2's F.6 power column, number for number."""
    arms = _ladder(_power_ladder(3.668, 15.0, 2.0))
    r2 = ins.w10_new_ladder(arms)["r2"]["power"]
    got = {q["key"]: q for q in r2["readings"]}
    assert got["declared"]["order"] == pytest.approx(0.7518, abs=2e-4)
    assert got["declared"]["limit_hz"] / 1e9 == pytest.approx(3.66901,
                                                              abs=2e-5)
    assert got["triple_0"]["limit_hz"] / 1e9 == pytest.approx(3.66450,
                                                              abs=2e-5)
    assert got["coarse_endpoint"]["limit_hz"] / 1e9 == pytest.approx(
        3.67183, abs=2e-5)


@pytest.mark.parametrize("shift_mhz,verdict", [(7.0, "HELD"), (-7.0, "HELD"),
                                               (7.7, "FIRED"),
                                               (-7.7, "FIRED")])
def test_w11_one_limit(shift_mhz, verdict):
    """|L_new - L_R2| against 0.20 % of the reference, 7.35 MHz."""
    l_r2 = ins.w11_one_limit(_ladder(_power_ladder(3.6, 20.0, 1.0)))[
        "limit_r2_hz"]
    arms = _ladder(_power_ladder(l_r2 / 1e9 + shift_mhz * 1e-3, 20.0, 1.0))
    r = ins.w11_one_limit(arms)
    assert r["difference_hz"] / 1e6 == pytest.approx(abs(shift_mhz), abs=1e-3)
    assert r["bar_hz"] == pytest.approx(0.002 * ins.reference_notch_hz("power"),
                                        rel=1e-12)
    assert r["verdict"] == verdict
    # Where the bar comes from: R2's five readings, largest minus smallest.
    assert r["r2_readings_spread_pct"] == pytest.approx(0.1996, abs=5e-4)


def test_w11_fires_when_a_limit_does_not_exist():
    """A new ladder with no monotone fall has no declared limit to compare."""
    new = {"Z6m": 3.70, "Z8m": 3.72, "Cm": 3.69, "Z16m": 3.71}
    r = ins.w11_one_limit(_ladder(new))
    assert math.isnan(r["limit_new_hz"]) and r["verdict"] == "FIRED"


def test_w12_is_w8s_derivation_per_reading():
    """FZ = (bar x limit / abs(A))^(1/p) and n_z = ceil(h / FZ), per row.

    First against the power law the ladder was BUILT from, whose limit,
    amplitude and order are known here without asking the instrument; then
    every row against its own three numbers.
    """
    f_inf, span, p = 3.668, 15.0, 2.0
    arms = _ladder(_power_ladder(f_inf, span, p))
    r = ins.w12_substrate_rule(arms)
    assert r["verdict"] is None
    h_um = [_R2_FZ_UM[ins.AFTER_1213_NAMESAKES[k]]
            for k in ins.AFTER_1213_LADDER]
    a_hz_per_m_p = (span * 1e6 / (h_um[0] ** p - h_um[-1] ** p)) * 1e6 ** p
    fz_known = (case.FREQ_BAR * f_inf * 1e9 / a_hz_per_m_p) ** (1.0 / p)
    declared = {q["key"]: q for q in r["readings"]["power"]}["declared"]
    assert declared["substrate_cell_for_the_bar_m"] == pytest.approx(
        fz_known, rel=1e-5)
    assert declared["n_substrate_cells_for_the_bar"] == int(np.ceil(
        case.SUBSTRATE_THICKNESS_M / fz_known - 1e-12))
    for t in ins.TRANSFORMS:
        for q in r["readings"][t]:
            if not (np.isfinite(q["order"]) and q["order"] > 0):
                continue
            fz = (case.FREQ_BAR * abs(q["limit_hz"]) / abs(q["amplitude"])) ** (
                1.0 / q["order"])
            assert q["substrate_cell_for_the_bar_m"] == pytest.approx(
                fz, rel=1e-9), (t, q["key"])
            assert q["n_substrate_cells_for_the_bar"] == int(np.ceil(
                case.SUBSTRATE_THICKNESS_M / fz - 1e-12)), (t, q["key"])
    fin = arms["Z16m"]
    ref = ins.reference_notch_hz("power")
    assert r["finest_from_reference_pct"]["power"] == pytest.approx(
        100.0 * (ins.notch_of(fin, "power")["f"] - ref) / ref, rel=1e-12)


def test_the_mesh_check_names_what_moved():
    """An arm and its namesake, identical, then one cell and one field off."""
    arms = _ladder(_power_ladder(3.668, 15.0, 2.0))
    for key in ins.AFTER_1213_LADDER:
        old = ins.AFTER_1213_NAMESAKES[key]
        arms[key]["profiles"] = json.loads(json.dumps(arms[old]["profiles"]))
    m = ins.mesh_against_namesake(arms, "Z8m")
    assert m["mesh_identical"] and m["profiles_bit_identical"]
    assert m["fields_differing"] == [] and m["realized_equal"]

    arms["Z8m"]["profiles"]["z"] = [arms["Z8m"]["profiles"]["z"][0]
                                    * (1.0 + 2.0 ** -52)]
    m = ins.mesh_against_namesake(arms, "Z8m")
    assert m["profiles_bit_identical"] is False and m["mesh_identical"] is False

    arms["Z8m"]["profiles"] = json.loads(json.dumps(arms["Z8"]["profiles"]))
    arms["Z8m"]["dt_s"] = 1e-15
    m = ins.mesh_against_namesake(arms, "Z8m")
    assert m["fields_differing"] == ["dt_s"] and m["mesh_identical"] is False
    # And it compares with the namesake the note names, not any rung.
    assert ins.mesh_against_namesake(arms, "Cm")["namesake"] == "C_off_re"


def test_the_provenance_names_the_tree_and_whether_the_export_was_it():
    """``rfx_tree_exported_matches`` is a comparison, not a copy of a flag."""
    a, b = "1" * 40, "2" * 40
    same = ins.provenance("7", commit="c" * 40, rfx_tree=a,
                          rfx_tree_exported=a)
    assert same["rfx_tree"] == a and same["rfx_tree_exported_matches"] is True
    diff = ins.provenance("7", commit="c" * 40, rfx_tree=a,
                          rfx_tree_exported=b)
    assert diff["rfx_tree_exported_matches"] is False
    with pytest.raises(SystemExit, match="tree hash"):
        ins.provenance("7", commit="c" * 40, rfx_tree="unknown")


def test_an_after_1213_arm_whose_export_is_not_its_tree_is_refused():
    """Refused before anything is built: the guard above would raise
    AssertionError on a build, so SystemExit here is the refusal itself."""
    with pytest.raises(SystemExit, match="refusing to solve Z6m"):
        ins.run_arm("Z6m", run_id="7", commit="c" * 40, rfx_tree="1" * 40,
                    rfx_tree_exported="2" * 40)


# ------------------------------------------------- the windows, on the record
#: What the committed record holds, transcribed from it and from the note's
#: Results.  Pinned means pinned: W11 FIRED and is pinned FIRED.
RECORDED_ARMS = ("Cm", "Z16m", "Z6m", "Z6r", "Z8m")
#: Each new arm's notch (GHz): the |S21|^2 vertex, then the log vertex.
RECORDED_NOTCH_GHZ = {
    "Z6m": (3.674809, 3.672677),
    "Z8m": (3.676144, 3.675747),
    "Cm": (3.677308, 3.678996),
    "Z16m": (3.677785, 3.679892),
    "Z6r": (3.752466, 3.749938),
}
RECORDED_RUN_IDS = {"Z6m": "369367263949", "Z6r": "369367263950",
                    "Z8m": "369367263956", "Cm": "369367263959",
                    "Z16m": "369367263960"}
#: The commit each job read, and ``git rev-parse <commit>:rfx`` of it.
_MAIN = ("17971050aa0d702e8636a1f456e53d00a6860cb8",
         "b91a81bf4fe2931fe94e7be255c12fc7ad45a710")
_REVERT = ("9c8fd9b605204b6272c262a4d49017c3c2306933",
           "2af37eafe9e1e1feefa3bb27ba80525bd81f53db")
RECORDED_COMMIT_AND_TREE = {"Z6m": _MAIN, "Z8m": _MAIN, "Cm": _MAIN,
                            "Z16m": _MAIN, "Z6r": _REVERT}
#: ``git rev-parse d558382f:rfx``, transcribed: the tree R2's Z6 ran.  Z6r ran
#: a different one.
R2_Z6_RFX_TREE = "544af8a8c41d18f14be119b424635c56dce515f0"
RECORDED_W9_VERDICT = "moved"
RECORDED_W9_ATTRIBUTION = "#1213 alone"
#: New minus R2 per rung (MHz): |S21|^2, then log.
RECORDED_W9_DELTA_MHZ = {
    "Z6m": (-77.6571, -77.2613),
    "Z8m": (-60.1437, -58.0334),
    "Cm": (-41.0999, -37.6389),
    "Z16m": (-31.0151, -30.4970),
}
RECORDED_W9_REST_OF_DIFF_MHZ = 0.0
RECORDED_W9_CHANGE_1213_MHZ = -77.6571
RECORDED_W10_VERDICT = "partly"
RECORDED_W10_SPAN_MHZ = -2.9760
RECORDED_W10_R2_SPAN_MHZ = 43.6659
RECORDED_W10_SPAN_RATIO = -0.0682
RECORDED_W10_HALF_BAR_MHZ = 21.8330
RECORDED_W10_MOST_BAR_MHZ = 34.9328
RECORDED_W10_ORDER = 1.4847
RECORDED_W10_ORDER_AT_COARSE_ENDPOINT = 1.5315
RECORDED_W10_LIMIT_GHZ = 3.678681
RECORDED_W10_STEPS_MHZ = [1.3354, 1.1634, 0.4772]
RECORDED_W10_MONOTONE = True
RECORDED_W11_VERDICT = "FIRED"
RECORDED_W11_LIMIT_NEW_GHZ = 3.678681
RECORDED_W11_LIMIT_R2_GHZ = 3.669008
RECORDED_W11_DIFFERENCE_MHZ = 9.6729
RECORDED_W11_DIFFERENCE_PCT = 0.2633
RECORDED_W11_BAR_MHZ = 7.3477
RECORDED_W11_R2_SPREAD_PCT = 0.1996
#: W12 per reading, in :func:`order_readings` order: declared, coarse
#: endpoint, the two triples, the three-parameter fit.
RECORDED_W12_KEYS = ["declared", "coarse_endpoint", "triple_0", "triple_1",
                     "three_parameter"]
RECORDED_W12_FZ_UM = {"power": [193.8880, 184.1512, 204.4266, 179.1327,
                                196.9212],
                      "log": [102.3486, 96.6780, 130.7974, 67.6662,
                              108.5079]}
RECORDED_W12_N_Z = {"power": [2, 2, 2, 2, 2], "log": [3, 3, 2, 4, 3]}
RECORDED_W12_FINEST_PCT = {"power": 0.1066, "log": 0.1507}


@pytest.fixture(scope="module")
def record_file():
    assert RECORD.is_file(), f"the record is missing: {RECORD}"
    with RECORD.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def arms():
    """The five new arms and R2's four rungs, as the instrument loads them."""
    return ins.load_after_1213_arms(RECORD, R2_RECORD)


@pytest.fixture(scope="module")
def v(arms):
    return ins.after_1213_verdicts(arms)


def test_the_record_holds_the_arms_and_windows_it_is_pinned_to_hold(
        record_file, arms):
    """Exactly the five arms of section 2, and the windows as section 4."""
    assert record_file["schema"] == "msl_notch_graded_fz_after_1213/1"
    assert record_file["note"] == ins.AFTER_1213_NOTE_PATH
    assert record_file["reused_from"] == "msl_notch_graded_fz.json"
    assert record_file["reused_arms"] == ["Z6", "Z8", "C_off_re", "Z16"]
    assert record_file["namesakes"] == ins.AFTER_1213_NAMESAKES
    assert record_file["commits"] == [dict(c) for c in ins.AFTER_1213_COMMITS]
    assert [c["commit"] for c in record_file["commits"]] == [
        _MAIN[0], _REVERT[0]]
    assert sorted(record_file["arms"]) == sorted(RECORDED_ARMS) == sorted(
        ins.AFTER_1213_ARMS)
    assert sorted(arms) == sorted(set(RECORDED_ARMS) | set(ins.R2_LADDER))
    w = record_file["windows"]
    assert w["primary_transform"] == "power"
    assert (w["W9_moved_bar_hz"], w["W9_attribution_bar_hz"]) == (1.0e6,
                                                                  0.05e6)
    assert w["W9_revert_arm"] == "Z6r"
    assert w["W10_ladder"] == ["Z6m", "Z8m", "Cm", "Z16m"]
    assert w["W10_r2_ladder"] == ["Z6", "Z8", "C_off_re", "Z16"]
    assert (w["W10_interface_span_fraction"], w["W10_interface_min_order"],
            w["W10_not_interface_span_fraction"]) == (0.5, 1.5, 0.8)
    assert w["W11_bar_pct"] == 0.20
    assert w["W12_freq_bar_pct"] == case.FREQ_BAR * 100.0
    for key, rec in sorted(record_file["arms"].items()):
        assert rec["arm"] == key
        assert rec["smoke"] is False, f"{key} was recorded from a capped run"
        assert rec["num_periods"] == case.NUM_PERIODS
        for field in ("freqs_hz", "s11_re", "s11_im", "s21_re", "s21_im"):
            assert len(rec[field]) == case.N_FREQS, (key, field)


def test_every_new_arm_names_its_run_its_commit_and_the_tree_it_ran(
        record_file):
    """Section 6's provenance: commit, ``rfx/`` tree, run, GPU, the export."""
    for key, rec in sorted(record_file["arms"].items()):
        p = rec["provenance"]
        commit, tree = RECORDED_COMMIT_AND_TREE[key]
        assert p["run_id"] == RECORDED_RUN_IDS[key], key
        assert p["git_sha"] == commit, key
        assert p["rfx_tree"] == p["rfx_tree_exported"] == tree, key
        assert p["rfx_tree_exported_matches"] is True, key
        assert p["jax_backend"] == "gpu" and p["gpu_device_kind"], key
        assert p["argv"][p["argv"].index("--arm") + 1] == key
        assert p["argv"][p["argv"].index("--commit") + 1] == commit
    assert len({r["provenance"]["run_id"]
                for r in record_file["arms"].values()}) == len(RECORDED_ARMS)


def test_the_witnesses_are_inside_their_bars_for_every_new_arm(record_file):
    for key, rec in sorted(record_file["arms"].items()):
        w = rec["witnesses"]
        assert w["worst_settling_db"] <= case.SETTLING_DB, key
        assert w["worst_sigma_excess"] <= case.PASSIVITY_EXCESS_BAR, key


@pytest.mark.parametrize("key", sorted(RECORDED_NOTCH_GHZ))
def test_each_new_notch_is_re_derived_from_the_curve(arms, key):
    """The shared estimator on the recorded |S21|, both ways, against the pin."""
    power, log = RECORDED_NOTCH_GHZ[key]
    assert ins.notch_of(arms[key], "power")["f"] / 1e9 == pytest.approx(
        power, abs=5e-7)
    got = ins.notch_of(arms[key], "log")
    assert got["f"] / 1e9 == pytest.approx(log, abs=5e-7)
    assert got["f"] == pytest.approx(arms[key]["notch"]["f"], abs=1.0)


@pytest.mark.parametrize("key", RECORDED_ARMS)
def test_each_new_arm_solved_its_namesakes_mesh(arms, v, key):
    """G.1 on the records: every cell of the three profiles, every declared
    field, and R3's realized block equal to the R2 namesake's.

    The profiles and the grid are compared here directly on the two records
    as well as through :func:`mesh_against_namesake`, so an emptied
    comparison is not the only witness.
    """
    old = arms[ins.AFTER_1213_NAMESAKES[key]]
    for axis in ("x", "y", "z"):
        assert arms[key]["profiles"][axis] == old["profiles"][axis], (
            key, axis)
    for field in ("grid_shape", "dt_s", "n_substrate_cells", "fine_cell_m",
                  "substrate_cell_m"):
        assert arms[key][field] == old[field], (key, field)
    m = v["mesh"][key]
    assert m["mesh_identical"] is True and m["profiles_bit_identical"] is True
    assert m["fields_differing"] == [] and m["realized_equal"] is True


def test_z6r_solved_r2s_z6_to_the_last_bit(arms):
    """The record fact under W9's 0.0000 MHz: Z6r's four S curves, its port
    impedance and its passivity curve are R2's Z6's bit for bit, though the
    two jobs ran different ``rfx/`` trees (the note's Conclusions say so)."""
    z6r, z6 = arms["Z6r"], arms["Z6"]
    for field in ("freqs_hz", "s11_re", "s11_im", "s21_re", "s21_im",
                  "s12_re", "s12_im", "s22_re", "s22_im",
                  "z0_re", "z0_im", "z0_median_ohm", "sigma_max_excess"):
        assert z6r[field] == z6[field], field
    assert z6r["provenance"]["rfx_tree"] == _REVERT[1] != R2_Z6_RFX_TREE
    assert arms["Z6m"]["s21_re"] != z6["s21_re"]


def test_w9_on_the_record_moved_and_1213_alone(v):
    r = v["W9"]
    assert r["verdict"] == RECORDED_W9_VERDICT
    assert r["transform"] == "power"
    assert r["delta_z6_hz"] / 1e6 == pytest.approx(
        RECORDED_W9_DELTA_MHZ["Z6m"][0], abs=5e-5)
    for row in r["rows"]:
        power, log = RECORDED_W9_DELTA_MHZ[row["arm"]]
        assert row["delta_hz"]["power"] / 1e6 == pytest.approx(power,
                                                               abs=5e-5)
        assert row["delta_hz"]["log"] / 1e6 == pytest.approx(log, abs=5e-5)
    assert r["revert_present"] is True
    assert r["attribution"] == RECORDED_W9_ATTRIBUTION
    assert r["rest_of_diff_hz"]["power"] == RECORDED_W9_REST_OF_DIFF_MHZ
    assert r["rest_of_diff_hz"]["log"] == RECORDED_W9_REST_OF_DIFF_MHZ
    assert r["change_1213_hz"]["power"] / 1e6 == pytest.approx(
        RECORDED_W9_CHANGE_1213_MHZ, abs=5e-5)


def test_w10_on_the_record_partly(v):
    """Partly: the span is under half of R2's and the order under 1.5."""
    r = v["W10"]
    nw = r["new"]["power"]
    assert r["verdict"] == RECORDED_W10_VERDICT
    assert r["span_hz"] / 1e6 == pytest.approx(RECORDED_W10_SPAN_MHZ,
                                               abs=5e-5)
    assert r["span_r2_hz"] / 1e6 == pytest.approx(RECORDED_W10_R2_SPAN_MHZ,
                                                  abs=5e-5)
    assert r["span_ratio"] == pytest.approx(RECORDED_W10_SPAN_RATIO,
                                            abs=5e-5)
    assert r["interface_span_bar_hz"] / 1e6 == pytest.approx(
        RECORDED_W10_HALF_BAR_MHZ, abs=5e-5)
    assert r["not_interface_span_bar_hz"] / 1e6 == pytest.approx(
        RECORDED_W10_MOST_BAR_MHZ, abs=5e-5)
    assert r["declared_order"] == pytest.approx(RECORDED_W10_ORDER, abs=5e-5)
    assert nw["declared"]["order"] == r["declared_order"]
    assert nw["declared"]["limit_hz"] / 1e9 == pytest.approx(
        RECORDED_W10_LIMIT_GHZ, abs=5e-7)
    assert {q["key"]: q for q in nw["readings"]}["coarse_endpoint"][
        "order"] == pytest.approx(RECORDED_W10_ORDER_AT_COARSE_ENDPOINT,
                                  abs=5e-5)
    assert [s / 1e6 for s in nw["steps_hz"]] == pytest.approx(
        RECORDED_W10_STEPS_MHZ, abs=5e-5)
    assert nw["monotone"] is RECORDED_W10_MONOTONE
    assert nw["arms"] == ["Z6m", "Z8m", "Cm", "Z16m"]


def test_w11_on_the_record_fired(v):
    r = v["W11"]
    assert r["verdict"] == RECORDED_W11_VERDICT
    assert r["limit_new_hz"] / 1e9 == pytest.approx(RECORDED_W11_LIMIT_NEW_GHZ,
                                                    abs=5e-7)
    assert r["limit_r2_hz"] / 1e9 == pytest.approx(RECORDED_W11_LIMIT_R2_GHZ,
                                                   abs=5e-7)
    assert r["difference_hz"] / 1e6 == pytest.approx(
        RECORDED_W11_DIFFERENCE_MHZ, abs=5e-5)
    assert r["difference_pct"] == pytest.approx(RECORDED_W11_DIFFERENCE_PCT,
                                                abs=5e-5)
    assert r["bar_hz"] / 1e6 == pytest.approx(RECORDED_W11_BAR_MHZ, abs=5e-5)
    assert r["r2_readings_spread_pct"] == pytest.approx(
        RECORDED_W11_R2_SPREAD_PCT, abs=5e-5)


def test_w12_on_the_record_is_reported_without_a_verdict(v):
    r = v["W12"]
    assert r["verdict"] is None
    for t in ins.TRANSFORMS:
        rows = r["readings"][t]
        assert [q["key"] for q in rows] == RECORDED_W12_KEYS
        # The recorded cells are 4-decimal roundings. The three-parameter row
        # takes its order from a search along an rms minimum so flat that the
        # ~1e-7 Hz rounding difference between OpenBLAS kernels in the
        # least-squares solve moves the order by ~3e-7: the recording machine's
        # SkylakeX kernel reads 196.92115262 um (196.9212), and CI's runners and
        # the Haswell, Sandybridge, Nehalem and Katmai kernels read
        # 196.92110674 um (196.9211). OPENBLAS_CORETYPE=Haswell reproduces CI
        # on a SkylakeX machine. A pin at half the rounding step cannot hold
        # across kernels; 5e-4 um (2.5e-6 relative) still pins every recorded
        # digit that means anything for a cell size.
        assert [q["substrate_cell_for_the_bar_m"] * 1e6 for q in rows] == (
            pytest.approx(RECORDED_W12_FZ_UM[t], abs=5e-4)), t
        assert [q["n_substrate_cells_for_the_bar"] for q in rows] == (
            RECORDED_W12_N_Z[t]), t
        assert r["finest_from_reference_pct"][t] == pytest.approx(
            RECORDED_W12_FINEST_PCT[t], abs=5e-5)
    assert r["finest_arm"] == "Z16m"


# ------------------------------- the second note's Results, byte for byte
R2_NOTE = _ROOT / ins.FZ_NOTE_PATH


def _r2_note_f0_rebuild(section: str) -> dict:
    """F.0's rebuild measurement as the second note printed it.

    F.0 is the one part of that section measured on the machine that prints
    it -- the float64 last bit of ramp cells rebuilt on the reader's numpy --
    so the byte comparison below reads it from the note rather than asking
    this machine's numpy, and compares everything else.  The rebuild itself
    is gated by ``test_msl_notch_graded_build.py``.
    """
    m = re.search(r"Measured by rebuilding all (\d+) of them against `([^`]+)`",
                  section)
    n_arms, record = int(m.group(1)), m.group(2)
    rows = re.findall(
        r"^\| (\w+) \| (\d+) / (\d+) / (\d+) \| (\d+) \| (\d+) \| (\w+) \|$",
        section[m.end():section.index("So at most")], flags=re.M)
    arms = {k: dict(ramp_cells=[int(a), int(b), int(c)],
                    n_cells_moved=int(moved), worst_ulp=float(ulp),
                    all_sums_equal=(ok == "True"))
            for k, a, b, c, moved, ulp, ok in rows}
    s = re.search(
        r"So at most (\d+) cells of one profile and (\d+) of one arm, (\d+) "
        r"across all \d+,\nout of profiles carrying (\d+) to (\d+) ramp cells "
        r"each; worst distance (\d+) ulp", section)
    return dict(arms=arms, record=record, n_arms=n_arms,
                worst_cells_moved_per_profile=int(s.group(1)),
                worst_cells_moved_per_arm=int(s.group(2)),
                total_cells_moved=int(s.group(3)),
                ramp_cells_min=int(s.group(4)), ramp_cells_max=int(s.group(5)),
                worst_ulp=float(s.group(6)))


def test_the_second_notes_results_are_still_what_its_tables_print(monkeypatch):
    """The third note may not change a byte of the second note's Results.

    The whole section, from its heading to the leader's Conclusions, against
    ``fz_markdown_tables`` on the second record today.  F.4 and F.4a ask git
    whether commits share an ``rfx/`` tree, so where git cannot see those
    commits the test skips and names them; F.0's ramp-cell measurement is read
    from the note (see ``_r2_note_f0_rebuild``).
    """
    with R2_RECORD.open() as fh:
        r2 = json.load(fh)
    arms = ins.load_fz_arms(R2_RECORD, BASE_RECORD)
    shas = sorted({a["provenance"]["git_sha"] for a in arms.values()})
    missing = ins.commits_absent(shas)
    if missing:
        pytest.skip("git cannot see " + ", ".join(s[:12] for s in missing)
                    + ", which F.4 asks about (a shallow clone, or history "
                    "squashed away on merge)")
    note = R2_NOTE.read_text()
    in_note = note[note.index("## Results (facts)"):
                   note.index("### Conclusions")]
    monkeypatch.setattr(ins, "rebuild_report",
                        lambda record_path=None: _r2_note_f0_rebuild(in_note))
    md = ins.fz_markdown_tables(arms, r2.get("build_only"),
                                ins.load_first_ladder(BASE_RECORD))
    printed = md[md.index("## Results (facts)"):
                 md.index("Conclusions: leader fills.")]
    assert printed == in_note


# ------------------------------------------- the two earlier records, unedited
#: The git blob hashes of the two earlier records and of R2's replay, as main
#: cca8ee5b holds them.  Section 6: "The R2 record and its replay are
#: untouched."  Computed from the bytes on disk, so no git is needed to check.
UNEDITED_BLOBS = {
    "validation/research/multiband_nu/results/msl_notch_graded_fz.json":
        "96692703d9d8fa535082d1ba9089578e0f7cf481",
    "validation/research/multiband_nu/results/msl_notch_graded.json":
        "16e6adf90962c8ee3b914b504440fbe06bba933d",
    "tests/unit/nonuniform/test_msl_notch_fz_replay.py":
        "e8733b79d9949efc8c56d27569eb5d011ec01a22",
}


def _blob(data: bytes) -> str:
    """``git hash-object`` of these bytes."""
    import hashlib

    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.mark.parametrize("path", sorted(UNEDITED_BLOBS))
def test_the_earlier_records_and_r2s_replay_are_byte_for_byte_mains(path):
    assert _blob((_ROOT / path).read_bytes()) == UNEDITED_BLOBS[path], path


# ---------------------------------------- this note's Results, byte for byte
NOTE = _ROOT / ins.AFTER_1213_NOTE_PATH
#: The note's blob as the pre-declaration commit 64ce5cd2 holds it, before
#: any arm ran.  Sections 0-7 are frozen from that commit.
PREDECLARATION_BLOB = "b7757db27eb06b434bd49a1124170fa2b41f84f7"


def test_sections_0_to_7_are_the_pre_declaration_as_committed():
    """Everything before the Results is 64ce5cd2's file, then one blank line."""
    data = NOTE.read_bytes()
    frozen = data[:data.index(b"## Results (facts)")]
    assert frozen.endswith(b"\n\n")
    assert _blob(frozen[:-1]) == PREDECLARATION_BLOB


def test_the_notes_results_are_what_tables_prints(capsys):
    """The note's Results, heading to Conclusions, against ``--tables``.

    ``--tables`` is run through the instrument's own command line on the two
    committed records.  The note's side ends where the leader's Conclusions
    begin: at ``Conclusions: leader fills.`` while it stands, or at a
    ``### Conclusions`` heading once the leader writes one in its place.
    """
    assert ins.main(["--tables", "--out", str(RECORD),
                     "--fz-out", str(R2_RECORD)]) == 0
    md = capsys.readouterr().out
    printed = md[md.index("## Results (facts)"):
                 md.index("Conclusions: leader fills.")]
    note = NOTE.read_text()
    start = note.index("## Results (facts)")
    end = min(i for i in (note.find("Conclusions: leader fills.", start),
                          note.find("### Conclusions", start)) if i >= 0)
    _same_text_and_numbers(note[start:end], printed)


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _same_text_and_numbers(note, printed):
    """The note's text equals ``--tables``' text, and every number agrees to one
    unit in its last printed place (integers exactly).

    ``--tables`` re-derives the numbers on the machine that runs it, and the
    fits behind them move in the last printed digit with the OpenBLAS kernel
    (see the W12 pin), so a byte comparison flips on that digit -- it did on
    CI's Python 3.10 shard at 9fe747a4. A number in e-notation gets one unit
    of its mantissa's last place (the W12 table's |A| prints 9.1633e+12, a
    quarter of a unit from its rounding edge on CI's value). Everything that
    is not a number, every integer (cell counts, keys) and every number
    printed without a decimal point is still compared exactly, and a number
    may not change its printed shape (decimals, e-notation).
    """
    assert _NUMBER.split(note) == _NUMBER.split(printed)
    a, b = _NUMBER.findall(note), _NUMBER.findall(printed)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if x == y:
            continue
        mx, _, ex = x.lower().partition("e")
        my, _, ey = y.lower().partition("e")
        assert "." in mx and "." in my, (x, y)
        places = len(mx.split(".")[1])
        assert places == len(my.split(".")[1]) and bool(ex) == bool(ey), (x, y)
        unit = 10.0 ** (max(int(ex or 0), int(ey or 0)) - places)
        assert abs(float(x) - float(y)) <= 1.000001 * unit, (x, y)

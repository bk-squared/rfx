"""The FZ ladder's windows, re-derived -- no FDTD.

``docs/design_notes/20260922_msl_notch_fz_ladder_predeclaration.md`` asks which
cell carries the notch filter's remaining first-order error: the cell through
the substrate at the strip's edge (FZ), the cell across the board (F), or the
0.35-cell edge offset.  Its four windows are arithmetic on eight measured
notch frequencies -- five arms solved for this note and three reused from the
first record -- and this file re-derives all of it with the instrument's own
functions and pins what the records say each window was.

Two kinds of test live here and they check different things.

* The estimator itself, on SYNTHETIC ladders whose order is known.  W5 fits a
  power ``f = f_inf + A FZ^p`` to four rungs, and a fit that cannot recover an
  order it was given is a fit whose answer means nothing.  These run without
  any record.
* The windows, on the committed records.  Pinned means pinned: a window that
  FIRED is pinned fired, because the pre-declaration allows one attempt per
  arm and the alternative to pinning it is losing the fact that it fired.

Nothing here builds a ``Simulation`` or steps a field -- the guard below
replaces the instrument's ``Simulation`` with one that raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.research.multiband_nu import msl_notch_graded as ins
from tests.crossval.msl_notch_filter import test_msl_notch_filter as case

_RESULTS = (Path(__file__).resolve().parents[3] / "validation" / "research"
            / "multiband_nu" / "results")
FZ_RECORD = _RESULTS / "msl_notch_graded_fz.json"
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


# ------------------------------------------------- the arms the note declares
def test_the_five_new_arms_are_the_ones_the_note_declares():
    """Section 2's table, as cells rather than as prose.

    Every F and FZ below is what the note prints in micrometres, recomputed
    from the trace width and the substrate thickness the case owns.  The arms
    are NOT in ``ARMS``: the first record's replay asserts that file holds
    exactly its own eight arms, and these five go in a separate record.
    """
    assert sorted(ins.FZ_ARMS) == ["C_off_re", "F16Z6", "ON13", "Z16", "Z6",
                                   "Z8"]
    assert set(ins.FZ_ARMS) & set(ins.ARMS) == set()
    assert ins.FZ_LADDER == ("Z6", "Z8", "C_off", "Z16")
    assert ins.FZ_LADDER_ONE_COMMIT == ("Z6", "Z8", "C_off_re", "Z16")
    assert ins.FZ_REMEASURED == ("C_off_re", "C_off")
    assert ins.FZ_REUSED_ARMS == ("A_on", "A_off", "C_off")

    h, w_metal = case.SUBSTRATE_THICKNESS_M, case.TRACE_WIDTH_M
    want = {
        # arm: (n_z, n, placement, F um, FZ um)
        "C_off_re": (12, 24, "offset", 24.2915, 21.1667),
        "Z6": (6, 24, "offset", 24.2915, 42.3333),
        "Z8": (8, 24, "offset", 24.2915, 31.7500),
        "Z16": (16, 24, "offset", 24.2915, 15.8750),
        "F16Z6": (6, 16, "offset", 35.9281, 42.3333),
        "ON13": (6, 13, "on-node", 46.1538, 42.3333),
    }
    for key, (n_z, n, placement, f_um, fz_um) in want.items():
        arm = ins.FZ_ARMS[key]
        assert arm.placement == placement, key
        assert arm.arm_length_m == case.ARM_LENGTH_M, key
        assert ins.rung_spec(arm.rung) == (n_z, n), key
        f = ins.fine_cell(arm.rung, arm.placement)
        assert f * 1e6 == pytest.approx(f_um, abs=5e-5), key
        assert h / n_z * 1e6 == pytest.approx(fz_um, abs=5e-5), key
        # The in-plane cell IS the metal divided by the cells across it, with
        # the offset arm's two half-cell overhangs counted.
        cells = n + (0.0 if placement == "on-node" else 2.0 * ins.EDGE_OFFSET)
        assert cells * f == pytest.approx(w_metal, abs=1e-15), key

    # What each arm holds fixed is the whole point of it.
    fz = {k: h / ins.rung_spec(ins.FZ_ARMS[k].rung)[0] for k in ins.FZ_ARMS}
    fine = {k: ins.fine_cell(ins.FZ_ARMS[k].rung, ins.FZ_ARMS[k].placement)
            for k in ins.FZ_ARMS}
    assert fine["Z6"] == fine["Z8"] == fine["Z16"] == fine["C_off_re"]
    assert fz["Z6"] == fz["F16Z6"] == fz["ON13"]            # F ladder: FZ fixed
    assert len({fz["Z6"], fz["Z8"], fz["Z16"]}) == 3        # and FZ moves


def test_the_re_measured_rung_is_the_first_records_rung_and_not_a_new_one():
    """Addendum A.2: C_off_re must declare C_off's mesh, not one like it.

    It shares the rung ENTRY, so there is no second place for the substrate
    cell or the in-plane cell to be written down and drift.  If the two meshes
    were not identical, the difference between their notches would be a
    difference of boards and would say nothing about the solver.
    """
    new, old = ins.FZ_ARMS["C_off_re"], ins.ARMS["C_off"]
    assert new == old, "C_off_re and C_off must be the same Arm"
    assert new.rung == "C" and ins.rung_spec(new.rung) == (12, 24)
    xn, yn, zn, bn = ins.profiles(new.rung, new.placement, new.arm_length_m)
    xo, yo, zo, bo = ins.profiles(old.rung, old.placement, old.arm_length_m)
    for axis, a, b in (("x", xn, xo), ("y", yn, yo), ("z", zn, zo)):
        assert a.tolist() == b.tolist(), axis
    for field in ("fine_cell_m", "substrate_cell_m", "z_tail_cell_m",
                  "edge_offset_m", "margin_cells", "n_across_metal",
                  "n_substrate_cells"):
        assert bn[field] == bo[field], field
    assert bn["declared"] == bo["declared"]
    assert bn["node"] == bo["node"]


def test_the_rung_table_still_holds_the_first_notes_three_rungs():
    """Generalizing the spec may not move a rung an existing arm was cut from.

    The bit-for-bit rebuild lives in the build test; this is the one line that
    says what those three pairs are, so a reader of either file can see it.
    """
    assert ins.RUNGS["A"] == (6, 12)
    assert ins.RUNGS["B"] == (8, 16)
    assert ins.RUNGS["C"] == (12, 24)
    assert ins.rung_spec("C") == ins.rung_spec((12, 24)) == (12, 24)
    assert [ins.RUNGS[ins.ARMS[k].rung] for k in ins.LADDER] == [
        (6, 12), (8, 16), (12, 24)]
    # The FZ ladder's third rung IS the first record's C_off, not a copy.
    assert ins.FZ_LADDER[2] == "C_off" and "C_off" in ins.ARMS
    assert ins.rung_spec(ins.ARMS["C_off"].rung) == (12, 24)
    assert ins.rung_spec(ins.FZ_ARMS["Z8"].rung)[1] == 24


def test_an_odd_cell_count_puts_no_node_on_the_metals_centre():
    """ON13's mesh fact, and the one arm it applies to.

    Thirteen cells across the 600 um metal cannot reach its centre line, so
    the board declares no centre node for that arm and the feed goes on the
    metal's own centre.  Every other arm lays an even count and is unchanged.
    """
    for key, arm in sorted(ins.ARM_TABLE.items()):
        n = ins.rung_spec(arm.rung)[1]
        b = ins.board(arm.rung, arm.placement, arm.arm_length_m)
        assert b["centre_is_a_node"] == (n % 2 == 0), key
        has_centre = "trace_y_centre_node" in b["declared"]
        assert has_centre == (n % 2 == 0), key
    assert ins.rung_spec(ins.FZ_ARMS["ON13"].rung)[1] % 2 == 1
    assert [k for k, a in sorted(ins.ARM_TABLE.items())
            if ins.rung_spec(a.rung)[1] % 2 == 1] == ["ON13"]


# -------------------------------------------------------------- the estimator
def _synthetic_curve(f0_hz: float, n: int = 400) -> dict:
    """An arm record carrying a |S21| whose notch is exactly ``f0_hz``.

    The estimator takes the deepest bin in the reference band and the vertex
    of the parabola through it and its two neighbours, fitted in log|S|.  A
    curve that IS a parabola in log|S| on an evenly spaced grid therefore
    returns its own centre, whichever three bins are used -- so these records
    carry a notch frequency that is known rather than measured.
    """
    lo, hi = case.REFERENCE_BAND_HZ
    f = np.linspace(lo, hi, n)
    y = -30.0 + 4e-17 * (f - f0_hz) ** 2
    s21 = np.power(10.0, y / 20.0)
    return dict(freqs_hz=f.tolist(), s21_re=s21.tolist(),
                s21_im=np.zeros_like(s21).tolist())


def test_the_synthetic_curve_carries_the_notch_it_claims():
    """The helper above, before anything is built on it."""
    for f0 in (3.60e9, 3.6744e9, 3.75e9):
        got = ins.notch_of(_synthetic_curve(f0))["f"]
        assert got == pytest.approx(f0, abs=1.0), f0


@pytest.mark.parametrize("p_true", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_the_order_fit_recovers_an_order_it_was_given(p_true):
    """W5's fit on a ladder built from a known power law.

    The four substrate cells are the ones the ladder really uses, so this is
    the fit as it will run.  Recovering ``p_true`` to six figures is what says
    the midpoint abscissa is unbiased on THIS ladder; the coarse-endpoint
    abscissa, reported beside it, is not, and the assertion below pins that
    difference so nobody quietly swaps them.
    """
    h = [case.SUBSTRATE_THICKNESS_M / n for n in (6, 8, 12, 16)]
    f_inf, amp = 3.6e9, -1.0e8
    f = [f_inf + amp * (v / h[0]) ** p_true for v in h]

    r = ins.fit_order_on_successive_differences(h, f)
    assert r["monotone"] and r["reason"] == ""
    assert r["order"] == pytest.approx(p_true, rel=1e-9)
    lim = ins.limit_at_order(h[2], h[3], f[2], f[3], r["order"])
    assert lim["limit"] == pytest.approx(f_inf, rel=1e-9)
    assert lim["amplitude"] == pytest.approx(amp / h[0] ** p_true, rel=1e-9)

    # The other reading of "log FZ", reported and not used.  It is biased by
    # about 5 % on this ladder, so the two are not interchangeable.
    assert r["order_at_coarse_endpoint"] != pytest.approx(p_true, rel=1e-3)
    assert r["order_at_coarse_endpoint"] > r["order"]


def test_the_fit_residuals_are_structural_and_not_a_goodness_measure():
    """Why the residuals are large even on an exact power law.

    ``ln|df|`` is ``p`` times the midpoint of the pair's logs plus a term in
    the pair's RATIO, and this ladder's ratios are 4/3, 3/2, 4/3.  The three
    points therefore do not lie on one line even when the data are an exact
    power.  They still give the exact slope, because the 3/2 pair sits at the
    mean abscissa and cannot lever it while the two 4/3 pairs carry equal
    terms that cancel.  A reader who takes these residuals for scatter would
    read a perfect ladder as a bad fit.
    """
    h = [case.SUBSTRATE_THICKNESS_M / n for n in (6, 8, 12, 16)]
    f = [3.6e9 - 1.0e8 * (v / h[0]) for v in h]
    r = ins.fit_order_on_successive_differences(h, f)
    assert r["order"] == pytest.approx(1.0, rel=1e-9)
    assert max(abs(v) for v in r["residuals"]) > 0.1
    ratios = [h[i] / h[i + 1] for i in range(3)]
    assert ratios[0] == pytest.approx(ratios[2])
    assert ratios[1] != pytest.approx(ratios[0])


def test_the_fit_refuses_a_ladder_it_cannot_read():
    """Two premises, each refused rather than returned as a number."""
    h = [case.SUBSTRATE_THICKNESS_M / n for n in (6, 8, 12, 16)]
    with pytest.raises(AssertionError, match="coarse to fine"):
        ins.fit_order_on_successive_differences(h[::-1], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(AssertionError, match="at least three rungs"):
        ins.fit_order_on_successive_differences(h[:2], [1.0, 2.0])
    # Not monotone: no order is fitted and the reason says so.
    r = ins.fit_order_on_successive_differences(h, [3.7e9, 3.6e9, 3.8e9, 3.5e9])
    assert not r["monotone"] and np.isnan(r["order"])
    assert "monotonically" in r["reason"]


def test_the_reference_notch_is_read_from_the_reference_file():
    """The 1 % bar's denominator, measured rather than typed.

    It has to be the same number the first note's W2 compared against, or the
    two notes would judge distance from two different references.
    """
    ref = ins.reference_notch_hz()
    with BASE_RECORD.open() as fh:
        base = json.load(fh)["arms"]
    assert ref / 1e9 == pytest.approx(
        ins.w2_comparison(base)["ref_notch_ghz"], abs=1e-9)
    # No numeric LITERAL in the instrument stands in for it.  Read from the
    # syntax tree, so the module's prose (which does quote the note's
    # 3.67436 GHz, in words) is not what is checked.
    import ast

    tree = ast.parse(open(ins.__file__).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            for scale in (1.0, 1e9):
                assert abs(node.value - ref / scale) > 1e-4 * abs(ref / scale), (
                    f"line {node.lineno}: {node.value} is the reference notch "
                    "re-typed as a literal; call reference_notch_hz()")


# ------------------------------------------------ the windows, on fake notches
def _fake_arms(notches_ghz: dict, fine_um: dict, fz_um: dict,
               shas: dict | None = None) -> dict:
    """Arm records carrying nothing but what W5-W8 read.

    ``shas`` gives an arm the commit it was solved at; the windows count the
    distinct ones so a reader can tell a ladder that spans two solver builds
    from one that does not.  Left out, every arm shares one commit.
    """
    shas = shas or {}
    return {k: dict(_synthetic_curve(v * 1e9),
                    arm=k, fine_cell_m=fine_um[k] * 1e-6,
                    substrate_cell_m=fz_um[k] * 1e-6,
                    n_substrate_cells=int(round(
                        case.SUBSTRATE_THICKNESS_M / (fz_um[k] * 1e-6))),
                    provenance=dict(git_sha=shas.get(k, "a" * 40),
                                    run_id="1"))
            for k, v in notches_ghz.items()}


def test_a_ladder_that_spans_two_solver_builds_says_so():
    """W5 counts the commits its rungs were solved at.

    The FZ ladder's third rung came from the first record and another build of
    ``rfx/``; addendum A.2 adds the same rung solved at this branch's commit.
    Which of the two a window read has to be visible in the window, not only
    in the provenance table.
    """
    f = {"Z6": 3.7250, "Z8": 3.7220, "C_off": 3.71663, "Z16": 3.7150}
    mixed = ins.w5_fz_ladder(_fake_arms(
        f, _LADDER_F, _LADDER_FZ, shas={"C_off": "b" * 40}))
    assert mixed["n_commits"] == 2 and mixed["one_solver_build"] is False
    assert mixed["commits"] == ["a" * 40, "b" * 40]
    same = ins.w5_fz_ladder(_fake_arms(f, _LADDER_F, _LADDER_FZ))
    assert same["n_commits"] == 1 and same["one_solver_build"] is True
    # The commit count is bookkeeping; it does not touch the verdict.
    assert mixed["order"] == pytest.approx(same["order"])
    assert mixed["verdict"] == same["verdict"]


def test_the_four_ways_of_reading_the_order_agree_on_an_exact_power_law():
    """The spread the record carries is a property of the data, not of the code.

    On an exact power law every reading must return the same order; on real
    rungs they need not, and that is what putting all four in the table is
    for.
    """
    h = [case.SUBSTRATE_THICKNESS_M / n for n in (6, 8, 12, 16)]
    for p_true in (0.8, 1.0, 1.9):
        f = [3.6e9 - 1.0e8 * (v / h[0]) ** p_true for v in h]
        est = ins.order_estimates(h, f)
        assert est["declared"] == pytest.approx(p_true, rel=1e-6)
        assert est["three_parameter_order"] == pytest.approx(p_true, rel=1e-4)
        assert est["three_parameter_rms_mhz"] < 1e-3
        assert len(est["adjacent_triples"]) == 2
        for t in est["adjacent_triples"]:
            assert t["order"] == pytest.approx(p_true, rel=1e-5)
            assert t["limit_ghz"] == pytest.approx(3.6, rel=1e-7)


def test_the_three_parameter_fit_is_least_squares_on_all_four_rungs():
    """It fits f_inf, A and p at once, and leaves a residual that means something."""
    h = [case.SUBSTRATE_THICKNESS_M / n for n in (6, 8, 12, 16)]
    # A ladder that is NOT a single power: two powers added.
    f = [3.6e9 - 1.0e8 * (v / h[0]) - 4.0e7 * (v / h[0]) ** 3 for v in h]
    r = ins.three_parameter_fit(h, f)
    assert 0.5 < r["order"] < 3.0
    assert r["rms_hz"] > 0.0 and r["n_points"] == 4
    # No other order fits these four better than the one it returned.
    import numpy as _np

    for p in _np.linspace(0.2, 4.0, 60):
        assert ins._fit_at_order(_np.asarray(h), _np.asarray(f),
                                 float(p))["rms_hz"] >= r["rms_hz"] * (1 - 1e-9)


_LADDER_FZ = {"Z6": 42.33333333333333, "Z8": 31.75,
              "C_off": 21.166666666666664, "Z16": 15.875}
_LADDER_F = dict.fromkeys(_LADDER_FZ, 24.29149797570851)


@pytest.mark.parametrize("order,verdict", [
    (0.45, "FIRED"), (0.55, "HELD"), (1.0, "HELD"), (2.45, "HELD"),
    (2.55, "FIRED"),
])
def test_w5_holds_inside_its_declared_order_window_and_fires_outside(order,
                                                                     verdict):
    """The window is [0.5, 2.5] and it is the order that decides.

    The orders tested sit clear of the two edges on purpose: a fitted order is
    a float and asking whether 2.5 exactly is inside would be asking about the
    last bit of a least-squares slope, not about the window.  The edges
    themselves are pinned as the constant below.
    """
    assert ins.P_Z_WINDOW == (0.5, 2.5)
    h = [_LADDER_FZ[k] * 1e-6 for k in ins.FZ_LADDER]
    f = {k: (3.6 + 0.1 * (v / h[0]) ** order) for k, v in zip(ins.FZ_LADDER, h)}
    r = ins.w5_fz_ladder(_fake_arms(f, _LADDER_F, _LADDER_FZ))
    assert r["order"] == pytest.approx(order, rel=1e-6)
    assert r["monotone"]
    assert r["verdict"] == verdict
    assert r["order_window"] == [0.5, 2.5]


def test_w5_fires_on_a_ladder_that_does_not_fall_monotonically():
    f = {"Z6": 3.74, "Z8": 3.70, "C_off": 3.72, "Z16": 3.68}
    r = ins.w5_fz_ladder(_fake_arms(f, _LADDER_F, _LADDER_FZ))
    assert not r["monotone"] and r["verdict"] == "FIRED"
    assert np.isnan(r["order"]) and np.isnan(r["limit_ghz"])


def test_w5_refuses_a_ladder_whose_in_plane_cell_also_moves():
    """The premise that makes the fit mean FZ and not "the mesh".

    This is the thing the previous note could not separate, so a ladder that
    cuts both cells has to be refused here and not fitted quietly.
    """
    f = {"Z6": 3.74, "Z8": 3.73, "C_off": 3.72, "Z16": 3.71}
    fine = dict(_LADDER_F, Z8=35.9281)
    with pytest.raises(AssertionError, match="one in-plane cell"):
        ins.w5_fz_ladder(_fake_arms(f, fine, _LADDER_FZ))


@pytest.mark.parametrize("f_z6,expect", [
    # A_off 3.74739, C_off 3.71663, so the total is 30.8 MHz in every row.
    # Z6 sits at one corner, and where it lands says which leg carried the
    # move: near C_off's notch the whole move happened on the F leg
    # (A_off -> Z6), near A_off's notch it happened on the FZ leg
    # (Z6 -> C_off).
    (3.71700, "F dominant"),
    (3.74700, "FZ dominant"),
    (3.73200, "shared"),
])
def test_w6_classifies_the_step_by_which_leg_carries_it(f_z6, expect):
    """The pre-declared classification, on each of its three outcomes."""
    notches = {"A_off": 3.74739, "Z6": f_z6, "C_off": 3.71663}
    fine = {"A_off": 47.2441, "Z6": 24.29150, "C_off": 24.29150}
    fz = {"A_off": 42.33333, "Z6": 42.33333, "C_off": 21.16667}
    r = ins.w6_attribution(_fake_arms(notches, fine, fz))
    assert r["classification"] == expect
    assert r["dominance_fraction"] == pytest.approx(2.0 / 3.0)
    # The two legs are the whole step, exactly, whatever the classification.
    assert r["delta_z_mhz"] + r["delta_f_mhz"] == pytest.approx(
        r["total_mhz"], abs=1e-6)
    assert abs(r["cross_term_hz"]) < 1e-3
    assert r["cross_term_is_algebraically_zero"] is True


def test_w6_refuses_legs_that_are_not_one_cell_each():
    """Each leg has to hold one cell fixed or it is not an attribution."""
    notches = {"A_off": 3.74739, "Z6": 3.73, "C_off": 3.71663}
    fine = {"A_off": 47.2441, "Z6": 24.29150, "C_off": 24.29150}
    fz = {"A_off": 42.33333, "Z6": 42.33333, "C_off": 21.16667}
    with pytest.raises(AssertionError, match="in-plane step"):
        ins.w6_attribution(_fake_arms(notches, fine, dict(fz, Z6=31.75)))
    with pytest.raises(AssertionError, match="substrate step"):
        ins.w6_attribution(_fake_arms(notches, dict(fine, C_off=35.93), fz))


@pytest.mark.parametrize("f_on13,lower", [(3.74000, True), (3.75000, False)])
def test_w7_reads_the_offsets_sign_off_the_two_declared_branches(f_on13,
                                                                 lower):
    """Both branches of W7, and the F slope it reports beside them."""
    notches = {"A_on": 3.73232, "A_off": 3.74739, "ON13": f_on13}
    fine = {"A_on": 50.0, "A_off": 47.2441, "ON13": 46.15385}
    fz = dict.fromkeys(notches, 42.33333)
    r = ins.w7_offset_sign(_fake_arms(notches, fine, fz))
    assert r["on_node_reads_lower"] is lower
    assert ("raises the notch on its own" in r["reading"]) is lower
    assert r["fine_cell_gap_pct"] == pytest.approx(
        100.0 * (47.2441 - 46.15385) / 46.15385, abs=1e-3)
    # The slope is the two ON-NODE arms only, so it carries no offset at all.
    slope = ((3.73232 - f_on13) * 1e9) / ((50.0 - 46.15385) * 1e-6)
    assert r["f_slope_mhz_per_um"] == pytest.approx(slope / 1e12, rel=1e-5)
    assert r["f_part_of_delta_mhz"] == pytest.approx(
        slope * (47.2441 - 46.15385) * 1e-6 / 1e6, rel=1e-5)


def test_w7_refuses_arms_that_do_not_share_the_substrate_cell():
    notches = {"A_on": 3.73232, "A_off": 3.74739, "ON13": 3.74}
    fine = {"A_on": 50.0, "A_off": 47.2441, "ON13": 46.15385}
    fz = dict.fromkeys(notches, 42.33333)
    with pytest.raises(AssertionError, match="one substrate cell"):
        ins.w7_offset_sign(_fake_arms(notches, fine, dict(fz, ON13=31.75)))


@pytest.mark.parametrize("limit_ghz,verdict", [(3.67436, "HELD"),
                                               (3.72000, "FIRED")])
def test_w8_asks_whether_the_fz_limit_reaches_the_bar(limit_ghz, verdict):
    """W8 on a ladder built to land on a chosen limit.

    Its bar is the case's own 1 % frequency bar, and its reference is the
    openEMS stage read from the reference file, so the only free thing here is
    where the ladder extrapolates to.
    """
    h = [_LADDER_FZ[k] * 1e-6 for k in ins.FZ_LADDER]
    f = {k: limit_ghz + 0.06 * (v / h[0]) for k, v in zip(ins.FZ_LADDER, h)}
    arms = _fake_arms(f, _LADDER_F, _LADDER_FZ)
    r = ins.w8_is_z_enough(arms)
    assert r["w5_verdict"] == "HELD"
    assert r["order"] == pytest.approx(1.0, rel=1e-6)
    assert r["limit_ghz"] == pytest.approx(limit_ghz, rel=1e-7)
    assert r["bar_pct"] == case.FREQ_BAR * 100.0
    assert r["verdict"] == verdict
    # The substrate rule it derives, re-derived here from its own numbers.
    # The bar is a fraction of the ladder's OWN limit, not of the reference.
    want = (case.FREQ_BAR * abs(r["limit_ghz"]) * 1e9
            / abs(r["amplitude"])) ** (1.0 / r["order"])
    assert r["substrate_cell_for_the_bar_m"] == pytest.approx(want, rel=1e-9)
    assert r["n_substrate_cells_for_the_bar"] == int(np.ceil(
        case.SUBSTRATE_THICKNESS_M / want - 1e-12))


def test_w8_fires_whenever_w5_does_however_close_the_limit_lands():
    """"If W5 holds and the limit is inside the bar" -- both, not either."""
    f = {"Z6": 3.74, "Z8": 3.70, "C_off": 3.72, "Z16": 3.68}
    r = ins.w8_is_z_enough(_fake_arms(f, _LADDER_F, _LADDER_FZ))
    assert r["w5_verdict"] == "FIRED" and r["verdict"] == "FIRED"


def test_fz_verdicts_reports_only_the_windows_whose_arms_are_present():
    """A record part-way through the campaign carries part of the arithmetic."""
    f = {"Z6": 3.7250, "Z8": 3.7220, "C_off": 3.71663, "Z16": 3.7150}
    arms = _fake_arms(f, _LADDER_F, _LADDER_FZ)
    for k in arms:
        arms[k].update(wall_s=1.0, grid_shape=[2, 2, 2])
    out = ins.fz_verdicts(arms)
    assert set(out) == {"W5", "W8", "cost"}
    assert "W6" not in out and "W7" not in out
    assert sorted(out["cost"]) == sorted(arms)


# ------------------------------------------------- the windows, on the records
#: What the two records hold, transcribed from the committed files.  Pinned
#: means pinned: a window that FIRED is pinned fired.
#: The arms the committed FZ record holds.
RECORDED_FZ_ARMS = ("C_off_re", "F16Z6", "ON13", "Z16", "Z6", "Z8")
#: Declared by the module but not yet solved.  A pre-declaration names its
#: arms before they run, so this is the honest state between the declaration
#: and the record, and it is asserted rather than left implicit: an arm that
#: quietly stopped being owed would be an arm nobody noticed was missing.
PENDING_ARMS = ()
RECORDED_NOTCH_GHZ = {
    "A_off": 3.747389,
    "A_on": 3.732318,
    "C_off": 3.716635,
    "F16Z6": 3.748281,
    "ON13": 3.733383,
    "Z16": 3.710389,
    "Z6": 3.749938,
    "Z8": 3.733780,
}
RECORDED_WALL_S = {
    "A_off": 106.1,
    "A_on": 101.4,
    "C_off": 1007.7,
    "F16Z6": 160.8,
    "ON13": 91.7,
    "Z16": 1081.9,
    "Z6": 357.0,
    "Z8": 715.9,
}
RECORDED_CELLS = {
    "A_off": 1545600,
    "A_on": 1502976,
    "C_off": 3291484,
    "F16Z6": 1758720,
    "ON13": 1559712,
    "Z16": 3935470,
    "Z6": 2289728,
    "Z8": 2647498,
}
RECORDED_VERDICTS = {"W5": "HELD", "W8": "HELD"}
RECORDED_W6_CLASS = "FZ dominant"
RECORDED_W5_ORDER = 1.3713
RECORDED_W5_ORDER_AT_COARSE_ENDPOINT = 1.4448
RECORDED_W5_LIMIT_GHZ = 3.69748
RECORDED_W5_MONOTONE = True
RECORDED_W5_FINEST_PCT = 0.9807
RECORDED_W5_LIMIT_PCT = 0.6292
RECORDED_W5_STEPS_MHZ = [16.158, 17.1453, 6.2457]
RECORDED_W6_DELTA_Z_MHZ = 33.3032
RECORDED_W6_DELTA_F_MHZ = -2.5492
RECORDED_W6_TOTAL_MHZ = 30.7541
RECORDED_W6_BAR_MHZ = 20.5027
RECORDED_W7_ON_NODE_READS_LOWER = True
RECORDED_W7_DELTA_MHZ = 14.0061
RECORDED_W7_DELTA_PCT = 0.3752
RECORDED_W7_FINE_CELL_GAP_PCT = 2.3622
RECORDED_W7_F_SLOPE_MHZ_PER_UM = -0.2767
RECORDED_W7_F_PART_MHZ = -0.3016
RECORDED_W8_N_Z_FOR_THE_BAR = 8
RECORDED_W8_FZ_FOR_THE_BAR_UM = 34.1871
RECORDED_W8_AMPLITUDE = 4.924565e+13
RECORDED_F_LADDER_GHZ = [3.747389, 3.748281, 3.749938]
RECORDED_F_LADDER_MONOTONE = True
RECORDED_REFERENCE_GHZ = 3.67436


@pytest.fixture(scope="module")
def arms():
    """The eight arms W5-W8 read, from the two records the instrument names."""
    assert FZ_RECORD.is_file(), f"the record is missing: {FZ_RECORD}"
    return ins.load_fz_arms(FZ_RECORD, BASE_RECORD)


@pytest.fixture(scope="module")
def fz_file():
    with FZ_RECORD.open() as fh:
        return json.load(fh)


def test_the_fz_record_holds_the_arms_it_is_pinned_to_hold(fz_file, arms):
    """Exactly the arms below, and every one of them declared by the module.

    ``RECORDED_FZ_ARMS`` is what the committed file holds; ``ins.FZ_ARMS`` is
    what the module declares.  They are asserted separately because an arm
    can be declared before it is solved -- that is the whole shape of a
    pre-declaration -- and the gap between the two is what says an arm is
    still owed.
    """
    assert fz_file["schema"] == "msl_notch_graded_fz/1"
    assert fz_file["note"].endswith(
        "20260922_msl_notch_fz_ladder_predeclaration.md")
    assert sorted(fz_file["arms"]) == sorted(RECORDED_FZ_ARMS)
    assert set(RECORDED_FZ_ARMS) <= set(ins.FZ_ARMS)
    assert set(RECORDED_FZ_ARMS) | set(PENDING_ARMS) == set(ins.FZ_ARMS)
    assert set(RECORDED_FZ_ARMS) & set(PENDING_ARMS) == set()
    assert sorted(set(ins.FZ_ARMS) - set(fz_file["arms"])) == sorted(
        PENDING_ARMS), "the arms still owed are not the ones declared pending"
    assert fz_file["reused_from"] == "msl_notch_graded.json"
    assert fz_file["reused_arms"] == list(ins.FZ_REUSED_ARMS)
    assert sorted(arms) == sorted(set(RECORDED_FZ_ARMS)
                                  | set(ins.FZ_REUSED_ARMS))
    for key, rec in sorted(fz_file["arms"].items()):
        assert rec["arm"] == key
        assert rec["smoke"] is False, f"{key} was recorded from a capped run"
        assert rec["num_periods"] == case.NUM_PERIODS
        for field in ("freqs_hz", "s11_re", "s11_im", "s21_re", "s21_im"):
            assert len(rec[field]) == case.N_FREQS, (key, field)
    # The windows' own constants, as the record froze them.
    w = fz_file["windows"]
    assert w["W5_order_window"] == list(ins.P_Z_WINDOW)
    assert w["W5_ladder"] == list(ins.FZ_LADDER)
    assert w["W6_dominance_fraction"] == pytest.approx(2.0 / 3.0)
    assert w["W8_freq_bar_pct"] == case.FREQ_BAR * 100.0


def test_the_first_record_was_not_edited():
    """The three reused arms come from a file this note does not write.

    The pre-declaration reuses A_on, A_off and C_off rather than re-running
    them, and the whole point of a separate record is that the first one stays
    as it was measured.
    """
    with BASE_RECORD.open() as fh:
        base = json.load(fh)
    assert sorted(base["arms"]) == sorted(ins.ALL_ARMS)
    assert set(base["arms"]) & set(ins.FZ_ARMS) == set()
    assert base["schema"] == "msl_notch_graded/1"
    assert base["note"].endswith(
        "20260922_msl_notch_graded_mesh_predeclaration.md")


def test_every_new_arm_names_the_run_and_the_commit_that_made_it(fz_file):
    """Provenance, read rather than assumed, and one commit for the five.

    The pre-declaration pins one commit for the whole ladder: two arms at two
    commits would be two instruments, and the FZ ladder's whole claim is that
    only the substrate cell moved between its rungs.
    """
    runs, shas = set(), set()
    for key, rec in sorted(fz_file["arms"].items()):
        p = rec["provenance"]
        assert ins._SHA_RE.match(p["git_sha"]), (key, p["git_sha"])
        assert p["run_id"] and str(p["run_id"]).isdigit(), (key, p["run_id"])
        assert p["jax_backend"] == "gpu", (key, p["jax_backend"])
        assert p["gpu_device_kind"], key
        assert p["run_id"] not in runs, f"{key} reuses run {p['run_id']}"
        runs.add(p["run_id"])
        shas.add(p["git_sha"])
    # One SIMULATOR, not one commit label.  The five arms of section 2 ran at
    # one commit; C_off_re was declared after the review and ran at a later
    # one that changes a note, this instrument and its tests and leaves rfx/
    # alone.  What would make two arms incomparable is a different solver, so
    # that is what is asserted.
    assert ins.solver_tree_equal(sorted(shas)) is True, (
        f"the arms span {len(shas)} commits whose rfx/ trees differ")
    assert len(shas) <= 2, f"the arms span {len(shas)} commits"


def test_the_witnesses_and_the_refusals_are_recorded_for_every_new_arm(fz_file):
    """R1, R2, R3 before the solve and R4 after it, as each arm recorded them."""
    for key, rec in sorted(fz_file["arms"].items()):
        w = rec["witnesses"]
        assert w["worst_settling_db"] <= case.SETTLING_DB, key
        assert w["worst_sigma_excess"] <= case.PASSIVITY_EXCESS_BAR, key
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
        assert g["edge_offset_residual_m"] <= 1e-16, key
        for port in rec["probe_runway"]["ports"]:
            assert port["n_cells_off_coarse"] == 0, (key, port["name"])
        lo, hi = rec["band_y_trace_m"]
        for port in rec["port_footprint"]["ports"]:
            assert lo <= port["y_lo_m"] and port["y_hi_m"] <= hi, key


def test_the_odd_rung_recorded_its_straddled_centre(fz_file):
    """ON13's mesh fact as the arm that solved it wrote it down."""
    rec = fz_file["arms"]["ON13"]
    assert rec["centre_is_a_node"] is False
    assert rec["n_across_metal"] % 2 == 1
    f = rec["fine_cell_m"]
    for label, st in sorted(rec["centre_straddle"].items()):
        lo, hi = st["nodes_m"]
        assert hi - lo == pytest.approx(f, abs=ins.NODE_TOL_M), label
        assert 0.5 * (lo + hi) == pytest.approx(st["declared_centre_m"],
                                                abs=ins.NODE_TOL_M), label
    assert rec["port_centre_y_m"] == pytest.approx(
        rec["centre_straddle"]["trace_y_centre"]["declared_centre_m"])
    for key, other in sorted(fz_file["arms"].items()):
        if key != "ON13":
            assert other["centre_is_a_node"] is True, key
            assert other["centre_straddle"] == {}, key
            assert other["n_across_metal"] % 2 == 0, key


@pytest.mark.parametrize("key", sorted(RECORDED_NOTCH_GHZ))
def test_each_notch_is_re_derived_from_the_curve(arms, key):
    """The shared estimator on the recorded |S21|, against the pinned value."""
    got = ins.notch_of(arms[key])
    assert got["f"] == pytest.approx(arms[key]["notch"]["f"], abs=1.0)
    assert got["f"] / 1e9 == pytest.approx(RECORDED_NOTCH_GHZ[key], abs=5e-6)
    assert got["depth_db"] == pytest.approx(arms[key]["notch"]["depth_db"],
                                            abs=1e-9)


def test_the_fz_ladder_moves_only_the_substrate_cell(arms):
    """W5's premise, on the records rather than on the declaration.

    Four rungs, one in-plane cell, four substrate cells that really do refine.
    If this were false the fitted order would be a property of the mesh and
    not of the substrate.
    """
    fine = [arms[k]["fine_cell_m"] for k in ins.FZ_LADDER]
    fz = [arms[k]["substrate_cell_m"] for k in ins.FZ_LADDER]
    assert max(fine) - min(fine) <= ins.NODE_TOL_M
    assert fz == sorted(fz, reverse=True)
    assert [arms[k]["n_substrate_cells"] for k in ins.FZ_LADDER] == [6, 8, 12,
                                                                     16]
    assert [arms[k]["n_across_metal"] for k in ins.FZ_LADDER] == [24] * 4
    assert [arms[k]["placement"] for k in ins.FZ_LADDER] == ["offset"] * 4
    assert [arms[k]["arm_length_m"] for k in ins.FZ_LADDER] == [
        case.ARM_LENGTH_M] * 4


def test_w5_the_fz_ladder_is_a_ladder(arms):
    r = ins.w5_fz_ladder(arms)
    assert r["monotone"] is RECORDED_W5_MONOTONE
    assert r["order"] == pytest.approx(RECORDED_W5_ORDER, abs=1e-4)
    assert r["order_at_coarse_endpoint"] == pytest.approx(
        RECORDED_W5_ORDER_AT_COARSE_ENDPOINT, abs=1e-4)
    assert r["limit_ghz"] == pytest.approx(RECORDED_W5_LIMIT_GHZ, abs=1e-5)
    assert r["finest_distance_pct"] == pytest.approx(RECORDED_W5_FINEST_PCT,
                                                     abs=1e-4)
    assert r["limit_distance_pct"] == pytest.approx(RECORDED_W5_LIMIT_PCT,
                                                    abs=1e-4)
    assert r["differences_mhz"] == pytest.approx(RECORDED_W5_STEPS_MHZ,
                                                 abs=1e-3)
    assert r["order_window"] == [0.5, 2.5]
    assert r["reference_ghz"] == pytest.approx(RECORDED_REFERENCE_GHZ, abs=1e-5)
    assert r["verdict"] == RECORDED_VERDICTS["W5"]


def test_w6_attribution_at_the_a_off_to_c_off_step(arms):
    r = ins.w6_attribution(arms)
    assert r["delta_z_mhz"] == pytest.approx(RECORDED_W6_DELTA_Z_MHZ, abs=1e-3)
    assert r["delta_f_mhz"] == pytest.approx(RECORDED_W6_DELTA_F_MHZ, abs=1e-3)
    assert r["total_mhz"] == pytest.approx(RECORDED_W6_TOTAL_MHZ, abs=1e-3)
    assert r["bar_mhz"] == pytest.approx(RECORDED_W6_BAR_MHZ, abs=1e-3)
    assert r["classification"] == RECORDED_W6_CLASS
    # The two legs are the whole step and the cross term cancels exactly.
    assert r["delta_z_mhz"] + r["delta_f_mhz"] == pytest.approx(
        r["total_mhz"], abs=1e-9)
    assert abs(r["cross_term_hz"]) < 1e-3


def test_w7_the_offsets_own_sign(arms):
    r = ins.w7_offset_sign(arms)
    assert r["on_node_reads_lower"] is RECORDED_W7_ON_NODE_READS_LOWER
    assert r["delta_mhz"] == pytest.approx(RECORDED_W7_DELTA_MHZ, abs=1e-3)
    assert r["delta_pct"] == pytest.approx(RECORDED_W7_DELTA_PCT, abs=1e-4)
    assert r["fine_cell_gap_pct"] == pytest.approx(
        RECORDED_W7_FINE_CELL_GAP_PCT, abs=1e-4)
    assert r["f_slope_mhz_per_um"] == pytest.approx(
        RECORDED_W7_F_SLOPE_MHZ_PER_UM, abs=1e-4)
    assert r["f_part_of_delta_mhz"] == pytest.approx(RECORDED_W7_F_PART_MHZ,
                                                     abs=1e-3)
    # Its three arms share the substrate cell, and only ON13 is on-node at a
    # cell near A_off's.
    assert r["substrate_cell_m"] == pytest.approx(
        case.SUBSTRATE_THICKNESS_M / 6)
    assert r["fine_cell_on13_m"] < r["fine_cell_a_off_m"] < r[
        "fine_cell_a_on_m"]


def test_w8_is_z_enough(arms):
    r = ins.w8_is_z_enough(arms)
    assert r["w5_verdict"] == RECORDED_VERDICTS["W5"]
    assert r["limit_ghz"] == pytest.approx(RECORDED_W5_LIMIT_GHZ, abs=1e-5)
    assert r["limit_distance_pct"] == pytest.approx(RECORDED_W5_LIMIT_PCT,
                                                    abs=1e-4)
    assert r["bar_pct"] == case.FREQ_BAR * 100.0
    assert r["verdict"] == RECORDED_VERDICTS["W8"]
    assert r["amplitude"] == pytest.approx(RECORDED_W8_AMPLITUDE, rel=1e-4)
    assert r["substrate_cell_for_the_bar_m"] * 1e6 == pytest.approx(
        RECORDED_W8_FZ_FOR_THE_BAR_UM, abs=1e-3)
    assert r["n_substrate_cells_for_the_bar"] == RECORDED_W8_N_Z_FOR_THE_BAR
    # The rule it derives, re-derived here from its own two numbers.
    want = (case.FREQ_BAR * abs(r["limit_ghz"]) * 1e9
            / abs(r["amplitude"])) ** (1.0 / r["order"])
    assert r["substrate_cell_for_the_bar_m"] == pytest.approx(want, rel=1e-9)


def test_the_f_ladder_reported_beside_w6(arms):
    """Three in-plane cells at one substrate cell, W6's dF end to end."""
    r = ins.fz_verdicts(arms)["f_ladder_at_fz_42um"]
    assert r["arms"] == ["A_off", "F16Z6", "Z6"]
    assert r["notches_ghz"] == pytest.approx(RECORDED_F_LADDER_GHZ, abs=5e-6)
    assert r["monotone"] is RECORDED_F_LADDER_MONOTONE
    assert r["fine_cells_m"] == sorted(r["fine_cells_m"], reverse=True)
    assert r["substrate_cell_m"] == pytest.approx(
        case.SUBSTRATE_THICKNESS_M / 6)


def test_the_cost_of_every_arm_is_cells_and_not_nodes(arms):
    """Both columns of F.1 and F.3, counted the way the first note counts."""
    cost = ins.fz_verdicts(arms)["cost"]
    for key, cells in sorted(RECORDED_CELLS.items()):
        assert cost[key]["n_grid_cells"] == cells, key
        assert cost[key]["n_grid_cells"] == ins.grid_cells(arms[key]), key
        assert cost[key]["wall_s"] == pytest.approx(RECORDED_WALL_S[key],
                                                    abs=0.05), key
        shape = arms[key]["grid_shape"]
        assert shape[0] * shape[1] * shape[2] > cells, key


def test_the_tables_the_note_carries_are_generated_from_the_records(arms):
    """`--tables` emits the Results section, and writes no conclusion.

    The note's Results section is this function's output; nothing in it is
    typed by hand, and the last line is the one that says whose the
    interpreting sentences are.
    """
    md = ins.fz_markdown_tables(arms)
    assert md.rstrip().endswith("Conclusions: leader fills.")
    for heading in ("## Results (facts)", "### F.0", "### F.1", "### F.2",
                    "### F.3", "### F.4", "### F.5"):
        assert heading in md, heading
    for key in sorted(arms):
        assert f"| {key} |" in md, key
    assert f"{RECORDED_W5_LIMIT_GHZ:.5f}" in md
    assert f"{RECORDED_REFERENCE_GHZ:.5f}" in md
    assert RECORDED_W6_CLASS in md
    assert f"{RECORDED_W8_N_Z_FOR_THE_BAR}" in md
    # Every window the verdicts carry appears in the section.  A local name
    # that shadowed `v` once emptied every window block at a stroke while the
    # section still had its headings, its F.0, F.1, F.2, F.3 and F.5 tables
    # and its closing line -- so "the tables were generated" is not enough to
    # check, the windows have to be in them.
    v = ins.fz_verdicts(arms)
    for key in sorted(v):
        if key.startswith("W"):
            assert f"**{key.split('_')[0]} --" in md, key
    assert "### F.4 The frozen windows\n\n### F.5" not in md

    # Every row of every table has as many cells as that table's header.  A
    # ragged row is a column silently dropped from the note.
    tables, block = [], []
    for ln in md.splitlines():
        if ln.startswith("|"):
            block.append(ln)
        elif block:
            tables.append(block)
            block = []
    if block:
        tables.append(block)
    assert len(tables) >= 8, f"only {len(tables)} tables in the section"
    for t in tables:
        assert len(t) >= 3, f"a table with no rows: {t[0]}"
        widths = {row.count("|") for row in t}
        assert len(widths) == 1, f"ragged table starting {t[0]!r}: {widths}"


# -------------------------------------- the review's P3 items, on their own
def test_the_rebuild_report_counts_what_f0_prints():
    """F.0's ramp-cell numbers come from the record, not from a sentence.

    The first version of that paragraph said "at most 6 of a profile's ramp
    cells", which confused the 6 ulp worst distance with a cell count.  The
    count is now measured on every read, so it cannot drift from the record
    again.
    """
    r = ins.rebuild_report()
    assert r["n_arms"] == len(ins.ARMS)
    assert r["record"] == ins.DEFAULT_OUT.name
    assert r["all_sums_equal"] is True
    assert r["worst_ulp"] <= ins.RECORD_ULP_FLOOR
    # Per arm and per profile, and they are different numbers.
    assert r["worst_cells_moved_per_profile"] <= r["worst_cells_moved_per_arm"]
    assert r["total_cells_moved"] == sum(
        v["n_cells_moved"] for v in r["arms"].values())
    for key, v in sorted(r["arms"].items()):
        assert len(v["ramp_cells"]) == 3
        assert v["n_cells_moved"] <= sum(v["ramp_cells"]), key
        assert v["worst_per_profile"] <= v["n_cells_moved"], key
    # Every arm of the first record is covered, and none of the FZ arms is.
    assert set(r["arms"]) == set(ins.ARMS)
    assert set(r["arms"]) & set(ins.FZ_ARMS) == set()


def test_the_substrate_rule_bar_is_a_fraction_of_the_ladders_own_limit(arms):
    """W8's derived rule bounds the term that is left, not the reference gap.

    The quantity is how far the notch still has to fall on THIS ladder, and
    the ladder converges to its own limit.  Reading the bar off the external
    reference instead bounds a different thing; where the two differ, so does
    the cell it asks for.
    """
    r = ins.w8_is_z_enough(arms)
    assert r["substrate_rule_bar_is"].endswith("own fitted limit")
    assert r["substrate_rule_bar_hz"] == pytest.approx(
        case.FREQ_BAR * r["limit_ghz"] * 1e9, rel=1e-12)
    # It is NOT the reference's one per cent, and the two differ.
    off_reference = case.FREQ_BAR * ins.reference_notch_hz()
    assert r["substrate_rule_bar_hz"] != pytest.approx(off_reference, rel=1e-6)
    # The cell it asks for follows from that bar and from nothing else.
    want = (r["substrate_rule_bar_hz"] / abs(r["amplitude"])) ** (
        1.0 / r["order"])
    assert r["substrate_cell_for_the_bar_m"] == pytest.approx(want, rel=1e-9)
    assert r["n_substrate_cells_for_the_bar"] == int(np.ceil(
        case.SUBSTRATE_THICKNESS_M / want - 1e-12))


# ------------------------------- addendum A.2, on the records
# --- addendum A.2: the re-measured rung and the one-commit ladder
RECORDED_REMEASURE_DELTA_MHZ = 0.0000
RECORDED_REMEASURE_DELTA_PCT = 0.00000
RECORDED_REMEASURE_NOTCH_GHZ = 3.716635
RECORDED_REMEASURE_MESH_IDENTICAL = True
RECORDED_REMEASURE_PROFILES_BIT_IDENTICAL = True
RECORDED_W5_ONE_COMMIT_ORDER = 1.3713
RECORDED_W5_ONE_COMMIT_LIMIT_GHZ = 3.69748
RECORDED_W5_ONE_COMMIT_LIMIT_PCT = 0.6292
RECORDED_W5_ONE_COMMIT_FINEST_PCT = 0.9807
RECORDED_W5_ONE_COMMIT_STEPS_MHZ = [16.158, 17.1453, 6.2457]
RECORDED_W5_ONE_COMMIT_VERDICT = 'HELD'
RECORDED_W8_ONE_COMMIT_VERDICT = 'HELD'
RECORDED_W8_ONE_COMMIT_N_Z = 8
RECORDED_W6_ONE_COMMIT_CLASS = 'FZ dominant'
RECORDED_W6_ONE_COMMIT_DELTA_Z_MHZ = 33.3032
RECORDED_W6_ONE_COMMIT_TOTAL_MHZ = 30.7541
#: order read four ways: declared, coarse endpoint, the two adjacent
#: triples, and the three-parameter fit -- for each ladder.
RECORDED_ORDERS_MIXED = dict(
    declared=1.3713, coarse_endpoint=1.4448,
    triples=[0.8258, 1.8888],
    three_parameter=1.1908,
    three_parameter_limit_ghz=3.69189,
    three_parameter_rms_mhz=0.5173)
RECORDED_ORDERS_ONE_COMMIT = dict(
    declared=1.3713, coarse_endpoint=1.4448,
    triples=[0.8258, 1.8888],
    three_parameter=1.1908,
    three_parameter_limit_ghz=3.69189,
    three_parameter_rms_mhz=0.5173)
RECORDED_REMEASURE_NOTCH_BIT_IDENTICAL = True
RECORDED_REMEASURE_CURVES_BIT_IDENTICAL = True



def test_the_one_commit_ladder_was_solved_by_one_build(arms):
    """Addendum A.2's reason for existing, read off the records.

    W5 fits an order to the differences between four notches, so a rung solved
    by another build of ``rfx/`` puts that build's difference into the order.
    The declared ladder spans two commits; the A.2 ladder spans one. Both are
    kept, and which is which has to be visible in the window itself.
    """
    mixed = ins.w5_fz_ladder(arms, ins.FZ_LADDER)
    one = ins.w5_fz_ladder(arms, ins.FZ_LADDER_ONE_COMMIT)
    # Both ladders carry two commit LABELS; only one of them carries two
    # simulators.  The label is not the thing that matters and the window
    # says so separately.
    assert mixed["n_commits"] == 2 and one["n_commits"] == 2
    assert mixed["solver_tree_equal"] is False
    assert mixed["one_solver_build"] is False
    assert one["solver_tree_equal"] is True
    assert one["one_solver_build"] is True
    # The commit C_off_re adds leaves rfx/ alone; C_off's does not.
    assert ins.solver_tree_equal(one["commits"]) is True
    assert ins.solver_tree_equal(mixed["commits"]) is False
    assert arms["C_off"]["provenance"]["git_sha"] not in one["commits"]
    assert arms["C_off_re"]["provenance"]["git_sha"] not in mixed["commits"]


def test_the_re_measured_rung_solved_the_same_mesh(arms):
    """C_off_re against C_off: same declared board, two solver builds.

    If the meshes differed, the distance between the two notches would be a
    distance between two boards and would say nothing about the solver. Both
    checks are on the RECORDS, not on the rung table: the profiles each arm
    wrote down are compared cell by cell.
    """
    r = ins.fz_verdicts(arms)["remeasurement"]
    assert r["arms"] == ["C_off_re", "C_off"]
    assert r["mesh_identical"] is RECORDED_REMEASURE_MESH_IDENTICAL
    assert (r["profiles_bit_identical"]
            is RECORDED_REMEASURE_PROFILES_BIT_IDENTICAL)
    assert r["commits"][0] != r["commits"][1]
    assert r["grid_cells"][0] == r["grid_cells"][1]
    assert r["notch_new_ghz"] == pytest.approx(RECORDED_REMEASURE_NOTCH_GHZ,
                                               abs=5e-6)
    assert r["delta_mhz"] == pytest.approx(RECORDED_REMEASURE_DELTA_MHZ,
                                           abs=1e-3)
    assert r["delta_pct"] == pytest.approx(RECORDED_REMEASURE_DELTA_PCT,
                                           abs=1e-5)
    # Same rung entry, so the two records must also agree on every cell size.
    for field in ("fine_cell_m", "substrate_cell_m", "z_tail_cell_m",
                  "edge_offset_m", "n_across_metal", "n_substrate_cells",
                  "margin_cells", "lateral_clearance_m"):
        assert arms["C_off_re"][field] == arms["C_off"][field], field


def test_w5_on_the_one_commit_ladder(arms):
    r = ins.w5_fz_ladder(arms, ins.FZ_LADDER_ONE_COMMIT)
    assert r["arms"] == ["Z6", "Z8", "C_off_re", "Z16"]
    assert r["monotone"] is True
    assert r["order"] == pytest.approx(RECORDED_W5_ONE_COMMIT_ORDER, abs=1e-4)
    assert r["limit_ghz"] == pytest.approx(RECORDED_W5_ONE_COMMIT_LIMIT_GHZ,
                                           abs=1e-5)
    assert r["limit_distance_pct"] == pytest.approx(
        RECORDED_W5_ONE_COMMIT_LIMIT_PCT, abs=1e-4)
    assert r["finest_distance_pct"] == pytest.approx(
        RECORDED_W5_ONE_COMMIT_FINEST_PCT, abs=1e-4)
    assert r["differences_mhz"] == pytest.approx(
        RECORDED_W5_ONE_COMMIT_STEPS_MHZ, abs=1e-3)
    assert r["verdict"] == RECORDED_W5_ONE_COMMIT_VERDICT
    # The declared ladder keeps its own verdict; A.2 adds a reading, it does
    # not replace one.
    assert ins.w5_fz_ladder(arms, ins.FZ_LADDER)["verdict"] == (
        RECORDED_VERDICTS["W5"])


def test_w6_and_w8_on_the_one_commit_rung(arms):
    v = ins.fz_verdicts(arms)
    w6 = v["W6_one_commit"]
    assert w6["fine_rung"] == "C_off_re"
    assert w6["classification"] == RECORDED_W6_ONE_COMMIT_CLASS
    assert w6["delta_z_mhz"] == pytest.approx(RECORDED_W6_ONE_COMMIT_DELTA_Z_MHZ,
                                              abs=1e-3)
    assert w6["total_mhz"] == pytest.approx(RECORDED_W6_ONE_COMMIT_TOTAL_MHZ,
                                            abs=1e-3)
    # Its substrate leg is now one simulator; its in-plane leg still is not,
    # because A_off comes from the first record either way.
    assert w6["substrate_leg_solver_equal"] is True
    assert w6["inplane_leg_solver_equal"] is False
    assert v["W6"]["substrate_leg_solver_equal"] is False
    w8 = v["W8_one_commit"]
    assert w8["verdict"] == RECORDED_W8_ONE_COMMIT_VERDICT
    assert w8["one_solver_build"] is True
    assert w8["n_substrate_cells_for_the_bar"] == RECORDED_W8_ONE_COMMIT_N_Z
    assert w8["bar_pct"] == case.FREQ_BAR * 100.0


@pytest.mark.parametrize("ladder,pins", [
    ("FZ_LADDER", "RECORDED_ORDERS_MIXED"),
    ("FZ_LADDER_ONE_COMMIT", "RECORDED_ORDERS_ONE_COMMIT"),
])
def test_the_order_is_recorded_four_ways_for_each_ladder(arms, ladder, pins):
    """Four rungs support several readings of one power; all of them are pinned.

    They are not expected to agree on real rungs -- that spread is the point
    of recording them -- so each is pinned separately. The window is judged on
    ``declared`` and on nothing else, which the last assertion holds.
    """
    want = globals()[pins]
    r = ins.w5_fz_ladder(arms, getattr(ins, ladder))
    assert r["order"] == pytest.approx(want["declared"], abs=1e-4)
    assert r["order_at_coarse_endpoint"] == pytest.approx(
        want["coarse_endpoint"], abs=1e-4)
    assert r["order_adjacent_triples"] == pytest.approx(want["triples"],
                                                        abs=1e-4)
    assert len(r["adjacent_triples"]) == 2
    assert r["order_three_parameter"] == pytest.approx(
        want["three_parameter"], abs=1e-3)
    assert r["limit_three_parameter_ghz"] == pytest.approx(
        want["three_parameter_limit_ghz"], abs=1e-4)
    assert r["rms_three_parameter_mhz"] == pytest.approx(
        want["three_parameter_rms_mhz"], abs=1e-3)
    # The verdict follows the declared reading, whatever the others say.
    lo, hi = r["order_window"]
    assert r["verdict"] == ("HELD" if r["monotone"] and lo <= r["order"] <= hi
                            else "FIRED")


def test_the_tables_carry_both_ladders_and_the_re_measurement(arms):
    """The note's Results section shows each window twice and says which is which."""
    md = ins.fz_markdown_tables(arms)
    assert "### F.4a The same rung solved twice" in md
    for needle in ("as section 4 declares it", "addendum A.2, one solver build",
                   "section 4's rung", "addendum A.2's rung",
                   "section 4's ladder", "addendum A.2's ladder"):
        assert needle in md, needle
    assert md.count("**W5 -- the FZ ladder is a ladder") == 2
    assert md.count("**W6 -- attribution at the A_off to") == 2
    assert md.count("**W8 -- is z enough?") == 2
    assert "| C_off_re |" in md
    assert "how the order is read" in md
    assert md.rstrip().endswith("Conclusions: leader fills.")


def test_the_re_measurement_found_no_difference_at_all(arms):
    """What the re-measured rung actually answers, pinned as measured.

    The same board was solved by two simulators eleven files and 742 changed
    lines apart, and the two agree to the last bit of a float64 -- not the
    notch only, the whole S matrix on the whole frequency grid. That is a
    stronger statement than "close enough", so it is pinned as bit equality
    and not as a tolerance.
    """
    r = ins.fz_verdicts(arms)["remeasurement"]
    assert r["solver_tree_equal"] is False, (
        "the two commits must differ in rfx/ or this measures nothing")
    assert r["notch_bit_identical"] is RECORDED_REMEASURE_NOTCH_BIT_IDENTICAL
    assert r["curves_bit_identical"] is RECORDED_REMEASURE_CURVES_BIT_IDENTICAL
    assert r["delta_mhz"] == 0.0
    for field in ("s21_re", "s21_im", "s11_re", "s11_im", "freqs_hz"):
        assert arms["C_off_re"][field] == arms["C_off"][field], field
    # So every window reads the same on both ladders, and that is a result
    # rather than a coincidence of rounding.
    for key in ("order", "limit_ghz", "limit_distance_pct"):
        assert ins.w5_fz_ladder(arms, ins.FZ_LADDER)[key] == pytest.approx(
            ins.w5_fz_ladder(arms, ins.FZ_LADDER_ONE_COMMIT)[key], rel=1e-12)


def test_the_solver_tree_check_tells_a_label_from_a_simulator():
    """``solver_tree_equal`` asks git about rfx/, not about the commit string.

    Two commits that differ only in a note or in this instrument leave the
    simulator alone. If this compared labels it would call the A.2 ladder two
    builds, which is the mistake the field exists to avoid.
    """
    assert ins.solver_tree_equal([]) is True
    assert ins.solver_tree_equal(["a" * 40]) is True
    # A sha git cannot resolve is "not checkable", not "equal".
    assert ins.solver_tree_equal(["0" * 40, "1" * 40]) is None

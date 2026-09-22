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
    assert sorted(ins.FZ_ARMS) == ["F16Z6", "ON13", "Z16", "Z6", "Z8"]
    assert set(ins.FZ_ARMS) & set(ins.ARMS) == set()
    assert ins.FZ_LADDER == ("Z6", "Z8", "C_off", "Z16")
    assert ins.FZ_REUSED_ARMS == ("A_on", "A_off", "C_off")

    h, w_metal = case.SUBSTRATE_THICKNESS_M, case.TRACE_WIDTH_M
    want = {
        # arm: (n_z, n, placement, F um, FZ um)
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
    assert fine["Z6"] == fine["Z8"] == fine["Z16"]          # FZ ladder: F fixed
    assert fz["Z6"] == fz["F16Z6"] == fz["ON13"]            # F ladder: FZ fixed
    assert len({fz["Z6"], fz["Z8"], fz["Z16"]}) == 3        # and FZ moves


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
def _fake_arms(notches_ghz: dict, fine_um: dict, fz_um: dict) -> dict:
    """Arm records carrying nothing but what W5-W8 read."""
    return {k: dict(_synthetic_curve(v * 1e9),
                    arm=k, fine_cell_m=fine_um[k] * 1e-6,
                    substrate_cell_m=fz_um[k] * 1e-6,
                    n_substrate_cells=int(round(
                        case.SUBSTRATE_THICKNESS_M / (fz_um[k] * 1e-6))))
            for k, v in notches_ghz.items()}


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
    # The substrate rule it derives, re-derived here from its own two numbers.
    want = (case.FREQ_BAR * ins.reference_notch_hz()
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

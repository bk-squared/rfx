"""The shared judging rules in ``tests/crossval/_v2_judging.py``, on synthetic
curves and on the feature frequencies logged by three rfx ladders.

No rfx, no solver: every input is written here.  The ladders are the judged
feature frequencies the cases' GPU runs printed (VESSL 369367264046, log on
the lab share under research/rfx/.omx/msl-cases-ladder-post1213/
20260923T144706Z-6e6f07a0/ladder.log, for the notch and Sheen filters; the
RT5880 patch antenna's ladder values as the crossval-lane brief of 2026-09-24
quotes them).  The input-resistance and level-crossing rules (PI 2026-09-24)
are tested on closed-form curves written here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.crossval._v2_judging import (
    CONVERGED,
    CROSSING_FIT_BINS,
    DEEP_NULL_DB,
    FLAT_STEP,
    FREQ_BAR,
    LADDER_AGREEMENT,
    MAG_BAR_DB,
    NOT_SHOWN_TO_CONVERGE,
    RESISTANCE_BAR,
    aligned_magnitude,
    input_resistance,
    level_crossing,
    mesh_statement,
    refined_remax,
    remax_half_grid_witness,
)

GHZ = 1e9


def _labels(n: int) -> list[str]:
    return [f"rung {i}" for i in range(n)]


# ------------------------------------------------------------ mesh statement
@pytest.mark.parametrize("reversal, flat, verdict", [
    (0.00099, [False, True], CONVERGED),               # just inside 0.1 %
    (0.00101, [False, False], NOT_SHOWN_TO_CONVERGE),  # just outside it
])
def test_a_reversal_within_the_flat_step_does_not_break_the_trend(reversal, flat, verdict):
    """A ladder falling by 0.5 % and then rising by ``reversal``: a rise of
    0.099 % is flat and the ladder is monotone; 0.101 % is a reversal."""
    f0 = 3.70 * GHZ
    f1 = f0 * (1.0 - 0.005)
    f2 = f1 * (1.0 + reversal)
    m = mesh_statement([f0, f1, f2], _labels(3))
    print(f"\n  reversal {reversal*100:.3f} %: {m['verdict']} -- {m['reason']}")
    assert m["flat"] == flat
    assert m["monotone"] is (verdict == CONVERGED)
    assert m["verdict"] == verdict


def test_a_real_reversal_is_not_shown_to_converge():
    """The Sheen low-pass filter's stopband null along its four rungs: it moves
    by about half a percent at every step and changes direction twice."""
    f = [8.19634 * GHZ, 8.23697 * GHZ, 8.20201 * GHZ, 8.23472 * GHZ]
    m = mesh_statement(f, ["264.67µm", "158.80µm", "113.43µm", "66.17µm"])
    print(f"\n  {m['verdict']} -- {m['reason']}")
    assert m["flat"] == [False, False, False]
    assert m["monotone"] is False
    # the last two rungs are 0.399 % apart, inside the 1 % bar: the reversal
    # alone is what refuses the ladder
    assert m["last_two_pct"] == pytest.approx(0.3988, abs=1e-3)
    assert m["verdict"] == NOT_SHOWN_TO_CONVERGE
    assert "reverses" in m["reason"]


@pytest.mark.parametrize("name, f_ghz, flat", [
    ("rising", (2.300, 2.330, 2.345), [False, False]),
    ("falling", (3.800, 3.770, 3.755), [False, False]),
    # the RT5880 patch antenna: rising, the last step 0.0955 % (flat)
    ("patch", (2.337070, 2.356146, 2.358396), [False, True]),
    # the MSL notch filter after #1213: +0.018 % (flat), then -0.298 %
    ("notch", (3.68600, 3.68666, 3.67568), [True, False]),
])
def test_monotone_ladders_converge(name, f_ghz, flat):
    m = mesh_statement([v * GHZ for v in f_ghz], _labels(len(f_ghz)))
    print(f"\n  {name}: {m['verdict']} -- {m['reason']}")
    assert m["flat"] == flat
    assert m["monotone"] is True
    assert m["last_two_pct"] < 1.0
    assert m["verdict"] == CONVERGED


def test_last_two_rungs_a_percent_apart_are_not_shown_to_converge():
    """Monotone, but the last two rungs differ by 1.31 % (the notch filter's
    ladder before #1213 ended 3.8034 -> 3.7537 GHz)."""
    m = mesh_statement([3.90 * GHZ, 3.8034 * GHZ, 3.7537 * GHZ], _labels(3))
    print(f"\n  {m['verdict']} -- {m['reason']}")
    assert m["monotone"] is True
    assert m["last_two_pct"] == pytest.approx(1.3067, abs=1e-3)
    assert m["verdict"] == NOT_SHOWN_TO_CONVERGE
    assert "last two rungs" in m["reason"]


# -------------------------------------- the mesh statement for another bar
OHM = ("Ω", 1.0, 3)


def test_the_keyword_defaults_are_the_frequency_bars():
    """Called without the keywords, the statement is the one it always was:
    FLAT_STEP 0.1 % and LADDER_AGREEMENT 1 %, the same verdicts and flags as
    passing them explicitly."""
    for f_ghz in ((3.68600, 3.68666, 3.67568), (8.19634, 8.23697, 8.20201, 8.23472),
                  (3.90, 3.8034, 3.7537)):
        f = [v * GHZ for v in f_ghz]
        a = mesh_statement(f, _labels(len(f)))
        b = mesh_statement(f, _labels(len(f)), flat_step=FLAT_STEP, agreement=LADDER_AGREEMENT)
        assert (a["verdict"], a["flat"], a["reason"]) == (b["verdict"], b["flat"], b["reason"])
        assert a["flat_step"] == FLAT_STEP == 0.001
        assert a["agreement"] == LADDER_AGREEMENT == 0.01


def test_a_resistance_ladder_takes_its_own_flat_step():
    """R 75.0 -> 76.0 -> 75.9 ohm: +1.333 %, then -0.132 %.  With R's flat step
    (RESISTANCE_BAR / 10 = 0.2 %) the second step is flat and the ladder
    converges; with the frequency's 0.1 % it is a reversal."""
    r = [75.0, 76.0, 75.9]
    own = mesh_statement(r, _labels(3), flat_step=RESISTANCE_BAR / 10,
                         agreement=RESISTANCE_BAR, unit=OHM)
    freq = mesh_statement(r, _labels(3), unit=OHM)
    print(f"\n  R's bars: {own['verdict']} -- {own['reason']}\n  frequency's bars: "
          f"{freq['verdict']} -- {freq['reason']}")
    assert own["flat"] == [False, True] and own["verdict"] == CONVERGED
    assert freq["flat"] == [False, False] and freq["verdict"] == NOT_SHOWN_TO_CONVERGE
    assert "75.900 Ω" in own["reason"] and "0.2 %" in own["reason"]


def test_a_resistance_ladder_takes_its_own_last_two_bar():
    """R 72.0 -> 74.0 -> 75.2 ohm: one direction, the last two 1.622 % apart:
    inside R's 2 % bar, outside the frequency's 1 %."""
    r = [72.0, 74.0, 75.2]
    own = mesh_statement(r, _labels(3), flat_step=RESISTANCE_BAR / 10,
                         agreement=RESISTANCE_BAR, unit=OHM)
    freq = mesh_statement(r, _labels(3), unit=OHM)
    print(f"\n  R's bars: {own['verdict']}; frequency's bars: {freq['verdict']} -- "
          f"{freq['reason']}")
    assert own["last_two_pct"] == pytest.approx(1.6216, abs=1e-3)
    assert own["verdict"] == CONVERGED
    assert freq["verdict"] == NOT_SHOWN_TO_CONVERGE and "bar 1 %" in freq["reason"]


# ---------------------------------------------------------- aligned magnitude
REF_F = np.linspace(1e6, 7e9, 1601)          # the openEMS record's grid
OUR_F = np.linspace(0.7e9, 7e9, 4001)        # fine, so interpolation adds ~nothing
BAND = (2e9, 7e9)
F_NOTCH = 3.6 * GHZ


def _dip(f, f0, half_width, depth_db):
    """|S|^2 of a Lorentzian dip reaching ``depth_db`` at ``f0``."""
    floor = 10.0 ** (depth_db / 10.0)
    return 1.0 - (1.0 - floor) * half_width ** 2 / ((f - f0) ** 2 + half_width ** 2)


def _s21_db(f, *, scale=1.0, depth_db=-45.0):
    """A notch at 3.6 GHz and a shallower, wider dip at 6.2 GHz, as |S21| in
    dB.  ``scale`` multiplies the frequency argument, so the curve's features
    sit at ``f0 / scale``: a uniform scale of the whole axis."""
    x = np.asarray(f, float) * scale
    p = _dip(x, F_NOTCH, 0.20 * GHZ, depth_db) * _dip(x, 6.2 * GHZ, 0.25 * GHZ, -15.0)
    return 10.0 * np.log10(p)


def test_a_pure_frequency_scale_aligns_to_zero():
    """rfx's curve is the reference's with every frequency 0.8 % lower (inside
    the 1 % bar).  Scaled back, it lies on the reference: both dips coincide.
    Unaligned, the notch's skirts differ by several dB; translated instead of
    scaled, the second dip would stay 23 MHz off."""
    s = 1.008
    ref_db = _s21_db(REF_F)
    our_db = _s21_db(OUR_F, scale=s)
    c = aligned_magnitude(REF_F, ref_db, OUR_F, our_db, F_NOTCH, F_NOTCH / s, BAND)
    print(f"\n  scale {c['scale']:.6f}: aligned max |ΔdB| {c['max_abs_delta_db']:.4f} dB "
          f"at {c['f_at_max_hz']/1e9:.4f} GHz over {c['n_compared']} bins; unaligned "
          f"{c['unaligned_max_abs_delta_db']:.3f} dB at "
          f"{c['unaligned_f_at_max_hz']/1e9:.4f} GHz over {c['unaligned_n_compared']}")
    assert c["scale"] == pytest.approx(s)
    assert c["n_compared"] > 1000
    assert c["max_abs_delta_db"] < 0.05
    assert c["unaligned_max_abs_delta_db"] > 3.0


def test_a_depth_only_change_is_still_caught():
    """Same frequency, a shallow notch (-8 dB instead of -45 dB): alignment
    moves nothing (scale 1), and the skirts above -20 dB differ by far more
    than 2 dB."""
    ref_db = _s21_db(REF_F)
    our_db = _s21_db(OUR_F, depth_db=-8.0)
    c = aligned_magnitude(REF_F, ref_db, OUR_F, our_db, F_NOTCH, F_NOTCH, BAND)
    print(f"\n  aligned max |ΔdB| {c['max_abs_delta_db']:.3f} dB at "
          f"{c['f_at_max_hz']/1e9:.4f} GHz over {c['n_compared']} bins")
    assert c["scale"] == 1.0
    assert c["max_abs_delta_db"] > MAG_BAR_DB
    assert c["max_abs_delta_db"] == pytest.approx(c["unaligned_max_abs_delta_db"])


def test_bins_below_the_deep_null_level_are_not_compared():
    """The two curves differ ONLY where one of them is below -20 dB: 15 dB
    apart in the notch, and one bin where the reference is at -19.5 dB and
    rfx at -20.5 dB.  None of those bins is compared, so the maximum is 0."""
    f = np.linspace(2e9, 7e9, 501)
    ref_db = _s21_db(f)
    our_db = ref_db.copy()
    deep = ref_db < DEEP_NULL_DB
    assert deep.sum() >= 3
    our_db[deep] += 15.0
    edge = int(np.flatnonzero(ref_db >= DEEP_NULL_DB)[0])
    ref_db[edge], our_db[edge] = -19.5, -20.5
    c = aligned_magnitude(f, ref_db, f, our_db, F_NOTCH, F_NOTCH, BAND)
    expected = int(((ref_db >= DEEP_NULL_DB) & (our_db >= DEEP_NULL_DB)).sum())
    print(f"\n  {deep.sum()} bins in the notch, 1 at the edge; compared "
          f"{c['n_compared']} of {c['n_in_band']}; max |ΔdB| {c['max_abs_delta_db']}")
    assert c["n_compared"] == expected == c["n_in_band"] - int(deep.sum()) - 1
    assert c["max_abs_delta_db"] == 0.0


def test_the_floor_is_applied_to_the_aligned_rfx_curve_not_the_unaligned_one():
    """rfx's notch sits 0.9 % below the reference's, and rfx's skirt is 3 dB
    too deep 33 MHz below the reference notch.  After alignment that skirt bin
    is at about -6 dB on both curves and must be compared: the bar reads 3 dB.
    On rfx's UNALIGNED axis the same reference bin falls in rfx's own notch
    (below -20 dB); a floor taken from the unaligned curve would drop the bin
    and hide the error (the review mutation of 2026-09-24)."""
    f0_ref = 3.674 * GHZ
    f0_our = f0_ref * 0.991
    scale = f0_ref / f0_our
    gamma = 30e6
    floor_lin = 10 ** (-45.0 / 20)

    def notch_db(f, f0):
        d = f - f0
        return 20 * np.log10(np.sqrt((d ** 2 + (gamma * floor_lin) ** 2)
                                     / (d ** 2 + gamma ** 2)))

    ref_f = np.arange(3.40e9, 3.90e9 + 1, 1e6)
    our_f = np.arange(3.30e9, 4.00e9 + 1, 0.5e6)
    ref_db = notch_db(ref_f, f0_ref)
    our_db = notch_db(our_f * scale, f0_ref)          # identical after alignment
    skirt = np.abs(our_f * scale - (f0_ref - 33e6)) <= 1.5e6
    our_db[skirt] -= 3.0                               # the shape error
    c = aligned_magnitude(ref_f, ref_db, our_f, our_db, f0_ref, f0_our,
                          (ref_f[0], ref_f[-1]))
    err_bins = np.abs(ref_f - (f0_ref - 33e6)) <= 1.5e6
    unaligned_at_err = np.interp(ref_f[err_bins], our_f, our_db)
    worst_unaligned_rfx = float(unaligned_at_err.max())
    print(f"\n  aligned max |ΔdB| {c['max_abs_delta_db']:.3f} dB at "
          f"{c['f_at_max_hz']/1e9:.4f} GHz; at the error bins the unaligned rfx curve reads at most "
          f"{worst_unaligned_rfx:.1f} dB")
    assert c["max_abs_delta_db"] == pytest.approx(3.0, abs=0.05)
    assert c["max_abs_delta_db"] > MAG_BAR_DB
    # the premise: on rfx's unaligned axis every error bin is below the floor
    assert worst_unaligned_rfx < DEEP_NULL_DB


# ------------------------------------------- resonance and input resistance
PATCH_F = np.linspace(1.6e9, 3.4e9, 901)        # the RT5880 record's grid, 2 MHz
PATCH_BAND = (1.9325e9, 2.8987e9)               # 0.8-1.2 x its TL-model TM010


def _zin(f, f0, r0, q, x_series):
    """A parallel resonance (R ``r0`` at ``f0``, quality ``q``) behind a series
    probe reactance ``x_series``: Re(Zin) peaks at ``r0`` exactly at ``f0``,
    whatever ``x_series`` is."""
    f = np.asarray(f, float)
    return r0 / (1.0 + 1j * q * (f / f0 - f0 / f)) + 1j * x_series


def _s11_dip_hz(f, zin, band=PATCH_BAND, z0=50.0):
    s = np.abs((zin - z0) / (zin + z0))
    k = (f >= band[0]) & (f <= band[1])
    return float(f[k][np.argmin(s[k])])


def test_the_re_max_does_not_move_with_the_probe_reactance():
    """The same cavity (f0 2.303804 GHz, R 75.7 ohm, Q 12) behind two probe
    reactances, +43 and +50.5 ohm (the two solvers' X at f0 on the RT5880
    patch).  Re(Zin) peaks at the same f0 and height for both; the |S11|
    minimum, computed from the same Zin against 50 ohm, sits elsewhere and
    moves with the reactance."""
    f0, r0, q = 2.303804e9, 75.7, 12.0
    a = _zin(PATCH_F, f0, r0, q, 43.0)
    b = _zin(PATCH_F, f0, r0, q, 50.5)
    ea, eb = refined_remax(PATCH_F, a, *PATCH_BAND), refined_remax(PATCH_F, b, *PATCH_BAND)
    dip_a, dip_b = _s11_dip_hz(PATCH_F, a), _s11_dip_hz(PATCH_F, b)
    print(f"\n  Re-max f0 {ea['f0_hz']/1e9:.6f} / {eb['f0_hz']/1e9:.6f} GHz, R "
          f"{ea['r_ohm']:.3f} / {eb['r_ohm']:.3f} ohm, X {ea['x_ohm']:.2f} / "
          f"{eb['x_ohm']:.2f} ohm; |S11| minimum {dip_a/1e9:.4f} / {dip_b/1e9:.4f} GHz")
    assert ea["flags"] == [] and eb["flags"] == []
    assert ea["f0_hz"] == pytest.approx(f0, rel=1e-5)
    assert eb["f0_hz"] == pytest.approx(f0, rel=1e-5)
    assert ea["r_ohm"] == pytest.approx(r0, rel=1e-4)
    assert eb["x_ohm"] - ea["x_ohm"] == pytest.approx(7.5, abs=0.05)
    # the premise: the |S11| minimum is 3 % away from f0 and moves with the
    # probe (by one 2 MHz bin here)
    assert abs(dip_a - f0) / f0 > 0.03
    assert dip_b - dip_a == pytest.approx(2e6)


def test_the_re_max_is_refined_between_bins():
    """f0 0.9 MHz from the nearest 2 MHz bin: the parabola vertex lands within
    0.001 % of it, the bin itself 0.039 % off; the half-grid witness puts the
    two half-density estimates well inside one fine bin of each other."""
    f0 = 2.3031e9
    z = _zin(PATCH_F, f0, 75.7, 12.0, 43.0)
    e = refined_remax(PATCH_F, z, *PATCH_BAND)
    w = remax_half_grid_witness(PATCH_F, z, *PATCH_BAND)
    print(f"\n  bin {e['bin_f_hz']/1e9:.6f} GHz, refined {e['f0_hz']/1e9:.6f} GHz "
          f"(shift {e['sub_bin_shift']:+.3f} bin); half grids "
          f"{[round(v/1e9, 6) for v in w['f0_even_odd_hz']]} GHz, spread "
          f"{w['spread_bins']:.4f} bin")
    assert abs(e["bin_f_hz"] - f0) / f0 > 3.5e-4
    assert abs(e["f0_hz"] - f0) / f0 < 1e-5
    assert w["spread_bins"] < 0.1


def test_f0_and_r_are_read_each_on_its_own_curve():
    """rfx's resonance 0.604 % above the reference's, the same 75.7 ohm
    resistance there, a probe reactance 7.5 ohm higher.  Each read at its own
    Re(Zin) peak the resistances agree; rfx's Re(Zin) read at the
    REFERENCE's f0 sits down its own resonance curve, 11.6 % low."""
    f_ref, f_our = 2.303804e9, 2.303804e9 * 1.00604
    ref = _zin(PATCH_F, f_ref, 75.7, 30.0, 43.0)
    ours = _zin(PATCH_F, f_our, 75.7, 30.0, 50.5)
    c = input_resistance(PATCH_F, ref, PATCH_F, ours, PATCH_BAND)
    wrong = float(np.interp(f_ref, PATCH_F, ours.real))
    print(f"\n  f0 {c['ref']['f0_hz']/1e9:.6f} -> {c['ours']['f0_hz']/1e9:.6f} GHz "
          f"({c['f0_pct']:+.3f} %); R {c['r_ref_ohm']:.3f} / {c['r_ours_ohm']:.3f} ohm "
          f"({c['rel']*100:.3f} %, bar {c['bar']*100:.0f} %); X {c['x_ref_ohm']:.2f} / "
          f"{c['x_ours_ohm']:.2f} ohm (reported); rfx's Re(Zin) at the reference's "
          f"f0: {wrong:.3f} ohm")
    assert c["f0_pct"] == pytest.approx(0.604, abs=2e-3)
    assert c["rel"] < 1e-3
    assert c["passed"] is True
    assert c["x_ours_ohm"] - c["x_ref_ohm"] == pytest.approx(7.5, abs=0.2)
    assert abs(wrong - 75.7) / 75.7 > 0.10       # the premise of the test


@pytest.mark.parametrize("ratio, passed", [(1.019, True), (1.021, False),
                                           (0.981, True), (0.979, False)])
def test_the_resistance_bar_is_two_percent(ratio, passed):
    """Same resonance, rfx's resistance ``ratio`` times the reference's."""
    f0 = 2.303804e9
    ref = _zin(PATCH_F, f0, 75.7, 12.0, 43.0)
    ours = _zin(PATCH_F, f0, 75.7 * ratio, 12.0, 43.0)
    c = input_resistance(PATCH_F, ref, PATCH_F, ours, PATCH_BAND)
    print(f"\n  ratio {ratio}: {c['rel']*100:.3f} % against {c['bar']*100:.0f} %")
    assert RESISTANCE_BAR == 0.02
    assert c["bar"] == RESISTANCE_BAR
    assert c["rel"] == pytest.approx(abs(ratio - 1.0), rel=1e-3)
    assert c["passed"] is passed


def test_a_peak_on_the_window_edge_is_refused():
    """A Re(Zin) that still rises at the window's top edge has no peak inside
    it: the estimate is flagged, and judging it is refused."""
    ref = _zin(PATCH_F, 2.303804e9, 75.7, 12.0, 43.0)
    ours = _zin(PATCH_F, 3.2e9, 75.7, 12.0, 43.0)
    e = refined_remax(PATCH_F, ours, *PATCH_BAND)
    assert e["flags"] == ["peak on the window's edge"]
    with pytest.raises(ValueError, match="cannot be judged"):
        input_resistance(PATCH_F, ref, PATCH_F, ours, PATCH_BAND)


# ---------------------------------------------------- level crossing (R-c)
SHEEN_F = np.linspace(0.5e9, 20e9, 801)         # the Sheen record's grid, 24.375 MHz
F_CORNER = 5.5717e9


def _falling_db(f, slope_db_per_ghz, f0=F_CORNER, level_db=-3.78, curvature=0.0):
    """A dB curve crossing ``level_db`` at ``f0`` with the given slope, plus an
    optional quadratic term (dB/GHz^2)."""
    x = (np.asarray(f, float) - f0) / 1e9
    return level_db + slope_db_per_ghz * x + curvature * x ** 2


def test_the_tolerance_is_the_magnitude_bar_over_the_reference_slope():
    """Reference falling at 12 dB/GHz: 2 dB / 12 dB/GHz = 166.7 MHz, 2.99 % of
    5.5717 GHz.  rfx crossing 2 % high passes, where the 1 % bar would fail
    it; 3.5 % high fails."""
    ref_db = _falling_db(SHEEN_F, -12.0)
    ok = level_crossing(SHEEN_F, ref_db, F_CORNER, F_CORNER * 1.02)
    bad = level_crossing(SHEEN_F, ref_db, F_CORNER, F_CORNER * 1.035)
    print(f"\n  slope {ok['slope_db_per_ghz']:.4f} dB/GHz over {ok['n_fit_bins']} bins "
          f"({ok['fit_f_hz'][0]/1e9:.5f}-{ok['fit_f_hz'][1]/1e9:.5f} GHz): tolerance "
          f"{ok['tol_hz']/1e6:.2f} MHz = {ok['tol_pct']:.3f} %; offset "
          f"{ok['offset_pct']:+.2f} % -> {ok['passed']}, {bad['offset_pct']:+.2f} % -> "
          f"{bad['passed']}")
    assert ok["slope_db_per_ghz"] == pytest.approx(-12.0, rel=1e-9)
    assert ok["tol_hz"] == pytest.approx(MAG_BAR_DB / 12.0 * 1e9, rel=1e-9)
    assert ok["tol_pct"] == pytest.approx(100 * (MAG_BAR_DB / 12.0 * 1e9) / F_CORNER, rel=1e-9)
    assert ok["tol_pct"] > 100 * FREQ_BAR          # the rule is not the 1 % bar
    assert ok["passed"] is True
    assert bad["passed"] is False


def test_the_slope_is_the_reference_s_not_rfx_s():
    """rfx's curve falls twice as steeply as the reference's.  The tolerance
    is the reference's (166.7 MHz), and rfx's own slope is only reported: with
    rfx's slope the tolerance would halve and the 2 % offset would fail."""
    ref_db = _falling_db(SHEEN_F, -12.0)
    f_our = F_CORNER * 1.02
    our_db = _falling_db(SHEEN_F, -24.0, f0=f_our)
    c = level_crossing(SHEEN_F, ref_db, F_CORNER, f_our, our_f=SHEEN_F, our_db=our_db)
    print(f"\n  reference slope {c['slope_db_per_ghz']:.3f} dB/GHz (judged), rfx "
          f"{c['our_slope_db_per_ghz']:.3f} dB/GHz (reported): tolerance "
          f"{c['tol_hz']/1e6:.2f} MHz, offset {c['offset_hz']/1e6:.2f} MHz -> {c['passed']}")
    assert c["slope_db_per_ghz"] == pytest.approx(-12.0, rel=1e-9)
    assert c["our_slope_db_per_ghz"] == pytest.approx(-24.0, rel=1e-9)
    assert c["tol_hz"] == pytest.approx(MAG_BAR_DB / 12.0 * 1e9, rel=1e-9)
    assert c["passed"] is True


def test_the_slope_is_a_five_bin_fit_not_one_bin():
    """A reference with a 0.3 dB step between two bins right at the crossing:
    the five-bin fit reads the local trend with the step spread over its
    97.5 MHz (-15.7 dB/GHz); a two-point difference across the step reads
    -24.3 dB/GHz, and one bin carries no slope at all."""
    ref_db = _falling_db(SHEEN_F, -12.0)
    k = int(np.argmin(np.abs(SHEEN_F - F_CORNER)))
    ref_db[k + 1:] -= 0.3
    c = level_crossing(SHEEN_F, ref_db, F_CORNER, F_CORNER)
    df = SHEEN_F[1] - SHEEN_F[0]
    two_point = (ref_db[k + 1] - ref_db[k]) / df * 1e9
    print(f"\n  {c['n_fit_bins']} bins {c['fit_f_hz'][0]/1e9:.5f}-{c['fit_f_hz'][1]/1e9:.5f} "
          f"GHz: slope {c['slope_db_per_ghz']:.3f} dB/GHz, residual "
          f"{c['fit_residual_db']:.3f} dB; two-point slope across the step {two_point:.3f}")
    assert CROSSING_FIT_BINS == 5
    assert c["n_fit_bins"] == 5
    assert c["fit_f_hz"][1] - c["fit_f_hz"][0] == pytest.approx(4 * df)
    assert c["slope_db_per_ghz"] == pytest.approx(-15.692, abs=1e-3)
    assert two_point == pytest.approx(-24.308, abs=1e-3)
    with pytest.raises(ValueError, match="three bins"):
        level_crossing(SHEEN_F, ref_db, F_CORNER, F_CORNER, n_fit=1)


@pytest.mark.parametrize("name, ref_db", [
    ("flat", np.full(SHEEN_F.size, -3.0)),
    # an extremum: the local line is flat, the curvature is the whole shape
    ("extremum", _falling_db(SHEEN_F, 0.0, curvature=-40.0)),
])
def test_a_flat_reference_is_refused(name, ref_db):
    """A slope that cannot carry a tolerance (infinite or meaningless) is a
    refusal, not a pass."""
    with pytest.raises(ValueError, match="flat"):
        level_crossing(SHEEN_F, ref_db, F_CORNER, F_CORNER * 1.001)
    print(f"\n  {name}: refused")


def test_a_crossing_at_the_edge_of_the_curve_is_refused():
    ref_db = _falling_db(SHEEN_F, -12.0, f0=SHEEN_F[1])
    with pytest.raises(ValueError, match="runs off"):
        level_crossing(SHEEN_F, ref_db, SHEEN_F[1], SHEEN_F[1])

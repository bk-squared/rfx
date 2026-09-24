"""The shared judging rules in ``tests/crossval/_v2_judging.py``, on synthetic
curves and on the feature frequencies logged by three rfx ladders.

No rfx, no solver: every input is written here.  The ladders are the judged
feature frequencies the cases' GPU runs printed (VESSL 369367264046, log on
the lab share under research/rfx/.omx/msl-cases-ladder-post1213/
20260923T144706Z-6e6f07a0/ladder.log, for the notch and Sheen filters; the
RT5880 patch antenna's ladder values as the crossval-lane brief of 2026-09-24
quotes them).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.crossval._v2_judging import (
    CONVERGED,
    DEEP_NULL_DB,
    MAG_BAR_DB,
    NOT_SHOWN_TO_CONVERGE,
    aligned_magnitude,
    mesh_statement,
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

"""#498 — locks on the two zero-compute findings, and on the comparator fix itself.

Pure NumPy over committed artifacts. No solver, no openEMS, no VESSL.

These tests exist so the next reader cannot re-open a closed branch by hand-waving:

  * ``kurokawa_renormalize`` is the identity exactly when the two reference
    impedances agree, and the sqrt(Z_j/Z_i) rule otherwise (fail-first contract).
  * Applying it to the COMMITTED openEMS Stage-2 artifact does NOT restore
    passivity and leaves a frequency-drifting residual. The fix is a real fix;
    it is not a gate pass, and this pins the honest number so nobody quotes it
    as one.
  * The lw-diagonal flux bracket is negative for an instrument reason, not a
    box-width reason: ``r1 > 1`` is non-physical on its own, and the bracket
    stays negative when the 5-face box is replaced by a box-independent
    three-plane power budget.

Nothing here pins an lw-diagonal VALUE (predeclaration §10 item 1) and nothing
here writes a branch up as "vindicated" (§10 item 11).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_MEAS = (_REPO / "scripts" / "diagnostics" / "_mixed_refplane_logs"
         / "measurement_369367257597_60p.json")
_REFEREE = (_REPO / "scripts" / "diagnostics" / "_probe_fed_msl_referee_logs"
            / "stage2_369367257643_referee.json")
_REFEREE_SRC = (_REPO / "scripts" / "diagnostics"
                / "probe_fed_msl_openems_referee.py")
_REANALYSIS_SRC = (_REPO / "scripts" / "diagnostics"
                   / "mixed_anchor_reanalysis.py")
_REANALYSIS_JSON = (_REPO / "scripts" / "diagnostics"
                    / "_mixed_anchor_reanalysis" / "anchor_reanalysis.json")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _doc(path: Path) -> dict:
    if not path.exists():  # pragma: no cover - artifacts are committed
        pytest.skip(f"artifact not present: {path}")
    return json.loads(path.read_text())


def _cx(pairs) -> np.ndarray:
    a = np.asarray(pairs, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


# ---------------------------------------------------------------------------
# The comparator fix, as arithmetic
# ---------------------------------------------------------------------------
def test_kurokawa_is_the_identity_when_the_reference_impedances_agree():
    ref = _load_module(_REFEREE_SRC, "probe_fed_msl_openems_referee")
    rng = np.random.default_rng(498)
    s = {k: rng.normal(size=6) + 1j * rng.normal(size=6)
         for k in ("s11", "s21", "s22", "s12")}
    out = ref.kurokawa_renormalize(**s, z_port1=50.0, z_port2=50.0)
    for k in ("s11", "s21", "s22", "s12"):
        np.testing.assert_allclose(out[k], s[k], rtol=0, atol=0)
    np.testing.assert_allclose(out["reciprocity_ratio_renormalized"],
                               out["reciprocity_ratio_raw"], rtol=1e-15)


def test_kurokawa_applies_exactly_sqrt_of_the_impedance_ratio():
    ref = _load_module(_REFEREE_SRC, "probe_fed_msl_openems_referee")
    s21 = np.asarray([1.0 + 0.0j, 0.5 - 0.25j])
    s12 = np.asarray([1.0 + 0.0j, 0.5 - 0.25j])
    z1, z2 = 50.0, 61.0 - 0.9j          # Re only, per the power-wave rule
    out = ref.kurokawa_renormalize(s11=np.zeros(2, complex), s21=s21,
                                   s22=np.zeros(2, complex), s12=s12,
                                   z_port1=z1, z_port2=z2)
    np.testing.assert_allclose(out["s21"], s21 * np.sqrt(z1 / np.real(z2)),
                               rtol=1e-14)
    np.testing.assert_allclose(out["s12"], s12 * np.sqrt(np.real(z2) / z1),
                               rtol=1e-14)
    # the diagonals are untouched by a diagonal similarity transform
    np.testing.assert_allclose(out["s11"], 0.0, atol=0)
    np.testing.assert_allclose(out["s22"], 0.0, atol=0)
    # and the renormalized reciprocity ratio IS the "required factor" drift
    np.testing.assert_allclose(out["reciprocity_ratio_renormalized"],
                               out["kurokawa_residual_drift"], rtol=1e-14)


def test_kurokawa_rejects_a_non_positive_reference_impedance():
    ref = _load_module(_REFEREE_SRC, "probe_fed_msl_openems_referee")
    z = np.zeros(3, complex)
    with pytest.raises(ref.ConfigError):
        ref.kurokawa_renormalize(s11=z, s21=z, s22=z, s12=z,
                                 z_port1=50.0, z_port2=-1.0)


def test_a_failing_report_only_renormalization_cannot_discard_a_solved_leg():
    """The additive field is attached AFTER the solve, so it must never raise.

    ``_run_stage2_leg`` runs this on a machine where the leg has just cost a
    VESSL slot. ``kurokawa_renormalize`` rejects ``Re Z <= 0``, and the
    measured ``z0_msl_measured_ohm`` swings (Re 33.19-81.21 ohm on the
    committed dx = 80 um leg), so an unguarded call is a new abort point
    after the expensive part. Here the renormalization is forced to fail and
    the solved record must survive verbatim.
    """
    ref = _load_module(_REFEREE_SRC, "probe_fed_msl_openems_referee")
    leg = _doc(_REFEREE)["stage2"]["legs"]["comparator_dx50um"]

    good = ref.attach_renormalization(json.loads(json.dumps(leg)))
    assert good["kurokawa_renormalization"]["n_bins"] == 48

    broken = json.loads(json.dumps(leg))
    broken["z0_msl_measured_ohm"][7] = [-1.0, 0.0]     # Re Z <= 0 at one bin
    solved_before = {k: v for k, v in broken.items()
                     if k != "kurokawa_renormalization"}
    out = ref.attach_renormalization(broken)          # must NOT raise

    assert out is broken
    rec = out["kurokawa_renormalization"]
    assert rec["status"].startswith("NOT COMPUTED")
    assert rec["error_type"] == "ConfigError"
    # every solved entry is byte-identical to what the solve produced
    assert {k: v for k, v in out.items()
            if k != "kurokawa_renormalization"} == solved_before


def test_the_offline_mode_has_its_own_output_default(tmp_path, monkeypatch):
    """A bare ``--renormalize-json`` must not overwrite the solver artifact.

    The two modes write differently shaped documents. They used to share
    ``--output``'s single default, so an offline report landed on
    ``.omx/probe-fed-msl-referee/referee.json`` -- the solver's path.
    """
    ref = _load_module(_REFEREE_SRC, "probe_fed_msl_openems_referee")
    assert ref.DEFAULT_OUTPUT_SOLVER != ref.DEFAULT_OUTPUT_RENORMALIZE

    monkeypatch.chdir(tmp_path)
    assert ref.main(["--renormalize-json", str(_REFEREE)]) == 0

    solver_default = tmp_path / ref.DEFAULT_OUTPUT_SOLVER
    offline_default = tmp_path / ref.DEFAULT_OUTPUT_RENORMALIZE
    assert not solver_default.exists()
    assert offline_default.exists()
    written = json.loads(offline_default.read_text())
    assert set(written["legs"]) == {"comparator_dx50um",
                                    "reported_only_dx80um"}


def test_the_committed_verbatim_balance_is_exactly_abs_s11_sq_plus_abs_s21_sq():
    """The gate's baseline and the artifact field are the SAME quantity.

    This is the check that stops a reader from comparing the 1.05 bar against a
    symmetrized surrogate (|S11|^2 + geomean^2) instead of the gate's formula.
    """
    leg = _doc(_REFEREE)["stage2"]["legs"]["comparator_dx50um"]
    recomputed = (np.abs(_cx(leg["s11"])) ** 2 + np.abs(_cx(leg["s21"])) ** 2)
    verbatim = np.asarray(leg["passivity_balance_verbatim"], dtype=float)
    assert float(np.max(np.abs(recomputed - verbatim))) == 0.0


def test_the_fix_does_not_restore_passivity_on_the_committed_comparator_leg():
    """THE HONEST NUMBER. Normalization was one defect; a second remains.

    If a future change makes this test fail by making the comparator PASS, that
    is a real result and the test must be updated deliberately, with the run
    that produced it named. It must never be updated to hide a regression.
    """
    ref = _load_module(_REFEREE_SRC, "probe_fed_msl_openems_referee")
    leg = _doc(_REFEREE)["stage2"]["legs"]["comparator_dx50um"]
    rec = ref.renormalized_leg_record(leg)

    assert rec["n_bins"] == 48
    lo, hi = rec["passivity_balance_renormalized_band"]
    assert lo == pytest.approx(1.0106, abs=5e-4)
    assert hi == pytest.approx(1.0579, abs=5e-4)
    assert rec["passivity_passed_renormalized"] is False
    assert rec["n_bins_balance_over_tol"] == 5
    assert rec["n_bins_abs_s21_over_unity"] == 48

    # The residual DRIFTS. Frequency dependence is the ground on which #498's
    # 2026-08-03 comment falsified a constant per-port rescale, and it is the
    # ONLY thing the two share: that factor was on rfx's own wave channel and
    # DECREASED 1.677 -> 1.605; this one is on the openEMS comparator, RISES
    # 1.0101 -> 1.0302 and is ~30x smaller. The asserted direction below is
    # this leg's own (rising above its minimum), not the 2026-08-03 curve's.
    dlo, dhi = rec["kurokawa_residual_drift_band"]
    assert dlo == pytest.approx(1.0101, abs=5e-4)
    assert dhi == pytest.approx(1.0302, abs=5e-4)
    assert rec["kurokawa_residual_drift_is_flat"] is False
    drift = np.asarray(rec["kurokawa_residual_drift"], dtype=float)
    assert np.all(np.diff(drift[int(np.argmin(drift)):]) > 0)

    # the committed verbatim record is never overwritten by the fix
    assert rec["passivity_balance_verbatim"] == leg["passivity_balance_verbatim"]


def test_the_same_fix_does_not_make_the_dx80um_leg_reciprocal_either():
    """A witness against 'normalization explains everything'."""
    ref = _load_module(_REFEREE_SRC, "probe_fed_msl_openems_referee")
    leg = _doc(_REFEREE)["stage2"]["legs"]["reported_only_dx80um"]
    rec = ref.renormalized_leg_record(leg)
    lo, hi = [float(x) for x in (min(rec["reciprocity_ratio_renormalized"]),
                                 max(rec["reciprocity_ratio_renormalized"]))]
    assert lo == pytest.approx(0.6535, abs=5e-4)
    assert hi == pytest.approx(0.8339, abs=5e-4)
    assert hi < 1.0   # never crosses unity: still not reciprocal


# ---------------------------------------------------------------------------
# The flux bracket: an instrument inconsistency, not a box width
# ---------------------------------------------------------------------------
def _flux(meas):
    lw = next(f for f in meas["flux"] if f["drive"] == "lw")
    ml = next(f for f in meas["flux"] if f["drive"] == "msl")
    g = lambda fl, k: np.asarray(fl[k + "_exact_f64"], dtype=float)
    return (g(lw, "box_net"), g(lw, "plane_msl"), g(lw, "plane_px_2p56mm"),
            g(lw, "plane_mx_1p44mm"), g(ml, "box_net"), g(ml, "plane_msl"),
            g(ml, "plane_px_2p56mm"), g(ml, "plane_mx_1p44mm"))


def test_r1_above_unity_is_non_physical_on_the_committed_run():
    """More power is dissipated in a SUBREGION than in the region containing it.

    MSL drive: the only source is at x = 5.52 mm, east of the MSL reference
    plane at 4.72 mm, so the net inward flux there is the total dissipation
    west of it. The lw flux box lies entirely west of that plane, so its own
    net inward flux -- the dissipation inside the box -- cannot exceed it.
    """
    box_lw, pm_lw, _, _, box_ml, pm_ml, _, _ = _flux(_doc(_MEAS))
    r0 = pm_lw / box_lw
    r1 = box_ml / pm_ml
    assert float(np.min(r1)) == pytest.approx(1.02484, abs=5e-5)
    assert float(np.max(r1)) == pytest.approx(1.02566, abs=5e-5)
    assert np.all(r1 > 1.0), "r1 <= 1 is the physical bound; it is violated"
    assert float(np.max(r1 / r0)) == pytest.approx(1.05074, abs=5e-5)


def test_the_bracket_is_negative_at_every_anchor_including_zero():
    meas = _doc(_MEAS)
    box_lw, pm_lw, _, _, box_ml, pm_ml, _, _ = _flux(meas)
    ratio = (box_ml / pm_ml) / (pm_lw / box_lw)
    n_f = len(meas["fixture"]["freqs_hz"])
    mc = meas["msl_channel"]
    v = _cx(np.asarray(mc["v0_msl"]).reshape(2, 1, n_f, 2))
    i = _cx(np.asarray(mc["i_msl"]).reshape(2, 1, n_f, 2))
    run = int(meas["refplane"]["msl_drive_run"])
    zc = float(np.real(_cx(meas["refplane"]["runs"][0]["zc"]))[0])
    s22 = np.abs((v[run, 0] - zc * i[run, 0]) / (v[run, 0] + zc * i[run, 0]))
    for s in (s22, np.asarray(mc["abs_S22_raw"], dtype=float),
              np.zeros(n_f)):
        assert np.all(1.0 - (1.0 - s ** 2) * ratio < 0.0)


def test_a_box_independent_three_plane_budget_keeps_the_bracket_negative():
    """Half the excess is the box and half is the planes -- neither is width.

    The run committed two FULL-cross-section x-planes (1.44 / 2.56 mm) that
    bound a slab containing the whole lw port and share no face with the box.
    Swapping the box out for that budget removes about half the log-excess in
    r1/r0 and the bracket is still negative at every bin.
    """
    meas = _doc(_MEAS)
    (box_lw, pm_lw, px_lw, mx_lw,
     box_ml, pm_ml, px_ml, mx_ml) = _flux(meas)
    q_lw = px_lw - mx_lw
    q_ml = (-px_ml) + mx_ml

    # the box reads high against the slab budget by the SAME factor on both
    # drives, and on the MSL drive the box is a subregion of that slab
    box_over_slab_lw = box_lw / q_lw
    box_over_slab_ml = np.abs(box_ml) / q_ml
    assert np.all(box_over_slab_ml > 1.0)
    assert float(np.max(np.abs(box_over_slab_lw - box_over_slab_ml))) < 1e-3

    ratio_box = (box_ml / pm_ml) / (pm_lw / box_lw)
    ratio_planes = (q_ml / np.abs(pm_ml)) / (pm_lw / q_lw)
    assert np.all(ratio_planes > 1.0)
    frac = 1.0 - np.log(ratio_planes) / np.log(ratio_box)
    assert np.all((frac > 0.50) & (frac < 0.54))

    n_f = len(meas["fixture"]["freqs_hz"])
    mc = meas["msl_channel"]
    v = _cx(np.asarray(mc["v0_msl"]).reshape(2, 1, n_f, 2))
    i = _cx(np.asarray(mc["i_msl"]).reshape(2, 1, n_f, 2))
    run = int(meas["refplane"]["msl_drive_run"])
    zc = float(np.real(_cx(meas["refplane"]["runs"][0]["zc"]))[0])
    s22 = np.abs((v[run, 0] - zc * i[run, 0]) / (v[run, 0] + zc * i[run, 0]))
    assert np.all(1.0 - (1.0 - s22 ** 2) * ratio_planes < 0.0)


def test_the_required_instrument_bias_is_about_two_percent_either_way():
    """The competing hypothesis the box plan omitted, quantified.

    r1/r0 = P_box(msl)*P_box(lw) / (P_plane(msl)*P_plane(lw)), so a common
    factor m on plane_msl scales it by 1/m^2. The m that flips the bracket and
    the m that makes r1 physical agree to well under a percent -- i.e. one
    instrument bias explains both, with no box change at all.
    """
    meas = _doc(_MEAS)
    box_lw, pm_lw, _, _, box_ml, pm_ml, _, _ = _flux(meas)
    r0, r1 = pm_lw / box_lw, box_ml / pm_ml
    n_f = len(meas["fixture"]["freqs_hz"])
    mc = meas["msl_channel"]
    v = _cx(np.asarray(mc["v0_msl"]).reshape(2, 1, n_f, 2))
    i = _cx(np.asarray(mc["i_msl"]).reshape(2, 1, n_f, 2))
    run = int(meas["refplane"]["msl_drive_run"])
    zc = float(np.real(_cx(meas["refplane"]["runs"][0]["zc"]))[0])
    s22 = np.abs((v[run, 0] - zc * i[run, 0]) / (v[run, 0] + zc * i[run, 0]))

    m_bracket = float(np.max(np.sqrt((r1 / r0) * (1.0 - s22 ** 2))))
    m_physical = float(np.max(r1))
    assert m_bracket == pytest.approx(1.0220, abs=5e-4)
    assert m_physical == pytest.approx(1.0257, abs=5e-4)
    assert abs(m_physical - m_bracket) < 0.005


def test_widening_the_box_has_the_wrong_sign_for_both_readings():
    """The box factor that would rescue either reading is BELOW one, measured.

    Two separate claims, and only the second is evidence:

    1. ALGEBRA (holds for any positive data, asserts nothing about this run).
       Scaling ``box_net`` by ``g`` on both drives sends r0 = plane/box to
       r0/g and r1 = box/plane to g*r1, so r1/r0 scales by g**2 and the
       bracket 1 - (1-|S22|^2)*r1/r0 falls monotonically in g. A wider box
       captures more, i.e. g > 1, so widening pushes both readings the wrong
       way -- PROVIDED the required g is not already above 1.

    2. THE COMMITTED MEASUREMENT, which is what makes (1) bite. Solve the
       algebra for the g that would rescue each reading:
         * bracket >= 0 needs g <= 1/sqrt((1-|S22|^2) * r1/r0);
         * r1 <= 1  needs g <= 1/r1.
       Both are computed here from the ``exact_f64`` faces and both come out
       BELOW one (0.9785 and 0.9750). A wider box moves g the other way, so
       the proposed run cannot produce the reading it would be spent on.

    Assertion 2 fails on perturbed data -- e.g. plane_msl x1.03, or box_net
    x0.97, either of which lifts a required factor above 1.
    """
    meas = _doc(_MEAS)
    box_lw, pm_lw, _, _, box_ml, pm_ml, _, _ = _flux(meas)
    n_f = len(meas["fixture"]["freqs_hz"])
    mc = meas["msl_channel"]
    v = _cx(np.asarray(mc["v0_msl"]).reshape(2, 1, n_f, 2))
    i = _cx(np.asarray(mc["i_msl"]).reshape(2, 1, n_f, 2))
    run = int(meas["refplane"]["msl_drive_run"])
    zc = float(np.real(_cx(meas["refplane"]["runs"][0]["zc"]))[0])
    s22 = np.abs((v[run, 0] - zc * i[run, 0]) / (v[run, 0] + zc * i[run, 0]))

    r0, r1 = pm_lw / box_lw, box_ml / pm_ml
    base = r1 / r0

    # (2) THE DATA-BOUND HALF. The box factor that would rescue each reading.
    g_bracket = float(np.min(1.0 / np.sqrt((1.0 - s22 ** 2) * base)))
    g_physical = float(np.min(1.0 / r1))
    assert g_bracket == pytest.approx(0.97849, abs=5e-5)
    assert g_physical == pytest.approx(0.97498, abs=5e-5)
    # BELOW one at every bin: only a box that reads LESS could close either,
    # and a wider box reads MORE. This is the sign claim, and it is measured.
    assert g_bracket < 1.0 and g_physical < 1.0
    assert np.all(1.0 / np.sqrt((1.0 - s22 ** 2) * base) < 1.0)
    assert np.all(1.0 / r1 < 1.0)

    # (1) THE ALGEBRA, stated so the reader can check the direction. Combined
    # with g_bracket < 1 above, this is what closes the branch.
    for gain in (1.005, 1.01, 1.02, 1.05):      # a strictly wider box
        wider = ((box_ml * gain) / pm_ml) / (pm_lw / (box_lw * gain))
        assert np.all(wider > base)
        assert np.all(1.0 - (1.0 - s22 ** 2) * wider
                      < 1.0 - (1.0 - s22 ** 2) * base)


# ---------------------------------------------------------------------------
# The anchor sweep reports a BOUND, never a pinned value
# ---------------------------------------------------------------------------
def test_the_anchor_sweep_bounds_the_anchor_and_does_not_pin_it():
    """57.463 ohm reproduces M2 comparably to 57.925 -- ~57-58 ohm, no tighter.

    The two anchors are PINNED here. Without that the "4.88 % vs 3.60 %"
    comparison is between two unnamed numbers: collapsing the realized-cell
    anchor onto the measured Zc would make the deviations identical and every
    other assertion in this test would still pass.
    """
    mod = _load_module(_REANALYSIS_SRC, "mixed_anchor_reanalysis_498")
    meas = _doc(_MEAS)
    sec = mod.section1_anchor(meas)
    sweep = sec["anchor_sweep"]

    # the two anchors being compared are DISTINCT and are these values
    z_cell = sweep["hj_realized_cell_560x320um"]["z_ohm"]
    z_meas = sweep["zc_measured_two_plane"]["z_ohm"]
    assert z_cell == pytest.approx(57.463, abs=5e-4)
    assert z_meas == pytest.approx(57.9252, abs=5e-4)
    assert abs(z_meas - z_cell) > 0.4     # not the same anchor twice
    assert sweep["hj_declared_600x254um"]["z_ohm"] == pytest.approx(
        47.8948, abs=5e-4)
    assert sweep["hj_realized_node_480x320um"]["z_ohm"] == pytest.approx(
        62.652, abs=5e-4)

    d_meas = sweep["zc_measured_two_plane"]["max_dev_vs_M2_pct"]
    d_cell = sweep["hj_realized_cell_560x320um"]["max_dev_vs_M2_pct"]
    d_decl = sweep["hj_declared_600x254um"]["max_dev_vs_M2_pct"]
    d_node = sweep["hj_realized_node_480x320um"]["max_dev_vs_M2_pct"]
    assert d_meas == pytest.approx(3.6020, abs=5e-4)
    assert d_cell == pytest.approx(4.8829, abs=5e-4)
    assert d_meas < 5.0 and d_cell < 5.0          # both inside the band
    assert abs(d_cell - d_meas) < 2.0             # NOT discriminated
    assert d_decl > 50.0 and d_node > 20.0        # both outside it
    assert "ANCHOR-CIRCULAR" in sec["reading"]
    assert "vindicat" not in json.dumps(sec).lower()   # §10 item 11


def test_the_word_vindicated_appears_only_inside_the_do_not_pin_fence():
    """§10 item 11, over the WHOLE committed artifact, not one section.

    The fence text itself is item 11 ("F2's consistent branch must not be
    written up as vindicated"), so the artifact does contain the string once,
    in ``do_not_pin``. Everywhere else -- all four sections, the headline, the
    source block -- it must be absent, and that is the half that can fail.
    """
    doc = _doc(_REANALYSIS_JSON)
    fence = json.dumps(doc["do_not_pin"], ensure_ascii=False).lower()
    assert fence.count("vindicat") == 1        # the fence is still there
    rest = json.dumps({k: v for k, v in doc.items() if k != "do_not_pin"},
                      ensure_ascii=False).lower()
    assert "vindicat" not in rest
    # and the fence is the predeclaration's own item 11, not a paraphrase
    assert "must not be written up as 'vindicated'" in fence


def test_the_591_ohm_premise_does_not_reproduce():
    mod = _load_module(_REANALYSIS_SRC, "mixed_anchor_reanalysis_498")
    sec = mod.section1_anchor(_doc(_MEAS))
    zin = np.asarray(sec["zin_msl_probe0_ohm"], dtype=float)
    re = zin[:, 0]
    assert 40.0 < float(np.min(re)) and float(np.max(re)) < 55.0
    assert float(np.max(np.asarray(sec["zin_im_over_re_pct"]))) < 5.0

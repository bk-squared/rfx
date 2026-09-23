"""Replay of the NU Lane 2b witness (no FDTD).

Frozen rules: docs/design_notes/20260915_nu_lane2b_predeclaration.md.
Every verdict is re-derived from the stored loss ladders with the SAME judges
the instrument used (AD-Q's fit_order / fd_budget with the recorded noise
floor, then the lane-2b lower-side order rule) and must equal the recorded
one -- FIRED and INCONCLUSIVE included; they are recorded results.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from validation.research.multiband_nu import adq_designvar as adq
from validation.research.multiband_nu import e6_inplane_designvar as e6
from validation.research.multiband_nu import e7_lane2b as e7

RESULTS = Path(__file__).resolve().parents[3] / "validation/research/multiband_nu/results"
ARMS = ("y", "pos", "zsmooth")
NAMES = {"y": ("w", "y_c"), "pos": ("w", "x_c"), "zsmooth": adq.NAMES[:4]}
SCALES = {"y": (e6.W0, e6.W0), "pos": (e6.W0, e6.W0), "zsmooth": tuple(float(s) for s in adq.P0[:4])}


def _load(arm):
    p = RESULTS / f"e7_{arm}.json"
    d = json.loads(p.read_text())
    assert d["arm"] == arm
    return d


def test_missing_required_artifact_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "RESULTS", tmp_path)
    with pytest.raises(FileNotFoundError):
        try:
            _load("y")
        except pytest.skip.Exception:
            pytest.fail("A completed witness must not skip missing evidence")


def test_main_refuses_existing_evidence_before_measurement(tmp_path, monkeypatch):
    out = tmp_path / "e7_y.json"
    original = '{"retained": true}\n'
    out.write_text(original)
    monkeypatch.setattr(e7, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["e7_lane2b", "--arm", "y"])

    def forbidden_measurement():
        pytest.fail("Existing evidence must be refused before measurement")

    monkeypatch.setattr(e7, "arm_y", forbidden_measurement)
    with pytest.raises(FileExistsError, match="e7_y.json"):
        e7.main()
    assert out.read_text() == original
    assert not out.with_suffix(".started").exists()


def test_main_writes_supplement_without_replacing_frozen_evidence(tmp_path, monkeypatch):
    frozen = tmp_path / "e7_y.json"
    original = '{"retained": true}\n'
    frozen.write_text(original)
    out = tmp_path / "supplement.json"
    monkeypatch.setattr(e7, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["e7_lane2b", "--arm", "y", "--out", str(out)])
    monkeypatch.setattr(e7, "arm_y", lambda: {"arm": "y", "sentinel": 17})
    e7.main()
    assert json.loads(out.read_text())["sentinel"] == 17
    assert frozen.read_text() == original


def test_floor_retains_the_actual_loss_samples():
    def loss(q):
        return jnp.float32(0.1306) + 0.37 * q[0] + 2.9 * q[0] ** 2

    x, v = np.zeros(1), np.ones(1)
    expected = [float(loss(jnp.asarray(x + h * v, jnp.float32))) for h in e7.FLOOR_HS]
    result = e7.judge_control(loss, x, v, 0.37, 1.0, float(loss(x)), True)
    floor = result["floor"]
    assert floor["hs"] == list(e7.FLOOR_HS)
    assert floor["losses"] == expected
    hs = np.asarray(e7.FLOOR_HS)
    residual = np.asarray(expected) - np.polyval(np.polyfit(hs, expected, 2), hs)
    np.testing.assert_array_equal(floor["quadratic_residuals"], residual)
    assert floor["sigma"] == pytest.approx(np.linalg.norm(residual) / np.sqrt(len(hs) - 3))


def test_revert_retains_gradient_and_judge_inputs():
    def loss_cells(cells, stop_dt=False):
        q = cells[0] - 1.0
        dt_term = jax.lax.stop_gradient(q) if stop_dt else q
        # The cubic gives central FD a nonzero h^2 truncation term; without
        # it Richardson's validity check correctly reports INCONCLUSIVE.
        return jnp.float32(0.1306) + 0.37 * q + 2.9 * q ** 2 + 0.5 * q ** 3 + 0.05 * dt_term

    result = e7.controls_arm(
        "synthetic", {"l1": (loss_cells, 1)}, lambda q: q,
        np.eye(1), np.ones(1), ("w",), (1.0,), np.array([0]),
        use_floor=True, revert_control="w",
    )
    rec = result["losses"]["l1"]
    rv = rec["revert"]
    np.testing.assert_allclose(rec["g_cell"], [0.42], rtol=1e-6)
    np.testing.assert_allclose(rv["g_cell"], [0.37], rtol=1e-6)
    assert rv["ad_relative"] == pytest.approx(0.37, rel=1e-6)
    assert rv["forward_identical"] is True
    assert [p["h"] for p in rv["points"]] == list(adq.HS)
    assert rv["floor"]["hs"] == list(e7.FLOOR_HS)
    order = adq.fit_order(rv["points"], rec["loss0"], rv["ad_relative"], sigma=rv["floor"]["sigma"])
    fd = adq.fd_budget(rv["points"], rv["ad_relative"], 1.0, sigma=rv["floor"]["sigma"])
    assert order["verdict"] == rv["order"]["verdict"]
    assert fd["verdict"] == rv["fd"]["verdict"] == rv["fd_verdict"] == "FIRED"


def _jacobian(arm):
    """d cells / d control, rebuilt from the same fixed-topology maps the arm used."""
    if arm == "y":
        return np.asarray(e7.MAP_Y.jacobian_seg() @ e6.A, float)
    if arm == "pos":
        return np.asarray(e6.J_PARAM, float)
    p0 = jnp.asarray(adq.P0, jnp.float32)
    return np.asarray(jax.jacfwd(adq.stack_cells)(p0), float)[:, :4]


def _tied(arm):
    if arm == "y":
        return np.flatnonzero(e7.MAP_Y.cells0 == e7.MAP_Y.cells0.min())
    if arm == "pos":
        return np.flatnonzero(e6.MAP.cells0 == e6.MAP.cells0.min())
    c0 = np.asarray(adq.stack_cells(jnp.asarray(adq.P0, jnp.float32)), float)[:64]
    return np.flatnonzero(c0 == c0.min())


def _points(r):
    return [{"h": p["h"], "loss_plus": p["loss_plus"], "loss_minus": p["loss_minus"]} for p in r["points"]]


def test_frozen_constants():
    assert e7.R0_BAND == (0.9, 1.1) and e7.R1_MIN == 1.8 and e7.CANCEL_MIN == 0.5
    assert e7.HS is adq.HS
    assert len(e7.FLOOR_HS) == 64 and e7.FLOOR_HS[0] == 1e-5 and e7.FLOOR_HS[-1] == 1e-4
    assert (e7.Z_COMB_DF, e7.Z_COMB_K, e7.Z_TW_FRACTION) == (0.25e9, 10, 0.8)
    assert len(e7.COMB_Z) == 21 and abs(e7.COMB_Z[10] - e7.F_NOM_Z) == 0.0


def test_lane_rule_is_lower_side_only():
    """The lane-2b order rule keeps AD-Q's lower edges and drops its upper R1
    edge (2.2): a wrong gradient drives R1 toward 1, never above 2."""
    v = e7.lane_order_verdict
    assert v({"R0": {"slope": 1.0}, "R1": {"slope": 2.5}}) == "HELD"
    assert v({"R0": {"slope": 1.0}, "R1": {"slope": 1.79}}) == "FIRED"
    assert v({"R0": {"slope": 0.89}, "R1": {"slope": 2.0}}) == "FIRED"
    assert v({"R0": {"slope": 1.11}, "R1": {"slope": 2.0}}) == "FIRED"
    assert v({"R0": {"slope": 1.0}}) == "INCONCLUSIVE"


@pytest.mark.parametrize("arm", ARMS)
def test_selfcheck_recorded_before_arm(arm):
    d = _load(arm)
    assert d["selfcheck"]["all_pass"] is True
    assert d["provenance"]["rfx_file"].endswith("rfx/__init__.py")


@pytest.mark.parametrize("arm", ARMS)
def test_induced_directions_constant_on_tied_set(arm):
    jac, tied = _jacobian(arm), _tied(arm)
    assert len(tied) >= 1
    for i, nm in enumerate(NAMES[arm]):
        assert np.ptp(jac[tied, i]) == 0.0, nm


@pytest.mark.parametrize("arm", ARMS)
def test_verdicts_replay(arm):
    d = _load(arm)
    for label, rec in d["losses"].items():
        loss0 = rec["loss0"]
        assert abs(rec["loss_ulp"] - adq.ulp(loss0)) == 0.0
        for i, nm in enumerate(NAMES[arm]):
            r = rec["controls"][nm]
            pts = _points(r)
            assert [p["h"] for p in pts] == list(adq.HS), (label, nm)
            sigma = r["floor"]["sigma"]
            # `reliable` is a numpy bool serialized through float -> 1.0 in the stored JSON
            assert r["floor"]["reliable"] == 1 and abs(r["floor"]["sigma_in_ulp"] - sigma / adq.ulp(loss0)) <= 1e-9
            order = adq.fit_order(pts, loss0, r["ad_relative"], sigma=sigma)
            fd = adq.fd_budget(pts, r["ad_relative"], SCALES[arm][i], sigma=sigma)
            assert order["verdict"] == r["order"]["verdict"], (label, nm)
            assert fd["verdict"] == r["fd"]["verdict"], (label, nm)
            for lab in ("R0", "R1"):
                if lab in r["order"]:
                    assert abs(order[lab]["slope"] - r["order"][lab]["slope"]) <= 1e-12, (label, nm, lab)
            lane = e7.lane_order_verdict(order)
            if r["lane_order_verdict"] == "INCONCLUSIVE (precheck)":
                assert d["precheck"][label]["cancellation_ratio"] < e7.CANCEL_MIN
            else:
                assert lane == r["lane_order_verdict"], (label, nm)


@pytest.mark.parametrize("arm", ARMS)
def test_chain_rule_and_coverage_replay(arm):
    d, jac = _load(arm), _jacobian(arm)
    names = NAMES[arm]
    for label, rec in d["losses"].items():
        g = np.asarray(rec["g_cell"], float)
        np.testing.assert_allclose(jac.T @ g, rec["g_x_chain"], rtol=1e-12, atol=0)
        assert abs(adq.coverage(g, jac)["fraction"] - rec["coverage"]["fraction"]) <= 1e-12, label
        held = [i for i, nm in enumerate(names)
                if rec["controls"][nm]["lane_order_verdict"] == "HELD" and rec["controls"][nm]["fd"]["verdict"] == "HELD"]
        if held:
            assert abs(adq.coverage(g, jac[:, held])["fraction"] - rec["coverage_verified"]["fraction"]) <= 1e-12, label
        else:
            assert rec["coverage_verified"] is None


@pytest.mark.parametrize("arm", ("y", "zsmooth"))
def test_revert_proof_has_power_against_the_dt_path(arm):
    """Same judges with stop_gradient(dt): the declared control must FIRE on
    order or FD while the forward value stays bit-identical."""
    d = _load(arm)
    for label, rec in d["losses"].items():
        rv = rec["revert"]
        assert rv["forward_identical"] is True, label
        assert rv["lane_order_verdict"] == "FIRED" or rv["fd_verdict"] == "FIRED", label
        # zsmooth's share is negative (dt path and geometry path of opposite sign)
        assert rv["dt_share"] is not None and abs(rv["dt_share"]) > 0.05, label


def test_y_width_gradient_verified_on_second_inplane_axis():
    """Recorded: the in-plane width gradient holds on y as it did on x (E6),
    dt path included; y_c is INCONCLUSIVE for the fixture-symmetry reason
    that the pos arm removes."""
    d = _load("y")
    for label in ("l1", "l2s"):
        w = d["losses"][label]["controls"]["w"]
        assert w["lane_order_verdict"] == "HELD" and w["fd"]["verdict"] == "HELD", label
        assert 1.9 <= w["order"]["R1"]["slope"] <= 2.0, label
        assert d["losses"][label]["controls"]["y_c"]["lane_order_verdict"] == "INCONCLUSIVE", label
        assert d["losses"][label]["revert"]["dt_share"] > 0.5, label


def test_pos_precheck_replay():
    """Eligibility (declared before the gradient): x_c must not be a lead/tail
    cancellation. Re-derived from the stored cell gradient and the map."""
    d, jac = _load("pos"), _jacobian("pos")
    lead, tail = e6.MAP.seg_id == 0, e6.MAP.seg_id == 2
    assert d["fixture"]["cancel_min"] == e7.CANCEL_MIN
    for label, rec in d["losses"].items():
        g = np.asarray(rec["g_cell"], float)
        gl, gt = float(jac[lead, 1] @ g[lead]), float(jac[tail, 1] @ g[tail])
        ratio = abs(gl + gt) / (abs(gl) + abs(gt))
        pre = d["precheck"][label]
        np.testing.assert_allclose([gl, gt, ratio], [pre["g_lead"], pre["g_tail"], pre["cancellation_ratio"]], rtol=1e-12, atol=0)
        assert ratio >= e7.CANCEL_MIN, label


def test_pos_position_gradient_order_verified():
    """Recorded: the first order-verified in-plane band POSITION gradient
    (E6's x_c was INCONCLUSIVE for the fixture's symmetry). AD-Q's judge, with
    its upper R1 edge 2.2, would have fired L1 x_c on that upper side (4-point
    window); the lane rule dropped the edge before any arm ran."""
    d = _load("pos")
    for label in ("l1", "l2s"):
        r = d["losses"][label]["controls"]["x_c"]
        assert r["lane_order_verdict"] == "HELD" and r["fd"]["verdict"] == "HELD", label
        assert r["order"]["R1"]["slope"] >= e7.R1_MIN, label
    l1 = d["losses"]["l1"]["controls"]["x_c"]["order"]
    assert l1["verdict"] == "FIRED" and l1["R1"]["slope"] > 2.2 and l1["points"] == 4
    l2 = d["losses"]["l2s"]["controls"]["x_c"]["order"]
    assert l2["verdict"] == "HELD" and l2["points"] == 5


def test_zsmooth_thin_thickness_verified_on_smooth_spectral_loss():
    """Recorded: AD-Q's "L2 thickness NOT verified" is closed on the Hann+comb
    observable -- h_thin order HELD with FD inside its bar on the 8000-step z
    stack (floor well under AD-Q's 14-44 ulp), and the dt revert-proof fires."""
    d = _load("zsmooth")
    rec = d["losses"]["l2s"]
    t = rec["controls"]["h_thin"]
    assert t["lane_order_verdict"] == "HELD" and t["fd"]["verdict"] == "HELD"
    assert 1.9 <= t["order"]["R1"]["slope"] <= 2.1 and t["order"]["points"] >= 6
    assert t["floor"]["sigma_in_ulp"] < 14.0
    for nm in ("h_core_left", "h_air"):
        assert rec["controls"][nm]["lane_order_verdict"] == "HELD", nm
    assert rec["revert"]["control"] == "h_thin" and rec["revert"]["order"]["R1"]["slope"] < 1.3
    assert d["fixture"]["n2"] == 8000 and len(d["fixture"]["comb_hz"]) == 21


def test_zsmooth_core_right_fire_is_the_one_sided_cubic_asymmetry():
    """Recorded FIRED (h_core_right R1 1.721), re-derived from the ladders: the
    direction is the exact negative of h_core_left's; the two ladders agree to
    a few ulp; the one-sided R1 lands on opposite sides of 2 at the two ends of
    the line while the even remainder (cubic term cancelled) fits ~2 for every
    control; and the central remainder falls as h^3, which bounds any gradient
    error along the line. A judge property, recorded -- not a rule change."""
    d = _load("zsmooth")
    rec = d["losses"]["l2s"]
    loss0, u = rec["loss0"], adq.ulp(rec["loss0"])
    L, R = rec["controls"]["h_core_left"], rec["controls"]["h_core_right"]
    assert np.array_equal(adq.A[:, 2], -adq.A[:, 0]) and L["ad_relative"] == -R["ad_relative"]
    assert R["lane_order_verdict"] == "FIRED" and R["fd"]["verdict"] == "HELD" and R["order"]["R1"]["slope"] < e7.R1_MIN
    assert L["lane_order_verdict"] == "HELD" and L["order"]["R1"]["slope"] > 2.1
    for pl, pr in zip(L["points"], R["points"]):
        assert abs(pr["loss_plus"] - pl["loss_minus"]) <= 16 * u and abs(pr["loss_minus"] - pl["loss_plus"]) <= 16 * u
    idx = L["order"]["indices"]
    assert idx == R["order"]["indices"] and len(idx) == 7

    def slope(hs, y):
        return float(np.polyfit(np.log(hs), np.log(y), 1)[0])

    for nm, r in rec["controls"].items():
        pts, g, ii = r["points"], r["ad_relative"], r["order"]["indices"]
        hs = np.array([pts[i]["h"] for i in ii])
        plus = [abs(pts[i]["loss_plus"] - loss0 - pts[i]["h"] * g) for i in ii]
        minus = [abs(pts[i]["loss_minus"] - loss0 + pts[i]["h"] * g) for i in ii]
        even = [abs(0.5 * (pts[i]["loss_plus"] + pts[i]["loss_minus"]) - loss0) for i in ii]
        assert abs(slope(hs, plus) - r["order"]["R1"]["slope"]) <= 1e-12, nm   # R1 is the one-sided plus remainder
        assert abs(slope(hs, even) - 2.0) <= 0.05, nm
        if nm == "h_core_left":
            assert slope(hs, minus) < e7.R1_MIN and abs(slope(hs, minus) - R["order"]["R1"]["slope"]) < 0.02
        # central remainder |L+ - L- - 2 h g.v|: h^3 above 32 sigma, no linear floor
        h = np.array([p["h"] for p in pts])
        cen = np.array([abs(p["loss_plus"] - p["loss_minus"] - 2 * p["h"] * g) for p in pts])
        ok = np.flatnonzero(cen > 32 * r["floor"]["sigma"])
        assert len(ok) >= 3 and slope(h[ok], cen[ok]) > 2.6, nm
        assert cen[ok[0]] / (2 * h[ok[0]]) / abs(g) < 1e-3, nm   # |delta| / |g.v| bound


def test_pos_l1_w_fire_is_on_the_r0_edge():
    """Recorded FIRED, not a pass: w on L1 fires the R0 lower edge alone
    (R0 0.8997 vs 0.9) with R1 2.025 and FD HELD. R0 is the first-difference
    sanity slope; a wrong gradient does not move it. Pinned so a change in
    either is noticed."""
    r = _load("pos")["losses"]["l1"]["controls"]["w"]
    assert r["lane_order_verdict"] == "FIRED" and r["fd"]["verdict"] == "HELD"
    assert 0.89 <= r["order"]["R0"]["slope"] < e7.R0_BAND[0]
    assert r["order"]["R1"]["slope"] >= 2.0
    assert _load("pos")["losses"]["l2s"]["controls"]["w"]["lane_order_verdict"] == "HELD"

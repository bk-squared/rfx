"""E6 replay — in-plane design-variable AD (Lane 2), no FDTD.

Frozen rules: docs/design_notes/20260913_nu_lane2_inplane_designvar_ad_predeclaration.md.
Every verdict is re-derived from the stored loss ladders with the SAME judges
the instrument used (``adq_designvar.fit_order`` / ``fd_budget`` verbatim, the
stored ``sigma_eff``), and must equal the recorded one -- including FIRED and
INCONCLUSIVE ones, which are recorded results, not failures to hide. The map
checks m1-m4 and m6 (analytic side) are re-derived live from the builder (host
arithmetic only); m5 / m8 from the stored numbers. Files not present are
skipped, so the test is meaningful at every commit of the lane.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.research.multiband_nu import adq_designvar as adq
from validation.research.multiband_nu import e6_inplane_designvar as e6

RESULTS = Path(__file__).resolve().parents[3] / "validation/research/multiband_nu/results"


def _load(arm):
    p = RESULTS / f"e6_{arm}.json"
    if not p.exists():
        pytest.skip(f"e6 arm {arm} not measured yet")
    d = json.loads(p.read_text())
    assert "instrument_error" not in d, d.get("instrument_error")
    return d


def _points(steps):
    return [{"h": s["h"], "loss_plus": s["loss_plus"], "loss_minus": s["loss_minus"]} for s in steps]


def test_frozen_constants():
    assert e6.HS == tuple(2.0 ** -k for k in range(17, 2, -1))
    assert e6.N_QUANTA == 32 and e6.RHO_BAND == (2.0, 8.0)
    assert e6.X_EDGES == (0.0, 8e-3, 10e-3, 18e-3) and e6.X_SIZES == (0.5e-3, 0.1e-3, 0.5e-3)
    assert e6.CAP == 1.3 and e6.PIN == 0.5e-3 and (e6.NY, e6.NZ) == (12, 24)
    assert (e6.N1, e6.N2, e6.CHECKPOINT_EVERY) == (600, 2400, 100)
    assert e6.T_W_FRACTION == 0.8 and e6.COMB_DF == 0.5e9 and e6.COMB_K == 10
    assert e6.F0 == 20e9 and e6.SIGMA_T == 20e-12 and e6.T0 == 5 * e6.SIGMA_T
    np.testing.assert_array_equal(e6.P0, [2e-3, 9e-3])
    np.testing.assert_array_equal(e6.A, [[-0.5, 1.0], [1.0, 0.0], [-0.5, -1.0]])
    assert e6.LADDER_SCALE == 2e-3
    assert (e6.M_LEN_TOL, e6.M_IFACE_TOL, e6.M_JAC_REL, e6.M_DT_REL, e6.M_STRIP_NODES, e6.M_FWD_REL) == (
        1e-9, 1e-9, 1e-5, 1e-6, 21, 1e-6)


def test_map_is_the_builder_output_and_fixed_topology():
    """The declared fixture: 60 cells, 20/20/20, interfaces at nodes 20/40/60,
    tied set = the band, pinned ends; the map at delta = 0 is cells0 exactly."""
    m = e6.MAP
    assert m.cells0.size == 60 and m.counts.tolist() == [20, 20, 20]
    assert m.edge_nodes.tolist() == [0, 20, 40, 60] and (e6.I_L, e6.I_R) == (20, 40)
    assert m.pinned.tolist() == [True] + [False] * 58 + [True]
    assert np.flatnonzero(m.cells0 == m.cells0.min()).tolist() == list(range(20, 40))
    np.testing.assert_allclose(m.free_len0, [7.5e-3, 2e-3, 7.5e-3], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(e6.cells_of(np.zeros(2, np.float32))),
                                  np.asarray(e6.CELLS0_F32))
    assert e6.PRB_IJK == (43, 6, 14) and e6.SRC_IJK == (8, 6, 6) and e6.GRID_SHAPE == (61, 13, 25)


def test_map_checks_m1_m4_live():
    """m1-m4 re-derived from the builder at every ladder point (host arithmetic)."""
    rows = e6.ladder_geometry()
    assert len(rows) == 2 * 2 * len(e6.HS)
    assert max(r["length_err_m"] for r in rows) <= e6.M_LEN_TOL
    assert all(r["tied_is_band"] for r in rows)
    tied = np.flatnonzero(e6.MAP.cells0 == e6.MAP.cells0.min())
    assert [float(np.ptp(e6.J_PARAM[tied, i])) for i in range(2)] == [0.0, 0.0]
    assert np.all(e6.J_PARAM[e6.MAP.pinned] == 0.0)
    for i in range(2):
        v = np.eye(2)[i] * e6.LADDER_SCALE
        for h in e6.HS:
            for sign in (1, -1):
                d = e6.cells_f64_of(sign * h * v)
                nodes = np.concatenate([[0.0], np.cumsum(d)])
                edges = np.concatenate([[0.0], np.cumsum(e6.SEG_LEN0 + e6.A @ (sign * h * v))])
                assert np.max(np.abs(nodes[e6.MAP.edge_nodes] - edges)) <= e6.M_IFACE_TOL


def test_strip_masks_index_attached():
    mx, my, mz = (np.asarray(m) for m in e6.MASKS)
    assert my[:, 6, e6.K_S].sum() == 21 and mx[:, 6, e6.K_S].sum() == 20 and mz.sum() == 0
    assert my.sum() == 21 * 13 and mx.sum() == 20 * 13
    assert not my[:, :, :e6.K_S].any() and not my[:, :, e6.K_S + 1:].any()


def test_model_json_matches_live_model():
    d = _load("model")
    assert d["selfcheck"]["all_pass"] is True
    np.testing.assert_array_equal(d["cells0"], e6.MAP.cells0)
    np.testing.assert_array_equal(d["jacobian_param_f64"], e6.J_PARAM)
    assert d["tied_cells_x"] == list(range(20, 40)) and d["tie_spread"] == [0.0, 0.0]
    assert d["dt0"] == e6.DT0 and d["t_w"] == e6.T_W
    assert abs(d["ddt_dw_analytic"] - e6.ddt_dw_analytic()) <= 1e-12 * e6.ddt_dw_analytic()
    assert d["strip_nodes"] == 21
    rows = e6.ladder_geometry()
    for a, b in zip(d["ladder"], rows):
        assert a["seam_ratio_lo"] == b["seam_ratio_lo"] and a["tied_set"] == b["tied_set"]
        assert a["dt"] == b["dt"]
    assert min(r["n2_dt_over_tw"] for r in rows) > 1.0


def test_map_checks_arm_replay():
    d = _load("map_checks")
    assert d["selfcheck"]["all_pass"] is True
    for k in range(1, 9):
        assert isinstance(d[f"m{k}_pass"], bool)
    assert d["all_pass"] == all(d[f"m{k}_pass"] for k in range(1, 9))
    # m5 / m6 / m8 re-derived from the stored numbers with the stored thresholds
    chain = e6.J_PARAM.T @ np.asarray(d["m5_g_cell"])
    np.testing.assert_allclose(chain, d["m5_chain_rule"], rtol=1e-12, atol=0)
    rel = [abs(c - g) / abs(g) for c, g in zip(chain, d["m5_g_param"])]
    assert d["m5_pass"] == all(r <= e6.M_JAC_REL for r in rel)
    assert d["m6_pass"] == (abs(d["m6_ddt_dp_ad"][0] - e6.ddt_dw_analytic()) / e6.ddt_dw_analytic() <= e6.M_DT_REL
                            and d["m6_ddt_dp_ad"][1] == 0.0)
    assert d["m8_pass"] == (abs(d["m8_loss_traced"] - d["m8_loss_concrete"]) / abs(d["m8_loss_concrete"]) <= e6.M_FWD_REL)
    assert d["m7_strip_nodes"] == 21


@pytest.mark.parametrize("arm", ["l1", "l2s"])
def test_arm_verdicts_replay(arm):
    d = _load(arm)
    assert d["selfcheck"]["all_pass"] is True
    loss0 = d["loss0"]
    for name, r in d["directions"].items():
        pts = _points(r["fd"]["steps"])
        assert [p["h"] for p in pts] == list(adq.HS)
        f = r["floor"]
        hs = np.asarray(f["floor_hs"])
        ys = np.asarray(f["floor_losses"])
        res2 = ys - np.polyval(np.polyfit(hs, ys, 2), hs)
        assert abs(np.sqrt(np.sum(res2 ** 2) / (len(hs) - 3)) - f["sigma"]) <= 1e-9 * max(f["sigma"], 1e-300)
        assert r["sigma_eff"] == max(adq.ulp(loss0), f["sigma"])
        assert r["ad_relative"] == r["ad_physical"] * e6.LADDER_SCALE
        order = adq.fit_order(pts, loss0, r["ad_relative"], sigma=r["sigma_eff"])
        fd = adq.fd_budget(pts, r["ad_relative"], e6.LADDER_SCALE, sigma=r["sigma_eff"])
        assert order["verdict"] == r["order"]["verdict"], name
        assert fd["verdict"] == r["fd"]["verdict"], name
        for lab in ("R0", "R1"):
            if lab in r["order"]:
                assert abs(order[lab]["slope"] - r["order"][lab]["slope"]) <= 1e-12, (name, lab)
        assert adq.fit_order(pts, loss0, r["ad_relative"])["verdict"] == r["order_1ulp"]["verdict"]
        assert adq.fd_budget(pts, r["ad_relative"], e6.LADDER_SCALE)["verdict"] == r["fd_1ulp"]["verdict"]
        assert r["both_held"] == (order["verdict"] == fd["verdict"] == "HELD")
        assert r["tie_spread_xyz"] == [0.0, 0.0, 0.0]
    held = [e6.NAMES.index(n) for n in d["verified_controls"]]
    assert held == [i for i, n in enumerate(e6.NAMES) if d["directions"][n]["both_held"]]


@pytest.mark.parametrize("arm", ["l1", "l2s"])
def test_chain_rule_and_coverage_replay(arm):
    d = _load(arm)
    jac, g, c0 = np.asarray(d["jacobian"]), np.asarray(d["g_cell"]), np.asarray(d["cell0"])
    np.testing.assert_array_equal(jac, e6.J_PARAM)
    np.testing.assert_allclose(jac.T @ g, d["chain_rule"], rtol=1e-12, atol=0)
    held = [e6.NAMES.index(n) for n in d["verified_controls"]]
    for key, cols in (("coverage_all", slice(None)), ("coverage_verified", held)):
        c = adq.coverage(g, jac[:, cols])
        assert c["rank"] == d[key]["rank"]
        if c["fraction"] is None:
            assert d[key]["fraction"] is None
        else:
            assert abs(c["fraction"] - d[key]["fraction"]) <= 1e-12
    c = adq.coverage(g * c0, jac / c0[:, None])
    assert abs(c["fraction"] - d["coverage_dimensionless"]["fraction"]) <= 1e-12


def test_revert_proof_replay():
    """Re-derive the no-dt verdicts and the declared requirement flags from the
    stored ladder; pin that the forward value is dt-cut invariant."""
    d = _load("revert_l1_nodt")
    first = _load("l1")
    assert d["forward_values_identical"] is True
    assert d["loss0_matches_first"] is True and d["loss0"] == first["loss0"]
    share = dict(zip(e6.NAMES, d["dt_path_share"]))
    # Pre-declared expectation (program 5.3): x_c share exactly 0. Measured
    # -1.48e-4 (float32 recompilation under stop_gradient(dt), amplified by the
    # 43x lead/tail cancellation; note "Diagnostic"). The record is pinned, the
    # expectation is not rewritten into a pass.
    assert share["x_c"] == -0.0001483932070827205
    assert share["x_c"] == (d["g_param_true"][1] - d["g_param_nodt"][1]) / d["g_param_true"][1]
    for name, r in d["directions"].items():
        pts = _points(r["fd"]["steps"])
        assert r["sigma_eff"] == first["directions"][name]["sigma_eff"]
        assert r["ad_relative_nodt"] == d["g_param_nodt"][e6.NAMES.index(name)] * e6.LADDER_SCALE
        order = adq.fit_order(pts, d["loss0"], r["ad_relative_nodt"], sigma=r["sigma_eff"])
        fd = adq.fd_budget(pts, r["ad_relative_nodt"], e6.LADDER_SCALE, sigma=r["sigma_eff"])
        assert order["verdict"] == r["order"]["verdict"], name
        assert fd["verdict"] == r["fd"]["verdict"], name
        assert r["first_attempt_order"] == first["directions"][name]["order"]["verdict"]
        assert r["first_attempt_fd"] == first["directions"][name]["fd"]["verdict"]
    w, xc = d["directions"]["w"], d["directions"]["x_c"]
    assert d["w_fired_on_order"] == (w["order"]["verdict"] != "HELD"
                                     and ("R1" not in w["order"] or w["order"]["R1"]["slope"] < 1.8))
    assert d["xc_unchanged"] == (xc["order"]["verdict"] == xc["first_attempt_order"]
                                 and xc["fd"]["verdict"] == xc["first_attempt_fd"])
    assert d["requirement_met"] == (d["w_fired_on_order"] and d["xc_unchanged"])


def test_recorded_outcomes_are_pinned():
    """Recorded results, not passes: w HELD on both gates on both observables;
    x_c INCONCLUSIVE on order (1 / 3 eligible points) and HELD on FD; the
    revert-proof fired on w (R1 1.022, dt share 0.588) with x_c unchanged;
    x_c's dt share is NOT exactly zero (-1.5e-4, diagnostic below)."""
    mc, l1, l2, rv = _load("map_checks"), _load("l1"), _load("l2s"), _load("revert_l1_nodt")
    assert mc["all_pass"] is True
    for d in (l1, l2):
        w, xc = d["directions"]["w"], d["directions"]["x_c"]
        assert w["both_held"] and 1.8 <= w["order"]["R1"]["slope"] <= 2.2 and w["fd"]["n_informative"] >= 4
        assert xc["order"]["verdict"] == "INCONCLUSIVE" and xc["fd"]["verdict"] == "HELD"
        assert d["verified_controls"] == ["w"]
    assert l1["directions"]["x_c"]["order"]["points"] == 1 and l2["directions"]["x_c"]["order"]["points"] == 3
    assert l2["directions"]["x_c"]["order_1ulp"]["verdict"] == "HELD"     # reported alongside, not primary
    assert rv["requirement_met"] is True and rv["directions"]["w"]["fd"]["verdict"] == "FIRED"
    assert rv["directions"]["w"]["order"]["R1"]["slope"] < 1.3 and rv["dt_path_share"][0] > 0.5
    assert rv["dt_path_share"][1] != 0.0 and abs(rv["dt_path_share"][1]) < 1e-3
    assert 0.80 <= l1["coverage_verified"]["fraction"] <= 0.81 and 0.67 <= l2["coverage_verified"]["fraction"] <= 0.68


def test_revert_cellwise_diagnostic_replay():
    d = _load("diag_revert_cellwise")
    l1, rv = _load("l1"), _load("revert_l1_nodt")
    gt, gn = np.asarray(d["g_cell_true"]), np.asarray(d["g_cell_nodt"])
    np.testing.assert_array_equal(gt, l1["g_cell"])
    diff = gt - gn
    non = np.r_[0:20, 40:60]
    assert not d["nontied_diff_is_zero"] and np.max(np.abs(diff[non])) == d["nontied_max_abs_diff"]
    assert abs(float(e6.J_PARAM[:, 1] @ diff / (e6.J_PARAM[:, 1] @ gt)) - d["chain_xc_share"]) <= 1e-12
    assert abs(float(e6.J_PARAM[:, 0] @ diff / (e6.J_PARAM[:, 0] @ gt)) - d["chain_w_share"]) <= 1e-12
    assert abs(d["chain_w_share"] - rv["dt_path_share"][0]) <= 1e-6
    assert abs(d["xc_lead_contrib_true"] + d["xc_tail_contrib_true"] - rv["g_param_true"][1]) <= 1e-6


def test_coverage_summary_replay():
    d = _load("coverage")
    for arm in ("l1", "l2s"):
        src = _load(arm)
        g, jac = np.asarray(src["g_cell"]), np.asarray(src["jacobian"])
        c = d["arms"][arm]
        assert abs(adq.coverage(g, jac)["fraction"] - c["coverage_all"]["fraction"]) <= 1e-12
        assert abs(adq.coverage(g, jac[:, :1])["fraction"] - c["coverage_w_only"]["fraction"]) <= 1e-12
        assert abs(np.linalg.norm(g[20:40]) - c["g_cell_norm_band"]) <= 1e-12 * np.linalg.norm(g)

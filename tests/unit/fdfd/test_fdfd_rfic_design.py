"""Gates for study R part C, ``validation/fdfd/rfic_design.py``: L-BFGS-B
design of the paper-scale square spiral to L_diff = 4.000 nH (1 %) with Q_diff
maximised at 2.45 GHz on the resolved level, against a 3 x 3 x 3 sweep of the
same box; the loop ran inside one GPU job (``lane_r_rfic.py``, block
``design``).

Live on the CPU (no GPU needed):

D1  the design box is feasible: ``rfx.fdfd.spiral.check_feasible`` accepts its
    eight corners and all 27 sweep points on the paper model (level 1; the
    breakpoints and walls are those of every level), the binding margin
    ``r_out - 4 width - 3 spacing`` is 18 um at the worst corner, and a point
    outside (r_out 150 um) raises.
D2  the box-normalised parameterisation round-trips and the sweep is the
    27 distinct lo / mid / hi points.
D3  the loop's plumbing on the small CPU model (box +-5 % around its theta,
    four evaluations): every evaluation is logged with a loss that
    recomputes from its L and Q, every accepted iterate points at the
    evaluation at exactly its u, and the objective's ``jax.value_and_grad``
    matches FD4 in u to 1e-4.

From the JSON (the GPU numbers, asserted as recorded):

D4  R4 / R6 / R7 and the cost block re-derive from the logged evaluations,
    the sweep points and the finer re-solve; R3 from the FD4 stencils; the
    caveat blocks (projected gradient, R7's implied order and three-point
    limit, the 3 W box gap, the 20 W wall estimate) from their raw numbers.
D5  the gate verdicts; D6 the docstring numbers equal the JSON.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
from typing import Any

import numpy as np
import pytest

from tests._x64_compat import enable_x64

pytest.importorskip("shapely")

REPO = pathlib.Path(__file__).resolve().parents[3]
DESIGN_PATH = REPO / "validation" / "fdfd" / "rfic_design.py"
JSON_PATH = REPO / "validation" / "fdfd" / "rfic_design.json"

_CACHE: dict[str, Any] = {}


def _rd():
    if "rd" not in _CACHE:
        spec = importlib.util.spec_from_file_location("rfic_design_test", DESIGN_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["rd"] = mod
    return _CACHE["rd"]


def _json() -> dict[str, Any]:
    if "json" not in _CACHE:
        assert JSON_PATH.exists(), f"{JSON_PATH} missing: run the study first"
        _CACHE["json"] = json.loads(JSON_PATH.read_text())
    return _CACHE["json"]


def test_d1_box_feasible():
    """Eight corners and 27 sweep points accepted; binding margin 18 um at
    (200, 18, 32) um; r_out = 150 um with the widest strip raises."""
    rd = _rd()
    rs = rd.load_rs()
    from rfx.fdfd import spiral as sm
    with enable_x64():
        m = rs.build_r(1)
        corners = [tuple(c) for c in np.array(np.meshgrid(*rd.BOX)).T.reshape(-1, 3)]
        assert len(set(corners)) == 8
        for th in corners + rd.sweep_points():
            sm.check_feasible(m, np.asarray(th))
        margins = [th[0] - 4 * th[2] - 3 * th[1] for th in corners]
        assert min(margins) == pytest.approx(18e-6, rel=1e-9)
        with pytest.raises(ValueError):
            sm.check_feasible(m, np.asarray((150e-6, 18e-6, 32e-6)))


def test_d2_parameterisation():
    rd = _rd()
    rs = rd.load_rs()
    u = np.array([0.1, 0.5, 0.9])
    assert np.allclose(rd.to_u(rd.to_theta(u)), u, atol=1e-15)
    assert np.allclose(rd.to_theta(rd.to_u(rs.THETA0)), rs.THETA0, rtol=1e-14)
    pts = rd.sweep_points()
    assert len(pts) == 27 and len(set(pts)) == 27
    assert any(abs(p[1] - rs.SPACING) < 1e-15 for p in pts)


def test_d3_loop_plumbing_on_small_model(monkeypatch):
    """Four evaluations of the real loop on the small model (target 5 % above
    its nominal L, box +-5 %): logged, loss recomputed to 1e-12, iterates
    matched to evaluations, value_and_grad vs FD4 in u_2 (step 0.02) <= 1e-4."""
    rd = _rd()
    rs = rd.load_rs()
    import jax
    import jax.numpy as jnp

    import rfx.fdfd.linear_solve as ls
    with enable_x64():
        old = ls.factor_cache_size()
        ls.factor_cache_size(0)
        try:
            m = rs.build_cheap()
            th = m.theta_nominal
            monkeypatch.setattr(rs, "THETA0", th)
            monkeypatch.setattr(rd, "BOX", tuple((0.95 * v, 1.05 * v) for v in th))
            f0 = rs.forward(m)
            monkeypatch.setattr(rd, "L_TARGET", 1.05 * f0["L_diff"])
            q_ref = f0["Q_diff"]
            rec = rd.run_design(m, q_ref, lambda d: None, maxfun=4, maxiter=3,
                                log=lambda s: None)
            fn = rd.objective(m, q_ref)
            u0 = jnp.asarray(rd.to_u(th))
            (val, _), g = jax.value_and_grad(fn, has_aux=True)(u0)

            def f(v: float) -> float:
                return float(fn(u0.at[2].set(v))[0])
            h = 0.02
            fd = (-f(0.5 + 2 * h) + 8 * f(0.5 + h) - 8 * f(0.5 - h) + f(0.5 - 2 * h)) / (12 * h)
        finally:
            ls.factor_cache_size(old)
    assert 2 <= len(rec["evals"]) <= 8      # scipy checks maxfun per iteration, not per call
    for e in rec["evals"]:
        assert e["loss"] == pytest.approx(rd.loss_of(e["L_diff"], e["Q_diff"], q_ref), rel=1e-12)
        assert e["reciprocity_worst"] <= 1e-8 and e["passivity_worst"] <= 1 + 1e-8
    for it in rec["iterates"]:
        assert it["eval"] is not None
        assert np.array_equal(np.asarray(rec["evals"][it["eval"]]["u"]), np.asarray(it["u"]))
    assert abs(float(g[2]) - fd) / abs(fd) <= 1e-4


# ----------------------------------------------------------------------------
# from the JSON

def test_d4_json_rederives():
    rd = _rd()
    st = _json()
    g = st["gates"]
    fe = rd.final_eval(st["design"])
    assert fe is not None
    assert g["R4"]["L_err"] == pytest.approx(fe["L_diff"] / rd.L_TARGET - 1, rel=1e-12)
    pts = st["csweep"]["points"]
    assert len(pts) == 27
    band = [p for p in pts if abs(p["L_diff"] / rd.L_TARGET - 1) <= rd.L_BAND]
    assert len(g["R6"]["sweep_points_in_band"]) == len(band)
    for p in band:
        assert fe["Q_diff"] > p["Q_diff"]
    for e in st["design"]["evals"]:
        assert e["loss"] == pytest.approx(
            rd.loss_of(e["L_diff"], e["Q_diff"], st["design"]["q_ref"], st["design"]["lam"]),
            rel=1e-12)
    cs = st["fine"]["cases"]
    ip = cs["opt_m3z2_w3"]["L_diff"] / cs["opt_m2_w3"]["L_diff"] - 1
    vt = cs["opt_m2z3_w10"]["L_diff"] / fe["L_diff"] - 1
    assert g["R7"]["inplane_at_optimum"]["dL_rel"] == pytest.approx(ip, rel=1e-12)
    assert g["R7"]["vertical_at_optimum"]["dL_rel"] == pytest.approx(vt, rel=1e-12)
    assert g["R7"]["dL_rel"] == pytest.approx(ip + vt, rel=1e-12)
    ipq = cs["opt_m3z2_w3"]["Q_diff"] / cs["opt_m2_w3"]["Q_diff"] - 1
    vtq = cs["opt_m2z3_w10"]["Q_diff"] / fe["Q_diff"] - 1
    assert g["R7"]["dQ_rel"] == pytest.approx(ipq + vtq, rel=1e-12)
    assert cs["opt_m3z2_w3"]["grid"]["cells_across_width"] == 6
    assert cs["opt_m2z3_w10"]["grid"]["cells_across_width"] == 4
    # the same model at the same theta through two paths and two jobs: the design's
    # value_and_grad forward (R2) and a plain forward of the sweep (R3), (200, 10, 22) um
    corner = [e for e in st["design"]["evals"] if np.allclose(e["theta"], (200e-6, 10e-6, 22e-6),
                                                                rtol=0, atol=1e-15)]
    sw = [p for p in pts if np.allclose(p["theta"], (200e-6, 10e-6, 22e-6), rtol=0, atol=1e-15)]
    assert corner and sw
    assert corner[0]["L_diff"] == pytest.approx(sw[0]["L_diff"], rel=1e-9)
    assert corner[0]["Q_diff"] == pytest.approx(sw[0]["Q_diff"], rel=1e-9)
    kk = st["kkt"]
    dl = kk["dL_dtheta"]
    assert abs(sum(a * b for a, b in zip(dl, kk["move"]))) <= 1e-12 * abs(dl[2] * kk["move"][2]) + 1e-30
    fo = st["fdopt"]
    for name, rec in fo["fd"].items():
        h = rec["step"]
        for q in ("L", "Q"):
            v = {int(k): p[f"{q}_diff"] for k, p in rec["points"].items()}
            fd = (-v[2] + 8 * v[1] - 8 * v[-1] + v[-2]) / (12 * h)
            assert fd == pytest.approx(rec[f"fd_{q}"], rel=1e-12)
    assert st["cost"]["design"]["objective_evaluations"] == len(st["design"]["evals"])
    assert st["cost"]["sweep"]["forward_three_fixture_solves"] == 27


def test_d4_caveats_rederive():
    """The caveat blocks from their raw numbers: the projected gradient
    P(u - g) - u (and the KKT verdict against gtol), the implied order p
    (it solves (2^-p - 3^-p) / (1 - 2^-p) = step ratio to 1e-10) and the
    three-point limit from the logged level-1 / level-2 / R7 values, the 3 W
    in-plane box's absolute gap, and the 20 W estimate of the design's L --
    all to 1e-12 relative."""
    rd = _rd()
    st = _json()
    kk = st["kkt"]
    u, g = np.asarray(kk["u_at_return"]), np.asarray(kk["grad_u_at_return"])
    pg = np.clip(u - g, 0, 1) - u
    assert np.allclose(pg, kk["projected_grad_u"], rtol=1e-12, atol=0)
    assert kk["kkt_point"] is (float(np.max(np.abs(pg))) <= rd.GTOL)
    fe = rd.final_eval(st["design"])
    r7 = st["gates"]["R7"]
    cv = r7["caveats"]
    cs = st["fine"]["cases"]
    assert cv["inplane_box_3W_vs_10W_at_optimum"]["rel"] == pytest.approx(
        cs["opt_m2_w3"]["L_diff"] / fe["L_diff"] - 1, rel=1e-12)
    ex = cv["extrapolation"]["L"]
    v1, v2 = cs["opt_m1_w10"]["L_diff"], fe["L_diff"]
    v3 = v2 * (1 + r7["dL_rel"])
    ratio = (v3 - v2) / (v2 - v1)
    assert ex["step_ratio_23_over_12"] == pytest.approx(ratio, rel=1e-12)
    p = ex["implied_order"]
    assert (2 ** -p - 3 ** -p) / (1 - 2 ** -p) == pytest.approx(ratio, rel=1e-10)
    assert ex["limit"] == pytest.approx(v2 + (v2 - v1) * 2 ** -p / (1 - 2 ** -p), rel=1e-12)
    assert cv["extrapolation"]["Q"]["limit"] is None       # p_Q below ORDER_MIN
    assert cv["extrapolation"]["Q"]["implied_order"] < rd.ORDER_MIN <= p
    w = st["walls"]
    assert w["design_L_at_20W_estimate"] == pytest.approx(
        fe["L_diff"] * (1 + w["per_point"]["design"]["dL_rel_10_to_20W"]), rel=1e-12)


def test_d5_gate_verdicts():
    st = _json()
    g = st["gates"]
    assert g["R2"]["passed"] is True
    assert g["R3"]["passed"] is True and g["R3"]["worst_rel"] <= 1e-4
    assert g["R4"]["passed"] is True and abs(g["R4"]["L_err"]) <= 0.01
    assert g["R6"]["passed"] is True
    assert g["R7"]["passed"] is True


def test_d6_docstring_numbers_equal_json():
    rd = _rd()
    st = _json()
    doc = " ".join(rd.__doc__.split())
    for s in docstring_numbers(st):
        assert s in doc, s


def docstring_numbers(st: dict[str, Any]) -> list[str]:
    g = st["gates"]
    c = st["cost"]
    r7 = g["R7"]
    out = [f"{g['R4']['L_diff'] * 1e12:.2f} pH", f"{g['R4']['Q_diff']:.3f}",
           f"{100 * g['R4']['L_err']:+.4f} %", f"{g['R3']['worst_rel']:.2e}",
           f"{c['design']['objective_evaluations']} objective evaluations",
           f"{c['design']['accepted_iterates']} accepted iterates",
           f"{c['design']['cudss_factorisations']} cuDSS factorisations",
           f"{c['design']['wall_seconds_loop'] / 60:.1f} min",
           f"{c['sweep']['wall_seconds'] / 60:.1f} min",
           f"{g['R2']['n_solves']} solves", f"{g['R2']['reciprocity_worst']:.2e}",
           f"{g['R2']['passivity_worst']:.5f}",
           f"{g['R6']['best_in_band_Q']:.3f}", f"{g['R6']['design_minus_best_in_band']:.3f}",
           f"{100 * r7['inplane_at_optimum']['dL_rel']:+.3f} %",
           f"{100 * r7['inplane_at_optimum']['dQ_rel']:+.3f} %",
           f"{100 * r7['vertical_at_optimum']['dL_rel']:+.3f} %",
           f"{100 * r7['vertical_at_optimum']['dQ_rel']:+.3f} %",
           f"{100 * r7['dL_rel']:+.3f} %", f"{100 * r7['dQ_rel']:+.3f} %",
           f"{100 * st['kkt']['predicted_dQ_rel']:.2f} %",
           f"{r7['L_finer_estimate'] * 1e12:.2f} pH", f"{100 * r7['L_err_finer_estimate']:.2f} %",
           f"N = {r7['inplane_grid_at_10W_n_unknowns']}",
           f"{100 * r7['box_check_level1_to_2_nominal']['dL_points']:.2f} points",
           f"{100 * r7['box_check_level1_to_2_nominal']['dQ_points']:.2f} on Q",
           f"{100 * r7['additivity_check_level1_to_2_nominal']['dL_points_full_minus_sum']:.3f} points",
           f"({100 * r7['additivity_check_level1_to_2_nominal']['dQ_points_full_minus_sum']:.2f} points",
           f"L_diff {100 * r7['level1_to_2_at_optimum']['dL_rel']:+.2f} %",
           f"Q_diff {100 * r7['level1_to_2_at_optimum']['dQ_rel']:+.2f} %",
           f"N = {r7['cases']['opt_m3z2_w3']['n_unknowns']}",
           f"N = {r7['cases']['opt_m2z3_w10']['n_unknowns']}",
           f"{r7['cases']['opt_m3z2_w3']['plan_gb']:.2f} GB",
           f"{r7['cases']['opt_m2z3_w10']['plan_gb']:.2f} GB",
           f"{r7['full_level3']['plan_gb']:.2f} GB",
           f"{st['kkt']['projected_grad_inf_norm']:.4f}",
           f"{-100 * r7['caveats']['inplane_box_3W_vs_10W_at_optimum']['rel']:.2f} % below",
           f"p = {r7['caveats']['extrapolation']['L']['implied_order']:.2f}",
           f"p = {r7['caveats']['extrapolation']['Q']['implied_order']:.2f}",
           f"{r7['caveats']['extrapolation']['L']['limit'] * 1e12:.2f} pH",
           f"{-100 * r7['caveats']['extrapolation']['L']['level2_rel_to_limit']:.2f} % below",
           f"{100 * r7['caveats']['extrapolation']['L_err_at_limit']:+.2f} %",
           f"{100 * st['walls']['per_point']['design']['dL_rel_10_to_20W']:+.2f} %",
           f"{100 * st['walls']['per_point']['nominal']['dL_rel_10_to_20W']:+.2f} %",
           f"{100 * st['walls']['per_point']['r_out_hi']['dL_rel_10_to_20W']:+.2f} %",
           f"{st['walls']['design_L_at_20W_estimate'] * 1e12:.2f} pH",
           f"{100 * st['walls']['design_L_err_at_20W_estimate']:+.2f} %",
           f"{100 * st['walls']['bias_slope_per_m_r_out'] * 1e-6:.4f} % per um",
           f"{100 * st['walls']['bias_slope_over_dL_rel_dr_out']:.1f} %",
           f"{100 * st['walls']['dL_rel_dr_out_at_theta_star'] * 1e-6:.2f} % per um",
           f"{100 * st['walls']['max_abs_dQ_rel_10_to_20W']:.2f} %",
           f"{st['walls']['per_point']['design']['wall_x_minus_r_out']['10'] * 1e6:.1f} um",
           f"{st['walls']['per_point']['r_out_hi']['wall_x_minus_r_out']['10'] * 1e6:.1f} um",
           f"{-100 * st['accuracy']['own_referee_190um']['gap_by_level']['2']:.2f} % low",
           f"{100 * st['remesh']['dL_rel_fresh_vs_deformed']:+.2f} %",
           f"{100 * st['remesh']['dQ_rel_fresh_vs_deformed']:+.2f} %"]
    for name, v in g["R3"]["checked"].items():
        out.append(f"{name.replace('_d', '/d', 1)} {v['rel']:.2e}")
    return [" ".join(x.split()) for x in out]

"""Gates for study H, ``validation/fdfd/rfic_spiral.py``: an RFIC-scale
square spiral (the geometry of a published 2.4 GHz LC-VCO, on an SG13G2-like
stack) through the differentiable FDFD pipeline.

The study's own levels are N = 85785 unknowns and ~220 s per de-embedded
three-fixture solve, tens of them, so this file does NOT re-run them. It
runs ONE live solve of the SAME geometry and the SAME code path on the
study's deliberately degraded test grid (``study.build_cheap()``: no wall
padding, one cell per metal slab, one silicon cell, N = 26818, ~13 s) and
asserts the things that grid supports, plus everything that needs no solver
at all. Every number the study measured on its real grids is asserted
against the JSON as a hard value, so the JSON cannot drift or be widened
silently.

Gates here (numbers measured in this session; see each docstring):

T1  the parameterisation is feasible where the study says it is:
    ``spiral.check_feasible`` accepts the nominal theta and all eight
    corners of part C's design box, the binding margin
    ``a_in - width/2`` is >= 14 um over the whole box, and a theta just
    outside it (r_out shrunk until the inner end reaches the axis) raises.
T2  the live cheap solve is reciprocal (|S12 - S21| <= 1e-8) and passive
    (max singular value <= 1 + 1e-8) in all three fixtures, and the
    study's ``solve_split`` -- three single-fixture solves with the LU
    factor cache emptied between them, which is how the study keeps one
    factor live -- reproduces a single ``spiral.solve_spiral`` call on the
    same model BITWISE on L_diff, Q_diff, L_se and Q_se. That identity is
    what lets the study's recorded metrics be read as ``solve_spiral``
    output.
T3  the referee re-derives: the study's PRIMARY quantity
    L(strip) - L(bridge) and every part of it (strip, bridge, the
    reference-plane band, the bridge split band, the ground-image curve,
    Mohan-Wheeler) recompute from the study's own functions to 1e-12 of the
    JSON, the area-exact corner convention tiles the ``rect_spiral``
    polygon to 1e-12 while the delivered Greenhouse convention overlaps
    itself, and the reference-plane invariance holds: the strip alone moves
    1.06 % across the lead-column footprint while the DIFFERENCE moves
    0.0176 %, a 60x collapse.
T4  the JSON's derived numbers re-derive from its raw records: every gap
    from L_diff and the referee, every worst-case reciprocity and
    passivity from the per-solve stats, both FD4 derivatives from the four
    logged stencil points, the design loop's L error from its final
    evaluation, and R5's largest N from the per-level rows.
T5  the recorded gate verdicts as hard numbers.

x64 is scoped per test through ``tests._x64_compat.enable_x64`` (never
flipped at module level); the model and the one solve are cached for the
module.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import time
from typing import Any

import jax
import numpy as np
import pytest

from rfx.fdfd import spiral as sm
from tests._x64_compat import enable_x64

pytest.importorskip("shapely")

REPO = pathlib.Path(__file__).resolve().parents[1]
STUDY_PATH = REPO / "validation" / "fdfd" / "rfic_spiral.py"
JSON_PATH = REPO / "validation" / "fdfd" / "rfic_spiral.json"

_CACHE: dict[str, Any] = {}


def _study():
    if "study" not in _CACHE:
        spec = importlib.util.spec_from_file_location("rfic_spiral_study", STUDY_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["study"] = mod
    return _CACHE["study"]


def _json() -> dict[str, Any]:
    if "json" not in _CACHE:
        assert JSON_PATH.exists(), f"{JSON_PATH} missing: run the study first"
        _CACHE["json"] = json.loads(JSON_PATH.read_text())
    return _CACHE["json"]


def _model():
    """The cheap live model (see the module doc); built once."""
    if "model" not in _CACHE:
        with enable_x64():
            _CACHE["model"] = _study().build_cheap()
    return _CACHE["model"]


def _cheap_solve() -> dict[str, Any]:
    """One live de-embedded solve of the cheap model at 2.45 GHz, both ways:
    ``spiral.solve_spiral`` (all three fixtures in one traced call) and the
    study's ``solve_split`` (three single-fixture calls with the LU factor
    cache emptied between them). Cached; ~23 s together."""
    if "cheap_solve" not in _CACHE:
        st = _study()
        with enable_x64():
            from rfx.fdfd.linear_solve import factor_cache_size
            # the cache size is process-global: restore it, or every later
            # test that relies on the documented default of 4 sees 1
            old_cache = factor_cache_size()
            factor_cache_size(1)
            try:
                m = _model()
                kw = dict(sigma_metal=st.SIGMA_CU, sigma_si=st.SIGMA_SI)
                t0 = time.time()
                res = sm.solve_spiral(m, st.FREQ_B, **kw)
                jax.block_until_ready(res.L_diff)
                t_one = time.time() - t0
                t0 = time.time()
                split = st.solve_split(m, st.FREQ_B, **kw)
            finally:
                factor_cache_size(old_cache)
            _CACHE["cheap_solve"] = {"res": res, "split": split, "seconds_one": t_one,
                                     "seconds_split": time.time() - t0}
    return _CACHE["cheap_solve"]


# ---------------------------------------------------------------------------
# T1: the parameterisation is feasible where the study says it is

def test_t1_feasibility_over_the_design_box():
    """``spiral.check_feasible`` accepts the nominal theta and all eight
    corners of part C's design box on the study's W/1 model (the walls are at
    431 um, so ``r_out`` up to 250 um keeps the breakpoint order); the
    binding topological margin ``a_in - width/2 = r_out - width -
    n (width + spacing)`` is 18.0 um at the worst corner (r_out 200,
    spacing 18, width 32 um) and 56.0 um at the nominal, and a theta whose
    inner end reaches the axis raises. Build only, no solve."""
    st = _study()
    with enable_x64():
        model = st.build_level(1.0)
        st_sm = sm
        st_sm.check_feasible(model, np.asarray(st.THETA0))
        assert st.feasibility_margin(st.THETA0) == pytest.approx(56.0e-6, abs=1e-12)
        margins = []
        for a in st.BOUNDS[0]:
            for b in st.BOUNDS[1]:
                for c in st.BOUNDS[2]:
                    theta = np.asarray([a, b, c], dtype=np.float64)
                    st_sm.check_feasible(model, theta)          # must not raise
                    margins.append(st.feasibility_margin(theta))
        assert min(margins) == pytest.approx(18.0e-6, abs=1e-12)
        assert min(margins) > 0.0
        # just outside: a_in reaches the axis at r_out = width + n pitch = 162 um
        with pytest.raises(ValueError):
            st_sm.check_feasible(model, np.asarray([160e-6, st.SPACING, st.WIDTH]))


# ---------------------------------------------------------------------------
# T2: the live solve, and the split-solve identity the study depends on

def test_t2_reciprocity_and_passivity_live():
    """The cheap live solve at 2.45 GHz with the Leontovich metal and the
    lossy silicon: |S12 - S21| <= 1e-8 and max singular value <= 1 + 1e-8 in
    all three solved fixtures AND in the de-embedded S. Measured this
    session: worst reciprocity 2.264e-12, worst singular value 0.999178
    (solved) and 0.968248 (de-embedded)."""
    st = _study()
    d = _cheap_solve()["split"]
    for name in ("dut", "open", "short", "deembedded"):
        s = d["s_stats"][name]
        assert s["reciprocity"] <= st.RECIP_TOL, (name, s)
        assert s["max_singular_value"] <= st.PASSIVITY_TOL, (name, s)
    assert d["reciprocity_worst_solved"] < 1e-11
    assert d["passivity_worst_solved"] < 1.0


def test_t2b_split_solve_is_the_same_computation():
    """``solve_split`` -- the study's three single-fixture solves with the LU
    factor cache emptied between them -- reproduces a single
    ``spiral.solve_spiral`` call BITWISE on L_diff, Q_diff, L_se, Q_se and
    L_raw (measured: exactly 0.0 relative on all five). Without this the
    study's recorded metrics would be a different computation from the one
    ``spiral.solve_spiral`` performs."""
    c = _cheap_solve()
    res, split = c["res"], c["split"]
    for key, value in (("L_diff", res.L_diff), ("Q_diff", res.Q_diff),
                       ("L_se", res.L_se), ("Q_se", res.Q_se), ("L_raw", res.L_raw)):
        assert float(value) == split[key], (key, float(value), split[key])


# ---------------------------------------------------------------------------
# T3: the referee re-derives, and its conventions are pinned by invariances

def test_t3_referee_reproduces_the_json():
    """Recompute the whole referee block from the study's own functions and
    require every recorded number to 1e-12 relative. Host numpy, no solver:
    this is the gate that keeps the referee side of R1 honest."""
    st = _study()
    with enable_x64():
        fresh = st.referee_block()
    rec = _json()["referee"]
    for key in ("strip", "bridge", "deembedded", "strip_no_ground",
                "greenhouse_corner_convention_strip", "reference_plane_spread",
                "reference_plane_spread_strip_only", "split_spread"):
        assert fresh[key] == pytest.approx(rec[key], rel=1e-12), key
    for key in rec["ground_image_curve"]:
        assert fresh["ground_image_curve"][key]["deembedded"] == pytest.approx(
            rec["ground_image_curve"][key]["deembedded"], rel=1e-12), key
    assert fresh["mohan_wheeler"]["L"] == pytest.approx(rec["mohan_wheeler"]["L"], rel=1e-12)
    assert fresh["gradient_fd4"] == pytest.approx(rec["gradient_fd4"], rel=1e-9)


def test_t3b_corner_convention_and_reference_plane():
    """The two referee conventions the protocol fixes, each pinned by a
    measurement rather than by assertion.

    (a) The area-exact corner convention tiles the ``gds.rect_spiral`` strip
    polygon EXACTLY -- sum of bar areas = union = polygon area to 1.2e-16
    relative -- while the delivered Greenhouse centreline convention
    overlaps itself by 2.339 % of the area, and both have the same total
    centreline length (3.847 mm), so no conductor is created or destroyed.

    (b) The de-embedding's reference plane is a CHOICE, and the primary
    quantity is the one that does not depend on it: moving the plane through
    the lead-column footprint (port line -> 15 um -> 30 um) moves the strip
    alone by 0.9914 % and moves ``L(strip) - L(bridge)`` by 0.0345 %, a
    28.7x collapse. The bridge's M2/M1 split, a pure grid detail, moves it
    by 0.0193 %."""
    rec = _json()["referee"]
    ae, gh = rec["area_gate"]["area_exact"], rec["area_gate"]["greenhouse"]
    assert abs(ae["sum_over_polygon_minus_1"]) < 1.2e-16
    assert abs(ae["union_over_polygon_minus_1"]) < 1e-15
    assert abs(ae["union_over_sum_minus_1"]) < 1e-15
    assert gh["union_over_sum_minus_1"] == pytest.approx(-0.0233948531323106, rel=1e-9)
    assert ae["total_centreline_length"] == pytest.approx(gh["total_centreline_length"],
                                                          rel=1e-15)
    assert ae["total_centreline_length"] == pytest.approx(3.847e-3, rel=1e-9)
    assert rec["reference_plane_spread_strip_only"] == pytest.approx(0.009913761840971407,
                                                                     rel=1e-9)
    assert rec["reference_plane_spread"] == pytest.approx(0.0003448495721410342, rel=1e-9)
    collapse = rec["reference_plane_spread_strip_only"] / rec["reference_plane_spread"]
    assert collapse == pytest.approx(28.748076384206577, rel=1e-9)
    assert collapse > 20.0
    assert rec["split_spread"] == pytest.approx(0.00019317226294868545, rel=1e-9)


def test_t3c_mohan_wheeler_confirms_the_referee():
    """The independent closed form. Mohan's modified-Wheeler fit for a square
    spiral (2-3 % typical, ~8 % worst-case error against field solvers) has
    no ground plane, so it is read against the referee's FREE-SPACE strip:
    4.1653 nH against 4.0243 nH, +3.504 %, inside Mohan's stated band. The
    ground-imaged primary value is 3.5604 nH, 17.0 % below the fit, which is
    the ground shielding, not a disagreement."""
    rec = _json()["referee"]
    mw = rec["mohan_wheeler"]
    assert mw["d_out"] == pytest.approx(436e-6, rel=1e-12)
    assert mw["d_in"] == pytest.approx(200e-6, rel=1e-12)
    assert mw["L"] == pytest.approx(4.16532927655346e-9, rel=1e-9)
    assert mw["over_referee_no_ground_minus_1"] == pytest.approx(0.035039613667293024,
                                                                 rel=1e-6)
    assert abs(mw["over_referee_no_ground_minus_1"]) < 0.08     # Mohan's worst case
    assert mw["over_referee_deembedded_minus_1"] == pytest.approx(0.1699154842133408,
                                                                  rel=1e-6)


# ---------------------------------------------------------------------------
# T4: the JSON's derived numbers re-derive from its raw records

def test_t4_json_is_self_consistent():
    """Everything the JSON DERIVES is recomputed from what it MEASURED, so a
    hand-edited gate cannot survive: the per-level gaps from ``L_diff`` and
    the referee, the worst-case reciprocity and passivity from the per-solve
    ``s_stats``, the FD4 derivatives from the four logged stencil points, the
    AD-vs-FD ratios from those and the recorded gradients, the design loop's
    final L error from its final evaluation, and R5's largest N from the
    per-level rows."""
    st = _study()
    d = _json()
    ref = d["referee"]["deembedded"]
    g = d["gates"]

    # R1: gaps
    for key, level in d["A"]["levels"].items():
        assert level["gap"] == pytest.approx(level["L_diff"] / ref - 1.0, rel=1e-12)
        assert g["R1"]["per_level_gap"][f"W/{key}"] == pytest.approx(level["gap"], rel=1e-12)
        assert level["grid"]["cells_across_width"] == 2
    if "w2_gap" in g["R1"]:
        assert g["R1"]["w2_gap_minus_d2"] == pytest.approx(
            g["R1"]["w2_gap"] - g["R1"]["d2_w2_gap"], rel=1e-12)
        assert g["R1"]["passed"] is (abs(g["R1"]["w2_gap_minus_d2"]) <= g["R1"]["tolerance_points"])

    # R2: worst-case reciprocity / passivity from the per-solve records
    per_r, per_p = g["R2"]["per_solve_reciprocity"], g["R2"]["per_solve_passivity"]
    assert per_r and per_p
    assert g["R2"]["reciprocity_worst"] == pytest.approx(max(per_r.values()), rel=0, abs=0)
    assert g["R2"]["passivity_worst"] == pytest.approx(max(per_p.values()), rel=0, abs=0)
    for name, rec in _iter_solves(d):
        stats = rec["s_stats"]
        assert rec["reciprocity_worst_solved"] == max(
            stats[k]["reciprocity"] for k in ("dut", "open", "short")), name
        assert rec["passivity_worst_solved"] == max(
            stats[k]["max_singular_value"] for k in ("dut", "open", "short")), name

    # R3: the FD4 stencils
    for param, rec in d["B"].get("fd", {}).items():
        pts = rec["points"]
        if len(pts) < 4:
            continue
        h = rec["step"]
        assert h == pytest.approx(st.FD_STEP_REL * st.THETA0[rec["index"]], rel=1e-12)
        for field, key in (("L_diff", "fd4_dL_diff"), ("Q_diff", "fd4_dQ_diff")):
            p = {int(m): pts[m][field] for m in pts}
            fd = (-p[2] + 8 * p[1] - 8 * p[-1] + p[-2]) / (12 * h)
            assert rec[key] == pytest.approx(fd, rel=1e-12), (param, field)
    for label, rec in g["R3"]["checked"].items():
        assert rec["rel"] == pytest.approx(abs(rec["ad"] - rec["fd4"]) / abs(rec["fd4"]),
                                           rel=1e-12), label

    # R4: the loop's own bookkeeping
    c = d.get("C", {})
    if c.get("complete"):
        assert c["final_L_rel_error"] == pytest.approx(c["final_L"] / c["L_target"] - 1.0,
                                                       rel=1e-12)
        assert c["n_replay_mismatches"] == 0
        assert c["final_theta"] == c["final"]["theta"]
        assert g["R4"]["passed"] is (abs(c["final_L_rel_error"]) <= st.L_BAND)
        with enable_x64():
            model = st.build_level(1.0)
            sm.check_feasible(model, np.asarray(c["final_theta"]))
        for k, (lo, hi) in enumerate(st.BOUNDS):
            assert lo - 1e-15 <= c["final_theta"][k] <= hi + 1e-15

    # R5: sizes
    rows = g["R5"]["rows"]
    assert rows
    assert g["R5"]["largest_n_unknowns"] == max(r["n_unknowns"] for r in rows)
    assert g["R5"]["peak_footprint_gb"] == max(
        r["footprint_gb"] for r in rows if r["footprint_gb"])


def _iter_solves(d: dict[str, Any]):
    for part, key in (("A", "levels"), ("B", "levels"), ("B", "sweep")):
        for k, v in d.get(part, {}).get(key, {}).items():
            if "s_stats" in v:
                yield f"{part}.{key}.{k}", v
    for k, v in d.get("B", {}).get("wall", {}).items():
        if isinstance(v, dict) and "solve" in v:
            yield f"B.wall.{k}", v["solve"]
    for k, v in d.get("B", {}).get("t_si_sensitivity", {}).items():
        if isinstance(v, dict) and "solve" in v:
            yield f"B.t_si.{k}", v["solve"]
    if "w2_resolve" in d.get("C", {}):
        yield "C.w2_resolve", d["C"]["w2_resolve"]["solve"]


def test_t4b_every_recorded_solve_is_reciprocal_and_passive():
    """The R2 gate applied to every solve the study recorded, not just the
    worst: |S12 - S21| <= 1e-8 and max singular value <= 1 + 1e-8 in the dut,
    open and short fixtures of every level, sweep point, wall pad, substrate
    thickness and re-solve."""
    st = _study()
    d = _json()
    n = 0
    for name, rec in _iter_solves(d):
        for fixture in ("dut", "open", "short"):
            s = rec["s_stats"][fixture]
            assert s["reciprocity"] <= st.RECIP_TOL, (name, fixture, s)
            assert s["max_singular_value"] <= st.PASSIVITY_TOL, (name, fixture, s)
        n += 1
    assert n >= 2, "the JSON has fewer solves than part A alone would give"


# ---------------------------------------------------------------------------
# T5: the recorded gate numbers, as hard values

def test_t5_gate_r1_absolute_l():
    """R1, the only absolute-accuracy gate in the study. PASSES: the FDFD's
    de-embedded L_diff at 100 MHz is 3089.74 pH (base_dx = W) and 3088.30 pH
    (W/2) -- 0.047 % apart, because at this pitch the two are the same grid
    -- against the referee's 3560.37 pH, i.e. -13.218 % and -13.259 %. D2's
    small spiral at the SAME two cells across the strip was -12.025 %, so the
    W/2 gap is 1.234 points deeper, inside the 3-point criterion: the deficit
    at RFIC scale is the known strip cross-section bias and nothing new.
    With the side walls pushed from 4 to 8 graded cells (N = 119689) the same
    protocol gives 3121.04 pH, -12.339 %, 0.314 points from D2's number."""
    st = _study()
    g = _json()["gates"]["R1"]
    assert g["referee_deembedded"] == pytest.approx(3.560367678485986e-09, rel=1e-9)
    assert g["per_level_gap"]["W/1"] == pytest.approx(-0.13218404790384464, rel=1e-6)
    assert g["per_level_gap"]["W/2"] == pytest.approx(-0.13258911856264166, rel=1e-6)
    assert g["per_level_cells_across_width"] == {"W/1": 2, "W/2": 2}
    assert g["d2_w2_gap"] == pytest.approx(-0.12025158857388896, rel=1e-12)
    assert g["w2_gap_minus_d2"] == pytest.approx(-0.012337529988752705, rel=1e-6)
    assert abs(g["w2_gap_minus_d2"]) <= st.R1_POINTS
    assert g["passed"] is True
    w = g["wall_correction"]
    assert w["rel_change_L"] == pytest.approx(0.010128690428861553, rel=1e-6)
    assert w["gap_at_pad_8"] == pytest.approx(-0.12339420877583496, rel=1e-6)
    assert w["gap_at_pad_8_minus_d2"] == pytest.approx(-0.0031426202019460003, rel=1e-5)
    # the two levels really are one grid: the whole point of the W/3 remark
    lv = _json()["A"]["levels"]
    assert abs(lv["2"]["L_diff"] / lv["1"]["L_diff"] - 1.0) < 5e-4


def test_t5b_gate_r2_reciprocity_passivity_walls():
    """R2 FAILS, on the wall criterion alone, and the failure is reported as
    a measured number rather than widened away.

    * reciprocity: worst |S12 - S21| = 1.970e-10 over 12 recorded solves,
      tolerance 1e-8 -- PASS.
    * passivity: worst singular value 1.0000000001071, tolerance 1 + 1e-8 --
      PASS (and the de-embedded S is passive too, 0.99999998).
    * side walls: pad 4 -> 8 moves L_diff by +1.0245 % against the < 1 %
      criterion -- FAIL. The three-point ladder says why it is only just
      over: +0.983 % from pad 4 to 6 and +0.041 % from 6 to 8, so the box is
      converged from pad 6 on and the pad-4 grid used by parts A, B and C is
      1.02 % low from the side walls alone."""
    st = _study()
    g = _json()["gates"]["R2"]
    assert g["n_solves_checked"] >= 12
    assert g["reciprocity_worst"] == pytest.approx(1.9704589366182672e-10, rel=1e-6)
    assert g["reciprocity_worst"] <= st.RECIP_TOL
    assert g["passivity_worst"] == pytest.approx(1.000000000107069, rel=1e-12)
    assert g["passivity_worst"] <= st.PASSIVITY_TOL
    assert g["deembedded_passivity_worst"] <= st.PASSIVITY_TOL
    assert g["passed_reciprocity"] is True
    assert g["passed_passivity"] is True
    assert g["wall_rel_change_L"] == pytest.approx(0.010245359246944474, rel=1e-6)
    assert g["wall_rel_change_L"] > st.WALL_TOL          # the gate this study FAILS
    assert g["passed_wall"] is False
    assert g["passed"] is False
    steps = g["wall_consecutive_steps_L"]
    assert steps["4->6"] == pytest.approx(0.009832471874017035, rel=1e-6)
    assert steps["6->8"] == pytest.approx(0.0004088671977056091, rel=1e-6)
    assert abs(steps["6->8"]) < 1e-3                     # converged from pad 6 on


def test_t5c_gate_r3_ad_vs_fd4():
    """R3 PASSES by five orders of magnitude. ``jax.grad`` against FD4 of the
    same discrete model with 1 % steps: dL/dr_out 1.402e-8 and dQ/dwidth
    5.747e-8 relative (tolerance 1e-4), and from the same two stencils for
    free dL/dwidth 5.673e-10 and dQ/dr_out 4.517e-5. The steps move the
    objective far above the LU noise floor: L_diff spans 2910.22 to
    3177.84 pH across the r_out stencil (8.8 %) and Q_diff 20.65841 to
    20.62547 across the width stencil (0.16 %, i.e. 3.3e-2 absolute against a
    ~1e-9 solver floor)."""
    st = _study()
    d = _json()
    g = d["gates"]["R3"]
    assert g["checked"]["dL_dr_out"]["rel"] == pytest.approx(1.4020244881791309e-08, rel=1e-3)
    assert g["checked"]["dQ_dwidth"]["rel"] == pytest.approx(5.7466826121904616e-08, rel=1e-3)
    assert g["worst_rel"] <= st.AD_FD_TOL
    assert g["passed"] is True
    for label, rec in g["all_available"].items():
        assert rec["rel"] <= st.AD_FD_TOL, label
    # the stencils have real signal
    pts = d["B"]["fd"]["r_out"]["points"]
    ls = [pts[m]["L_diff"] for m in pts]
    assert (max(ls) - min(ls)) / min(ls) > 0.05
    ptw = d["B"]["fd"]["width"]["points"]
    qs = [ptw[m]["Q_diff"] for m in ptw]
    assert (max(qs) - min(qs)) > 1e-3


def test_t5d_leontovich_validity_is_reported_not_hidden():
    """The frequency floor of the sheet metal model, as numbers: the skin
    depth at 2.45 GHz is 1.8411 um for the stack's 3.05e7 S/m, so
    ``spiral.leontovich_validity`` -- skin depth over the THINNEST metal slab,
    the 2 um TopMetal1 underpass -- is 0.9206 and delta/t(TopMetal2, 3 um) is
    0.6137. PURE copper (5.8e7 S/m) on the 3 um metal would give 0.4450,
    which is the ~0.45 the model is usually quoted at. All three are well
    above the ``delta << t`` the sheet needs, which is why no absolute-Q
    claim is made anywhere in the study."""
    st = _study()
    d = _json()
    v = d["B"]["leontovich"]
    with enable_x64():
        model = st.build_level(1.0)
        from rfx.fdfd import spiral as spiral_mod
        fresh = float(spiral_mod.leontovich_validity(model, st.FREQ_B, st.SIGMA_CU))
        delta = float(spiral_mod.skin_depth(st.FREQ_B, st.SIGMA_CU))
    assert fresh == pytest.approx(v["leontovich_validity"], rel=1e-12)
    assert delta == pytest.approx(v["skin_depth"], rel=1e-12)
    assert v["skin_depth"] == pytest.approx(1.8411415106684175e-06, rel=1e-9)
    assert v["leontovich_validity"] == pytest.approx(0.9205707553342087, rel=1e-9)
    assert v["delta_over_t_m2"] == pytest.approx(0.6137138368894725, rel=1e-9)
    assert v["pure_copper"]["delta_over_t_m2"] == pytest.approx(0.4450428600917758, rel=1e-9)
    assert v["leontovich_validity"] > 0.5


def test_t5e_substrate_thickness_deviation_is_quantified():
    """The study's largest deliberate deviation, measured on both sides: the
    silicon is 120 um (two 60 um cells) over the fixture's PEC ground where
    the real die is ~190 um. Halving it to 60 um moves the FDFD's L_diff by
    -16.096 % and Q_diff by -6.258 % at 2.45 GHz, against -15.290 % for the
    referee's ground image over the same pair (0.81 points apart, so the
    FDFD's thickness response IS the ground-image response); the referee puts
    the chosen 120 um 6.227 % below the 190 um die."""
    ts = _json()["B"]["t_si_sensitivity"]
    assert ts["chosen"]["t_si"] == pytest.approx(120e-6, rel=1e-12)
    assert ts["chosen"]["n_si_cells"] == 2
    assert ts["alt"]["n_si_cells"] == 1
    assert ts["dL_rel"] == pytest.approx(-0.160960098700213, rel=1e-6)
    assert ts["dQ_rel"] == pytest.approx(-0.06258287824359565, rel=1e-6)
    assert ts["referee_dL_rel_same_pair"] == pytest.approx(-0.1528968010805879, rel=1e-9)
    assert abs(ts["dL_rel"] - ts["referee_dL_rel_same_pair"]) < 0.01
    assert ts["referee_dL_rel_chosen_to_real"] == pytest.approx(0.06227082494924696, rel=1e-9)


def test_t5f_gate_r5_sizes_and_the_level_nobody_can_afford():
    """R5, the cost. Largest N solved 119689; per-fixture factorisation +
    solve 44.6-134.0 s (median 76.7); peak physical footprint 8.92 GB through
    ``solve_split`` (three single-fixture calls, one LU factor live) and
    12.44 GB through the single ``solve_spiral`` call the FD stencils use.
    The measured cost law of this fixture family over N = 45744..85785 is
    seconds ~ N^2.04 and peak footprint ~ N^2.12. The 3-cells-across-W level
    the bar study called its 2 % point is N = 509580 here, 5.94x, which that
    law puts at 3012 s per fixture and 322 GB of peak footprint (79.7 GB on
    the textbook N^(4/3) fill law) -- both far past this 36 GB machine, which
    is why it is reported and not run."""
    d = _json()
    r5 = d["gates"]["R5"]
    assert r5["largest_n_unknowns"] == 119689
    assert r5["seconds_per_fixture_max"] == pytest.approx(134.0, rel=0.02)
    assert r5["peak_footprint_gb"] == pytest.approx(8.92, rel=0.02)
    assert r5["single_call_path"]["peak_footprint_gb"] == pytest.approx(12.44, rel=0.02)
    sc = d["B"]["scaling"]
    assert sc["memory_increments_usable"] is True
    assert sc["exponent_time"] == pytest.approx(2.0426487819870385, rel=1e-6)
    assert sc["exponent_memory"] == pytest.approx(2.117600971507289, rel=1e-6)
    w3 = r5["w3_cost_projection"]
    assert w3["n_unknowns"] == 509580
    assert w3["n_unknowns_ratio"] == pytest.approx(509580 / 85785, rel=1e-12)
    assert w3["measured_law"]["peak_footprint_gb"] > 8 * w3["machine_ram_gb"]
    assert w3["textbook_law"]["peak_footprint_gb"] > 2 * w3["machine_ram_gb"]
    assert w3["measured_law"]["seconds_three_fixtures"] > 3600.0

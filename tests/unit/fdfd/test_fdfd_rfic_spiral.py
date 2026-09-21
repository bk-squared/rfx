"""Gates for study R parts A and B, ``validation/fdfd/rfic_spiral.py``: the
paper-scale square spiral at 2.45 GHz on the level-invariant fixture with a
graded silicon, a Leontovich metal and a lossy substrate, solved through
cuDSS on the GPU (the VESSL lanes of ``validation/vessl/lane_r_rfic.py``).

The study's own levels are N = 106k-773k unknowns on a GPU; this file does
not re-run them. It re-runs, live on the CPU, everything that needs no GPU:

S1  the graded-substrate hook: the silicon z lines of the paper model are a
    geometric grading of neighbour ratio <= SI_GRADE pinned to the port-gap
    line, every level-1 z line is a level-2 z line (the levels stay nested),
    ``rfx.fdfd.spiral._stack_z_lines`` is restored after the build (also
    after an exception), and the unknown counts equal the JSON's.
S2  the pipeline on the small CPU model (``build_cheap``, same code path):
    every S-matrix reciprocal to 1e-8 and passive to 1 + 1e-8, and the
    ``jax.vjp`` gradients of L_diff and Q_diff against FD4 (1 % steps) to
    1e-4 in r_out and width.
S3  the analytic Leontovich statement and the referee curve recompute to the
    JSON, and the legacy referee the ladder imports is unchanged.
S4  the JSON's derived numbers re-derive from its raw records (R2 worst
    cases, R3 relative errors and the FD4 stencils, the thickness and
    grading changes, the self-resonance fit), and the gate verdicts.
S4b the CPU wall check and the fresh-mesh check: the models are rebuilt
    (walls and lid where the fixture puts them, unknown counts as in the
    JSON, the fresh mesh's walls at the deformed one's), and every derived
    change re-derives from the recorded solves; the 10 W CPU solves equal
    the GPU lanes' to 1e-9.
S5  the numbers quoted in the module docstring equal the JSON.
S6  (GPU only; skipped here) cuDSS equals SuperLU on the small model.

x64 is scoped per test (``tests._x64_compat.enable_x64``).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
from typing import Any

import numpy as np
import pytest

from tests._x64_compat import enable_x64

pytest.importorskip("shapely")

REPO = pathlib.Path(__file__).resolve().parents[3]
STUDY_PATH = REPO / "validation" / "fdfd" / "rfic_spiral.py"
JSON_PATH = REPO / "validation" / "fdfd" / "rfic_spiral.json"
LADDER_JSON = REPO / "validation" / "fdfd" / "invariant_ladder.json"

_CACHE: dict[str, Any] = {}


def _rs():
    if "rs" not in _CACHE:
        spec = importlib.util.spec_from_file_location("rfic_spiral_test", STUDY_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["rs"] = mod
    return _CACHE["rs"]


def _json() -> dict[str, Any]:
    if "json" not in _CACHE:
        assert JSON_PATH.exists(), f"{JSON_PATH} missing: run the study first"
        _CACHE["json"] = json.loads(JSON_PATH.read_text())
    return _CACHE["json"]


def _cheap():
    """Forward, gradients and FD4 on the small model; cached (~35 s)."""
    if "cheap" not in _CACHE:
        rs = _rs()
        with enable_x64():
            from rfx.fdfd.linear_solve import factor_cache_size
            old = factor_cache_size()
            factor_cache_size(0)
            try:
                m = rs.build_cheap()
                fwd = rs.forward(m)
                gr = rs.metrics_with_grads(m)
                fd = {k: rs.fd4_param(m, k) for k in (0, 2)}
            finally:
                factor_cache_size(old)
        _CACHE["cheap"] = (m, fwd, gr, fd)
    return _CACHE["cheap"]


# ----------------------------------------------------------------------------
# S1

def test_s1_graded_substrate_hook():
    """The graded silicon: neighbour ratio <= SI_GRADE (1 + 1e-9, float
    slack) through the graded run, the oxide's first cell to the first
    silicon cell too, the port-gap and silicon-top lines exact to 1e-12 m,
    level 1 nested in level 2 to 1e-12 m; the private builder restored."""
    rs = _rs()
    from rfx.fdfd import spiral as sm
    orig = sm._stack_z_lines
    with enable_x64():
        m1 = rs.build_r(1)
        m2 = rs.build_r(2)
        mu = rs.build_r(1, grade=None)
    assert sm._stack_z_lines is orig
    z1, z2, zu = (np.asarray(m.z) for m in (m1, m2, mu))
    tsi, pg = rs.T_SI_R, rs.PORT_GAP
    assert np.min(np.abs(z1 - tsi)) <= 1e-12 and np.min(np.abs(z1 - pg)) <= 1e-12
    graded = z1[(z1 >= pg - 1e-12) & (z1 <= tsi + 1e-12)]
    d = np.diff(graded)
    d_ox = rs.T_OX_LOW
    # pad_to's run (before _stack_z_lines splits any cell above base_dz):
    # the un-split cells shrink toward the oxide by at most SI_GRADE each
    top = np.concatenate([d, [d_ox]])[::-1]          # from the oxide cell downward
    ratios = top[1:] / top[:-1]
    assert np.all(ratios[:3] <= rs.SI_GRADE * (1 + 1e-9)), ratios
    assert np.all(d <= rs.BASE_DZ * (1 + 1e-12))
    # nesting: every level-1 line is a level-2 line
    assert max(np.min(np.abs(z2 - v)) for v in z1) <= 1e-12
    # the layer-uniform build has fewer, equal silicon cells above the port gap
    ug = zu[(zu >= pg - 1e-12) & (zu <= tsi + 1e-12)]
    assert np.allclose(np.diff(ug), np.diff(ug)[0], rtol=1e-12)
    assert len(ug) < len(graded)
    # restored after an exception inside the context too
    with pytest.raises(RuntimeError):
        with rs.graded_substrate():
            raise RuntimeError("inside")
    assert sm._stack_z_lines is orig
    st = _json()
    assert m1.n_unknowns == st["levels"]["1"]["grid"]["n_unknowns"]
    assert m2.n_unknowns == st["levels"]["2"]["grid"]["n_unknowns"]
    assert rs.grid_record(m2)["cells_across_width"] == st["levels"]["2"]["grid"]["cells_across_width"] == 4
    assert rs.grid_record(m1)["cells_across_width"] == 2


# ----------------------------------------------------------------------------
# S2

def test_s2_cheap_reciprocity_passivity():
    """Every S-matrix of the live small-model solve (the three solved
    fixtures and the de-embedded one): |S12 - S21| <= 1e-8 (measured
    ~1.2e-12) and the largest singular value <= 1 + 1e-8 (measured 0.9983:
    the lossy silicon and the sheet make it strictly passive)."""
    _, fwd, gr, _ = _cheap()
    rs = _rs()
    for rec in (fwd, gr):
        assert rec["reciprocity_worst"] <= rs.RECIP_TOL
        assert rec["passivity_worst"] <= rs.PASSIVITY_TOL
    assert fwd["L_diff"] == pytest.approx(gr["L_diff"], rel=1e-12)
    assert fwd["Q_diff"] == pytest.approx(gr["Q_diff"], rel=1e-10)


def test_s2_cheap_ad_vs_fd4():
    """``jax.vjp`` of L_diff and Q_diff against FD4 with 1 % steps on the
    small model, in r_out and width: <= 1e-4 (measured 1.8e-9 / 5.0e-10 in
    width; the 1 % step moves L by ~1 %, six decades above the LU noise)."""
    _, _, gr, fd = _cheap()
    for k, rec in fd.items():
        for q, key in (("L", "dL"), ("Q", "dQ")):
            rel = abs(gr[key][k] - rec[f"fd_{q}"]) / abs(rec[f"fd_{q}"])
            assert rel <= 1e-4, (k, q, rel)


# ----------------------------------------------------------------------------
# S3

def test_s3_leontovich_and_referee_recompute():
    """The analytic Leontovich block and the referee curve recompute to the
    JSON to 1e-12; the slab ratio has its two limits (t >> delta -> 1,
    t << delta -> 2 delta / t to 1e-3); the LEGACY referee the ladder's
    ``paper_referee`` imports (ground 131.54 um) still gives the ladder's
    3560.37 pH to 1e-12."""
    rs = _rs()
    st = _json()
    lb = rs.leontovich_block()
    assert lb["skin_depth"] == pytest.approx(st["leontovich"]["skin_depth"], rel=1e-12)
    for k in ("M2", "M1"):
        for kk in ("R_exact_over_R_sheet", "X_exact_over_X_sheet"):
            assert lb["slab_1d"][k][kk] == pytest.approx(st["leontovich"]["slab_1d"][k][kk],
                                                         rel=1e-12)
    big = rs.slab_ratio(100e-6, 1e-6)
    assert big["R_exact_over_R_sheet"] == pytest.approx(1.0, abs=1e-12)
    small = rs.slab_ratio(1e-6, 100e-6)
    assert small["R_exact_over_R_sheet"] == pytest.approx(small["R_dc_over_R_sheet"], rel=1e-3)
    ref = rs.referee_curve()
    for k, v in st["referee"].items():
        if isinstance(v, dict):
            assert ref[k]["deembedded"] == pytest.approx(v["deembedded"], rel=1e-12)
    il = json.loads(LADDER_JSON.read_text())
    sg, cv = rs.load_referee(), rs.bind_referee()
    assert rs.GROUND_H == pytest.approx(131.54e-6, rel=1e-12)
    assert rs.referee_deembedded(sg, cv, rs.THETA0) == pytest.approx(
        il["paper"]["referee"]["total"], rel=1e-12)


# ----------------------------------------------------------------------------
# S4

def test_s4_json_rederives():
    """R2's worst cases from the per-solve table, R3 from AD and the FD4
    stencils recomputed from their four logged points, the thickness and
    grading changes and the self-resonance fit from their records -- all
    to 1e-12 relative."""
    rs = _rs()
    st = _json()
    g = st["gates"]
    ps = g["R2"]["per_solve"]
    assert g["R2"]["reciprocity_worst"] == max(v["reciprocity"] for v in ps.values())
    assert g["R2"]["passivity_worst"] == max(v["passivity"] for v in ps.values())
    for name, rec in st["fd"].items():
        h = rec["step"]
        for q in ("L", "Q"):
            v = {int(m): p[f"{q}_diff"] for m, p in rec["points"].items()}
            fd = (-v[2] + 8 * v[1] - 8 * v[-1] + v[-2]) / (12 * h)
            assert fd == pytest.approx(rec[f"fd_{q}"], rel=1e-12)
            ad = st["gradients"][f"d{q}"][rec["k"]]
            assert g["R3"]["checked"][f"d{q}_d{name}"]["rel"] == pytest.approx(
                abs(ad - fd) / abs(fd), rel=1e-9)
    l1 = st["tsi_sensitivity"]["level1"]
    kc = f"{rs.T_SI_R * 1e6:g}"
    for k, row in l1.items():
        assert row["dL_rel_to_chosen"] == pytest.approx(row["L_diff"] / l1[kc]["L_diff"] - 1,
                                                        rel=1e-12, abs=1e-15)
    sw = st["sweep"]
    up = [i for i, f in enumerate(sw["freqs_ghz"]) if f * 1e9 >= rs.FREQ - 1]
    fit = rs.srf_fit([sw["freqs_ghz"][i] * 1e9 for i in up], [sw["L_diff"][i] for i in up])
    assert fit["f_self_resonance"] == pytest.approx(sw["srf_fit_upper"]["f_self_resonance"],
                                                    rel=1e-12)


def test_s4_gate_verdicts():
    """The recorded verdicts and their headline numbers (hard values)."""
    st = _json()
    g = st["gates"]
    assert g["R2"]["passed"] is True
    assert g["R2"]["reciprocity_worst"] <= 1e-8
    assert g["R2"]["passivity_worst"] <= 1 + 1e-8
    assert g["R3"]["passed"] is True
    assert g["R3"]["n_checked"] >= 2
    assert g["R3"]["worst_rel"] <= 1e-4


# ----------------------------------------------------------------------------
# S4b

def test_s4b_walls_and_remesh_models():
    """The wall-check models (5 / 10 / 20 W) have their x walls at
    +-(r_out + w W) and the lid w W above the stack to 1e-12 m, with the
    JSON's unknown counts; the fresh mesh at theta* has the walls and lid of
    the nominal 10 W model to 1e-12 m, 3 cells across its 22 um strip and
    the JSON's unknown count."""
    rs = _rs()
    st = _json()
    cases = st["walls_cpu"]["cases"]
    with enable_x64():
        like = None
        for w in sorted({c["wall_w"] for c in cases.values()}):
            m = rs.build_r(1, wall_w=w)
            x, z = np.asarray(m.x_nom), np.asarray(m.z)
            assert x[-1] == pytest.approx(rs.R_OUT + w * rs.WIDTH, abs=1e-12)
            assert x[0] == pytest.approx(-(rs.R_OUT + w * rs.WIDTH), abs=1e-12)
            assert z[-1] == pytest.approx(m.spec.stack.z_top + w * rs.WIDTH, abs=1e-12)
            for c in cases.values():
                if c["wall_w"] == w:
                    assert c["grid"]["n_unknowns"] == m.n_unknowns
            if w == rs.WALL_W:
                like = m
        assert like is not None
        th = rs.wall_check_points()["design"]
        fr = rs.build_fresh(th, like)
    g = rs.grid_record(fr)
    gl = rs.grid_record(like)
    for k in ("wall_x", "lid_z"):
        assert np.allclose(g[k], gl[k], rtol=0, atol=1e-12)
    assert g["wall_y"][1] == pytest.approx(gl["wall_y"][1], abs=1e-12)
    assert g["cells_across_width"] == 3
    assert g["n_unknowns"] == st["remesh_cpu"]["fresh_design_w10"]["grid"]["n_unknowns"]


def test_s4b_walls_and_remesh_rederive():
    """Every relative change of ``walls`` and ``remesh`` from its two
    records to 1e-12; the 10 W CPU (SuperLU) solves against the GPU (cuDSS)
    lanes' records of the same model and theta: <= 1e-9 relative on L and
    Q (the two backends factor the same matrix; measured in the JSON)."""
    st = _json()
    cases = st["walls_cpu"]["cases"]
    wb = st["walls"]
    for p, blk in wb["per_point"].items():
        rows = {c["wall_w"]: c for c in cases.values() if c["point"] == p}
        for a, b in ((10.0, 20.0), (5.0, 10.0)):
            if a in rows and b in rows:
                assert blk[f"dL_rel_{a:g}_to_{b:g}W"] == pytest.approx(
                    rows[b]["L_diff"] / rows[a]["L_diff"] - 1, rel=1e-12)
                assert blk[f"dQ_rel_{a:g}_to_{b:g}W"] == pytest.approx(
                    rows[b]["Q_diff"] / rows[a]["Q_diff"] - 1, rel=1e-12)
    for k in ("nominal_level1", "design_level1"):
        xb = wb["cross_backend_10W"][k]
        assert abs(xb["rel_L"]) <= 1e-9 and abs(xb["rel_Q"]) <= 1e-9, (k, xb)
    rm = st["remesh"]
    assert rm["dL_rel_fresh_vs_deformed"] == pytest.approx(
        rm["fresh"]["L_diff"] / rm["deformed"]["L_diff"] - 1, rel=1e-12)
    assert rm["deformed"]["L_diff"] == cases["design_w10"]["L_diff"]
    for c in list(cases.values()) + [st["remesh_cpu"]["fresh_design_w10"]]:
        assert c["reciprocity_worst"] <= 1e-8 and c["passivity_worst"] <= 1 + 1e-8
    ref = st["referee"]["190"]["deembedded"]
    for k, v in st["accuracy"]["own_referee_190um"]["gap_by_level"].items():
        assert v == pytest.approx(st["levels"][k]["L_diff"] / ref - 1, rel=1e-12)


# ----------------------------------------------------------------------------
# S5

def test_s5_docstring_numbers_equal_json():
    """Every headline number of the module docstring, re-formatted from the
    JSON, appears verbatim in it."""
    rs = _rs()
    doc = re.sub(r"\s+", " ", rs.__doc__)
    st = _json()
    for s in docstring_numbers(st):
        assert s in doc, s


def docstring_numbers(st: dict[str, Any]) -> list[str]:
    lv = st["levels"]
    g = st["gates"]
    sw = st["sweep"]
    ts = st["tsi_sensitivity"]
    gr = st["gradients"]
    le = st["leontovich"]
    out = [f"{lv['1']['L_diff'] * 1e12:.2f} pH", f"{lv['2']['L_diff'] * 1e12:.2f} pH",
           f"{lv['1']['Q_diff']:.3f}", f"{lv['2']['Q_diff']:.3f}",
           f"{lv['1']['L_se'] * 1e12:.2f} pH", f"{lv['1']['Q_se']:.3f}",
           f"{lv['2']['L_se'] * 1e12:.2f} pH", f"{lv['2']['Q_se']:.3f}",
           f"N = {lv['1']['grid']['n_unknowns']}", f"N = {lv['2']['grid']['n_unknowns']}",
           f"{100 * (lv['2']['L_diff'] / lv['1']['L_diff'] - 1):.2f} %",
           f"{g['R2']['reciprocity_worst']:.2e}", f"{g['R2']['passivity_worst']:.5f}",
           f"{g['R2']['n_solves']} solves", f"({g['R2']['n_s_matrices']}:",
           f"{g['R3']['worst_rel']:.2e}",
           f"{le['skin_depth'] * 1e6:.3f} um", f"{le['delta_over_t']['M2']:.3f}",
           f"{le['delta_over_t']['M1']:.3f}",
           f"x {le['slab_1d']['M2']['R_exact_over_R_sheet']:.3f}",
           f"x {le['slab_1d']['M1']['R_exact_over_R_sheet']:.3f}",
           f"{sw['srf_fit_upper']['f_self_resonance'] / 1e9:.2f} GHz",
           f"{sw['srf_fit_upper']['L0'] * 1e12:.2f} pH",
           f"{100 * ts['resolved']['dL_rel']:.2f} %", f"{100 * ts['resolved']['dQ_rel']:.2f} %",
           f"{100 * ts['resolved']['referee_dL_rel']:.2f} %",
           f"{100 * ts['level1']['120']['dL_rel_to_chosen']:+.2f} %",
           f"{100 * ts['level1']['300']['dL_rel_to_chosen']:+.2f} %",
           f"{100 * ts['level1']['120']['referee_dL_rel_to_chosen']:+.2f} %",
           f"{100 * ts['level1']['300']['referee_dL_rel_to_chosen']:+.2f} %",
           f"{100 * st['grading']['uniform']['dL_rel_to_chosen']:+.2f} %".replace("-", ""),
           f"{st['plans']['190:1.5:2']['permanent_device_memory_gb']:.2f} GB",
           f"{st['plans']['190:2:2']['permanent_device_memory_gb']:.2f} GB",
           f"{st['plans']['190:2:3']['permanent_device_memory_gb']:.2f} GB",
           f"{gr['dL'][0]:.4e}", f"{gr['dL'][1]:.4e}", f"{gr['dL'][2]:.4e}",
           f"{gr['dQ'][0]:.1f}", f"{gr['dQ'][1]:+.1f}", f"{gr['dQ'][2]:.1f}"]
    wp = st["walls"]["per_point"]
    rm = st["remesh"]
    ac = st["accuracy"]
    out += [f"{100 * wp['nominal']['dL_rel_10_to_20W']:+.2f} %",
            f"{100 * wp['nominal']['dL_rel_5_to_10W']:+.2f} %",
            f"{100 * wp['design']['dL_rel_10_to_20W']:+.2f} %",
            f"{100 * wp['r_out_hi']['dL_rel_10_to_20W']:+.2f} %",
            f"{100 * wp['nominal']['dQ_rel_10_to_20W']:+.2f} %",
            f"{100 * st['walls']['ladder_paper_protocol']['dL_rel_10_to_20W']:+.2f} %",
            f"{wp['design']['wall_x_minus_r_out']['10'] * 1e6:.1f} um",
            f"{wp['r_out_hi']['wall_x_minus_r_out']['10'] * 1e6:.1f} um",
            f"{100 * rm['dL_rel_fresh_vs_deformed']:+.2f} %",
            f"{100 * rm['dQ_rel_fresh_vs_deformed']:+.2f} %",
            f"N = {rm['fresh']['n_unknowns']}",
            f"{100 * rm['fraction_of_level_step_L']:.1f} %",
            f"{100 * rm['fraction_of_level_step_Q']:.1f} %",
            f"{100 * rm['level1_to_2_at_theta_star']['dL_rel']:+.2f} %",
            f"{100 * rm['level1_to_2_at_theta_star']['dQ_rel']:+.2f} %",
            f"{rm['fresh']['L_diff'] * 1e12:.2f} against {rm['deformed']['L_diff'] * 1e12:.2f} pH",
            f"{100 * wp['nominal']['tail_beyond_20W_geometric']:.2f} %",
            f"{g['R2']['reciprocity_worst_gpu']:.2e}",
            f"{-100 * ac['own_referee_190um']['gap_by_level']['1']:.2f} % / "
            f"{-100 * ac['own_referee_190um']['gap_by_level']['2']:.2f} % low",
            f"{ac['own_referee_190um']['referee_deembedded'] * 1e12:.2f} pH"]
    for f, lval, q in zip(sw["freqs_ghz"], sw["L_diff"], sw["Q_diff"]):
        out += [f"{lval * 1e12:.2f}", f"{q:.3f}"]
    for name, v in g["R3"]["checked"].items():
        out.append(f"{name.replace('_d', '/d', 1)} {v['rel']:.2e}")
    return [re.sub(r"\s+", " ", s) for s in out]


# ----------------------------------------------------------------------------
# S6

def test_s6_cudss_equals_superlu_on_cheap_model():
    """GPU only: one forward solve of the small model through cuDSS equals
    SuperLU's to 1e-8 relative on L_diff and Q_diff (skipped without cuDSS)."""
    import rfx.fdfd.linear_solve as ls
    if not ls.backend_available("cudss"):
        pytest.skip("cuDSS not available on this machine")
    rs = _rs()
    m, fwd, _, _ = _cheap()
    with enable_x64():
        with ls.default_backend("cudss"):
            r = rs.forward(m)
    assert r["L_diff"] == pytest.approx(fwd["L_diff"], rel=1e-8)
    assert r["Q_diff"] == pytest.approx(fwd["Q_diff"], rel=1e-8)

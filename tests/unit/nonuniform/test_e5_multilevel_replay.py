"""E5 replay — multi-level unequal bands on x, y, z (Lane 1), chain model
only, no FDTD.

Replays ``validation/research/multiband_nu/results/e5_model.json`` (the
frozen predictions), ``e5_z.json`` / ``e5_x.json`` / ``e5_y.json`` (the
measured cells), ``e5_relabel.json`` (L1-R) and ``e5_pinbridge.json``
(W5 pin bridge) written by ``e5_multilevel_axes.py`` under
``docs/design_notes/20260913_nu_lane1_multilevel_xyz_predeclaration.md``:

* model side: every (pattern, axis) cell regenerates through
  ``model_cell`` (same dt, n_steps, layouts, declared vectors to 1e-12 m,
  builder flag, ``R_model`` to 1e-8, windows, gate geometry to 1e-9
  relative, law side, ``c_model`` re-fit to 1e-9 m); independently, every
  stored profile's structure slice is re-solved by ``chain_model.
  scattering`` and must return the stored ``R_model``;
* results side: every W1 / W2 / W3 verdict, the floor-rule flag, the
  cell conjunctions, the W4 control rows against E1's JSON, the W5
  relabel comparison from the stored traces and the pin-bridge rows are
  recomputed from the stored numbers with the module's own judge
  functions and must equal the recorded ones — fired verdicts included.
  Instrument preconditions (``dt_matches_b``, ``source_probe_in_lead``,
  ``lead_f32_identical``, ``builder_matches_declared_vector``) are
  asserted true;
* fixture identity (no FDTD): the three axis fixtures of one profile have
  the same dt and bit-equal float32 inverse-spacing arrays.

Tolerances, declared before the first run of this file: chain floats
1e-8 relative (the E1 replay bound re-declared 2026-09-11); cells 1e-12 m;
``c`` re-fits 1e-9 m; gate times 1e-9 relative; W4 control 1e-6 (R_meas)
and 1e-8 (R_model) relative; W5 1e-6 relative; pin bridge 1.5e-4 absolute.
Files not present are skipped, so the test is meaningful at every commit
of the lane. Runtime: 19 cell regenerations (~0.4 s each) plus stored
re-solves — about 20 s.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from validation.research.multiband_nu import e5_multilevel_axes as e5
from validation.research.multiband_nu import w6_band_builder as w6
from validation.research.multiband_nu.chain_model import scattering

_REPO = pathlib.Path(__file__).resolve().parents[3]
_RESULTS = _REPO / "validation/research/multiband_nu/results"
_MODEL = _RESULTS / "e5_model.json"
_AXIS = {a: _RESULTS / f"e5_{a}.json" for a in ("z", "x", "y")}
_RELABEL = _RESULTS / "e5_relabel.json"
_PINBRIDGE = _RESULTS / "e5_pinbridge.json"
_E1 = _RESULTS / "e1_band_law_sweep.json"

REL = 1e-8
CELL_TOL = 1e-12
C_TOL_M = 1e-9
GATE_REL = 1e-9


def _load(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    with open(path) as fh:
        return json.load(fh)


def _rel_close(x, y, rel=REL):
    return abs(x - y) <= rel * max(abs(x), abs(y))


@pytest.fixture(scope="module")
def model_json():
    return _load(_MODEL)


def _model_cells():
    if not _MODEL.exists():
        return []
    with open(_MODEL) as fh:
        m = json.load(fh)
    return [(ax, p) for ax in m["axes"] for p in m["axes"][ax]]


def _axis_cells(axis):
    p = _AXIS[axis]
    if not p.exists():
        return []
    with open(p) as fh:
        return sorted(json.load(fh)["cells"])


# --- helpers shared by model and results replays ----------------------------------------

def _arms(cell):
    return list(cell["singles"].values()) + cell["bands"]


def _check_arm_model(rec, dt, key):
    prof = e5.unrle(rec["profile_rle"])
    assert len(prof) == rec["n_cells"]
    assert rec["builder_matches_declared_vector"] is True, key
    assert rec["max_abs_cell_dev_m"] <= CELL_TOL
    lay = rec["layout"]
    struct = prof[rec["m_lo"]:len(prof) - rec["m_hi"]]
    assert len(struct) == rec["n_struct_cells"]
    r = abs(scattering(struct, lay["n_lead"], lay["n_tail"], w6.F0, dt, e5.D_T, e5.fx.B_Y)[0])
    assert _rel_close(r, rec["R_model"]), (key, r, rec["R_model"])
    half = e5.w1_window(rec["R_model"])
    assert _rel_close(rec["half"], half)
    assert _rel_close(rec["window"][0], rec["R_model"] - half)
    assert _rel_close(rec["window"][1], rec["R_model"] + half)
    assert rec["k_src_used"] == rec["m_lo"] + lay["k_src"]
    assert rec["k_prb_used"] == rec["m_lo"] + lay["k_prb"]
    # layout is the E1 1c re-cut of the physical planes in the lead cell
    regen = e5.layout(lay["lead_cell_m"], lay["tail_cell_m"])
    assert regen == lay
    assert all(v > 0 for v in rec["gate_margins_ns"].values()) and rec["gates_hold"] is True, key
    assert rec["max_adjacent_ratio"] <= e5.CAP + 1e-9


def _check_cell_against_regen(cell, key):
    """Regenerate the cell through the instrument (no FDTD) and compare."""
    spec = e5.AXES[cell["axis"]]
    regen = e5.model_cell(cell["pattern"], spec, cell["pinned"])
    assert _rel_close(regen["dt_s"], cell["dt_s"], 1e-12)
    assert regen["n_steps"] == cell["n_steps"]
    assert regen["layout_band"] == cell["layout_band"]
    for name in cell["singles"]:
        a, b = regen["singles"][name], cell["singles"][name]
        _compare_arm(a, b, f"{key}/{name}")
    assert [r["name"] for r in regen["bands"]] == [r["name"] for r in cell["bands"]]
    for a, b in zip(regen["bands"], cell["bands"]):
        _compare_arm(a, b, f"{key}/{b['name']}")
    for name in cell["mirrors"]:
        assert _rel_close(regen["mirrors"][name]["R_model"], cell["mirrors"][name]["R_model"])
    law_r, law_c = regen["law"], cell["law"]
    assert _rel_close(law_r["bound_sum"], law_c["bound_sum"])
    if cell["pattern"] in e5.FP_PATTERNS:
        for k in ("R_L", "R_R", "k_g_fine_per_m", "chain_max_over_sum", "chain_min_0_80"):
            assert _rel_close(law_r[k], law_c[k]), (key, k)
        assert abs(law_r["c_model_m"] - law_c["c_model_m"]) <= C_TOL_M
        assert abs(law_r["c_closed_form_m"] - law_c["c_closed_form_m"]) <= 1e-12
        assert _rel_close(law_c["c_window_m"], e5.E1_C_TOL_DR * e5.DR)
        assert len(law_c["chain_sweep_R"]) == e5.E1_LAW_SWEEP_MAX + 1
        for a, b in zip(law_r["chain_sweep_R"], law_c["chain_sweep_R"]):
            assert _rel_close(a, b)
        # c_model re-fit from the STORED sweep with the stored amplitudes
        c = e5.fit_c_two_amp(list(range(len(law_c["chain_sweep_R"]))), law_c["chain_sweep_R"], e5.D_F,
                             law_c["R_L"], law_c["R_R"], law_c["k_g_fine_per_m"], e5.DR)
        assert abs(c - law_c["c_model_m"]) <= C_TOL_M, (key, c, law_c["c_model_m"])
        assert max(law_c["chain_sweep_R"]) <= (law_c["R_L"] + law_c["R_R"]) * (1 + 1e-9)
        assert _rel_close(law_c["bound_sum"], (1 + e5.E1_BOUND_SLACK) * (law_c["R_L"] + law_c["R_R"]))
    else:
        for k in law_c["amplitudes"]:
            assert _rel_close(law_r["amplitudes"][k], law_c["amplitudes"][k])
        assert _rel_close(law_c["bound_sum"], (1 + e5.E1_BOUND_SLACK) * sum(law_c["amplitudes"].values()))
    assert regen["in_accuracy_class"] == cell["in_accuracy_class"]
    for k, b in cell["b_refs_model"].items():
        assert b["builder_matches_declared_vector"] is True
        assert np.max(np.abs(e5.unrle(regen["b_refs_model"][k]["profile_rle"]) - e5.unrle(b["profile_rle"]))) <= CELL_TOL


def _compare_arm(a, b, key):
    pa, pb = e5.unrle(a["profile_rle"]), e5.unrle(b["profile_rle"])
    assert len(pa) == len(pb) and np.max(np.abs(pa - pb)) <= CELL_TOL, key
    assert (a["m_lo"], a["m_hi"]) == (b["m_lo"], b["m_hi"])
    assert a["layout"] == b["layout"]
    assert _rel_close(a["R_model"], b["R_model"]), (key, a["R_model"], b["R_model"])
    for k, v in b["gates_ns"].items():
        assert _rel_close(a["gates_ns"][k], v, GATE_REL) or abs(a["gates_ns"][k] - v) <= 1e-12, (key, k)
    for k, v in b["gate_margins_ns"].items():
        assert abs(a["gate_margins_ns"][k] - v) <= GATE_REL * max(1.0, abs(v)), (key, k)
    assert a["n_gate"] == b["n_gate"] and a["n_inc"] == b["n_inc"]
    if "fp_form_model" in b:
        assert _rel_close(a["fp_form_model"], b["fp_form_model"])


# --- model side ----------------------------------------------------------------------------

def test_model_provenance(model_json):
    assert model_json["rfx_file"].endswith("rfx/__init__.py")
    assert len(model_json["git_sha"]) == 40
    assert model_json["model_only"] is True
    assert model_json["windows"]["W1_per_arm"] == "|R_meas - R_model| <= 0.20 R_model + 3e-5"
    assert model_json["windows"]["W3_c_window_m"] == pytest.approx(0.14e-3)
    assert set(model_json["axes"]) == {"z", "x", "y", "z_pinned"}
    for ax in ("z", "x", "y"):
        assert list(model_json["axes"][ax]) == list(e5.PATTERNS)


@pytest.mark.parametrize("axis,pattern", _model_cells())
def test_model_cell_arms_resolve_from_stored_profiles(model_json, axis, pattern):
    cell = model_json["axes"][axis][pattern]
    assert cell["pinned"] is (axis != "z")
    for name, rec in cell["singles"].items():
        _check_arm_model(rec, cell["dt_s"], f"{axis}/{pattern}/{name}")
    for rec in cell["bands"]:
        _check_arm_model(rec, cell["dt_s"], f"{axis}/{pattern}/{rec['name']}")
        assert _rel_close(rec["bound_sum"], cell["law"]["bound_sum"])
    assert min(min(r["gate_margins_ns"].values()) for r in _arms(cell)) >= e5.W6_EXPECT_MIN_MARGIN_NS - 1e-6


@pytest.mark.parametrize("axis,pattern", _model_cells())
def test_model_cell_regenerates(model_json, axis, pattern):
    _check_cell_against_regen(model_json["axes"][axis][pattern], f"{axis}/{pattern}")


def test_model_predictions_identical_across_axes(model_json):
    z = model_json["axes"]["z"]
    for ax in ("x", "y"):
        for p in e5.PATTERNS:
            for name in z[p]["singles"]:
                assert _rel_close(model_json["axes"][ax][p]["singles"][name]["R_model"], z[p]["singles"][name]["R_model"])
            for a, b in zip(model_json["axes"][ax][p]["bands"], z[p]["bands"]):
                assert _rel_close(a["R_model"], b["R_model"])
            assert _rel_close(model_json["axes"][ax][p]["dt_s"], z[p]["dt_s"], 1e-12)
            assert model_json["axes"][ax][p]["n_steps"] == z[p]["n_steps"]


def test_model_s_cell_is_the_e1_control(model_json):
    e1 = _load(_E1)["cells"]["N30_r1.4"]
    s = model_json["axes"]["z"]["S"]
    assert s["layout_band"]["k_src"] == 85 and s["layout_band"]["k_prb"] == 100
    assert (s["layout_band"]["n_lead"], s["layout_band"]["n_tail"], s["layout_band"]["n_B"]) == (140, 150, 400)
    assert s["n_steps"] == 1200 and _rel_close(s["dt_s"], e1["model"]["dt_s"], 1e-12)
    assert _rel_close(s["singles"]["L"]["R_model"], e1["model"]["R_single_model"])
    e1_rows = {r["n_b"]: r for r in e1["model"]["rows"]}
    for rec in s["bands"]:
        assert _rel_close(rec["R_model"], e1_rows[rec["n_b"]]["R_model"])
    assert abs(s["law"]["c_model_m"] - e1["model"]["c_model_m"]) <= C_TOL_M
    # the two-amplitude fit reduces to E1's one-amplitude fit on E1's own sweep
    sweep = e1["model"]["chain_sweep_R"]
    c2 = e5.fit_c_two_amp(list(range(len(sweep))), sweep, e5.D_F, e1["model"]["R_single_model"],
                          e1["model"]["R_single_model"], e1["model"]["k_g_fine_per_m"], e5.DR)
    assert abs(c2 - e1["model"]["c_model_m"]) <= C_TOL_M


def test_fixture_relabel_identity_no_fdtd():
    segs = [e5._seg_lead(e5.DC), e5.Seg("band", e5.D_F, 4, True), e5._seg_tail(e5.DC, False)]
    prof = e5.build_declared(segs)["profile"]
    grids = {ax: e5.build_pec_fixture_axis(prof, e5.AXES[ax])[0] for ax in ("z", "x", "y")}
    assert (grids["z"].nx, grids["z"].ny, grids["z"].nz) == (4, 21, len(prof) + 1)
    assert (grids["x"].nx, grids["x"].ny, grids["x"].nz) == (len(prof) + 1, 4, 21)
    assert (grids["y"].nx, grids["y"].ny, grids["y"].nz) == (21, len(prof) + 1, 4)
    for ax in ("x", "y"):
        assert float(grids[ax].dt) == float(grids["z"].dt)
        ia, iah = e5.AXES[ax].inv_arrays(grids[ax])
        iz, izh = e5.AXES["z"].inv_arrays(grids["z"])
        assert np.array_equal(np.asarray(ia), np.asarray(iz)) and np.array_equal(np.asarray(iah), np.asarray(izh))


# --- results side ------------------------------------------------------------------------------

@pytest.mark.parametrize("axis", ["z", "x", "y"])
def test_results_provenance(axis):
    res = _load(_AXIS[axis])
    assert res["rfx_file"].endswith("rfx/__init__.py")
    assert len(res["git_sha"]) == 40
    assert res["model_only"] is False and res["axis"] == axis
    assert res["pinned"] is (axis != "z")


@pytest.mark.parametrize("axis,pattern", [(ax, p) for ax in ("z", "x", "y") for p in _axis_cells(ax)])
def test_results_cell_replays_verdicts(model_json, axis, pattern):
    res = _load(_AXIS[axis])
    cell = res["cells"][pattern]
    frozen = model_json["axes"][axis][pattern]
    key = f"{axis}/{pattern}"
    assert _rel_close(cell["dt_s"], frozen["dt_s"], 1e-12) and cell["n_steps"] == frozen["n_steps"]
    # the results JSON carries the frozen predictions
    for name in frozen["singles"]:
        assert _rel_close(cell["singles"][name]["R_model"], frozen["singles"][name]["R_model"])
    for a, b in zip(cell["bands"], frozen["bands"]):
        assert a["name"] == b["name"] and _rel_close(a["R_model"], b["R_model"])
    if pattern in e5.FP_PATTERNS:
        assert abs(cell["law"]["c_model_m"] - frozen["law"]["c_model_m"]) <= C_TOL_M
    assert _rel_close(cell["law"]["bound_sum"], frozen["law"]["bound_sum"])
    # per-arm verdicts from the stored numbers with the module's judge
    for rec in _arms(cell):
        _check_arm_model(rec, cell["dt_s"], key)
        m = rec["meas"]
        v = e5.w1_verdict(m["R_meas"], rec["R_model"])
        assert m["fired"] is v["fired"] and m["fired_at_floor"] is v["fired_at_floor"], (key, rec.get("name"))
        assert _rel_close(m["deviation"], v["deviation"], 1e-6) and _rel_close(m["half"], v["half"])
        assert m["dt_matches_b"] is (m["dt_diff_s"] < 1e-20)
        assert m["dt_matches_b"] is True and m["source_probe_in_lead"] is True, (key, rec.get("name"))
        assert m["lead_f32_identical"] is True and m["lead_f32_slice"] == rec["m_lo"] + rec["layout"]["n_lead"]
        assert m["gates_hold"] is (rec["gates_hold"] and m["dt_matches_b"] and m["source_probe_in_lead"]
                                   and m["lead_f32_identical"])
        if rec["kind"] == "band":
            assert _rel_close(m["bound_sum"], cell["law"]["bound_sum"])
            assert m["bound_fired"] is (m["R_meas"] > m["bound_sum"])
    # cell conjunctions and the c fit
    v = e5.cell_verdicts(cell)
    stored = cell["verdicts"]
    for k in ("any_w1_fired", "any_w2_fired", "all_gates_hold", "w3_fired", "in_law_domain",
              "w1_fired_arms", "w1_fired_at_floor_arms"):
        assert v[k] == stored[k], (key, k, v[k], stored[k])
    if pattern in e5.FP_PATTERNS:
        assert abs(v["c_meas_m"] - stored["c_meas_m"]) <= C_TOL_M, (key, v["c_meas_m"], stored["c_meas_m"])
        assert _rel_close(stored["c_window_m"], e5.E1_C_TOL_DR * e5.DR)
        assert stored["w3_fired"] is (abs(stored["c_meas_m"] - stored["c_model_m"]) > stored["c_window_m"])
    # every B reference: declared vector, dt equal to the cell's
    for b in cell["b_refs"].values():
        assert b["builder_matches_declared_vector"] is True
        assert abs(b["dt_s"] - cell["dt_s"]) < 1e-20


def test_results_w4_control_against_e1():
    res = _load(_AXIS["z"])
    if "S" not in res["cells"]:
        pytest.skip("S not measured")
    e1 = _load(_E1)
    regen = e5.w4_control(res["cells"]["S"], e1)
    stored = res["w4_control"]
    assert len(regen["rows"]) == len(stored["rows"]) == 6
    for a, b in zip(regen["rows"], stored["rows"]):
        assert a["arm"] == b["arm"]
        assert a["R_meas"] == b["R_meas"] and a["R_meas_e1"] == b["R_meas_e1"]
        assert a["meas_fired"] is b["meas_fired"] and a["model_fired"] is b["model_fired"]
        assert b["meas_fired"] is (b["meas_rel"] > e5.W4_MEAS_REL)
        assert b["model_fired"] is (b["model_rel"] > e5.W4_MODEL_REL)
    assert regen["any_fired"] is stored["any_fired"]
    # the control is a precondition of the lane: it held, or the lane stopped
    assert stored["any_fired"] is False
    assert stored["c_meas_diff_m"] <= C_TOL_M and stored["c_model_diff_m"] <= C_TOL_M


def test_results_relabel_replays():
    rel = _load(_RELABEL)
    arms = rel["arms"]
    assert set(arms) == {"z", "x", "y"}
    for ax, a in arms.items():
        assert a["comp"] == e5.AXES[ax].comp
        prof = e5.unrle(a["a_profile_rle"])
        lay = e5.layout(e5.DC, e5.DC)
        r = abs(scattering(prof, lay["n_lead"], lay["n_tail"], w6.F0, a["dt_s"], e5.D_T, e5.fx.B_Y)[0])
        assert _rel_close(r, a["R_model"])
        v = e5.w1_verdict(a["R_meas"], a["R_model"])
        assert a["fired"] is v["fired"]
        assert a["a_builder_exact"] is True and a["b_builder_exact"] is True
        assert abs(a["dt_s"] - a["dt_b_s"]) < 1e-20
        b_sym = e5.unrle(a["b_sym_profile_rle"])
        assert abs(b_sym[0] - e5.DC) <= CELL_TOL and abs(b_sym[-1] - e5.DC) <= CELL_TOL
    for ax in ("x", "y"):
        regen = e5.relabel_compare(arms["z"], arms[ax])
        stored = rel["compare"][ax]
        for k in ("trace_a_rel_maxdiff", "trace_b_rel_maxdiff", "R_meas_rel_diff"):
            assert _rel_close(regen[k], stored[k], 1e-9) or abs(regen[k] - stored[k]) <= 1e-15
        assert regen["w5_fired"] is stored["w5_fired"]
        assert stored["w5_fired"] is (stored["R_meas_rel_diff"] > e5.W5_REL
                                      or stored["trace_a_rel_maxdiff"] > e5.W5_REL)
    if "z_bridge" in rel:
        zb = rel["z_bridge"]
        assert _rel_close(zb["rel_diff"], abs(zb["R_meas_b_sym"] - zb["R_meas_e1_b"]) / zb["R_meas_e1_b"], 1e-9)


def test_results_pin_bridge_replays():
    pb = _load(_PINBRIDGE)
    zres = _load(_AXIS["z"])
    unp = {a["n_b"]: a for a in zres["cells"]["S"]["bands"]}
    cell = pb["cell"]
    assert cell["pinned"] is True and cell["axis"] == "z"
    pinned = {a["n_b"]: a for a in cell["bands"]}
    assert [r["n_b"] for r in pb["rows"]] == list(e5.WIDTHS)
    for r in pb["rows"]:
        assert r["R_meas_pinned"] == pinned[r["n_b"]]["meas"]["R_meas"]
        assert r["R_meas_unpinned"] == unp[r["n_b"]]["meas"]["R_meas"]
        d = abs(r["R_meas_pinned"] - r["R_meas_unpinned"])
        assert _rel_close(r["abs_diff"], d, 1e-9) or abs(r["abs_diff"] - d) <= 1e-18
        assert r["fired"] is (d > e5.W5_PIN_ABS)
        assert _rel_close(pinned[r["n_b"]]["R_model"], unp[r["n_b"]]["R_model"])
    assert pb["any_fired"] is any(r["fired"] for r in pb["rows"])
    for rec in _arms(cell):
        assert rec["m_lo"] == 2 and rec["k_src_used"] == 87
        _check_arm_model(rec, cell["dt_s"], f"pinbridge/{rec.get('name')}")


# --- second pass (2026-09-14): the numbers the note quotes, pinned to the JSONs ------------
# Tolerances declared with the second pass: ratios of stored floats 1e-9 relative;
# the prose ranges below are the corrected sentences of the note, re-derived here.

def _stored_arm(res, pattern, name):
    cell = res["cells"][pattern]
    if name in cell["singles"]:
        return cell["singles"][name]
    return {b["name"]: b for b in cell["bands"]}[name]


def test_results_prose_numbers_second_pass():
    res = {ax: _load(_AXIS[ax]) for ax in ("z", "x", "y")}
    # P1 vs P2 per width: z 0.02-0.11 %, x/y 0.13-1.84 % (|P1 - P2| / P2), growing with n_b in-plane
    for ax, lo, hi in (("z", 1.0e-4, 1.2e-3), ("x", 1.3e-3, 1.85e-2), ("y", 1.3e-3, 1.85e-2)):
        p1 = {b["n_b"]: b["meas"]["R_meas"] for b in res[ax]["cells"]["P1"]["bands"]}
        p2 = {b["n_b"]: b["meas"]["R_meas"] for b in res[ax]["cells"]["P2"]["bands"]}
        spread = [abs(p1[n] - p2[n]) / p2[n] for n in e5.WIDTHS]
        assert lo <= min(spread) and max(spread) <= hi, (ax, spread)
        if ax != "z":
            assert spread == sorted(spread), (ax, spread)
    # y vs x on all 40 arm pairs: largest 4.08e-5 (quoted <= 4.1e-5), median 3.4e-6
    rels = []
    for p in res["x"]["cells"]:
        cx, cy = res["x"]["cells"][p], res["y"]["cells"][p]
        for k in cx["singles"]:
            rels.append(abs(cx["singles"][k]["meas"]["R_meas"] - cy["singles"][k]["meas"]["R_meas"])
                        / cx["singles"][k]["meas"]["R_meas"])
        for bx, by in zip(cx["bands"], cy["bands"]):
            rels.append(abs(bx["meas"]["R_meas"] - by["meas"]["R_meas"]) / bx["meas"]["R_meas"])
    assert len(rels) == 40 and 4.0e-5 < max(rels) <= 4.1e-5 and float(np.median(rels)) < 4e-6
    # T ratio to the four-amplitude bound: 0.8767 z, 0.8779 x/y (quoted 0.878)
    for ax, want in (("z", 0.8767), ("x", 0.8779), ("y", 0.8779)):
        ratio = max(b["meas"]["R_meas"] / b["meas"]["bound_sum"] for b in res[ax]["cells"]["T"]["bands"])
        assert abs(ratio - want) < 5e-5, (ax, ratio)
    # single-arm bookkeeping: 11 entries per axis, 8 distinct FDTD runs
    for ax in ("z", "x", "y"):
        entries = [s["meas"]["run_id"] for c in res[ax]["cells"].values() for s in c["singles"].values()]
        assert len(entries) == 11 and len(set(entries)) == 8, ax
    # no arm within 0.30 of its half-window of a W1 edge (closest 0.26, x S n_b = 32)
    worst = 0.0
    for ax in ("z", "x", "y"):
        for c in res[ax]["cells"].values():
            for rec in _arms(c):
                worst = max(worst, abs(rec["meas"]["R_meas"] - rec["R_model"]) / rec["meas"]["half"])
    assert 0.25 < worst <= 0.30, worst
    # the L1-Z re-run of the S single reproduced L1-0 to the bit in a new process
    z = res["z"]["cells"]
    assert z["P1"]["singles"]["L"]["meas"]["fdtd_run_cached"] is False
    assert z["P1"]["singles"]["L"]["meas"]["R_meas"] == z["S"]["singles"]["L"]["meas"]["R_meas"]
    # the regenerated W4 block carries its dirty-tree annotation
    blk = res["z"]["w4_control_recomputed_from_stored"]
    assert blk["git_dirty_at_refresh"] is True and len(blk["git_sha"]) == 40


def test_results_window_scan_replays():
    scan = _load(_RESULTS / "e5_window_scan.json")
    assert scan["diagnostic"] == "window_scan" and scan["rfx_file"].endswith("rfx/__init__.py")
    assert [tuple(a) for a in scan["arms_declared"]] == list(e5.WINDOW_SCAN_ARMS)
    assert scan["inc_scan_sigmas_after_arrival"] == list(e5.WINDOW_SCAN_INC_SIGMAS)
    assert len(scan["rows"]) == 5 and scan["all_reruns_bit_identical"] is True
    res = {ax: _load(_AXIS[ax]) for ax in ("z", "x")}
    for r in scan["rows"]:
        stored = _stored_arm(res[r["axis"]], r["pattern"], r["arm"])
        # precondition: the rerun reproduced the recorded arm (same run ids, same float64)
        assert r["run_id"] == stored["meas"]["run_id"] and r["b_run_id"] == stored["meas"]["b_run_id"]
        assert r["R_meas_rerun"] == r["R_meas_stored"] == stored["meas"]["R_meas"]
        assert r["rerun_bit_identical"] is True and r["rerun_rel_diff"] == 0.0
        assert _rel_close(r["R_model"], stored["R_model"])
        # the scan's own arithmetic from its stored pieces
        assert _rel_close(r["R_meas_rerun"], r["refl_abs"] / r["inc_abs"], 1e-9)
        assert _rel_close(r["inc_peak_to_peak_rel"], r["inc_ratio_max"] - r["inc_ratio_min"], 1e-9)
        assert r["inc_ratio_min"] <= r["inc_ratio_4_8_min"] <= r["inc_ratio_4_8_max"] <= r["inc_ratio_max"]
        assert r["inc_ratio_min"] <= 1.0 <= r["inc_ratio_max"] and r["refl_ratio_min"] <= 1.0 <= r["refl_ratio_max"]
        assert _rel_close(r["R_range_min"], r["refl_ratio_min"] * r["refl_abs"] / (r["inc_ratio_max"] * r["inc_abs"]), 1e-9)
        assert _rel_close(r["R_range_max"], r["refl_ratio_max"] * r["refl_abs"] / (r["inc_ratio_min"] * r["inc_abs"]), 1e-9)
        assert abs(r["R_range_rel_model"][0] - (r["R_range_min"] / r["R_model"] - 1)) <= 1e-12
        assert abs(r["R_range_rel_model"][1] - (r["R_range_max"] / r["R_model"] - 1)) <= 1e-12
        assert abs(abs(r["R_nominal_rel_model"]) - stored["meas"]["deviation_rel"]) <= 1e-12
        # window ends: incident scan is arrival + 4..12 sigma; reflection scan ends at the gate
        assert r["inc_scan_steps"][0] < r["n_inc_nominal"] < r["inc_scan_steps"][1]
        assert r["refl_scan_steps"][1] == r["n_gate_nominal"] == stored["n_gate"]
        # the discrete TE10 cutoff of the grid and the scan's beat agree within one FFT bin
        assert _rel_close(r["te10_cutoff_grid_hz"], e5.te10_cutoff_hz(r["dt_s"]), 1e-12)
        assert 4.9e9 < r["te10_cutoff_grid_hz"] < 5.1e9
        assert abs(r["inc_scan_beat_hz"] - r["te10_cutoff_grid_hz"]) <= r["inc_scan_fft_resolution_hz"]
    # the summary the note quotes: incident 3.2-3.3 % p-p on every arm, R half-range 2-4.5 %
    assert all(0.032 <= r["inc_peak_to_peak_rel"] <= 0.033 for r in scan["rows"])
    half = max(max(abs(r["R_range_rel_model"][0]), abs(r["R_range_rel_model"][1])) for r in scan["rows"])
    assert abs(half - scan["max_half_range_rel_model"]) <= 1e-12 and 0.04 < half < 0.05
    assert all(0.019 < max(abs(r["R_range_rel_model"][0]), abs(r["R_range_rel_model"][1])) for r in scan["rows"])

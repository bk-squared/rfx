"""Replay of the E2 thin-layer subpixel witness
(``validation/research/multiband_nu/e2_thin_layer_subpixel.py``), pre-declared
in ``docs/design_notes/20260907_nu_exp2_thin_layer_subpixel_predeclaration.md``.

Written and committed before the ladder finished; the windows below are the
note's section 3 numbers verbatim and are never widened here.

* **JSON-free** (always run): the frozen windows are pinned against the
  instrument's constants; the transfer-matrix oracle passes (i)/(i')/(i'')
  on the E2 box; ``K_pert`` and the S1 fixed-fill model ladder (order 1.902)
  are recomputed from first principles; every declared mesh regenerates.
* **Replay** (skipped cleanly while the results JSON is absent): every
  mesh is regenerated cell for cell (1e-12 m) and every subpixel / dual
  column re-assembled bit-exactly from the JSON's own profile; every
  ``f_model`` is recomputed from the JSON's profile and column (1e-9
  relative); every rule of note section 3 is re-evaluated through the
  instrument's judge and must agree with the verdict the JSON carries and
  HOLD. A rule that FIRED in the committed JSON turns its test red by
  design; the answer is a PI decision recorded in the note, never a
  tolerance edit here.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from validation.research.multiband_nu import e2_thin_layer_subpixel as e2
from validation.research.multiband_nu import w7_accuracy_ad as w7

_REPO = pathlib.Path(__file__).resolve().parents[3]
_JSON = _REPO / "validation/research/multiband_nu/results/e2_thin_layer_subpixel.json"

CELL_TOL = 1e-12
MODEL_REL = 1e-9
JUDGE_REL = 1e-9

FROZEN = {
    "P_S1_WINDOW": (1.8, 2.2), "RHO_MAX": 2.0, "LAW_REL": 0.25, "LAW_ABS_HZ": 0.3e6, "S1B_RATIO_MAX": 1.0,
    "G3_MODEL_RESIDUAL_HZ": 0.15e6, "INVARIANCE_HZ": 0.1e6, "E_FLOOR_HZ": 0.3e6,
    "T_TOTAL": 15e-9, "T_TRUNC": 10e-9, "SIGMA_T": 200e-12, "ORACLE_REL": 1e-12, "SELFCHECK_REL": 1e-7,
    "Z1": 15.4e-3, "Z_AIR": 29.4e-3, "L_Z": 43.4e-3, "A_X": 30e-3, "B_Y": 3e-3, "EPS_C": 4.3, "EPS_T": 3.0,
    "DC0": 0.7e-3, "DXY0": 0.25e-3, "R_CAP": 1.4, "Z_SRC": 35.0e-3, "Z_PRB": 32.2e-3,
    "SCALES": (0.5, 1.0, 2.0), "THICK_UM": (175, 350, 700), "H1": 0.7e-3,
    "F_REF_DECLARED": 10_680_141_503.244,
}
MODEL_ORDER_S1_L1 = 1.902
K_PERT_MHZ_PER_MM2 = 17.608


@pytest.fixture(scope="module")
def e2_json() -> dict:
    if not _JSON.exists():
        pytest.skip(f"{_JSON.name} not present yet — the E2 units have not been run on this tree")
    with open(_JSON) as fh:
        return json.load(fh)


def _real(units: dict) -> dict:
    return {k: r for k, r in units.items() if not r.get("smoke") and "|superseded|" not in k}


def _close(x, y, rel):
    if x is None or y is None:
        return x is None and y is None
    return abs(x - y) <= rel * max(abs(x), abs(y), 1e-300)


# =============================================================================
# JSON-free
# =============================================================================

def test_frozen_windows_pinned_verbatim():
    for name, value in FROZEN.items():
        got = getattr(e2, name)
        if isinstance(value, tuple):
            assert tuple(got) == value, (name, got, value)
        else:
            assert got == value, (name, got, value)
    assert e2.stack(0) == ([0.0, 29.4e-3, 43.4e-3], [4.3, 1.0])
    assert e2.stack(350e-6) == ([0.0, 15.4e-3, 15.4e-3 + 350e-6, 29.4e-3, 43.4e-3], [4.3, 3.0, 4.3, 1.0])
    assert len(e2.declared_units()) == 49
    assert sorted(e2.law_units()) == sorted(["s1|0.5|175", "s1|1|175", "s1|1|350", "s1|2|175", "s1|2|350", "s1|2|700"])


def test_oracle_self_checks_on_the_e2_box():
    sc = e2.oracle_selfcheck()
    assert sc["i_pass"] and sc["i_worst_rel"] <= e2.ORACLE_REL, sc["i_worst_rel"]
    assert sc["ip_pass"] and sc["ip_worst_rel"] <= e2.ORACLE_REL, sc["ip_worst_rel"]
    assert sc["ipp_pass"] and sc["ipp_worst_rel"] <= e2.ORACLE_REL, sc["ipp_worst_rel"]
    assert sc["f_ref_matches_declared"] and sc["f_true_match_declared"], sc["f_true_hz"]
    assert sc["p_half_waves"] == 5
    assert abs(sc["psi_at_14_15p4_prb_src"][0]) < 0.05           # the A1 layer plane is a near-null here
    assert abs(sc["psi_at_14_15p4_prb_src"][3]) > 0.99            # source at the air antinode
    assert sc["other_families_t350"][0]["rel_to_f_true"] != 0.0
    assert abs(sc["other_families_t350"][0]["rel_to_f_true"]) < 0.01


def test_k_pert_and_s1_fixed_fill_model_ladder_from_first_principles():
    k = e2.k_pert()
    assert abs(k["K_pert_mhz_per_mm2"] - K_PERT_MHZ_PER_MM2) <= 1e-3 * K_PERT_MHZ_PER_MM2, k
    f_ref = e2.f_ref_true()
    es = []
    for s in e2.SCALES:
        t_um = int(round(350 * s))
        prof = e2.mesh_u(s)
        dx = e2.DXY0 * s
        from rfx.nonuniform import make_nonuniform_grid
        dt = float(make_nonuniform_grid((e2.A_X, e2.B_Y), prof, dx, cpml_layers=0).dt)
        f_true = e2.f_true_of(t_um, False)
        edges, eps = e2.stack(t_um * 1e-6)
        col, _ = e2.column_for("s1", prof, edges, eps, s, f_true)
        colr, _ = e2.column_for("u", prof, *e2.stack(0), s, f_ref)
        fm = e2.model(prof, col, dx, dt, edges, eps, f_true)["f_model"]
        fmr = e2.model(prof, colr, dx, dt, *e2.stack(0), f_ref)["f_model"]
        es.append((fm - fmr) - (f_true - f_ref))
        # the law from first principles, before any measurement: within 10 % on the model
        h, t = e2.DC0 * s, t_um * 1e-6
        law = k["K_pert_hz_per_m2"] * t * (h - t)
        assert abs(es[-1] - law) <= 0.10 * abs(law), (s, es[-1], law)
    p = np.polyfit(np.log10([e2.DC0 * s for s in e2.SCALES]), np.log10(np.abs(es)), 1)[0]
    assert abs(p - MODEL_ORDER_S1_L1) <= 1e-3, p


def test_declared_meshes_regenerate_on_this_tree():
    seen = set()
    for u in e2.declared_units():
        mk = (u["mesh"], u["scale"], u["t_mesh_um"])
        if mk in seen:
            continue
        seen.add(mk)
        prof = e2.mesh_of(u["mesh"], u["scale"], u["t_mesh_um"] * 1e-6)
        nz_key = ("u", u["scale"]) if u["mesh"] == "u" else (u["mesh"] if u["mesh"] != "p4" else "r4", u["scale"], u["t_mesh_um"])
        assert len(prof) == e2.MESH_NZ[nz_key], mk
        zn = np.concatenate([[0.0], np.cumsum(prof)])
        for z in (e2.Z1, e2.Z_AIR, e2.Z_SRC, e2.Z_PRB):
            assert np.min(np.abs(zn - z)) <= CELL_TOL, (mk, z)
        rr = np.maximum(prof[1:] / prof[:-1], prof[:-1] / prof[1:])
        assert rr.max() <= e2.R_CAP + 1e-9, mk
        air = prof[zn[:-1] >= e2.Z_AIR - 1e-12]
        assert np.allclose(air, e2.DC0 * u["scale"], atol=1e-12), mk


# =============================================================================
# Replay from the committed JSON
# =============================================================================

def test_replay_provenance_and_selfcheck(e2_json):
    sc = e2_json["selfcheck"]
    assert sc["all_pass"] and not sc["failed"]
    assert sc["rfx_file"].endswith("rfx/__init__.py")
    for run in e2_json["runs"]:
        for key in ("rfx_file", "git_sha", "git_dirty", "argv", "started_utc"):
            assert key in run
    for r in _real(e2_json["units"]).values():
        assert r["rfx_file"].endswith("rfx-nu-exp2/rfx/__init__.py"), r["key"]


def test_replay_meshes_and_columns(e2_json):
    units = _real(e2_json["units"])
    assert units, "no units in the JSON"
    for key, r in units.items():
        prof = np.asarray(r["profile_m"])
        regen = e2.mesh_of(r["mesh"], r["scale"], r["t_mesh_um"] * 1e-6)
        assert len(regen) == len(prof) and np.max(np.abs(regen - prof)) <= CELL_TOL, key
        col = np.asarray(r["eps_column"])
        assert len(col) == len(prof) + 1, key
        edges, eps = r["edges_m"], r["eps_layers"]
        if r["rule"] == "subpixel":
            expect = w7.f32_values(e2.dual_from_cells(prof, e2.cell_eps_fill(prof, edges, eps)))
            assert np.max(np.abs(expect - col)) == 0.0, key
        elif r["rule"] == "dual":
            expect = w7.f32_values(e2.dual_from_cells(prof, w7.cell_eps(prof, edges, eps)))
            assert np.max(np.abs(expect - col)) == 0.0, key
        else:
            assert r["rule"] == "production" and r["interior_transversely_uniform"], key
            if r["layer"] and r["t_mesh_um"] == int(round(350 * r["scale"])):
                tab = tuple(round(v["eps"], 1) for v in r["interface_table"].values())
                assert tab == e2.P4_INTERFACE_TABLE[r["scale"]], (key, tab)
        assert r["node_err_m"] <= CELL_TOL, key


def test_replay_model_frequencies_recompute(e2_json):
    for key, r in _real(e2_json["units"]).items():
        m = e2.model(np.asarray(r["profile_m"]), np.asarray(r["eps_column"]), r["dx"], r["dt"],
                     r["edges_m"], r["eps_layers"], r["f_true"])
        assert _close(m["f_model"], r["f_model"], MODEL_REL), key
        assert _close(r["e_total_model"], r["f_model"] - r["f_true"], 1e-6), key


def test_replay_rules(e2_json):
    units = _real(e2_json["units"])
    j_json = e2_json["judge"]
    j = e2.judge(units, K=j_json["K_pert_hz_per_m2"], f_ref=j_json["f_ref_hz"])
    assert j["verdict"] == j_json["verdict"], (j["verdict"], j_json["verdict"])
    assert _close(j["e2_f1"]["p_s1"], j_json["e2_f1"]["p_s1"], JUDGE_REL)
    for s, v in j["e2_f2"]["rho_per_scale"].items():
        assert _close(v, j_json["e2_f2"]["rho_per_scale"][s], JUDGE_REL)
    for k, v in j["e2_f3"]["units"].items():
        assert _close(v["e_shift_hz"], j_json["e2_f3"]["units"][k]["e_shift_hz"], JUDGE_REL), k
    # --- the frozen rules of note section 3, from the measured numbers ------------
    g = j["gates"]
    assert g["g3_pass_all"], g["g3_failed_units"]
    assert g["g3_max_residual_subpixel_dual_hz"] <= e2.G3_MODEL_RESIDUAL_HZ
    assert g["max_invariance_hz"] <= e2.INVARIANCE_HZ, g["max_invariance_hz"]
    assert not g["inconclusive_units"], g["inconclusive_units"]
    f1 = j["e2_f1"]
    assert f1["n_points"] == 3, f1
    assert e2.P_S1_WINDOW[0] <= f1["p_s1"] <= e2.P_S1_WINDOW[1], f1["p_s1"]
    f2 = j["e2_f2"]
    assert f2["n_ratios"] == 6 and f2["fired"] is False, f2
    assert max(list(f2["rho_per_scale"].values()) + list(f2["rho_per_thickness_s1"].values())) <= e2.RHO_MAX
    f3 = j["e2_f3"]
    assert f3["n_evaluated"] == 6 and f3["fired"] is False, f3["fired_units"]
    for k, v in f3["units"].items():
        assert abs(v["dev_hz"]) <= v["tol_hz"], (k, v)
    f4 = j["e2_f4"]
    assert len(f4["ratio_per_scale"]) == 3 and f4["fired"] is False, f4
    assert max(f4["ratio_per_scale"].values()) <= e2.S1B_RATIO_MAX


def test_replay_measured_shift_errors_follow_the_model(e2_json):
    """Reported, pinned at the G3 level: every measured e_shift is within
    2 x G3 (two units per difference) of the exact-model e_shift."""
    j = e2_json["judge"]
    for k, e in j["e_shift_hz"].items():
        m = j["e_shift_model_hz"].get(k)
        if e is None or m is None:
            continue
        assert abs(e - m) <= 2 * e2.G3_MODEL_RESIDUAL_HZ, (k, e, m)

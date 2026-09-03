"""cv26 oblique slab -- gate replay of the committed VESSL artifacts.

Skips while ``validation/crossval/_26_oblique_results/rfx.json`` is absent.
Each replay recomputes the gates from the artifact's raw per-bin R, T with
the comparator (windows, oracle, lattice and records re-derived, never read
from the artifact's own copies) and must reproduce the stored verdicts:
the baseline passes on every arm; every falsifier artifact fails for its
pre-declared reason; the Meep k_2pi leg fails E4 with ``precheck.passed ==
False``; the depth-ladder rungs reproduce their lattice predictions.

No FDTD runs here. Pre-declaration:
``docs/design_notes/20260902_cv26_oblique_fresnel_predeclaration.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_RESULTS = _REPO / "validation/crossval/_26_oblique_results"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _load("cv26_gates_oblique_fresnel", "validation/crossval/comparators/oblique_fresnel.py")


def _artifact(name: str) -> dict:
    p = _RESULTS / name
    if not p.is_file():
        pytest.skip(f"{p.relative_to(_REPO)} not committed yet (VESSL round pending)")
    doc = json.loads(p.read_text())
    if doc.get("smoke"):
        pytest.skip("smoke artifact is never evidence")
    return doc


def _replay_arm(arm: str, ad: dict, *, oracle_pol=None, oracle_ky=None):
    spec = O.arm_spec(arm)
    run = ad["run"]
    cells = O.rig_cells(run["nx_interior"], run["n_cpml"], dx_div=run["dx_div"])
    e2 = O.evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], spec, run["dt_s"], tail=ad["tail"], cells=cells,
                       n_cpml=run["n_cpml"], oracle_pol=oracle_pol, oracle_ky=oracle_ky)
    return spec, run, cells, e2


def test_baseline_replays_and_passes_on_every_arm():
    doc = _artifact("rfx.json")
    assert doc["schema"] == O.SCHEMA and doc["falsifier"] is None and doc["commit"] not in ("", "unknown")
    assert set(O.ARM_ORDER + O.GRAZE_ARMS) <= set(doc["arms"])
    assert doc["verdict"]["exit_code"] == 0
    for arm in O.ARM_ORDER:
        ad = doc["arms"][arm]
        spec, run, cells, e2 = _replay_arm(arm, ad)
        assert run["dx_div"] == O.ARM_DX_DIV[arm] and run["n_cpml"] == O.N_CPML
        rec = O.derive_record(spec, run["dt_s"], dx_div=run["dx_div"])
        assert run["record"]["n_steps_min"] == rec["n_steps"]
        # round 2 (note section 13): the record is the DECLARED lattice settling step, and
        # the absorber echo INSIDE it is what the arm is admissible on
        assert run["record"]["record_source"].startswith("declared"), (arm, run["record"]["record_source"])
        assert run["record"]["absorber_ok"] and ad["gates_all"]["G3_absorber"] is True, arm
        assert max(run["record"]["W_absorber_R_max"], run["record"]["W_absorber_T_max"]) <= O.W_BIN, arm
        assert run["n_steps"] == rec["n_steps"] + run["record"]["extensions"] * run["record"]["extend_steps"]
        assert run["n_steps"] <= run["record"]["cap_steps"] and not run["record"]["cap_reached"]
        assert ad["tail"]["scat_refl_rel"] < O.SETTLING_LIMIT and ad["tail"]["total_trans_rel"] < O.SETTLING_LIMIT
        assert ad["tail"]["purity_inc_rel"] < O.TAIL_PURITY_LIMIT
        assert e2["gates"] == ad["gates"] and e2["e2_ok"] and all(e2["gates"].values()), arm
        for k in ("mean_dR_gated", "mean_dT_gated", "max_dR_gated", "mean_window_R"):
            assert e2[k] == pytest.approx(ad[k], rel=1e-9), (arm, k)
        # the lattice witness (reported): rfx equals its own exact discrete model
        assert e2["lattice"]["mean_dR_lattice_gated"] <= 3e-4 and e2["lattice"]["mean_dT_lattice_gated"] <= 3e-4, arm
        if arm == O.BREWSTER_ARM:
            bw = O.evaluate_brewster(e2)
            assert bw["ok"] and ad["brewster"]["ok"] and abs(bw["theta_bin_deg"] - bw["theta_brewster_deg"]) < 0.1
        if arm == "te_00":
            assert ad["swap_ref_at_normal"]["e2_ok"]
        m = ad["meep"]
        if arm in O.MEEP_ARMS:
            assert m["present"], (arm, m.get("unavailable_reason"))
            assert m["gates"]["precheck_passed"] and m["gates"]["k_point_matches_declared"] and m["e4_ok"], (arm, m["gates"])
            assert m["resolution"] == O.MEEP_PRIMARY_RESOLUTION
    # grazing arms
    ad = doc["arms"]["graze_vac"]
    assert ad["leak"]["G_leak"] and ad["leak"]["max_leak_gated"] <= O.LEAK_BAR
    ad = doc["arms"]["graze_pec"]
    spec, run, cells, _ = _replay_arm("graze_pec", ad)
    pg = O.evaluate_grazing_pec(ad["freqs_hz"], ad["R_rfx"], spec, run["dt_s"],
                                O.rig_cells(spec["nx_interior"], O.N_CPML, dx_div=run["dx_div"]), n_cpml=run["n_cpml"])
    assert pg["G6_absorber"] and ad["grazing_pec"]["G6_absorber"]
    assert pg["max_abs_dev_gated"] == pytest.approx(ad["grazing_pec"]["max_abs_dev_gated"], rel=1e-9)
    assert pg["max_cpml3d_term_band"] > 0.1                     # the 3-D absorber term at 80-85 deg, a priori
    ad = doc["arms"]["graze_te"]
    spec, run, cells, _ = _replay_arm("graze_te", ad)
    gs = O.evaluate_grazing_slab(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], spec, run["dt_s"], cells, n_cpml=run["n_cpml"])
    assert gs["G7_R"] and gs["G7_T"] and ad["grazing_slab"]["G7_R"] and ad["grazing_slab"]["G7_T"]


@pytest.mark.parametrize("name", sorted(O.FALSIFIER_MUST_EXIT_1))
def test_each_falsifier_artifact_fails_for_its_declared_reason(name):
    doc = _artifact(O.rfx_json_name(name))
    assert doc["falsifier"] == name and doc["verdict"]["exit_code"] == 1
    arm, _, run_def, or_def = O.FALSIFIERS[name]
    ad = doc["arms"][arm]
    if arm == "graze_pec":
        assert ad["run"]["n_cpml"] == run_def.get("n_cpml", O.N_CPML)
        assert ad["gates_all"]["G6_absorber"] is False
        return
    if "theta0_deg" in run_def:
        assert ad["run"]["theta0_run_deg"] == run_def["theta0_deg"]
    spec, run, cells, e2 = _replay_arm(arm, ad, oracle_pol=or_def.get("oracle_pol"))
    assert not e2["e2_ok"] and e2["gates"] == ad["gates"]
    assert not (e2["gates"]["G2_R"] and e2["gates"]["G2_T"]), "the declared defect must fail the band-mean gate"
    if name == "tm_60_swap_te":
        assert ad["brewster"]["ok"] is False


def test_meep_k_2pi_falsifier_fails_e4_with_a_failed_precheck():
    doc = _artifact(O.rfx_json_name("meep_te_45_k_2pi"))
    m = doc["arms"][O.MEEP_FALSIFIER_ARM]["meep"]
    assert m["present"] and m["precheck"]["passed"] is False and not m["gates"]["k_point_matches_declared"]
    assert not m["e4_ok"] and doc["verdict"]["exit_code"] == 1
    leg = _artifact(O.meep_json_name(O.MEEP_FALSIFIER_ARM, "k_2pi"))
    assert leg["precheck"]["passed"] is False and leg["k_point"][1] == pytest.approx(O.meep_k_point_wrong_2pi(O.TFSF_F0_HZ, 45.0)[1])


@pytest.mark.parametrize("depth", O.CPML_DEPTH_LADDER)
def test_depth_ladder_rungs_reproduce_their_own_lattice_prediction(depth):
    doc = _artifact(f"rfx__graze_pec_d{depth}.json")
    ad = doc["arms"]["graze_pec"]
    assert ad["run"]["n_cpml"] == depth
    spec = O.arm_spec("graze_pec")
    cells = O.rig_cells(spec["nx_interior"], depth, dx_div=ad["run"]["dx_div"])
    lat = O.yee_lattice_full(np.asarray(ad["freqs_hz"]), spec["ky"], cells, dx=cells["dx"], dt=ad["run"]["dt_s"],
                             n_cpml=depth, pec=True)
    g = np.asarray(ad["gated"], bool)
    dev = np.abs(np.asarray(ad["R_rfx"]) - lat["R"])[g]
    assert dev.max() <= O.PML_REL * np.abs(lat["R"] - 1.0)[g].max() + O.PML_FLOOR_R


@pytest.mark.parametrize("arm", O.MEEP_ARMS)
def test_every_meep_leg_vouched_for_its_own_output(arm):
    """Round 2 (note section 14). Round 1's te_00 and te_30 legs wrote R = -inf,
    T = +inf for all 400 bins and the E4 gate read them.  The artifact now
    carries the leg's own acceptance verdict, and a rejected one carries no
    arrays at all."""
    doc = _artifact(O.meep_json_name(arm))
    assert doc["schema"] == "cv26-meep-leg/v2", arm
    assert doc["accepted"] is True and doc["rejection_reasons"] == [], (arm, doc["rejection_reasons"])
    assert O.meep_unavailable_reason(doc, str(_RESULTS / O.meep_json_name(arm))) is None
    R = np.asarray(doc["R"], float); T = np.asarray(doc["T"], float)
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T)), arm
    spec = O.arm_spec(arm)
    acc = O.meep_accept(doc["freqs_hz"], R, T, doc["inc_flux"], spec, inc_flux_refl=doc["inc_flux_refl"])
    assert acc["accepted"], (arm, acc["reasons"])
    # the cross-box flux identity of the EMPTY run: round 2's measurement of it
    assert acc["vacuum_flux_ratio_dev"] <= O.MEEP_ACCEPT_TOL, (arm, acc["vacuum_flux_ratio_dev"])
    # the geometry fix: the PML sits where rfx's CPML sits, so the source is clear of it
    geo = doc["geometry"]
    assert geo["sx"] == pytest.approx((O.NX_INTERIOR + 2 * O.N_CPML) * O.DX_M / doc["a_m"])
    assert geo["src_x"] >= -geo["sx"] / 2 + geo["dpml"], arm
    # and the stop condition could not fire before first arrival
    assert doc["run"]["t_min_after_sources"] >= abs(geo["trans_x"] - geo["src_x"]), arm

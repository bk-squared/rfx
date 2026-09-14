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


def _assert_compact_declaration(doc: dict, arm: str, expect_judged: dict) -> None:
    """A compact-box arm's artifact must carry the whole verdict rule, not only its
    PASS: the gates it was judged on with their values, and the gates DECLARED not
    applicable shown as ``N/A`` rather than as bare Falses beside a PASS.

    graze_pec and graze_te were moved onto this footing by the PI decision of
    2026-09-13 (issue #905, pre-declaration section 18): they are judged on
    G6 / G7 plus the tail witness, and G3_passivity / G3_closure are N/A because
    their record is designed to contain the 20-cell absorber's echo.  graze_vac
    keeps the four oracle gates it already declared.
    """
    ad = doc["arms"][arm]
    na = list(O.COMPACT_GATES_NOT_JUDGED[arm])
    assert ad["gates_not_applicable"] == na, (arm, ad["gates_not_applicable"])
    assert ad["gates_not_applicable_reason"].strip(), arm
    for g in na:
        assert ad["gates_all"][g] == "N/A", (arm, g, ad["gates_all"][g])
    rule = ad["verdict_rule"]
    assert rule["judged_on"] == list(O.COMPACT_GATES_JUDGED_ON[arm]), (arm, rule["judged_on"])
    assert rule["gate_values"] == expect_judged, (arm, rule["gate_values"])
    # the stored verdict is what the rule gives when replayed on the stored gates
    assert O.compact_arm_verdict(arm, rule["gate_values"])["ok"] is bool(ad["e2_ok"]), arm


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
        # The lattice witness (reported): rfx equals its own exact discrete model.
        #
        # The bar is a tenth of the arm's DECLARED band-mean window, which resolves
        # through cv04's adoption record (``O.W_MEAN_R`` / ``O.W_MEAN_T``), not a
        # literal.  The ``3e-4`` that stood here is cv23's rig-specific number -- 10x
        # THAT case's round-1 residual, ``test_cv23_lossy_slab_gates.py``'s
        # ``_R2_LATTICE_RESIDUAL_BAR`` -- and it was never re-derived for this rig.
        # It was also never exercised: this file skips until the artifacts land, so
        # the literal survived from the authoring commit to the first real lane.
        # On the 80-cell primary rig three arms sit above it with nothing wrong --
        # the two dx controls (3.34e-04 te_00, 3.41e-04 tm_00) and te_60 at dx/2
        # (4.05e-04).  Close note section 10 already records that the inherited
        # reading rule misreads this case.
        #
        # Against the window the arm is actually judged on, every arm is far inside:
        # 0.44-4.05 % (R) and 1.53-4.94 % (T), so the E2 gates judge the solver
        # against Fresnel and not against a solver-vs-own-model gap.  The window is
        # declared a priori, so this is not the run certifying its own bar.
        lat = e2["lattice"]
        assert lat["mean_dR_lattice_gated"] <= 0.1 * O.W_MEAN_R, (arm, lat["mean_dR_lattice_gated"])
        assert lat["mean_dT_lattice_gated"] <= 0.1 * O.W_MEAN_T, (arm, lat["mean_dT_lattice_gated"])
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
    # The four oracle gates are DECLARED not-judged on the vacuum arm (no R/T
    # oracle), and shown as N/A rather than as False beside a PASS. A gate that
    # stopped being judged without this declaration is the failure mode this
    # asserts against. (Checked below with the other two compact arms.)
    assert ad["gates_all"]["G_leak"] is True and ad["gates_all"]["G3_tail"] is True
    ad = doc["arms"]["graze_pec"]
    spec, run, cells, _ = _replay_arm("graze_pec", ad)
    # the DECLARED absorber of the compact box is 20 cells, not the primary rig's 80
    pg = O.evaluate_grazing_pec(ad["freqs_hz"], ad["R_rfx"], spec, run["dt_s"],
                                O.rig_cells(spec["nx_interior"], O.N_CPML_COMPACT, dx_div=run["dx_div"]),
                                n_cpml=run["n_cpml"])
    assert run["n_cpml"] == O.N_CPML_COMPACT
    assert pg["G6_absorber"] and ad["grazing_pec"]["G6_absorber"]
    assert pg["max_abs_dev_gated"] == pytest.approx(ad["grazing_pec"]["max_abs_dev_gated"], rel=1e-9)
    assert pg["max_cpml3d_term_band"] > 0.1                     # the 3-D absorber term at 80-85 deg, a priori
    # PI decision 2026-09-13 (issue #905), pre-declaration section 18: the measured
    # excess sits on the a-priori absorber term -- that agreement is the reason the
    # arm is judged on G6 rather than on passivity, so it is asserted, not assumed.
    assert ad["grazing_pec"]["max_excess_meas_band"] == pytest.approx(
        ad["grazing_pec"]["max_absorber_term_band"], rel=0.05)
    ad = doc["arms"]["graze_te"]
    spec, run, cells, _ = _replay_arm("graze_te", ad)
    gs = O.evaluate_grazing_slab(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], spec, run["dt_s"], cells, n_cpml=run["n_cpml"])
    assert gs["G7_R"] and gs["G7_T"] and ad["grazing_slab"]["G7_R"] and ad["grazing_slab"]["G7_T"]
    _assert_compact_declaration(doc, "graze_pec", {"G6_absorber": True, "G3_tail": True})
    _assert_compact_declaration(doc, "graze_te", {"G7_R": True, "G7_T": True, "G3_tail": True})
    _assert_compact_declaration(doc, "graze_vac", {"G_leak": True, "G3_tail": True})


@pytest.mark.parametrize("name", sorted(O.FALSIFIER_MUST_EXIT_1))
def test_each_falsifier_artifact_fails_for_its_declared_reason(name):
    doc = _artifact(O.rfx_json_name(name))
    assert doc["falsifier"] == name and doc["verdict"]["exit_code"] == 1
    arm, _, run_def, or_def = O.FALSIFIERS[name]
    ad = doc["arms"][arm]
    if arm == "graze_pec":
        assert ad["run"]["n_cpml"] == run_def.get("n_cpml", O.N_CPML_COMPACT)
        assert ad["gates_all"]["G6_absorber"] is False
        # The falsifier of the PI decision itself (issue #905, pre-declaration
        # section 18): passivity and closure are N/A on this arm, and a defect that
        # breaks the witness it IS judged on must still take the arm and the run
        # down. An N/A that made the arm unfalsifiable would read green here.
        assert ad["gates_all"]["G3_passivity"] == "N/A" and ad["gates_all"]["G3_closure"] == "N/A"
        assert ad["verdict_rule"]["gate_values"]["G6_absorber"] is False
        assert ad["e2_ok"] is False and doc["verdict"]["exit_code"] == 1
        return
    if "theta0_deg" in run_def:
        assert ad["run"]["theta0_run_deg"] == run_def["theta0_deg"]
    spec, run, cells, e2 = _replay_arm(arm, ad, oracle_pol=or_def.get("oracle_pol"))
    assert not e2["e2_ok"] and e2["gates"] == ad["gates"]
    assert not (e2["gates"]["G2_R"] and e2["gates"]["G2_T"]), "the declared defect must fail the band-mean gate"
    if name == "tm_60_swap_te":
        # The pre-declaration named the Brewster bin as this falsifier's target and
        # this line asserted the gate FAILS.  The measurement refuted that, and close
        # note section 4.3 carries the refutation: ``evaluate_brewster`` compares
        # rfx's own R at the Brewster bin against a floor built from the ORACLE, and
        # never compares the oracle's R.  Swapping the oracle to TE only WIDENS that
        # floor, so the swap makes the gate EASIER -- it cannot fire on an
        # oracle-swap falsifier at all.  Asserting the mechanism instead of the
        # refuted expectation, so a change that made the gate fire here would still
        # be caught.  The defect is carried by the band-mean gates asserted above.
        clean = _artifact("rfx.json")["arms"][arm]["brewster"]
        assert ad["brewster"]["floor"] > clean["floor"]                    # widened, not tightened
        assert ad["brewster"]["R_rfx_at_brewster"] == pytest.approx(       # rfx side untouched
            clean["R_rfx_at_brewster"], rel=1e-12)
        assert clean["R_an_at_brewster"] < 1e-6                            # TM oracle: a true null
        assert ad["brewster"]["R_an_at_brewster"] > 0.1                    # TE oracle: no null left
        assert ad["brewster"]["ok"] is True


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

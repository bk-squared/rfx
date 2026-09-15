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
_LW = _load("cv26_gates_lattice_witness", "validation/crossval/comparators/lattice_witness.py")

# GL2 (band mean) against the DERIVED window of the lattice-witness standard section 3,
# per arm, as MEASURED on the committed lane.  This is a record of the outcome, not a
# claim that every arm passes.
#
# tm_60 FAILS GL2_R: its residual 5.060e-05 is 244 % of a window 2.070e-05.  It is pinned
# as a failure rather than excluded or widened away.  What that does and does not mean:
# the arm's Fresnel verdict is a SEPARATE gate and passes as measured (G1, G2, G3 all
# true), so what fails is rfx-versus-its-own-discrete-model on this arm, not
# rfx-versus-Fresnel.  ``validation/crossval/manifest.json``'s claim_scope says so.
#
# The failure sits well inside a term the standard's budget does not model: tm_60's own
# absorber term is 5.048e-04, 24x its window and 10x its residual.  Tracked as #1015;
# no mechanism attempt was made, so it is reported as a failure and not explained away.
_GL2_EXPECTED = {
    "te_00": (True, True), "te_30": (True, True), "te_45": (True, True),
    "te_60": (True, True), "tm_00": (True, True), "tm_45": (True, True),
    "tm_60": (False, True),
}

# #1015 / standard section 13: GL1's validity domain on R, as MEASURED --
# (bins in the domain, breaches inside it, breaches outside it).  A record of the
# outcome, not a claim that the domain closes GL1: te_30's R domain is the whole
# band, so all 18 of its R breaches lie INSIDE it, which is why the domain is
# stated as a necessary condition and GL1 stayed ungated here.  Of this case's
# 1291 R+T breaches, 1156 (89.5 %) fall outside the domain and 135 inside.
_DOMAIN_EXPECTED = {
    "te_00": (290, 0, 0), "te_30": (441, 0, 0), "te_45": (341, 16, 59),
    "te_60": (214, 8, 90), "tm_00": (290, 0, 0), "tm_45": (211, 13, 177),
    "tm_60": (42, 14, 343),
}

# #1015 section 14: WHICH lattice each arm's witness is judged against, and the
# GEOMETRY that decides it -- `record["t_safe_cpml_steps"] >= n_steps` AND the
# #892 auxiliary-echo arrival `> n_steps`.  An arm whose two absorber echoes are
# both outside its record cannot have measured either, so it is judged against
# the ABSORBER-FREE lattice (the slab family's own reference) and its unmodelled
# term is zero BY ARRIVAL; an arm that admits an echo by AMPLITUDE keeps the
# realized lattice and section 13's validity domain.
#
# This table is pinned against the GEOMETRY, not against a breach count: a rig
# change that moved the CPML standoff or the auxiliary absorber distance would
# otherwise silently re-pick an arm's reference and change what GL1 is measured
# against with nothing going red.  That is the silent-default hazard standard
# section 13.6 warns about for `unmodelled_term=None`, and this is the guard.
#
# te_30 is the arm the correction moves: (441, 18, 0) -> (441, 0, 0) above.
# te_00 and tm_00 were already (290, 0, 0) and cannot move.
_WITNESS_REFERENCE = {
    "te_00": "absorber_free_by_arrival", "te_30": "absorber_free_by_arrival",
    "tm_00": "absorber_free_by_arrival",
    "te_45": "realized_absorbers", "te_60": "realized_absorbers",
    "tm_45": "realized_absorbers", "tm_60": "realized_absorbers",
    "graze_vac": "realized_absorbers", "graze_pec": "realized_absorbers",
    "graze_te": "realized_absorbers",
}

# The two settle-60 rungs, which NO test pinned before #1015 section 14 and
# which are the entries the correction moves MOST: 76.6 % / 75.5 % domain
# coverage with 4/66 and 5/66 in/out-of-domain R breaches, against the realized
# lattice, become 100 % coverage and 0/0 against the absorber-free one.  Each
# tuple is (bins in domain, in-domain breaches, out-of-domain breaches) on R,
# then on T, then the all-bin GL1 (R, T) counts.
_SETTLE60_EXPECTED = {
    "te_00__settle60": ("rfx__te_00_settle60.json", "te_00", 1611, (290, 0, 0), (290, 0, 0), (0, 0)),
    "tm_00__settle60": ("rfx__tm_00_settle60.json", "tm_00", 1612, (290, 0, 0), (290, 0, 0), (0, 0)),
}


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
                       n_cpml=run["n_cpml"], oracle_pol=oracle_pol, oracle_ky=oracle_ky,
                       inc_amp_rel=ad["inc_amp_rel"], record=run["record"])
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
        # The lattice witness: rfx equals its own exact discrete model, inside a
        # window DERIVED from this arm's own committed witnesses.
        #
        # What stood here was ``<= 3e-4``, which is cv23's rig-specific number --
        # 10x THAT case's round-1 residual, ``test_cv23_lossy_slab_gates.py``'s
        # ``_R2_LATTICE_RESIDUAL_BAR`` -- never re-derived for this rig, and never
        # exercised, because this file skips until the artifacts land.  Lane
        # 20260913b refutes it on both sides: three of seven arms above it in R,
        # FIVE of seven in T.  The full table is in pre-declaration section 19,
        # which also records that section 12 of that same note already carried
        # 3.9e-4 / 4.8e-4 / 6.8e-4 for its own recipes.
        #
        # The replacement is not another literal and is not a multiple of the
        # continuum window.  ``docs/design_notes/20260903_lattice_witness_standard.md``
        # section 3 derives ``W_witness`` for exactly this witness out of the arm's
        # own tail (record truncation, incident-reference truncation, float32), and
        # section 4 gates it as GL1 per bin and GL2 per band mean.  The comparator
        # computes it with that standard's own primitives.
        #
        # GL2 is the gate here, because the band mean is what the retired literal
        # also bounded.  GL1 is reported with its breach count and NOT gated; the
        # artifact carries the reason, and section 19 records the one attempt that
        # was made to close it and failed.
        lat = e2["lattice"]
        # GL2 is asserted against the MEASURED table, not asserted to pass.  On the
        # corrected windows tm_60 FAILS it (residual 244 % of its window) and that is
        # pinned as a failure, so a change that fixed it or broke another arm shows
        # up here instead of being absorbed.  The arm's E2 / Fresnel verdict is
        # separate and unchanged; see ``_GL2_EXPECTED``'s note.
        assert (lat["GL2_R"], lat["GL2_T"]) == _GL2_EXPECTED[arm], (
            arm, lat["GL2_R"], lat["GL2_T"], lat["mean_dR_lattice_gated"],
            lat["mean_W_witness_R_gated"], lat["mean_dT_lattice_gated"],
            lat["mean_W_witness_T_gated"])
        # the window is built from witnesses, never from the residual it bounds
        assert lat["witness_rate_source"] == "derived", (arm, lat["witness_rate_source"])
        assert lat["GL1_gated"] is False and lat["GL1_not_gated_reason"]
        # #1015: GL1's validity domain (standard section 13).  This rig admits its
        # absorber echo INSIDE the record by AMPLITUDE, so the standard's section-3
        # "zero by construction" does not hold here and the window is only a valid
        # bound where the omitted term sits inside it.  The domain is REPORTED --
        # GL1 stays ungated -- and the breach split is pinned per arm so a change
        # that moved the coverage shows up here.  The domain is necessary, not
        # sufficient: every arm below with a partial domain still has in-domain
        # breaches, which is exactly why GL1 did not become a gate.
        for obs, beyond in (("R", lat["GL1_R_bins_beyond"]), ("T", lat["GL1_T_bins_beyond"])):
            dom = lat[f"domain_{obs}"]
            assert dom["n_bins_gated"] == e2["n_bins_gated"], (arm, obs, dom)
            assert (dom["n_bins_beyond_in_domain"]
                    + dom["n_bins_beyond_outside_domain"]) == beyond, (arm, obs, dom)
            assert 0.0 <= dom["domain_fraction"] <= 1.0, (arm, obs, dom)
        assert _DOMAIN_EXPECTED[arm] == (
            lat["domain_R"]["n_bins_in_domain"],
            lat["domain_R"]["n_bins_beyond_in_domain"],
            lat["domain_R"]["n_bins_beyond_outside_domain"]), (arm, lat["domain_R"])
        # #1015 section 14: the reference is chosen by ARRIVAL, and the choice is
        # pinned here beside the domain it decides.  The two ratio keys must
        # SURVIVE the choice: on an arrival-safe arm the domain is total and the
        # absorber magnitude is reported through `report_term`, so a reader still
        # sees how big the content the chosen reference EXCLUDES is.
        assert lat["witness_reference"] == _WITNESS_REFERENCE[arm], (arm, lat["witness_reference"])
        assert lat["witness_reference_reason"].strip(), arm
        for obs in ("R", "T"):
            dom = lat[f"domain_{obs}"]
            assert "max_unmodelled_over_window_gated" in dom, (arm, obs, sorted(dom))
            assert "mean_unmodelled_over_mean_window_gated" in dom, (arm, obs, sorted(dom))
            if lat["witness_reference_arrival_safe"]:
                assert dom["unmodelled_term_is_reported_only"] is True, (arm, obs)
                assert dom["n_bins_in_domain"] == e2["n_bins_gated"], (arm, obs, dom)
            else:
                assert "unmodelled_term_is_reported_only" not in dom, (arm, obs)
        # the budget must use the ARM's source, not the slab family's constant
        assert lat["tau_src_s"] == pytest.approx(run["record"]["src_tau_s"], rel=1e-12), arm
        assert lat["tau_src_s"] != pytest.approx(_LW.TAU_SRC_S, rel=1e-9), (
            arm, "cv26 drives its own bandwidth; the family tau must not be what was used")
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


def test_the_witness_reference_is_chosen_by_arrival_geometry_not_by_a_residual():
    """#1015 section 14: which lattice each arm's witness is judged against is
    decided by GEOMETRY -- `record["t_safe_cpml_steps"]` and the #892 auxiliary
    echo's flight-minus-lead arrival, both against the arm's own record length.

    The predicate is recomputed here from those two numbers ALONE, with no
    window, no residual and no breach count in sight, and required to agree with
    what the comparator chose on every arm.  That is the whole content of the
    mechanism claim, so it is pinned rather than described: an arm that started
    matching a different reference would have to red here first.

    The two clauses are collinear on this rig -- every arm that clears the CPML
    round trip also clears the auxiliary arrival -- so this is ten entries of
    evidence, not twenty independent bits, and the docstring says so instead of
    letting the "10/10" read stronger than it is.
    """
    doc = _artifact("rfx.json")
    for arm in O.ARM_ORDER + O.GRAZE_ARMS:
        ad = doc["arms"][arm]
        spec, run, cells, e2 = _replay_arm(arm, ad)
        rec = run["record"]
        n = int(rec["n_steps"])
        ae = O.aux_echo_arrival_report(spec, cells, dx=float(cells["dx"]), dt=float(run["dt_s"]), n_steps=n)
        expect = _LW.reference_is_absorber_free(rec["t_safe_cpml_steps"], ae["arrival_steps"], n)
        lat = e2["lattice"]
        assert lat["witness_reference_arrival_safe"] is bool(expect), (
            arm, rec["t_safe_cpml_steps"], ae["arrival_steps"], n)
        assert lat["witness_reference"] == _WITNESS_REFERENCE[arm], (arm, lat["witness_reference"])
        # the geometry the comparator recorded is the geometry recomputed here
        wi = lat["witness_reference_inputs"]
        assert wi["n_steps"] == n and wi["t_safe_cpml_steps"] == rec["t_safe_cpml_steps"], (arm, wi)
        assert wi["aux_echo_arrival_steps"] == pytest.approx(float(ae["arrival_steps"]), rel=1e-12), (arm, wi)
        # and the decisive witness: the chosen reference is the one the record
        # actually matches.  R and T are checked separately -- the sign has to
        # agree on both or the mechanism claim is a coincidence on one observable.
        #
        # graze_vac is EXCLUDED here, and the exclusion is asserted rather than
        # assumed: it is the vacuum arm, so it has no scatterer and its two
        # references are the same lattice to 1e-8 relative.  There is no sign to
        # agree with there, and counting it would inflate the claim.  It is also
        # one of the two arms with no defined W_witness, so it is not one of the
        # ten entries the mechanism claim is made over.
        if arm == "graze_vac":
            for obs in ("R", "T"):
                a_, b_ = float(lat[f"mean_d{obs}_lattice_gated"]), float(lat[f"mean_d{obs}_lattice_gated_alt"])
                assert a_ == pytest.approx(b_, rel=1e-8), (arm, obs, a_, b_)
            continue
        for obs in ("R", "T"):
            chosen = float(lat[f"mean_d{obs}_lattice_gated"])
            other = float(lat[f"mean_d{obs}_lattice_gated_alt"])
            assert chosen < other, (arm, obs, chosen, other,
                                    "the arrival predicate picked the reference the record matches WORSE")


def test_the_two_settle60_rungs_are_judged_against_the_absorber_free_lattice():
    """The two entries #1015 section 14 moves MOST, and which no test pinned before.

    `rfx__te_00_settle60.json` / `rfx__tm_00_settle60.json` are the 60-step
    settling rungs of the two normal-incidence arms.  Both have
    `t_safe_cpml_steps` 3141 and an auxiliary echo arriving at 3197 steps
    against records of 1611 and 1612 steps -- neither absorber echo can be in
    either record -- yet they were judged against the lattice WITH both
    absorbers, which put 76.6 % and 75.5 % of their gated bins in the validity
    domain and left 4 and 5 in-domain R breaches (66 more outside it each).

    Against the reference their arrival entitles them to, the domain is the
    whole gated band and the breach count is zero on both observables, with the
    worst per-bin ratio 0.25 rather than 3.06.  Pinned so the correction cannot
    regress silently on the rungs nothing else covers.
    """
    for key, (fname, arm, n_steps, exp_R, exp_T, exp_gl1) in _SETTLE60_EXPECTED.items():
        doc = _artifact(fname)
        ad = doc["arms"][arm]
        spec, run, cells, e2 = _replay_arm(arm, ad)
        assert int(run["n_steps"]) == n_steps, (key, run["n_steps"])
        lat = e2["lattice"]
        assert lat["witness_reference"] == "absorber_free_by_arrival", (key, lat["witness_reference"])
        assert lat["witness_reference_inputs"]["t_safe_cpml_steps"] >= n_steps, key
        assert lat["witness_reference_inputs"]["aux_echo_arrival_steps"] > n_steps, key
        for obs, exp in (("R", exp_R), ("T", exp_T)):
            dom = lat[f"domain_{obs}"]
            got = (dom["n_bins_in_domain"], dom["n_bins_beyond_in_domain"],
                   dom["n_bins_beyond_outside_domain"])
            assert got == exp, (key, obs, got, exp)
            assert dom["domain_fraction"] == 1.0, (key, obs, dom["domain_fraction"])
            assert dom["worst_ratio_in_domain"] <= 0.90, (key, obs, dom["worst_ratio_in_domain"])
        assert (lat["GL1_R_bins_beyond"], lat["GL1_T_bins_beyond"]) == exp_gl1, key
        # GL2 was passing before and must still pass; nothing here may turn a
        # reported failure green, which is the same A2 rule `_GL2_EXPECTED` pins
        assert lat["GL2_R"] and lat["GL2_T"], key


def test_the_family_default_reference_rule_is_unchanged_by_the_cv26_correction():
    """`reference_is_absorber_free` states the rule; `domain_report`'s default
    output must be untouched by it.

    The slab family (cv04, cv22, cv23) reaches `domain_report` with neither
    `unmodelled_term` nor `report_term`, and its dict must keep exactly the six
    keys it had -- no `max_unmodelled_over_window_gated`, no
    `unmodelled_term_is_reported_only`.  `report_term` must also never be able
    to move the domain itself.
    """
    W = np.full(8, 1e-3)
    resid = np.linspace(0.0, 2e-3, 8)
    base = _LW.domain_report(W, resid)
    assert sorted(base) == sorted([
        "n_bins_gated", "n_bins_in_domain", "domain_fraction",
        "n_bins_beyond_in_domain", "n_bins_beyond_outside_domain", "worst_ratio_in_domain"])
    assert base["n_bins_in_domain"] == 8 and base["domain_fraction"] == 1.0

    # report_term reports and does not restrict: the domain stays total even
    # when the reported magnitude is 100x the window
    rep = _LW.domain_report(W, resid, None, report_term=np.full(8, 0.1))
    for k in base:
        assert rep[k] == base[k], k
    assert rep["max_unmodelled_over_window_gated"] == pytest.approx(100.0)
    assert rep["unmodelled_term_is_reported_only"] is True

    # the same magnitude as an unmodelled_term DOES restrict, and says nothing
    # about being reported-only
    real = _LW.domain_report(W, resid, np.full(8, 0.1))
    assert real["n_bins_in_domain"] == 0 and "unmodelled_term_is_reported_only" not in real

    # and the predicate itself is geometry: equal arrival is not safe, strictly
    # greater is, and a non-applicable (negative) arrival is never safe
    assert _LW.reference_is_absorber_free(3141, 3197.0, 1611) is True
    assert _LW.reference_is_absorber_free(1000, 3197.0, 1611) is False      # CPML echo inside
    assert _LW.reference_is_absorber_free(3141, 1611.0, 1611) is False      # aux echo not strictly outside
    assert _LW.reference_is_absorber_free(4776, -8641.0, 21835) is False    # cv26's compact boxes
    assert _LW.reference_is_absorber_free(float("nan"), 3197.0, 1611) is False


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

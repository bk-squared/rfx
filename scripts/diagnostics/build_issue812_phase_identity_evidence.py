#!/usr/bin/env python3
"""Emit the #812 P1 (cv20/cv21) re-gate evidence as a committed JSON artifact.

Round 2 of #812 changed the process, not the physics: numbers live in a
committed artifact the harness writes, and prose REFERENCES an artifact key
instead of restating digits (issue #812 round-1 retrospective; the reference
syntax is the one ``tests/test_evidence_numeric_provenance.py`` resolves,
``path.json::key.path = value``).

This script runs NO FDTD and no external solver. It replays the four
committed configurations of the round-1 lane through the referees' OWN
witness functions and records what they return:

  cv21 registered mesh   VESSL 369367251629 (run-3 stage_b_partial, committed
                         verbatim as the ``_RUN3_*`` literals in
                         tests/crossval/test_coax_two_port_referee_header.py)
  cv21 1.5x refinement   validation/crossval/_21_coax_two_port_referee_logs/
                         mesh_refinement_369367251845_result.json
  cv20 declared board    validation/crossval/_20_msl_phase_referee_logs/
                         20260804T055009Z_result.json (run-1, VESSL 369367251705)
  cv20 realized board    validation/crossval/_20_msl_phase_referee_logs/
                         20260827T102342Z_result.json (run-2, VESSL 369367256520)

Usage::

    PYTHONPATH=<worktree> python scripts/diagnostics/build_issue812_phase_identity_evidence.py \
        [--output validation/crossval/_issue812_phase_identity/regate_evidence.json]

``tests/crossval/test_issue812_phase_identity_evidence.py`` re-runs this builder into a
scratch path and asserts the committed artifact still equals what it produces,
so the artifact cannot go stale against the code it describes.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import pathlib
import sys
from types import ModuleType

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "validation" / "crossval" / "_issue812_phase_identity" / "regate_evidence.json"
)

CV20_PATH = REPO_ROOT / "validation" / "crossval" / "20_msl_phase_referee.py"
CV21_PATH = REPO_ROOT / "validation" / "crossval" / "21_coax_two_port_referee.py"
CV20_RUN1 = REPO_ROOT / "validation/crossval/_20_msl_phase_referee_logs/20260804T055009Z_result.json"
CV20_RUN2 = REPO_ROOT / "validation/crossval/_20_msl_phase_referee_logs/20260827T102342Z_result.json"
CV20_RFX_FIXTURE = REPO_ROOT / "tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json"
CV21_REFINED = REPO_ROOT / (
    "validation/crossval/_21_coax_two_port_referee_logs/mesh_refinement_369367251845_result.json"
)


def _load(name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cx(pairs) -> np.ndarray:
    return np.array([complex(re, im) for re, im in pairs], dtype=np.complex128)


def _failed_result(exc: RuntimeError) -> dict:
    """The witness's own result dict, recovered at FULL precision.

    Every witness in both referees ends its ``RuntimeError`` with
    ``Full result: {result}``. Reading that back beats re-deriving the
    number here: the artifact then records exactly what the gate computed,
    not a second implementation of it.
    """
    return ast.literal_eval(str(exc).split("Full result: ", 1)[1])


# ---------------------------------------------------------------------------
# cv21 -- coax two-port referee
# ---------------------------------------------------------------------------
def _cv21_run3_arrays(cv21: ModuleType):
    """Run-3's own committed field data.

    The arrays live as the ``_RUN3_*`` literals in the header test, which is
    where this repo committed run-3's ``stage_b_partial`` verbatim; importing
    them keeps this builder from re-typing a single digit of them.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from tests.crossval import test_coax_two_port_referee_header as hdr  # noqa: PLC0415

    return hdr._run3_arrays(cv21)


def _cv21_refined_arrays(cv21: ModuleType):
    stage_b = json.loads(CV21_REFINED.read_text())["stage_b"]
    freqs_hz = np.asarray(stage_b["freqs_ghz"], dtype=float) * 1e9
    s21 = _cx(stage_b["s21"])
    d1, d2 = stage_b["drive1_diagnostics"], stage_b["drive2_diagnostics"]
    beta = 0.25 * (
        _cx(d1["beta_port1"]) + _cx(d1["beta_port2"])
        + _cx(d2["beta_port1"]) + _cx(d2["beta_port2"])
    )
    return freqs_hz, s21, beta, cv21.B_L12_MM * 1e-3, float(stage_b["annulus_cells"])


def _analytic_leg(cv21: ModuleType, freqs, s21, beta, l_m, cells, label) -> dict:
    res = cv21._analytic_beta_witness(
        freqs, s21, beta, L_m=l_m, eps_r=cv21.B_PTFE_EPS_R,
        annulus_cells=cells, label=label)
    return {
        "annulus_cells": res["annulus_cells"],
        "envelope_bound_frac": res["envelope_bound_frac"],
        "beta_max_abs_dev_frac": res["beta_max_abs_dev_frac"],
        "gd_max_abs_dev_frac": res["gd_max_abs_dev_frac"],
        "beta_margin_x": res["envelope_bound_frac"] / res["beta_max_abs_dev_frac"],
        "gd_margin_x": res["envelope_bound_frac"] / res["gd_max_abs_dev_frac"],
        "implied_phase_error_deg": res["implied_phase_error_deg"],
        "passed": res["passed"],
    }


def _cv21_block(cv21: ModuleType) -> dict:
    pre = cv21.MESH_REFINEMENT_PREDECLARATION
    freqs3, s21_3, beta3, l_m = _cv21_run3_arrays(cv21)
    cells_reg = float(cv21._stage_b_layout(dx_scale=1.0)["annulus_cells"])
    registered = _analytic_leg(cv21, freqs3, s21_3, beta3, l_m, cells_reg, "registered")

    freqs_r, s21_r, beta_r, l_r, cells_ref = _cv21_refined_arrays(cv21)
    refined = _analytic_leg(cv21, freqs_r, s21_r, beta_r, l_r, cells_ref, "refined")

    # Criterion (B): the coherent phase-velocity error the audit measured the
    # E1 witness blind to, replayed through both witnesses at every k.
    criterion_b = []
    baseline_e1 = cv21._matched_through_witness(
        freqs3, s21_3, L_m=l_m, eps_r=cv21.B_PTFE_EPS_R,
        mag_band=cv21.B_S21_THRU_BAND, label="e1_baseline", beta=beta3)
    for k in (0.50, 1.02, 1.10, 1.30, 1.50, 1.57):
        s21_k = s21_3 * np.exp(-1j * (k - 1.0) * np.real(beta3) * l_m)
        beta_k = beta3 * k
        e1 = cv21._matched_through_witness(
            freqs3, s21_k, L_m=l_m, eps_r=cv21.B_PTFE_EPS_R,
            mag_band=cv21.B_S21_THRU_BAND, label=f"e1_k{k}", beta=beta_k)
        e2_reason = None
        try:
            e2 = cv21._analytic_beta_witness(
                freqs3, s21_k, beta_k, L_m=l_m, eps_r=cv21.B_PTFE_EPS_R,
                annulus_cells=cells_reg, label=f"e2_k{k}")
            e2_fired = False
        except RuntimeError as exc:
            e2_fired = True
            e2 = _failed_result(exc)
            e2_reason = str(exc).split(" Full result:")[0]
        criterion_b.append({
            "k": k,
            "e1_max_phase_dev_deg": e1["max_phase_dev_deg"],
            "e1_fired": not e1["passed"],
            "e2_beta_max_abs_dev_frac": e2["beta_max_abs_dev_frac"],
            "e2_gd_max_abs_dev_frac": e2["gd_max_abs_dev_frac"],
            "e2_fired": e2_fired,
            "e2_failure_reason": e2_reason,
        })

    # The margin is the declared headroom BY CONSTRUCTION, at BOTH meshes: the
    # committed convergence order is itself the two-point fit through the two
    # committed excesses, so the refined-mesh check restates the registered one.
    excess_before = float(pre["excess_before"])
    excess_after = float(pre["measured_excess_after"])
    n_before = float(pre["annulus_cells_before"])
    n_after = float(pre["annulus_cells_after"])
    p_recovered = math.log(excess_before / excess_after) / math.log(n_after / n_before)
    by_construction = {
        "headroom_declared": float(cv21.BETA_ENVELOPE_HEADROOM),
        "committed_excess_before": excess_before,
        "committed_excess_after": excess_after,
        "committed_order_p": float(pre["implied_convergence_order"]),
        "order_p_recovered_from_the_two_committed_excesses": p_recovered,
        "order_p_recovery_abs_error": abs(p_recovered - float(pre["implied_convergence_order"])),
        "bound_at_n_after_minus_headroom_times_excess_after": (
            cv21._beta_envelope_bound(n_after) - cv21.BETA_ENVELOPE_HEADROOM * excess_after
        ),
    }

    def _floor(bound: float, excess: float) -> dict:
        return {"k_hi": (1.0 + bound) / (1.0 + excess),
                "k_lo": (1.0 - bound) / (1.0 + excess)}

    # Round-2 review (B1): the detection floor is the declared HEADROOM's
    # choice, not a property of the annulus. Same formula, same committed
    # data, other headrooms -- and the tightest envelope the registered data
    # itself admits, which is the only floor physics forces at this mesh.
    headroom_declared = float(cv21.BETA_ENVELOPE_HEADROOM)
    bound_reg_declared = cv21._beta_envelope_bound(n_before)
    headroom_dependence = []
    for h in (1.30, 1.10, 1.04):
        bound_reg_h = bound_reg_declared * h / headroom_declared
        bound_ref_h = refined["envelope_bound_frac"] * h / headroom_declared
        headroom_dependence.append({
            "headroom": h,
            "envelope_bound_frac_registered": bound_reg_h,
            "k_hi": (1.0 + bound_reg_h) / (1.0 + excess_before),
            "k_lo": (1.0 - bound_reg_h) / (1.0 + excess_before),
            "criterion_a_still_passes_registered": bool(
                registered["beta_max_abs_dev_frac"] <= bound_reg_h
                and registered["gd_max_abs_dev_frac"] <= bound_reg_h),
            "criterion_a_still_passes_refined": bool(
                refined["beta_max_abs_dev_frac"] <= bound_ref_h
                and refined["gd_max_abs_dev_frac"] <= bound_ref_h),
        })
    physics_forced_bound = max(registered["beta_max_abs_dev_frac"],
                               registered["gd_max_abs_dev_frac"])
    physics_forced = {
        "convention": (
            "the tightest envelope that still admits the committed registered-mesh "
            "data (per-bin max over the beta and group-delay legs); no headroom "
            "choice can place the floor closer to k = 1 than this at this mesh."
        ),
        "bound_frac": physics_forced_bound,
        "implied_headroom": physics_forced_bound / (bound_reg_declared / headroom_declared),
        "k_hi": (1.0 + physics_forced_bound) / (1.0 + excess_before),
        "k_lo": (1.0 - physics_forced_bound) / (1.0 + excess_before),
    }

    return {
        "registered_mesh": registered,
        "refined_mesh": refined,
        "margin_is_the_declared_headroom_by_construction": by_construction,
        "detection_floor": {
            "convention": (
                "k fires when |k*(1+excess) - 1| > BOUND; the declared floor uses the "
                "committed record's own excess and rounded cell count, the measured "
                "floors use each configuration's replayed per-bin max."
            ),
            "declared_registered": _floor(
                cv21._beta_envelope_bound(n_before), excess_before),
            "measured_registered": _floor(
                registered["envelope_bound_frac"], registered["beta_max_abs_dev_frac"]),
            "measured_refined": _floor(
                refined["envelope_bound_frac"], refined["beta_max_abs_dev_frac"]),
            "predeclared_k_hi": float(cv21.BETA_ENVELOPE_PREDECLARATION["detection_floor_k_hi"]),
            "predeclared_k_lo": float(cv21.BETA_ENVELOPE_PREDECLARATION["detection_floor_k_lo"]),
            "headroom_dependence": headroom_dependence,
            "physics_forced_registered": physics_forced,
        },
        "criterion_b": criterion_b,
        "e1_baseline_max_phase_dev_deg": baseline_e1["max_phase_dev_deg"],
        "evidence_levels_supported_by_a_leg_in_this_case": ["E1", "E2"],
        "e4_supporting_leg_count": 0,
        "e4_not_supported_because": (
            "the script imports no rfx module and reads no rfx fixture, so no leg "
            "compares an rfx quantity against the external solver; the E4 comparison "
            "that consumes this referee's openEMS output lives downstream, in the "
            "compute_coaxial_two_port label-lift chain recorded in "
            "docs/guides/sparameter_support_matrix.md"
        ),
    }


# ---------------------------------------------------------------------------
# cv20 -- MSL phase referee
# ---------------------------------------------------------------------------
def _cv20_legs(cv20: ModuleType, result_path: pathlib.Path, eps_eff_openems: float) -> dict:
    stage_b = json.loads(result_path.read_text())["stage_b"]
    freqs = np.asarray(stage_b["freqs_hz"], dtype=float)
    cross = stage_b["cross_solver_report"]
    fixture = cv20._load_rfx_fixture(str(CV20_RFX_FIXTURE))
    eps_rfx = cv20._hammerstad_jensen_eps_eff(
        fixture["meta"]["w_trace_realized_m"], fixture["meta"]["h_sub_realized_m"], cv20.B_EPS_R)
    rfx = cv20._analytic_beta_witness(
        freqs, np.asarray(cross["beta_rfx_real"], dtype=float), eps_eff=eps_rfx,
        tol_frac=cv20.B_BETA_ANALYTIC_TOL_FRAC, label="replay", solver="rfx")
    oe = cv20._analytic_beta_witness(
        freqs, np.asarray(cross["beta_openems_real"], dtype=float), eps_eff=eps_eff_openems,
        tol_frac=cv20.B_BETA_ANALYTIC_TOL_FRAC, label="replay", solver="openems")
    xs = cv20._cross_solver_phase_witness(
        freqs, np.asarray(cross["raw_phase_diff_deg"], dtype=float),
        tol_deg=cv20.B_CROSS_SOLVER_PHASE_TOL_DEG, label="replay")
    return {
        "analytic_beta_rfx_max_abs_dev_frac": rfx["max_abs_dev_frac"],
        "analytic_beta_openems_max_abs_dev_frac": oe["max_abs_dev_frac"],
        "analytic_beta_tol_frac": float(cv20.B_BETA_ANALYTIC_TOL_FRAC),
        "cross_solver_max_abs_raw_phase_diff_deg": xs["max_abs_raw_phase_diff_deg"],
        "cross_solver_tol_deg": float(cv20.B_CROSS_SOLVER_PHASE_TOL_DEG),
        "cross_solver_margin_x": (
            float(cv20.B_CROSS_SOLVER_PHASE_TOL_DEG) / xs["max_abs_raw_phase_diff_deg"]),
        "all_three_passed": bool(rfx["passed"] and oe["passed"] and xs["passed"]),
    }


def _cv20_block(cv20: ModuleType) -> dict:
    eps_declared = cv20._hammerstad_jensen_eps_eff(600e-6, 254e-6, cv20.B_EPS_R)
    fixture = cv20._load_rfx_fixture(str(CV20_RFX_FIXTURE))
    eps_realized = cv20._hammerstad_jensen_eps_eff(
        fixture["meta"]["w_trace_realized_m"], fixture["meta"]["h_sub_realized_m"], cv20.B_EPS_R)

    run1 = _cv20_legs(cv20, CV20_RUN1, eps_declared)
    run2 = _cv20_legs(cv20, CV20_RUN2, eps_realized)

    # Criterion (B), on run-2: the rfx side's phase velocity scaled coherently.
    stage_b = json.loads(CV20_RUN2.read_text())["stage_b"]
    freqs = np.asarray(stage_b["freqs_hz"], dtype=float)
    l12 = float(stage_b["layout"]["l12_m"])
    s21_openems = _cx(stage_b["s21"])
    s21_rfx = _cx(fixture["s21"])
    beta_rfx = np.real(_cx(fixture["beta_first_port"]))

    criterion_b = []
    for k in (0.5, 2.0):
        beta_bad = beta_rfx * k
        s21_bad = s21_rfx * np.exp(-1j * (k - 1.0) * beta_rfx * l12)
        e2_reason = None
        try:
            e2 = cv20._analytic_beta_witness(
                freqs, beta_bad, eps_eff=eps_realized,
                tol_frac=cv20.B_BETA_ANALYTIC_TOL_FRAC, label="perturbed", solver="rfx")
            e2_fired = False
        except RuntimeError as exc:
            e2_fired = True
            e2 = _failed_result(exc)
            e2_reason = str(exc).split(" Full result:")[0]
        e2_dev = e2["max_abs_dev_frac"]
        raw = np.degrees(np.angle(np.exp(
            1j * (np.unwrap(np.angle(s21_bad)) - np.unwrap(np.angle(s21_openems))))))
        e4_reason = None
        try:
            xs = cv20._cross_solver_phase_witness(
                freqs, raw, tol_deg=cv20.B_CROSS_SOLVER_PHASE_TOL_DEG, label="perturbed")
            e4_fired = False
        except RuntimeError as exc:
            e4_fired = True
            xs = _failed_result(exc)
            e4_reason = str(exc).split(" Full result:")[0]
        e4_dev = xs["max_abs_raw_phase_diff_deg"]
        e1 = cv20._self_consistency_witness(
            freqs, s21_bad, beta_bad, l12_m=l12, mag_band=cv20.B_S21_MAG_BAND,
            phase_tol_deg=cv20.B_PHASE_TOL_DEG, gd_tol_ps=cv20.B_GD_TOL_PS, label="old")
        criterion_b.append({
            "k": k,
            "e1_self_consistency_fired": not e1["passed"],
            "e1_max_phase_dev_deg": e1["max_phase_dev_deg"],
            "e2_analytic_beta_rfx_dev_frac": e2_dev,
            "e2_fired": e2_fired,
            "e4_cross_solver_max_abs_raw_phase_diff_deg": e4_dev,
            "e4_fired": e4_fired,
            "e2_failure_reason": e2_reason,
            "e4_failure_reason": e4_reason,
            "e4_attributes_to_a_solver": xs["attributes_to_a_solver"],
        })

    # Round-2 review (B2): the audit's own blindness construction (scale the
    # de-embedded phase, which scales the extraction residual with it) and the
    # dispersion-corrected residual, both previously restated in prose with no
    # artifact key. Same replay data as criterion (B) above.
    s21_scaled = np.abs(s21_rfx) * np.exp(2.0j * np.angle(s21_rfx))
    audit = cv20._self_consistency_witness(
        freqs, s21_scaled, beta_rfx * 2.0, l12_m=l12, mag_band=cv20.B_S21_MAG_BAND,
        phase_tol_deg=cv20.B_PHASE_TOL_DEG, gd_tol_ps=cv20.B_GD_TOL_PS,
        label="audit_construction")
    beta_openems = np.asarray(stage_b["cross_solver_report"]["beta_openems_real"], dtype=float)
    mask = cv20._gate_band_mask(freqs)

    def _residual_after_dispersion(s21_r, beta_r):
        raw = np.degrees(np.angle(np.exp(
            1j * (np.unwrap(np.angle(s21_r)) - np.unwrap(np.angle(s21_openems))))))
        return np.degrees(np.angle(np.exp(
            1j * (np.radians(raw) - (beta_openems - beta_r) * l12))))

    resid_baseline = _residual_after_dispersion(s21_rfx, beta_rfx)
    resid_k2 = _residual_after_dispersion(
        s21_rfx * np.exp(-1j * beta_rfx * l12), 2.0 * beta_rfx)
    blindness = {
        "audit_construction_e1_max_phase_dev_deg": audit["max_phase_dev_deg"],
        "audit_construction_e1_passed": bool(audit["passed"]),
        "e1_tol_deg": float(cv20.B_PHASE_TOL_DEG),
        "dispersion_corrected_residual_max_abs_deg_baseline": float(np.max(np.abs(resid_baseline[mask]))),
        "dispersion_corrected_residual_max_abs_deg_k2": float(np.max(np.abs(resid_k2[mask]))),
        "dispersion_corrected_residual_max_abs_change_deg_k2": float(np.max(np.abs(resid_k2 - resid_baseline))),
    }

    return {
        "blindness": blindness,
        "eps_eff_hammerstad_jensen_declared_board": eps_declared,
        "eps_eff_hammerstad_jensen_realized_board": eps_realized,
        "run1_declared_board": run1,
        "run2_realized_board": run2,
        "criterion_b": criterion_b,
        "cross_solver_raw_phase_difference_is_gated": True,
        "evidence_levels_supported_by_a_leg_in_this_case": ["E1", "E2", "E4"],
        "e4_supporting_leg_count": 1,
    }


def build() -> dict:
    cv20 = _load("cv20_msl_phase_referee", CV20_PATH)
    cv21 = _load("cv21_coax_two_port_referee", CV21_PATH)
    return {
        "schema": "issue812_phase_identity_regate_evidence/1",
        "issue": 812,
        "phase": "P1 self-referential phase gates (cv20, cv21)",
        "round": 2,
        "generated_by": "scripts/diagnostics/build_issue812_phase_identity_evidence.py",
        "runs_no_fdtd": True,
        "sources": {
            "cv21_registered_mesh": "VESSL 369367251629 (run-3 stage_b_partial, committed "
                                    "verbatim as tests/crossval/test_coax_two_port_referee_header.py's "
                                    "_RUN3_* literals)",
            "cv21_refined_mesh": "validation/crossval/_21_coax_two_port_referee_logs/"
                                 "mesh_refinement_369367251845_result.json",
            "cv20_run1_declared_board": "validation/crossval/_20_msl_phase_referee_logs/"
                                        "20260804T055009Z_result.json",
            "cv20_run2_realized_board": "validation/crossval/_20_msl_phase_referee_logs/"
                                        "20260827T102342Z_result.json",
            "cv20_rfx_fixture": "tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json",
        },
        "cv20": _cv20_block(cv20),
        "cv21": _cv21_block(cv21),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    payload = build()
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

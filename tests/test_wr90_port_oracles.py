from __future__ import annotations

import json
from argparse import Namespace

import jax.numpy as jnp
import numpy as np
import scripts._wr90_port_oracle_matrix as oracle


def test_ref_free_gamma_solver_recovers_synthetic_three_plane_line():
    x = np.array([0.02, 0.04, 0.06])
    beta = 121.5
    a_plus = 1.2 - 0.3j
    gamma_true = 0.997 * np.exp(1j * 0.42)
    a_minus = gamma_true * a_plus
    u = a_plus * np.exp(-1j * beta * x) + a_minus * np.exp(+1j * beta * x)

    fit = oracle.solve_ref_free_gamma(x, u, beta)

    assert fit.rank[0] == 2
    assert fit.residual_norm[0] < 1e-12
    assert abs(fit.gamma[0] - gamma_true) < 1e-12


def test_ref_free_gamma_solver_handles_discrete_beta_band_and_noisy_overdetermined_fit():
    rng = np.random.default_rng(1234)
    x = np.array([0.018, 0.031, 0.047, 0.064, 0.079])
    beta = np.array([97.0, 123.0, 148.0])
    a_plus = np.array([1.0 + 0.1j, 0.9 - 0.2j, 1.1 + 0.05j])
    gamma_true = np.array([0.94 * np.exp(0.2j), 0.97 * np.exp(-0.3j), 1.01 * np.exp(0.1j)])
    samples = np.column_stack([
        a_plus[i] * np.exp(-1j * beta[i] * x)
        + gamma_true[i] * a_plus[i] * np.exp(+1j * beta[i] * x)
        for i in range(beta.size)
    ])
    samples += 1e-5 * (rng.normal(size=samples.shape) + 1j * rng.normal(size=samples.shape))

    fit = oracle.solve_ref_free_gamma(x, samples, beta)

    assert fit.gamma.shape == (3,)
    assert np.max(np.abs(fit.gamma - gamma_true)) < 1e-4
    assert np.max(fit.residual_norm) < 1e-4
    assert np.all(np.isfinite(fit.condition))


def test_integer_cycle_lockin_recovers_known_amplitude_and_phase():
    freq = 2.0e9
    samples_per_period = 64
    dt = 1.0 / (freq * samples_per_period)
    amp = 0.73
    phase = -0.61
    n = np.arange(samples_per_period * 9)
    signal = amp * np.cos(2.0 * np.pi * freq * n * dt + phase)

    phasor = oracle.integer_cycle_lockin(signal, freq, dt, start_index=samples_per_period, n_cycles=6)

    assert abs(abs(phasor) - amp) < 1e-12
    assert abs(np.angle(phasor * np.exp(-1j * phase))) < 1e-12


def test_emit_jsonl_rows_is_machine_readable(tmp_path):
    rows = [oracle.synthetic_least_squares_control(), oracle.cw_lockin_control_row()]
    path = tmp_path / "rows.jsonl"

    written = oracle.emit_jsonl_rows(rows, path)

    assert written == str(path)
    payload = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["status"] for row in payload] == ["control", "control"]
    assert payload[0]["case"] == "synthetic_ref_free_least_squares_control"
    assert payload[1]["case"] == "cw_lockin_synthetic_control"


def test_quick_matrix_contract_with_physical_rows_monkeypatched(monkeypatch):
    baseline = oracle.OracleRow(
        "baseline_current_2run_internal_mask_current_cpml",
        "current_2run",
        "ok",
        {"mean_abs_s11": 0.955, "monitor_backend": "production_two_run_waveguide_s_matrix"},
        "B_or_C_if_ref_free_good",
    )
    ref_free = oracle.OracleRow(
        "ref_free_3plane_internal_mask_current_cpml",
        "ref_free_multiplane",
        "ok",
        {
            "mean_abs_s11": 0.999,
            "monitor_backend": "passive_waveguide_ref_voltage",
            "monitor_distances_m": [0.02, 0.03, 0.04],
            "source_short_distance_m": 0.075,
            "short_type": "internal_mask",
            "cpml_layers": 10,
            "beta_type": "yee_discrete",
            "dft_window": "rect_full_record",
            "fit_residual": 1e-3,
            "fit_cond": 3.0,
        },
        "A_D_or_C3_if_ref_free_deficit_persists",
    )
    monkeypatch.setattr(oracle, "run_current_2run_baseline", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(oracle, "run_reference_free_case", lambda *args, **kwargs: ref_free)
    args = Namespace(
        synthetic_only=False,
        cpml_layers=10,
        num_periods=40.0,
        dx=None,
        full=False,
        monitor_x_m=[0.03, 0.045, 0.06],
        center_freq_hz=None,
        freq_min_hz=5.0e9,
        freq_max_hz=7.0e9,
        n_freqs=3,
    )

    rows = oracle.run_matrix(args)

    cases = {row.case for row in rows}
    assert "baseline_current_2run_internal_mask_current_cpml" in cases
    assert "ref_free_3plane_internal_mask_current_cpml" in cases
    assert "synthetic_ref_free_least_squares_control" in cases
    ref_payload = next(row.to_jsonable() for row in rows if row.case == "ref_free_3plane_internal_mask_current_cpml")
    for key in (
        "monitor_backend",
        "monitor_distances_m",
        "source_short_distance_m",
        "short_type",
        "cpml_layers",
        "beta_type",
        "dft_window",
        "fit_residual",
        "fit_cond",
    ):
        assert key in ref_payload


def test_full_matrix_emits_explicit_deferred_rows(monkeypatch):
    ok = oracle.OracleRow("ref_free_3plane_internal_mask_current_cpml", "ref_free_multiplane", "ok", {"mean_abs_s11": 0.99})
    monkeypatch.setattr(oracle, "run_current_2run_baseline", lambda *args, **kwargs: oracle.OracleRow("baseline_current_2run_internal_mask_current_cpml", "current_2run", "ok"))
    monkeypatch.setattr(oracle, "run_reference_free_case", lambda *args, **kwargs: ok)
    monkeypatch.setattr(oracle, "run_source_purity_empty_line_sweep", lambda *args, **kwargs: oracle.OracleRow("source_purity_empty_line_sweep", "source_purity_line_sweep", "ok"))
    args = Namespace(
        synthetic_only=False,
        cpml_layers=10,
        num_periods=40.0,
        dx=None,
        full=True,
        monitor_x_m=[0.03, 0.045, 0.06],
        center_freq_hz=None,
        freq_min_hz=5.0e9,
        freq_max_hz=7.0e9,
        n_freqs=3,
        cw_warmup_cycles=20,
        cw_lockin_cycles=20,
    )

    rows = oracle.run_matrix(args)

    payload = [row.to_jsonable() for row in rows]
    cases = {row["case"] for row in payload}
    assert "pml_sweep_current_2run_layers_10_20_40" in cases
    assert "face_short_ref_free_no_cpml_or_irrelevant" in cases
    assert any(row["status"] == "skipped" and row.get("skip_reason") for row in payload)


def test_reference_free_phase_shift_convention_recovers_gamma_at_reporting_plane():
    reporting_planes = {
        "source": 0.011,
        "reference": 0.037,
        "short": 0.075,
    }
    beta = 132.0
    gamma_at_origin = 0.88 * np.exp(0.73j)
    a_plus = 0.9 - 0.2j
    x = np.array([0.021, 0.034, 0.052, 0.071])
    samples = (
        a_plus * np.exp(-1j * beta * x)
        + gamma_at_origin * a_plus * np.exp(+1j * beta * x)
    )

    fit = oracle.solve_ref_free_gamma(x, samples, beta)

    assert oracle.PHASE_SHIFT_CONVENTION == "gamma_ref=gamma_origin*exp(+2j*beta*x_ref)"
    assert abs(fit.gamma[0] - gamma_at_origin) < 1e-12
    for x_ref in reporting_planes.values():
        gamma_at_ref = gamma_at_origin * np.exp(+2j * beta * x_ref)
        shifted = oracle.shift_gamma_to_reference_plane(fit.gamma, beta, x_ref)
        assert abs(shifted[0] - gamma_at_ref) < 1e-12


def test_current_normalization_formula_dissection_handles_safe_denominators():
    a_ref = np.array([2.0 + 0.0j, 0.0 + 0.0j])
    b_ref = np.array([0.1 + 0.2j, 0.3 + 0.0j])
    b_dev = np.array([1.1 + 0.2j, 0.7 + 0.0j])
    a_dev = np.array([1.0 + 0.0j, 4.0 + 0.0j])

    formulas = oracle.assemble_current_normalization_formulas(
        a_inc_ref_drive=a_ref,
        b_ref_drive=b_ref,
        b_dev_drive=b_dev,
        a_inc_dev_drive=a_dev,
    )

    assert np.allclose(formulas["current_formula"][0], 0.5 + 0.0j)
    # Zero reference denominator falls back to 1.0, keeping diagnostics finite.
    assert np.allclose(formulas["current_formula"][1], 0.4 + 0.0j)
    assert np.allclose(formulas["no_subtraction_formula"][0], 0.55 + 0.1j)
    assert np.allclose(formulas["device_denominator_formula"][1], 0.1 + 0.0j)


def test_bc_diagnostics_matrix_contract_monkeypatched(monkeypatch):
    baseline = oracle.OracleRow("baseline_current_2run_internal_mask_current_cpml", "current_2run", "ok")
    ref_free = oracle.OracleRow("ref_free_3plane_internal_mask_current_cpml", "ref_free_multiplane", "ok")
    dissection = oracle.OracleRow(
        "current_norm_dissection_pec_short",
        "current_2run_formula_dissection",
        "ok",
        {
            "current_formula": [1 + 0j],
            "reference_free_gamma_at_current_reference_plane": [1 + 0j],
            "gamma_fit_origin_phase_deg": [0.0],
            "phase_reference_plane_m": 0.01,
            "gamma_at_phase_reference_plane_phase_deg": [1.0],
            "phase_shift_convention": oracle.PHASE_SHIFT_CONVENTION,
        },
    )
    period = oracle.OracleRow("period_sweep_current_vs_ref_free", "period_sensitivity", "ok")
    pml = oracle.OracleRow("pml_sweep_current_vs_ref_free_layers_8_10_12", "cpml_sensitivity", "ok")
    ref_plane = oracle.OracleRow("reference_plane_sweep_current_2run", "reference_plane_sensitivity", "ok")
    monkeypatch.setattr(oracle, "run_current_2run_baseline", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(oracle, "run_reference_free_case", lambda *args, **kwargs: ref_free)
    monkeypatch.setattr(oracle, "run_current_norm_dissection", lambda *args, **kwargs: dissection)
    monkeypatch.setattr(oracle, "run_period_sweep_current_vs_ref_free", lambda *args, **kwargs: period)
    monkeypatch.setattr(oracle, "run_pml_sweep_current_vs_ref_free", lambda *args, **kwargs: pml)
    monkeypatch.setattr(oracle, "run_reference_plane_sweep_current_2run", lambda *args, **kwargs: ref_plane)
    args = Namespace(
        synthetic_only=False,
        bc_diagnostics=True,
        cpml_layers=10,
        num_periods=40.0,
        dx=None,
        full=False,
        monitor_x_m=[0.03, 0.045, 0.06],
        center_freq_hz=None,
        freq_min_hz=5.0e9,
        freq_max_hz=7.0e9,
        n_freqs=3,
    )

    rows = oracle.run_matrix(args)

    payload = {row.case: row.to_jsonable() for row in rows}
    assert "current_norm_dissection_pec_short" in payload
    assert "period_sweep_current_vs_ref_free" in payload
    assert "pml_sweep_current_vs_ref_free_layers_8_10_12" in payload
    assert "reference_plane_sweep_current_2run" in payload
    assert payload["current_norm_dissection_pec_short"]["phase_shift_convention"] == oracle.PHASE_SHIFT_CONVENTION


def test_phase2a_windowed_dft_full_window_matches_rect_dft():
    freq = 1.5e9
    dt = 1.0 / (freq * 32)
    n = np.arange(32 * 6)
    signal = np.cos(2.0 * np.pi * freq * n * dt + 0.4)

    windowed = oracle._rect_dft_windowed(signal, np.array([freq]), dt, 0, signal.size)
    reference = np.asarray(oracle._rect_dft(jnp.asarray(signal), jnp.asarray([freq]), dt, signal.size))

    assert np.max(np.abs(windowed - reference)) < 1e-12
    assert oracle._window_bounds(signal.size, "late_half", dt=dt) == (signal.size // 2, signal.size)
    metrics = oracle._window_energy_metrics(signal, signal.size // 2, signal.size, signal.size)
    assert 0.0 < metrics["window_energy_fraction_vs_full"] < 1.0


def test_phase2a_verdict_allows_ranked_inconclusive_mixed_signals():
    rows = [
        oracle.OracleRow(
            "phase2a_ref_free_period_sweep_40_80_120",
            "ref_free_period_sensitivity",
            "ok",
            {"ref_free_delta_mean_abs_s11": 0.05, "ref_free_delta_min_abs_s11": 0.01},
        ),
        oracle.OracleRow(
            "phase2a_ref_free_same_run_center_vs_band",
            "same_run_ref_free_center_vs_band",
            "ok",
            {"same_run_center_minus_band_mean_abs": 0.04},
        ),
    ]

    verdict = oracle.run_phase2a_oracle_stability_verdict(rows).to_jsonable()

    assert verdict["primary_classification"] == "inconclusive_keep_no_fix_gate"
    assert "dft_window_settling_likely" in verdict["secondary_signals"]
    assert "band_vs_center_artifact_likely" in verdict["secondary_signals"]
    assert verdict["no_production_fix_gate"] == "closed"
    assert verdict["issues_13_17_resolved"] is False


def test_phase2a_diagnostics_matrix_contract_and_artifact_mode(monkeypatch):
    baseline = oracle.OracleRow("baseline_current_2run_internal_mask_current_cpml", "current_2run", "ok")
    ref_free = oracle.OracleRow("ref_free_3plane_internal_mask_current_cpml", "ref_free_multiplane", "ok")
    verdict = oracle.OracleRow(
        "phase2a_oracle_stability_verdict",
        "ranked_oracle_stability_classification",
        "ok",
        {
            "primary_classification": "inconclusive_keep_no_fix_gate",
            "no_production_fix_gate": "closed",
            "issues_13_17_resolved": False,
            "strict_closure_claimed": False,
        },
    )
    gate = oracle.OracleRow("phase2a_no_production_fix_gate", "no_production_fix_gate", "ok")
    monkeypatch.setattr(oracle, "run_current_2run_baseline", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(oracle, "run_reference_free_case", lambda *args, **kwargs: ref_free)
    monkeypatch.setattr(oracle, "run_phase2a_diagnostics", lambda *args, **kwargs: [verdict, gate])
    args = Namespace(
        synthetic_only=False,
        bc_diagnostics=False,
        phase2a_diagnostics=True,
        cpml_layers=10,
        num_periods=40.0,
        dx=None,
        full=False,
        monitor_x_m=[0.03, 0.045, 0.06],
        center_freq_hz=None,
        freq_min_hz=5.0e9,
        freq_max_hz=7.0e9,
        n_freqs=3,
    )

    rows = oracle.run_matrix(args)
    payload = {row.case: row.to_jsonable() for row in rows}

    assert "phase2a_oracle_stability_verdict" in payload
    assert "phase2a_no_production_fix_gate" in payload
    assert payload["phase2a_oracle_stability_verdict"]["issues_13_17_resolved"] is False

    captured: dict[str, str] = {}

    def fake_emit(rows_arg, path_or_stdout=None):
        captured["path"] = str(path_or_stdout)
        return str(path_or_stdout)

    monkeypatch.setattr(oracle, "run_matrix", lambda args_arg: [verdict, gate])
    monkeypatch.setattr(oracle, "emit_jsonl_rows", fake_emit)

    parsed = oracle.parse_args(["--matrix", "--phase2a-diagnostics"])
    assert parsed.phase2a_diagnostics is True
    assert oracle.main(["--matrix", "--phase2a-diagnostics"]) == 0
    assert "phase2a" in captured["path"]
    assert "quick" not in captured["path"]
    assert "full" not in captured["path"]
    assert "bc" not in captured["path"]


def test_phase2b_row_schema_and_verdict_reject_missing_fields():
    good = oracle._phase2b_row(
        case="phase2b_schema_good",
        method="schema_test",
        stage_id="stage_schema",
        invariant="schema_complete_rows_are_interpretable",
        hypothesis="missing_fields_make_phase2b_evidence_non_interpretable",
        physical_expected="all required fields present",
        observed={"value": 1.0},
        threshold={"value": 1.0},
        threshold_rationale="unit-test schema contract",
        geometry_scope="profile_only",
        passed=True,
    )
    bad = oracle.OracleRow("phase2b_schema_bad", "schema_test", "ok", {"stage_id": "stage_schema"})

    assert oracle.phase2b_schema_errors(good) == []
    assert "invariant" in oracle.phase2b_schema_errors(bad)

    verdict = oracle.run_phase2b_verdict([good, bad]).to_jsonable()
    assert verdict["observed"]["primary_classification"] == "schema_invalid_no_interpretation"
    assert verdict["pass"] is False


def test_phase2b_diagnostics_matrix_contract_and_artifact_mode(monkeypatch):
    baseline = oracle.OracleRow("baseline_current_2run_internal_mask_current_cpml", "current_2run", "ok")
    ref_free = oracle.OracleRow("ref_free_3plane_internal_mask_current_cpml", "ref_free_multiplane", "ok")
    verdict = oracle._phase2b_row(
        case="phase2b_physics_ladder_verdict",
        method="ranked_physics_ladder_classification",
        stage_id="stage_final_verdict",
        invariant="schema_complete_stage_ordered_physics_evidence",
        hypothesis="classification_only",
        physical_expected="no strict closure",
        observed={"primary_classification": "blocked_at_stage2_source_empty_guide_purity"},
        threshold={"schema_errors": 0},
        threshold_rationale="classification row",
        geometry_scope="production_path",
        passed=True,
        blocks_next_stage=True,
    )
    gate = oracle._phase2b_row(
        case="phase2b_no_production_fix_gate",
        method="no_production_fix_gate",
        stage_id="stage7_production_fix_gate",
        invariant="production_fix_requires_invariant",
        hypothesis="gate_closed",
        physical_expected="closed",
        observed={"no_production_fix_gate": "closed", "issues_13_17_resolved": False},
        threshold={"stage7_gate": "closed"},
        threshold_rationale="diagnostic only",
        geometry_scope="production_path",
        passed=True,
    )
    monkeypatch.setattr(oracle, "run_current_2run_baseline", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(oracle, "run_reference_free_case", lambda *args, **kwargs: ref_free)
    monkeypatch.setattr(oracle, "run_phase2b_diagnostics", lambda *args, **kwargs: [verdict, gate])
    args = Namespace(
        synthetic_only=False,
        bc_diagnostics=False,
        phase2a_diagnostics=False,
        phase2b_diagnostics=True,
        cpml_layers=10,
        num_periods=40.0,
        dx=None,
        full=False,
        monitor_x_m=[0.03, 0.045, 0.06],
        center_freq_hz=None,
        freq_min_hz=5.0e9,
        freq_max_hz=7.0e9,
        n_freqs=3,
    )

    rows = oracle.run_matrix(args)
    payload = {row.case: row.to_jsonable() for row in rows}

    assert "phase2b_physics_ladder_verdict" in payload
    assert "phase2b_no_production_fix_gate" in payload
    assert payload["phase2b_no_production_fix_gate"]["observed"]["issues_13_17_resolved"] is False

    captured: dict[str, str] = {}

    def fake_emit(rows_arg, path_or_stdout=None):
        captured["path"] = str(path_or_stdout)
        written_rows = list(rows_arg)
        assert oracle.validate_phase2b_rows(written_rows) == []
        return str(path_or_stdout)

    monkeypatch.setattr(oracle, "run_matrix", lambda args_arg: [verdict, gate])
    monkeypatch.setattr(oracle, "emit_jsonl_rows", fake_emit)

    parsed = oracle.parse_args(["--matrix", "--phase2b-physics-ladder"])
    assert parsed.phase2b_diagnostics is True
    assert oracle.main(["--matrix", "--phase2b-physics-ladder"]) == 0
    assert "phase2b" in captured["path"]
    assert "phase2a" not in captured["path"]
    assert "quick" not in captured["path"]

"""Freeze tracked scripts output paths; new logs, arrays and JSON need review.

Existing evidence stays beside its readers. JSON is deliberately classified by
suffix, including configuration JSON, so a differently named result cannot evade
this contract. This checks the git index and skips exports without git metadata.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._git_tracked import git_available

REPO = Path(__file__).resolve().parents[2]
OUTPUT_SUFFIXES = {".log", ".out", ".txt", ".npz", ".json"}
# Survivors after the B10 scripts cleanup at base 1f88121fc.
ALLOWED_OUTPUTS = frozenset("""\
scripts/ci/file_size_baseline.json
scripts/ci/layer_baseline.json
scripts/diagnostics/_artifacts/cpml_subpixel_stability/amplification.json
scripts/diagnostics/_artifacts/cpml_subpixel_stability/coefficients_main.json
scripts/diagnostics/_artifacts/cpml_subpixel_stability/cv01_head.json
scripts/diagnostics/_artifacts/cpml_subpixel_stability/cv01_main.json
scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json
scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json
scripts/diagnostics/_artifacts/cv03_seam_facet/cv01.json
scripts/diagnostics/_artifacts/cv03_seam_facet/pad_replication_shape.json
scripts/diagnostics/_artifacts/cv03_seam_facet/profile.json
scripts/diagnostics/_artifacts/cv03_seam_facet/sweep.json
scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json
scripts/diagnostics/_artifacts/palace_notch/postpro/notch_full_4090/palace.json
scripts/diagnostics/_artifacts/palace_notch/postpro/notch_full_mid/palace.json
scripts/diagnostics/_artifacts/palace_notch/postpro/notch_probe_4090/palace.json
scripts/diagnostics/_artifacts/palace_notch/postpro/notch_probe_fine/palace.json
scripts/diagnostics/_artifacts/palace_notch/postpro/notch_probe_mid/palace.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/corrected_369367257613_attempt3_x64-0_result.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/corrected_369367257613_attempt3_x64-0_witnesses.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/corrected_369367257614_attempt3b_x64-0_result.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/corrected_369367257614_attempt3b_x64-0_run.log
scripts/diagnostics/_coax_msl_transition_settled_run_logs/corrected_369367257614_attempt3b_x64-0_witnesses.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/remeasure_369367257263_attempt2_x64-0_result.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/remeasure_369367257264_attempt2_wide_x64-0_result.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/remeasure_369367257283_attempt3_x64-0_result.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/settled_run_369367252283_result.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/settled_run_369367252283_run.log
scripts/diagnostics/_coax_msl_transition_settled_run_logs/witnesses_369367257533_attempt3_x64-0_ladders.npz
scripts/diagnostics/_coax_msl_transition_settled_run_logs/witnesses_369367257533_attempt3_x64-0_result.json
scripts/diagnostics/_coax_msl_transition_settled_run_logs/witnesses_369367257533_attempt3_x64-0_witnesses.json
scripts/diagnostics/_mixed_refplane_logs/measurement_369367257597_60p.json
scripts/diagnostics/_msl_passive_port_reflection_logs/20260830_vessl_369367257265/summary.json
scripts/diagnostics/boundary_model/B0/MATRIX.json
scripts/diagnostics/boundary_model/B0/WITNESS.json
scripts/diagnostics/boundary_model/B0/witness/W1__forward__0p5.json
scripts/diagnostics/boundary_model/B0/witness/W1__forward__1.json
scripts/diagnostics/boundary_model/B0/witness/W1__nonuniform__0p5.json
scripts/diagnostics/boundary_model/B0/witness/W1__nonuniform__0p5_nu_measured.json
scripts/diagnostics/boundary_model/B0/witness/W1__nonuniform__1.json
scripts/diagnostics/boundary_model/B0/witness/W1__nonuniform__1_nu_measured.json
scripts/diagnostics/boundary_model/B0/witness/W1__nonuniform__1_smoke_nu_measured.json
scripts/diagnostics/boundary_model/B0/witness/W1__run__0p5.json
scripts/diagnostics/boundary_model/B0/witness/W1__run__1.json
scripts/diagnostics/boundary_model/B0/witness/W1__sweep__0p5.json
scripts/diagnostics/boundary_model/B0/witness/W1__sweep__1.json
scripts/diagnostics/boundary_model/B0/witness/W1__sweep__1_smoke.json
scripts/diagnostics/boundary_model/B0/witness/W2_NOT_MEASURED.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__lumped__0p5__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__lumped__0p5__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__lumped__1__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__lumped__1__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__lumped__2__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__lumped__2__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__wire__0p5__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__wire__0p5__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__wire__1__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__wire__1__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__wire__2__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__forward__wire__2__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__lumped__0p5__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__lumped__0p5__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__lumped__1__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__lumped__1__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__lumped__2__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__lumped__2__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__wire__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__wire__0p5__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__wire__0p5__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__wire__1__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__wire__1__1.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__wire__2__0.json
scripts/diagnostics/boundary_model/B0/witness/W3__run__wire__2__1.json
scripts/diagnostics/boundary_model/B0/witness/W4__forward.json
scripts/diagnostics/boundary_model/B0/witness/W4__forward_smoke.json
scripts/diagnostics/boundary_model/B0/witness/W4__run-no-backing.json
scripts/diagnostics/boundary_model/B0/witness/W4__run.json
scripts/diagnostics/boundary_model/B0b/measurements.json
scripts/diagnostics/boundary_model/B0c/RESULTS.json
scripts/diagnostics/boundary_model/B0c/gpu_relaunch/RESULTS.json
scripts/diagnostics/boundary_model/B1/MATRIX.json
scripts/diagnostics/boundary_model/B1/battery/PROVENANCE.json
scripts/diagnostics/boundary_model/reviews/codex_review/cavity.json
scripts/diagnostics/boundary_model/reviews/codex_review/checks.json
scripts/diagnostics/boundary_model/reviews/codex_review/verification.json
scripts/diagnostics/gradient_dx_ladder/desk_feasibility.json
scripts/diagnostics/issue169_decay/issue169_decay_vs_flux.json
scripts/diagnostics/lumped_rlc_adjacent_to_port_nan.json
scripts/diagnostics/msl_ad_z0_anchor_probe.json
scripts/diagnostics/msl_beta_fit_synthetic_accuracy/msl_beta_fit_synthetic_accuracy.json
scripts/diagnostics/msl_vi_flux_oracle/msl_vi_flux_oracle.json
scripts/diagnostics/msl_z0_bias_floor_sweep/msl_z0_bias_floor_sweep.json
scripts/diagnostics/msl_z0_bias_floor_sweep/msl_z0_bias_floor_sweep_readability_witness_2026-09-20.json
scripts/diagnostics/msl_z0_bias_floor_sweep/msl_z0_bias_floor_sweep_realized_anchor.json
scripts/diagnostics/msl_z0_bias_floor_sweep/msl_z0_bias_floor_sweep_realized_anchor_2026-09-19.json
scripts/diagnostics/open_boundary_contract/cont/_cpu_smoke/result.json
scripts/diagnostics/open_boundary_contract/cont/n2_pad0_cpml4_a/result.json
scripts/diagnostics/open_boundary_contract/cont/n2_pad0_cpml4_a/summary.json
scripts/diagnostics/open_boundary_contract/cont/n2_pad10_cpml4_a/result.json
scripts/diagnostics/open_boundary_contract/cont/n2_pad10_cpml4_a/summary.json
scripts/diagnostics/open_boundary_contract/cont/n3_pad10_cpml6_a/result.json
scripts/diagnostics/open_boundary_contract/cont/n3_pad10_cpml6_a/summary.json
scripts/diagnostics/open_boundary_contract/cont/n4_pad10_cpml8_a/result.json
scripts/diagnostics/open_boundary_contract/cont/n4_pad10_cpml8_a/summary.json
scripts/diagnostics/open_boundary_contract/dumps/_cpu_smoke/meta.json
scripts/diagnostics/open_boundary_contract/dumps/n2_pad0_main/meta.json
scripts/diagnostics/open_boundary_contract/dumps/n2_pad0_main/summary.json
scripts/diagnostics/open_boundary_contract/dumps/n2_pad0_main_ceil/meta.json
scripts/diagnostics/open_boundary_contract/dumps/n2_pad0_main_ceil/summary.json
scripts/diagnostics/open_boundary_contract/dumps/n2_pad10_main/meta.json
scripts/diagnostics/open_boundary_contract/dumps/n2_pad10_main/summary.json
scripts/diagnostics/open_boundary_contract/dumps/n2_pad10_main1012/meta.json
scripts/diagnostics/open_boundary_contract/dumps/n3_pad10_main/meta.json
scripts/diagnostics/open_boundary_contract/dumps/n3_pad10_main/summary.json
scripts/diagnostics/open_boundary_contract/dumps/n3_pad10_main1012/meta.json
scripts/diagnostics/open_boundary_contract/dumps/n3_pad10_main1012/summary.json
scripts/diagnostics/open_boundary_contract/msl/verification.json
scripts/diagnostics/open_boundary_contract/option_b/COMPARATOR_COMPARISONS_3.json
scripts/diagnostics/open_boundary_contract/option_b/FIELD_SPATIAL_DETAIL_3.json
scripts/diagnostics/open_boundary_contract/option_b/FIELD_SUMMARY_3.json
scripts/diagnostics/open_boundary_contract/option_b/PROVENANCE.txt
scripts/diagnostics/open_boundary_contract/option_b/dry/msl/0_4/one_port/result.json
scripts/diagnostics/open_boundary_contract/option_b/dry/msl/0_4/result.json
scripts/diagnostics/open_boundary_contract/option_b/dry/msl/0_4/two_port/result.json
scripts/diagnostics/open_boundary_contract/option_b/dry/patch/1_6/result.json
scripts/diagnostics/open_boundary_contract/option_b/dry/plane/0_4/result.json
scripts/diagnostics/open_boundary_contract/option_b/dry/plane/0_4/small/result.json
scripts/diagnostics/open_boundary_contract/option_b/dry/waveguide/0.3_8/result.json
scripts/diagnostics/open_boundary_contract/option_b/dry/waveguide/1_8/result.json
scripts/diagnostics/open_boundary_contract/option_b/job_source_hashes.json
scripts/diagnostics/open_boundary_contract/option_b/job_source_hashes_3.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_2/msl/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_2/patch/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_2/plane/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_2/waveguide/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_3/frequency_1/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_3/frequency_2/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_3/frequency_4/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_3/msl/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_3/oracle/summary.json
scripts/diagnostics/open_boundary_contract/option_b/jobs_3/waveguide/summary.json
scripts/diagnostics/open_boundary_contract/option_b/log_replay_precision_3.json
scripts/diagnostics/open_boundary_contract/option_b/raw/msl/0_4/one_port/result.json
scripts/diagnostics/open_boundary_contract/option_b/raw/plane_reference/calibration.json
scripts/diagnostics/open_boundary_contract/option_b/raw_3/diagnostic_msl/result.json
scripts/diagnostics/open_boundary_contract/option_b/raw_3/diagnostic_waveguide/result.json
scripts/diagnostics/open_boundary_contract/option_b/reference_sizing_3.json
scripts/diagnostics/open_boundary_contract/option_b/repeat_comparison_3.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.01_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.01_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.01_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.03_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.03_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.03_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.1_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.1_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.1_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.3_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.3_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0.3_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/0_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/1_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/1_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/1_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/3_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/3_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl/3_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/msl_low/0.01_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.01_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.01_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.01_6.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.01_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.03_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.03_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.03_6.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.03_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.1_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.1_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.1_6.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.1_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.3_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.3_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.3_6.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0.3_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0_6.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/0_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/1_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/1_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/1_6.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/1_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/3_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/3_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/3_6.json
scripts/diagnostics/open_boundary_contract/option_b/results/patch/3_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.01_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.01_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.01_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.03_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.03_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.03_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.1_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.1_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.1_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.3_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.3_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0.3_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/0_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/1_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/1_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/1_8.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/3_16.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/3_4.json
scripts/diagnostics/open_boundary_contract/option_b/results/waveguide/3_8.json
scripts/diagnostics/open_boundary_contract/option_b/run_ids.json
scripts/diagnostics/open_boundary_contract/option_b/run_ids_2.json
scripts/diagnostics/open_boundary_contract/option_b/scale1_profile_verification_3.json
scripts/diagnostics/open_boundary_contract/option_b/static_verification_3.json
scripts/diagnostics/open_boundary_contract/option_b/variant_trace_replay_3.json
scripts/diagnostics/open_boundary_contract/option_b/verification.json
scripts/diagnostics/open_boundary_contract/option_b/verification_2.json
scripts/diagnostics/open_boundary_contract/option_b/verification_3.json
scripts/diagnostics/open_boundary_contract/ports/verification.json
scripts/diagnostics/open_boundary_contract/result_main.json
scripts/diagnostics/open_boundary_contract/result_main_pad0_ceil_sizing.json
scripts/diagnostics/open_boundary_contract/result_main_plus_1012.json
scripts/diagnostics/openems_thru_referee/results/thru_openems_dx500um.json
scripts/diagnostics/palace_notch_referee/notch_s11_4090.json
scripts/diagnostics/palace_notch_referee/notch_s11_mid.json
scripts/diagnostics/palace_notch_referee/notch_s11_probe_4090.json
scripts/diagnostics/palace_notch_referee/notch_s11_probe_mid.json
scripts/diagnostics/palace_sheen_referee/sheen_s21_4090.json
scripts/diagnostics/palace_sheen_referee/sheen_s21_mid.json
scripts/diagnostics/palace_sheen_referee/sheen_s21_probe_4090.json
scripts/diagnostics/palace_sheen_referee/sheen_s21_probe_mid.json
scripts/diagnostics/pec_sheet_edge_offset/pec_sheet_edge_offset.json
scripts/diagnostics/physics_gate_vessl_shards.json
scripts/diagnostics/port_external_reference_requirements.json
scripts/diagnostics/wr90_dz_dispatch_falsifier_prefix_baseline.json
scripts/diagnostics/wr90_dz_dispatch_falsifier_results.json
scripts/requirements-public-docs.txt
""".splitlines())


def _tracked_outputs(repo: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "scripts/"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    return {
        path for path in result.stdout.split("\0")
        if Path(path).suffix.lower() in OUTPUT_SUFFIXES
    }


def _check_outputs(repo: Path) -> None:
    actual = _tracked_outputs(repo)
    added = sorted(actual - ALLOWED_OUTPUTS)
    missing = sorted(ALLOWED_OUTPUTS - actual)
    assert not added and not missing, (
        f"scripts output closed list changed: added={added}, missing={missing}. "
        "Keep new run outputs out of scripts/; review survivor removals."
    )


pytestmark = pytest.mark.skipif(
    not git_available(REPO), reason="tracked outputs require a git checkout",
)


def test_scripts_output_files_are_a_closed_list() -> None:
    _check_outputs(REPO)


@pytest.mark.parametrize("path", [
    "scripts/__b10_mutant__.log",
    "scripts/diagnostics/__b10_mutant_result__.json",
])
def test_new_tracked_output_fails(path: str, tmp_path: Path, monkeypatch) -> None:
    """Mutate a private copy of the real index without changing the worktree."""
    index = subprocess.run(
        ["git", "rev-parse", "--git-path", "index"], cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    scratch_index = tmp_path / "index"
    shutil.copyfile(REPO / index, scratch_index)
    monkeypatch.setenv("GIT_INDEX_FILE", os.fspath(scratch_index))
    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"], cwd=REPO, input="mutation\n",
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    assert path in _tracked_outputs(REPO)
    with pytest.raises(AssertionError, match="added=.*__b10_mutant"):
        _check_outputs(REPO)

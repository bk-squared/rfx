"""Regression locks for Milestone 9 promotion artifacts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMOTION = ROOT / "docs/guides/sbp_sat_support_promotion_proposal.md"
CAVEATS = ROOT / "docs/guides/sbp_sat_release_migration_caveats.md"
VERIFIER = ROOT / "docs/guides/sbp_sat_final_verifier_report.md"
FULL_SPEC = ROOT / "docs/guides/sbp_sat_zslab_phase1_full_spec.md"
SUPPORT_MATRIX = ROOT / "docs/guides/support_matrix.md"
TRUE_RT_SPEC = ROOT / "docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md"
CHANGELOG = ROOT / "docs/public/guide/changelog.mdx"
MIGRATION = ROOT / "docs/public/guide/migration.md"


def _text(path: Path) -> str:
    return path.read_text()


def test_promotion_proposal_recommends_retaining_experimental_status():
    text = _text(PROMOTION)

    for token in (
        "Do not promote SBP-SAT / subgridding beyond `experimental`",
        "status: experimental",
        "boundary: all_pec_plus_selected_reflector_periodic_cpml_subset",
        "geometry: axis_aligned_arbitrary_box_with_cpml_guard_for_absorbing_faces",
        "claim_level: experimental_proxy_validated_only",
        "retain experimental status",
        "true R/T benchmark",
        "deferred",
        "Milestones 7-8 still remain **RFC/spec gates**",
        "arbitrary-box runtime, selected reflector/periodic subset, and bounded",
        "CPML absorbing subset now exist",
        "arbitrary-box lane with selected reflector/periodic and bounded CPML boundary subsets",
    ):
        assert token in text


def test_release_and_migration_caveats_lock_current_surface():
    text = _text(CAVEATS)

    for token in (
        "experimental only",
        "selected reflector/periodic and bounded CPML boundary subsets",
        "one axis-aligned refinement box only",
        "soft point source + point probe only",
        "proxy numerical-equivalence benchmark only",
        "boundary=\"pec\"",
        "selected reflector/periodic subset",
        "bounded CPML subset",
        "DFT planes",
        "flux monitors",
        "NTFF / Huygens-box far field",
        "material-scaled SAT",
        "sub-stepped or multi-rate",
    ):
        assert token in text


def test_final_verifier_report_ties_claims_to_evidence():
    text = _text(VERIFIER)

    for token in (
        "experimental, proxy-only, axis-aligned arbitrary-box",
        "selected reflector/periodic and bounded CPML boundary subsets",
        "Claim-to-evidence map",
        "tests/test_support_matrix_sbp_sat.py",
        "tests/test_public_subgridding_docs_contract.py",
        "tests/test_subgrid_crossval.py",
        "tests/test_sbp_sat_boundary_crossval.py",
        "tests/test_sbp_sat_absorbing_crossval.py",
        "tests/test_sbp_sat_api_guards.py",
        "Milestone 5-8 RFC docs + contract tests",
        "does **not** recommend promotion",
        "Anything broader would exceed the presently verified evidence",
    ):
        assert token in text


def test_full_spec_references_milestone9_artifacts_and_contract_test():
    text = _text(FULL_SPEC)

    assert "docs/guides/sbp_sat_support_promotion_proposal.md" in text
    assert "docs/guides/sbp_sat_release_migration_caveats.md" in text
    assert "docs/guides/sbp_sat_final_verifier_report.md" in text
    assert "tests/test_sbp_sat_promotion_artifacts_contract.py" in text


def test_internal_flux_dft_gate_does_not_promote_public_observables():
    support_text = _text(SUPPORT_MATRIX)
    spec_text = _text(TRUE_RT_SPEC)

    assert "private analytic-sheet flux/DFT R/T benchmark gate" in support_text
    assert "private analytic sheet/source" in spec_text
    for token in (
        "public true R/T",
        "public DFT-plane and flux-monitor APIs still hard-fail",
        "mixed periodic+CPML is rejected",
        "support matrix continues to mark true R/T as deferred",
    ):
        assert token in spec_text


def test_public_changelog_and_migration_keep_sbp_sat_narrow():
    assert "SBP-SAT subgridding remains experimental" in _text(CHANGELOG)
    assert "experimental arbitrary-box" in _text(MIGRATION)

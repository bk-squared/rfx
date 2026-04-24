"""Regression locks for Milestone 9 promotion artifacts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMOTION = ROOT / "docs/guides/sbp_sat_support_promotion_proposal.md"
CAVEATS = ROOT / "docs/guides/sbp_sat_release_migration_caveats.md"
VERIFIER = ROOT / "docs/guides/sbp_sat_final_verifier_report.md"
FULL_SPEC = ROOT / "docs/guides/sbp_sat_zslab_phase1_full_spec.md"
CHANGELOG = ROOT / "docs/public/guide/changelog.mdx"
MIGRATION = ROOT / "docs/public/guide/migration.md"


def _text(path: Path) -> str:
    return path.read_text()


def test_promotion_proposal_recommends_retaining_experimental_status():
    text = _text(PROMOTION)

    for token in (
        "Do not promote SBP-SAT / subgridding beyond `experimental`",
        "status: experimental",
        "boundary: all_pec_only",
        "geometry: full_span_xy_z_slab_only",
        "claim_level: experimental_proxy_validated_only",
        "retain experimental status",
        "true R/T benchmark",
        "deferred",
        "Milestones 5-8 produced **RFC/spec gates**, not runtime implementations",
    ):
        assert token in text


def test_release_and_migration_caveats_lock_current_surface():
    text = _text(CAVEATS)

    for token in (
        "experimental only",
        "all-PEC only",
        "one full-span x/y refined z slab only",
        "soft point source + point probe only",
        "proxy numerical-equivalence benchmark only",
        "boundary=\"pec\"",
        "all-PEC `BoundarySpec`",
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
        "experimental, proxy-only, all-PEC z-slab",
        "Claim-to-evidence map",
        "tests/test_support_matrix_sbp_sat.py",
        "tests/test_public_subgridding_docs_contract.py",
        "tests/test_subgrid_crossval.py",
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


def test_public_changelog_and_migration_keep_sbp_sat_narrow():
    assert "SBP-SAT subgridding remains experimental" in _text(CHANGELOG)
    assert "experimental all-PEC z-slab, proxy-only" in _text(MIGRATION)

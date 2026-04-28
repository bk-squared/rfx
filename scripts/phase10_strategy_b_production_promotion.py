"""Phase X Strategy B production-promotion decision harness.

This script is intentionally a thin policy layer over hardened Phase IX
artifacts.  It does not infer physical validity from generic execution success;
it consumes Phase IX readiness/physical artifacts, candidate-family stale status,
worktree promotability, and family-specific full-floor physical oracle fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase9_strategy_b_full_floor as phase9  # noqa: E402

SCHEMA_VERSION = 1
PROMOTION_CONTRACT = "phase_x_strategy_b_production_promotion"
SUMMARY_FILENAME = "phase10_strategy_b_production_promotion.json"
PROMOTION_ELIGIBLE_FAMILIES = (
    "source_probe",
    "cpml_topology",
    "pec_topology",
    "port_proxy",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": phase9.file_sha256(path)}


def refresh_phase9_summary(artifact_dir: Path) -> dict[str, Any]:
    summary_path = artifact_dir / phase9.SUMMARY_FILENAME
    return phase9.summarize_artifacts(artifact_dir=artifact_dir, output=summary_path)


def candidate_stale_entries(summary: dict[str, Any], family: str) -> dict[str, Any]:
    stale = summary.get("stale_context", {})
    return {
        key: value
        for key, value in stale.items()
        if key in {f"{family}:readiness", f"{family}:physical"}
    }


def unrelated_stale_entries(summary: dict[str, Any], family: str) -> dict[str, Any]:
    stale = summary.get("stale_context", {})
    return {
        key: value
        for key, value in stale.items()
        if key not in {f"{family}:readiness", f"{family}:physical"}
    }


def _row(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = artifact.get("rows") or []
    return rows[0] if rows else {}


def _execution(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("full_floor_execution", {})


def _artifact_paths(artifact_dir: Path, family: str) -> dict[str, Path]:
    return phase9.artifact_paths(artifact_dir, family)


def _missing_artifacts(paths: dict[str, Path]) -> list[str]:
    return [kind for kind, path in paths.items() if not path.exists()]


def _split_source_valid(row: dict[str, Any]) -> bool:
    gradient = row.get("required_gradient_evidence", {})
    return (
        gradient.get("required") is True
        and gradient.get("state") == "pass"
        and gradient.get("source_artifact_provenance_valid") is True
        and not gradient.get("source_artifact_provenance_errors")
    )


def _physical_oracle_valid(row: dict[str, Any]) -> bool:
    oracle = row.get("required_physical_oracle", {})
    return oracle.get("present") is True and oracle.get("passed") is True


def _topology_optimized_density_replay_valid(row: dict[str, Any]) -> bool:
    oracle = row.get("required_physical_oracle", {})
    return (
        oracle.get("present") is True
        and oracle.get("passed") is True
        and oracle.get("replay_mode") == phase9.TOPOLOGY_REPLAY_MODE
        and oracle.get("replay_density_source") == phase9.TOPOLOGY_REPLAY_DENSITY_SOURCE
        and oracle.get("replay_supported") is True
        and oracle.get("replay_finite") is True
        and oracle.get("replay_beta_matches_optimizer") is True
        and oracle.get("material_consistency_passed") is True
        and oracle.get("replay_source_is_required_physical_oracle") is True
    )


def _worktree_clean(artifact: dict[str, Any]) -> bool:
    return (
        artifact.get("promotion_worktree_status") == phase9.PROMOTION_WORKTREE_CLEAN
        and artifact.get("provenance", {}).get("worktree_signature", {}).get("dirty")
        is False
    )


def _fail_closed_pass(artifact: dict[str, Any]) -> bool:
    return artifact.get("fail_closed_evidence", {}).get("row_state") == "pass"


def _family_decision(
    family: str, artifact_dir: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    paths = _artifact_paths(artifact_dir, family)
    missing = _missing_artifacts(paths)
    warnings: list[str] = []
    unrelated_stale = unrelated_stale_entries(summary, family)
    if unrelated_stale:
        warnings.append("unrelated_family_stale_context_present")
    if missing:
        return {
            "family": family,
            "decision": "not_evaluated",
            "failed_gates": [f"missing_{kind}_artifact" for kind in missing],
            "reasons": ["required Phase IX candidate-family artifacts are missing"],
            "warnings": warnings,
            "unrelated_stale_context": unrelated_stale,
            "artifacts": {kind: str(path) for kind, path in paths.items()},
        }

    readiness = read_json(paths["readiness"])
    physical = read_json(paths["physical"])
    readiness_row = _row(readiness)
    physical_row = _row(physical)
    readiness_execution = _execution(readiness_row)
    readiness_executed = (
        readiness_execution.get("executed") is True
        and readiness_execution.get("status") == "completed"
    )
    physical_executed = physical_row.get("strategy_b_executed") is True
    promotion_eligible_family = family in PROMOTION_ELIGIBLE_FAMILIES

    failed: list[str] = []
    reasons: list[str] = []
    stale = candidate_stale_entries(summary, family)
    if stale:
        failed.append("candidate_family_stale_provenance")
        reasons.append("candidate-family stale_context entries are present")
    if family == "pec_topology" and not phase9._floor_for_family(family).get(
        "representative", True
    ):
        failed.append("representative_pec_topology_floor_missing")
        reasons.append("PEC topology has no approved representative Phase IX floor")
    for kind, artifact in (("readiness", readiness), ("physical", physical)):
        if not _worktree_clean(artifact):
            failed.append(f"{kind}_debug_only_dirty_worktree")
            reasons.append(
                f"{kind} artifact is not a clean-worktree promotion candidate"
            )
        if not _fail_closed_pass(artifact):
            failed.append(f"{kind}_fail_closed_not_pass")
            reasons.append(f"{kind} artifact fail-closed source is not passing")

    if not readiness_executed:
        failed.append("readiness_full_floor_not_executed")
        reasons.append(
            "readiness artifact does not contain completed executed full-floor evidence"
        )
    if readiness.get("summary", {}).get("family_status") != "production_ready_limited":
        failed.append("readiness_not_production_ready_limited")
        reasons.append("readiness artifact did not reach production_ready_limited")
    if (
        promotion_eligible_family
        and readiness_executed
        and not _split_source_valid(readiness_row)
    ):
        failed.append("split_source_evidence_not_valid")
        reasons.append(
            "split-source gradient evidence is missing, stale, or not passing"
        )

    if not physical_executed:
        failed.append("physical_strategy_b_not_executed")
        reasons.append(
            "physical artifact does not show Strategy B full-floor execution"
        )
    if physical.get("summary", {}).get("family_status") != "physics_validated_limited":
        failed.append("physical_not_validated_limited")
        reasons.append("physical artifact did not reach physics_validated_limited")
    if (
        promotion_eligible_family
        and readiness_executed
        and physical_executed
        and not _physical_oracle_valid(physical_row)
    ):
        failed.append("family_specific_physical_oracle_not_valid")
        reasons.append(
            "family-specific full-floor physical oracle is missing or not passing"
        )
    if (
        family in phase9.TOPOLOGY_FAMILIES
        and readiness_executed
        and physical_executed
        and not _topology_optimized_density_replay_valid(physical_row)
    ):
        failed.append("topology_optimized_density_replay_not_valid")
        reasons.append(
            "topology physical oracle is not sourced from valid optimized-density replay"
        )

    if not failed:
        decision = "promoted_limited"
        reasons = ["candidate family passed all Phase X production-promotion gates"]
    elif any(
        gate in failed
        for gate in (
            "candidate_family_stale_provenance",
            "split_source_evidence_not_valid",
            "family_specific_physical_oracle_not_valid",
            "topology_optimized_density_replay_not_valid",
        )
    ) or any("dirty_worktree" in gate or "fail_closed" in gate for gate in failed):
        decision = "blocked"
    elif family == "pec_topology" or any(
        "not_executed" in gate or "not_" in gate for gate in failed
    ):
        decision = "experimental_limited"
    else:
        decision = "blocked"

    return {
        "family": family,
        "decision": decision,
        "failed_gates": failed,
        "reasons": reasons,
        "warnings": warnings,
        "candidate_stale_context": stale,
        "unrelated_stale_context": unrelated_stale,
        "artifacts": {
            "readiness": artifact_ref(paths["readiness"]),
            "physical": artifact_ref(paths["physical"]),
        },
        "statuses": {
            "readiness": readiness.get("summary", {}).get("family_status"),
            "physical": physical.get("summary", {}).get("family_status"),
            "readiness_execution": readiness.get("summary", {}).get("execution_status"),
            "physical_execution": physical.get("summary", {}).get("execution_status"),
            "readiness_worktree": readiness.get("promotion_worktree_status"),
            "physical_worktree": physical.get("promotion_worktree_status"),
        },
    }


def build_promotion_decision(
    *,
    artifact_dir: Path,
    families: tuple[str, ...] = phase9.FAMILIES,
    command: list[str] | None = None,
) -> dict[str, Any]:
    summary_path = artifact_dir / phase9.SUMMARY_FILENAME
    summary = refresh_phase9_summary(artifact_dir)
    decisions = {
        family: _family_decision(family, artifact_dir, summary) for family in families
    }
    selected_eligible = [
        family for family in PROMOTION_ELIGIBLE_FAMILIES if family in decisions
    ]
    promoted = sorted(
        family
        for family, decision in decisions.items()
        if decision["decision"] == "promoted_limited"
    )
    blocked = sorted(
        family
        for family, decision in decisions.items()
        if decision["decision"] == "blocked"
    )
    experimental = sorted(
        family
        for family, decision in decisions.items()
        if decision["decision"] == "experimental_limited"
    )
    not_evaluated = sorted(
        family
        for family, decision in decisions.items()
        if decision["decision"] == "not_evaluated"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "promotion_contract": PROMOTION_CONTRACT,
        "generated_at": phase9.utc_now(),
        "artifact_dir": str(artifact_dir),
        "command": command or sys.argv,
        "phase9_summary": artifact_ref(summary_path) if summary_path.exists() else None,
        "phase9_stale_context": summary.get("stale_context", {}),
        "decisions": decisions,
        "summary": {
            "promoted_families": promoted,
            "blocked_families": blocked,
            "experimental_limited_families": experimental,
            "not_evaluated_families": not_evaluated,
            "eligible_families": list(PROMOTION_ELIGIBLE_FAMILIES),
            "selected_eligible_families": selected_eligible,
            "all_eligible_promoted": bool(selected_eligible)
            and all(
                decisions.get(family, {}).get("decision") == "promoted_limited"
                for family in selected_eligible
            ),
            "has_hygiene_warnings": any(
                decision.get("warnings") for decision in decisions.values()
            ),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path(".omx/artifacts"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--family", choices=("all", *phase9.FAMILIES), default="all")
    parser.add_argument("--require-promotion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    families = phase9.FAMILIES if args.family == "all" else (args.family,)
    payload = build_promotion_decision(
        artifact_dir=args.artifact_dir,
        families=families,
        command=sys.argv,
    )
    output = args.output or args.artifact_dir / SUMMARY_FILENAME
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_promotion:
        required = [
            family for family in PROMOTION_ELIGIBLE_FAMILIES if family in families
        ]
        failed = [
            family
            for family in required
            if payload["decisions"].get(family, {}).get("decision")
            != "promoted_limited"
        ]
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

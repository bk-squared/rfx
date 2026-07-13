#!/usr/bin/env python3
"""Audit the active "all RF port / S-parameter infra reaches E5" goal.

This is a completion-audit gate, not a physics solver.  It converts the broad
goal into a prompt-to-artifact checklist, reads the current S-parameter support
matrix, verifies referenced local evidence artifacts where possible, and marks
the goal complete only if every public port family has an unblocked E5-level
claims-bearing envelope.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from check_port_external_references import build_external_reference_audit


REPO_ROOT = Path(__file__).resolve().parents[2]

OBJECTIVE = "모든 port 및 infra E5 수준까지 개선"
DEFAULT_EXTERNAL_REFERENCE_MANIFEST = (
    "scripts/diagnostics/port_external_reference_requirements.json"
)

SUCCESS_CRITERIA = (
    {
        "id": "all_current_and_planned_port_families_e5",
        "requirement": (
            "Every current public impedance-defined port / S-parameter family "
            "and every planned generalized RF port surface tracked for this "
            "goal has an unblocked claims-bearing E5 envelope, or is explicitly "
            "removed from the goal scope."
        ),
    },
    {
        "id": "e2_e3_e4_e5_artifacts_present",
        "requirement": (
            "Each promoted E5 envelope cites analytic/oracle evidence where "
            "available, independent raw dump replay, external cross-solver or "
            "equivalent evidence, and mesh/frequency/geometry envelope artifacts."
        ),
    },
    {
        "id": "unsupported_paths_not_silent",
        "requirement": (
            "Unsupported, shadow, or non-port S-parameter paths are explicitly "
            "documented and cannot be mistaken for claims-bearing E5 outputs."
        ),
    },
    {
        "id": "docs_and_machine_matrix_match",
        "requirement": (
            "Machine-readable support matrix and public docs agree with the "
            "actual evidence level and blocked claims."
        ),
    },
)

E5_BLOCKING_TOKENS = (
    "blocked",
    "partial",
    "limited",
    "experimental",
    "shadow",
    "only",
)


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _is_pathish(value: str) -> bool:
    return (
        value.startswith((".", "/", "docs/", "tests/", "scripts/", "rfx/", "examples/"))
        or "/" in value
    )


def _artifact_status(value: str) -> dict[str, Any] | None:
    if not _is_pathish(value):
        return None
    # Keep pytest node ids path-checkable at the file portion.
    file_part = value.split("::", 1)[0].split(" ", 1)[0]
    if not file_part or file_part.endswith(")"):
        return None
    path = _repo_path(file_part)
    status: dict[str, Any] = {
        "artifact": value,
        "path_checked": _rel(path),
        "exists": path.exists(),
    }
    if path.exists() and path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            status.update(
                {
                    "json_parse_status": "invalid",
                    "json_error": str(exc),
                    "reported_status": "",
                    "reported_evidence_level": "",
                }
            )
        else:
            reported_status = payload.get("status")
            reported_evidence = payload.get("evidence_level", payload.get("claim_level", ""))
            status.update(
                {
                    "json_parse_status": "parsed",
                    "reported_status": "" if reported_status is None else str(reported_status),
                    "reported_evidence_level": str(reported_evidence),
                }
            )
    return status


def _evidence_is_unblocked_e5(level: str) -> bool:
    normalized = level.lower()
    if not normalized.startswith("e5"):
        return False
    return not any(token in normalized for token in E5_BLOCKING_TOKENS)


def _evidence_is_narrow_e5(level: str) -> bool:
    normalized = level.lower()
    return normalized.split("/", 1)[0].startswith("e5-narrow")


def _family_result(entry: dict[str, Any]) -> dict[str, Any]:
    family = str(entry.get("family", ""))
    is_port = bool(entry.get("is_port", False))
    evidence_level = str(entry.get("evidence_level", ""))
    artifacts = [
        status
        for artifact in entry.get("evidence_artifacts", [])
        for status in [_artifact_status(str(artifact))]
        if status is not None
    ]
    missing_artifacts = [item for item in artifacts if not item["exists"]]
    blockers: list[str] = []

    if not is_port:
        status = "not_applicable"
        if evidence_level != "not_a_port":
            blockers.append("non-port family should be marked not_a_port")
    elif _evidence_is_unblocked_e5(evidence_level):
        # For this broad goal, E5-narrow is useful evidence but still does not
        # prove every branch/mode/nonuniform variant.  Keep it partial unless
        # the level is plain E5 or the docs explicitly say no broader blockers.
        if _evidence_is_narrow_e5(evidence_level):
            status = "partial"
            blockers.append("E5-narrow is not broad all-family E5")
        else:
            status = "passed"
    else:
        status = "blocked"
        blockers.append(f"evidence level is {evidence_level!r}, not unblocked E5")

    known_limits = [str(x) for x in entry.get("known_limits", [])]
    caveats = [str(x) for x in entry.get("caveats", [])]
    validation_gaps = [
        str(x)
        for x in entry.get(
            "validation_gaps", entry.get("promotion_requirements", [])
        )
    ]
    for text in known_limits + caveats:
        lowered = text.lower()
        if any(token in lowered for token in ("blocked", "requires", "needs", "not promoted", "no broad", "caveat")):
            if text not in blockers:
                blockers.append(text)

    for text in validation_gaps:
        if text not in blockers:
            blockers.append(text)

    # A broad-E5 result can be complete for its documented API while still
    # being incomplete for this auditor's broader all-port-family objective.
    # Do not silently pass that objective when the matrix names an outstanding
    # requirement (for example, a separate arbitrary-launch or multi-port API).
    if validation_gaps and status == "passed":
        status = "partial"

    if missing_artifacts and is_port:
        blockers.append(
            "referenced local evidence artifacts are missing: "
            + ", ".join(item["path_checked"] for item in missing_artifacts)
        )
        if status == "passed":
            status = "partial"

    invalid_json_artifacts = [
        item for item in artifacts if item.get("json_parse_status") == "invalid"
    ]
    explicitly_failed_json_artifacts = [
        item
        for item in artifacts
        if item.get("json_parse_status") == "parsed"
        and item.get("reported_status")
        and item.get("reported_status") != "passed"
    ]
    if invalid_json_artifacts and is_port:
        blockers.append(
            "referenced JSON evidence artifacts are invalid: "
            + ", ".join(item["path_checked"] for item in invalid_json_artifacts)
        )
        if status == "passed":
            status = "partial"
    if explicitly_failed_json_artifacts and is_port:
        blockers.append(
            "referenced JSON evidence artifacts did not report status=passed: "
            + ", ".join(
                f"{item['path_checked']}={item.get('reported_status')!r}"
                for item in explicitly_failed_json_artifacts
            )
        )
        if status == "passed":
            status = "partial"

    unstamped_json_artifacts = [
        item
        for item in artifacts
        if item.get("json_parse_status") == "parsed"
        and "reported_status" in item
        and not item.get("reported_status")
    ]
    if status == "passed" and unstamped_json_artifacts and is_port:
        blockers.append(
            "promoted E5 family cites JSON artifacts without a reported status: "
            + ", ".join(item["path_checked"] for item in unstamped_json_artifacts)
        )
        status = "partial"

    return {
        "family": family,
        "primitive": entry.get("primitive"),
        "is_port": is_port,
        "evidence_level": evidence_level,
        "status": status,
        "validation_status": entry.get("validation_status"),
        "validated_scope": entry.get(
            "validated_scope", entry.get("claim_envelope")
        ),
        "artifact_checks": artifacts,
        "blockers": blockers,
    }


def _future_family_result(entry: dict[str, Any]) -> dict[str, Any]:
    artifacts = [
        status
        for artifact in entry.get("evidence_artifacts", [])
        for status in [_artifact_status(str(artifact))]
        if status is not None
    ]
    missing_artifacts = [item for item in artifacts if not item["exists"]]
    validation_gaps = entry.get(
        "validation_gaps", entry.get("required_evidence", [])
    )
    blockers = [
        "port family has no implemented public S-parameter API",
        *[str(x) for x in validation_gaps],
    ]
    if missing_artifacts:
        blockers.append(
            "referenced local planning/evidence artifacts are missing: "
            + ", ".join(item["path_checked"] for item in missing_artifacts)
        )
    return {
        "family": entry.get("family"),
        "status": "blocked",
        "evidence_level": entry.get("evidence_level", entry.get("status", "not_implemented")),
        "blockers": blockers,
        "validation_gaps": validation_gaps,
        "artifact_checks": artifacts,
        "public_sparameter_api": entry.get("public_sparameter_api"),
    }


def _prompt_to_artifact_checklist(
    *,
    matrix_path: Path,
    external_manifest_path: Path,
    external_audit: dict[str, Any],
    family_results: list[dict[str, Any]],
    future_results: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matrix_check = _artifact_status(_rel(matrix_path))
    support_doc = _artifact_status("docs/guides/sparameter_support_matrix.md")
    evidence_rule = _artifact_status("docs/guides/physics_validation_evidence_rule.md")
    external_manifest = _artifact_status(_rel(external_manifest_path))
    current_blocked = [
        item["family"]
        for item in family_results
        if item["status"] not in ("passed", "not_applicable")
    ]
    future_blocked = [item["family"] for item in future_results]
    return [
        {
            "requirement_id": "all_current_and_planned_port_families_e5",
            "evidence": [matrix_check, support_doc, external_manifest],
            "covered_family_count": len(family_results) + len(future_results),
            "blocked_families": current_blocked + future_blocked,
            "status": "passed" if not incomplete else "blocked",
        },
        {
            "requirement_id": "e2_e3_e4_e5_artifacts_present",
            "evidence": [matrix_check, evidence_rule, external_manifest],
            "missing_or_blocked_count": len(incomplete),
            "status": "passed" if not incomplete else "blocked",
        },
        {
            "requirement_id": "broad_e5_external_reference_manifest",
            "evidence": [external_manifest],
            "external_reference_status": external_audit["status"],
            "external_reference_schema_status": external_audit["schema_status"],
            "incomplete_count": external_audit["incomplete_count"],
            "missing_passed_comparison_artifact_families": external_audit[
                "missing_passed_comparison_artifact_families"
            ],
            "missing_broad_e4_comparison_artifact_families": external_audit[
                "missing_broad_e4_comparison_artifact_families"
            ],
            "missing_broad_e5_envelope_artifact_families": external_audit[
                "missing_broad_e5_envelope_artifact_families"
            ],
            "status": "passed" if external_audit["status"] == "passed" else "blocked",
        },
        {
            "requirement_id": "unsupported_paths_not_silent",
            "evidence": [support_doc, _artifact_status("tests/test_sparameter_support_contract.py")],
            "status": "passed",
        },
        {
            "requirement_id": "docs_and_machine_matrix_match",
            "evidence": [matrix_check, support_doc, _artifact_status("scripts/diagnostics/audit_sparameter_claims.py")],
            "status": "passed" if matrix_check and matrix_check["exists"] else "blocked",
        },
    ]


def build_goal_audit(
    matrix_path: Path,
    external_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if external_manifest_path is None:
        external_manifest_path = _repo_path(DEFAULT_EXTERNAL_REFERENCE_MANIFEST)
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    external_audit = build_external_reference_audit(
        external_manifest_path,
        matrix_path,
    )
    family_results = [_family_result(entry) for entry in data.get("port_families", [])]
    unavailable_results = [
        _future_family_result(entry)
        for entry in data.get(
            "unavailable_port_families", data.get("future_port_families", [])
        )
    ]
    all_results = family_results + unavailable_results
    incomplete = [
        item
        for item in all_results
        if item["status"] not in ("passed", "not_applicable")
    ]
    port_families = [item for item in family_results if item["is_port"]]
    passed_ports = [item for item in port_families if item["status"] == "passed"]
    external_reference_status = external_audit["status"]
    external_reference_schema_status = external_audit["schema_status"]
    external_reference_blocked = (
        external_reference_status != "passed"
        or external_reference_schema_status != "passed"
    )
    prompt_checklist = _prompt_to_artifact_checklist(
        matrix_path=matrix_path,
        external_manifest_path=external_manifest_path,
        external_audit=external_audit,
        family_results=family_results,
        future_results=unavailable_results,
        incomplete=incomplete,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": OBJECTIVE,
        "matrix_path": _rel(matrix_path),
        "external_manifest": _rel(external_manifest_path),
        "external_reference_audit": external_audit,
        "status": (
            "complete" if not incomplete and not external_reference_blocked else "blocked"
        ),
        "success_criteria": SUCCESS_CRITERIA,
        "prompt_to_artifact_checklist": prompt_checklist,
        "summary": {
            "port_family_count": len(port_families),
            "passed_port_family_count": len(passed_ports),
            "future_family_count": len(unavailable_results),
            "incomplete_count": len(incomplete),
            "external_reference_status": external_reference_status,
            "external_reference_schema_status": external_reference_schema_status,
            "external_reference_incomplete_count": external_audit["incomplete_count"],
            "not_applicable_count": len([x for x in family_results if x["status"] == "not_applicable"]),
        },
        "family_results": family_results,
        "future_family_results": unavailable_results,
        "incomplete": incomplete,
        "completion_decision": (
            "Do not call update_goal: at least one port family, planned RF "
            "S-parameter surface, or external-reference manifest gate remains "
            "below unblocked broad E5."
            if incomplete or external_reference_blocked
            else "Objective evidence is complete; update_goal may be called."
        ),
    }


def write_goal_audit(audit: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rf_infra_e5_goal_audit.json"
    md_path = output_dir / "rf_infra_e5_goal_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lines = [
        "# RF infra E5 goal completion audit",
        "",
        f"- generated_at: `{audit['generated_at']}`",
        f"- objective: {audit['objective']}",
        f"- status: `{audit['status']}`",
        f"- matrix_path: `{audit['matrix_path']}`",
        f"- external_manifest: `{audit['external_manifest']}`",
        f"- external_reference_status: `{audit['summary']['external_reference_status']}`",
        f"- external_reference_incomplete_count: `{audit['summary']['external_reference_incomplete_count']}`",
        f"- completion_decision: {audit['completion_decision']}",
        "",
        "## Success criteria",
        "",
    ]
    for criterion in audit["success_criteria"]:
        lines.append(f"- `{criterion['id']}`: {criterion['requirement']}")
    lines.extend(["", "## Prompt-to-artifact checklist", ""])
    for item in audit["prompt_to_artifact_checklist"]:
        evidence = [
            f"`{entry['artifact']}`={'OK' if entry['exists'] else 'MISSING'}"
            for entry in item.get("evidence", [])
            if entry
        ]
        lines.append(
            f"- `{item['requirement_id']}`: `{item['status']}`; "
            + "; ".join(evidence)
        )
    lines.extend(
        [
            "",
            "## Family checklist",
            "",
            "| Family | Evidence | Status | Main blockers |",
            "|---|---:|---:|---|",
        ]
    )
    for item in audit["family_results"] + audit["future_family_results"]:
        blockers = "; ".join(item.get("blockers", [])[:3])
        if len(item.get("blockers", [])) > 3:
            blockers += "; ..."
        lines.append(
            "| "
            f"`{item.get('family')}` | `{item.get('evidence_level')}` | "
            f"`{item.get('status')}` | {blockers or '—'} |"
        )
    lines.extend(["", "## Incomplete requirements", ""])
    if audit["incomplete"]:
        for item in audit["incomplete"]:
            lines.append(
                f"- `{item.get('family')}` (`{item.get('evidence_level')}`): "
                + "; ".join(item.get("blockers", [])[:5])
            )
    else:
        lines.append("- none")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_rel(json_path)}")
    print(f"wrote {_rel(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="docs/guides/sparameter_support_matrix.json")
    parser.add_argument(
        "--external-manifest",
        default=DEFAULT_EXTERNAL_REFERENCE_MANIFEST,
        help="Broad-E5 external-reference manifest that must pass before completion.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return nonzero when the goal remains blocked",
    )
    args = parser.parse_args(argv)

    audit = build_goal_audit(
        _repo_path(args.matrix),
        _repo_path(args.external_manifest),
    )
    write_goal_audit(audit, _repo_path(args.output_dir))
    print(f"status={audit['status']}")
    if args.require_complete and audit["status"] != "complete":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

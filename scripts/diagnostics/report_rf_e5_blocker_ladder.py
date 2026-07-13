#!/usr/bin/env python3
"""Report a machine-readable blocker ladder for the all-port RF E5 goal.

This diagnostic decomposes the remaining broad-E5 blockers into ordered stages
(API, E3 replay, E4 external reference, broad-E4 comparison, E5 envelope, and
solver dependency).  It is a planning/audit artifact only: it does not run a
solver, does not create physics evidence, and must not be used to promote any
family to E5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from audit_rf_infra_e5_goal import DEFAULT_EXTERNAL_REFERENCE_MANIFEST
from check_external_solver_dependencies import build_dependency_audit
from check_port_external_references import build_external_reference_audit


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPPORT_MATRIX = "docs/guides/sparameter_support_matrix.json"

STAGE_ORDER = {
    "api_surface": 10,
    "e2_analytic_oracle": 20,
    "e3_replay_or_calibration": 30,
    "dependency": 35,
    "e4_external_reference": 40,
    "broad_e4_comparison": 50,
    "e5_envelope": 60,
    "scope_decision": 70,
}

FAMILY_DEPENDENCIES = {
    "lumped_port": ["openems_crossval"],
    "wire_port": ["openems_crossval"],
    "microstrip_line_port": ["openems_crossval"],
    "rectangular_waveguide_port": ["palace_crossval", "openems_crossval"],
    "coaxial_port": ["openems_crossval", "palace_crossval"],
    "floquet_port": ["rcwa_floquet"],
    "generalized_planar_ports": ["openems_crossval", "palace_crossval"],
}

LOCAL_NEXT_HINTS = {
    "lumped_port": (
        "Build a matched/open/short/load openEMS fixture family and aggregate it "
        "through compare_sparameter_reference.py before any broad-E5 envelope."
    ),
    "wire_port": (
        "Define the absolute wire-port calibration convention, then add an "
        "external fixture beyond patch resonance."
    ),
    "microstrip_line_port": (
        "Land the MSL eigenmode source/extractor or explicitly keep broad MSL "
        "out of scope; do not promote the current laplace lane beyond E5-narrow."
    ),
    "rectangular_waveguide_port": (
        "Add a branch/T-junction or multimode external reference geometry, then "
        "state the uniform/nonuniform/mode envelope."
    ),
    "coaxial_port": (
        "Promote the M67 distributed transverse E/M plane-source prototype into "
        "a real coaxial port/feed contract, then run calibrated matched/open/short/load "
        "coax external fixtures."
    ),
    "floquet_port": (
        "Make RCWA/S4 available or generate an equivalent external periodic-cell "
        "reference, then compare a non-empty FSS/RIS cell."
    ),
    "generalized_planar_ports": (
        "Choose the first generalized planar family and implement a minimal API "
        "plus dump/replay before external references."
    ),
}


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_missing_evidence(text: str) -> list[str]:
    lowered = text.lower()
    stages: list[str] = []
    if "implemented" in lowered or "api" in lowered or "promoted" in lowered:
        stages.append("api_surface")
    if "analytic" in lowered or "oracle" in lowered:
        stages.append("e2_analytic_oracle")
    if "raw" in lowered or "dump" in lowered or "v/i" in lowered or "calibrated tem" in lowered:
        stages.append("e3_replay_or_calibration")
    if "external" in lowered or "rcwa" in lowered or "full-wave" in lowered or "reference geometry" in lowered:
        stages.append("e4_external_reference")
    if "broad" in lowered or "absolute" in lowered or "branch" in lowered or "multi-mode" in lowered or "matched/open/short/load" in lowered:
        stages.append("broad_e4_comparison")
    if "mesh" in lowered or "frequency" in lowered or "geometry envelope" in lowered or "scan-angle" in lowered or "polarization" in lowered or "substrate-cell" in lowered or "nonuniform" in lowered:
        stages.append("e5_envelope")
    return stages or ["scope_decision"]


def _dependency_rows(family: str, dependency_audit: dict[str, Any]) -> list[dict[str, Any]]:
    capabilities = dependency_audit.get("capabilities", {})
    rows = []
    for capability in FAMILY_DEPENDENCIES.get(family, []):
        item = capabilities.get(capability, {})
        rows.append(
            {
                "stage": "dependency",
                "capability": capability,
                "status": item.get("status", "unknown"),
                "blockers": item.get("blockers", []),
                "order": STAGE_ORDER["dependency"],
            }
        )
    return rows


def _support_lookup(support_matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in support_matrix.get("port_families", []):
        family = str(row.get("family", ""))
        if family:
            rows[family] = row
    for row in support_matrix.get(
        "unavailable_port_families",
        support_matrix.get("future_port_families", []),
    ):
        family = str(row.get("family", ""))
        if family:
            rows[family] = row
    return rows


def _manifest_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("family", "")): row
        for row in manifest.get("requirements", [])
        if row.get("family")
    }


def _family_ladder(
    *,
    family: str,
    manifest_row: dict[str, Any],
    support_row: dict[str, Any] | None,
    external_row: dict[str, Any] | None,
    dependency_audit: dict[str, Any],
) -> dict[str, Any]:
    stage_rows: list[dict[str, Any]] = []
    for missing in manifest_row.get("missing_evidence", []):
        for stage in _classify_missing_evidence(str(missing)):
            stage_rows.append(
                {
                    "stage": stage,
                    "source": "missing_evidence",
                    "item": str(missing),
                    "status": "missing",
                    "order": STAGE_ORDER[stage],
                }
            )

    comparison_count = len(manifest_row.get("external_comparison_artifacts", []))
    envelope_count = len(manifest_row.get("broad_e5_envelope_artifacts", []))
    if comparison_count == 0:
        stage_rows.append(
            {
                "stage": "e4_external_reference",
                "source": "external_comparison_artifacts",
                "item": "no listed external comparison artifacts",
                "status": "missing",
                "order": STAGE_ORDER["e4_external_reference"],
            }
        )
    if envelope_count == 0:
        stage_rows.append(
            {
                "stage": "e5_envelope",
                "source": "broad_e5_envelope_artifacts",
                "item": "no listed broad E5 envelope artifacts",
                "status": "missing",
                "order": STAGE_ORDER["e5_envelope"],
            }
        )

    stage_rows.extend(_dependency_rows(family, dependency_audit))
    stage_rows.sort(key=lambda row: (int(row["order"]), str(row.get("item", ""))))
    first_missing = next(
        (
            row
            for row in stage_rows
            if row.get("status") in {"missing", "blocked", "unknown"}
        ),
        None,
    )
    blocked_deps = [
        row for row in stage_rows if row.get("stage") == "dependency" and row.get("status") != "available"
    ]
    stage_counts = Counter(str(row["stage"]) for row in stage_rows)
    return {
        "family": family,
        "primitive": manifest_row.get("primitive") or (support_row or {}).get("primitive"),
        "current_status": manifest_row.get("current_status"),
        "support_evidence_level": (support_row or {}).get("evidence_level", (support_row or {}).get("status")),
        "external_audit_status": None if external_row is None else external_row.get("status"),
        "external_comparison_artifact_count": comparison_count,
        "broad_e5_envelope_artifact_count": envelope_count,
        "stage_counts": dict(sorted(stage_counts.items())),
        "blocked_dependency_count": len(blocked_deps),
        "first_blocking_stage": None if first_missing is None else first_missing.get("stage"),
        "first_blocking_item": None if first_missing is None else first_missing.get("item", first_missing.get("capability")),
        "local_next_hint": LOCAL_NEXT_HINTS.get(family, "Inspect family blockers and add the next missing evidence artifact."),
        "stages": stage_rows,
    }


def build_blocker_ladder(
    *,
    manifest_path: Path,
    support_matrix_path: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    support_matrix = _read_json(support_matrix_path)
    support_rows = _support_lookup(support_matrix)
    manifest_rows = _manifest_lookup(manifest)
    external_audit = build_external_reference_audit(manifest_path, support_matrix_path)
    external_rows = {row["family"]: row for row in external_audit.get("requirements", [])}
    dependency_audit = build_dependency_audit()

    required_families = sorted(
        family for family, row in manifest_rows.items() if row.get("required_for_e5")
    )
    family_ladders = [
        _family_ladder(
            family=family,
            manifest_row=manifest_rows[family],
            support_row=support_rows.get(family),
            external_row=external_rows.get(family),
            dependency_audit=dependency_audit,
        )
        for family in required_families
    ]
    stage_counts = Counter(
        stage
        for family in family_ladders
        for stage, count in family["stage_counts"].items()
        for _ in range(count)
    )
    no_comparison = [
        row["family"] for row in family_ladders if row["external_comparison_artifact_count"] == 0
    ]
    no_envelope = [
        row["family"] for row in family_ladders if row["broad_e5_envelope_artifact_count"] == 0
    ]
    blocked_dependencies = [
        {
            "family": row["family"],
            "dependencies": [
                stage
                for stage in row["stages"]
                if stage.get("stage") == "dependency" and stage.get("status") != "available"
            ],
        }
        for row in family_ladders
        if row["blocked_dependency_count"]
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if external_audit.get("status") != "passed" else "passed",
        "evidence_level": "planning-only",
        "claim_scope": (
            "ordered blocker ladder for the all-port broad-E5 campaign; not "
            "physics evidence, not external validation, and not an E5 promotion"
        ),
        "manifest": _display(manifest_path),
        "support_matrix": _display(support_matrix_path),
        "external_reference_status": external_audit.get("status"),
        "external_reference_schema_status": external_audit.get("schema_status"),
        "dependency_status": dependency_audit.get("status"),
        "required_family_count": len(family_ladders),
        "families_without_external_comparison_artifacts": no_comparison,
        "families_without_broad_e5_envelope_artifacts": no_envelope,
        "blocked_dependency_families": blocked_dependencies,
        "stage_counts": dict(sorted(stage_counts.items())),
        "family_ladders": family_ladders,
        "ranked_next_actions": [
            {
                "family": row["family"],
                "first_blocking_stage": row["first_blocking_stage"],
                "first_blocking_item": row["first_blocking_item"],
                "hint": row["local_next_hint"],
            }
            for row in family_ladders
        ],
        "completion_decision": (
            "Do not call update_goal from this ladder. It is planning-only and "
            "the external-reference audit remains blocked."
            if external_audit.get("status") != "passed"
            else "Ladder has no external-reference blockers; run the full RF-infra E5 audit before update_goal."
        ),
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rf_e5_blocker_ladder.json"
    md_path = output_dir / "rf_e5_blocker_ladder.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RF infra broad-E5 blocker ladder",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- external_reference_status: `{payload['external_reference_status']}`",
        f"- dependency_status: `{payload['dependency_status']}`",
        f"- required_family_count: `{payload['required_family_count']}`",
        "",
        payload["claim_scope"],
        "",
        "## Families without artifact classes",
        "",
        "- without external comparison artifacts: "
        + ", ".join(f"`{family}`" for family in payload["families_without_external_comparison_artifacts"]),
        "- without broad E5 envelope artifacts: "
        + ", ".join(f"`{family}`" for family in payload["families_without_broad_e5_envelope_artifacts"]),
        "",
        "## Ranked next actions",
        "",
        "| Family | First blocking stage | First blocking item | Hint |",
        "|---|---|---|---|",
    ]
    for row in payload["ranked_next_actions"]:
        item = str(row["first_blocking_item"] or "—").replace("|", "\\|")
        hint = str(row["hint"]).replace("|", "\\|")
        lines.append(
            f"| `{row['family']}` | `{row['first_blocking_stage']}` | {item} | {hint} |"
        )
    lines.extend(["", "## Family ladders", ""])
    for row in payload["family_ladders"]:
        lines.append(f"### `{row['family']}`")
        lines.append("")
        lines.append(f"- current_status: `{row['current_status']}`")
        lines.append(f"- support_evidence_level: `{row['support_evidence_level']}`")
        lines.append(f"- external_audit_status: `{row['external_audit_status']}`")
        for stage in row["stages"]:
            label = stage.get("item", stage.get("capability", ""))
            lines.append(f"  - `{stage['stage']}` `{stage['status']}`: {label}")
        lines.append("")
    lines.append(f"Decision: {payload['completion_decision']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_display(json_path)}")
    print(f"wrote {_display(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_EXTERNAL_REFERENCE_MANIFEST)
    parser.add_argument("--support-matrix", default=DEFAULT_SUPPORT_MATRIX)
    parser.add_argument("--output-dir", default=".omx/physics-gate/latest-rf-e5-blocker-ladder")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    payload = build_blocker_ladder(
        manifest_path=_repo_path(args.manifest),
        support_matrix_path=_repo_path(args.support_matrix),
    )
    _write_report(payload, _repo_path(args.output_dir))
    print(
        "status={status} required_family_count={required_family_count} no_external={no_external} no_envelope={no_envelope}".format(
            status=payload["status"],
            required_family_count=payload["required_family_count"],
            no_external=len(payload["families_without_external_comparison_artifacts"]),
            no_envelope=len(payload["families_without_broad_e5_envelope_artifacts"]),
        )
    )
    if args.require_complete and payload["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

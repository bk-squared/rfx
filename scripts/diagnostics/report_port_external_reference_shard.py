#!/usr/bin/env python3
"""Emit one family-specific broad-E5 external-reference shard report.

The VESSL YAMLs for port external references call this script so each port
family can be launched and tracked independently.  The script is deliberately
claim-safe: it reports the current blocker state from
``port_external_reference_requirements.json`` and only exits as complete when
the family is explicitly marked ``broad_e5_passed`` by the manifest/audit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from check_port_external_references import (
    DEFAULT_SUPPORT_MATRIX,
    _display,
    _repo_path,
    build_external_reference_audit,
)


def build_family_reference_shard(
    family: str,
    manifest_path: Path,
    support_matrix_path: Path,
    require_committed: bool = False,
) -> dict[str, Any]:
    audit = build_external_reference_audit(
        manifest_path, support_matrix_path, require_committed=require_committed
    )
    rows = {row["family"]: row for row in audit["requirements"]}
    row = rows.get(family)
    if row is None:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "family": family,
            "status": "untracked",
            "manifest": audit["manifest"],
            "support_matrix": audit["support_matrix"],
            "required_scope": "",
            "current_status": "",
            "recommended_vessl_shard_id": "",
            "recommended_reference": "",
            "artifact_checks": [],
            "yaml_checks": [],
            "blockers": [
                "family is not tracked in the broad-E5 external-reference manifest"
            ],
            "completion_decision": (
                "Do not treat this shard as E5 evidence until the family is "
                "tracked and marked broad_e5_passed."
            ),
        }

    status = "passed" if row["status"] == "passed" else "blocked"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "status": status,
        "manifest": audit["manifest"],
        "support_matrix": audit["support_matrix"],
        "required_scope": row["required_scope"],
        "current_status": row["current_status"],
        "recommended_vessl_shard_id": row["recommended_vessl_shard_id"],
        "recommended_reference": row["recommended_reference"],
        "artifact_checks": row["artifact_checks"],
        "yaml_checks": row["yaml_checks"],
        "comparison_artifact_checks": row["comparison_artifact_checks"],
        "broad_e5_envelope_artifact_checks": row[
            "broad_e5_envelope_artifact_checks"
        ],
        "passed_comparison_artifact_count": row["passed_comparison_artifact_count"],
        "passed_broad_e4_comparison_artifact_count": row[
            "passed_broad_e4_comparison_artifact_count"
        ],
        "passed_broad_e5_envelope_artifact_count": row[
            "passed_broad_e5_envelope_artifact_count"
        ],
        "blockers": row["blockers"],
        "notes": row["notes"],
        "completion_decision": (
            "Family has broad-E5 external/reference evidence in the manifest."
            if status == "passed"
            else "Do not claim broad E5 for this family; complete the listed "
            "external/reference evidence first."
        ),
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{payload['family']}_external_reference_shard"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Port external-reference shard: `{payload['family']}`",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- status: `{payload['status']}`",
        f"- current_status: `{payload['current_status']}`",
        f"- required_scope: `{payload['required_scope']}`",
        f"- passed_comparison_artifact_count: `{payload.get('passed_comparison_artifact_count', 0)}`",
        f"- passed_broad_e4_comparison_artifact_count: `{payload.get('passed_broad_e4_comparison_artifact_count', 0)}`",
        f"- passed_broad_e5_envelope_artifact_count: `{payload.get('passed_broad_e5_envelope_artifact_count', 0)}`",
        f"- recommended_vessl_shard_id: `{payload['recommended_vessl_shard_id']}`",
        f"- recommended_reference: {payload['recommended_reference']}",
        "",
        "## Blockers",
        "",
    ]
    blockers = payload.get("blockers", [])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none recorded")
    lines.extend(["", f"Decision: {payload['completion_decision']}", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_display(json_path)}")
    print(f"wrote {_display(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--manifest",
        default="scripts/diagnostics/port_external_reference_requirements.json",
    )
    parser.add_argument("--support-matrix", default=DEFAULT_SUPPORT_MATRIX)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit 2 unless the selected family is broad-E5 complete.",
    )
    parser.add_argument(
        "--require-committed",
        action="store_true",
        help=(
            "T2.5: require the family's gating artifacts to be committed to HEAD, "
            "not merely present on disk (catches gitignored .omx). REQUIRES git — "
            "do NOT use on the git-less VESSL run image (git ls-tree returns empty "
            "there, which fail-closed would block everything). Use in the fast-"
            "suite / GitHub-Actions CI lanes that have git."
        ),
    )
    args = parser.parse_args(argv)

    payload = build_family_reference_shard(
        args.family,
        _repo_path(args.manifest),
        _repo_path(args.support_matrix),
        require_committed=args.require_committed,
    )
    _write_report(payload, _repo_path(args.output_dir))
    print(f"status={payload['status']} family={payload['family']}")
    if args.require_complete and payload["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

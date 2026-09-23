#!/usr/bin/env python3
"""Build an execution manifest for broad-E5 port external-reference shards.

This is orchestration evidence, not physics evidence.  It reads the broad-E5
external-reference backlog, inspects the referenced VESSL YAMLs, and emits the
expected shard IDs, output directories, result JSON paths, and diagnostic
commands that should run before the family readiness report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "scripts/diagnostics/port_external_reference_requirements.json"
DEFAULT_OUTPUT_ROOT = ".omx/physics-gate/vessl-2026-05-09-port-external"

DIAGNOSTIC_COMMAND_MARKERS = (
    "build_lumped_openems_sparameter_comparison.py",
    "build_lumped_openems_sweep_comparison.py",
    "build_patch_openems_wire_sparameter_comparison.py",
    "build_floquet_empty_space_analytic_comparison.py",
    "build_floquet_slab_analytic_comparison.py",
    "report_floquet_periodic_slab_oracles.py",
    "report_generalized_planar_quasitem_oracles.py",
    "build_msl_openems_sparameter_comparison.py",
    "build_waveguide_wr90_external_sparameter_comparison.py",
    "report_floquet_modal_oracles.py",
    "generate_floquet_modal_field_dump.py",
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a YAML mapping")
    return data


def _inspect_yaml(path: Path, *, family: str, shard_id: str) -> dict[str, Any]:
    exists = path.exists()
    blockers: list[str] = []
    if not exists:
        return {
            "yaml_path": _display(path),
            "exists": False,
            "status": "missing",
            "blockers": ["YAML file is missing"],
        }

    try:
        data = _load_yaml(path)
    except Exception as exc:
        return {
            "yaml_path": _display(path),
            "exists": True,
            "status": "invalid",
            "blockers": [f"YAML parse failed: {type(exc).__name__}: {exc}"],
        }

    run = str(data.get("run", ""))
    if f"--family {family}" not in run and f"--family\n    {family}" not in run:
        blockers.append(f"run script does not clearly report --family {family}")
    if shard_id not in run:
        blockers.append(f"run script does not reference expected shard id {shard_id}")
    if "report_port_external_reference_shard.py" not in run:
        blockers.append("run script does not emit family readiness shard JSON")

    diagnostic_markers = [marker for marker in DIAGNOSTIC_COMMAND_MARKERS if marker in run]
    expected_result_json = f"{DEFAULT_OUTPUT_ROOT}/{shard_id}/{family}_external_reference_shard.json"
    return {
        "yaml_path": _display(path),
        "exists": True,
        "status": "passed" if not blockers else "blocked",
        "name": data.get("name"),
        "description": data.get("description"),
        "resource_preset": data.get("resources", {}).get("preset") if isinstance(data.get("resources"), dict) else None,
        "image": data.get("image"),
        "expected_output_dir": f"{DEFAULT_OUTPUT_ROOT}/{shard_id}",
        "expected_result_json": expected_result_json,
        "diagnostic_markers": diagnostic_markers,
        "has_diagnostic_command": bool(diagnostic_markers),
        "blockers": blockers,
    }


def build_execution_manifest(
    manifest_path: Path,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in manifest.get("requirements", []) if row.get("required_for_e5")]
    shard_rows = []
    for row in rows:
        family = row["family"]
        shard_id = row["recommended_vessl_shard_id"]
        yaml_checks = [
            _inspect_yaml(_repo_path(path), family=family, shard_id=shard_id)
            for path in row.get("existing_vessl_yamls", [])
        ]
        has_launchable_yaml = any(check["status"] == "passed" for check in yaml_checks)
        has_diagnostic_command = any(check.get("has_diagnostic_command") for check in yaml_checks)
        blockers = []
        if not yaml_checks:
            blockers.append("no VESSL YAML listed")
        if not has_launchable_yaml:
            blockers.append("no launchable YAML matches family/shard result contract")
        shard_rows.append(
            {
                "family": family,
                "recommended_vessl_shard_id": shard_id,
                "current_status": row.get("current_status"),
                "expected_result_json": f"{output_root}/{shard_id}/{family}_external_reference_shard.json",
                "has_launchable_yaml": has_launchable_yaml,
                "has_diagnostic_command": has_diagnostic_command,
                "yaml_checks": yaml_checks,
                "status": "passed" if not blockers else "blocked",
                "blockers": blockers,
            }
        )

    blocked = [row for row in shard_rows if row["status"] != "passed"]
    missing_diagnostics = [row for row in shard_rows if not row["has_diagnostic_command"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not blocked else "blocked",
        "claim_scope": (
            "VESSL shard execution manifest only; proves launch/result-path wiring, "
            "not physics correctness or broad E5 completion"
        ),
        "manifest": _display(manifest_path),
        "output_root": output_root,
        "required_family_count": len(shard_rows),
        "launchable_family_count": sum(1 for row in shard_rows if row["has_launchable_yaml"]),
        "diagnostic_command_family_count": sum(1 for row in shard_rows if row["has_diagnostic_command"]),
        "blocked_family_count": len(blocked),
        "missing_diagnostic_command_families": [row["family"] for row in missing_diagnostics],
        "shards": shard_rows,
        "completion_decision": (
            "Do not call update_goal from this manifest; run shard result and full RF-infra E5 audits."
        ),
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "port_external_shard_execution_manifest.json"
    md_path = output_dir / "port_external_shard_execution_manifest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Port external-reference shard execution manifest",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- status: `{payload['status']}`",
        f"- required_family_count: `{payload['required_family_count']}`",
        f"- launchable_family_count: `{payload['launchable_family_count']}`",
        f"- diagnostic_command_family_count: `{payload['diagnostic_command_family_count']}`",
        "",
        payload["claim_scope"],
        "",
        "| Family | Shard ID | Launchable | Has diagnostics | Expected result JSON |",
        "|---|---|---:|---:|---|",
    ]
    for row in payload["shards"]:
        lines.append(
            f"| `{row['family']}` | `{row['recommended_vessl_shard_id']}` | "
            f"`{row['has_launchable_yaml']}` | `{row['has_diagnostic_command']}` | "
            f"`{row['expected_result_json']}` |"
        )
    lines.extend(["", "## Suggested parallel launch commands", ""])
    seen_yamls: set[str] = set()
    for row in payload["shards"]:
        for check in row["yaml_checks"]:
            if check["status"] != "passed":
                continue
            yaml_path = check["yaml_path"]
            if yaml_path in seen_yamls:
                continue
            seen_yamls.add(yaml_path)
            lines.append(f"- `vessl run create -f {yaml_path}`")
            break
    lines.extend(
        [
            "",
            "After runs finish, collect the shard output root and run:",
            "",
            "```bash",
            "python scripts/diagnostics/check_port_external_shard_results.py \\",
            "  --result-root .omx/physics-gate/vessl-2026-05-09-port-external \\",
            "  --directory-layout shard_id \\",
            "  --output-dir .omx/physics-gate/latest-port-external-shard-result-audit \\",
            "  --require-complete",
            "```",
            "",
        ]
    )
    if payload["missing_diagnostic_command_families"]:
        lines.extend(["", "## Families without diagnostic pre-run commands", ""])
        for family in payload["missing_diagnostic_command_families"]:
            lines.append(f"- `{family}`")
    lines.append(f"\nDecision: {payload['completion_decision']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_display(json_path)}")
    print(f"wrote {_display(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-port-external-shard-execution-manifest",
    )
    parser.add_argument("--require-launchable", action="store_true")
    args = parser.parse_args(argv)

    payload = build_execution_manifest(_repo_path(args.manifest), output_root=args.output_root)
    _write_report(payload, _repo_path(args.output_dir))
    print(
        "status={status} launchable_family_count={launchable} diagnostic_command_family_count={diag}".format(
            status=payload["status"],
            launchable=payload["launchable_family_count"],
            diag=payload["diagnostic_command_family_count"],
        )
    )
    if args.require_launchable and payload["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

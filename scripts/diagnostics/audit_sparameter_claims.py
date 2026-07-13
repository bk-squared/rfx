#!/usr/bin/env python3
"""Audit S-parameter support claims against evidence-level rules."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_FAMILY_LEVELS = {
    "lumped_port": "E2/E3/E4-partial",
    "wire_port": "E2/E3/E4-partial",
    "microstrip_line_port": "E5-narrow/eigenmode-blocked",
    "rectangular_waveguide_port": "E5-broad-magnitude/E4-external/phase-narrow",
    "coaxial_port": "E5-broad/E4-broad/differentiable",
    "floquet_port": "E2/E3-modal/slab-analytic/no-public-api",
    "soft_current_source": "not_a_port",
    "tfsf_plane_wave_source": "not_a_port",
    "non_port_observable": "not_a_port",
}

CURRENT_DOC_REQUIRED_SNIPPETS = {
    "README.md": [
        "docs/guides/sparameter_support_matrix.md",
        "compute_coaxial_line_reflection",
    ],
    "docs/guides/sparameter_support_matrix.md": [
        "broad magnitude evidence",
        "raw voltage/current replay",
    ],
    "docs/public/validation/reference-lane.mdx": [
        "uniform Cartesian Yee",
        "compute_coaxial_line_reflection(...)"
    ],
    "docs/public/api/support-boundaries.mdx": [
        "compute_msl_s_matrix()",
        "compute_waveguide_s_matrix()",
        "compute_coaxial_line_reflection(...)"
    ],
}


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def audit_support_matrix(matrix_path: Path) -> list[dict[str, Any]]:
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    families = {entry.get("family"): entry for entry in data.get("port_families", [])}

    missing_families = sorted(set(EXPECTED_FAMILY_LEVELS) - set(families))
    for family in missing_families:
        findings.append(
            {
                "severity": "error",
                "path": _rel(matrix_path),
                "check": "family_present",
                "message": f"missing expected family {family}",
            }
        )

    for family, expected_level in EXPECTED_FAMILY_LEVELS.items():
        entry = families.get(family)
        if not entry:
            continue
        actual = entry.get("evidence_level")
        if actual != expected_level:
            findings.append(
                {
                    "severity": "error",
                    "path": _rel(matrix_path),
                    "family": family,
                    "check": "evidence_level",
                    "message": f"expected {expected_level!r}, got {actual!r}",
                }
            )
        for key, legacy_key in (
            ("validation_status", "validation_status"),
            ("validated_scope", "claim_envelope"),
        ):
            if not entry.get(key, entry.get(legacy_key)):
                findings.append(
                    {
                        "severity": "error",
                        "path": _rel(matrix_path),
                        "family": family,
                        "check": key,
                        "message": f"missing {key}",
                    }
                )
        if entry.get("calculation_api"):
            for key in ("evidence_artifacts", "numeric_metrics", "caveats"):
                if not entry.get(key):
                    findings.append(
                        {
                            "severity": "error",
                            "path": _rel(matrix_path),
                            "family": family,
                            "check": key,
                            "message": f"family with a calculation API lacks {key}",
                        }
                    )
    return findings


def audit_current_docs() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for rel_path, snippets in CURRENT_DOC_REQUIRED_SNIPPETS.items():
        path = REPO_ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                findings.append(
                    {
                        "severity": "error",
                        "path": rel_path,
                        "check": "required_snippet",
                        "message": f"missing snippet: {snippet}",
                    }
                )

    for rel_path in ("README.md", "docs/public/index.mdx"):
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        if "0 xfails" in text:
            findings.append(
                {
                    "severity": "error",
                    "path": rel_path,
                    "check": "hidden_xfails_wording",
                    "message": "current overview must not claim 0 xfails",
                }
            )
    return findings


def build_audit(matrix_path: Path) -> dict[str, Any]:
    findings = audit_support_matrix(matrix_path) + audit_current_docs()
    status = "passed" if not any(f["severity"] == "error" for f in findings) else "failed"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "matrix_path": _rel(matrix_path),
        "expected_family_levels": EXPECTED_FAMILY_LEVELS,
        "findings": findings,
    }


def write_audit(audit: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sparameter_claim_audit.json"
    md_path = output_dir / "sparameter_claim_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lines = [
        "# S-parameter claim audit",
        "",
        f"- status: `{audit['status']}`",
        f"- matrix_path: `{audit['matrix_path']}`",
        "",
        "## Expected family levels",
        "",
    ]
    for family, level in audit["expected_family_levels"].items():
        lines.append(f"- `{family}`: `{level}`")
    lines.extend(["", "## Findings", ""])
    if audit["findings"]:
        for finding in audit["findings"]:
            lines.append(
                f"- `{finding['severity']}` {finding['path']} "
                f"{finding['check']}: {finding['message']}"
            )
    else:
        lines.append("- none")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_rel(json_path)}")
    print(f"wrote {_rel(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default="docs/guides/sparameter_support_matrix.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    audit = build_audit(_repo_path(args.matrix))
    write_audit(audit, _repo_path(args.output_dir))
    print(f"status={audit['status']}")
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

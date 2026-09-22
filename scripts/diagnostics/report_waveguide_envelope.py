#!/usr/bin/env python3
"""Build a claims-aware WaveguidePort envelope report from gate artifacts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_MAG_RE = re.compile(
    r"^\[(?P<label>[^\]]+)\] \|S\|: max_diff=(?P<max>[0-9.]+) "
    r"mean_diff=(?P<mean>[0-9.]+) \(gate (?P<gate>[0-9.]+)\)"
)
_PHASE_RE = re.compile(
    r"^\[(?P<label>[^\]]+)\] ∠S: max_diff=(?P<max>[0-9.]+)° "
    r"mean_diff=(?P<mean>[0-9.]+)° \(gate (?P<gate>[0-9.]+)°\)"
)
_COMPLEX_RE = re.compile(
    r"^\[(?P<label>[^\]]+)\] \|S_rfx−S_ref\|: max=(?P<max>[0-9.]+) "
    r"mean=(?P<mean>[0-9.]+) \(gate (?P<gate>[0-9.]+)\)"
)


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|")


def parse_cv11_stdout(text: str) -> dict[str, Any]:
    metrics: dict[str, dict[str, float]] = {}
    loaded_refs: list[str] = []
    status = "failed"
    for line in text.splitlines():
        if line.startswith("[meep-ref] loaded"):
            loaded_refs.append("MEEP")
        elif line.startswith("[openems-ref] loaded"):
            loaded_refs.append("OpenEMS")
        elif line.startswith("[palace-ref] loaded"):
            loaded_refs.append("Palace")
        elif "CROSSVAL-11 PASS" in line:
            status = "passed"
        elif "CROSSVAL-11 FAIL" in line:
            status = "failed"

        for kind, regex in (
            ("magnitude", _MAG_RE),
            ("phase", _PHASE_RE),
            ("complex", _COMPLEX_RE),
        ):
            match = regex.match(line)
            if match:
                label = match.group("label")
                entry = metrics.setdefault(label, {})
                entry[f"{kind}_max"] = float(match.group("max"))
                entry[f"{kind}_mean"] = float(match.group("mean"))
                entry[f"{kind}_gate"] = float(match.group("gate"))
    return {
        "status": status,
        "loaded_external_references": loaded_refs,
        "metrics": metrics,
    }


def build_report(gate_json: Path, cv11_stdout: Path, cv11_rc: int) -> dict[str, Any]:
    gate = json.loads(gate_json.read_text(encoding="utf-8"))
    cv11 = parse_cv11_stdout(cv11_stdout.read_text(encoding="utf-8"))
    gate_result = gate["results"][0]
    status = (
        "passed"
        if gate.get("status") == "passed"
        and gate_result.get("coverage_status") == "full"
        and cv11["status"] == "passed"
        and cv11_rc == 0
        else "failed"
    )
    external_refs = cv11["loaded_external_references"]
    e4_status = "present" if external_refs else "blocked"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "claim_level": "E5-narrow" if status == "passed" and e4_status == "present" else "E2/E3",
        "claim_scope": (
            "Uniform-Yee rectangular waveguide TE10/TE/TM modal-port envelope "
            "covered by empty-guide, PEC-short, slab Airy/reference-plane, "
            "passivity, reciprocity, dump-diagnostic, and external-reference "
            "artifacts. This is not a branch/T-junction, multimode-normalized, "
            "or broad nonuniform claim."
        ),
        "gate_result_json": _rel(gate_json),
        "cv11_stdout": _rel(cv11_stdout),
        "cv11_returncode": cv11_rc,
        "waveguide_gate": {
            "status": gate.get("status"),
            "coverage_status": gate_result.get("coverage_status"),
            "passed": gate_result.get("pytest_summary", {}).get("counts", {}).get("passed"),
            "summary_line": gate_result.get("pytest_summary", {}).get("summary_line"),
        },
        "cv11": cv11,
        "evidence_inventory": [
            {
                "level": "E2",
                "claim": "empty-guide, PEC-short, slab Airy/reference-plane analytic gates",
                "artifact": _rel(cv11_stdout),
            },
            {
                "level": "E2",
                "claim": "waveguide-port oracle, passivity, reciprocity, and source-directionality gates",
                "artifact": _rel(gate_json),
            },
            {
                "level": "E3",
                "claim": "independent WR90 dump/projection diagnostic harness",
                "artifact": "out_cross_tool_audit/cross_tool_half_step_audit_R1.json",
            },
            {
                "level": "E4",
                "claim": "external MEEP/OpenEMS/Palace WR90 references loaded by cv11",
                "artifact": _rel(cv11_stdout),
                "status": e4_status,
                "references": external_refs,
            },
        ],
        "blocked_claims": [
            {
                "claim": "branch/T-junction or arbitrary multiport waveguide networks",
                "reason": "per-port reference geometry and validation ladder not defined",
            },
            {
                "claim": "multi-mode normalize=True envelope",
                "reason": "unsupported by support matrix and not covered by current M4 artifacts",
            },
            {
                "claim": "broad nonuniform waveguide-port S-parameters",
                "reason": "nonuniform lane remains restricted shadow normalize=True single-mode",
            },
        ],
    }


def _write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "waveguide_envelope_report.json"
    md_path = output_dir / "waveguide_envelope_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# WaveguidePort claims envelope report",
        "",
        f"- status: `{report['status']}`",
        f"- claim_level: `{report['claim_level']}`",
        f"- gate_result_json: `{report['gate_result_json']}`",
        f"- cv11_stdout: `{report['cv11_stdout']}`",
        f"- cv11_returncode: `{report['cv11_returncode']}`",
        f"- external_references: `{', '.join(report['cv11']['loaded_external_references']) or 'none'}`",
        "",
        report["claim_scope"],
        "",
        "## Key cv11 gates",
        "",
        "| Label | mag mean/gate | phase mean/gate | complex max/gate |",
        "|---|---:|---:|---:|",
    ]
    for label, metric in report["cv11"]["metrics"].items():
        complex_cell = (
            "—"
            if "complex_max" not in metric
            else f"{metric['complex_max']:.4f}/{metric['complex_gate']:.3f}"
        )
        lines.append(
            "| "
            f"`{_md_cell(label)}` | "
            f"{metric.get('magnitude_mean', 0):.4f}/{metric.get('magnitude_gate', 0):.3f} | "
            f"{metric.get('phase_mean', 0):.2f}/{metric.get('phase_gate', 0):.1f} | "
            f"{complex_cell} |"
        )
    lines.extend(["", "## Evidence inventory", ""])
    for item in report["evidence_inventory"]:
        suffix = ""
        if item.get("status"):
            suffix = f" ({item['status']}: {', '.join(item.get('references', []))})"
        lines.append(f"- `{item['level']}` {item['claim']}: `{item['artifact']}`{suffix}")
    lines.extend(["", "## Blocked claims", ""])
    for item in report["blocked_claims"]:
        lines.append(f"- {item['claim']}: {item['reason']}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_rel(json_path)}")
    print(f"wrote {_rel(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--cv11-stdout", required=True)
    parser.add_argument("--cv11-rc", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = build_report(
        _repo_path(args.gate_json),
        _repo_path(args.cv11_stdout),
        args.cv11_rc,
    )
    _write_reports(report, _repo_path(args.output_dir))
    print(f"status={report['status']} claim_level={report['claim_level']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

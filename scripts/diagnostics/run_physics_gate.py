#!/usr/bin/env python3
"""Run the rfx physics-validation gate as explicit pytest groups.

This script is intentionally a thin orchestrator over pytest.  It does not
convert failing physics checks into success: every group is run independently,
the raw stdout/stderr and return code are persisted, and the script exits
non-zero whenever any group fails, times out, or errors.

The purpose is to avoid another "manifest complete == physics complete" mistake:
the durable result artifact must say exactly which physics surfaces were
executed and which ones are still blockers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GateGroup:
    """One independently executed physics-gate pytest group."""

    group_id: str
    description: str
    tests: tuple[str, ...]
    pytest_args: tuple[str, ...] = ()
    claim_level: str = "E1"
    coverage_scope: str = "regression"
    requires_external_reference: bool = False
    validated_claims: tuple[dict[str, str], ...] = ()
    blocked_claims: tuple[dict[str, str], ...] = ()


@dataclass
class GateResult:
    """Durable result for one gate group."""

    group_id: str
    description: str
    tests: list[str]
    command: list[str]
    status: str
    returncode: int | None
    duration_s: float
    stdout_path: str
    stderr_path: str
    coverage_status: str
    claim_level: str
    required_skip_count: int
    optional_skip_count: int
    strict_xfail_count: int
    pytest_summary: dict[str, Any]
    skip_reasons: list[dict[str, Any]]
    xfail_reasons: list[dict[str, Any]]
    validated_claims: list[dict[str, str]]
    blocked_claims: list[dict[str, str]]
    vessl_run_ids: list[str]


GATE_GROUPS: tuple[GateGroup, ...] = (
    GateGroup(
        group_id="maxwell_integrity",
        description=(
            "Discrete Maxwell residuals, energy/Poynting behavior, "
            "convergence, and reciprocal-network invariants."
        ),
        tests=(
            "tests/test_physics.py",
            "tests/test_physics_integrity.py",
            "tests/test_conservation_laws.py",
        ),
        claim_level="E1",
        validated_claims=(
            {
                "claim": "discrete Maxwell and conservation-law regression gate",
                "evidence_level": "E1",
                "artifact": "physics_gate_results.json plus raw pytest stdout/stderr",
            },
        ),
    ),
    GateGroup(
        group_id="sparameter_core",
        description=(
            "Core lumped/wire S-parameter contracts and physical S11 "
            "optimization regressions."
        ),
        tests=(
            "tests/test_sparam.py",
            "tests/test_s11_at_freq.py",
            "tests/test_minimize_s11_at_freq_physical.py",
            "tests/test_lumped_port_sparams_jit.py",
            "tests/test_wire_sparam.py",
            "tests/test_wire_port_sparams_forward.py",
            "tests/test_twoport_wire_port.py",
        ),
        claim_level="E1",
        validated_claims=(
            {
                "claim": "core lumped/wire S-parameter software and invariant regression gate",
                "evidence_level": "E1",
                "artifact": "physics_gate_results.json plus raw pytest stdout/stderr",
            },
        ),
        blocked_claims=(
            {
                "claim": "broad calibrated lumped/wire S-parameter physics envelope",
                "evidence_level": "E5",
                "reason": "requires analytic and independent V-I replay evidence per campaign D6",
            },
        ),
    ),
    GateGroup(
        group_id="waveguide_ports",
        description=(
            "Waveguide-port S-matrix extraction, WR-90 oracles, two-port "
            "contracts, and forward-path behavior."
        ),
        tests=(
            "tests/test_waveguide_port.py",
            "tests/test_waveguide_port_validation_battery.py",
            "tests/test_waveguide_twoport_contract_v1.py",
            "tests/test_waveguide_forward.py",
            "tests/test_wr90_port_oracles.py",
        ),
        claim_level="E2",
        validated_claims=(
            {
                "claim": "waveguide-port oracle and reciprocity regression envelope",
                "evidence_level": "E2",
                "artifact": "tests/test_wr90_port_oracles.py and waveguide-port gate stdout",
            },
        ),
        blocked_claims=(
            {
                "claim": "claims-bearing WaveguidePort E5 envelope",
                "evidence_level": "E5",
                "reason": "requires final command/artifact/envelope audit and external reference if E4 is claimed",
            },
        ),
    ),
    GateGroup(
        group_id="msl_ports",
        description=(
            "Microstrip line port API, integration, eigenmode, plane "
            "extractor, and de-embedding helpers."
        ),
        tests=(
            "tests/test_msl_port.py",
            "tests/test_msl_port_integration.py",
            "tests/test_msl_port_preflight.py",
            "tests/test_msl_eigenmode_solver.py",
            "tests/test_msl_plane_extractor_jax.py",
            "tests/test_msl_wave_decomp_jvp.py",
        ),
        claim_level="E2",
        validated_claims=(
            {
                "claim": "MSL laplace/quasi-TEM regression and analytic helper gate",
                "evidence_level": "E2",
                "artifact": "MSL gate stdout and integration tests",
            },
        ),
        blocked_claims=(
            {
                "claim": "MSL eigenmode Path-B source/extractor support",
                "evidence_level": "E2",
                "reason": "currently strict-xfailed/unsupported until implementation and validation land",
            },
        ),
    ),
    GateGroup(
        group_id="modal_normalization",
        description=(
            "Shared modal decomposition, normalization, flux, and PEC-mask "
            "extraction checks used by port calculators."
        ),
        tests=(
            "tests/test_deembed.py",
            "tests/test_eigenmode.py",
            "tests/test_eigenmode_port.py",
            "tests/test_multimode_waveguide.py",
            "tests/test_normalization.py",
            "tests/test_normalize_flux.py",
            "tests/test_extract_s_matrix_pec_mask.py",
        ),
        claim_level="E1",
        validated_claims=(
            {
                "claim": "modal decomposition and normalization regression gate",
                "evidence_level": "E1",
                "artifact": "modal_normalization stdout/stderr artifacts",
            },
        ),
    ),
    GateGroup(
        group_id="unsupported_or_shadow_lanes",
        description=(
            "Coaxial/Floquet/nonuniform/support-matrix checks that should "
            "remain unsupported, experimental, or shadow unless separately "
            "promoted by evidence."
        ),
        tests=(
            "tests/test_coaxial_port.py",
            "tests/test_floquet.py",
            "tests/test_nonuniform_grad_sparams.py",
            "tests/test_port_dump_replay.py",
            "tests/test_port_observable_validation.py",
            "tests/test_sparameter_support_contract.py",
        ),
        claim_level="E0",
        coverage_scope="not_claims_bearing",
        validated_claims=(
            {
                "claim": "unsupported/shadow lanes fail loudly or remain contract-covered",
                "evidence_level": "E0",
                "artifact": "support-contract and observable-validation tests",
            },
        ),
        blocked_claims=(
            {
                "claim": "coaxial, Floquet, nonuniform/distributed/AD claims-bearing S-parameters",
                "evidence_level": "E5",
                "reason": "promotion ladders remain campaign D7 work",
            },
        ),
    ),
    GateGroup(
        group_id="external_crossval",
        description=(
            "Cross-validation harnesses and external-reference handling. "
            "Missing external references must report skip/unknown, not pass."
        ),
        tests=(
            "tests/test_crossval_migration_smoke.py",
            "tests/test_crossval_comprehensive.py",
            "tests/test_meep_crossval.py",
            "tests/test_meep_crossval_dielectric_cavity.py",
            "tests/test_openems_crossval.py",
        ),
        claim_level="E0",
        coverage_scope="not_claims_bearing",
        validated_claims=(
            {
                "claim": "external cross-validation harness smoke/contract gate",
                "evidence_level": "E0",
                "artifact": "external_crossval stdout/stderr artifacts",
            },
        ),
        blocked_claims=(
            {
                "claim": "external full-wave cross-solver validation",
                "evidence_level": "E4",
                "reason": "requires actual Meep/openEMS/CSXCAD reference artifact, not only smoke tests",
            },
        ),
    ),
    GateGroup(
        group_id="slow_physics_release",
        description=(
            "Opt-in release-tag physics checks that are intentionally outside "
            "the fast local gate."
        ),
        tests=("tests",),
        pytest_args=("-m", "slow_physics"),
        claim_level="E1",
        validated_claims=(
            {
                "claim": "release-tag physics regression gate",
                "evidence_level": "E1",
                "artifact": "slow_physics_release VESSL/local result JSON",
            },
        ),
        blocked_claims=(
            {
                "claim": "GPU V173-A bit-identity baseline",
                "evidence_level": "E5",
                "reason": "GPU parity needs a separate justified baseline from CPU bit-identity",
            },
        ),
    ),
    GateGroup(
        group_id="slow_boundary_absorber",
        description="Slow CPML/PML reflectivity and absorber-regression checks.",
        tests=(
            "tests/test_cpml.py",
            "tests/test_pml_reflectivity.py",
        ),
        pytest_args=("-m", "slow"),
        claim_level="E1",
        validated_claims=(
            {
                "claim": "slow boundary absorber regression gate",
                "evidence_level": "E1",
                "artifact": "slow_boundary_absorber result JSON",
            },
        ),
    ),
    GateGroup(
        group_id="slow_external_crossval",
        description=(
            "Slow Meep/OpenEMS cross-validation tests. Missing external "
            "solver binaries must be reported by pytest as skip/error rather "
            "than hidden by the gate runner."
        ),
        tests=(
            "tests/test_meep_crossval.py",
            "tests/test_openems_crossval.py",
        ),
        pytest_args=("-m", "slow"),
        claim_level="E4",
        requires_external_reference=True,
        blocked_claims=(
            {
                "claim": "slow external full-wave cross-validation",
                "evidence_level": "E4",
                "reason": "Meep/openEMS/CSXCAD must be importable or equivalent reference artifacts must be present",
            },
        ),
    ),
    GateGroup(
        group_id="slow_nonuniform_subgrid_subpixel",
        description=(
            "Slow nonuniform, subgrid, and subpixel convergence/crossval "
            "checks."
        ),
        tests=(
            "tests/test_nonuniform_convergence.py",
            "tests/test_subgrid_crossval.py",
            "tests/test_subpixel.py",
            "tests/test_subpixel_pec.py",
        ),
        pytest_args=("-m", "slow"),
        claim_level="E1",
        validated_claims=(
            {
                "claim": "slow nonuniform/subpixel regression gate",
                "evidence_level": "E1",
                "artifact": "slow_nonuniform_subgrid_subpixel result JSON",
            },
        ),
        blocked_claims=(
            {
                "claim": "SBP-SAT/subgrid RMS accuracy promotion",
                "evidence_level": "E5",
                "reason": "strict-xfailed until the subgrid crossval ladder is restored",
            },
        ),
    ),
    GateGroup(
        group_id="slow_sbp_sat",
        description="Slow SBP-SAT stability and energy-conservation checks.",
        tests=(
            "tests/test_sbp_sat_1d.py",
            "tests/test_sbp_sat_2d.py",
        ),
        pytest_args=("-m", "slow"),
        claim_level="E1",
        validated_claims=(
            {
                "claim": "slow SBP-SAT stability/energy regression gate",
                "evidence_level": "E1",
                "artifact": "slow_sbp_sat result JSON",
            },
        ),
    ),
    GateGroup(
        group_id="slow_msl",
        description="Slow MSL thru-line passive/eigenmode integration gates.",
        tests=("tests/test_msl_port_integration.py",),
        pytest_args=("-m", "slow"),
        claim_level="E2",
        validated_claims=(
            {
                "claim": "slow MSL laplace thru-line passive gate",
                "evidence_level": "E2",
                "artifact": "slow_msl result JSON",
            },
        ),
        blocked_claims=(
            {
                "claim": "slow MSL eigenmode gate",
                "evidence_level": "E2",
                "reason": "mode='eigenmode' remains strict-xfailed until implemented",
            },
        ),
    ),
)


_SUMMARY_OUTCOMES = {
    "passed": "passed",
    "failed": "failed",
    "error": "errors",
    "errors": "errors",
    "skipped": "skipped",
    "xfailed": "xfailed",
    "xpassed": "xpassed",
    "deselected": "deselected",
    "warning": "warnings",
    "warnings": "warnings",
}


def gate_groups_by_id() -> dict[str, GateGroup]:
    """Return known gate groups keyed by stable id."""

    return {group.group_id: group for group in GATE_GROUPS}


def _empty_pytest_counts() -> dict[str, int]:
    return {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
        "warnings": 0,
    }


def parse_pytest_report(stdout: str) -> dict[str, Any]:
    """Extract pytest outcome counts and skip/xfail reasons from stdout.

    The runner intentionally avoids a hard dependency on pytest-json-report.
    Pytest's `-ra` summary is enough for the gate-level coverage semantics we
    need here: a zero return code can still contain skipped external solver
    checks or strict xfails that must not be hidden behind `passed`.
    """

    counts = _empty_pytest_counts()
    summary_line = ""
    skipped: list[dict[str, Any]] = []
    xfailed: list[dict[str, Any]] = []

    for line in stdout.splitlines():
        if line.startswith("SKIPPED ["):
            match = re.match(r"SKIPPED \[(\d+)\] (.*)", line)
            skipped.append(
                {
                    "count": int(match.group(1)) if match else 1,
                    "reason": match.group(2).strip() if match else line.strip(),
                }
            )
        elif line.startswith("XFAIL "):
            item, _, reason = line.partition(" - ")
            xfailed.append(
                {
                    "count": 1,
                    "test": item.removeprefix("XFAIL ").strip(),
                    "reason": reason.strip(),
                }
            )

    for line in reversed(stdout.splitlines()):
        if " in " not in line:
            continue
        found = re.findall(
            r"(\d+)\s+"
            r"(passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)",
            line,
        )
        if not found:
            continue
        summary_line = line.strip()
        for raw_count, raw_name in found:
            counts[_SUMMARY_OUTCOMES[raw_name]] = int(raw_count)
        break

    if not counts["skipped"] and skipped:
        counts["skipped"] = sum(item["count"] for item in skipped)
    if not counts["xfailed"] and xfailed:
        counts["xfailed"] = sum(item["count"] for item in xfailed)

    return {
        "counts": counts,
        "summary_line": summary_line,
        "skipped": skipped,
        "xfailed": xfailed,
    }


def _coverage_status(
    group: GateGroup,
    *,
    execution_status: str,
    counts: dict[str, int],
) -> str:
    """Map execution status plus pytest outcomes to claim-coverage status."""

    if execution_status != "passed":
        return "blocked"
    if group.requires_external_reference and counts["skipped"] > 0:
        return "blocked"
    if group.coverage_scope == "not_claims_bearing":
        return "not_claims_bearing"
    if counts["skipped"] > 0:
        return "passed_with_skips"
    if counts["xfailed"] > 0 or counts["xpassed"] > 0:
        return "passed_with_xfails"
    return "full"


def _blocked_claims(
    group: GateGroup,
    *,
    execution_status: str,
    pytest_report: dict[str, Any],
) -> list[dict[str, str]]:
    claims = [dict(claim) for claim in group.blocked_claims]
    counts = pytest_report["counts"]

    if execution_status != "passed":
        claims.append(
            {
                "claim": f"{group.group_id} physics-gate execution",
                "evidence_level": group.claim_level,
                "reason": f"group execution status is {execution_status}",
            }
        )

    if group.requires_external_reference and counts["skipped"] > 0:
        for item in pytest_report["skipped"]:
            claims.append(
                {
                    "claim": f"{group.group_id} required external reference",
                    "evidence_level": "E4",
                    "reason": str(item["reason"]),
                }
            )

    for item in pytest_report["xfailed"]:
        test = item.get("test") or group.group_id
        claims.append(
            {
                "claim": f"unpromoted strict-xfail lane: {test}",
                "evidence_level": group.claim_level,
                "reason": str(item.get("reason", "pytest xfail")),
            }
        )

    return claims


def coverage_metadata(
    group: GateGroup,
    *,
    execution_status: str,
    stdout: str,
    vessl_run_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build claims-aware coverage metadata for one group result."""

    pytest_report = parse_pytest_report(stdout)
    counts = pytest_report["counts"]
    required_skip_count = (
        counts["skipped"] if group.requires_external_reference else 0
    )
    optional_skip_count = counts["skipped"] - required_skip_count
    coverage_status = _coverage_status(
        group,
        execution_status=execution_status,
        counts=counts,
    )

    return {
        "coverage_status": coverage_status,
        "claim_level": group.claim_level,
        "required_skip_count": required_skip_count,
        "optional_skip_count": optional_skip_count,
        "strict_xfail_count": counts["xfailed"],
        "pytest_summary": {
            "counts": counts,
            "summary_line": pytest_report["summary_line"],
        },
        "skip_reasons": pytest_report["skipped"],
        "xfail_reasons": pytest_report["xfailed"],
        "validated_claims": [dict(claim) for claim in group.validated_claims],
        "blocked_claims": _blocked_claims(
            group,
            execution_status=execution_status,
            pytest_report=pytest_report,
        ),
        "vessl_run_ids": vessl_run_ids or [],
    }


def aggregate_coverage_status(results: list[dict[str, Any]]) -> str:
    """Summarize group coverage without hiding blocked/skipped/xfail states."""

    statuses = [str(result.get("coverage_status", "blocked")) for result in results]
    if any(status == "blocked" for status in statuses):
        return "blocked"
    if any(status == "passed_with_skips" for status in statuses):
        return "passed_with_skips"
    if any(status == "passed_with_xfails" for status in statuses):
        return "passed_with_xfails"
    if statuses and all(status == "not_claims_bearing" for status in statuses):
        return "not_claims_bearing"
    if any(status == "not_claims_bearing" for status in statuses):
        return "passed_with_non_claims_bearing_groups"
    return "full"


def parse_vessl_run_ids(
    values: list[str] | None,
    *,
    selected_groups: list[GateGroup],
) -> dict[str, list[str]]:
    """Parse `--vessl-run-id` values.

    Accepts either `GROUP=ID` for one group or a bare `ID`, which is attached to
    every selected group. Environment variables are intentionally optional: VESSL
    images are not guaranteed to expose a stable name, so explicit CLI IDs are
    preferred when recording release evidence.
    """

    mapping: dict[str, list[str]] = {group.group_id: [] for group in selected_groups}
    env_run_id = (
        os.environ.get("VESSL_RUN_ID")
        or os.environ.get("VESSEL_RUN_ID")
        or os.environ.get("VESSL_JOB_ID")
    )
    all_values = list(values or [])
    if env_run_id:
        all_values.append(env_run_id)

    for value in all_values:
        if "=" in value:
            group_id, run_id = value.split("=", 1)
            group_id = group_id.strip()
            if group_id not in mapping:
                raise SystemExit(f"unknown --vessl-run-id group: {group_id}")
            mapping[group_id].append(run_id.strip())
        else:
            for group_run_ids in mapping.values():
                group_run_ids.append(value.strip())
    return mapping


def _selected_groups(names: list[str] | None) -> list[GateGroup]:
    if not names:
        return list(GATE_GROUPS)
    by_id = gate_groups_by_id()
    unknown = sorted(set(names) - set(by_id))
    if unknown:
        raise SystemExit(f"unknown gate group(s): {', '.join(unknown)}")
    return [by_id[name] for name in names]


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path.relative_to(REPO_ROOT))


def run_group(
    group: GateGroup,
    *,
    output_dir: Path,
    timeout_s: int | None,
    extra_pytest_args: list[str],
    vessl_run_ids: list[str] | None = None,
) -> GateResult:
    """Run one group and persist stdout/stderr."""

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-ra",
        *group.tests,
        *group.pytest_args,
        *extra_pytest_args,
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        duration_s = time.monotonic() - start
        status = "passed" if completed.returncode == 0 else "failed"
        returncode: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        duration_s = time.monotonic() - start
        status = "timeout"
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stderr += (
            f"\n[TIMEOUT] group {group.group_id!r} exceeded "
            f"{timeout_s} seconds.\n"
        )

    stdout_path = _write_text(output_dir / f"{group.group_id}.stdout.txt", stdout)
    stderr_path = _write_text(output_dir / f"{group.group_id}.stderr.txt", stderr)
    metadata = coverage_metadata(
        group,
        execution_status=status,
        stdout=stdout,
        vessl_run_ids=vessl_run_ids,
    )

    return GateResult(
        group_id=group.group_id,
        description=group.description,
        tests=list(group.tests),
        command=command,
        status=status,
        returncode=returncode,
        duration_s=duration_s,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        **metadata,
    )


def write_reports(output_dir: Path, results: list[GateResult]) -> None:
    """Persist JSON and Markdown summaries."""

    generated_at = datetime.now(timezone.utc).isoformat()
    result_dicts = [asdict(r) for r in results]
    payload = {
        "generated_at": generated_at,
        "repo_root": str(REPO_ROOT),
        "status": "passed" if all(r.status == "passed" for r in results) else "failed",
        "coverage_status": aggregate_coverage_status(result_dicts),
        "required_skip_count": sum(r.required_skip_count for r in results),
        "optional_skip_count": sum(r.optional_skip_count for r in results),
        "strict_xfail_count": sum(r.strict_xfail_count for r in results),
        "validated_claims": [
            claim for result in results for claim in result.validated_claims
        ],
        "blocked_claims": [
            claim for result in results for claim in result.blocked_claims
        ],
        "vessl_run_ids": sorted(
            {
                run_id
                for result in results
                for run_id in result.vessl_run_ids
            }
        ),
        "results": result_dicts,
    }
    (output_dir / "physics_gate_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# rfx physics gate results",
        "",
        f"- generated_at: `{generated_at}`",
        f"- overall_status: `{payload['status']}`",
        f"- coverage_status: `{payload['coverage_status']}`",
        f"- required_skip_count: `{payload['required_skip_count']}`",
        f"- optional_skip_count: `{payload['optional_skip_count']}`",
        f"- strict_xfail_count: `{payload['strict_xfail_count']}`",
        "",
        "| Group | Status | Coverage | E-level | Req. skips | Opt. skips | Xfails | Duration (s) | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            f"`{result.group_id}` | `{result.status}` | "
            f"`{result.coverage_status}` | `{result.claim_level}` | "
            f"{result.required_skip_count} | {result.optional_skip_count} | "
            f"{result.strict_xfail_count} | "
            f"{result.duration_s:.1f} | "
            f"[stdout]({Path(result.stdout_path).name}) / "
            f"[stderr]({Path(result.stderr_path).name}) |"
        )
    if payload["blocked_claims"]:
        lines.extend(["", "## Blocked or unpromoted claims", ""])
        for claim in payload["blocked_claims"]:
            lines.append(
                "- "
                f"`{claim.get('evidence_level', '?')}` "
                f"{claim.get('claim', 'claim')}: "
                f"{claim.get('reason', 'no reason recorded')}"
            )
    lines.extend(
        [
            "",
            "A `failed` or `timeout` group is a blocking physics-gate result, "
            "not a completed validation claim.",
            "A `passed` execution status can still be `passed_with_skips`, "
            "`passed_with_xfails`, `blocked`, or `not_claims_bearing` for "
            "claim coverage.",
            "",
        ]
    )
    (output_dir / "physics_gate_results.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        help="Run only the named group. May be repeated.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List gate groups and exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest",
        help="Directory for JSON/Markdown and raw stdout/stderr artifacts.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=900,
        help="Per-group timeout in seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--vessl-run-id",
        action="append",
        default=[],
        help=(
            "Record a VESSL run id in the result JSON. Use GROUP=ID for one "
            "group or a bare ID for all selected groups. May be repeated."
        ),
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra pytest args after '--', for example -- -k waveguide.",
    )
    args = parser.parse_args(argv)

    groups = _selected_groups(args.groups)
    if args.list:
        for group in groups:
            print(f"{group.group_id}: {group.description}")
            for test in group.tests:
                print(f"  - {test}")
        return 0

    extra_args = list(args.pytest_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    timeout_s = None if args.timeout_s == 0 else args.timeout_s
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vessl_run_ids = parse_vessl_run_ids(args.vessl_run_id, selected_groups=groups)

    results = [
        run_group(
            group,
            output_dir=output_dir,
            timeout_s=timeout_s,
            extra_pytest_args=extra_args,
            vessl_run_ids=vessl_run_ids[group.group_id],
        )
        for group in groups
    ]
    write_reports(output_dir, results)

    for result in results:
        print(
            f"{result.group_id}: {result.status} "
            f"/ {result.coverage_status} "
            f"({result.duration_s:.1f}s, rc={result.returncode})"
        )
    return 0 if all(result.status == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

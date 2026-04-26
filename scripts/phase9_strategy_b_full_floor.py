"""Phase IX Strategy B full workload-floor coordination harness.

This script coordinates representative full-floor evidence without changing the
existing Phase VII/Phase VIII metadata-only ``--mode full`` behavior.  It writes
schema-distinct readiness and physical artifacts per family plus an optional
Phase IX coordination summary.

Default local execution is conservative: workload-floor rows are artifacted as
honest deferred/not-evaluated outcomes unless ``--execute-workload`` is used and
resource guards allow the run.  This prevents accidental multi-billion
cell-step runs while preserving a real execution path for split runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from scripts import phase7_strategy_b_readiness as phase7  # noqa: E402
from scripts import phase8_strategy_b_physical_validation as phase8  # noqa: E402

SCHEMA_VERSION = 1
PHASE_IX_CONTRACT = "phase_ix_strategy_b_full_floor_execution"
SUMMARY_FILENAME = "phase9_strategy_b_full_floor_summary.json"
FAMILIES = ("source_probe", "cpml_topology", "pec_topology", "port_proxy")
DEFAULT_FAIL_CLOSED_ARTIFACT = Path(
    ".omx/artifacts/phase7_strategy_b_readiness_quick.json"
)
DEFAULT_MAX_CELL_STEPS = 50_000_000
DEFAULT_SPLIT_SOURCE_ARTIFACT = Path(
    ".omx/artifacts/phase7_strategy_b_readiness_quick.json"
)
IMMUTABLE_SOURCE_REF_FIELDS = (
    "path",
    "sha256",
    "source_contract",
    "source_schema_version",
    "source_generated_at",
)

READINESS_ARTIFACT_TEMPLATE = "phase9_{family}_readiness_full.json"
PHYSICAL_ARTIFACT_TEMPLATE = "phase9_{family}_physical_full.json"


@dataclass(frozen=True)
class FloorExecutionResult:
    row_state: str
    execution_status: str
    executed: bool
    reason: str
    runtime_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance must be best-effort in non-git archives.
        return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json_hash(payload: Any) -> str:
    return _sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    )


def worktree_signature() -> dict[str, Any]:
    head = _run_git(["rev-parse", "HEAD"]) or "unknown"
    status = _run_git(["status", "--short"]) or ""
    return {
        "head": head,
        "dirty": bool(status),
        "status_sha256": _sha256_text(status),
    }


def environment_summary() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
    }


def _case_for_family(family: str) -> Any:
    for case in phase7.CASES:
        if case.family == family:
            return case
    raise ValueError(f"unknown family {family!r}")


def _floor_for_family(family: str) -> dict[str, Any]:
    return _case_for_family(family).floor.as_dict()


def _floor_cell_count(family: str) -> int:
    sim = phase7._sim_for_floor(_case_for_family(family))
    return math.prod(phase7._grid_shape(sim))


def artifact_paths(artifact_dir: Path, family: str) -> dict[str, Path]:
    return {
        "readiness": artifact_dir / READINESS_ARTIFACT_TEMPLATE.format(family=family),
        "physical": artifact_dir / PHYSICAL_ARTIFACT_TEMPLATE.format(family=family),
    }


def import_fail_closed_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit = payload.get("fail_closed_audit")
    if not isinstance(audit, dict):
        raise ValueError(f"fail_closed_audit missing from {path}")
    source = {
        "mode": "imported",
        "path": str(path),
        "sha256": file_sha256(path),
        "source_contract": payload.get("benchmark_contract"),
        "source_schema_version": payload.get("schema_version"),
        "source_generated_at": payload.get("generated_at"),
        "row_state": audit.get("row_state"),
        "audit": audit,
    }
    source["id"] = stable_json_hash({k: v for k, v in source.items() if k != "audit"})
    return source


def rerun_fail_closed_evidence() -> dict[str, Any]:
    started = time.perf_counter()
    audit = phase7.build_fail_closed_audit()
    source = {
        "mode": "rerun",
        "path": None,
        "sha256": stable_json_hash(audit),
        "source_contract": phase7.BENCHMARK_CONTRACT,
        "source_schema_version": phase7.SCHEMA_VERSION,
        "source_generated_at": utc_now(),
        "row_state": audit.get("row_state"),
        "runtime_s": round(time.perf_counter() - started, 6),
        "audit": audit,
    }
    source["id"] = stable_json_hash({k: v for k, v in source.items() if k != "audit"})
    return source


def choose_fail_closed_evidence(
    *,
    source: Literal["import", "rerun", "none"],
    artifact: Path = DEFAULT_FAIL_CLOSED_ARTIFACT,
    imported: dict[str, Any] | None = None,
    rerun: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if rerun is not None:
        return rerun
    if imported is not None and source != "rerun":
        return imported
    if source == "rerun":
        return rerun_fail_closed_evidence()
    if source == "import":
        return import_fail_closed_evidence(artifact)
    source_payload = {
        "mode": "none",
        "row_state": "not_evaluated",
        "reason": "fail-closed evidence was not requested",
        "audit": {"row_state": "not_evaluated", "rows": [], "runtime_s": 0.0},
    }
    source_payload["id"] = stable_json_hash(source_payload)
    return source_payload


def fail_closed_pass(fail_closed: dict[str, Any]) -> bool:
    return fail_closed.get("row_state") == "pass"


def source_artifact_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "source_contract": payload.get("benchmark_contract"),
        "source_schema_version": payload.get("schema_version"),
        "source_generated_at": payload.get("generated_at"),
    }


def default_split_source_references(
    path: Path = DEFAULT_SPLIT_SOURCE_ARTIFACT,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "readiness_quick_artifact": {
                "path": str(path),
                "missing": True,
            }
        }
    return {"readiness_quick_artifact": source_artifact_reference(path)}


def _iter_source_refs(refs: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for name, ref in refs.items():
        if isinstance(ref, dict):
            yield name, ref


def validate_split_source_references(refs: dict[str, Any]) -> list[str]:
    """Require immutable artifact identity before split-source promotion."""

    ref_items = list(_iter_source_refs(refs))
    if not ref_items:
        return ["no_source_artifacts"]
    errors: list[str] = []
    for name, ref in ref_items:
        if ref.get("missing") is True:
            errors.append(f"{name}.missing")
            continue
        for field_name in IMMUTABLE_SOURCE_REF_FIELDS:
            if ref.get(field_name) in (None, ""):
                errors.append(f"{name}.{field_name}")
    return errors


def derive_split_source_gradient_state(
    family: str, refs: dict[str, Any]
) -> tuple[str, list[str]]:
    """Derive required gradient state from the referenced source artifact contents."""

    errors = validate_split_source_references(refs)
    if errors:
        return "invalid_source_artifact_provenance", errors
    for name, ref in _iter_source_refs(refs):
        path = Path(str(ref["path"]))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report artifact-read failures as evidence failures.
            errors.append(f"{name}.read_error:{exc.__class__.__name__}")
            continue
        if file_sha256(path) != ref.get("sha256"):
            errors.append(f"{name}.sha256_mismatch")
            continue
        matching_rows = [
            row for row in payload.get("rows", []) if row.get("family") == family
        ]
        if not matching_rows:
            errors.append(f"{name}.family_missing:{family}")
            continue
        for row in matching_rows:
            gradient = row.get("required_gradient_evidence", {})
            if (
                row.get("row_state") == "pass"
                and gradient.get("required") is True
                and gradient.get("state") == "pass"
            ):
                return "pass", []
        errors.append(f"{name}.required_gradient_not_pass:{family}")
    return "not_evaluated", errors


def build_provenance(
    *,
    family: str,
    mode: str,
    command: list[str] | None,
    fail_closed: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    floor = _floor_for_family(family)
    return {
        "phase_ix_schema_version": SCHEMA_VERSION,
        "worktree_signature": worktree_signature(),
        "script": "scripts/phase9_strategy_b_full_floor.py",
        "script_contract": PHASE_IX_CONTRACT,
        "family": family,
        "mode": mode,
        "thresholds": thresholds or {},
        "workload_floor": floor,
        "workload_floor_id": floor["case_id"],
        "n_steps": floor["n_steps"],
        "checkpoint_every": floor["checkpoint_every"],
        "domain_m": floor["domain_m"],
        "dx_m": floor["dx_m"],
        "boundary": floor["boundary"],
        "command": command or sys.argv,
        "generated_at": utc_now(),
        "environment": environment_summary(),
        "fail_closed_source_id": fail_closed.get("id"),
        "fail_closed_source_mode": fail_closed.get("mode"),
        "split_source_artifacts": default_split_source_references(),
    }


def validate_provenance(
    artifact: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    provenance = artifact.get("provenance", {})
    mismatches: list[str] = []
    for key in (
        "family",
        "mode",
        "workload_floor_id",
        "workload_floor",
        "n_steps",
        "checkpoint_every",
        "boundary",
        "domain_m",
        "dx_m",
        "fail_closed_source_id",
    ):
        if provenance.get(key) != expected.get(key):
            mismatches.append(key)
    if provenance.get("thresholds") != expected.get("thresholds"):
        mismatches.append("thresholds")
    for sig_key in ("head", "dirty", "status_sha256"):
        if provenance.get("worktree_signature", {}).get(sig_key) != expected.get(
            "worktree_signature", {}
        ).get(sig_key):
            mismatches.append(f"worktree_signature.{sig_key}")
    if provenance.get("phase_ix_schema_version") != expected.get(
        "phase_ix_schema_version"
    ):
        mismatches.append("schema_version")
    provenance_hash = stable_json_hash(provenance)
    for row in artifact.get("rows", []):
        if (
            row.get("phase_ix_provenance_hash")
            and row.get("phase_ix_provenance_hash") != provenance_hash
        ):
            mismatches.append("provenance_hash")
            break
    return mismatches


def _quarantine_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.stale-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.stale-{stamp}.{counter}")
        counter += 1
    return candidate


def fence_existing_artifacts(
    *,
    paths: dict[str, Path],
    expected: dict[str, Any],
    stale_policy: Literal["reject", "quarantine", "overwrite"] = "quarantine",
) -> dict[str, Any]:
    """Fence stale rerun/resume artifacts before writing replacements.

    Phase IX artifacts are provenance-bearing evidence.  A rerun must therefore
    not silently overwrite artifacts whose floor, threshold, fail-closed source,
    or worktree signature no longer matches the current run.  The default policy
    quarantines stale files, preserving evidence while allowing an intentional
    rerun to continue.
    """

    stale: dict[str, Any] = {}
    for kind, path in paths.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        mismatches = validate_provenance(payload, expected)
        if not mismatches:
            continue
        if stale_policy == "reject":
            raise RuntimeError(f"stale Phase IX {kind} artifact {path}: {mismatches}")
        record: dict[str, Any] = {
            "path": str(path),
            "mismatches": mismatches,
            "policy": stale_policy,
        }
        if stale_policy == "quarantine":
            quarantined = _quarantine_path(path)
            path.rename(quarantined)
            record["quarantined_path"] = str(quarantined)
        stale[kind] = record
    return stale


def _floor_guard_result(
    family: str, *, execute_workload: bool, max_cell_steps: int
) -> FloorExecutionResult | None:
    floor = _floor_for_family(family)
    cell_count = _floor_cell_count(family)
    cell_steps = cell_count * int(floor["n_steps"])
    if family == "pec_topology" and not floor.get("representative", True):
        return FloorExecutionResult(
            row_state="not_evaluated",
            execution_status="defer_limited",
            executed=False,
            reason="PEC topology has no approved representative Phase IX floor; family remains limited by disposition gate",
            metrics={"cell_count": cell_count, "cell_steps": cell_steps},
        )
    if not execute_workload:
        return FloorExecutionResult(
            row_state="not_evaluated",
            execution_status="deferred_requires_explicit_execution",
            executed=False,
            reason="representative workload floor not executed; rerun with --execute-workload and resource guard approval",
            metrics={
                "cell_count": cell_count,
                "cell_steps": cell_steps,
                "max_cell_steps": max_cell_steps,
            },
        )
    if cell_steps > max_cell_steps:
        return FloorExecutionResult(
            row_state="not_evaluated",
            execution_status="deferred_by_resource_guard",
            executed=False,
            reason=f"representative workload has {cell_steps} cell-steps, above local guard {max_cell_steps}",
            metrics={
                "cell_count": cell_count,
                "cell_steps": cell_steps,
                "max_cell_steps": max_cell_steps,
            },
        )
    return None


def _series_metrics(series: Any) -> dict[str, Any]:
    arr = np.asarray(series).reshape(-1)
    return {
        "time_series_shape": list(np.asarray(series).shape),
        "finite": bool(np.all(np.isfinite(arr))),
        "peak_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "observable_energy_proxy": float(np.sum(np.abs(arr.astype(np.float64)) ** 2)),
    }


def _execute_source_probe_floor() -> FloorExecutionResult:
    case = _case_for_family("source_probe")
    sim = phase7._sim_for_floor(case)
    started = time.perf_counter()
    inputs = sim.build_hybrid_phase1_inputs(n_steps=case.floor.n_steps)
    result = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=case.floor.checkpoint_every,
    )
    runtime_s = time.perf_counter() - started
    metrics = _series_metrics(result.time_series)
    cell_count = _floor_cell_count("source_probe")
    strategy_a_gb, strategy_b_gb = phase7._estimate_memory(
        sim,
        n_steps=case.floor.n_steps,
        checkpoint_every=case.floor.checkpoint_every,
        budget_gb=phase7.DEFAULT_BUDGET_GB,
    )
    metrics.update(
        {
            "cell_count": cell_count,
            "cell_steps": cell_count * case.floor.n_steps,
            "strategy_a_estimated_memory_gb": strategy_a_gb,
            "strategy_b_estimated_memory_gb": strategy_b_gb,
        }
    )
    passed = bool(metrics["finite"])
    return FloorExecutionResult(
        row_state="pass" if passed else "fail",
        execution_status="completed" if passed else "failed_nonfinite",
        executed=True,
        reason="representative source/probe Strategy B workload-floor executed"
        if passed
        else "representative source/probe run produced nonfinite output",
        runtime_s=runtime_s,
        metrics=metrics,
    )


def _execute_topology_floor(family: str) -> FloorExecutionResult:
    case = _case_for_family(family)
    floor = case.floor
    sim = phase7._make_source_probe_sim(
        boundary=floor.boundary,
        domain=floor.domain_m,
        freq_max=floor.freq_max_hz,
        dx=floor.dx_m,
        cpml_layers=floor.cpml_layers,
    )
    sim.add_material("phase9_diel", eps_r=4.0, sigma=0.0)
    cx = floor.domain_m[0] * 0.60
    region = phase7.TopologyDesignRegion(
        corner_lo=(cx, floor.domain_m[1] * 0.25, floor.domain_m[2] * 0.25),
        corner_hi=(
            min(cx + 3 * floor.dx_m, floor.domain_m[0] * 0.85),
            floor.domain_m[1] * 0.45,
            floor.domain_m[2] * 0.45,
        ),
        material_bg="air",
        material_fg="phase9_diel",
        beta_projection=1.0,
    )
    started = time.perf_counter()
    result = phase7.topology_optimize(
        sim,
        region,
        phase7._topology_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=floor.n_steps,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=floor.checkpoint_every,
    )
    runtime_s = time.perf_counter() - started
    history = np.asarray(result.history)
    metrics = {
        "history_shape": list(history.shape),
        "history_finite": bool(np.all(np.isfinite(history))),
        "history_last": float(history[-1]) if history.size else None,
        "density_shape": list(np.asarray(result.density).shape),
    }
    cell_count = _floor_cell_count(family)
    strategy_a_gb, strategy_b_gb = phase7._estimate_memory(
        sim,
        n_steps=floor.n_steps,
        checkpoint_every=floor.checkpoint_every,
        budget_gb=phase7.DEFAULT_BUDGET_GB,
    )
    metrics.update(
        {
            "cell_count": cell_count,
            "cell_steps": cell_count * floor.n_steps,
            "strategy_a_estimated_memory_gb": strategy_a_gb,
            "strategy_b_estimated_memory_gb": strategy_b_gb,
        }
    )
    passed = bool(metrics["history_finite"])
    return FloorExecutionResult(
        row_state="pass" if passed else "fail",
        execution_status="completed" if passed else "failed_nonfinite",
        executed=True,
        reason=f"representative {family} Strategy B topology workload-floor executed"
        if passed
        else f"representative {family} topology run produced nonfinite output",
        runtime_s=runtime_s,
        metrics=metrics,
    )


def _execute_port_proxy_floor() -> FloorExecutionResult:
    case = _case_for_family("port_proxy")
    floor = case.floor
    sim = phase7._sim_for_floor(case)
    region = phase7.DesignRegion(
        corner_lo=(
            floor.domain_m[0] * 0.50,
            floor.domain_m[1] * 0.25,
            floor.domain_m[2] * 0.25,
        ),
        corner_hi=(
            floor.domain_m[0] * 0.65,
            floor.domain_m[1] * 0.45,
            floor.domain_m[2] * 0.45,
        ),
        eps_range=(1.0, 4.4),
    )
    started = time.perf_counter()
    inputs = sim.build_hybrid_phase1_inputs(n_steps=floor.n_steps)
    forward_result = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=floor.checkpoint_every,
    )
    result = phase7.optimize(
        sim,
        region,
        phase7._probe_objective,
        n_iters=1,
        lr=0.01,
        n_steps=floor.n_steps,
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=floor.checkpoint_every,
    )
    runtime_s = time.perf_counter() - started
    passive_metrics = _series_metrics(forward_result.time_series)
    excited_power = 0.0
    for raw_source in inputs.raw_sources:
        waveform = np.asarray(raw_source[-1], dtype=np.float64)
        excited_power += float(np.sum(waveform**2))
    passive_power = float(passive_metrics["observable_energy_proxy"])
    passive_ratio = passive_power / max(excited_power, 1e-30)
    loss = np.asarray(result.loss_history)
    metrics = {
        "loss_history_shape": list(loss.shape),
        "loss_history_finite": bool(np.all(np.isfinite(loss))),
        "loss_last": float(loss[-1]) if loss.size else None,
        "passive_load_no_gain_metric": "passive_to_excited_power_ratio",
        "passive_gain_limit": phase8.DEFAULT_PASSIVE_GAIN_LIMIT,
        "excited_power_proxy": excited_power,
        "passive_observable_power_proxy": passive_power,
        "passive_to_excited_power_ratio": passive_ratio,
        "passive_no_gain_pass": bool(passive_metrics["finite"])
        and passive_ratio <= phase8.DEFAULT_PASSIVE_GAIN_LIMIT,
    }
    cell_count = _floor_cell_count("port_proxy")
    strategy_a_gb, strategy_b_gb = phase7._estimate_memory(
        sim,
        n_steps=floor.n_steps,
        checkpoint_every=floor.checkpoint_every,
        budget_gb=phase7.DEFAULT_BUDGET_GB,
    )
    metrics.update(
        {
            "cell_count": cell_count,
            "cell_steps": cell_count * floor.n_steps,
            "strategy_a_estimated_memory_gb": strategy_a_gb,
            "strategy_b_estimated_memory_gb": strategy_b_gb,
        }
    )
    passed = bool(metrics["loss_history_finite"] and metrics["passive_no_gain_pass"])
    return FloorExecutionResult(
        row_state="pass" if passed else "fail",
        execution_status="completed" if passed else "failed_nonfinite",
        executed=True,
        reason="representative port-proxy Strategy B workload-floor executed"
        if passed
        else "representative port-proxy run produced nonfinite output",
        runtime_s=runtime_s,
        metrics=metrics,
    )


def _execute_workload_floor(family: str) -> FloorExecutionResult:
    if family == "source_probe":
        return _execute_source_probe_floor()
    if family in {"cpml_topology", "pec_topology"}:
        return _execute_topology_floor(family)
    if family == "port_proxy":
        return _execute_port_proxy_floor()
    raise ValueError(f"unknown family {family!r}")


def _deferred_result(
    family: str, *, execute_workload: bool, max_cell_steps: int
) -> FloorExecutionResult:
    guard = _floor_guard_result(
        family, execute_workload=execute_workload, max_cell_steps=max_cell_steps
    )
    if guard is not None:
        return guard
    return _execute_workload_floor(family)


def simulated_pass_result(family: str) -> FloorExecutionResult:
    cell_count = _floor_cell_count(family)
    floor = _floor_for_family(family)
    return FloorExecutionResult(
        row_state="pass",
        execution_status="completed",
        executed=True,
        reason="test fixture: representative workload-floor evidence completed",
        runtime_s=1.0,
        metrics={
            "cell_count": cell_count,
            "cell_steps": cell_count * int(floor["n_steps"]),
            "observed_host_rss_peak_mb": 1.0,
            "strategy_b_estimated_memory_gb": 1.0,
            "strategy_a_estimated_memory_gb": 2.0,
        },
    )


def build_readiness_artifact(
    *,
    family: str,
    execution: FloorExecutionResult,
    fail_closed: dict[str, Any],
    provenance: dict[str, Any],
    split_source_gradient_state: str | None = None,
    split_source_refs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case = _case_for_family(family)
    split_source_refs = split_source_refs or default_split_source_references()
    derived_gradient_state, split_source_errors = derive_split_source_gradient_state(
        family, split_source_refs
    )
    if split_source_gradient_state is None:
        effective_gradient_state = derived_gradient_state
    elif split_source_gradient_state == "pass" and derived_gradient_state == "pass":
        effective_gradient_state = "pass"
    elif split_source_gradient_state == "pass":
        effective_gradient_state = derived_gradient_state
    else:
        effective_gradient_state = split_source_gradient_state
    split_source_valid = not split_source_errors and effective_gradient_state == "pass"
    floor = case.floor.as_dict()
    floor_representative = bool(floor.get("representative", True))
    memory_pass = (
        execution.metrics.get("strategy_b_estimated_memory_gb", 0)
        < execution.metrics.get("strategy_a_estimated_memory_gb", math.inf)
        if execution.executed
        else False
    )
    if execution.executed and execution.row_state == "pass" and floor_representative:
        required_gradient_state = (
            effective_gradient_state if split_source_valid else effective_gradient_state
        )
        readiness_status = phase7.classify_family(
            [
                {
                    "mode": "full",
                    "row_state": "pass",
                    "workload_floor": {**floor, "meets_full_floor": True},
                    "required_gradient_evidence": {
                        "required": True,
                        "state": required_gradient_state,
                    },
                }
            ],
            fail_closed_pass=fail_closed_pass(fail_closed),
        )["status"]
    elif execution.row_state == "fail" or not fail_closed_pass(fail_closed):
        required_gradient_state = "not_evaluated"
        readiness_status = "blocked"
    elif family == "pec_topology" and not floor_representative:
        required_gradient_state = "not_evaluated"
        readiness_status = "experimental_limited"
    else:
        required_gradient_state = "not_evaluated"
        readiness_status = "experimental_limited"

    row = {
        "case_id": case.case_id,
        "family": family,
        "mode": "full",
        "boundary": floor["boundary"],
        "objective_family": case.objective_family,
        "n_steps": floor["n_steps"],
        "checkpoint_every": floor["checkpoint_every"],
        "row_state": execution.row_state,
        "reason": execution.reason,
        "workload_floor": {
            **floor,
            "meets_full_floor": bool(execution.executed and floor_representative),
        },
        "full_floor_execution": {
            "executed": execution.executed,
            "status": execution.execution_status,
            "runtime_s": execution.runtime_s,
            "metrics": execution.metrics,
        },
        "required_gradient_evidence": {
            "required": True,
            "state": required_gradient_state,
            "source": case.required_gradient_source,
            "split_source_allowed": True,
            "source_artifacts": split_source_refs,
            "source_artifact_provenance_valid": split_source_valid,
            "source_artifact_provenance_errors": split_source_errors,
        },
        "memory_status": "pass" if memory_pass else "not_evaluated",
        "phase_ix_provenance_hash": stable_json_hash(provenance),
    }
    return {
        "schema_version": phase7.SCHEMA_VERSION,
        "benchmark_contract": phase7.BENCHMARK_CONTRACT,
        "contract_status": "phase_ix_full_floor_readiness_evidence",
        "phase_ix_contract": PHASE_IX_CONTRACT,
        "preserves_phase_vii_metadata_full_mode": True,
        "generated_at": provenance["generated_at"],
        "family": family,
        "provenance": provenance,
        "fail_closed_evidence": fail_closed,
        "rows": [row],
        "summary": {
            "family_status": readiness_status,
            "overall_status": readiness_status,
            "execution_status": execution.execution_status,
            "metadata_only_rows_satisfy_full_floor": False,
            "production_ready_limited_requires_executed_full_floor": True,
        },
    }


def build_physical_artifact(
    *,
    family: str,
    execution: FloorExecutionResult,
    fail_closed: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    case = _case_for_family(family)
    floor = case.floor.as_dict()
    port_proxy_physics_pass = family != "port_proxy" or (
        execution.metrics.get("passive_no_gain_pass") is True
        and execution.metrics.get("passive_to_excited_power_ratio") is not None
    )
    if execution.executed and execution.row_state == "pass" and port_proxy_physics_pass:
        row_state = "pass"
        physical_status = (
            "physics_validated_limited"
            if fail_closed_pass(fail_closed)
            else "physics_blocked"
        )
        evidence_strength = "representative_full"
    elif execution.executed and execution.row_state == "pass":
        row_state = "not_evaluated"
        physical_status = "physics_experimental"
        evidence_strength = "representative_full_missing_physical_oracle"
    elif execution.row_state == "fail" or not fail_closed_pass(fail_closed):
        row_state = "fail"
        physical_status = "physics_blocked"
        evidence_strength = "representative_full"
    else:
        row_state = "not_evaluated"
        physical_status = "physics_experimental"
        evidence_strength = "full_floor_not_evaluated"
    row = {
        "case_id": f"{family}_phase_ix_representative_physics",
        "family": family,
        "mode": "full",
        "boundary": floor["boundary"],
        "objective_family": case.objective_family,
        "oracle_type": "phase_ix_representative_physical_floor",
        "metric_name": "representative_full_floor_execution",
        "comparator": "executed representative physical oracle must pass",
        "tolerance": None,
        "measured_value": None,
        "physical_metrics": execution.metrics,
        "row_state": row_state,
        "reason": (
            execution.reason
            if port_proxy_physics_pass or family != "port_proxy"
            else "port-proxy full-floor execution did not include required passive-load no-gain evidence"
        ),
        "evidence_strength": evidence_strength,
        "raw_parity_supplemental": None,
        "strategy_b_executed": execution.executed,
        "runtime_surface": "Phase IX coordinator over landed Strategy B family",
        "n_steps": floor["n_steps"],
        "checkpoint_every": floor["checkpoint_every"],
        "runtime_s": execution.runtime_s,
        "phase_ix_provenance_hash": stable_json_hash(provenance),
    }
    return {
        "schema_version": phase8.SCHEMA_VERSION,
        "benchmark_contract": phase8.BENCHMARK_CONTRACT,
        "contract_status": "phase_ix_full_floor_physical_evidence",
        "phase_ix_contract": PHASE_IX_CONTRACT,
        "preserves_phase_viii_metadata_full_mode": True,
        "generated_at": provenance["generated_at"],
        "family": family,
        "provenance": provenance,
        "fail_closed_evidence": fail_closed,
        "rows": [row],
        "readiness_integration": {
            "phase_ix_status_is_physical_only": True,
            "does_not_set_production_ready_limited": True,
        },
        "summary": {
            "family_status": physical_status,
            "overall_status": physical_status,
            "execution_status": execution.execution_status,
            "merged_status_taxonomy": False,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_family_artifacts(
    *,
    family: str,
    artifact_dir: Path,
    mode: str = "full",
    fail_closed: dict[str, Any] | None = None,
    fail_closed_source: Literal["import", "rerun", "none"] = "import",
    fail_closed_artifact: Path = DEFAULT_FAIL_CLOSED_ARTIFACT,
    execute_workload: bool = False,
    max_cell_steps: int = DEFAULT_MAX_CELL_STEPS,
    execution: FloorExecutionResult | None = None,
    command: list[str] | None = None,
    stale_policy: Literal["reject", "quarantine", "overwrite"] = "quarantine",
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    fail_closed = fail_closed or choose_fail_closed_evidence(
        source=fail_closed_source,
        artifact=fail_closed_artifact,
    )
    thresholds = {
        "max_cell_steps": max_cell_steps,
        "execute_workload": execute_workload,
    }
    provenance = build_provenance(
        family=family,
        mode=mode,
        command=command,
        fail_closed=fail_closed,
        thresholds=thresholds,
    )
    paths = artifact_paths(artifact_dir, family)
    stale_context = fence_existing_artifacts(
        paths=paths, expected=provenance, stale_policy=stale_policy
    )
    execution = execution or _deferred_result(
        family,
        execute_workload=execute_workload,
        max_cell_steps=max_cell_steps,
    )
    readiness = build_readiness_artifact(
        family=family,
        execution=execution,
        fail_closed=fail_closed,
        provenance=provenance,
    )
    physical = build_physical_artifact(
        family=family,
        execution=execution,
        fail_closed=fail_closed,
        provenance=provenance,
    )
    write_json(paths["readiness"], readiness)
    write_json(paths["physical"], physical)
    return {
        "readiness": readiness,
        "physical": physical,
        "paths": {k: str(v) for k, v in paths.items()},
        "stale_context": stale_context,
    }


def _artifact_file_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in paths if path.exists()}


def summarize_artifacts(
    *, artifact_dir: Path, output: Path | None = None, allow_stale: bool = False
) -> dict[str, Any]:
    readiness_artifacts: dict[str, Any] = {}
    physical_artifacts: dict[str, Any] = {}
    stale_context: dict[str, Any] = {}
    fail_closed_sources: dict[str, Any] = {}
    for family in FAMILIES:
        paths = artifact_paths(artifact_dir, family)
        if paths["readiness"].exists():
            payload = json.loads(paths["readiness"].read_text(encoding="utf-8"))
            expected = build_provenance(
                family=family,
                mode=payload.get("provenance", {}).get("mode", "full"),
                command=payload.get("provenance", {}).get("command"),
                fail_closed=payload.get("fail_closed_evidence", {}),
                thresholds=payload.get("provenance", {}).get("thresholds", {}),
            )
            mismatches = validate_provenance(payload, expected)
            if mismatches and not allow_stale:
                stale_context[f"{family}:readiness"] = mismatches
            readiness_artifacts[family] = {
                "path": str(paths["readiness"]),
                "sha256": file_sha256(paths["readiness"]),
                "status": payload.get("summary", {}).get("family_status"),
                "execution_status": payload.get("summary", {}).get("execution_status"),
            }
            source = payload.get("fail_closed_evidence", {})
            if source.get("id"):
                fail_closed_sources[source["id"]] = {
                    k: source.get(k) for k in ("mode", "row_state", "path", "sha256")
                }
        if paths["physical"].exists():
            payload = json.loads(paths["physical"].read_text(encoding="utf-8"))
            expected = build_provenance(
                family=family,
                mode=payload.get("provenance", {}).get("mode", "full"),
                command=payload.get("provenance", {}).get("command"),
                fail_closed=payload.get("fail_closed_evidence", {}),
                thresholds=payload.get("provenance", {}).get("thresholds", {}),
            )
            mismatches = validate_provenance(payload, expected)
            if mismatches and not allow_stale:
                stale_context[f"{family}:physical"] = mismatches
            physical_artifacts[family] = {
                "path": str(paths["physical"]),
                "sha256": file_sha256(paths["physical"]),
                "status": payload.get("summary", {}).get("family_status"),
                "execution_status": payload.get("summary", {}).get("execution_status"),
            }
            source = payload.get("fail_closed_evidence", {})
            if source.get("id"):
                fail_closed_sources[source["id"]] = {
                    k: source.get(k) for k in ("mode", "row_state", "path", "sha256")
                }
    summary_paths = [Path(v["path"]) for v in readiness_artifacts.values()] + [
        Path(v["path"]) for v in physical_artifacts.values()
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_ix_contract": PHASE_IX_CONTRACT,
        "contract_status": "phase_ix_coordination_summary",
        "generated_at": utc_now(),
        "merged_status_taxonomy": False,
        "readiness_artifacts": readiness_artifacts,
        "physical_artifacts": physical_artifacts,
        "artifact_hashes": _artifact_file_hashes(summary_paths),
        "fail_closed_evidence": {
            "sources": fail_closed_sources,
            "all_known_sources_pass": bool(fail_closed_sources)
            and all(
                src.get("row_state") == "pass" for src in fail_closed_sources.values()
            ),
        },
        "stale_context": stale_context,
        "summary": {
            "families_with_readiness_artifacts": sorted(readiness_artifacts),
            "families_with_physical_artifacts": sorted(physical_artifacts),
            "has_stale_context": bool(stale_context),
            "next_decision": "fix_or_collect_more_evidence"
            if stale_context
            else "review_family_outcomes",
        },
    }
    if output is not None:
        write_json(output, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("all", *FAMILIES), default="all")
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--artifact-dir", type=Path, default=Path(".omx/artifacts"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize existing Phase IX artifacts instead of running a family.",
    )
    parser.add_argument(
        "--execute-workload",
        action="store_true",
        help="Attempt representative execution subject to resource guard.",
    )
    parser.add_argument("--max-cell-steps", type=int, default=DEFAULT_MAX_CELL_STEPS)
    parser.add_argument(
        "--fail-closed-source", choices=("import", "rerun", "none"), default="import"
    )
    parser.add_argument(
        "--fail-closed-artifact", type=Path, default=DEFAULT_FAIL_CLOSED_ARTIFACT
    )
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument(
        "--stale-policy",
        choices=("reject", "quarantine", "overwrite"),
        default="quarantine",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.summarize:
        output = args.output or args.artifact_dir / SUMMARY_FILENAME
        payload = summarize_artifacts(
            artifact_dir=args.artifact_dir, output=output, allow_stale=args.allow_stale
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    families = FAMILIES if args.family == "all" else (args.family,)
    results: dict[str, Any] = {}
    for family in families:
        results[family] = build_family_artifacts(
            family=family,
            artifact_dir=args.artifact_dir,
            mode=args.mode,
            fail_closed_source=args.fail_closed_source,
            fail_closed_artifact=args.fail_closed_artifact,
            execute_workload=args.execute_workload,
            max_cell_steps=args.max_cell_steps,
            command=sys.argv,
            stale_policy="overwrite" if args.allow_stale else args.stale_policy,
        )
    summary_path = args.output or args.artifact_dir / SUMMARY_FILENAME
    summary = summarize_artifacts(
        artifact_dir=args.artifact_dir,
        output=summary_path,
        allow_stale=args.allow_stale,
    )
    print(
        json.dumps(
            {"families": sorted(results), "summary": summary}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()

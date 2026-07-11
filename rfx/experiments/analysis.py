"""Cited, limitation-aware RF analysis over immutable run artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def analyze_sparameters(service, run_id: str) -> dict[str, Any]:
    linked = service.application_repository.get_linked_run(run_id)
    revision = service.application_repository.get_revision(linked.revision_id)
    artifact = next(
        (
            item
            for item in service.application_repository.list_artifacts(run_id)
            if item.kind in {"s11", "sparameters"}
        ),
        None,
    )
    if artifact is None:
        raise ValueError(f"run {run_id} has no S-parameter artifact")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    if artifact.kind == "s11":
        points = payload["points"]
        s21_values = None
        observable = "S11"
    else:
        points = [
            {"frequency_hz": item["frequency_hz"], **item["matrix"][0][0]}
            for item in payload["points"]
        ]
        s21_values = (
            [
                math.hypot(item["matrix"][1][0]["real"], item["matrix"][1][0]["imag"])
                for item in payload["points"]
            ]
            if len(payload["port_names"]) > 1
            else None
        )
        observable = "full S-matrix"
    minimum = min(points, key=lambda item: item["magnitude_db"])
    max_abs = max(math.hypot(item["real"], item["imag"]) for item in points)
    definitions = {
        "minimum_s11_db": "minimum sampled 20*log10(abs(S11))",
        "minimum_s11_frequency_hz": "frequency at the minimum sampled S11 magnitude",
        "max_s11_abs": "maximum sampled complex reflection magnitude",
    }
    metrics = {
        "minimum_s11_db": minimum["magnitude_db"],
        "minimum_s11_frequency_hz": minimum["frequency_hz"],
        "max_s11_abs": max_abs,
    }
    if s21_values is not None:
        definitions["min_s21_abs"] = "minimum sampled forward transmission magnitude"
        metrics["min_s21_abs"] = min(s21_values)
    return {
        "schema_version": "rfx-run-analysis/v1",
        "analysis_kind": "sparameters",
        "input_run_ids": [run_id],
        "revision_ids": [linked.revision_id],
        "metric_definition": definitions,
        "metrics": metrics,
        "validation_status": revision.validation_state,
        "support_lane": revision.spec["validation"]["support_lane"],
        "limitations": _limitations(revision.spec),
        "citations": [
            {
                "run_id": run_id,
                "artifact_id": artifact.id,
                "sha256": artifact.sha256,
                "observable": observable,
            }
        ],
    }


def analyze_reflection_transmission(service, run_id: str) -> dict[str, Any]:
    linked = service.application_repository.get_linked_run(run_id)
    revision = service.application_repository.get_revision(linked.revision_id)
    artifact = next(
        (
            item
            for item in service.application_repository.list_artifacts(run_id)
            if item.kind == "reflection-transmission"
        ),
        None,
    )
    if artifact is None:
        raise ValueError(f"run {run_id} has no reflection/transmission artifact")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    points = [item for item in payload["points"] if item["signal_valid"]]
    signal_limitation = []
    if not points:
        points = payload["points"]
        signal_limitation.append(
            "No frequency bin passed the structural fixture signal-floor witness; metrics are diagnostic only."
        )
    reflection_errors = [
        abs(item["reflection"] - item["analytic_reflection"]) for item in points
    ]
    transmission_errors = [
        abs(item["transmission"] - item["analytic_transmission"]) for item in points
    ]
    closure_errors = [
        abs(item["reflection"] + item["transmission"] - 1.0) for item in points
    ]
    return {
        "schema_version": "rfx-run-analysis/v1",
        "analysis_kind": "reflection-transmission",
        "input_run_ids": [run_id],
        "revision_ids": [linked.revision_id],
        "metric_definition": {
            "mean_reflectance_error": "mean abs(R_fdfd - R_exact_transfer_matrix)",
            "mean_transmittance_error": "mean abs(T_fdtd - T_exact_transfer_matrix)",
            "mean_energy_closure_error": "mean abs(R_fdtd + T_fdtd - 1)",
        },
        "metrics": {
            "mean_reflectance_error": sum(reflection_errors) / len(reflection_errors),
            "mean_transmittance_error": sum(transmission_errors)
            / len(transmission_errors),
            "mean_energy_closure_error": sum(closure_errors) / len(closure_errors),
        },
        "validation_status": revision.validation_state,
        "support_lane": revision.spec["validation"]["support_lane"],
        "limitations": _limitations(revision.spec) + signal_limitation,
        "citations": [
            {
                "run_id": run_id,
                "artifact_id": artifact.id,
                "sha256": artifact.sha256,
                "observable": "reflection/transmission and exact transfer-matrix reference",
            }
        ],
    }


def analyze_run(service, run_id: str) -> dict[str, Any]:
    kinds = {
        item.kind for item in service.application_repository.list_artifacts(run_id)
    }
    if "reflection-transmission" in kinds:
        return analyze_reflection_transmission(service, run_id)
    return analyze_sparameters(service, run_id)


def compare_sparameter_runs(service, run_ids: Iterable[str]) -> dict[str, Any]:
    identifiers = list(dict.fromkeys(run_ids))
    if len(identifiers) < 2 or len(identifiers) > 20:
        raise ValueError("comparison requires 2..20 unique run ids")
    analyses = [analyze_sparameters(service, run_id) for run_id in identifiers]
    fields = [_field_evidence(service, run_id) for run_id in identifiers]
    baseline = analyses[0]
    baseline_field = fields[0]
    rows = []
    for analysis, field in zip(analyses, fields):
        metrics = analysis["metrics"]
        field_delta = None
        if baseline_field is not None and field is not None:
            if field["values"].shape != baseline_field["values"].shape:
                raise ValueError("field comparison requires matching plane shapes")
            denominator = max(float(np.linalg.norm(baseline_field["values"])), 1e-30)
            field_delta = float(
                np.linalg.norm(field["values"] - baseline_field["values"]) / denominator
            )
        rows.append(
            {
                "run_id": analysis["input_run_ids"][0],
                "revision_id": analysis["revision_ids"][0],
                **metrics,
                "delta_minimum_s11_db_vs_baseline": metrics["minimum_s11_db"]
                - baseline["metrics"]["minimum_s11_db"],
                "delta_resonance_hz_vs_baseline": metrics["minimum_s11_frequency_hz"]
                - baseline["metrics"]["minimum_s11_frequency_hz"],
                "maximum_field_abs": (
                    field["maximum_absolute"] if field is not None else None
                ),
                "field_normalized_l2_delta_vs_baseline": field_delta,
                "validation_status": analysis["validation_status"],
            }
        )
    metric_definition = {
        **baseline["metric_definition"],
        "maximum_field_abs": "maximum absolute value in the cited final-state field plane",
        "field_normalized_l2_delta_vs_baseline": "L2(field - baseline) / max(L2(baseline), 1e-30) on matching persisted planes",
    }
    return {
        "schema_version": "rfx-run-comparison/v1",
        "analysis_kind": "sparameter-comparison",
        "input_run_ids": identifiers,
        "baseline_run_id": identifiers[0],
        "metric_definition": metric_definition,
        "rows": rows,
        "validation_status": {
            item["input_run_ids"][0]: item["validation_status"] for item in analyses
        },
        "limitations": sorted(
            {limitation for item in analyses for limitation in item["limitations"]}
        ),
        "citations": [
            *[citation for item in analyses for citation in item["citations"]],
            *[
                {
                    "run_id": run_id,
                    "artifact_id": field["artifact_id"],
                    "sha256": field["sha256"],
                    "observable": "final-state field plane",
                }
                for run_id, field in zip(identifiers, fields)
                if field is not None
            ],
        ],
    }


def explain_validation(service, run_id: str) -> dict[str, Any]:
    linked = service.application_repository.get_linked_run(run_id)
    revision = service.application_repository.get_revision(linked.revision_id)
    return {
        "schema_version": "rfx-validation-explanation/v1",
        "input_run_ids": [run_id],
        "revision_ids": [revision.id],
        "validation_status": revision.validation_state,
        "support_lane": revision.spec["validation"]["support_lane"],
        "required_checks": revision.spec["validation"]["required_checks"],
        "declared_metrics": revision.spec["validation"]["metrics"],
        "preflight": revision.preflight,
        "limitations": _limitations(revision.spec),
    }


def _limitations(spec: dict[str, Any]) -> list[str]:
    limitations = []
    claim = spec["metadata"].get("claims")
    fidelity = spec["metadata"].get("fidelity")
    if claim:
        limitations.append(str(claim))
    if fidelity == "structural-cpu-smoke":
        limitations.append(
            "The small CPU fixture establishes lifecycle behavior, not converged quantitative RF accuracy."
        )
    limitations.append(
        "Metrics describe only declared, persisted observables for the cited immutable revision."
    )
    return limitations


def _field_evidence(service, run_id: str) -> dict[str, Any] | None:
    artifact = next(
        (
            item
            for item in service.application_repository.list_artifacts(run_id)
            if item.kind == "field-slice"
        ),
        None,
    )
    if artifact is None:
        return None
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    values = np.asarray(payload["values"], dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("field comparison requires a finite 2-D artifact")
    return {
        "artifact_id": artifact.id,
        "sha256": artifact.sha256,
        "maximum_absolute": float(payload["maximum_absolute"]),
        "values": values,
    }

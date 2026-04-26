"""Phase VIII Strategy B physical-validation evidence harness.

This harness is deliberately separate from Phase VII readiness reporting.  It
checks that currently landed Strategy B families produce physically plausible
observables against explicit electromagnetic sanity oracles, while preserving
Phase VII's fail-closed boundaries and avoiding production-readiness promotion.

Run:
    python scripts/phase8_strategy_b_physical_validation.py --mode quick --indent 2
    python scripts/phase8_strategy_b_physical_validation.py --mode full --family source_probe --indent 2
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import jax  # noqa: E402
import numpy as np  # noqa: E402

from rfx.grid import C0  # noqa: E402
from rfx.topology import _inspect_topology_hybrid_support  # noqa: E402

from scripts import phase7_strategy_b_readiness as phase7  # noqa: E402

SCHEMA_VERSION = 1
BENCHMARK_CONTRACT = "phase_viii_strategy_b_physical_validation"
CONTRACT_STATUS = "phase_viii_physical_validation_evidence"
BASELINE_COMMIT = phase7.BASELINE_COMMIT
QUICK_N_STEPS = 32
QUICK_CHECKPOINT_EVERY = 4
TOPOLOGY_PHYSICAL_N_STEPS = 384
TOPOLOGY_PHYSICAL_CHECKPOINT_EVERY = 16
TOPOLOGY_SOURCE_OFF_THRESHOLD_RATIO = 1e-3
TOPOLOGY_MIN_POST_SOURCE_SAMPLES = 32
DEFAULT_CAUSAL_THRESHOLD_RATIO = 0.05
DEFAULT_CPML_TAIL_GROWTH_LIMIT = 1.10
DEFAULT_PEC_ENERGY_GROWTH_LIMIT = 1.10
DEFAULT_PASSIVE_GAIN_LIMIT = 1.25
DEFAULT_PHYSICAL_PARITY_THRESHOLD = 1e-5

ROW_STATES = ("pass", "warn", "fail", "not_evaluated")
FAMILY_STATUSES = (
    "physics_validated_limited",
    "physics_experimental",
    "physics_blocked",
    "not_evaluated",
)
FAMILIES = ("source_probe", "cpml_topology", "pec_topology", "port_proxy")

REQUIRED_ROW_FIELDS = (
    "case_id",
    "family",
    "mode",
    "boundary",
    "objective_family",
    "oracle_type",
    "metric_name",
    "comparator",
    "tolerance",
    "measured_value",
    "physical_metrics",
    "row_state",
    "reason",
    "evidence_strength",
    "raw_parity_supplemental",
    "strategy_b_executed",
    "runtime_surface",
    "n_steps",
    "checkpoint_every",
    "runtime_s",
)


@dataclass(frozen=True)
class Thresholds:
    causal_threshold_ratio: float = DEFAULT_CAUSAL_THRESHOLD_RATIO
    cpml_tail_growth_limit: float = DEFAULT_CPML_TAIL_GROWTH_LIMIT
    pec_energy_growth_limit: float = DEFAULT_PEC_ENERGY_GROWTH_LIMIT
    passive_gain_limit: float = DEFAULT_PASSIVE_GAIN_LIMIT
    physical_parity: float = DEFAULT_PHYSICAL_PARITY_THRESHOLD


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _finite_series_metrics(series: np.ndarray) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(series)))
    abs_series = np.abs(series.astype(np.float64, copy=False))
    energy = float(np.sum(abs_series**2))
    peak = float(np.max(abs_series)) if abs_series.size else 0.0
    return {
        "finite": finite,
        "sample_count": int(abs_series.size),
        "peak_abs": peak,
        "observable_energy_proxy": energy,
    }


def _source_probe_distance_m(sim: Any) -> float:
    if not sim._ports or not sim._probes:
        return math.nan
    src = sim._ports[0].position
    probe = sim._probes[0].position
    return float(math.dist(src, probe))


def _run_source_probe_quick(thresholds: Thresholds) -> dict[str, Any]:
    sim = phase7._make_source_probe_sim(boundary="cpml")
    inputs = sim.build_hybrid_phase1_inputs(n_steps=QUICK_N_STEPS)
    before = _rss_mb()
    started = time.perf_counter()
    result = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )
    runtime_s = time.perf_counter() - started
    after = _rss_mb()

    series = np.asarray(result.time_series).reshape(-1)
    finite_metrics = _finite_series_metrics(series)
    dt = float(inputs.grid.dt)
    distance_m = _source_probe_distance_m(sim)
    travel_time_s = float(distance_m / C0)
    threshold_abs = max(
        finite_metrics["peak_abs"] * thresholds.causal_threshold_ratio,
        1e-15,
    )
    significant = np.flatnonzero(np.abs(series) >= threshold_abs)
    first_index = int(significant[0]) if significant.size else None
    peak_index = int(np.argmax(np.abs(series))) if series.size else None
    first_time_s = None if first_index is None else float(first_index * dt)
    peak_time_s = None if peak_index is None else float(peak_index * dt)
    lower_bound_s = travel_time_s - 2.0 * dt
    causal_margin_steps = (
        None if first_time_s is None else float((first_time_s - lower_bound_s) / dt)
    )
    causal_pass = (
        bool(finite_metrics["finite"])
        and first_time_s is not None
        and first_time_s >= lower_bound_s
    )
    row_state = "pass" if causal_pass else "fail"
    return {
        "case_id": "source_probe_causal_propagation_strategy_b_physics",
        "family": "source_probe",
        "mode": "quick",
        "boundary": "cpml",
        "objective_family": "source_probe",
        "oracle_type": "causal_propagation_and_bounded_observable",
        "metric_name": "causal_margin_steps",
        "comparator": ">= 0 after distance/C0 - 2*dt lower bound",
        "tolerance": {
            "causal_threshold_ratio": thresholds.causal_threshold_ratio,
            "early_arrival_slack_steps": 2.0,
        },
        "measured_value": None if causal_margin_steps is None else round(causal_margin_steps, 6),
        "physical_metrics": {
            **finite_metrics,
            "dt_s": dt,
            "source_probe_distance_m": distance_m,
            "c0_travel_time_s": travel_time_s,
            "earliest_allowed_response_s": lower_bound_s,
            "first_significant_response_index": first_index,
            "first_significant_response_s": first_time_s,
            "peak_response_index": peak_index,
            "peak_response_s": peak_time_s,
            "significance_threshold_abs": threshold_abs,
            "causal_margin_steps": causal_margin_steps,
        },
        "row_state": row_state,
        "reason": (
            "Strategy B probe response is finite and not earlier than the causal lower bound"
            if row_state == "pass"
            else "Strategy B probe response is nonfinite or violates the causal lower bound"
        ),
        "evidence_strength": "quick_physical_oracle",
        "raw_parity_supplemental": None,
        "strategy_b_executed": True,
        "runtime_surface": "Simulation.forward_hybrid_phase1_from_inputs(..., strategy='b')",
        "n_steps": QUICK_N_STEPS,
        "checkpoint_every": QUICK_CHECKPOINT_EVERY,
        "runtime_s": round(runtime_s, 6),
        "observed_host_rss_delta_mb": round(max(0.0, after - before), 3),
        "observed_host_rss_peak_mb": round(after, 3),
    }


def _quarter_energy_ratios(series: np.ndarray) -> dict[str, Any]:
    energy = np.abs(series.astype(np.float64, copy=False)) ** 2
    if energy.size == 0:
        return {
            "quarter_energy": [],
            "tail_to_total_energy_ratio": math.nan,
            "tail_to_previous_quarter_ratio": math.nan,
        }
    quarters = [float(np.sum(chunk)) for chunk in np.array_split(energy, 4)]
    total = float(np.sum(energy))
    tail = quarters[-1]
    previous = quarters[-2] if len(quarters) > 1 else 0.0
    return {
        "quarter_energy": quarters,
        "tail_to_total_energy_ratio": tail / max(total, 1e-30),
        "tail_to_previous_quarter_ratio": tail / max(previous, 1e-30),
    }


def _source_envelope_from_inputs(inputs: Any, sample_count: int) -> np.ndarray:
    envelope = np.zeros(sample_count, dtype=np.float64)
    for raw_source in inputs.raw_sources:
        waveform = np.asarray(raw_source[-1], dtype=np.float64).reshape(-1)
        width = min(sample_count, waveform.size)
        envelope[:width] = np.maximum(envelope[:width], np.abs(waveform[:width]))
    return envelope


def _post_source_window(series: np.ndarray, inputs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    source_envelope = _source_envelope_from_inputs(inputs, series.size)
    source_peak = float(np.max(source_envelope)) if source_envelope.size else 0.0
    threshold_abs = source_peak * TOPOLOGY_SOURCE_OFF_THRESHOLD_RATIO
    active = np.flatnonzero(source_envelope > threshold_abs) if threshold_abs > 0.0 else np.array([], dtype=int)
    source_active_last_index = int(active[-1]) if active.size else -1
    start = min(source_active_last_index + 1, series.size)
    graded = series[start:]
    tail_source_peak = float(np.max(source_envelope[start:])) if start < source_envelope.size else 0.0
    verified = bool(
        graded.size >= TOPOLOGY_MIN_POST_SOURCE_SAMPLES
        and (threshold_abs == 0.0 or tail_source_peak <= threshold_abs)
    )
    return graded, {
        "full_sample_count": int(series.size),
        "graded_sample_count": int(graded.size),
        "source_peak_abs": source_peak,
        "source_off_threshold_ratio": TOPOLOGY_SOURCE_OFF_THRESHOLD_RATIO,
        "source_off_threshold_abs": threshold_abs,
        "source_active_last_index": source_active_last_index,
        "graded_window_start_index": int(start),
        "max_source_abs_in_graded_window": tail_source_peak,
        "min_post_source_samples": TOPOLOGY_MIN_POST_SOURCE_SAMPLES,
        "source_off_window_verified": verified,
    }


def _topology_strategy_b_series(boundary: str) -> tuple[Any, np.ndarray, float, dict[str, Any]]:
    sim, region = phase7._make_topology_case(boundary)
    inputs, report, *_ = _inspect_topology_hybrid_support(
        sim,
        region,
        n_steps=TOPOLOGY_PHYSICAL_N_STEPS,
    )
    if not report.supported or inputs.materials is None:
        raise RuntimeError(report.reason_text)
    started = time.perf_counter()
    result = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=TOPOLOGY_PHYSICAL_CHECKPOINT_EVERY,
    )
    runtime_s = time.perf_counter() - started
    full_series = np.asarray(result.time_series).reshape(-1)
    graded_series, window_metrics = _post_source_window(full_series, inputs)
    return sim, graded_series, runtime_s, window_metrics


def _run_cpml_topology_quick(thresholds: Thresholds) -> dict[str, Any]:
    before = _rss_mb()
    topo_started = time.perf_counter()
    phase7._run_topology_quick("cpml", phase7.Thresholds())
    topology_runtime_s = time.perf_counter() - topo_started
    sim, series, forward_runtime_s, window_metrics = _topology_strategy_b_series("cpml")
    after = _rss_mb()

    finite_metrics = _finite_series_metrics(series)
    ratio_metrics = _quarter_energy_ratios(series)
    tail_ratio = float(ratio_metrics["tail_to_previous_quarter_ratio"])
    metric_pass = (
        bool(finite_metrics["finite"])
        and bool(window_metrics["source_off_window_verified"])
        and tail_ratio <= thresholds.cpml_tail_growth_limit
    )
    row_state = "pass" if metric_pass else "fail"
    return {
        "case_id": "cpml_topology_tail_energy_strategy_b_physics",
        "family": "cpml_topology",
        "mode": "quick",
        "boundary": "cpml",
        "objective_family": "cpml_topology_probe_energy",
        "oracle_type": "cpml_tail_energy_absorption_proxy",
        "metric_name": "tail_to_previous_quarter_ratio",
        "comparator": "<= cpml_tail_growth_limit",
        "tolerance": {"cpml_tail_growth_limit": thresholds.cpml_tail_growth_limit},
        "measured_value": round(tail_ratio, 12),
        "physical_metrics": {
            **finite_metrics,
            **ratio_metrics,
            "dt_s": float(sim.build_hybrid_phase1_inputs(n_steps=1).grid.dt),
            "post_source_window": window_metrics,
            "executed_topology_optimize_strategy_b": True,
            "topology_optimize_runtime_s": round(topology_runtime_s, 6),
        },
        "row_state": row_state,
        "reason": (
            "CPML Strategy B observable is finite and non-growing over a verified post-source quick window"
            if row_state == "pass"
            else "CPML Strategy B observable is nonfinite, lacks a verified post-source window, or exceeds the quick tail-energy bound"
        ),
        "evidence_strength": "quick_physical_oracle",
        "raw_parity_supplemental": {
            "topology_optimize_strategy_b_executed": True,
            "parity_source": "Phase VII topology quick parity/gradient helper executed for this row",
        },
        "strategy_b_executed": True,
        "runtime_surface": "topology_optimize(..., adjoint_mode='hybrid', strategy='b') + Strategy B forward replay",
        "n_steps": TOPOLOGY_PHYSICAL_N_STEPS,
        "checkpoint_every": TOPOLOGY_PHYSICAL_CHECKPOINT_EVERY,
        "runtime_s": round(topology_runtime_s + forward_runtime_s, 6),
        "observed_host_rss_delta_mb": round(max(0.0, after - before), 3),
        "observed_host_rss_peak_mb": round(after, 3),
    }


def _run_pec_topology_quick(thresholds: Thresholds) -> dict[str, Any]:
    before = _rss_mb()
    topo_started = time.perf_counter()
    phase7._run_topology_quick("pec", phase7.Thresholds())
    topology_runtime_s = time.perf_counter() - topo_started
    _sim, series, forward_runtime_s, window_metrics = _topology_strategy_b_series("pec")
    after = _rss_mb()

    finite_metrics = _finite_series_metrics(series)
    ratio_metrics = _quarter_energy_ratios(series)
    observable_growth_ratio = float(ratio_metrics["tail_to_previous_quarter_ratio"])
    metric_pass = (
        bool(finite_metrics["finite"])
        and bool(window_metrics["source_off_window_verified"])
        and observable_growth_ratio <= thresholds.pec_energy_growth_limit
    )
    row_state = "pass" if metric_pass else "fail"
    return {
        "case_id": "pec_topology_bounded_energy_strategy_b_physics",
        "family": "pec_topology",
        "mode": "quick",
        "boundary": "pec",
        "objective_family": "pec_topology_probe_energy",
        "oracle_type": "pec_lossless_bounded_observable_proxy",
        "metric_name": "tail_to_previous_quarter_ratio",
        "comparator": "<= pec_energy_growth_limit",
        "tolerance": {"pec_energy_growth_limit": thresholds.pec_energy_growth_limit},
        "measured_value": round(observable_growth_ratio, 12),
        "physical_metrics": {
            **finite_metrics,
            **ratio_metrics,
            "post_source_window": window_metrics,
            "executed_topology_optimize_strategy_b": True,
            "topology_optimize_runtime_s": round(topology_runtime_s, 6),
        },
        "row_state": row_state,
        "reason": (
            "PEC Strategy B observable is finite and bounded over a verified post-source quick window; representative PEC floor remains undefined"
            if row_state == "pass"
            else "PEC Strategy B observable is nonfinite, lacks a verified post-source window, or exceeds the quick bounded-energy proxy"
        ),
        "evidence_strength": "quick_physical_oracle",
        "raw_parity_supplemental": {
            "topology_optimize_strategy_b_executed": True,
            "parity_source": "Phase VII topology quick parity/gradient helper executed for this row",
        },
        "strategy_b_executed": True,
        "runtime_surface": "topology_optimize(..., adjoint_mode='hybrid', strategy='b') + Strategy B forward replay",
        "n_steps": TOPOLOGY_PHYSICAL_N_STEPS,
        "checkpoint_every": TOPOLOGY_PHYSICAL_CHECKPOINT_EVERY,
        "runtime_s": round(topology_runtime_s + forward_runtime_s, 6),
        "observed_host_rss_delta_mb": round(max(0.0, after - before), 3),
        "observed_host_rss_peak_mb": round(after, 3),
    }


def _run_port_proxy_quick(thresholds: Thresholds) -> dict[str, Any]:
    sim, _region = phase7._make_port_proxy_case()
    before = _rss_mb()
    opt_started = time.perf_counter()
    phase7._run_port_proxy_quick(phase7.Thresholds())
    optimize_runtime_s = time.perf_counter() - opt_started
    inputs = sim.build_hybrid_phase1_inputs(n_steps=QUICK_N_STEPS)
    forward_started = time.perf_counter()
    result = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )
    forward_runtime_s = time.perf_counter() - forward_started
    after = _rss_mb()

    series = np.asarray(result.time_series).reshape(-1)
    finite_metrics = _finite_series_metrics(series)
    excited_power = 0.0
    for raw_source in inputs.raw_sources:
        waveform = np.asarray(raw_source[-1], dtype=np.float64)
        excited_power += float(np.sum(waveform**2))
    passive_power = float(finite_metrics["observable_energy_proxy"])
    ratio = passive_power / max(excited_power, 1e-30)
    metric_pass = bool(finite_metrics["finite"]) and ratio <= thresholds.passive_gain_limit
    row_state = "warn" if metric_pass else "fail"
    return {
        "case_id": "one_passive_port_no_gain_strategy_b_physics",
        "family": "port_proxy",
        "mode": "quick",
        "boundary": "pec",
        "objective_family": "one_excited_one_passive_port_proxy",
        "oracle_type": "passive_load_no_gain_proxy",
        "metric_name": "passive_to_excited_power_ratio",
        "comparator": "<= passive_gain_limit",
        "tolerance": {"passive_gain_limit": thresholds.passive_gain_limit},
        "measured_value": round(ratio, 12),
        "physical_metrics": {
            **finite_metrics,
            "excited_power_proxy": excited_power,
            "passive_observable_power_proxy": passive_power,
            "passive_to_excited_power_ratio": ratio,
            "excited_lumped_port_count": sum(1 for port in sim._ports if port.excite),
            "passive_lumped_port_count": sum(1 for port in sim._ports if not port.excite),
            "executed_optimize_strategy_b": True,
            "optimize_runtime_s": round(optimize_runtime_s, 6),
        },
        "row_state": row_state,
        "reason": (
            "one-excited/one-passive Strategy B proxy is finite and does not show passive-load gain; this is not generic multi-port validation"
            if row_state == "warn"
            else "one-excited/one-passive Strategy B proxy is nonfinite or exceeds the passive no-gain limit"
        ),
        "evidence_strength": "quick_proxy",
        "raw_parity_supplemental": {
            "optimize_strategy_b_executed": True,
            "parity_source": "Phase VII port-proxy quick parity/gradient helper executed for this row",
        },
        "strategy_b_executed": True,
        "runtime_surface": "optimize(..., adjoint_mode='hybrid', strategy='b') + Strategy B forward replay",
        "n_steps": QUICK_N_STEPS,
        "checkpoint_every": QUICK_CHECKPOINT_EVERY,
        "runtime_s": round(optimize_runtime_s + forward_runtime_s, 6),
        "observed_host_rss_delta_mb": round(max(0.0, after - before), 3),
        "observed_host_rss_peak_mb": round(after, 3),
    }


def _build_quick_row(family: str, thresholds: Thresholds) -> dict[str, Any]:
    if family == "source_probe":
        return _run_source_probe_quick(thresholds)
    if family == "cpml_topology":
        return _run_cpml_topology_quick(thresholds)
    if family == "pec_topology":
        return _run_pec_topology_quick(thresholds)
    if family == "port_proxy":
        return _run_port_proxy_quick(thresholds)
    raise ValueError(f"unknown family {family!r}")


def _build_full_row(family: str, thresholds: Thresholds) -> dict[str, Any]:
    del thresholds
    boundary = "cpml" if family in {"source_probe", "cpml_topology"} else "pec"
    objective = {
        "source_probe": "source_probe",
        "cpml_topology": "cpml_topology_probe_energy",
        "pec_topology": "pec_topology_probe_energy",
        "port_proxy": "one_excited_one_passive_port_proxy",
    }[family]
    return {
        "case_id": f"{family}_full_physical_floor_strategy_b_physics",
        "family": family,
        "mode": "full",
        "boundary": boundary,
        "objective_family": objective,
        "oracle_type": "full_physical_floor_metadata",
        "metric_name": "full_physical_floor_execution",
        "comparator": "requires dedicated full/slow physical runner",
        "tolerance": None,
        "measured_value": None,
        "physical_metrics": {},
        "row_state": "not_evaluated",
        "reason": "full physical floor execution is intentionally split from the local quick gate",
        "evidence_strength": "metadata_only",
        "raw_parity_supplemental": None,
        "strategy_b_executed": False,
        "runtime_surface": _runtime_surface_name(family),
        "n_steps": None,
        "checkpoint_every": None,
        "runtime_s": 0.0,
        "full_evidence_requirement": {
            "status": "requires_split_run_physical_floor_execution",
            "minimum_requirement": "run representative physical source/probe, topology, or port-proxy floor with independent physical metrics before production promotion",
        },
    }


def _runtime_surface_name(family: str) -> str:
    if family == "source_probe":
        return "Simulation.forward_hybrid_phase1_from_inputs(..., strategy='b')"
    if family in {"cpml_topology", "pec_topology"}:
        return "topology_optimize(..., adjoint_mode='hybrid', strategy='b')"
    return "optimize(..., adjoint_mode='hybrid', strategy='b')"


def classify_family(rows: Iterable[dict[str, Any]], *, fail_closed_pass: bool = True) -> dict[str, Any]:
    family_rows = list(rows)
    if not family_rows:
        return {"status": "not_evaluated", "reasons": ["no evidence rows for family"]}
    if any(row.get("row_state") == "fail" for row in family_rows):
        return {
            "status": "physics_blocked",
            "reasons": ["one or more mandatory physical metrics failed"],
        }
    if not fail_closed_pass:
        return {"status": "physics_blocked", "reasons": ["global fail-closed audit failed"]}

    meaningful_rows = [row for row in family_rows if row.get("row_state") in {"pass", "warn"}]
    if not meaningful_rows:
        return {"status": "not_evaluated", "reasons": ["no meaningful physical evidence"]}
    if any(row.get("row_state") == "warn" for row in meaningful_rows):
        return {
            "status": "physics_experimental",
            "reasons": ["quick or weak physical proxy evidence requires full physical floor follow-up"],
        }
    return {
        "status": "physics_validated_limited",
        "reasons": ["all mandatory quick physical metrics passed; not production readiness"],
    }


def classify_overall(
    family_statuses: dict[str, dict[str, Any]],
    *,
    fail_closed_pass: bool,
) -> dict[str, Any]:
    if not family_statuses:
        return {"status": "not_evaluated", "reasons": ["no landed family evidence"]}
    statuses = {name: payload["status"] for name, payload in family_statuses.items()}
    if all(status == "not_evaluated" for status in statuses.values()):
        return {"status": "not_evaluated", "reasons": ["no meaningful physical evidence"]}
    if not fail_closed_pass or any(status == "physics_blocked" for status in statuses.values()):
        return {
            "status": "physics_blocked",
            "reasons": ["family or global fail-closed audit blocked physical validation"],
        }
    if any(status in {"physics_experimental", "not_evaluated"} for status in statuses.values()):
        return {
            "status": "physics_experimental",
            "reasons": ["at least one family has weak, partial, or missing physical evidence"],
        }
    return {
        "status": "physics_validated_limited",
        "reasons": ["all landed families cleared mandatory physical metrics; not production readiness"],
    }


def build_phase8_report(
    *,
    mode: Literal["quick", "full"] = "quick",
    family: str | None = None,
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or Thresholds()
    selected = [name for name in FAMILIES if family in {None, "all", name}]
    if not selected:
        allowed = ", ".join(["all", *FAMILIES])
        raise ValueError(f"unknown family {family!r}; expected one of {allowed}")

    rows = [
        _build_quick_row(name, thresholds) if mode == "quick" else _build_full_row(name, thresholds)
        for name in selected
    ]
    fail_closed_audit = phase7.build_fail_closed_audit() if mode == "quick" else {
        "row_state": "not_evaluated",
        "runtime_s": 0.0,
        "rows": [],
        "reason": "fail-closed audit runs in quick mode",
    }
    fail_closed_pass = fail_closed_audit["row_state"] in {"pass", "not_evaluated"}
    family_statuses = {
        name: classify_family(
            [row for row in rows if row["family"] == name],
            fail_closed_pass=fail_closed_pass,
        )
        for name in selected
    }
    overall = classify_overall(family_statuses, fail_closed_pass=fail_closed_pass)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_contract": BENCHMARK_CONTRACT,
        "contract_status": CONTRACT_STATUS,
        "preserves_phase_vii_readiness_contract": True,
        "phase_vii_contract_reference": phase7.BENCHMARK_CONTRACT,
        "baseline_commit": BASELINE_COMMIT,
        "generated_at": _utc_now(),
        "mode": mode,
        "selected_family": family or "all",
        "row_states": list(ROW_STATES),
        "family_statuses_allowed": list(FAMILY_STATUSES),
        "thresholds": {
            "causal_threshold_ratio": thresholds.causal_threshold_ratio,
            "cpml_tail_growth_limit": thresholds.cpml_tail_growth_limit,
            "pec_energy_growth_limit": thresholds.pec_energy_growth_limit,
            "passive_gain_limit": thresholds.passive_gain_limit,
            "physical_parity": thresholds.physical_parity,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "required_row_fields": list(REQUIRED_ROW_FIELDS),
        "rows": rows,
        "fail_closed_audit": fail_closed_audit,
        "readiness_integration": {
            "phase_viii_status_is_physical_only": True,
            "does_not_set_production_ready_limited": True,
            "production_readiness_requires_phase_vii_and_phase_ix_floor_evidence": True,
        },
        "summary": {
            "row_count": len(rows),
            "rows_with_required_fields": sum(all(field in row for field in REQUIRED_ROW_FIELDS) for row in rows),
            "family_statuses": family_statuses,
            "overall_status": overall["status"],
            "overall_reasons": overall["reasons"],
            "fail_closed_audit_state": fail_closed_audit["row_state"],
            "phase_viii_never_maps_directly_to_production_ready_limited": True,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--family",
        choices=("all", *FAMILIES),
        default="all",
        help="Limit report to one landed family in full/diagnostic runs.",
    )
    parser.add_argument("--causal-threshold-ratio", type=float, default=DEFAULT_CAUSAL_THRESHOLD_RATIO)
    parser.add_argument("--cpml-tail-growth-limit", type=float, default=DEFAULT_CPML_TAIL_GROWTH_LIMIT)
    parser.add_argument("--pec-energy-growth-limit", type=float, default=DEFAULT_PEC_ENERGY_GROWTH_LIMIT)
    parser.add_argument("--passive-gain-limit", type=float, default=DEFAULT_PASSIVE_GAIN_LIMIT)
    parser.add_argument("--physical-parity-threshold", type=float, default=DEFAULT_PHYSICAL_PARITY_THRESHOLD)
    parser.add_argument("--output", type=Path, help="Optional JSON output path. Report is always printed to stdout.")
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds = Thresholds(
        causal_threshold_ratio=args.causal_threshold_ratio,
        cpml_tail_growth_limit=args.cpml_tail_growth_limit,
        pec_energy_growth_limit=args.pec_energy_growth_limit,
        passive_gain_limit=args.passive_gain_limit,
        physical_parity=args.physical_parity_threshold,
    )
    report = build_phase8_report(
        mode=args.mode,
        family=args.family,
        thresholds=thresholds,
    )
    text = json.dumps(report, indent=args.indent, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

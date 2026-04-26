"""Phase VII Strategy B production-readiness evidence harness.

This harness deliberately lives beside, not inside, the locked Phase III and
Phase VI benchmark contracts.  It gathers Phase VII readiness evidence for the
currently landed Strategy B families, classifies that evidence with a canonical
row -> family -> overall roll-up, and refuses to promote quick/tiny fixtures to
production-ready status.

Run:
    python scripts/phase7_strategy_b_readiness.py --mode quick --indent 2
    python scripts/phase7_strategy_b_readiness.py --mode full --family source_probe --indent 2
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
from typing import Any, Callable, Iterable, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rfx import Box, DebyePole, GaussianPulse, Simulation  # noqa: E402
from rfx.grid import C0  # noqa: E402
from rfx.materials.lorentz import LorentzPole  # noqa: E402
from rfx.optimize import DesignRegion, optimize  # noqa: E402
from rfx.topology import TopologyDesignRegion, topology_optimize  # noqa: E402

SCHEMA_VERSION = 1
BENCHMARK_CONTRACT = "phase_vii_strategy_b_production_readiness"
CONTRACT_STATUS = "phase_vii_readiness_evidence"
BASELINE_COMMIT = "eeb365e"
DEFAULT_BUDGET_GB = 24.0
DEFAULT_CHECKPOINT_EVERY = 1000
QUICK_N_STEPS = 8
QUICK_CHECKPOINT_EVERY = 3
DEFAULT_PARITY_THRESHOLD = 1e-5
DEFAULT_GRADIENT_THRESHOLD = 1e-4

ROW_STATES = ("pass", "warn", "fail", "not_evaluated")
FAMILY_STATUSES = (
    "production_ready_limited",
    "experimental_limited",
    "blocked",
    "not_evaluated",
)

REQUIRED_ROW_FIELDS = (
    "case_id",
    "family",
    "mode",
    "boundary",
    "objective_family",
    "grid_shape",
    "cell_count",
    "n_steps",
    "checkpoint_every",
    "row_state",
    "reason",
    "workload_floor",
    "correctness_metric",
    "correctness_threshold",
    "gradient_metric",
    "gradient_threshold",
    "required_gradient_evidence",
    "strategy_a_estimated_memory_gb",
    "strategy_b_estimated_memory_gb",
    "memory_status",
    "runtime_s",
    "observed_host_rss_delta_mb",
    "observed_host_rss_peak_mb",
    "host_memory_caveat",
    "evidence",
)


@dataclass(frozen=True)
class Thresholds:
    parity: float = DEFAULT_PARITY_THRESHOLD
    gradient: float = DEFAULT_GRADIENT_THRESHOLD
    budget_gb: float = DEFAULT_BUDGET_GB


@dataclass(frozen=True)
class WorkloadFloor:
    case_id: str
    source: str
    boundary: str
    domain_m: tuple[float, float, float]
    dx_m: float
    freq_max_hz: float
    n_steps: int
    checkpoint_every: int
    cpml_layers: int = 0
    representative: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source": self.source,
            "boundary": self.boundary,
            "domain_m": list(self.domain_m),
            "dx_m": self.dx_m,
            "freq_max_hz": self.freq_max_hz,
            "n_steps": self.n_steps,
            "checkpoint_every": self.checkpoint_every,
            "cpml_layers": self.cpml_layers,
            "representative": self.representative,
        }


@dataclass(frozen=True)
class ReadinessCase:
    family: str
    case_id: str
    objective_family: str
    builder: Literal["source_probe", "topology", "port_proxy"]
    boundary: str
    quick_domain_m: tuple[float, float, float]
    quick_freq_max_hz: float
    quick_dx_m: float | None
    quick_cpml_layers: int
    floor: WorkloadFloor
    required_gradient_source: str
    full_mode_supported: bool = True


FLOORS: dict[str, WorkloadFloor] = {
    "source_probe": WorkloadFloor(
        case_id="source_probe_optimize_patch",
        source="Phase III Gate 0 primary workload + Phase VI runtime evidence",
        boundary="cpml",
        domain_m=(0.030, 0.030, 0.030),
        dx_m=0.5e-3,
        freq_max_hz=10e9,
        n_steps=10_000,
        checkpoint_every=1_000,
        cpml_layers=8,
    ),
    "cpml_topology": WorkloadFloor(
        case_id="cpml_topology_probe_energy",
        source="Phase III Gate 0 secondary workload + Phase VI runtime evidence",
        boundary="cpml",
        domain_m=(0.040, 0.040, 0.040),
        dx_m=0.6e-3,
        freq_max_hz=10e9,
        n_steps=10_000,
        checkpoint_every=1_000,
        cpml_layers=8,
    ),
    "pec_topology": WorkloadFloor(
        case_id="phase_vi_pec_topology_parity_fixture",
        source="Phase VI PEC topology parity fixture; no Phase III representative PEC topology floor",
        boundary="pec",
        domain_m=(0.015, 0.015, 0.015),
        dx_m=0.75e-3,
        freq_max_hz=5e9,
        n_steps=8,
        checkpoint_every=3,
        cpml_layers=0,
        representative=False,
    ),
    "port_proxy": WorkloadFloor(
        case_id="one_passive_port_proxy",
        source="Phase III Gate 0 tertiary workload + Phase VI runtime evidence",
        boundary="pec",
        domain_m=(0.018, 0.018, 0.018),
        dx_m=0.35e-3,
        freq_max_hz=8e9,
        n_steps=8_000,
        checkpoint_every=1_000,
        cpml_layers=0,
    ),
}

CASES: tuple[ReadinessCase, ...] = (
    ReadinessCase(
        family="source_probe",
        case_id="source_probe_cpml_strategy_b_readiness",
        objective_family="source_probe",
        builder="source_probe",
        boundary="cpml",
        quick_domain_m=(0.015, 0.015, 0.015),
        quick_freq_max_hz=5e9,
        quick_dx_m=None,
        quick_cpml_layers=8,
        floor=FLOORS["source_probe"],
        required_gradient_source="Phase III source/probe Strategy B custom-VJP gradient tests",
    ),
    ReadinessCase(
        family="cpml_topology",
        case_id="cpml_topology_probe_energy_strategy_b_readiness",
        objective_family="cpml_topology_probe_energy",
        builder="topology",
        boundary="cpml",
        quick_domain_m=(0.015, 0.015, 0.015),
        quick_freq_max_hz=5e9,
        quick_dx_m=None,
        quick_cpml_layers=8,
        floor=FLOORS["cpml_topology"],
        required_gradient_source="Phase VI CPML topology Strategy B gradient parity test",
    ),
    ReadinessCase(
        family="pec_topology",
        case_id="pec_topology_probe_energy_strategy_b_readiness",
        objective_family="pec_topology_probe_energy",
        builder="topology",
        boundary="pec",
        quick_domain_m=(0.015, 0.015, 0.015),
        quick_freq_max_hz=5e9,
        quick_dx_m=None,
        quick_cpml_layers=0,
        floor=FLOORS["pec_topology"],
        required_gradient_source="Phase VI PEC topology Strategy B parity evidence; representative gradient floor deferred",
        full_mode_supported=False,
    ),
    ReadinessCase(
        family="port_proxy",
        case_id="one_passive_port_proxy_strategy_b_readiness",
        objective_family="one_excited_one_passive_port_proxy",
        builder="port_proxy",
        boundary="pec",
        quick_domain_m=(0.015, 0.015, 0.015),
        quick_freq_max_hz=5e9,
        quick_dx_m=None,
        quick_cpml_layers=0,
        floor=FLOORS["port_proxy"],
        required_gradient_source="Phase VI one-passive-port proxy Strategy B gradient parity test",
    ),
)


class ReadinessError(RuntimeError):
    """Raised when a readiness row cannot be evaluated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rss_mb() -> float:
    # Linux returns KiB, macOS returns bytes.  The CI containers for this repo are
    # Linux, but keep the branch explicit for portable local runs.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _center(domain: tuple[float, float, float]) -> tuple[float, float, float]:
    return (domain[0] / 2.0, domain[1] / 2.0, domain[2] / 2.0)


def _grid_shape(sim: Simulation) -> tuple[int, int, int]:
    dx = sim._dx or (C0 / sim._freq_max / 20.0)

    def axis_cells(extent: float, profile: object | None) -> int:
        if profile is not None:
            return len(profile) + 1 + 2 * sim._cpml_layers  # type: ignore[arg-type]
        return int(math.ceil(extent / dx)) + 1 + 2 * sim._cpml_layers

    return (
        axis_cells(sim._domain[0], sim._dx_profile),
        axis_cells(sim._domain[1], sim._dy_profile),
        axis_cells(sim._domain[2], sim._dz_profile),
    )


def _make_source_probe_sim(
    *,
    boundary: str = "cpml",
    domain: tuple[float, float, float] = (0.015, 0.015, 0.015),
    freq_max: float = 5e9,
    dx: float | None = None,
    cpml_layers: int = 8,
) -> Simulation:
    kwargs: dict[str, Any] = {
        "freq_max": freq_max,
        "domain": domain,
        "boundary": boundary,
        "cpml_layers": cpml_layers,
    }
    if dx is not None:
        kwargs["dx"] = dx
    sim = Simulation(**kwargs)
    x_mid, y_mid, z_mid = _center(domain)
    sim.add_source(
        (max(2 * (dx or 0.001), domain[0] / 3.0), y_mid, z_mid),
        "ez",
        waveform=GaussianPulse(f0=freq_max * 0.6, bandwidth=0.5),
    )
    sim.add_probe((min(domain[0] * 0.67, domain[0] - 2 * (dx or 0.001)), y_mid, z_mid), "ez")
    return sim


def _make_topology_case(boundary: str) -> tuple[Simulation, TopologyDesignRegion]:
    sim = _make_source_probe_sim(boundary=boundary, cpml_layers=8 if boundary == "cpml" else 0)
    sim.add_material("phase7_diel", eps_r=4.0, sigma=0.0)
    region = TopologyDesignRegion(
        corner_lo=(0.009, 0.003, 0.003),
        corner_hi=(0.012, 0.006, 0.006),
        material_bg="air",
        material_fg="phase7_diel",
        beta_projection=1.0,
    )
    return sim, region


def _make_port_proxy_case() -> tuple[Simulation, DesignRegion]:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port(
        (0.005, 0.0075, 0.0075),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_port((0.010, 0.0075, 0.0075), "ez", impedance=50.0, excite=False)
    sim.add_probe((0.012, 0.0075, 0.0075), "ez")
    region = DesignRegion(
        corner_lo=(0.009, 0.003, 0.003),
        corner_hi=(0.012, 0.006, 0.006),
        eps_range=(1.0, 4.4),
    )
    return sim, region


def _topology_objective(result):
    return -jnp.sum(result.time_series ** 2)


def _probe_objective(result):
    return -jnp.sum(result.time_series ** 2)


def _single_cell_eps(grid, base_eps: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    i, j, k = grid.position_to_index((0.0075, 0.0075, 0.0075))
    return base_eps.at[i, j, k].add(alpha)


def _estimate_memory(sim: Simulation, *, n_steps: int, checkpoint_every: int, budget_gb: float) -> tuple[float, float]:
    est = sim.estimate_ad_memory(
        n_steps=n_steps,
        available_memory_gb=budget_gb,
        checkpoint_every=checkpoint_every,
    )
    return float(est.ad_full_gb), float(est.ad_segmented_gb)


def _run_source_probe_quick(thresholds: Thresholds) -> dict[str, Any]:
    sim = _make_source_probe_sim(boundary="cpml")
    inputs = sim.build_hybrid_phase1_inputs(n_steps=QUICK_N_STEPS)
    before = _rss_mb()
    started = time.perf_counter()
    strategy_a = sim.forward_hybrid_phase1_from_inputs(inputs)
    strategy_b = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )
    runtime_s = time.perf_counter() - started
    after = _rss_mb()
    parity = float(jnp.max(jnp.abs(strategy_b.time_series - strategy_a.time_series)))
    grad_metric = _source_probe_gradient_metric(sim, inputs)
    return {
        "sim": sim,
        "runtime_s": runtime_s,
        "rss_delta_mb": max(0.0, after - before),
        "rss_peak_mb": after,
        "correctness_metric": parity,
        "correctness_pass": parity <= thresholds.parity,
        "gradient_metric": grad_metric,
        "gradient_pass": grad_metric <= thresholds.gradient,
    }


def _source_probe_gradient_metric(sim: Simulation, inputs) -> float:
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)

    def pure_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=QUICK_N_STEPS, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def strategy_b_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps,
            strategy="b",
            checkpoint_every=QUICK_CHECKPOINT_EVERY,
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_strategy_b = jax.grad(strategy_b_loss)(alpha0)
    denom = max(float(jnp.abs(grad_pure)), 1e-12)
    return float(jnp.abs(grad_strategy_b - grad_pure) / denom)


def _run_topology_quick(boundary: str, thresholds: Thresholds) -> dict[str, Any]:
    sim, region = _make_topology_case(boundary)
    before = _rss_mb()
    started = time.perf_counter()
    pure = topology_optimize(
        sim,
        region,
        _topology_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=QUICK_N_STEPS,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="pure_ad",
    )
    strategy_b = topology_optimize(
        sim,
        region,
        _topology_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=QUICK_N_STEPS,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )
    runtime_s = time.perf_counter() - started
    after = _rss_mb()
    history_diff = float(np.max(np.abs(np.asarray(strategy_b.history) - np.asarray(pure.history))))
    density_diff = float(np.max(np.abs(np.asarray(strategy_b.density) - np.asarray(pure.density))))
    metric = max(history_diff, density_diff)
    grad_metric = _topology_gradient_metric(sim, region)
    return {
        "sim": sim,
        "runtime_s": runtime_s,
        "rss_delta_mb": max(0.0, after - before),
        "rss_peak_mb": after,
        "correctness_metric": metric,
        "correctness_pass": metric <= thresholds.parity,
        "gradient_metric": grad_metric,
        "gradient_pass": grad_metric <= thresholds.gradient,
    }


def _topology_gradient_metric(sim: Simulation, region: TopologyDesignRegion) -> float:
    # Reuse the same scalar cell-gradient parity used in Phase VI tests, but keep
    # it local so this harness owns the emitted evidence artifact.
    from rfx.topology import _inspect_topology_hybrid_support

    inputs, report, grid, *_ = _inspect_topology_hybrid_support(
        sim,
        region,
        n_steps=QUICK_N_STEPS,
    )
    if not report.supported or inputs.materials is None:
        raise ReadinessError(report.reason_text)

    def pure_loss(alpha):
        eps = _single_cell_eps(grid, inputs.materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=QUICK_N_STEPS, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def strategy_b_loss(alpha):
        eps = _single_cell_eps(grid, inputs.materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps,
            strategy="b",
            checkpoint_every=QUICK_CHECKPOINT_EVERY,
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_strategy_b = jax.grad(strategy_b_loss)(alpha0)
    denom = max(float(jnp.abs(grad_pure)), 1e-12)
    return float(jnp.abs(grad_strategy_b - grad_pure) / denom)


def _run_port_proxy_quick(thresholds: Thresholds) -> dict[str, Any]:
    sim, region = _make_port_proxy_case()
    before = _rss_mb()
    started = time.perf_counter()
    pure = optimize(
        sim,
        region,
        _probe_objective,
        n_iters=1,
        lr=0.01,
        n_steps=QUICK_N_STEPS,
        verbose=False,
        adjoint_mode="pure_ad",
    )
    strategy_b = optimize(
        sim,
        region,
        _probe_objective,
        n_iters=1,
        lr=0.01,
        n_steps=QUICK_N_STEPS,
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )
    runtime_s = time.perf_counter() - started
    after = _rss_mb()
    loss_diff = float(
        np.max(np.abs(np.asarray(strategy_b.loss_history) - np.asarray(pure.loss_history)))
    )
    latent_diff = float(np.max(np.abs(np.asarray(strategy_b.latent) - np.asarray(pure.latent))))
    metric = max(loss_diff, latent_diff)
    grad_metric = _port_proxy_gradient_metric(sim)
    return {
        "sim": sim,
        "runtime_s": runtime_s,
        "rss_delta_mb": max(0.0, after - before),
        "rss_peak_mb": after,
        "correctness_metric": metric,
        "correctness_pass": metric <= thresholds.parity,
        "gradient_metric": grad_metric,
        "gradient_pass": grad_metric <= thresholds.gradient,
    }


def _port_proxy_gradient_metric(sim: Simulation) -> float:
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=QUICK_N_STEPS)

    def pure_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=QUICK_N_STEPS, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def strategy_b_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps,
            strategy="b",
            checkpoint_every=QUICK_CHECKPOINT_EVERY,
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_strategy_b = jax.grad(strategy_b_loss)(alpha0)
    denom = max(float(jnp.abs(grad_pure)), 1e-12)
    return float(jnp.abs(grad_strategy_b - grad_pure) / denom)


def _quick_evidence(case: ReadinessCase, thresholds: Thresholds) -> dict[str, Any]:
    if case.builder == "source_probe":
        return _run_source_probe_quick(thresholds)
    if case.builder == "topology":
        return _run_topology_quick(case.boundary, thresholds)
    if case.builder == "port_proxy":
        return _run_port_proxy_quick(thresholds)
    raise ValueError(f"unknown builder {case.builder!r}")


def _build_quick_row(case: ReadinessCase, thresholds: Thresholds) -> dict[str, Any]:
    evidence = _quick_evidence(case, thresholds)
    sim = evidence["sim"]
    shape = _grid_shape(sim)
    strategy_a_gb, strategy_b_gb = _estimate_memory(
        sim,
        n_steps=QUICK_N_STEPS,
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
        budget_gb=thresholds.budget_gb,
    )
    memory_pass = strategy_b_gb < strategy_a_gb and strategy_b_gb <= thresholds.budget_gb
    correctness_pass = bool(evidence["correctness_pass"])
    gradient_pass = bool(evidence["gradient_pass"])
    row_state = "pass" if correctness_pass and gradient_pass and memory_pass else "fail"
    reason = (
        "quick Strategy B runtime/correctness/gradient evidence passed; family remains experimental_limited until full floor clears"
        if row_state == "pass"
        else "quick Strategy B readiness evidence failed one or more thresholds"
    )
    return {
        "case_id": case.case_id,
        "family": case.family,
        "mode": "quick",
        "boundary": case.boundary,
        "objective_family": case.objective_family,
        "grid_shape": list(shape),
        "cell_count": math.prod(shape),
        "n_steps": QUICK_N_STEPS,
        "checkpoint_every": QUICK_CHECKPOINT_EVERY,
        "row_state": row_state,
        "reason": reason,
        "workload_floor": {
            **case.floor.as_dict(),
            "meets_full_floor": False,
            "floor_status": "quick_evidence_only",
        },
        "correctness_metric": round(float(evidence["correctness_metric"]), 12),
        "correctness_threshold": thresholds.parity,
        "gradient_metric": round(float(evidence["gradient_metric"]), 12),
        "gradient_threshold": thresholds.gradient,
        "required_gradient_evidence": {
            "required": True,
            "state": "pass" if gradient_pass else "fail",
            "source": case.required_gradient_source,
        },
        "strategy_a_estimated_memory_gb": round(strategy_a_gb, 6),
        "strategy_b_estimated_memory_gb": round(strategy_b_gb, 6),
        "memory_status": "pass" if memory_pass else "fail",
        "runtime_s": round(float(evidence["runtime_s"]), 6),
        "observed_host_rss_delta_mb": round(float(evidence["rss_delta_mb"]), 3),
        "observed_host_rss_peak_mb": round(float(evidence["rss_peak_mb"]), 3),
        "host_memory_caveat": "stdlib ru_maxrss is a coarse process peak, not JAX device memory",
        "evidence": {
            "runtime_surface": _runtime_surface_name(case),
            "strategy_a_comparator": "pure_ad/custom-vjp Strategy A surface",
            "quick_fixture_note": "CI-safe tiny fixture; not representative full-mode production evidence",
        },
    }


def _build_full_row(case: ReadinessCase, thresholds: Thresholds) -> dict[str, Any]:
    floor = case.floor
    sim = _sim_for_floor(case)
    shape = _grid_shape(sim)
    strategy_a_gb, strategy_b_gb = _estimate_memory(
        sim,
        n_steps=floor.n_steps,
        checkpoint_every=floor.checkpoint_every,
        budget_gb=thresholds.budget_gb,
    )
    reason = (
        "representative PEC topology full floor is not defined in Phase III; family remains experimental_limited"
        if not case.full_mode_supported
        else "full-mode workload-floor execution is intentionally not performed by this local metadata report"
    )
    return {
        "case_id": case.case_id,
        "family": case.family,
        "mode": "full",
        "boundary": floor.boundary,
        "objective_family": case.objective_family,
        "grid_shape": list(shape),
        "cell_count": math.prod(shape),
        "n_steps": floor.n_steps,
        "checkpoint_every": floor.checkpoint_every,
        "row_state": "not_evaluated",
        "reason": reason,
        "workload_floor": {
            **floor.as_dict(),
            "meets_full_floor": bool(floor.representative and case.full_mode_supported),
            "floor_status": "declared_not_executed",
        },
        "correctness_metric": None,
        "correctness_threshold": thresholds.parity,
        "gradient_metric": None,
        "gradient_threshold": thresholds.gradient,
        "required_gradient_evidence": {
            "required": True,
            "state": "not_evaluated",
            "source": case.required_gradient_source,
        },
        "strategy_a_estimated_memory_gb": round(strategy_a_gb, 6),
        "strategy_b_estimated_memory_gb": round(strategy_b_gb, 6),
        "memory_status": "pass" if strategy_b_gb < strategy_a_gb else "fail",
        "runtime_s": 0.0,
        "observed_host_rss_delta_mb": None,
        "observed_host_rss_peak_mb": round(_rss_mb(), 3),
        "host_memory_caveat": "full workload not executed by default; estimator evidence only",
        "full_evidence_requirement": {
            "status": "requires_split_run_floor_execution",
            "reason": "local full mode records locked floor metadata and estimator evidence only; production evidence requires a separate floor runner that executes correctness, memory, and gradient gates at this workload size",
            "minimum_floor": floor.as_dict(),
            "suggested_runner_contract": (
                "Implement or invoke a dedicated split-run full-floor runner for this family, then import "
                "its measured correctness, required-gradient, runtime, and memory evidence into the Phase VII schema."
            ),
        },
        "evidence": {
            "runtime_surface": _runtime_surface_name(case),
            "full_mode_note": "metadata and estimator evidence only; no full workload-floor execution is claimed",
        },
    }


def _runtime_surface_name(case: ReadinessCase) -> str:
    if case.builder == "source_probe":
        return "Simulation.forward_hybrid_phase1_from_inputs(..., strategy='b')"
    if case.builder == "topology":
        return "topology_optimize(..., adjoint_mode='hybrid', strategy='b')"
    return "optimize(..., adjoint_mode='hybrid', strategy='b')"


def _sim_for_floor(case: ReadinessCase) -> Simulation:
    floor = case.floor
    if case.builder in {"source_probe", "topology"}:
        return _make_source_probe_sim(
            boundary=floor.boundary,
            domain=floor.domain_m,
            freq_max=floor.freq_max_hz,
            dx=floor.dx_m,
            cpml_layers=floor.cpml_layers,
        )
    sim = Simulation(
        freq_max=floor.freq_max_hz,
        domain=floor.domain_m,
        boundary=floor.boundary,
        dx=floor.dx_m,
        cpml_layers=floor.cpml_layers,
    )
    x_mid, y_mid, z_mid = _center(floor.domain_m)
    sim.add_port(
        (floor.domain_m[0] * 0.33, y_mid, z_mid),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=floor.freq_max_hz / 2.0, bandwidth=0.5),
    )
    sim.add_port((floor.domain_m[0] * 0.67, y_mid, z_mid), "ez", impedance=50.0, excite=False)
    sim.add_probe((x_mid, y_mid, z_mid), "ez")
    return sim


def classify_family(rows: Iterable[dict[str, Any]], *, fail_closed_pass: bool = True) -> dict[str, Any]:
    family_rows = list(rows)
    if not family_rows:
        return {"status": "not_evaluated", "reasons": ["no evidence rows for family"]}
    if any(row.get("row_state") == "fail" for row in family_rows):
        return {"status": "blocked", "reasons": ["one or more required evidence items failed"]}
    if not fail_closed_pass:
        return {"status": "blocked", "reasons": ["global fail-closed audit failed"]}

    meaningful_rows = [row for row in family_rows if row.get("row_state") in {"pass", "warn"}]
    if not meaningful_rows:
        return {"status": "not_evaluated", "reasons": ["no meaningful execution evidence"]}

    full_floor_pass = any(
        row.get("mode") == "full"
        and row.get("row_state") == "pass"
        and row.get("workload_floor", {}).get("meets_full_floor") is True
        for row in family_rows
    )
    required_gradient_pass = any(
        row.get("required_gradient_evidence", {}).get("required") is True
        and row.get("required_gradient_evidence", {}).get("state") == "pass"
        for row in family_rows
    )
    if full_floor_pass and required_gradient_pass:
        return {"status": "production_ready_limited", "reasons": ["full floor and required gradient evidence passed"]}

    reasons: list[str] = []
    if not full_floor_pass:
        reasons.append("missing full-mode workload-floor evidence")
    if not required_gradient_pass:
        reasons.append("missing required gradient evidence")
    return {"status": "experimental_limited", "reasons": reasons}


def classify_overall(family_statuses: dict[str, dict[str, Any]], *, fail_closed_pass: bool) -> dict[str, Any]:
    if not family_statuses:
        return {"status": "not_evaluated", "reasons": ["no landed family evidence"]}
    statuses = {name: payload["status"] for name, payload in family_statuses.items()}
    if all(status == "not_evaluated" for status in statuses.values()):
        return {"status": "not_evaluated", "reasons": ["no meaningful readiness execution evidence"]}
    if not fail_closed_pass or any(status == "blocked" for status in statuses.values()):
        return {"status": "blocked", "reasons": ["family or global fail-closed audit blocked readiness"]}
    if any(status in {"experimental_limited", "not_evaluated"} for status in statuses.values()):
        return {
            "status": "experimental_limited",
            "reasons": ["at least one family lacks full workload-floor or required gradient evidence"],
        }
    return {"status": "production_ready_limited", "reasons": ["all landed families cleared readiness gates"]}


def _expect_explicit_strategy_b_raise(
    name: str,
    fn: Callable[[], Any],
    expected: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - fail-closed audit intentionally captures any rejection.
        message = str(exc)
        passed = expected in message
        return {
            "case_id": name,
            "row_state": "pass" if passed else "fail",
            "expected_reason_contains": expected,
            "observed_reason": message,
            "runtime_s": round(time.perf_counter() - started, 6),
            "audit_kind": "explicit_strategy_b_raise",
        }
    return {
        "case_id": name,
        "row_state": "fail",
        "expected_reason_contains": expected,
        "observed_reason": "explicit Strategy B request completed instead of raising",
        "runtime_s": round(time.perf_counter() - started, 6),
        "audit_kind": "explicit_strategy_b_raise",
    }


def _run_explicit_strategy_b_from_inputs(sim: Simulation) -> None:
    inputs = sim.build_hybrid_phase1_inputs(n_steps=QUICK_N_STEPS)
    sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )


def _audit_nonuniform_strategy_b() -> Any:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary="cpml",
        dx=0.0025,
        dz_profile=np.array([0.0020, 0.0016, 0.0013, 0.0016, 0.0020], dtype=float),
    )
    sim.add_source((0.005, 0.0075, 0.0075), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_ntff_strategy_b() -> Any:
    sim = _make_source_probe_sim(boundary="cpml")
    sim.add_ntff_box((0.003, 0.003, 0.003), (0.012, 0.012, 0.012), n_freqs=2)
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_extra_passive_port_strategy_b() -> Any:
    sim, _ = _make_port_proxy_case()
    sim.add_port((0.013, 0.0075, 0.0075), "ez", impedance=50.0, excite=False)
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_excited_port_overlap_auto_strategy_b() -> None:
    sim, _ = _make_port_proxy_case()
    overlap_region = DesignRegion(
        corner_lo=(0.004, 0.006, 0.006),
        corner_hi=(0.006, 0.009, 0.009),
        eps_range=(1.0, 4.4),
    )
    optimize(
        sim,
        overlap_region,
        _probe_objective,
        n_iters=1,
        lr=0.01,
        n_steps=QUICK_N_STEPS,
        verbose=False,
        adjoint_mode="auto",
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )


def _audit_port_overlap_auto_strategy_b() -> None:
    sim, _ = _make_port_proxy_case()
    overlap_region = DesignRegion(
        corner_lo=(0.009, 0.006, 0.006),
        corner_hi=(0.011, 0.009, 0.009),
        eps_range=(1.0, 4.4),
    )
    optimize(
        sim,
        overlap_region,
        _probe_objective,
        n_iters=1,
        lr=0.01,
        n_steps=QUICK_N_STEPS,
        verbose=False,
        adjoint_mode="auto",
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )


def _audit_lossy_topology_auto_strategy_b() -> None:
    sim, region = _make_topology_case("cpml")
    sim.add_material("phase7_lossy", eps_r=4.0, sigma=0.02)
    lossy_region = TopologyDesignRegion(
        corner_lo=region.corner_lo,
        corner_hi=region.corner_hi,
        material_bg=region.material_bg,
        material_fg="phase7_lossy",
        beta_projection=region.beta_projection,
    )
    topology_optimize(
        sim,
        lossy_region,
        _topology_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=QUICK_N_STEPS,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="auto",
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )


def _audit_lossy_source_probe_strategy_b() -> None:
    sim = _make_source_probe_sim(boundary="cpml")
    sim.add_material("phase7_lossy_blocked", eps_r=2.0, sigma=0.01)
    sim.add(Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)), material="phase7_lossy_blocked")
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_debye_strategy_b() -> None:
    sim = _make_source_probe_sim(boundary="cpml")
    sim.add_material(
        "phase7_debye_blocked",
        eps_r=2.0,
        debye_poles=[DebyePole(delta_eps=1.0, tau=8e-12)],
    )
    sim.add(Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)), material="phase7_debye_blocked")
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_lorentz_strategy_b() -> None:
    sim = _make_source_probe_sim(boundary="cpml")
    omega_0 = 2.0 * math.pi * 6e9
    sim.add_material(
        "phase7_lorentz_blocked",
        eps_r=2.0,
        lorentz_poles=[LorentzPole(omega_0=omega_0, delta=1e8, kappa=omega_0**2)],
    )
    sim.add(Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)), material="phase7_lorentz_blocked")
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_mixed_dispersion_strategy_b() -> None:
    sim = _make_source_probe_sim(boundary="cpml")
    omega_0 = 2.0 * math.pi * 6e9
    sim.add_material(
        "phase7_mixed_dispersion_blocked",
        eps_r=2.0,
        debye_poles=[DebyePole(delta_eps=1.0, tau=8e-12)],
        lorentz_poles=[LorentzPole(omega_0=omega_0, delta=1e8, kappa=omega_0**2)],
    )
    sim.add(
        Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)),
        material="phase7_mixed_dispersion_blocked",
    )
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_pec_topology_strategy_b() -> None:
    sim = _make_source_probe_sim(boundary="pec", cpml_layers=0)
    region = TopologyDesignRegion(
        corner_lo=(0.006, 0.006, 0.006),
        corner_hi=(0.009, 0.009, 0.009),
        material_bg="air",
        material_fg="pec",
        beta_projection=1.0,
    )
    topology_optimize(
        sim,
        region,
        _topology_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=QUICK_N_STEPS,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="auto",
        strategy="b",
        checkpoint_every=QUICK_CHECKPOINT_EVERY,
    )


def _audit_wire_port_strategy_b() -> None:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port(
        (0.005, 0.0075, 0.005),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
        extent=0.003,
    )
    sim.add_probe((0.010, 0.0075, 0.0075), "ez")
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_waveguide_port_strategy_b() -> None:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="cpml", cpml_layers=8)
    sim.add_waveguide_port(
        0.003,
        y_range=(0.004, 0.011),
        z_range=(0.004, 0.011),
        direction="+x",
        n_freqs=2,
        probe_offset=1,
        ref_offset=1,
    )
    _run_explicit_strategy_b_from_inputs(sim)


def _audit_floquet_port_strategy_b() -> None:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="cpml", cpml_layers=8)
    sim.add_floquet_port(0.003, axis="z", n_freqs=2)
    _run_explicit_strategy_b_from_inputs(sim)


def build_fail_closed_audit() -> dict[str, Any]:
    rows = [
        _expect_explicit_strategy_b_raise("nonuniform_strategy_b", _audit_nonuniform_strategy_b, "supports only uniform grids"),
        _expect_explicit_strategy_b_raise("ntff_directivity_strategy_b", _audit_ntff_strategy_b, "does not support NTFF/directivity"),
        _expect_explicit_strategy_b_raise("generic_multi_port_strategy_b", _audit_extra_passive_port_strategy_b, "supports exactly one excited lumped port"),
        _expect_explicit_strategy_b_raise("excited_port_design_region_overlap_auto_strategy_b", _audit_excited_port_overlap_auto_strategy_b, "design region overlaps the excited lumped-port cell"),
        _expect_explicit_strategy_b_raise("port_design_region_overlap_auto_strategy_b", _audit_port_overlap_auto_strategy_b, "design region overlaps a passive lumped-port cell"),
        _expect_explicit_strategy_b_raise("lossy_topology_auto_strategy_b", _audit_lossy_topology_auto_strategy_b, "zero sigma"),
        _expect_explicit_strategy_b_raise("lossy_source_probe_strategy_b", _audit_lossy_source_probe_strategy_b, "supports conductivity only for the bounded lumped-port proxy"),
        _expect_explicit_strategy_b_raise("debye_strategy_b", _audit_debye_strategy_b, "supports only lossless nondispersive materials"),
        _expect_explicit_strategy_b_raise("lorentz_strategy_b", _audit_lorentz_strategy_b, "supports only lossless nondispersive materials"),
        _expect_explicit_strategy_b_raise("mixed_dispersion_strategy_b", _audit_mixed_dispersion_strategy_b, "supports only lossless nondispersive materials"),
        _expect_explicit_strategy_b_raise("topology_pec_occupancy_strategy_b", _audit_pec_topology_strategy_b, "pec_occupancy replay is unsupported"),
        _expect_explicit_strategy_b_raise("wire_port_strategy_b", _audit_wire_port_strategy_b, "supports exactly one excited lumped port"),
        _expect_explicit_strategy_b_raise("waveguide_port_strategy_b", _audit_waveguide_port_strategy_b, "waveguide/wire/floquet port accumulation is unsupported"),
        _expect_explicit_strategy_b_raise("floquet_port_strategy_b", _audit_floquet_port_strategy_b, "supports only add_source()/probe workflows"),
    ]
    passed = all(row["row_state"] == "pass" for row in rows)
    return {
        "row_state": "pass" if passed else "fail",
        "runtime_s": round(sum(row["runtime_s"] for row in rows), 6),
        "rows": rows,
    }


def build_phase7_report(
    *,
    mode: Literal["quick", "full"] = "quick",
    family: str | None = None,
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or Thresholds()
    selected = [case for case in CASES if family in {None, "all", case.family}]
    if not selected:
        allowed = ", ".join(["all", *(case.family for case in CASES)])
        raise ValueError(f"unknown family {family!r}; expected one of {allowed}")

    rows = [
        _build_quick_row(case, thresholds) if mode == "quick" else _build_full_row(case, thresholds)
        for case in selected
    ]
    fail_closed_audit = build_fail_closed_audit() if mode == "quick" else {
        "row_state": "not_evaluated",
        "runtime_s": 0.0,
        "rows": [],
        "reason": "fail-closed audit runs in quick mode",
    }
    fail_closed_pass = fail_closed_audit["row_state"] in {"pass", "not_evaluated"}
    family_statuses = {
        case.family: classify_family(
            [row for row in rows if row["family"] == case.family],
            fail_closed_pass=fail_closed_pass,
        )
        for case in selected
    }
    overall = classify_overall(family_statuses, fail_closed_pass=fail_closed_pass)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_contract": BENCHMARK_CONTRACT,
        "contract_status": CONTRACT_STATUS,
        "preserves_phase_iii_gate0_contract": True,
        "baseline_commit": BASELINE_COMMIT,
        "generated_at": _utc_now(),
        "mode": mode,
        "selected_family": family or "all",
        "row_states": list(ROW_STATES),
        "family_statuses_allowed": list(FAMILY_STATUSES),
        "thresholds": {
            "parity": thresholds.parity,
            "gradient": thresholds.gradient,
            "budget_gb": thresholds.budget_gb,
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
        "summary": {
            "row_count": len(rows),
            "rows_with_required_fields": sum(all(field in row for field in REQUIRED_ROW_FIELDS) for row in rows),
            "family_statuses": family_statuses,
            "overall_status": overall["status"],
            "overall_reasons": overall["reasons"],
            "fail_closed_audit_state": fail_closed_audit["row_state"],
            "production_ready_limited_requires_full_mode": True,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--family",
        choices=("all", *(case.family for case in CASES)),
        default="all",
        help="Limit report to one landed family in full/diagnostic runs.",
    )
    parser.add_argument("--budget-gb", type=float, default=DEFAULT_BUDGET_GB)
    parser.add_argument("--parity-threshold", type=float, default=DEFAULT_PARITY_THRESHOLD)
    parser.add_argument("--gradient-threshold", type=float, default=DEFAULT_GRADIENT_THRESHOLD)
    parser.add_argument("--output", type=Path, help="Optional JSON output path. Report is always printed to stdout.")
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds = Thresholds(
        parity=args.parity_threshold,
        gradient=args.gradient_threshold,
        budget_gb=args.budget_gb,
    )
    report = build_phase7_report(
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

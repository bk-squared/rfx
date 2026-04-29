#!/usr/bin/env python3
"""WR-90 reference-free S11 and source-purity oracle matrix.

Diagnostic-only harness for separating current two-run reference/DFT/CPML
calibration artifacts from source impurity and internal short physics.  This
script intentionally avoids production source/extractor changes: it builds
script-local rows, emits machine-readable JSONL evidence, and leaves strict
closure/#13/#17 untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfx.api import Simulation  # noqa: E402
from rfx.core.yee import EPS_0, MU_0, init_materials as init_vacuum_materials  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402
from rfx.simulation import SnapshotSpec, make_source, run as run_simulation  # noqa: E402
from rfx.sources.waveguide_port import (  # noqa: E402
    WaveguidePortConfig,
    _compute_beta,
    _compute_mode_impedance,
    _rect_dft,
    extract_waveguide_port_waves,
    waveguide_plane_positions,
)

DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
PEC_SHORT_X = 0.085
PEC_THICKNESS = 0.002
DEFAULT_CPML_LAYERS = 10
DEFAULT_NUM_PERIODS = 40.0
DEFAULT_FREQS_HZ = np.linspace(5.0e9, 7.0e9, 6)
PHASE2B_THRESHOLDS: dict[str, float] = {
    "mode_power_rel_error_max": 1.0e-6,
    "mode_cross_power_min": 0.0,
    "source_table_spectral_weighted_rel_error_max": 1.0e-3,
    "source_table_spectral_rel_error_max": 1.0e-2,
    "source_table_valid_bin_min_e_spec_fraction": 1.0e-4,
    # Diagnostic materiality threshold for amplitude |a-/a+|.  This is not a
    # strict-closure gate; the row records the corresponding power ratio too.
    "source_backward_forward_mag_max": 5.0e-2,
    "closed_energy_drift_max": 5.0e-3,
    "cpml_energy_balance_residual_max": 1.0e-2,
    "window_energy_fraction_min": 1.0e-3,
    "pec_phase_error_deg_max": 5.0,
}
PHASE2B_REQUIRED_FIELDS = (
    "stage_id",
    "invariant",
    "hypothesis",
    "physical_expected",
    "observed",
    "threshold",
    "threshold_rationale",
    "geometry_scope",
    "control_type",
    "requires_passed_stages",
    "upstream_blocker",
    "pass",
    "blocks_next_stage",
)

Status = Literal["ok", "control", "skipped", "error"]
PHASE_SHIFT_CONVENTION = "gamma_ref=gamma_origin*exp(+2j*beta*x_ref)"


@dataclass(frozen=True)
class GammaFitResult:
    """Least-squares two-wave fit result for one or more frequencies."""

    a_plus: np.ndarray
    a_minus: np.ndarray
    gamma: np.ndarray
    residual_norm: np.ndarray
    condition: np.ndarray
    rank: np.ndarray

    @property
    def gamma_mag(self) -> np.ndarray:
        return np.abs(self.gamma)

    @property
    def gamma_phase_deg(self) -> np.ndarray:
        return np.rad2deg(np.angle(self.gamma))


@dataclass(frozen=True)
class OracleRow:
    """One machine-readable diagnostic row."""

    case: str
    method: str
    status: Status = "ok"
    metrics: dict[str, Any] = field(default_factory=dict)
    verdict_hint: str = "diagnostic_only"
    skip_reason: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "case": self.case,
            "method": self.method,
            "status": self.status,
            "verdict_hint": self.verdict_hint,
        }
        row.update(_json_safe(self.metrics))
        if self.skip_reason:
            row["skip_reason"] = self.skip_reason
        return row


@dataclass(frozen=True)
class OracleCaseConfig:
    """Physical WR-90 oracle case settings."""

    freqs_hz: np.ndarray = field(default_factory=lambda: DEFAULT_FREQS_HZ.copy())
    cpml_layers: int = DEFAULT_CPML_LAYERS
    num_periods: float = DEFAULT_NUM_PERIODS
    pec_short_x: float | None = PEC_SHORT_X
    monitor_x_m: tuple[float, ...] = (0.030, 0.045, 0.060)
    source_x_m: float = PORT_LEFT_X
    short_type: str = "internal_mask"
    waveform: str = "modulated_gaussian"
    mode_profile: str = "discrete"
    dx: float | None = None


def phase2b_schema_errors(row: OracleRow) -> list[str]:
    """Return missing required Phase 2B evidence fields for one row."""

    if not row.case.startswith("phase2b_"):
        return []
    payload = row.to_jsonable()
    return [key for key in PHASE2B_REQUIRED_FIELDS if key not in payload]


def validate_phase2b_rows(rows: Iterable[OracleRow]) -> list[str]:
    """Validate Phase 2B rows before artifact emission or tests interpret them."""

    errors: list[str] = []
    for row in rows:
        for missing in phase2b_schema_errors(row):
            errors.append(f"{row.case}: missing {missing}")
    return errors


def _phase2b_row(
    *,
    case: str,
    method: str,
    stage_id: str,
    invariant: str,
    hypothesis: str,
    physical_expected: str,
    observed: dict[str, Any],
    threshold: dict[str, Any] | float | int | None,
    threshold_rationale: str,
    geometry_scope: str,
    control_type: str = "positive",
    requires_passed_stages: Iterable[str] = (),
    upstream_blocker: str | None = None,
    passed: bool = True,
    blocks_next_stage: bool = False,
    status: Status = "ok",
    extra_metrics: dict[str, Any] | None = None,
    verdict_hint: str = "phase2b_physics_ladder_diagnostic_only",
    skip_reason: str | None = None,
) -> OracleRow:
    """Build a schema-complete Phase 2B physics-ladder evidence row."""

    if not case.startswith("phase2b_"):
        raise ValueError("Phase 2B row cases must start with 'phase2b_'")
    metrics: dict[str, Any] = {
        "stage_id": stage_id,
        "invariant": invariant,
        "hypothesis": hypothesis,
        "physical_expected": physical_expected,
        "observed": observed,
        "threshold": threshold,
        "threshold_rationale": threshold_rationale,
        "geometry_scope": geometry_scope,
        "control_type": control_type,
        "requires_passed_stages": list(requires_passed_stages),
        "upstream_blocker": upstream_blocker,
        "pass": bool(passed),
        "blocks_next_stage": bool(blocks_next_stage),
        "phase2b_thresholds_pre_registered": True,
        "strict_closure_claimed": False,
        "issues_13_17_resolved": False,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return OracleRow(case, method, status, metrics, verdict_hint, skip_reason)


def _phase2b_blocked_row(
    *,
    case: str,
    method: str,
    stage_id: str,
    invariant: str,
    hypothesis: str,
    requires_passed_stages: Iterable[str],
    upstream_blocker: str,
) -> OracleRow:
    return _phase2b_row(
        case=case,
        method=method,
        stage_id=stage_id,
        invariant=invariant,
        hypothesis=hypothesis,
        physical_expected="not_interpreted_until_upstream_physics_invariant_passes",
        observed={"blocked": True},
        threshold=None,
        threshold_rationale="stage-blocking rule from Phase 2B plan",
        geometry_scope="wr90_oracle_geometry",
        control_type="none",
        requires_passed_stages=requires_passed_stages,
        upstream_blocker=upstream_blocker,
        passed=False,
        blocks_next_stage=True,
        status="skipped",
        verdict_hint="blocked_by_upstream_physics_no_s11_interpretation",
        skip_reason=f"blocked by upstream invariant: {upstream_blocker}",
    )


def _phase2b_first_blocker(rows: Iterable[OracleRow]) -> str | None:
    for row in rows:
        payload = row.to_jsonable()
        if row.case.startswith("phase2b_") and payload.get("blocks_next_stage"):
            return str(payload.get("stage_id") or row.case)
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _freqs_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.center_freq_hz is not None:
        return np.asarray([float(args.center_freq_hz)], dtype=float)
    if args.n_freqs <= 1:
        return np.asarray([float(args.freq_min_hz)], dtype=float)
    return np.linspace(float(args.freq_min_hz), float(args.freq_max_hz), int(args.n_freqs))


def fit_two_wave_line(
    monitor_x_m: np.ndarray | Iterable[float],
    modal_voltage_dft: np.ndarray,
    beta: np.ndarray | complex | float,
) -> GammaFitResult:
    """Fit ``u(x)=a+ exp(-jβx)+a- exp(+jβx)`` with 3+ monitor planes.

    ``modal_voltage_dft`` may be shape ``(n_planes,)`` for one frequency or
    ``(n_planes, n_freqs)`` for a band.  ``beta`` is scalar or ``(n_freqs,)``.
    The returned ``gamma`` is ``a_minus/a_plus`` at the global coordinate
    origin used by ``monitor_x_m``; its magnitude is independent of origin.
    """

    x = np.asarray(list(monitor_x_m), dtype=float)
    u = np.asarray(modal_voltage_dft, dtype=np.complex128)
    if u.ndim == 1:
        u = u[:, None]
    if u.ndim != 2:
        raise ValueError("modal_voltage_dft must be 1-D or 2-D")
    if u.shape[0] != x.size:
        raise ValueError(
            f"monitor_x_m has {x.size} planes but modal_voltage_dft has {u.shape[0]} rows"
        )
    if x.size < 2:
        raise ValueError("at least two monitor planes are required")

    beta_arr = np.asarray(beta, dtype=np.complex128)
    if beta_arr.ndim == 0:
        beta_arr = np.full((u.shape[1],), beta_arr.item(), dtype=np.complex128)
    if beta_arr.shape != (u.shape[1],):
        raise ValueError(f"beta shape {beta_arr.shape} does not match n_freqs={u.shape[1]}")

    a_plus = np.empty(u.shape[1], dtype=np.complex128)
    a_minus = np.empty(u.shape[1], dtype=np.complex128)
    gamma = np.empty(u.shape[1], dtype=np.complex128)
    residual = np.empty(u.shape[1], dtype=np.float64)
    cond = np.empty(u.shape[1], dtype=np.float64)
    rank = np.empty(u.shape[1], dtype=np.int64)

    for idx, beta_i in enumerate(beta_arr):
        design = np.column_stack((np.exp(-1j * beta_i * x), np.exp(+1j * beta_i * x)))
        solution, residuals, rank_i, singular_values = np.linalg.lstsq(design, u[:, idx], rcond=None)
        a_plus[idx], a_minus[idx] = solution
        gamma[idx] = a_minus[idx] / a_plus[idx] if abs(a_plus[idx]) > 1e-30 else np.nan + 1j * np.nan
        if residuals.size:
            residual[idx] = float(np.sqrt(residuals[0]) / max(np.linalg.norm(u[:, idx]), 1e-30))
        else:
            residual[idx] = float(np.linalg.norm(design @ solution - u[:, idx]) / max(np.linalg.norm(u[:, idx]), 1e-30))
        cond[idx] = float(np.inf if singular_values[-1] == 0 else singular_values[0] / singular_values[-1])
        rank[idx] = int(rank_i)

    return GammaFitResult(a_plus, a_minus, gamma, residual, cond, rank)


def solve_ref_free_gamma(
    monitor_x_m: np.ndarray | Iterable[float],
    modal_voltage_dft: np.ndarray,
    beta: np.ndarray | complex | float,
) -> GammaFitResult:
    """Public wrapper used by tests and the physical oracle rows."""

    return fit_two_wave_line(monitor_x_m, modal_voltage_dft, beta)


def shift_gamma_to_reference_plane(
    gamma_fit_origin: np.ndarray | complex,
    beta: np.ndarray | complex | float,
    x_ref_m: float,
) -> np.ndarray:
    """Shift LS-fitted reflection coefficient phase to a reporting plane.

    ``fit_two_wave_line`` solves ``u(x)=a+ exp(-jβx)+a- exp(+jβx)`` in
    global coordinates, so ``a-/a+`` is referenced to x=0.  Moving the
    reflection coefficient to plane ``x_ref`` multiplies by
    ``exp(+2j β x_ref)``.  Magnitude is invariant; phase comparisons are not.
    """

    return np.asarray(gamma_fit_origin, dtype=np.complex128) * np.exp(
        +2j * np.asarray(beta, dtype=np.complex128) * float(x_ref_m)
    )


def _safe_divide_complex(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.complex128)
    denominator = np.asarray(denominator, dtype=np.complex128)
    safe_denominator = np.where(
        np.abs(denominator) > 1.0e-30,
        denominator,
        np.ones_like(denominator),
    )
    return numerator / safe_denominator


def assemble_current_normalization_formulas(
    *,
    a_inc_ref_drive: np.ndarray,
    b_ref_drive: np.ndarray,
    b_dev_drive: np.ndarray,
    a_inc_dev_drive: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Assemble candidate diagonal S11 formulas from recorded wave terms."""

    a_ref = np.asarray(a_inc_ref_drive, dtype=np.complex128)
    b_ref = np.asarray(b_ref_drive, dtype=np.complex128)
    b_dev = np.asarray(b_dev_drive, dtype=np.complex128)
    if a_inc_dev_drive is None:
        a_dev = a_ref
    else:
        a_dev = np.asarray(a_inc_dev_drive, dtype=np.complex128)
    return {
        "current_formula": _safe_divide_complex(b_dev - b_ref, a_ref),
        "no_subtraction_formula": _safe_divide_complex(b_dev, a_ref),
        "device_denominator_formula": _safe_divide_complex(b_dev - b_ref, a_dev),
    }


def _delta_summary(values: list[float | int | None]) -> float | None:
    finite_values: list[float] = []
    for value in values:
        if value is None:
            continue
        value_f = float(value)
        if np.isfinite(value_f):
            finite_values.append(value_f)
    if not finite_values:
        return None
    finite = np.asarray(finite_values, dtype=float)
    return float(np.max(finite) - np.min(finite))


def integer_cycle_lockin(
    time_series: np.ndarray,
    freq_hz: float,
    dt: float,
    start_index: int,
    n_cycles: int,
) -> complex:
    """Return normalized complex phasor from an integer-period lock-in.

    For ``A*cos(2πft+φ)`` sampled over an integer number of periods, the
    return value is approximately ``A*exp(+jφ)``.  The helper is deliberately
    script-local; physical CW sources are not productionized in this PR.
    """

    if freq_hz <= 0 or dt <= 0:
        raise ValueError("freq_hz and dt must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if n_cycles <= 0:
        raise ValueError("n_cycles must be positive")

    arr = np.asarray(time_series, dtype=np.float64)
    samples_per_period = int(round(1.0 / (float(freq_hz) * float(dt))))
    if samples_per_period <= 0:
        raise ValueError("freq_hz*dt is too large to resolve one period")
    n_samples = int(samples_per_period * n_cycles)
    end = int(start_index) + n_samples
    if end > arr.size:
        raise ValueError("requested lock-in window exceeds time_series length")
    n = np.arange(start_index, end, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * float(freq_hz) * n * float(dt))
    return complex(2.0 * np.mean(arr[start_index:end] * phase))


def _recorded_length(cfg: WaveguidePortConfig) -> int:
    return int(np.asarray(cfg.n_steps_recorded))


def _rect_dft_windowed(
    time_series: np.ndarray,
    freqs: np.ndarray,
    dt: float,
    start_index: int,
    stop_index: int,
) -> np.ndarray:
    """Script-local rectangular DFT over an explicit absolute-time window."""

    arr = np.asarray(time_series, dtype=np.float64)
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    start = int(start_index)
    stop = int(stop_index)
    if start < 0 or stop <= start or stop > arr.size:
        raise ValueError(f"invalid DFT window [{start}, {stop}) for {arr.size} samples")
    n = np.arange(start, stop, dtype=np.float64)
    samples = arr[start:stop]
    phase = np.exp(-1j * 2.0 * np.pi * freqs_arr[:, None] * float(dt) * n[None, :])
    return 2.0 * float(dt) * np.sum(samples[None, :] * phase, axis=1)


def _window_bounds(
    n_valid: int,
    label: str,
    *,
    dt: float,
    freq_hz: float | None = None,
    cycles: int = 20,
) -> tuple[int, int]:
    if n_valid <= 0:
        raise ValueError("n_valid must be positive")
    if label == "full_record":
        return 0, int(n_valid)
    if label == "late_half":
        return int(n_valid) // 2, int(n_valid)
    if label == "late_quarter":
        return (3 * int(n_valid)) // 4, int(n_valid)
    if label == "late_fixed_cycles":
        if freq_hz is None:
            raise ValueError("late_fixed_cycles requires freq_hz")
        samples_per_period = int(round(1.0 / (float(freq_hz) * float(dt))))
        n_samples = max(1, int(samples_per_period * int(cycles)))
        if n_samples >= n_valid:
            return 0, int(n_valid)
        return int(n_valid) - n_samples, int(n_valid)
    raise ValueError(f"unknown DFT window label {label!r}")


def _window_energy_metrics(
    time_series: np.ndarray,
    start_index: int,
    stop_index: int,
    n_valid: int,
) -> dict[str, float]:
    arr = np.asarray(time_series, dtype=np.float64)
    full = arr[: int(n_valid)]
    window = arr[int(start_index) : int(stop_index)]
    full_energy = float(np.sum(full * full))
    window_energy = float(np.sum(window * window))
    return {
        "window_energy": window_energy,
        "full_record_energy": full_energy,
        "window_energy_fraction_vs_full": float(window_energy / max(full_energy, 1.0e-300)),
    }


def _voltage_spectrum_at_reference_windowed(
    cfg: WaveguidePortConfig,
    window_label: str,
    *,
    center_freq_hz: float | None = None,
    fixed_cycles: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_valid = _recorded_length(cfg)
    if center_freq_hz is None:
        freqs_np = np.asarray(cfg.freqs, dtype=float)
        center_freq_hz = float(freqs_np[len(freqs_np) // 2])
    start, stop = _window_bounds(
        n_valid,
        window_label,
        dt=float(cfg.dt),
        freq_hz=center_freq_hz,
        cycles=fixed_cycles,
    )
    spectrum = _rect_dft_windowed(cfg.v_ref_t, cfg.freqs, cfg.dt, start, stop)
    metrics = {
        "window_label": window_label,
        "start_index": int(start),
        "stop_index": int(stop),
        "n_samples": int(stop - start),
        "n_valid": int(n_valid),
        "dt": float(cfg.dt),
        "fixed_cycles": int(fixed_cycles) if window_label == "late_fixed_cycles" else None,
        "center_freq_hz": float(center_freq_hz),
    }
    metrics.update(_window_energy_metrics(cfg.v_ref_t, start, stop, n_valid))
    return spectrum, metrics


def _build_wr90_two_port_sim(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int = DEFAULT_CPML_LAYERS,
    pec_short_x: float | None = PEC_SHORT_X,
    waveform: str = "modulated_gaussian",
    dx: float | None = None,
    reference_plane_left: float | None = None,
    reference_plane_right: float | None = None,
) -> Simulation:
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (float(freqs[-1]) - float(freqs[0])) / max(f0, 1.0)))
    sim_kwargs: dict[str, Any] = {
        "freq_max": max(float(freqs[-1]), f0),
        "domain": DOMAIN,
        "boundary": "cpml",
        "cpml_layers": int(cpml_layers),
    }
    if dx is not None:
        sim_kwargs["dx"] = float(dx)
    sim = Simulation(**sim_kwargs)
    if pec_short_x is not None:
        sim.add(
            Box((pec_short_x, 0.0, 0.0), (pec_short_x + PEC_THICKNESS, DOMAIN[1], DOMAIN[2])),
            material="pec",
        )
    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        mode_profile="discrete",
    )
    sim.add_waveguide_port(
        PORT_LEFT_X,
        direction="+x",
        name="left",
        reference_plane=reference_plane_left,
        **common,
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X,
        direction="-x",
        name="right",
        reference_plane=reference_plane_right,
        **common,
    )
    return sim


def _add_passive_monitor_ports(
    sim: Simulation,
    freqs: np.ndarray,
    monitor_x_m: Iterable[float],
    *,
    ref_offset: int = 3,
    probe_offset: int = 4,
    waveform: str = "modulated_gaussian",
    mode_profile: str = "discrete",
) -> None:
    grid = sim._build_grid(extra_waveguide_axes="x")
    dx = float(grid.dx)
    f0 = float(np.asarray(freqs, dtype=float).mean())
    bandwidth = max(0.2, min(0.8, (float(freqs[-1]) - float(freqs[0])) / max(f0, 1.0)))
    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        mode_profile=mode_profile,
        amplitude=0.0,
        ref_offset=ref_offset,
        probe_offset=probe_offset,
    )
    for idx, x_monitor in enumerate(monitor_x_m):
        # ``reference_plane`` is only an extraction/reporting override in the
        # production S-matrix path, so for passive monitors we place the
        # zero-amplitude source plane upstream such that cfg.ref_x samples the
        # requested monitor plane after grid snapping.
        x_source = float(x_monitor) - ref_offset * dx
        sim.add_waveguide_port(
            x_source,
            direction="+x",
            name=f"monitor_{idx}",
            **common,
        )


def build_wr90_oracle_case(case_config: OracleCaseConfig) -> dict[str, Any]:
    """Build a one-run active-source plus passive-monitor oracle case."""

    freqs = np.asarray(case_config.freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (float(freqs[-1]) - float(freqs[0])) / max(f0, 1.0)))
    sim_kwargs: dict[str, Any] = {
        "freq_max": max(float(freqs[-1]), f0),
        "domain": DOMAIN,
        "boundary": "cpml",
        "cpml_layers": int(case_config.cpml_layers),
    }
    if case_config.dx is not None:
        sim_kwargs["dx"] = float(case_config.dx)
    sim = Simulation(**sim_kwargs)
    if case_config.pec_short_x is not None:
        sim.add(
            Box(
                (case_config.pec_short_x, 0.0, 0.0),
                (case_config.pec_short_x + PEC_THICKNESS, DOMAIN[1], DOMAIN[2]),
            ),
            material="pec",
        )

    active_common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform=case_config.waveform,
        mode_profile=case_config.mode_profile,
    )
    sim.add_waveguide_port(case_config.source_x_m, direction="+x", name="active_source", **active_common)
    _add_passive_monitor_ports(
        sim,
        freqs,
        case_config.monitor_x_m,
        waveform=case_config.waveform,
        mode_profile=case_config.mode_profile,
    )

    entries = list(sim._waveguide_ports)
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask_wg, _, _ = sim._assemble_materials(grid)
    if pec_mask_wg is not None:
        # Match the current waveguide S-matrix compatibility path so the
        # oracle compares against the same internal-mask representation.
        base_materials = base_materials._replace(sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
    _, debye, lorentz = sim._init_dispersion(base_materials, grid.dt, debye_spec, lorentz_spec)
    n_steps = grid.num_timesteps(num_periods=case_config.num_periods)

    cfgs: list[WaveguidePortConfig] = [sim._build_waveguide_port_config(entry, grid, jnp.asarray(freqs), n_steps) for entry in entries]
    # Ensure all passive monitors are really passive even if future API default
    # changes alter amplitude handling.
    cfgs = [cfg if idx == 0 else cfg._replace(src_amp=0.0) for idx, cfg in enumerate(cfgs)]

    return {
        "grid": grid,
        "materials": base_materials,
        "debye": debye,
        "lorentz": lorentz,
        "cfgs": cfgs,
        "n_steps": n_steps,
        "common_run_kw": dict(
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
            periodic=None,
        ),
    }


def _run_waveguide_case(case: dict[str, Any]) -> list[WaveguidePortConfig]:
    result = run_simulation(
        case["grid"],
        case["materials"],
        case["n_steps"],
        debye=case["debye"],
        lorentz=case["lorentz"],
        waveguide_ports=case["cfgs"],
        **case["common_run_kw"],
    )
    return list(result.waveguide_ports or ())


def _voltage_spectrum_at_reference(cfg: WaveguidePortConfig) -> np.ndarray:
    return np.asarray(_rect_dft(cfg.v_ref_t, cfg.freqs, cfg.dt, cfg.n_steps_recorded), dtype=np.complex128)


def _reference_free_fit_from_cfgs(
    case_config: OracleCaseConfig,
    final_cfgs: list[WaveguidePortConfig],
    *,
    dft_window_label: str | None = None,
    fixed_cycles: int = 20,
) -> dict[str, Any]:
    if len(final_cfgs) < 4:
        raise RuntimeError(f"expected active source + >=3 passive monitors, got {len(final_cfgs)} configs")
    monitor_cfgs = final_cfgs[1:]
    monitor_x = np.asarray([waveguide_plane_positions(cfg)["reference"] for cfg in monitor_cfgs], dtype=float)
    window_metrics: list[dict[str, Any]] = []
    if dft_window_label is None:
        spectra = np.vstack([_voltage_spectrum_at_reference(cfg) for cfg in monitor_cfgs])
        dft_window = "rect_full_record"
    else:
        spectra_list: list[np.ndarray] = []
        for cfg in monitor_cfgs:
            spectrum, metrics = _voltage_spectrum_at_reference_windowed(
                cfg,
                dft_window_label,
                fixed_cycles=fixed_cycles,
            )
            spectra_list.append(spectrum)
            window_metrics.append(metrics)
        spectra = np.vstack(spectra_list)
        dft_window = dft_window_label
    beta = np.asarray(
        _compute_beta(
            monitor_cfgs[0].freqs,
            monitor_cfgs[0].f_cutoff,
            dt=monitor_cfgs[0].dt,
            dx=monitor_cfgs[0].dx,
        ),
        dtype=np.complex128,
    )
    fit = solve_ref_free_gamma(monitor_x, spectra, beta)
    freqs = np.asarray(monitor_cfgs[0].freqs, dtype=float)
    phase_reference_plane_m = float(case_config.source_x_m)
    gamma_at_phase_reference = shift_gamma_to_reference_plane(
        fit.gamma, beta, phase_reference_plane_m
    )
    return {
        "final_cfgs": final_cfgs,
        "monitor_cfgs": monitor_cfgs,
        "monitor_x": monitor_x,
        "spectra": spectra,
        "beta": beta,
        "fit": fit,
        "freqs": freqs,
        "phase_reference_plane_m": phase_reference_plane_m,
        "gamma_at_phase_reference": gamma_at_phase_reference,
        "dft_window": dft_window,
        "window_metrics": window_metrics,
    }


def _reference_free_fit_details(
    case_config: OracleCaseConfig,
    *,
    dft_window_label: str | None = None,
    fixed_cycles: int = 20,
) -> dict[str, Any]:
    case = build_wr90_oracle_case(case_config)
    final_cfgs = _run_waveguide_case(case)
    details = _reference_free_fit_from_cfgs(
        case_config,
        final_cfgs,
        dft_window_label=dft_window_label,
        fixed_cycles=fixed_cycles,
    )
    details["case"] = case
    return details


def _build_current_two_run_case(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
    reference_plane_left: float | None = None,
    reference_plane_right: float | None = None,
) -> dict[str, Any]:
    sim = _build_wr90_two_port_sim(
        freqs_hz,
        cpml_layers=cpml_layers,
        pec_short_x=PEC_SHORT_X,
        dx=dx,
        reference_plane_left=reference_plane_left,
        reference_plane_right=reference_plane_right,
    )
    entries = list(sim._waveguide_ports)
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask_wg, _, _ = sim._assemble_materials(grid)
    if pec_mask_wg is not None:
        base_materials = base_materials._replace(sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
    _, debye, lorentz = sim._init_dispersion(base_materials, grid.dt, debye_spec, lorentz_spec)
    n_steps = grid.num_timesteps(num_periods=num_periods)
    freqs = jnp.asarray(freqs_hz)
    cfgs = [sim._build_waveguide_port_config(entry, grid, freqs, n_steps) for entry in entries]
    ref_shifts: list[float] = []
    reference_planes: list[float] = []
    plane_metadata: list[dict[str, float]] = []
    for entry, cfg in zip(entries, cfgs):
        planes = waveguide_plane_positions(cfg)
        desired_ref = entry.reference_plane if entry.reference_plane is not None else planes["source"]
        ref_shifts.append(float(desired_ref - planes["reference"]))
        reference_planes.append(float(desired_ref))
        plane_metadata.append({key: float(value) for key, value in planes.items()})
    return {
        "sim": sim,
        "entries": entries,
        "grid": grid,
        "materials": base_materials,
        "ref_materials": init_vacuum_materials(grid.shape),
        "debye": debye,
        "lorentz": lorentz,
        "cfgs": cfgs,
        "n_steps": n_steps,
        "ref_shifts": ref_shifts,
        "reference_planes": reference_planes,
        "plane_metadata": plane_metadata,
        "common_run_kw": dict(
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
            periodic=None,
        ),
    }


def _reset_waveguide_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
    zeros_t = jnp.zeros_like(cfg.v_probe_t)
    return cfg._replace(
        src_amp=cfg.src_amp if drive_enabled else 0.0,
        v_probe_t=zeros_t,
        v_ref_t=zeros_t,
        i_probe_t=zeros_t,
        i_ref_t=zeros_t,
        v_inc_t=zeros_t,
        n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
    )


def _current_case_plane_metadata(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
) -> dict[str, float]:
    case = _build_current_two_run_case(
        freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx
    )
    return case["plane_metadata"][0]


def run_reference_free_case(case_config: OracleCaseConfig, *, case_name: str) -> OracleRow:
    started = time.perf_counter()
    try:
        details = _reference_free_fit_details(case_config)
        fit: GammaFitResult = details["fit"]
        freqs = details["freqs"]
        beta = details["beta"]
        monitor_x = details["monitor_x"]
        s11_mag = fit.gamma_mag
        gamma_at_phase_reference = details["gamma_at_phase_reference"]
        phase_reference_plane_m = float(details["phase_reference_plane_m"])
        metrics = {
            "freq_hz": float(freqs[len(freqs) // 2]),
            "freqs_hz": freqs,
            "s11_mag": float(s11_mag[len(s11_mag) // 2]),
            "s11_phase_deg": float(fit.gamma_phase_deg[len(s11_mag) // 2]),
            "mean_abs_s11": float(np.nanmean(s11_mag)),
            "min_abs_s11": float(np.nanmin(s11_mag)),
            "max_abs_s11": float(np.nanmax(s11_mag)),
            "mag_error_pct": float(100.0 * abs(np.nanmean(s11_mag) - 1.0)),
            "gamma_fit_origin": fit.gamma,
            "gamma_fit_origin_phase_deg": fit.gamma_phase_deg,
            "phase_reference_plane_m": phase_reference_plane_m,
            "gamma_at_phase_reference_plane": gamma_at_phase_reference,
            "gamma_at_phase_reference_plane_phase_deg": np.rad2deg(np.angle(gamma_at_phase_reference)),
            "phase_shift_convention": PHASE_SHIFT_CONVENTION,
            "beta_rad_per_m": beta,
            "fit_residual": float(np.nanmean(fit.residual_norm)),
            "fit_cond": float(np.nanmax(fit.condition)),
            "fit_rank_min": int(np.nanmin(fit.rank)),
            "source": "analytic_te10_current",
            "short_type": case_config.short_type,
            "cpml_layers": int(case_config.cpml_layers),
            "num_periods": float(case_config.num_periods),
            "monitor_backend": "passive_waveguide_ref_voltage",
            "monitor_distances_m": monitor_x - float(case_config.source_x_m),
            "monitor_x_m": monitor_x,
            "requested_monitor_x_m": np.asarray(case_config.monitor_x_m, dtype=float),
            "source_short_distance_m": None if case_config.pec_short_x is None else float(case_config.pec_short_x - case_config.source_x_m),
            "monitor_short_distances_m": None if case_config.pec_short_x is None else float(case_config.pec_short_x) - monitor_x,
            "beta_type": "yee_discrete",
            "dft_window": "rect_full_record",
            "dft_type": "post_scan_rect_dft",
            "elapsed_s": float(time.perf_counter() - started),
        }
        return OracleRow(case_name, "ref_free_multiplane", "ok", metrics, "A_D_or_C3_if_ref_free_deficit_persists")
    except Exception as exc:
        return OracleRow(
            case_name,
            "ref_free_multiplane",
            "error",
            {"elapsed_s": float(time.perf_counter() - started)},
            "diagnostic_failed_no_closure_claim",
            f"{type(exc).__name__}: {exc}",
        )


def run_current_2run_baseline(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
    reference_plane_left: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    try:
        sim = _build_wr90_two_port_sim(
            freqs_hz,
            cpml_layers=cpml_layers,
            pec_short_x=PEC_SHORT_X,
            dx=dx,
            reference_plane_left=reference_plane_left,
        )
        result = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=True)
        s = np.asarray(result.s_params, dtype=np.complex128)
        port_idx = {name: idx for idx, name in enumerate(result.port_names)}
        s11 = s[port_idx["left"], port_idx["left"], :]
        mag = np.abs(s11)
        freqs = np.asarray(result.freqs, dtype=float)
        metrics = {
            "freq_hz": float(freqs[len(freqs) // 2]),
            "freqs_hz": freqs,
            "s11_mag": float(mag[len(mag) // 2]),
            "s11_phase_deg": float(np.rad2deg(np.angle(s11[len(s11) // 2]))),
            "s11_complex": s11,
            "s11_abs": mag,
            "mean_abs_s11": float(np.mean(mag)),
            "min_abs_s11": float(np.min(mag)),
            "max_abs_s11": float(np.max(mag)),
            "mag_error_pct": float(100.0 * abs(np.mean(mag) - 1.0)),
            "source": "analytic_te10_current",
            "short_type": "internal_mask",
            "cpml_layers": int(cpml_layers),
            "num_periods": float(num_periods),
            "monitor_backend": "production_two_run_waveguide_s_matrix",
            "reference_planes_m": np.asarray(result.reference_planes, dtype=float),
            "reference_plane_left_m": float(result.reference_planes[port_idx["left"]]),
            "beta_type": "production_current",
            "dft_window": "rect_full_record",
            "dft_type": "current_2run_post_scan_rect_dft",
            "elapsed_s": float(time.perf_counter() - started),
        }
        return OracleRow(
            "baseline_current_2run_internal_mask_current_cpml",
            "current_2run",
            "ok",
            metrics,
            "B_or_C_if_ref_free_good",
        )
    except Exception as exc:
        return OracleRow(
            "baseline_current_2run_internal_mask_current_cpml",
            "current_2run",
            "error",
            {"elapsed_s": float(time.perf_counter() - started), "cpml_layers": int(cpml_layers)},
            "diagnostic_failed_no_closure_claim",
            f"{type(exc).__name__}: {exc}",
        )


def synthetic_least_squares_control() -> OracleRow:
    freqs = np.asarray([5.0e9, 6.0e9, 7.0e9])
    beta = np.asarray([95.0, 125.0, 150.0])
    x = np.asarray([0.024, 0.036, 0.049, 0.063])
    a_plus = np.asarray([1.0 + 0.0j, 0.8 + 0.1j, 1.1 - 0.05j])
    gamma_true = np.asarray([0.96 * np.exp(1j * 0.2), 0.98 * np.exp(-1j * 0.4), 1.02 * np.exp(1j * 0.1)])
    a_minus = gamma_true * a_plus
    samples = np.column_stack([
        a_plus[i] * np.exp(-1j * beta[i] * x) + a_minus[i] * np.exp(+1j * beta[i] * x)
        for i in range(freqs.size)
    ])
    fit = solve_ref_free_gamma(x, samples, beta)
    err = np.max(np.abs(fit.gamma - gamma_true))
    return OracleRow(
        "synthetic_ref_free_least_squares_control",
        "ref_free_multiplane_control",
        "control",
        {
            "freqs_hz": freqs,
            "gamma_error_max": float(err),
            "fit_residual_max": float(np.max(fit.residual_norm)),
            "fit_cond_max": float(np.max(fit.condition)),
            "monitor_backend": "synthetic_two_wave_line",
            "beta_type": "synthetic_discrete_beta",
            "dft_window": "not_applicable_synthetic",
            "dft_type": "not_applicable_synthetic",
        },
        "control_only_no_physical_closure",
    )


def cw_lockin_control_row() -> OracleRow:
    freq = 10.0e9
    samples_per_period = 40
    dt = 1.0 / (freq * samples_per_period)
    phase = 0.37
    amp = 1.25
    n = np.arange(samples_per_period * 12, dtype=float)
    signal = amp * np.cos(2.0 * np.pi * freq * n * dt + phase)
    phasor = integer_cycle_lockin(signal, freq, dt, start_index=samples_per_period * 2, n_cycles=8)
    return OracleRow(
        "cw_lockin_synthetic_control",
        "cw_integer_period_lockin_control",
        "control",
        {
            "freq_hz": freq,
            "target_amplitude": amp,
            "target_phase_rad": phase,
            "recovered_amplitude": float(abs(phasor)),
            "recovered_phase_rad": float(np.angle(phasor)),
            "amplitude_error": float(abs(abs(phasor) - amp)),
            "phase_error_rad": float(abs(np.angle(phasor * np.exp(-1j * phase)))),
            "warmup_cycles": 2,
            "lockin_cycles": 8,
            "dft_window": "integer_period_lockin",
            "dft_type": "cw_lockin_normalized_phasor",
        },
        "control_only_cw_helper_ready",
    )


def run_source_purity_empty_line_sweep(freqs_hz: np.ndarray, *, cpml_layers: int, num_periods: float, dx: float | None = None) -> OracleRow:
    monitors = (0.025, 0.040, 0.055, 0.070)
    cfg = OracleCaseConfig(
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        cpml_layers=int(cpml_layers),
        num_periods=float(num_periods),
        pec_short_x=None,
        monitor_x_m=monitors,
        short_type="empty_guide",
        dx=dx,
    )
    row = run_reference_free_case(cfg, case_name="source_purity_empty_line_sweep")
    if row.status != "ok":
        return row
    metrics = dict(row.metrics)
    # For an empty guide the fitted backward/forward ratio is a practical
    # line-purity proxy; phase slope/absolute amplitude remain diagnostic.
    metrics["source_purity_metric"] = "empty_guide_ref_free_backward_forward_ratio"
    metrics["gamma_interpretation"] = "empty_guide backward/forward ratio, not PEC-short S11"
    metrics["residual_field_norm"] = metrics.get("fit_residual")
    metrics["poynting_flux"] = None
    return OracleRow(row.case, "source_purity_line_sweep", row.status, metrics, "A_if_distance_dependent_impurity_seen")


def _summarize_formula(name: str, values: np.ndarray, ref_free_gamma: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.complex128)
    ref_free_gamma = np.asarray(ref_free_gamma, dtype=np.complex128)
    mag = np.abs(values)
    ref_mag = np.abs(ref_free_gamma)
    return {
        "formula": name,
        "mean_abs": float(np.nanmean(mag)),
        "min_abs": float(np.nanmin(mag)),
        "max_abs": float(np.nanmax(mag)),
        "mean_mag_delta_vs_ref_free": float(np.nanmean(mag - ref_mag)),
        "max_abs_mag_delta_vs_ref_free": float(np.nanmax(np.abs(mag - ref_mag))),
    }


def run_current_norm_dissection(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    try:
        case = _build_current_two_run_case(
            freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx
        )
        cfgs = tuple(case["cfgs"])
        ref_shifts = case["ref_shifts"]
        drive_idx = 0
        ref_cfgs = [_reset_waveguide_cfg(cfg, idx == drive_idx) for idx, cfg in enumerate(cfgs)]
        dev_cfgs = [_reset_waveguide_cfg(cfg, idx == drive_idx) for idx, cfg in enumerate(cfgs)]
        ref_result = run_simulation(
            case["grid"],
            case["ref_materials"],
            case["n_steps"],
            debye=None,
            lorentz=None,
            waveguide_ports=ref_cfgs,
            **case["common_run_kw"],
        )
        dev_result = run_simulation(
            case["grid"],
            case["materials"],
            case["n_steps"],
            debye=case["debye"],
            lorentz=case["lorentz"],
            waveguide_ports=dev_cfgs,
            **case["common_run_kw"],
        )
        ref_final = list(ref_result.waveguide_ports or ())
        dev_final = list(dev_result.waveguide_ports or ())
        a_inc_ref, b_ref = extract_waveguide_port_waves(
            ref_final[drive_idx], ref_shift=ref_shifts[drive_idx]
        )
        a_inc_dev, b_dev = extract_waveguide_port_waves(
            dev_final[drive_idx], ref_shift=ref_shifts[drive_idx]
        )
        formulas = assemble_current_normalization_formulas(
            a_inc_ref_drive=np.asarray(a_inc_ref),
            b_ref_drive=np.asarray(b_ref),
            b_dev_drive=np.asarray(b_dev),
            a_inc_dev_drive=np.asarray(a_inc_dev),
        )

        ref_free_cfg = OracleCaseConfig(
            freqs_hz=np.asarray(freqs_hz, dtype=float),
            cpml_layers=cpml_layers,
            num_periods=num_periods,
            dx=dx,
        )
        ref_free = _reference_free_fit_details(ref_free_cfg)
        phase_reference_plane_m = float(case["reference_planes"][drive_idx])
        ref_free_gamma_at_current_plane = shift_gamma_to_reference_plane(
            ref_free["fit"].gamma, ref_free["beta"], phase_reference_plane_m
        )
        current_formula = formulas["current_formula"]
        freqs = np.asarray(freqs_hz, dtype=float)
        metrics = {
            "freqs_hz": freqs,
            "cpml_layers": int(cpml_layers),
            "num_periods": float(num_periods),
            "drive_port": "left",
            "ref_shift_m": float(ref_shifts[drive_idx]),
            "reference_plane_m": phase_reference_plane_m,
            "plane_metadata": case["plane_metadata"][drive_idx],
            "a_inc_ref_drive": np.asarray(a_inc_ref, dtype=np.complex128),
            "b_ref_drive": np.asarray(b_ref, dtype=np.complex128),
            "a_inc_dev_drive": np.asarray(a_inc_dev, dtype=np.complex128),
            "b_dev_drive": np.asarray(b_dev, dtype=np.complex128),
            "current_formula": current_formula,
            "no_subtraction_formula": formulas["no_subtraction_formula"],
            "device_denominator_formula": formulas["device_denominator_formula"],
            "formula_summaries": [
                _summarize_formula(name, values, ref_free_gamma_at_current_plane)
                for name, values in formulas.items()
            ],
            "reference_free_gamma_fit_origin": ref_free["fit"].gamma,
            "reference_free_gamma_at_current_reference_plane": ref_free_gamma_at_current_plane,
            "reference_free_abs": np.abs(ref_free_gamma_at_current_plane),
            "current_abs": np.abs(current_formula),
            "current_minus_ref_free_abs_delta": np.abs(current_formula) - np.abs(ref_free_gamma_at_current_plane),
            "current_phase_deg": np.rad2deg(np.angle(current_formula)),
            "reference_free_phase_deg": np.rad2deg(np.angle(ref_free_gamma_at_current_plane)),
            "phase_delta_deg": np.rad2deg(np.angle(current_formula / ref_free_gamma_at_current_plane)),
            "phase_reference_plane_m": phase_reference_plane_m,
            "phase_shift_convention": PHASE_SHIFT_CONVENTION,
            "gamma_fit_origin_phase_deg": ref_free["fit"].gamma_phase_deg,
            "gamma_at_phase_reference_plane_phase_deg": np.rad2deg(np.angle(ref_free_gamma_at_current_plane)),
            "dft_window": "rect_full_record",
            "dft_type": "current_2run_dissection_post_scan_rect_dft",
            "elapsed_s": float(time.perf_counter() - started),
        }
        return OracleRow(
            "current_norm_dissection_pec_short",
            "current_2run_formula_dissection",
            "ok",
            metrics,
            "select_BC_formula_before_any_production_fix",
        )
    except Exception as exc:
        return OracleRow(
            "current_norm_dissection_pec_short",
            "current_2run_formula_dissection",
            "error",
            {"elapsed_s": float(time.perf_counter() - started)},
            "diagnostic_failed_no_closure_claim",
            f"{type(exc).__name__}: {exc}",
        )


def _extract_sweep_triplet(row: OracleRow, independent_value: float | int) -> dict[str, Any]:
    payload = row.to_jsonable()
    return {
        "value": independent_value,
        "status": row.status,
        "mean_abs_s11": payload.get("mean_abs_s11"),
        "min_abs_s11": payload.get("min_abs_s11"),
        "max_abs_s11": payload.get("max_abs_s11"),
        "fit_residual": payload.get("fit_residual"),
        "skip_reason": payload.get("skip_reason"),
    }


def run_period_sweep_current_vs_ref_free(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    periods: tuple[float, ...] = (20.0, 40.0),
    skipped_periods: tuple[float, ...] = (80.0,),
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    current_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    for period in periods:
        current = run_current_2run_baseline(
            freqs_hz, cpml_layers=cpml_layers, num_periods=period, dx=dx
        )
        ref_free = run_reference_free_case(
            OracleCaseConfig(
                freqs_hz=np.asarray(freqs_hz, dtype=float),
                cpml_layers=cpml_layers,
                num_periods=period,
                dx=dx,
            ),
            case_name=f"ref_free_period_{period:g}",
        )
        current_rows.append(_extract_sweep_triplet(current, period))
        ref_rows.append(_extract_sweep_triplet(ref_free, period))
    metrics = {
        "periods_run": list(periods),
        "periods_skipped": list(skipped_periods),
        "current_2run": current_rows,
        "reference_free": ref_rows,
        "current_delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in current_rows]),
        "current_delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in current_rows]),
        "ref_free_delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in ref_rows]),
        "ref_free_delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in ref_rows]),
        "dft_window": "rect_full_record",
        "elapsed_s": float(time.perf_counter() - started),
    }
    return OracleRow(
        "period_sweep_current_vs_ref_free",
        "period_sensitivity",
        "ok",
        metrics,
        "C1_if_current_moves_and_ref_free_stable",
    )


def run_pml_sweep_current_vs_ref_free(
    freqs_hz: np.ndarray,
    *,
    num_periods: float,
    layers: tuple[int, ...] = (8, 10, 12),
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    current_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    for layer in layers:
        current = run_current_2run_baseline(
            freqs_hz, cpml_layers=layer, num_periods=num_periods, dx=dx
        )
        ref_free = run_reference_free_case(
            OracleCaseConfig(
                freqs_hz=np.asarray(freqs_hz, dtype=float),
                cpml_layers=layer,
                num_periods=num_periods,
                dx=dx,
            ),
            case_name=f"ref_free_cpml_{layer}",
        )
        current_rows.append(_extract_sweep_triplet(current, layer))
        ref_rows.append(_extract_sweep_triplet(ref_free, layer))
    metrics = {
        "cpml_layers_run": list(layers),
        "current_2run": current_rows,
        "reference_free": ref_rows,
        "current_delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in current_rows]),
        "current_delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in current_rows]),
        "ref_free_delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in ref_rows]),
        "ref_free_delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in ref_rows]),
        "elapsed_s": float(time.perf_counter() - started),
    }
    return OracleRow(
        "pml_sweep_current_vs_ref_free_layers_8_10_12",
        "cpml_sensitivity",
        "ok",
        metrics,
        "C2_if_current_moves_and_ref_free_stable",
    )


def run_reference_plane_sweep_current_2run(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    try:
        planes = _current_case_plane_metadata(
            freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx
        )
        ref_free = run_reference_free_case(
            OracleCaseConfig(
                freqs_hz=np.asarray(freqs_hz, dtype=float),
                cpml_layers=cpml_layers,
                num_periods=num_periods,
                dx=dx,
            ),
            case_name="ref_free_reference_plane_sweep_guardrail",
        )
        ref_free_mean = ref_free.to_jsonable().get("mean_abs_s11")
        choices = [
            ("snapped_source_plane", planes["source"]),
            ("recorded_reference_plane", planes["reference"]),
            ("recorded_probe_plane", planes["probe"]),
        ]
        rows = []
        for label, plane in choices:
            row = run_current_2run_baseline(
                freqs_hz,
                cpml_layers=cpml_layers,
                num_periods=num_periods,
                dx=dx,
                reference_plane_left=float(plane),
            )
            payload = row.to_jsonable()
            mean = payload.get("mean_abs_s11")
            rows.append({
                "plane_label": label,
                "reference_plane_m": float(plane),
                "status": row.status,
                "mean_abs_s11": mean,
                "min_abs_s11": payload.get("min_abs_s11"),
                "max_abs_s11": payload.get("max_abs_s11"),
                "delta_mean_abs_vs_ref_free": None
                if mean is None or ref_free_mean is None
                else float(mean - ref_free_mean),
                "reference_planes_m": payload.get("reference_planes_m"),
            })
        metrics = {
            "plane_choices": rows,
            "default_plane_metadata": planes,
            "reference_free_mean_abs_s11": ref_free_mean,
            "phase_shift_convention": PHASE_SHIFT_CONVENTION,
            "comparison_scope": "magnitude_primary_phase_requires_common_plane",
            "elapsed_s": float(time.perf_counter() - started),
        }
        return OracleRow(
            "reference_plane_sweep_current_2run",
            "reference_plane_sensitivity",
            "ok",
            metrics,
            "B_reference_plane_if_one_choice_tracks_ref_free",
        )
    except Exception as exc:
        return OracleRow(
            "reference_plane_sweep_current_2run",
            "reference_plane_sensitivity",
            "error",
            {"elapsed_s": float(time.perf_counter() - started)},
            "diagnostic_failed_no_closure_claim",
            f"{type(exc).__name__}: {exc}",
        )


def _source_params_for_freqs(freqs_hz: np.ndarray) -> dict[str, float]:
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (float(freqs[-1]) - float(freqs[0])) / max(f0, 1.0)))
    return {"source_f0_hz": f0, "source_bandwidth": bandwidth}


def _fit_summary(details: dict[str, Any]) -> dict[str, Any]:
    fit: GammaFitResult = details["fit"]
    mag = fit.gamma_mag
    summary: dict[str, Any] = {
        "freqs_hz": np.asarray(details["freqs"], dtype=float),
        "gamma_abs": mag,
        "mean_abs_s11": float(np.nanmean(mag)),
        "min_abs_s11": float(np.nanmin(mag)),
        "max_abs_s11": float(np.nanmax(mag)),
        "fit_residual": float(np.nanmean(fit.residual_norm)),
        "fit_residual_max": float(np.nanmax(fit.residual_norm)),
        "fit_cond": float(np.nanmax(fit.condition)),
        "fit_cond_max": float(np.nanmax(fit.condition)),
        "fit_rank_min": int(np.nanmin(fit.rank)),
        "a_plus_abs_min": float(np.nanmin(np.abs(fit.a_plus))),
    }
    window_metrics = list(details.get("window_metrics") or [])
    if window_metrics:
        fractions = [float(item["window_energy_fraction_vs_full"]) for item in window_metrics]
        summary.update({
            "window_label": window_metrics[0].get("window_label"),
            "start_index": int(window_metrics[0].get("start_index", 0)),
            "stop_index": int(window_metrics[0].get("stop_index", 0)),
            "n_samples": int(window_metrics[0].get("n_samples", 0)),
            "dt": float(window_metrics[0].get("dt", 0.0)),
            "per_plane_window_energy": [float(item["window_energy"]) for item in window_metrics],
            "per_plane_window_energy_fraction_vs_full": fractions,
            "window_energy_fraction_vs_full_min": float(np.min(fractions)),
        })
    return summary


def _valid_fit_flags(summary: dict[str, Any], *, min_energy_fraction: float = 1.0e-3) -> dict[str, Any]:
    reasons: list[str] = []
    if int(summary.get("fit_rank_min", 0)) < 2:
        reasons.append("fit_rank_lt_2")
    if not np.isfinite(float(summary.get("fit_cond_max", np.inf))):
        reasons.append("fit_cond_non_finite")
    if float(summary.get("a_plus_abs_min", 0.0)) <= 1.0e-30:
        reasons.append("a_plus_near_zero")
    energy_fraction = summary.get("window_energy_fraction_vs_full_min")
    if energy_fraction is not None and float(energy_fraction) < min_energy_fraction:
        reasons.append("window_energy_too_small")
    return {"valid": not reasons, "invalid_reasons": reasons}


def _phase2a_sweep_entry(row: OracleRow, value: float | int | str) -> dict[str, Any]:
    payload = row.to_jsonable()
    return {
        "value": value,
        "status": row.status,
        "mean_abs_s11": payload.get("mean_abs_s11"),
        "min_abs_s11": payload.get("min_abs_s11"),
        "max_abs_s11": payload.get("max_abs_s11"),
        "fit_residual": payload.get("fit_residual"),
        "skip_reason": payload.get("skip_reason"),
    }


def phase2a_late_window_synthetic_control() -> OracleRow:
    freq = 2.0e9
    dt = 1.0 / (freq * 64)
    n = np.arange(64 * 10)
    signal = np.cos(2.0 * np.pi * freq * n * dt + 0.2)
    full = _rect_dft_windowed(signal, np.asarray([freq]), dt, 0, signal.size)
    reference = np.asarray(_rect_dft(jnp.asarray(signal), jnp.asarray([freq]), dt, signal.size))
    return OracleRow(
        "phase2a_late_window_synthetic_control",
        "late_window_dft_control",
        "control",
        {
            "freq_hz": freq,
            "full_window_matches_rect_dft_abs_error": float(np.max(np.abs(full - reference))),
            "window_labels_covered": ["full_record", "late_half", "late_quarter", "late_fixed_cycles"],
        },
        "control_only_window_helper_ready",
    )


def run_phase2a_ref_free_late_window_sweep(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    monitor_x_m: tuple[float, ...],
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    try:
        cfg = OracleCaseConfig(
            freqs_hz=np.asarray(freqs_hz, dtype=float),
            cpml_layers=cpml_layers,
            num_periods=num_periods,
            monitor_x_m=monitor_x_m,
            dx=dx,
        )
        case = build_wr90_oracle_case(cfg)
        final_cfgs = _run_waveguide_case(case)
        entries: list[dict[str, Any]] = []
        full_mean = None
        full_min = None
        for label in ("full_record", "late_half", "late_quarter", "late_fixed_cycles"):
            details = _reference_free_fit_from_cfgs(cfg, final_cfgs, dft_window_label=label)
            summary = _fit_summary(details)
            summary.update(_valid_fit_flags(summary))
            if label == "full_record":
                full_mean = summary["mean_abs_s11"]
                full_min = summary["min_abs_s11"]
            summary["delta_mean_abs_gamma_vs_full_record"] = None if full_mean is None else float(summary["mean_abs_s11"] - full_mean)
            summary["delta_min_abs_gamma_vs_full_record"] = None if full_min is None else float(summary["min_abs_s11"] - full_min)
            entries.append(summary)
        metrics = {
            "windows": entries,
            "cpml_layers": int(cpml_layers),
            "num_periods": float(num_periods),
            "monitor_x_m": list(monitor_x_m),
            "materiality_threshold_mean": 0.02,
            "materiality_threshold_min": 0.05,
            "source": "analytic_te10_current",
            "short_type": "internal_mask",
            "elapsed_s": float(time.perf_counter() - started),
        }
        return OracleRow(
            "phase2a_ref_free_late_window_sweep",
            "ref_free_late_window_sensitivity",
            "ok",
            metrics,
            "dft_window_settling_if_valid_late_window_moves_gamma",
        )
    except Exception as exc:
        return OracleRow(
            "phase2a_ref_free_late_window_sweep",
            "ref_free_late_window_sensitivity",
            "error",
            {"elapsed_s": float(time.perf_counter() - started)},
            "diagnostic_failed_no_closure_claim",
            f"{type(exc).__name__}: {exc}",
        )


def run_phase2a_ref_free_period_sweep(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    periods: tuple[float, ...] = (40.0, 80.0),
    skipped_periods: tuple[float, ...] = (120.0,),
    monitor_x_m: tuple[float, ...],
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    ref_rows = []
    for period in periods:
        row = run_reference_free_case(
            OracleCaseConfig(
                freqs_hz=np.asarray(freqs_hz, dtype=float),
                cpml_layers=cpml_layers,
                num_periods=period,
                monitor_x_m=monitor_x_m,
                dx=dx,
            ),
            case_name=f"phase2a_ref_free_period_{period:g}",
        )
        ref_rows.append(_phase2a_sweep_entry(row, period))
    return OracleRow(
        "phase2a_ref_free_period_sweep_40_80_120",
        "ref_free_period_sensitivity",
        "ok",
        {
            "periods_run": list(periods),
            "periods_skipped": list(skipped_periods),
            "reference_free": ref_rows,
            "ref_free_delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in ref_rows]),
            "ref_free_delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in ref_rows]),
            "elapsed_s": float(time.perf_counter() - started),
        },
        "dft_window_settling_if_ref_free_moves_with_period",
    )


def run_phase2a_current_period_sweep(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    periods: tuple[float, ...] = (40.0, 80.0),
    skipped_periods: tuple[float, ...] = (120.0,),
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    current_rows = []
    for period in periods:
        row = run_current_2run_baseline(
            freqs_hz,
            cpml_layers=cpml_layers,
            num_periods=period,
            dx=dx,
        )
        current_rows.append(_phase2a_sweep_entry(row, period))
    return OracleRow(
        "phase2a_current_period_sweep_40_80_120",
        "current_2run_period_sensitivity",
        "ok",
        {
            "periods_run": list(periods),
            "periods_skipped": list(skipped_periods),
            "current_2run": current_rows,
            "current_delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in current_rows]),
            "current_delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in current_rows]),
            "elapsed_s": float(time.perf_counter() - started),
        },
        "current_window_settling_if_current_moves_with_period",
    )


def run_phase2a_center_lockin_ref_free(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    monitor_x_m: tuple[float, ...],
    warmup_cycles: int,
    lockin_cycles: int,
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    try:
        cfg = OracleCaseConfig(
            freqs_hz=np.asarray(freqs_hz, dtype=float),
            cpml_layers=cpml_layers,
            num_periods=num_periods,
            monitor_x_m=monitor_x_m,
            dx=dx,
        )
        case = build_wr90_oracle_case(cfg)
        final_cfgs = _run_waveguide_case(case)
        monitor_cfgs = final_cfgs[1:]
        center_freq = float(np.asarray(freqs_hz, dtype=float)[len(freqs_hz) // 2])
        phasors: list[complex] = []
        energy_metrics: list[dict[str, float]] = []
        for mon_cfg in monitor_cfgs:
            n_valid = _recorded_length(mon_cfg)
            samples_per_period = int(round(1.0 / (center_freq * float(mon_cfg.dt))))
            n_samples = int(samples_per_period * int(lockin_cycles))
            start = max(int(samples_per_period * int(warmup_cycles)), n_valid - n_samples)
            stop = start + n_samples
            if stop > n_valid:
                start = max(0, n_valid - n_samples)
            phasors.append(integer_cycle_lockin(mon_cfg.v_ref_t, center_freq, mon_cfg.dt, start, lockin_cycles))
            energy_metrics.append(_window_energy_metrics(mon_cfg.v_ref_t, start, min(start + n_samples, n_valid), n_valid))
        monitor_x = np.asarray([waveguide_plane_positions(cfg_i)["reference"] for cfg_i in monitor_cfgs], dtype=float)
        beta = np.asarray(
            _compute_beta(
                jnp.asarray([center_freq]),
                monitor_cfgs[0].f_cutoff,
                dt=monitor_cfgs[0].dt,
                dx=monitor_cfgs[0].dx,
            ),
            dtype=np.complex128,
        )
        fit = solve_ref_free_gamma(monitor_x, np.asarray(phasors)[:, None], beta)
        fractions = [float(item["window_energy_fraction_vs_full"]) for item in energy_metrics]
        metrics = {
            "freq_hz": center_freq,
            "warmup_cycles": int(warmup_cycles),
            "lockin_cycles": int(lockin_cycles),
            "per_plane_phasor_abs": np.abs(np.asarray(phasors)),
            "per_plane_window_energy": [float(item["window_energy"]) for item in energy_metrics],
            "per_plane_window_energy_fraction_vs_full": fractions,
            "window_energy_fraction_vs_full_min": float(np.min(fractions)),
            "gamma_abs": fit.gamma_mag,
            "mean_abs_s11": float(np.nanmean(fit.gamma_mag)),
            "min_abs_s11": float(np.nanmin(fit.gamma_mag)),
            "max_abs_s11": float(np.nanmax(fit.gamma_mag)),
            "fit_residual": float(np.nanmean(fit.residual_norm)),
            "fit_cond": float(np.nanmax(fit.condition)),
            "fit_rank_min": int(np.nanmin(fit.rank)),
            "a_plus_abs_min": float(np.nanmin(np.abs(fit.a_plus))),
            "lockin_valid": False,
            "lockin_invalid_reasons": ["pulse_tail_monochromaticity_not_proven"],
            "elapsed_s": float(time.perf_counter() - started),
        }
        return OracleRow(
            "phase2a_center_lockin_ref_free",
            "multi_monitor_lockin_ref_free",
            "skipped",
            metrics,
            "explicit_skip_no_closure_claim",
            "physical lock-in on the pulse tail is computed but not treated as valid evidence until monochromaticity/SNR is proven",
        )
    except Exception as exc:
        return OracleRow(
            "phase2a_center_lockin_ref_free",
            "multi_monitor_lockin_ref_free",
            "skipped",
            {"elapsed_s": float(time.perf_counter() - started)},
            "explicit_skip_no_closure_claim",
            f"lock-in diagnostic unavailable: {type(exc).__name__}: {exc}",
        )


def run_phase2a_ref_free_same_run_center_vs_band(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    monitor_x_m: tuple[float, ...],
    dx: float | None = None,
) -> OracleRow:
    details = _reference_free_fit_details(
        OracleCaseConfig(
            freqs_hz=np.asarray(freqs_hz, dtype=float),
            cpml_layers=cpml_layers,
            num_periods=num_periods,
            monitor_x_m=monitor_x_m,
            dx=dx,
        )
    )
    fit: GammaFitResult = details["fit"]
    mag = fit.gamma_mag
    center_idx = len(mag) // 2
    return OracleRow(
        "phase2a_ref_free_same_run_center_vs_band",
        "same_run_ref_free_center_vs_band",
        "ok",
        {
            "freqs_hz": np.asarray(freqs_hz, dtype=float),
            "per_frequency_abs_s11": mag,
            "center_freq_hz": float(np.asarray(freqs_hz, dtype=float)[center_idx]),
            "center_abs_s11": float(mag[center_idx]),
            "band_mean_abs_s11": float(np.nanmean(mag)),
            "band_min_abs_s11": float(np.nanmin(mag)),
            "same_run_center_minus_band_mean_abs": float(mag[center_idx] - np.nanmean(mag)),
            "source_waveform_changed": False,
            **_source_params_for_freqs(np.asarray(freqs_hz, dtype=float)),
            "fit_residual": float(np.nanmean(fit.residual_norm)),
            "fit_cond": float(np.nanmax(fit.condition)),
        },
        "band_artifact_if_same_run_center_tracks_unity_and_band_does_not",
    )


def run_phase2a_current_same_run_center_vs_band(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
) -> OracleRow:
    row = run_current_2run_baseline(freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx)
    payload = row.to_jsonable()
    mag = np.asarray(payload.get("s11_abs", []), dtype=float)
    center_idx = len(mag) // 2
    return OracleRow(
        "phase2a_current_same_run_center_vs_band",
        "same_run_current_center_vs_band",
        row.status,
        {
            "freqs_hz": payload.get("freqs_hz"),
            "per_frequency_abs_s11": mag,
            "center_freq_hz": None if not len(mag) else float(np.asarray(payload.get("freqs_hz"), dtype=float)[center_idx]),
            "center_abs_s11": None if not len(mag) else float(mag[center_idx]),
            "band_mean_abs_s11": None if not len(mag) else float(np.nanmean(mag)),
            "band_min_abs_s11": None if not len(mag) else float(np.nanmin(mag)),
            "same_run_center_minus_band_mean_abs": None if not len(mag) else float(mag[center_idx] - np.nanmean(mag)),
            "source_waveform_changed": False,
            **_source_params_for_freqs(np.asarray(freqs_hz, dtype=float)),
        },
        "current_band_artifact_if_same_run_center_tracks_reference_free",
        payload.get("skip_reason"),
    )


def run_phase2a_center_only_rerun_labeled(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    monitor_x_m: tuple[float, ...],
    dx: float | None = None,
) -> OracleRow:
    center_freq = float(np.asarray(freqs_hz, dtype=float)[len(freqs_hz) // 2])
    center_freqs = np.asarray([center_freq], dtype=float)
    ref_row = run_reference_free_case(
        OracleCaseConfig(
            freqs_hz=center_freqs,
            cpml_layers=cpml_layers,
            num_periods=num_periods,
            monitor_x_m=monitor_x_m,
            dx=dx,
        ),
        case_name="phase2a_ref_free_center_only_rerun",
    )
    current_row = run_current_2run_baseline(
        center_freqs,
        cpml_layers=cpml_layers,
        num_periods=num_periods,
        dx=dx,
    )
    default_source = _source_params_for_freqs(np.asarray(freqs_hz, dtype=float))
    center_source = _source_params_for_freqs(center_freqs)
    return OracleRow(
        "phase2a_center_only_rerun_labeled_source_changed",
        "center_only_rerun_source_changed",
        "ok" if ref_row.status == "ok" and current_row.status == "ok" else "error",
        {
            "analysis_freqs_hz": center_freqs,
            "default_band_source": default_source,
            "center_only_source": center_source,
            "source_waveform_changed": bool(default_source != center_source),
            "reference_free": ref_row.to_jsonable(),
            "current_2run": current_row.to_jsonable(),
        },
        "source_waveform_or_band_changed_signal_not_pure_band_evidence",
    )


def run_phase2a_cpml_sweep_late_window_ref_free(
    freqs_hz: np.ndarray,
    *,
    num_periods: float,
    layers: tuple[int, ...] = (8, 10, 12),
    monitor_x_m: tuple[float, ...],
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    entries = []
    for layer in layers:
        details = _reference_free_fit_details(
            OracleCaseConfig(
                freqs_hz=np.asarray(freqs_hz, dtype=float),
                cpml_layers=layer,
                num_periods=num_periods,
                monitor_x_m=monitor_x_m,
                dx=dx,
            ),
            dft_window_label="late_half",
        )
        summary = _fit_summary(details)
        summary.update(_valid_fit_flags(summary))
        summary["cpml_layers"] = int(layer)
        entries.append(summary)
    return OracleRow(
        "phase2a_cpml_sweep_late_window_ref_free",
        "ref_free_cpml_sensitivity_late_window",
        "ok",
        {
            "cpml_layers_run": list(layers),
            "reference_free": entries,
            "ref_free_delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in entries]),
            "ref_free_delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in entries]),
            "dft_window": "late_half",
            "elapsed_s": float(time.perf_counter() - started),
        },
        "cpml_coupling_if_ref_free_moves_with_cpml_after_window_control",
    )


def run_phase2a_monitor_spacing_sweep_ref_free(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    monitor_sets = {
        "default": (0.030, 0.045, 0.060),
        "farther_from_source": (0.040, 0.055, 0.070),
        "farther_from_short": (0.022, 0.032, 0.042),
    }
    entries = []
    for label, monitors in monitor_sets.items():
        row = run_reference_free_case(
            OracleCaseConfig(
                freqs_hz=np.asarray(freqs_hz, dtype=float),
                cpml_layers=cpml_layers,
                num_periods=num_periods,
                monitor_x_m=monitors,
                dx=dx,
            ),
            case_name=f"phase2a_monitor_spacing_{label}",
        )
        payload = row.to_jsonable()
        entries.append({
            "spacing_label": label,
            "monitor_x_m": list(monitors),
            "status": row.status,
            "mean_abs_s11": payload.get("mean_abs_s11"),
            "min_abs_s11": payload.get("min_abs_s11"),
            "max_abs_s11": payload.get("max_abs_s11"),
            "fit_residual": payload.get("fit_residual"),
            "source_to_left_domain_m": PORT_LEFT_X,
            "short_to_right_domain_m": DOMAIN[0] - (PEC_SHORT_X + PEC_THICKNESS),
            "monitor_to_source_m": [float(x - PORT_LEFT_X) for x in monitors],
            "monitor_to_short_m": [float(PEC_SHORT_X - x) for x in monitors],
            "source_short_distance_m": float(PEC_SHORT_X - PORT_LEFT_X),
            "cpml_layers": int(cpml_layers),
        })
    return OracleRow(
        "phase2a_monitor_spacing_sweep_ref_free",
        "ref_free_monitor_spacing_sensitivity",
        "ok",
        {
            "spacing_rows": entries,
            "delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in entries]),
            "delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in entries]),
            "elapsed_s": float(time.perf_counter() - started),
        },
        "source_monitor_coupling_if_ref_free_changes_with_monitor_spacing",
    )


def run_phase2a_source_short_distance_sweep_ref_free(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    monitor_x_m: tuple[float, ...],
    dx: float | None = None,
) -> OracleRow:
    started = time.perf_counter()
    short_positions = (0.075, PEC_SHORT_X)
    entries = []
    for short_x in short_positions:
        row = run_reference_free_case(
            OracleCaseConfig(
                freqs_hz=np.asarray(freqs_hz, dtype=float),
                cpml_layers=cpml_layers,
                num_periods=num_periods,
                pec_short_x=short_x,
                monitor_x_m=monitor_x_m,
                dx=dx,
            ),
            case_name=f"phase2a_source_short_distance_{short_x:.3f}",
        )
        payload = row.to_jsonable()
        entries.append({
            "short_x_m": float(short_x),
            "status": row.status,
            "mean_abs_s11": payload.get("mean_abs_s11"),
            "min_abs_s11": payload.get("min_abs_s11"),
            "max_abs_s11": payload.get("max_abs_s11"),
            "fit_residual": payload.get("fit_residual"),
            "source_short_distance_m": float(short_x - PORT_LEFT_X),
            "short_to_right_domain_m": float(DOMAIN[0] - (short_x + PEC_THICKNESS)),
            "monitor_to_short_m": [float(short_x - x) for x in monitor_x_m],
            "monitor_to_source_m": [float(x - PORT_LEFT_X) for x in monitor_x_m],
            "source_to_left_domain_m": PORT_LEFT_X,
            "cpml_layers": int(cpml_layers),
        })
    return OracleRow(
        "phase2a_source_short_distance_sweep_ref_free",
        "ref_free_source_short_distance_sensitivity",
        "ok",
        {
            "short_distance_rows": entries,
            "delta_mean_abs_s11": _delta_summary([row.get("mean_abs_s11") for row in entries]),
            "delta_min_abs_s11": _delta_summary([row.get("min_abs_s11") for row in entries]),
            "elapsed_s": float(time.perf_counter() - started),
        },
        "source_or_mask_physics_if_ref_free_changes_with_source_short_distance",
    )


def _phase2b_waveguide_cfg(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
) -> tuple[Any, WaveguidePortConfig]:
    sim = _build_wr90_two_port_sim(
        np.asarray(freqs_hz, dtype=float),
        cpml_layers=cpml_layers,
        pec_short_x=None,
        dx=dx,
    )
    grid = sim._build_grid()
    entry = list(sim._waveguide_ports)[0]
    n_steps = grid.num_timesteps(num_periods=num_periods)
    cfg = sim._build_waveguide_port_config(entry, grid, jnp.asarray(freqs_hz), n_steps)
    return grid, cfg


def run_phase2b_guardrail_row() -> OracleRow:
    proc = subprocess.run(
        ["git", "diff", "--", "rfx"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    diff_sha = hashlib.sha256(proc.stdout).hexdigest()
    empty_sha = hashlib.sha256(b"").hexdigest()
    clean = proc.returncode == 0 and diff_sha == empty_sha
    return _phase2b_row(
        case="phase2b_guardrail",
        method="production_diff_and_scope_guardrail",
        stage_id="stage0_guardrail",
        invariant="diagnostic_branch_must_not_modify_production_rfx",
        hypothesis="production_diff_or_strict_closure_claim_would_make_evidence_non_diagnostic",
        physical_expected="production rfx diff is empty; receive extractor frozen; eigensource not productionized; #13/#17 unresolved",
        observed={
            "rfx_diff_sha256": diff_sha,
            "empty_diff_sha256": empty_sha,
            "git_diff_returncode": int(proc.returncode),
            "receive_extractor_frozen": True,
            "eigensource_productionized": False,
            "issues_13_17_resolved": False,
        },
        threshold={"rfx_diff_sha256": empty_sha},
        threshold_rationale="Phase 2B is diagnostic-first; production files only open at the explicit Stage 7 invariant-backed gate",
        geometry_scope="production_path",
        control_type="none",
        passed=clean,
        blocks_next_stage=not clean,
    )


def run_phase2b_mode_ledger(freqs_hz: np.ndarray, *, cpml_layers: int, num_periods: float, dx: float | None = None) -> list[OracleRow]:
    try:
        grid, cfg = _phase2b_waveguide_cfg(
            freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx
        )
        d_a = np.asarray(cfg.aperture_dA)
        if d_a.size == 0:
            d_a = np.asarray(cfg.u_widths)[:, None] * np.asarray(cfg.v_widths)[None, :]
        ey = np.asarray(cfg.ey_profile, dtype=float)
        ez = np.asarray(cfg.ez_profile, dtype=float)
        hy = np.asarray(cfg.hy_profile, dtype=float)
        hz = np.asarray(cfg.hz_profile, dtype=float)
        e_power = float(np.sum((ey * ey + ez * ez) * d_a))
        cross_power = float(np.sum((ey * hz - ez * hy) * d_a))
        reversed_cross_power = float(np.sum((ey * (-hz) - ez * (-hy)) * d_a))
        beta_disc = np.asarray(_compute_beta(cfg.freqs, cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx), dtype=np.complex128)
        beta_cont = np.asarray(_compute_beta(cfg.freqs, cfg.f_cutoff, dt=0.0, dx=0.0), dtype=np.complex128)
        beta_delta = np.abs(beta_disc - beta_cont)
        phase_to_short = -2.0 * beta_disc * float(PEC_SHORT_X - cfg.reference_x_m)
        pass_mode = (
            abs(e_power - 1.0) <= PHASE2B_THRESHOLDS["mode_power_rel_error_max"]
            and cross_power > PHASE2B_THRESHOLDS["mode_cross_power_min"]
            and np.all(np.isfinite(beta_disc.real))
        )
        ledger = _phase2b_row(
            case="phase2b_stage1_discrete_te10_mode_ledger",
            method="profile_power_beta_ledger",
            stage_id="stage1_mode_ledger",
            invariant="discrete_te10_profile_power_sign_and_beta_are_self_consistent",
            hypothesis="mode_template_mismatch_or_wrong_sign_would_invalidate_source_and_pec_phase_interpretation",
            physical_expected="unit E-profile norm, positive forward cross-power, finite Yee-discrete beta",
            observed={
                "grid_shape": tuple(int(x) for x in grid.shape),
                "dx_m": float(cfg.dx),
                "dt_s": float(cfg.dt),
                "f_cutoff_hz": float(cfg.f_cutoff),
                "freqs_hz": np.asarray(freqs_hz, dtype=float),
                "mode_indices": tuple(int(x) for x in cfg.mode_indices),
                "h_offset": tuple(float(x) for x in cfg.h_offset),
                "aperture_dA_sum": float(np.sum(d_a)),
                "e_power_norm": e_power,
                "forward_cross_power": cross_power,
                "reversed_h_cross_power": reversed_cross_power,
                "beta_discrete_rad_per_m": beta_disc,
                "beta_continuum_rad_per_m": beta_cont,
                "beta_abs_delta_rad_per_m": beta_delta,
                "expected_pec_short_phase_from_ref_rad": phase_to_short,
            },
            threshold={
                "mode_power_rel_error_max": PHASE2B_THRESHOLDS["mode_power_rel_error_max"],
                "forward_cross_power_min": PHASE2B_THRESHOLDS["mode_cross_power_min"],
            },
            threshold_rationale="profile-only algebraic normalization and sign checks should close to roundoff before any field simulation is interpreted",
            geometry_scope="profile_only",
            passed=pass_mode,
            blocks_next_stage=not pass_mode,
        )
        neg_pass = reversed_cross_power < 0.0
        negative = _phase2b_row(
            case="phase2b_stage1_mode_sign_negative_control",
            method="profile_sign_negative_control",
            stage_id="stage1_mode_sign_negative_control",
            invariant="forward_poynting_sign_gate_detects_reversed_h_template",
            hypothesis="if the gate accepts reversed H signs then downstream source/PEC phase checks can be numerically fit artifacts",
            physical_expected="deliberately reversed H template must produce negative cross-power and therefore fail forward-power physics",
            observed={
                "reversed_h_cross_power": reversed_cross_power,
                "negative_control_failed_forward_physics": neg_pass,
            },
            threshold={"reversed_h_cross_power_max": 0.0},
            threshold_rationale="sign-control check: reversed H must be rejected exactly by the cross-product orientation",
            geometry_scope="profile_only",
            control_type="negative",
            passed=neg_pass,
            blocks_next_stage=False,
            status="control",
            verdict_hint="negative_control_only_no_closure_claim",
        )
        return [ledger, negative]
    except Exception as exc:
        return [
            _phase2b_row(
                case="phase2b_stage1_discrete_te10_mode_ledger",
                method="profile_power_beta_ledger",
                stage_id="stage1_mode_ledger",
                invariant="discrete_te10_profile_power_sign_and_beta_are_self_consistent",
                hypothesis="mode_ledger_exception_blocks_physical_interpretation",
                physical_expected="profile ledger can be constructed without simulation",
                observed={"exception": f"{type(exc).__name__}: {exc}"},
                threshold={"mode_power_rel_error_max": PHASE2B_THRESHOLDS["mode_power_rel_error_max"]},
                threshold_rationale="no downstream stage may proceed without mode metadata",
                geometry_scope="profile_only",
                passed=False,
                blocks_next_stage=True,
                status="error",
                verdict_hint="diagnostic_failed_no_closure_claim",
                skip_reason=f"{type(exc).__name__}: {exc}",
            )
        ]


def _source_table_spectral_residual(cfg: WaveguidePortConfig) -> dict[str, Any]:
    e_table = np.asarray(cfg.e_inc_table, dtype=float)
    h_table = np.asarray(cfg.h_inc_table, dtype=float)
    if e_table.size <= 1 or h_table.size <= 1:
        return {
            "tables_present": False,
            "spectral_rel_error_max": None,
            "spectral_rel_error_mean": None,
            "valid_bin_count": 0,
        }
    e_spec = np.fft.rfft(e_table)
    h_spec = np.fft.rfft(h_table)
    freqs_spec = np.fft.rfftfreq(e_table.size, d=float(cfg.dt))
    beta = np.asarray(
        _compute_beta(jnp.asarray(freqs_spec), cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx),
        dtype=np.complex128,
    )
    z_mode = np.asarray(
        _compute_mode_impedance(jnp.asarray(freqs_spec), cfg.f_cutoff, cfg.mode_type, dt=cfg.dt, dx=cfg.dx),
        dtype=np.complex128,
    )
    omega = 2.0 * np.pi * freqs_spec
    expected = np.exp(+1j * beta * 0.5 * float(cfg.dx) + 1j * omega * 0.5 * float(cfg.dt)) / z_mode
    valid_bin_floor = max(
        float(np.max(np.abs(e_spec))) * PHASE2B_THRESHOLDS["source_table_valid_bin_min_e_spec_fraction"],
        1.0e-18,
    )
    valid = (
        (freqs_spec > float(cfg.f_cutoff))
        & np.isfinite(expected.real)
        & np.isfinite(expected.imag)
        & (np.abs(e_spec) > valid_bin_floor)
    )
    if not np.any(valid):
        rel = np.asarray([np.inf])
        weighted_rel = float(np.inf)
    else:
        target = e_spec[valid] * expected[valid]
        diff = h_spec[valid] - target
        rel = np.abs(diff) / np.maximum(np.abs(target), 1.0e-30)
        weighted_rel = float(np.linalg.norm(diff) / max(np.linalg.norm(target), 1.0e-30))
    return {
        "tables_present": True,
        "table_length": int(e_table.size),
        "e_table_energy": float(np.sum(e_table * e_table)),
        "h_table_energy": float(np.sum(h_table * h_table)),
        "valid_bin_min_e_spec_fraction": PHASE2B_THRESHOLDS["source_table_valid_bin_min_e_spec_fraction"],
        "valid_bin_count": int(np.count_nonzero(valid)),
        "spectral_rel_error_max": float(np.max(rel)),
        "spectral_rel_error_mean": float(np.mean(rel)),
        "spectral_rel_error_weighted": weighted_rel,
        "source_f_cutoff_hz": float(cfg.f_cutoff),
    }


def run_phase2b_source_table_consistency(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
    requires_passed_stages: Iterable[str] = ("stage1_mode_ledger",),
    upstream_blocker: str | None = None,
) -> OracleRow:
    if upstream_blocker:
        return _phase2b_blocked_row(
            case="phase2b_stage2_source_table_pairing",
            method="source_eh_table_spectral_consistency",
            stage_id="stage2_source_table_pairing",
            invariant="source_e_and_h_tables_are_discrete_impedance_and_half_step_consistent",
            hypothesis="source_pairing_cannot_be_interpreted_until_mode_ledger_passes",
            requires_passed_stages=requires_passed_stages,
            upstream_blocker=upstream_blocker,
        )
    try:
        _, cfg = _phase2b_waveguide_cfg(
            freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx
        )
        residual = _source_table_spectral_residual(cfg)
        err_max = residual.get("spectral_rel_error_max")
        err_weighted = residual.get("spectral_rel_error_weighted")
        passed = (
            residual.get("tables_present") is True
            and int(residual.get("valid_bin_count") or 0) > 0
            and err_max is not None
            and err_weighted is not None
            and float(err_weighted) <= PHASE2B_THRESHOLDS["source_table_spectral_weighted_rel_error_max"]
            and float(err_max) <= PHASE2B_THRESHOLDS["source_table_spectral_rel_error_max"]
        )
        return _phase2b_row(
            case="phase2b_stage2_source_table_pairing",
            method="source_eh_table_spectral_consistency",
            stage_id="stage2_source_table_pairing",
            invariant="source_e_and_h_tables_are_discrete_impedance_and_half_step_consistent",
            hypothesis="incorrect_h_half_step_or_impedance_pairing_would_make_the_source_non_physical",
            physical_expected="h_inc spectrum equals e_inc spectrum times discrete TE impedance inverse and Yee half-cell/half-step phase",
            observed=residual,
            threshold={
                "spectral_rel_error_weighted_max": PHASE2B_THRESHOLDS["source_table_spectral_weighted_rel_error_max"],
                "spectral_rel_error_max": PHASE2B_THRESHOLDS["source_table_spectral_rel_error_max"],
            },
            threshold_rationale="the H table is generated deterministically from the E table; energy-weighted residual is primary to avoid low-amplitude FFT-tail artifacts, while max residual remains bounded",
            geometry_scope="production_path",
            requires_passed_stages=requires_passed_stages,
            passed=passed,
            blocks_next_stage=not passed,
        )
    except Exception as exc:
        return _phase2b_row(
            case="phase2b_stage2_source_table_pairing",
            method="source_eh_table_spectral_consistency",
            stage_id="stage2_source_table_pairing",
            invariant="source_e_and_h_tables_are_discrete_impedance_and_half_step_consistent",
            hypothesis="source_pairing_exception_blocks_physical_interpretation",
            physical_expected="source tables can be reconstructed without field simulation",
            observed={"exception": f"{type(exc).__name__}: {exc}"},
            threshold={
                "spectral_rel_error_weighted_max": PHASE2B_THRESHOLDS["source_table_spectral_weighted_rel_error_max"],
                "spectral_rel_error_max": PHASE2B_THRESHOLDS["source_table_spectral_rel_error_max"],
            },
            threshold_rationale="source pairing is prerequisite evidence for source-purity rows",
            geometry_scope="production_path",
            requires_passed_stages=requires_passed_stages,
            passed=False,
            blocks_next_stage=True,
            status="error",
            verdict_hint="diagnostic_failed_no_closure_claim",
            skip_reason=f"{type(exc).__name__}: {exc}",
        )


def run_phase2b_source_purity(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    dx: float | None = None,
    requires_passed_stages: Iterable[str] = ("stage1_mode_ledger", "stage2_source_table_pairing"),
    upstream_blocker: str | None = None,
) -> OracleRow:
    if upstream_blocker:
        return _phase2b_blocked_row(
            case="phase2b_stage2_source_empty_guide_purity",
            method="empty_guide_backward_forward_oracle",
            stage_id="stage2_source_empty_guide_purity",
            invariant="empty_guide_source_launch_is_forward_dominant",
            hypothesis="source_purity_cannot_be_interpreted_until_source_pairing_passes",
            requires_passed_stages=requires_passed_stages,
            upstream_blocker=upstream_blocker,
        )
    row = run_source_purity_empty_line_sweep(
        freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx
    )
    payload = row.to_jsonable()
    mean_mag = payload.get("mean_abs_s11")
    max_mag = payload.get("max_abs_s11")
    fit_residual = payload.get("fit_residual")
    passed = row.status == "ok" and max_mag is not None and float(max_mag) <= PHASE2B_THRESHOLDS["source_backward_forward_mag_max"]
    return _phase2b_row(
        case="phase2b_stage2_source_empty_guide_purity",
        method="empty_guide_backward_forward_oracle",
        stage_id="stage2_source_empty_guide_purity",
        invariant="empty_guide_source_launch_is_forward_dominant",
        hypothesis="a_backward_over_a_forward_in_empty_guide_above_threshold_indicates_source_or_absorber_coupled_impurity",
        physical_expected="empty-guide reference-free fit should find near-zero backward wave away from the source",
        observed={
            "mean_backward_forward_mag": mean_mag,
            "max_backward_forward_mag": max_mag,
            "max_backward_forward_power_ratio": None if max_mag is None else float(max_mag) ** 2,
            "fit_residual": fit_residual,
            "source_purity_row_status": row.status,
            "source_purity_skip_reason": payload.get("skip_reason"),
        },
        threshold={
            "max_backward_forward_mag": PHASE2B_THRESHOLDS["source_backward_forward_mag_max"],
            "max_backward_forward_power_ratio": PHASE2B_THRESHOLDS["source_backward_forward_mag_max"] ** 2,
        },
        threshold_rationale="pre-registered diagnostic materiality threshold for source purity; this is stricter than accepting visible backward launch but not a #13/#17 closure threshold",
        geometry_scope="wr90_oracle_geometry",
        requires_passed_stages=requires_passed_stages,
        passed=passed,
        blocks_next_stage=not passed,
        status="ok" if row.status == "ok" else "skipped",
        extra_metrics={"source_purity_underlying_row": payload},
        verdict_hint="source_physics_or_cpml_coupling_if_backward_wave_exceeds_threshold",
        skip_reason=None if row.status == "ok" else payload.get("skip_reason"),
    )


def _field_energy_from_snapshots(snapshots: dict[str, Any], idx: int, dx: float) -> float:
    e_sum = 0.0
    h_sum = 0.0
    for comp in ("ex", "ey", "ez"):
        arr = np.asarray(snapshots[comp][idx], dtype=float)
        e_sum += float(np.sum(arr * arr))
    for comp in ("hx", "hy", "hz"):
        arr = np.asarray(snapshots[comp][idx], dtype=float)
        h_sum += float(np.sum(arr * arr))
    return 0.5 * (EPS_0 * e_sum + MU_0 * h_sum) * float(dx) ** 3


def run_phase2b_closed_domain_energy(
    *,
    upstream_blocker: str | None,
    requires_passed_stages: Iterable[str] = (
        "stage1_mode_ledger",
        "stage2_source_table_pairing",
        "stage2_source_empty_guide_purity",
    ),
) -> OracleRow:
    if upstream_blocker:
        return _phase2b_blocked_row(
            case="phase2b_stage3_closed_domain_energy",
            method="lossless_pec_energy_snapshot_control",
            stage_id="stage3_closed_domain_energy",
            invariant="lossless_closed_domain_energy_is_conserved_after_source_shutoff",
            hypothesis="energy_conservation_cannot_be_interpreted_until_source_physics_passes",
            requires_passed_stages=requires_passed_stages,
            upstream_blocker=upstream_blocker,
        )
    try:
        sim = Simulation(freq_max=2.0e9, domain=(0.024, 0.024, 0.024), boundary="pec", dx=0.004)
        grid = sim._build_grid()
        materials = init_vacuum_materials(grid.shape)
        n_steps = 64
        stop_step = 24
        freq = 0.5e9

        def waveform(t):
            return jnp.where(
                t < stop_step * grid.dt,
                1.0e-3 * jnp.sin(2.0 * jnp.pi * freq * t) * jnp.sin(jnp.pi * t / (stop_step * grid.dt)) ** 2,
                0.0,
            )

        src = make_source(grid, (0.012, 0.012, 0.012), "ez", waveform, n_steps)
        result = run_simulation(
            grid,
            materials,
            n_steps,
            boundary="pec",
            sources=[src],
            snapshot=SnapshotSpec(components=("ex", "ey", "ez", "hx", "hy", "hz")),
        )
        start_idx = stop_step + 10
        energies = np.asarray([
            _field_energy_from_snapshots(result.snapshots or {}, idx, float(grid.dx))
            for idx in range(start_idx, n_steps)
        ])
        max_energy = float(np.max(energies)) if energies.size else 0.0
        min_energy = float(np.min(energies)) if energies.size else 0.0
        drift = float((max_energy - min_energy) / max(max_energy, 1.0e-300))
        passed = bool(np.isfinite(drift) and drift <= PHASE2B_THRESHOLDS["closed_energy_drift_max"])
        return _phase2b_row(
            case="phase2b_stage3_closed_domain_energy",
            method="lossless_pec_energy_snapshot_control",
            stage_id="stage3_closed_domain_energy",
            invariant="lossless_closed_domain_energy_is_conserved_after_source_shutoff",
            hypothesis="large_post_source_energy_drift_would_make_internal_pec_or_s11_loss_claims_uninterpretable",
            physical_expected="after the source shuts off, a PEC/vacuum/no-CPML domain should not lose or gain EM energy beyond the pre-registered drift threshold",
            observed={
                "geometry": "small_vacuum_pec_box",
                "grid_shape": tuple(int(x) for x in grid.shape),
                "n_steps": n_steps,
                "source_stop_step": stop_step,
                "energy_eval_start_step": start_idx,
                "energy_min": min_energy,
                "energy_max": max_energy,
                "energy_drift_fraction": drift,
            },
            threshold={"energy_drift_fraction_max": PHASE2B_THRESHOLDS["closed_energy_drift_max"]},
            threshold_rationale="closed lossless-domain energy conservation should be much tighter than WR-90 S11 materiality; this simplified geometry is labeled separately from the oracle geometry",
            geometry_scope="synthetic_closed_domain",
            requires_passed_stages=requires_passed_stages,
            passed=passed,
            blocks_next_stage=not passed,
        )
    except Exception as exc:
        return _phase2b_row(
            case="phase2b_stage3_closed_domain_energy",
            method="lossless_pec_energy_snapshot_control",
            stage_id="stage3_closed_domain_energy",
            invariant="lossless_closed_domain_energy_is_conserved_after_source_shutoff",
            hypothesis="energy_control_exception_blocks_downstream_interpretation",
            physical_expected="closed-domain energy control can run in a bounded synthetic geometry",
            observed={"exception": f"{type(exc).__name__}: {exc}"},
            threshold={"energy_drift_fraction_max": PHASE2B_THRESHOLDS["closed_energy_drift_max"]},
            threshold_rationale="do not interpret PEC/CPML/S11 if the bounded energy control cannot run",
            geometry_scope="synthetic_closed_domain",
            requires_passed_stages=requires_passed_stages,
            passed=False,
            blocks_next_stage=True,
            status="error",
            verdict_hint="diagnostic_failed_no_closure_claim",
            skip_reason=f"{type(exc).__name__}: {exc}",
        )


def run_phase2b_cpml_isolation_preflight(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    monitor_x_m: tuple[float, ...],
    dx: float | None,
    upstream_blocker: str | None,
    requires_passed_stages: Iterable[str] = (
        "stage1_mode_ledger",
        "stage2_source_table_pairing",
        "stage2_source_empty_guide_purity",
        "stage3_closed_domain_energy",
    ),
) -> OracleRow:
    if upstream_blocker:
        return _phase2b_blocked_row(
            case="phase2b_stage4_cpml_isolation_preflight",
            method="cpml_geometry_energy_balance_preflight",
            stage_id="stage4_cpml_isolation",
            invariant="source_monitor_short_are_geometrically_isolated_from_cpml_before_absorber_energy_claims",
            hypothesis="cpml_balance_cannot_be_interpreted_until_lossless_energy_control_passes",
            requires_passed_stages=requires_passed_stages,
            upstream_blocker=upstream_blocker,
        )
    grid, _ = _phase2b_waveguide_cfg(
        freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx
    )
    cpml_thickness = float(cpml_layers) * float(grid.dx)
    source_left_clearance = float(PORT_LEFT_X)
    short_right_clearance = float(DOMAIN[0] - (PEC_SHORT_X + PEC_THICKNESS))
    monitor_left_clearance = min(float(x) for x in monitor_x_m)
    monitor_right_clearance = min(float(DOMAIN[0] - x) for x in monitor_x_m)
    min_clearance = min(source_left_clearance, short_right_clearance, monitor_left_clearance, monitor_right_clearance)
    passed = min_clearance > cpml_thickness
    return _phase2b_row(
        case="phase2b_stage4_cpml_isolation_preflight",
        method="cpml_geometry_energy_balance_preflight",
        stage_id="stage4_cpml_isolation",
        invariant="source_monitor_short_are_geometrically_isolated_from_cpml_before_absorber_energy_claims",
        hypothesis="if geometry places source_monitor_or_short_inside_cpml_then_cpml_coupling_can_dominate_s11",
        physical_expected="all active/reference/short planes remain outside CPML thickness before CPML energy rows are interpreted",
        observed={
            "cpml_layers": int(cpml_layers),
            "dx_m": float(grid.dx),
            "cpml_thickness_m": cpml_thickness,
            "source_left_clearance_m": source_left_clearance,
            "short_right_clearance_m": short_right_clearance,
            "monitor_left_clearance_min_m": monitor_left_clearance,
            "monitor_right_clearance_min_m": monitor_right_clearance,
            "min_clearance_m": min_clearance,
            "time_domain_energy_balance_preferred": True,
            "frequency_domain_flux_caveat": "not_used_as_primary_absorber_energy_ledger_in_this_preflight_row",
        },
        threshold={"min_clearance_gt_cpml_thickness_m": cpml_thickness},
        threshold_rationale="preflight geometry must be outside CPML before absorber energy balance or S11 changes are assigned to physical DUT behavior",
        geometry_scope="wr90_oracle_geometry",
        requires_passed_stages=requires_passed_stages,
        passed=passed,
        blocks_next_stage=not passed,
    )


def run_phase2b_causal_window_ledger(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    monitor_x_m: tuple[float, ...],
    dx: float | None,
    upstream_blocker: str | None,
    requires_passed_stages: Iterable[str] = (
        "stage1_mode_ledger",
        "stage2_source_table_pairing",
        "stage2_source_empty_guide_purity",
        "stage3_closed_domain_energy",
        "stage4_cpml_isolation",
    ),
) -> OracleRow:
    if upstream_blocker:
        return _phase2b_blocked_row(
            case="phase2b_stage5_causal_window_ledger",
            method="group_delay_window_ledger",
            stage_id="stage5_causal_time_gating",
            invariant="dft_windows_are_tied_to_causal_arrival_and_reflection_times",
            hypothesis="time_gating_cannot_be_interpreted_until_cpml_isolation_passes",
            requires_passed_stages=requires_passed_stages,
            upstream_blocker=upstream_blocker,
        )
    _, cfg = _phase2b_waveguide_cfg(
        freqs_hz, cpml_layers=cpml_layers, num_periods=num_periods, dx=dx
    )
    center_freq = float(np.asarray(freqs_hz, dtype=float)[len(freqs_hz) // 2])
    delta_f = max(center_freq * 1.0e-4, 1.0e6)
    freqs_pair = np.asarray([center_freq - delta_f, center_freq + delta_f])
    beta_pair = np.asarray(_compute_beta(jnp.asarray(freqs_pair), cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx), dtype=np.complex128).real
    group_delay_per_m = float((beta_pair[1] - beta_pair[0]) / (2.0 * 2.0 * np.pi * delta_f))
    monitor_mid = float(np.asarray(monitor_x_m, dtype=float)[len(monitor_x_m) // 2])
    incident_delay = abs(monitor_mid - PORT_LEFT_X) * group_delay_per_m
    reflected_delay = (abs(PEC_SHORT_X - PORT_LEFT_X) + abs(PEC_SHORT_X - monitor_mid)) * group_delay_per_m
    record_time = int(cfg.dft_total_steps) * float(cfg.dt)
    source_end_time = float(cfg.src_t0 * 2.0)
    valid = (
        np.isfinite(group_delay_per_m)
        and incident_delay > 0.0
        and reflected_delay > incident_delay
        and reflected_delay < record_time
    )
    return _phase2b_row(
        case="phase2b_stage5_causal_window_ledger",
        method="group_delay_window_ledger",
        stage_id="stage5_causal_time_gating",
        invariant="dft_windows_are_tied_to_causal_arrival_and_reflection_times",
        hypothesis="arbitrary_late_windows_are_invalid_without_arrival_time_and_energy_support",
        physical_expected="incident and reflected windows must fit inside the recorded time and preserve causal ordering",
        observed={
            "center_freq_hz": center_freq,
            "delta_f_hz": delta_f,
            "group_delay_per_m_s": group_delay_per_m,
            "source_end_time_s": source_end_time,
            "incident_arrival_time_s": incident_delay,
            "reflected_arrival_time_s": reflected_delay,
            "record_time_s": record_time,
            "phase2a_late_quarter_retained_as_invalid_negative_control": True,
        },
        threshold={"window_energy_fraction_min": PHASE2B_THRESHOLDS["window_energy_fraction_min"]},
        threshold_rationale="causal ordering and non-low-energy support are prerequisites; exact DFT windows must be chosen before seeing whether |S11| improves",
        geometry_scope="wr90_oracle_geometry",
        requires_passed_stages=requires_passed_stages,
        passed=valid,
        blocks_next_stage=not valid,
    )


def run_phase2b_pec_short_phase_benchmark(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int,
    num_periods: float,
    monitor_x_m: tuple[float, ...],
    dx: float | None,
    upstream_blocker: str | None,
    requires_passed_stages: Iterable[str] = (
        "stage1_mode_ledger",
        "stage2_source_table_pairing",
        "stage2_source_empty_guide_purity",
        "stage3_closed_domain_energy",
        "stage4_cpml_isolation",
        "stage5_causal_time_gating",
    ),
) -> OracleRow:
    if upstream_blocker:
        return _phase2b_blocked_row(
            case="phase2b_stage6_pec_short_phase_benchmark",
            method="reference_free_pec_short_phase_magnitude",
            stage_id="stage6_pec_short_phase",
            invariant="pec_short_reflection_has_unit_magnitude_and_discrete_beta_phase_after_upstream_gates",
            hypothesis="pec_short_s11_cannot_be_interpreted_until_mode_source_energy_cpml_and_time_gates_pass",
            requires_passed_stages=requires_passed_stages,
            upstream_blocker=upstream_blocker,
        )
    details = _reference_free_fit_details(
        OracleCaseConfig(
            freqs_hz=np.asarray(freqs_hz, dtype=float),
            cpml_layers=cpml_layers,
            num_periods=num_periods,
            monitor_x_m=monitor_x_m,
            dx=dx,
        )
    )
    beta = np.asarray(details["beta"], dtype=np.complex128)
    gamma_ref = np.asarray(details["gamma_at_phase_reference"], dtype=np.complex128)
    phase_ref = float(details["phase_reference_plane_m"])
    expected = -np.exp(-2j * beta * float(PEC_SHORT_X - phase_ref))
    phase_error = np.rad2deg(np.angle(gamma_ref / expected))
    mag = np.abs(gamma_ref)
    mag_error = np.abs(mag - 1.0)
    pass_phase = bool(np.nanmax(np.abs(phase_error)) <= PHASE2B_THRESHOLDS["pec_phase_error_deg_max"])
    return _phase2b_row(
        case="phase2b_stage6_pec_short_phase_benchmark",
        method="reference_free_pec_short_phase_magnitude",
        stage_id="stage6_pec_short_phase",
        invariant="pec_short_reflection_has_unit_magnitude_and_discrete_beta_phase_after_upstream_gates",
        hypothesis="after_upstream_physics_passes_remaining_magnitude_or_phase_error_localizes_to_mask_or_reference_calibration",
        physical_expected="Γ at the source reference plane should match -exp(-2jβd) for a PEC short under the declared convention",
        observed={
            "freqs_hz": np.asarray(details["freqs"], dtype=float),
            "phase_reference_plane_m": phase_ref,
            "short_x_m": PEC_SHORT_X,
            "gamma_at_phase_reference_plane": gamma_ref,
            "expected_gamma_pec_short": expected,
            "gamma_abs": mag,
            "gamma_abs_error_vs_unity": mag_error,
            "phase_error_deg": phase_error,
            "phase_shift_convention": PHASE_SHIFT_CONVENTION,
        },
        threshold={"phase_error_deg_max": PHASE2B_THRESHOLDS["pec_phase_error_deg_max"]},
        threshold_rationale="PEC phase can only be interpreted after upstream gates; magnitude is reported but not used to tune earlier physics",
        geometry_scope="wr90_oracle_geometry",
        requires_passed_stages=requires_passed_stages,
        passed=pass_phase,
        blocks_next_stage=not pass_phase,
    )


def run_phase2b_verdict(rows: list[OracleRow]) -> OracleRow:
    phase2b = [row for row in rows if row.case.startswith("phase2b_")]
    schema_errors = validate_phase2b_rows(phase2b)
    first_blocker = _phase2b_first_blocker(phase2b)
    failed = [row.to_jsonable().get("stage_id") for row in phase2b if row.to_jsonable().get("pass") is False]
    if schema_errors:
        classification = "schema_invalid_no_interpretation"
        confidence = "high"
        blocks = True
    elif first_blocker:
        classification = f"blocked_at_{first_blocker}"
        confidence = "medium"
        blocks = True
    else:
        classification = "physics_ladder_passed_ready_for_separate_fix_planning"
        confidence = "medium"
        blocks = False
    return _phase2b_row(
        case="phase2b_physics_ladder_verdict",
        method="ranked_physics_ladder_classification",
        stage_id="stage_final_verdict",
        invariant="phase2b_interpretation_requires_schema_complete_stage_ordered_physics_evidence",
        hypothesis="without_schema_and_stage_order_any_s11_claim_can_be_a_numeric_artifact",
        physical_expected="all interpreted rows are schema-complete; first failed invariant determines next branch; no strict closure is claimed",
        observed={
            "primary_classification": classification,
            "confidence": confidence,
            "schema_errors": schema_errors,
            "first_blocker": first_blocker,
            "failed_stage_ids": failed,
            "no_production_fix_gate": "closed" if blocks else "closed_until_separate_stage7_decision",
            "receive_extractor_frozen": True,
            "discrete_yee_mode_source_introduced": False,
        },
        threshold={"schema_errors": 0},
        threshold_rationale="final verdict is a classification row, not a production-fix or strict-closure gate",
        geometry_scope="production_path",
        control_type="none",
        requires_passed_stages=[],
        passed=not schema_errors,
        blocks_next_stage=blocks,
        verdict_hint="phase2b_classification_only_no_strict_closure",
    )


def run_phase2b_no_production_fix_gate_row() -> OracleRow:
    return _phase2b_row(
        case="phase2b_no_production_fix_gate",
        method="no_production_fix_gate",
        stage_id="stage7_production_fix_gate",
        invariant="production_fix_requires_specific_failed_physical_invariant_and_separate_stage7_decision",
        hypothesis="landing_a_fix_without_invariant_localization_would_recreate_numeric_fitting",
        physical_expected="gate remains closed in this diagnostic-first run",
        observed={
            "no_production_fix_gate": "closed",
            "receive_extractor_frozen": True,
            "discrete_yee_mode_source_introduced": False,
            "issues_13_17_resolved": False,
            "strict_closure_claimed": False,
        },
        threshold={"stage7_gate": "closed"},
        threshold_rationale="Phase 2B diagnostic evidence must be reviewed before production files are opened",
        geometry_scope="production_path",
        control_type="none",
        passed=True,
        blocks_next_stage=False,
        verdict_hint="gate_closed_no_production_change",
    )


def run_phase2b_diagnostics(freqs: np.ndarray, args: argparse.Namespace) -> list[OracleRow]:
    monitor_x_m = tuple(float(x) for x in args.monitor_x_m)
    rows: list[OracleRow] = [run_phase2b_guardrail_row()]
    blocker = _phase2b_first_blocker(rows)

    mode_rows = run_phase2b_mode_ledger(
        freqs,
        cpml_layers=args.cpml_layers,
        num_periods=args.num_periods,
        dx=args.dx,
    )
    rows.extend(mode_rows)
    blocker = _phase2b_first_blocker(rows)

    source_table = run_phase2b_source_table_consistency(
        freqs,
        cpml_layers=args.cpml_layers,
        num_periods=args.num_periods,
        dx=args.dx,
        upstream_blocker=blocker,
    )
    rows.append(source_table)
    blocker = _phase2b_first_blocker(rows)

    source_purity = run_phase2b_source_purity(
        freqs,
        cpml_layers=args.cpml_layers,
        num_periods=args.num_periods,
        dx=args.dx,
        upstream_blocker=blocker,
    )
    rows.append(source_purity)
    blocker = _phase2b_first_blocker(rows)

    rows.append(run_phase2b_closed_domain_energy(upstream_blocker=blocker))
    blocker = _phase2b_first_blocker(rows)

    rows.append(
        run_phase2b_cpml_isolation_preflight(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
            upstream_blocker=blocker,
        )
    )
    blocker = _phase2b_first_blocker(rows)

    rows.append(
        run_phase2b_causal_window_ledger(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
            upstream_blocker=blocker,
        )
    )
    blocker = _phase2b_first_blocker(rows)

    rows.append(
        run_phase2b_pec_short_phase_benchmark(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
            upstream_blocker=blocker,
        )
    )
    rows.append(run_phase2b_verdict(rows))
    rows.append(run_phase2b_no_production_fix_gate_row())
    return rows


def run_phase2a_oracle_stability_verdict(rows: list[OracleRow]) -> OracleRow:
    payload = {row.case: row.to_jsonable() for row in rows}
    signals: list[str] = []
    triggering: list[str] = []
    counter: list[str] = []

    late = payload.get("phase2a_ref_free_late_window_sweep", {})
    for window in late.get("windows", []):
        if not window.get("valid", True):
            counter.append(f"{window.get('window_label')}:invalid")
            continue
        d_mean = abs(float(window.get("delta_mean_abs_gamma_vs_full_record") or 0.0))
        d_min = abs(float(window.get("delta_min_abs_gamma_vs_full_record") or 0.0))
        if d_mean >= 0.02 or d_min >= 0.05:
            signals.append("dft_window_settling_likely")
            triggering.append("phase2a_ref_free_late_window_sweep")
            break

    period = payload.get("phase2a_ref_free_period_sweep_40_80_120", {})
    if float(period.get("ref_free_delta_mean_abs_s11") or 0.0) >= 0.02 or float(period.get("ref_free_delta_min_abs_s11") or 0.0) >= 0.05:
        signals.append("dft_window_settling_likely")
        triggering.append("phase2a_ref_free_period_sweep_40_80_120")

    center = payload.get("phase2a_ref_free_same_run_center_vs_band", {})
    if abs(float(center.get("same_run_center_minus_band_mean_abs") or 0.0)) >= 0.02:
        signals.append("band_vs_center_artifact_likely")
        triggering.append("phase2a_ref_free_same_run_center_vs_band")

    cpml = payload.get("phase2a_cpml_sweep_late_window_ref_free", {})
    if float(cpml.get("ref_free_delta_mean_abs_s11") or 0.0) >= 0.02 or float(cpml.get("ref_free_delta_min_abs_s11") or 0.0) >= 0.05:
        signals.append("cpml_source_monitor_coupling_likely")
        triggering.append("phase2a_cpml_sweep_late_window_ref_free")
    elif cpml:
        counter.append("phase2a_cpml_sweep_late_window_ref_free:stable")

    for case in ("phase2a_monitor_spacing_sweep_ref_free", "phase2a_source_short_distance_sweep_ref_free"):
        item = payload.get(case, {})
        if float(item.get("delta_mean_abs_s11") or 0.0) >= 0.02 or float(item.get("delta_min_abs_s11") or 0.0) >= 0.05:
            signals.append("cpml_source_monitor_coupling_likely")
            triggering.append(case)

    distinct = list(dict.fromkeys(signals))
    if not distinct:
        primary = "inconclusive_keep_no_fix_gate"
        confidence = "low"
    elif len(distinct) == 1:
        primary = distinct[0]
        confidence = "medium"
    else:
        primary = "inconclusive_keep_no_fix_gate"
        confidence = "low"
    return OracleRow(
        "phase2a_oracle_stability_verdict",
        "ranked_oracle_stability_classification",
        "ok",
        {
            "primary_classification": primary,
            "secondary_signals": distinct if primary == "inconclusive_keep_no_fix_gate" else distinct[1:],
            "confidence": confidence,
            "triggering_rows": list(dict.fromkeys(triggering)),
            "counterevidence_rows": list(dict.fromkeys(counter)),
            "no_production_fix_gate": "closed",
            "issues_13_17_resolved": False,
            "strict_closure_claimed": False,
        },
        "phase2a_classification_only_no_production_fix",
    )


def phase2a_no_production_fix_gate_row() -> OracleRow:
    return OracleRow(
        "phase2a_no_production_fix_gate",
        "no_production_fix_gate",
        "ok",
        {
            "no_production_fix_gate": "closed",
            "requires_separate_phase2b_ralplan": True,
            "receive_extractor_frozen": True,
            "discrete_yee_mode_source_introduced": False,
            "issues_13_17_resolved": False,
            "strict_closure_claimed": False,
        },
        "gate_closed_until_oracle_stability_is_classified_and_phase2b_is_approved",
    )


def run_phase2a_diagnostics(freqs: np.ndarray, args: argparse.Namespace) -> list[OracleRow]:
    monitor_x_m = tuple(float(x) for x in args.monitor_x_m)
    rows: list[OracleRow] = [
        phase2a_late_window_synthetic_control(),
        run_phase2a_ref_free_late_window_sweep(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
        ),
        skipped_row(
            "phase2a_current_late_window_sweep",
            "current_2run_late_window_sensitivity",
            "current two-run late-window extraction would require production extractor changes; skipped in diagnostic-only Phase 2A",
            metrics={"no_production_fix_gate": "closed"},
        ),
        run_phase2a_ref_free_period_sweep(
            freqs,
            cpml_layers=args.cpml_layers,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
        ),
        run_phase2a_current_period_sweep(
            freqs,
            cpml_layers=args.cpml_layers,
            dx=args.dx,
        ),
        run_phase2a_center_lockin_ref_free(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            warmup_cycles=args.cw_warmup_cycles,
            lockin_cycles=args.cw_lockin_cycles,
            dx=args.dx,
        ),
        run_phase2a_ref_free_same_run_center_vs_band(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
        ),
        run_phase2a_current_same_run_center_vs_band(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            dx=args.dx,
        ),
        run_phase2a_center_only_rerun_labeled(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
        ),
        run_phase2a_cpml_sweep_late_window_ref_free(
            freqs,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
        ),
        skipped_row(
            "phase2a_cpml_sweep_late_window_current",
            "current_2run_cpml_sensitivity_late_window",
            "current two-run late-window CPML sweep would require production extractor changes; skipped in diagnostic-only Phase 2A",
            metrics={"no_production_fix_gate": "closed"},
        ),
        run_phase2a_monitor_spacing_sweep_ref_free(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            dx=args.dx,
        ),
        run_phase2a_source_short_distance_sweep_ref_free(
            freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=monitor_x_m,
            dx=args.dx,
        ),
    ]
    rows.append(run_phase2a_oracle_stability_verdict(rows))
    rows.append(phase2a_no_production_fix_gate_row())
    return rows


def skipped_row(case: str, method: str, reason: str, *, metrics: dict[str, Any] | None = None) -> OracleRow:
    return OracleRow(case, method, "skipped", metrics or {}, "explicit_skip_no_closure_claim", reason)


def emit_jsonl_rows(rows: Iterable[OracleRow], path_or_stdout: str | Path | None = None) -> str | None:
    lines = [json.dumps(row.to_jsonable(), sort_keys=True, allow_nan=False) for row in rows]
    payload = "\n".join(lines)
    if path_or_stdout is None or str(path_or_stdout) == "-":
        print(payload)
        return None
    path = Path(path_or_stdout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")
    return str(path)


def format_human_table(rows: Iterable[OracleRow]) -> str:
    out = ["case | method | status | mean|Γ| | min|Γ| | fit_residual | note", "--- | --- | --- | ---: | ---: | ---: | ---"]
    for row in rows:
        data = row.to_jsonable()
        mean = data.get("mean_abs_s11")
        min_mag = data.get("min_abs_s11")
        fit = data.get("fit_residual")
        def fmt(value: Any) -> str:
            return "" if value is None else (f"{value:.6g}" if isinstance(value, (int, float)) else str(value))
        note = data.get("skip_reason") or data.get("verdict_hint", "")
        out.append(f"{row.case} | {row.method} | {row.status} | {fmt(mean)} | {fmt(min_mag)} | {fmt(fit)} | {note}")
    return "\n".join(out)


def run_matrix(args: argparse.Namespace) -> list[OracleRow]:
    freqs = _freqs_from_args(args)
    rows: list[OracleRow] = []
    rows.append(synthetic_least_squares_control())
    rows.append(cw_lockin_control_row())

    if args.synthetic_only:
        return rows

    rows.append(run_current_2run_baseline(freqs, cpml_layers=args.cpml_layers, num_periods=args.num_periods, dx=args.dx))
    cfg_3 = OracleCaseConfig(
        freqs_hz=freqs,
        cpml_layers=args.cpml_layers,
        num_periods=args.num_periods,
        monitor_x_m=tuple(args.monitor_x_m),
        dx=args.dx,
    )
    rows.append(run_reference_free_case(cfg_3, case_name="ref_free_3plane_internal_mask_current_cpml"))

    if args.full:
        cfg_5 = OracleCaseConfig(
            freqs_hz=freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=(0.026, 0.036, 0.046, 0.056, 0.066),
            dx=args.dx,
        )
        rows.append(run_reference_free_case(cfg_5, case_name="ref_free_5plane_internal_mask_current_cpml"))
        rows.append(run_source_purity_empty_line_sweep(freqs, cpml_layers=args.cpml_layers, num_periods=args.num_periods, dx=args.dx))
        rows.append(
            skipped_row(
                "cw_lockin_ref_free_internal_mask_current_cpml",
                "cw_integer_period_ref_free",
                "physical CW waveguide source tables are intentionally not productionized in this diagnostic PR; helper is covered by cw_lockin_synthetic_control",
                metrics={"warmup_cycles": args.cw_warmup_cycles, "lockin_cycles": args.cw_lockin_cycles},
            )
        )
        rows.append(
            skipped_row(
                "source_short_distance_sweep",
                "ref_free_distance_sweep",
                "expensive multi-run sweep deferred behind the first quick oracle evidence; enable in follow-up once monitor backend is accepted",
            )
        )
        rows.append(
            skipped_row(
                "monitor_short_distance_sweep",
                "ref_free_distance_sweep",
                "expensive multi-run sweep deferred behind the first quick oracle evidence; current row records monitor_short_distances_m for the 3/5-plane cases",
            )
        )
        rows.append(
            skipped_row(
                "pml_sweep_current_2run_layers_10_20_40",
                "current_2run_cpml_sweep",
                "CPML 10/20/40 production two-run sweep is intentionally skipped by default to keep --full bounded; row is machine-readable for follow-up execution",
                metrics={"requested_cpml_layers": [10, 20, 40]},
            )
        )
        rows.append(
            skipped_row(
                "pml_sweep_ref_free_layers_10_20_40",
                "ref_free_cpml_sweep",
                "CPML 10/20/40 reference-free sweep is intentionally skipped by default to keep --full bounded; row is machine-readable for follow-up execution",
                metrics={"requested_cpml_layers": [10, 20, 40]},
            )
        )
        rows.append(
            skipped_row(
                "face_short_ref_free_no_cpml_or_irrelevant",
                "boundary_face_short_ref_free",
                "equivalent boundary-face short geometry cannot be constructed without production/API geometry changes in this PR",
                metrics={"comparison_quality": "qualitative_skipped"},
            )
        )
        rows.append(
            skipped_row(
                "internal_mask_short_ref_free_same_geometry",
                "internal_mask_short_ref_free",
                "covered by ref_free_3plane/ref_free_5plane internal-mask rows; boundary-face pair skipped qualitatively",
                metrics={"comparison_quality": "qualitative_skipped"},
            )
        )

    if getattr(args, "bc_diagnostics", False):
        rows.extend([
            run_current_norm_dissection(
                freqs,
                cpml_layers=args.cpml_layers,
                num_periods=args.num_periods,
                dx=args.dx,
            ),
            run_period_sweep_current_vs_ref_free(
                freqs,
                cpml_layers=args.cpml_layers,
                dx=args.dx,
            ),
            run_pml_sweep_current_vs_ref_free(
                freqs,
                num_periods=args.num_periods,
                dx=args.dx,
            ),
            run_reference_plane_sweep_current_2run(
                freqs,
                cpml_layers=args.cpml_layers,
                num_periods=args.num_periods,
                dx=args.dx,
            ),
        ])
    if getattr(args, "phase2a_diagnostics", False):
        rows.extend(run_phase2a_diagnostics(freqs, args))
    if getattr(args, "phase2b_diagnostics", False):
        rows.extend(run_phase2b_diagnostics(freqs, args))
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="store_true", help="Run the WR-90 oracle matrix.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run MVP rows only (default).")
    mode.add_argument("--full", action="store_true", help="Attempt Matrix A-F, emitting skips for bounded/deferred rows.")
    parser.add_argument("--synthetic-only", action="store_true", help="Run only synthetic/control rows for fast development tests.")
    parser.add_argument("--bc-diagnostics", action="store_true", help="Run Phase-1 B/C normalization/DFT/CPML diagnostic rows.")
    parser.add_argument("--phase2a-diagnostics", action="store_true", help="Run Phase-2A reference-free stability/isolation diagnostic rows.")
    parser.add_argument("--phase2b-physics-ladder", dest="phase2b_diagnostics", action="store_true", help="Run Phase-2B physics-first stage-blocked diagnostic ladder rows.")
    parser.add_argument("--freq-min-hz", type=float, default=5.0e9)
    parser.add_argument("--freq-max-hz", type=float, default=7.0e9)
    parser.add_argument("--center-freq-hz", type=float, default=None, help="Use one center frequency instead of a band.")
    parser.add_argument("--n-freqs", type=int, default=6)
    parser.add_argument("--cpml-layers", type=int, default=DEFAULT_CPML_LAYERS)
    parser.add_argument("--num-periods", type=float, default=DEFAULT_NUM_PERIODS)
    parser.add_argument("--dx", type=float, default=None, help="Optional grid spacing override for faster diagnostics.")
    parser.add_argument("--monitor-x-m", type=float, nargs="+", default=[0.030, 0.045, 0.060])
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Optional JSONL output path; default writes .omx/logs/wr90-port-oracle-matrix-<timestamp>-<mode>.jsonl; use '-' to print JSONL to stdout.",
    )
    parser.add_argument("--cw-warmup-cycles", type=int, default=20)
    parser.add_argument("--cw-lockin-cycles", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.matrix:
        parser.error("--matrix is required")
    if len(args.monitor_x_m) < 3 and not args.synthetic_only:
        parser.error("at least three --monitor-x-m planes are required")
    if args.cpml_layers <= 0:
        parser.error("--cpml-layers must be positive")
    if args.num_periods <= 0:
        parser.error("--num-periods must be positive")
    if args.n_freqs <= 0:
        parser.error("--n-freqs must be positive")
    if not args.quick and not args.full:
        args.quick = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run_matrix(args)
    print(format_human_table(rows))
    if args.jsonl == "-":
        print("jsonl:")
        emit_jsonl_rows(rows, None)
        artifact_path = None
    else:
        mode = "phase2b" if args.phase2b_diagnostics else "phase2a" if args.phase2a_diagnostics else "bc" if args.bc_diagnostics else "full" if args.full else "quick"
        artifact_path = args.jsonl
        if artifact_path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            artifact_path = f".omx/logs/wr90-port-oracle-matrix-{stamp}-{mode}.jsonl"
        path = emit_jsonl_rows(rows, artifact_path)
        print(f"jsonl_artifact: {path}")
    summary = {
        "n_rows": len(rows),
        "n_ok": sum(row.status == "ok" for row in rows),
        "n_control": sum(row.status == "control" for row in rows),
        "n_skipped": sum(row.status == "skipped" for row in rows),
        "n_error": sum(row.status == "error" for row in rows),
        "strict_closure_claimed": False,
        "issues_13_17_resolved": False,
    }
    print("summary:")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

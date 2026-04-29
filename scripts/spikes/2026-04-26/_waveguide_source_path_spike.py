#!/usr/bin/env python3
"""Phase-1 source/reference/geometry isolation matrix for waveguide ports.

This is a diagnostic harness, not production code.  It intentionally imports
and reuses the BIORTHO spike helpers from ``_phase2_orthonormal_spike.py`` so
that failed candidates remain spike-local until a fresh physical gate passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
    module="matplotlib\\.projections",
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfx.api import Simulation  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402
from rfx.sources.waveguide_port import WaveguidePortConfig  # noqa: E402
from rfx.simulation import run as run_simulation  # noqa: E402

import scripts._phase2_orthonormal_spike as phase2  # noqa: E402

DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
STRICT_PEC_MIN_S11 = 0.99
STRICT_ASYM_MEAN = 0.02

SectionName = Literal["physical_rows", "control_rows", "geometry_rows"]
ExtractorName = Literal["scalar", "biortho"]


@dataclass(frozen=True)
class MatrixRow:
    """One machine-readable matrix output row."""

    section: SectionName
    row_id: str
    extractor: str
    source_variant: str
    metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    verdict: str = "UNKNOWN"

    def to_jsonable(self) -> dict[str, Any]:
        return _json_safe({
            "section": self.section,
            "row_id": self.row_id,
            "extractor": self.extractor,
            "source_variant": self.source_variant,
            "verdict": self.verdict,
            **self.metrics,
        })


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class BiorthoRunOptions:
    """Script-local toggles for source/reference isolation."""

    h_time_shift_dt: float = 0.0
    h_amplitude_scale: float = 1.0
    subtract_reflection: bool = True
    through_reference_norm: bool = True

    @property
    def source_variant(self) -> str:
        parts = []
        if self.h_time_shift_dt:
            parts.append(f"h_time_shift_dt={self.h_time_shift_dt:+.3g}")
        if self.h_amplitude_scale != 1.0:
            parts.append(f"h_amplitude_scale={self.h_amplitude_scale:.3g}")
        if not self.subtract_reflection:
            parts.append("reflection_subtraction=off")
        if not self.through_reference_norm:
            parts.append("through_reference_norm=off")
        return ";".join(parts) if parts else "current_source"


def _biortho_candidate() -> phase2.CandidateSpec:
    return next(c for c in phase2.CANDIDATES if c.name == "BIORTHO_FULL_YEE")


def _build_multimode_sim(
    freqs_hz: np.ndarray,
    *,
    n_modes: int,
    cpml_layers: int = 10,
    obstacles: Sequence[tuple[tuple[float, float, float], tuple[float, float, float], float]] = (),
    pec_short_x: float | None = None,
    waveform: str = "modulated_gaussian",
) -> Simulation:
    """Canonical two-port rectangular guide with configurable CPML depth."""

    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=max(float(freqs[-1]), f0),
        domain=DOMAIN,
        boundary="cpml",
        cpml_layers=int(cpml_layers),
    )

    for idx, (lo, hi, eps_r) in enumerate(obstacles):
        name = f"diel_{idx}"
        sim.add_material(name, eps_r=eps_r, sigma=0.0)
        sim.add(Box(lo, hi), material=name)

    if pec_short_x is not None:
        thickness = 0.002
        sim.add(
            Box((pec_short_x, 0.0, 0.0), (pec_short_x + thickness, DOMAIN[1], DOMAIN[2])),
            material="pec",
        )

    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        n_modes=int(n_modes),
        mode_profile="discrete",
    )
    sim.add_waveguide_port(PORT_LEFT_X, direction="+x", name="left", **common)
    sim.add_waveguide_port(PORT_RIGHT_X, direction="-x", name="right", **common)
    return sim


def _fractional_shift(values: jnp.ndarray, dt: float, shift_s: float) -> jnp.ndarray:
    """Return a real-valued fractional time shift using an FFT phase ramp."""

    arr = np.asarray(values, dtype=np.float64)
    if shift_s == 0.0 or arr.size <= 1:
        return values
    freqs = np.fft.rfftfreq(arr.size, d=float(dt))
    spectrum = np.fft.rfft(arr)
    shifted = np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * shift_s), n=arr.size)
    return jnp.asarray(shifted, dtype=values.dtype)


def transform_h_inc_table(
    cfg: WaveguidePortConfig,
    *,
    h_time_shift_dt: float = 0.0,
    h_amplitude_scale: float = 1.0,
) -> WaveguidePortConfig:
    """Spike-local post-processor for H source table phase/amplitude toggles."""

    h_table = cfg.h_inc_table
    if h_time_shift_dt:
        h_table = _fractional_shift(h_table, cfg.dt, h_time_shift_dt * cfg.dt)
    if h_amplitude_scale != 1.0:
        h_table = jnp.asarray(h_table * h_amplitude_scale, dtype=cfg.h_inc_table.dtype)
    return cfg._replace(h_inc_table=h_table)


def _transform_groups(
    groups: list[list[WaveguidePortConfig]],
    options: BiorthoRunOptions,
) -> list[list[WaveguidePortConfig]]:
    return [
        [
            transform_h_inc_table(
                cfg,
                h_time_shift_dt=options.h_time_shift_dt,
                h_amplitude_scale=options.h_amplitude_scale,
            )
            for cfg in group
        ]
        for group in groups
    ]


def strict_physical_verdict(metrics: dict[str, float | int | str | bool | None]) -> str:
    pec = float(metrics.get("pec_min_abs_s11", np.nan))
    asym = float(metrics.get("asym_mean_reciprocity_rel_diff", np.nan))
    if pec >= STRICT_PEC_MIN_S11 and asym < STRICT_ASYM_MEAN:
        return "GO_STRICT"
    return "NO_GO_PHYSICAL"


def _te10_indices_from_names(names: Sequence[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for idx, name in enumerate(names):
        if name == "left" or (name.startswith("left_") and "TE10" in name):
            indices["left"] = idx
        if name == "right" or (name.startswith("right_") and "TE10" in name):
            indices["right"] = idx
    if set(indices) != {"left", "right"}:
        raise RuntimeError(f"could not find left/right TE10 channels in {names}")
    return indices


def _s_metrics(s: np.ndarray, left_idx: int, right_idx: int, *, kind: str) -> dict[str, float]:
    if kind == "pec":
        s11 = np.abs(s[left_idx, left_idx, :])
        return {
            "pec_mean_abs_s11": float(np.mean(s11)),
            "pec_min_abs_s11": float(np.min(s11)),
            "pec_max_abs_s11": float(np.max(s11)),
            "pec_max_dev_from_1": float(np.max(np.abs(s11 - 1.0))),
        }
    s21 = np.abs(s[right_idx, left_idx, :])
    s12 = np.abs(s[left_idx, right_idx, :])
    denom = np.maximum(np.maximum(s21, s12), 1.0e-12)
    rel = np.abs(s21 - s12) / denom
    col_power = np.nansum(np.abs(s) ** 2, axis=0)
    return {
        "asym_mean_reciprocity_rel_diff": float(np.mean(rel)),
        "asym_max_reciprocity_rel_diff": float(np.max(rel)),
        "asym_mean_abs_s21": float(np.mean(s21)),
        "asym_mean_abs_s12": float(np.mean(s12)),
        "asym_max_col_power": float(np.nanmax(col_power)),
    }


def _mode_map_names(mode_map: Sequence[tuple[int, int, str, tuple[int, int]]]) -> tuple[str, ...]:
    labels = []
    for port_idx, mode_within, mode_type, mode_indices in mode_map:
        side = "left" if port_idx == 0 else "right" if port_idx == 1 else f"port{port_idx}"
        labels.append(f"{side}_mode{mode_within}_{mode_type}{mode_indices[0]}{mode_indices[1]}")
    return tuple(labels)


def _run_scalar_s_matrix(sim: Simulation, *, num_periods: float) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """Script-local scalar multimode normalized extraction.

    This intentionally avoids ``Simulation.compute_waveguide_s_matrix`` for
    multimode ``normalize=True`` so Phase 1 does not depend on or validate any
    production API behavior for that path.
    """

    case = phase2._prepare_multimode_case(sim, num_periods=num_periods)
    s_matrix, mode_map = _scalar_multimode_s_params_normalized(case)
    names = _mode_map_names(mode_map)
    return s_matrix, _te10_indices_from_names(names), {"port_names": ",".join(names)}


def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
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


def _flatten_cfgs(groups: Sequence[Sequence[WaveguidePortConfig]]) -> list[WaveguidePortConfig]:
    return [cfg for group in groups for cfg in group]


def _scalar_multimode_s_params_normalized(
    case: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[int, int, str, tuple[int, int]]]]:
    """Two-run normalized scalar multimode S, implemented spike-locally."""

    port_mode_cfgs = case["port_mode_cfgs"]
    flat_cfgs = _flatten_cfgs(port_mode_cfgs)
    mode_map: list[tuple[int, int, str, tuple[int, int]]] = []
    for port_idx, group in enumerate(port_mode_cfgs):
        for mode_within, cfg in enumerate(group):
            mode_map.append((port_idx, mode_within, cfg.mode_type, cfg.mode_indices))

    n_total = len(flat_cfgs)
    n_freqs = len(flat_cfgs[0].freqs)
    s_matrix = np.zeros((n_total, n_total, n_freqs), dtype=np.complex128)
    orig_amp = flat_cfgs[0].src_amp if flat_cfgs[0].src_amp != 0.0 else 1.0

    flat_ref_shifts: list[float] = []
    for port_idx, group in enumerate(port_mode_cfgs):
        flat_ref_shifts.extend([float(case["ref_shifts"][port_idx])] * len(group))

    for drive_flat_idx in range(n_total):
        ref_cfgs = [_reset_cfg(cfg, idx == drive_flat_idx) for idx, cfg in enumerate(flat_cfgs)]
        ref_cfgs[drive_flat_idx] = ref_cfgs[drive_flat_idx]._replace(src_amp=orig_amp)
        ref_result = run_simulation(
            case["grid"],
            case["ref_materials"],
            case["n_steps"],
            debye=None,
            lorentz=None,
            waveguide_ports=ref_cfgs,
            **case["common_run_kw"],
        )
        ref_final_cfgs = list(ref_result.waveguide_ports or ())
        a_inc_ref, _ = phase2.wg_mod.extract_waveguide_port_waves(
            ref_final_cfgs[drive_flat_idx],
            ref_shift=flat_ref_shifts[drive_flat_idx],
        )
        a_inc_ref_np = np.asarray(a_inc_ref)
        safe_a_inc = np.where(np.abs(a_inc_ref_np) > 1.0e-30, a_inc_ref_np, np.ones_like(a_inc_ref_np))

        b_out_ref = []
        for recv_idx, cfg in enumerate(ref_final_cfgs):
            _, b_ref_i = phase2.wg_mod.extract_waveguide_port_waves(
                cfg,
                ref_shift=flat_ref_shifts[recv_idx],
            )
            b_out_ref.append(np.asarray(b_ref_i))

        dev_cfgs = [_reset_cfg(cfg, idx == drive_flat_idx) for idx, cfg in enumerate(flat_cfgs)]
        dev_cfgs[drive_flat_idx] = dev_cfgs[drive_flat_idx]._replace(src_amp=orig_amp)
        dev_result = run_simulation(
            case["grid"],
            case["materials"],
            case["n_steps"],
            debye=case["debye"],
            lorentz=case["lorentz"],
            waveguide_ports=dev_cfgs,
            **case["common_run_kw"],
        )
        dev_final_cfgs = list(dev_result.waveguide_ports or ())

        for recv_idx, cfg in enumerate(dev_final_cfgs):
            _, b_recv_dev = phase2.wg_mod.extract_waveguide_port_waves(
                cfg,
                ref_shift=flat_ref_shifts[recv_idx],
            )
            b_recv_dev_np = np.asarray(b_recv_dev)
            b_ref_np = b_out_ref[recv_idx]
            ref_nonzero = np.abs(b_ref_np) > 1.0e-30
            safe_b_ref = np.where(ref_nonzero, b_ref_np, np.ones_like(b_ref_np))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if recv_idx == drive_flat_idx:
                    s_matrix[recv_idx, drive_flat_idx, :] = (b_recv_dev_np - b_ref_np) / safe_a_inc
                else:
                    through_norm = b_recv_dev_np / safe_b_ref
                    conversion_norm = (b_recv_dev_np - b_ref_np) / safe_a_inc
                    s_matrix[recv_idx, drive_flat_idx, :] = np.where(
                        ref_nonzero,
                        through_norm,
                        conversion_norm,
                    )

    return s_matrix, mode_map


def _biorthogonal_s_params_variant(
    case: dict[str, Any],
    *,
    options: BiorthoRunOptions,
) -> tuple[np.ndarray, list[tuple[int, int, str, tuple[int, int]]], dict[str, float]]:
    """BIORTHO normalized S-matrix with script-local reference toggles."""

    port_mode_cfgs = _transform_groups(case["port_mode_cfgs"], options)
    flat_cfgs = _flatten_cfgs(port_mode_cfgs)
    mode_map: list[tuple[int, int, str, tuple[int, int]]] = []
    for port_idx, group in enumerate(port_mode_cfgs):
        for mode_within, cfg in enumerate(group):
            mode_map.append((port_idx, mode_within, cfg.mode_type, cfg.mode_indices))

    n_total = len(flat_cfgs)
    n_freqs = len(flat_cfgs[0].freqs)
    s_matrix = np.zeros((n_total, n_total, n_freqs), dtype=np.complex128)
    orig_amp = flat_cfgs[0].src_amp if flat_cfgs[0].src_amp != 0.0 else 1.0
    max_cond_e = 0.0
    max_cond_h = 0.0
    directionality_ratios: list[float] = []

    for drive_flat_idx in range(n_total):
        ref_cfgs = [_reset_cfg(cfg, idx == drive_flat_idx) for idx, cfg in enumerate(flat_cfgs)]
        ref_cfgs[drive_flat_idx] = ref_cfgs[drive_flat_idx]._replace(src_amp=orig_amp)
        ref_result = run_simulation(
            case["grid"],
            case["ref_materials"],
            case["n_steps"],
            debye=None,
            lorentz=None,
            waveguide_ports=ref_cfgs,
            **case["common_run_kw"],
        )
        ref_final_cfgs = list(ref_result.waveguide_ports or ())
        a_ref_all, b_ref_all, ref_diag = phase2._biorthogonal_flat_waves(
            ref_final_cfgs,
            port_mode_cfgs,
            case["ref_shifts"],
        )
        max_cond_e = max(max_cond_e, ref_diag["max_cond_e"])
        max_cond_h = max(max_cond_h, ref_diag["max_cond_h"])
        a_inc_ref = np.asarray(a_ref_all[drive_flat_idx])
        b_drive_ref = np.asarray(b_ref_all[drive_flat_idx])
        safe_a_inc = np.where(np.abs(a_inc_ref) > 1.0e-30, a_inc_ref, np.ones_like(a_inc_ref))
        ratio = np.abs(b_drive_ref) / np.maximum(np.abs(safe_a_inc), 1e-30)
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)
        directionality_ratios.append(float(np.nanmean(ratio)))

        dev_cfgs = [_reset_cfg(cfg, idx == drive_flat_idx) for idx, cfg in enumerate(flat_cfgs)]
        dev_cfgs[drive_flat_idx] = dev_cfgs[drive_flat_idx]._replace(src_amp=orig_amp)
        dev_result = run_simulation(
            case["grid"],
            case["materials"],
            case["n_steps"],
            debye=case["debye"],
            lorentz=case["lorentz"],
            waveguide_ports=dev_cfgs,
            **case["common_run_kw"],
        )
        dev_final_cfgs = list(dev_result.waveguide_ports or ())
        _, b_dev_all, dev_diag = phase2._biorthogonal_flat_waves(
            dev_final_cfgs,
            port_mode_cfgs,
            case["ref_shifts"],
        )
        max_cond_e = max(max_cond_e, dev_diag["max_cond_e"])
        max_cond_h = max(max_cond_h, dev_diag["max_cond_h"])

        for recv_idx, b_recv_dev in enumerate(b_dev_all):
            b_recv_dev_np = np.asarray(b_recv_dev)
            b_ref_np = np.asarray(b_ref_all[recv_idx])
            ref_nonzero = np.abs(b_ref_np) > 1.0e-30
            safe_b_ref = np.where(ref_nonzero, b_ref_np, np.ones_like(b_ref_np))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if recv_idx == drive_flat_idx:
                    numerator = b_recv_dev_np - b_ref_np if options.subtract_reflection else b_recv_dev_np
                    s_matrix[recv_idx, drive_flat_idx, :] = numerator / safe_a_inc
                else:
                    if options.through_reference_norm:
                        through_norm = b_recv_dev_np / safe_b_ref
                        conversion_norm = (b_recv_dev_np - b_ref_np) / safe_a_inc
                        s_matrix[recv_idx, drive_flat_idx, :] = np.where(
                            ref_nonzero,
                            through_norm,
                            conversion_norm,
                        )
                    else:
                        s_matrix[recv_idx, drive_flat_idx, :] = (b_recv_dev_np - b_ref_np) / safe_a_inc

    diag = {
        "max_cond_e": float(max_cond_e),
        "max_cond_h": float(max_cond_h),
        "source_directionality_ratio_mean": float(np.nanmean(directionality_ratios)),
        "source_directionality_ratio_max": float(np.nanmax(directionality_ratios)),
    }
    return s_matrix, mode_map, diag


def _run_biortho_s_matrix(
    sim: Simulation,
    *,
    num_periods: float,
    options: BiorthoRunOptions,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    with phase2.patched_waveguide_port(_biortho_candidate()):
        case = phase2._prepare_multimode_case(sim, num_periods=num_periods)
        s_matrix, mode_map, diag = _biorthogonal_s_params_variant(case, options=options)
    indices = phase2._te10_channel_indices(mode_map)
    return s_matrix, indices, diag


def _run_pair(
    *,
    n_modes: int,
    num_periods: float,
    cpml_layers: int,
    extractor: ExtractorName,
    options: BiorthoRunOptions | None = None,
) -> dict[str, float | int | str | bool | None]:
    options = options or BiorthoRunOptions()

    def _run(sim: Simulation) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
        if extractor == "scalar":
            return _run_scalar_s_matrix(sim, num_periods=num_periods)
        return _run_biortho_s_matrix(sim, num_periods=num_periods, options=options)

    pec_freqs = np.linspace(5.0e9, 7.0e9, 6)
    pec_sim = _build_multimode_sim(
        pec_freqs,
        n_modes=n_modes,
        cpml_layers=cpml_layers,
        pec_short_x=0.085,
    )
    pec_s, pec_idx, pec_diag = _run(pec_sim)

    asym_freqs = np.linspace(4.5e9, 8.0e9, 10)
    asym_obstacles = [((0.03, 0.0, 0.0), (0.05, 0.02, 0.02), 6.0)]
    asym_sim = _build_multimode_sim(
        asym_freqs,
        n_modes=n_modes,
        cpml_layers=cpml_layers,
        obstacles=asym_obstacles,
    )
    asym_s, asym_idx, asym_diag = _run(asym_sim)

    metrics: dict[str, float | int | str | bool | None] = {
        "n_modes": int(n_modes),
        "n_total_channels": int(pec_s.shape[0]),
        "num_periods": float(num_periods),
        "cpml_layers": int(cpml_layers),
    }
    metrics.update(_s_metrics(pec_s, pec_idx["left"], pec_idx["right"], kind="pec"))
    metrics.update(_s_metrics(asym_s, asym_idx["left"], asym_idx["right"], kind="asym"))
    for prefix, diag in (("pec", pec_diag), ("asym", asym_diag)):
        for key, value in diag.items():
            if isinstance(value, (float, int, str, bool)) or value is None:
                metrics[f"{prefix}_{key}"] = value
    return metrics


def _control_profiles() -> tuple[phase2.CandidateSpec, np.ndarray, np.ndarray, np.ndarray]:
    candidate = _biortho_candidate()
    a = 22.86e-3
    b = 10.16e-3
    dx = 1.0e-3
    u_widths = np.full(int(round(a / dx)), dx, dtype=np.float64)
    v_widths = np.full(int(round(b / dx)), dx, dtype=np.float64)
    d_a = phase2._candidate_dA(candidate, u_widths, v_widths)
    return candidate, u_widths, v_widths, d_a


def run_control_rows() -> list[MatrixRow]:
    candidate, u_widths, v_widths, d_a = _control_profiles()
    a = 22.86e-3
    b = 10.16e-3
    scalar_candidate = next(c for c in phase2.CANDIDATES if c.name == "FULL_YEE_DROP_GS")
    scalar_profiles = [
        phase2._profile_for_candidate(scalar_candidate, a, b, mode, u_widths, v_widths)
        for mode in [(1, 0), (2, 0)]
    ]
    scalar_gram = phase2._lorentz_gram(scalar_profiles, d_a)
    scalar_err = phase2._synthetic_projection_error(scalar_candidate, scalar_profiles, scalar_gram, d_a)

    biortho_profiles = [
        phase2._profile_for_candidate(candidate, a, b, mode, u_widths, v_widths, h_offset=(0.5, 0.5))
        for mode in [(1, 0), (0, 1), (2, 0)]
    ]
    gram_e = phase2._mode_dot_gram_np([(p[0], p[1]) for p in biortho_profiles], d_a)
    gram_h = phase2._mode_dot_gram_np([(p[2], p[3]) for p in biortho_profiles], d_a)
    coeff = np.array([0.75, -0.30, 0.10], dtype=np.float64)
    recovered_e = phase2._solve_biorthogonal_coefficients(gram_e, gram_e.T @ coeff)
    recovered_h = phase2._solve_biorthogonal_coefficients(gram_h, gram_h.T @ coeff)
    biortho_err = float(max(np.max(np.abs(recovered_e - coeff)), np.max(np.abs(recovered_h - coeff))))

    pure = biortho_profiles[0]
    amp = 0.625
    voltage = float(np.sum(((amp * pure[0]) * pure[0] + (amp * pure[1]) * pure[1]) * d_a))
    current = float(np.sum(((amp * pure[2]) * pure[2] + (amp * pure[3]) * pure[3]) * d_a))

    return [
        MatrixRow(
            "control_rows",
            "control_known_modal_coeffs_scalar",
            "scalar",
            "synthetic_modal_vi",
            {"coefficient_error_max": float(scalar_err), "backward_leakage_abs": 0.0},
            "CONTROL_ONLY",
        ),
        MatrixRow(
            "control_rows",
            "control_known_modal_coeffs_biortho",
            "biortho",
            "synthetic_modal_vi",
            {
                "coefficient_error_max": biortho_err,
                "gram_e_cond": float(np.linalg.cond(gram_e)),
                "gram_h_cond": float(np.linalg.cond(gram_h)),
            },
            "CONTROL_ONLY",
        ),
        MatrixRow(
            "control_rows",
            "control_probe_only_field_snapshot",
            "scalar+biortho",
            "constructed_aperture_fields",
            {
                "modal_voltage_recovery": voltage,
                "modal_current_recovery": current,
                "target_amplitude": amp,
                "vi_recovery_error_max": float(max(abs(voltage - amp), abs(current - amp))),
            },
            "CONTROL_ONLY",
        ),
    ]


def _physical_specs(include_geometry: bool) -> list[tuple[SectionName, str, ExtractorName, int, BiorthoRunOptions]]:
    specs: list[tuple[SectionName, str, ExtractorName, int, BiorthoRunOptions]] = [
        ("physical_rows", "phys_scalar_baseline", "scalar", 10, BiorthoRunOptions()),
        ("physical_rows", "phys_biortho_current_source", "biortho", 10, BiorthoRunOptions()),
        ("physical_rows", "phys_h_phase_minus", "biortho", 10, BiorthoRunOptions(h_time_shift_dt=-0.5)),
        ("physical_rows", "phys_h_phase_plus", "biortho", 10, BiorthoRunOptions(h_time_shift_dt=0.5)),
        ("physical_rows", "phys_h_impedance_scale_0p9", "biortho", 10, BiorthoRunOptions(h_amplitude_scale=0.9)),
        ("physical_rows", "phys_h_impedance_scale_1p0", "biortho", 10, BiorthoRunOptions(h_amplitude_scale=1.0)),
        ("physical_rows", "phys_h_impedance_scale_1p1", "biortho", 10, BiorthoRunOptions(h_amplitude_scale=1.1)),
        (
            "physical_rows",
            "phys_reference_no_reflection_subtract",
            "biortho",
            10,
            BiorthoRunOptions(subtract_reflection=False),
        ),
        (
            "physical_rows",
            "phys_reference_no_through_norm",
            "biortho",
            10,
            BiorthoRunOptions(through_reference_norm=False),
        ),
    ]
    if include_geometry:
        for layers in (8, 10, 12):
            specs.append(("geometry_rows", f"phys_cpml_geometry_sanity_{layers}", "biortho", layers, BiorthoRunOptions()))
    return specs


def run_physical_rows(
    *,
    n_modes: int,
    num_periods: float,
    include_geometry: bool = True,
    row_filter: set[str] | None = None,
) -> list[MatrixRow]:
    rows: list[MatrixRow] = []
    for section, row_id, extractor, cpml_layers, options in _physical_specs(include_geometry):
        if row_filter is not None and row_id not in row_filter:
            continue
        started = time.perf_counter()
        try:
            metrics = _run_pair(
                n_modes=n_modes,
                num_periods=num_periods,
                cpml_layers=cpml_layers,
                extractor=extractor,
                options=options,
            )
            metrics["elapsed_s"] = float(time.perf_counter() - started)
            rows.append(
                MatrixRow(
                    section,
                    row_id,
                    extractor,
                    options.source_variant if extractor == "biortho" else "current_source",
                    metrics,
                    strict_physical_verdict(metrics),
                )
            )
        except Exception as exc:  # keep matrix machine-readable on failures
            rows.append(
                MatrixRow(
                    section,
                    row_id,
                    extractor,
                    options.source_variant if extractor == "biortho" else "current_source",
                    {
                        "skipped": True,
                        "skip_reason": f"{type(exc).__name__}: {exc}",
                        "cpml_layers": int(cpml_layers),
                        "elapsed_s": float(time.perf_counter() - started),
                    },
                    "SKIP_OR_ERROR",
                )
            )
    return rows


def format_sections(rows: Sequence[MatrixRow]) -> str:
    lines: list[str] = []
    for section in ("physical_rows", "control_rows", "geometry_rows"):
        lines.append(f"{section}:")
        for row in rows:
            if row.section == section:
                lines.append(json.dumps(row.to_jsonable(), sort_keys=True, allow_nan=False))
    return "\n".join(lines)


def run_matrix(args: argparse.Namespace) -> list[MatrixRow]:
    rows: list[MatrixRow] = []
    if not args.skip_controls:
        rows.extend(run_control_rows())
    if not args.skip_physical:
        row_filter = set(args.row) if args.row else None
        rows.extend(
            run_physical_rows(
                n_modes=args.n_modes,
                num_periods=args.num_periods,
                include_geometry=not args.skip_geometry,
                row_filter=row_filter,
            )
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="store_true", help="Run the source-path matrix.")
    parser.add_argument("--n-modes", type=int, default=3, help="Multimode count for physical rows.")
    parser.add_argument("--num-periods", type=float, default=40.0, help="FDTD run length in periods.")
    parser.add_argument("--skip-physical", action="store_true", help="Emit only synthetic/control rows.")
    parser.add_argument("--skip-controls", action="store_true", help="Run only physical/geometry rows.")
    parser.add_argument("--skip-geometry", action="store_true", help="Skip CPML geometry rows.")
    parser.add_argument("--row", action="append", default=[], help="Run only named physical row(s). Repeatable.")
    args = parser.parse_args(argv)
    if not args.matrix:
        parser.error("--matrix is required")
    if args.n_modes < 1:
        parser.error("--n-modes must be >= 1")
    if args.num_periods <= 0:
        parser.error("--num-periods must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run_matrix(args)
    print(format_sections(rows))
    physical = [row for row in rows if row.section == "physical_rows"]
    strict_pass = [row.row_id for row in physical if row.verdict == "GO_STRICT"]
    print("summary:")
    print(json.dumps({"strict_pass_rows": strict_pass, "n_rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Crossval 13: production subgrid material-validation example.

This example is intentionally conservative.  The current production subgrid
API is validated only for a guarded one-sided z-slab refinement envelope.  It
does **not** use centered/two-interface material subgridding as a production
claim: material cases outside the guarded envelope are rejected until their
observable and external-crossval evidence is promoted.

The script demonstrates both sides:

1. a guarded one-sided vacuum source/probe z-slab refinement is accepted by
   ``Simulation.validate_subgrid()``,
2. a centered homogeneous dielectric cavity is rejected by production
   validation with precise fail-closed reasons,
3. the same dielectric cavity is solved on the uniform reference lane and
   checked against the analytic TM110 resonance, proving that the material
   setup is physically meaningful even though that centered subgrid case is
   not a promoted production case.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np

from rfx import Box, GaussianPulse, Simulation
from rfx.grid import C0

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "docs" / "research_notes" / "subgrid_material_validation_examples.json"


@dataclass(frozen=True)
class MaterialValidationExampleResult:
    status: str
    vacuum_subgrid_supported: bool
    dielectric_subgrid_supported: bool
    dielectric_rejection_codes: list[str]
    dielectric_uniform_analytic_hz: float
    dielectric_uniform_peak_hz: float
    dielectric_uniform_error_pct: float
    claim_scope: str


def _fft_peak(time_series: np.ndarray, dt: float, center_hz: float) -> float:
    ts = np.asarray(time_series, dtype=np.float64).ravel()
    ts = ts - float(np.mean(ts))
    spectrum = np.abs(np.fft.rfft(ts * np.hanning(len(ts))))
    freqs = np.fft.rfftfreq(len(ts), d=dt)
    band = (freqs >= 0.5 * center_hz) & (freqs <= 1.5 * center_hz)
    if not np.any(band):
        raise RuntimeError("empty FFT search band")
    return float(freqs[band][int(np.argmax(spectrum[band]))])


def _tm110_hz(a: float, b: float, eps_r: float = 1.0) -> float:
    return float((C0 / (2.0 * math.sqrt(eps_r))) * math.sqrt((1.0 / a) ** 2 + (1.0 / b) ** 2))


def run_example() -> MaterialValidationExampleResult:
    a = b = 40e-3
    d = 24e-3
    eps_r = 2.25

    vacuum = Simulation(freq_max=10e9, domain=(a, b, d), boundary="pec", dx=2e-3, cpml_layers=0)
    guarded_z_range = (2e-3, d)
    zlo, zhi = guarded_z_range
    span = zhi - zlo
    vacuum.add_refinement(z_range=guarded_z_range, ratio=2, validation="production")
    vacuum.add_source((a / 3, b / 3, zlo + 0.45 * span), "ez")
    vacuum.add_probe((2 * a / 3, 2 * b / 3, zlo + 0.55 * span), "ez")
    vacuum_report = vacuum.validate_subgrid()

    dielectric_subgrid = Simulation(
        freq_max=8e9, domain=(a, b, d), boundary="pec", dx=2e-3, cpml_layers=0,
    )
    dielectric_subgrid.add_material("dielectric", eps_r=eps_r)
    # Deliberately pad the box so every Yee cell in the finite domain is dielectric.
    dielectric_subgrid.add(
        Box((-1e-3, -1e-3, -1e-3), (a + 1e-3, b + 1e-3, d + 1e-3)),
        material="dielectric",
    )
    dielectric_subgrid.add_refinement(z_range=(8e-3, 16e-3), ratio=2, validation="production")
    dielectric_subgrid.add_source((a / 3, b / 3, d / 2), "ez")
    dielectric_subgrid.add_probe((2 * a / 3, 2 * b / 3, d / 2), "ez")
    dielectric_report = dielectric_subgrid.validate_subgrid()
    rejection_codes = [issue.code for issue in dielectric_report.errors]

    f_ref = _tm110_hz(a, b, eps_r)
    uniform = Simulation(freq_max=2.0 * f_ref, domain=(a, b, d), boundary="pec", dx=2e-3, cpml_layers=0)
    uniform.add_material("dielectric", eps_r=eps_r)
    uniform.add(
        Box((-1e-3, -1e-3, -1e-3), (a + 1e-3, b + 1e-3, d + 1e-3)),
        material="dielectric",
    )
    uniform.add_source((a / 3, b / 3, d / 2), "ez", waveform=GaussianPulse(f0=f_ref, bandwidth=0.8))
    uniform.add_probe((2 * a / 3, 2 * b / 3, d / 2), "ez")
    uniform_result = uniform.run(num_periods=60, compute_s_params=False)
    peak = _fft_peak(np.asarray(uniform_result.time_series).ravel(), float(uniform_result.dt), f_ref)
    err_pct = 100.0 * abs(peak - f_ref) / f_ref

    gates = (
        vacuum_report.supported
        and not dielectric_report.supported
        and "material_weighted_sat_missing" in rejection_codes
        and err_pct < 1.0
    )
    return MaterialValidationExampleResult(
        status="passed" if gates else "failed",
        vacuum_subgrid_supported=bool(vacuum_report.supported),
        dielectric_subgrid_supported=bool(dielectric_report.supported),
        dielectric_rejection_codes=rejection_codes,
        dielectric_uniform_analytic_hz=f_ref,
        dielectric_uniform_peak_hz=peak,
        dielectric_uniform_error_pct=float(err_pct),
        claim_scope=(
            "production subgrid validates only guarded one-sided z slabs; this "
            "example keeps centered material subgridding fail-closed and uses "
            "a uniform analytic reference for the material setup"
        ),
    )


def main() -> int:
    result = run_example()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Crossval 13 subgrid material validation: {result.status.upper()}")
    print(f"  vacuum subgrid supported:      {result.vacuum_subgrid_supported}")
    print(f"  dielectric subgrid supported:  {result.dielectric_subgrid_supported}")
    print(f"  dielectric rejection codes:    {', '.join(result.dielectric_rejection_codes)}")
    print(f"  uniform dielectric analytic:   {result.dielectric_uniform_analytic_hz/1e9:.6f} GHz")
    print(f"  uniform dielectric FFT peak:   {result.dielectric_uniform_peak_hz/1e9:.6f} GHz")
    print(f"  uniform dielectric error:      {result.dielectric_uniform_error_pct:.4f}%")
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

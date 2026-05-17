#!/usr/bin/env python3
"""Stage 1 non-uniform mesh physical evidence gate.

A closed PEC air cavity has an analytic TM110 resonance. This bounded
case stresses the Stage-1 memory-reduction story by using a strongly
non-uniform z profile with 0.25 mm fine cells but 1.0 mm x/y cells. A
uniform mesh at the finest spacing would refine all axes, so the report
should show a large cell-count saving while the FDTD run still recovers
the analytic cavity resonance.

This is a clean physics/CFL/report gate, not a dielectric-interface
accuracy claim: TM110 has p=0, so the resonance mainly validates that
non-uniform fine z cells do not corrupt the closed-cavity RF result.

Run:
    python scripts/stage1_nu_cavity_physics_gate.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from rfx import GaussianPulse, Simulation
from rfx.auto_config import smooth_grading
from rfx.grid import C0


@dataclass(frozen=True)
class Stage1NUGateResult:
    analytic_freq_hz: float
    fft_peak_hz: float
    resonance_error_pct: float
    cells: int
    uniform_fine_cells: int
    cell_savings_factor: float
    segmented_ad_gb: float
    full_ad_gb: float
    n_steps: int
    preflight_issues: tuple[str, ...]


def _fft_peak(time_series: np.ndarray, dt: float, center_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(time_series))
    freqs = np.fft.rfftfreq(len(time_series), d=dt)
    mask = (freqs >= 0.7 * center_hz) & (freqs <= 1.3 * center_hz)
    if not np.any(mask):
        raise RuntimeError("FFT search band contains no bins")
    band_freqs = freqs[mask]
    band_spec = spectrum[mask]
    return float(band_freqs[int(np.argmax(band_spec))])


def run_gate() -> Stage1NUGateResult:
    a = b = 40e-3
    # Coarse air bulk with a deliberately fine z band. Smooth grading
    # avoids artificial reflections from abrupt 4:1 jumps.
    dz_raw = np.array([1.0e-3] * 14 + [0.25e-3] * 8 + [1.0e-3] * 14)
    dz_profile = smooth_grading(dz_raw, max_ratio=1.3)
    d = float(np.sum(dz_profile))
    f_tm110 = float((C0 / 2.0) * np.sqrt((1.0 / a) ** 2 + (1.0 / b) ** 2))

    sim = Simulation(
        freq_max=2.0 * f_tm110,
        domain=(a, b, d),
        dx=1.0e-3,
        dz_profile=dz_profile,
        boundary="pec",
        cpml_layers=0,
    )
    sim.add_source(
        (a / 3.0, b / 3.0, d / 2.0),
        "ez",
        waveform=GaussianPulse(f0=f_tm110, bandwidth=0.8),
    )
    sim.add_probe((2.0 * a / 3.0, 2.0 * b / 3.0, d / 2.0), "ez")

    report = sim.mesh_intelligence_report(n_steps=5_000, checkpoint_every=500)
    result = sim.run(num_periods=20, compute_s_params=False)
    ts = np.asarray(result.time_series).ravel()
    if not np.all(np.isfinite(ts)) or float(np.max(np.abs(ts))) <= 0.0:
        raise RuntimeError("non-uniform cavity run produced invalid or zero signal")

    peak_hz = _fft_peak(ts, float(result.dt), f_tm110)
    err_pct = 100.0 * abs(peak_hz - f_tm110) / f_tm110
    seg_gb = report.ad_memory.ad_segmented_gb if report.ad_memory else None
    if seg_gb is None:
        raise RuntimeError("segmented AD estimate missing from mesh report")

    gate = Stage1NUGateResult(
        analytic_freq_hz=f_tm110,
        fft_peak_hz=peak_hz,
        resonance_error_pct=float(err_pct),
        cells=report.cells,
        uniform_fine_cells=report.uniform_fine_cells,
        cell_savings_factor=float(report.cell_savings_factor),
        segmented_ad_gb=float(seg_gb),
        full_ad_gb=float(report.ad_memory.ad_full_gb),
        n_steps=int(len(ts)),
        preflight_issues=report.preflight_issues,
    )

    if gate.preflight_issues:
        raise AssertionError(f"preflight issues present: {gate.preflight_issues!r}")
    if gate.cell_savings_factor < 40.0:
        raise AssertionError(
            f"cell savings {gate.cell_savings_factor:.2f}x below 40x gate"
        )
    if gate.resonance_error_pct > 1.0:
        raise AssertionError(
            f"resonance error {gate.resonance_error_pct:.3f}% exceeds 1% gate"
        )
    if gate.segmented_ad_gb >= gate.full_ad_gb:
        raise AssertionError(
            f"segmented AD {gate.segmented_ad_gb:.3f} GB should be below "
            f"full AD {gate.full_ad_gb:.3f} GB"
        )
    return gate


def main() -> int:
    gate = run_gate()
    print("Stage 1 NU cavity physics gate: PASS")
    print(f"  analytic TM110:       {gate.analytic_freq_hz/1e9:.6f} GHz")
    print(f"  FDTD FFT peak:        {gate.fft_peak_hz/1e9:.6f} GHz")
    print(f"  resonance error:      {gate.resonance_error_pct:.4f} %")
    print(f"  cells:                {gate.cells:,}")
    print(f"  uniform-fine cells:   {gate.uniform_fine_cells:,}")
    print(f"  cell savings:         {gate.cell_savings_factor:.2f}x")
    print(f"  segmented AD memory:  {gate.segmented_ad_gb:.4f} GB")
    print(f"  full AD memory:       {gate.full_ad_gb:.4f} GB")
    print(f"  n_steps:              {gate.n_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

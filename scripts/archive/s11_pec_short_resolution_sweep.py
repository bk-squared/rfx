#!/usr/bin/env python3
"""Experiment 11: resolve the remaining S11 residuals (PEC-short + slab).

After experiment 10 pinned the slab |S21| deficit as resolution-
limited (dx 1→0.5 mm halved the error), two items remain open:

  - PEC-short |S11| = 0.93 ± 7 %  (target 1.00 — known-issues #3)
  - slab S11 phase mean 22°       (crossval/11 slab reports
                                    phase-gated 5°, but mean is 22°
                                    even after the proper slab-edge
                                    → port-plane de-embedding)

This script runs BOTH geometries at dx = 1 mm and dx = 0.5 mm (CPML
layers scaled so physical PML stays 20 mm) and reports:

  - PEC-short |S11|:  rfx vs unity (ideal full reflection).
  - Slab S11:         rfx vs corrected-β_d analytic Airy,
                      de-embedded by exp(−j·β_v · 2 · d_left)
                      to the rfx reference plane at 50 mm.

Verdict logic:
  - If halving dx halves the error on BOTH → resolution-limited, same
    mechanism as slab S21; the preflight tightening covers it and the
    battery gates can be re-tightened when users run at dx ≤ 0.5 mm.
  - If halving dx helps slab S11 but not PEC-short → PEC-short is a
    mode-projection / staircase artefact, distinct mechanism.
  - If neither improves → neither is resolution-limited; a deeper
    extractor audit is needed.

Run:
    python scripts/s11_pec_short_resolution_sweep.py
Compute time: ~6 min (2× dx × 2 geometries ≈ 4 runs, last one ~5 min).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np


C0 = 2.998e8


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit_phase_residual(s_rfx, s_ref, f_hz, fc):
    """RMS of phase difference after wrap; return (mean_abs_deg, max_deg)."""
    phi_rfx = np.angle(s_rfx)
    phi_ref = np.angle(s_ref)
    d = np.abs(phi_rfx - phi_ref)
    d = np.minimum(d, 2 * np.pi - d)
    return float(np.degrees(d.mean())), float(np.degrees(d.max()))


def main() -> int:
    cv11 = _load_cv11()

    results = {}
    for dx in (0.001, 0.0005):
        cv11.DX_M = dx
        cv11.CPML_LAYERS = int(round(0.020 / dx))
        print(
            f"\n=== dx = {dx*1e3:.2f} mm, CPML_LAYERS = {cv11.CPML_LAYERS} ===",
            flush=True,
        )

        print(f"  [pec-short] run …", flush=True)
        t0 = time.time()
        f_pec, s11_pec, _ = cv11.run_rfx_pec_short()
        dt_pec = time.time() - t0
        print(f"  [pec-short] done in {dt_pec:.1f}s", flush=True)

        print(f"  [slab] run …", flush=True)
        t0 = time.time()
        f_slab, s11_slab, _s21_slab = cv11.run_rfx_slab(2.0, 0.010)
        dt_slab = time.time() - t0
        print(f"  [slab] done in {dt_slab:.1f}s", flush=True)

        # De-embed analytic S11 from slab-edge to port-1 reference plane.
        s11_airy_edge, _ = cv11.analytic_slab_s(f_slab, 2.0, 0.010)
        omega = 2.0 * np.pi * f_slab
        kc = 2.0 * np.pi * cv11.F_CUTOFF_TE10 / C0
        beta_v = np.sqrt(np.maximum((omega / C0) ** 2 - kc ** 2, 0.0))
        slab_center = 0.5 * (cv11.PORT_LEFT_X + cv11.PORT_RIGHT_X)
        d_left = slab_center - 0.5 * 0.010 - 0.050
        s11_airy = s11_airy_edge * np.exp(-1j * beta_v * 2.0 * d_left)

        results[dx] = {
            "pec": (f_pec, s11_pec, dt_pec),
            "slab": (f_slab, s11_slab, s11_airy, dt_slab),
        }

    # PEC-short report.
    print()
    print("=== PEC-short |S11| vs unity (target 1.000) ===")
    print(" f/GHz    |S11|_dx1mm  |S11|_dx0.5mm    deficit_1mm   deficit_0.5mm")
    print("-" * 78)
    _, s11_pec_1, _ = results[0.001]["pec"]
    _, s11_pec_05, _ = results[0.0005]["pec"]
    f_pec = results[0.001]["pec"][0]
    for i in range(len(f_pec)):
        a1 = float(np.abs(s11_pec_1[i]))
        a05 = float(np.abs(s11_pec_05[i]))
        d1 = (a1 - 1.0) * 100.0
        d05 = (a05 - 1.0) * 100.0
        print(
            f" {f_pec[i]/1e9:5.2f}    {a1:.4f}       {a05:.4f}         "
            f"{d1:+6.2f} %      {d05:+6.2f} %"
        )
    rms_pec_1 = float(np.sqrt(np.mean((np.abs(s11_pec_1) - 1.0) ** 2))) * 100.0
    rms_pec_05 = float(np.sqrt(np.mean((np.abs(s11_pec_05) - 1.0) ** 2))) * 100.0
    max_pec_1 = float(np.max(np.abs(np.abs(s11_pec_1) - 1.0))) * 100.0
    max_pec_05 = float(np.max(np.abs(np.abs(s11_pec_05) - 1.0))) * 100.0

    # Slab S11 report.
    print()
    print("=== slab S11 vs corrected-β_d Airy (de-embedded) ===")
    f_slab, s11_slab_1, s11_airy_1, _ = results[0.001]["slab"]
    _, s11_slab_05, s11_airy_05, _ = results[0.0005]["slab"]
    fc = cv11.F_CUTOFF_TE10
    mean_ph_1, max_ph_1 = _fit_phase_residual(s11_slab_1, s11_airy_1, f_slab, fc)
    mean_ph_05, max_ph_05 = _fit_phase_residual(s11_slab_05, s11_airy_05, f_slab, fc)
    mean_mag_1 = float(np.mean(np.abs(np.abs(s11_slab_1) - np.abs(s11_airy_1))))
    mean_mag_05 = float(np.mean(np.abs(np.abs(s11_slab_05) - np.abs(s11_airy_05))))
    print(f"  dx = 1.0 mm:  mag mean diff {mean_mag_1:.4f},  "
          f"phase mean {mean_ph_1:.2f}°, max {max_ph_1:.2f}°")
    print(f"  dx = 0.5 mm:  mag mean diff {mean_mag_05:.4f},  "
          f"phase mean {mean_ph_05:.2f}°, max {max_ph_05:.2f}°")

    # Verdict.
    print()
    print("=== Verdict ===")
    print(f"PEC-short |S11| deficit vs unity:")
    print(f"  dx = 1.0 mm:  RMS {rms_pec_1:.2f} %,  max {max_pec_1:.2f} %")
    print(f"  dx = 0.5 mm:  RMS {rms_pec_05:.2f} %,  max {max_pec_05:.2f} %")
    pec_ratio = rms_pec_05 / max(rms_pec_1, 1e-6)
    print(f"  ratio 0.5 mm / 1 mm = {pec_ratio:.2f}")

    slab_phase_ratio = mean_ph_05 / max(mean_ph_1, 1e-6)
    print(f"Slab S11 phase mean {mean_ph_1:.1f}° → {mean_ph_05:.1f}°, "
          f"ratio = {slab_phase_ratio:.2f}")
    print()

    if pec_ratio < 0.7 and slab_phase_ratio < 0.7:
        print("Verdict: BOTH residuals resolution-limited (ratio < 0.7).")
        print("         Same mechanism as slab S21; preflight covers it.")
    elif slab_phase_ratio < 0.7 and pec_ratio > 0.85:
        print("Verdict: slab S11 is resolution-limited, PEC-short is NOT.")
        print("         PEC-short is a DIFFERENT mechanism — likely mode")
        print("         projection at the staircased PEC boundary, or")
        print("         CPML residue for the total-reflection case.")
    elif pec_ratio < 0.7 and slab_phase_ratio > 0.85:
        print("Verdict: PEC-short is resolution-limited, slab S11 is NOT.")
        print("         Slab S11 needs a dedicated investigation.")
    else:
        print("Verdict: neither scaled cleanly — revisit.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

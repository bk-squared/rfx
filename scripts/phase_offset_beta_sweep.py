#!/usr/bin/env python3
"""Experiment G: is the rfx vs Meep S21 phase offset linear in β(f) or constant?

RESOLVED (W3.4, 2026-07-02): the offset is a TIME-CONVENTION difference,
not a reference-plane shift. Meep fields carry exp(-iωt) (physics
convention) while rfx reports engineering exp(+jωt) S-parameters, so
S21_meep ≈ conj(S21_rfx) at the matched insertion reference. The fitted
(slope ≈ −5.97 mm, intercept ≈ −60.8°, RMS 2.4°) is the affine-in-β
shadow of 2×∠S21_rfx (twice the slab insertion phase) — the fit is good
only because the insertion phase is nearly affine in β over 8.2–12.4 GHz.
Witnesses: rfx ∠S21 matches the analytic Airy insertion phase to ≤0.89°
across the band; ∠conj(S21_meep) matches rfx ∠S21 to ≤2.64° (max over
the 21-frequency band, full precision from the cv11 report row). Do NOT
de-embed by the fitted slope/intercept — conjugate the Meep data instead.
See the time-convention comment and corrected-phase rows in
validation/crossval/11_waveguide_port_wr90.py (slab section).

Physical hypothesis test (original framing, kept for reproducibility):
  - If ∠(S21_rfx / S21_meep) is linear in β(f) → it's a reference-plane
    offset `exp(-jβ·Δx)` — a CONVENTION ISSUE that can be absorbed by
    aligning the reference planes (or documented as a fixed β·Δx shift).
  - If ∠(S21_rfx / S21_meep) is constant in β → it's a fixed PHASE
    CONVENTION difference — sign flip somewhere, source pulse t0
    mismatch, or mode normalization sign. Not a reference plane issue.
  - If neither fits → there's an interaction (perhaps β-dependent but
    not linear) — investigate further.

This script loads the existing Meep reference JSON (produced by the
VESSL job in microwave-energy), runs the same rfx crossval case, and
prints:
  - Per-freq  ∠(S21_rfx) - ∠(S21_meep)
  - A linear fit Δφ(f) = a·β(f) + b (with a = equivalent reference
    shift in metres; b = constant phase offset)
  - Residuals from the fit, to judge goodness

No pytest, no external plotting dependency — pure stdout table.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

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


def _meep_complex(block):
    return np.array([complex(r, i) for r, i in block], dtype=np.complex128)


def _load_meep():
    path = ("/root/workspace/byungkwan-workspace/research/microwave-energy/"
            "results/rfx_crossval_wr90_meep/wr90_meep_reference.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _unwrap_phase_diff_deg(phases_deg: np.ndarray) -> np.ndarray:
    """Unwrap phase differences (in degrees) to avoid 2π discontinuities."""
    rad = np.radians(phases_deg)
    unwrapped = np.unwrap(rad)
    return np.degrees(unwrapped)


def main():
    cv11 = _load_cv11()
    meep = _load_meep()
    if meep is None:
        print("Meep reference JSON not found. Run the VESSL job first.",
              file=sys.stderr)
        return 2

    # Get rfx S21 (slab case) + Meep S21 (pick the highest resolution).
    f_hz, s11_rfx, s21_rfx = cv11.run_rfx_slab(2.0, 0.010)
    r_keys = sorted(k for k in meep.keys() if k.startswith("r") and k != "meta")
    rk = r_keys[-1]  # highest resolution
    s21_meep = _meep_complex(meep[rk]["slab"]["s21"])

    # β(f) in the vacuum-filled WR-90 (continuous form is fine for this
    # diagnostic — the β scaling doesn't change between rfx and Meep).
    fc = cv11.F_CUTOFF_TE10
    omega = 2 * np.pi * f_hz
    k0 = omega / C0
    beta = np.sqrt(np.maximum(k0**2 - (2*np.pi*fc/C0)**2, 0.0))

    # Raw + unwrapped phase difference, sorted by β (monotonic in f here).
    phase_rfx_deg = np.degrees(np.angle(s21_rfx))
    phase_meep_deg = np.degrees(np.angle(s21_meep))
    raw_diff = phase_rfx_deg - phase_meep_deg
    wrapped_diff = ((raw_diff + 180) % 360) - 180   # principal branch [-180, 180)
    unwrapped_diff = _unwrap_phase_diff_deg(wrapped_diff)

    # Linear fit Δφ(rad) = a·β + b   (a = Δx in metres, b in radians).
    rad_diff = np.radians(unwrapped_diff)
    A = np.vstack([beta, np.ones_like(beta)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, rad_diff, rcond=None)
    fit = A @ np.array([slope, intercept])
    fit_resid = rad_diff - fit
    fit_rms_deg = float(np.degrees(np.sqrt(np.mean(fit_resid**2))))
    fit_max_deg = float(np.degrees(np.max(np.abs(fit_resid))))

    print("=== Phase-offset β-linearity test ===")
    print(f"Meep resolution used: {rk}")
    print(f"Frequencies: {f_hz[0]/1e9:.2f}–{f_hz[-1]/1e9:.2f} GHz "
          f"({len(f_hz)} pts)")
    print()
    print(" f/GHz    β/rad·m⁻¹  ∠S21_rfx   ∠S21_meep  Δφ_wrapped  Δφ_unwrapped  fit")
    print("-" * 94)
    for i in range(len(f_hz)):
        print(f" {f_hz[i]/1e9:5.2f}   {beta[i]:9.2f}  "
              f"{phase_rfx_deg[i]:+8.2f}°  {phase_meep_deg[i]:+8.2f}°  "
              f"{wrapped_diff[i]:+8.2f}°   {unwrapped_diff[i]:+8.2f}°  "
              f"{np.degrees(fit[i]):+8.2f}°")

    print()
    print("=== Linear fit Δφ(rad) = slope·β + intercept ===")
    print(f"  slope     = {slope:.5e} m      (= {slope*1e3:.2f} mm equivalent reference shift)")
    print(f"  intercept = {intercept:.4f} rad  (= {np.degrees(intercept):.2f}°)")
    print(f"  fit residuals: RMS={fit_rms_deg:.2f}°, max={fit_max_deg:.2f}°")
    print()
    print("Interpretation (magnitudes, not causality):")
    if abs(slope) * 1e3 > 2.0:
        print(f"  • LARGE β-dependent component (~{slope*1e3:.1f} mm equivalent shift).")
        print(f"    Consistent with a REFERENCE-PLANE misalignment of that magnitude")
        print(f"    between rfx's and Meep's S21 reporting planes.")
    else:
        print(f"  • Small β-dependent component ({slope*1e3:.2f} mm equivalent).")
    if abs(np.degrees(intercept)) > 30.0:
        print(f"  • LARGE constant offset ({np.degrees(intercept):.1f}°).")
        print(f"    Consistent with a fixed PHASE CONVENTION difference — source")
        print(f"    pulse t0, mode normalization sign, or similar.")
    else:
        print(f"  • Small constant offset ({np.degrees(intercept):.1f}°).")

    print()
    print("Verdict guidance:")
    if fit_rms_deg < 10.0:
        print(f"  Linear fit explains the phase offset within RMS {fit_rms_deg:.1f}°.")
        print(f"  The (slope, intercept) pair captures the full offset well. Use")
        print(f"  the reported {slope*1e3:.1f} mm and {np.degrees(intercept):.0f}° to")
        print(f"  de-embed rfx S21 onto Meep's convention (or vice versa).")
    else:
        print(f"  Linear fit residual RMS {fit_rms_deg:.1f}° is nontrivial. The phase")
        print(f"  offset has β-dependence beyond a single reference shift — the")
        print(f"  remaining residuals themselves need characterisation.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

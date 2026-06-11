#!/usr/bin/env python3
"""Experiment 8: rfx slab S21 phase vs analytic Airy — same β-linear fit.

Purpose: isolate whether the -5.87 mm / -57° slab S21 phase offset
observed between rfx and Meep is (a) rfx diverging from *truth* (analytic
Airy) or (b) rfx and Meep both diverging from truth in opposite
directions by similar amounts.

If the rfx-vs-analytic fit matches rfx-vs-Meep → Meep agrees with
analytic, and rfx is the oddball. Direct evidence the next session
should be spent on rfx's extractor, not on the reference convention.

If rfx-vs-analytic fit differs → Meep itself has an offset, and the
rfx vs Meep gap is a convention disagreement where "truth" sits in
between.

Uses the same analytic formula as examples/crossval/11, same
frequency grid. No VESSL, no external data needed beyond rfx
itself.

Run:
    python scripts/rfx_vs_analytic_slab_phase.py
"""

from __future__ import annotations

import importlib.util
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


def _de_embed_slab_edges(s21_airy, f_hz, fc_hz, slab_length_m, ref_left_m,
                         ref_right_m, slab_center_m):
    """Port rfx/meep reference planes at ref_left/ref_right (50/150 mm),
    but analytic Airy formula is referenced to the slab edges.
    Propagate analytic S21 from slab-edge convention out to the
    reference planes using the empty-guide β.
    """
    omega = 2 * np.pi * f_hz
    k0 = omega / C0
    kc = 2 * np.pi * fc_hz / C0
    beta_empty = np.sqrt(np.maximum(k0 ** 2 - kc ** 2, 0.0))
    dist_left = (slab_center_m - 0.5 * slab_length_m) - ref_left_m
    dist_right = ref_right_m - (slab_center_m + 0.5 * slab_length_m)
    # S21 at left-ref .. slab .. right-ref adds exp(-jβ·(dist_left+dist_right))
    return s21_airy * np.exp(-1j * beta_empty * (dist_left + dist_right))


def fit_phase_offset(s21_rfx, s21_ref, f_hz, fc_hz):
    omega = 2 * np.pi * f_hz
    k0 = omega / C0
    beta = np.sqrt(np.maximum(k0 ** 2 - (2 * np.pi * fc_hz / C0) ** 2, 0.0))
    raw = np.degrees(np.angle(s21_rfx)) - np.degrees(np.angle(s21_ref))
    wrapped = ((raw + 180) % 360) - 180
    unwrapped = np.degrees(np.unwrap(np.radians(wrapped)))
    rad = np.radians(unwrapped)
    A = np.vstack([beta, np.ones_like(beta)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, rad, rcond=None)
    fit = A @ np.array([slope, intercept])
    rms = float(np.degrees(np.sqrt(np.mean((rad - fit) ** 2))))
    return slope, intercept, rms, beta, unwrapped


def main() -> int:
    cv11 = _load_cv11()
    print("=== Running rfx slab (this takes a minute) ===", flush=True)
    eps_r, slab_L = 2.0, 0.010
    f_hz, _s11_rfx, s21_rfx = cv11.run_rfx_slab(eps_r, slab_L)
    fc = cv11.F_CUTOFF_TE10
    _s11_airy, s21_airy_slabedge = cv11.analytic_slab_s(f_hz, eps_r, slab_L)

    # De-embed from slab-edges to rfx reference planes (0.050, 0.150).
    s21_airy = _de_embed_slab_edges(
        s21_airy_slabedge, f_hz, fc, slab_L,
        ref_left_m=0.050, ref_right_m=0.150,
        slab_center_m=0.5 * (cv11.PORT_LEFT_X + cv11.PORT_RIGHT_X),
    )

    slope, intercept, rms, beta, diff = fit_phase_offset(
        s21_rfx, s21_airy, f_hz, fc,
    )

    print()
    print(f"  WR-90 slab (εr={eps_r}, L={slab_L*1e3:.0f} mm), "
          f"rfx reference planes at 50/150 mm (matched to Meep monitors)")
    print()
    print(" f/GHz    β/m⁻¹    |S21|_rfx  |S21|_airy   ∠S21_rfx    ∠S21_airy    Δφ")
    print("-" * 92)
    for i in range(len(f_hz)):
        print(f" {f_hz[i]/1e9:5.2f}    {beta[i]:7.2f}   "
              f"{abs(s21_rfx[i]):.4f}     {abs(s21_airy[i]):.4f}     "
              f"{np.degrees(np.angle(s21_rfx[i])):+8.2f}°   "
              f"{np.degrees(np.angle(s21_airy[i])):+8.2f}°   "
              f"{diff[i]:+7.2f}°")

    print()
    print("=== Linear fit Δφ(rad) = slope·β + intercept ===")
    print(f"  rfx − analytic (slab-edge de-embedded):")
    print(f"    slope     = {slope*1e3:+7.3f} mm equivalent reference shift")
    print(f"    intercept = {np.degrees(intercept):+7.2f}°")
    print(f"    fit RMS   = {rms:.2f}°")
    print()
    print("Reference — handover v2 rfx vs Meep fit for same geometry:")
    print(f"    slope     = -5.873 mm")
    print(f"    intercept = -57.27°")
    print(f"    fit RMS   = 2.28°")
    print()
    if abs(slope * 1e3 - (-5.873)) < 1.0 and abs(np.degrees(intercept) - (-57.27)) < 10.0:
        print("Verdict: rfx-vs-analytic ≈ rfx-vs-Meep → Meep ≈ analytic.")
        print("         rfx is the oddball; next session should audit")
        print("         `rfx/sources/waveguide_port.py` slab extraction.")
    else:
        print("Verdict: rfx-vs-analytic DIFFERS from rfx-vs-Meep.")
        print("         Both Meep and rfx differ from analytic; truth is")
        print("         not simply on Meep's side. Needs a third independent")
        print("         extractor (OpenEMS MSL-style or direct field overlap).")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

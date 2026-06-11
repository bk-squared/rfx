#!/usr/bin/env python3
"""Experiment 4: fit (slope, intercept) to the phase offset PER GEOMETRY.

Slab fit was (slope=-5.87 mm, intercept=-57°). Applying it to PEC short
didn't help — correction doesn't transfer. That was the "REJECT universal
shift" result.

This experiment fits each geometry SEPARATELY and reports (slope_i,
intercept_i) per case. Patterns to look for:

- Common intercept, different slopes → "β-dependent geometry factor".
- Common slope, different intercepts → "universal reference shift +
  geometry-specific constant".
- Both differ → fully geometry-specific.

Empty, PEC short, and slab are run. Empty is a trivial case — if rfx
and Meep agree there (Δφ ≈ 0), then we know the convention shift is
NOT universal. If empty shows a nonzero slope/intercept, that's the
universal piece.
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


def _beta_from_freqs(f_hz, fc):
    omega = 2 * np.pi * f_hz
    k0 = omega / C0
    return np.sqrt(np.maximum(k0**2 - (2*np.pi*fc/C0)**2, 0.0))


def _fit_phase(s_rfx, s_meep, beta):
    """Unwrap Δφ and linear-fit to a·β + b."""
    raw = np.degrees(np.angle(s_rfx)) - np.degrees(np.angle(s_meep))
    wrapped = ((raw + 180) % 360) - 180
    rad = np.unwrap(np.radians(wrapped))
    A = np.vstack([beta, np.ones_like(beta)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, rad, rcond=None)
    fit = A @ np.array([slope, intercept])
    resid = rad - fit
    rms = float(np.degrees(np.sqrt(np.mean(resid**2))))
    return slope, intercept, rms


def main():
    cv11 = _load_cv11()
    meep = _load_meep()
    if meep is None:
        print("Meep JSON not found.", file=sys.stderr)
        return 2
    r_keys = sorted(k for k in meep.keys() if k.startswith("r") and k != "meta")
    rk = r_keys[-1]
    fc = cv11.F_CUTOFF_TE10

    cases = []

    # Empty guide — S21 ratio; rfx and Meep BOTH measure dev/ref which
    # is two copies of an empty guide => ideally S21=1.0+0j exactly.
    # Any phase deviation is numerical noise; this is the "zero-point"
    # check.
    print("Running empty guide (rfx)...", flush=True)
    f_hz, s11_emp_rfx, s21_emp_rfx = cv11.run_rfx_empty()
    beta = _beta_from_freqs(f_hz, fc)
    s21_emp_meep = _meep_complex(meep[rk]["empty"]["s21"])
    s11_emp_meep = _meep_complex(meep[rk]["empty"]["s11"])
    slope, intercept, rms = _fit_phase(s21_emp_rfx, s21_emp_meep, beta)
    cases.append(("empty  S21", slope, intercept, rms,
                  float(np.abs(s21_emp_rfx).mean()), float(np.abs(s21_emp_meep).mean())))
    # S11 empty: both ≈0, phase is noise — skip slope fit, just report magnitudes.

    # PEC short — S11 is the dominant (|S11|≈1).
    print("Running PEC short (rfx)...", flush=True)
    f_hz, s11_ps_rfx, s21_ps_rfx = cv11.run_rfx_pec_short()
    s11_ps_meep = _meep_complex(meep[rk]["pec_short"]["s11"])
    slope, intercept, rms = _fit_phase(s11_ps_rfx, s11_ps_meep, beta)
    cases.append(("pec-sh S11", slope, intercept, rms,
                  float(np.abs(s11_ps_rfx).mean()), float(np.abs(s11_ps_meep).mean())))

    # Slab — S21 (the existing fit target).
    print("Running slab (rfx)...", flush=True)
    f_hz, s11_sl_rfx, s21_sl_rfx = cv11.run_rfx_slab(2.0, 0.010)
    s21_sl_meep = _meep_complex(meep[rk]["slab"]["s21"])
    s11_sl_meep = _meep_complex(meep[rk]["slab"]["s11"])
    slope, intercept, rms = _fit_phase(s21_sl_rfx, s21_sl_meep, beta)
    cases.append(("slab   S21", slope, intercept, rms,
                  float(np.abs(s21_sl_rfx).mean()), float(np.abs(s21_sl_meep).mean())))
    # Slab S11 too — different fit.
    slope, intercept, rms = _fit_phase(s11_sl_rfx, s11_sl_meep, beta)
    cases.append(("slab   S11", slope, intercept, rms,
                  float(np.abs(s11_sl_rfx).mean()), float(np.abs(s11_sl_meep).mean())))

    print()
    print("=== Per-geometry linear fit of Δφ(rad) = slope·β + intercept ===")
    print(" geometry    slope[mm]   intercept[°]   RMS[°]   |rfx|_mean   |meep|_mean")
    print("-" * 78)
    for name, slope, intercept, rms, m_rfx, m_meep in cases:
        print(f" {name}  {slope*1e3:+8.2f}   {np.degrees(intercept):+7.1f}      "
              f"{rms:6.2f}     {m_rfx:.4f}      {m_meep:.4f}")

    print()
    print("=== Interpretation checks ===")

    # Collect slopes and intercepts for non-trivial cases
    slopes_mm = [c[1]*1e3 for c in cases]
    intercepts_deg = [np.degrees(c[2]) for c in cases]
    print(f"  slopes range      : [{min(slopes_mm):+.2f}, {max(slopes_mm):+.2f}] mm")
    print(f"  intercepts range  : [{min(intercepts_deg):+.1f}, {max(intercepts_deg):+.1f}]°")

    # If ALL geometries share a slope within ~1 mm → universal β-shift
    # exists (hypothesis 4 of session). If they cluster near 0 for
    # empty/PEC but spread for slab, the offset is slab-specific.
    if max(slopes_mm) - min(slopes_mm) < 1.5:
        print("  → slope is roughly CONSTANT across geometries: universal β-shift.")
    else:
        print(f"  → slope varies by {max(slopes_mm)-min(slopes_mm):.2f} mm across geometries: "
              f"geometry-specific component dominates.")
    if max(intercepts_deg) - min(intercepts_deg) < 15.0:
        print("  → intercept is roughly CONSTANT across geometries: universal "
              "constant phase shift.")
    else:
        print(f"  → intercept varies by {max(intercepts_deg)-min(intercepts_deg):.1f}° "
              f"across geometries: geometry-specific phase.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

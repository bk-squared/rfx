"""rfx slab S11 phase vs analytic Airy — linear-β fit, mirror of
scripts/rfx_vs_analytic_slab_phase.py for S21 (which already shows RMS
0.22°).

Crossval/11 reports slab S11 phase mean=22°, max=143° vs analytic, but
S21 clears 5° gate easily (mean 0.36°). The 4-way table shows rfx and
OpenEMS phase agreement within ~5° at most frequencies, with up to 40°
discrepancy at a couple of mid-band points. This spike isolates:

  (1) Per-frequency rfx vs analytic S11 phase difference, with the
      same -2β_v·d_left de-embedding the crossval/11 script applies.
  (2) Whether a single-parameter linear-β shift gives a tight RMS,
      mirroring the S21 result. If yes, the shift formula in
      crossval/11 has a constant-distance error. If no, rfx S11
      phase has a frequency-dependent error not matched by a simple
      reference-plane shift.
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
        "..", "..", "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    cv = _load_cv11()
    eps_r = 2.0
    slab_L = 0.010
    f_hz, s11_rfx, s21_rfx = cv.run_rfx_slab(eps_r, slab_L)
    s11_airy_edge, s21_airy_edge = cv.analytic_slab_s(f_hz, eps_r, slab_L)

    omega = 2.0 * np.pi * f_hz
    kc = 2.0 * np.pi * cv.F_CUTOFF_TE10 / C0
    beta_v = np.sqrt(np.maximum((omega / C0) ** 2 - kc ** 2, 0.0))

    slab_center = 0.5 * (cv.PORT_LEFT_X + cv.PORT_RIGHT_X)
    d_left = slab_center - 0.5 * slab_L - 0.050  # script: 45 mm

    s11_ref = s11_airy_edge * np.exp(-1j * beta_v * 2.0 * d_left)

    raw = np.degrees(np.angle(s11_rfx)) - np.degrees(np.angle(s11_ref))
    wrapped = ((raw + 180) % 360) - 180

    print(f"\n=== rfx slab S11 vs analytic-de-embedded (d_left={d_left*1e3:.1f}mm) ===")
    print(f"{'f_GHz':>6} {'beta':>8} {'|rfx|':>7} {'|airy|':>8}  "
          f"{'rfx∠':>9} {'airy_shifted∠':>14} {'wrapped_diff':>12}")
    for i, f in enumerate(f_hz):
        print(f"{f/1e9:6.2f} {beta_v[i]:8.2f} {abs(s11_rfx[i]):7.4f} "
              f"{abs(s11_airy_edge[i]):8.4f}  "
              f"{np.degrees(np.angle(s11_rfx[i])):+9.2f}° "
              f"{np.degrees(np.angle(s11_ref[i])):+13.2f}° "
              f"{wrapped[i]:+12.2f}°")

    # Linear-β fit on UNWRAPPED phase difference, mirroring S21 script.
    unwrapped = np.degrees(np.unwrap(np.radians(wrapped)))
    rad = np.radians(unwrapped)
    A = np.vstack([beta_v, np.ones_like(beta_v)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, rad, rcond=None)
    fit = A @ np.array([slope, intercept])
    rms = float(np.degrees(np.sqrt(np.mean((rad - fit) ** 2))))
    print(f"\n=== Linear-β fit Δφ(rad) = slope·β + intercept ===")
    print(f"  slope     = {slope*1000:.3f} mm equivalent reference shift")
    print(f"  intercept = {np.degrees(intercept):.2f}°")
    print(f"  fit RMS   = {rms:.2f}°")
    if rms < 5.0:
        print(f"\nVerdict: linear-β fit captures the discrepancy. The crossval/11")
        print(f"  S11 shift formula is missing a {slope*1000:.2f} mm constant offset.")
    else:
        print(f"\nVerdict: linear-β fit RMS too large ({rms:.2f}°) — the discrepancy")
        print(f"  is NOT a simple reference-plane shift; rfx S11 has a frequency-")
        print(f"  dependent error not explained by constant distance.")


if __name__ == "__main__":
    main()

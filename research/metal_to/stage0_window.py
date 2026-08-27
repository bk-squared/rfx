"""Phase-2 Stage 0 — how long a record does the dual-band spec actually need?

The Phase-1 window (10 periods at F_MAX) gives a DFT resolution of 0.90 GHz. The
WLAN band centres are 0.525 GHz apart, so that window cannot even separate the two
notches, let alone resolve the 200/100 MHz notch bandwidths. This script measures
the requirement instead of guessing it, on the classical two-stub design that will
also serve as baseline D.

Method: build two hard-PEC open stubs (lambda/4 at 5.25 and 5.775 GHz) inside the
bounded 12 x 9 mm design box, solve with the imperative compute_msl_s_matrix at
increasing num_periods, and record for each window:
  * rfx's ring-down settling witness (must be <= -40 dB before any number is quoted)
  * the depth and centre of each notch
  * whether the two notches are separately resolved

Output: research/metal_to/out_vessl/stage0/window_<periods>.json

Run:  python research/metal_to/stage0_window.py --periods 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "validation" / "tmtt_paper"))

import msl_stub_notch_tuning as notch  # noqa: E402
from rfx import Box  # noqa: E402

OUT = Path(os.environ.get("OUTPUT_DIR", HERE / "out_vessl" / "stage0"))
OUT.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8
F_LO, F_HI = 5.25e9, 5.775e9          # WLAN band centres (5.15-5.35, 5.725-5.825)
BOX_X, BOX_Y = 12.0e-3, 9.0e-3        # bounded design box (see PLAN_phase2_dualband.md)
SMOKE = os.environ.get("SMOKE", "0") == "1"


def quarter_wave(f):
    return C0 / (f * np.sqrt(notch.EPS_EFF)) / 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=float, required=True)
    ap.add_argument("--sep_mm", type=float, default=8.0,
                    help="centre-to-centre stub separation along the line (mm). "
                         "Schiffman & Matthaei prescribe 3*lambda_g/4 = 24 mm to "
                         "avoid interaction; the 12 mm box forbids that, which is "
                         "the point of the benchmark.")
    args = ap.parse_args()

    import jax.numpy as jnp
    nf = 21 if SMOKE else 161
    freqs = np.linspace(4.0e9, 8.0e9, nf)
    freqs_j = jnp.asarray(freqs, dtype=jnp.float32)
    sim, _, trace_y_hi, _, _ = notch.build_sim(freqs_j)
    grid = sim._build_grid()

    l_lo, l_hi = quarter_wave(F_LO), quarter_wave(F_HI)
    x_mid = notch.LX / 2.0
    sep = args.sep_mm * 1e-3
    half_w = notch.W_TRACE / 2.0
    for x_c, L, tag in ((x_mid - sep / 2, l_lo, "lo"), (x_mid + sep / 2, l_hi, "hi")):
        sim.add(Box((x_c - half_w, trace_y_hi, notch.H_SUB),
                    (x_c + half_w, trace_y_hi + L, notch.H_SUB + notch.DX)),
                material="pec")
    object.__setattr__(sim._msl_ports[1], "excite", True)

    in_box = (sep / 2 + half_w) * 2 <= BOX_X and max(l_lo, l_hi) <= BOX_Y
    print(f"[stage0] stubs {l_lo*1e3:.2f} / {l_hi*1e3:.2f} mm, separation "
          f"{args.sep_mm:.1f} mm -> fits the {BOX_X*1e3:.0f}x{BOX_Y*1e3:.0f} mm box: {in_box}")
    print(f"[stage0] 3*lambda_g/4 anti-coupling spacing would need "
          f"{3*quarter_wave(5.5e9)*1e3:.1f} mm — the box forbids it")
    sim.preflight()

    t0 = time.time()
    res = sim.compute_msl_s_matrix(freqs=freqs_j, num_periods=float(args.periods))
    wall = time.time() - t0

    settling = np.asarray(getattr(res, "settling_db", []), dtype=float).ravel()
    reliable = np.asarray(getattr(res, "reliable", []), dtype=bool).ravel()
    worst = float(np.max(settling)) if settling.size else float("nan")
    f = np.asarray(res.freqs)
    db21 = 20 * np.log10(np.abs(np.asarray(res.S[1, 0, :])) + 1e-30)

    def band_min(lo, hi):
        m = (f >= lo) & (f <= hi)
        if not m.any():
            return None, None
        i = int(np.argmin(np.where(m, db21, 1e9)))
        return float(db21[i]), float(f[i] / 1e9)

    d_lo, f_lo = band_min(4.6e9, 5.5e9)
    d_hi, f_hi = band_min(5.5e9, 6.4e9)
    # are the two notches separately resolved? require a peak between them
    mid = (f > min(f_lo, f_hi) * 1e9) & (f < max(f_lo, f_hi) * 1e9)
    sep_db = float(np.max(db21[mid]) - max(d_lo, d_hi)) if mid.any() else 0.0
    dt = float(grid.dt)
    n_steps = int(round(args.periods * (1.0 / notch.F_MAX) / dt))

    out = dict(periods=args.periods, n_steps=n_steps,
               record_ns=n_steps * dt * 1e9, dft_res_GHz=1.0 / (n_steps * dt) / 1e9,
               settling_worst_db=worst, settled=bool(worst <= -40.0),
               reliable_bins=[int(reliable.sum()), int(reliable.size)],
               stub_lo_mm=l_lo * 1e3, stub_hi_mm=l_hi * 1e3, sep_mm=args.sep_mm,
               notch_lo_db=d_lo, notch_lo_GHz=f_lo,
               notch_hi_db=d_hi, notch_hi_GHz=f_hi,
               separation_db=sep_db, resolved=bool(sep_db >= 3.0),
               wall_s=round(wall, 1),
               freqs_GHz=[float(x) / 1e9 for x in f],
               s21_db=[float(x) for x in db21])
    (OUT / f"window_{args.periods:.0f}.json").write_text(json.dumps(out, indent=2))
    print(f"[stage0] periods={args.periods:.0f} n_steps={n_steps} "
          f"T={out['record_ns']:.2f}ns res={out['dft_res_GHz']:.3f}GHz")
    print(f"[stage0] settling worst={worst:.1f} dB -> "
          f"{'SETTLED' if out['settled'] else 'NOT SETTLED'}; reliable "
          f"{out['reliable_bins'][0]}/{out['reliable_bins'][1]}")
    print(f"[stage0] notch lo {d_lo:+.1f} dB @ {f_lo:.3f} GHz (target 5.250) · "
          f"hi {d_hi:+.1f} dB @ {f_hi:.3f} GHz (target 5.775)")
    print(f"[stage0] peak between notches {sep_db:.1f} dB -> "
          f"{'RESOLVED' if out['resolved'] else 'MERGED — window too short'} "
          f"({wall:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

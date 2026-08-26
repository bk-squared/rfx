"""Cross-validation tier 1 — rebuild the optimized design as real PEC boxes.

Every verdict number so far came from the SAME differentiable operator at its
PEC limit (occupancy = 1 through the Kottke fold). That cannot catch an
artifact of the operator itself. This script closes that hole the way the
validated notch example does: it re-creates the binarized design as explicit
``Box(material="pec")`` geometry and solves it with the imperative
``compute_msl_s_matrix`` path — independent geometry rasterization, independent
S-parameter extractor, absolute (not empty-line-normalized) numbers.

Designs (all on the identical line/substrate/ports):
  empty   through-line only                    -> normalization reference
  oracle  analytic lambda/4 stub as one Box     -> classical baseline
  B_stub  phase-1c winner (damped gray, seeded) -> per-cell mask -> boxes
  C_low   phase-1c free-form discovery          -> per-cell mask -> boxes

The per-cell binary mask is run-length merged along y within each x column, so
a ~280-cell design becomes a few dozen boxes rather than 280.

Run:  python research/metal_to/xval1_imperative.py --design B_stub
Out:  research/metal_to/out_vessl/xval1/<design>.json
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

OUT = Path(os.environ.get("OUTPUT_DIR", HERE / "out_vessl" / "xval1"))
OUT.mkdir(parents=True, exist_ok=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
FREQS = np.linspace(4.5e9, 8.5e9, 9 if SMOKE else 41)
NUM_PERIODS = float(os.environ.get("XV_PERIODS", "8" if SMOKE else "80"))

DESIGNS = {
    "B_stub": ("phase1c_B_stub", "phase1c_stub_i150_B_final.npz"),
    "C_low": ("phase1c_C_low", "phase1c_low_i150_C_final.npz"),
}


def mask_to_boxes(hard, ix_lo, iy_lo, iz, grid):
    """Run-length merge a per-cell metal mask into PEC boxes (cell edges)."""
    pad_x, pad_y, pad_z = grid.axis_pads
    dx = notch.DX
    boxes = []
    ndx, ndy = hard.shape
    for i in range(ndx):
        j = 0
        while j < ndy:
            if hard[i, j] < 0.5:
                j += 1
                continue
            j0 = j
            while j < ndy and hard[i, j] >= 0.5:
                j += 1
            gi, gj0, gj1 = ix_lo + i, iy_lo + j0, iy_lo + j
            x_lo = (gi - pad_x) * dx
            x_hi = (gi + 1 - pad_x) * dx
            y_lo = (gj0 - pad_y) * dx
            y_hi = (gj1 - pad_y) * dx
            z_lo = (iz - pad_z) * dx
            z_hi = (iz + 1 - pad_z) * dx
            boxes.append(((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi)))
    return boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=("empty", "oracle", "B_stub", "C_low"),
                    required=True)
    ap.add_argument("--stub_mm", type=float, default=None,
                    help="oracle only: override the stub length (mm). The analytic "
                         "lambda/4 length does NOT land the notch on target at this "
                         "mesh (eps_eff staircase), so a fair classical baseline is "
                         "the mesh-CALIBRATED length, swept with this flag.")
    args = ap.parse_args()
    d = args.design

    import jax.numpy as jnp
    freqs_j = jnp.asarray(FREQS, dtype=jnp.float32)
    sim, y_trace, trace_y_hi, _, _ = notch.build_sim(freqs_j)
    grid = sim._build_grid()

    # region indices, identical to the optimization script
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    xc = (np.arange(nx) - pad_x + 0.5) * notch.DX
    yc = (np.arange(ny) - pad_y + 0.5) * notch.DX
    zc = (np.arange(nz) - pad_z + 0.5) * notch.DX
    x_mid = notch.LX / 2.0
    ix = np.where((xc >= x_mid - 3.0e-3 / 2) & (xc <= x_mid + 3.0e-3 / 2))[0]
    iy = np.where((yc >= trace_y_hi) & (yc <= trace_y_hi + 12.0e-3))[0]
    iz = int(np.argmin(np.abs(zc - (notch.H_SUB + 0.5 * notch.DX))))

    n_boxes = 0
    stub_len = None
    if d == "oracle":
        stub_len = (args.stub_mm * 1e-3 if args.stub_mm is not None
                    else float(notch.L_TARGET_AN))
        stub_x_lo = x_mid - notch.W_TRACE / 2.0
        stub_x_hi = x_mid + notch.W_TRACE / 2.0
        sim.add(Box((stub_x_lo, trace_y_hi, notch.H_SUB),
                    (stub_x_hi, trace_y_hi + stub_len,
                     notch.H_SUB + notch.DX)), material="pec")
        n_boxes = 1
        print(f"[xval1:oracle] stub length {stub_len*1e3:.3f} mm "
              f"(analytic {float(notch.L_TARGET_AN)*1e3:.3f} mm)")
    elif d in DESIGNS:
        sub, fname = DESIGNS[d]
        hard = np.load(HERE / "out_vessl" / sub / fname)["hard"]
        boxes = mask_to_boxes(hard, int(ix[0]), int(iy[0]), iz, grid)
        for lo, hi in boxes:
            sim.add(Box(lo, hi), material="pec")
        n_boxes = len(boxes)
        print(f"[xval1:{d}] mask fill={hard.mean():.3f} -> {n_boxes} PEC boxes")

    # compute_msl_s_matrix drives both ports; build_sim disabled port 1
    object.__setattr__(sim._msl_ports[1], "excite", True)
    for m in sim.preflight():
        print(f"  preflight: {m[:110]}")

    print(f"[xval1:{d}] grid={grid.shape} freqs={len(FREQS)} "
          f"periods={NUM_PERIODS} boxes={n_boxes} smoke={SMOKE}")
    t0 = time.time()
    res = sim.compute_msl_s_matrix(freqs=freqs_j, num_periods=NUM_PERIODS)
    wall = time.time() - t0

    # Carry the solver's own reliability verdict into the record: a high-Q
    # notch read from a truncated record is a DFT artifact, and rfx says so.
    settling = np.asarray(getattr(res, "settling_db", []), dtype=float).ravel()
    reliable = np.asarray(getattr(res, "reliable", []), dtype=bool).ravel()
    pcorr = np.asarray(getattr(res, "passivity_correction", []), dtype=float).ravel()
    settled = bool(settling.size and float(np.max(settling)) <= -40.0)
    print(f"[xval1:{d}] settling worst={float(np.max(settling)) if settling.size else float('nan'):.1f} dB "
          f"(need <= -40) -> {'SETTLED' if settled else 'NOT SETTLED — do not quote'}; "
          f"reliable bins {int(reliable.sum())}/{reliable.size}; "
          f"worst passivity corr {float(np.max(pcorr)) if pcorr.size else 0.0:.3f}")

    f = np.asarray(res.freqs)
    s21 = np.asarray(res.S[1, 0, :])
    s11 = np.asarray(res.S[0, 0, :])
    db21 = 20 * np.log10(np.abs(s21) + 1e-30)
    it = int(np.argmin(np.abs(f - notch.F_TARGET)))
    out = dict(design=d, stub_mm=(None if stub_len is None else stub_len*1e3), n_boxes=n_boxes, wall_s=round(wall, 1),
               num_periods=NUM_PERIODS, freqs_GHz=[float(x) / 1e9 for x in f],
               s21_db=[float(x) for x in db21],
               s21_db_at_target=float(db21[it]),
               s11_db=[float(20 * np.log10(abs(x) + 1e-30)) for x in s11],
               f_min_GHz=float(f[int(np.argmin(db21))] / 1e9),
               depth_min_db=float(np.min(db21)),
               settling_worst_db=(float(np.max(settling)) if settling.size else None),
               settled=settled,
               reliable_bins=[int(reliable.sum()), int(reliable.size)],
               passivity_worst=(float(np.max(pcorr)) if pcorr.size else None))
    tag = d if stub_len is None else f"{d}_{stub_len*1e3:.2f}mm"
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[xval1:{d}] |S21|(6 GHz)={db21[it]:+.2f} dB · "
          f"min {out['depth_min_db']:+.2f} dB @ {out['f_min_GHz']:.2f} GHz · "
          f"{wall:.0f}s -> {OUT}/{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
  oracle  analytic lambda/4 stub                -> classical baseline
  B_stub  phase-1c winner (damped gray, seeded) -> per-cell mask -> boxes
  C_low   phase-1c free-form discovery          -> per-cell mask -> boxes

ONE geometry pathway for every arm (2026-08-27 review, defect D6). The oracle
used to be added as a single continuous ``Box`` while the mask arms went
through ``floor(L/dx)``. rfx rasterizes a ``Box`` on NODES, half-open
``[lo, hi)`` (see ``rfx.geometry.csg.Box``), so a stub drawn from the trace
edge realizes ``floor(L/dx)`` or ``floor(L/dx)+1`` rows depending on
``frac(L/dx)``. MEASURED on this fixture with rfx's own rasterizer, the
continuous path leaves the ``floor(L/dx)`` lattice for **7 of 10** fractions
(the stub root sits at y/dx = 16.7244, so the extra row appears from
``frac(L/dx) >= 0.276``; e.g. L/dx = 58.30 -> 59 rows Box, 58 rows mask).
One row is 127 um, 1.5 % of a lambda/4 stub at 5.25 GHz — about 80 MHz,
comparable to the entire 100 MHz upper WLAN band. A baseline swept through
one path against designs evaluated through the other sits on a different
length lattice: the exact asymmetry that forced the Phase-1 retraction.

The oracle now goes through ``phase2_fixture.mask_from_stubs`` and every arm
is emitted by ``phase2_fixture.boxes_from_mask``, which reproduces the
cell-edge, per-x-column run-length convention this file used to implement
locally (checked box-for-box against the deleted ``mask_to_boxes`` on both
phase-1c designs: 6 and 50 boxes, identical coordinates). Two consequences of
routing the oracle through the mask lattice, both measured and both intended:
its length is now ``floor(L/dx)`` at every fraction, and at ``x = LX/2`` its
5-node x footprint moves from nodes 137-141 to 136-140 — a 127 um shift ALONG
the line, which is where the mask arms' own metal sits. Stub position along a
uniform line does not set the transmission zero (stub length does), and the
stub stays rooted on the trace either way.

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
sys.path.insert(0, str(HERE))

import msl_stub_notch_tuning as notch  # noqa: E402
import phase2_fixture as fx  # noqa: E402

# SMOKE runs must never write into the tracked results tree: a 0.5-period
# smoke silently clobbered three committed result files during review.
OUT = Path(os.environ.get(
    "OUTPUT_DIR",
    HERE / "out_smoke" / "xval1" if os.environ.get("SMOKE", "0") == "1"
    else HERE / "out_vessl" / "xval1"))
OUT.mkdir(parents=True, exist_ok=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
FREQS = np.linspace(4.5e9, 8.5e9, 9 if SMOKE else 41)
NUM_PERIODS = float(os.environ.get("XV_PERIODS", "8" if SMOKE else "80"))

DESIGNS = {
    "B_stub": ("phase1c_B_stub", "phase1c_stub_i150_B_final.npz"),
    "C_low": ("phase1c_C_low", "phase1c_low_i150_C_final.npz"),
}


def design_box_from_region(grid, ix, iy, iz, trace_y_hi):
    """A ``phase2_fixture.DesignBox`` over the phase-1 design region.

    ``phase2_fixture.design_box()`` is anchored on the two-sided fixture's own
    trace coordinates and cannot be used on this one-sided fixture, so the
    ``hi`` (+y) side is built from the very ``ix``/``iy`` index arrays the
    optimization script used — the mask files index exactly this block. The
    ``lo`` side is the -y clearance strip: real space, never a design region
    here, asserted empty by every caller.
    """
    pad_x, pad_y, pad_z = grid.axis_pads
    ny = grid.shape[1]
    dx = notch.DX
    yc = (np.arange(ny) - pad_y + 0.5) * dx
    ix_lo, ix_hi = int(ix[0]), int(ix[-1]) + 1
    iy_lo, iy_hi = int(iy[0]), int(iy[-1]) + 1
    hi = fx.BoxSide("hi", ix_lo, ix_hi, iy_lo, iy_hi, iz, grid.axis_pads, dx)
    # -y clearance strip: substrate edge up to the first rasterized trace cell.
    j_tr_lo = int(np.where(yc >= trace_y_hi - notch.W_TRACE)[0][0])
    lo = fx.BoxSide("lo", ix_lo, ix_hi, pad_y, j_tr_lo, iz, grid.axis_pads, dx)
    if lo.ny <= 0:
        raise RuntimeError("no -y clearance cells below the trace")
    return fx.DesignBox(lo=lo, hi=hi, iz=iz, grid_shape=tuple(grid.shape), dx=dx)


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

    box = design_box_from_region(grid, ix, iy, iz, trace_y_hi)
    mask = box.empty_mask()

    stub_len = None
    stub_cells = None
    if d == "oracle":
        stub_len = (args.stub_mm * 1e-3 if args.stub_mm is not None
                    else float(notch.L_TARGET_AN))
        # SAME pathway as every mask arm: floor(L/dx) cells, no continuous Box.
        mask = fx.mask_from_stubs(
            [("hi", x_mid, notch.W_TRACE, stub_len)], box)
        stub_cells = max(1, int(np.floor(stub_len / notch.DX + 1e-9)))
        realized = int(np.asarray(mask["hi"]).any(axis=0).sum())
        print(f"[xval1:oracle] stub length {stub_len*1e3:.3f} mm "
              f"(analytic {float(notch.L_TARGET_AN)*1e3:.3f} mm) -> "
              f"{stub_cells} cells = floor(L/dx); realized {realized} rows, "
              f"{int(np.asarray(mask['hi']).any(axis=1).sum())} columns")
    elif d in DESIGNS:
        sub, fname = DESIGNS[d]
        hard = np.load(HERE / "out_vessl" / sub / fname)["hard"]
        if tuple(hard.shape) != tuple(box.hi.shape):
            raise ValueError(f"mask shape {hard.shape} != design region "
                             f"{box.hi.shape}")
        mask["hi"] = (np.asarray(hard) >= 0.5).astype(np.uint8)
        print(f"[xval1:{d}] mask fill={np.asarray(hard).mean():.3f}")

    if int(np.asarray(mask["lo"]).sum()) != 0:
        raise RuntimeError("design metal leaked onto the -y clearance strip")
    n_boxes = fx.add_pec_boxes(sim, fx.boxes_from_mask(mask, box))
    print(f"[xval1:{d}] design region {box.hi.shape} -> {n_boxes} PEC boxes")

    # compute_msl_s_matrix drives both ports; build_sim disabled port 1
    object.__setattr__(sim._msl_ports[1], "excite", True)
    pre = [(getattr(m, "code", "uncoded"), getattr(m, "severity", "warning"),
            str(m)) for m in sim.preflight()]
    for code, sev, msg in pre:
        print(f"  preflight [{sev}/{code}]: {msg[:110]}")

    print(f"[xval1:{d}] grid={grid.shape} freqs={len(FREQS)} "
          f"periods={NUM_PERIODS} boxes={n_boxes} smoke={SMOKE}")
    t0 = time.time()
    res = sim.compute_msl_s_matrix(freqs=freqs_j, num_periods=NUM_PERIODS)
    wall = time.time() - t0

    # Carry the solver's own reliability verdict into the record: a high-Q
    # notch read from a truncated record is a DFT artifact, and rfx says so.
    f = np.asarray(res.freqs)
    n_f = int(f.size)
    settling = np.asarray(getattr(res, "settling_db", []), dtype=float).ravel()
    reliable = np.asarray(getattr(res, "reliable", []), dtype=bool).ravel()
    _pc = getattr(res, "passivity_correction", None)
    # rfx sets passivity_correction = None when NO bin needed projecting, i.e.
    # exactly when the extraction was PERFECTLY PASSIVE. The old idiom
    # np.asarray(None, dtype=float) is array(nan), size 1, so passivity_worst
    # was recorded as nan for the cleanest possible run — and
    # score_dualband.check_validity evaluates nan <= PASSIVITY_MAX as False,
    # declaring a perfectly passive run NOT QUOTABLE. None == nothing
    # projected == the correction is identically zero.
    pcorr = (np.asarray(_pc, dtype=float).ravel() if _pc is not None
             else np.zeros(n_f, dtype=float))
    passivity_projected = _pc is not None
    p_worst = float(np.max(pcorr)) if pcorr.size else 0.0
    settled = bool(settling.size and float(np.max(settling)) <= -40.0)
    print(f"[xval1:{d}] settling worst={float(np.max(settling)) if settling.size else float('nan'):.1f} dB "
          f"(need <= -40) -> {'SETTLED' if settled else 'NOT SETTLED — do not quote'}; "
          f"reliable bins {int(reliable.sum())}/{reliable.size}; "
          f"worst passivity corr {p_worst:.3f} (projected={passivity_projected})")

    s21 = np.asarray(res.S[1, 0, :])
    s11 = np.asarray(res.S[0, 0, :])
    db21 = 20 * np.log10(np.abs(s21) + 1e-30)
    it = int(np.argmin(np.abs(f - notch.F_TARGET)))
    out = dict(design=d, stub_mm=(None if stub_len is None else stub_len*1e3),
               stub_cells=stub_cells,
               stub_pathway="mask_from_stubs -> boxes_from_mask (floor(L/dx))",
               n_boxes=n_boxes, wall_s=round(wall, 1),
               num_periods=NUM_PERIODS, freqs_GHz=[float(x) / 1e9 for x in f],
               s21_db=[float(x) for x in db21],
               s21_db_at_target=float(db21[it]),
               s11_db=[float(20 * np.log10(abs(x) + 1e-30)) for x in s11],
               f_min_GHz=float(f[int(np.argmin(db21))] / 1e9),
               depth_min_db=float(np.min(db21)),
               settling_worst_db=(float(np.max(settling)) if settling.size else None),
               settled=settled,
               reliable_bins=[int(reliable.sum()), int(reliable.size)],
               passivity_worst=p_worst,
               passivity_projected=bool(passivity_projected),
               preflight=[dict(code=c, severity=s, message=m) for c, s, m in pre])
    tag = d if stub_len is None else f"{d}_{stub_len*1e3:.2f}mm"
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[xval1:{d}] |S21|(6 GHz)={db21[it]:+.2f} dB · "
          f"min {out['depth_min_db']:+.2f} dB @ {out['f_min_GHz']:.2f} GHz · "
          f"{wall:.0f}s -> {OUT}/{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

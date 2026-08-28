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
  * the pre-registered score M on the SPEC bands (score_dualband, frozen grid)
  * whether the two SPEC bands are separately resolved (gap recovery)

Three things this script deliberately does NOT do any more (2026-08-27 review):

  1. **It does not place the stubs as continuous ``Box`` geometry.** rfx
     rasterizes a ``Box`` on NODES, half-open ``[lo, hi)`` (see
     ``rfx.geometry.csg.Box``), so a stub drawn from the trace edge realizes
     ``floor(L/dx)`` OR ``floor(L/dx)+1`` rows depending on ``frac(L/dx)``,
     while every mask-based arm realizes ``floor(L/dx)`` exactly. MEASURED on
     this fixture with rfx's own rasterizer, the continuous path leaves the
     ``floor(L/dx)`` lattice for **7 of 10** fractions (the stub root sits at
     y/dx = 16.7244, so the extra row appears from ``frac(L/dx) >= 0.276``);
     the two lambda/4 stubs used here realized 67 and 61 rows through the Box
     path against 66 and 60 through the mask path. One row is 127 um, 1.5 % of
     a lambda/4 stub at 5.25 GHz -- about 80 MHz, comparable to the whole
     100 MHz upper WLAN band. A classical baseline swept through one path
     against optimized arms evaluated through the other would sit on a
     different length lattice: the asymmetry class that forced the Phase-1
     retraction. Stubs now enter through ``phase2_fixture.mask_from_stubs``
     -> ``boxes_from_mask``, the same and only pathway every other arm uses,
     so the classical arm here is ONE ROW SHORTER than the numbers in
     NOTE_stage0_window.md, which were taken through the Box path.

  2. **It does not score on an ad-hoc grid around the observed notches.** The
     old grid was ``linspace(4.0, 8.0 GHz)``, which never evaluates the
     3.1-4.0 / 8.0-8.6 GHz passband at all, and the old depth extraction
     searched 4.6-5.5 / 5.5-6.4 GHz windows placed around wherever the notches
     happened to land -- so a notch tuned to 5.55 GHz was attributed to the
     lower band. Scoring is now ``score_dualband.score()`` on the frozen
     scoring grid and the frozen SPEC bands.

  3. **It does not discard ``sim.preflight()``'s return value**, and it never
     lets a ``None`` passivity correction reach the validity gate (rfx returns
     None when NOTHING needed projecting -- ``np.asarray(None, dtype=float)``
     is nan and ``nan <= 0.05`` is False, so the cleanest possible run was
     being declared NOT QUOTABLE).

Caveat carried in the JSON: insertion loss here is ``-|S21|_dB``, i.e. the
empty line is ASSUMED to be 0.00 dB. The two-sided fixture's empty-line
calibration has not been re-measured (LY changed), so ``empty_cal_max_db`` is
left None and the frozen empty-calibration gate is SKIPPED rather than faked.

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
sys.path.insert(0, str(HERE))

import msl_stub_notch_tuning as notch  # noqa: E402
import phase2_fixture as fx  # noqa: E402
import score_dualband as sd  # noqa: E402

OUT = Path(os.environ.get("OUTPUT_DIR", HERE / "out_vessl" / "stage0"))
OUT.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8
F_LO, F_HI = 5.25e9, 5.775e9          # WLAN band centres (5.15-5.35, 5.725-5.825)
BOX_X, BOX_Y = 12.0e-3, 9.0e-3        # bounded design box (see PLAN_phase2_dualband.md)
SMOKE = os.environ.get("SMOKE", "0") == "1"

GAP_MHZ = (sd.BAND_L_MHZ[1] + sd.GUARD_MHZ, sd.BAND_U_MHZ[0] - sd.GUARD_MHZ)


def quarter_wave(f):
    return C0 / (f * np.sqrt(notch.EPS_EFF)) / 4.0


def one_sided_design_box(grid, y_root, nx_cells, ny_cells, x_centre):
    """A ``phase2_fixture.DesignBox`` on THIS (one-sided) fixture's grid.

    ``phase2_fixture.design_box()`` is anchored on the two-sided fixture's own
    trace coordinates, so it cannot be used here; this builds the same object
    from the rasterized trace of whatever grid it is handed. The design region
    is the ``hi`` (+y) side, which is the only side this fixture has room for
    (the -y side is 12 cells of clearance). The ``lo`` side is that clearance
    strip: it is real space, it is never a design region here, and callers
    assert its mask stays empty.
    """
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    dx = notch.DX
    yc = (np.arange(ny) - pad_y + 0.5) * dx
    zc = (np.arange(nz) - pad_z + 0.5) * dx
    iz = int(np.argmin(np.abs(zc - (notch.H_SUB + 0.5 * dx))))

    p0 = int(round(x_centre / dx - nx_cells / 2.0))
    ix_lo, ix_hi = p0 + pad_x, p0 + nx_cells + pad_x
    if ix_lo < 0 or ix_hi > nx:
        raise RuntimeError(f"design box x range [{ix_lo},{ix_hi}) outside nx={nx}")

    j_above = np.where(yc >= y_root)[0]
    if j_above.size == 0:
        raise RuntimeError("no cells above the trace -- geometry is wrong")
    j0 = int(j_above[0])
    if j0 + ny_cells > ny:
        raise RuntimeError(f"design box needs cells up to {j0+ny_cells}, ny={ny}")

    hi = fx.BoxSide("hi", ix_lo, ix_hi, j0, j0 + ny_cells, iz,
                    grid.axis_pads, dx)
    # -y clearance strip: from the substrate edge up to the first trace cell.
    j_tr_lo = int(np.where(yc >= y_root - notch.W_TRACE)[0][0])
    lo = fx.BoxSide("lo", ix_lo, ix_hi, pad_y, j_tr_lo, iz, grid.axis_pads, dx)
    if lo.ny <= 0:
        raise RuntimeError("no -y clearance cells below the trace")
    return fx.DesignBox(lo=lo, hi=hi, iz=iz, grid_shape=tuple(grid.shape), dx=dx)


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
    # Frozen grid. The descent grid is used ONLY under SMOKE, where nothing is
    # quotable anyway; score_dualband says so itself.
    grid_mhz = sd.descent_grid_mhz() if SMOKE else sd.scoring_grid_mhz()
    freqs = grid_mhz.astype(float) * 1e6
    freqs_j = jnp.asarray(freqs, dtype=jnp.float32)
    sim, _, trace_y_hi, _, _ = notch.build_sim(freqs_j)
    grid = sim._build_grid()

    l_lo, l_hi = quarter_wave(F_LO), quarter_wave(F_HI)
    x_mid = notch.LX / 2.0
    sep = args.sep_mm * 1e-3

    # ---- ONE pathway: stubs -> per-cell mask -> run-length boxes -> PEC -----
    box = one_sided_design_box(
        grid, trace_y_hi,
        nx_cells=int(np.floor(BOX_X / notch.DX + 1e-9)),
        ny_cells=int(np.floor(BOX_Y / notch.DX + 1e-9)),
        x_centre=x_mid)
    stubs = [("hi", x_mid - sep / 2, notch.W_TRACE, l_lo),
             ("hi", x_mid + sep / 2, notch.W_TRACE, l_hi)]
    mask = fx.mask_from_stubs(stubs, box)
    if int(np.asarray(mask["lo"]).sum()) != 0:
        raise RuntimeError("stubs leaked onto the -y clearance strip")
    n_boxes = fx.add_pec_boxes(sim, fx.boxes_from_mask(mask, box))

    cells_lo = max(1, int(np.floor(l_lo / notch.DX + 1e-9)))
    cells_hi = max(1, int(np.floor(l_hi / notch.DX + 1e-9)))
    realized_cells = int(np.asarray(mask["hi"]).any(axis=0).sum())

    in_box = (sep / 2 + notch.W_TRACE / 2.0) * 2 <= BOX_X and max(l_lo, l_hi) <= BOX_Y
    print(f"[stage0] stubs {l_lo*1e3:.3f} / {l_hi*1e3:.3f} mm -> "
          f"{cells_lo} / {cells_hi} cells (floor(L/dx), mask pathway), "
          f"separation {args.sep_mm:.1f} mm -> fits the "
          f"{BOX_X*1e3:.0f}x{BOX_Y*1e3:.0f} mm box: {in_box}")
    print(f"[stage0] design box {box.hi.shape} cells on the +y side -> "
          f"{n_boxes} PEC boxes, {realized_cells} occupied rows")
    print(f"[stage0] 3*lambda_g/4 anti-coupling spacing would need "
          f"{3*quarter_wave(5.5e9)*1e3:.1f} mm — the box forbids it")

    # preflight is EVIDENCE, not a side effect: keep every message.
    pre = [(getattr(m, "code", "uncoded"), getattr(m, "severity", "warning"),
            str(m)) for m in sim.preflight()]
    for code, sev, msg in pre:
        print(f"  preflight [{sev}/{code}]: {msg[:110]}")

    object.__setattr__(sim._msl_ports[1], "excite", True)

    t0 = time.time()
    res = sim.compute_msl_s_matrix(freqs=freqs_j, num_periods=float(args.periods))
    wall = time.time() - t0

    f = np.asarray(res.freqs)
    n_f = int(f.size)
    settling = np.asarray(getattr(res, "settling_db", None)
                          if getattr(res, "settling_db", None) is not None
                          else [], dtype=float)
    _rel = getattr(res, "reliable", None)
    reliable = np.asarray(_rel, dtype=bool) if _rel is not None else None
    _pc = getattr(res, "passivity_correction", None)
    # rfx returns None when NO bin needed projecting, i.e. when the extraction
    # was PERFECTLY PASSIVE. np.asarray(None, dtype=float) is nan and
    # nan <= PASSIVITY_MAX is False, which would fail the cleanest run there
    # is. None == nothing projected == correction is 0.
    pcorr = (np.asarray(_pc, dtype=float).ravel() if _pc is not None
             else np.zeros(n_f, dtype=float))
    projected = _pc is not None
    worst = float(np.max(settling)) if settling.size else float("nan")
    p_worst = float(np.max(pcorr)) if pcorr.size else 0.0

    s21 = np.asarray(res.S[1, 0, :])
    s11 = np.asarray(res.S[0, 0, :])
    db21 = 20 * np.log10(np.abs(s21) + 1e-30)
    db11 = 20 * np.log10(np.abs(s11) + 1e-30)

    f_mhz = np.round(f / 1e6).astype(int)
    if not np.array_equal(f_mhz, grid_mhz.astype(int)):
        raise RuntimeError("solver returned bins off the frozen grid")

    # IL vs the empty line. NOT calibrated on this fixture (gap G-A): the
    # empty line is assumed 0.00 dB, so empty_cal_max_db stays None and the
    # frozen empty-calibration gate is skipped rather than faked.
    il = -db21
    il_clipped = np.minimum(il, sd.SCORE.r_cap_db)
    validity = sd.check_validity(settling, pcorr, reliable, f_mhz, il_clipped,
                                 thr=sd.SCORE, empty_cal_max_db=None)
    r = sd.score(f_mhz, il, s11_db=db11, s21_db_abs=db21, thr=sd.SCORE,
                 validity=validity)
    rr = sd.score(f_mhz, il, s11_db=db11, s21_db_abs=db21, thr=sd.RELAXED)

    # Window-adequacy witness, anchored on the SPEC bands (never on observed
    # notch positions): how far transmission recovers in the inter-band gap
    # relative to the worse of the two stopbands. Unclipped, so it is not
    # flattened by the 25 dB cap. Diagnostic only -- not part of M.
    mL = (f_mhz >= sd.BAND_L_MHZ[0]) & (f_mhz <= sd.BAND_L_MHZ[1])
    mU = (f_mhz >= sd.BAND_U_MHZ[0]) & (f_mhz <= sd.BAND_U_MHZ[1])
    mG = (f_mhz >= GAP_MHZ[0]) & (f_mhz <= GAP_MHZ[1])
    gap_recovery = (float(min(il[mL].min(), il[mU].min()) - il[mG].max())
                    if mG.any() else 0.0)

    dt = float(grid.dt)
    n_steps = int(round(args.periods * (1.0 / notch.F_MAX) / dt))

    out = dict(periods=args.periods, n_steps=n_steps,
               record_ns=n_steps * dt * 1e9, dft_res_GHz=1.0 / (n_steps * dt) / 1e9,
               settling_worst_db=(worst if settling.size else None),
               settled=bool(settling.size and worst <= sd.SETTLING_MAX_DB),
               reliable_bins=[int(reliable.sum()), int(reliable.size)]
               if reliable is not None else [0, 0],
               passivity_worst=p_worst, passivity_projected=bool(projected),
               stub_lo_mm=l_lo * 1e3, stub_hi_mm=l_hi * 1e3, sep_mm=args.sep_mm,
               stub_cells=[cells_lo, cells_hi],
               stub_pathway="mask_from_stubs -> boxes_from_mask (floor(L/dx))",
               n_boxes=n_boxes,
               grid_name="descent (SMOKE, NOT QUOTABLE)" if SMOKE else "scoring",
               n_freqs=n_f,
               il_reference=("absolute |S21| (empty line assumed 0.00 dB; the "
                             "two-sided empty-line calibration is not measured "
                             "yet -- gap G-A)"),
               M=r.M, M_relaxed=rr.M, score=r.as_dict(),
               validity_ok=bool(validity.ok),
               gap_recovery_db=gap_recovery,
               resolved=bool(gap_recovery >= 3.0),
               preflight=[dict(code=c, severity=s, message=m) for c, s, m in pre],
               wall_s=round(wall, 1),
               freqs_MHz=[int(x) for x in f_mhz],
               freqs_GHz=[float(x) / 1e9 for x in f],
               s21_db=[float(x) for x in db21],
               s11_db=[float(x) for x in db11])
    (OUT / f"window_{args.periods:.0f}.json").write_text(json.dumps(out, indent=2))
    print(f"[stage0] periods={args.periods:.0f} n_steps={n_steps} "
          f"T={out['record_ns']:.2f}ns res={out['dft_res_GHz']:.3f}GHz")
    print(f"[stage0] settling worst={worst:.1f} dB -> "
          f"{'SETTLED' if out['settled'] else 'NOT SETTLED'}; reliable "
          f"{out['reliable_bins'][0]}/{out['reliable_bins'][1]}; passivity "
          f"worst={p_worst:.3f} (projected={projected}) -> validity "
          f"{'OK' if validity.ok else 'NOT QUOTABLE'}")
    print(f"[stage0] M={r.M:.2f} (relaxed {rr.M:.2f}) "
          f"[S_L={r.S_L:.2f} S_U={r.S_U:.2f} S_G={r.S_G:.2f} S_P={r.S_P:.2f}] "
          f"R_L={r.R_L_raw:+.1f} R_U={r.R_U_raw:+.1f} dB")
    print(f"[stage0] spec-band notches {r.f_notch_L_MHz:.0f} / "
          f"{r.f_notch_U_MHz:.0f} MHz (targets 5250 / 5775)")
    verdict = "RESOLVED" if out["resolved"] else "MERGED (window too short, or the design is one wide notch)"
    print(f"[stage0] gap recovery {gap_recovery:.1f} dB -> {verdict} ({wall:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Arm D — the calibrated classical baseline on the two-sided Phase-2 fixture.

Everything the review demanded before a number is quotable happens here by
construction, because every solve goes through ``phase2_calibrate.score_design``:
the empty-line reference is solved once and cached, insertion loss is computed
against it (nothing assumes the empty line is 0.00 dB), ``empty_cal_max_db`` is
populated so the frozen gate can actually fire, and the design reaches the solver
as a per-cell mask through the single shared geometry pathway.

Modes
-----
``--mode window``   Re-measure the window table on the TWO-SIDED fixture. The
                    Stage-0 table was taken on the one-sided fixture through the
                    continuous-Box path, which realizes one row more than the
                    mask path, so it does not transfer. Solves the empty line and
                    the textbook two-stub design at each requested window.
``--mode sweep``    The calibrated sweep: stub lengths, widths and separation on
                    the cell lattice, both sides of the trace in play.

Every result carries the validity block (settling, reliable bins, passivity,
empty-line calibration) and the realized CELL INDICES, not the requested metres —
a mask cell realizes metal half a cell below its centre, so logging the request
would carry half-cell jitter into the position axis.

Run:
  python research/metal_to/armD_classical.py --mode window --periods 45 90
  python research/metal_to/armD_classical.py --mode sweep --periods 45
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "validation" / "tmtt_paper"))

import phase2_calibrate as cal  # noqa: E402
import phase2_fixture as fx  # noqa: E402
import score_dualband as sd  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = Path(os.environ.get(
    "OUTPUT_DIR",
    HERE / "out_smoke" / "armD" if SMOKE else HERE / "out_vessl" / "armD"))
OUT.mkdir(parents=True, exist_ok=True)
# The empty-line cache follows the same gating: a SMOKE reference is an
# unsettled ring-down artifact and must never sit where a real run looks for it.
CACHE = OUT.parent / "empty_ref"
CACHE.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8
F_LO, F_HI = 5.25e9, 5.775e9


def quarter_wave(f, eps_eff):
    return C0 / (f * np.sqrt(eps_eff)) / 4.0


def _grid_hz():
    g = sd.descent_grid_mhz() if SMOKE else sd.scoring_grid_mhz()
    return np.asarray(g, dtype=float) * 1e6


def _realized(mask):
    """What the lattice actually made, not what was requested.

    A requested width can straddle cell boundaries and realize one column more
    (w=1 cell -> 2 columns here), so the f(L, W) fit must be keyed on the
    realized geometry or the calibration inherits a half-cell bias.
    """
    out = {}
    for side in ("lo", "hi"):
        m = np.asarray(mask[side])
        if not m.any():
            out[f"{side}_cells"] = 0
            continue
        cols = np.where(m.any(axis=1))[0]
        rows = np.where(m.any(axis=0))[0]
        out[f"{side}_cells"] = int(m.sum())
        out[f"{side}_cols"] = int(len(cols))
        out[f"{side}_rows"] = int(len(rows))
        out[f"{side}_col_range"] = [int(cols.min()), int(cols.max())]
        out[f"{side}_row_range"] = [int(rows.min()), int(rows.max())]
    return out


def x_c_default(box):
    pad_x = box.hi.pads[0]
    return (0.5 * (box.hi.ix_lo + box.hi.ix_hi) - pad_x) * box.dx


def _stub_pair(box, sep_m, l_lo, l_hi, w_lo, w_hi, two_sided: bool):
    """Textbook pair: one stub per band. ``two_sided`` puts them on opposite
    sides of the trace, which the box now allows and the Stage-0 fixture did not."""
    pad_x = box.hi.pads[0]
    x_c = (0.5 * (box.hi.ix_lo + box.hi.ix_hi) - pad_x) * box.dx
    stubs = [
        ("lo" if two_sided else "hi", x_c - sep_m / 2.0, w_lo, l_lo),
        ("hi", x_c + sep_m / 2.0, w_hi, l_hi),
    ]
    return fx.mask_from_stubs(stubs, box)


def _record(tag, scored, extra):
    d = dict(tag=tag, **extra)
    d.update(scored.to_json() if hasattr(scored, "to_json") else
             json.loads(json.dumps(scored, default=lambda o: getattr(o, "__dict__", str(o)))))
    (OUT / f"{tag}.json").write_text(json.dumps(d, indent=2, default=str))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("window", "d1", "d3"), required=True)
    ap.add_argument("--periods", type=float, nargs="+", default=[45.0])
    ap.add_argument("--sep_mm", type=float, default=8.0)
    ap.add_argument("--two_sided", action="store_true",
                    help="place the two stubs on opposite sides of the trace "
                         "(the PI relaxation); default keeps both on +y so the "
                         "number is comparable with the Stage-0 table")
    args = ap.parse_args()

    freqs = _grid_hz()
    fixture = fx.build_sim(freqs)
    grid = fixture.sim._build_grid()
    box = fx.design_box(grid)
    eps_eff = fx.EPS_EFF if hasattr(fx, "EPS_EFF") else 2.8694

    l_lo, l_hi = quarter_wave(F_LO, eps_eff), quarter_wave(F_HI, eps_eff)
    print(f"[armD] grid={grid.shape} box lo={box.lo.nx}x{box.lo.ny} "
          f"hi={box.hi.nx}x{box.hi.ny} ({box.lo.nx*box.lo.ny*2} cells) "
          f"lambda/4 = {l_lo*1e3:.3f} / {l_hi*1e3:.3f} mm  smoke={SMOKE}")

    w = fx.W_TRACE if hasattr(fx, "W_TRACE") else 600e-6
    results = []

    if args.mode == "window":
        for periods in args.periods:
            t0 = time.time()
            mask = _stub_pair(box, args.sep_mm * 1e-3, l_lo, l_hi, w, w,
                              args.two_sided)
            sc = cal.score_design(mask, freqs_hz=freqs, num_periods=periods,
                                  label=f"armD_window_{periods:.0f}",
                                  cache_dir=CACHE, verbose=True)
            r = _record(f"window_{periods:.0f}{'_2s' if args.two_sided else ''}",
                        sc, dict(mode="window", periods=periods,
                                 sep_mm=args.sep_mm, two_sided=args.two_sided,
                                 stub_lo_mm=l_lo * 1e3, stub_hi_mm=l_hi * 1e3,
                                 wall_s=round(time.time() - t0, 1)))
            results.append(r)
            print(f"[armD] periods={periods:.0f} -> "
                  f"{time.time()-t0:.0f}s  (record written)")
    elif args.mode == "d1":
        # Calibrate the single-stub law f(L, W) on THIS lattice. The analytic
        # quarter wave is ~5 % off (open-end + T-junction fringing + the stub's
        # own eps_eff), and the whole point of arm D is that the classical
        # design gets to remove that offset before being judged.
        periods = args.periods[0]
        widths = [1, 2, 3, 5, 8, 12]          # cells; Z_s ~ 103 -> 27 ohm
        lengths = [40, 48, 56, 63, 70]        # cells; 5.08 - 8.89 mm
        dx = box.dx
        for wc in widths:
            for lc in lengths:
                t0 = time.time()
                tag = f"d1_w{wc}_l{lc}"
                if (OUT / f"{tag}.json").exists():
                    print(f"[armD] {tag} cached, skipping"); continue
                mask = fx.mask_from_stubs(
                    [("hi", x_c_default(box), wc * dx, lc * dx)], box)
                sc = cal.score_design(mask, freqs_hz=freqs, num_periods=periods,
                                      label=tag, cache_dir=CACHE, verbose=False)
                r = _record(tag, sc, dict(mode="d1", periods=periods,
                                          w_cells=wc, l_cells=lc,
                                          w_mm=wc * dx * 1e3, l_mm=lc * dx * 1e3,
                                          **_realized(mask),
                                          wall_s=round(time.time() - t0, 1)))
                res = r.get("result", {})
                print(f"[armD] {tag}: notch f_L={res.get('f_notch_L_MHz')} "
                      f"f_U={res.get('f_notch_U_MHz')} R_L={res.get('R_L')} "
                      f"({time.time()-t0:.0f}s)")
                results.append(r)
    else:  # d3 — the calibrated two-stub sweep
        # Lengths come from the D-1 law f = K/(L+dL), fitted per width to 0.1%:
        #   5.25  GHz -> 63.3-63.8 cells      5.775 GHz -> 57.5-58.1 cells
        # Both land between cells, so the lattice bounds notch placement to
        # +-42 MHz (one cell = 127 um = ~83 MHz here). Width is the vernier:
        # at fixed length it trims ~22 MHz across the realizable width range.
        periods = args.periods[0]
        dx = box.dx
        L_LO = [62, 63, 64]
        L_HI = [57, 58, 59]
        COLS = [2, 4]          # narrow = high-Z = sharper skirts = better gap
        SEPS = [24, 40, 63]    # cells; 63 = lambda_g/4 at 5.5 GHz
        for two_sided in (False, True):
            for sep_c in SEPS:
                for wc in COLS:
                    for llo in L_LO:
                        for lhi in L_HI:
                            tag = (f"d3_{'2s' if two_sided else '1s'}_sep{sep_c}"
                                   f"_w{wc}_llo{llo}_lhi{lhi}")
                            if (OUT / f"{tag}.json").exists():
                                continue
                            t0 = time.time()
                            mask = _stub_pair(box, sep_c * dx, llo * dx,
                                              lhi * dx, wc * dx, wc * dx,
                                              two_sided)
                            sc = cal.score_design(
                                mask, freqs_hz=freqs, num_periods=periods,
                                label=tag, cache_dir=CACHE, verbose=False)
                            r = _record(tag, sc, dict(
                                mode="d3", periods=periods, two_sided=two_sided,
                                sep_cells=sep_c, w_cells=wc,
                                l_lo_cells=llo, l_hi_cells=lhi,
                                **_realized(mask),
                                wall_s=round(time.time() - t0, 1)))
                            res = r.get("result", {})
                            print(f"[armD] {tag}: M={res.get('M'):.2f} "
                                  f"S=({res.get('S_L'):.1f},{res.get('S_U'):.1f},"
                                  f"{res.get('S_G'):.1f},{res.get('S_P'):.1f}) "
                                  f"({time.time()-t0:.0f}s)", flush=True)
                            results.append(r)

    print(f"[armD] wrote {len(results)} record(s) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

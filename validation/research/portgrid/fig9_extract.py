#!/usr/bin/env python3
"""Window-provenance instrument: vector-data extraction of arXiv:1606.08761 Fig. 9.

Extracts the reflection curves of Fig. 9 (both panels) directly from the
arXiv PDF's vector path data (no reading-by-eye), calibrates axes from the
tick-mark path coordinates, and prints/stores the class anchors used by the
F-M1b retry pre-declaration:

  bottom panel ("Sub-grid interface only", r in {2,4,6}):
    worst-curve max |S11| over [2,20] GHz and [2,30] GHz  ->  +5 dB rule
  top panel ("With scatterer", subgridding r in {2,4,6} vs all-coarse and
  all-fine r=6):
    max LINEAR |S11_r - S11_allfine| over [2,30] GHz      ->  +5 dB rule

Dependencies: pdfminer.six (NOT in the repo venv on purpose -- this is a
one-shot provenance instrument, not part of the battery).  Run with any
Python that has pdfminer.six:

  python3 fig9_extract.py /path/to/1606.08761.pdf --out fig9_extraction.json

Axis calibration facts recovered from the PDF (page index 9):
  bottom frame x: 357.09792 .. 552.0151 pt  = 0 .. 30 GHz (7 ticks, 5 GHz)
  bottom y ticks: -100 dB @ 366.90918, -80 @ 389.26932, -60 @ 411.70463,
                  -40 @ 434.06459 pt;  frame top 445.2076 pt = -30.05 dB,
                  frame bottom 355.6913 pt = -110.0 dB
  (the -40 dB tick is merely the topmost LABEL; the frame extends to -30 dB
  and every curve is fully inside the frame -- Correction 2 confirmed.)
  top frame y: 457.40405 (-35 dB) .. 546.91978 (-5 dB), ticks every 5 dB.
Curve colors: r=2 green (0.100098,0.5,0.100098), r=4 blue (0,0,1),
r=6 red (1,0,0); top panel adds all-coarse black (0.0) and all-fine r=6
gray (0.5).  Color-to-ratio mapping cross-checked against the 2022-review
endpoint reading (-36.5 / -34.6 / -34.3 dB at 30 GHz for r=2/4/6).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def collect(page, kinds):
    out = []

    def walk(el):
        if isinstance(el, kinds):
            out.append(el)
        if hasattr(el, "_objs"):
            for o in el._objs:
                walk(o)

    for el in page:
        walk(el)
    return out


def main() -> int:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTCurve, LTLine

    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="local path to the arXiv:1606.08761 PDF")
    ap.add_argument("--page", type=int, default=9, help="0-based page index of Fig. 9")
    ap.add_argument("--out", default="")
    ap.add_argument("--abs-out", dest="abs_out", default="",
                    help="write ONLY the top-panel absolute anchors here "
                         "(Correction R3(b) / F-M1b-abs provenance)")
    args = ap.parse_args()

    pages = list(extract_pages(args.pdf))
    curves = collect(pages[args.page], (LTLine, LTCurve))

    # ---- calibration (vector tick coordinates, see module docstring) ----
    x0pt, x30pt = 357.09792, 552.0151
    yb = {-100: 366.90918, -80: 389.26932, -60: 411.704626208, -40: 434.06458552}
    ys = np.array(sorted(yb.values()))
    dbs = np.array(sorted(yb.keys()))
    slope, icept = np.polyfit(ys, dbs, 1)
    yt_lo, yt_hi = 457.40405448, 546.919780688  # top panel frame: -35 .. -5 dB

    def to_ghz(x):
        return (np.asarray(x) - x0pt) / (x30pt - x0pt) * 30.0

    def bot_db(y):
        return slope * np.asarray(y) + icept

    def top_db(y):
        return -35.0 + (np.asarray(y) - yt_lo) / (yt_hi - yt_lo) * 30.0

    colors = {
        (0.100098, 0.5, 0.100098): "r2",
        (0.0, 0.0, 1.0): "r4",
        (1.0, 0.0, 0.0): "r6",
        (0.0,): "all_coarse",
        (0.5,): "all_fine6",
    }

    def gather(y_lo, y_hi, wanted):
        data = {}
        for c in curves:
            col = c.stroking_color
            col = tuple(col) if isinstance(col, (list, tuple)) else (col,)
            name = colors.get(col)
            if (name in wanted and c.pts and len(c.pts) > 5
                    and c.x0 > 350 and y_lo < c.y0 and c.y1 < y_hi):
                data.setdefault(name, []).extend(c.pts)
        series = {}
        for name, pts in data.items():
            a = np.array(sorted(set(pts)))
            order = np.argsort(a[:, 0])
            series[name] = a[order]
        return series

    result = {"pdf": args.pdf, "page_index": args.page}

    # ---- bottom panel: interface-only ----
    bot = gather(348, 446, ("r2", "r4", "r6"))
    banner = {}
    for name, a in bot.items():
        ghz, db = to_ghz(a[:, 0]), bot_db(a[:, 1])
        row = {"n_points": int(len(ghz)), "db_at_30GHz": round(float(db[np.argmax(ghz)]), 2)}
        for lo, hi in ((2, 20), (2, 30)):
            m = (ghz >= lo) & (ghz <= hi)
            row[f"max_db_{lo}_{hi}GHz"] = round(float(np.max(db[m])), 2)
        banner[name] = row
    worst20 = max(v["max_db_2_20GHz"] for v in banner.values())
    worst30 = max(v["max_db_2_30GHz"] for v in banner.values())
    result["bottom_panel_interface_only"] = banner
    result["worst_curve_max_db"] = {"2_20GHz": worst20, "2_30GHz": worst30}
    result["retry_windows_db_worst_plus_5dB"] = {
        "2_20GHz": round(worst20 + 5.0, 2), "2_30GHz": round(worst30 + 5.0, 2)}

    # ---- top panel: with scatterer ----
    top = gather(448, 556, tuple(colors.values()))
    f = np.linspace(2.0, 29.8, 600)
    interp = {}
    for name, a in top.items():
        interp[name] = np.interp(f, to_ghz(a[:, 0]), top_db(a[:, 1]))
    lin = {k: 10 ** (v / 20.0) for k, v in interp.items()}
    mism = {}
    for k in ("r2", "r4", "r6", "all_coarse"):
        d = float(np.max(np.abs(lin[k] - lin["all_fine6"])))
        mism[k] = {"max_linear_diff_vs_allfine": round(d, 4),
                   "as_db": round(20 * np.log10(d), 2)}
    result["top_panel_mismatch_vs_allfine_2_29p8GHz"] = mism
    r6 = mism["r6"]["max_linear_diff_vs_allfine"]
    result["rod_arm_window_linear_r6_paper_plus_5dB"] = round(r6 * 10 ** (5.0 / 20.0), 4)

    # ---- top panel: ABSOLUTE anchors (Correction R3(b), F-M1b-abs) ----
    # The all-fine curve of the top panel is a pure-Yee run of the paper's own
    # Sec. V-C fixture: no subgridding anywhere in it.  Comparing OUR all-fine
    # run against it is a fixture/observable-fidelity check with no scheme
    # content, which is why it is the check that catches a probe-projection
    # error.  Anchors: max over the extracted band and the values at 10/25/29
    # GHz.  Recorded for every top-panel curve; all_fine6 is the one F-M1b-abs
    # judges against.
    absolute = {"band_GHz": [float(f[0]), float(f[-1])]}
    for name, db in interp.items():
        row = {"max_db": round(float(np.max(db)), 2),
               "max_linear": round(float(np.max(10 ** (db / 20.0))), 4),
               "f_at_max_GHz": round(float(f[int(np.argmax(db))]), 2)}
        for fq in (10.0, 25.0, 29.0):
            row[f"db_at_{int(fq)}GHz"] = round(float(np.interp(fq, f, db)), 2)
        absolute[name] = row
    result["top_panel_absolute_anchors"] = absolute

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    if args.abs_out:
        # separate file so the frozen window-provenance JSON stays byte-identical
        with open(args.abs_out, "w") as fh:
            json.dump({"pdf": args.pdf, "page_index": args.page,
                       "top_panel_absolute_anchors": result["top_panel_absolute_anchors"]},
                      fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

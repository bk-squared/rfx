"""3-way S11-trend comparison for the cv05 2.4 GHz FR4 probe-fed patch:
rfx (mesh sweep + port-extent sweep) vs openEMS (independent reference) vs
Balanis analytic resonance.

LOADS (degrades gracefully — compares whatever is present):
  (a) rfx mesh-sweep JSONs   scripts/research/calibration/crossval/out_gpu/*.json
  (b) rfx port-extent JSONs  scripts/research/calibration/crossval/out_gpu_port/ext*/*.json
  (c) openEMS reference      scripts/research/calibration/crossval/out_ref/openems_patch_s11.json
  (d) Balanis analytic f_res (from any JSON's analytic_resonance_hz)

PRODUCES:
  * overlay plot |S11| dB + phase vs f (one curve per solver/config)
        -> scripts/research/calibration/crossval/out_ref/compare_patch_s11.png
  * agreement table printed to stdout:
      - resonance-frequency % diff: rfx-Harminv vs openEMS-Harminv vs analytic
      - on the SHARED 1.5-3.5 GHz / 101-pt grid: max and mean |dS11| and
        |dphase| between each rfx config and openEMS (the S11-TREND metric),
        with the best-matching rfx config flagged.

Run (works with only openEMS+analytic present; fills in rfx when JSONs land):
  python scripts/research/calibration/crossval/compare_patch_s11.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GPU = os.path.join(SCRIPT_DIR, "out_gpu")
OUT_GPU_PORT = os.path.join(SCRIPT_DIR, "out_gpu_port")
OUT_REF = os.path.join(SCRIPT_DIR, "out_ref")
OPENEMS_JSON = os.path.join(OUT_REF, "openems_patch_s11.json")
BALANIS_FALLBACK_HZ = 2.424e9   # cv05 Balanis TL model


def _load(path):
    with open(path) as f:
        return json.load(f)


def _complex(pairs):
    return np.array([complex(a, b) for a, b in pairs])


def _s11_dB(s11):
    return 20 * np.log10(np.maximum(np.abs(s11), 1e-9))


def _load_rfx_configs():
    """Return a list of rfx config dicts, tagged by source (mesh vs port)."""
    configs = []
    # (a) mesh sweep
    for path in sorted(glob.glob(os.path.join(OUT_GPU, "*.json"))):
        d = _load(path)
        configs.append({
            "kind": "mesh",
            "label": f"rfx mesh dx={d['dx_fine_mm']:.2f}mm",
            "sort": (0, -d["dx_fine_mm"]),
            "freqs": np.array(d["freqs_hz"]),
            "s11": _complex(d["s11"]),
            "harminv_hz": d.get("harminv_hz"),
            "s11_dip_hz": d.get("s11_dip_hz"),
            "s11_dip_db": d.get("s11_dip_db"),
            "port_extent_cells": d.get("port_extent_cells"),
            "dx_fine_mm": d.get("dx_fine_mm"),
        })
    # (b) port-extent sweep (ext*/ subdirs)
    for extdir in sorted(glob.glob(os.path.join(OUT_GPU_PORT, "ext*"))):
        ext_tag = os.path.basename(extdir)
        for path in sorted(glob.glob(os.path.join(extdir, "*.json"))):
            d = _load(path)
            pec = d.get("port_extent_cells", "?")
            configs.append({
                "kind": "port",
                "label": f"rfx port {ext_tag} (pec={pec}, dx={d['dx_fine_mm']:.2f}mm)",
                "sort": (1, pec if isinstance(pec, (int, float)) else 0),
                "freqs": np.array(d["freqs_hz"]),
                "s11": _complex(d["s11"]),
                "harminv_hz": d.get("harminv_hz"),
                "s11_dip_hz": d.get("s11_dip_hz"),
                "s11_dip_db": d.get("s11_dip_db"),
                "port_extent_cells": pec,
                "dx_fine_mm": d.get("dx_fine_mm"),
            })
    configs.sort(key=lambda c: c["sort"])
    return configs


def _trend_metrics(rfx_freqs, rfx_s11, oe_freqs, oe_s11):
    """max/mean |dS11| and |dphase| (deg) on the shared grid. If the two
    grids differ, rfx is interpolated (re/im) onto the openEMS grid."""
    if len(rfx_freqs) == len(oe_freqs) and np.allclose(rfx_freqs, oe_freqs):
        r = rfx_s11
    else:
        r = (np.interp(oe_freqs, rfx_freqs, rfx_s11.real)
             + 1j * np.interp(oe_freqs, rfx_freqs, rfx_s11.imag))
    dmag = np.abs(r - oe_s11)
    # wrapped phase difference, robust to 2pi offsets
    dphase = np.abs(np.angle(r / oe_s11)) * 180 / np.pi
    return (float(np.max(dmag)), float(np.mean(dmag)),
            float(np.max(dphase)), float(np.mean(dphase)))


def main():
    have_oe = os.path.exists(OPENEMS_JSON)
    oe = _load(OPENEMS_JSON) if have_oe else None
    rfx_configs = _load_rfx_configs()

    # analytic reference — every producer stores the same Balanis f_res
    f_analytic = BALANIS_FALLBACK_HZ
    for src in ([OPENEMS_JSON] if have_oe else []) + \
            sorted(glob.glob(os.path.join(OUT_GPU, "*.json"))) + \
            sorted(glob.glob(os.path.join(OUT_GPU_PORT, "ext*", "*.json"))):
        val = _load(src).get("analytic_resonance_hz")
        if val:
            f_analytic = float(val)
            break

    print("=" * 74)
    print("3-way patch S11-trend comparison: rfx vs openEMS vs Balanis analytic")
    print("=" * 74)
    print(f"Balanis analytic resonance: {f_analytic/1e9:.4f} GHz")
    print(f"openEMS reference present:  {have_oe}")
    print(f"rfx configs found:          {len(rfx_configs)} "
          f"({sum(c['kind']=='mesh' for c in rfx_configs)} mesh, "
          f"{sum(c['kind']=='port' for c in rfx_configs)} port-extent)")
    print()

    # ---- resonance-frequency agreement ----
    print("-" * 74)
    print("RESONANCE FREQUENCY (Harminv) vs analytic")
    print("-" * 74)
    if oe is not None and oe.get("harminv_hz") is not None and not np.isnan(oe["harminv_hz"]):
        f_oe = float(oe["harminv_hz"])
        oe_vs_an = 100 * abs(f_oe - f_analytic) / f_analytic
        print(f"  openEMS Harminv:  {f_oe/1e9:.4f} GHz   "
              f"(vs analytic: {oe_vs_an:+.2f} %, Q={oe.get('harminv_Q', float('nan')):.1f})")
    else:
        f_oe = None
        print("  openEMS Harminv:  (not available)")
    for c in rfx_configs:
        fh = c.get("harminv_hz")
        if fh is None or (isinstance(fh, float) and np.isnan(fh)):
            print(f"  {c['label']:<40} Harminv: (n/a)")
            continue
        vs_an = 100 * abs(fh - f_analytic) / f_analytic
        vs_oe = (100 * abs(fh - f_oe) / f_oe) if f_oe else float("nan")
        print(f"  {c['label']:<40} {fh/1e9:.4f} GHz  "
              f"(vs analytic {vs_an:+.2f} %, vs openEMS {vs_oe:+.2f} %)")
    print()

    # ---- S11-trend agreement (the metric the professor wants) ----
    best = None
    if oe is not None and rfx_configs:
        oe_freqs = np.array(oe["freqs_hz"])
        oe_s11 = _complex(oe["s11"])
        print("-" * 74)
        print("S11-TREND AGREEMENT rfx-config vs openEMS (shared 1.5-3.5 GHz grid)")
        print("-" * 74)
        hdr = f"  {'config':<40} {'max|dS11|':>10} {'mean|dS11|':>11} {'max dph':>9} {'mean dph':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for c in rfx_configs:
            mx, mn, mxp, mnp = _trend_metrics(c["freqs"], c["s11"], oe_freqs, oe_s11)
            c["_trend"] = (mx, mn, mxp, mnp)
            print(f"  {c['label']:<40} {mx:>10.4f} {mn:>11.4f} "
                  f"{mxp:>8.1f}d {mnp:>8.1f}d")
            if best is None or mn < best["_trend"][1]:
                best = c
        print()
        print(f"  BEST-matching rfx config (min mean|dS11| vs openEMS): "
              f"{best['label']}")
        mx, mn, mxp, mnp = best["_trend"]
        print(f"    max|dS11| = {mx:.4f},  mean|dS11| = {mn:.4f},  "
              f"max|dphase| = {mxp:.1f} deg,  mean|dphase| = {mnp:.1f} deg")
        print()
    elif oe is not None:
        print("-" * 74)
        print("S11-TREND AGREEMENT: openEMS present but NO rfx JSONs yet.")
        print("  -> rfx-vs-openEMS |dS11| / |dphase| will fill in when the GPU")
        print("     mesh (out_gpu/) and port-extent (out_gpu_port/ext*/) JSONs land.")
        print("-" * 74)
        print()

    # ---- overlay plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = plt.get_cmap("viridis")
    n_rfx = max(1, len(rfx_configs))
    for i, c in enumerate(rfx_configs):
        col = cmap(i / n_rfx)
        f_g = c["freqs"] / 1e9
        ls = "-" if c["kind"] == "mesh" else "--"
        axes[0].plot(f_g, _s11_dB(c["s11"]), color=col, lw=1.4, ls=ls, label=c["label"])
        axes[1].plot(f_g, np.unwrap(np.angle(c["s11"])) * 180 / np.pi,
                     color=col, lw=1.4, ls=ls, label=c["label"])
    if oe is not None:
        f_g = np.array(oe["freqs_hz"]) / 1e9
        s11 = _complex(oe["s11"])
        axes[0].plot(f_g, _s11_dB(s11), "r-", lw=2.4, label="openEMS (ref)")
        axes[1].plot(f_g, np.unwrap(np.angle(s11)) * 180 / np.pi,
                     "r-", lw=2.4, label="openEMS (ref)")
    for ax in axes:
        ax.axvline(f_analytic / 1e9, color="k", ls=":", alpha=0.6,
                   label=f"Balanis {f_analytic/1e9:.3f} GHz")
        ax.set_xlabel("f (GHz)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("|S11| (dB)"); axes[0].set_title("Return loss |S11|")
    axes[0].set_ylim(-40, 5)
    axes[1].set_ylabel("phase(S11) (deg)"); axes[1].set_title("S11 phase")
    fig.suptitle("2.4 GHz FR4 patch — S11 trend: rfx vs openEMS vs analytic",
                 fontweight="bold")
    plt.tight_layout()
    png = os.path.join(OUT_REF, "compare_patch_s11.png")
    os.makedirs(OUT_REF, exist_ok=True)
    plt.savefig(png, dpi=140); plt.close()
    print(f"Saved overlay plot: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

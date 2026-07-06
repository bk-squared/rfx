"""X-band INSET-FED matched patch — the issue80 validated frame + inset feed.

WHY X-BAND
----------
The msl extraction is gate-validated exactly in the issue80 frame (RO4003C
eps_r=3.38, h=0.787 mm, dx=0.197 mm, 1.5-15 GHz, patch ~9.2 GHz): rerun this
session it gives sane, passive-marginal, physics-consistent S11 (in-band
|S11|~0.92-0.96 for the EDGE-fed patch = correctly unmatched; dip above band
= the match point; Harminv/openEMS/Balanis triple-agree on the resonance).
The same extractor in a 2.1 GHz / h=1.524 mm frame produces ACTIVE |S11|
(+4.7..+11.5 dB even on a matched thru; probe params and band/waveform swaps
do not fix it) — a separate main-repo finding, not fought here.

So the matched-patch design moves to X-band: same issue80 patch, but fed
through a Balanis INSET that transforms the high edge resistance down to
50 ohm -> a genuine deep |S11| null at the resonance, in the validated frame.

    R_in(d) = R_edge * cos^2(pi d / L);  R_edge ~ 230-470 ohm  ->  d ~ 3 mm.

GEOMETRY (issue80 verbatim except the feed junction)
----------------------------------------------------
substrate RO4003C eps_r=3.38 h=0.787 (4 cells @ dx=0.197), ground+substrate
full domain footprint, 50-ohm feed W_MSL=1.8 mm from x=0 (CPML-absorbed
backward wave) to the patch edge, patch W=10.129 (y) x L=8.595 (x, resonant),
CPML 8, GaussianPulse f0=8.5 bw=1.6, msl port at PORT_MARGIN=5 mm.

Inset: the feed strip penetrates depth d into the patch through two etched
notch slots of gap g (PEC built additively; slots = unmetallised strips).

Run (GPU):  python scripts/crossval/rfx_patch_inset_xband.py \
                --inset-list-mm 0,2.4,2.8,3.2,3.6 --output scripts/crossval/out_xband
(0 = edge-fed reference, reproduces issue80 behaviour.)
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- issue80 frame, verbatim -------------------------------------------------
EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3            # patch width (y)
L = 8.595e-3             # patch length (x, resonant)
W_MSL = 1.8e-3
L_MSL = 8.0e-3
PORT_MARGIN = 5.0e-3
DX = 0.197e-3            # overridable via --dx-um
DOM_X = 29.747e-3
DOM_Y = 18.130e-3
DOM_Z = 12.787e-3
Y_C = DOM_Y / 2.0
Z_GND = 4e-3             # ground PEC at [Z_GND, Z_GND+DX]
Z_SUB_LO = Z_GND + DX
Z_SUB_HI = Z_SUB_LO + H_SUB
Z_MET_LO = Z_SUB_HI
Z_MET_HI = Z_SUB_HI + DX
X_PATCH_LO = PORT_MARGIN + L_MSL
X_PATCH_HI = X_PATCH_LO + L

NOTCH_GAP = 0.9e-3       # etched slot width beside the feed strip (~4.6 cells)
TARGET_GHZ = 9.21        # Balanis analytic (issue80)


def build_sim(inset_m):
    from rfx import Box, Simulation
    from rfx.sources import GaussianPulse

    sim = Simulation(
        freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
        dx=DX, cpml_layers=8, boundary="cpml",
    )
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, Z_GND), (DOM_X, DOM_Y, Z_GND + DX)), material="pec")
    sim.add(Box((0, 0, Z_SUB_LO), (DOM_X, DOM_Y, Z_SUB_HI)),
            material="ro4003c")

    yf_lo, yf_hi = Y_C - W_MSL / 2, Y_C + W_MSL / 2
    yp_lo, yp_hi = Y_C - W / 2, Y_C + W / 2
    yn_lo, yn_hi = yf_lo - NOTCH_GAP, yf_hi + NOTCH_GAP
    if inset_m < 0:
        # THRU: bare 50-ohm line across the whole domain, no patch. The
        # line's true Z0 discriminator: if the line is really ~70 ohm,
        # |S11| vs the 50-ohm split plateaus at ~-15.5 dB; if ~50, low.
        sim.add(Box((0, yf_lo, Z_MET_LO), (DOM_X, yf_hi, Z_MET_HI)),
                material="pec")
        sim.add_msl_port(
            position=(PORT_MARGIN, Y_C, Z_SUB_LO),
            width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
            waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
        )
        return sim
    x_conn = X_PATCH_LO + inset_m

    # feed strip: domain edge -> inset connection plane (PEC additive union)
    sim.add(Box((0, yf_lo, Z_MET_LO), (x_conn, yf_hi, Z_MET_HI)),
            material="pec")
    if inset_m > 0:
        # patch body beyond the inset + two flanks beside the notch slots
        sim.add(Box((x_conn, yp_lo, Z_MET_LO), (X_PATCH_HI, yp_hi, Z_MET_HI)),
                material="pec")
        sim.add(Box((X_PATCH_LO, yp_lo, Z_MET_LO), (x_conn, yn_lo, Z_MET_HI)),
                material="pec")
        sim.add(Box((X_PATCH_LO, yn_hi, Z_MET_LO), (x_conn, yp_hi, Z_MET_HI)),
                material="pec")
    else:
        # edge-fed reference (issue80 verbatim)
        sim.add(Box((X_PATCH_LO, yp_lo, Z_MET_LO), (X_PATCH_HI, yp_hi, Z_MET_HI)),
                material="pec")

    sim.add_msl_port(
        position=(PORT_MARGIN, Y_C, Z_SUB_LO),
        width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
    )
    return sim


def run_one(inset_mm, args, out_dir):
    print("=" * 72)
    label = ("THRU (no patch)" if inset_mm < 0 else
             ("edge-fed reference" if inset_mm == 0 else f"gap={NOTCH_GAP*1e3:.2f} mm"))
    print(f"INSET d = {inset_mm:.2f} mm ({label})")
    sim = build_sim(inset_mm * 1e-3)
    print("  preflight:")
    sim.preflight(strict=False)

    t0 = time.time()
    res = sim.compute_msl_s_matrix(n_freqs=args.nfreq,
                                   num_periods=args.num_periods)
    t_run = time.time() - t0

    freqs = np.asarray(res.freqs, dtype=float)
    z0_fit = np.asarray(res.Z0, dtype=complex) if getattr(res, "Z0", None) is not None else None
    if z0_fit is not None:
        z0f = z0_fit.reshape(-1, z0_fit.shape[-1])[0]
        print(f"  fitted line Z0: median {np.median(z0f.real):.1f}"
              f"{np.median(z0f.imag):+.1f}j ohm  "
              f"(range {z0f.real.min():.1f}..{z0f.real.max():.1f})")
    s11 = np.asarray(res.S)[0, 0, :]
    mag = np.abs(s11)
    db = 20 * np.log10(np.maximum(mag, 1e-9))
    zin = 50.0 * (1 + s11) / (1 - s11)

    i_dip = int(np.argmin(db))
    # value near the analytic resonance
    i_res = int(np.argmin(np.abs(freqs - TARGET_GHZ * 1e9)))
    print(f"  dip {db[i_dip]:.2f} dB @ {freqs[i_dip]/1e9:.3f} GHz  "
          f"Zin={zin[i_dip].real:.0f}{zin[i_dip].imag:+.0f}j | "
          f"@{TARGET_GHZ} GHz: {db[i_res]:.2f} dB  "
          f"Zin={zin[i_res].real:.0f}{zin[i_res].imag:+.0f}j | "
          f"max|S11|={mag.max():.3f}  ({t_run:.0f}s)")

    payload = {
        "case": "xband_inset_patch_issue80_frame",
        "inset_mm": float(inset_mm), "notch_gap_mm": NOTCH_GAP * 1e3,
        "runtime_s": t_run, "num_periods": args.num_periods,
        "s11_dip_db": float(db[i_dip]), "s11_dip_hz": float(freqs[i_dip]),
        "s11_at_target_db": float(db[i_res]),
        "zin_at_dip_ohm": [float(zin[i_dip].real), float(zin[i_dip].imag)],
        "zin_at_target_ohm": [float(zin[i_res].real), float(zin[i_res].imag)],
        "s11_max_abs": float(mag.max()),
        "z0_fitted": ([[float(v.real), float(v.imag)] for v in
                       z0_fit.reshape(-1, z0_fit.shape[-1])[0]]
                      if z0_fit is not None else None),
        "freqs_hz": freqs.tolist(),
        "s11": [[float(v.real), float(v.imag)] for v in s11],
        "z11": [[float(v.real), float(v.imag)] for v in zin],
    }
    dx_tag = "" if abs(DX - 0.197e-3) < 1e-9 else f"_dx{DX*1e6:.0f}um"
    jp = os.path.join(out_dir, f"xband_inset_{inset_mm:.1f}mm{dx_tag}.json")
    with open(jp, "w") as fh:
        json.dump(payload, fh)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inset-list-mm", default="0,2.4,2.8,3.2,3.6")
    p.add_argument("--output", default=os.path.join(SCRIPT_DIR, "out_xband"))
    p.add_argument("--num-periods", type=float, default=200.0)
    p.add_argument("--nfreq", type=int, default=161)
    p.add_argument("--dx-um", type=float, default=197.0,
                   help="uniform cell size (um); 98.5 = half-mesh refinement")
    args = p.parse_args()
    global DX, Z_SUB_LO, Z_SUB_HI, Z_MET_LO, Z_MET_HI
    DX = args.dx_um * 1e-6
    # recompute DX-derived stack coordinates (module-level values were baked
    # with the default DX; stale values leave an air gap under the substrate)
    Z_SUB_LO = Z_GND + DX
    Z_SUB_HI = Z_SUB_LO + H_SUB
    Z_MET_LO = Z_SUB_HI
    Z_MET_HI = Z_SUB_HI + DX

    os.makedirs(args.output, exist_ok=True)
    insets = [(-1.0 if v.strip().lower() == "thru" else float(v))
              for v in args.inset_list_mm.split(",") if v.strip()]
    results = [run_one(d, args, args.output) for d in insets]

    print("\n" + "=" * 72)
    print(f"{'inset':>7} {'dip dB':>8} {'f_dip GHz':>10} {'@9.21 dB':>9} "
          f"{'Zin@9.21':>16} {'max|S11|':>9}")
    fig, ax = plt.subplots(figsize=(9, 5.5), tight_layout=True)
    for r in results:
        f = np.array(r["freqs_hz"]) / 1e9
        db = 20 * np.log10(np.maximum(
            np.abs(np.array([complex(a, b) for a, b in r["s11"]])), 1e-9))
        z = r["zin_at_target_ohm"]
        print(f"{r['inset_mm']:>6.1f}m {r['s11_dip_db']:>7.2f} "
              f"{r['s11_dip_hz']/1e9:>10.3f} {r['s11_at_target_db']:>9.2f} "
              f"{z[0]:>8.0f}{z[1]:+7.0f}j {r['s11_max_abs']:>9.3f}")
        lbl = ("edge-fed (ref)" if r["inset_mm"] == 0
               else f"inset {r['inset_mm']:.1f} mm")
        ax.plot(f, db, label=lbl)
    ax.axvline(TARGET_GHZ, color="k", ls="--", alpha=0.4,
               label=f"Balanis {TARGET_GHZ} GHz")
    ax.set_xlabel("f (GHz)"); ax.set_ylabel("|S11| (dB)")
    ax.set_title("rfx X-band inset-fed patch (issue80 frame) — inset sweep")
    ax.grid(alpha=0.3); ax.legend()
    png = os.path.join(args.output, "xband_inset_sweep.png")
    fig.savefig(png, dpi=110)
    print(f"overlay -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

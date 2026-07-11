"""Inset-fed microstrip patch on the canonical Rogers substrate — rfx MSL port.

SUPERSEDED-BY: rfx_patch_xband_canonical_frame.py (#273). This script's patch
path still carries the retired interior-ground frame (ground PEC box inside
the domain with air+CPML below) that the PR #272 post-mortem replaced with the
repo's z_lo PEC boundary idiom. Kept for the inset-sweep history; do not
extend.

WHY THIS DESIGN / THIS PORT
---------------------------
Probe-feed replicas of the openEMS canonical patch through rfx's lumped/wire
`add_port` failed structurally: that extractor derives S11 from the port
cell's own V/I, which in a SERIES feed topology (via/probe column through the
substrate) measures the port element itself — Zin came out frequency-flat
(15.3 ohm across 1.5-3.0 GHz; S11 = (Z0-Zcell)/(Z0+Zcell) = flat -5.5 dB)
while Harminv saw the patch ringing at 2.34 GHz. The antenna never enters the
local V/I ratio. (Measured on VESSL runs 369367245963/964/967.)

The VALIDATED patch S11 path in rfx is the MSL port (issue #80): a microstrip
feed line + `add_msl_port` + `compute_msl_s_matrix` — N-probe wave split on
the line with reference-plane de-embedding, gate-verified against openEMS and
the Balanis analytic resonance, and AD-traceable via eps_override. That is a
feed-line extractor measuring propagating waves, exactly what a patch needs.

So: a classic Balanis INSET-FED rectangular patch (all-rectangular PEC — no
thin notch cells beyond the mesh resolution, rfx CPML is stable here), swept
over the inset depth to find the 50-ohm match:

    R_in(d) = R_edge * cos^2(pi * d / L)   ->  deep |S11| null when R_in ~ 50.

GEOMETRY (canonical Rogers substrate, resonance ~2.1 GHz)
---------------------------------------------------------
  * substrate eps_r = 3.38, h = 1.524 mm, tan_delta ~ 1e-3 (openEMS kappa)
  * patch: L = 40 mm along +x (RESONANT dimension, feed enters the -x
    radiating edge), W = 32 mm along y. Same plate as the openEMS canonical
    tutorial patch, rotated so the resonant dimension is along the feed.
  * 50-ohm feed line: w50 ~ 3.66 mm (12 cells; Hammerstad w/h ~ 2.32),
    running +x from the domain edge (x=0, backward wave absorbed by CPML,
    issue80 pattern) into the patch edge through two etched notch slots of
    gap g (6 cells ~ 1.83 mm), penetrating to inset depth d (CLI sweep).
  * ground + substrate: full domain footprint (issue80 pattern).
  * UNIFORM mesh dx = h/5 = 0.3048 mm, every coordinate snapped to integer
    cells (the NU lane is a minefield for ports: unfolded port impedance +
    smooth_grading misalignment — see rfx_patch_canonical.py history).
  * CPML 8 all faces; air above ~30 mm (~lambda/4.7 at 2.1 GHz).

Run (GPU):
    python scripts/research/calibration/crossval/rfx_patch_inset_msl.py \
        --inset-list-mm 10,12,14,16 \
        --output scripts/research/calibration/crossval/out_inset
"""

import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPS0 = 8.8541878128e-12

# --- canonical substrate ----------------------------------------------------
EPS_R = 3.38
H_SUB = 1.524e-3
SUB_SIGMA = 1e-3 * 2 * math.pi * 2.45e9 * EPS0 * EPS_R   # tan_delta ~ 1e-3

# --- uniform mesh: everything in integer cells of DX ------------------------
N_SUB = 5
DX = H_SUB / N_SUB                    # 0.3048 mm

def C(n):                              # n cells -> metres
    return n * DX

def cells(length_m):
    return max(1, int(round(length_m / DX)))

# --- plate ------------------------------------------------------------------
N_PATCH_L = cells(40.0e-3)            # resonant dimension along x (131)
N_PATCH_W = cells(32.0e-3)            # width along y (105)
N_W50 = 12                            # 50-ohm line width (3.658 mm)
N_GAP = 6                             # notch gap (1.829 mm)

# --- feed line / domain -----------------------------------------------------
N_PORT_MARGIN = 20                    # x cells before the msl port (~6.1 mm; >2*h_sub from CPML)
N_LINE = 76                           # port -> patch edge (~23.2 mm run)
N_MARGIN_X_HI = 66                    # air beyond patch (+x) (~20 mm)
N_MARGIN_Y = 66                       # air beside patch (~20 mm)
N_AIR_BELOW = 13                      # below ground (~4 mm, issue80 pattern)
N_AIR_ABOVE = 98                      # above patch (~30 mm)

N_DOM_X = N_PORT_MARGIN + N_LINE + N_PATCH_L + N_MARGIN_X_HI
N_DOM_Y = N_PATCH_W + 2 * N_MARGIN_Y
N_DOM_Z = N_AIR_BELOW + 1 + N_SUB + N_AIR_ABOVE   # +1 ground cell

DOM_X, DOM_Y, DOM_Z = C(N_DOM_X), C(N_DOM_Y), C(N_DOM_Z)
Y_C = C(N_DOM_Y // 2)                 # snapped centre line

Z_GND_LO = C(N_AIR_BELOW)
Z_GND_HI = C(N_AIR_BELOW + 1)
Z_SUB_LO = Z_GND_HI
Z_SUB_HI = C(N_AIR_BELOW + 1 + N_SUB)
Z_MET_LO = Z_SUB_HI                   # 1-cell metal layer
Z_MET_HI = Z_SUB_HI + DX

X_PATCH_LO = C(N_PORT_MARGIN + N_LINE)
X_PATCH_HI = X_PATCH_LO + C(N_PATCH_L)
Y_PATCH_LO = Y_C - C(N_PATCH_W // 2)
Y_PATCH_HI = Y_PATCH_LO + C(N_PATCH_W)
Y_FEED_LO = Y_C - C(N_W50 // 2)
Y_FEED_HI = Y_FEED_LO + C(N_W50)
Y_NOTCH_LO = Y_FEED_LO - C(N_GAP)
Y_NOTCH_HI = Y_FEED_HI + C(N_GAP)

REF_F_HZ = 2.085e9                    # openEMS canonical resonance (same plate)
PROBE_KW = None                       # set from CLI (--probe-offset etc.)


def build_sim(inset_cells, pulse_f0, pulse_bw):
    from rfx import Box, Simulation
    from rfx.sources import GaussianPulse

    x_conn = X_PATCH_LO + C(inset_cells)

    sim = Simulation(
        freq_max=4e9, domain=(DOM_X, DOM_Y, DOM_Z),
        dx=DX, cpml_layers=8, boundary="cpml",
    )
    sim.add_material("rogers", eps_r=EPS_R, sigma=SUB_SIGMA)
    # Ground + substrate: full domain footprint (issue80 pattern)
    sim.add(Box((0, 0, Z_GND_LO), (DOM_X, DOM_Y, Z_GND_HI)), material="pec")
    sim.add(Box((0, 0, Z_SUB_LO), (DOM_X, DOM_Y, Z_SUB_HI)), material="rogers")

    # The former "thru" arm (one-port line run into the far CPML) is deleted
    # (#273): that wiring was the PR #272 miswiring — one port and no matched
    # second port gives no thru observable at all. The canonical matched-thru
    # is scripts/diagnostics/msl_thru_mesh_convergence.py (two add_msl_port
    # ends on a z_lo PEC ground boundary — the second port IS the matched
    # termination; pattern of tests/test_msl_port_integration.py). Do not
    # rebuild a duplicate thru fixture here.

    # Metal layer (PEC is OR-accumulated -> build ADDITIVELY, the two notch
    # slots are simply never metallised):
    # (1) 50-ohm feed strip: domain edge -> inset connection plane
    sim.add(Box((0, Y_FEED_LO, Z_MET_LO), (x_conn, Y_FEED_HI, Z_MET_HI)),
            material="pec")
    # (2) patch body beyond the inset depth (full width)
    sim.add(Box((x_conn, Y_PATCH_LO, Z_MET_LO),
                (X_PATCH_HI, Y_PATCH_HI, Z_MET_HI)), material="pec")
    # (3) lower flank beside the lower notch slot
    sim.add(Box((X_PATCH_LO, Y_PATCH_LO, Z_MET_LO),
                (x_conn, Y_NOTCH_LO, Z_MET_HI)), material="pec")
    # (4) upper flank beside the upper notch slot
    sim.add(Box((X_PATCH_LO, Y_NOTCH_HI, Z_MET_LO),
                (x_conn, Y_PATCH_HI, Z_MET_HI)), material="pec")

    sim.add_msl_port(
        position=(C(N_PORT_MARGIN), Y_C, Z_SUB_LO),
        width=C(N_W50), height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=pulse_f0, bandwidth=pulse_bw),
        **(PROBE_KW or {}),
    )
    return sim


def run_one(inset_mm, args, out_dir):
    inset_c = cells(inset_mm * 1e-3)
    print("=" * 72)
    print(f"INSET d = {inset_mm:.1f} mm ({inset_c} cells; snapped "
          f"{C(inset_c)*1e3:.3f} mm)   patch L={C(N_PATCH_L)*1e3:.2f} x "
          f"W={C(N_PATCH_W)*1e3:.2f} mm   w50={C(N_W50)*1e3:.3f} mm "
          f"gap={C(N_GAP)*1e3:.3f} mm")
    print(f"  domain {DOM_X*1e3:.1f} x {DOM_Y*1e3:.1f} x {DOM_Z*1e3:.1f} mm "
          f"= {N_DOM_X}x{N_DOM_Y}x{N_DOM_Z} cells "
          f"({N_DOM_X*N_DOM_Y*N_DOM_Z/1e6:.1f}M interior)")

    sim = build_sim(inset_c, args.pulse_f0_ghz * 1e9, args.pulse_bw)
    print("  preflight:")
    sim.preflight(strict=False)

    import jax.numpy as jnp
    t0 = time.time()
    res = sim.compute_msl_s_matrix(
        freqs=jnp.linspace(args.freq_lo_ghz * 1e9, args.freq_hi_ghz * 1e9,
                           args.nfreq),
        num_periods=args.num_periods,
    )
    t_run = time.time() - t0

    freqs = np.asarray(res.freqs, dtype=float)
    s11 = np.asarray(res.S)[0, 0, :]
    mag = np.abs(s11)
    db = 20 * np.log10(np.maximum(mag, 1e-9))
    zin = 50.0 * (1 + s11) / (1 - s11)

    i_dip = int(np.argmin(db))
    print(f"  |S11| dip: {db[i_dip]:.2f} dB @ {freqs[i_dip]/1e9:.4f} GHz  "
          f"Zin={zin[i_dip].real:.1f}{zin[i_dip].imag:+.1f}j  "
          f"max|S11|={mag.max():.3f}  ({t_run:.0f}s)")

    payload = {
        "case": "inset_fed_patch_msl_canonical_substrate",
        "inset_mm": float(inset_mm), "inset_cells": int(inset_c),
        "geometry": {
            "eps_r": EPS_R, "h_sub_mm": H_SUB * 1e3,
            "patch_L_mm": C(N_PATCH_L) * 1e3, "patch_W_mm": C(N_PATCH_W) * 1e3,
            "w50_mm": C(N_W50) * 1e3, "notch_gap_mm": C(N_GAP) * 1e3,
            "feed_line_mm": C(N_LINE) * 1e3, "dx_mm": DX * 1e3,
            "dom_mm": [DOM_X * 1e3, DOM_Y * 1e3, DOM_Z * 1e3],
        },
        "runtime_s": t_run,
        "num_periods": args.num_periods,
        "s11_dip_db": float(db[i_dip]),
        "s11_dip_hz": float(freqs[i_dip]),
        "zin_at_dip_ohm": [float(zin[i_dip].real), float(zin[i_dip].imag)],
        "s11_max_abs": float(mag.max()),
        "freqs_hz": freqs.tolist(),
        "s11": [[float(v.real), float(v.imag)] for v in s11],
        "z11": [[float(v.real), float(v.imag)] for v in zin],
    }
    jp = os.path.join(out_dir, f"rfx_patch_inset_{inset_mm:.0f}mm.json")
    with open(jp, "w") as fh:
        json.dump(payload, fh)
    print(f"  saved {jp}")
    return payload


# run_thru() deleted (#273): the one-port-line-into-CPML "thru" was the
# PR #272 miswiring (see the note in build_sim). Canonical matched-thru:
# scripts/diagnostics/msl_thru_mesh_convergence.py.


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inset-list-mm", default="10,12,14,16")
    p.add_argument("--output", default=os.path.join(SCRIPT_DIR, "out_inset"))
    p.add_argument("--num-periods", type=float, default=120.0)
    p.add_argument("--nfreq", type=int, default=121)
    p.add_argument("--freq-lo-ghz", type=float, default=1.5)
    p.add_argument("--freq-hi-ghz", type=float, default=3.0)
    p.add_argument("--pulse-f0-ghz", type=float, default=2.25)
    p.add_argument("--pulse-bw", type=float, default=1.2)
    p.add_argument("--probe-offset", type=int, default=None,
                   help="explicit n_probe_offset (cells)")
    p.add_argument("--probe-spacing", type=int, default=None)
    p.add_argument("--n-probes", type=int, default=None)
    args = p.parse_args()
    global PROBE_KW
    kw = {}
    if args.probe_offset is not None:
        kw["n_probe_offset"] = args.probe_offset
    if args.probe_spacing is not None:
        kw["n_probe_spacing"] = args.probe_spacing
    if args.n_probes is not None:
        kw["n_probes"] = args.n_probes
    PROBE_KW = kw

    os.makedirs(args.output, exist_ok=True)
    # "thru" CLI token deleted with the thru arm (#273, PR #272 post-mortem).
    insets = [float(v) for v in args.inset_list_mm.split(",") if v.strip()]
    results = [run_one(d, args, args.output) for d in insets]

    # summary + overlay
    print("\n" + "=" * 72)
    print(f"{'inset':>8} {'dip dB':>9} {'f_dip GHz':>10} {'Zin@dip':>18}")
    fig, ax = plt.subplots(figsize=(9, 5.5), tight_layout=True)
    for r in results:
        f = np.array(r["freqs_hz"]) / 1e9
        db = 20 * np.log10(np.maximum(
            np.abs(np.array([complex(a, b) for a, b in r["s11"]])), 1e-9))
        z = r["zin_at_dip_ohm"]
        print(f"{r['inset_mm']:>7.1f}m {r['s11_dip_db']:>8.2f} "
              f"{r['s11_dip_hz']/1e9:>10.4f} "
              f"{z[0]:>9.1f}{z[1]:+8.1f}j")
        ax.plot(f, db, label=f"inset {r['inset_mm']:.0f} mm")
    ax.axvline(REF_F_HZ / 1e9, color="k", ls="--", alpha=0.4,
               label="openEMS canonical 2.085 GHz")
    ax.set_xlabel("f (GHz)"); ax.set_ylabel("|S11| (dB)")
    ax.set_title("rfx inset-fed patch (MSL port) — inset matching sweep")
    ax.grid(alpha=0.3); ax.legend()
    png = os.path.join(args.output, "rfx_patch_inset_sweep.png")
    fig.savefig(png, dpi=110)
    print(f"overlay -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

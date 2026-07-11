"""X-band inset patch in the CANONICAL rfx msl frame + common 3-solver box.

Corrects the retracted fixture (PR #272 post-mortem): this build follows the
repo's validated msl idiom (scripts/diagnostics/msl_thru_mesh_convergence.py)
- ground = z_lo PEC BOUNDARY FACE (BoundarySpec z lo="pec"), NOT an interior
  ground box with air below;
- substrate spans z = 0 .. H_SUB over the full footprint;
- CPML only on x/y/z_hi.
and the COMMON box shared with the Palace mesh and the openEMS replica:
  x in [-12, 18.595] -> rfx [0, 30.595] mm (feed wall at x=0)
  y +-15.0645        -> rfx [0, 30.129] mm
  z [0, 10.787] mm   (ground plane at z=0)

Observables per run:
  * full complex S11(f) 7-12 GHz via add_msl_port + compute_msl_s_matrix
    (dip location documented mesh-limited -> we compare the CURVE + mesh
    series, not dip equality);
  * Harminv ring-down resonance (separate unported run) for the Level-1
    resonance cross-check.

Modes: --mode s11 (default) | harminv | shielded-harminv (all-PEC box for
the Level-1 shielded resonance match).

Run: python rfx_patch_xband_canonical_frame.py --mode s11 --dx-um 197
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- common-box design (mm; matches palace_patch/mesh_patch.py) --------------
EPS_R = 3.38
H_SUB = 0.787e-3
PATCH_L = 8.595e-3        # x, resonant
PATCH_W = 10.129e-3       # y
W_MSL = 1.8e-3
NOTCH_GAP = 0.9e-3
INSET_D = 2.4e-3
L_LINE = 12.0e-3          # feed wall (x=0) -> patch edge
MARGIN_XY = 10.0e-3
AIR_ABOVE = 10.0e-3

DOM_X = L_LINE + PATCH_L + MARGIN_XY            # 30.595 mm
DOM_Y = PATCH_W + 2 * MARGIN_XY                 # 30.129 mm
DOM_Z = H_SUB + AIR_ABOVE                       # 10.787 mm
Y_C = DOM_Y / 2
X_PATCH_LO = L_LINE
X_PATCH_HI = L_LINE + PATCH_L
X_CONN = X_PATCH_LO + INSET_D
PORT_MARGIN = 5.0e-3      # msl port x (feed plane), probes downstream

TARGET_GHZ = 9.21


def build(mode, dx):
    from rfx import Box, Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.sources import GaussianPulse

    if mode == "shielded-harminv":
        bspec = BoundarySpec.uniform("pec")
        cpml = 0
    else:
        # canonical msl frame: ground = z_lo PEC face, CPML elsewhere
        bspec = BoundarySpec(x="cpml", y="cpml",
                             z=Boundary(lo="pec", hi="cpml"))
        cpml = 8
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z), dx=dx,
                     cpml_layers=cpml, boundary=bspec)
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    # substrate: full footprint, z = 0 .. H_SUB (ground is the z_lo face)
    sim.add(Box((0, 0, 0), (DOM_X, DOM_Y, H_SUB)), material="ro4003c")

    zm_lo, zm_hi = H_SUB, H_SUB + dx
    yf_lo, yf_hi = Y_C - W_MSL / 2, Y_C + W_MSL / 2
    yp_lo, yp_hi = Y_C - PATCH_W / 2, Y_C + PATCH_W / 2
    yn_lo, yn_hi = yf_lo - NOTCH_GAP, yf_hi + NOTCH_GAP
    # feed strip: wall (x=0) -> inset connection plane
    sim.add(Box((0, yf_lo, zm_lo), (X_CONN, yf_hi, zm_hi)), material="pec")
    # patch body + flanks (additive PEC; notch slots stay unmetallised)
    sim.add(Box((X_CONN, yp_lo, zm_lo), (X_PATCH_HI, yp_hi, zm_hi)),
            material="pec")
    sim.add(Box((X_PATCH_LO, yp_lo, zm_lo), (X_CONN, yn_lo, zm_hi)),
            material="pec")
    sim.add(Box((X_PATCH_LO, yn_hi, zm_lo), (X_CONN, yp_hi, zm_hi)),
            material="pec")

    if mode == "s11":
        sim.add_msl_port(
            position=(PORT_MARGIN, Y_C, 0.0), width=W_MSL, height=H_SUB,
            direction="+x", impedance=50.0,
            waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
        )
    else:
        zmid = H_SUB / 2
        sim.add_source(position=(X_PATCH_LO + 0.75 * PATCH_L,
                                 Y_C + 0.25 * PATCH_W, zmid),
                       component="ez",
                       waveform=GaussianPulse(f0=9.5e9, bandwidth=0.8))
        sim.add_probe(position=(X_PATCH_LO + 0.35 * PATCH_L,
                                Y_C - 0.30 * PATCH_W, zmid), component="ez")
    return sim


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["s11", "harminv", "shielded-harminv"],
                   default="s11")
    p.add_argument("--dx-um", type=float, default=197.0)
    p.add_argument("--num-periods", type=float, default=200.0)
    p.add_argument("--nfreq", type=int, default=201)
    p.add_argument("--output", default=os.path.join(SCRIPT_DIR, "out_canonical_frame"))
    args = p.parse_args()

    dx = args.dx_um * 1e-6
    os.makedirs(args.output, exist_ok=True)
    print(f"mode={args.mode}  dx={args.dx_um}um  "
          f"box {DOM_X*1e3:.3f} x {DOM_Y*1e3:.3f} x {DOM_Z*1e3:.3f} mm")
    sim = build(args.mode, dx)
    print("preflight:")
    sim.preflight(strict=False)

    t0 = time.time()
    if args.mode == "s11":
        import jax.numpy as jnp
        res = sim.compute_msl_s_matrix(
            freqs=jnp.linspace(7e9, 12e9, args.nfreq),
            num_periods=args.num_periods,
        )
        freqs = np.asarray(res.freqs, dtype=float)
        s11 = np.asarray(res.S)[0, 0, :]
        db = 20 * np.log10(np.maximum(np.abs(s11), 1e-9))
        i = int(np.argmin(db))
        print(f"  |S11| dip {db[i]:.2f} dB @ {freqs[i]/1e9:.4f} GHz  "
              f"max|S11|={np.abs(s11).max():.3f}  ({time.time()-t0:.0f}s)")
        out = {
            "case": "canonical_frame_inset_s11", "dx_um": args.dx_um,
            "s11_dip_db": float(db[i]), "s11_dip_hz": float(freqs[i]),
            "s11_max_abs": float(np.abs(s11).max()),
            "freqs_hz": freqs.tolist(),
            "s11": [[float(v.real), float(v.imag)] for v in s11],
        }
        jp = os.path.join(args.output,
                          f"rfx_cf_s11_dx{args.dx_um:.0f}um.json")
        json.dump(out, open(jp, "w"))
        print(f"  saved {jp}")
    else:
        from rfx.harminv import harminv
        res = sim.run(num_periods=120)
        ts = np.asarray(res.time_series)[:, 0]
        dt = float(getattr(res, "dt", 0) or sim._build_grid().dt)
        n0 = int(len(ts) * 0.35)
        modes = [m for m in harminv(ts[n0:], dt, f_min=7e9, f_max=12e9)
                 if m.Q > 5]
        print(f"  ring-down modes ({args.mode}, {time.time()-t0:.0f}s):")
        for m in sorted(modes, key=lambda m: -abs(m.amplitude))[:6]:
            print(f"    {m.freq/1e9:.4f} GHz  Q={m.Q:.1f}  "
                  f"amp={abs(m.amplitude):.3e}")
        out = {
            "case": f"canonical_frame_{args.mode}", "dx_um": args.dx_um,
            "modes": [[float(m.freq), float(m.Q), float(abs(m.amplitude))]
                      for m in modes],
        }
        jp = os.path.join(args.output,
                          f"rfx_cf_{args.mode}_dx{args.dx_um:.0f}um.json")
        json.dump(out, open(jp, "w"))
        print(f"  saved {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

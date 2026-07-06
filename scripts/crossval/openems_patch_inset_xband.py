"""openEMS replication of the rfx X-band INSET-FED matched patch.

Replicates the rfx design (scripts/crossval/rfx_patch_inset_xband.py, issue80
frame): RO4003C eps_r=3.38 h=0.787 mm, patch L=8.595 (x, resonant) x
W=10.129 mm (y), 50-ohm feed w=1.8 mm entering through two 0.9 mm etched
notch slots to inset depth d (default 2.4 mm = the rfx match:
rfx dip -30.25 dB @ 10.359 GHz).

openEMS side follows the official MSL_NotchFilter tutorial feed pattern:
  * FDTD.AddMSLPort(..., 'x', 'z', excite=-1, FeedShift, MeasPlaneShift) —
    the port DRAWS the feed strip and de-embeds to the measurement plane.
  * boundaries ['PML_8','MUR','MUR','MUR','PEC','MUR']: the line's far end
    runs into the x_lo PML (matched termination behind the feed);
    z_lo = PEC IS the ground plane (no explicit ground box).
  * thirds-rule mesh lines on the strip edges; substrate 4 z-cells.

Difference to rfx frame (documented): rfx has a finite full-footprint ground
inside the domain with air+CPML below; here the ground is the PEC boundary.
Both look "infinite" to the patch within the domain; noted as a caveat.

Run: python scripts/crossval/openems_patch_inset_xband.py [--inset-mm 2.4]
"""

import argparse
import json
import math
import os
import shutil
import time

import numpy as np
np.float = float; np.int = int  # openEMS 0.0.35 bindings shim for numpy>=1.24  # noqa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "out_xband")

C0 = 299792458.0
RES_SCALE = 1.0     # CLI --res-scale (0.5 = half-mesh refinement)
METAL_T = 0.0       # CLI --metal-t-mm: metal slab thickness (0 = sheet)

# --- design (matches rfx_patch_inset_xband.py) -------------------------------
EPS_R = 3.38
H_SUB = 0.787          # mm
PATCH_L = 8.595        # x, resonant
PATCH_W = 10.129       # y
W_MSL = 1.8
NOTCH_GAP = 0.9
L_LINE = 12.0          # feed line length before the patch edge (x<0)
# Box-matching defaults = the rfx issue80 frame (patch sees the same walls):
#   rfx domain 29.747 x 18.130 x 12.787, patch y-margins (18.130-10.129)/2,
#   x beyond patch 29.747-13-8.595, air above patch ~7.6 mm.
MARGIN_X_HI = 8.152    # air beyond patch (rfx frame)
MARGIN_Y = 4.0005      # air beside patch/line (rfx frame)
AIR_ABOVE = 7.6        # air above substrate (rfx frame)

F_LO, F_HI, NFREQ = 1.5e9, 15e9, 161


def run(inset_mm, nrts, sim_dir, calc_only=False):
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS as OEMS

    d = float(inset_mm)
    x_edge = 0.0                     # patch leading edge
    x_conn = x_edge + d              # inset connection plane
    x_hi = x_edge + PATCH_L + MARGIN_X_HI
    y_lo, y_hi = -PATCH_W / 2 - MARGIN_Y, PATCH_W / 2 + MARGIN_Y
    yf_lo, yf_hi = -W_MSL / 2, W_MSL / 2
    yn_lo, yn_hi = yf_lo - NOTCH_GAP, yf_hi + NOTCH_GAP
    yp_lo, yp_hi = -PATCH_W / 2, PATCH_W / 2

    f0 = 0.5 * (F_LO + F_HI)
    fc = 0.5 * (F_HI - F_LO) * 1.15
    FDTD = OEMS(NrTS=nrts, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    # x_lo PML terminates the feed line; z_lo PEC = ground plane.
    FDTD.SetBoundaryCond(['PML_8', 'MUR', 'MUR', 'MUR', 'PEC', 'MUR'])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)          # mm

    # substrate: full footprint, z in [0, H_SUB]
    sub = CSX.AddMaterial('ro4003c', epsilon=EPS_R)
    sub.AddBox([-L_LINE, y_lo, 0], [x_hi, y_hi, H_SUB], priority=1)

    pec = CSX.AddMetal('pec')
    # metal layer: sheet at z=H_SUB (t=0) or an rfx-mimicking THICK slab
    # [H_SUB, H_SUB+t] (t>0) to test the metal-thickness junction hypothesis.
    zt = H_SUB + METAL_T
    if d > 0:
        pec.AddBox([x_edge, yf_lo, H_SUB], [x_conn, yf_hi, zt], priority=10)
        pec.AddBox([x_conn, yp_lo, H_SUB], [x_edge + PATCH_L, yp_hi, zt],
                   priority=10)
        pec.AddBox([x_edge, yp_lo, H_SUB], [x_conn, yn_lo, zt], priority=10)
        pec.AddBox([x_edge, yn_hi, H_SUB], [x_conn, yp_hi, zt], priority=10)
    else:
        pec.AddBox([x_edge, yp_lo, H_SUB], [x_edge + PATCH_L, yp_hi, zt],
                   priority=10)

    # ---- mesh (tutorial pattern) ----
    res = C0 / (F_HI * math.sqrt(EPS_R)) / 1e-3 / 50.0 * RES_SCALE
    third = np.array([2 * res / 3, -res / 3])
    mesh.AddLine('x', [-L_LINE, x_hi])
    for xl in (x_edge, x_conn, x_edge + PATCH_L):
        mesh.AddLine('x', xl + third)
        mesh.AddLine('x', xl - third[::-1] * 0)   # ensure the exact line too
        mesh.AddLine('x', xl)
    mesh.SmoothMeshLines('x', res)
    mesh.AddLine('y', [y_lo, y_hi, 0.0])
    for yl in (yf_lo, yf_hi, yn_lo, yn_hi, yp_lo, yp_hi):
        mesh.AddLine('y', yl + third)
        mesh.AddLine('y', yl)
    mesh.SmoothMeshLines('y', res)
    zl = np.linspace(0, H_SUB, 5).tolist()
    if METAL_T > 0:
        zl += [H_SUB + METAL_T]
    mesh.AddLine('z', zl)
    mesh.AddLine('z', AIR_ABOVE)
    mesh.SmoothMeshLines('z', res)

    # ---- MSL port: draws the strip from the PML end to the patch edge ----
    portstart = [-L_LINE, yf_lo, H_SUB]
    portstop = [x_edge, yf_hi, 0]
    port = FDTD.AddMSLPort(1, pec, portstart, portstop, 'x', 'z',
                           excite=-1, FeedShift=10 * res,
                           MeasPlaneShift=L_LINE / 3, priority=10)

    if not calc_only:
        if os.path.exists(sim_dir):
            shutil.rmtree(sim_dir)
        os.makedirs(sim_dir)
    nx, ny, nz = (len(mesh.GetLines('x')), len(mesh.GetLines('y')),
                  len(mesh.GetLines('z')))
    print(f"  mesh {nx} x {ny} x {nz} = {nx*ny*nz/1e3:.0f}k cells, "
          f"res={res:.3f} mm")
    t0 = time.time()
    if not calc_only:
        FDTD.Run(sim_dir, verbose=0, cleanup=True)
    t_run = time.time() - t0

    f = np.linspace(F_LO, F_HI, NFREQ)
    port.CalcPort(sim_dir, f, ref_impedance=50)
    s11 = port.uf_ref / port.uf_inc
    return f, s11, t_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inset-mm", type=float, default=2.4)
    p.add_argument("--nrts", type=int, default=60000)
    p.add_argument("--calc-only", action="store_true", help="post-process existing port dumps in sim_dir (no FDTD run)")
    p.add_argument("--margin-y", type=float, default=None)
    p.add_argument("--margin-x-hi", type=float, default=None)
    p.add_argument("--air-above", type=float, default=None)
    p.add_argument("--res-scale", type=float, default=1.0)
    p.add_argument("--metal-t-mm", type=float, default=0.0)
    args = p.parse_args()

    global MARGIN_Y, MARGIN_X_HI, AIR_ABOVE, RES_SCALE, METAL_T
    RES_SCALE = args.res_scale
    METAL_T = args.metal_t_mm
    if args.margin_y is not None:
        MARGIN_Y = args.margin_y
    if args.margin_x_hi is not None:
        MARGIN_X_HI = args.margin_x_hi
    if args.air_above is not None:
        AIR_ABOVE = args.air_above
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"openEMS X-band inset patch: d={args.inset_mm} mm "
          f"(margins y={MARGIN_Y}, x_hi={MARGIN_X_HI}, above={AIR_ABOVE})")
    sim_dir = os.path.join(SCRIPT_DIR, f"oems_xband_tmp_d{args.inset_mm:.1f}")
    f, s11, t_run = run(args.inset_mm, args.nrts, sim_dir, calc_only=args.calc_only)

    db = 20 * np.log10(np.maximum(np.abs(s11), 1e-9))
    i = int(np.argmin(db))
    print(f"  dip {db[i]:.2f} dB @ {f[i]/1e9:.3f} GHz   max|S11|="
          f"{np.abs(s11).max():.3f}   ({t_run:.0f}s)")

    out = {
        "case": "openems_xband_inset_patch",
        "inset_mm": args.inset_mm, "runtime_s": t_run,
        "s11_dip_db": float(db[i]), "s11_dip_hz": float(f[i]),
        "s11_max_abs": float(np.abs(s11).max()),
        "freqs_hz": f.tolist(),
        "s11": [[float(v.real), float(v.imag)] for v in s11],
    }
    rs_tag = "" if abs(args.res_scale - 1.0) < 1e-9 else f"_rs{args.res_scale:g}"
    if args.metal_t_mm > 0:
        rs_tag += f"_mt{args.metal_t_mm:g}"
    jp = os.path.join(OUT_DIR, f"openems_xband_inset_{args.inset_mm:.1f}mm{rs_tag}.json")
    with open(jp, "w") as fh:
        json.dump(out, fh)
    print(f"  saved {jp}")
    if not args.calc_only:
        shutil.rmtree(sim_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""openEMS design iterator: MATCHED inset-fed microstrip patch (cv05 2.4 GHz FR4).

The cv05 probe-fed patch with a fixed 8 mm vertical lumped port is essentially
UNMATCHED (|S11| ~ 0 dB flat, no resonant null) so its S11 has no feature to
cross-validate. This script designs a MATCHED variant: the standard Balanis
inset-fed rectangular microstrip patch.

Feed topology (in the x-y plane, patch metal at z = h_sub):
  * The resonant (half-wave) dimension L is along x; W is along y.
  * A 50 ohm microstrip line (width `w_feed`, synthesized for FR4/h) runs along
    +x on the substrate over the finite ground plane, enters the patch through
    the radiating edge at x = patch_x_lo, and PENETRATES the patch by an inset
    depth y0.
  * Two etched notch slots (each `notch_gap` wide, y0 long) flank the feed strip
    so the strip is isolated from the patch body for the first y0 of its
    penetration and connects galvanically only at depth y0. This transforms the
    high edge resistance toward 50 ohm:  R_in(y0) = R_edge * cos^2(pi*y0/L).
  * Excitation: a 50 ohm vertical lumped port (ground -> strip) at the outer end
    of the microstrip line. Because the line is ~50 ohm and low-loss, the port
    |S11| magnitude equals the |S11| at the inset connection (the line only
    rotates the reflection phase), so the null depth is preserved.

Sweep y0 over {6, 8, 10, 11, 12, 13} mm, extract |S11|(f), and find the y0 with
the deepest null near the ~2.4-2.5 GHz resonance. Emits:
  * scripts/crossval/out_ref/openems_patch_inset_sweep.json  (full sweep)
  * scripts/crossval/out_ref/openems_patch_matched.json      (best y0)
  * scripts/crossval/matched_patch_geometry.json             (geometry spec)

Run (system python — openEMS / CSXCAD bindings live there):
  python scripts/crossval/openems_patch_inset_sweep.py
"""

import json
import math
import os
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "out_ref")
os.makedirs(OUT_DIR, exist_ok=True)
C0 = 2.998e8
Z0_PORT = 50.0

# =============================================================================
# Fixed substrate / patch geometry — identical to cv05 openems_patch_s11.py
# =============================================================================
f_design = 2.4e9
eps_r = 4.3
tan_delta = 0.02
h_sub = 1.5e-3
W = 38.0e-3
L = 29.5e-3
gx = 60.0e-3
gy = 55.0e-3

# =============================================================================
# Inset-feed parameters
# =============================================================================
# 50 ohm microstrip width synthesized via Hammerstad (W/h < 2 branch).
_A = Z0_PORT / 60.0 * math.sqrt((eps_r + 1) / 2) + \
     (eps_r - 1) / (eps_r + 1) * (0.23 + 0.11 / eps_r)
_Woh = 8 * math.exp(_A) / (math.exp(2 * _A) - 2)
W_FEED = _Woh * h_sub                # ~2.92 mm for FR4 h=1.5 mm
NOTCH_GAP = 1.0e-3                   # etched slot each side of the feed strip
FEED_LEN = 8.0e-3                    # external microstrip length (edge -> port)

# Balanis edge-resistance estimate (single-slot conductance, Ch. 14) and the
# closed-form starting inset depth. Used only for reporting / sanity.
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_sub / W) ** (-0.5)
delta_L = 0.412 * h_sub * ((eps_eff + 0.3) * (W / h_sub + 0.264)) / \
          ((eps_eff - 0.258) * (W / h_sub + 0.8))
L_eff = L + 2 * delta_L
f_resonance_an = C0 / (2 * L_eff * math.sqrt(eps_eff))
_lam0 = C0 / f_resonance_an
_G1 = (1.0 / 90.0) * (W / _lam0) ** 2          # single-slot conductance
R_EDGE_EST = 1.0 / (2.0 * _G1)                 # no mutual term (upper bound)
Y0_BALANIS = (L / math.pi) * math.acos(math.sqrt(min(Z0_PORT / R_EDGE_EST, 1.0)))

# Sweep of inset depths (mm)
Y0_SWEEP_MM = [6.0, 8.0, 10.0, 11.0, 12.0, 13.0]

# Shared S11 frequency grid (1.5-3.5 GHz); finer than cv05 to resolve the null.
FREQS_HZ = np.linspace(1.5e9, 3.5e9, 201)

NUM_THREADS = min(int(os.environ.get("OPENEMS_THREADS", "32")), os.cpu_count() or 16)
NRTS = int(os.environ.get("OPENEMS_NRTS", "30000"))


def _numpy_compat_shim():
    for _n in ("float", "int", "complex"):
        if not hasattr(np, _n):
            setattr(np, _n, {"float": float, "int": int, "complex": complex}[_n])


def _merge_lines(lines, tol):
    """Sort, round, and drop mesh lines closer than ``tol`` mm to the previous
    kept line. Fixing geometry edges AND a fine arange onto the same axis leaves
    near-coincident lines (e.g. 0.017 mm apart); such micro-cells collapse the
    Courant dt and truncate the excitation pulse. Merging keeps min cell >= tol.
    """
    lines = np.unique(np.round(np.sort(np.asarray(lines, float)), 4))
    kept = [lines[0]]
    for v in lines[1:]:
        if v - kept[-1] >= tol:
            kept.append(v)
    return np.array(kept)


def build_inset_patch(y0_mm, sim_path):
    """Construct the inset-fed microstrip patch for one inset depth y0 (mm).

    Returns (FDTD, port, mesh_info_dict).
    """
    from CSXCAD.CSXCAD import ContinuousStructure
    from CSXCAD.SmoothMeshLines import SmoothMeshLines
    from openEMS.openEMS import openEMS as OEMS

    UNIT = 1e-3  # mm

    # Geometry in mm
    L_mm, W_mm, h_mm = L * 1e3, W * 1e3, h_sub * 1e3
    gx_mm, gy_mm = gx * 1e3, gy * 1e3
    wf_mm = W_FEED * 1e3
    g_mm = NOTCH_GAP * 1e3
    feed_len_mm = FEED_LEN * 1e3
    y0 = float(y0_mm)

    # Domain: ground plane + >= lambda/2 air margin (PML) + radiation air above.
    margin_mm = 50.0
    air_above_mm = 40.0
    dom_x_mm = gx_mm + 2 * margin_mm
    dom_y_mm = gy_mm + 2 * margin_mm
    dom_z_mm = h_mm + air_above_mm
    x_c, y_c = dom_x_mm / 2, dom_y_mm / 2

    # Solver. Matched antenna decays fast -> EndCriteria terminates early;
    # NrTS caps the worst-case (unmatched, longer ring-down) runs.
    f0_hz, fc_hz = f_design, 1.2e9
    FDTD = OEMS(NrTS=NRTS, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0_hz, fc_hz)
    FDTD.SetBoundaryCond(['PML_8'] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    # FR4 substrate (lossy, matching cv05 sigma model)
    sub_mat = CSX.AddMaterial('FR4')
    sub_mat.SetMaterialProperty(
        epsilon=eps_r,
        kappa=2 * math.pi * f0_hz * 8.8541878128e-12 * eps_r * tan_delta,
    )
    sub_lo = [x_c - gx_mm / 2, y_c - gy_mm / 2, 0]
    sub_hi = [x_c + gx_mm / 2, y_c + gy_mm / 2, h_mm]
    sub_mat.AddBox(sub_lo, sub_hi, priority=1)

    # Finite ground plane (2D PEC at z=0)
    gnd = CSX.AddMetal('gnd')
    gnd.AddBox([sub_lo[0], sub_lo[1], 0], [sub_hi[0], sub_hi[1], 0], priority=10)

    # --- Patch metal at z=h_sub, built from blocks that leave two notch slots
    #     and the connecting feed strip, giving a Balanis inset feed. ---
    x0 = x_c - L_mm / 2          # radiating (feed) edge
    x1 = x_c + L_mm / 2
    y0p = y_c - W_mm / 2
    y1p = y_c + W_mm / 2
    x_conn = x0 + y0             # inset connection plane (strip -> patch body)
    yf_lo = y_c - wf_mm / 2      # feed-strip edges
    yf_hi = y_c + wf_mm / 2
    yn_lo = yf_lo - g_mm         # outer edges of the two notch slots
    yn_hi = yf_hi + g_mm

    patch = CSX.AddMetal('patch')
    # (1) patch body beyond the inset depth (full width)
    patch.AddBox([x_conn, y0p, h_mm], [x1, y1p, h_mm], priority=10)
    # (2) lower flank beside the lower notch
    patch.AddBox([x0, y0p, h_mm], [x_conn, yn_lo, h_mm], priority=10)
    # (3) upper flank beside the upper notch
    patch.AddBox([x0, yn_hi, h_mm], [x_conn, y1p, h_mm], priority=10)
    # (4) connecting feed strip inside the patch (between the two notches)
    patch.AddBox([x0, yf_lo, h_mm], [x_conn, yf_hi, h_mm], priority=10)
    #     -> the two notch slots are the un-metallized gaps:
    #        y in [yn_lo, yf_lo] and [yf_hi, yn_hi], x in [x0, x_conn].

    # (5) external 50 ohm microstrip line (edge -> port), continuous with (4)
    feed_x_start = x0 - feed_len_mm
    feed = CSX.AddMetal('feed')
    feed.AddBox([feed_x_start, yf_lo, h_mm], [x0, yf_hi, h_mm], priority=10)

    # --- Mesh: fine in the feed/notch region, coarse in air/PML ---
    fine_feed = 0.3     # resolve 1.0 mm notch gap with ~3 cells
    fine_patch = 0.6    # patch body
    coarse_mm = 5.0
    sub_cells = 6       # 0.25 mm z cells across the 1.5 mm substrate

    # x fixed lines: geometry edges + fine feed/notch region + patch body
    x_feed_fine = np.arange(feed_x_start, x_conn + 2.0, fine_feed)
    x_patch = np.arange(x0, x1 + fine_patch / 2, fine_patch)
    x_fixed = np.concatenate([
        [0.0, dom_x_mm, sub_lo[0], sub_hi[0], x0, x1, x_conn, feed_x_start],
        x_feed_fine, x_patch,
    ])
    # y fixed lines: notch/feed edges + fine band + patch body
    y_feed_fine = np.arange(yn_lo - 2.0, yn_hi + 2.0, fine_feed)
    y_patch = np.arange(y0p, y1p + fine_patch / 2, fine_patch)
    y_fixed = np.concatenate([
        [0.0, dom_y_mm, sub_lo[1], sub_hi[1], y0p, y1p,
         yn_lo, yf_lo, y_c, yf_hi, yn_hi],
        y_feed_fine, y_patch,
    ])
    z_fixed = np.concatenate([
        [0.0, dom_z_mm, h_mm], np.linspace(0, h_mm, sub_cells + 1),
    ])

    # Merge near-coincident lines (tol 0.2 mm) so no micro-cell shrinks dt.
    x_lines = SmoothMeshLines(_merge_lines(x_fixed, 0.2), coarse_mm, ratio=1.4)
    y_lines = SmoothMeshLines(_merge_lines(y_fixed, 0.2), coarse_mm, ratio=1.4)
    z_lines = SmoothMeshLines(np.unique(np.round(z_fixed, 4)), coarse_mm, ratio=1.4)

    # Snap the lumped port to FINAL mesh lines so its excitation box coincides
    # with mesh edges (otherwise openEMS drops the excitation -> S11 = NaN).
    port_x = float(x_lines[np.argmin(np.abs(x_lines - feed_x_start))])
    port_y = float(y_lines[np.argmin(np.abs(y_lines - y_c))])
    port = FDTD.AddLumpedPort(
        port_nr=1, R=Z0_PORT,
        start=[port_x, port_y, 0.0],
        stop=[port_x, port_y, h_mm],
        p_dir='z', excite=1.0,
    )

    mesh.SetLines('x', x_lines)
    mesh.SetLines('y', y_lines)
    mesh.SetLines('z', z_lines)

    ncells = len(x_lines) * len(y_lines) * len(z_lines)
    info = {
        "nx": len(x_lines), "ny": len(y_lines), "nz": len(z_lines),
        "ncells": ncells, "fine_feed_mm": fine_feed, "fine_patch_mm": fine_patch,
        "sub_cells": sub_cells, "port_x_mm": port_x, "port_y_mm": port_y,
        "x_conn_mm": x_conn, "feed_x_start_mm": feed_x_start,
    }
    return FDTD, port, info


def run_one(y0_mm):
    """Run one inset depth; return dict with S11(f) and null metrics."""
    sim_path = os.path.join(SCRIPT_DIR, f"inset_tmp_y{int(round(y0_mm))}")
    FDTD, port, info = build_inset_patch(y0_mm, sim_path)
    print(f"  y0={y0_mm:.1f} mm  mesh {info['nx']}x{info['ny']}x{info['nz']} "
          f"= {info['ncells']:,} cells; running (threads={NUM_THREADS})...", flush=True)
    t0 = time.time()
    FDTD.Run(sim_path, verbose=0, cleanup=True, numThreads=NUM_THREADS)
    runtime_s = time.time() - t0

    port.CalcPort(sim_path, FREQS_HZ)
    s11 = np.asarray(port.uf_ref) / np.asarray(port.uf_inc)
    s11_dB = 20 * np.log10(np.maximum(np.abs(s11), 1e-9))

    # Null near the analytic resonance (+/- 12 %) and global minimum.
    lo = int(np.searchsorted(FREQS_HZ, f_resonance_an * 0.88))
    hi = int(np.searchsorted(FREQS_HZ, f_resonance_an * 1.12))
    idx = lo + int(np.argmin(s11_dB[lo:hi]))
    idx_g = int(np.argmin(s11_dB))
    res = {
        "y0_mm": float(y0_mm), "runtime_s": runtime_s,
        "mesh": info,
        "s11_dip_hz": float(FREQS_HZ[idx]), "s11_dip_db": float(s11_dB[idx]),
        "s11_dip_global_hz": float(FREQS_HZ[idx_g]),
        "s11_dip_global_db": float(s11_dB[idx_g]),
        "s11_max_abs": float(np.max(np.abs(s11))),
        "freqs_hz": [float(v) for v in FREQS_HZ],
        "s11": [[float(v.real), float(v.imag)] for v in s11],
    }
    print(f"    done {runtime_s:.1f}s  null near-res: |S11|={res['s11_dip_db']:.2f} dB "
          f"@ {res['s11_dip_hz']/1e9:.4f} GHz  (global {res['s11_dip_global_db']:.2f} dB "
          f"@ {res['s11_dip_global_hz']/1e9:.4f} GHz)", flush=True)
    return res


def main():
    print("=" * 74)
    print("openEMS inset-fed microstrip patch — y0 matching sweep (cv05 FR4)")
    print("=" * 74)
    print(f"Substrate: eps_r={eps_r}, tan_delta={tan_delta}, h={h_sub*1e3:.1f} mm")
    print(f"Patch: W={W*1e3:.1f} x L={L*1e3:.1f} mm; ground {gx*1e3:.0f}x{gy*1e3:.0f} mm")
    print(f"50 ohm feed width: {W_FEED*1e3:.3f} mm (Hammerstad); notch gap "
          f"{NOTCH_GAP*1e3:.2f} mm; feed line {FEED_LEN*1e3:.1f} mm")
    print(f"Analytic resonance (Balanis TL): {f_resonance_an/1e9:.4f} GHz")
    print(f"R_edge estimate (single-slot): {R_EDGE_EST:.0f} ohm  -> "
          f"Balanis y0 = {Y0_BALANIS*1e3:.2f} mm")
    print(f"Sweeping y0 = {Y0_SWEEP_MM} mm")
    print()

    _numpy_compat_shim()

    sweep = []
    for y0_mm in Y0_SWEEP_MM:
        sweep.append(run_one(y0_mm))

    # Best matched = deepest near-resonance null.
    best = min(sweep, key=lambda r: r["s11_dip_db"])

    print()
    print("  y0 sweep  |  |S11| null depth (near resonance)  |  null freq")
    print("  " + "-" * 64)
    for r in sweep:
        flag = "  <== deepest" if r is best else ""
        print(f"  y0 = {r['y0_mm']:5.1f} mm  |  {r['s11_dip_db']:8.2f} dB  |  "
              f"{r['s11_dip_hz']/1e9:.4f} GHz{flag}")
    print()
    reached = best["s11_dip_db"] < -15.0
    print(f"  BEST: y0 = {best['y0_mm']:.1f} mm, |S11| dip = {best['s11_dip_db']:.2f} dB "
          f"@ {best['s11_dip_hz']/1e9:.4f} GHz")
    print(f"  Reached < -15 dB match: {'YES' if reached else 'NO'}")
    print()

    # --- Save full sweep ---
    sweep_payload = {
        "solver": "openems",
        "feed": "inset_microstrip",
        "geometry": {
            "eps_r": eps_r, "tan_delta": tan_delta, "h_sub_mm": h_sub * 1e3,
            "patch_W_mm": W * 1e3, "patch_L_mm": L * 1e3,
            "ground_x_mm": gx * 1e3, "ground_y_mm": gy * 1e3,
            "feed_width_mm": W_FEED * 1e3, "notch_gap_mm": NOTCH_GAP * 1e3,
            "feed_len_mm": FEED_LEN * 1e3,
        },
        "analytic_resonance_hz": float(f_resonance_an),
        "r_edge_estimate_ohm": float(R_EDGE_EST),
        "y0_balanis_mm": float(Y0_BALANIS * 1e3),
        "reached_match_-15dB": bool(reached),
        "best_y0_mm": float(best["y0_mm"]),
        "best_s11_dip_db": float(best["s11_dip_db"]),
        "best_s11_dip_hz": float(best["s11_dip_hz"]),
        "sweep": [
            {k: v for k, v in r.items() if k not in ("s11", "freqs_hz")}
            for r in sweep
        ],
    }
    p1 = os.path.join(OUT_DIR, "openems_patch_inset_sweep.json")
    with open(p1, "w") as f:
        json.dump(sweep_payload, f, indent=2)
        f.write("\n")
    print(f"  wrote {p1}")

    # --- Save best matched S11(f) (full complex + z11) ---
    s11_best = np.array([complex(a, b) for a, b in best["s11"]])
    z11 = Z0_PORT * (1 + s11_best) / (1 - s11_best)
    matched_payload = {
        "solver": "openems",
        "feed": "inset_microstrip",
        "y0_mm": float(best["y0_mm"]),
        "s11_dip_db": float(best["s11_dip_db"]),
        "s11_dip_hz": float(best["s11_dip_hz"]),
        "s11_dip_global_db": float(best["s11_dip_global_db"]),
        "s11_dip_global_hz": float(best["s11_dip_global_hz"]),
        "s11_max_abs": float(best["s11_max_abs"]),
        "reached_match_-15dB": bool(reached),
        "analytic_resonance_hz": float(f_resonance_an),
        "runtime_s": float(best["runtime_s"]),
        "mesh": best["mesh"],
        "geometry": sweep_payload["geometry"],
        "freqs_hz": [float(v) for v in FREQS_HZ],
        "s11": best["s11"],
        "z11": [[float(v.real), float(v.imag)] for v in z11],
    }
    p2 = os.path.join(OUT_DIR, "openems_patch_matched.json")
    with open(p2, "w") as f:
        json.dump(matched_payload, f, indent=2)
        f.write("\n")
    print(f"  wrote {p2}")

    # --- Save the matched geometry spec (for replication in rfx) ---
    geom = {
        "description": "Balanis inset-fed microstrip patch, cv05 2.4 GHz FR4, "
                       "matched via openEMS y0 sweep. Coordinates centered; L "
                       "along x (resonant), W along y. Patch metal at z=h_sub, "
                       "ground at z=0. Feed enters radiating edge x=patch_x_lo, "
                       "penetrates inset depth y0, connected at x=patch_x_lo+y0 "
                       "with two notch slots (notch_gap wide, y0 long) flanking "
                       "the feed strip. Port = 50 ohm vertical lumped port at the "
                       "outer end of the microstrip line.",
        "eps_r": eps_r,
        "tan_delta": tan_delta,
        "h_sub_mm": h_sub * 1e3,
        "patch_W_mm": W * 1e3,
        "patch_L_mm": L * 1e3,
        "ground_x_mm": gx * 1e3,
        "ground_y_mm": gy * 1e3,
        "feed_width_mm": W_FEED * 1e3,
        "notch_gap_mm": NOTCH_GAP * 1e3,
        "feed_len_mm": FEED_LEN * 1e3,
        "inset_depth_y0_mm": float(best["y0_mm"]),
        "port_impedance_ohm": Z0_PORT,
        "matched_s11_dip_db": float(best["s11_dip_db"]),
        "matched_s11_dip_hz": float(best["s11_dip_hz"]),
        "reached_match_-15dB": bool(reached),
        "provenance": {
            "solver": "openems",
            "y0_sweep_mm": Y0_SWEEP_MM,
            "r_edge_estimate_ohm": float(R_EDGE_EST),
            "y0_balanis_mm": float(Y0_BALANIS * 1e3),
        },
    }
    p3 = os.path.join(SCRIPT_DIR, "matched_patch_geometry.json")
    with open(p3, "w") as f:
        json.dump(geom, f, indent=2)
        f.write("\n")
    print(f"  wrote {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

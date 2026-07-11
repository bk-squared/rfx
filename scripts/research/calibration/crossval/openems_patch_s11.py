"""Standalone high-resolution openEMS reference: FULL complex S11(f) for the
cv05 2.4 GHz FR4 probe-fed rectangular patch antenna.

This is the INDEPENDENT reference leg of the 3-way S11-trend cross-validation
(rfx GPU mesh/port sweeps + openEMS + Balanis analytic). It reuses the exact
geometry, lumped-port and S11/Harminv extraction of
``examples/crossval/05_patch_antenna.py`` PART 2, but:

  * refines the mesh to a FINE resolution near the patch (thirds-rule edge
    lines + ~0.3 mm cells across the patch bounding box, comparable to rfx's
    dx_fine ~ 0.25 mm) and 6 cells across the 1.5 mm FR4 substrate;
  * extracts the FULL complex S11(f) on the SAME 1.5-3.5 GHz / 101-pt grid
    that the rfx convergence sweep writes, so the two curves are directly
    subtractable;
  * computes Z11(f) = Z0 (1+S11)/(1-S11) and the Harminv resonance from the
    port V(t) ring-down;
  * emits scripts/research/calibration/crossval/out_ref/openems_patch_s11.json.

Run (system python — openEMS/CSXCAD bindings live there):
  python scripts/research/calibration/crossval/openems_patch_s11.py
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
# Geometry — copied verbatim from examples/crossval/05_patch_antenna.py
# =============================================================================
f_design = 2.4e9
eps_r = 4.3
tan_delta = 0.02              # cv05 FR4 loss tangent
h_sub = 1.5e-3
W = 38.0e-3
L = 29.5e-3
gx = 60.0e-3
gy = 55.0e-3
probe_inset = 8.0e-3

# Analytic (Balanis Ch. 14) reference resonance
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_sub / W) ** (-0.5)
delta_L = 0.412 * h_sub * ((eps_eff + 0.3) * (W / h_sub + 0.264)) / \
          ((eps_eff - 0.258) * (W / h_sub + 0.8))
L_eff = L + 2 * delta_L
f_resonance_an = C0 / (2 * L_eff * math.sqrt(eps_eff))

# Shared S11 frequency grid — IDENTICAL to rfx_patch_s11_convergence.py so the
# two S11(f) curves live on the same abscissa and |dS11| is a direct diff.
FREQS_HZ = np.linspace(1.5e9, 3.5e9, 101)


def main():
    print("=" * 72)
    print("openEMS high-res reference — 2.4 GHz FR4 patch S11(f)")
    print("=" * 72)
    print(f"Substrate: eps_r={eps_r}, tan_delta={tan_delta}, h={h_sub*1e3:.1f} mm (FR4)")
    print(f"Patch: W={W*1e3:.1f} x L={L*1e3:.1f} mm; ground {gx*1e3:.0f}x{gy*1e3:.0f} mm; "
          f"inset {probe_inset*1e3:.1f} mm")
    print(f"Analytic (Balanis TL): eps_eff={eps_eff:.3f}, f_res={f_resonance_an/1e9:.4f} GHz")
    print()

    # numpy compat shim for openEMS v0.0.35 (expects np.float etc)
    for _n in ("float", "int", "complex"):
        if not hasattr(np, _n):
            setattr(np, _n, {"float": float, "int": int, "complex": complex}[_n])

    from CSXCAD.CSXCAD import ContinuousStructure
    from CSXCAD.SmoothMeshLines import SmoothMeshLines
    from openEMS.openEMS import openEMS as OEMS

    UNIT = 1e-3  # mm
    sim_path_oe = os.path.join(SCRIPT_DIR, "openems_hires_tmp")

    f0_hz = f_design
    # widen the Gaussian half-bandwidth so the source covers the full
    # 1.5-3.5 GHz comparison band (2.4 +/- 1.2 = 1.2-3.6 GHz).
    fc_hz = 1.2e9

    # Geometry in mm
    L_mm = L * 1000
    W_mm = W * 1000
    h_sub_mm = h_sub * 1000
    gx_mm = gx * 1000
    gy_mm = gy * 1000
    inset_mm = probe_inset * 1000

    # Domain: ground plane + >= lambda/2 air margin (PML) + radiation air above.
    margin_mm = 50.0
    air_above_mm = 40.0
    dom_x_mm = gx_mm + 2 * margin_mm
    dom_y_mm = gy_mm + 2 * margin_mm
    dom_z_mm = h_sub_mm + air_above_mm
    x_c = dom_x_mm / 2
    y_c = dom_y_mm / 2

    # FDTD solver. NrTS caps the ring-down; loose EndCriteria so near-resonance
    # energy oscillation does not truncate the run early.
    # (OPENEMS_NRTS env override is for fast port-excitation smoke tests.)
    nrts = int(os.environ.get("OPENEMS_NRTS", "30000"))
    FDTD = OEMS(NrTS=nrts, EndCriteria=1e-6)
    FDTD.SetGaussExcite(f0_hz, fc_hz)
    FDTD.SetBoundaryCond(['PML_8'] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh_oe = CSX.GetGrid()
    mesh_oe.SetDeltaUnit(UNIT)

    # FR4 substrate (with loss tangent, matching cv05 convergence sigma model)
    sub_mat = CSX.AddMaterial('FR4')
    sub_mat.SetMaterialProperty(epsilon=eps_r, kappa=2 * math.pi * f0_hz * 8.8541878128e-12 * eps_r * tan_delta)
    sub_lo = [x_c - gx_mm / 2, y_c - gy_mm / 2, 0]
    sub_hi = [x_c + gx_mm / 2, y_c + gy_mm / 2, h_sub_mm]
    sub_mat.AddBox(sub_lo, sub_hi, priority=1)

    # Ground plane (2D PEC at z=0)
    gnd = CSX.AddMetal('gnd')
    gnd.AddBox([sub_lo[0], sub_lo[1], 0],
               [sub_hi[0], sub_hi[1], 0], priority=10)

    # Patch (2D PEC at z=h_sub)
    patch_lo_oe = [x_c - L_mm / 2, y_c - W_mm / 2, h_sub_mm]
    patch_hi_oe = [x_c + L_mm / 2, y_c + W_mm / 2, h_sub_mm]
    patch = CSX.AddMetal('patch')
    patch.AddBox(patch_lo_oe, patch_hi_oe, priority=10)

    # --- FINE mesh: ~0.6 mm cells across the patch bounding box ---
    # (comparable to rfx's dx_fine near the patch). The 0.25 mm z substrate cell
    # sets the Courant dt, so in-plane cells stay >= 0.5 mm to avoid shrinking dt
    # and blowing up the step count. A dense fine block spans the patch; the
    # air/PML region is filled coarse (<= coarse_mm) with graded transitions.
    fine_mm = 0.6
    x_patch = np.arange(patch_lo_oe[0], patch_hi_oe[0] + fine_mm / 2, fine_mm)
    y_patch = np.arange(patch_lo_oe[1], patch_hi_oe[1] + fine_mm / 2, fine_mm)

    x_fixed = np.concatenate([[0.0, dom_x_mm, sub_lo[0], sub_hi[0]], x_patch])
    y_fixed = np.concatenate([[0.0, dom_y_mm, sub_lo[1], sub_hi[1]], y_patch])
    # z: resolve the 1.5 mm substrate with 6 cells (0.25 mm), coarse air above
    sub_cells = 6
    z_fixed = np.concatenate([
        [0.0, dom_z_mm, h_sub_mm],
        np.linspace(0, h_sub_mm, sub_cells + 1),
    ])

    # SmoothMeshLines keeps the fixed lines and fills the coarse regions up to
    # `coarse_mm`, grading with ratio<=1.4. Fine near patch, coarse in air/PML.
    coarse_mm = 5.0
    x_lines = SmoothMeshLines(np.unique(np.round(x_fixed, 4)), coarse_mm, ratio=1.4)
    y_lines = SmoothMeshLines(np.unique(np.round(y_fixed, 4)), coarse_mm, ratio=1.4)
    z_lines = SmoothMeshLines(np.unique(np.round(z_fixed, 4)), coarse_mm, ratio=1.4)

    # Snap the feed to the FINAL mesh lines so the lumped-port excitation box
    # coincides EXACTLY with a mesh edge. If it does not, openEMS drops the
    # excitation ("Unused primitive ... port_excite_1"), the port never radiates
    # and every S11 sample is 0/0 = NaN. (This is why the feed must be snapped
    # AFTER SmoothMeshLines + rounding, not to the pre-rounded grid.)
    feed_x_mm = float(x_lines[np.argmin(np.abs(x_lines - (patch_lo_oe[0] + inset_mm)))])
    feed_y_mm = float(y_lines[np.argmin(np.abs(y_lines - y_c))])

    # Lumped 50 ohm port: vertical, ground-to-patch at the (snapped) feed inset
    port = FDTD.AddLumpedPort(
        port_nr=1, R=Z0_PORT,
        start=[feed_x_mm, feed_y_mm, 0.0],
        stop=[feed_x_mm, feed_y_mm, h_sub_mm],
        p_dir='z', excite=1.0,
    )

    mesh_oe.SetLines('x', x_lines)
    mesh_oe.SetLines('y', y_lines)
    mesh_oe.SetLines('z', z_lines)

    ncells = len(x_lines) * len(y_lines) * len(z_lines)
    print(f"openEMS mesh: {len(x_lines)} x {len(y_lines)} x {len(z_lines)} "
          f"= {ncells:,} cells")
    print(f"  fine ~{fine_mm} mm across patch, {sub_cells} cells across substrate, "
          f"coarse <= {coarse_mm} mm in air/PML")

    print(f"  Running openEMS (GaussExcite f0={f0_hz/1e9:.2f} GHz, "
          f"fc={fc_hz/1e9:.2f} GHz, NrTS<=30000)...")
    t0 = time.time()
    FDTD.Run(sim_path_oe, verbose=0, cleanup=True, numThreads=16)
    runtime_s = time.time() - t0
    print(f"  done in {runtime_s:.1f}s")

    # --- Post-process: FULL complex S11 on the shared 1.5-3.5 GHz grid ---
    port.CalcPort(sim_path_oe, FREQS_HZ)
    s11 = np.asarray(port.uf_ref) / np.asarray(port.uf_inc)
    s11_dB = 20 * np.log10(np.maximum(np.abs(s11), 1e-9))
    z11 = Z0_PORT * (1 + s11) / (1 - s11)

    # S11 dip near analytic resonance (local, +/-10 %)
    lo = int(np.searchsorted(FREQS_HZ, f_resonance_an * 0.90))
    hi = int(np.searchsorted(FREQS_HZ, f_resonance_an * 1.10))
    idx_min = lo + int(np.argmin(s11_dB[lo:hi]))
    f_dip = float(FREQS_HZ[idx_min])
    s11_dip_dB = float(s11_dB[idx_min])
    # global minimum too (informative — the true antenna match dip)
    idx_gmin = int(np.argmin(s11_dB))
    f_dip_global = float(FREQS_HZ[idx_gmin])
    s11_dip_global_dB = float(s11_dB[idx_gmin])

    # --- Harminv on the openEMS port voltage time series ---
    from rfx.harminv import harminv

    def _read_probe(fname):
        return np.loadtxt(fname, comments="%")

    _ut = _read_probe(os.path.join(sim_path_oe, "port_ut_1"))
    t_oe = _ut[:, 0]
    ut_oe = _ut[:, 1]
    dt_oe = float(t_oe[1] - t_oe[0])
    _skip = int(0.2 * len(ut_oe))
    modes = harminv(ut_oe[_skip:], dt_oe, 1.5e9, 3.5e9)
    modes_good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-8]
    if modes_good:
        modes_good.sort(key=lambda m: abs(m.freq - f_resonance_an))
        f_harminv = float(modes_good[0].freq)
        Q_harminv = float(modes_good[0].Q)
    else:
        f_harminv = float("nan")
        Q_harminv = float("nan")

    an_err = 100 * abs(f_harminv - f_resonance_an) / f_resonance_an if not np.isnan(f_harminv) else float("nan")

    print()
    print(f"  openEMS Harminv (port V(t)) modes (Q>2) near target:")
    for m in sorted(modes_good, key=lambda m: m.freq)[:6]:
        print(f"    f = {m.freq/1e9:.4f} GHz, Q = {m.Q:.1f}, amp = {m.amplitude:.2e}")
    print()
    print(f"  openEMS Harminv:    f = {f_harminv/1e9:.4f} GHz, Q = {Q_harminv:.1f}")
    print(f"  openEMS S11 dip (near target): f = {f_dip/1e9:.4f} GHz, "
          f"|S11| = {s11_dip_dB:.2f} dB")
    print(f"  openEMS S11 dip (global):      f = {f_dip_global/1e9:.4f} GHz, "
          f"|S11| = {s11_dip_global_dB:.2f} dB")
    print(f"  Analytic target:    f = {f_resonance_an/1e9:.4f} GHz")
    print(f"  openEMS Harminv vs analytic: {an_err:.2f} %")
    print(f"  |S11| passive (<=1): max|S11| = {np.max(np.abs(s11)):.4f}")

    payload = {
        "solver": "openems",
        "mesh": {
            "nx": len(x_lines), "ny": len(y_lines), "nz": len(z_lines),
            "ncells": ncells, "fine_mm": fine_mm, "sub_cells": sub_cells,
            "coarse_mm": coarse_mm,
        },
        "runtime_s": runtime_s,
        "dt_s": dt_oe,
        "n_time_samples": int(len(ut_oe)),
        "analytic_resonance_hz": float(f_resonance_an),
        "harminv_hz": f_harminv,
        "harminv_Q": Q_harminv,
        "harminv_vs_analytic_pct": float(an_err),
        "s11_dip_hz": f_dip,
        "s11_dip_db": s11_dip_dB,
        "s11_dip_global_hz": f_dip_global,
        "s11_dip_global_db": s11_dip_global_dB,
        "s11_max_abs": float(np.max(np.abs(s11))),
        "s11_passive": bool(np.all(np.abs(s11) < 1.05)),
        "freqs_hz": [float(v) for v in FREQS_HZ],
        "s11": [[float(v.real), float(v.imag)] for v in s11],
        "z11": [[float(v.real), float(v.imag)] for v in z11],
    }
    out_path = os.path.join(OUT_DIR, "openems_patch_s11.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

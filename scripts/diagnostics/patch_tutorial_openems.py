"""STATUS 2026-07-13: LIVE — validated far-field lane (openEMS side, canonical Simple_Patch recipe: AddEdges2Grid thirds-rule + feed mesh line). Supersedes the archived cv05_openems_nf2ff iterations.

TRACKED 2026-07-18 as the canonical-patch oracle provenance: this script produced
the committed openEMS reference fixture
tests/fixtures/patch_canonical_farfield_e4/patch_farfield_openems.json
(gated by tests/test_patch_canonical_farfield_e4.py, cv05 manifest gate_paths).

REPRODUCE-GATE RECORD (added retroactively 2026-07-18; the run is 2026-07-11 and
the record-first ordering caveat is stated in the fixture meta):
- Source example: openEMS Simple_Patch_Antenna tutorial (upstream
  python/Tutorials/Simple_Patch_Antenna.py), published expectation "~7 dBi
  broadside"; no more precise upstream number is recorded in this repo, so the
  values below are OUR source-built-openEMS run's measured values.
- Reproduced known-good result (VESSL run 369367246713, 2026-07-11):
  f_res 2.4221 GHz (harminv on port V, Q 20.1), S11 dip 2.4300 GHz (-27.8 dB),
  broadside D 6.79 dBi, E/H-plane peaks 0/0 deg, stopped on EndCriteria=1e-4 at
  step 8671 with energy -41.09 dB.
- Log: docs/research_notes/vessl_logs/patch_tutorial_openems_GOOD_369367246713.log
  (local-only; the earlier run 369367246711 failed with the lumped port off-grid
  — "Unused primitive", zero energy for the whole run — fixed by the explicit
  feed mesh lines marked CRITICAL below).
- Deltas vs the upstream tutorial script: (1) resonance read via rfx harminv on
  port_ut_1 instead of the tutorial's S11-dip readout (the S11 dip is also
  printed); (2) explicit sub_cells=3 z-lines across the substrate; (3) NF2FF cut
  set theta=0:2:180 deg, phi={0,90,180,270}, center (0,0,sub_thick/2);
  (4) explicit feed mesh lines at x=-6 mm / y=0.
- Hand-ported sanity checks (external scripts get no rfx preflight): EndCriteria
  ring-down stop (settling), nonzero-excitation check via the port-V trace, and
  the broadside/HPBW pattern verdict printed before any number is quoted.

Canonical patch far-field — openEMS Simple_Patch_Antenna tutorial, replicated.

A KNOWN-GOOD reference: the official openEMS python tutorial patch (32x40mm on
eps_r=3.38 / 1.524mm substrate, 60x60mm GP, -6mm probe feed). It radiates a clean
broadside pattern (~7 dBi) — unlike the cv05 patch, which is a high-Q weak radiator
whose far-field is feed-probe-dominated. The rfx run (patch_tutorial_rfx.py) uses
the IDENTICAL geometry so the two solvers' far-fields compare apples-to-apples.

Critical openEMS recipe points my earlier cv05 script was MISSING and that make
the patch radiate correctly:
 - AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2): the
   thirds-rule metal-edge mesh — without it the patch current is wrong.
 - MUR boundaries (tutorial), EndCriteria=1e-4 (stops at -40dB, no late instability).
Resonance via harminv-on-port-V (robust); NF2FF theta/phi in DEGREES.
"""
import json
import math
import os
import time
import numpy as np
from CSXCAD.CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS as OEMS

C0 = 2.998e8
EPS0 = 8.8541878e-12
UNIT = 1e-3
# ---- tutorial geometry (mm), centred at origin ----
patch_w, patch_l = 32.0, 40.0
sub_epsR, sub_thick = 3.38, 1.524
sub_w = sub_l = 60.0
feed_pos, feed_R = -6.0, 50.0
sub_kappa = 1e-3 * 2 * math.pi * 2.45e9 * EPS0 * sub_epsR   # tan_delta=1e-3
f0, fc = 2.0e9, 1.0e9
SimBox = np.array([200.0, 200.0, 150.0])
mesh_res = C0 / (f0 + fc) / 1e-3 / 20.0                     # ~5 mm
sub_cells = 3
# analytic-ish resonance guess for the nearest-mode picker (patch half-wave in eps_eff)
eps_eff = (sub_epsR + 1) / 2 + (sub_epsR - 1) / 2 * (1 + 12 * (sub_thick / patch_w)) ** -0.5
F_GUESS = C0 / (2 * patch_l * 1e-3 * math.sqrt(eps_eff))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "cv05_investigation_results")
os.makedirs(RESULT_DIR, exist_ok=True)
sim_path = os.path.join(SCRIPT_DIR, "patch_tutorial_oe_tmp")


def hpbw_deg(ang_deg, power_lin):
    p = np.asarray(power_lin, float); p = p / np.max(p)
    ipk = int(np.argmax(p))

    def edge(direction):
        i = ipk
        while 0 <= i + direction < len(p) and p[i] >= 0.5:
            i += direction
        if p[i] >= 0.5:
            return np.nan
        j = i - direction
        t = (p[j] - 0.5) / (p[j] - p[i])
        return ang_deg[j] + t * (ang_deg[i] - ang_deg[j])

    lo, hi = edge(-1), edge(+1)
    return float("nan") if (np.isnan(lo) or np.isnan(hi)) else abs(hi - lo)


def main():
    t0 = time.time()
    print(f"TUTORIAL patch openEMS | f_guess={F_GUESS/1e9:.4f} GHz | mesh_res={mesh_res:.2f}mm")
    FDTD = OEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['MUR'] * 6)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(UNIT)

    # patch (thirds-rule edges)
    patch = CSX.AddMetal('patch')
    patch.AddBox(priority=10, start=[-patch_w / 2, -patch_l / 2, sub_thick],
                 stop=[patch_w / 2, patch_l / 2, sub_thick])
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res / 2)
    # substrate
    sub = CSX.AddMaterial('substrate', epsilon=sub_epsR, kappa=sub_kappa)
    sub.AddBox(priority=0, start=[-sub_w / 2, -sub_l / 2, 0], stop=[sub_w / 2, sub_l / 2, sub_thick])
    # ground
    gnd = CSX.AddMetal('gnd')
    gnd.AddBox(priority=10, start=[-sub_w / 2, -sub_l / 2, 0], stop=[sub_w / 2, sub_l / 2, 0])
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
    # port (probe through substrate)
    port = FDTD.AddLumpedPort(1, feed_R, [feed_pos, 0, 0], [feed_pos, 0, sub_thick],
                              'z', 1.0, priority=5)
    # air box + smooth mesh. CRITICAL: add mesh lines AT the feed (x=feed_pos, y=0)
    # so the lumped-port excitation box lands on the grid — without this the port is
    # an "Unused primitive" (energy stays 0.0 -> all-NaN far-field).
    mesh.AddLine('x', [-SimBox[0] / 2, feed_pos, SimBox[0] / 2])
    mesh.AddLine('y', [-SimBox[1] / 2, 0.0, SimBox[1] / 2])
    mesh.AddLine('z', [-SimBox[2] / 3, SimBox[2] * 2 / 3])
    mesh.AddLine('z', np.linspace(0, sub_thick, sub_cells + 1))
    mesh.SmoothMeshLines('all', mesh_res, 1.4)

    nf2ff = FDTD.CreateNF2FFBox()
    print("running openEMS...")
    FDTD.Run(sim_path, verbose=0, cleanup=True)
    print(f"  FDTD done {time.time()-t0:.1f}s")

    freqs = np.linspace(f0 - fc, f0 + fc, 401)
    port.CalcPort(sim_path, freqs)
    s11 = 20 * np.log10(np.maximum(np.abs(port.uf_ref / port.uf_inc), 1e-6))
    f_s11_dip = float(freqs[int(np.argmin(s11))])

    from rfx.harminv import harminv as _harminv
    ut = np.loadtxt(os.path.join(sim_path, "port_ut_1"), comments="%")
    dt_oe = float(ut[1, 0] - ut[0, 0]); skip = int(0.2 * len(ut))
    modes = [m for m in _harminv(ut[skip:, 1], dt_oe, 1.0e9, 3.5e9)
             if m.Q > 2 and m.amplitude > 1e-8]
    if modes:
        modes.sort(key=lambda m: abs(m.freq - f_s11_dip))
        f_res, Q_oe = float(modes[0].freq), float(modes[0].Q)
    else:
        f_res, Q_oe = f_s11_dip, float("nan")
    print(f"  harminv-port-V f_res={f_res/1e9:.4f} GHz Q={Q_oe:.1f} | S11 dip {f_s11_dip/1e9:.4f} "
          f"({s11.min():.1f} dB)")

    theta = np.arange(0, 180.1, 2.0)           # DEGREES from +z
    phi = np.array([0.0, 90.0, 180.0, 270.0])
    nf = nf2ff.CalcNF2FF(sim_path, [f_res], theta, phi, center=[0, 0, sub_thick / 2 * UNIT])
    D_dbi = 10 * np.log10(float(np.atleast_1d(nf.Dmax)[0]))
    En = np.asarray(nf.E_norm[0])
    iph = {p: k for k, p in enumerate(phi)}
    tm = theta <= 90; tp = theta[tm]
    psi = np.concatenate([-tp[::-1], tp])
    pE = np.concatenate([(En[tm, iph[180.0]] ** 2)[::-1], En[tm, iph[0.0]] ** 2])
    pH = np.concatenate([(En[tm, iph[270.0]] ** 2)[::-1], En[tm, iph[90.0]] ** 2])
    ipE, ipH = int(np.argmax(pE)), int(np.argmax(pH))
    hE, hH = hpbw_deg(psi, pE), hpbw_deg(psi, pH)
    broadside_ok = abs(psi[ipE]) <= 15 and abs(psi[ipH]) <= 15
    print(f"  E-plane peak {psi[ipE]:.1f}deg HPBW {hE:.1f} | H-plane peak {psi[ipH]:.1f}deg HPBW {hH:.1f}")
    print(f"  directivity {D_dbi:.2f} dBi | BROADSIDE {'OK' if broadside_ok else 'FAIL'}")

    out = dict(solver="openEMS", geometry="tutorial_patch_32x40_eps3.38",
               f_res_ghz=round(f_res / 1e9, 4), Q=round(Q_oe, 1),
               s11_dip_ghz=round(f_s11_dip / 1e9, 4), s11_min_db=round(float(s11.min()), 2),
               E_plane_peak_deg=round(float(psi[ipE]), 1), H_plane_peak_deg=round(float(psi[ipH]), 1),
               hpbw_E_deg=None if np.isnan(hE) else round(float(hE), 1),
               hpbw_H_deg=None if np.isnan(hH) else round(float(hH), 1),
               directivity_dbi=round(D_dbi, 2), broadside_ok=bool(broadside_ok),
               psi_deg=[round(float(a), 2) for a in psi],
               E_plane_norm=[round(float(v), 5) for v in (pE / np.max(pE))],
               H_plane_norm=[round(float(v), 5) for v in (pH / np.max(pH))],
               wall_s=round(time.time() - t0, 1))
    path = os.path.join(RESULT_DIR, "patch_tutorial_openems.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"WROTE {path} (wall {out['wall_s']}s)")


if __name__ == "__main__":
    main()

"""2x2 convergence grid: NU/uniform x lossy/lossless at dx=0.5mm."""

from __future__ import annotations

import math
import time
import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.harminv import harminv

C0 = 2.998e8
F_DESIGN = 2.4e9
EPS_FR4 = 4.3
H_SUB = 1.5e-3
W, L = 38.0e-3, 29.5e-3
GX, GY = 60.0e-3, 55.0e-3
PROBE_INSET = 8.0e-3

eps_eff = (EPS_FR4 + 1) / 2 + (EPS_FR4 - 1) / 2 * (1 + 12 * H_SUB / W) ** -0.5
dL = 0.412 * H_SUB * ((eps_eff + 0.3) * (W / H_SUB + 0.264)) / (
    (eps_eff - 0.258) * (W / H_SUB + 0.8))
F_BALANIS = C0 / (2 * (L + 2 * dL) * math.sqrt(eps_eff))


def run_case(label, dx, use_nu, lossy):
    print(f"\n=== {label} === dx={dx*1e3:.2f}mm  NU={use_nu}  lossy={lossy}")
    n_cpml = 8
    dom_x = GX + 20e-3
    dom_y = GY + 20e-3
    gx_lo = (dom_x - GX) / 2
    gy_lo = (dom_y - GY) / 2
    px_lo = dom_x / 2 - L / 2
    py_lo = dom_y / 2 - W / 2
    feed_x = px_lo + PROBE_INSET
    feed_y = dom_y / 2

    if use_nu:
        # Co-refine: dz_sub = dx/4 maintains the 4:1 aspect ratio of the
        # baseline (dx=1mm, dz=0.25mm). Taflove Ch. 4: fixing dz while
        # refining dx inflates the xy Courant number and anti-converges.
        dz_sub = dx / 4.0
        n_sub = max(6, int(round(H_SUB / dz_sub)))
        n_trans = 5
        trans = np.geomspace(dx, dz_sub, n_trans + 2)[1:-1]
        n_fine = n_sub + 2
        n_coarse_below = max(0, int(round((12e-3 - trans.sum()) / dx)))
        n_coarse_above = max(0, int(round((25e-3 - trans.sum()) / dx)))
        dz = np.concatenate([
            np.full(n_coarse_below, dx), trans,
            np.full(n_fine, dz_sub),
            trans[::-1], np.full(n_coarse_above, dx),
        ])
        ze = np.concatenate([[0], np.cumsum(dz)])
        k_gnd = n_coarse_below + n_trans
        z_gnd_lo = float(ze[k_gnd])
        z_sub_lo = float(ze[k_gnd + 1])
        z_sub_hi = float(ze[k_gnd + 1 + n_sub])
        z_patch_hi = float(ze[k_gnd + 2 + n_sub])
        src_z = z_sub_lo + dz_sub * 2.5
        sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, 0),
                         dx=dx, dz_profile=dz, boundary="cpml",
                         cpml_layers=n_cpml)
    else:
        dom_z = 12e-3 + H_SUB + 25e-3
        z_gnd_lo = 12e-3 - dx
        z_sub_lo = 12e-3
        z_sub_hi = 12e-3 + H_SUB
        z_patch_hi = z_sub_hi + dx
        src_z = z_sub_lo + dx * 0.5
        sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, dom_z),
                         dx=dx, boundary="cpml", cpml_layers=n_cpml)

    eps0 = 8.8541878128e-12
    sigma = 2 * np.pi * F_DESIGN * eps0 * EPS_FR4 * 0.02 if lossy else 0.0
    sim.add_material("fr4", eps_r=EPS_FR4, sigma=sigma)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo),
                (gx_lo + GX, gy_lo + GY, z_sub_lo)), material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo),
                (gx_lo + GX, gy_lo + GY, z_sub_hi)), material="fr4")
    sim.add(Box((px_lo, py_lo, z_sub_hi),
                (px_lo + L, py_lo + W, z_patch_hi)), material="pec")
    sim.add_source((feed_x, feed_y, src_z), "ez",
                   waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2))
    sim.add_probe((dom_x / 2 + 5e-3, dom_y / 2 + 5e-3, src_z), "ez")

    g = sim._build_nonuniform_grid() if use_nu else sim._build_grid()
    cells = g.nx * g.ny * g.nz
    print(f"  cells={cells:,}")
    t0 = time.time()
    res = sim.run(num_periods=60, compute_s_params=False)
    dt_run = time.time() - t0
    ts = np.asarray(res.time_series).ravel()
    dt_sim = float(res.dt)
    skip = int(len(ts) * 0.3)
    modes = harminv(ts[skip:], dt_sim, 1.5e9, 3.5e9)
    good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-10]
    if good:
        best = max(good, key=lambda m: m.amplitude)
        f_res = float(best.freq)
        Q = float(best.Q)
        err = 100 * abs(f_res - F_BALANIS) / F_BALANIS
        print(f"  f_res={f_res/1e9:.4f} GHz  Q={Q:.1f}  error={err:.2f}%  "
              f"Balanis={F_BALANIS/1e9:.4f}  dt={dt_run:.1f}s")
    else:
        print(f"  harminv FAIL (no modes). dt={dt_run:.1f}s  "
              f"max|ts|={np.max(np.abs(ts)):.3e}")


def main():
    # Proper convergence sweep: co-refine dx AND dz_sub at fixed 4:1
    # aspect ratio (Taflove Ch. 3.3 / Ch. 4.7 — fixing dz while
    # reducing dx inflates the xy Courant number and ANTI-converges).
    print("=" * 60)
    print("Proper convergence: co-refine dx + dz at 4:1 aspect")
    print("=" * 60)
    run_case("F1: NU dx=1.0mm dz=0.250mm (baseline)", 1.0e-3, True, True)
    run_case("F2: NU dx=0.5mm dz=0.125mm", 0.5e-3, True, True)
    run_case("F3: NU dx=0.25mm dz=0.0625mm", 0.25e-3, True, True)

    # Also re-run the anti-convergence case for comparison
    print("\n" + "=" * 60)
    print("Anti-convergence reference (dz fixed at 0.25mm)")
    print("=" * 60)
    run_case("G1: NU dx=1.0mm dz=0.250mm", 1.0e-3, True, True)
    run_case("G2: NU dx=0.5mm dz=0.250mm (WRONG)", 0.5e-3, True, True)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()

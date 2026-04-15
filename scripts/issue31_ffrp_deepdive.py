"""Issue #48/49 deep dive: why does patch NTFF peak at θ≈87° (grazing)?

Isolates whether the problem is:
  (A) NU NTFF math bug → uniform should peak at θ≈0
  (B) simulation not exciting patch mode → uniform also peaks at ~90°
  (C) NTFF missing patch PEC surface currents → any antenna with a PEC
      surface will look like a bare dipole

Runs three minimal configurations and reports peak direction for each:

  1. Uniform cavity with no PEC — pure ez dipole in vacuum.
     Expected: peak at θ=90° (dipole signature).
  2. Uniform patch antenna (dx=1mm, dz=0.25mm everywhere).
     Expected: peak at θ≈0° (broadside).
  3. NU patch antenna (same geometry, graded dz).
     Expected: peak at θ≈0° if NU NTFF works correctly.

Configuration (1) is the baseline. (2) vs (3) tells us whether the
problem is NU-specific.
"""

from __future__ import annotations

import math
import time
import numpy as np

from rfx import Simulation, Box
from rfx.auto_config import smooth_grading
from rfx.sources.sources import GaussianPulse
from rfx.farfield import compute_far_field


C0 = 2.998e8
F_DESIGN = 2.4e9


def _peak(ff, theta, phi):
    E_t = np.asarray(ff.E_theta[0]); E_p = np.asarray(ff.E_phi[0])
    mag = np.sqrt(np.abs(E_t) ** 2 + np.abs(E_p) ** 2)
    i_peak, j_peak = np.unravel_index(np.argmax(mag), mag.shape)
    return np.degrees(theta[i_peak]), np.degrees(phi[j_peak]), mag


def run_case(sim, label, *, f_ntff=F_DESIGN, num_periods=40):
    g = (sim._build_nonuniform_grid() if sim._dz_profile is not None
         else sim._build_grid())
    cells = g.nx * g.ny * g.nz
    print(f"\n=== {label} === cells={cells:,}")
    t0 = time.time()
    res = sim.run(num_periods=num_periods, compute_s_params=False)
    print(f"[run] {time.time() - t0:.1f}s")
    if res.ntff_data is None:
        print("[warn] no NTFF box, skipping pattern check")
        return
    theta = np.linspace(0.01, np.pi / 2, 60)
    phi = np.linspace(0, 2 * np.pi, 121)
    ff = compute_far_field(res.ntff_data, res.ntff_box, g, theta, phi)
    th_p, ph_p, mag = _peak(ff, theta, phi)
    mag_n = mag / np.max(mag)
    # Broadside ratio: |E_far| at θ=0 vs peak
    broadside = float(mag_n[np.argmin(np.abs(theta - 0.0)), 0])
    print(f"[peak] θ={th_p:.1f}°  φ={ph_p:.1f}°   "
          f"broadside_ratio={broadside:.3f}   "
          f"(≈1.0 means broadside peak, ≈0 means grazing-only)")
    return th_p, broadside


# ---------------------------------------------------------------------
# Case 1 — Uniform pure ez dipole in vacuum
# ---------------------------------------------------------------------
def case1_dipole():
    sim = Simulation(freq_max=4e9, domain=(0.08, 0.075, 0.040),
                     dx=1e-3, boundary="cpml", cpml_layers=8)
    sim.add_source((0.04, 0.0375, 0.020), "ez",
                   waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2))
    sim.add_ntff_box(corner_lo=(0.010, 0.010, 0.005),
                     corner_hi=(0.070, 0.065, 0.035),
                     freqs=[F_DESIGN])
    return run_case(sim, "Case 1 — uniform, ez dipole (NO patch)")


# ---------------------------------------------------------------------
# Case 2 — Uniform patch antenna
# ---------------------------------------------------------------------
def _patch_geometry():
    eps_r = 4.3
    h_sub = 1.5e-3
    W, L = 38.0e-3, 29.5e-3
    gx, gy = 60.0e-3, 55.0e-3
    air_above, air_below = 25.0e-3, 12.0e-3
    probe_inset = 8.0e-3
    dom_x = gx + 20e-3
    dom_y = gy + 20e-3
    geom = dict(
        eps_r=eps_r, h_sub=h_sub, W=W, L=L, gx=gx, gy=gy,
        air_above=air_above, air_below=air_below,
        dom_x=dom_x, dom_y=dom_y,
        gx_lo=(dom_x - gx) / 2, gy_lo=(dom_y - gy) / 2,
        px_lo=dom_x / 2 - L / 2, py_lo=dom_y / 2 - W / 2,
        feed_x=(dom_x / 2 - L / 2) + probe_inset, feed_y=dom_y / 2,
    )
    return geom


def _add_patch(sim, G, dz_sub, z_gnd_lo, z_sub_lo, z_sub_hi, z_patch_lo,
               z_patch_hi, src_z):
    sim.add_material("fr4", eps_r=G["eps_r"])
    sim.add(Box((G["gx_lo"], G["gy_lo"], z_gnd_lo),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], z_sub_lo)),
            material="pec")
    sim.add(Box((G["gx_lo"], G["gy_lo"], z_sub_lo),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], z_sub_hi)),
            material="fr4")
    sim.add(Box((G["px_lo"], G["py_lo"], z_patch_lo),
                (G["px_lo"] + G["L"], G["py_lo"] + G["W"], z_patch_hi)),
            material="pec")
    sim.add_source(position=(G["feed_x"], G["feed_y"], src_z),
                   component="ez",
                   waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2))


def case2_uniform_patch():
    G = _patch_geometry()
    dx = 1e-3
    dz_sub = G["h_sub"] / 6
    # Use uniform z at dz_sub → ~160 cells in z, but domain height 38.5mm / 0.25mm
    # is 154 cells. Keep dx=1mm in x,y.
    dz_uniform = dz_sub
    dom_z = G["air_below"] + G["h_sub"] + G["air_above"]
    sim = Simulation(freq_max=4e9, domain=(G["dom_x"], G["dom_y"], dom_z),
                     dx=dx, boundary="cpml", cpml_layers=8)
    # With no dz_profile, z uses dx (not dz_sub). So dielectric resolution
    # is worse than NU, but we keep it to compare NTFF behaviour.
    z_gnd_lo = G["air_below"] - dx
    z_sub_lo = G["air_below"]
    z_sub_hi = G["air_below"] + G["h_sub"]
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dx
    src_z = z_sub_lo + dx * 0.5
    _add_patch(sim, G, dx, z_gnd_lo, z_sub_lo, z_sub_hi, z_patch_lo,
               z_patch_hi, src_z)
    margin = 3 * dx
    sim.add_ntff_box(
        corner_lo=(max(G["px_lo"] - 8e-3, margin),
                   max(G["py_lo"] - 8e-3, margin),
                   max(z_gnd_lo - 2 * dx, margin)),
        corner_hi=(min(G["px_lo"] + G["L"] + 8e-3, G["dom_x"] - margin),
                   min(G["py_lo"] + G["W"] + 8e-3, G["dom_y"] - margin),
                   min(z_patch_hi + 15e-3, dom_z - margin)),
        freqs=[F_DESIGN],
    )
    return run_case(sim, "Case 2 — UNIFORM patch (dx=1mm everywhere)",
                    num_periods=40)


def case3_nu_patch():
    G = _patch_geometry()
    dx = 1e-3
    n_cpml = 8
    n_sub = 6; dz_sub = G["h_sub"] / n_sub
    n_below = int(math.ceil(G["air_below"] / dx))
    n_above = int(math.ceil(G["air_above"] / dx))
    dz = np.asarray(smooth_grading(np.concatenate([
        np.full(n_below, dx), np.full(n_sub, dz_sub), np.full(n_above, dx)
    ])), dtype=np.float64)
    sim = Simulation(freq_max=4e9, domain=(G["dom_x"], G["dom_y"], 0),
                     dx=dx, dz_profile=dz, boundary="cpml",
                     cpml_layers=n_cpml)
    z_gnd_lo = G["air_below"] - dz_sub
    z_sub_lo = G["air_below"]
    z_sub_hi = G["air_below"] + G["h_sub"]
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dz_sub
    src_z = z_sub_lo + dz_sub * 2.5
    _add_patch(sim, G, dz_sub, z_gnd_lo, z_sub_lo, z_sub_hi, z_patch_lo,
               z_patch_hi, src_z)
    margin = 3 * dx
    dom_z = float(np.sum(dz))
    sim.add_ntff_box(
        corner_lo=(max(G["px_lo"] - 8e-3, margin),
                   max(G["py_lo"] - 8e-3, margin),
                   max(z_gnd_lo - 2 * dz_sub, 2 * dx)),
        corner_hi=(min(G["px_lo"] + G["L"] + 8e-3, G["dom_x"] - margin),
                   min(G["py_lo"] + G["W"] + 8e-3, G["dom_y"] - margin),
                   min(z_patch_hi + 15e-3, dom_z - 2 * dx)),
        freqs=[F_DESIGN],
    )
    return run_case(sim, "Case 3 — NU patch (same geometry, graded dz)",
                    num_periods=40)


def main():
    print("=" * 70)
    print("Patch NTFF root cause deep dive")
    print("=" * 70)
    case1_dipole()
    case2_uniform_patch()
    case3_nu_patch()
    print("\n=== Verdict ===")
    print("  Case 1 should peak at θ≈90° (dipole equatorial).")
    print("  Cases 2 and 3 should peak at θ≈0° if patch mode is excited.")
    print("  If 2 is broadside but 3 is grazing → NU NTFF bug.")
    print("  If both 2 and 3 are grazing → patch mode not reaching NTFF.")
    print("  Record the numbers in issue #48.")


if __name__ == "__main__":
    main()

"""Resolution-convergence study of the FULL S11(f) curve for the cv05
2.4 GHz FR4 probe-fed patch antenna (rfx, CPU / non-uniform mesh).

WHY
---
At the coarse dx = 1 mm baseline (examples/crossval/05_patch_antenna.py)
the rfx lumped-port S11 dip did NOT agree with openEMS in either dip
frequency or depth (rfx 2.66 GHz / -0.32 dB vs openEMS 2.19 GHz /
-0.53 dB); only the Harminv *resonance* agreed (2.65 %). The single-cell
lumped port carries a parasitic cell reactance that produces a monotonic
S11 background and a shallow, noisy dip. This script asks a sharper
question: is the FULL complex S11(f) TREND (both |S11| and phase / input
impedance Z11) mesh-CONVERGED, i.e. does the curve-to-curve difference
shrink as the substrate/patch/feed cells are refined?

METHOD
------
Same geometry as cv05 (2.4 GHz rectangular patch, FR4 eps_r=4.3, h=1.5 mm,
W=38.0 x L=29.5 mm, 8 mm probe inset, finite 60x55 mm PEC ground, CPML +
air padding). CPU-only, so a GRADED non-uniform mesh keeps the air/PML
coarse (~1 mm) while the substrate/patch/feed region is refined to
dx_fine in {0.5, 0.33, 0.25} mm. The z stack is already non-uniform and
fine (0.25 mm across the substrate), identical to cv05, so the Courant dt
is set by that 0.25 mm z cell and stays constant for dx_fine >= 0.25 mm
(refining xy does NOT shrink dt until dx_fine < 0.25 mm) -> the physical
ring-down duration for a fixed n_steps is preserved across the sweep.

For each resolution:
  * FULL complex S11(f) over 1.5-3.5 GHz via the rfx lumped-port path
    (sim.run(compute_s_params=True, s_param_freqs=...)), exactly the path
    cv05 PART 3 uses for this probe patch.
  * Input impedance Z11(f) = Z0 * (1 + S11) / (1 - S11), Z0 = 50 ohm.
  * Harminv ring-down resonance from a separate broadband source run
    (cv05 PART 1 path).

Convergence is quantified curve-to-curve (successive resolutions, on the
shared 101-point frequency grid): max and mean |dS11| over the band, the
S11-dip frequency shift, and the dip-depth shift.

Run (resolution-parametrized CLI; NU graded mesh keeps cells bounded):
  # CPU sanity at one coarse level
  JAX_PLATFORMS=cpu python scripts/crossval/rfx_patch_s11_convergence.py \
      --dx-fine-list-mm 0.5
  # GPU high-res sweep (VESSL): just change the list
  python scripts/crossval/rfx_patch_s11_convergence.py \
      --dx-fine-list-mm 0.5,0.33,0.25,0.2 --output scripts/crossval/out
Passing no --dx-fine-list-mm (empty) only (re)builds the overlay plot +
convergence table from whatever per-resolution JSONs are already on disk.
"""

import argparse
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)
C0 = 2.998e8
Z0_PORT = 50.0

# =============================================================================
# Geometry — copied verbatim from examples/crossval/05_patch_antenna.py
# =============================================================================
f_design = 2.4e9
eps_r = 4.3
tan_delta = 0.02              # cv05's FR4 loss tangent (folded into sigma below)
h_sub = 1.5e-3
W = 38.0e-3
L = 29.5e-3
gx = 60.0e-3
gy = 55.0e-3
probe_inset = 8.0e-3

dx_coarse = 1.0e-3            # air / PML cell size (baseline)
n_cpml = 8
n_sub = 6
dz_sub = h_sub / n_sub       # 0.25 mm per cell inside substrate
air_below = 12.0e-3
air_above = 25.0e-3
n_below = int(math.ceil(air_below / dx_coarse))
n_above = int(math.ceil(air_above / dx_coarse))

dom_x = gx + 2 * 10e-3
dom_y = gy + 2 * 10e-3
dom_z = air_below + h_sub + air_above

gx_lo = (dom_x - gx) / 2
gx_hi = gx_lo + gx
gy_lo = (dom_y - gy) / 2
gy_hi = gy_lo + gy
patch_x_lo = dom_x / 2 - L / 2
patch_x_hi = dom_x / 2 + L / 2
patch_y_lo = dom_y / 2 - W / 2
patch_y_hi = dom_y / 2 + W / 2
feed_x = patch_x_lo + probe_inset
feed_y = dom_y / 2

z_gnd_lo = air_below - dz_sub
z_gnd_hi = air_below
z_sub_lo = air_below
z_sub_hi = air_below + h_sub
z_patch_lo = z_sub_hi
z_patch_hi = z_sub_hi + dz_sub

# FR4 loss: sigma = 2*pi*f*eps0*eps_r*tan_delta at design freq
EPS0 = 8.8541878128e-12
fr4_sigma = 2 * math.pi * f_design * EPS0 * eps_r * tan_delta

# Analytic (Balanis Ch. 14) reference resonance
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_sub / W) ** (-0.5)
delta_L = 0.412 * h_sub * ((eps_eff + 0.3) * (W / h_sub + 0.264)) / \
          ((eps_eff - 0.258) * (W / h_sub + 0.8))
L_eff = L + 2 * delta_L
f_resonance_an = C0 / (2 * L_eff * math.sqrt(eps_eff))

# Shared S11 frequency grid (identical across resolutions -> direct dS11)
FREQS_HZ = np.linspace(1.5e9, 3.5e9, 101)

# Fine (patch+feed) region bounds: patch extent +/- 2 mm margin
_MARGIN = 2.0e-3
fine_x_lo = patch_x_lo - _MARGIN
fine_x_hi = patch_x_hi + _MARGIN
fine_y_lo = patch_y_lo - _MARGIN
fine_y_hi = patch_y_hi + _MARGIN


def _graded_profile(dom_len, interior_lo, interior_hi, dx_fine, dx_coarse):
    """dx profile: coarse (dx_coarse) outside [interior_lo, interior_hi],
    fine (dx_fine) inside, with smooth grading (max cell ratio 1.3).

    Mirrors cv05's `_refined_xy_profile`. First/last cells are dx_coarse
    (CPML cells must be uniform for make_nonuniform_grid).
    """
    from rfx.auto_config import smooth_grading
    lo_len = max(interior_lo, 0.0)
    hi_len = max(dom_len - interior_hi, 0.0)
    fine_len = max(interior_hi - interior_lo, 0.0)
    n_lo = max(1, int(round(lo_len / dx_coarse)))
    n_fine = max(1, int(round(fine_len / dx_fine)))
    n_hi = max(1, int(round(hi_len / dx_coarse)))
    raw = np.concatenate([
        np.full(n_lo, dx_coarse),
        np.full(n_fine, dx_fine),
        np.full(n_hi, dx_coarse),
    ])
    return smooth_grading(raw, max_ratio=1.3)


def _build_dz_profile():
    from rfx.auto_config import smooth_grading
    raw_dz = np.concatenate([
        np.full(n_below, dx_coarse),
        np.full(1, dz_sub),
        np.full(n_sub, dz_sub),
        np.full(n_above, dx_coarse),
    ])
    return smooth_grading(raw_dz, max_ratio=1.3)


def build_patch(with_port, dx_profile, dy_profile, dz_profile,
                port_extent_cells=1):
    """Build the cv05 patch stack with a graded xy mesh.

    port_extent_cells only affects the with_port build: 1 = cv05's
    single-cell z extent (from mid-substrate to patch); >1 kept for the
    port-parasitic probe (extends the z span of the wire port).
    """
    from rfx import Simulation, Box
    from rfx.boundaries.spec import BoundarySpec
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(
        freq_max=4e9,
        domain=(dom_x, dom_y, 0),
        dx=dx_coarse,
        dz_profile=dz_profile,
        dx_profile=dx_profile,
        dy_profile=dy_profile,
        boundary=BoundarySpec.uniform("cpml"),
        cpml_layers=n_cpml,
    )
    sim.add_material("fr4", eps_r=eps_r, sigma=fr4_sigma)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo),
                (gx_hi, gy_hi, z_gnd_hi)), material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo),
                (gx_hi, gy_hi, z_sub_hi)), material="fr4")
    sim.add(Box((patch_x_lo, patch_y_lo, z_patch_lo),
                (patch_x_hi, patch_y_hi, z_patch_hi)), material="pec")

    if with_port:
        port_z0 = z_sub_lo + dz_sub * 1.5
        port_extent = z_sub_hi - port_z0
        if port_extent_cells > 1:
            # extend downward toward the ground plane -> longer wire port
            port_z0 = z_sub_lo + dz_sub * 0.5
            port_extent = z_sub_hi - port_z0
        sim.add_port(
            position=(feed_x, feed_y, port_z0),
            component="ez",
            impedance=Z0_PORT,
            extent=port_extent,
            waveform=GaussianPulse(f0=f_design, bandwidth=0.8),
        )
    else:
        src_z = z_sub_lo + dz_sub * 2.5
        sim.add_source(
            position=(feed_x, feed_y, src_z),
            component="ez",
            waveform=GaussianPulse(f0=f_design, bandwidth=1.2),
        )
        sim.add_probe(
            position=(dom_x / 2 + 5e-3, dom_y / 2 + 5e-3, src_z),
            component="ez",
        )
    return sim


def _tag(dx_fine_mm):
    return f"{dx_fine_mm:.2f}mm"


def _json_path(out_dir, dx_fine_mm):
    return os.path.join(out_dir, f"rfx_patch_s11_conv_{_tag(dx_fine_mm)}.json")


def run_one(dx_fine_mm, out_dir=OUT_DIR, freqs_hz=FREQS_HZ,
            n_steps=12000, num_periods=60.0, port_extent_cells=1):
    """Run Harminv + S11 at one interior resolution; write JSON; return dict."""
    import jax.numpy as jnp
    from rfx.harminv import harminv

    os.makedirs(out_dir, exist_ok=True)
    freqs_hz = np.asarray(freqs_hz)
    dx_fine = dx_fine_mm * 1e-3
    dz_profile = _build_dz_profile()
    dx_profile = _graded_profile(dom_x, fine_x_lo, fine_x_hi, dx_fine, dx_coarse)
    dy_profile = _graded_profile(dom_y, fine_y_lo, fine_y_hi, dx_fine, dx_coarse)
    nx, ny, nz = len(dx_profile), len(dy_profile), len(dz_profile)
    ncells = nx * ny * nz

    print("=" * 72)
    print(f"RESOLUTION dx_fine = {dx_fine_mm:.3f} mm  "
          f"(port_extent_cells={port_extent_cells})")
    print(f"  grid: {nx} x {ny} x {nz} = {ncells:,} cells "
          f"(air/PML {dx_coarse*1e3:.1f} mm)")
    print("=" * 72)

    # ---- Harminv ring-down (clean resonance) ----
    sim_h = build_patch(False, dx_profile, dy_profile, dz_profile)
    t0 = time.time()
    res_h = sim_h.run(num_periods=num_periods)
    t_harminv = time.time() - t0
    ts = np.asarray(res_h.time_series).ravel()
    dt_h = float(res_h.dt)
    skip = int(len(ts) * 0.3)
    modes = harminv(ts[skip:], dt_h, 1.5e9, 3.5e9)
    modes_good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-8]
    if modes_good:
        modes_good.sort(key=lambda m: abs(m.freq - f_resonance_an))
        f_res_harminv = float(modes_good[0].freq)
        Q_harminv = float(modes_good[0].Q)
    else:
        f_res_harminv = float("nan")
        Q_harminv = float("nan")
    print(f"  Harminv: f={f_res_harminv/1e9:.4f} GHz Q={Q_harminv:.1f} "
          f"({t_harminv:.1f}s, dt={dt_h*1e12:.4f} ps)")

    # ---- Full complex S11 via lumped port ----
    sim_p = build_patch(True, dx_profile, dy_profile, dz_profile,
                        port_extent_cells=port_extent_cells)
    freqs_s = jnp.asarray(freqs_hz)
    t0 = time.time()
    result = sim_p.run(
        n_steps=n_steps,
        compute_s_params=True,
        s_param_freqs=freqs_s,
        s_param_n_steps=n_steps,
    )
    t_s11 = time.time() - t0
    S = np.asarray(result.s_params)
    S11 = S[0, 0, :]
    S11_dB = 20 * np.log10(np.maximum(np.abs(S11), 1e-6))
    Z11 = Z0_PORT * (1 + S11) / (1 - S11)

    # dip near analytic resonance (local, ±10 %)
    lo = int(np.searchsorted(freqs_hz, f_resonance_an * 0.90))
    hi = int(np.searchsorted(freqs_hz, f_resonance_an * 1.10))
    local_idx = lo + int(np.argmin(S11_dB[lo:hi]))
    f_dip = float(freqs_hz[local_idx])
    s11_dip_dB = float(S11_dB[local_idx])
    passive = bool(np.all(np.abs(S11) < 1.05))
    print(f"  S11 dip: f={f_dip/1e9:.4f} GHz depth={s11_dip_dB:.3f} dB "
          f"max|S11|={np.max(np.abs(S11)):.3f} ({t_s11:.1f}s)")

    payload = {
        "dx_fine_mm": dx_fine_mm,
        "port_extent_cells": port_extent_cells,
        "grid": {"nx": nx, "ny": ny, "nz": nz, "ncells": ncells},
        "dt_s": dt_h,
        "runtime_s": {"harminv": t_harminv, "s11": t_s11},
        "analytic_resonance_hz": float(f_resonance_an),
        "harminv_hz": f_res_harminv,
        "harminv_Q": Q_harminv,
        "s11_dip_hz": f_dip,
        "s11_dip_db": s11_dip_dB,
        "s11_max_abs": float(np.max(np.abs(S11))),
        "s11_passive": passive,
        "freqs_hz": [float(v) for v in freqs_hz],
        "s11": [[float(v.real), float(v.imag)] for v in S11],
        "z11": [[float(v.real), float(v.imag)] for v in Z11],
    }
    path = _json_path(out_dir, dx_fine_mm)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"  wrote {path}")
    return payload


def _load_all(out_dir):
    """Load every conv JSON on disk, sorted fine->coarse dx (coarse first)."""
    out = []
    for fn in os.listdir(out_dir):
        if fn.startswith("rfx_patch_s11_conv_") and fn.endswith(".json"):
            with open(os.path.join(out_dir, fn)) as f:
                out.append(json.load(f))
    out.sort(key=lambda d: -d["dx_fine_mm"])   # coarsest first
    return out


def _assess_and_plot(out_dir):
    data = _load_all(out_dir)
    if not data:
        print("No conv JSONs on disk yet.")
        return
    # Reconstruct complex arrays
    for d in data:
        d["_S11"] = np.array([complex(a, b) for a, b in d["s11"]])
        d["_freqs"] = np.array(d["freqs_hz"])

    print("\n" + "=" * 72)
    print("CONVERGENCE SUMMARY (curve-to-curve on shared 1.5-3.5 GHz grid)")
    print("=" * 72)
    hdr = (f"{'dx_fine':>8} {'cells':>10} {'t_s11':>7} {'harminv':>9} "
           f"{'dip_f':>8} {'dip_dB':>8} {'max|dS11|':>10} {'mean|dS11|':>11} "
           f"{'d_dipf':>8} {'d_dipdB':>8}")
    print(hdr)
    print("-" * len(hdr))
    prev = None
    for d in data:
        if prev is not None:
            dS = np.abs(d["_S11"] - prev["_S11"])
            max_d = float(np.max(dS))
            mean_d = float(np.mean(dS))
            d_dipf = (d["s11_dip_hz"] - prev["s11_dip_hz"]) / 1e9
            d_dipdB = d["s11_dip_db"] - prev["s11_dip_db"]
        else:
            max_d = mean_d = d_dipf = d_dipdB = float("nan")
        print(f"{d['dx_fine_mm']:>7.2f}m {d['grid']['ncells']:>10,} "
              f"{d['runtime_s']['s11']:>6.0f}s "
              f"{d['harminv_hz']/1e9:>8.4f}G "
              f"{d['s11_dip_hz']/1e9:>7.3f}G {d['s11_dip_db']:>8.3f} "
              f"{max_d:>10.4f} {mean_d:>11.4f} {d_dipf:>+8.4f} {d_dipdB:>+8.3f}")
        prev = d

    # ---- overlay plot: |S11| dB and phase ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(data):
        c = cmap(i / max(1, len(data) - 1))
        f_g = d["_freqs"] / 1e9
        s11 = d["_S11"]
        s11_dB = 20 * np.log10(np.maximum(np.abs(s11), 1e-6))
        ph = np.unwrap(np.angle(s11)) * 180 / np.pi
        lbl = f"dx_fine={d['dx_fine_mm']:.2f} mm ({d['grid']['ncells']/1e3:.0f}k)"
        axes[0].plot(f_g, s11_dB, color=c, lw=1.6, label=lbl)
        axes[1].plot(f_g, ph, color=c, lw=1.6, label=lbl)
    axes[0].axvline(f_resonance_an / 1e9, color="k", ls="--", alpha=0.5,
                    label=f"analytic {f_resonance_an/1e9:.3f} GHz")
    axes[0].set_xlabel("f (GHz)"); axes[0].set_ylabel("|S11| (dB)")
    axes[0].set_title("|S11| vs frequency — resolution sweep")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].axvline(f_resonance_an / 1e9, color="k", ls="--", alpha=0.5)
    axes[1].set_xlabel("f (GHz)"); axes[1].set_ylabel("phase(S11) (deg)")
    axes[1].set_title("phase(S11) vs frequency — resolution sweep")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)
    fig.suptitle("2.4 GHz FR4 probe-fed patch — rfx S11(f) mesh convergence "
                 "(non-uniform CPU mesh)", fontweight="bold")
    plt.tight_layout()
    png = os.path.join(out_dir, "rfx_patch_s11_convergence.png")
    plt.savefig(png, dpi=140); plt.close()
    print(f"\nSaved overlay plot: {png}")
    return png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dx-fine-list-mm", default="",
                   help="comma-separated interior (substrate/patch/feed) cell "
                        "sizes in mm for the NU graded mesh, e.g. 0.5,0.33,0.25 "
                        "(air/PML stay coarse ~1mm). Empty = only rebuild the "
                        "plot/table from existing JSONs.")
    p.add_argument("--output", default=OUT_DIR,
                   help="output directory for per-resolution JSON "
                        "(rfx_patch_s11_conv_<dx>.json) + overlay PNG")
    p.add_argument("--n-steps", type=int, default=12000,
                   help="FDTD steps for the S-parameter (S11) run")
    p.add_argument("--num-periods", type=float, default=60.0,
                   help="ring-down periods for the Harminv run")
    p.add_argument("--freq-lo-ghz", type=float, default=1.5)
    p.add_argument("--freq-hi-ghz", type=float, default=3.5)
    p.add_argument("--nfreq", type=int, default=101)
    p.add_argument("--port-extent-cells", type=int, default=1,
                   help="1 = cv05 single-cell z port; >1 extends the wire-port "
                        "z span toward the ground plane (port-parasitic probe)")
    args = p.parse_args()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    freqs_hz = np.linspace(args.freq_lo_ghz * 1e9, args.freq_hi_ghz * 1e9,
                           args.nfreq)
    dx_list = [float(v) for v in args.dx_fine_list_mm.split(",") if v.strip()]
    for dx_fine_mm in dx_list:
        run_one(dx_fine_mm, out_dir=out_dir, freqs_hz=freqs_hz,
                n_steps=args.n_steps, num_periods=args.num_periods,
                port_extent_cells=args.port_extent_cells)
    _assess_and_plot(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Issue #31 — quantitative + qualitative validation of NU patch antenna.

Produces three artifacts from the same 2.4 GHz FR4 patch on the NU-z
mesh (the same geometry the segmented-scan smoke used):

  1. S11(f) magnitude via ``add_port`` + ``compute_s_params=True``.
  2. Far-field radiation pattern (azimuth + elevation) at f_res via
     ``add_ntff_box`` + ``compute_far_field``.
  3. Field-evolution animation: Ez on the xz mid-slice sampled at
     increasing n_steps snapshots.

Outputs (docs/research_notes/issue31_figs/):
  - issue31_patch_s11.png
  - issue31_patch_farfield.png
  - issue31_patch_field_evolution.gif
"""

from __future__ import annotations

import math
import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from rfx import Simulation, Box
from rfx.auto_config import smooth_grading
from rfx.sources.sources import GaussianPulse


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "docs", "research_notes", "issue31_figs")
os.makedirs(OUT_DIR, exist_ok=True)


def build(*, dx_mm=1.5, with_port=True, with_ntff=False):
    f_design = 2.4e9
    eps_r_fr4 = 4.3
    h_sub = 1.5e-3
    W, L = 38.0e-3, 29.5e-3
    gx, gy = 60.0e-3, 55.0e-3
    air_above, air_below = 25.0e-3, 12.0e-3
    probe_inset = 8.0e-3

    dx = dx_mm * 1e-3
    n_cpml = 8
    n_sub = 6
    dz_sub = h_sub / n_sub
    n_below = int(math.ceil(air_below / dx))
    n_above = int(math.ceil(air_above / dx))
    dz_raw = np.concatenate([np.full(n_below, dx), np.full(n_sub, dz_sub),
                             np.full(n_above, dx)])
    dz_profile = np.asarray(smooth_grading(dz_raw), dtype=np.float64)

    dom_x = gx + 20e-3
    dom_y = gy + 20e-3
    gx_lo = (dom_x - gx) / 2;  gx_hi = gx_lo + gx
    gy_lo = (dom_y - gy) / 2;  gy_hi = gy_lo + gy
    px_lo = dom_x / 2 - L / 2; px_hi = dom_x / 2 + L / 2
    py_lo = dom_y / 2 - W / 2; py_hi = dom_y / 2 + W / 2
    feed_x = px_lo + probe_inset
    feed_y = dom_y / 2
    z_gnd_lo = air_below - dz_sub
    z_sub_lo = air_below
    z_sub_hi = air_below + h_sub
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dz_sub
    src_z = z_sub_lo + dz_sub * 2.5

    sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, 0), dx=dx,
                     dz_profile=dz_profile, boundary="cpml", cpml_layers=n_cpml)
    sim.add_material("fr4", eps_r=eps_r_fr4)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_sub_lo)),
            material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)),
            material="fr4")
    sim.add(Box((px_lo, py_lo, z_patch_lo), (px_hi, py_hi, z_patch_hi)),
            material="pec")
    if with_port:
        port_z0 = z_sub_lo + dz_sub * 1.5
        sim.add_port(position=(feed_x, feed_y, port_z0), component="ez",
                     impedance=50.0, extent=z_sub_hi - port_z0,
                     waveform=GaussianPulse(f0=f_design, bandwidth=0.8))
    else:
        sim.add_source(position=(feed_x, feed_y, src_z), component="ez",
                       waveform=GaussianPulse(f0=f_design, bandwidth=1.2))
        sim.add_probe(position=(dom_x / 2 + 5e-3, dom_y / 2 + 5e-3, src_z),
                      component="ez")
    if with_ntff:
        # NTFF box around the patch, 2 cells outside PEC geometry.
        margin = max(2 * dx, 2 * dz_sub)
        sim.add_ntff_box(
            corner_lo=(gx_lo - margin, gy_lo - margin, z_gnd_lo - margin),
            corner_hi=(gx_hi + margin, gy_hi + margin, z_patch_hi + 10e-3),
            freqs=np.array([2.4e9, 2.5e9]),
        )
    layout = dict(
        dom_x=dom_x, dom_y=dom_y,
        gnd=(gx_lo, gy_lo, z_gnd_lo, gx, gy, dz_sub),
        sub=(gx_lo, gy_lo, z_sub_lo, gx, gy, h_sub),
        patch=(px_lo, py_lo, z_patch_lo, L, W, dz_sub),
        feed=(feed_x, feed_y, src_z),
        f_design=f_design,
    )
    return sim, layout


# -----------------------------------------------------------------------------
# 1) S11(f)
# -----------------------------------------------------------------------------
def run_s11(*, dx_mm):
    print("\n=== [1/3] S11(f) via lumped port ===")
    sim, layout = build(dx_mm=dx_mm, with_port=True)
    g = sim._build_nonuniform_grid()
    print(f"[cfg] cells={g.nx * g.ny * g.nz:,}")
    t0 = time.time()
    res = sim.run(num_periods=40, compute_s_params=True)
    print(f"[run] done in {time.time() - t0:.1f}s")
    freqs = np.asarray(res.freqs)
    s11 = np.asarray(res.s_params)[0, 0, :]
    s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-6))
    mask = (freqs > 1.5e9) & (freqs < 3.5e9)
    f_band = freqs[mask]; s_band = s11_db[mask]
    idx = int(np.argmin(s_band))
    f_res_s11 = float(f_band[idx])
    print(f"[result] S11 dip at f = {f_res_s11/1e9:.4f} GHz  "
          f"|S11|_min = {s_band[idx]:.2f} dB")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs / 1e9, s11_db, lw=1.5)
    ax.axvline(layout["f_design"] / 1e9, ls="--", c="gray",
               label=f"design f₀ = {layout['f_design']/1e9:.2f} GHz")
    ax.axvline(f_res_s11 / 1e9, ls="--", c="red",
               label=f"rfx S11 dip = {f_res_s11/1e9:.3f} GHz")
    ax.set_xlim(1.5, 3.5); ax.set_ylim(-30, 0.5)
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("|S11| (dB)")
    ax.set_title("Patch S11 on NU mesh (lumped port)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "issue31_patch_s11.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[out] {out}")
    return f_res_s11


# -----------------------------------------------------------------------------
# 2) Far-field radiation pattern at f_res
# -----------------------------------------------------------------------------
def run_farfield(*, dx_mm, f_res):
    print("\n=== [2/3] Far-field radiation pattern at f_res ===")
    sim, layout = build(dx_mm=dx_mm, with_port=True, with_ntff=True)
    g = sim._build_nonuniform_grid()
    print(f"[cfg] cells={g.nx * g.ny * g.nz:,}, NTFF f={f_res/1e9:.3f} GHz")
    t0 = time.time()
    res = sim.run(num_periods=40, compute_s_params=False)
    print(f"[run] done in {time.time() - t0:.1f}s")

    from rfx.farfield import compute_far_field
    theta = np.linspace(0, np.pi / 2, 91)
    phi = np.linspace(0, 2 * np.pi, 181)
    th_grid, ph_grid = np.meshgrid(theta, phi, indexing="ij")
    # NTFF gives |E_far| at (theta, phi); use the closer of the two stored freqs.
    freqs_ntff = np.asarray(res.ntff_box.freqs)
    f_idx = int(np.argmin(np.abs(freqs_ntff - f_res)))
    ef = compute_far_field(res.ntff_data, res.ntff_box,
                           freq_idx=f_idx,
                           theta=th_grid.ravel(),
                           phi=ph_grid.ravel(),
                           r=1.0)
    mag = np.sqrt(np.abs(ef.E_theta) ** 2 + np.abs(ef.E_phi) ** 2)
    mag = mag.reshape(th_grid.shape)
    mag_db = 20 * np.log10(np.maximum(mag / np.max(mag), 1e-3))

    # Elevation cut (phi=0 = +x plane)
    cut_phi0 = mag_db[:, 0]
    cut_phi90 = mag_db[:, 90]
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax1.plot(theta, cut_phi0, label="φ=0° (xz-plane)")
    ax1.plot(theta, cut_phi90, label="φ=90° (yz-plane)")
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_rlim(-30, 0)
    ax1.set_title(f"Elevation cuts at {freqs_ntff[f_idx]/1e9:.3f} GHz")
    ax1.legend(loc="lower right", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(mag_db, origin="lower", aspect="auto",
                    extent=[0, 360, 0, 90], cmap="viridis", vmin=-30, vmax=0)
    fig.colorbar(im, ax=ax2, label="|E_far| (dB, normalized)")
    ax2.set_xlabel("φ (deg)"); ax2.set_ylabel("θ (deg)")
    ax2.set_title("|E_far| over upper hemisphere")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "issue31_patch_farfield.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[out] {out}")


# -----------------------------------------------------------------------------
# 3) Field evolution GIF — Ez on xz mid-slice
# -----------------------------------------------------------------------------
def run_field_evolution(*, dx_mm, n_frames=16, n_steps_total=2000):
    print("\n=== [3/3] Field evolution GIF (Ez xz-slice) ===")
    sim, layout = build(dx_mm=dx_mm, with_port=False)
    g = sim._build_nonuniform_grid()
    print(f"[cfg] cells={g.nx * g.ny * g.nz:,}  frames={n_frames}")
    j_mid = g.ny // 2

    step_schedule = np.linspace(n_steps_total // n_frames,
                                n_steps_total, n_frames).astype(int)
    frames_ez = []
    t0_all = time.time()
    for ns in step_schedule:
        t0 = time.time()
        res = sim.run(n_steps=int(ns), compute_s_params=False)
        ez_slice = np.asarray(res.state.ez[:, j_mid, :])
        frames_ez.append(ez_slice)
        print(f"  n_steps={ns:4d}  max|Ez|={np.max(np.abs(ez_slice)):.3e}  "
              f"dt={time.time() - t0:.1f}s")
    print(f"[run] total {time.time() - t0_all:.1f}s")

    vmax = float(np.percentile([np.max(np.abs(f)) for f in frames_ez], 95) or 1.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(frames_ez[0].T, origin="lower", aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Ez (V/m)")
    title = ax.set_title(f"Ez xz-slice | step {step_schedule[0]}")

    def update(k):
        im.set_data(frames_ez[k].T)
        title.set_text(f"Ez xz-slice | step {step_schedule[k]}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(frames_ez), interval=200, blit=False)
    out = os.path.join(OUT_DIR, "issue31_patch_field_evolution.gif")
    anim.save(out, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"[out] {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-mm", type=float, default=1.5)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--n-steps-anim", type=int, default=1500)
    ap.add_argument("--skip-farfield", action="store_true")
    ap.add_argument("--skip-anim", action="store_true")
    args = ap.parse_args()

    f_res = run_s11(dx_mm=args.dx_mm)
    if not args.skip_farfield:
        run_farfield(dx_mm=args.dx_mm, f_res=f_res)
    if not args.skip_anim:
        run_field_evolution(dx_mm=args.dx_mm, n_frames=args.frames,
                            n_steps_total=args.n_steps_anim)


if __name__ == "__main__":
    main()

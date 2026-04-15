"""Issue #31 — quantitative + qualitative validation of NU patch antenna.

Produces three artifacts from the 2.4 GHz FR4 patch on a non-uniform
z mesh. All three agree on the same f_res, which comes from Harminv
(ringdown-based extraction) — NOT from the lumped-port S11 dip (which
is a circuit-matching artifact, see #46).

Artifacts in docs/research_notes/issue31_figs/:
  1. issue31_patch_s11.png — |S11|(f) with harminv + analytic markers.
  2. issue31_patch_farfield.png — NTFF at harminv f_res:
        - polar elevation cuts φ=0° and φ=90°, −90° ≤ θ ≤ +90°
        - full (θ, φ) hemisphere as a 2-D imshow
        - peak direction annotated.
  3. issue31_patch_field_evolution.gif — Ez xz-slice with structure
     outlines, per-frame vmax so propagation is visible throughout.
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
from rfx.harminv import harminv


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "docs", "research_notes", "issue31_figs")
os.makedirs(OUT_DIR, exist_ok=True)

C0 = 2.998e8


# -----------------------------------------------------------------------------
# Geometry (shared by all three runs; matches nonuniform_patch_demo.py)
# -----------------------------------------------------------------------------
def _geometry(dx_mm):
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
    g_box = dict(
        dom_x=dom_x, dom_y=dom_y, dx_mm=dx_mm, dz_profile=dz_profile,
        n_cpml=n_cpml, dz_sub=dz_sub, f_design=f_design,
        gx_lo=(dom_x - gx) / 2, gy_lo=(dom_y - gy) / 2, gx=gx, gy=gy,
        z_gnd_lo=air_below - dz_sub, z_sub_lo=air_below,
        z_sub_hi=air_below + h_sub, z_patch_lo=air_below + h_sub,
        z_patch_hi=air_below + h_sub + dz_sub,
        px_lo=dom_x / 2 - L / 2, py_lo=dom_y / 2 - W / 2, L=L, W=W,
        feed_x=(dom_x / 2 - L / 2) + probe_inset, feed_y=dom_y / 2,
        src_z=air_below + dz_sub * 2.5, eps_r_fr4=eps_r_fr4,
    )
    # Analytic Balanis f_res
    eps_eff = (eps_r_fr4 + 1) / 2 + (eps_r_fr4 - 1) / 2 * (
        1 + 12 * h_sub / W) ** (-0.5)
    delta_L = 0.412 * h_sub * ((eps_eff + 0.3) * (W / h_sub + 0.264)) / \
              ((eps_eff - 0.258) * (W / h_sub + 0.8))
    g_box["f_analytic"] = C0 / (2 * (L + 2 * delta_L) * math.sqrt(eps_eff))
    return g_box


def _build(G, *, with_port, with_ntff, ntff_freqs=None):
    sim = Simulation(freq_max=4e9, domain=(G["dom_x"], G["dom_y"], 0),
                     dx=G["dx_mm"] * 1e-3, dz_profile=G["dz_profile"],
                     boundary="cpml", cpml_layers=G["n_cpml"])
    sim.add_material("fr4", eps_r=G["eps_r_fr4"])
    sim.add(Box((G["gx_lo"], G["gy_lo"], G["z_gnd_lo"]),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], G["z_sub_lo"])),
            material="pec")
    sim.add(Box((G["gx_lo"], G["gy_lo"], G["z_sub_lo"]),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], G["z_sub_hi"])),
            material="fr4")
    sim.add(Box((G["px_lo"], G["py_lo"], G["z_patch_lo"]),
                (G["px_lo"] + G["L"], G["py_lo"] + G["W"], G["z_patch_hi"])),
            material="pec")
    if with_port:
        port_z0 = G["z_sub_lo"] + G["dz_sub"] * 1.5
        sim.add_port(position=(G["feed_x"], G["feed_y"], port_z0),
                     component="ez", impedance=50.0,
                     extent=G["z_sub_hi"] - port_z0,
                     waveform=GaussianPulse(f0=G["f_design"], bandwidth=0.8))
    else:
        sim.add_source(position=(G["feed_x"], G["feed_y"], G["src_z"]),
                       component="ez",
                       waveform=GaussianPulse(f0=G["f_design"], bandwidth=1.2))
        sim.add_probe(position=(G["dom_x"] / 2 + 5e-3,
                                G["dom_y"] / 2 + 5e-3, G["src_z"]),
                      component="ez")
    if with_ntff:
        margin = max(2 * G["dx_mm"] * 1e-3, 2 * G["dz_sub"])
        sim.add_ntff_box(
            corner_lo=(G["gx_lo"] - margin, G["gy_lo"] - margin,
                       G["z_gnd_lo"] - margin),
            corner_hi=(G["gx_lo"] + G["gx"] + margin,
                       G["gy_lo"] + G["gy"] + margin,
                       G["z_patch_hi"] + 10e-3),
            freqs=np.asarray(ntff_freqs or [G["f_design"]]),
        )
    return sim


# -----------------------------------------------------------------------------
# Step 1 — Harminv from source+probe ringdown → true f_res
# -----------------------------------------------------------------------------
def run_harminv(G):
    print("\n=== [1/4] Harminv ringdown → true f_res ===")
    sim = _build(G, with_port=False, with_ntff=False)
    g = sim._build_nonuniform_grid()
    print(f"[cfg] cells={g.nx * g.ny * g.nz:,}")
    t0 = time.time()
    res = sim.run(num_periods=60, compute_s_params=False)
    print(f"[run] done in {time.time() - t0:.1f}s")
    ts = np.asarray(res.time_series).ravel()
    dt = float(res.dt)
    skip = int(len(ts) * 0.3)
    modes = harminv(ts[skip:], dt, 1.5e9, 3.5e9)
    good = [m for m in modes if m.Q > 5 and m.amplitude > 1e-8]
    if not good:
        raise RuntimeError("Harminv failed to extract a mode")
    best = max(good, key=lambda m: m.amplitude)
    f_res = float(best.freq)
    Q = float(best.Q)
    err = 100 * abs(f_res - G["f_analytic"]) / G["f_analytic"]
    print(f"[result] Harminv f_res = {f_res/1e9:.4f} GHz  Q = {Q:.1f}  "
          f"error = {err:.2f} %  (analytic = {G['f_analytic']/1e9:.4f} GHz)")
    return f_res, Q, err


# -----------------------------------------------------------------------------
# Step 2 — S11(f) via lumped port (secondary pin, context only)
# -----------------------------------------------------------------------------
def run_s11(G, *, f_res_harminv):
    print("\n=== [2/4] S11(f) via lumped port ===")
    sim = _build(G, with_port=True, with_ntff=False)
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
    print(f"[result] S11 dip: f = {f_res_s11/1e9:.4f} GHz, "
          f"|S11| = {s_band[idx]:.2f} dB (shallow — port matching artifact)")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(freqs / 1e9, s11_db, lw=1.5, label="rfx |S11|")
    ax.axvline(G["f_analytic"] / 1e9, ls="--", c="black",
               label=f"analytic f_res = {G['f_analytic']/1e9:.3f} GHz")
    ax.axvline(f_res_harminv / 1e9, ls="-", c="green",
               label=f"harminv f_res = {f_res_harminv/1e9:.3f} GHz")
    ax.axvline(f_res_s11 / 1e9, ls=":", c="red",
               label=f"rfx S11 dip = {f_res_s11/1e9:.3f} GHz (port artifact)")
    ax.set_xlim(1.5, 3.5); ax.set_ylim(-30, 0.5)
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("|S11| (dB)")
    ax.set_title("Patch |S11| on NU mesh (lumped port, secondary pin)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "issue31_patch_s11.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[out] {out}")


# -----------------------------------------------------------------------------
# Step 3 — Far-field at harminv f_res
# -----------------------------------------------------------------------------
def run_farfield(G, *, f_res):
    print(f"\n=== [3/4] Far-field radiation pattern at f_res = "
          f"{f_res/1e9:.3f} GHz ===")
    sim = _build(G, with_port=True, with_ntff=True, ntff_freqs=[f_res])
    t0 = time.time()
    res = sim.run(num_periods=40, compute_s_params=False)
    print(f"[run] done in {time.time() - t0:.1f}s")

    from rfx.farfield import compute_far_field
    theta = np.linspace(0, np.pi / 2, 91)   # upper hemisphere only (ground below)
    phi = np.linspace(0, 2 * np.pi, 181)
    grid = sim._build_nonuniform_grid()
    ef = compute_far_field(res.ntff_data, res.ntff_box, grid, theta, phi)
    E_t = np.asarray(ef.E_theta[0])  # (n_theta, n_phi)
    E_p = np.asarray(ef.E_phi[0])
    mag = np.sqrt(np.abs(E_t) ** 2 + np.abs(E_p) ** 2)
    mag_db = 20 * np.log10(np.maximum(mag / np.max(mag), 1e-3))

    # Peak direction
    i_peak, j_peak = np.unravel_index(np.argmax(mag), mag.shape)
    theta_peak_deg = np.degrees(theta[i_peak])
    phi_peak_deg = np.degrees(phi[j_peak])

    # Polar elevation cuts: unfold θ∈[0, π/2] to [-π/2, +π/2] by
    # taking φ=0 for positive θ side and φ=π for negative side.
    idx_phi0 = int(np.argmin(np.abs(phi - 0.0)))
    idx_phi180 = int(np.argmin(np.abs(phi - np.pi)))
    idx_phi90 = int(np.argmin(np.abs(phi - np.pi / 2)))
    idx_phi270 = int(np.argmin(np.abs(phi - 3 * np.pi / 2)))

    theta_full = np.concatenate([-theta[::-1], theta])
    def _unfold(i_left, i_right):
        return np.concatenate([mag_db[::-1, i_left], mag_db[:, i_right]])
    cut_xz = _unfold(idx_phi180, idx_phi0)   # φ=0/180 → xz-plane
    cut_yz = _unfold(idx_phi270, idx_phi90)  # φ=90/270 → yz-plane

    fig = plt.figure(figsize=(14, 5))
    # Polar cut
    ax1 = fig.add_subplot(1, 3, 1, projection="polar")
    ax1.plot(theta_full, cut_xz, label="φ=0° (xz-plane)", lw=1.5)
    ax1.plot(theta_full, cut_yz, label="φ=90° (yz-plane)", lw=1.5, ls="--")
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_thetamin(-90); ax1.set_thetamax(90)
    ax1.set_rlim(-30, 0)
    ax1.set_title(f"Elevation cuts @ {f_res/1e9:.3f} GHz\n"
                  f"(broadside = θ=0)")
    ax1.legend(loc="lower center", fontsize=8)

    # Azimuth cut at broadside
    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    # θ slightly off broadside to avoid axis singularity
    theta_plot_idx = min(5, len(theta) - 1)
    az_cut = mag_db[theta_plot_idx, :]
    ax2.plot(phi, az_cut, lw=1.5)
    ax2.set_title(f"Azimuth cut @ θ={np.degrees(theta[theta_plot_idx]):.0f}°")
    ax2.set_rlim(-30, 0)

    # 2D θ×φ map
    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.imshow(mag_db, origin="lower", aspect="auto",
                    extent=[0, 360, 0, 90], cmap="viridis", vmin=-30, vmax=0)
    ax3.plot(phi_peak_deg, theta_peak_deg, "r*", markersize=14,
             label=f"peak θ={theta_peak_deg:.0f}°, φ={phi_peak_deg:.0f}°")
    fig.colorbar(im, ax=ax3, label="|E_far| (dB, norm)")
    ax3.set_xlabel("φ (deg)"); ax3.set_ylabel("θ (deg)")
    ax3.set_title("Upper-hemisphere radiation")
    ax3.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Patch far-field, harminv f_res = {f_res/1e9:.3f} GHz",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "issue31_patch_farfield.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[out] {out}  peak at θ={theta_peak_deg:.1f}°, φ={phi_peak_deg:.1f}°")


# -----------------------------------------------------------------------------
# Step 4 — Field evolution GIF with per-frame vmax + geometry outlines
# -----------------------------------------------------------------------------
def run_field_evolution(G, *, n_frames, n_steps_total):
    print("\n=== [4/4] Field evolution GIF ===")
    sim = _build(G, with_port=False, with_ntff=False)
    g = sim._build_nonuniform_grid()
    print(f"[cfg] cells={g.nx * g.ny * g.nz:,}  frames={n_frames}")
    j_mid = g.ny // 2

    # Cell centre positions (for extent)
    dx = float(g.dx)
    x_centres = (np.arange(g.nx) - G["n_cpml"] + 0.5) * dx
    dz_arr = np.asarray(g.dz)
    z_edges = np.concatenate([[0], np.cumsum(dz_arr)])
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    z_centres = z_centres - z_centres[G["n_cpml"]]

    step_schedule = np.linspace(n_steps_total // n_frames,
                                n_steps_total, n_frames).astype(int)
    frames = []
    t0_all = time.time()
    for ns in step_schedule:
        t0 = time.time()
        res = sim.run(n_steps=int(ns), compute_s_params=False)
        ez_slice = np.asarray(res.state.ez[:, j_mid, :])
        frames.append(ez_slice)
        print(f"  n_steps={ns:4d}  max|Ez|={np.max(np.abs(ez_slice)):.3e}  "
              f"dt={time.time() - t0:.1f}s")
    print(f"[run] total {time.time() - t0_all:.1f}s")

    # Per-frame symmetric log normalisation: signed-log10 preserves sign.
    def _to_signed_log(a, floor):
        sign = np.sign(a)
        mag = np.log10(np.maximum(np.abs(a), floor) / floor)
        return sign * mag

    fig, ax = plt.subplots(figsize=(10, 5))
    # First frame setup
    floor0 = max(np.max(np.abs(frames[0])) * 1e-3, 1e-6)
    img0 = _to_signed_log(frames[0].T, floor0)
    vmax0 = max(float(np.max(np.abs(img0))), 0.5)
    im = ax.imshow(
        img0, origin="lower", aspect="auto",
        extent=[x_centres[0] * 1e3, x_centres[-1] * 1e3,
                z_centres[0] * 1e3, z_centres[-1] * 1e3],
        cmap="RdBu_r", vmin=-vmax0, vmax=vmax0,
    )
    fig.colorbar(im, ax=ax, label="sign × log10(|Ez| / floor)")

    # Structure overlays (drawn once — static geometry)
    def _rect(x_m, z_m, w_m, h_m, **kw):
        return Rectangle((x_m * 1e3, z_m * 1e3), w_m * 1e3, h_m * 1e3,
                         fill=False, **kw)
    ax.add_patch(_rect(G["gx_lo"], G["z_gnd_lo"], G["gx"], G["dz_sub"],
                       edgecolor="black", lw=1.2, label="ground PEC"))
    ax.add_patch(_rect(G["gx_lo"], G["z_sub_lo"], G["gx"],
                       G["z_sub_hi"] - G["z_sub_lo"], edgecolor="dimgray",
                       lw=1.0, ls="--", label="FR4"))
    ax.add_patch(_rect(G["px_lo"], G["z_patch_lo"], G["L"], G["dz_sub"],
                       edgecolor="black", lw=2.0, label="patch PEC"))
    ax.plot(G["feed_x"] * 1e3, G["src_z"] * 1e3, "g^", ms=8, label="source")
    ax.plot((G["dom_x"] / 2 + 5e-3) * 1e3, G["src_z"] * 1e3, "ms",
            ms=8, label="probe")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    title = ax.set_title(f"Ez xz-slice (per-frame log scale) | step {step_schedule[0]}")
    ax.legend(loc="upper right", fontsize=8)

    def update(k):
        fr = frames[k]
        floor = max(np.max(np.abs(fr)) * 1e-3, 1e-6)
        img = _to_signed_log(fr.T, floor)
        vmax = max(float(np.max(np.abs(img))), 0.5)
        im.set_data(img)
        im.set_clim(-vmax, vmax)
        title.set_text(f"Ez xz-slice (per-frame log) | step {step_schedule[k]}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=250,
                         blit=False)
    out = os.path.join(OUT_DIR, "issue31_patch_field_evolution.gif")
    anim.save(out, writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"[out] {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-mm", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--n-steps-anim", type=int, default=3000)
    ap.add_argument("--skip-farfield", action="store_true")
    ap.add_argument("--skip-anim", action="store_true")
    ap.add_argument("--skip-s11", action="store_true")
    args = ap.parse_args()

    G = _geometry(args.dx_mm)
    print(f"[cfg] analytic f_res = {G['f_analytic']/1e9:.4f} GHz")
    f_res, _, _ = run_harminv(G)
    if not args.skip_s11:
        run_s11(G, f_res_harminv=f_res)
    if not args.skip_farfield:
        run_farfield(G, f_res=f_res)
    if not args.skip_anim:
        run_field_evolution(G, n_frames=args.frames,
                            n_steps_total=args.n_steps_anim)


if __name__ == "__main__":
    main()

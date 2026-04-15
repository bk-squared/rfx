"""Issue #48 — uniform-grid patch antenna NTFF to compare with the NU run.

Same OpenEMS-validated geometry as examples/crossval/05_patch_antenna.py
but forced onto a uniform mesh. Generates harminv f_res, full 2D FFRP,
and a 3D plotly lobe + structure visualization.

Purpose: see whether the uniform path also produces a non-broadside
patch FFRP. If uniform is broadside → NU code path is the bug. If
uniform is also off-broadside → the simulation setup itself is not
producing a radiating patch mode (likely a cavity-trapped mode).
"""

from __future__ import annotations

import math
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.harminv import harminv
from rfx.farfield import compute_far_field


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "docs", "research_notes", "issue48_figs")
os.makedirs(OUT_DIR, exist_ok=True)

C0 = 2.998e8
F_DESIGN = 2.4e9


def _geom():
    eps_r = 4.3
    h_sub = 1.5e-3
    W, L = 38.0e-3, 29.5e-3
    gx, gy = 60.0e-3, 55.0e-3
    air_above, air_below = 25.0e-3, 12.0e-3
    probe_inset = 8.0e-3
    dom_x = gx + 20e-3; dom_y = gy + 20e-3
    dom_z = air_above + air_below + h_sub
    return dict(
        eps_r=eps_r, h_sub=h_sub, W=W, L=L, gx=gx, gy=gy,
        air_above=air_above, air_below=air_below, probe_inset=probe_inset,
        dom_x=dom_x, dom_y=dom_y, dom_z=dom_z,
        gx_lo=(dom_x - gx) / 2, gy_lo=(dom_y - gy) / 2,
        px_lo=dom_x / 2 - L / 2, py_lo=dom_y / 2 - W / 2,
        feed_x=(dom_x / 2 - L / 2) + probe_inset, feed_y=dom_y / 2,
    )


def _build(G, *, dx, with_port, with_ntff=False, f_ntff=F_DESIGN):
    n_cpml = 8
    sim = Simulation(freq_max=4e9, domain=(G["dom_x"], G["dom_y"], G["dom_z"]),
                     dx=dx, boundary="cpml", cpml_layers=n_cpml)
    z_gnd_lo = G["air_below"] - dx
    z_sub_lo = G["air_below"]
    z_sub_hi = G["air_below"] + G["h_sub"]
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dx
    src_z = z_sub_lo + dx * 0.5
    # Realistic FR4 with loss tangent 0.02 — crucial for a radiating
    # patch (otherwise energy is trapped in the cavity, Q ~ 1000 instead
    # of the expected ~30-60).
    eps0 = 8.8541878128e-12
    omega = 2 * np.pi * F_DESIGN
    tan_delta = 0.02
    sigma_fr4 = omega * eps0 * G["eps_r"] * tan_delta
    sim.add_material("fr4", eps_r=G["eps_r"], sigma=sigma_fr4)
    sim.add(Box((G["gx_lo"], G["gy_lo"], z_gnd_lo),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], z_sub_lo)),
            material="pec")
    sim.add(Box((G["gx_lo"], G["gy_lo"], z_sub_lo),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], z_sub_hi)),
            material="fr4")
    sim.add(Box((G["px_lo"], G["py_lo"], z_patch_lo),
                (G["px_lo"] + G["L"], G["py_lo"] + G["W"], z_patch_hi)),
            material="pec")
    if with_port:
        port_z0 = z_sub_lo + dx * 0.5
        sim.add_port(position=(G["feed_x"], G["feed_y"], port_z0),
                     component="ez", impedance=50.0,
                     extent=z_sub_hi - port_z0,
                     waveform=GaussianPulse(f0=F_DESIGN, bandwidth=0.8))
    else:
        sim.add_source(position=(G["feed_x"], G["feed_y"], src_z),
                       component="ez",
                       waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2))
        sim.add_probe(position=(G["dom_x"] / 2 + 5e-3,
                                G["dom_y"] / 2 + 5e-3, src_z),
                      component="ez")
    if with_ntff:
        margin = 3 * dx
        sim.add_ntff_box(
            corner_lo=(max(G["px_lo"] - 8e-3, margin),
                       max(G["py_lo"] - 8e-3, margin),
                       max(z_gnd_lo - 2 * dx, margin)),
            corner_hi=(min(G["px_lo"] + G["L"] + 8e-3, G["dom_x"] - margin),
                       min(G["py_lo"] + G["W"] + 8e-3, G["dom_y"] - margin),
                       min(z_patch_hi + 15e-3, G["dom_z"] - margin)),
            freqs=[f_ntff])
    return sim


def run_harminv_uniform(G, dx):
    print(f"\n=== Harminv (uniform, dx={dx*1e3:.2f}mm) ===")
    sim = _build(G, dx=dx, with_port=False)
    g = sim._build_grid()
    print(f"[cfg] cells={g.nx * g.ny * g.nz:,}")
    t0 = time.time()
    res = sim.run(num_periods=60, compute_s_params=False)
    print(f"[run] {time.time() - t0:.1f}s")
    ts = np.asarray(res.time_series).ravel()
    modes = harminv(ts[int(len(ts) * 0.3):], float(res.dt), 1.5e9, 3.5e9)
    good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-10]
    if not good:
        print(f"[warn] harminv found no modes; falling back to F_DESIGN. "
              f"All modes: {[(m.freq/1e9, m.Q, m.amplitude) for m in modes]}")
        f_res, Q = F_DESIGN, float("nan")
    else:
        best = max(good, key=lambda m: m.amplitude)
        f_res, Q = float(best.freq), float(best.Q)
    # Analytic Balanis f_res
    eps_eff = (G["eps_r"] + 1) / 2 + (G["eps_r"] - 1) / 2 * (
        1 + 12 * G["h_sub"] / G["W"]) ** (-0.5)
    dL = 0.412 * G["h_sub"] * (
        (eps_eff + 0.3) * (G["W"] / G["h_sub"] + 0.264)) / (
        (eps_eff - 0.258) * (G["W"] / G["h_sub"] + 0.8))
    f_an = C0 / (2 * (G["L"] + 2 * dL) * math.sqrt(eps_eff))
    err = 100 * abs(f_res - f_an) / f_an
    print(f"[result] f_res={f_res/1e9:.4f} GHz  Q={Q:.1f}  "
          f"error vs Balanis {f_an/1e9:.4f} GHz = {err:.2f}%")
    return f_res, Q


def run_farfield_uniform(G, dx, f_res):
    print(f"\n=== Far-field (uniform, dx={dx*1e3:.2f}mm, f={f_res/1e9:.3f} GHz) ===")
    sim = _build(G, dx=dx, with_port=True, with_ntff=True, f_ntff=f_res)
    g = sim._build_grid()
    print(f"[cfg] cells={g.nx * g.ny * g.nz:,}")
    t0 = time.time()
    res = sim.run(num_periods=40, compute_s_params=False)
    print(f"[run] {time.time() - t0:.1f}s")
    theta = np.linspace(0.01, np.pi / 2, 80)
    phi = np.linspace(0, 2 * np.pi, 161)
    ff = compute_far_field(res.ntff_data, res.ntff_box, g, theta, phi)
    E_t = np.asarray(ff.E_theta[0]); E_p = np.asarray(ff.E_phi[0])
    mag = np.sqrt(np.abs(E_t) ** 2 + np.abs(E_p) ** 2)
    mag_n = mag / np.max(mag)
    mag_db = 20 * np.log10(np.maximum(mag_n, 1e-3))
    i_p, j_p = np.unravel_index(np.argmax(mag), mag.shape)
    print(f"[peak] θ={np.degrees(theta[i_p]):.1f}°  "
          f"φ={np.degrees(phi[j_p]):.1f}°  "
          f"broadside_ratio={float(mag_n[0, 0]):.3f}")

    # 2D plot
    th_full = np.concatenate([-theta[::-1], theta])
    idx0 = int(np.argmin(np.abs(phi - 0)))
    idx_pi = int(np.argmin(np.abs(phi - np.pi)))
    idx_90 = int(np.argmin(np.abs(phi - np.pi / 2)))
    idx_270 = int(np.argmin(np.abs(phi - 3 * np.pi / 2)))
    cut_xz = np.concatenate([mag_db[::-1, idx_pi], mag_db[:, idx0]])
    cut_yz = np.concatenate([mag_db[::-1, idx_270], mag_db[:, idx_90]])

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax1.plot(th_full, cut_xz, label="xz-plane φ=0°", lw=1.5)
    ax1.plot(th_full, cut_yz, label="yz-plane φ=90°", ls="--", lw=1.5)
    ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
    ax1.set_thetamin(-90); ax1.set_thetamax(90); ax1.set_rlim(-30, 0)
    ax1.legend(loc="lower center", fontsize=8)
    ax1.set_title(f"Uniform patch FFRP (dx={dx*1e3:.2f}mm)\n"
                  f"peak θ={np.degrees(theta[i_p]):.0f}° φ={np.degrees(phi[j_p]):.0f}°")
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(mag_db, origin="lower", aspect="auto",
                    extent=[0, 360, 0, 90], cmap="viridis", vmin=-30, vmax=0)
    ax2.plot(np.degrees(phi[j_p]), np.degrees(theta[i_p]),
             "r*", ms=14, label=f"peak")
    fig.colorbar(im, ax=ax2, label="|E_far| (dB, norm)")
    ax2.set_xlabel("φ (deg)"); ax2.set_ylabel("θ (deg)")
    ax2.legend(loc="upper right")
    fig.suptitle(f"Uniform grid @ {f_res/1e9:.3f} GHz")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"uniform_ffrp_dx{dx*1e3:.2f}mm.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[out] {out}")

    # 3D plotly
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[3D] plotly missing, skipping")
        return
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    r = mag_n
    cx = (G["px_lo"] + G["L"] / 2) * 1e3
    cy = (G["py_lo"] + G["W"] / 2) * 1e3
    cz = (G["air_below"] + G["h_sub"]) * 1e3
    scale = 40.0
    X = cx + scale * r * np.sin(TH) * np.cos(PH)
    Y = cy + scale * r * np.sin(TH) * np.sin(PH)
    Z = cz + scale * r * np.cos(TH)
    fig3d = go.Figure()
    fig3d.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=20 * np.log10(np.maximum(r, 1e-2)),
        colorscale="Viridis", cmin=-40, cmax=0, opacity=0.75,
        colorbar=dict(title="|E_far| (dB, norm)"),
        name=f"|E_far| @ {f_res/1e9:.3f} GHz"))

    def _cuboid(x0, y0, z0, w, d, h, color, opacity, name):
        xs = [x0, x0 + w, x0 + w, x0, x0, x0 + w, x0 + w, x0]
        ys = [y0, y0, y0 + d, y0 + d, y0, y0, y0 + d, y0 + d]
        zs = [z0, z0, z0, z0, z0 + h, z0 + h, z0 + h, z0 + h]
        i = [0, 0, 1, 1, 2, 2, 4, 4, 0, 0, 1, 2]
        j = [1, 2, 2, 5, 3, 6, 5, 6, 4, 5, 5, 3]
        k = [2, 3, 5, 6, 6, 7, 6, 7, 5, 1, 6, 7]
        return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                         color=color, opacity=opacity, name=name,
                         showlegend=True, flatshading=True)

    fig3d.add_trace(_cuboid(G["gx_lo"] * 1e3, G["gy_lo"] * 1e3,
                            (G["air_below"] - dx) * 1e3,
                            G["gx"] * 1e3, G["gy"] * 1e3, dx * 1e3,
                            "black", 0.5, "ground PEC"))
    fig3d.add_trace(_cuboid(G["gx_lo"] * 1e3, G["gy_lo"] * 1e3,
                            G["air_below"] * 1e3, G["gx"] * 1e3,
                            G["gy"] * 1e3, G["h_sub"] * 1e3,
                            "tan", 0.18, "FR4"))
    fig3d.add_trace(_cuboid(G["px_lo"] * 1e3, G["py_lo"] * 1e3,
                            (G["air_below"] + G["h_sub"]) * 1e3,
                            G["L"] * 1e3, G["W"] * 1e3, dx * 1e3,
                            "goldenrod", 0.85, "patch PEC"))
    fig3d.update_layout(
        title=f"Uniform (dx={dx*1e3:.2f}mm) @ {f_res/1e9:.3f} GHz — "
              f"peak θ={np.degrees(theta[i_p]):.0f}°",
        scene=dict(xaxis=dict(title="x (mm)"),
                   yaxis=dict(title="y (mm)"),
                   zaxis=dict(title="z (mm)"),
                   aspectmode="data"),
    )
    html_out = os.path.join(OUT_DIR, f"uniform_ffrp_dx{dx*1e3:.2f}mm.html")
    fig3d.write_html(html_out, include_plotlyjs="cdn")
    print(f"[out] {html_out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-mm", type=float, default=0.5)
    args = ap.parse_args()
    G = _geom()
    dx = args.dx_mm * 1e-3
    f_res, Q = run_harminv_uniform(G, dx)
    run_farfield_uniform(G, dx, f_res)


if __name__ == "__main__":
    main()

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
def _geometry(dx_mm, gx_mm=60.0, gy_mm=55.0):
    f_design = 2.4e9
    eps_r_fr4 = 4.3
    h_sub = 1.5e-3
    W, L = 38.0e-3, 29.5e-3
    gx, gy = gx_mm * 1e-3, gy_mm * 1e-3
    air_above, air_below = 25.0e-3, 12.0e-3
    probe_inset = 8.0e-3

    dx = dx_mm * 1e-3
    n_cpml = 8
    n_sub = 6
    dz_sub = h_sub / n_sub

    # Meep/OpenEMS convention (issue #48): metal surfaces must sit on
    # cell edges with symmetric neighbouring cells. Build dz EXPLICITLY
    # so that substrate cells align with z=air_below..(air_below+h_sub).
    #
    # Transition cells (coarse dx → fine dz_sub) consume from the
    # adjacent coarse region so the fine block lies at the expected
    # physical z. Ratio ~= (dz_sub/dx)^(1/ntrans) per step.
    n_trans = 5
    trans_down = np.geomspace(dx, dz_sub, n_trans + 2, dtype=np.float64)[1:-1]
    trans_up = trans_down[::-1]
    trans_sum = float(trans_down.sum())
    # Use coarse cells for the bulk of air_below, ending with transitions
    # that land exactly at z = air_below. Same for air_above (starts
    # with transitions).
    n_coarse_below = max(0, int(round((air_below - trans_sum) / dx)))
    n_coarse_above = max(0, int(round((air_above - trans_sum) / dx)))
    # Fine block spans (1 ground) + (n_sub substrate) + (1 patch) cells
    # all at dz_sub so the metal planes sit on cell edges with
    # symmetric neighbours (Meep/OpenEMS convention).
    n_fine_total = n_sub + 2
    dz_profile = np.concatenate([
        np.full(n_coarse_below, dx), trans_down,
        np.full(n_fine_total, dz_sub),
        trans_up, np.full(n_coarse_above, dx),
    ]).astype(np.float64)
    z_edges = np.concatenate([[0.0], np.cumsum(dz_profile)])
    k_ground = n_coarse_below + n_trans             # ground cell index
    k_sub_lo = k_ground + 1                         # first substrate cell
    k_sub_hi = k_sub_lo + n_sub                     # one past last substrate cell
    k_patch = k_sub_hi                              # patch cell index
    z_gnd_lo = float(z_edges[k_ground])
    z_sub_lo = float(z_edges[k_sub_lo])
    z_sub_hi = float(z_edges[k_sub_hi])
    z_patch_lo = z_sub_hi
    z_patch_hi = float(z_edges[k_patch + 1])
    src_z = z_sub_lo + dz_sub * 2.5

    dom_x = gx + 20e-3
    dom_y = gy + 20e-3
    g_box = dict(
        dom_x=dom_x, dom_y=dom_y, dx_mm=dx_mm, dz_profile=dz_profile,
        n_cpml=n_cpml, dz_sub=dz_sub, f_design=f_design,
        gx_lo=(dom_x - gx) / 2, gy_lo=(dom_y - gy) / 2, gx=gx, gy=gy,
        z_gnd_lo=z_gnd_lo, z_sub_lo=z_sub_lo,
        z_sub_hi=z_sub_hi, z_patch_lo=z_patch_lo,
        z_patch_hi=z_patch_hi,
        px_lo=dom_x / 2 - L / 2, py_lo=dom_y / 2 - W / 2, L=L, W=W,
        feed_x=(dom_x / 2 - L / 2) + probe_inset, feed_y=dom_y / 2,
        src_z=src_z, eps_r_fr4=eps_r_fr4,
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
    # Lossy FR4 (tan δ = 0.02). Essential for physical radiation —
    # lossless dielectric traps energy in the patch cavity (Q ~ 1000)
    # and produces a near-grazing NTFF pattern (issue #48 deep dive).
    eps0 = 8.8541878128e-12
    tan_delta = 0.02
    sigma_fr4 = 2 * np.pi * G["f_design"] * eps0 * G["eps_r_fr4"] * tan_delta
    sim.add_material("fr4", eps_r=G["eps_r_fr4"], sigma=sigma_fr4)
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
        # NTFF box MUST be strictly inside the CPML region (issue #48).
        # Interior x range is [0, dom_x] and CPML is padded outside; the
        # safety buffer keeps us well off the boundary.
        dx_m = G["dx_mm"] * 1e-3
        safety_xy = 3 * dx_m   # 3-cell buffer inside interior
        safety_z = 2 * dx_m
        px_lo, py_lo = G["px_lo"], G["py_lo"]
        px_hi = px_lo + G["L"]; py_hi = py_lo + G["W"]
        # Tight box around the patch + margin for the near field.
        ntff_lo_x = max(px_lo - 8e-3, safety_xy)
        ntff_hi_x = min(px_hi + 8e-3, G["dom_x"] - safety_xy)
        ntff_lo_y = max(py_lo - 8e-3, safety_xy)
        ntff_hi_y = min(py_hi + 8e-3, G["dom_y"] - safety_xy)
        # Enclose from just below ground to ~15 mm above the patch.
        ntff_lo_z = max(G["z_gnd_lo"] - 2 * G["dz_sub"], safety_z)
        dom_z = float(np.sum(G["dz_profile"]))
        ntff_hi_z = min(G["z_patch_hi"] + 15e-3, dom_z - safety_z)
        sim.add_ntff_box(
            corner_lo=(ntff_lo_x, ntff_lo_y, ntff_lo_z),
            corner_hi=(ntff_hi_x, ntff_hi_y, ntff_hi_z),
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
# Step 4 — 3D far-field lobe + structure (issue #49)
# -----------------------------------------------------------------------------
def run_farfield_3d(G, *, f_res):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[3D-FFRP] plotly missing, skipping.")
        return
    print(f"\n=== [4/5] 3D far-field + structure at {f_res/1e9:.3f} GHz ===")
    sim = _build(G, with_port=True, with_ntff=True, ntff_freqs=[f_res])
    t0 = time.time()
    res = sim.run(num_periods=40, compute_s_params=False)
    print(f"[run] done in {time.time() - t0:.1f}s")

    from rfx.farfield import compute_far_field
    theta = np.linspace(0.01, np.pi / 2, 60)  # avoid theta=0 singularity
    phi = np.linspace(0, 2 * np.pi, 121)
    grid = sim._build_nonuniform_grid()
    ef = compute_far_field(res.ntff_data, res.ntff_box, grid, theta, phi)
    E_t = np.asarray(ef.E_theta[0]); E_p = np.asarray(ef.E_phi[0])
    mag = np.sqrt(np.abs(E_t) ** 2 + np.abs(E_p) ** 2)
    mag_norm = mag / np.max(mag)

    # Sphere coords deformed by magnitude → farfield surface.
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    r = mag_norm
    # Centre the lobe above the patch (origin at the patch centre, mm).
    cx = (G["px_lo"] + G["L"] / 2) * 1e3
    cy = (G["py_lo"] + G["W"] / 2) * 1e3
    cz = G["z_patch_hi"] * 1e3
    scale = 40.0  # mm — visual size of the lobe at r=1
    X = cx + scale * r * np.sin(TH) * np.cos(PH)
    Y = cy + scale * r * np.sin(TH) * np.sin(PH)
    Z = cz + scale * r * np.cos(TH)

    fig = go.Figure()

    # Structure boxes (PEC ground, FR4 substrate, PEC patch) as semi-
    # transparent cuboids. Plotly Mesh3d expects vertex + face triples.
    def _cuboid(x0, y0, z0, w, d, h, color, opacity, name):
        # 8 vertices of a box
        xs = [x0, x0 + w, x0 + w, x0, x0, x0 + w, x0 + w, x0]
        ys = [y0, y0, y0 + d, y0 + d, y0, y0, y0 + d, y0 + d]
        zs = [z0, z0, z0, z0, z0 + h, z0 + h, z0 + h, z0 + h]
        # 12 triangles
        i = [0, 0, 1, 1, 2, 2, 4, 4, 0, 0, 1, 2]
        j = [1, 2, 2, 5, 3, 6, 5, 6, 4, 5, 5, 3]
        k = [2, 3, 5, 6, 6, 7, 6, 7, 5, 1, 6, 7]
        return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                         color=color, opacity=opacity, name=name, showlegend=True,
                         flatshading=True)

    gx_lo, gy_lo = G["gx_lo"] * 1e3, G["gy_lo"] * 1e3
    fig.add_trace(_cuboid(gx_lo, gy_lo, G["z_gnd_lo"] * 1e3,
                          G["gx"] * 1e3, G["gy"] * 1e3, G["dz_sub"] * 1e3,
                          "black", 0.5, "ground PEC"))
    fig.add_trace(_cuboid(gx_lo, gy_lo, G["z_sub_lo"] * 1e3,
                          G["gx"] * 1e3, G["gy"] * 1e3,
                          (G["z_sub_hi"] - G["z_sub_lo"]) * 1e3,
                          "tan", 0.18, "FR4 substrate"))
    fig.add_trace(_cuboid(G["px_lo"] * 1e3, G["py_lo"] * 1e3,
                          G["z_patch_lo"] * 1e3, G["L"] * 1e3, G["W"] * 1e3,
                          G["dz_sub"] * 1e3, "goldenrod", 0.85, "patch PEC"))

    # Far-field lobe
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, surfacecolor=20 * np.log10(np.maximum(mag_norm, 1e-2)),
        colorscale="Viridis", cmin=-40, cmax=0,
        colorbar=dict(title="|E_far| (dB, norm)"),
        opacity=0.7, name=f"|E_far| @ {f_res/1e9:.3f} GHz",
    ))

    # Source / probe markers
    fig.add_trace(go.Scatter3d(
        x=[G["feed_x"] * 1e3], y=[G["feed_y"] * 1e3], z=[G["src_z"] * 1e3],
        mode="markers", marker=dict(size=5, color="green", symbol="diamond"),
        name="feed port"))

    # NTFF integration box (wireframe) — visible=legendonly by default so
    # it doesn't clutter the scene; click the legend to toggle.
    ntff_lo = res.ntff_box
    grid = sim._build_nonuniform_grid()
    dx_m = float(grid.dx); dy_m = float(getattr(grid, "dy", grid.dx))
    dz_arr = np.asarray(grid.dz)
    z_edges = np.concatenate([[0], np.cumsum(dz_arr)]) - np.cumsum(dz_arr)[G["n_cpml"] - 1]
    nx0 = (ntff_lo.i_lo - G["n_cpml"]) * dx_m * 1e3
    nx1 = (ntff_lo.i_hi - G["n_cpml"]) * dx_m * 1e3
    ny0 = (ntff_lo.j_lo - G["n_cpml"]) * dy_m * 1e3
    ny1 = (ntff_lo.j_hi - G["n_cpml"]) * dy_m * 1e3
    nz0 = z_edges[ntff_lo.k_lo] * 1e3
    nz1 = z_edges[ntff_lo.k_hi] * 1e3
    # 12 edges of a box
    box_lines_x, box_lines_y, box_lines_z = [], [], []
    corners = [(nx0, ny0, nz0), (nx1, ny0, nz0), (nx1, ny1, nz0), (nx0, ny1, nz0),
               (nx0, ny0, nz1), (nx1, ny0, nz1), (nx1, ny1, nz1), (nx0, ny1, nz1)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        box_lines_x += [corners[a][0], corners[b][0], None]
        box_lines_y += [corners[a][1], corners[b][1], None]
        box_lines_z += [corners[a][2], corners[b][2], None]
    fig.add_trace(go.Scatter3d(
        x=box_lines_x, y=box_lines_y, z=box_lines_z, mode="lines",
        line=dict(color="orange", width=3),
        name="NTFF box", visible="legendonly"))

    # CPML boundary (outer domain wireframe). Physical interior starts
    # at (0, 0, 0) and extends to (dom_x, dom_y, dom_z).
    dom_z = float(np.sum(G["dz_profile"]))
    cx0, cy0, cz0 = 0.0, 0.0, 0.0
    cx1, cy1, cz1 = G["dom_x"] * 1e3, G["dom_y"] * 1e3, dom_z * 1e3
    pml_corners = [(cx0, cy0, cz0), (cx1, cy0, cz0), (cx1, cy1, cz0), (cx0, cy1, cz0),
                   (cx0, cy0, cz1), (cx1, cy0, cz1), (cx1, cy1, cz1), (cx0, cy1, cz1)]
    pml_x, pml_y, pml_z = [], [], []
    for a, b in edges:
        pml_x += [pml_corners[a][0], pml_corners[b][0], None]
        pml_y += [pml_corners[a][1], pml_corners[b][1], None]
        pml_z += [pml_corners[a][2], pml_corners[b][2], None]
    fig.add_trace(go.Scatter3d(
        x=pml_x, y=pml_y, z=pml_z, mode="lines",
        line=dict(color="purple", width=2, dash="dash"),
        name="CPML outer domain", visible="legendonly"))

    # Peak direction annotation
    i_peak, j_peak = np.unravel_index(np.argmax(mag), mag.shape)
    th_peak, ph_peak = theta[i_peak], phi[j_peak]
    xp = cx + scale * 1.1 * np.sin(th_peak) * np.cos(ph_peak)
    yp = cy + scale * 1.1 * np.sin(th_peak) * np.sin(ph_peak)
    zp = cz + scale * 1.1 * np.cos(th_peak)
    fig.add_trace(go.Scatter3d(
        x=[cx, xp], y=[cy, yp], z=[cz, zp],
        mode="lines+markers",
        line=dict(color="red", width=5),
        marker=dict(size=[3, 6], color="red"),
        name=f"peak θ={np.degrees(th_peak):.0f}°, φ={np.degrees(ph_peak):.0f}°"))

    fig.update_layout(
        title=f"3D far-field + structure @ {f_res/1e9:.3f} GHz "
              f"(peak θ={np.degrees(th_peak):.0f}°, φ={np.degrees(ph_peak):.0f}°)",
        scene=dict(
            xaxis=dict(title="x (mm)"), yaxis=dict(title="y (mm)"),
            zaxis=dict(title="z (mm)"), aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    html_out = os.path.join(OUT_DIR, "issue31_patch_farfield_3d.html")
    fig.write_html(html_out, include_plotlyjs="cdn")
    print(f"[out] {html_out}")
    # Optional static snapshot if kaleido is installed
    try:
        png_out = os.path.join(OUT_DIR, "issue31_patch_farfield_3d.png")
        fig.write_image(png_out, width=1100, height=800)
        print(f"[out] {png_out}")
    except Exception as e:
        print(f"[3D-FFRP] PNG snapshot skipped ({e}); HTML is interactive.")


# -----------------------------------------------------------------------------
# Step 5 — Field evolution GIF with per-frame vmax + geometry outlines
# -----------------------------------------------------------------------------
def run_field_evolution(G, *, n_frames, n_steps_total):
    print("\n=== [5/5] Field evolution GIF ===")
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
    ap.add_argument("--gx-mm", type=float, default=60.0)
    ap.add_argument("--gy-mm", type=float, default=55.0)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--n-steps-anim", type=int, default=3000)
    ap.add_argument("--skip-farfield", action="store_true")
    ap.add_argument("--skip-3d", action="store_true")
    ap.add_argument("--skip-anim", action="store_true")
    ap.add_argument("--skip-s11", action="store_true")
    args = ap.parse_args()

    G = _geometry(args.dx_mm, gx_mm=args.gx_mm, gy_mm=args.gy_mm)
    print(f"[cfg] analytic f_res = {G['f_analytic']/1e9:.4f} GHz")
    f_res, _, _ = run_harminv(G)
    if not args.skip_s11:
        run_s11(G, f_res_harminv=f_res)
    if not args.skip_farfield:
        run_farfield(G, f_res=f_res)
    if not args.skip_3d:
        run_farfield_3d(G, f_res=f_res)
    if not args.skip_anim:
        run_field_evolution(G, n_frames=args.frames,
                            n_steps_total=args.n_steps_anim)


if __name__ == "__main__":
    main()

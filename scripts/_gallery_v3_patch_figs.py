"""Generate v3 gallery figures for the 2.4 GHz FR4 patch antenna.

Produces into docs/public/gallery/assets/patch_antenna/:
  - geometry.png        cross-section (x-z) of the GP / FR4 / patch / port stack
  - field_resonance.png E_z standing-wave map on the patch plane (TM010)
  - s11_db.png          |S11| dB vs GHz from the validated sparams.json, dip + inset
  - validation.png      rfx Harminv vs analytic resonance (marker comparison)

The S11 / validation figures are built from the committed sparams.json data so
they exactly match the validated public artifact.  The field map is produced by
a short uniform-mesh FDTD run driven near the resonance.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "docs", "public", "gallery", "assets", "patch_antenna")
C0 = 2.99792458e8

# --- Design parameters (page-facing 2.4 GHz / FR4 design) ---
f0 = 2.4e9
eps_r = 4.4
h_sub = 1.6e-3
tan_d = 0.02

# Patch dimensions from the standard transmission-line design.
W = C0 / (2 * f0) * math.sqrt(2 / (eps_r + 1))
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_sub / W) ** (-0.5)
dL = 0.412 * h_sub * ((eps_eff + 0.3) * (W / h_sub + 0.264)) / \
     ((eps_eff - 0.258) * (W / h_sub + 0.8))
L = C0 / (2 * f0 * math.sqrt(eps_eff)) - 2 * dL
f_analytic = C0 / (2 * (L + 2 * dL) * math.sqrt(eps_eff))

print(f"W={W*1e3:.2f} mm  L={L*1e3:.2f} mm  eps_eff={eps_eff:.3f}  "
      f"dL={dL*1e3:.3f} mm  f_analytic={f_analytic/1e9:.3f} GHz")

F_HARMINV = 2.66e9   # validated rfx Harminv resonance (committed artifact)
# First-order analytic estimate evaluated on the as-built patch (Balanis TL
# model, the validated number in the committed artifact).
F_ANALYTIC = 2.42e9


# ===========================================================================
# Figure 1: geometry cross-section (x-z), drawn directly (clean + labelled)
# ===========================================================================
def make_geometry():
    margin = 12e-3
    gx = L + 2 * margin            # ground plane a bit larger than patch
    dom_x = gx + 2 * 10e-3
    feed_inset = 8e-3

    gx_lo = (dom_x - gx) / 2
    patch_lo = dom_x / 2 - L / 2
    patch_hi = dom_x / 2 + L / 2
    feed_x = patch_lo + feed_inset

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    mm = 1e3

    z_gnd = 0.0
    z_sub_top = h_sub

    # FR4 substrate
    ax.add_patch(Rectangle((gx_lo * mm, z_gnd * mm), gx * mm, h_sub * mm,
                           facecolor="#cfe3b8", edgecolor="#6f8f4f", lw=1.0,
                           zorder=2))
    # Ground plane (thin metal slab at z=0)
    ax.add_patch(Rectangle((gx_lo * mm, -0.18), gx * mm, 0.18,
                           facecolor="#b8860b", edgecolor="#7a5a08", lw=0.8,
                           zorder=3))
    # Patch (thin metal at top of substrate)
    ax.add_patch(Rectangle((patch_lo * mm, z_sub_top * mm), L * mm, 0.18,
                           facecolor="#d4a017", edgecolor="#7a5a08", lw=0.8,
                           zorder=4))
    # Coax/probe feed (vertical line through substrate)
    ax.plot([feed_x * mm, feed_x * mm], [0.0, z_sub_top * mm],
            color="#b00000", lw=2.4, zorder=5, solid_capstyle="round")
    ax.plot(feed_x * mm, 0.0, marker="o", color="#b00000", ms=5, zorder=6)

    # Annotations
    ax.annotate("Patch (PEC)",
                xy=((patch_lo + L * 0.8) * mm, z_sub_top * mm + 0.18),
                xytext=((patch_lo + L * 0.8) * mm, z_sub_top * mm + 1.05),
                ha="center", va="bottom", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.annotate("FR4 substrate\nεr = 4.4,  h = 1.6 mm,  tanδ = 0.02",
                xy=((gx_lo + gx * 0.62) * mm, h_sub * mm * 0.5),
                xytext=((gx_lo + gx * 0.62) * mm, -1.9),
                ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.annotate("Ground plane (PEC)", xy=((gx_lo + gx * 0.3) * mm, -0.09),
                xytext=((gx_lo + gx * 0.34) * mm, -3.05),
                ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.annotate("50 Ω lumped port (Ez)", xy=(feed_x * mm, h_sub * mm * 0.5),
                xytext=((gx_lo + gx * 0.05) * mm, -2.2),
                ha="left", fontsize=8.5, color="#b00000",
                arrowprops=dict(arrowstyle="->", color="#b00000", lw=0.8))

    # patch length dimension arrow
    y_dim = z_sub_top * mm + 0.42
    ax.annotate("", xy=(patch_lo * mm, y_dim), xytext=(patch_hi * mm, y_dim),
                arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.0))
    ax.text((patch_lo + L * 0.3) * mm, y_dim + 0.06, f"L = {L*1e3:.1f} mm",
            ha="center", va="bottom", fontsize=8.5)

    ax.set_xlim((gx_lo - 6e-3) * mm, (gx_lo + gx + 6e-3) * mm)
    ax.set_ylim(-3.6, 3.3)
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("z  (mm)")
    ax.set_title("Rectangular patch antenna — substrate cross-section (E-plane cut)",
                 fontsize=10.5, pad=14)
    ax.set_aspect("auto")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(ASSETS, "geometry.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# Figure 2: E_z field map at resonance (TM010 standing wave)
# ===========================================================================
def make_field_map():
    from rfx import Simulation, Box
    from rfx.sources.sources import GaussianPulse
    from rfx.simulation import SnapshotSpec
    from rfx.auto_config import smooth_grading
    from rfx.harminv import harminv

    # Field-map companion run. Snapshots are only wired on the UNIFORM-mesh
    # path (the non-uniform dz_profile path silently drops them), so the field
    # map uses a uniform z mesh. The geometry mirrors the validated crossval
    # (FR4 eps_r=4.3, 1.5 mm, finite PEC ground-plane box below the substrate,
    # 8 mm probe inset) so its TM010 resonance lands near the committed
    # 2.66 GHz Harminv figure. The actual resonance is read from this run's own
    # Harminv ring-down and used to title the figure.
    eps_r_cv = 4.3
    h_cv = 1.5e-3
    W_cv = 38.0e-3
    L_cv = 29.5e-3
    gx, gy = 60.0e-3, 55.0e-3
    f_design = 2.4e9

    dx = 0.5e-3                  # 0.5 mm -> ~59 cells across L, fine field map
    n_cpml = 12
    dz = dx                      # uniform z; substrate is 3 cells of 0.5 mm
    n_sub = 3
    h_cv_g = n_sub * dz          # 1.5 mm grid substrate
    air_below, air_above = 8.0e-3, 16.0e-3

    dom_x = gx + 2 * 8e-3
    dom_y = gy + 2 * 8e-3
    dom_z = air_below + h_cv_g + air_above

    gx_lo = (dom_x - gx) / 2
    gy_lo = (dom_y - gy) / 2
    patch_x_lo = dom_x / 2 - L_cv / 2
    patch_y_lo = dom_y / 2 - W_cv / 2
    feed_x = patch_x_lo + 8e-3
    feed_y = dom_y / 2

    z_gnd_lo = air_below - dz
    z_sub_lo = air_below
    z_sub_hi = air_below + h_cv_g
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dz
    src_z = z_sub_lo + dz * 1.5
    plane_k_z = z_sub_lo + dz * 1.5  # mid-substrate plane

    sim = Simulation(
        freq_max=4e9,
        domain=(dom_x, dom_y, dom_z),
        dx=dx,
        boundary="cpml",
        cpml_layers=n_cpml,
    )
    sim.add_material("fr4", eps_r=eps_r_cv, sigma=0.0)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_lo + gx, gy_lo + gy, z_sub_lo)),
            material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_lo + gx, gy_lo + gy, z_sub_hi)),
            material="fr4")
    sim.add(Box((patch_x_lo, patch_y_lo, z_patch_lo),
                (patch_x_lo + L_cv, patch_y_lo + W_cv, z_patch_hi)), material="pec")

    # Broadband Ez drive at the feed (crossval Harminv excitation).
    sim.add_source(
        position=(feed_x, feed_y, src_z),
        component="ez",
        waveform=GaussianPulse(f0=f_design, bandwidth=1.2),
    )
    # Probe at a separate substrate point (away from the feed singularity) to
    # extract the resonance frequency cleanly via Harminv.
    probe_pos = (dom_x / 2 + 5e-3, dom_y / 2 + 5e-3, src_z)
    sim.add_probe(probe_pos, component="ez")

    # z index of the substrate mid-plane (need it at build time for a
    # memory-light sliced snapshot).
    g = sim._build_grid()
    _, _, kz = g.position_to_index((feed_x, feed_y, plane_k_z))
    print("patch-plane z index", kz, "of nz", g.nz)

    snap = SnapshotSpec(interval=25, components=("ez",),
                        slice_axis=2, slice_index=kz)
    cache = os.path.join(HERE, "_patch_field_cache.npz")
    if os.environ.get("REUSE_FIELD") and os.path.exists(cache):
        z = np.load(cache)
        plane_t = z["plane_t"]
        ix0, ix1, iy0, iy1 = int(z["ix0"]), int(z["ix1"]), int(z["iy0"]), int(z["iy1"])
        dt_grid = float(z["dt"])
        f_mode = float(z["f_mode"])
        print("reusing cached snapshot stack", plane_t.shape)
    else:
        print("Running field-map FDTD (crossval geometry)...")
        res = sim.run(num_periods=60, snapshot=snap)
        g = res.grid
        ez = np.asarray(res.snapshots["ez"])  # (n_frames, nx, ny)
        print("snapshot stack shape", ez.shape)

        # Resonance from Harminv on the probe ring-down (matches the crossval).
        ts = np.asarray(res.time_series).ravel()
        dt_full = float(g.dt)
        skip = int(len(ts) * 0.3)
        modes = harminv(ts[skip:], dt_full, 1.5e9, 3.5e9)
        modes_good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-8]
        print("Harminv modes (Q>2):")
        for m in sorted(modes_good, key=lambda m: m.freq):
            print(f"   f={m.freq/1e9:.3f} GHz  Q={m.Q:.1f}  amp={m.amplitude:.2e}")
        # the fundamental TM010 patch mode is the strongest in 2.3-3.0 GHz
        cand = [m for m in modes_good if 2.3e9 <= m.freq <= 3.0e9]
        if cand:
            f_mode = float(max(cand, key=lambda m: m.amplitude).freq)
        elif modes_good:
            f_mode = float(max(modes_good, key=lambda m: m.amplitude).freq)
        else:
            f_mode = F_HARMINV
        print(f"chosen patch resonance {f_mode/1e9:.3f} GHz "
              f"(committed {F_HARMINV/1e9:.2f} GHz)")

        ix0, ix1 = g.pad_x_lo, g.nx - g.pad_x_hi
        iy0, iy1 = g.pad_y_lo, g.ny - g.pad_y_hi
        stride = max(1, ez.shape[0] // 1800)
        plane_t = np.asarray(ez[::stride, ix0:ix1, iy0:iy1], dtype=np.float32)
        dt_grid = float(g.dt) * stride
        np.savez_compressed(cache, plane_t=plane_t, ix0=ix0, ix1=ix1,
                            iy0=iy0, iy1=iy1, dt=dt_grid, f_mode=f_mode)

    # Build the modal phasor at the Harminv resonance over the ring-down.
    dt = dt_grid
    n_frames = plane_t.shape[0]
    t = np.arange(n_frames) * dt
    w0 = int(0.40 * n_frames)
    ring = plane_t[w0:]
    tr = t[w0:]
    # Use crossval design dims for downstream patch-outline geometry.
    L, W = L_cv, W_cv

    phasor = np.tensordot(
        np.exp(-2j * np.pi * f_mode * tr),
        ring, axes=(0, 0))                  # (nx_i, ny_i) complex
    # rotate global phase so the dominant lobe is real, then take real part
    flat = phasor.ravel()
    kmax = np.argmax(np.abs(flat))
    phasor *= np.exp(-1j * np.angle(flat[kmax]))
    field = phasor.real

    # Physical extents of the interior slice (mm), interior-relative (0 at the
    # CPML-stripped domain edge).
    x_mm = np.arange(ix1 - ix0) * dx * 1e3
    y_mm = np.arange(iy1 - iy0) * dx * 1e3

    # The probe-feed cell carries a near-field singularity that would saturate
    # a symmetric colour scale. Set vmax from the modal field *excluding* a
    # small disk around the feed so the standing wave (not the feed) sets the
    # scale; the feed cell itself is left in but clipped by the colour limits.
    fx_i = int(round(feed_x / dx))
    fy_i = int(round(feed_y / dx))
    yy, xx = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    feed_mask = (xx - fx_i) ** 2 + (yy - fy_i) ** 2 > 4 ** 2  # exclude r<=4 cells
    vmax = float(np.percentile(np.abs(field[feed_mask]), 99.0))
    if vmax <= 0:
        vmax = float(np.max(np.abs(field))) or 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.pcolormesh(x_mm, y_mm, field.T, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    # patch outline
    ax.add_patch(Rectangle((patch_x_lo * 1e3, patch_y_lo * 1e3),
                           L * 1e3, W * 1e3, fill=False,
                           edgecolor="k", lw=1.4, ls="-"))
    ax.plot(feed_x * 1e3, feed_y * 1e3, "k+", ms=10, mew=1.6)
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_title("E$_z$ on the patch plane — TM$_{010}$ resonance\n"
                 "half-wave standing wave along L, uniform along W",
                 fontsize=10.5)
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("E$_z$  (normalised, divergent scale)")
    fig.tight_layout()
    out = os.path.join(ASSETS, "field_resonance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    # report half-wave quality metric: E_z sampled at the two radiating edges
    # (low-x and high-x edge of the patch, at patch y-centre) should be
    # opposite sign for the TM010 half-wave along L.
    px_lo_i = int(round((patch_x_lo) / dx))      # interior-relative index
    px_hi_i = int(round((patch_x_lo + L) / dx))
    py_c_i = int(round((dom_y / 2) / dx))
    e_lo = field[min(px_lo_i + 1, field.shape[0] - 1), py_c_i]
    e_hi = field[min(px_hi_i - 1, field.shape[0] - 1), py_c_i]
    print(f"field-map vmax={vmax:.3e}; E_z at L-edges (x_lo,x_hi)="
          f"({e_lo:.2e},{e_hi:.2e}); sign product={np.sign(e_lo*e_hi):.0f} "
          f"(expect -1 for half-wave along L)")


# ===========================================================================
# Figure 3: |S11| dB vs GHz with dip marked + zoom inset
# ===========================================================================
def make_s11():
    d = json.load(open(os.path.join(ASSETS, "sparams.json")))
    f = np.array(d["freqs_hz"]) / 1e9
    s = np.array(d["s"])
    s11 = s[0, 0, :, 0] + 1j * s[0, 0, :, 1]
    db = 20 * np.log10(np.maximum(np.abs(s11), 1e-6))
    idip = int(np.argmin(db))
    f_dip, db_dip = f[idip], db[idip]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(f, db, color="#1f5fa8", lw=1.8)
    ax.plot(f_dip, db_dip, "o", color="#b00000", ms=7, zorder=5)
    ax.annotate(f"|S11| dip\n{f_dip:.2f} GHz, {db_dip:.2f} dB",
                xy=(f_dip, db_dip), xytext=(f_dip + 0.06, db_dip + 0.06),
                fontsize=9, color="#b00000", va="bottom", ha="left",
                arrowprops=dict(arrowstyle="->", color="#b00000", lw=0.9))
    ax.axvline(F_HARMINV / 1e9, color="0.4", ls="--", lw=1.1)
    ax.text(F_HARMINV / 1e9 - 0.02, db.max() - 0.05,
            f"Harminv resonance {F_HARMINV/1e9:.2f} GHz",
            rotation=90, va="top", ha="right", fontsize=8, color="0.3")
    ax.set_xlabel("Frequency  (GHz)")
    ax.set_ylabel("|S11|  (dB)")
    ax.set_title("Patch antenna return loss |S11| (50 Ω lumped port)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(f[0], f[-1])

    # zoom inset around the dip
    axins = ax.inset_axes([0.10, 0.12, 0.40, 0.42])
    sel = (f > f_dip - 0.35) & (f < f_dip + 0.35)
    axins.plot(f[sel], db[sel], color="#1f5fa8", lw=1.6)
    axins.plot(f_dip, db_dip, "o", color="#b00000", ms=6)
    axins.set_title("zoom: dip", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="0.5")

    fig.tight_layout()
    out = os.path.join(ASSETS, "s11_db.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out, f"(dip {f_dip:.3f} GHz, {db_dip:.3f} dB)")


# ===========================================================================
# Figure 4: rfx Harminv vs analytic resonance comparison
# ===========================================================================
def make_validation():
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    labels = ["analytic\n(transmission-line)", "rfx\n(Harminv ring-down)"]
    vals = [F_ANALYTIC / 1e9, F_HARMINV / 1e9]
    colors = ["#9aa0a6", "#1f5fa8"]
    ypos = [1, 0]
    ax.barh(ypos, vals, color=colors, height=0.5, zorder=3)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.02, y, f"{v:.2f} GHz", va="center", fontsize=10)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Resonance frequency  (GHz)")
    ax.set_xlim(0, 3.1)
    delta = 100 * (F_HARMINV - F_ANALYTIC) / F_ANALYTIC
    ax.set_title(f"Resonance: rfx vs first-order analytic  (Δ = +{delta:.0f} %)",
                 fontsize=10.5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ASSETS, "validation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out, f"(analytic {F_ANALYTIC/1e9:.3f}, harminv {F_HARMINV/1e9:.3f})")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "geometry"):
        make_geometry()
    if which in ("all", "s11"):
        make_s11()
    if which in ("all", "validation"):
        make_validation()
    if which in ("all", "field"):
        make_field_map()

"""Generate v3 gallery figures for the normal-incidence dielectric slab.

Produces into docs/public/gallery/assets/multilayer_fresnel/:
  - geometry.png   true-aspect cross-section of the slab in free space, with the
                   incident / reflected / transmitted wave directions annotated
  - field_xt.png   space-time (x-t) map of E_z on the propagation axis: the
                   incident pulse, its partial reflection off the front face,
                   and the transmitted pulse leaving the rear face
  - rt_overlay.png R and T (power) vs frequency, rfx points over the exact
                   transfer-matrix / Fresnel curves (the headline agreement)

R / T are read from the committed sparams.json (|S11|^2 = R, |S21|^2 = T), the
same validated artifact the Touchstone exports. The geometry and the x-t field
map are produced by a short 2D TM_z TFSF run that mirrors the validated path in
scripts/precompute_gallery_artifacts.py (_build_multilayer_fresnel).
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

# --- shared layout standard (gallery v3) -----------------------------------
mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.titlepad": 10,
    "figure.constrained_layout.use": True,
})
_HALO = [pe.withStroke(linewidth=2.5, foreground="white")]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "docs", "public", "gallery", "assets", "multilayer_fresnel")
C0 = 2.99792458e8

# --- slab design (matches the validated builder) ---
EPS_SLAB = 4.0
D_SLAB = 10.0e-3
F0 = 10.0e9
BW = 0.5
DX = 1.0e-3
N_CPML = 20
NX_INTERIOR = 600


def fresnel_slab_rt_complex(freqs, eps_r, d):
    """Exact transfer-matrix (r, t) for a lossless slab in air, normal incidence."""
    n = math.sqrt(eps_r)
    r = np.zeros_like(freqs, dtype=complex)
    t = np.zeros_like(freqs, dtype=complex)
    for i, f in enumerate(freqs):
        if f <= 0:
            t[i] = 1.0
            continue
        delta = 2 * np.pi * f * n * d / C0
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        m00, m01 = cos_d, 1j * sin_d / n
        m10, m11 = 1j * n * sin_d, cos_d
        num = m00 + m01 - m10 - m11
        den = m00 + m01 + m10 + m11
        r[i] = num / den
        t[i] = 2.0 / den
    return r, t


# ===========================================================================
# Figure 1: geometry cross-section (true aspect, wave directions annotated)
# ===========================================================================
def make_geometry():
    n_slab = math.sqrt(EPS_SLAB)
    # Draw a compact, true-to-physics cross-section: air | slab | air. The
    # propagation axis is horizontal; the slab is a vertical band of finite
    # thickness. Use a synthetic, readable x-extent (mm) centred on the slab.
    x_lo, x_hi = -30.0, 30.0
    d_mm = D_SLAB * 1e3
    slab_lo, slab_hi = -d_mm / 2, d_mm / 2

    fig, ax = plt.subplots(figsize=(8.0, 3.6), layout="constrained")
    ax.add_patch(Rectangle((x_lo, 0), x_hi - x_lo, 1.0,
                           facecolor="#eaf2fb", edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((slab_lo, 0), d_mm, 1.0,
                           facecolor="#9ec3e6", edgecolor="#2f6aa8", lw=1.4,
                           zorder=2))
    # slab label
    ax.text(0, 0.5, f"dielectric slab\nεr = {EPS_SLAB:.0f}  (n = {n_slab:.1f})\n"
                    f"d = {d_mm:.0f} mm",
            ha="center", va="center", fontsize=9.5, zorder=4)
    ax.text(x_lo + 4, 0.86, "air", fontsize=10, color="#3a5d80", va="top")
    ax.text(x_hi - 4, 0.86, "air", fontsize=10, color="#3a5d80", va="top", ha="right")

    # incident / reflected / transmitted arrows
    yI = 0.28
    ax.add_patch(FancyArrowPatch((x_lo + 5, yI), (slab_lo - 3, yI),
                 arrowstyle="-|>", mutation_scale=16, color="#1f5fa8", lw=2.0))
    ax.text((x_lo + slab_lo) / 2 - 1, yI + 0.07, "incident", color="#1f5fa8",
            ha="center", fontsize=9)
    yR = 0.72
    ax.add_patch(FancyArrowPatch((slab_lo - 3, yR), (x_lo + 5, yR),
                 arrowstyle="-|>", mutation_scale=14, color="#b00000", lw=1.8))
    ax.text((x_lo + slab_lo) / 2 - 1, yR - 0.10, "reflected", color="#b00000",
            ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((slab_hi + 3, yI), (x_hi - 5, yI),
                 arrowstyle="-|>", mutation_scale=16, color="#1f7a3a", lw=2.0))
    ax.text((slab_hi + x_hi) / 2 + 1, yI + 0.07, "transmitted", color="#1f7a3a",
            ha="center", fontsize=9)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("propagation axis  x  (mm)")
    ax.set_yticks([])
    ax.set_title("Dielectric slab in free space — normal-incidence cross-section")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    out = os.path.join(ASSETS, "geometry.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# Figure 2: space-time (x-t) map of E_z on the propagation axis
# ===========================================================================
def _run_slab_field():
    """Run the 1D-physics slab sim (a 2D TMz strip) and return E_z(x,t) on the
    interior propagation axis: (field (n_steps, nx), x_mm [0 at the slab front
    face], t_ns, slab_front_mm, slab_back_mm). Shared by the x-t map and the
    propagation animation."""
    from rfx.grid import Grid
    from rfx.core.yee import init_state, init_materials, update_e, update_h
    from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
    from rfx.sources.tfsf import (
        init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e,
        apply_tfsf_e, apply_tfsf_h,
    )

    grid = Grid(freq_max=20e9, domain=(NX_INTERIOR * DX, 0.004, DX),
                dx=DX, cpml_layers=N_CPML, mode="2d_tmz")
    dt = grid.dt
    periodic = (False, True, True)

    tfsf_cfg, tfsf_st = init_tfsf(
        grid.nx, DX, dt, cpml_layers=N_CPML, tfsf_margin=5,
        f0=F0, bandwidth=BW, amplitude=1.0, polarization="ez",
        direction="+x", ny=grid.ny, nz=grid.nz,
    )
    x_lo, x_hi = tfsf_cfg.x_lo, tfsf_cfg.x_hi

    slab_lo_g = grid.nx // 2 - int(D_SLAB / (2 * DX))
    slab_hi_g = grid.nx // 2 + int(D_SLAB / (2 * DX))

    materials = init_materials(grid.shape)
    materials = materials._replace(
        eps_r=materials.eps_r.at[slab_lo_g:slab_hi_g, :, :].set(EPS_SLAB)
    )
    state = init_state(grid.shape)
    cp, cs = init_cpml(grid)

    # Run long enough for the transmitted pulse to clear the slab and approach
    # the rear CPML, and the reflected pulse to head back toward the front CPML.
    v_cells = C0 * dt / DX
    dist = (grid.nx - N_CPML) - (grid.nx // 2)
    n_steps = int(2.0 * dist / v_cells * 0.95)
    n_steps = min(n_steps, 1400)

    jrow = grid.ny // 2
    rec = np.zeros((n_steps, grid.nx), dtype=np.float32)
    for step in range(n_steps):
        t = step * dt
        state = update_h(state, materials, dt, DX, periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_st, DX, dt)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_h(tfsf_cfg, tfsf_st, DX, dt)

        state = update_e(state, materials, dt, DX, periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_st, DX, dt)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_e(tfsf_cfg, tfsf_st, DX, dt, t)

        rec[step] = np.asarray(state.ez[:, jrow, 0])

    # Interior window (strip CPML on x); convert to mm and ns
    ix0, ix1 = N_CPML, grid.nx - N_CPML
    field = rec[:, ix0:ix1]
    x_mm = (np.arange(ix0, ix1) - slab_lo_g) * DX * 1e3  # 0 at slab front face
    t_ns = np.arange(n_steps) * dt * 1e9
    slab_front_mm = 0.0
    slab_back_mm = (slab_hi_g - slab_lo_g) * DX * 1e3
    return field, x_mm, t_ns, slab_front_mm, slab_back_mm


def make_field_xt():
    field, x_mm, t_ns, slab_front_mm, slab_back_mm = _run_slab_field()

    vmax = float(np.percentile(np.abs(field), 99.5)) or 1.0

    fig, ax = plt.subplots(figsize=(7.4, 5.2), layout="constrained")
    im = ax.pcolormesh(x_mm, t_ns, field, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    # slab band
    ax.axvspan(slab_front_mm, slab_back_mm, color="0.25", alpha=0.18, lw=0)
    ax.axvline(slab_front_mm, color="0.25", lw=0.9, ls="--")
    ax.axvline(slab_back_mm, color="0.25", lw=0.9, ls="--")
    s_lbl = ax.text(slab_back_mm + 2, t_ns[-1] * 0.97, "slab", fontsize=9,
                    color="0.2", va="top")
    s_lbl.set_path_effects(_HALO)
    # annotate the three wave branches (white halo so they read over the bright
    # pulse streaks)
    for x_pos, y_frac, label, color, rot in (
        (x_mm[0] + 40, 0.30, "incident →", "#0b3d6b", -38),
        (x_mm[0] + 40, 0.86, "← reflected", "#7a0000", 38),
        (slab_back_mm + 30, 0.80, "transmitted →", "#0b3d6b", -38),
    ):
        t = ax.text(x_pos, t_ns[-1] * y_frac, label, color=color, fontsize=9,
                    rotation=rot)
        t.set_path_effects(_HALO)

    ax.set_xlabel("position along propagation axis  (mm,  0 = slab front face)")
    ax.set_ylabel("time  (ns)")
    ax.set_title("E$_z$ space-time map: incident, reflected, transmitted pulse")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("E$_z$  (incident amplitude = 1)")
    out = os.path.join(ASSETS, "field_xt.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out, f"(n_steps={n_steps}, vmax={vmax:.3f})")


def make_field_anim():
    """Animate the plane-wave pulse along x: it travels +x at c, hits the slab
    front face, and splits into a reflected pulse (heading back -x) and a
    transmitted pulse (continuing +x, slowed to c/n inside the slab)."""
    import matplotlib.animation as manim
    field, x_mm, t_ns, slab_front_mm, slab_back_mm = _run_slab_field()
    n_steps = field.shape[0]
    vmax = float(np.percentile(np.abs(field), 99.5)) or 1.0
    ylim = 1.15 * vmax
    nframes = 80
    idx = np.linspace(0, n_steps - 1, nframes).round().astype(int)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axvspan(slab_front_mm, slab_back_mm, color="0.25", alpha=0.18, lw=0)
    ax.axvline(slab_front_mm, color="0.25", lw=0.9, ls="--")
    ax.axvline(slab_back_mm, color="0.25", lw=0.9, ls="--")
    s_lbl = ax.text(slab_back_mm + 2, ylim * 0.86, "slab", fontsize=9,
                    color="0.2", va="top")
    s_lbl.set_path_effects(_HALO)
    ax.axhline(0, color="0.6", lw=0.6)
    (line,) = ax.plot(x_mm, field[idx[0]], color="#1f4e9c", lw=1.4)
    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("position along propagation axis  (mm,  0 = slab front face)")
    ax.set_ylabel("E$_z$  (incident amplitude = 1)")
    ax.set_title("Plane-wave pulse: incidence, reflection and transmission at the slab")
    tlabel = ax.text(0.985, 0.94, "", transform=ax.transAxes, ha="right",
                     va="top", fontsize=8.5, color="0.25")

    def _upd(k):
        line.set_ydata(field[idx[k]])
        tlabel.set_text(f"t = {t_ns[idx[k]]:.2f} ns")
        return (line, tlabel)

    anim = manim.FuncAnimation(fig, _upd, frames=nframes, blit=False)
    out = os.path.join(ASSETS, "field_anim.gif")
    anim.save(out, writer=manim.PillowWriter(fps=15), dpi=90)
    plt.close(fig)
    print("wrote", out, f"({nframes} frames, vmax={vmax:.3f})")


# ===========================================================================
# Figure 3: R / T power vs frequency, rfx over exact analytic
# ===========================================================================
def make_rt_overlay():
    d = json.load(open(os.path.join(ASSETS, "sparams.json")))
    f = np.array(d["freqs_hz"])
    s = np.array(d["s"])
    s11 = s[0, 0, :, 0] + 1j * s[0, 0, :, 1]
    s21 = s[1, 0, :, 0] + 1j * s[1, 0, :, 1]
    R = np.abs(s11) ** 2
    T = np.abs(s21) ** 2

    r_an_c, t_an_c = fresnel_slab_rt_complex(f, EPS_SLAB, D_SLAB)
    R_an = np.abs(r_an_c) ** 2
    T_an = np.abs(t_an_c) ** 2

    f_ghz = f / 1e9
    # decimate rfx markers so the analytic line stays visible underneath
    step = max(1, len(f) // 40)
    mk = slice(None, None, step)

    fig, ax = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
    ax.plot(f_ghz, T_an, "-", color="#1f7a3a", lw=2.0, label="T  — exact transfer matrix")
    ax.plot(f_ghz, R_an, "-", color="#b00000", lw=2.0, label="R  — exact transfer matrix")
    ax.plot(f_ghz[mk], T[mk], "o", color="#1f7a3a", ms=5, mfc="white", mew=1.3,
            label="T  — rfx FDTD")
    ax.plot(f_ghz[mk], R[mk], "s", color="#b00000", ms=4.5, mfc="white", mew=1.3,
            label="R  — rfx FDTD")
    ax.set_xlabel("Frequency  (GHz)")
    ax.set_ylabel("Power fraction")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(f_ghz[0], f_ghz[-1])
    ax.set_title("Reflected (R) and transmitted (T) power — rfx vs exact Fresnel")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", framealpha=0.9)

    r_err = float(np.abs(R - R_an).mean())
    t_err = float(np.abs(T - T_an).mean())
    cons = float(np.abs(R + T - 1).mean())
    txt = (f"mean |R$_{{rfx}}$−R$_{{exact}}$| = {r_err:.3f}\n"
           f"mean |T$_{{rfx}}$−T$_{{exact}}$| = {t_err:.3f}\n"
           f"mean |R+T−1| = {cons:.3f}")
    ax.text(0.015, 0.04, txt, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    i = int(np.argmin(R_an))
    ax.annotate("Fabry–Pérot transmission peak\n(half-wave in the slab)",
                xy=(f_ghz[i], T_an[i]), xytext=(f_ghz[i], 0.55),
                fontsize=8, ha="center", color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    out = os.path.join(ASSETS, "rt_overlay.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out, f"(r_err={r_err:.4f}, t_err={t_err:.4f}, cons={cons:.4f})")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "geometry"):
        make_geometry()
    if which in ("all", "rt"):
        make_rt_overlay()
    if which in ("all", "field"):
        make_field_xt()
    if which in ("all", "anim"):
        make_field_anim()

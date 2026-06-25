"""Generate v3 gallery figures for the 2.4 GHz FR4 patch antenna.

A SINGLE uniform-mesh patch run drives every figure, so the return-loss dip,
the Harminv ring-down resonance, and the E_z field map are all read at ONE
self-consistent frequency. (The non-uniform production path cannot snapshot the
fields, which is what made the earlier dip / resonance / field-map frequencies
disagree.)

The feed is a multi-cell wire port spanning the substrate from the ground plane
to the patch. Its inset along the resonant length L sets the input resistance:
high at the radiating edge, falling toward the patch centre. ``sweep`` finds the
inset that lands nearest 50 Ohm (deepest |S11| dip at resonance); ``final``
re-runs that inset and emits every asset.

Produces into docs/public/gallery/assets/patch_antenna/:
  - geometry.png        substrate cross-section (GP / FR4 / patch / probe)
  - field_resonance.png E_z on the patch plane at f_res (TM010 half-wave)
  - s11_db.png          |S11| dB vs GHz with the matched dip + zoom inset
  - validation.png      rfx Harminv resonance vs first-order analytic estimate
  - sparams.json        the matched |S11| sweep (committed artifact)
  - sparams.s1p         Touchstone of the same sweep

Modes:
  python _gallery_v3_patch_figs.py sweep   # inset sweep -> pick the match
  python _gallery_v3_patch_figs.py final   # chosen inset -> all assets
  python _gallery_v3_patch_figs.py geometry|s11|field|validation  # single fig
"""

import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
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
ASSETS = os.path.join(ROOT, "docs", "public", "gallery", "assets", "patch_antenna")
CACHE = os.path.join(HERE, "_patch_v3_cache.npz")
C0 = 2.99792458e8

# --- geometry (matches examples/crossval/05_patch_antenna.py) ---
EPS_R = 4.3
TAN_D = 0.02
H_SUB = 1.5e-3
W = 38.0e-3
L = 29.5e-3
GX, GY = 60.0e-3, 55.0e-3
F_DESIGN = 2.4e9

# Uniform cubic cells (dx = dy = dz). Snapshots require a uniform mesh, and a
# single resolution for the S11 sweep, the Harminv ring-down and the field map
# is what keeps all three at one self-consistent frequency. 1 mm gives ~30 cells
# across L (the TM010 half-wave is well sampled) and runs in ~1 min/case on CPU.
DX = 1.0e-3                # uniform cubic Yee cells (dx = dy = dz); snapshots
N_CPML = 8                 #   require a uniform mesh, and one resolution for the
DZ = DX                    #   S11 sweep, the Harminv ring-down and the field map
#                            keeps all three at one self-consistent frequency.
# The 1.5 mm substrate is a sub-cell layer represented with 0.5 mm-thick metal
# cells (ground + patch) stamped onto the 1 mm propagation mesh — the standard
# coarse-substrate patch model. ~30 cells span L, so the TM010 half-wave is well
# sampled, and a case runs in ~1 min on CPU.
DZ_THIN = 0.5e-3           # ground / patch metal cell thickness + substrate
N_SUB = 3                  # 3 * 0.5 mm = 1.5 mm FR4 substrate
H_G = N_SUB * DZ_THIN
AIR_BELOW, AIR_ABOVE = 8.0e-3, 16.0e-3

DOM_X = GX + 2 * 8e-3
DOM_Y = GY + 2 * 8e-3
DOM_Z = AIR_BELOW + H_G + AIR_ABOVE

GX_LO = (DOM_X - GX) / 2
GY_LO = (DOM_Y - GY) / 2
PATCH_X_LO = DOM_X / 2 - L / 2
PATCH_Y_LO = DOM_Y / 2 - W / 2
FEED_Y = DOM_Y / 2

Z_GND_LO = AIR_BELOW - DZ_THIN
Z_SUB_LO = AIR_BELOW
Z_SUB_HI = AIR_BELOW + H_G
Z_PATCH_LO = Z_SUB_HI
Z_PATCH_HI = Z_SUB_HI + DZ_THIN
SRC_Z = Z_SUB_LO + DZ_THIN * 1.5    # mid-substrate drive / probe plane

# FR4 conductivity from the loss tangent (gives a finite, physical Q).
SIGMA_FR4 = 2 * math.pi * F_DESIGN * 8.8541878128e-12 * EPS_R * TAN_D

# first-order transmission-line (Balanis) analytic resonance
EPS_EFF = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 * H_SUB / W) ** -0.5
DL = 0.412 * H_SUB * ((EPS_EFF + 0.3) * (W / H_SUB + 0.264)) / \
     ((EPS_EFF - 0.258) * (W / H_SUB + 0.8))
F_ANALYTIC = C0 / (2 * (L + 2 * DL) * math.sqrt(EPS_EFF))

# Matched inset chosen from `sweep`: the dip deepens monotonically toward the
# radiating edge (2 mm: -12.6 dB ... 11 mm: -8.4 dB, all at 2.35 GHz). 3 mm gives
# essentially the same match as 2 mm (-12.3 vs -12.6 dB) but sits a cell further
# from the edge, so it is the robust choice. Override with PATCH_INSET_M.
MATCHED_INSET = float(os.environ.get("PATCH_INSET_M", 3.0e-3))


def _build(inset, *, with_port):
    """Build the uniform-mesh patch. ``with_port`` -> wire-port S11 run;
    else a broadband Ez source + probe for the Harminv ring-down."""
    from rfx import Simulation, Box
    from rfx.sources.sources import GaussianPulse

    feed_x = PATCH_X_LO + inset
    sim = Simulation(freq_max=4e9, domain=(DOM_X, DOM_Y, DOM_Z), dx=DX,
                     boundary="cpml", cpml_layers=N_CPML)
    sim.add_material("fr4", eps_r=EPS_R, sigma=SIGMA_FR4)
    sim.add(Box((GX_LO, GY_LO, Z_GND_LO), (GX_LO + GX, GY_LO + GY, Z_SUB_LO)),
            material="pec")
    sim.add(Box((GX_LO, GY_LO, Z_SUB_LO), (GX_LO + GX, GY_LO + GY, Z_SUB_HI)),
            material="fr4")
    sim.add(Box((PATCH_X_LO, PATCH_Y_LO, Z_PATCH_LO),
                (PATCH_X_LO + L, PATCH_Y_LO + W, Z_PATCH_HI)), material="pec")
    if with_port:
        sim.add_port(position=(feed_x, FEED_Y, Z_SUB_LO), component="ez",
                     impedance=50.0, extent=Z_SUB_HI - Z_SUB_LO,
                     waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.0))
    else:
        sim.add_source(position=(feed_x, FEED_Y, SRC_Z), component="ez",
                       waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2))
        sim.add_probe((DOM_X / 2 + 5e-3, DOM_Y / 2 + 5e-3, SRC_Z),
                      component="ez")
    return sim


def _run_s11(inset, freqs, n_steps=6000):
    import jax.numpy as jnp
    sim = _build(inset, with_port=True)
    res = sim.run(n_steps=n_steps, compute_s_params=True,
                  s_param_freqs=jnp.asarray(freqs), s_param_n_steps=n_steps,
                  skip_preflight=True)
    S = np.asarray(res.s_params)
    s11 = S[0, 0, :] if S.ndim == 3 else np.asarray(S).ravel()
    return s11


def _harminv_resonance(inset):
    from rfx.harminv import harminv
    sim = _build(inset, with_port=False)
    res = sim.run(num_periods=60, skip_preflight=True)
    ts = np.asarray(res.time_series).ravel()
    dt = float(res.dt)
    skip = int(len(ts) * 0.3)
    modes = harminv(ts[skip:], dt, 1.5e9, 3.5e9)
    good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-8]
    cand = [m for m in good if 1.8e9 <= m.freq <= 3.0e9]
    if cand:
        m = max(cand, key=lambda m: m.amplitude)
    elif good:
        m = max(good, key=lambda m: m.amplitude)
    else:
        return float("nan"), float("nan")
    return float(m.freq), float(m.Q)


# ===========================================================================
# sweep: find the inset that matches the feed (deepest dip at resonance)
# ===========================================================================
def sweep():
    print(f"analytic f_res = {F_ANALYTIC/1e9:.3f} GHz  (eps_eff={EPS_EFF:.3f}, "
          f"dL={DL*1e3:.3f} mm)")
    freqs = np.linspace(1.5e9, 3.5e9, 81)
    f_ghz = freqs / 1e9
    results = []
    for inset in (2e-3, 3e-3, 5e-3, 8e-3, 11e-3):
        t0 = time.time()
        s11 = _run_s11(inset, freqs, n_steps=6000)
        db = 20 * np.log10(np.maximum(np.abs(s11), 1e-6))
        idip = int(np.argmin(db))
        results.append((inset, f_ghz[idip], db[idip], float(np.abs(s11).max())))
        print(f"  inset={inset*1e3:4.1f} mm  dip={db[idip]:7.2f} dB @ "
              f"{f_ghz[idip]:.3f} GHz  max|S11|={np.abs(s11).max():.3f}  "
              f"({time.time()-t0:.0f}s)")
    best = min(results, key=lambda r: r[2])
    print(f"\nbest match: inset={best[0]*1e3:.1f} mm, dip={best[2]:.2f} dB @ "
          f"{best[1]:.3f} GHz")
    print(f"-> set PATCH_INSET_M={best[0]:.4f} and run `final`")


# ===========================================================================
# final: chosen inset -> S11 sweep + Harminv + field map (all coherent)
# ===========================================================================
def final():
    inset = MATCHED_INSET
    print(f"FINAL run @ inset = {inset*1e3:.1f} mm")
    # 1) S11 sweep (committed artifact resolution)
    freqs = np.linspace(1.5e9, 3.5e9, 101)
    t0 = time.time()
    s11 = _run_s11(inset, freqs, n_steps=7000)
    print(f"  S11 sweep done ({time.time()-t0:.0f}s)")
    db = 20 * np.log10(np.maximum(np.abs(s11), 1e-6))
    idip = int(np.argmin(db))
    f_dip, db_dip = float(freqs[idip]), float(db[idip])

    # 2) Harminv ring-down on the SAME geometry
    t0 = time.time()
    f_res, q_res = _harminv_resonance(inset)
    print(f"  Harminv f_res={f_res/1e9:.3f} GHz Q={q_res:.1f} "
          f"({time.time()-t0:.0f}s)")

    # 3) field map at f_res (capture E_z on the substrate mid-plane)
    field_payload = _field_map(inset, f_res)

    # cache everything so the plotters reuse it without recomputing
    np.savez_compressed(
        CACHE, inset=inset, freqs=freqs,
        s11_re=s11.real, s11_im=s11.imag,
        f_dip=f_dip, db_dip=db_dip, f_res=f_res, q_res=q_res,
        **field_payload,
    )
    print(f"  cached -> {CACHE}")
    # write artifacts + figures
    _write_sparams(freqs, s11)
    make_s11()
    make_field()
    make_validation()
    make_geometry()
    print(f"\nSUMMARY  inset={inset*1e3:.1f}mm  dip={db_dip:.2f}dB @ "
          f"{f_dip/1e9:.3f}GHz  Harminv f_res={f_res/1e9:.3f}GHz Q={q_res:.1f}  "
          f"analytic={F_ANALYTIC/1e9:.3f}GHz")


def _field_map(inset, f_res):
    """Run a snapshot forward and return the modal E_z field arrays."""
    from rfx.simulation import SnapshotSpec
    sim = _build(inset, with_port=False)
    g = sim._build_grid()
    _, _, kz = g.position_to_index((PATCH_X_LO + inset, FEED_Y, SRC_Z))
    snap = SnapshotSpec(interval=25, components=("ez",), slice_axis=2,
                        slice_index=int(kz))
    res = sim.run(num_periods=60, snapshot=snap, skip_preflight=True)
    g = res.grid
    ez = np.asarray(res.snapshots["ez"])  # (n_frames, nx, ny)
    ix0, ix1 = g.pad_x_lo, g.nx - g.pad_x_hi
    iy0, iy1 = g.pad_y_lo, g.ny - g.pad_y_hi
    stride = max(1, ez.shape[0] // 1600)
    plane_t = np.asarray(ez[::stride, ix0:ix1, iy0:iy1], dtype=np.float32)
    dt_grid = float(g.dt) * stride
    # modal phasor over the ring-down at f_res
    n_frames = plane_t.shape[0]
    t = np.arange(n_frames) * dt_grid
    w0 = int(0.40 * n_frames)
    ring, tr = plane_t[w0:], t[w0:]
    phasor = np.tensordot(np.exp(-2j * np.pi * f_res * tr), ring, axes=(0, 0))
    flat = phasor.ravel()
    phasor *= np.exp(-1j * np.angle(flat[int(np.argmax(np.abs(flat)))]))
    field = np.asarray(phasor.real, dtype=np.float32)
    return {"field": field, "fm_inset": inset, "fm_fres": f_res,
            "ix_n": ix1 - ix0, "iy_n": iy1 - iy0}


def _write_sparams(freqs, s11):
    payload = {
        "case_id": "patch_antenna",
        "freqs_hz": [float(v) for v in freqs],
        "n_ports": 1,
        "s": [[[[float(v.real), float(v.imag)] for v in s11]]],
    }
    with open(os.path.join(ASSETS, "sparams.json"), "w") as f:
        json.dump(payload, f)
    # Touchstone .s1p (magnitude/angle, S11)
    lines = ["! Patch antenna |S11| — matched inset, uniform-mesh rfx run",
             "# Hz S MA R 50"]
    for fr, v in zip(freqs, s11):
        lines.append(f"{fr:.6e} {abs(v):.6e} {math.degrees(np.angle(v)):.4f}")
    with open(os.path.join(ASSETS, "sparams.s1p"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote sparams.json + sparams.s1p")


def _load_cache():
    if not os.path.exists(CACHE):
        raise SystemExit("no cache — run `final` first")
    z = np.load(CACHE)
    return z


# ===========================================================================
# Figure 1: geometry cross-section (x-z)
# ===========================================================================
def make_geometry():
    inset = float(_load_cache()["inset"]) if os.path.exists(CACHE) else MATCHED_INSET
    margin = 12e-3
    gx = L + 2 * margin
    feed_x = inset
    fig, ax = plt.subplots(figsize=(8.0, 3.4), layout="constrained")
    mm = 1e3
    z_sub_top = H_SUB
    # draw centred on patch
    x0 = -gx / 2
    ax.add_patch(Rectangle((x0 * mm, 0), gx * mm, H_SUB * mm,
                 facecolor="#cfe3b8", edgecolor="#6f8f4f", lw=1.0, zorder=2))
    ax.add_patch(Rectangle((x0 * mm, -0.18), gx * mm, 0.18,
                 facecolor="#b8860b", edgecolor="#7a5a08", lw=0.8, zorder=3))
    ax.add_patch(Rectangle((-L / 2 * mm, z_sub_top * mm), L * mm, 0.18,
                 facecolor="#d4a017", edgecolor="#7a5a08", lw=0.8, zorder=4))
    fx = (-L / 2 + feed_x)
    ax.plot([fx * mm, fx * mm], [0.0, z_sub_top * mm], color="#b00000",
            lw=2.4, zorder=5, solid_capstyle="round")
    ax.plot(fx * mm, 0.0, marker="o", color="#b00000", ms=5, zorder=6)

    # --- TOP band: patch label + L dimension + FR4 substrate label ---------
    ax.annotate("Patch (PEC)", xy=(L * 0.32 * mm, z_sub_top * mm + 0.18),
                xytext=(L * 0.32 * mm, z_sub_top * mm + 1.55), ha="center",
                va="bottom", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    # FR4 label points UP into the substrate band from the top band (away from
    # the ground-plane arrow, which lives in the bottom band).
    ax.annotate(f"FR4  εr = {EPS_R},  h = {H_SUB*1e3:.1f} mm,  tanδ = {TAN_D}",
                xy=(-gx * 0.30 * mm, H_SUB * mm * 0.5),
                xytext=(-gx * 0.30 * mm, z_sub_top * mm + 1.45), ha="center",
                va="bottom", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    # --- BOTTOM band: ground plane + wire port (no collision with FR4) ------
    ax.annotate("Ground plane (PEC)", xy=(gx * 0.18 * mm, -0.09),
                xytext=(gx * 0.18 * mm, -1.7), ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.annotate(f"50 Ω wire port (Ez)\ninset {inset*1e3:.1f} mm from edge",
                xy=(fx * mm, H_SUB * mm * 0.5), xytext=(-gx * 0.46 * mm, -1.95),
                ha="left", fontsize=8.5, color="#b00000",
                arrowprops=dict(arrowstyle="->", color="#b00000", lw=0.8))
    y_dim = z_sub_top * mm + 0.45
    ax.annotate("", xy=(-L / 2 * mm, y_dim), xytext=(L / 2 * mm, y_dim),
                arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.0))
    ax.text(-L * 0.18 * mm, y_dim + 0.08, f"L = {L*1e3:.1f} mm", ha="center",
            va="bottom", fontsize=8.5)
    ax.set_xlim((-gx / 2 - 6e-3) * mm, (gx / 2 + 6e-3) * mm)
    # Tighten to the structure (ground at -0.18, patch top at z_sub_top+0.18)
    # plus just enough headroom for the top/bottom annotation bands.
    ax.set_ylim(-2.4, z_sub_top * mm + 2.2)
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("z  (mm)")
    ax.set_title("Rectangular patch antenna — substrate cross-section (E-plane cut)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = os.path.join(ASSETS, "geometry.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# Figure 2: E_z field map at resonance (TM010)
# ===========================================================================
def _despeckle(frame):
    """Replace lone single-cell spikes (a probe/corner cell can show up as one
    near-zero pixel inside a strong lobe) with their local 3x3 median, leaving
    the smooth mode untouched. Touches only cells deviating from the local
    median by both >6x the local MAD and >20% of the global peak.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    f = np.asarray(frame, dtype=np.float32)
    pad = np.pad(f, 1, mode="edge")
    win = sliding_window_view(pad, (3, 3))
    med = np.median(win, axis=(-1, -2))
    mad = np.median(np.abs(win - med[..., None, None]), axis=(-1, -2))
    gmax = float(np.abs(f).max()) or 1.0
    spike = (np.abs(f - med) > 6.0 * mad) & (np.abs(f - med) > 0.20 * gmax)
    return np.where(spike, med, f).astype(np.float32)


def make_field():
    z = _load_cache()
    field = _despeckle(np.asarray(z["field"]))
    inset = float(z["fm_inset"])
    f_res = float(z["fm_fres"])
    x_mm = np.arange(field.shape[0]) * DX * 1e3
    y_mm = np.arange(field.shape[1]) * DX * 1e3

    feed_x = PATCH_X_LO + inset
    fx_i = int(round(feed_x / DX))
    fy_i = int(round(FEED_Y / DX))
    yy, xx = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    feed_mask = (xx - fx_i) ** 2 + (yy - fy_i) ** 2 > 4 ** 2
    vmax = float(np.percentile(np.abs(field[feed_mask]), 99.0))
    if vmax <= 0:
        vmax = float(np.max(np.abs(field))) or 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    im = ax.pcolormesh(x_mm, y_mm, field.T, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    # patch outline: centred box L x W
    cx, cy = field.shape[0] * DX / 2 * 1e3, field.shape[1] * DX / 2 * 1e3
    ax.add_patch(Rectangle((cx - L / 2 * 1e3, cy - W / 2 * 1e3),
                           L * 1e3, W * 1e3, fill=False, edgecolor="k", lw=1.4))
    ax.plot((cx - L / 2 * 1e3) + inset * 1e3, cy, "k+", ms=10, mew=1.6)
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_title(f"E$_z$ on the patch plane — TM$_{{010}}$ at {f_res/1e9:.2f} GHz\n"
                 "half-wave standing wave along L, uniform along W")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("E$_z$  (normalised, divergent scale)")
    out = os.path.join(ASSETS, "field_resonance.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out, f"(vmax={vmax:.3e})")


def make_field_anim():
    """Animate the TM010 standing wave oscillating in place over one RF period:
    E_z(x,t) = E_mode(x) * cos(wt). The two radiating edges swap sign each half
    period (red <-> blue through a zero crossing) — the standing-wave signature.
    """
    import matplotlib.animation as manim
    z = _load_cache()
    field = _despeckle(np.asarray(z["field"]))
    inset = float(z["fm_inset"])
    f_res = float(z["fm_fres"])
    x_mm = np.arange(field.shape[0]) * DX * 1e3
    y_mm = np.arange(field.shape[1]) * DX * 1e3
    feed_x = PATCH_X_LO + inset
    fx_i = int(round(feed_x / DX))
    fy_i = int(round(FEED_Y / DX))
    yy, xx = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    feed_mask = (xx - fx_i) ** 2 + (yy - fy_i) ** 2 > 4 ** 2
    vmax = float(np.percentile(np.abs(field[feed_mask]), 99.0))
    if vmax <= 0:
        vmax = float(np.max(np.abs(field))) or 1.0

    nframes = 40
    phase = np.linspace(0.0, 2 * np.pi, nframes, endpoint=False)
    cx, cy = field.shape[0] * DX / 2 * 1e3, field.shape[1] * DX / 2 * 1e3

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    mesh = ax.pcolormesh(x_mm, y_mm, (field * np.cos(phase[0])).T,
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    ax.add_patch(Rectangle((cx - L / 2 * 1e3, cy - W / 2 * 1e3),
                           L * 1e3, W * 1e3, fill=False, edgecolor="k", lw=1.4))
    ax.plot((cx - L / 2 * 1e3) + inset * 1e3, cy, "k+", ms=10, mew=1.6)
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"TM$_{{010}}$ standing wave at {f_res/1e9:.2f} GHz — E$_z$ over "
                 "one RF period")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("E$_z$  (normalised, divergent scale)")

    def _upd(i):
        mesh.set_array((field * np.cos(phase[i])).T.ravel())
        return (mesh,)

    anim = manim.FuncAnimation(fig, _upd, frames=nframes, blit=False)
    out = os.path.join(ASSETS, "field_anim.gif")
    anim.save(out, writer=manim.PillowWriter(fps=12), dpi=90)
    plt.close(fig)
    print("wrote", out, f"({nframes} frames, vmax={vmax:.3e})")


# ===========================================================================
# Figure 3: |S11| dB with matched dip + zoom inset
# ===========================================================================
def make_s11():
    z = _load_cache()
    f = np.asarray(z["freqs"]) / 1e9
    s11 = np.asarray(z["s11_re"]) + 1j * np.asarray(z["s11_im"])
    db = 20 * np.log10(np.maximum(np.abs(s11), 1e-6))
    f_dip, db_dip = float(z["f_dip"]) / 1e9, float(z["db_dip"])
    f_res = float(z["f_res"]) / 1e9

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    ax.plot(f, db, color="#1f5fa8", lw=1.8)
    ax.plot(f_dip, db_dip, "o", color="#b00000", ms=7, zorder=5)
    # Annotate INSIDE the axes in the empty lower-left quadrant (left of the
    # dip, below the shallow left shoulder of the curve), with the arrow
    # reaching across to the dip marker. Left-aligned and clear of the y-tick
    # gutter; the white bbox keeps it readable over the grid.
    ax.annotate(f"matched |S11| dip\n{f_dip:.2f} GHz, {db_dip:.1f} dB",
                xy=(f_dip, db_dip), xytext=(f[0] + 0.22, db_dip + 0.35),
                fontsize=9, color="#b00000", va="center", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="#b00000", lw=0.9))
    ax.axvline(f_res, color="0.4", ls="--", lw=1.1)
    # Rotated label moved OFF the vline (to its left) and high on the plot where
    # the curve is flat, with a white halo so it reads cleanly over the grid.
    t = ax.text(f_res - 0.06, db.max() - 0.25,
                f"Harminv resonance {f_res:.2f} GHz", rotation=90, va="top",
                ha="right", fontsize=8, color="0.25")
    t.set_path_effects(_HALO)
    ax.axhline(-10, color="0.6", ls=":", lw=0.9)
    ax.text(f[0] + 0.05, -10.0, "−10 dB", fontsize=7.5, color="0.5",
            va="bottom", ha="left")
    ax.set_xlabel("Frequency  (GHz)")
    ax.set_ylabel("|S11|  (dB)")
    ax.set_title("Patch antenna return loss |S11| (50 Ω matched wire port)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(db_dip - 1.5, db.max() + 0.8)

    axins = ax.inset_axes([0.58, 0.12, 0.38, 0.42])
    sel = (f > f_dip - 0.35) & (f < f_dip + 0.35)
    axins.plot(f[sel], db[sel], color="#1f5fa8", lw=1.6)
    axins.plot(f_dip, db_dip, "o", color="#b00000", ms=6)
    axins.set_title("zoom: matched dip", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="0.5")
    out = os.path.join(ASSETS, "s11_db.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out, f"(dip {f_dip:.3f} GHz, {db_dip:.2f} dB)")


# ===========================================================================
# Figure 4: rfx Harminv vs analytic resonance
# ===========================================================================
def make_validation():
    z = _load_cache()
    f_res = float(z["f_res"])
    fig, ax = plt.subplots(figsize=(6.6, 3.2), layout="constrained")
    labels = ["analytic\n(transmission-line)", "rfx\n(Harminv ring-down)"]
    vals = [F_ANALYTIC / 1e9, f_res / 1e9]
    colors = ["#9aa0a6", "#1f5fa8"]
    ypos = [1, 0]
    ax.barh(ypos, vals, color=colors, height=0.5, zorder=3)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.03, y, f"{v:.2f} GHz", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Resonance frequency  (GHz)")
    ax.set_xlim(0, max(vals) * 1.3)
    delta = 100 * (f_res - F_ANALYTIC) / F_ANALYTIC
    sign = "+" if delta >= 0 else "−"
    ax.set_title(f"Resonance: rfx vs first-order analytic  (Δ = {sign}{abs(delta):.0f} %)")
    ax.grid(True, axis="x", alpha=0.3)
    out = os.path.join(ASSETS, "validation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out, f"(analytic {F_ANALYTIC/1e9:.3f}, rfx {f_res/1e9:.3f})")


# ===========================================================================
# Figure 5: autodiff — d|S11|^2/d(eps_r) of the substrate, AD vs central FD
# ===========================================================================
# Differentiate |S11(f0)|^2 of the matched patch w.r.t. the FR4 substrate
# permittivity, straight through Simulation.forward(port_s11_freqs=...) with the
# wave-decomposition objective (minimize_s11_at_freq_wave_decomp). Validated
# against central finite differences (5% gate, per differentiable_s11_design.py).
#
# The full production figure (the headline 3D run at the gallery mesh) is a GPU
# job — reverse-mode AD through the 3D patch plus two FD forwards needs the
# checkpointed GPU path. This routine runs a CPU-feasible MODERATE-resolution
# version so the code path and the AD/FD numbers are real; set PATCH_AD_DX to
# tighten the mesh on a GPU run.
F0_S11_AD = 2.35e9


def _build_ad(deps, *, dx_ad, n_cpml_ad, n_sub_ad):
    """Coarsened patch for the AD demo (geometry mirrors `_build`)."""
    from rfx import Simulation, Box
    from rfx.sources.sources import GaussianPulse

    air_below, air_above = 6.0e-3, 10.0e-3
    h_g = n_sub_ad * dx_ad
    dom_x = GX + 2 * 6e-3
    dom_y = GY + 2 * 6e-3
    dom_z = air_below + h_g + air_above
    gx_lo, gy_lo = (dom_x - GX) / 2, (dom_y - GY) / 2
    px_lo, py_lo = dom_x / 2 - L / 2, dom_y / 2 - W / 2
    z_gnd_lo = air_below - dx_ad
    z_sub_lo, z_sub_hi = air_below, air_below + h_g
    feed_x = px_lo + MATCHED_INSET

    sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, dom_z), dx=dx_ad,
                     boundary="cpml", cpml_layers=n_cpml_ad)
    sim.add_material("fr4", eps_r=EPS_R, sigma=SIGMA_FR4)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_lo + GX, gy_lo + GY, z_sub_lo)),
            material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_lo + GX, gy_lo + GY, z_sub_hi)),
            material="fr4")
    sim.add(Box((px_lo, py_lo, z_sub_hi),
                (px_lo + L, py_lo + W, z_sub_hi + dx_ad)), material="pec")
    sim.add_port(position=(feed_x, dom_y / 2, z_sub_lo),
                 component="ez", impedance=50.0, extent=z_sub_hi - z_sub_lo,
                 waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.0))
    return sim


def make_autodiff():
    import time as _time
    import jax
    import jax.numpy as jnp
    from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp

    dx_ad = float(os.environ.get("PATCH_AD_DX", 2.0e-3))
    n_cpml_ad = int(os.environ.get("PATCH_AD_CPML", 6))
    n_sub_ad = int(os.environ.get("PATCH_AD_NSUB", 2))
    n_steps = int(os.environ.get("PATCH_AD_STEPS", 1500))

    sim = _build_ad(0.0, dx_ad=dx_ad, n_cpml_ad=n_cpml_ad, n_sub_ad=n_sub_ad)
    grid = sim._build_grid()
    mats = sim._assemble_materials(grid)[0]
    eps_base = np.asarray(mats.eps_r)
    sub_mask = jnp.asarray((eps_base > 1.5).astype(np.float32))   # FR4 cells
    eps_base_j = jnp.asarray(eps_base)
    print(f"  AD grid {grid.shape}  FR4 cells {int(sub_mask.sum())}  dx={dx_ad*1e3:.1f}mm")

    obj = minimize_s11_at_freq_wave_decomp(F0_S11_AD, port_idx=0)
    freqs = jnp.asarray([F0_S11_AD])

    def s11_sq(deps):
        sim2 = _build_ad(deps, dx_ad=dx_ad, n_cpml_ad=n_cpml_ad, n_sub_ad=n_sub_ad)
        eps = eps_base_j * (1.0 + sub_mask * deps)
        res = sim2.forward(eps_override=eps, n_steps=n_steps,
                           port_s11_freqs=freqs, skip_preflight=True)
        return obj(res)

    deps0 = jnp.float32(0.0)
    t0 = _time.time()
    val, g = jax.value_and_grad(s11_sq)(deps0)
    val, g_ad = float(val), float(g)
    print(f"  AD: |S11|^2={val:.5f}  d/deps={g_ad:+.5e}  ({_time.time()-t0:.0f}s)")

    h = 0.05
    fp = float(s11_sq(jnp.float32(h)))
    fm = float(s11_sq(jnp.float32(-h)))
    g_fd = (fp - fm) / (2 * h)
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-12)
    sign_ok = (g_ad * g_fd) > 0
    print(f"  FD: {g_fd:+.5e}  rel(AD vs FD)={rel:.4f}  sign_agree={'YES' if sign_ok else 'NO'}")

    fig, ax = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    xs = ["AD\n(jax.grad)", "central FD\n(h = 0.05)"]
    vals = [g_ad, g_fd]
    colors = ["#1f5fa8", "#9aa0a6"]
    bars = ax.bar(xs, vals, color=colors, width=0.55, zorder=3)
    span = max(abs(v) for v in vals)
    sgn = 1 if vals[0] >= 0 else -1
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + sgn * 0.04 * span,
                f"{v:+.4f}", ha="center",
                va="bottom" if sgn > 0 else "top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_ylabel(r"$\partial\,|S_{11}|^2 / \partial\,\varepsilon_r$")
    ax.set_title("Patch return-loss sensitivity to the substrate $\\varepsilon_r$ — AD vs FD")
    ax.grid(True, axis="y", alpha=0.3)
    if sgn > 0:
        ax.set_ylim(-span * 0.30, span * 1.45)
    else:
        ax.set_ylim(-span * 1.45, span * 0.30)
    txt = (f"|S11|^2 = {val:.4f}  at {F0_S11_AD/1e9:.2f} GHz\n"
           f"rel(AD vs FD) = {rel*100:.2f} %  (gate < 5%)\n"
           f"sign agreement: {'YES' if sign_ok else 'NO'}\n"
           f"uniform mesh  dx = {dx_ad*1e3:.1f} mm")
    ax.text(0.5, 0.04, txt, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="center", family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))
    out = os.path.join(ASSETS, "autodiff.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)
    return dict(val=val, g_ad=g_ad, g_fd=g_fd, rel=rel, sign_ok=sign_ok)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "final"
    if which == "sweep":
        sweep()
    elif which == "final":
        final()
    elif which == "autodiff":
        make_autodiff()
    elif which == "geometry":
        make_geometry()
    elif which == "s11":
        make_s11()
    elif which == "field":
        make_field()
    elif which == "anim":
        make_field_anim()
    elif which == "validation":
        make_validation()
    else:
        raise SystemExit(f"unknown mode {which!r}")

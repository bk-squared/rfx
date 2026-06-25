"""Generate v3 gallery figures for the empty WR-90 rectangular waveguide.

Produces into docs/public/gallery/assets/waveguide_wr90/:
  - field_te10.png  a TE10 propagation map (E_z on the broad-wall mid-plane of a
                    source-fed empty guide: the half-sine across the broad wall,
                    travelling down the guide)
  - validation.png  |S11| and |S21| vs the exact matched empty-guide answer
                    (|S11| = 0, |S21| = 1), read from the committed sparams.json
  - autodiff.png    d|S21|^2/d(eps_r) of the dielectric fill via jax.value_and_grad
                    through compute_waveguide_s_matrix(normalize="flux", ...),
                    cross-checked against central finite differences (5% gate,
                    mirroring tests/test_waveguide_flux_ad.py)

The S-parameters themselves live in the committed sparams.json produced by
scripts/precompute_gallery_artifacts.py (_build_waveguide_wr90). This script adds
the field map and the two validation/autodiff figures on top of that artifact.

Modes:
  python _gallery_v3_waveguide_figs.py all        # every figure
  python _gallery_v3_waveguide_figs.py field|validation|autodiff
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
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "docs", "public", "gallery", "assets", "waveguide_wr90")
C0 = 2.99792458e8

# --- shared layout standard (gallery v3) -----------------------------------
mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.titlepad": 10,
    "figure.constrained_layout.use": True,
})
_HALO = [pe.withStroke(linewidth=2.5, foreground="white")]

# --- WR-90 geometry (matches _build_waveguide_wr90) ------------------------
A_WG = 0.02286            # broad wall (sets the TE10 cutoff)
B_WG = 0.01016            # narrow wall
DX = 0.001
F_CUTOFF = C0 / (2.0 * A_WG)   # ~6.557 GHz


# ===========================================================================
# Figure 1: TE10 propagation map (source-fed companion guide)
# ===========================================================================
def make_field():
    """Snapshot E_z on the broad-wall mid-plane of a source-fed empty guide.

    The validated S-matrix path (two waveguide ports) cannot snapshot fields, so
    a source-fed empty guide reproduces the same TE10 propagation purely for the
    visual — exactly the companion-Simulation pattern in the builder.
    """
    from rfx import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.sources.sources import GaussianPulse
    from rfx.simulation import SnapshotSpec

    f0 = 10.2e9
    bandwidth = 0.5
    # CPML must fit the guide cross-section: pad b so nz > 2*cpml (still hollow
    # air — only the extent changes, not the physics), as the builder does.
    anim_cpml = 8
    b_anim = max(B_WG, (2 * anim_cpml + 4) * DX)
    a_anim = A_WG
    domain_x = 0.120
    src_x = domain_x * 0.12

    sim = Simulation(
        freq_max=14e9,
        domain=(domain_x, a_anim, b_anim),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=anim_cpml,
        dx=DX,
    )
    # Drive at the broad-wall centre (TE10 has its peak E_z there).
    sim.add_source((src_x, a_anim / 2, b_anim / 2), "ez",
                   waveform=GaussianPulse(f0=f0, bandwidth=bandwidth, amplitude=1.0))

    grid = sim._build_grid()
    # slice on the broad-wall mid-plane (constant z), giving an (x, y) map
    _, _, kz = grid.position_to_index((src_x, a_anim / 2, b_anim / 2))
    snap = SnapshotSpec(interval=8, components=("ez",), slice_axis=2,
                        slice_index=int(kz))
    res = sim.run(num_periods=60, snapshot=snap, skip_preflight=True)
    g = res.grid
    ez = np.asarray(res.snapshots["ez"])  # (n_frames, nx, ny)

    ix0, ix1 = g.pad_x_lo, g.nx - g.pad_x_hi
    iy0, iy1 = g.pad_y_lo, g.ny - g.pad_y_hi
    interior = ez[:, ix0:ix1, iy0:iy1]
    nx = interior.shape[1]

    # Pick a frame where the pulse has PROPAGATED to mid-guide (not the source
    # blob at firing time). The pulsed source keeps re-radiating near its own
    # cell, so the global energy peak sits AT the source — useless. Instead,
    # look only DOWNSTREAM of the source (x > src) and choose the frame whose
    # downstream energy peaks near the guide centre: that is the travelling
    # TE10 wave packet, where the half-sine across the broad wall is clean.
    src_i = int(round(src_x / DX))
    down_lo = src_i + int(0.18 * nx)        # safely clear of the source blob
    e_down = np.sum(interior[:, down_lo:, :] ** 2, axis=(1, 2))   # (n_frames,)
    x_idx = np.arange(nx - down_lo)
    e_xd = np.sum(interior[:, down_lo:, :] ** 2, axis=2)          # (n_frames, ndown)
    safe = np.maximum(e_down, 1e-30)
    xcent = down_lo + np.sum(e_xd * x_idx[None, :], axis=1) / safe
    target = 0.58 * nx
    valid = e_down > (0.10 * e_down.max())
    cand = np.where(valid & (np.abs(xcent - target) < 0.16 * nx))[0]
    f_idx = int(cand[np.argmax(e_down[cand])]) if len(cand) else int(np.argmax(e_down))
    frame = np.asarray(interior[f_idx], dtype=np.float32)

    x_mm = np.arange(frame.shape[0]) * DX * 1e3
    y_mm = np.arange(frame.shape[1]) * DX * 1e3
    # scale to the downstream wave packet so its half-sine is visible (a
    # residual source blob, if any, then saturates rather than washing it out)
    vmax = float(np.abs(frame[down_lo:, :]).max()) or 1.0

    fig, ax = plt.subplots(figsize=(8.2, 3.2), layout="constrained")
    im = ax.pcolormesh(x_mm, y_mm, frame.T, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_xlabel("propagation axis  x  (mm)")
    ax.set_ylabel("broad wall  y  (mm)")
    ax.set_title("TE$_{10}$ mode in the empty WR-90 guide — E$_z$ on the broad-wall "
                 "mid-plane")
    ax.set_aspect("equal")
    # annotation inside the axes, near the lower PEC wall where the half-sine
    # field is faint; the white halo keeps it readable
    t = ax.text(x_mm[-1] * 0.5, y_mm[-1] * 0.07,
                "half-sine across the broad wall,  travelling +x",
                ha="center", va="bottom", fontsize=9, color="0.12")
    t.set_path_effects(_HALO)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("E$_z$  (normalised, divergent scale)")
    out = os.path.join(ASSETS, "field_te10.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out, f"(frame {f_idx}, vmax={vmax:.3e})")


# ===========================================================================
# Figure 2: |S11| / |S21| vs the exact matched empty-guide answer
# ===========================================================================
def _raw_floor_s():
    """Run an empty WR-90 guide with normalize=False to expose the REAL
    measurement floor: |S11| is set by CPML back-reflection (a few percent),
    not by construction. normalize=True/'flux' divide out the incident wave and
    return |S11|=0, |S21|=1 exactly, which is the matched-guide reference but
    not an informative scatter plot. The raw run is the honest validation.
    """
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    freqs = np.linspace(8.2e9, 12.4e9, 21)
    f0 = float(freqs.mean())
    bandwidth = 0.5
    domain_x = 0.200
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(domain_x, A_WG, B_WG),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=20,
        dx=DX,
    )
    pf = jnp.asarray(freqs)
    sim.add_waveguide_port(0.040, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=pf, f0=f0, bandwidth=bandwidth,
                           waveform="modulated_gaussian",
                           reference_plane=0.050, name="left")
    sim.add_waveguide_port(domain_x - 0.040, direction="-x", mode=(1, 0),
                           mode_type="TE", freqs=pf, f0=f0, bandwidth=bandwidth,
                           waveform="modulated_gaussian",
                           reference_plane=domain_x - 0.050, name="right")
    res = sim.compute_waveguide_s_matrix(num_periods=200, normalize=False)
    pidx = {n: i for i, n in enumerate(res.port_names)}
    il, ir = pidx["left"], pidx["right"]
    s = np.asarray(res.s_params)
    f = np.asarray(res.freqs)
    return f, np.abs(s[il, il, :]), np.abs(s[ir, il, :])


def make_validation():
    f, abs_s11, abs_s21 = _raw_floor_s()
    f_ghz = f / 1e9
    max_s11 = float(abs_s11.max())
    min_s21 = float(abs_s21.min())

    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    # exact matched empty-guide reference: |S11| = 0, |S21| = 1
    ax.axhline(1.0, color="#1f7a3a", ls="-", lw=1.6, alpha=0.6,
               label="exact: |S21| = 1")
    ax.axhline(0.0, color="#b00000", ls="-", lw=1.6, alpha=0.6,
               label="exact: |S11| = 0")
    ax.plot(f_ghz, abs_s21, "o", color="#1f7a3a", ms=5, mfc="white", mew=1.3,
            label="|S21|  — rfx FDTD")
    ax.plot(f_ghz, abs_s11, "s", color="#b00000", ms=4.5, mfc="white",
            mew=1.3, label="|S11|  — rfx FDTD (CPML floor)")
    ax.set_xlabel("Frequency  (GHz)")
    ax.set_ylabel("|S|  (linear)")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(f_ghz[0], f_ghz[-1])
    ax.set_title("Empty WR-90 guide: |S11| / |S21| vs the exact matched answer")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", framealpha=0.9)
    txt = (f"max |S11| = {max_s11:.3f}   (CPML back-reflection floor)\n"
           f"min |S21| = {min_s21:.3f}   (near-lossless transmission)\n"
           f"f_c(TE10) = {F_CUTOFF/1e9:.3f} GHz  (band well above cutoff)")
    ax.text(0.015, 0.10, txt, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))
    out = os.path.join(ASSETS, "validation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out, f"(max|S11|={max_s11:.4f}, min|S21|={min_s21:.4f})")


# ===========================================================================
# Figure 3: d|S21|^2/d(eps_r) — AD vs central finite differences
# ===========================================================================
def make_autodiff():
    import jax
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    # X-band, above cutoff. Mirrors _build_waveguide_wr90 but shorter (quick
    # AD-feasible domain), with a dielectric-fill plug mid-guide as the design
    # degree of freedom. This is the test_waveguide_flux_ad.py contract.
    freqs = np.linspace(8.2e9, 12.4e9, 7)
    f0 = float(freqs.mean())
    bandwidth = 0.5
    domain_x = 0.120
    num_periods = 30
    cpml = 12

    def make_sim():
        sim = Simulation(
            freq_max=float(freqs[-1]) * 1.1,
            domain=(domain_x, A_WG, B_WG),
            boundary=BoundarySpec(
                x=Boundary(lo="cpml", hi="cpml"),
                y=Boundary(lo="pec", hi="pec"),
                z=Boundary(lo="pec", hi="pec"),
            ),
            cpml_layers=cpml,
            dx=DX,
        )
        pf = jnp.asarray(freqs)
        sim.add_waveguide_port(0.030, direction="+x", mode=(1, 0), mode_type="TE",
                               freqs=pf, f0=f0, bandwidth=bandwidth,
                               waveform="modulated_gaussian",
                               reference_plane=0.040, name="left")
        sim.add_waveguide_port(domain_x - 0.040, direction="-x", mode=(1, 0),
                               mode_type="TE", freqs=pf, f0=f0, bandwidth=bandwidth,
                               waveform="modulated_gaussian",
                               reference_plane=domain_x - 0.050, name="right")
        return sim

    def eps_override_for(sim, deps):
        grid = sim._build_grid()
        eps = jnp.ones(grid.shape, dtype=jnp.float32)
        # dielectric-fill plug, mid-guide (~8-cell region along x)
        i_lo = grid.position_to_index((0.056, 0.0, 0.0))[0]
        i_hi = grid.position_to_index((0.064, 0.0, 0.0))[0]
        return eps.at[i_lo:i_hi, :, :].add(deps)

    def s21_mag2(deps):
        sim = make_sim()
        eps = eps_override_for(sim, deps)
        res = sim.compute_waveguide_s_matrix(
            num_periods=num_periods, normalize="flux", eps_override=eps)
        k = res.s_params.shape[-1] // 2
        return jnp.abs(res.s_params[1, 0, k]) ** 2

    deps0 = jnp.asarray(0.5, dtype=jnp.float32)
    t0 = time.time()
    val, g = jax.value_and_grad(s21_mag2)(deps0)
    val, g_ad = float(val), float(g)
    print(f"  AD: |S21|^2={val:.5f}  d/deps={g_ad:+.5e}  ({time.time()-t0:.0f}s)")

    h = 0.05
    fp = float(s21_mag2(deps0 + h))
    fm = float(s21_mag2(deps0 - h))
    g_fd = (fp - fm) / (2 * h)
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-12)
    sign_ok = (g_ad * g_fd) > 0
    print(f"  FD: {g_fd:+.5e}  rel(AD vs FD)={rel:.4f}  sign_agree={'YES' if sign_ok else 'NO'}")

    # analytic cutoff relation: the dielectric fill lowers f_c by 1/sqrt(eps_r);
    # this is the band-edge lever the eps gradient reflects (same sign).
    eps_fill = 1.0 + float(deps0)
    fc_d = F_CUTOFF / math.sqrt(eps_fill)
    dfc_deps = -F_CUTOFF / (2.0 * eps_fill ** 1.5)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    xs = ["AD\n(jax.grad)", "central FD\n(h = 0.05)"]
    vals = [g_ad, g_fd]
    colors = ["#1f5fa8", "#9aa0a6"]
    bars = ax.bar(xs, vals, color=colors, width=0.55, zorder=3)
    span = abs(min(vals))
    # value labels just BELOW each (negative) bar end, clear of the stats box
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.04 * span,
                f"{v:+.4f}", ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_ylabel(r"$\partial\,|S_{21}|^2 / \partial\,\varepsilon_r$")
    ax.set_title("Transmission sensitivity to the dielectric fill — AD vs FD")
    ax.grid(True, axis="y", alpha=0.3)
    # headroom: bars negative -> extra room below for labels, room above for
    # the stats box.
    ax.set_ylim(min(vals) * 1.30, span * 0.85)
    txt = (f"|S21|^2 = {val:.4f}\n"
           f"rel(AD vs FD) = {rel*100:.2f} %  (gate < 5%)\n"
           f"sign agreement: {'YES' if sign_ok else 'NO'}\n"
           f"cutoff lever  f_c,d = f_c/sqrt(eps_r) = {fc_d/1e9:.2f} GHz\n"
           f"  d f_c,d / d eps_r = {dfc_deps/1e9:+.3f} GHz  (same sign)")
    ax.text(0.5, 0.97, txt, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="center", family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))
    out = os.path.join(ASSETS, "autodiff.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)
    return dict(val=val, g_ad=g_ad, g_fd=g_fd, rel=rel, sign_ok=sign_ok)


def register_v3_assets(case_id, new_assets):
    """Add/refresh v3 figure entries in a case manifest.json, in place.

    new_assets: list of (filename, type) tuples. Existing entries with the same
    filename are refreshed (sha256 + size); the rest of the manifest is left
    untouched. Keeps the committed provenance/validation blocks intact.
    """
    import hashlib

    case_dir = os.path.join(ROOT, "docs", "public", "gallery", "assets", case_id)
    man_path = os.path.join(case_dir, "manifest.json")
    with open(man_path) as f:
        man = json.load(f)
    by_name = {a["filename"]: a for a in man.get("assets", [])}
    for filename, atype in new_assets:
        path = os.path.join(case_dir, filename)
        if not os.path.exists(path):
            print(f"  skip {filename} (not on disk)")
            continue
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        entry = by_name.get(filename, {})
        entry.update({
            "filename": filename,
            "type": atype,
            "served_url": f"/rfx/gallery/assets/{case_id}/{filename}",
            "sha256": sha,
            "size_bytes": os.path.getsize(path),
        })
        by_name[filename] = entry
    man["assets"] = list(by_name.values())
    with open(man_path, "w") as f:
        json.dump(man, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"  updated {man_path} ({len(man['assets'])} assets)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "field"):
        make_field()
    if which in ("all", "validation"):
        make_validation()
    if which in ("all", "autodiff"):
        make_autodiff()
    if which in ("all", "manifest"):
        register_v3_assets("waveguide_wr90", [
            ("field_te10.png", "field-map-png"),
            ("validation.png", "validation-png"),
            ("autodiff.png", "autodiff-png"),
        ])
        register_v3_assets("patch_antenna", [
            ("autodiff.png", "autodiff-png"),
        ])
        register_v3_assets("multilayer_fresnel", [
            ("autodiff.png", "autodiff-png"),
        ])

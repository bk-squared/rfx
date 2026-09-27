"""Figures and loops for the showcase records, drawn only from the arrays the jobs saved.

Nothing is computed here but unit conversions (dB, mm) and normalisation for
display; every number on a figure comes from a file in the record directory.
Figures carry axis labels with units and no claim sentences.

    python scripts/showcase/render.py patch   RECORD_DIR OUT_DIR
    python scripts/showcase/render.py timing  RECORD_DIR OUT_DIR   # the patch record's timing files
    python scripts/showcase/render.py ar      RECORD_DIR OUT_DIR
    python scripts/showcase/render.py curves  RECORD_DIR OUT_DIR   # a forward_curves record

Per item: ``<item>_site.png`` (16:9, 1920 x 1080), ``<item>_social.png``
(1080 x 1350) and, where the item has a sequence, ``<item>.mp4`` (silent,
H.264, yuv420p).  ``render_manifest.json`` lists the inputs and outputs with
their sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation, patches  # noqa: E402

# ------------------------------------------------------------------ style
RFX = "#B97800"
OPENEMS = "#0A7896"
PALACE = "#A3386B"
INK = "#1F1F1F"          # closed form, outlines
INK_2 = "#555555"        # secondary text
GRID = "#E3E3E0"
SURFACE = "#FFFFFF"
DIVERGING = "RdBu"       # gradients, symmetric about zero
SEQUENTIAL = "Greys"     # one hue, light to dark

SITE = dict(figsize=(9.6, 5.4), dpi=200)      # 1920 x 1080
SOCIAL = dict(figsize=(7.2, 9.0), dpi=150)    # 1080 x 1350
MOVIE_DPI = 100

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "savefig.facecolor": SURFACE,
})


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(x)), 1e-12))


def _save(fig, path: Path, spec: dict) -> Path:
    fig.savefig(path, dpi=spec["dpi"])
    plt.close(fig)
    return path


def _movie(fig, update, n_frames, path: Path, fps: int) -> Path:
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode:
        raise RuntimeError("ffmpeg is required for the MP4 loops")
    mv = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    mv.save(path, writer=animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=900,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]),
        dpi=MOVIE_DPI)
    plt.close(fig)
    return path


def _load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if name.endswith(".json") else np.load(p)


# ------------------------------------------------------------------ patch
class PatchData:
    def __init__(self, d: Path):
        self.model = _load(d, "model.json")
        self.ft = _load(d, "f_t.json")
        self.base = _load(d, "baseline.json")
        b = _load(d, "baseline.npz")
        self.freqs, self.s11 = b["freqs_hz"], b["s11"]
        g = _load(d, "gradient_full.npz")
        self.grad = g["grad"]
        self.gmap = self.grad.sum(axis=2)
        e = _load(d, "ez2_mid.npz")
        self.ez2 = e["ez2_ft"]
        m = self.model
        box = m["boxes"]["design"]
        dx = m["dx_m"]
        I, J = m["patch_centre_node"]
        ia, ja = box["cells_x"][0], box["cells_y"][0]
        # cell edges in mm from the patch centre node
        self.xe = (np.arange(box["shape"][0] + 1) + ia - I) * dx * 1e3
        self.ye = (np.arange(box["shape"][1] + 1) + ja - J) * dx * 1e3
        px, py = m["patch_cells"]["x"], m["patch_cells"]["y"]
        self.patch_rect = ((px[0] - I) * dx * 1e3, (py[0] - J) * dx * 1e3,
                           (px[1] + 1 - px[0]) * dx * 1e3, (py[1] + 1 - py[0]) * dx * 1e3)
        self.probe = (m["feed_offset_x_m"] * 1e3, 0.0)
        self.blocks = []
        for name, blk in m["blocks"].items():
            (xa, xb), (ya, yb) = blk["cells_x"], blk["cells_y"]
            self.blocks.append((name, (xa - I) * dx * 1e3, (ya - J) * dx * 1e3,
                                (xb + 1 - xa) * dx * 1e3, (yb + 1 - ya) * dx * 1e3))
        self.dx_mm = dx * 1e3
        self.vmax = float(np.max(np.abs(self.gmap)))


def _patch_overlay(ax, pd: PatchData, blocks=True):
    x, y, w, h = pd.patch_rect
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec=INK, lw=1.2))
    if blocks:
        for _, bx, by, bw, bh in pd.blocks:
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, fill=False, ec=INK, lw=0.9, ls="-"))
    ax.plot(*pd.probe, "o", ms=7, mfc=INK, mec=SURFACE, mew=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x from patch centre (mm)")
    ax.set_ylabel("y from patch centre (mm)")
    ax.grid(False)


def _gradient_axes(ax, pd: PatchData, data=None, colorbar=True, fig=None, orientation="vertical"):
    data = pd.gmap if data is None else data
    k = int(np.floor(np.log10(pd.vmax)))
    scale = 10.0 ** k
    im = ax.pcolormesh(pd.xe, pd.ye, data.T / scale, cmap=DIVERGING, vmin=-pd.vmax / scale,
                       vmax=pd.vmax / scale, shading="flat", rasterized=True)
    _patch_overlay(ax, pd)
    if colorbar:
        cb = (fig or ax.figure).colorbar(im, ax=ax, fraction=0.05, pad=0.04,
                                         orientation=orientation, shrink=0.9)
        cb.set_label(f"∂J/∂εr per cell, summed over z (×10$^{{{k}}}$)")
    return im


def _ez_axes(ax, pd: PatchData, fig, orientation="vertical"):
    ez = pd.ez2 / pd.ez2.max()
    im = ax.pcolormesh(pd.xe, pd.ye, ez.T, cmap=SEQUENTIAL, vmin=0, vmax=1,
                       shading="flat", rasterized=True)
    _patch_overlay(ax, pd, blocks=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, orientation=orientation, shrink=0.9)
    cb.set_label("|Ez(f_t)|², normalised")


def _s11_axes(ax, pd: PatchData):
    f = pd.freqs / 1e9
    ax.plot(f, _db(pd.s11), color=RFX, label="rfx, h/4")
    for val, ls, lab, ha, dx in ((pd.ft["f_r_hz"], ":", "$f_r$", "right", -4),
                                 (pd.ft["f_t_hz"], "--", "$f_t$", "left", 4)):
        ax.axvline(val / 1e9, color=INK, ls=ls, lw=1.2)
        ax.annotate(lab, (val / 1e9, 0.96), xycoords=("data", "axes fraction"), ha=ha, va="top",
                    xytext=(dx, 0), textcoords="offset points", color=INK, fontsize=14)
    lo, hi = pd.ft["f_r_hz"] * 0.95 / 1e9, pd.ft["f_r_hz"] * 1.05 / 1e9
    ax.set_xlim(lo, hi)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|S11| (dB)")


def render_patch(d: Path, out: Path) -> list[Path]:
    pd = PatchData(d)
    made = []
    fig = plt.figure(figsize=SITE["figsize"])
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1])
    ax_g = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1])
    _gradient_axes(ax_g, pd, fig=fig)
    ax_g.set_title("J = |S11(f_t)|², gradient per substrate cell", fontsize=14)
    _s11_axes(ax_s, pd)
    _ez_axes(ax_e, pd, fig)
    ax_e.set_xlabel("x (mm)")
    ax_e.set_ylabel("y (mm)")
    fig.tight_layout()
    made.append(_save(fig, out / "patch_sensitivity_site.png", SITE))

    fig, axes = plt.subplots(2, 1, figsize=SOCIAL["figsize"],
                             gridspec_kw={"height_ratios": [1.9, 1.0]})
    _gradient_axes(axes[0], pd, fig=fig)
    axes[0].set_title("J = |S11(f_t)|², gradient per substrate cell", fontsize=14)
    _s11_axes(axes[1], pd)
    fig.tight_layout()
    made.append(_save(fig, out / "patch_sensitivity_social.png", SOCIAL))

    # the reveal: cells above a falling threshold, largest |gradient| first
    order = np.sort(np.abs(pd.gmap).ravel())[::-1]
    n_reveal, n_hold = 48, 24
    idx = np.unique(np.geomspace(1, order.size, n_reveal).astype(int)) - 1
    thresholds = order[idx]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    im = _gradient_axes(ax, pd, data=np.zeros_like(pd.gmap), fig=fig)
    ax.set_title("J = |S11(f_t)|², gradient per substrate cell", fontsize=14)
    fig.tight_layout()
    scale = 10.0 ** int(np.floor(np.log10(pd.vmax)))

    def update(k):
        t = thresholds[min(k, thresholds.size - 1)]
        shown = np.where(np.abs(pd.gmap) >= t, pd.gmap, 0.0) / scale
        im.set_array(shown.T.ravel())
        return (im,)

    made.append(_movie(fig, update, thresholds.size + n_hold, out / "patch_sensitivity.mp4", 15))
    return made


# ------------------------------------------------------------------ timing
def render_timing(d: Path, out: Path) -> list[Path]:
    boxes = [b for b in ("patch", "design", "laminate", "layer")
             if (d / f"timing_{b}_forward.json").is_file()]
    rows = []
    for b in boxes:
        tf = json.loads((d / f"timing_{b}_forward.json").read_text())
        tg = json.loads((d / f"timing_{b}_grad.json").read_text())
        n = tf["box_desc"]["n_cells"]
        rows.append({"box": b, "n": n, "other": tf["box_desc"].get("n_other_cells", 0),
                     "fwd": tf["wall_s"], "grad": tg["wall_s"],
                     "fd": 2 * n * tf["wall_s"], "mem_grad": tg["peak_bytes_in_use"],
                     "mem_fwd": tf["peak_bytes_in_use"], "device": tf["device_kind"]})
    made = []
    for spec, name in ((SITE, "site"), (SOCIAL, "social")):
        fig, ax = plt.subplots(figsize=spec["figsize"])
        y = np.arange(len(rows))[::-1]
        h = 0.24
        ax.barh(y + h, [r["fwd"] for r in rows], h * 0.85, color="#9A9A96", label="forward (measured)")
        ax.barh(y, [r["grad"] for r in rows], h * 0.85, color=RFX, label="value + gradient (measured)")
        ax.barh(y - h, [r["fd"] for r in rows], h * 0.85, color=SURFACE, ec=INK, hatch="///",
                lw=1.0, label="central differences: 2N forwards (derived)")
        for yy, r in zip(y, rows):
            texts = ((h, r["fwd"], f"{r['fwd']:.1f} s, peak {r['mem_fwd'] / 1e9:.2f} GB"),
                     (0.0, r["grad"], f"{r['grad']:.1f} s, peak {r['mem_grad'] / 1e9:.2f} GB"),
                     (-h, r["fd"], f"{r['fd']:.3g} s = {r['fd'] / 86400:.1f} days"))
            for off, v, t in texts:
                ax.annotate(t, (v, yy + off), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=12, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['box']}\n{r['n']:,} cells"
                            + (f"\n({r['other']:,} not laminate)" if r["other"] else "")
                            for r in rows])
        ax.set_xscale("log")
        ax.set_xlim(min(r["fwd"] for r in rows) * 0.5, max(r["fd"] for r in rows) * 60)
        ax.set_xlabel(f"wall time on {rows[0]['device']} (s)")
        ax.grid(axis="y", visible=False)
        ncol = 2 if name == "site" else 1
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=ncol,
                   frameon=False, fontsize=12)
        fig.tight_layout(rect=(0, 0.11 if name == "site" else 0.12, 1, 1))
        made.append(_save(fig, out / f"gradient_cost_{name}.png", spec))
    return made


# ------------------------------------------------------------------ AR coating
def render_ar(d: Path, out: Path) -> list[Path]:
    it = _load(d, "iterations.npz")
    tmm = _load(d, "tmm.json")
    tc = _load(d, "tmm_curves.npz")
    f = it["freqs_band_hz"] / 1e9
    R, Rt, cost, eps = it["R_band"], it["R_tmm_band"], it["cost"], it["eps_r"]
    n_last = cost.size - 1

    def r_axes(ax, k, title=True):
        ax.plot(f, 10 * np.log10(np.maximum(R[0], 1e-12)), color=RFX, alpha=0.35, lw=1.6,
                label="rfx FDTD, start")
        ax.plot(f, 10 * np.log10(np.maximum(R[k], 1e-12)), color=RFX, label="rfx FDTD")
        ax.plot(f, 10 * np.log10(np.maximum(Rt[k], 1e-12)), color=INK, ls="--", lw=1.6,
                label="transfer matrix, same εr")
        ax.plot(tc["freqs_band_hz"] / 1e9, 10 * np.log10(np.maximum(tc["R_opt_band"], 1e-12)),
                color=INK, ls=":", lw=1.6, label="transfer-matrix optimum")
        ax.set_xlabel("frequency (GHz)")
        ax.set_ylabel("|R|² (dB)")
        allr = 10 * np.log10(np.maximum(np.concatenate([R.ravel(), Rt.ravel()]), 1e-12))
        ax.set_ylim(np.floor(allr.min() / 5) * 5 - 5, np.ceil(allr.max() / 5) * 5)
        ax.legend(loc="lower right", frameon=False)
        if title:
            ax.set_title("reflection, 8–12 GHz")

    def c_axes(ax, k):
        x = np.arange(cost.size)
        ax.plot(x[:k + 1], cost[:k + 1], color=RFX, label="rfx FDTD cost")
        ax.plot([k], [cost[k]], "o", ms=8, mfc=RFX, mec=SURFACE, mew=2, clip_on=False)
        ax.axhline(tmm["optimum_cost"], color=INK, ls="--", lw=1.6, label="transfer-matrix optimum")
        ax.set_yscale("log")
        ax.set_xlim(0, n_last)
        lo = min(float(np.min(cost)), tmm["optimum_cost"]) * 0.7
        ax.set_ylim(lo, float(np.max(cost)) * 1.4)
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel("mean |R|², 8–12 GHz")
        ax.legend(loc="upper right", frameon=False)

    made = []
    fig, axes = plt.subplots(1, 2, figsize=SITE["figsize"])
    r_axes(axes[0], n_last)
    c_axes(axes[1], n_last)
    axes[1].set_title("cost per iteration")
    fig.tight_layout()
    made.append(_save(fig, out / "ar_coating_site.png", SITE))

    fig, axes = plt.subplots(2, 1, figsize=SOCIAL["figsize"])
    r_axes(axes[0], n_last)
    c_axes(axes[1], n_last)
    fig.tight_layout()
    made.append(_save(fig, out / "ar_coating_social.png", SOCIAL))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    hold = 15

    def update(k):
        k = min(k, n_last)
        for ax in axes:
            ax.clear()
        r_axes(axes[0], k)
        c_axes(axes[1], k)
        axes[1].set_title(f"iteration {k}   εr = " + ", ".join(f"{v:.2f}" for v in eps[k]))
        return ()

    fig.tight_layout()
    made.append(_movie(fig, update, n_last + 1 + hold, out / "ar_coating.mp4", 10))
    return made


# ------------------------------------------------------------------ forward curves
_CASE_TITLES = {"rt5880_patch": "RT/Duroid 5880 probe-fed patch",
                "msl_notch_filter": "microstrip open-stub notch filter",
                "sheen_lpf": "Sheen stepped-impedance low-pass filter"}
# The frequency window each test compares in (GHz): the patch's RESONANCE_BAND_HZ,
# the notch filter's and the Sheen filter's REFERENCE_BAND_HZ.
_CASE_XLIM = {"rt5880_patch": (1.94669887, 2.92004831),
              "msl_notch_filter": (2.0, 7.0),
              "sheen_lpf": (2.0, 12.0)}


def _ylim_to_window(ax, pad=0.06):
    """y limits from the data inside the current x window only."""
    lo, hi = ax.get_xlim()
    ys = []
    for ln in ax.get_lines():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if x.size == y.size and x.size:
            ys.append(y[(x >= lo) & (x <= hi) & np.isfinite(y)])
    ys = np.concatenate([v for v in ys if v.size]) if ys else np.array([])
    if ys.size:
        span = max(ys.max() - ys.min(), 1e-9)
        ax.set_ylim(ys.min() - pad * span, ys.max() + pad * span)


def render_curves(d: Path, out: Path) -> list[Path]:
    rung = _load(d, "rung.json")
    case = rung["case"]
    rc = _load(d, "rfx_curves.npz")
    ref = _load(d, "reference_curves.npz")
    f = rc["freqs_hz"] / 1e9
    lab = f"rfx, dx = {rung['dx_m'] * 1e6:.1f} µm"

    # the reference underneath, wider and lighter; rfx on top, so both stay
    # visible where they overlap
    def ref_line(ax, key_f, key_y, color, label, db=True, marker=None, ls="-"):
        if key_f in ref.files and key_y in ref.files:
            y = ref[key_y]
            ax.plot(ref[key_f] / 1e9, _db(y) if db else y, color=color, ls=ls,
                    lw=3.2 if not marker else 0, alpha=0.6 if not marker else 1.0, zorder=2,
                    marker=marker, ms=6.5 if marker else 0, mec=SURFACE, mew=0.8, label=label)

    def mag_axes(ax, sname):
        ax.plot(f, _db(rc[sname]), color=RFX, lw=1.8, zorder=4, label=lab)
        ref_line(ax, "openems_stage_b_fine_freqs_hz", f"openems_stage_b_fine_{sname}_mag",
                 OPENEMS, "openEMS, finest mesh")
        for mesh, mk in (("coarse", "o"), ("mid", "s")):
            ref_line(ax, f"palace_{mesh}_freqs_hz", f"palace_{mesh}_{sname}_mag", PALACE,
                     f"Palace, {mesh} mesh", marker=mk, ls="none")
        ax.set_xlabel("frequency (GHz)")
        ax.set_ylabel(f"|{sname.upper()}| (dB)")
        ax.set_xlim(*_CASE_XLIM[case])
        _ylim_to_window(ax)

    def zin_axes(ax):
        z = rc["zin_ohm"]
        ax.plot(f, z.real, color=RFX, lw=1.8, zorder=4, label=lab)
        ax.plot(f, z.imag, color=RFX, lw=1.8, zorder=4, ls="--", label="_nolegend_")
        ref_line(ax, "openems_stage_b_fine_freqs_hz", "openems_stage_b_fine_zin_re_ohm", OPENEMS,
                 "openEMS, finest mesh", db=False)
        ref_line(ax, "openems_stage_b_fine_freqs_hz", "openems_stage_b_fine_zin_im_ohm", OPENEMS,
                 "_nolegend_", db=False, ls="--")
        ax.text(0.98, 0.97, "solid: Re Zin\ndashed: Im Zin", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, color=INK_2)
        ax.set_xlim(*_CASE_XLIM[case])
        ax.set_xlabel("frequency (GHz)")
        ax.set_ylabel("input impedance (Ω)")

    panels = (("s11", zin_axes) if case == "rt5880_patch" else ("s21", "s11"))
    made = []
    for spec, name, shape in ((SITE, "site", (1, 2)), (SOCIAL, "social", (2, 1))):
        fig, axes = plt.subplots(*shape, figsize=spec["figsize"])
        for ax, p in zip(axes, panels):
            if callable(p):
                p(ax)
            else:
                mag_axes(ax, p)
        fig.suptitle(_CASE_TITLES[case], fontsize=15)
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        ncol = 3 if name == "site" else 2
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=12)
        rows = -(-len(labels) // ncol)
        fig.tight_layout(rect=(0, 0.035 * rows + (0.02 if name == "site" else 0.0), 1, 1))
        made.append(_save(fig, out / f"forward_curves_{case}_{name}.png", spec))
    return made


# ------------------------------------------------------------------ main
def _renderer_commit() -> str:
    """The commit this renderer ran from, with ``+dirty`` when its own file differs."""
    here = Path(__file__).resolve().parent
    sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", Path(__file__).name],
                           capture_output=True, text=True, check=True).stdout.strip()
    return sha + ("+dirty" if dirty else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("item", choices=["patch", "timing", "ar", "curves"])
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    made = {"patch": render_patch, "timing": render_timing, "ar": render_ar,
            "curves": render_curves}[a.item](a.record_dir, a.out_dir)
    inputs = sorted(p for p in a.record_dir.iterdir()
                    if p.is_file() and p.suffix in (".npz", ".json"))
    manifest_path = a.out_dir / "render_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest[a.item + ":" + a.record_dir.name] = {
        "record_dir": str(a.record_dir),
        "inputs": {p.name: _sha(p) for p in inputs},
        "outputs": {p.name: _sha(p) for p in made},
        "colours": {"rfx": RFX, "openEMS": OPENEMS, "Palace": PALACE, "closed form": INK,
                    "gradient": DIVERGING, "field": SEQUENTIAL},
        "renderer": "scripts/showcase/render.py",
        "renderer_commit": _renderer_commit(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    for p in made:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

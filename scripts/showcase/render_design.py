"""Films and contact sheets of the two design loops (issue 1359), drawn from the saved arrays.

Three panels share one clock, left to right on the site film and stacked on the
social one: the design being changed, the response it produces, and the
gradient that sets the next change.  Nothing is computed here except unit
conversions (dB, mm), the E-plane cut of the stored directivity, and the
display scaling of the gradient.  Every number on a frame comes from a file in
the record directory.

    python scripts/showcase/render_design.py taper RECORD_DIR OUT_DIR
    python scripts/showcase/render_design.py beam  RECORD_DIR OUT_DIR

Per case: ``<case>_site.mp4`` (1920 x 1080, gradient on a scale fixed from the
first iterations), ``<case>_site_normscale.mp4`` (the same, gradient normalized
per frame with its norm printed), ``<case>_social.mp4`` (1080 x 1350, fixed
scale), ``<case>_poster.png`` (the site film's last frame) and
``<case>_contact.png`` (rows: iterations 0, ~25 %, ~50 % and the final design;
columns: design, result, gradient on the fixed scale, gradient normalized).
H.264, yuv420p, no audio.  ``render_manifest_<case>.json`` lists inputs and
outputs with their sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation, patches  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

# ------------------------------------------------------------------ style
RFX = "#B97800"            # the design's own response
REF = "#0A7896"            # the reference / baseline
INK = "#1F1F1F"
INK_2 = "#555555"
MUTED = "#8A8A86"          # the start curve
GRID = "#E3E3E0"
SURFACE = "#FFFFFF"
EPS_CMAP = LinearSegmentedColormap.from_list(
    "rfx_amber", ["#FBF5EA", "#EBC983", "#B97800", "#5A3B00"])   # one hue, light to dark
GRAD_CMAP = "RdBu_r"       # positive (raising eps_r raises the cost) red, negative blue

MOVIE_DPI = 100
FPS = 30
# point sizes at 100 dpi: 22 pt = 30.6 px, the smallest text on either film
PT = dict(tick=22, label=24, title=26, header=32, sub=27, legend=22, note=22)
SITE = dict(size=(19.2, 10.8), intro_s=3.0, main_s=None, hold_s=2.0, frames_per_iter=3)
SOCIAL = dict(size=(10.8, 13.5), intro_s=2.0, main_s=None, hold_s=2.0, frames_per_iter=2)
FIXED_SCALE_ITERS = 3      # the fixed gradient scale: max |g| over iterations 0 .. 2

plt.rcParams.update({
    "font.size": PT["tick"], "axes.titlesize": PT["title"], "axes.labelsize": PT["label"],
    "xtick.labelsize": PT["tick"], "ytick.labelsize": PT["tick"], "legend.fontsize": PT["legend"],
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "lines.linewidth": 2.5,
    "lines.solid_capstyle": "round", "savefig.facecolor": SURFACE,
    "axes.titlepad": 12, "axes.titleweight": "bold",
})


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _db10(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-30))


def _db20(x):
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(x, dtype=float)), 1e-30))


def _num(v: float, fmt: str = ".1f") -> str:
    """A number with a typographic minus sign."""
    return format(v, fmt).replace("-", "\u2212")


def _titles(ax, title: str, subtitle: str | None = None):
    """Panel title (bold) and an optional plain line under it, above the axes."""
    off = 8 + (PT["note"] + 10 if subtitle else 0)
    t = ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, off),
                    textcoords="offset points", fontsize=PT["title"], weight="bold",
                    va="bottom", ha="left", annotation_clip=False)
    sub = None
    if subtitle:
        sub = ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 8),
                          textcoords="offset points", fontsize=PT["note"], color=INK_2,
                          va="bottom", ha="left", annotation_clip=False)
    return t, sub


def _sci(v: float) -> str:
    e = int(np.floor(np.log10(abs(v)))) if v else 0
    return f"{v / 10 ** e:.2f}×10$^{{{e}}}$"


# =================================================================== taper
class TaperData:
    FILES = ("iterations.npz", "baselines_a36.npz", "baselines_a36.json", "model.json")

    def __init__(self, d: Path):
        it = np.load(d / "iterations.npz")
        self.eps = it["eps_sec"]
        self.s11 = it["abs_s11"]
        self.J = it["J"]
        self.grad = it["grad_eps"]
        self.freqs = it["freqs_hz"] / 1e9
        self.edges = np.asarray(it["section_edges_mm"], dtype=float)
        b = np.load(d / "baselines_a36.npz")
        self.s11_klop = np.abs(b["s11_klopfenstein"])
        self.s11_start = np.abs(b["s11_flat_start"])
        bj = json.loads((d / "baselines_a36.json").read_text())
        self.J_klop = float(bj["klopfenstein"]["J"])
        m = json.loads((d / "model.json").read_text())
        self.fill_mm = float(m["fill_mm"])
        self.eps_load = float(m["eps_load"])
        self.N = len(self.J) - 1                       # iterate N: the final design
        self.n_grad = int(np.sum(np.all(np.isfinite(self.grad), axis=1)))
        self.gfix = float(np.nanmax(np.abs(self.grad[:min(FIXED_SCALE_ITERS, self.n_grad)])))
        allc = np.concatenate([_db20(self.s11).ravel(), _db20(self.s11_klop)])
        self.ylo = float(np.floor((allc.min() - 3.0) / 10.0) * 10.0)
        self.goal = "Match WR-90 to a dielectric load with 30 graded sections"
        self.subgoal = None
        self.x_lo, self.x_hi = self.edges[0] - 14.0, self.fill_mm + 24.0

    def objective(self, k: int, short: bool = False) -> str:
        v = _num(_db10(self.J[k]))
        return f"band-mean |S11|² = {v} dB" if short else f"band-mean |S11|², 8.2–12.4 GHz = {v} dB"

    def strip(self, short: bool = False) -> str:
        a, b, c = (_num(_db10(x)) for x in (self.J[0], self.J[-1], self.J_klop))
        if short:
            return (f"band-mean |S11|²: start {a} dB → final {b} dB\n"
                    f"Klopfenstein profile, A tuned for this band: {c} dB")
        return (f"band-mean |S11|²:  start {a} dB  →  final {b} dB"
                f"      Klopfenstein profile, A tuned for this band: {c} dB")

    def legend(self, short: bool = False):
        h = [plt.Line2D([], [], color=RFX, lw=3.5, marker="o", ms=6),
             plt.Line2D([], [], color=REF, lw=2.5, ls="--"),
             plt.Line2D([], [], color=MUTED, lw=2.5, ls=":")]
        if short:
            return h, ["this iteration", "Klopfenstein (A tuned)", "start"]
        return h, ["this iteration", "Klopfenstein profile, A tuned for this band (closed form)",
                   "start (εr = 5 in every section)"]


class TaperDesign:
    def __init__(self, fig, rect, td: TaperData, fmt: str):
        x, y, w, h = rect
        self.td = td
        self.ax = fig.add_axes((x, y + 0.27 * h, w, 0.73 * h))
        self.strip = fig.add_axes((x, y, w, 0.15 * h), sharex=self.ax)
        ax, st = self.ax, self.strip
        e = td.edges
        self.norm = Normalize(1.0, td.eps_load)
        self.bars = [ax.add_patch(patches.Rectangle((e[i], 0), e[i + 1] - e[i], 1.0, lw=1.5,
                                                    ec=SURFACE, fc=EPS_CMAP(0.0)))
                     for i in range(len(e) - 1)]
        ax.add_patch(patches.Rectangle((td.fill_mm, 0), td.x_hi - td.fill_mm, td.eps_load, lw=0,
                                       fc=EPS_CMAP(1.0)))
        ax.plot([td.x_lo, e[0]], [1, 1], color=INK_2, lw=3)
        ax.text(td.x_lo + 1, 1.4, "air", fontsize=PT["note"], color=INK_2, va="bottom")
        ax.text(td.x_hi - 1.5, td.eps_load - 0.4, f"load\nεr {td.eps_load:g}", fontsize=PT["note"],
                color=SURFACE, ha="right", va="top", weight="bold")
        ax.set_xlim(td.x_lo, td.x_hi)
        ax.set_ylim(0, 10.5)
        ax.set_yticks([1, 5, 9])
        ax.set_ylabel("εr")
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelbottom=False)
        _titles(ax, "Design", "permittivity of each section" if fmt != "social" else None)
        # the side view of the guide: the same sections between its broad walls
        self.cells = [st.add_patch(patches.Rectangle((e[i], 0), e[i + 1] - e[i], 1.0, lw=0,
                                                     fc=EPS_CMAP(0.0)))
                      for i in range(len(e) - 1)]
        st.add_patch(patches.Rectangle((td.fill_mm, 0), td.x_hi - td.fill_mm, 1.0, lw=0,
                                       fc=EPS_CMAP(1.0)))
        for yy in (0, 1):
            st.plot([td.x_lo, td.x_hi], [yy, yy], color=INK, lw=5, solid_capstyle="butt")
        st.annotate("TE10 →", (td.x_lo, 0.5), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=PT["note"], color=INK, annotation_clip=False)
        st.set_ylim(-0.15, 1.15)
        st.set_yticks([])
        st.grid(False)
        for sp in ("left", "right", "top", "bottom"):
            st.spines[sp].set_visible(False)
        st.set_xlabel("position along the guide (mm)")
        self.axes = (ax, st)

    def update(self, k: int):
        for b, c, v in zip(self.bars, self.cells, self.td.eps[k]):
            col = EPS_CMAP(self.norm(v))
            b.set_height(v)
            b.set_facecolor(col)
            c.set_facecolor(col)

    def set_visible(self, v: bool):
        for a in self.axes:
            a.set_visible(v)


class TaperResult:
    def __init__(self, fig, rect, td: TaperData, fmt: str):
        self.td = td
        ax = self.ax = fig.add_axes(rect)
        f = td.freqs
        ax.plot(f, _db20(td.s11_start), color=MUTED, ls=":", lw=2.5)
        ax.plot(f, _db20(td.s11_klop), color=REF, ls="--", lw=2.5)
        (self.line,) = ax.plot(f, _db20(td.s11[0]), color=RFX, lw=3.5, marker="o", ms=6)
        ax.set_xlim(8.2, 12.4)
        ax.set_xticks([9, 10, 11, 12])
        ax.set_ylim(td.ylo, 0)
        step = 10 if td.ylo >= -50 else 20
        ax.set_yticks(np.arange(td.ylo - td.ylo % step, 1, step))
        ax.set_xlabel("frequency (GHz)")
        ax.set_ylabel("|S11| (dB)")
        if fmt != "social":
            _titles(ax, "Result", "|S11| across X band")
        else:
            _titles(ax, "Result")
            h, lab = td.legend(short=True)
            ax.legend(h, lab, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
                      fontsize=PT["legend"], handlelength=1.1, columnspacing=0.7,
                      handletextpad=0.35, borderaxespad=0.1)

    def update(self, k: int):
        self.line.set_ydata(_db20(self.td.s11[k]))

    def set_visible(self, v: bool):
        self.ax.set_visible(v)


class TaperGradient:
    def __init__(self, fig, rect, td: TaperData, fmt: str, mode: str):
        self.td, self.mode = td, mode
        ax = self.ax = fig.add_axes(rect)
        e = td.edges
        self.centres, self.widths = 0.5 * (e[:-1] + e[1:]), np.diff(e)
        self.bars = ax.bar(self.centres, np.zeros(len(self.centres)), self.widths * 0.82,
                           color=MUTED)
        ax.axhline(0, color=INK_2, lw=1.5)
        ax.set_xlim(td.x_lo, td.x_hi)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("position along the guide (mm)")
        if mode == "fixed":
            e10 = int(np.floor(np.log10(td.gfix)))
            self.scale = 10.0 ** e10
            lim = td.gfix / self.scale * 1.08
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f"∂J/∂εr (×10$^{{{e10}}}$)")
        else:
            self.scale = None
            ax.set_ylim(-1.08, 1.08)
            ax.set_yticks([-1, 0, 1])
            ax.set_ylabel("∂J/∂εr ÷ its max")
            self.norm_text = ax.text(0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top",
                                     fontsize=PT["note"], color=INK)
        ax.text(0.5 * (td.fill_mm + td.x_hi), 0, "load", ha="center", va="center",
                rotation=90, fontsize=PT["note"], color=INK_2)
        self.fmt = fmt
        self.sub_text = "J = band-mean |S11|²; red raises J"
        self.title, self.sub = _titles(ax, "Gradient", self.sub_text if fmt != "social" else None)
        self.cmap = plt.get_cmap(GRAD_CMAP)

    def update(self, k: int):
        kk = min(k, self.td.n_grad - 1)
        g = self.td.grad[kk]
        if self.mode == "fixed":
            vals = g / self.scale
            cols = self.cmap(0.5 + 0.5 * np.clip(g / self.td.gfix, -1, 1))
        else:
            m = float(np.max(np.abs(g)))
            vals = g / m
            cols = self.cmap(0.5 + 0.5 * vals)
            self.norm_text.set_text(f"max |∂J/∂εr| = {_sci(m)}")
        for b, v, c in zip(self.bars, vals, cols):
            b.set_height(v)
            b.set_color(c)
        _last_step_title(self, kk, k)

    def set_visible(self, v: bool):
        self.ax.set_visible(v)


def _last_step_title(panel, kk: int, k: int):
    """The final design has no gradient of its own: its frame shows the last one."""
    if panel.sub is not None:
        panel.sub.set_text(panel.sub_text if kk == k else f"last step, from iteration {kk}")
    else:
        panel.title.set_text("Gradient" if kk == k else f"Gradient, iteration {kk}")


# =================================================================== beam
class BeamData:
    FILES = ("iterations.npz", "baselines_l20.npz", "baselines_l20.json", "model.json")

    def __init__(self, d: Path):
        it = np.load(d / "iterations.npz")
        self.cover = it["eps"].mean(axis=3)                # plan view: the mean of the 3 layers
        self.D = it["D"]
        self.D30 = it["D30_dbi"]
        self.grad = it["grad_eps_map"]                     # dL/d eps_r per cell, summed over the layers
        th = np.asarray(it["theta_deg"], dtype=float)
        self.theta_e = np.concatenate([-th[::-1], th[1:]])
        m = json.loads((d / "model.json").read_text())
        self.dx_mm = float(m["dx_m"]) * 1e3
        self.half_mm = float(m["plate"]["half"]) * 1e3
        self.lam_mm = float(m["lambda_m"]) * 1e3
        self.n_nodes = self.cover.shape[1]
        b = np.load(d / "baselines_l20.npz")
        bj = json.loads((d / "baselines_l20.json").read_text())
        self.D_bare = b["D_uniform_eps1"]
        self.D30_bare = float(bj["uniform_eps1"]["D30_dbi"])
        best = bj["best_uniform"]
        self.best_name, self.best_eps = best, float(bj[best]["eps_r"])
        self.D30_best = float(bj[best]["D30_dbi"])
        self.N = len(self.D30) - 1
        self.n_grad = int(np.sum(np.all(np.isfinite(self.grad.reshape(len(self.grad), -1)), axis=1)))
        self.gfix = float(np.nanmax(np.abs(self.grad[:min(FIXED_SCALE_ITERS, self.n_grad)])))
        cuts = np.array([self.e_plane(self.D[k]) for k in range(len(self.D))])
        self.ymax = float(np.ceil((max(cuts.max(), self.e_plane(self.D_bare).max()) + 2) / 5) * 5)
        self.goal = "Turn a dipole's beam to 30° with a dielectric cover"
        self.subgoal = "objective: raise D(30°) while holding broadside and back lobes down"

    @staticmethod
    def e_plane(D):
        """E-plane (phi = 0 and 180 deg) cut of D(theta, phi) in dBi, theta in -180..180."""
        return 10 * np.log10(np.maximum(np.concatenate([D[::-1, 36], D[1:, 0]]), 1e-6))

    def objective(self, k: int, short: bool = False) -> str:
        v = _num(self.D30[k])
        return f"D(30°) = {v} dBi" if short else f"directivity toward 30° = {v} dBi"

    def strip(self, short: bool = False) -> str:
        a, b, c, d = (_num(x) for x in (self.D30[0], self.D30[-1], self.D30_bare, self.D30_best))
        if short:
            return (f"D(30°): start {a} dBi → final {b} dBi\n"
                    f"no cover {c} dBi · best uniform cover {d} dBi")
        return (f"D(30°):  start {a} dBi  →  final {b} dBi      no cover {c} dBi      "
                f"best uniform cover (εr {self.best_eps:g}) {d} dBi")

    def legend(self, short: bool = False):
        h = [plt.Line2D([], [], color=RFX, lw=3.5), plt.Line2D([], [], color=REF, lw=2.5, ls="--"),
             plt.Line2D([], [], color=MUTED, lw=2.5, ls=":")]
        if short:
            return h, ["this iteration", "no cover", "start"]
        return h, ["this iteration", "no cover", "start (εr ramp 2 → 9 along x)"]


def _cover_axes(ax, bd: BeamData):
    ax.add_patch(patches.Rectangle((-bd.half_mm, -bd.half_mm), 2 * bd.half_mm, 2 * bd.half_mm,
                                   fill=False, ec=INK, lw=2.0, ls="--"))
    q = bd.lam_mm / 8
    ax.plot([-q, q], [0, 0], color=INK, lw=5, solid_capstyle="butt")
    ax.plot([0], [0], "o", ms=10, mfc=INK, mec=SURFACE, mew=2)
    ax.set_aspect("equal")
    lim = bd.half_mm + bd.dx_mm
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([-60, 0, 60])
    ax.set_yticks([-60, 0, 60])
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(False)


def _map_axes(fig, rect):
    """A square map and its colour bar inside ``rect``, centred."""
    x, y, w, h = rect
    W, H = fig.get_size_inches()
    side = min(w * W * 0.78, h * H) / W          # figure fraction, along x
    sh = side * W / H
    x0 = x + (w - side / 0.78) / 2
    y0 = y + h - sh                                # top-aligned, so panel titles line up
    ax = fig.add_axes((x0, y0, side, sh))
    cax = fig.add_axes((x0 + side + 0.03 * w, y0 + 0.1 * sh, 0.035 * w, 0.8 * sh))
    return ax, cax


class BeamDesign:
    def __init__(self, fig, rect, bd: BeamData, fmt: str):
        self.bd = bd
        ax, self.cax = _map_axes(fig, rect)
        self.ax = ax
        n = bd.n_nodes
        e = (np.arange(n + 1) - n / 2) * bd.dx_mm
        self.im = ax.pcolormesh(e, e, bd.cover[0].T, cmap=EPS_CMAP, vmin=1, vmax=10,
                                shading="flat", rasterized=True)
        _cover_axes(ax, bd)
        cb = fig.colorbar(self.im, cax=self.cax)
        cb.set_ticks([1, 5, 10])
        cb.set_label("εr")
        _titles(ax, "Design", "cover εr, mean of its 3 layers" if fmt != "social" else None)
        if fmt == "social":           # beside the colour bar: the space under the map is taken
            ax.annotate("bar: dipole\ndashed: plate\nbelow the cover", (1.68, 0.5),
                        xycoords="axes fraction", fontsize=PT["note"], color=INK_2, ha="left",
                        va="center", annotation_clip=False)
        elif fmt == "site":           # the contact sheet says it once, in its header
            ax.annotate("bar: dipole   dashed: plate below", (0, 0), xycoords="axes fraction",
                        xytext=(0, -78), textcoords="offset points",
                        fontsize=PT["note"], color=INK_2, ha="left",
                        va="top", annotation_clip=False)

    def update(self, k: int):
        self.im.set_array(self.bd.cover[k].T.ravel())

    def set_visible(self, v: bool):
        self.ax.set_visible(v)
        self.cax.set_visible(v)


class BeamResult:
    def __init__(self, fig, rect, bd: BeamData, fmt: str):
        self.bd = bd
        ax = self.ax = fig.add_axes(rect)
        t = bd.theta_e
        ax.plot(t, bd.e_plane(bd.D[0]), color=MUTED, ls=":", lw=2.5)
        ax.plot(t, bd.e_plane(bd.D_bare), color=REF, ls="--", lw=2.5)
        (self.line,) = ax.plot(t, bd.e_plane(bd.D[0]), color=RFX, lw=3.5)
        (self.dot,) = ax.plot([30], [bd.D30[0]], "o", ms=13, mfc=RFX, mec=SURFACE, mew=2.5)
        ax.axvline(30, color=INK, lw=2)
        ax.annotate("30°", (30, 1), xycoords=("data", "axes fraction"), xytext=(6, -6),
                    textcoords="offset points", ha="left", va="top", fontsize=PT["note"], color=INK)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_ylim(-25, bd.ymax)
        ax.set_yticks(np.arange(-20, bd.ymax + 0.1, 10))
        ax.set_xlabel("θ in the E-plane (deg), + toward +x")
        ax.set_ylabel("D (dBi)")
        if fmt != "social":
            _titles(ax, "Result", "directivity pattern, E-plane")
        else:
            _titles(ax, "Result")
            h, lab = bd.legend(short=True)
            ax.legend(h, lab, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
                      fontsize=PT["legend"], handlelength=1.1, columnspacing=0.7,
                      handletextpad=0.35, borderaxespad=0.1)

    def update(self, k: int):
        self.line.set_ydata(self.bd.e_plane(self.bd.D[k]))
        self.dot.set_ydata([self.bd.D30[k]])

    def set_visible(self, v: bool):
        self.ax.set_visible(v)


class BeamGradient:
    def __init__(self, fig, rect, bd: BeamData, fmt: str, mode: str):
        self.bd, self.mode, self.fmt = bd, mode, fmt
        ax, self.cax = _map_axes(fig, rect)
        self.ax = ax
        n = bd.n_nodes
        e = (np.arange(n + 1) - n / 2) * bd.dx_mm
        if mode == "fixed":
            e10 = int(np.floor(np.log10(bd.gfix)))
            self.scale = 10.0 ** e10
            lim = bd.gfix / self.scale
            label = f"∂L/∂εr (×10$^{{{e10}}}$)"
        else:
            self.scale, lim, label = None, 1.0, "∂L/∂εr ÷ its max"
        self.im = ax.pcolormesh(e, e, np.zeros((n, n)), cmap=GRAD_CMAP, vmin=-lim, vmax=lim,
                                shading="flat", rasterized=True)
        _cover_axes(ax, bd)
        cb = fig.colorbar(self.im, cax=self.cax)
        cb.set_label(label)
        if mode != "fixed":
            cb.set_ticks([-1, 0, 1])
        self.sub_text = "per cell, sum of the 3 layers"
        self.title, self.sub = _titles(ax, "Gradient", self.sub_text if fmt != "social" else None)

    def update(self, k: int):
        kk = min(k, self.bd.n_grad - 1)
        g = self.bd.grad[kk]
        if self.mode == "fixed":
            v = g / self.scale
        else:
            m = float(np.max(np.abs(g)))
            v = g / m
            self.sub_text = f"sum over 3 layers · max {_sci(m)}"
        self.im.set_array(v.T.ravel())
        _last_step_title(self, kk, k)

    def set_visible(self, v: bool):
        self.ax.set_visible(v)
        self.cax.set_visible(v)


# =================================================================== films
CASES = {"taper": (TaperData, TaperDesign, TaperResult, TaperGradient),
         "beam": (BeamData, BeamDesign, BeamResult, BeamGradient)}


def _layout(fmt: str):
    """Figure-fraction rectangles: header lines, three panels, the footer."""
    if fmt == "site":
        return dict(goal=(0.03, 0.965), iter=(0.03, 0.895), obj=(0.97, 0.895),
                    panels=[(0.08, 0.20, 0.245, 0.55), (0.40, 0.20, 0.245, 0.55),
                            (0.725, 0.20, 0.245, 0.55)],
                    legend=(0.5, 0.015), strip=(0.5, 0.025), hint=None)
    return dict(goal=(0.04, 0.975), iter=(0.04, 0.878), obj=(0.96, 0.878),
                panels=[(0.17, 0.64, 0.78, 0.165), (0.17, 0.39, 0.78, 0.135),
                        (0.17, 0.155, 0.78, 0.13)],
                legend=None, strip=(0.5, 0.008), hint=(0.5, 0.012))


class Film:
    def __init__(self, case: str, data, fmt: str, grad_mode: str):
        Data, Design, Result, Gradient = CASES[case]
        spec = SITE if fmt == "site" else SOCIAL
        self.data, self.fmt, self.spec = data, fmt, spec
        self.fig = plt.figure(figsize=spec["size"], dpi=MOVIE_DPI)
        L = _layout(fmt)
        if fmt == "social" and case == "beam":
            L["panels"] = [(0.10, 0.61, 0.84, 0.16), (0.17, 0.39, 0.78, 0.115),
                           (0.10, 0.135, 0.84, 0.155)]
            L["hint"] = None
            L["subgoal_y"] = 0.897
            L["iter"], L["obj"] = (0.04, 0.838), (0.96, 0.838)
        if fmt == "site" and case == "beam":
            L["panels"] = [(x, 0.18, w, 0.53) for x, _, w, _ in L["panels"]]
            L["subgoal_y"] = 0.905
            L["iter"], L["obj"] = (0.03, 0.852), (0.97, 0.852)
        short = fmt == "social"
        goal = data.goal.replace(" with ", "\nwith ") if short else data.goal
        self.fig.text(*L["goal"], goal, fontsize=PT["header"], weight="bold", va="top",
                      linespacing=1.15)
        if data.subgoal:
            sub = data.subgoal.replace(" holding ", " holding\n") if short else data.subgoal
            self.fig.text(L["goal"][0], L["subgoal_y"], sub, fontsize=PT["sub"] - (6 if short else 3),
                          va="top", color=INK_2, linespacing=1.1)
        self.t_iter = self.fig.text(*L["iter"], "", fontsize=PT["sub"], va="top", color=INK_2)
        self.t_obj = self.fig.text(*L["obj"], "", fontsize=PT["sub"], va="top", ha="right",
                                   color=INK, weight="bold")
        self.t_hint = None
        if L.get("hint") is not None:
            sym = "J" if case == "taper" else "L"
            what = "J = band-mean |S11|²" if case == "taper" else "L = steering cost"
            self.t_hint = self.fig.text(*L["hint"], f"{what}; red: more εr raises {sym}, blue lowers it",
                                        ha="center", va="bottom",
                                        fontsize=PT["note"], color=INK_2)
        self.design = Design(self.fig, L["panels"][0], data, fmt)
        self.result = Result(self.fig, L["panels"][1], data, fmt)
        self.grad = Gradient(self.fig, L["panels"][2], data, fmt, grad_mode)
        self.leg = None
        if L["legend"] is not None:
            h, lab = data.legend()
            self.leg = self.fig.legend(h, lab, loc="lower center", bbox_to_anchor=L["legend"], ncol=3,
                                       frameon=False, fontsize=PT["legend"], handlelength=2.2,
                                       columnspacing=1.6)
        self.t_strip = self.fig.text(*L["strip"], data.strip(short=True), ha="center",
                                     va="bottom", fontsize=PT["sub"] - 3,
                                     weight="bold", color=INK, linespacing=1.3,
                                     bbox=dict(fc="#FBF5EA", ec=RFX, lw=2.5, pad=12))
        self.t_strip.set_visible(False)

    def draw(self, k: int, show=(True, True, True), final: bool = False):
        d = self.data
        self.design.update(k)
        self.result.update(k)
        self.grad.update(k)
        self.design.set_visible(show[0])
        self.result.set_visible(show[1])
        self.grad.set_visible(show[2])
        tail = "  (final design)" if final and self.fmt == "site" else ""
        self.t_iter.set_text(f"iteration {k} / {d.N}{tail}")
        self.t_obj.set_text(d.objective(k, short=self.fmt == "social"))
        if self.leg is not None:
            self.leg.set_visible(show[1] and not final)
        if self.t_hint is not None:
            self.t_hint.set_visible(show[2] and not final)
        self.t_strip.set_visible(final)

    def schedule(self):
        spec = self.spec
        n_intro = int(round(spec["intro_s"] * FPS))
        third = n_intro // 3
        frames = [(0, (True, i >= third, i >= 2 * third), False) for i in range(n_intro)]
        for k in range(self.data.n_grad):
            frames += [(k, (True, True, True), False)] * spec["frames_per_iter"]
        frames += [(self.data.N, (True, True, True), True)] * int(round(spec["hold_s"] * FPS))
        return frames

    def save(self, path: Path) -> Path:
        if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode:
            raise RuntimeError("ffmpeg is required")
        frames = self.schedule()
        writer = animation.FFMpegWriter(fps=FPS, codec="libx264", bitrate=6000, extra_args=[
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", "-profile:v", "high"])
        with writer.saving(self.fig, str(path), dpi=MOVIE_DPI):
            last = None
            for f in frames:
                if f != last:
                    self.draw(*f)
                    last = f
                writer.grab_frame()
        return path

    def still(self, path: Path, k: int, final: bool = False) -> Path:
        self.draw(k, final=final)
        self.fig.savefig(path, dpi=MOVIE_DPI)
        return path


def contact_sheet(case: str, data, path: Path) -> Path:
    """Rows: iterations 0, ~25 %, ~50 % and the final design. Columns: the three
    panels, the gradient drawn on both scales. Panels are the site film's."""
    Data, Design, Result, Gradient = CASES[case]
    N = data.N
    rows = [0, int(round(0.25 * N)), int(round(0.5 * N)), N]
    fig = plt.figure(figsize=(32.0, 30.0), dpi=MOVIE_DPI)
    fig.text(0.015, 0.992, f"{data.goal}: storyboard", fontsize=PT["header"] + 4, weight="bold",
             va="top")
    if data.subgoal:
        fig.text(0.40, 0.975, data.subgoal, fontsize=PT["sub"], va="top", color=INK_2)
    col_x = [0.10, 0.335, 0.565, 0.79]
    heads = ["Design", "Result", "Gradient, fixed scale", "Gradient, per-frame scale"]
    for c, hname in enumerate(heads):
        fig.text(col_x[c], 0.958, hname, fontsize=PT["title"] + 2, weight="bold", color=RFX)
    fig.text(0.565, 0.945, f"fixed: max |gradient| over iterations 0–{FIXED_SCALE_ITERS - 1}; "
             "per-frame: each frame divided by its own max", fontsize=PT["note"] - 2, color=INK_2,
             va="top")
    if case == "beam":
        fig.text(0.10, 0.945, "bar: dipole · dashed: reflector plate below the cover",
                 fontsize=PT["note"] - 2, color=INK_2, va="top")
    for r, k in enumerate(rows):
        y = 0.735 - r * 0.225
        if r:
            fig.add_artist(plt.Line2D([0.01, 0.99], [y + 0.197, y + 0.197], color=GRID, lw=3))
        fig.text(0.008, y + 0.145, f"iteration\n{k} / {N}" + ("\n(final)" if k == N else ""),
                 fontsize=PT["label"], va="top", weight="bold")
        fig.text(0.008, y + 0.09, data.objective(k, short=True).replace(" = ", "\n= "),
                 fontsize=PT["note"], va="top", color=INK_2)
        rect = lambda c: (col_x[c], y, 0.19, 0.145)  # noqa: E731
        p = [Design(fig, rect(0), data, "sheet"), Result(fig, rect(1), data, "sheet"),
             Gradient(fig, rect(2), data, "sheet", "fixed"), Gradient(fig, rect(3), data, "sheet", "norm")]
        for panel in p:
            panel.update(k)
    h, lab = data.legend()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=PT["legend"] + 2)
    fig.savefig(path, dpi=MOVIE_DPI)
    plt.close(fig)
    return path


def _renderer_commit() -> str:
    repo = Path(__file__).resolve().parents[2]
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() or "unknown (not a git checkout)"


def render(case: str, d: Path, out: Path) -> list[Path]:
    Data = CASES[case][0]
    data = Data(d)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for fmt, mode, name in (("site", "fixed", f"{case}_site.mp4"),
                            ("site", "norm", f"{case}_site_normscale.mp4"),
                            ("social", "fixed", f"{case}_social.mp4")):
        film = Film(case, data, fmt, mode)
        made.append(film.save(out / name))
        if name == f"{case}_site.mp4":
            made.append(film.still(out / f"{case}_poster.png", data.N, final=True))
        plt.close(film.fig)
    made.append(contact_sheet(case, data, out / f"{case}_contact.png"))
    manifest = {"renderer": "scripts/showcase/render_design.py", "renderer_commit": _renderer_commit(),
                "case": case, "record_dir": str(d),
                "inputs": {n: _sha(d / n) for n in Data.FILES},
                "outputs": {p.name: _sha(p) for p in made},
                "display": {"fps": FPS, "fixed_gradient_scale": "max |gradient| over iterations "
                            f"0..{FIXED_SCALE_ITERS - 1}; bars and cells beyond it saturate",
                            "gradient_colormap": GRAD_CMAP, "eps_colormap": "one hue, #FBF5EA -> "
                            "#5A3B00 through #B97800"}}
    (out / f"render_manifest_{case}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)
    for p in render(a.case, a.record_dir, a.out_dir):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

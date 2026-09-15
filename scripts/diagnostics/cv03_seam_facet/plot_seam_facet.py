"""#831 figure: the assembled absorber profile, the permittivity the update
actually reads, and what removing the guide's vacuum end facet does.

Reads the artifacts this lane's driver wrote and emits one four-panel PNG.
No FDTD, no rfx import for the measured panels -- panel A re-derives the two
permittivity arrays from rfx so the "which array does the update read" claim is
shown rather than asserted.

Run:
    PYTHONPATH=<repo> python3 \
        scripts/diagnostics/cv03_seam_facet/plot_seam_facet.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
ART = HERE.parent / "_artifacts" / "cv03_seam_facet"
OUT = ART / "seam_facet.png"

prof = json.load(open(ART / "profile.json"))["stages"]["profile"]
sweep = json.load(open(ART / "sweep.json"))["stages"]["sweep"]
bprime = json.load(open(ART / "bprime.json"))["stages"]["bprime"]
cv01 = json.load(open(ART / "cv01.json"))["stages"]["cv01"]

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# --- A: the two permittivity arrays, centre row -----------------------------
ax = axes[0, 0]
try:
    from rfx import Simulation, Box
    from rfx.boundaries.spec import BoundarySpec
    from rfx.geometry.smoothing import compute_smoothed_eps
    C0, a = 2.998e8, 1.0e-6
    dx = a / 10
    for n, col in ((20, "tab:blue"), (60, "tab:red")):
        sim = Simulation(freq_max=0.25 * C0 / a, domain=(16 * a, 9 * a, dx),
                         dx=dx, boundary=BoundarySpec.uniform("upml"),
                         cpml_layers=n, mode="2d_tmz")
        sim.add_material("wg", eps_r=12.0)
        sim.add(Box((0, 4.0 * a, 0), (16 * a, 5.0 * a, dx)), material="wg")
        g = sim._build_grid()
        m, *_ = sim._assemble_materials(g)
        pairs = [(e.shape, sim._resolve_material(e.material_name).eps_r)
                 for e in sim._geometry]
        _, _, az = compute_smoothed_eps(g, pairs, background_eps=1.0)
        jy = int(g.pad_y_lo) + 45
        x = (np.arange(g.shape[0]) - g.pad_x_lo) * dx / a
        # NOTE the leading space: matplotlib silently drops any legend label
        # that starts with "_", which is exactly how this method is spelled.
        ax.plot(x, np.asarray(m.eps_r)[:, jy, 0], col, lw=2.2,
                label=f" _assemble_materials (built), {n} layers")
        ax.plot(x, np.asarray(az)[:, jy, 0], col, lw=1.4, ls="--",
                label=f"compute_smoothed_eps (SOLVED), {n} layers")
except Exception as exc:  # pragma: no cover - figure-only fallback
    ax.text(0.5, 0.5, f"rfx unavailable: {exc}", ha="center", va="center",
            transform=ax.transAxes)
_ic = cv01["instrument_check"]["control"]
ax.plot([], [], " ",
        label=(f"cv01, same build: solved {_ic['solved_n_at_eps_wg']}"
               f"/{_ic['nx']} vs built {_ic['built_n_at_eps_wg']}"
               f"/{_ic['nx']}"))
ax.axvspan(-6, 0, color="0.9", zorder=0)
ax.axvspan(16, 22, color="0.9", zorder=0, label="absorber pad")
ax.set_xlabel("x (a), 0 = interior lo edge")
ax.set_ylabel(r"$\varepsilon_r$ on the guide centre row")
ax.set_title("A. The pad extension never reaches the solved array\n"
             "(subpixel_smoothing=True rebuilds eps from the geometry)")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)

# --- B: the assembled absorber profile --------------------------------------
ax = axes[0, 1]
for n, col in ((20, "tab:blue"), (40, "tab:green"), (60, "tab:red")):
    u = prof["arms"][f"upml_{n}"]
    sE = np.asarray(u["sigma_E_centre_row"], dtype=float)
    phx = u["pad_x_hi"]
    d = np.arange(phx)
    ax.semilogy(d, np.maximum(sE[-phx:], 1e-6), col, lw=2,
                label=f"UPML $\\sigma_E$, {n} layers "
                      f"($\\int\\sigma dx$={u['int_sigma_dx']:.5f})")
    c = prof["arms"][f"cpml_{n}"]
    sc = np.asarray(c["sigma"], dtype=float)[::-1]
    ax.semilogy(np.arange(len(sc)), np.maximum(sc, 1e-6), col, lw=1.2, ls=":",
                label=f"CPML $\\sigma$, {n} layers "
                      f"($\\int\\sigma dx$={c['int_sigma_dx']:.5f})")
ax.set_xlabel("cells into the hi-face pad (0 = the seam)")
ax.set_ylabel(r"$\sigma$ (S/m)")
ax.set_title("B. Assembled absorber profile\n"
             r"$\sigma_{max}\propto 1/N$ keeps $\int\sigma dx$ fixed, so the "
             "near-seam $\\sigma$ collapses with depth")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

# --- C: |Ez(x)| with and without the facet ----------------------------------
ax = axes[1, 0]
for n, col in ((20, "tab:blue"), (60, "tab:red")):
    for key, ls, lab in ((f"sweep_upml_{n}", "-", "facet (committed)"),
                         (f"guidecont_upml_{n}", "--", "guide continued")):
        v = (sweep if key.startswith("sweep") else bprime)[key]
        ez = np.asarray(v["ez_abs_carrier_full_x"], dtype=float)
        xx = np.asarray(v["x_full_m"], dtype=float) / 1.0e-6
        ax.semilogy(xx, np.maximum(ez, 1e-32), col, ls=ls, lw=1.7,
                    label=f"{n} layers, {lab}")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(16, color="k", lw=0.8)
ax.set_xlabel("x (a), 0 and 16 = the interior/pad seams")
ax.set_ylabel(r"$|E_z|$ at the carrier bin")
ax.set_ylim(1e-30, 3e-19)
ax.set_title("C. The facet arms decay only after the seam and ripple inside;\n"
             "continued through the pad the guide decays into the absorber")
ax.legend(fontsize=7.5, loc="lower center")
ax.grid(alpha=0.3, which="both")

# Interior detail, linear: the standing-wave ripple the facet makes.
axin = ax.inset_axes([0.30, 0.62, 0.42, 0.33])
for n, col in ((20, "tab:blue"), (60, "tab:red")):
    for key, ls in ((f"sweep_upml_{n}", "-"), (f"guidecont_upml_{n}", "--")):
        v = (sweep if key.startswith("sweep") else bprime)[key]
        ez = np.asarray(v["ez_abs_carrier_full_x"], dtype=float)
        xx = np.asarray(v["x_full_m"], dtype=float) / 1.0e-6
        sel = (xx >= 4) & (xx <= 12)
        axin.plot(xx[sel], ez[sel] / ez[sel].mean(), col, ls=ls, lw=1.2)
axin.set_title("fit window, normalised", fontsize=7)
axin.tick_params(labelsize=6)
axin.grid(alpha=0.3)

# --- D: the headline, both ways ---------------------------------------------
ax = axes[1, 1]
ns = [20, 40, 60]
ba_facet = [sweep[f"sweep_upml_{n}"]["b_over_a_carrier_bin"] for n in ns]
ba_cpml = [sweep[f"sweep_cpml_{n}"]["b_over_a_carrier_bin"] for n in ns]
ba_cont = [bprime[f"guidecont_upml_{n}"]["b_over_a_carrier_bin"] for n in ns]
ba_nosub = [bprime[f"nosub_upml_{n}"]["b_over_a_carrier_bin"] for n in ns]


def a10(v):
    ez = np.asarray(v["ez_abs_carrier_full_x"], dtype=float)
    seam = len(ez) - v["pad_x_hi"] - 1
    return float(ez[seam + 10] / ez[seam])


att = [a10(sweep[f"sweep_upml_{n}"]) for n in ns]
ax.plot(ns, ba_facet, "o-", color="tab:red", lw=2,
        label="|B/A|, committed cv03 (UPML) — the facet")
ax.plot(ns, ba_cpml, "s--", color="tab:orange", lw=1.5,
        label="|B/A|, committed cv03 (CPML)")
ax.plot(ns, ba_cont, "o-", color="tab:blue", lw=2,
        label="|B/A|, guide continued (B1)")
ax.plot(ns, ba_nosub, "^--", color="tab:cyan", lw=1.5,
        label="|B/A|, subpixel off (B2)")
ax.plot(ns, att, "d:", color="0.4", lw=1.5,
        label=r"$|E_z|$ 10 cells into the pad / at the seam")
_ba01 = cv01["arms"]["control"]["b_over_a_carrier_bin"]
ax.plot([20], [_ba01], "*", color="k", ms=16, zorder=5,
        label=f"cv01 committed rig (Arm E): |B/A| = {_ba01:.4f}")
ax.set_yscale("log")
ax.set_xticks(ns)
ax.set_xlabel("cpml_layers")
ax.set_ylabel("amplitude ratio")
ax.set_title("D. The trend is the facet being loaded by near-seam loss.\n"
             "Remove the facet and depth helps again (0.030 -> 0.002)")
ax.set_ylim(1e-3, 1.2)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

fig.suptitle("Issue #831 — the far-end return is a vacuum end facet left at "
             "the interior/pad seam when subpixel smoothing is on\n"
             "(cv03 measured at three depths under two absorber families; "
             "cv01's committed rig carries the same facet)", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")

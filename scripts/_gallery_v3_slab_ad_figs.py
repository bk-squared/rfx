"""Slab autodiff demo: dT/dd through the FDTD solver, exact-derivative check.

The cleanest autodiff case in the gallery. The design variable is the slab
thickness d; the scalar is the transmission T(d) = |X_trans/X_inc|^2 at a single
frequency. d enters the permittivity as a soft top-hat (a differentiable
thickness), so jax.value_and_grad flows straight through Simulation.forward and
returns dT/dd.

The slab transfer matrix gives T(d) in closed form, so dT/dd has an EXACT
analytic expression:

    T(d)     = 1 / (1 + F sin^2 d),   F = ((n - 1/n)^2)/4,  delta = 2*pi*f*n*d/c
    dT/dd    = -T^2 * F * sin(2 delta) * (2*pi*f*n/c)

We report BOTH cross-checks, which answer two different questions:
  * AD vs central FD on the SAME discretised forward  -> is the AD machinery
    correct? (machine-exact, sub-0.1%)
  * AD vs the continuum analytic derivative           -> has the FDTD physics
    converged? (single-digit %, tightens as dx halves; the residual is Yee
    dispersion plus the one-cell soft edge)

The working point d0 is chosen on a steep mid-slope of T(d) (delta ~ 2 rad), not
at a transmission peak where dT/dd -> 0 and the relative comparison is ill-posed.

Produces docs/public/gallery/assets/multilayer_fresnel/autodiff.png.

Run: python scripts/_gallery_v3_slab_ad_figs.py
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "docs", "public", "gallery", "assets", "multilayer_fresnel")

# --- shared layout standard (gallery v3) -----------------------------------
mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.titlepad": 10,
    "figure.constrained_layout.use": True,
})

C0 = 2.99792458e8
EPS_SLAB = 4.0
N_SLAB = math.sqrt(EPS_SLAB)
F0 = 8.0e9                      # single-bin probe frequency
DX = 0.25e-3                    # 0.25 mm cells -> ~15 cells/wavelength in slab
NX = int(round(0.6 / DX))       # 0.6 m domain (single-pass record before bounce)
N_STEPS = 8000
SMOOTH_CELLS = 1.0              # soft top-hat edge width (~1 cell)
SRC_OFFSET = 0.040             # source / probe set-back from the absorbers

# Working point on a steep mid-slope of T(d): delta ~ 2.0 rad (T ~ 0.68), away
# from the transmission peak where dT/dd -> 0.
D0 = 2.0 * C0 / (2.0 * np.pi * F0 * N_SLAB)


def _build_sim():
    """A thin (1-cell transverse) periodic strip = a 1-D normal-incidence slab."""
    domain = (NX * DX, DX, DX)
    sim = Simulation(
        freq_max=16e9, domain=domain, dx=DX,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="periodic", hi="periodic"),
            z=Boundary(lo="periodic", hi="periodic"),
        ),
        cpml_layers=24,
    )
    sim.add_source((SRC_OFFSET, DX / 2, DX / 2), "ez",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.7, amplitude=1.0))
    sim.add_probe((NX * DX - SRC_OFFSET, DX / 2, DX / 2), "ez")
    return sim


def _make_pieces(sim):
    grid = sim._build_grid()
    nx = grid.shape[0]
    xs = jnp.arange(nx, dtype=jnp.float64) * DX
    x_c = float(xs[nx // 2])
    dt = float(grid.dt)
    n = jnp.arange(N_STEPS)
    win = 0.5 - 0.5 * jnp.cos(2 * jnp.pi * n / (N_STEPS - 1))     # Hann window
    phasor = jnp.exp(-1j * 2 * jnp.pi * F0 * n * dt) * win
    ones = jnp.ones(grid.shape, dtype=jnp.float64)

    def eps_from_d(d):
        """Soft top-hat permittivity for a slab of thickness d (differentiable)."""
        lo, hi = x_c - d / 2, x_c + d / 2
        w = SMOOTH_CELLS * DX
        window = jax.nn.sigmoid((xs - lo) / w) * jax.nn.sigmoid((hi - xs) / w)
        return ((1.0 + (EPS_SLAB - 1.0) * window)[:, None, None] * ones)

    return grid, ones, phasor, eps_from_d


def _analytic_T(d):
    n = N_SLAB
    F = ((n - 1.0 / n) ** 2) / 4.0
    delta = 2.0 * np.pi * F0 * n * d / C0
    return 1.0 / (1.0 + F * np.sin(delta) ** 2)


def _analytic_dT(d):
    n = N_SLAB
    F = ((n - 1.0 / n) ** 2) / 4.0
    delta = 2.0 * np.pi * F0 * n * d / C0
    T = _analytic_T(d)
    return -(T ** 2) * F * np.sin(2.0 * delta) * (2.0 * np.pi * F0 * n / C0)


def compute():
    sim = _build_sim()
    grid, ones, phasor, eps_from_d = _make_pieces(sim)

    # vacuum reference forward -> incident phasor at the probe
    ts_vac = sim.forward(eps_override=ones, n_steps=N_STEPS,
                         skip_preflight=True).time_series
    x_inc = jnp.sum(ts_vac[:, 0] * phasor)

    def T_of_d(d):
        ts = sim.forward(eps_override=eps_from_d(d), n_steps=N_STEPS,
                         skip_preflight=True).time_series
        x_t = jnp.sum(ts[:, 0] * phasor)
        return (jnp.abs(x_t) / jnp.abs(x_inc)) ** 2

    print(f"grid {grid.shape}  N_STEPS {N_STEPS}  d0 = {D0*1e3:.3f} mm")

    t0 = time.time()
    T_val, dT_ad = jax.value_and_grad(T_of_d)(jnp.float64(D0))
    T_val, dT_ad = float(T_val), float(dT_ad)
    print(f"  AD: T(d0) = {T_val:.4f}  dT/dd_AD = {dT_ad:.4e}  ({time.time()-t0:.1f}s)")

    # central FD on the same discretised forward
    h = 0.05e-3
    fd = (float(T_of_d(jnp.float64(D0 + h))) - float(T_of_d(jnp.float64(D0 - h)))) / (2 * h)
    rel_fd = abs(dT_ad - fd) / max(abs(fd), 1e-30)
    # continuum analytic derivative
    dT_an = _analytic_dT(D0)
    rel_an = abs(dT_ad - dT_an) / max(abs(dT_an), 1e-30)
    print(f"  FD: dT/dd_FD = {fd:.4e}  rel(AD vs FD) = {rel_fd:.3e}")
    print(f"  analytic: dT/dd = {dT_an:.4e}  rel(AD vs analytic) = {rel_an:.3e}")

    # AD markers across a Fabry-Perot half-period for the sensitivity overlay
    d_grid = np.linspace(D0 * 0.45, D0 * 1.85, 9)
    dT_markers = []
    for dv in d_grid:
        _, gv = jax.value_and_grad(T_of_d)(jnp.float64(dv))
        dT_markers.append(float(gv))
    dT_markers = np.array(dT_markers)

    return dict(
        T_val=T_val, dT_ad=dT_ad, dT_fd=fd, dT_an=dT_an,
        rel_fd=rel_fd, rel_an=rel_an,
        d_grid=d_grid, dT_markers=dT_markers,
    )


def make_figure(data):
    # dense analytic curves over a full Fabry-Perot half-period
    d_lo, d_hi = D0 * 0.40, D0 * 1.95
    dd = np.linspace(d_lo, d_hi, 400)
    T_curve = np.array([_analytic_T(x) for x in dd])
    dT_curve = np.array([_analytic_dT(x) for x in dd])
    mm = 1e3

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")

    # --- (a) sensitivity overlay: AD markers on the exact dT/dd curve --------
    axL.plot(dd * mm, dT_curve, "-", color="#1f5fa8", lw=2.0,
             label="exact  dT/dd  (transfer matrix)")
    axL.plot(data["d_grid"] * mm, data["dT_markers"], "o", color="#b00000",
             ms=6.5, mfc="white", mew=1.5, label="jax.grad  (AD, FDTD)")
    axL.axhline(0.0, color="0.6", ls=":", lw=0.9)
    axL.set_xlabel("slab thickness  d  (mm)")
    axL.set_ylabel("dT/dd   (1/m)")
    axL.set_title("Transmission sensitivity dT/dd — AD vs exact derivative")
    axL.grid(True, alpha=0.3)
    axL.legend(loc="upper right", framealpha=0.9)
    txt = (f"at d0 = {D0*mm:.2f} mm\n"
           f"AD     = {data['dT_ad']:+.2f}\n"
           f"exact  = {data['dT_an']:+.2f}\n"
           f"rel(AD vs FD)       = {data['rel_fd']*100:.2f} %\n"
           f"rel(AD vs analytic) = {data['rel_an']*100:.1f} %")
    axL.text(0.03, 0.04, txt, transform=axL.transAxes, fontsize=8.5,
             va="bottom", ha="left", family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))

    # --- (b) T(d) and dT/dd on twin axes: gradient is zero at the T peak -----
    axR.plot(dd * mm, T_curve, "-", color="#1f7a3a", lw=2.0, label="T(d)")
    axR.set_xlabel("slab thickness  d  (mm)")
    axR.set_ylabel("transmission  T", color="#1f7a3a")
    axR.tick_params(axis="y", labelcolor="#1f7a3a")
    axR.set_ylim(0.0, 1.05)
    axR.grid(True, alpha=0.3)
    axR.set_title("T(d) and its gradient through one half-period")

    axR2 = axR.twinx()
    axR2.plot(dd * mm, dT_curve, "--", color="#b00000", lw=1.6, label="dT/dd")
    axR2.axhline(0.0, color="#b00000", ls=":", lw=0.8, alpha=0.6)
    axR2.set_ylabel("dT/dd   (1/m)", color="#b00000")
    axR2.tick_params(axis="y", labelcolor="#b00000")
    # mark the half-wave transmission peak where dT/dd crosses zero
    i_pk = int(np.argmax(T_curve))
    axR.plot(dd[i_pk] * mm, T_curve[i_pk], "o", color="#1f7a3a", ms=6)
    axR.annotate("half-wave peak: T = 1, dT/dd = 0",
                 xy=(dd[i_pk] * mm, T_curve[i_pk]),
                 xytext=(dd[i_pk] * mm, 0.55), ha="center", fontsize=8.5,
                 color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    lines = [axR.get_lines()[0], axR2.get_lines()[0]]
    axR.legend(lines, [ln.get_label() for ln in lines],
               loc="lower right", framealpha=0.9)

    out = os.path.join(ASSETS, "autodiff.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    data = compute()
    make_figure(data)

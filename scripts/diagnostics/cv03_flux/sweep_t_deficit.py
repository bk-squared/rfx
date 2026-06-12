"""Issue #160 diagnostic: cv03 straight-waveguide T=0.915 deficit.

One-attempt falsifier matrix (issue #160 acceptance):
  - resolution in {10, 15, 20}  (mesh-convergence axis: does T -> 1?)
  - monitor extent in {full-plane (cv03 as-is), bounded 2a (Meep-matched)}
    (comparator axis: is the deficit radiation captured by the oversized
    flux_in plane rather than a flux-normalization bug?)

Geometry/source/run-length identical to examples/crossval/03 PART 2,
parameterized by resolution. Dumps full T(f) curves (R5) to JSON + PNG.

Run: JAX_ENABLE_X64=1 python scripts/diagnostics/cv03_flux/sweep_t_deficit.py
"""

import os
import json
import math
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# ---- cv03 parameters (Meep tutorial) ----
eps_wg = 12.0
wg_width = 1.0
pad = 4.0
dpml = 2.0
sx = 16.0
sy = 2 * (pad + dpml + wg_width / 2)
a = 1.0e-6
fcen = 0.15
df = 0.1
n_freqs = 50

interior_x = sx
interior_y = sy - 2 * dpml
OFFSET_X = interior_x / 2.0
OFFSET_Y = interior_y / 2.0

bw_rfx = df / (fcen * math.pi * math.sqrt(2))
fcen_hz = fcen * C0 / a

src_x_meep = -7.0
flux_in_meep = -5.0
flux_out_meep = +5.0

freqs_norm = np.linspace(fcen - df / 2, fcen + df / 2, n_freqs)
meep_total_t = 400.0


def run_case(resolution: int, monitor: str) -> dict:
    from rfx import Simulation, Box, flux_spectrum
    from rfx.boundaries.spec import BoundarySpec
    from rfx.sources.sources import ModulatedGaussian
    import jax.numpy as jnp

    dx = a / resolution
    cpml_n = int(dpml * resolution)
    domain_x = interior_x * a
    domain_y = interior_y * a

    src_x_rfx = (src_x_meep + OFFSET_X) * a
    flux_in_rfx = (flux_in_meep + OFFSET_X) * a
    flux_out_rfx = (flux_out_meep + OFFSET_X) * a

    sim = Simulation(freq_max=0.25 * C0 / a,
                     domain=(domain_x, domain_y, dx), dx=dx,
                     boundary=BoundarySpec.uniform("upml"),
                     cpml_layers=cpml_n, mode="2d_tmz")
    sim.add_material("wg", eps_r=eps_wg)

    wg_y_lo = (OFFSET_Y - wg_width / 2) * a
    wg_y_hi = (OFFSET_Y + wg_width / 2) * a
    sim.add(Box((0, wg_y_lo, 0), (domain_x, wg_y_hi, dx)), material="wg")

    n_src = int(wg_width * resolution)
    for i in range(n_src):
        y = wg_y_lo + (i + 0.5) * dx
        sim.add_source(position=(src_x_rfx, y, 0), component="ez",
                       waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                                                  amplitude=1.0 / n_src,
                                                  cutoff=5.0 / math.sqrt(2)))

    freqs_rfx = jnp.asarray(freqs_norm * C0 / a)
    mon_kwargs = {}
    if monitor == "bounded2a":
        # Meep-matched flux region: 2*wg_width tall, centered on the guide.
        mon_kwargs = dict(size=(2 * wg_width * a, 10 * dx),
                          center=(OFFSET_Y * a, dx / 2))
    sim.add_flux_monitor(axis="x", coordinate=flux_in_rfx,
                         freqs=freqs_rfx, name="flux_in", **mon_kwargs)
    sim.add_flux_monitor(axis="x", coordinate=flux_out_rfx,
                         freqs=freqs_rfx, name="flux_out", **mon_kwargs)

    sim.preflight(strict=False)

    rfx_total_t = meep_total_t * a / C0
    dt = dx / (C0 * math.sqrt(2)) * 0.99
    n_steps = int(rfx_total_t / dt) + 200

    t0 = time.time()
    res = sim.run(n_steps=n_steps, subpixel_smoothing=True)
    wall = time.time() - t0

    flux_in = np.asarray(flux_spectrum(res.flux_monitors["flux_in"]))
    flux_out = np.asarray(flux_spectrum(res.flux_monitors["flux_out"]))
    eps_f = float(np.max(np.abs(flux_in))) * 1e-6
    T = flux_out / np.where(np.abs(flux_in) > eps_f, flux_in, eps_f)
    peak_idx = int(np.argmax(np.abs(flux_in)))

    out = {
        "resolution": resolution,
        "monitor": monitor,
        "n_steps": n_steps,
        "wall_s": round(wall, 1),
        "peak_idx": peak_idx,
        "f_peak": float(freqs_norm[peak_idx]),
        "T_peak": float(T[peak_idx]),
        "freqs_norm": freqs_norm.tolist(),
        "T": T.tolist(),
        "flux_in": flux_in.tolist(),
        "flux_out": flux_out.tolist(),
    }
    print(f"[case] res={resolution:3d} monitor={monitor:10s} "
          f"steps={n_steps} wall={wall:.0f}s  T(f_peak)={T[peak_idx]:.4f}",
          flush=True)
    return out


def main():
    cases = []
    for monitor in ("fullplane", "bounded2a"):
        for resolution in (10, 15, 20):
            cases.append(run_case(resolution, monitor))

    out_json = os.path.join(SCRIPT_DIR, "sweep_t_deficit.json")
    with open(out_json, "w") as f:
        json.dump(cases, f, indent=1)
    print(f"saved {out_json}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {10: "tab:blue", 15: "tab:orange", 20: "tab:red"}
    for c in cases:
        ax = axes[0] if c["monitor"] == "fullplane" else axes[1]
        ax.plot(c["freqs_norm"], c["T"], color=colors[c["resolution"]],
                label=f"res={c['resolution']}  T(fp)={c['T_peak']:.3f}")
    for ax, title in zip(axes, ("full-plane monitor (cv03 as-is)",
                                "bounded 2a monitor (Meep-matched)")):
        ax.axhline(1.0, color="k", ls=":", alpha=0.5)
        ax.axhspan(0.95, 1.05, color="g", alpha=0.08)
        ax.set_xlabel("Frequency (c/a)")
        ax.set_ylabel("T(f)")
        ax.set_title(title)
        ax.set_ylim(0.7, 1.3)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Issue #160: cv03 T deficit — mesh x monitor-extent matrix",
                 fontweight="bold")
    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "sweep_t_deficit.png")
    plt.savefig(out_png, dpi=150)
    print(f"saved {out_png}")

    print("\nSummary (T at peak bin):")
    for c in cases:
        print(f"  res={c['resolution']:3d}  {c['monitor']:10s}  "
              f"T={c['T_peak']:.4f}")


if __name__ == "__main__":
    main()

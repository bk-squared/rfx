"""WR-90 PEC-short |S11| spread vs PEC short cell position.

Hypothesis under test: the residual per-frequency |S11| oscillation in
rfx (~0.13 spread at R=1, vs OpenEMS 0.0036) is caused by staircase
phase error from the PEC short being snapped to an integer cell
boundary. If true, sweeping PEC_SHORT_X across whole-cell positions
should show the spread varying systematically (monotonic with phase
or sinusoidal with cell offset modulo λ_g/2).

If spread is invariant w.r.t. PEC_SHORT_X → PEC end-wall placement is
NOT the bottleneck; look elsewhere.

If spread varies systematically → the candidate FDTD-core fix is
axis-aligned PEC subpixel handling at end walls (memory
``project_wr90_architectural_candidates.md`` item #3).

Output: per-position |S11(f)| trace overlay + spread-vs-position
summary table + JSON record. Runs at R=1 (dx=1 mm) for ~5 positions,
~5-8 minutes wall-time on CPU.

Run:
  python scripts/diagnostics/wr90_port/pec_short_position_sweep.py
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_pec_short_sweep"
OUT.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8
MU_0 = 1.2566370614e-6


def _load_cv11():
    spec = importlib.util.spec_from_file_location(
        "cv11", REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _s11_cell_recipe(E_z, H_y, y, z, freqs, a):
    """Cell-centred TE10 V/I projection recipe — same as
    scripts/diagnostics/wr90_port/s11_from_dumps.py.
    """
    Ny, Nz = E_z.shape[1], E_z.shape[2]
    sin_y = np.sin(np.pi * y / a)
    dy = float(y[1] - y[0]) if Ny > 1 else a
    dz = float(z[1] - z[0]) if Nz > 1 else 1.0
    weight = sin_y[:, None] * np.ones((1, Nz)) * dy * dz
    V = np.sum(E_z * weight[None, :, :], axis=(1, 2))
    I = np.sum(H_y * weight[None, :, :], axis=(1, 2))
    omega = 2.0 * np.pi * freqs
    k0 = omega / C0
    kc = np.pi / a
    beta = np.sqrt(np.maximum(k0 ** 2 - kc ** 2, 1e-30))
    Z = omega * MU_0 / beta
    a_fwd = 0.5 * (V + I * Z)
    a_ref = V - a_fwd
    s11 = np.abs(a_ref / np.where(np.abs(a_fwd) > 1e-30, a_fwd, 1e-30))
    return s11


def run_rfx_at_pec_short_x(pec_short_x: float, R: int = 1) -> dict:
    """Run rfx PEC-short with PEC_SHORT_X overridden, return |S11(f)|."""
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    cv = _load_cv11()
    dx_m = 1e-3 / R
    plane_x = cv.MON_LEFT_X
    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2,
                             float(freqs[-1] - freqs[0]) / max(f0, 1.0)))

    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS, dx=dx_m,
    )
    sim.add(
        Box((pec_short_x, 0.0, 0.0),
            (pec_short_x + 2 * dx_m, cv.DOMAIN_Y, cv.DOMAIN_Z)),
        material="pec",
    )
    pf = jnp.asarray(freqs)
    sim.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050, name="left",
        mode_profile="discrete",
    )
    for comp in ("ez", "hy"):
        sim.add_dft_plane_probe(axis="x", coordinate=plane_x,
                                component=comp, freqs=pf, name=f"{comp}_p")
    res = sim.run(num_periods=200, compute_s_params=False)

    Ez = np.asarray(res.dft_planes["ez_p"].accumulator)
    Hy = np.asarray(res.dft_planes["hy_p"].accumulator)
    Ny, Nz = Ez.shape[1], Ez.shape[2]
    y = np.linspace(0.0, cv.A_WG, Ny)
    z = np.linspace(0.0, cv.B_WG, Nz)
    s11 = _s11_cell_recipe(Ez, Hy, y, z, np.asarray(freqs), cv.A_WG)
    return {
        "pec_short_x_m": pec_short_x,
        "freqs_hz": np.asarray(freqs),
        "s11": s11,
        "spread": float(s11.max() - s11.min()),
        "min": float(s11.min()),
        "mean": float(s11.mean()),
        "max": float(s11.max()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--R", type=int, default=1,
                   help="resolution multiplier (R=1 → dx=1 mm)")
    p.add_argument("--positions-mm", type=str,
                   default="142,143,144,145,146,147,148",
                   help="comma-separated PEC_SHORT_X values in mm")
    args = p.parse_args()

    positions_m = [float(s) * 1e-3 for s in args.positions_mm.split(",")]

    print(f"=== PEC-short position sweep (R={args.R}, dx={1.0/args.R:.3f} mm) ===")
    print(f"positions: {[f'{p*1000:.1f}mm' for p in positions_m]}")
    print()

    results = []
    for x in positions_m:
        t0 = time.time()
        r = run_rfx_at_pec_short_x(x, R=args.R)
        dt = time.time() - t0
        results.append(r)
        print(f"PEC_SHORT_X = {x*1000:6.1f} mm  |  "
              f"|S11| min/mean/max = {r['min']:.4f}/{r['mean']:.4f}/{r['max']:.4f}  |  "
              f"spread = {r['spread']:.4f}  |  {dt:.1f}s")

    # Summary table
    print("\n=== Summary ===")
    spreads = [r["spread"] for r in results]
    print(f"spread min={min(spreads):.4f}  max={max(spreads):.4f}  "
          f"max/min ratio = {max(spreads)/max(min(spreads),1e-9):.2f}")
    if max(spreads) - min(spreads) < 0.005:
        verdict = "INVARIANT — PEC end-wall position is NOT the cause"
    elif (max(spreads) - min(spreads)) / max(spreads) > 0.30:
        verdict = "VARYING — PEC end-wall staircase is a strong candidate"
    else:
        verdict = "PARTIAL — PEC end-wall contributes but other effects dominate"
    print(f"verdict: {verdict}")

    # Plot per-position |S11(f)| overlay
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    f_ghz = results[0]["freqs_hz"] / 1e9
    cmap = plt.get_cmap("viridis")
    ax = axes[0]
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal PEC-short")
    for i, r in enumerate(results):
        col = cmap(i / max(len(results) - 1, 1))
        ax.plot(f_ghz, r["s11"], "o-", color=col, lw=1.4, ms=4,
                label=f"x={r['pec_short_x_m']*1000:.1f}mm  spread={r['spread']:.4f}")
    ax.set_ylabel("|S11(f)|")
    ax.set_title(f"WR-90 PEC-short |S11| vs PEC end-wall position (R={args.R})")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=1, ls="--")
    for i, r in enumerate(results):
        col = cmap(i / max(len(results) - 1, 1))
        ax.plot(f_ghz, r["s11"] - 1.0, "o-", color=col, lw=1.4, ms=4,
                label=f"x={r['pec_short_x_m']*1000:.1f}mm")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("|S11| − 1")
    ax.set_title("Departure from ideal — per-freq oscillation by position")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / f"pec_short_position_sweep_R{args.R}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

    # JSON record
    out_json = OUT / f"pec_short_position_sweep_R{args.R}.json"
    out_json.write_text(json.dumps([
        {
            "pec_short_x_m": r["pec_short_x_m"],
            "freqs_hz": r["freqs_hz"].tolist(),
            "s11": r["s11"].tolist(),
            "spread": r["spread"],
            "min": r["min"], "mean": r["mean"], "max": r["max"],
        } for r in results
    ], indent=2))
    print(f"[json] {out_json}")
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()

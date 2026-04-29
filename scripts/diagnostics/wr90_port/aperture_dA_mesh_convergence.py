"""WR-90 ``_aperture_dA`` integral vs dx — mesh-convergence baseline.

Item C scoping diagnostic for the
``test_mesh_convergence_s21_scaled_cpml`` xfail. The xfail comment
identifies the residual as the DROP-weight integral changing 7%/5%/4%
across dx={3, 2, 1.5} mm — an integral normalisation that doesn't
converge to a well-defined dx → 0 limit. A fractional boundary-cell
weight (OpenEMS user-pinned integration box) would fix it.

This script does not implement the fix. It MEASURES the integral
``sum(aperture_dA)`` and ``sum(aperture_dA · ez_profile²)`` (the latter
is the modal-template normalisation that propagates into V/I scaling)
across a sweep of dx values, fits a 1st-order convergence model, and
reports both the absolute integral and its rate of change vs dx.

Output gives the next session a concrete baseline to compare a
fractional-weight implementation against.

This is a non-FDTD measurement — just port-init bookkeeping — so it
runs in a few seconds.
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_aperture_dA_conv"
OUT.mkdir(parents=True, exist_ok=True)


def _load_cv11():
    spec = importlib.util.spec_from_file_location(
        "cv11", REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure_at(dx_m: float):
    """Return aperture_dA and ez_profile integrals at the given dx."""
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    cv = _load_cv11()
    freqs = cv.FREQS_HZ[:1]   # only one frequency needed for init
    f0 = float(freqs.mean())
    bandwidth = 0.5
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
    pf = jnp.asarray(freqs)
    sim.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050, name="left",
        mode_profile="discrete",
    )
    # Trigger a single FDTD step's worth of config materialisation so
    # _aperture_dA and ez_profile are populated. compute_s_params=False
    # plus num_periods=1 keeps it cheap (~few seconds per dx).
    res = sim.run(num_periods=1, compute_s_params=False)
    cfg = res.waveguide_ports["left"]
    dA = np.asarray(cfg.aperture_dA)
    ez = np.asarray(cfg.ez_profile)
    Ny, Nz = dA.shape
    return {
        "dx_m": dx_m,
        "Ny": Ny, "Nz": Nz,
        "sum_dA": float(np.sum(dA)),
        "norm_ez_sq_dA": float(np.sum(dA * ez ** 2)),
        "n_zero_cells": int(np.sum(dA == 0.0)),
        "physical_aperture": cv.A_WG * cv.B_WG,
    }


def main():
    dxs = [3.0e-3, 2.0e-3, 1.5e-3, 1.0e-3, 0.75e-3, 0.5e-3]
    cv = _load_cv11()
    physical = cv.A_WG * cv.B_WG
    print(f"physical aperture A·B = {cv.A_WG*1e3:.3f} × {cv.B_WG*1e3:.3f} "
          f"= {physical*1e6:.4f} mm²\n")

    rows = []
    print(f"{'dx[mm]':>7s} | {'Ny':>3s} | {'Nz':>3s} | "
          f"{'∑dA[mm²]':>10s} | {'∑dA·ez²':>10s} | {'#0 cells':>9s} | "
          f"{'∑dA / phys − 1':>15s}")
    print("-" * 80)
    for dx in dxs:
        try:
            r = measure_at(dx)
            rows.append(r)
            rel_err = (r["sum_dA"] - physical) / physical
            print(f"{dx*1e3:7.3f} | {r['Ny']:3d} | {r['Nz']:3d} | "
                  f"{r['sum_dA']*1e6:10.4f} | {r['norm_ez_sq_dA']*1e6:10.4f} | "
                  f"{r['n_zero_cells']:9d} | {rel_err:+15.4%}")
        except Exception as e:
            print(f"{dx*1e3:7.3f} | ERROR: {e}")

    if len(rows) < 2:
        return

    # Fit 1st-order convergence: |sum_dA(dx) − sum_dA(0)| vs dx.
    # Use the finest dx as proxy for the dx → 0 limit.
    fine = rows[-1]
    dx_arr = np.array([r["dx_m"] for r in rows[:-1]])
    err_arr = np.array([
        abs(r["sum_dA"] - fine["sum_dA"]) / fine["sum_dA"] for r in rows[:-1]
    ])
    # Power law: err ~ C · dx^p
    if np.all(err_arr > 0):
        logdx = np.log(dx_arr)
        logerr = np.log(err_arr)
        p, logc = np.polyfit(logdx, logerr, 1)
        print(f"\nfit: |∑dA(dx) − ∑dA(fine)| / ∑dA(fine) ≈ "
              f"{np.exp(logc):.3e} · dx^{p:.2f}")
        if p < 0.5:
            print("verdict: integral is NOT converging (rate < 0.5) — "
                  "DROP-weight binary boundary handling is the bottleneck. "
                  "A fractional boundary weight (OpenEMS user-pinned box) "
                  "should restore 1st-order convergence at minimum.")
        elif p < 1.5:
            print("verdict: ~1st-order convergence — DROP-weight is "
                  "marginally acceptable but not optimal.")
        else:
            print("verdict: 2nd-order or better — boundary handling is fine.")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(physical * 1e6, color="k", lw=1, ls="--",
               label=f"physical A·B = {physical*1e6:.4f} mm²")
    sums = [r["sum_dA"] * 1e6 for r in rows]
    dxs_mm = [r["dx_m"] * 1e3 for r in rows]
    ax.plot(dxs_mm, sums, "o-", color="tab:orange",
            label="∑ aperture_dA (DROP-weight)")
    norms = [r["norm_ez_sq_dA"] * 1e6 for r in rows]
    ax.plot(dxs_mm, norms, "s-", color="tab:blue",
            label="∑ aperture_dA · ez_profile² (mode norm)")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("integral [mm²]")
    ax.set_xscale("log")
    ax.set_title("WR-90 aperture integral mesh-convergence "
                 "(item C scoping diagnostic, 2026-04-29)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_png = OUT / "aperture_dA_mesh_convergence.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {out_png}")

    out_json = OUT / "aperture_dA_mesh_convergence.json"
    out_json.write_text(json.dumps({
        "physical_aperture_m2": physical,
        "rows": rows,
    }, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()

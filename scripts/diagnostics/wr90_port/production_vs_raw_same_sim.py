"""WR-90 PEC-short |S11(f)| — production extractor vs dump recipe on same sim.

Two prior diagnostics on this thread came back null:
- production DROP +y/+z weight: zero contribution on PEC fields, raw
  trim has no effect (boundary_cell_trim.py).
- H normal-averaging across two x-planes: ratio 0.96x, makes the spread
  marginally worse (h_normal_average_test.py).

Yet codex verified production ``compute_waveguide_s_matrix`` on the same
geometry returns spread ≈ 0.0004 (300× tighter than the dump recipe's
0.13). Something in the production pipeline is doing real work, and
neither DROP nor normal-averaging is it on this setup.

Test: run rfx PEC-short ONCE, then extract |S11(f)| both ways:
  - production: ``sim.compute_waveguide_s_matrix(normalize=False)``.
  - dump recipe: cell-centred TE10 V/I projection on the same DFT
    planes (the s11_from_dumps recipe).

If production's spread really is ≈ 0.0004 on this geometry, the
dump-vs-production gap is structural, not a bookkeeping artifact, and
the next diagnostic must localize WHICH production step is doing the
work (mode template, Z choice, post-scan time-series construction, ...).
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
OUT = Path(__file__).parent / "out_production_vs_raw"
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


def _s11_dump_recipe(E_z, H_y, y, z, freqs, a):
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


def main():
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    cv = _load_cv11()
    dx_m = 1e-3
    plane_x = cv.MON_LEFT_X
    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2,
                             float(freqs[-1] - freqs[0]) / max(f0, 1.0)))

    # ----- Run 1: production extractor via crossval/11's two-port pipeline -----
    print("=== Run 1 (production): crossval/11 run_rfx_pec_short() ===")
    freqs_prod, s11_prod_complex, _ = cv.run_rfx_pec_short()
    s11_prod = np.abs(np.asarray(s11_prod_complex))

    # ----- Run 2: single-port + DFT plane probes for raw dump recipe -----
    print("\n=== Run 2 (dump fields): single-port + DFT plane probes ===")
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
        Box((cv.PEC_SHORT_X, 0.0, 0.0),
            (cv.PEC_SHORT_X + 2 * dx_m, cv.DOMAIN_Y, cv.DOMAIN_Z)),
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

    # Capture dt from the sim's grid for the Yee half-step correction.
    if hasattr(sim, "_grid") and hasattr(sim._grid, "dt"):
        dt_sim = float(sim._grid.dt)
    else:
        dt_sim = 0.5 * dx_m / C0  # default Courant 0.5

    # ----- Variant: dump recipe with Yee half-step time-stagger correction -----
    # Production's _co_located_current_spectrum applies exp(+j*omega*dt/2) to I
    # before wave decomp to undo the half-cell H sample timing. The dump-DFT
    # plane probe inherits the same offset from the probe-step pairing.
    omega = 2.0 * np.pi * np.asarray(freqs)
    Hy_corr_factor = np.exp(+1j * omega * (0.5 * dt_sim))
    print(f"\nYee half-step correction: dt={dt_sim*1e12:.3f} ps, "
          f"max ω·dt/2 = {(omega.max() * 0.5 * dt_sim):.4f} rad "
          f"(= {np.degrees(omega.max() * 0.5 * dt_sim):.2f}°)")

    # Dump recipe on the single-port sim.
    Ez = np.asarray(res.dft_planes["ez_p"].accumulator)
    Hy = np.asarray(res.dft_planes["hy_p"].accumulator)
    Ny, Nz = Ez.shape[1], Ez.shape[2]
    y = np.linspace(0.0, cv.A_WG, Ny)
    z = np.linspace(0.0, cv.B_WG, Nz)
    s11_dump = _s11_dump_recipe(Ez, Hy, y, z, np.asarray(freqs), cv.A_WG)

    # Apply Yee half-step correction to H_y spectrum, then re-run recipe.
    Hy_corrected = Hy * Hy_corr_factor[:, None, None]
    s11_dump_corr = _s11_dump_recipe(
        Ez, Hy_corrected, y, z, np.asarray(freqs), cv.A_WG
    )

    rows = [
        ("production compute_waveguide_s_matrix(normalize=False)", s11_prod),
        ("dump recipe (no half-step correction)", s11_dump),
        ("dump recipe + exp(+jω·dt/2) on H_y (production-style)",
         s11_dump_corr),
    ]

    print("\n%-58s | %s | %s | %s | %s"
          % ("variant", "min   ", "mean  ", "max   ", "spread"))
    print("-" * 100)
    for label, s11 in rows:
        print(f"{label:<58s} | {s11.min():.4f} | {s11.mean():.4f} | "
              f"{s11.max():.4f} | {s11.max()-s11.min():.4f}")

    print("\nper-frequency table (production vs dump):")
    f_ghz = np.asarray(freqs) / 1e9
    print("  freq[GHz]  prod        dump       diff")
    for fi, fr in enumerate(f_ghz):
        print(f"  {fr:6.2f}    {s11_prod[fi]:.4f}     "
              f"{s11_dump[fi]:.4f}    {s11_dump[fi]-s11_prod[fi]:+.4f}")

    spread_prod = float(s11_prod.max() - s11_prod.min())
    spread_dump = float(s11_dump.max() - s11_dump.min())
    print(f"\nproduction spread {spread_prod:.4f} | dump spread {spread_dump:.4f} "
          f"| ratio {spread_dump/max(spread_prod, 1e-9):.1f}x")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal PEC-short")
    cmap = plt.get_cmap("tab10")
    for i, (label, s11) in enumerate(rows):
        ax.plot(f_ghz, s11, "o-", color=cmap(i), lw=1.5, ms=4,
                label=f"{label}  (spread {s11.max()-s11.min():.4f})")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("|S11(f)|")
    ax.set_title("WR-90 PEC-short |S11| — production vs dump on the same sim")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / "production_vs_raw_R1.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

    out_json = OUT / "production_vs_raw_R1.json"
    out_json.write_text(json.dumps({
        "freqs_hz": np.asarray(freqs).tolist(),
        "s11_prod": s11_prod.tolist(),
        "s11_dump": s11_dump.tolist(),
        "spread_prod": spread_prod,
        "spread_dump": spread_dump,
    }, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()

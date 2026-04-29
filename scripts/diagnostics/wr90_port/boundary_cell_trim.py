"""WR-90 PEC-short |S11| spread vs V/I integration boundary-cell trimming.

Hypothesis: Codex 2026-04-28 noted that production's DROP +face PEC
weight (zeroing the boundary row in the V/I integral) is what hides
the 0.13 → 0.0004 spread. This script tests the converse on the same
field dump: progressively trim boundary rows from the dump-based V/I
recipe and watch whether the spread drops.

If trimming the +y or +z boundary row alone collapses the spread
toward OpenEMS-class (~0.005) → the residual is concentrated in one
cell-row of integration error at the side-wall, NOT a global field
artifact. The structural fix is then SIDE-wall PEC subpixel handling
(integral domain trimming on the boundary row), not END-wall
subpixel.

If trimming has little effect → the residual is distributed across
the full aperture and side-wall placement isn't the issue.

Single rfx run at PEC_SHORT_X = 145 mm, dx = 1 mm (~10 s).

Run:
  python scripts/diagnostics/wr90_port/boundary_cell_trim.py
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
OUT = Path(__file__).parent / "out_boundary_cell_trim"
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


def _s11(E_z, H_y, y, z, freqs, a, weight_mask):
    """Run the s11_from_dumps recipe with an explicit (Ny, Nz) mask."""
    Ny, Nz = E_z.shape[1], E_z.shape[2]
    sin_y = np.sin(np.pi * y / a)
    dy = float(y[1] - y[0]) if Ny > 1 else a
    dz = float(z[1] - z[0]) if Nz > 1 else 1.0
    base = sin_y[:, None] * np.ones((1, Nz)) * dy * dz
    weight = base * weight_mask
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


def run_rfx_once():
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

    Ez = np.asarray(res.dft_planes["ez_p"].accumulator)
    Hy = np.asarray(res.dft_planes["hy_p"].accumulator)
    Ny, Nz = Ez.shape[1], Ez.shape[2]
    y = np.linspace(0.0, cv.A_WG, Ny)
    z = np.linspace(0.0, cv.B_WG, Nz)
    return {
        "freqs_hz": np.asarray(freqs),
        "y": y, "z": z, "E_z": Ez, "H_y": Hy,
        "a": cv.A_WG, "Ny": Ny, "Nz": Nz,
    }


def main():
    print("=== Single rfx run @ PEC_SHORT_X = 145 mm, dx = 1 mm ===")
    f = run_rfx_once()
    Ny, Nz = f["Ny"], f["Nz"]
    print(f"DFT plane shape: ({Ny}, {Nz}) cells in (y, z)")

    # Trimming variants. Each mask is (Ny, Nz) binary multiplier on dA.
    full = np.ones((Ny, Nz))
    variants = [("full aperture (raw recipe)", full)]

    # Single-edge trims
    m = full.copy(); m[-1, :] = 0
    variants.append(("drop +y row (last y)", m))
    m = full.copy(); m[:, -1] = 0
    variants.append(("drop +z column (last z)", m))
    m = full.copy(); m[0, :] = 0
    variants.append(("drop -y row (first y)", m))
    m = full.copy(); m[:, 0] = 0
    variants.append(("drop -z column (first z)", m))

    # Both +faces (matches production DROP convention)
    m = full.copy(); m[-1, :] = 0; m[:, -1] = 0
    variants.append(("drop +y AND +z (production DROP)", m))

    # All four boundary rows
    m = full.copy(); m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    variants.append(("drop all 4 boundary rows", m))

    print("\n%-40s | %s | %s | %s | %s"
          % ("trim variant", "min   ", "mean  ", "max   ", "spread"))
    print("-" * 90)

    rows = []
    for label, mask in variants:
        s11 = _s11(f["E_z"], f["H_y"], f["y"], f["z"], f["freqs_hz"], f["a"], mask)
        spread = float(s11.max() - s11.min())
        rows.append({
            "label": label,
            "spread": spread,
            "min": float(s11.min()),
            "mean": float(s11.mean()),
            "max": float(s11.max()),
            "s11": s11.tolist(),
        })
        print(f"{label:<40s} | {s11.min():.4f} | {s11.mean():.4f} | "
              f"{s11.max():.4f} | {spread:.4f}")

    # Verdict
    raw = rows[0]["spread"]
    drop_yz = next(r for r in rows
                   if r["label"] == "drop +y AND +z (production DROP)")["spread"]
    print(f"\nraw spread {raw:.4f} → production-DROP spread {drop_yz:.4f}  "
          f"(ratio {raw/max(drop_yz, 1e-9):.1f}x)")
    if drop_yz < 0.020:
        verdict = ("CONFIRMED — boundary cells dominate. "
                   "Spread collapses to OE-class (<0.020) when +y,+z rows "
                   "are dropped from the V/I integral. The structural cause "
                   "is the boundary-cell contribution; the architectural fix "
                   "is SIDE-wall PEC subpixel handling at +y/+z, not END-wall.")
    elif drop_yz / raw < 0.5:
        verdict = ("PARTIAL — boundary cells contribute substantially "
                   "(>50% of spread) but other terms remain at this level.")
    else:
        verdict = ("REFUTED — dropping the boundary rows does not collapse "
                   "the spread; the residual is distributed across the "
                   "interior of the aperture, not concentrated at +y/+z.")
    print(f"\nverdict: {verdict}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal PEC-short")
    f_ghz = f["freqs_hz"] / 1e9
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(rows):
        col = cmap(i % 10)
        ax.plot(f_ghz, r["s11"], "o-", color=col, lw=1.4, ms=3,
                label=f"{r['label']}  (spread {r['spread']:.4f})")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("|S11(f)|")
    ax.set_title("WR-90 PEC-short |S11| vs V/I integration boundary-cell trim")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / "boundary_cell_trim_R1.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

    out_json = OUT / "boundary_cell_trim_R1.json"
    out_json.write_text(json.dumps({
        "freqs_hz": f["freqs_hz"].tolist(),
        "Ny": Ny, "Nz": Nz,
        "rows": rows,
        "verdict": verdict,
    }, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()

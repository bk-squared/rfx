"""WR-90 PEC-short |S11| spread vs H normal-averaging across Yee staggering.

Hypothesis: production ``modal_current`` averages H between the port
plane (``x_idx``) and one cell upstream (``x_idx - 1``) to co-locate H
with E on the staggered Yee grid. The dump-derived ``s11_from_dumps``
recipe uses a single H plane and likely picks up the half-cell phase
mismatch as a per-frequency oscillation — exactly the rfx residual
signature (spread 0.13 vs OE 0.004).

Test: add a SECOND DFT probe at ``mon_left - dx``, average H_y across
the two planes, recompute |S11(f)|. If spread drops toward OE-class
(<0.020), H staggering is the dominant residual and the structural
fix is to standardise the normal-averaging convention in
``s11_from_dumps`` — NOT a multi-week FDTD-core change.

Single rfx run at PEC_SHORT_X = 145 mm, dx = 1 mm (~10 s).
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
OUT = Path(__file__).parent / "out_h_normal_average"
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


def _s11(E_z, H_y, y, z, freqs, a):
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
    plane_x_minus = cv.MON_LEFT_X - dx_m
    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2,
                             float(freqs[-1] - freqs[0]) / max(f0, 1.0)))

    print("=== Single rfx run @ PEC_SHORT_X = 145 mm, dx = 1 mm ===")
    print(f"H_y dumped at TWO planes: x = {plane_x*1000:.2f} mm "
          f"and {plane_x_minus*1000:.2f} mm")

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
    sim.add_dft_plane_probe(axis="x", coordinate=plane_x,
                            component="ez", freqs=pf, name="ez_main")
    sim.add_dft_plane_probe(axis="x", coordinate=plane_x,
                            component="hy", freqs=pf, name="hy_main")
    sim.add_dft_plane_probe(axis="x", coordinate=plane_x_minus,
                            component="hy", freqs=pf, name="hy_prev")
    res = sim.run(num_periods=200, compute_s_params=False)

    Ez = np.asarray(res.dft_planes["ez_main"].accumulator)
    Hy_main = np.asarray(res.dft_planes["hy_main"].accumulator)
    Hy_prev = np.asarray(res.dft_planes["hy_prev"].accumulator)
    Ny, Nz = Ez.shape[1], Ez.shape[2]
    y = np.linspace(0.0, cv.A_WG, Ny)
    z = np.linspace(0.0, cv.B_WG, Nz)

    Hy_avg = 0.5 * (Hy_main + Hy_prev)

    s11_raw = _s11(Ez, Hy_main, y, z, np.asarray(freqs), cv.A_WG)
    s11_avg = _s11(Ez, Hy_avg, y, z, np.asarray(freqs), cv.A_WG)
    s11_prev = _s11(Ez, Hy_prev, y, z, np.asarray(freqs), cv.A_WG)

    rows = [
        ("H_y at x_idx (raw recipe)", s11_raw),
        ("H_y at x_idx - dx (upstream)", s11_prev),
        ("0.5 * (H_y[x_idx] + H_y[x_idx-dx]) (production-style)", s11_avg),
    ]

    print("\n%-55s | %s | %s | %s | %s"
          % ("variant", "min   ", "mean  ", "max   ", "spread"))
    print("-" * 100)
    for label, s11 in rows:
        print(f"{label:<55s} | {s11.min():.4f} | {s11.mean():.4f} | "
              f"{s11.max():.4f} | {s11.max()-s11.min():.4f}")

    spread_raw = float(s11_raw.max() - s11_raw.min())
    spread_avg = float(s11_avg.max() - s11_avg.min())
    print(f"\nraw spread {spread_raw:.4f} → normal-averaged spread {spread_avg:.4f}  "
          f"(ratio {spread_raw/max(spread_avg, 1e-9):.2f}x)")
    if spread_avg < 0.020:
        verdict = ("CONFIRMED — H normal-averaging is the dominant fix. "
                   "Spread drops to OE-class (<0.020) when the production "
                   "convention is replicated on dump fields. Architectural "
                   "fix is post-processing convention, NOT FDTD-core; "
                   "update s11_from_dumps + research notes accordingly.")
    elif spread_avg / spread_raw < 0.5:
        verdict = "PARTIAL — averaging cuts >50% of the spread but not all."
    else:
        verdict = ("REFUTED — H normal-averaging contributes <50% of the "
                   "spread; the residual is elsewhere in the production "
                   "pipeline.")
    print(f"\nverdict: {verdict}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal PEC-short")
    f_ghz = np.asarray(freqs) / 1e9
    cmap = plt.get_cmap("tab10")
    for i, (label, s11) in enumerate(rows):
        ax.plot(f_ghz, s11, "o-", color=cmap(i), lw=1.5, ms=4,
                label=f"{label}  (spread {s11.max()-s11.min():.4f})")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("|S11(f)|")
    ax.set_title("WR-90 PEC-short |S11| vs H normal-averaging convention")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / "h_normal_average_R1.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

    out_json = OUT / "h_normal_average_R1.json"
    out_json.write_text(json.dumps({
        "freqs_hz": np.asarray(freqs).tolist(),
        "s11_raw": s11_raw.tolist(),
        "s11_prev": s11_prev.tolist(),
        "s11_avg": s11_avg.tolist(),
        "spread_raw": spread_raw,
        "spread_avg": spread_avg,
        "verdict": verdict,
    }, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()

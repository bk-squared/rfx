"""Spike: WR-90 PEC-short with dy aligned to physical wall (dy = A_WG / N).

Hypothesis: rfx with cubic dx=1mm gives effective WR-90 broad wall at
+11.5mm (23×1mm) instead of +11.43mm (22.86mm). The 0.14mm overshoot
shifts the simulated TE10 cutoff/β/mode-shape and produces ±6-13% per-freq
oscillation in PEC-short |S11| via standing-wave Q-amplification.

Test: pass `dy_profile = np.full(N_y, A_WG/N_y)` and `dz_profile = np.full(
N_z, B_WG/N_z)` to Simulation. Domain becomes exactly A_WG × B_WG with
cells slightly non-cubic (dy=0.9939mm, dz=1.016mm at dx_x=1mm). Walls
exactly at the physical aperture.

Compare: production cubic (dx=dy=dz=1mm) vs wall-aligned (dy/dz adjusted).
Both on PEC-short, same recipe. If wall-aligned drops spread to OE/Meep
class (<2%), root cause confirmed.

Output: out_compare/wall_aligned_r1.json with per-freq |S11| both ways.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_compare"
OUT.mkdir(parents=True, exist_ok=True)


def _load_cv11():
    spec = importlib.util.spec_from_file_location(
        "cv11", REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def s11_from_field(E_z, H_y, y, z, freqs, a):
    Nf, Ny, Nz = E_z.shape
    y_ref = y - y.min()
    e_func = np.sin(np.pi * y_ref / a)
    dy_v = float(np.abs(y[1] - y[0]) if Ny > 1 else 1.0)
    dz_v = float(np.abs(z[1] - z[0]) if Nz > 1 else 1.0)
    dA = dy_v * dz_v
    weight = e_func[None, :, None]
    V = np.sum(E_z * weight, axis=(1, 2)) * dA
    I = np.sum(H_y * weight, axis=(1, 2)) * dA
    omega = 2 * np.pi * freqs
    C0 = 2.998e8; MU_0 = 1.2566370614e-6
    f_c = C0 / (2 * a)
    k0 = omega / C0
    kc = 2 * np.pi * f_c / C0
    beta = np.sqrt(np.maximum(k0**2 - kc**2, 0.0) + 0j)
    Z = omega * MU_0 / beta
    a_fwd = 0.5 * (V + I * Z)
    a_ref = V - a_fwd
    return np.abs(a_ref / a_fwd)


def run(cv, *, wall_aligned: bool, dx_m: float = 1e-3):
    """Run rfx PEC-short, return (freqs, E_z, H_y, y, z) at mon_left."""
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2,
                             float(freqs[-1] - freqs[0]) / max(f0, 1.0)))

    sim_kwargs = dict(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS,
        dx=dx_m,
    )
    if wall_aligned:
        # Override y/z cell counts to exactly span A_WG / B_WG.
        N_y = int(round(cv.DOMAIN_Y / dx_m))
        N_z = int(round(cv.DOMAIN_Z / dx_m))
        sim_kwargs["dy_profile"] = np.full(N_y, cv.DOMAIN_Y / N_y, dtype=float)
        sim_kwargs["dz_profile"] = np.full(N_z, cv.DOMAIN_Z / N_z, dtype=float)
        print(f"  wall_aligned: N_y={N_y}, dy={cv.DOMAIN_Y/N_y*1e3:.4f}mm; "
              f"N_z={N_z}, dz={cv.DOMAIN_Z/N_z*1e3:.4f}mm")
    else:
        print(f"  cubic: dx=dy=dz={dx_m*1e3:.3f}mm; effective domain"
              f" y={int(round(cv.DOMAIN_Y/dx_m))*dx_m*1e3:.2f}mm, "
              f"z={int(round(cv.DOMAIN_Z/dx_m))*dx_m*1e3:.2f}mm")

    sim = Simulation(**sim_kwargs)
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
    )
    for comp in ("ez", "hy"):
        sim.add_dft_plane_probe(axis="x", coordinate=cv.MON_LEFT_X,
                                component=comp, freqs=pf, name=f"{comp}_p")
    res = sim.run(num_periods=200, compute_s_params=False)
    Ez = np.asarray(res.dft_planes["ez_p"].accumulator)  # (Nf, Ny, Nz)
    Hy = np.asarray(res.dft_planes["hy_p"].accumulator)
    Ny, Nz = Ez.shape[1], Ez.shape[2]
    if wall_aligned:
        y = np.linspace(0.0, cv.DOMAIN_Y, Ny)  # exact
        z = np.linspace(0.0, cv.DOMAIN_Z, Nz)
    else:
        y = np.arange(Ny) * dx_m
        z = np.arange(Nz) * dx_m
    return np.asarray(freqs), Ez, Hy, y, z


def main():
    cv = _load_cv11()
    print("=== cubic (production canonical) ===", flush=True)
    f1, Ez1, Hy1, y1, z1 = run(cv, wall_aligned=False)
    s11_cubic = s11_from_field(Ez1, Hy1, y1, z1, f1, cv.A_WG)

    print("\n=== wall-aligned (dy=A_WG/N) ===", flush=True)
    f2, Ez2, Hy2, y2, z2 = run(cv, wall_aligned=True)
    s11_wa = s11_from_field(Ez2, Hy2, y2, z2, f2, cv.A_WG)

    print("\n=== Summary ===")
    print(f"cubic       |S11| min/mean/max/spread: "
          f"{s11_cubic.min():.4f}/{s11_cubic.mean():.4f}/"
          f"{s11_cubic.max():.4f}/{s11_cubic.max()-s11_cubic.min():.4f}")
    print(f"wall-aligned|S11| min/mean/max/spread: "
          f"{s11_wa.min():.4f}/{s11_wa.mean():.4f}/"
          f"{s11_wa.max():.4f}/{s11_wa.max()-s11_wa.min():.4f}")

    f_ghz = f1 / 1e9
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal")
    ax.plot(f_ghz, s11_cubic, "o-", color="#c52", lw=1.6, ms=4,
            label=f"cubic (dx=dy=dz=1mm) — spread {s11_cubic.max()-s11_cubic.min():.4f}")
    ax.plot(f_ghz, s11_wa, "s-", color="#0a6", lw=1.6, ms=4,
            label=f"wall-aligned (dy=A_WG/N) — spread {s11_wa.max()-s11_wa.min():.4f}")
    ax.set_ylabel("|S11(f)|")
    ax.set_title("rfx WR-90 PEC-short |S11| — cubic vs wall-aligned dy/dz "
                 "(dump-derived recipe at mon_left)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.plot(f_ghz, s11_cubic - 1, "o-", color="#c52", label="cubic")
    ax.plot(f_ghz, s11_wa - 1, "s-", color="#0a6", label="wall-aligned")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("|S11| − 1")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "wall_aligned_r1.png", dpi=150)
    plt.close(fig)
    print(f"\n[plot] {OUT / 'wall_aligned_r1.png'}")


if __name__ == "__main__":
    main()

"""CPML layer count sweep on dump-based |S11|.

Hypothesis: rfx left-side CPML reflection contributes to the 13% per-freq
spread. Memory says CPML_LAYERS=20 has 4% residual reflection (was 12%
at 10). If CPML reflection is the bottleneck, doubling to 40 layers
should drop spread substantially.

Compares CPML_LAYERS={20, 40, 60} on PEC-short dump-derived |S11|.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp
import importlib.util

REPO = Path(__file__).resolve().parents[3]


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
    k0 = omega / C0; kc = 2 * np.pi * f_c / C0
    beta = np.sqrt(np.maximum(k0**2 - kc**2, 0.0) + 0j)
    Z = omega * MU_0 / beta
    a_fwd = 0.5 * (V + I * Z)
    a_ref = V - a_fwd
    return np.abs(a_ref / a_fwd)


def run(cv, *, cpml_layers, dx_m=1e-3):
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cpml_layers, dx=dx_m,
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
    )
    for comp in ("ez", "hy"):
        sim.add_dft_plane_probe(axis="x", coordinate=cv.MON_LEFT_X,
                                component=comp, freqs=pf, name=f"{comp}_p")
    res = sim.run(num_periods=200, compute_s_params=False)
    Ez = np.asarray(res.dft_planes["ez_p"].accumulator)
    Hy = np.asarray(res.dft_planes["hy_p"].accumulator)
    Ny, Nz = Ez.shape[1], Ez.shape[2]
    y = np.arange(Ny) * dx_m
    z = np.arange(Nz) * dx_m
    return np.asarray(freqs), Ez, Hy, y, z


def main():
    cv = _load_cv11()
    print("=== CPML sweep ===")
    rows = []
    for n_cpml in (20, 40, 60):
        print(f"\n--- CPML_LAYERS={n_cpml} ---", flush=True)
        f, Ez, Hy, y, z = run(cv, cpml_layers=n_cpml)
        s11 = s11_from_field(Ez, Hy, y, z, f, cv.A_WG)
        spread = s11.max() - s11.min()
        rows.append((n_cpml, s11.min(), s11.mean(), s11.max(), spread))
        print(f"  CPML={n_cpml}: spread={spread:.4f}  mean={s11.mean():.4f}",
              flush=True)
    print(f"\n=== Summary ===")
    print(f"{'CPML':<8}{'min':>10}{'mean':>10}{'max':>10}{'spread':>10}")
    for n, mn, me, mx, sp in rows:
        print(f"{n:<8}{mn:>10.4f}{me:>10.4f}{mx:>10.4f}{sp:>10.4f}")


if __name__ == "__main__":
    main()

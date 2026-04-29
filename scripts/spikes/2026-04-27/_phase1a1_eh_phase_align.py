"""Phase 1A.1 follow-up: explicit E↔H half-cell phase alignment.

Yee staggering: E_z at (i, j, k+0.5), H_y at (i+0.5, j, k+0.5). On the
axis=0 port-plane probe, both arrays are sampled at probe.index, but
the H_y values are physically at x=(probe.index+0.5)·dx while E_z is
at x=probe.index·dx. For a forward TE10 wave e^{-jβx}, the H_y phase
relative to E_z is e^{-jβ·dx/2}.

Fix: multiply I by exp(+jβ·dx/2) before the wave decomposition to
align H to E's x position. Test: if per-freq oscillation collapses
to <1%, the half-cell x-offset was the mechanism.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp
import importlib.util


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    cv = _load_cv11()
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    dx_m = 1e-3
    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    pec_short_x = cv.PORT_RIGHT_X - 0.005
    sim = Simulation(
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
        reference_plane=0.050,
        name="left",
    )
    for comp in ("ey", "ez", "hy", "hz"):
        sim.add_dft_plane_probe(
            axis="x", coordinate=cv.PORT_LEFT_X, component=comp,
            freqs=pf, name=f"{comp}_port",
        )
    res = sim.run(num_periods=200, compute_s_params=False)

    e_z = np.transpose(np.asarray(res.dft_planes["ez_port"].accumulator), (1, 2, 0))
    h_y = np.transpose(np.asarray(res.dft_planes["hy_port"].accumulator), (1, 2, 0))
    ny, nz, n_freqs = e_z.shape
    a = (ny - 1) * dx_m
    y_int = np.arange(ny) * dx_m
    e_func = np.sin(np.pi * y_int / a)[:, None] * np.ones(nz)[None, :]
    h_func = e_func.copy()
    dA = dx_m * dx_m * np.ones((ny, nz))

    V = np.zeros(n_freqs, dtype=complex)
    I = np.zeros(n_freqs, dtype=complex)
    for f_idx in range(n_freqs):
        V[f_idx] = np.sum(e_z[:, :, f_idx] * e_func * dA)
        I[f_idx] = np.sum(h_y[:, :, f_idx] * h_func * dA)

    f_hz = np.asarray(freqs)
    omega = 2 * np.pi * f_hz
    C0 = 2.998e8
    MU_0 = 1.2566370614e-6
    f_c = C0 / (2 * a)
    k = omega / C0
    kc = 2 * np.pi * f_c / C0
    beta = np.sqrt(np.maximum(k**2 - kc**2, 0.0) + 0j)
    Z = omega * MU_0 / beta

    print(f"\nPEC-short PHASE-ALIGN comparison (dx={dx_m*1e3:.1f} mm):")

    # Variant 1: no alignment (control — Phase 1A.1 baseline)
    a_fwd = 0.5 * (V + I * Z)
    a_ref = V - a_fwd
    s11_no_align = np.abs(a_ref / a_fwd)
    print(f"  no align:                       min={s11_no_align.min():.4f} max={s11_no_align.max():.4f} osc={s11_no_align.max()-s11_no_align.min():.4f}")

    # Variant 2: align H to E plane (phase forward by β·dx/2)
    I_align_fwd = I * np.exp(+1j * beta * dx_m / 2)
    a_fwd = 0.5 * (V + I_align_fwd * Z)
    a_ref = V - a_fwd
    s11_align_fwd = np.abs(a_ref / a_fwd)
    print(f"  H aligned forward by β·dx/2:    min={s11_align_fwd.min():.4f} max={s11_align_fwd.max():.4f} osc={s11_align_fwd.max()-s11_align_fwd.min():.4f}")

    # Variant 3: align H to E plane (phase backward — opposite sign)
    I_align_bwd = I * np.exp(-1j * beta * dx_m / 2)
    a_fwd = 0.5 * (V + I_align_bwd * Z)
    a_ref = V - a_fwd
    s11_align_bwd = np.abs(a_ref / a_fwd)
    print(f"  H aligned backward by β·dx/2:   min={s11_align_bwd.min():.4f} max={s11_align_bwd.max():.4f} osc={s11_align_bwd.max()-s11_align_bwd.min():.4f}")

    # Variant 4: align E forward to H plane (E·exp(-jβ·dx/2))
    V_align = V * np.exp(-1j * beta * dx_m / 2)
    a_fwd = 0.5 * (V_align + I * Z)
    a_ref = V_align - a_fwd
    s11_e_align = np.abs(a_ref / a_fwd)
    print(f"  E aligned to H plane:           min={s11_e_align.min():.4f} max={s11_e_align.max():.4f} osc={s11_e_align.max()-s11_e_align.min():.4f}")

    # Variant 5: align both E and H to the midpoint
    V_mid = V * np.exp(-1j * beta * dx_m / 4)
    I_mid = I * np.exp(+1j * beta * dx_m / 4)
    a_fwd = 0.5 * (V_mid + I_mid * Z)
    a_ref = V_mid - a_fwd
    s11_midpoint = np.abs(a_ref / a_fwd)
    print(f"  E,H aligned to midpoint:        min={s11_midpoint.min():.4f} max={s11_midpoint.max():.4f} osc={s11_midpoint.max()-s11_midpoint.min():.4f}")

    print(f"\n=== Verdict ===")
    osc_min = min(
        s11_no_align.max() - s11_no_align.min(),
        s11_align_fwd.max() - s11_align_fwd.min(),
        s11_align_bwd.max() - s11_align_bwd.min(),
        s11_e_align.max() - s11_e_align.min(),
        s11_midpoint.max() - s11_midpoint.min(),
    )
    if osc_min < 0.01:
        print("PASS — explicit E/H phase alignment collapses oscillation to <1%.")
        print("E/H half-cell offset IS the mechanism for the per-freq residual.")
    elif osc_min < 0.05:
        print("PARTIAL — alignment helps significantly. E/H offset is part of the story.")
    else:
        print("FAIL — alignment doesn't change oscillation. Mechanism is elsewhere.")


if __name__ == "__main__":
    main()

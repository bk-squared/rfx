"""Phase 1A.1 — position-aware analytic template evaluation.

Hypothesis: rfx's analytic mode_profile evaluates E_func at cell
centers ((j+0.5)·dy), but the simulation's E_z lives at integer y
(j·dy) per Yee staggering and the +1 fence-post in grid.py:144-149.
The half-cell offset adds a position-mismatch error to the V/I
projection that may explain most of the remaining ~9% gap on KEEP-both
PEC-short.

Test: bypass the runtime V/I extractor entirely. Capture E and H at
the port plane via add_dft_plane_probe, compute V(f) and I(f) in
Python post-processing using the analytic Pozar formulas evaluated at
the EXACT Yee positions (E_z at integer y, integer x, half-z;
H_y at half-x, integer y, half-z; etc.) with full-aperture
integration (no DROP). Decompose with analytic β/Z. Compare to
production V/I extraction (DROP-both canonical).

If the position-aware projection gives PEC-short |S11| ≥ 0.99 on
KEEP-equivalent integration → Phase 1A.1 verified, the half-cell
offset was the dominant gap. Implementation work: ~3 days to
properly thread sample-position metadata through `_te_mode_profiles`
and the runtime extractor.
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


def _build_pec_short(cv):
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
        cpml_layers=cv.CPML_LAYERS,
        dx=cv.DX_M,
    )
    pec_short_x = cv.PORT_RIGHT_X - 0.005
    sim.add(
        Box((pec_short_x, 0.0, 0.0),
            (pec_short_x + 2 * cv.DX_M, cv.DOMAIN_Y, cv.DOMAIN_Z)),
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
    # DFT plane probes at port plane for E_y, E_z, H_y, H_z
    for comp in ("ey", "ez", "hy", "hz"):
        sim.add_dft_plane_probe(
            axis="x", coordinate=cv.PORT_LEFT_X, component=comp,
            freqs=pf, name=f"{comp}_port",
        )
    return sim


def main():
    cv = _load_cv11()
    sim_dut = _build_pec_short(cv)
    res_dut = sim_dut.run(num_periods=cv.NUM_PERIODS_LONG, compute_s_params=False)

    # Empty reference run
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim_ref = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS,
        dx=cv.DX_M,
    )
    pf = jnp.asarray(freqs)
    sim_ref.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050,
        name="left",
    )
    for comp in ("ey", "ez", "hy", "hz"):
        sim_ref.add_dft_plane_probe(
            axis="x", coordinate=cv.PORT_LEFT_X, component=comp,
            freqs=pf, name=f"{comp}_port",
        )
    res_ref = sim_ref.run(num_periods=cv.NUM_PERIODS_LONG, compute_s_params=False)

    def extract_VI(result):
        # Field accumulators: shape (n_freqs, ny, nz)
        e_z = np.transpose(np.asarray(result.dft_planes["ez_port"].accumulator), (1, 2, 0))
        h_y = np.transpose(np.asarray(result.dft_planes["hy_port"].accumulator), (1, 2, 0))
        ny, nz, n_freqs = e_z.shape
        # rfx uses uniform cubic cells dx
        dx = cv.DX_M
        # Effective aperture from +1 fence-post: a = (ny-1)*dx, b = (nz-1)*dx
        a = (ny - 1) * dx
        b = (nz - 1) * dx
        # Yee positions: E_z at integer y (j*dy) integer x, half-integer z.
        # H_y at half-integer x, integer y, half-integer z.
        # On the same x port plane: E_z at (j*dy, (k+0.5)*dz) for j=0..ny-1, k=0..nz-1
        # H_y at (j*dy, (k+0.5)*dz) — same in-plane positions on the y/z plane.
        y_int = np.arange(ny) * dx
        z_half = (np.arange(nz) + 0.5) * dx
        # Analytic Pozar TE10: E_z = sin(pi*y/a) * 1, H_y proportional to E_z / Z_TE
        e_func = np.sin(np.pi * y_int / a)[:, None] * np.ones(nz)[None, :]
        h_func = e_func.copy()  # same shape for TE10 (Z_TE absorbs the prefactor)
        # Cell-area weights — uniform dx*dx per cell
        dA = dx * dx * np.ones((ny, nz))
        # NOTE: with +1 fence-post, j=0 at y=0 AND j=ny-1 at y=a are physical
        # PEC nodes where E_z = 0. The analytic e_func at those positions also
        # vanishes (sin(0)=sin(pi)=0). So no DROP needed — full integration.
        # Per-frequency V, I
        V = np.zeros(n_freqs, dtype=complex)
        I = np.zeros(n_freqs, dtype=complex)
        for f_idx in range(n_freqs):
            V[f_idx] = np.sum(e_z[:, :, f_idx] * e_func * dA)
            I[f_idx] = np.sum(h_y[:, :, f_idx] * h_func * dA)
        return V, I, a, b

    V_dut, I_dut, a_eff, b_eff = extract_VI(res_dut)
    V_ref, I_ref, _, _ = extract_VI(res_ref)
    print(f"Effective aperture: a={a_eff*1e3:.3f} mm, b={b_eff*1e3:.3f} mm")

    # Analytic β and Z_TE for TE10 at each frequency
    f_hz = np.asarray(cv.FREQS_HZ)
    omega = 2 * np.pi * f_hz
    C0 = 2.998e8
    EPS_0 = 8.854187817e-12
    MU_0 = 1.2566370614e-6
    f_c = C0 / (2 * a_eff)  # TE10 cutoff
    print(f"TE10 f_cutoff = {f_c/1e9:.3f} GHz")
    k = omega / C0
    kc = 2 * np.pi * f_c / C0
    beta = np.sqrt(np.maximum(k**2 - kc**2, 0.0) + 0j)
    Z_TE = omega * MU_0 / beta

    # Wave decomposition (OpenEMS convention)
    a_fwd_dut = 0.5 * (V_dut + I_dut * Z_TE)
    a_ref_dut = V_dut - a_fwd_dut

    a_fwd_ref = 0.5 * (V_ref + I_ref * Z_TE)
    a_ref_ref = V_ref - a_fwd_ref

    # S11 = (a_ref_dut - a_ref_ref) / a_fwd_ref  (two-run normalize)
    # OR for normalize=False: S11 = a_ref_dut / a_fwd_dut
    s11_norm_false = a_ref_dut / a_fwd_dut
    s11_norm_true = (a_ref_dut - a_ref_ref) / a_fwd_ref

    print(f"\nPhase 1A.1 (position-aware analytic projection, no DROP):")
    print(f"  S11 (normalize=False): min |S11| = {np.abs(s11_norm_false).min():.4f}")
    print(f"                          max |S11| = {np.abs(s11_norm_false).max():.4f}")
    print(f"                          mean      = {np.abs(s11_norm_false).mean():.4f}")
    print(f"  S11 (normalize=True):  min |S11| = {np.abs(s11_norm_true).min():.4f}")
    print(f"                          max |S11| = {np.abs(s11_norm_true).max():.4f}")
    print(f"                          mean      = {np.abs(s11_norm_true).mean():.4f}")
    print(f"  per-freq S11 (normalize=False): {' '.join(f'{abs(x):.4f}' for x in s11_norm_false)}")

    minS = np.abs(s11_norm_false).min()
    print(f"\n=== Phase 1A.1 verdict ===")
    if minS >= 0.99:
        print("PASS — position-aware analytic projection reaches Meep class on KEEP-equivalent.")
    elif minS >= 0.95:
        print("PARTIAL — substantial improvement but not yet Meep class. Source symmetry likely needed.")
    else:
        print("INSUFFICIENT — analytic position-aware projection alone doesn't close the gap.")


if __name__ == "__main__":
    main()

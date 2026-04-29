"""Phase 1A.1 resolution-sweep validation.

Hypothesis (from commit 45c671e): the +/-7% per-freq oscillation in
Phase 1A.1's analytic projection on PEC-short comes from Yee numerical
dispersion at dx=1mm. Yee error scales as (k·dx)^2 per wavelength.
Halving dx should reduce per-freq oscillation by 4x; quartering dx by
16x. If observed scaling matches this prediction, the "Yee dispersion
is the residual gap" claim is verified.

Test: re-run the Phase 1A.1 spike at dx={1.0, 0.5, 0.25} mm. Measure
per-freq |S11| min/max and oscillation amplitude (max - min). Report
the scaling vs dx.
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


def run_at_dx(cv, dx_m, num_periods):
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    pec_short_x = cv.PORT_RIGHT_X - 0.005

    def build_sim(with_pec):
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
        if with_pec:
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
        return sim

    sim_dut = build_sim(with_pec=True)
    res_dut = sim_dut.run(num_periods=num_periods, compute_s_params=False)

    def extract_VI(result):
        e_z = np.transpose(np.asarray(result.dft_planes["ez_port"].accumulator), (1, 2, 0))
        h_y = np.transpose(np.asarray(result.dft_planes["hy_port"].accumulator), (1, 2, 0))
        ny, nz, n_freqs = e_z.shape
        a = (ny - 1) * dx_m
        b = (nz - 1) * dx_m
        y_int = np.arange(ny) * dx_m
        e_func = np.sin(np.pi * y_int / a)[:, None] * np.ones(nz)[None, :]
        h_func = e_func.copy()
        dA = dx_m * dx_m * np.ones((ny, nz))
        V = np.zeros(n_freqs, dtype=complex)
        I = np.zeros(n_freqs, dtype=complex)
        for f_idx in range(n_freqs):
            V[f_idx] = np.sum(e_z[:, :, f_idx] * e_func * dA)
            I[f_idx] = np.sum(h_y[:, :, f_idx] * h_func * dA)
        return V, I, a, b

    V_dut, I_dut, a_eff, _ = extract_VI(res_dut)

    f_hz = np.asarray(cv.FREQS_HZ)
    omega = 2 * np.pi * f_hz
    C0 = 2.998e8
    MU_0 = 1.2566370614e-6
    f_c = C0 / (2 * a_eff)
    k = omega / C0
    kc = 2 * np.pi * f_c / C0
    beta = np.sqrt(np.maximum(k**2 - kc**2, 0.0) + 0j)
    Z_TE = omega * MU_0 / beta

    a_fwd_dut = 0.5 * (V_dut + I_dut * Z_TE)
    a_ref_dut = V_dut - a_fwd_dut
    s11 = a_ref_dut / a_fwd_dut
    return np.abs(s11), a_eff, f_c


def main():
    cv = _load_cv11()
    print(f"Resolution sweep on PEC-short, Phase 1A.1 analytic projection")
    print(f"{'dx_mm':>7} {'a_eff_mm':>9} {'fc_GHz':>8} {'min':>8} {'max':>8} {'mean':>8} {'osc(max-min)':>13}")
    print("-" * 70)
    rows = []
    # Allow CLI override: python ... 1.0  → run only 1mm
    requested = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1.0, 0.5]
    for dx_mm in requested:
        num_periods = max(int(round(cv.NUM_PERIODS_LONG)), 200)
        try:
            s11, a_eff, f_c = run_at_dx(cv, dx_mm * 1e-3, num_periods=num_periods)
        except Exception as e:
            print(f"{dx_mm:>7.2f}  FAIL: {type(e).__name__}: {e}")
            continue
        osc = float(s11.max() - s11.min())
        rows.append((dx_mm, float(s11.min()), float(s11.max()), float(s11.mean()), osc))
        print(f"{dx_mm:>7.2f} {a_eff*1e3:>9.3f} {f_c/1e9:>8.3f} "
              f"{s11.min():>8.4f} {s11.max():>8.4f} {s11.mean():>8.4f} {osc:>13.4f}",
              flush=True)
    if len(rows) < 2:
        print("\nInsufficient data for scaling analysis.")
        return

    print(f"\n=== Yee dispersion scaling check ===")
    print(f"Prediction (dispersion ~ dx^2): osc(0.5mm)/osc(1mm) ≈ 0.25; "
          f"osc(0.25mm)/osc(1mm) ≈ 0.0625")
    print(f"\nrun pairs (osc, dx):")
    for dx, mn, mx, me, o in rows:
        print(f"  dx={dx:.2f} mm  osc={o:.4f}")
    if len(rows) >= 2:
        for i in range(1, len(rows)):
            ratio = rows[i][4] / rows[0][4]
            scale = rows[i][0] / rows[0][0]
            predicted = scale ** 2
            print(f"  dx ratio {scale:.3f} -> osc ratio {ratio:.4f}  (Yee predicts {predicted:.4f})")


if __name__ == "__main__":
    main()

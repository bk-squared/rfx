"""Phase 1A.1 follow-up: Yee-discrete Z_TE_disc vs analytic Z_TE.

Resolution sweep showed Phase 1A.1 per-freq oscillation scales as dx¹
(not dx² as Yee dispersion would predict). Linear-in-dx is the
signature of a Yee E/H staggering mismatch: in Yee, E and H samples
are at half-cell offset in space and half-step offset in time;
analytic Z_TE = ωμ/β assumes continuous E/H, while Yee-discrete
Z_TE_disc = μ·dx·sin(ω·dt/2)/(dt·sin(β·dx/2)) compensates the
staggering.

Test: re-run Phase 1A.1 with Yee-discrete Z_TE_disc (using rfx's
existing _compute_mode_impedance for Yee path). If oscillation
collapses to <1%, the hypothesis is confirmed and OpenEMS-class
accuracy IS achievable on Yee staggered FDTD with continuous-coord
analytic templates + Yee-discrete Z.
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


def run(cv, dx_m, num_periods, *, use_yee_z):
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box
    from rfx.sources.waveguide_port import _compute_beta, _compute_mode_impedance

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
    res = sim.run(num_periods=num_periods, compute_s_params=False)
    dt = float(res.dt) if hasattr(res, "dt") else dx_m / (2.998e8 * np.sqrt(3.0))

    e_z = np.transpose(np.asarray(res.dft_planes["ez_port"].accumulator), (1, 2, 0))
    h_y = np.transpose(np.asarray(res.dft_planes["hy_port"].accumulator), (1, 2, 0))
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

    f_hz = jnp.asarray(freqs)
    f_c = 2.998e8 / (2 * a)
    if use_yee_z:
        Z = np.asarray(_compute_mode_impedance(f_hz, f_c, "TE", dt=dt, dx=dx_m))
    else:
        omega = 2 * np.pi * np.asarray(freqs)
        MU_0 = 1.2566370614e-6
        C0 = 2.998e8
        k = omega / C0
        kc = 2 * np.pi * f_c / C0
        beta = np.sqrt(np.maximum(k**2 - kc**2, 0.0) + 0j)
        Z = omega * MU_0 / beta

    a_fwd = 0.5 * (V + I * Z)
    a_ref = V - a_fwd
    s11 = a_ref / a_fwd
    return np.abs(s11), float(a)


def main():
    cv = _load_cv11()
    requested = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1.0, 0.5]
    print(f"\n{'dx_mm':>7} {'Z':>10} {'min':>8} {'max':>8} {'mean':>8} {'osc':>10}")
    print("-" * 60)
    rows = {}
    for dx_mm in requested:
        for use_yee_z, label in [(False, "analytic"), (True, "Yee-disc")]:
            num_periods = 200
            try:
                s11, a_eff = run(cv, dx_mm * 1e-3, num_periods=num_periods, use_yee_z=use_yee_z)
            except Exception as e:
                print(f"{dx_mm:>7.2f} {label:>10}  FAIL: {type(e).__name__}: {e}")
                continue
            osc = float(s11.max() - s11.min())
            rows[(dx_mm, label)] = (float(s11.min()), float(s11.max()), float(s11.mean()), osc)
            print(f"{dx_mm:>7.2f} {label:>10} {s11.min():>8.4f} {s11.max():>8.4f} "
                  f"{s11.mean():>8.4f} {osc:>10.4f}",
                  flush=True)

    # Verdict
    print(f"\n=== Verdict ===")
    if (1.0, "Yee-disc") in rows:
        osc_yee_1 = rows[(1.0, "Yee-disc")][3]
        osc_an_1 = rows[(1.0, "analytic")][3]
        print(f"At dx=1mm: oscillation analytic Z = {osc_an_1:.4f}, Yee-disc Z = {osc_yee_1:.4f}")
        if osc_yee_1 < 0.01:
            print("PASS — Yee-discrete Z collapses per-freq oscillation. "
                  "OpenEMS-class accuracy achievable with continuous-coord templates "
                  "+ Yee-discrete Z at rfx's typical resolution.")
        elif osc_yee_1 < osc_an_1 * 0.3:
            print("PARTIAL — Yee-discrete Z helps significantly but residual remains. "
                  "Likely need additional staggering corrections.")
        else:
            print("FAIL — Yee-discrete Z alone doesn't close the gap; "
                  "the position-offset hypothesis is wrong or incomplete.")


if __name__ == "__main__":
    main()

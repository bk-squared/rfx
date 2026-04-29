"""Phase 1A.1 combined fix test: E/H phase align + h_offset variation.

Mechanism C confirmed in commit 05f5b31: E/H half-cell phase
alignment halves the oscillation (0.133 → 0.060). Research agent
predicts Mechanism A (h_offset roll-averaging asymmetry in
_shift_profile_to_dual at waveguide_port.py:463-479) accounts for
the remaining ~50%.

This spike runs PEC-short at dx=1mm with E/H phase align ON,
testing four h_offset configurations:
  (0, 0)    — no roll shift (test Mechanism A by removing it)
  (0.5, 0)  — roll u-axis only
  (0, 0.5)  — roll v-axis only
  (0.5, 0.5) — both (current default, baseline of 1A.1)

If h_offset=(0,0) drops oscillation toward <1%, the rfx h_offset
mechanism is the remaining residual. If it doesn't change much,
mechanism is elsewhere.

Note: h_offset is a parameter of init_waveguide_port. The spike's
analytic projection uses uniform-in-z e_func/h_func so the H
template's z half-cell shift is moot for TE10 — but the H template
USED IN THE TFSF SOURCE INJECTION (rfx side) has h_offset baked in.
This is what may create the source-vs-extractor asymmetry.
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


def _patch_h_offset(target_offset):
    """Monkey-patch init_waveguide_port to override h_offset."""
    import rfx.sources.waveguide_port as wp
    import rfx.api as api_mod
    orig = wp.init_waveguide_port

    def patched(*args, **kwargs):
        kwargs["h_offset"] = target_offset
        return orig(*args, **kwargs)

    wp.init_waveguide_port = patched
    api_mod.init_waveguide_port = patched


def run(cv, dx_m, h_offset):
    _patch_h_offset(h_offset)
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

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

    # Mechanism C fix: spatial half-cell phase H * exp(+j*beta*dx/2)
    # NEW: also time-stagger compensation — H DFT is at (n+0.5)*dt physical
    # time, so multiply by exp(+j*omega*dt/2) to align with E DFT at n*dt.
    # CFL: dt = dx/(c*sqrt(3)) for cubic-cell uniform Yee.
    C0 = 2.998e8
    dt_yee = dx_m / (C0 * np.sqrt(3.0))
    I_aligned = I * np.exp(+1j * beta * dx_m / 2) * np.exp(-1j * omega * dt_yee / 2)
    a_fwd = 0.5 * (V + I_aligned * Z)
    a_ref = V - a_fwd
    s11 = np.abs(a_ref / a_fwd)
    return s11, float(a)


def main():
    cv = _load_cv11()
    dx_m = 1e-3
    print(f"\nPEC-short at dx={dx_m*1e3:.1f} mm, with E/H phase align ON:")
    print(f"{'h_offset':<14} {'min':>8} {'max':>8} {'mean':>8} {'osc':>8}")
    print("-" * 50)
    rows = []
    for h_off in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]:
        try:
            s11, a_eff = run(cv, dx_m, h_off)
        except Exception as e:
            print(f"{str(h_off):<14}  FAIL: {type(e).__name__}: {e}")
            continue
        osc = float(s11.max() - s11.min())
        rows.append((h_off, float(s11.min()), float(s11.max()), float(s11.mean()), osc))
        print(f"{str(h_off):<14} {s11.min():>8.4f} {s11.max():>8.4f} "
              f"{s11.mean():>8.4f} {osc:>8.4f}", flush=True)

    print(f"\n=== Verdict ===")
    if rows:
        osc_min = min(r[4] for r in rows)
        if osc_min < 0.01:
            print(f"PASS — combined fix achieves <1% per-freq oscillation. "
                  f"OpenEMS-class accuracy IS achievable in rfx via E/H phase "
                  f"align + appropriate h_offset.")
        elif osc_min < 0.05:
            print(f"PARTIAL — combined fix reduces residual to ~5%. Real "
                  f"improvement; further work likely closes remaining gap.")
        else:
            print(f"INSUFFICIENT — h_offset alone doesn't close the residual. "
                  f"Mechanism A may need a different intervention.")


if __name__ == "__main__":
    main()

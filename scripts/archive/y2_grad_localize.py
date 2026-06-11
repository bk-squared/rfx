"""Localize the wrong-sign AD gradient.

After patching `_solve_q` with custom_jvp (commit c747700), end-to-end
AD gradient is unchanged.  This means the bug is NOT in the q-solver.

This script computes FD vs AD on intermediate quantities to identify
where the sign-flip happens.  At L=9.5mm (where AD wrong, FD says +97):

  Stage A: |s21|²  (cost — known wrong)
  Stage B: Re(s21)
  Stage C: Im(s21)
  Stage D: |alpha_p|² (just one wave amplitude)
  Stage E: |alpha_d|²
  Stage F: alpha_p_real
  Stage G: alpha_d_real

If FD/AD agrees at some stage but not later, the bug is between them.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/root/workspace/byungkwan-workspace/research/rfx")
from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.msl_wave_decomp import (
    register_msl_plane_probes, _v_from_plane, _solve_3probe_jax,
)


EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX = 127e-6
L_LINE = 30e-3
PORT_MARGIN = 1.6e-3
F_MAX = 9e9
F_TARGET = 6e9
NUM_PERIODS = 10
SIGMOID_BETA = max(DX * 0.25, 0.05 * H_SUB)


def build_sim(freqs):
    LX = L_LINE + 2 * PORT_MARGIN
    L_STUB_MAX = 14e-3
    LY = W_TRACE + 4 * (2 * H_SUB + 8 * DX) + L_STUB_MAX
    LZ = H_SUB + 1.0e-3
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = (2 * H_SUB + 8 * DX) + W_TRACE / 2
    trace_y_hi = y_trace + W_TRACE / 2
    sim.add(Box((0, y_trace - W_TRACE / 2, H_SUB),
                (LX, y_trace + W_TRACE / 2, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, y_trace, 0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_trace, 0),
                     width=W_TRACE, height=H_SUB, direction="-x",
                     impedance=50.0)
    object.__setattr__(sim._msl_ports[1], "excite", False)
    d_set = register_msl_plane_probes(sim, port_index=0, freqs=freqs,
                                      name_prefix="d")
    p_set = register_msl_plane_probes(sim, port_index=1, freqs=freqs,
                                      name_prefix="p")
    return sim, trace_y_hi, d_set, p_set


def build_stub_occ(grid, trace_y_hi, L_stub):
    LX = L_LINE + 2 * PORT_MARGIN
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    stub_xc = LX / 2
    z_patch = H_SUB + 0.5 * DX
    x_centres = (np.arange(nx) - pad_x + 0.5) * DX
    y_centres = (np.arange(ny) - pad_y + 0.5) * DX
    z_centres = (np.arange(nz) - pad_z + 0.5) * DX
    in_x = ((x_centres >= stub_xc - W_TRACE / 2) &
            (x_centres <= stub_xc + W_TRACE / 2)).astype(np.float32)
    in_z = (np.abs(z_centres - z_patch) <= 0.5 * DX).astype(np.float32)
    y_far = jnp.asarray(y_centres - trace_y_hi, dtype=jnp.float32)
    sig_low = jax.nn.sigmoid(y_far / SIGMOID_BETA)
    sig_high = jax.nn.sigmoid((L_stub - y_far) / SIGMOID_BETA)
    sig_y = sig_low * sig_high
    return (jnp.asarray(in_x)[:, None, None] * sig_y[None, :, None]
            * jnp.asarray(in_z)[None, None, :]).astype(jnp.float32)


def main():
    freqs = jnp.asarray([F_TARGET], dtype=jnp.float32)
    sim, trace_y_hi, d_set, p_set = build_sim(freqs)
    grid = sim._build_grid()
    period = 1.0 / float(sim._freq_max)
    n_steps_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
    K = max(8, int(math.isqrt(n_steps_raw)))
    n_steps = ((n_steps_raw + K - 1) // K) * K
    print(f"dx={DX*1e6:.0f}µm, n_steps={n_steps} ({K} segs)")

    def fwd(L_stub):
        occ = build_stub_occ(grid, trace_y_hi, L_stub)
        return sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                           checkpoint_segments=K, skip_preflight=True)

    def alpha_d_alpha_p(L_stub):
        fr = fwd(L_stub)
        v1d = _v_from_plane(fr, d_set.ez1_name, d_set)[0]
        v2d = _v_from_plane(fr, d_set.ez2_name, d_set)[0]
        v3d = _v_from_plane(fr, d_set.ez3_name, d_set)[0]
        v1p = _v_from_plane(fr, p_set.ez1_name, p_set)[0]
        v2p = _v_from_plane(fr, p_set.ez2_name, p_set)[0]
        v3p = _v_from_plane(fr, p_set.ez3_name, p_set)[0]
        alpha_d, _, _ = _solve_3probe_jax(v1d, v2d, v3d, None)
        alpha_p, _, _ = _solve_3probe_jax(v1p, v2p, v3p, None)
        return alpha_d, alpha_p, v1d, v2d, v3d, v1p, v2p, v3p

    def cost_full(L_stub):
        ad, ap, *_ = alpha_d_alpha_p(L_stub)
        s21 = ap / (ad + 1e-30)
        return jnp.abs(s21) ** 2

    def cost_alpha_p_sq(L_stub):
        _, ap, *_ = alpha_d_alpha_p(L_stub)
        return jnp.abs(ap) ** 2

    def cost_alpha_d_sq(L_stub):
        ad, _, *_ = alpha_d_alpha_p(L_stub)
        return jnp.abs(ad) ** 2

    def cost_v1p_sq(L_stub):
        _, _, _, _, _, v1p, _, _ = alpha_d_alpha_p(L_stub)
        return jnp.abs(v1p) ** 2

    def cost_v1d_sq(L_stub):
        _, _, v1d, _, _, _, _, _ = alpha_d_alpha_p(L_stub)
        return jnp.abs(v1d) ** 2

    L = 9.5e-3
    delta = 0.05e-3

    print("\n  Localization at L=9.5mm where AD is wrong-signed")
    print("  " + "=" * 78)
    print(f"  {'stage':<25}  {'value':>12}  {'FD grad':>12}  "
          f"{'AD grad':>12}  {'sign FD':>8}  {'sign AD':>8}")

    for label, fn in [
        ("|s21|² (cost)", cost_full),
        ("|alpha_p|²", cost_alpha_p_sq),
        ("|alpha_d|²", cost_alpha_d_sq),
        ("|v1_passive|² (DFT)", cost_v1p_sq),
        ("|v1_driven|² (DFT)", cost_v1d_sq),
    ]:
        t0 = time.time()
        L_jax = jnp.asarray(L, dtype=jnp.float32)
        c_lo = float(fn(jnp.asarray(L - delta, dtype=jnp.float32)))
        c_hi = float(fn(jnp.asarray(L + delta, dtype=jnp.float32)))
        fd = (c_hi - c_lo) / (2.0 * delta)
        c_ad, ad = jax.value_and_grad(fn)(L_jax)
        c_ad = float(c_ad); ad = float(ad)
        sign_fd = "+" if fd > 0 else "-"
        sign_ad = "+" if ad > 0 else "-"
        match = "MATCH" if sign_fd == sign_ad else "*** DISAGREE ***"
        print(f"  {label:<25}  {c_ad:12.4e}  {fd:+12.4e}  "
              f"{ad:+12.4e}  {sign_fd:>8}  {sign_ad:>8}  {match}  "
              f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

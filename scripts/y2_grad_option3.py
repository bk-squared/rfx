"""Y2 FD-vs-AD with option 3 (always pick |q|≤1) q-root selection.

Tests whether replacing the err-based root selector in
`_solve_3probe_jax` with |q|≤1 selection fixes the wrong-sign AD
gradient surfaced by run #605.

Architect's analysis predicts option 3 fails at near-resonance because
both |q₊| and |q₋| are near unity (FP32 noise dominates).  This
script provides empirical confirmation either way.

Compares to baseline (current err-based selector) at the same L values.
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
from rfx.probes import msl_wave_decomp as mwd
from rfx.probes.msl_wave_decomp import (
    register_msl_plane_probes, _v_from_plane,
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


def _solve_3probe_v3(v1, v2, v3, eps: float = 1e-30):
    """Option 3: always pick |q| ≤ 1 root, no err-based selection."""
    coeff = (v1 + v3) / (v2 + eps)
    disc = coeff ** 2 - 4.0
    sqrt_disc = jnp.sqrt(disc.astype(jnp.complex64))
    q_plus = (coeff + sqrt_disc) / 2.0
    q_minus = (coeff - sqrt_disc) / 2.0
    use_plus = jnp.abs(q_plus) <= jnp.abs(q_minus)
    q = jnp.where(use_plus, q_plus, q_minus)
    denom = (q * q - 1.0) + eps
    alpha = (q * v2 - v1) / denom
    gamma = q * (v1 * q - v2) / denom
    return alpha, gamma, q


def extract_v3(fr, driven, passive):
    v1d = _v_from_plane(fr, driven.ez1_name, driven)
    v2d = _v_from_plane(fr, driven.ez2_name, driven)
    v3d = _v_from_plane(fr, driven.ez3_name, driven)
    v1p = _v_from_plane(fr, passive.ez1_name, passive)
    v2p = _v_from_plane(fr, passive.ez2_name, passive)
    v3p = _v_from_plane(fr, passive.ez3_name, passive)
    alpha_d, gamma_d, _ = _solve_3probe_v3(v1d, v2d, v3d)
    alpha_p, _, _ = _solve_3probe_v3(v1p, v2p, v3p)
    eps = 1e-30
    s11 = gamma_d / (alpha_d + eps)
    s21 = alpha_p / (alpha_d + eps)
    return s11, s21


def main():
    freqs = jnp.asarray([F_TARGET], dtype=jnp.float32)
    sim, trace_y_hi, d_set, p_set = build_sim(freqs)
    grid = sim._build_grid()
    period = 1.0 / float(sim._freq_max)
    n_steps_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
    K = max(8, int(math.isqrt(n_steps_raw)))
    n_steps = ((n_steps_raw + K - 1) // K) * K
    print(f"dx={DX*1e6:.0f}µm, n_steps={n_steps} ({K} segs)")

    def cost_v3(L_stub):
        occ = build_stub_occ(grid, trace_y_hi, L_stub)
        fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                         checkpoint_segments=K, skip_preflight=True)
        _, s21 = extract_v3(fr, d_set, p_set)
        return jnp.abs(s21[0]) ** 2

    def cost_baseline(L_stub):
        # Uses current `_solve_3probe_jax` (err-based selector)
        occ = build_stub_occ(grid, trace_y_hi, L_stub)
        fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                         checkpoint_segments=K, skip_preflight=True)
        _, s21 = mwd.extract_msl_s_params_jax_plane(fr, d_set, p_set)
        return jnp.abs(s21[0]) ** 2

    L_test = [5.0e-3, 7.0e-3, 9.5e-3, 11.0e-3]
    delta = 0.05e-3

    print("\n  Option 3: always pick |q| ≤ 1 root")
    print("  " + "=" * 78)
    print(f"  {'L (mm)':>8}  {'cost':>10}  {'FD grad':>14}  "
          f"{'AD grad (v3)':>14}  {'sign FD':>8}  {'sign AD':>8}")
    grad_v3 = jax.value_and_grad(cost_v3)
    for L in L_test:
        t0 = time.time()
        L_jax = jnp.asarray(L, dtype=jnp.float32)
        c_lo = float(cost_v3(jnp.asarray(L - delta, dtype=jnp.float32)))
        c_hi = float(cost_v3(jnp.asarray(L + delta, dtype=jnp.float32)))
        fd = (c_hi - c_lo) / (2.0 * delta)
        c_ad, ad = grad_v3(L_jax)
        c_ad = float(c_ad); ad = float(ad)
        sign_fd = "+" if fd > 0 else "-" if fd < 0 else "0"
        sign_ad = "+" if ad > 0 else "-" if ad < 0 else "0"
        match = "  MATCH" if sign_fd == sign_ad else "  *** DISAGREE ***"
        print(f"  {L*1e3:8.2f}  {c_ad:10.4e}  {fd:+14.4e}  "
              f"{ad:+14.4e}  {sign_fd:>8}  {sign_ad:>8}  "
              f"({time.time()-t0:.1f}s){match}")


if __name__ == "__main__":
    main()

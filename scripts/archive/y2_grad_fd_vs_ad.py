"""Y2 finite-difference vs AD gradient comparison at dx=127µm.

Run #600 found that on the σ-loading-reverted branch, Adam goes the
wrong direction from L=9.5mm: cost rises with L (real grad > 0) but
AD reports negative grad.  The brute-scan cost surface and the L-sweep
diagnostic both show the cost is monotonically increasing in
L ∈ [7mm, 12mm], so the wrong-sign cannot be a non-monotonicity issue.

This script computes:
  - cost(L) via forward + JAX plane extractor at multiple L values
  - finite-difference gradient: (cost(L+δ) - cost(L-δ)) / (2δ)
  - AD gradient: jax.value_and_grad(cost_of_L)(L)

If FD and AD disagree on sign at any L, the AD pipeline has a bug.
We also compare with and without `checkpoint_segments` to localize
whether the bug is in the segmented checkpointing reverse-pass or
in the underlying chain.

Test L values:
  - L=5.0mm (descent toward notch, real grad < 0)
  - L=7.0mm (at notch, real grad ≈ 0)
  - L=9.5mm (Adam start, real grad > 0)
  - L=11.0mm (deeper ascent, real grad > 0)
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
    register_msl_plane_probes, extract_msl_s_params_jax_plane,
)


EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX = 127e-6  # clean alignment, sub_cells = 2.000
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
    print(f"dx={DX*1e6:.0f}µm, n_steps={n_steps} ({K} segs), "
          f"f_target={F_TARGET/1e9:.1f}GHz, β={SIGMOID_BETA*1e6:.1f}µm")

    def cost_of_L(L_stub, *, K_seg):
        occ = build_stub_occ(grid, trace_y_hi, L_stub)
        if K_seg is None:
            fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                             skip_preflight=True)
        else:
            fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                             checkpoint_segments=K_seg, skip_preflight=True)
        _, s21 = extract_msl_s_params_jax_plane(fr, d_set, p_set)
        return jnp.abs(s21[0]) ** 2

    L_test = [5.0e-3, 7.0e-3, 9.5e-3, 11.0e-3]
    delta = 0.05e-3   # 50 µm — well below SIGMOID_BETA-scale wobble

    print("\n" + "=" * 80)
    print(" Test 1: cost(L) — fine brute scan")
    print("=" * 80)
    L_brute = np.linspace(4e-3, 12e-3, 17)
    for L in L_brute:
        t0 = time.time()
        c = float(cost_of_L(jnp.asarray(L, dtype=jnp.float32), K_seg=K))
        c_db = 10 * math.log10(max(c, 1e-12))
        print(f"  L={L*1e3:6.3f}mm  cost={c:.4e}  ({c_db:+6.1f} dB)  "
              f"({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 80)
    print(" Test 2: FD vs AD gradient at key L values")
    print("=" * 80)
    print(f"  {'L (mm)':>8}  {'cost':>10}  {'FD grad':>14}  "
          f"{'AD grad (K=K)':>16}  {'sign FD':>8}  {'sign AD':>8}")

    grad_fn_K = jax.value_and_grad(lambda L: cost_of_L(L, K_seg=K))

    for L in L_test:
        t0 = time.time()
        L_jax = jnp.asarray(L, dtype=jnp.float32)
        c_lo = float(cost_of_L(jnp.asarray(L - delta, dtype=jnp.float32),
                               K_seg=K))
        c_hi = float(cost_of_L(jnp.asarray(L + delta, dtype=jnp.float32),
                               K_seg=K))
        fd_grad = (c_hi - c_lo) / (2.0 * delta)
        c_ad, ad_grad = grad_fn_K(L_jax)
        ad_grad = float(ad_grad); c_ad = float(c_ad)
        sign_fd = "+" if fd_grad > 0 else "-" if fd_grad < 0 else "0"
        sign_ad = "+" if ad_grad > 0 else "-" if ad_grad < 0 else "0"
        match = "  MATCH" if sign_fd == sign_ad else "  *** DISAGREE ***"
        print(f"  {L*1e3:8.2f}  {c_ad:10.4e}  {fd_grad:+14.4e}  "
              f"{ad_grad:+16.4e}  {sign_fd:>8}  {sign_ad:>8}  "
              f"({time.time()-t0:.1f}s){match}")


if __name__ == "__main__":
    main()

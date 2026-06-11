"""Brute-scan-only diagnostic for the dx=80 µm plane-lane bug.

Hypotheses (see rfx-known-issues.md "Y2 Phase 4 follow-up"):

  A — substrate-cell alignment.  At dx=h_sub/dx fractional, the trace
      PEC spans two Yee cells (h_sub/dx = 3.175 at 80 µm sets the trace
      bottom 14 µm into a substrate cell).  Test: run the same brute
      scan at dx ∈ {127, 85, 64} µm where h_sub/dx is near-integer
      (alignment offset ≤ 2 µm); if all three give |S21|² ≤ 1 while
      dx=80 µm gives |S21|² > 1, A holds.

  D — 3-probe per-Δ phasor numerical floor.  At smaller dx the
      n_probe_spacing × dx physical Δ shrinks, so β·Δ shrinks and the
      `q_plus`/`q_minus` roots of the 3-probe quadratic land closer
      together.  Test: re-run dx=80 µm with n_probe_spacing in cells
      bumped so the physical Δ matches dx=127 µm's natural spacing
      (3 cells × 127 µm = 381 µm → 5 cells × 80 µm = 400 µm).

Output: |S21|² at each probed L_stub on each dx / spacing config; a
clean PASS in only the near-integer dx values is the smoking gun for
A; a PASS at dx=80 µm with n_probe_spacing=5 is the smoking gun for D.
"""
from __future__ import annotations

import os
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
L_LINE = 30e-3
PORT_MARGIN = 1.6e-3
F_MAX = 9e9
F_TARGET = 6e9


def build_sim(dx: float, n_probe_spacing: int, freqs):
    LX = L_LINE + 2 * PORT_MARGIN
    L_STUB_MAX = 14e-3
    LY = W_TRACE + 2 * (2 * H_SUB + 8 * dx) + L_STUB_MAX + 2 * (2 * H_SUB + 8 * dx)
    LZ = H_SUB + 1.0e-3
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro")
    y_trace = (2 * H_SUB + 8 * dx) + W_TRACE / 2
    trace_y_hi = y_trace + W_TRACE / 2
    sim.add(Box((0, y_trace - W_TRACE / 2, H_SUB),
                (LX, y_trace + W_TRACE / 2, H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, y_trace, 0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0,
                     n_probe_spacing=n_probe_spacing)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_trace, 0), width=W_TRACE,
                     height=H_SUB, direction="-x", impedance=50.0,
                     n_probe_spacing=n_probe_spacing)
    object.__setattr__(sim._msl_ports[1], "excite", False)
    d_set = register_msl_plane_probes(sim, port_index=0, freqs=freqs, name_prefix="d")
    p_set = register_msl_plane_probes(sim, port_index=1, freqs=freqs, name_prefix="p")
    return sim, trace_y_hi, d_set, p_set


def build_stub_occ(grid, dx, trace_y_hi, L_stub):
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    LX = L_LINE + 2 * PORT_MARGIN
    stub_x_centre = LX / 2.0
    z_patch = H_SUB + 0.5 * dx
    x_centres = (np.arange(nx) - pad_x + 0.5) * dx
    y_centres = (np.arange(ny) - pad_y + 0.5) * dx
    z_centres = (np.arange(nz) - pad_z + 0.5) * dx
    in_x = ((x_centres >= stub_x_centre - W_TRACE / 2) &
            (x_centres <= stub_x_centre + W_TRACE / 2)).astype(np.float32)
    in_z = (np.abs(z_centres - z_patch) <= 0.5 * dx).astype(np.float32)
    in_x_j = jnp.asarray(in_x); in_z_j = jnp.asarray(in_z)
    sigmoid_beta = dx * 0.7
    y_far = jnp.asarray(y_centres - trace_y_hi, dtype=jnp.float32)
    sig_low = jax.nn.sigmoid(y_far / sigmoid_beta)
    sig_high = jax.nn.sigmoid((L_stub - y_far) / sigmoid_beta)
    sig_y = sig_low * sig_high
    return (in_x_j[:, None, None] * sig_y[None, :, None]
            * in_z_j[None, None, :]).astype(jnp.float32)


def diagnostic(dx_um: float, n_sp: int, label: str):
    import math
    dx = dx_um * 1e-6
    print(f"\n{'='*70}")
    print(f"  [{label}]  dx = {dx_um:.0f} µm   n_probe_spacing = {n_sp}")
    print(f"  h_sub/dx = {H_SUB/dx:.4f}  (integer-aligned if integer)")
    print(f"  physical Δ = {n_sp * dx * 1e6:.1f} µm")
    print(f"{'='*70}", flush=True)
    freqs = jnp.asarray([F_TARGET], dtype=jnp.float32)
    sim, trace_y_hi, d_set, p_set = build_sim(dx, n_sp, freqs)
    grid = sim._build_grid()
    print(f"  grid {grid.shape}, n_cells = {np.prod(grid.shape):,}", flush=True)
    period = 1.0 / float(sim._freq_max)
    n_steps_raw = int(math.ceil(10.0 * period / float(grid.dt)))
    K = max(8, int(math.isqrt(n_steps_raw)))
    n_steps_use = ((n_steps_raw + K - 1) // K) * K
    print(f"  n_steps = {n_steps_use} ({K} segments × {n_steps_use//K} steps)",
          flush=True)
    L_scan = np.array([4.0, 6.0, 7.0, 8.0, 10.0, 12.0]) * 1e-3
    print(f"  scan L_stub = {(L_scan*1e3).tolist()} mm", flush=True)
    for L in L_scan:
        t0 = time.time()
        occ = build_stub_occ(grid, dx, trace_y_hi, jnp.asarray(L, dtype=jnp.float32))
        fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps_use,
                         checkpoint_segments=K, skip_preflight=True)
        s11, s21 = extract_msl_s_params_jax_plane(fr, d_set, p_set)
        s11v = float(jnp.abs(s11[0]))
        s21v = float(jnp.abs(s21[0]))
        s21sq = s21v ** 2
        flag = "❌" if s21sq > 1.0 else "✅"
        print(f"    L = {L*1e3:5.2f} mm   |S11| = {s11v:.3f}   "
              f"|S21| = {s21v:.3f}   |S21|² = {s21sq:.4f}  {flag}  "
              f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    print("Y2 plane-lane dx-fragility diagnostic")
    print("Hypothesis A: substrate-cell alignment (h_sub/dx near-integer ⇒ pass)")
    print("Hypothesis D: physical Δ shrink at smaller dx (n_probe_spacing↑ ⇒ pass)")
    diagnostic(127.0, 3, "control / dx=127 / aligned")
    diagnostic(80.0,  3, "broken / dx=80 / misaligned")
    diagnostic(85.0,  3, "A test / dx=85 / near-aligned (3 cells)")
    diagnostic(64.0,  3, "A test / dx=64 / near-aligned (4 cells)")
    diagnostic(80.0,  5, "D test / dx=80 / n_sp=5 (Δ ≈ 400 µm matches dx=127 nat.)")

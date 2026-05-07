"""Phase 4 hypothesis F test — sigmoid PEC mask vs hard PEC Box.

Hypothesis F: Y2 demo's `pec_occupancy_override` (sigmoid density mask)
interacts with the Laplace-mode source distribution differently from a
hard-PEC Box.  Imperative cross-solver `compute_msl_s_matrix` always
runs with a hard Box stub and returns physical |S21| (-43.7 dB notch
at dx=80, -31.9 dB at dx=127); plane lane uses the sigmoid override
and gives |S21|² > 1.  Test:

  v2a — sigmoid PEC mask (current Y2 demo, control)
  v2b — hard PEC Box stub, no override (pure Box geometry)

Both fed through the same `extract_msl_s_params_jax_plane` extractor,
same plane DFT probes, same forward call.  If v2b returns physical
|S21|² ≤ 1 across all L while v2a stays > 1, hypothesis F is confirmed
and the Y2 demo's AD pipeline (which needs the sigmoid override for
gradient flow) needs a re-architecture for plane-lane accuracy.
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


import os
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX = float(os.environ.get("RFX_TEST_DX_UM", "127.0")) * 1e-6
L_LINE = 30e-3
PORT_MARGIN = 1.6e-3
F_MAX = 9e9
F_TARGET = 6e9


def build_thru_sim(freqs, *, with_hard_stub_L=None):
    """Build sim.  If `with_hard_stub_L` is set, add a hard PEC Box
    stub of that length at LX/2; otherwise build the bare thru-line."""
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
    if with_hard_stub_L is not None:
        stub_xc = LX / 2
        sim.add(Box((stub_xc - W_TRACE / 2, trace_y_hi, H_SUB),
                    (stub_xc + W_TRACE / 2,
                     trace_y_hi + with_hard_stub_L, H_SUB + DX)),
                material="pec")
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


def build_sigmoid_occ(grid, trace_y_hi, L_stub):
    LX = L_LINE + 2 * PORT_MARGIN
    # Allow override via env to compare β regimes (Phase 4 hypothesis F'):
    #   default = substrate-anchored floor (≈76 µm at dx=80 µm)
    #   small β (e.g. 5 µm) approaches a binary PEC mask
    SIGMOID_BETA = float(os.environ.get(
        "RFX_SIGMOID_BETA_M", str(max(DX * 0.7, 0.3 * H_SUB))
    ))
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


def run_brute_scan(label, build_fn, freqs, L_scan):
    print(f"\n{'='*70}\n  [{label}]\n{'='*70}", flush=True)
    n_steps = None
    K = None
    out = []
    for L in L_scan:
        t0 = time.time()
        sim, trace_y_hi, d_set, p_set = build_fn(freqs, L)
        grid = sim._build_grid()
        if n_steps is None:
            period = 1.0 / float(sim._freq_max)
            n_steps_raw = int(math.ceil(10.0 * period / float(grid.dt)))
            K = max(8, int(math.isqrt(n_steps_raw)))
            n_steps = ((n_steps_raw + K - 1) // K) * K
            print(f"  grid {grid.shape}, n_steps={n_steps} ({K} segs)",
                  flush=True)
        if label.startswith("v2a"):
            occ = build_sigmoid_occ(grid, trace_y_hi,
                                    jnp.asarray(L, dtype=jnp.float32))
            fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                             checkpoint_segments=K, skip_preflight=True)
        else:
            fr = sim.forward(n_steps=n_steps, checkpoint_segments=K,
                             skip_preflight=True)
        s11, s21 = extract_msl_s_params_jax_plane(fr, d_set, p_set)
        s11v = float(jnp.abs(s11[0]))
        s21v = float(jnp.abs(s21[0]))
        flag = "❌" if s21v ** 2 > 1.0 else "✅"
        print(f"    L={L*1e3:5.2f}mm  |S11|={s11v:.3f}  "
              f"|S21|={s21v:.3f}  |S21|²={s21v**2:.4f}  {flag}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        out.append((L, s11v, s21v))
    return out


def main():
    freqs = jnp.asarray([F_TARGET], dtype=jnp.float32)
    L_scan = np.array([4.0, 6.0, 7.0, 8.0, 10.0, 12.0]) * 1e-3
    sb = float(os.environ.get("RFX_SIGMOID_BETA_M",
                               str(max(DX * 0.7, 0.3 * H_SUB))))
    print("Phase 4 hypothesis F test — sigmoid override vs hard PEC Box stub")
    print(f"dx={DX*1e6:.0f}µm, F_TARGET={F_TARGET/1e9:.1f}GHz, NUM_PERIODS=10")
    print(f"SIGMOID_BETA = {sb*1e6:.2f} µm  ({sb/DX:.3f} × dx)")

    def v2a_build(freqs, L_stub):
        # Sigmoid PEC mask path — current Y2 demo style
        return build_thru_sim(freqs, with_hard_stub_L=None)

    def v2b_build(freqs, L_stub):
        # Hard-PEC Box stub built per L
        return build_thru_sim(freqs, with_hard_stub_L=float(L_stub))

    print("\n[v2a] sigmoid PEC mask via pec_occupancy_override "
          "(Y2 demo path)")
    a = run_brute_scan("v2a / sigmoid mask", v2a_build, freqs, L_scan)

    print("\n[v2b] hard PEC Box stub, no override (forward-only)")
    b = run_brute_scan("v2b / hard PEC Box", v2b_build, freqs, L_scan)

    print("\n" + "=" * 70)
    print("  Summary — v2a (sigmoid) vs v2b (hard Box) at dx=127µm")
    print("=" * 70)
    print(f"  {'L (mm)':>8}  {'v2a |S21|':>10}  {'v2a |S21|²':>10}  "
          f"{'v2b |S21|':>10}  {'v2b |S21|²':>10}  {'verdict'}")
    for (L, _, s21a), (_, _, s21b) in zip(a, b):
        a2 = s21a ** 2; b2 = s21b ** 2
        verdict = "F supported" if (a2 > 1.0 and b2 <= 1.0) else (
            "F rejected" if (a2 > 1.0 and b2 > 1.0) else "ambiguous"
        )
        print(f"  {L*1e3:8.2f}  {s21a:10.3f}  {a2:10.4f}  "
              f"{s21b:10.3f}  {b2:10.4f}  {verdict}")


if __name__ == "__main__":
    main()

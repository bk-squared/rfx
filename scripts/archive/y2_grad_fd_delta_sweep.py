"""Y2 FD step-size sweep at L=9.5mm.

Localizer (run #665) showed the wrong-sign AD gradient is already
present at |v1_driven|² (DFT'd voltage at one probe plane) — i.e.
inside `sim.forward(pec_occupancy_override=occ)`, before any extractor.

Two candidate causes:
  (1) Sub-β wiggle from sigmoid edge sliding across discrete cells.
      AD computes local gradient (wiggle-dominated, arbitrary sign);
      FD with δ ≫ β averages out wiggle (captures global trend).
  (2) Genuine sign bug in `apply_pec_occupancy` reverse pass.

This script runs FD with δ ∈ {1, 10, 50, 200, 500} µm and AD at L=9.5mm.
If FD sign flips between small and large δ, hypothesis (1) is correct
and AD is locally fine.  If FD stays positive at all scales but AD
stays negative, there's a real sign bug to fix in `pec.py`.

Cost function: |v1_driven|² (simplest meaningful quantity that
exhibits the bug per the localizer).
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


def main():
    freqs = jnp.asarray([F_TARGET], dtype=jnp.float32)
    sim, trace_y_hi, d_set, p_set = build_sim(freqs)
    grid = sim._build_grid()
    period = 1.0 / float(sim._freq_max)
    n_steps_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
    K = max(8, int(math.isqrt(n_steps_raw)))
    n_steps = ((n_steps_raw + K - 1) // K) * K
    print(f"dx={DX*1e6:.0f}µm, β={SIGMOID_BETA*1e6:.1f}µm, "
          f"n_steps={n_steps} ({K} segs)")

    def cost(L_stub):
        occ = build_stub_occ(grid, trace_y_hi, L_stub)
        fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                         checkpoint_segments=K, skip_preflight=True)
        v1d = _v_from_plane(fr, d_set.ez1_name, d_set)[0]
        return jnp.abs(v1d) ** 2

    L = 9.5e-3
    L_jax = jnp.asarray(L, dtype=jnp.float32)

    print("\n  FD step-size sweep at L=9.5mm")
    print(f"  β = {SIGMOID_BETA*1e6:.1f} µm,  dx = {DX*1e6:.0f} µm")
    print("  " + "=" * 68)
    print(f"  {'δL (µm)':>9}  {'δL/β':>6}  {'δL/dx':>6}  "
          f"{'FD grad':>14}  {'sign':>5}")

    deltas_um = [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0]
    for dum in deltas_um:
        delta = dum * 1e-6
        t0 = time.time()
        c_lo = float(cost(jnp.asarray(L - delta, dtype=jnp.float32)))
        c_hi = float(cost(jnp.asarray(L + delta, dtype=jnp.float32)))
        fd = (c_hi - c_lo) / (2.0 * delta)
        sign = "+" if fd > 0 else "-"
        print(f"  {dum:9.1f}  {dum/(SIGMOID_BETA*1e6):6.2f}  "
              f"{dum/(DX*1e6):6.2f}  {fd:+14.4e}  {sign:>5}  "
              f"({time.time()-t0:.1f}s)")

    print("\n  AD gradient at L=9.5mm")
    t0 = time.time()
    c_ad, ad = jax.value_and_grad(cost)(L_jax)
    print(f"  AD: cost={float(c_ad):.4e}  grad={float(ad):+.4e}  "
          f"sign={'+'if float(ad)>0 else '-'}  "
          f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

"""Single-forward Kottke path debug — capture intermediates and find NaN."""
from __future__ import annotations

import math
import os
import sys

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


def main():
    freqs = jnp.asarray([F_TARGET], dtype=jnp.float32)
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

    grid = sim._build_grid()
    period = 1.0 / float(sim._freq_max)
    n_steps_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
    K = max(8, int(math.isqrt(n_steps_raw)))
    n_steps = ((n_steps_raw + K - 1) // K) * K
    print(f"grid {grid.shape}, n_steps={n_steps} ({K} segs)", flush=True)

    # Sigmoid stub at L=7mm
    L_stub = 7.0e-3
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
    occ = (jnp.asarray(in_x)[:, None, None] * sig_y[None, :, None]
           * jnp.asarray(in_z)[None, None, :]).astype(jnp.float32)
    print(f"occ stats: min={float(jnp.min(occ)):.3e} "
          f"max={float(jnp.max(occ)):.3e} "
          f"mean={float(jnp.mean(occ)):.3e}", flush=True)

    # Single forward
    print(f"\nForward: KOTTKE={os.environ.get('RFX_PEC_OCC_KOTTKE', '0')}, "
          f"DEBUG={os.environ.get('RFX_PEC_OCC_KOTTKE_DEBUG', '0')}", flush=True)
    fr = sim.forward(
        pec_occupancy_override=occ,
        n_steps=n_steps,
        checkpoint_segments=K,
        skip_preflight=True,
    )
    # Extract |s21|
    from rfx.probes.msl_wave_decomp import extract_msl_s_params_jax_plane
    _, s21 = extract_msl_s_params_jax_plane(fr, d_set, p_set)
    s21_v = float(jnp.abs(s21[0]))
    print(f"\n|s21|={s21_v:.4e}, |s21|²={s21_v**2:.4e}, "
          f"NaN: {math.isnan(s21_v)}", flush=True)


if __name__ == "__main__":
    main()

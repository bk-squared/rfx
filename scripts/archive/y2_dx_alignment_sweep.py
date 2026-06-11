"""Y2 fine-grain dx sweep — localize the dx=80µm alignment artifact.

Run #563 found that on the σ-loading-reverted branch:
  - dx=127: Box ≡ binary-occ (argmin agree at L=7mm)
  - dx=80:  Box gives argmin L=7mm, |S21|² physical;
            binary-occ gives flat |S21|² ≈ 1.22 across ALL L (UNPHYSICAL)
  - dx=64:  Box ≡ binary-occ (argmin agree at L=7mm)

So the override-path bug is NOT a generic fine-mesh issue — it surfaces
specifically at dx=80µm.  Hypothesis: substrate height H_SUB=254µm
divides cleanly at dx=127 (2 cells) and dx=64 (3.97≈4 cells), but at
dx=80 it gives 3.175 cells (substrate-air boundary in the *middle* of
a Yee cell).  The override path may be sensitive to which cell the
stub lands in.

This script localizes the bug.  For each dx ∈ {70, 75, 80, 85, 90}µm:
  - Build Box-PEC stub at L=7mm — record |S21|² at f_target
  - Build same geometry but with binary occupancy override at L=7mm —
    record |S21|² at f_target
  - Print both + cell-alignment info (substrate cells, trace cells)

If the override path is broken at dx ∈ {78, 79, 80, 81, 82} only,
it's a Yee-alignment singularity and the fix is a guardrail.  If it's
broken broadly, it's a deeper override-path issue.
"""
from __future__ import annotations

import math
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
NUM_PERIODS = 10
L_TEST = 7.0e-3  # where the notch is


def build_sim(dx, freqs, *, with_hard_stub_L=None):
    LX = L_LINE + 2 * PORT_MARGIN
    L_STUB_MAX = 14e-3
    LY = W_TRACE + 4 * (2 * H_SUB + 8 * dx) + L_STUB_MAX
    LZ = H_SUB + 1.0e-3
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = (2 * H_SUB + 8 * dx) + W_TRACE / 2
    trace_y_hi = y_trace + W_TRACE / 2
    sim.add(Box((0, y_trace - W_TRACE / 2, H_SUB),
                (LX, y_trace + W_TRACE / 2, H_SUB + dx)), material="pec")
    if with_hard_stub_L is not None:
        stub_xc = LX / 2
        sim.add(Box((stub_xc - W_TRACE / 2, trace_y_hi, H_SUB),
                    (stub_xc + W_TRACE / 2,
                     trace_y_hi + with_hard_stub_L, H_SUB + dx)),
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


def build_binary_occ(grid, dx, trace_y_hi, L_stub):
    LX = L_LINE + 2 * PORT_MARGIN
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    stub_xc = LX / 2
    z_patch = H_SUB + 0.5 * dx
    x_centres = (np.arange(nx) - pad_x + 0.5) * dx
    y_centres = (np.arange(ny) - pad_y + 0.5) * dx
    z_centres = (np.arange(nz) - pad_z + 0.5) * dx
    in_x = ((x_centres >= stub_xc - W_TRACE / 2) &
            (x_centres <= stub_xc + W_TRACE / 2))
    in_z = (np.abs(z_centres - z_patch) <= 0.5 * dx)
    in_y = ((y_centres >= trace_y_hi) &
            (y_centres <= trace_y_hi + L_stub))
    occ = (in_x[:, None, None] & in_y[None, :, None] &
           in_z[None, None, :]).astype(np.float32)
    return jnp.asarray(occ)


def run_one(dx, label, freqs, n_steps_cache):
    """Single forward + extract for one (dx, path) at L=7mm."""
    t0 = time.time()
    if label == "Box":
        sim, _, d_set, p_set = build_sim(dx, freqs,
                                          with_hard_stub_L=L_TEST)
    else:
        sim, trace_y_hi, d_set, p_set = build_sim(dx, freqs,
                                                    with_hard_stub_L=None)
    grid = sim._build_grid()
    if dx not in n_steps_cache:
        period = 1.0 / float(sim._freq_max)
        n_steps_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
        K = max(8, int(math.isqrt(n_steps_raw)))
        n_steps = ((n_steps_raw + K - 1) // K) * K
        n_steps_cache[dx] = (n_steps, K)
    n_steps, K = n_steps_cache[dx]
    if label == "Binary":
        occ = build_binary_occ(grid, dx, trace_y_hi,
                                jnp.asarray(L_TEST, dtype=jnp.float32))
        fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                         checkpoint_segments=K, skip_preflight=True)
    else:
        fr = sim.forward(n_steps=n_steps, checkpoint_segments=K,
                         skip_preflight=True)
    s11, s21 = extract_msl_s_params_jax_plane(fr, d_set, p_set)
    s11v = float(jnp.abs(s11[0]))
    s21v = float(jnp.abs(s21[0]))
    s21_sq = s21v ** 2
    s21_db = 10 * math.log10(max(s21_sq, 1e-12))
    return s11v, s21v, s21_sq, s21_db, time.time() - t0, grid.shape


def main():
    freqs = jnp.asarray([F_TARGET], dtype=jnp.float32)
    dx_list_um = [70.0, 75.0, 80.0, 82.0, 85.0, 90.0, 95.0]
    print("Y2 fine-grain dx alignment sweep")
    print(f"  L_stub fixed at {L_TEST*1e3:.1f} mm (notch location)")
    print(f"  H_SUB = {H_SUB*1e6:.0f} µm  →  substrate cells = H_SUB / dx")
    print()
    print(f"  {'dx (µm)':>8}  {'sub cells':>10}  "
          f"{'|S21|²_box':>11}  {'box dB':>8}  "
          f"{'|S21|²_bin':>11}  {'bin dB':>8}  "
          f"{'verdict':>10}")
    n_steps_cache = {}
    for dx_um in dx_list_um:
        dx = dx_um * 1e-6
        sub_cells = H_SUB / dx
        try:
            box = run_one(dx, "Box", freqs, n_steps_cache)
            bin_ = run_one(dx, "Binary", freqs, n_steps_cache)
        except Exception as e:
            print(f"  {dx_um:8.1f}  ERROR: {e}")
            continue
        s_box = box[2]; s_bin = bin_[2]
        box_db = box[3]; bin_db = bin_[3]
        ratio = s_bin / max(s_box, 1e-12)
        if abs(math.log10(max(ratio, 1e-12))) < 0.3:
            verdict = "agree"
        elif s_bin > 1.05:
            verdict = "BIN>>1"
        else:
            verdict = f"x{ratio:.1f}"
        print(f"  {dx_um:8.1f}  {sub_cells:10.3f}  "
              f"{s_box:11.4f}  {box_db:+8.1f}  "
              f"{s_bin:11.4f}  {bin_db:+8.1f}  "
              f"{verdict:>10}")


if __name__ == "__main__":
    main()

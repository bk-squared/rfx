"""Y2 diagnostic — does sigmoid+σ override match a hard PEC Box?

Run #541 surfaced a 30% ε_eff disagreement between the differentiable
JAX path (sigmoid PEC mask via ``pec_occupancy_override``) and the
imperative ``compute_msl_s_matrix`` (hard PEC Box stub).  The σ-loading
fix (commit 8d65786) folds ``occ × 1e10`` into ``materials.sigma`` —
this is *not* a hard short.  Skin depth at f=6 GHz with σ=1e10 is
δ ≈ 36 µm, ~28 % of dx=127 µm.  That's enough penetration to shift the
effective electrical length of the stub.

This diagnostic answers:

  1. At the same L, does sigmoid+σ produce the same |S21|² as Box PEC?
     Both run through the *same* JAX plane extractor.  Any gap is
     pure forward-physics.

  2. Where is the actual notch in f for L=6 mm and L=10 mm with Box
     PEC?  Via ``compute_msl_s_matrix`` freq sweep.  Pins down what
     ε_eff the FDTD really sees on this mesh, so we can compare to the
     ~3.13 implied by the prior run #541 imperative result and the
     ~4.04 implied by the JAX brute scan minimum at L=6 mm.

Outputs:
  - L sweep table (Box vs sigmoid, |S21|² at f_target=6 GHz)
  - argmin(L) for each path — should agree if sigmoid ≡ Box
  - freq sweep at L=6 mm and L=10 mm — locates physical notch
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
DX = float(os.environ.get("RFX_TEST_DX_UM", "127.0")) * 1e-6
L_LINE = 30e-3
PORT_MARGIN = 1.6e-3
F_MAX = 9e9
F_TARGET = 6e9
NUM_PERIODS = 10
SIGMOID_BETA = float(os.environ.get(
    "RFX_SIGMOID_BETA_M", str(max(DX * 0.25, 0.05 * H_SUB))
))


def build_sim(freqs, *, with_hard_stub_L=None):
    """Build sim. If `with_hard_stub_L` is None, geometry has no stub
    (ready for sigmoid override).  Otherwise add a hard PEC Box stub."""
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


def run_L_sweep(label, build_fn, freqs, L_scan):
    """Run forward + JAX plane extract for each L; return |S21|²."""
    print(f"\n{'='*70}\n  [{label}]  (NUM_PERIODS={NUM_PERIODS})\n{'='*70}",
          flush=True)
    n_steps = None
    K = None
    out = []
    for L in L_scan:
        t0 = time.time()
        sim, trace_y_hi, d_set, p_set = build_fn(freqs, L)
        grid = sim._build_grid()
        if n_steps is None:
            period = 1.0 / float(sim._freq_max)
            n_steps_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
            K = max(8, int(math.isqrt(n_steps_raw)))
            n_steps = ((n_steps_raw + K - 1) // K) * K
            print(f"  grid {grid.shape}, n_steps={n_steps} ({K} segs)",
                  flush=True)
        if label.startswith("sigmoid"):
            occ = build_sigmoid_occ(grid, trace_y_hi,
                                    jnp.asarray(L, dtype=jnp.float32))
            fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                             checkpoint_segments=K, skip_preflight=True)
        elif label.startswith("binary"):
            occ_smooth = build_sigmoid_occ(
                grid, trace_y_hi, jnp.asarray(L, dtype=jnp.float32))
            occ = jnp.where(occ_smooth > 0.5, 1.0, 0.0).astype(jnp.float32)
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
        print(f"    L={L*1e3:5.2f}mm  |S11|={s11v:.3f}  "
              f"|S21|²={s21_sq:.4f}  ({s21_db:+6.1f} dB)  "
              f"({time.time()-t0:.1f}s)", flush=True)
        out.append((L, s11v, s21v, s21_sq, s21_db))
    return out


def run_freq_sweep_box(L, n_freqs=80):
    """Use compute_msl_s_matrix to find where the notch ACTUALLY is at L."""
    print(f"\n{'='*70}\n  [Box freq sweep at L={L*1e3:.1f}mm]\n{'='*70}",
          flush=True)
    f_grid = jnp.linspace(F_MAX / 10, F_MAX, n_freqs)
    sim, _, _, _ = build_sim(f_grid, with_hard_stub_L=L)
    object.__setattr__(sim._msl_ports[1], "excite", True)  # both-port-driven
    t0 = time.time()
    res = sim.compute_msl_s_matrix(n_freqs=n_freqs, num_periods=20.0)
    s21 = np.asarray(res.S[1, 0, :])
    s21_sq = np.abs(s21) ** 2
    i_min = int(np.argmin(s21_sq))
    f_min = float(np.asarray(f_grid)[i_min])
    s21_min_db = 10 * math.log10(max(float(s21_sq[i_min]), 1e-12))
    s21_at_target_idx = int(np.argmin(np.abs(np.asarray(f_grid) - F_TARGET)))
    s21_at_target_sq = float(s21_sq[s21_at_target_idx])
    s21_at_target_db = 10 * math.log10(max(s21_at_target_sq, 1e-12))
    print(f"  notch at f={f_min/1e9:.3f} GHz  ({s21_min_db:+.1f} dB)  "
          f"→ implied ε_eff = {(3e8/(4*L*f_min))**2:.3f}", flush=True)
    print(f"  |S21|² at f_target={F_TARGET/1e9:.2f} GHz: {s21_at_target_sq:.4f} "
          f"({s21_at_target_db:+6.1f} dB)  ({time.time()-t0:.1f}s)", flush=True)
    return f_min, s21_at_target_sq


def main():
    freqs = jnp.asarray([F_TARGET], dtype=jnp.float32)
    L_scan = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]) * 1e-3
    print("Y2 extractor-agreement diagnostic")
    print(f"  dx={DX*1e6:.0f}µm, F_TARGET={F_TARGET/1e9:.1f}GHz, "
          f"NUM_PERIODS={NUM_PERIODS}")
    print(f"  SIGMOID_BETA = {SIGMOID_BETA*1e6:.2f} µm "
          f"({SIGMOID_BETA/DX:.3f} × dx)")
    print(f"  σ-PEC = 1e10 S/m  →  skin depth δ@6GHz ≈ "
          f"{1.0/math.sqrt(math.pi*F_TARGET*4*math.pi*1e-7*1e10)*1e6:.1f} µm")
    print(f"            (δ/dx = "
          f"{1.0/math.sqrt(math.pi*F_TARGET*4*math.pi*1e-7*1e10)/DX:.3f})")

    # Path α: Box PEC + JAX extractor
    a = run_L_sweep(
        "α / Box PEC + JAX extractor",
        lambda fr, L: build_sim(fr, with_hard_stub_L=float(L)),
        freqs, L_scan,
    )

    # Path β: sigmoid+σ override + JAX extractor
    b = run_L_sweep(
        "sigmoid+σ override + JAX extractor",
        lambda fr, L: build_sim(fr, with_hard_stub_L=None),
        freqs, L_scan,
    )

    # Path γ: binary-occupancy override (sigmoid > 0.5) + JAX extractor
    # Tests whether the 1mm shift comes from the sigmoid edge (half-PEC at
    # occ=0.5) or from the override mechanism itself (σ-loading + H-damp).
    c = run_L_sweep(
        "binary-occ override + JAX extractor",
        lambda fr, L: build_sim(fr, with_hard_stub_L=None),
        freqs, L_scan,
    )

    # Summary table
    print("\n" + "=" * 80)
    print("  L sweep — Box (α) vs sigmoid+σ (β) vs binary-occ (γ), "
          "same JAX extractor")
    print("=" * 80)
    print(f"  {'L (mm)':>7}  {'|S21|²_α':>9}  {'α dB':>7}  "
          f"{'|S21|²_β':>9}  {'β dB':>7}  "
          f"{'|S21|²_γ':>9}  {'γ dB':>7}")
    for (L, _, _, sa_sq, sa_db), (_, _, _, sb_sq, sb_db), \
            (_, _, _, sc_sq, sc_db) in zip(a, b, c):
        print(f"  {L*1e3:7.2f}  {sa_sq:9.4f}  {sa_db:+7.1f}  "
              f"{sb_sq:9.4f}  {sb_db:+7.1f}  "
              f"{sc_sq:9.4f}  {sc_db:+7.1f}")

    # Find argmin in each
    a_min_i = int(np.argmin([row[3] for row in a]))
    b_min_i = int(np.argmin([row[3] for row in b]))
    c_min_i = int(np.argmin([row[3] for row in c]))
    print(f"\n  argmin α (Box):       L = {a[a_min_i][0]*1e3:.2f} mm  "
          f"|S21|² = {a[a_min_i][3]:.4f}")
    print(f"  argmin β (sigmoid):   L = {b[b_min_i][0]*1e3:.2f} mm  "
          f"|S21|² = {b[b_min_i][3]:.4f}")
    print(f"  argmin γ (binary):    L = {c[c_min_i][0]*1e3:.2f} mm  "
          f"|S21|² = {c[c_min_i][3]:.4f}")
    print(f"\n  Δ argmin (β − α) = "
          f"{(b[b_min_i][0] - a[a_min_i][0])*1e3:+.2f} mm  "
          "(sigmoid vs Box)")
    print(f"  Δ argmin (γ − α) = "
          f"{(c[c_min_i][0] - a[a_min_i][0])*1e3:+.2f} mm  "
          "(binary vs Box)")
    print(f"  Δ argmin (γ − β) = "
          f"{(c[c_min_i][0] - b[b_min_i][0])*1e3:+.2f} mm  "
          "(binary vs sigmoid → sigmoid-edge contribution)")

    # Freq sweep at L=6mm and L=10mm with Box PEC
    print("\n" + "=" * 80)
    print("  Freq sweep — Box PEC, where is the actual notch?")
    print("=" * 80)
    for L in (6e-3, 10e-3):
        run_freq_sweep_box(L)


if __name__ == "__main__":
    main()

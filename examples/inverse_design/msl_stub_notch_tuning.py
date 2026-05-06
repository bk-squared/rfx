"""Inverse design — microstrip-line open-stub notch frequency tuning
via density-based PEC mask + JAX-traceable post-forward S-extraction.

Microwave-engineering inverse-design demo on a 2-port MSL filter:
tune the open-stub length so the notch lands at a chosen design
frequency.  Stub length is naturally a *discrete* variable (an integer
number of Yee cells along ``y``); we reformulate it as a continuous
sigmoid PEC density mask via :meth:`Simulation.forward(pec_occupancy_override=…)`
and optimise the half-length scalar with Adam through ``jax.grad``.

Design variable: scalar ``L_stub ∈ [4, 12] mm`` (mapped from a real
latent via sigmoid).  Sigmoid mask sharpness ≈ 0.7 dx — soft enough
for AD, hard enough for the open-circuit boundary to register.

Cost: ``|S21(f_target)|²`` extracted *post-* :meth:`Simulation.forward`
by :func:`rfx.probes.msl_wave_decomp.extract_msl_s_params_jax` (a
JAX-friendly port of the validated 3-probe quadratic recurrence in
:mod:`rfx.sources.msl_port`, exposed because the imperative
:meth:`compute_msl_s_matrix` uses ``np.asarray`` and is not
``jax.grad``-traceable).

Validation chain (the actual point of this example):
  1. Adam minimises ``|S21_jax(f_target)|²`` with respect to ``L_stub``.
  2. Brute-force scan (same JAX extractor) confirms the optimum.
  3. **Cross-solver gate**: at the Adam-final ``L_opt``, run the
     validated imperative :meth:`Simulation.compute_msl_s_matrix` and
     check that the *imperative* notch is real (deep ≤ -15 dB) and
     near the design target.

What this demo intentionally surfaces (the actual deliverable —
infrastructure gaps for differentiable MSL inverse design):

  * `pec_occupancy_override` reformulation of L_stub *works* end-to-end
    on `Simulation.forward()` — gradients flow finite, Adam descends.
  * Adam lands on a *real* notch (imperative |S21| ≤ -50 dB at L_opt
    on this mesh).
  * **But** the scalar Ez point-probe proxy used by the JAX 3-probe
    extractor under-resolves the quasi-TEM mode integral compared to
    the validated plane-integrated path in
    :meth:`Simulation.compute_msl_s_matrix`.  This biases the JAX
    extractor's notch *frequency* by ~15-20 % at this mesh, so Adam
    converges to a notch that is real but ~15-20 % off the requested
    ``f_target``.  Tightening the demo gates to engineering tolerance
    (≤ 5 %) requires :class:`ForwardResult` to carry plane-DFT-probe
    accumulators (issue surfaced — needs `dft_planes` plumbing on the
    differentiable forward path).
  * `forward(port_s11_freqs=…)` is uniform-mesh-only (issue #72).
  * `compute_msl_s_matrix` is imperative (uses ``np.asarray``); not
    JAX-differentiable.  This is what motivates the new helper
    :mod:`rfx.probes.msl_wave_decomp`.

These four points together motivate the `dft_planes` plumbing and
"plane-integrated JAX S-extractor" follow-up tasks.  Until they land,
this demo is the canonical reproducer of "what's missing for
engineering-grade differentiable MSL inverse design".

Geometry: cv06b-class (uniform dx=127 µm = h_sub/2 = 2 substrate
cells, L_LINE=30 mm).  Long enough for each MSL port's 3-probe
extractor to sit outside the stub-junction standing-wave region
(λ_g/4 reflector-clearance check, enforced by `sim.preflight()` —
see `tests/test_msl_port_preflight.py::test_reflector_clearance_*`).
Earlier 5 mm short-line variant gave |S11|@notch ≈ -7 dB instead
of the physical 0 dB; that bias is closed at this geometry.

Run: ``python examples/inverse_design/msl_stub_notch_tuning.py``
"""

from __future__ import annotations

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.msl_wave_decomp import (
    register_msl_wave_probes,
    extract_msl_s_params_jax,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# ---------------------------------------------------------------------------
# Problem constants — shrunken cv06b
# ---------------------------------------------------------------------------
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX = 127e-6                            # h_sub / 2 → 2 substrate cells
L_LINE = 30.0e-3                       # cv06b-class line length.  Each
                                        # MSL port's V₃ probe must sit
                                        # ≥ λ_g/4 from the stub PEC
                                        # reflector for the imperative
                                        # `compute_msl_s_matrix` 3-probe
                                        # extractor to read |S11|@notch
                                        # cleanly; preflight enforces
                                        # the bound (see
                                        # `_check_msl_port_geometry`,
                                        # `tests/test_msl_port_preflight
                                        # .py::test_reflector_clearance_*`).
PORT_MARGIN = 1.6e-3                   # ≥ cpml_extent (8·dx ≈ 1.02 mm)
                                        # + 2·h_sub safety; preflight
                                        # x-CPML clearance check.
F_MAX = 9e9

L_STUB_MAX = 14.0e-3
L_MIN, L_MAX = 4.0e-3, 12.0e-3
L_INIT = 6.0e-3                        # off-target on the short side
SIGMOID_BETA = DX * 0.7

u = W_TRACE / H_SUB
EPS_EFF = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5

# Default cost target — quarter-wave at 6.0 GHz → L_stub ≈ 7.4 mm,
# inside the design-bound interior.
F_TARGET = 6.0e9
L_TARGET_AN = C0 / (4 * F_TARGET * np.sqrt(EPS_EFF))

# Domain
LX = L_LINE + 2 * PORT_MARGIN
msl_clearance = 2 * (2 * H_SUB + 8 * DX)
LY = W_TRACE + msl_clearance + L_STUB_MAX + 2 * (2 * H_SUB + 8 * DX)
LZ = H_SUB + 1.0e-3


# ---------------------------------------------------------------------------
# Sim builder + sigmoid stub mask
# ---------------------------------------------------------------------------
def build_sim() -> tuple[Simulation, float, float, "MSLWaveProbeSet", "MSLWaveProbeSet"]:
    """Build the through-line geometry (no stub Box) with 2 MSL ports
    + 8 wave probes (4 per port).  Returns the sim, the trace-y centre
    and trace-y top, and the two probe sets for post-forward extraction."""
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")

    y_trace = (2 * H_SUB + 8 * DX) + W_TRACE / 2.0
    trace_y_lo = y_trace - W_TRACE / 2.0
    trace_y_hi = y_trace + W_TRACE / 2.0
    sim.add(Box((0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
            material="pec")

    sim.add_msl_port(position=(PORT_MARGIN, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0)

    z_ez = 0.5 * H_SUB
    z_hy = H_SUB + 0.5 * DX
    d_set = register_msl_wave_probes(
        sim, feed_x=PORT_MARGIN, direction="+x",
        y_centre=y_trace, z_ez=z_ez, z_hy=z_hy,
    )
    p_set = register_msl_wave_probes(
        sim, feed_x=PORT_MARGIN + L_LINE, direction="-x",
        y_centre=y_trace, z_ez=z_ez, z_hy=z_hy,
    )

    # Drive only port 0 — disable port-1 excitation in the underlying
    # MSLPortEntry (frozen dataclass; bypass with object.__setattr__).
    object.__setattr__(sim._msl_ports[1], "excite", False)
    return sim, y_trace, trace_y_hi, d_set, p_set


def build_stub_occ(grid, trace_y_hi: float, L_stub: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid soft-PEC stub mask of commanded length ``L_stub``,
    rooted at ``trace_y_hi`` along +y, in the trace-x footprint."""
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    stub_x_centre = LX / 2.0
    stub_x_lo = stub_x_centre - W_TRACE / 2.0
    stub_x_hi = stub_x_centre + W_TRACE / 2.0
    z_patch = H_SUB + 0.5 * DX

    x_centres = (np.arange(nx) - pad_x + 0.5) * DX
    y_centres = (np.arange(ny) - pad_y + 0.5) * DX
    z_centres = (np.arange(nz) - pad_z + 0.5) * DX

    in_x = ((x_centres >= stub_x_lo) & (x_centres <= stub_x_hi)).astype(np.float32)
    in_z = (np.abs(z_centres - z_patch) <= 0.5 * DX).astype(np.float32)
    in_x_j = jnp.asarray(in_x); in_z_j = jnp.asarray(in_z)
    y_far = jnp.asarray(y_centres - trace_y_hi, dtype=jnp.float32)
    sig_low = jax.nn.sigmoid(y_far / SIGMOID_BETA)
    sig_high = jax.nn.sigmoid((L_stub - y_far) / SIGMOID_BETA)
    sig_y = sig_low * sig_high
    return (in_x_j[:, None, None] * sig_y[None, :, None]
            * in_z_j[None, None, :]).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Cost + Adam
# ---------------------------------------------------------------------------
def main() -> int:
    # Defaults sized for ~12-15 min wall on this dev box.  Adam converges
    # in 2-4 iters because the cost landscape is convex around L_target
    # (single-frequency notch tuning).
    NUM_PERIODS = float(os.environ.get("RFX_Y2B_PERIODS", 10.0))
    N_ITERS = int(os.environ.get("RFX_Y2B_ITERS", 4))
    LR = float(os.environ.get("RFX_Y2B_LR", 0.4))
    N_SCAN = int(os.environ.get("RFX_Y2B_SCAN", 5))

    sim, y_trace, trace_y_hi, d_set, p_set = build_sim()
    grid = sim._build_grid()
    print(f"Grid {grid.shape}  ({np.prod(grid.shape):,d} cells)  dt={float(grid.dt)*1e12:.2f}ps")

    # Preflight: surfaces MSL geometry warnings (lateral clearance, substrate
    # cells, x-CPML, reflector clearance — the last is what protects the
    # 3-probe extractor from sitting in the stub-junction standing-wave
    # region; see `_check_msl_port_geometry` in rfx/api.py).
    pre_msgs = sim.preflight()
    if pre_msgs:
        print("\nPreflight warnings:")
        for m in pre_msgs:
            print(f"  - {m}")
    else:
        print("\nPreflight: clean.")
    print(f"εr={EPS_R}, h_sub={H_SUB*1e6:.0f}µm, W={W_TRACE*1e6:.0f}µm, "
          f"dx={DX*1e6:.0f}µm  ε_eff={EPS_EFF:.3f}")
    print(f"L_stub bounds [{L_MIN*1e3:.1f}, {L_MAX*1e3:.1f}] mm   "
          f"init={L_INIT*1e3:.1f} mm")
    print(f"f_target={F_TARGET/1e9:.2f} GHz   "
          f"analytic L_target={L_TARGET_AN*1e3:.3f} mm")
    print(f"Pipeline: {NUM_PERIODS:.0f} periods × {N_ITERS} Adam iters  (lr={LR})")

    f_target_arr = jnp.asarray([F_TARGET], dtype=jnp.float32)

    def s21_at_f_target(L_stub):
        occ = build_stub_occ(grid, trace_y_hi, L_stub)
        fr = sim.forward(
            pec_occupancy_override=occ,
            num_periods=NUM_PERIODS,
            skip_preflight=True,
        )
        _, s21 = extract_msl_s_params_jax(
            fr.time_series, d_set, p_set,
            dt=float(grid.dt), freqs=f_target_arr,
        )
        return s21[0]

    def cost_from_latent(latent):
        L_stub = L_MIN + (L_MAX - L_MIN) * jax.nn.sigmoid(latent)
        s21 = s21_at_f_target(L_stub)
        return jnp.abs(s21) ** 2

    def latent_from_L(L: float) -> float:
        u = (L - L_MIN) / (L_MAX - L_MIN)
        u = max(min(u, 0.999), 0.001)
        return float(math.log(u / (1.0 - u)))

    # ---- Adam optimisation ----
    print("\n" + "=" * 70)
    print("Adam optimisation")
    print("=" * 70)
    latent = jnp.asarray(latent_from_L(L_INIT), dtype=jnp.float32)
    m_buf = jnp.zeros_like(latent); v_buf = jnp.zeros_like(latent)
    beta1, beta2, ae = 0.9, 0.999, 1e-8
    cost_grad_fn = jax.value_and_grad(cost_from_latent)

    history = {"L": [], "cost": [], "db": []}
    t_total = time.time()
    for it in range(N_ITERS):
        t0 = time.time()
        loss, grad = cost_grad_fn(latent)
        loss_v = float(loss); grad_v = float(grad)
        L_now = float(L_MIN + (L_MAX - L_MIN) * jax.nn.sigmoid(latent))
        db = 10.0 * math.log10(max(loss_v, 1e-12))
        history["L"].append(L_now * 1e3); history["cost"].append(loss_v); history["db"].append(db)
        print(f"  iter {it:3d}  L={L_now*1e3:6.3f}mm  |S21|²={loss_v:.4e}  "
              f"S21={db:+6.1f}dB  grad={grad_v:+.3e}  ({time.time()-t0:.1f}s)")
        m_buf = beta1 * m_buf + (1 - beta1) * grad
        v_buf = beta2 * v_buf + (1 - beta2) * grad ** 2
        m_hat = m_buf / (1 - beta1 ** (it + 1))
        v_hat = v_buf / (1 - beta2 ** (it + 1))
        latent = latent - LR * m_hat / (jnp.sqrt(v_hat) + ae)
    L_opt = float(L_MIN + (L_MAX - L_MIN) * jax.nn.sigmoid(latent))
    cost_opt = history["cost"][-1]
    print(f"\nAdam done in {time.time() - t_total:.1f}s — "
          f"L_opt={L_opt*1e3:.3f} mm,  |S21|²={cost_opt:.4e}")

    # ---- Brute-force scan ----
    print("\n" + "=" * 70)
    print("Reference: brute-force scan via the same JAX extractor")
    print("=" * 70)
    L_scan = np.linspace(L_MIN, L_MAX, N_SCAN)
    scan_costs = np.zeros(N_SCAN)
    t0 = time.time()
    for i, L in enumerate(L_scan):
        s21 = s21_at_f_target(jnp.asarray(L, dtype=jnp.float32))
        scan_costs[i] = float(jnp.abs(s21) ** 2)
        db = 10.0 * math.log10(max(scan_costs[i], 1e-12))
        print(f"  L={L*1e3:6.3f} mm  |S21|²={scan_costs[i]:.4e}  S21={db:+6.1f}dB")
    i_min = int(np.argmin(scan_costs))
    L_ref = float(L_scan[i_min])
    print(f"\nScan done in {time.time()-t0:.1f}s  →  L_ref={L_ref*1e3:.3f} mm")

    # ---- Cross-solver gate: imperative compute_msl_s_matrix at L_opt ----
    print("\n" + "=" * 70)
    print("Cross-solver gate: imperative compute_msl_s_matrix at L_opt")
    print("=" * 70)
    sim_imp, _, trace_y_hi_imp, _, _ = build_sim()
    # add a HARD-PEC stub Box of length L_opt to sim_imp
    stub_x_centre = LX / 2.0
    stub_x_lo = stub_x_centre - W_TRACE / 2.0
    stub_x_hi = stub_x_centre + W_TRACE / 2.0
    sim_imp.add(Box((stub_x_lo, trace_y_hi_imp, H_SUB),
                    (stub_x_hi, trace_y_hi_imp + L_opt, H_SUB + DX)),
                material="pec")
    # restore default both-port-driven for compute_msl_s_matrix
    object.__setattr__(sim_imp._msl_ports[1], "excite", True)
    t0 = time.time()
    res = sim_imp.compute_msl_s_matrix(n_freqs=80, num_periods=20.0)
    t_imp = time.time() - t0
    f_imp = np.asarray(res.freqs)
    s21_imp = np.asarray(res.S[1, 0, :])
    db_imp = 20 * np.log10(np.abs(s21_imp) + 1e-30)
    i_notch = int(np.argmin(db_imp))
    f_notch_imp = float(f_imp[i_notch])
    depth_imp = float(db_imp[i_notch])
    print(f"  imperative S-matrix done in {t_imp:.1f}s")
    print(f"  imperative notch: f={f_notch_imp/1e9:.3f} GHz  "
          f"|S21|={depth_imp:+.1f} dB")
    f_err = abs(f_notch_imp - F_TARGET) / F_TARGET * 100
    print(f"  Δf vs f_target=({F_TARGET/1e9:.2f} GHz): {f_err:.2f} %")

    # ---- Acceptance ----
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    L_err_ref = abs(L_opt - L_ref) / max(L_ref, 1e-12) * 100
    L_err_an = abs(L_opt - L_TARGET_AN) / L_TARGET_AN * 100
    cost_init = history["cost"][0]
    cost_drop_db = 10.0 * math.log10(cost_init / max(cost_opt, 1e-12))
    on_rail = L_opt <= L_MIN * 1.005 or L_opt >= L_MAX * 0.995
    # Gates G1, G4, G5 are tight — they verify the AD pipeline works
    # and that Adam descends to a *real* notch.  Gates G2, G3 are loose
    # (≤ 25 %) because the scalar Ez point-probe proxy used by the
    # JAX extractor under-resolves the quasi-TEM mode integral compared
    # to the validated plane-integrated `compute_msl_s_matrix`, biasing
    # the JAX-extractor notch frequency by ~15-20 % at this mesh.
    # See the docstring header for the full diagnosis and the
    # rfx-known-issues entry it surfaces.
    g1 = cost_drop_db >= 1.0
    g2 = L_err_an <= 25.0
    g3 = f_err <= 25.0
    g4 = depth_imp <= -15.0
    g5 = not on_rail
    print(f"  G1  Adam cost ↓ ≥ 1 dB:                  "
          f"{cost_drop_db:.1f} dB  ({'PASS' if g1 else 'FAIL'})")
    print(f"  G2  L_opt ≈ analytic L_target (≤ 25%):   "
          f"err={L_err_an:.2f}%  ({'PASS' if g2 else 'FAIL'})")
    print(f"  G3  Imperative notch ≈ f_target (≤ 25%): "
          f"err={f_err:.2f}%  ({'PASS' if g3 else 'FAIL'})")
    print(f"  G4  Imperative notch depth ≤ -15 dB:     "
          f"{depth_imp:+.1f} dB  ({'PASS' if g4 else 'FAIL'})")
    print(f"  G5  L_opt strictly interior:             "
          f"{L_opt*1e3:.3f} mm  ({'PASS' if g5 else 'FAIL'})")
    all_ok = g1 and g2 and g3 and g4 and g5
    print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}")
    print(f"  L_opt={L_opt*1e3:.3f}mm  L_ref={L_ref*1e3:.3f}mm  "
          f"L_target(an)={L_TARGET_AN*1e3:.3f}mm")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    iters = np.arange(len(history["db"]))

    ax = axes[0]
    ax.semilogy(L_scan * 1e3, scan_costs, "k-o", lw=1.6, ms=5,
                label="brute-force scan (JAX extr.)")
    ax.plot(history["L"], history["cost"], "b.-", lw=1.2, alpha=0.7,
            label="Adam path")
    ax.axvline(L_TARGET_AN * 1e3, color="g", ls="--", alpha=0.6,
               label=f"L_target_an={L_TARGET_AN*1e3:.2f}mm")
    ax.axvline(L_opt * 1e3, color="r", ls=":", alpha=0.8,
               label=f"L_opt={L_opt*1e3:.2f}mm")
    ax.set_xlabel("L_stub (mm)")
    ax.set_ylabel(f"|S21(f={F_TARGET/1e9:.1f} GHz)|² (JAX extractor)")
    ax.set_title("Single-frequency match cost vs L_stub")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters, history["db"], "b-o", lw=1.4, ms=4)
    ax.set_xlabel("Adam iter"); ax.set_ylabel("|S21(f_target)| dB (JAX)")
    ax.set_title("Adam convergence")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(f_imp / 1e9, db_imp, "k-", lw=1.5,
            label=f"imperative @ L_opt={L_opt*1e3:.2f}mm")
    ax.axvline(F_TARGET / 1e9, color="r", ls=":", alpha=0.8,
               label=f"f_target={F_TARGET/1e9:.2f} GHz")
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("|S21| dB (imperative)")
    ax.set_title("Cross-solver gate — imperative notch at L_opt")
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        f"MSL stub notch tuning — density-PEC reformulation + JAX 3-probe extractor\n"
        f"L_opt={L_opt*1e3:.2f}mm vs L_target_an={L_TARGET_AN*1e3:.2f}mm   "
        f"imperative notch @ {f_notch_imp/1e9:.2f}GHz ({depth_imp:+.1f}dB) — "
        f"{'PASS' if all_ok else 'FAIL'}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "msl_stub_notch_tuning.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nWrote: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

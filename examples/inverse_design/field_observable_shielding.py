"""Inverse-design a shielding stack against worst-case leaked field (#579).

Demonstrates ``rfx.observables``: a normal-incidence TFSF plane wave
illuminates a multilayer dielectric stack; two internal DFT-plane probes
sit BEHIND the stack at two different depths ("near" and "far" leakage
monitors). ``field_softmax`` pools BOTH planes (all space + all monitored
frequencies) into one smooth worst-case-leakage scalar, and
``jax.grad`` descent shrinks it by reshaping the stack's per-layer
permittivity.

Why two planes, not one multi-component probe: the original ask (a
``components=`` argument selecting several field components per probe) is
NOT how ``add_dft_plane_probe`` works in rfx today -- each plane probe is
single-component (issue #579 design doc). The register-N-planes-then-pool
pattern is the workaround: register one plane per physical monitor
location (or, for a true multi-component leakage measure, one plane per
component you care about) and pool them together:

    objective = field_softmax(["leak_near", "leak_far"], beta=BETA)
    loss_fn = lambda p: objective(sim.forward(eps_override=_build_eps(p)))

That composition IS the whole point of this module: an accessor/objective
factory that plugs directly into ``sim.forward(eps_override=...)`` +
``jax.grad``, the same way ``rfx.optimize_objectives`` factories plug into
``result.time_series`` / ``result.s_params``.

Design-region safety note: ``forward()``'s TFSF vacuum-boundary preflight
check runs on the CONCRETE (baseline, un-perturbed) config, not on
``eps_override`` -- it cannot see where the design region will end up once
``p`` moves. This script keeps the design region's INDEX RANGE fixed for
the whole run (only per-cell permittivity VALUES are trainable), and
asserts once, before optimizing, that fixed range sits strictly inside the
TFSF total-field box computed with the same formula
``rfx/sources/tfsf.py:255-256`` uses internally.

Run: python examples/inverse_design/field_observable_shielding.py
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation
from rfx.observables import field_softmax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Geometry. Normal-incidence TFSF forces periodic y/z + CPML-x internally
# (rfx/runners/uniform.py, "all other TFSF" branch) regardless of the
# boundary spec passed at construction, so a thin transverse domain (the
# same 1D-equivalent trick examples/inverse_design/multilayer_ar_coating.py
# uses) is valid here too -- it keeps this CPU-fast.
# ---------------------------------------------------------------------------
DX = 2.5e-3          # m
NX_INTERIOR = 60      # requested-domain cell count in x (CPML pads OUTSIDE this)
NY_INTERIOR = 6
NZ_INTERIOR = 6
CPML_LAYERS = 8
TFSF_MARGIN = 3        # add_tfsf_source(margin=) default, named explicitly below
F0 = 6.0e9
BANDWIDTH = 0.5
FREQ_MAX = 8.0e9

# Shield (design region): a 7-layer dielectric stack, physical x-range.
SHIELD_X_LO_M = 0.050
SHIELD_X_HI_M = 0.0675   # 7 layers * DX
N_LAYERS = 7

# Leakage monitors, both BEHIND the shield, two depths.
LEAK_NEAR_X_M = 0.080
LEAK_FAR_X_M = 0.130

EPS_MIN, EPS_MAX = 1.0, 6.0
# field_softmax sharpness: BETA is now DIMENSIONLESS (rfx.observables.
# field_softmax auto-scales internally as beta_eff = beta /
# stop_gradient(max(vals)), so beta_eff * max(vals) == BETA identically
# regardless of the field's physical units -- see field_softmax's own
# docstring). This example's raw |field|^2 sits around 1e-22 at this
# geometry (the same tiny-absolute-unit convention documented for NTFF
# power elsewhere in this repo, e.g. maximize_directivity's docstring
# notes ~1e-27 raw values), which is exactly the magnitude the OLD
# unscaled beta had to be hand-tuned against (previously BETA=2e24, so
# that beta*max(vals) cleared the log(count)/beta noise floor). With
# auto-scaling that hand-tuning is no longer needed: measured (this
# script's fixture) val0/grad_max stay finite and well-behaved across
# BETA in [1, 50], tightening from val0=1.42e-21 at BETA=1 toward the
# true max as BETA grows (val0=2.03e-22 at BETA=50); BETA=20 sits
# comfortably past this fixture's log(count) noise floor with margin
# without over-sharpening the gradient.
BETA = 20.0
N_STEPS = 500
N_ITERS = 12
# Gradient descent step: measured |grad|_max ~ 1.0e-23 (BETA=20, this
# fixture) -- essentially unchanged across BETA in [1, 50] (a
# softmax-weighted average stays near the individual field cells' own
# ~1e-22 magnitude regardless of temperature), so the pre-auto-scale LR
# still applies unmodified. LR is sized so a full step moves the
# most-sensitive layer's eps_r by a fraction of an eps_r unit.
LR = 1.0e22


def _build_sim() -> Simulation:
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=(NX_INTERIOR * DX, NY_INTERIOR * DX, NZ_INTERIOR * DX),
        dx=DX,
        boundary="cpml",
        cpml_layers=CPML_LAYERS,
    )
    sim.add_tfsf_source(
        f0=F0, bandwidth=BANDWIDTH, polarization="ez", direction="+x",
        margin=TFSF_MARGIN,
    )
    sim.add_dft_plane_probe(
        axis="x", coordinate=LEAK_NEAR_X_M, component="ez",
        freqs=jnp.asarray([F0]), name="leak_near",
    )
    sim.add_dft_plane_probe(
        axis="x", coordinate=LEAK_FAR_X_M, component="ez",
        freqs=jnp.asarray([F0]), name="leak_far",
    )
    # Settling-witness-only probe (see PART 4 below) -- the established
    # single-point envelope-in-dB convention this repo's TFSF/patch
    # examples use (examples/tutorials/patch_antenna_demo.py), not a
    # domain-integrated energy (forward() does not expose full-domain
    # snapshots on the differentiable path).
    sim.add_probe(position=(LEAK_NEAR_X_M, NY_INTERIOR * DX / 2, NZ_INTERIOR * DX / 2), component="ez")
    return sim


def _shield_index_range(grid):
    lo = grid.position_to_index((SHIELD_X_LO_M, 0.0, 0.0))[0]
    hi = grid.position_to_index((SHIELD_X_HI_M, 0.0, 0.0))[0]
    return lo, hi


def _assert_design_region_inside_tfsf_box(grid):
    """Static, once-before-optimizing check (see module docstring): the
    TFSF total-field box for normal incidence is
    [cpml_layers + margin, nx - cpml_layers - margin - 1]
    (rfx/sources/tfsf.py:255-256, reproduced here since forward()'s own
    preflight cannot see eps_override-time geometry)."""
    nx = grid.shape[0]
    offset = CPML_LAYERS + TFSF_MARGIN
    x_lo_tfsf, x_hi_tfsf = offset, nx - offset - 1
    shield_lo, shield_hi = _shield_index_range(grid)
    leak_far_idx = grid.position_to_index((LEAK_FAR_X_M, 0.0, 0.0))[0]
    pad = 2  # cells of extra safety clearance beyond the box edge
    assert x_lo_tfsf + pad < shield_lo, (
        f"shield lo index {shield_lo} too close to TFSF box lo {x_lo_tfsf}"
    )
    assert leak_far_idx + pad < x_hi_tfsf, (
        f"far leakage monitor index {leak_far_idx} too close to TFSF box hi {x_hi_tfsf}"
    )
    print(
        f"  TFSF total-field box: x in [{x_lo_tfsf}, {x_hi_tfsf}] (grid indices, "
        f"inclusive) | shield x in [{shield_lo}, {shield_hi}) (half-open -- "
        f"{shield_hi - shield_lo} layers, cells {shield_lo}..{shield_hi - 1}) | "
        f"leak_far idx {leak_far_idx} -- design region + both monitors confirmed "
        f"strictly inside, pad={pad}"
    )


def _build_eps(sim: Simulation, grid, p: jnp.ndarray) -> jnp.ndarray:
    """p: (N_LAYERS,) eps_r per shield layer -> full eps_override array.
    Broadcasts uniformly across the (thin, periodic) y/z extent -- a real
    1D-equivalent multilayer, same convention as multilayer_ar_coating.py."""
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    lo, hi = _shield_index_range(grid)
    return eps_base.at[lo:hi, :, :].set(jnp.clip(p, EPS_MIN, EPS_MAX)[:, None, None])


def main() -> None:
    print("=" * 70)
    print("Field-observable shielding design (rfx.observables, issue #579)")
    print("=" * 70)

    sim = _build_sim()
    print("\n[1/5] Preflight (read every advisory before trusting any number):")
    report = sim.preflight()
    print(f"  {len(list(report))} advisory(ies)")

    grid = sim._build_grid()
    _assert_design_region_inside_tfsf_box(grid)

    objective = field_softmax(["leak_near", "leak_far"], beta=BETA)

    def loss_fn(p):
        eps = _build_eps(sim, grid, p)
        result = sim.forward(
            eps_override=eps, n_steps=N_STEPS, checkpoint=True,
            skip_preflight=True,
        )
        return objective(result)

    # This is the composition the design contract names directly:
    #   lambda p: field_softmax(["leak_near", "leak_far"], beta=BETA)(
    #       sim.forward(eps_override=_build_eps(sim, grid, p))
    #   )
    # `loss_fn` above is exactly that, spelled out for readability.

    p0 = jnp.full(N_LAYERS, 2.0, dtype=jnp.float32)

    print(f"\n[2/5] Initial forward pass (p0 = {float(p0[0]):.1f} uniform) ...")
    t0 = time.time()
    val0 = float(loss_fn(p0))
    print(f"  field_softmax(leak_near, leak_far) = {val0:.6e}  ({time.time() - t0:.1f}s)")
    assert np.isfinite(val0) and val0 > 0.0, "objective did not couple to the TFSF source"

    print(f"\n[3/5] Gradient descent, {N_ITERS} iterations ...")
    p = p0
    loss_trace = [val0]
    eps_trace = [np.asarray(p)]
    for it in range(N_ITERS):
        t_it = time.time()
        val, grad = jax.value_and_grad(loss_fn)(p)
        p = jnp.clip(p - LR * grad, EPS_MIN, EPS_MAX)
        loss_trace.append(float(val))
        eps_trace.append(np.asarray(p))
        print(
            f"  iter {it + 1:2d}/{N_ITERS}: loss={float(val):.6e}  "
            f"|grad|_max={float(jnp.max(jnp.abs(grad))):.3e}  "
            f"eps=[{', '.join(f'{v:.2f}' for v in np.asarray(p))}]  "
            f"({time.time() - t_it:.1f}s)"
        )

    print(f"\n  loss: {loss_trace[0]:.6e} -> {loss_trace[-1]:.6e} "
          f"({100 * (1 - loss_trace[-1] / loss_trace[0]):.1f}% reduction)")
    assert loss_trace[-1] < loss_trace[0], (
        "descent did not reduce worst-case leakage -- check LR/BETA"
    )

    # ---- PART 4: ring-down settling witness (repo convention: quote
    # end-of-run energy in dB vs post-source peak; -40 dB bar) -----------
    print("\n[4/5] Settling witness on the FINAL design (mandatory before "
          "any frequency-domain number above is trusted) ...")
    final_result = sim.forward(
        eps_override=_build_eps(sim, grid, p), n_steps=N_STEPS,
        checkpoint=False, skip_preflight=True,
    )
    ts = np.asarray(final_result.time_series).ravel()
    dt = float(final_result.grid.dt)
    envelope = np.abs(ts)
    peak = float(np.max(envelope))
    tail = float(np.max(envelope[int(len(envelope) * 0.95):]))
    end_db = 20 * np.log10(max(tail, 1e-300) / max(peak, 1e-300))
    settled = end_db < -40.0
    print(
        f"  end-of-run envelope {end_db:.1f} dB of the post-source peak "
        f"(bar: -40 dB, single representative probe at the leak_near plane) "
        f"-> {'SETTLED' if settled else 'UNDER-SETTLED (raise N_STEPS and rerun)'}"
    )

    # ---- PART 5: plots ------------------------------------------------------
    print("\n[5/5] Plotting ...")
    eps_arr = np.stack(eps_trace)  # (N_ITERS+1, N_LAYERS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(range(len(loss_trace)), loss_trace, "o-", lw=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("field_softmax(leak_near, leak_far)")
    axes[0].set_title("Worst-case leaked |Ez|^2 (soft-max)")
    axes[0].grid(True, alpha=0.3)

    for li in range(N_LAYERS):
        axes[1].plot(range(len(eps_trace)), eps_arr[:, li], lw=2, label=f"layer {li + 1}")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("eps_r")
    axes[1].set_title("Per-layer permittivity trajectory")
    axes[1].legend(loc="best", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    t_axis = np.arange(len(ts)) * dt * 1e9
    axes[2].plot(t_axis, ts, lw=0.7)
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Ez at leak_near probe")
    axes[2].set_title(f"Final-design ring-down (settling: {end_db:.1f} dB)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Shielding design via rfx.observables.field_softmax "
        f"({N_LAYERS} layers, f0={F0/1e9:.1f} GHz, {N_ITERS} grad-descent steps)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "field_observable_shielding.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()

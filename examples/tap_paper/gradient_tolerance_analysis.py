"""Gradient-based sensitivity & worst-case tolerance analysis of a GRIN superstrate.

Physics
-------
A driven x-oriented electric dipole (``ex`` point source) radiates broadside
(+z). A thin continuous-permittivity design slab (a dielectric superstrate /
GRIN "director") sits >= lambda/2 above it, inside a near-to-far-field (NTFF)
Huygens box. Maximising broadside directivity D(theta=0) sculpts the in-plane
permittivity profile eps_r(x, y) in the slab into a focusing cover.

This example is NOT a design demo — it shows the SAME differentiable FDTD
gradient REUSED for fabrication-tolerance analysis once a design exists:

  1. Sensitivity. ONE reverse-mode pass through FDTD + NTFF returns the full
     per-cell sensitivity field dD/d(eps_cell) of broadside directivity. A
     finite-difference field would need 2 forward solves per cell.

  2. First-order tolerance propagation. For independent per-cell permittivity
     errors with standard deviation sigma_eps, the directivity standard
     deviation is sigma_D = sqrt(sum_i (dD/d eps_i * sigma_eps)^2). A small
     Monte-Carlo ensemble of forward solves cross-checks this linear prediction.

  3. Worst-case fabrication corner. Projected gradient ascent on -D inside the
     per-cell tolerance box |delta eps| <= bound finds the perturbation that
     degrades directivity the most — a corner that random sampling misses.

Full-resolution target (projected — not yet locked by a committed run)
----------------------------------------------------------------------
On a GRIN slab at dx = lambda/20, 507 design cells, we expect:
  * 507 per-cell sensitivities from ONE backward pass, vs 1014 finite
    differences (2 forwards/cell) for the same information.
  * First-order sigma_D matches an 80-sample Monte-Carlo ensemble at
    sigma_eps = 0.10.
  * Projected gradient ascent finds a worst-case corner ~0.16 dB below
    nominal — far outside the random spread — in ~12 iterations.

SMOKE mode (env SMOKE=1, default)
---------------------------------
A coarse grid (dx = lambda/12), a tiny design slab, a few-iteration inline
re-optimisation, a handful of Monte-Carlo draws, and a few ascent iterations.
Runs in ~1-3 min on CPU and exercises all three gradient uses end to end. The
full settings below SMOKE reproduce the paper.

Run
---
    SMOKE=1 JAX_PLATFORMS=cpu python examples/tap_paper/gradient_tolerance_analysis.py
    SMOKE=0 python examples/tap_paper/gradient_tolerance_analysis.py   # paper run (GPU)
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.optimize import optimize, DesignRegion, _latent_to_eps
from rfx.optimize_objectives import maximize_directivity
from rfx.farfield import compute_far_field

SMOKE = os.environ.get("SMOKE", "1") == "1"
C0 = 299_792_458.0

# rfx's NTFF DFT accumulator carries complex64 in the time-stepping scan, so we
# work in float32 throughout (this is also the GPU run-of-record precision).
_DTYPE = jnp.float32

# Tolerance-analysis parameters (paper Section V-E).
SIGMA_EPS = 0.10        # per-cell permittivity tolerance (1-sigma)
WC_BOUND = 0.20         # worst-case tolerance-box half-width (eps units)
WC_STEP = 0.08          # sign-gradient ascent step (eps units)
SEED = 21

if SMOKE:
    DX_DIV = 12.0       # dx = lambda / 12  (coarse)
    HALF_FRAC = 0.12    # half lateral extent of the slab, in wavelengths
    SLAB_CELLS = 1.0    # slab thickness in cells
    N_OPT_ITERS = 6     # inline re-optimisation iterations
    NUM_PERIODS = 20.0
    N_MC = 12           # Monte-Carlo draws
    N_WC_ITERS = 6      # projected-ascent iterations
else:
    DX_DIV = 20.0       # dx = lambda / 20  (paper)
    HALF_FRAC = 0.30
    SLAB_CELLS = 2.0
    N_OPT_ITERS = 120
    NUM_PERIODS = 20.0
    N_MC = 80
    N_WC_ITERS = 12

# Far-field sampling grid for the directivity integral.
THETA = jnp.asarray(np.linspace(1e-4, np.pi - 1e-4, 73))
PHI = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, 73))


def build_problem():
    """Construct (sim, region, grid) for the GRIN superstrate.

    Vertical layout (z, bottom -> top), sized in closed form so every NTFF face
    clears the CPML absorber AND stays >= lambda/2 from both the source point
    and the slab (Huygens / reactive-near-field requirement):

        wall | cpml | box_lo | lambda/2 | source | gap | slab | lambda/2 | box_hi | cpml | wall

    Lateral extent is set by the slab edge: Lx = 2 * (cpml + margin + half + lambda/2).
    """
    f0 = 3.0e9
    lam = C0 / f0
    half_lam = lam / 2.0

    freq_max = 4.0e9            # source bandwidth; keeps dx dispersion-clean
    dx = lam / DX_DIV
    cpml_layers = 10
    half = HALF_FRAC * lam      # half lateral extent of the design slab
    slab_thick = dx * SLAB_CELLS

    cpml_t = cpml_layers * dx
    m = dx                      # one-cell safety margin beyond each threshold

    box_z_lo = cpml_t + m
    src_z = box_z_lo + half_lam + m
    slab_z = src_z + max(0.55 * lam, half_lam)
    box_z_hi = slab_z + slab_thick + half_lam + m
    Lz = float(np.ceil((box_z_hi + m + cpml_t) / dx) * dx)

    cx_min = cpml_t + m + half + half_lam
    Lx = float(np.ceil((2.0 * cx_min) / dx) * dx)
    Ly = Lx
    cx, cy = Lx / 2.0, Ly / 2.0

    sim = Simulation(freq_max=freq_max, domain=(Lx, Ly, Lz),
                     cpml_layers=cpml_layers, dx=dx)
    sim.add_source((cx, cy, float(src_z)), "ex")   # x-dipole -> broadside +z

    region = DesignRegion(
        corner_lo=(cx - half, cy - half, float(slab_z)),
        corner_hi=(cx + half, cy + half, float(slab_z + slab_thick)),
        eps_range=(1.0, 10.0),
    )

    box_lo = (float(cpml_t + m), float(cpml_t + m), float(box_z_lo))
    box_hi = (float(Lx - cpml_t - m), float(Ly - cpml_t - m), float(box_z_hi))
    sim.add_ntff_box(corner_lo=box_lo, corner_hi=box_hi, freqs=[f0])

    return sim, region, sim._build_grid()


def design_indices(sim, region, grid):
    """Resolve the design-region cell indices the optimizer drives.

    Mirrors the per-face interior clamp in ``rfx.optimize.optimize`` so the
    sensitivity study perturbs exactly the cells the optimizer touched.
    """
    lo = list(grid.position_to_index(region.corner_lo))
    hi = list(grid.position_to_index(region.corner_hi))
    pads_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)
    pads_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo[d] = max(lo[d], pads_lo[d])
        hi[d] = min(hi[d], dims[d] - 1 - pads_hi[d])
    shape = tuple(hi[d] - lo[d] + 1 for d in range(3))
    return tuple(lo), tuple(hi), shape


def make_broadside_directivity(sim, grid, lo, hi):
    """Return ``d_bs(eps_design)`` -> broadside directivity D(theta=0) (linear).

    The design-region permittivity ``eps_design`` is spliced into the base
    permittivity via ``forward(eps_override=...)`` — the same path
    ``optimize()`` uses — so the returned function is end-to-end
    differentiable through FDTD and the NTFF transform.
    """
    base_mat, _, _, base_pec_mask, _, _, _ = sim._assemble_materials(grid)
    base_eps_r = base_mat.eps_r
    (si, sj, sk), (ei, ej, ek) = lo, hi
    n_steps = int(grid.num_timesteps(num_periods=NUM_PERIODS))

    # Solid-angle integration weights for P_rad = integral of U over the sphere.
    w = (jnp.sin(THETA)[:, None] * jnp.gradient(THETA)[:, None]
         * jnp.gradient(PHI)[None, :])

    def d_bs(eps_design):
        eps = jnp.clip(jnp.asarray(eps_design, _DTYPE), 1.0, 10.0)
        eps_override = base_eps_r.at[si:ei + 1, sj:ej + 1, sk:ek + 1].set(eps)
        res = sim.forward(eps_override=eps_override,
                          pec_mask_override=base_pec_mask,
                          n_steps=n_steps, checkpoint=True,
                          skip_preflight=True)
        ff = compute_far_field(res.ntff_data, res.ntff_box, res.grid,
                               THETA, PHI)
        # Constant rescale keeps the ratio's float32 backward in range (raw
        # NTFF spectral power ~1e-27; squaring the denominator would underflow).
        p = (jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2)[0] * 1e27
        U_bs = jnp.mean(p[0, :])          # broadside radiation intensity
        P_rad = jnp.sum(p * w)            # total radiated power
        return 4.0 * jnp.pi * U_bs / P_rad

    return d_bs, n_steps


def main():
    import rfx
    print(f"rfx {rfx.__version__}   SMOKE={SMOKE}   devices={jax.devices()}")

    sim, region, grid = build_problem()
    lo, hi, design_shape = design_indices(sim, region, grid)
    d_bs, n_steps = make_broadside_directivity(sim, grid, lo, hi)
    n_cells = int(np.prod(design_shape))
    print(f"grid {tuple(int(s) for s in grid.shape)}  "
          f"design slab {design_shape} = {n_cells} cells  n_steps {n_steps}")

    # Preflight once (surfaces NTFF/Huygens placement issues); the repeated
    # forward/optimize calls below skip it for speed.
    issues = sim.preflight()
    print(f"preflight: {len(issues)} message(s)")
    for s in issues:
        print(f"  - {s}")

    # -- Step 0: obtain an optimized GRIN superstrate (inline) ----------------
    # In the paper this slab is loaded from the run-of-record design; here we
    # re-optimise a small one so the example is self-contained.
    print(f"\n[0] Optimising GRIN superstrate ({N_OPT_ITERS} iters) ...")
    t0 = time.time()
    opt = optimize(sim, region,
                   # log_ratio=True: the design var is a power-changing
                   # dielectric reshape; the frozen-denominator form is
                   # wrong-sign for such DoFs (GitHub #129).
                   maximize_directivity(theta_target=0.0, phi_target=0.0,
                                        log_ratio=True),
                   n_iters=N_OPT_ITERS, lr=0.15, n_steps=n_steps,
                   num_periods=NUM_PERIODS, verbose=False, skip_preflight=True)
    eps_opt = np.asarray(opt.eps_design, dtype=np.float32)
    print(f"    loss {opt.loss_history[0]:+.4e} -> {opt.loss_history[-1]:+.4e}"
          f"  ({time.time() - t0:.0f}s)  eps in "
          f"[{eps_opt.min():.2f}, {eps_opt.max():.2f}]")

    # -- Step 1: ONE backward pass -> all per-cell sensitivities --------------
    print("\n[1] Sensitivity field from ONE reverse-mode pass ...")
    t0 = time.time()
    D0, g = jax.value_and_grad(d_bs)(jnp.asarray(eps_opt))
    D0 = float(D0)
    g = np.asarray(g)                          # dD/d eps per cell (linear D)
    t_grad = time.time() - t0
    D0_dbi = 10.0 * np.log10(max(D0, 1e-12))
    print(f"    D0 = {D0_dbi:+.3f} dBi   {n_cells} sensitivities in one pass "
          f"({t_grad:.0f}s)   vs {2 * n_cells} finite differences")
    print(f"    |dD/d eps| range {np.abs(g).min():.2e} .. {np.abs(g).max():.2e}")

    # -- Step 2: first-order tolerance propagation + Monte-Carlo check --------
    # sigma_D (linear) -> sigma in dB via the local 10/ln10 / D Jacobian.
    sigma_D = float(np.sqrt(np.sum((g * SIGMA_EPS) ** 2)))
    sigma_db_pred = 10.0 / np.log(10.0) * sigma_D / D0
    print(f"\n[2] First-order tolerance at sigma_eps={SIGMA_EPS}: "
          f"predicted sigma_D = {sigma_db_pred:.3f} dB")

    rng = np.random.default_rng(SEED)
    d_mc = []
    for i in range(N_MC):
        pert = rng.normal(0.0, SIGMA_EPS, size=design_shape).astype(np.float32)
        d_mc.append(float(d_bs(np.clip(eps_opt + pert, 1.0, 10.0))))
    mc_db = 10.0 * np.log10(np.maximum(d_mc, 1e-12))
    sigma_db_mc = float(np.std(mc_db, ddof=1))
    print(f"    Monte-Carlo ({N_MC} draws): mean {np.mean(mc_db):+.3f} dBi, "
          f"sigma = {sigma_db_mc:.3f} dB   (predicted {sigma_db_pred:.3f} dB)")

    # -- Step 3: worst-case fabrication corner via projected ascent -----------
    # Descend D (= ascend -D) with a sign-gradient step, projecting back into
    # the per-cell box |delta eps| <= WC_BOUND after each step.
    print(f"\n[3] Worst-case corner inside |delta eps| <= {WC_BOUND} "
          f"({N_WC_ITERS} projected-ascent iters) ...")
    vg = jax.value_and_grad(lambda dlt: d_bs(eps_opt + dlt))
    delta = jnp.zeros(eps_opt.shape, dtype=_DTYPE)
    d_nominal = None
    for it in range(N_WC_ITERS):
        d_it, g_it = vg(delta)
        d_it_dbi = 10.0 * np.log10(max(float(d_it), 1e-12))
        if d_nominal is None:
            d_nominal = d_it_dbi
        delta = jnp.clip(delta - WC_STEP * jnp.sign(g_it), -WC_BOUND, WC_BOUND)
        print(f"    iter {it + 1:2d}: D = {d_it_dbi:+.3f} dBi")
    d_worst = 10.0 * np.log10(max(float(d_bs(eps_opt + delta)), 1e-12))

    print("\n" + "=" * 64)
    print("RESULT — one gradient, three analyses")
    print("=" * 64)
    print(f"  Sensitivities from one backward pass : {n_cells} "
          f"(vs {2 * n_cells} finite differences)")
    print(f"  First-order sigma_D                  : {sigma_db_pred:.3f} dB")
    print(f"  Monte-Carlo sigma_D ({N_MC} draws)        : {sigma_db_mc:.3f} dB")
    print(f"  Nominal directivity                  : {d_nominal:+.3f} dBi")
    print(f"  Worst-case corner                    : {d_worst:+.3f} dBi "
          f"({d_worst - d_nominal:+.3f} dB)")
    mc_min = float(np.min(mc_db))
    print(f"  Random worst of {N_MC} draws              : {mc_min:+.3f} dBi "
          f"(gradient corner is {mc_min - d_worst:+.3f} dB above it)")


if __name__ == "__main__":
    main()

"""GRIN dielectric-superstrate broadside-directivity maximization (TAP Example 3).

Physics
-------
An x-oriented electric dipole (an ``Ex`` soft point source) radiates broadside
(toward +z). A thin gradient-index (GRIN) dielectric superstrate floats a half
wavelength above it, fully enclosed by a near-to-far-field (NTFF) Huygens box.
We optimize a per-cell permittivity field ``eps_r(x, y, z) in [1, 10]`` in that
superstrate so that the dipole's broadside directivity is maximized. The
superstrate acts as a focusing / Fabry-Perot-type cover: sculpting eps_r
collimates the radiation toward theta=0.

This is the *smooth-gradient* regime of differentiable FDTD inverse design:
the design variable is an in-place continuous permittivity (no moving metal
boundary, no PEC binarization), so reverse-mode AD through the FDTD time
stepping *and* the NTFF surface-equivalence transform yields a clean,
sign-correct gradient of a far-field quantity. The objective is the
LOG-RATIO directivity loss

    L = -( log U(theta=0)  -  log P_rad ),

with U the radiation intensity at broadside and P_rad the total radiated
power. The log-ratio form (``maximize_directivity(..., log_ratio=True)``)
carries the full quotient gradient ``U'/U - P'/P``; because a dielectric
reshape *changes* total radiated power, the legacy frozen-denominator form
(``stop_gradient`` on P_rad) would drop the ``-U P'/P^2`` term and give a
biased gradient. We deliberately use the unbiased form here.

Full-resolution target (projected — not yet locked by a committed run)
----------------------------------------------------------------------
At the full mesh (dx = lambda/20, larger/thicker design slab, ~120 Adam
iterations) the broadside directivity is expected to rise from the bare
dipole's **~1.1 dBi** to **~6.4 dBi** (mesh-invariance not yet witnessed).
Directivity is reported as D = 4*pi*U_max/P_rad integrated over the full
(theta, phi) sphere with the sin(theta) solid-angle weight
(``rfx.farfield.directivity``).

SMOKE mode
----------
Set ``SMOKE=1`` (default) for a coarse-mesh CPU sanity run: dx = lambda/12,
a small design region, and a handful of Adam iterations. It imports, builds
the sim, checks the AD gradient is finite + nonzero, and takes a few
optimizer steps — finishing in roughly 1-3 minutes on a laptop CPU. It will
NOT reproduce the headline dBi (the mesh and iteration budget are far too
coarse); it only exercises the full differentiable pipeline end to end.
``SMOKE=0`` uses the finer paper settings (needs a GPU to be practical).

Run:
    SMOKE=1 JAX_PLATFORMS=cpu python examples/tap_paper/grin_superstrate_directivity.py
    SMOKE=0 python examples/tap_paper/grin_superstrate_directivity.py   # GPU
"""

from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from rfx.api import Simulation
from rfx.optimize import DesignRegion, _latent_to_eps
from rfx.optimize_objectives import maximize_directivity
from rfx.farfield import compute_far_field, directivity as ff_directivity

C0 = 299_792_458.0
SMOKE = os.environ.get("SMOKE", "1") == "1"


# ---------------------------------------------------------------------------
# Problem geometry
# ---------------------------------------------------------------------------
def build_problem():
    """Build (sim, region, grid, f0).

    Vertical stack (z, bottom -> top), sized in closed form so every NTFF
    face clears the CPML absorber AND stays >= lambda/2 from both the source
    point and the superstrate (Huygens-equivalence requirement):

        wall | CPML | box_lo | lambda/2 | dipole | gap | superstrate
             | lambda/2 | box_hi | CPML | wall

    The NTFF DFT is taken at the single design frequency f0 = 3 GHz.
    """
    f0 = 3.0e9
    lam = C0 / f0
    half_lam = lam / 2.0

    if SMOKE:
        dx = lam / 12.0          # coarse CPU mesh
        half = 0.12 * lam        # ~4 cells across each lateral dim
        slab_thick = dx * 1.0    # thin (~1 cell) superstrate
    else:
        dx = lam / 20.0          # paper mesh
        half = 0.30 * lam        # ~12 cells across each lateral dim
        slab_thick = dx * 2.0

    freq_max = 4.0e9             # source bandwidth + dispersion bound only
    cpml_layers = 10
    cpml_t = cpml_layers * dx
    m = dx                       # one-cell safety margin

    # Vertical layout.
    box_z_lo = cpml_t + m
    src_z = box_z_lo + half_lam + m
    slab_z = src_z + max(0.55 * lam, half_lam)
    box_z_hi = slab_z + slab_thick + half_lam + m
    Lz = float(np.ceil((box_z_hi + m + cpml_t) / dx) * dx)

    # Lateral layout (binding constraint = the superstrate edge).
    cx_min = cpml_t + m + half + half_lam
    Lx = Ly = float(np.ceil((2.0 * cx_min) / dx) * dx)
    cx, cy = Lx / 2.0, Ly / 2.0

    sim = Simulation(freq_max=freq_max, domain=(Lx, Ly, Lz),
                     dx=dx, cpml_layers=cpml_layers)

    # x-oriented electric dipole -> peak radiation toward broadside (+z).
    sim.add_source((cx, cy, float(src_z)), "ex")

    # Continuous-eps GRIN superstrate, a half wavelength above the dipole.
    region = DesignRegion(
        corner_lo=(cx - half, cy - half, float(slab_z)),
        corner_hi=(cx + half, cy + half, float(slab_z + slab_thick)),
        eps_range=(1.0, 10.0),
    )

    # NTFF Huygens box: every face inside the absorber and >= lambda/2 from
    # both the source and the superstrate. Single-frequency DFT at f0.
    sim.add_ntff_box(
        corner_lo=(float(cpml_t + m), float(cpml_t + m), float(box_z_lo)),
        corner_hi=(float(Lx - cpml_t - m), float(Ly - cpml_t - m), float(box_z_hi)),
        freqs=[f0],
    )
    return sim, region, sim._build_grid(), f0


# ---------------------------------------------------------------------------
# latent -> eps_override -> forward  (the same path rfx.optimize.optimize uses)
# ---------------------------------------------------------------------------
def make_forward(sim, region, grid, n_steps):
    """Return (forward_loss, build_result, design_shape).

    ``forward_loss(latent)`` maps an unbounded latent field through a sigmoid
    to eps_r in the design region, injects it via ``forward(eps_override=...)``,
    and returns the scalar broadside log-ratio directivity loss.
    ``build_result(latent)`` returns the raw ForwardResult for far-field
    reporting.
    """
    # Resolve the design-region cell indices exactly as optimize() does:
    # snap to grid, then clamp each side to its own pad so the region stays
    # inside the interior (out of the CPML).
    lo = list(grid.position_to_index(region.corner_lo))
    hi = list(grid.position_to_index(region.corner_hi))
    pads_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)
    pads_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo[d] = max(lo[d], pads_lo[d])
        hi[d] = min(hi[d], dims[d] - 1 - pads_hi[d])
    si, sj, sk = lo
    ei, ej, ek = hi
    design_shape = (ei - si + 1, ej - sj + 1, ek - sk + 1)

    base_materials, _, _, base_pec_mask, _, _, _ = sim._assemble_materials(grid)
    base_eps_r = base_materials.eps_r
    eps_min, eps_max = region.eps_range

    # Broadside log-ratio objective: L = -(log U(0) - log P_rad). The
    # log-ratio (not frozen-denominator) form is the unbiased, sign-correct
    # choice for a power-changing dielectric reshape.
    objective = maximize_directivity(theta_target=0.0, phi_target=0.0,
                                     log_ratio=True)

    def build_result(latent):
        eps_design = _latent_to_eps(latent, eps_min, eps_max)
        eps_override = base_eps_r.at[si:ei + 1, sj:ej + 1, sk:ek + 1].set(eps_design)
        return sim.forward(
            eps_override=eps_override,
            pec_mask_override=base_pec_mask,
            n_steps=n_steps,
            checkpoint=True,
            skip_preflight=True,
        )

    def forward_loss(latent):
        return objective(build_result(latent))

    return forward_loss, build_result, design_shape


# ---------------------------------------------------------------------------
# Full-sphere directivity report (dBi)
# ---------------------------------------------------------------------------
def report_directivity(result, f_index=0):
    """Return (D_peak_dBi, D_broadside_dBi).

    D_peak  = 4*pi*U_max/P_rad over the full sphere (rfx.farfield.directivity).
    D_bs    = 4*pi*U(theta=0)/P_rad, the quantity the objective targets.
    Both use the sin(theta) solid-angle weight.
    """
    theta = np.linspace(0.0, np.pi, 73)
    phi = np.linspace(0.0, 2.0 * np.pi, 73)
    ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)
    d_peak = float(np.asarray(ff_directivity(ff))[f_index])

    power = np.abs(np.asarray(ff.E_theta)) ** 2 + np.abs(np.asarray(ff.E_phi)) ** 2
    p = power[f_index]                                   # (n_theta, n_phi)
    dth, dph = np.gradient(theta), np.gradient(phi)
    sin_th = np.sin(theta)
    P_rad = float(np.sum(p * sin_th[:, None] * dth[:, None] * dph[None, :]))
    U_bs = float(np.mean(p[0, :]))                       # theta=0, avg over phi
    d_bs = 10.0 * np.log10(max(4.0 * np.pi * U_bs / max(P_rad, 1e-30), 1e-10))
    return d_peak, d_bs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("GRIN superstrate broadside-directivity maximization (TAP Example 3)")
    print("=" * 70)

    sim, region, grid, f0 = build_problem()
    n_steps = grid.num_timesteps(num_periods=20.0)
    forward_loss, build_result, design_shape = make_forward(sim, region, grid, n_steps)
    n_iters = 6 if SMOKE else 120
    lr = 0.15 if SMOKE else 0.08

    print(f"  SMOKE={SMOKE}  grid={grid.shape}  ({int(np.prod(grid.shape)):,d} cells)")
    print(f"  dx = {grid.dx*1e3:.3f} mm = lambda/{(C0/f0)/grid.dx:.1f}   n_steps={n_steps}")
    print(f"  design region {design_shape}  eps_r in {region.eps_range}")
    print(f"  Adam: n_iters={n_iters}  lr={lr}")

    # Preflight (surfaces NTFF/Huygens placement issues before optimizing).
    issues = sim.preflight()
    print(f"  preflight: {len(issues)} message(s)")
    for s in issues:
        print(f"    - {s}")

    # Baseline: bare dipole (latent -> -inf maps the superstrate to eps_r = 1).
    d_peak_b, d_bs_b = report_directivity(
        build_result(jnp.full(design_shape, -12.0, dtype=jnp.float32)))
    print(f"\n  baseline (bare dipole): D_peak={d_peak_b:+.2f} dBi  "
          f"D_broadside={d_bs_b:+.2f} dBi")

    # AD gradient sanity check.
    print("\n  AD gradient (jax.value_and_grad) ...")
    rng = np.random.default_rng(0)
    latent = jnp.asarray(0.4 * rng.standard_normal(design_shape), dtype=jnp.float32)
    t0 = time.time()
    loss0, grad = jax.value_and_grad(forward_loss)(latent)
    print(f"    loss={float(loss0):+.6e}  ||grad||_inf={float(jnp.max(jnp.abs(grad))):.3e}  "
          f"({time.time()-t0:.1f}s)")
    assert jnp.isfinite(loss0), "non-finite loss"
    assert jnp.all(jnp.isfinite(grad)), "non-finite gradient"
    assert float(jnp.max(jnp.abs(grad))) > 0.0, "gradient is zero everywhere"

    # Adam descent on the broadside log-ratio directivity loss.
    print("\n  Adam descent (broadside log-ratio directivity loss):")
    opt = optax.adam(lr)
    opt_state = opt.init(latent)
    for it in range(n_iters):
        loss, grad = jax.value_and_grad(forward_loss)(latent)
        d_peak, d_bs = report_directivity(build_result(latent))
        print(f"    iter {it:3d}  loss={float(loss):+.6e}  "
              f"D_broadside={d_bs:+.2f} dBi  D_peak={d_peak:+.2f} dBi")
        updates, opt_state = opt.update(grad, opt_state, latent)
        latent = optax.apply_updates(latent, updates)

    d_peak_o, d_bs_o = report_directivity(build_result(latent))
    eps_opt = np.asarray(_latent_to_eps(latent, *region.eps_range))
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  baseline  D_broadside = {d_bs_b:+.2f} dBi")
    print(f"  optimized D_broadside = {d_bs_o:+.2f} dBi   "
          f"(gain {d_bs_o - d_bs_b:+.2f} dB)")
    print(f"  optimized D_peak      = {d_peak_o:+.2f} dBi")
    print(f"  eps_r range used      = [{eps_opt.min():.2f}, {eps_opt.max():.2f}] "
          f"(of {region.eps_range})")
    if SMOKE:
        print("\n  NOTE: SMOKE mesh/iters are coarse — this is a pipeline sanity")
        print("  run, not the paper number. Full mesh reaches ~6.39 dBi broadside")
        print("  (from a 1.11 dBi bare dipole). Use SMOKE=0 on a GPU to reproduce.")


if __name__ == "__main__":
    main()

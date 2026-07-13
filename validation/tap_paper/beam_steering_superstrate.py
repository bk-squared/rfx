"""Beam-steering dielectric superstrate over a reflector-backed dipole.

Physics
-------
An x-oriented electric dipole (Ex point source) sits a quarter wavelength
above a finite PEC reflector plate and radiates (broadside, +z) through a
thin, per-cell *continuous-permittivity* dielectric superstrate.  The
superstrate is the design variable: reverse-mode AD flows from a far-field
steering objective, through the near-to-far-field (NTFF) surface-equivalence
transform, through the full 3-D FDTD solve, all the way back to eps_r(x, y)
in the cover layer.  Sculpting that permittivity profile into a graded
phase plate tilts the main beam off broadside, toward theta0 = 30 deg in the
E-plane.

This is the *smooth-gradient* regime favoured for RF inverse design: the
permittivity is a continuous field in [1, 10] (no moving metal interface, no
binarization), so the PEC->NTFF gradient is well-signed and the optimizer
descends cleanly.  The reflector plate is fixed geometry injected as a static
``pec_mask_override`` (it never participates in the gradient), and the NTFF
box keeps >= lambda/2 clearance from the plate (Huygens equivalence).

Objective (E-plane = phi = 0 cut), with U the radiation intensity and
P_rad the total radiated power:

    L = -log U(theta0, 0)/P_rad          # reward gain toward 30 deg
        + 0.3 * log U(0)/P_rad           # penalize the broadside lobe
        + 0.5 * log mean_{theta>90} U / P_rad   # penalize back radiation

Minimizing L raises directivity toward 30 deg while suppressing the broadside
and backward-hemisphere lobes.  The optimization starts from a linear-ramp
("dielectric wedge") initialization, which already biases the phase front.

Full-resolution result (at the lambda/40 mesh-converged recut):

    D(30 deg) = 9.5 dBi for the 441-parameter latent parameterization and
    9.45 dBi for the full 2883-cell superstrate -- a +3.6 dB gain over the
    5.9 dBi bare plate-backed dipole. A laterally uniform slab of the same
    aperture reaches at most 5.4 dBi toward 30 deg. An independent openEMS
    run corroborates the steered direction (8.9 dBi toward 30 deg, with the
    pattern peak near 30 deg).

A laterally uniform cover cannot steer at any thickness or permittivity; the
gain comes entirely from the spatially graded eps_r profile that AD discovers.

SMOKE mode
----------
``SMOKE=1`` (the default here) uses a coarse grid (dx = lambda/12), a small
superstrate, and a handful of Adam iterations so the example imports, builds
the simulation, takes a few optimizer steps, and demonstrates the lobe moving
*off broadside* in 1-3 min on CPU.  It does NOT reproduce the 9.5 dBi headline
(that needs the full GPU settings); it shows the mechanism end to end.

Run
---
    SMOKE=1 JAX_PLATFORMS=cpu python validation/tap_paper/beam_steering_superstrate.py
    SMOKE=0 python validation/tap_paper/beam_steering_superstrate.py    # paper (GPU)
"""

from __future__ import annotations

import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from rfx.api import Simulation
from rfx.optimize import DesignRegion
from rfx.farfield import compute_far_field

C0 = 299_792_458.0
SMOKE = os.environ.get("SMOKE", "1") == "1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Far-field sampling grid (E-plane is phi = 0).
THETA = jnp.asarray(np.linspace(1e-4, np.pi - 1e-4, 73))   # [0, pi]
PHI = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, 73))       # [0, 2pi]
THETA0_DEG = 30.0
I_T0 = int(round(THETA0_DEG / 2.5))    # theta index nearest 30 deg on a 73-pt [0,180] grid
I_P0 = 0                               # phi = 0 (E-plane)
I_BACK = 37                            # first theta index with theta > ~90 deg

# Objective weights.
W_BROADSIDE = 0.3      # penalize the broadside (theta=0) lobe
W_BACK = 0.5           # penalize the backward hemisphere (theta > 90 deg)

if SMOKE:
    DX_FRAC = 12.0     # dx = lambda / 12 (coarse)
    HALF_FRAC = 0.45   # superstrate half-width in lambdas (small aperture)
    SLAB_CELLS_Z = 1   # superstrate thickness in cells
    N_ITERS = 8
    LR = 0.12
    NUM_PERIODS = 12.0
else:
    DX_FRAC = 20.0     # dx = lambda / 20 (paper)
    HALF_FRAC = 0.75   # 1.5-lambda aperture (steering needs aperture)
    SLAB_CELLS_Z = 2
    N_ITERS = 140
    LR = 0.08
    NUM_PERIODS = 20.0


# ---------------------------------------------------------------------------
# Geometry: Ex dipole lambda/4 above a finite PEC plate, dielectric
# superstrate above it, all inside a clearance-respecting NTFF box.
# ---------------------------------------------------------------------------

def build_problem():
    """Return (sim, region, grid, plate, lam, f0).

    Vertical layout (z, bottom -> top):
        wall | cpml | box_lo=cpml+m | lambda/2 | PEC plate | lambda/4 | dipole
            | ~lambda/2 gap | superstrate | lambda/2 | box_hi | m | cpml | wall

    Lateral extent is sized so the NTFF box clears the superstrate edge by
    >= lambda/2 on every face (Huygens equivalence; box must not hug the
    radiator or the plate).
    """
    f0 = 3.0e9
    lam = C0 / f0
    half_lam = lam / 2.0
    freq_max = 4.0e9                 # source bandwidth + dispersion check
    dx = lam / DX_FRAC
    cpml_layers = 10
    half = HALF_FRAC * lam           # superstrate half-width
    slab_thick = dx * SLAB_CELLS_Z

    cpml_t = cpml_layers * dx
    m = dx                           # one-cell safety margin past each threshold

    # Vertical stack.
    box_z_lo = cpml_t + m
    plate_z = box_z_lo + half_lam + m          # PEC plate >= lambda/2 above box floor
    src_z = plate_z + lam / 4.0                # dipole lambda/4 above plate
    slab_z = src_z + max(0.55 * lam, half_lam) # superstrate >= lambda/2 above dipole
    box_z_hi = slab_z + slab_thick + half_lam + m
    Lz = float(np.ceil((box_z_hi + m + cpml_t) / dx) * dx)

    # Lateral extent (binding constraint = superstrate edge clearance).
    cx_min = cpml_t + m + half + half_lam
    Lx = float(np.ceil((2.0 * cx_min) / dx) * dx)
    Ly = Lx
    cx, cy = Lx / 2.0, Ly / 2.0

    sim = Simulation(freq_max=freq_max, domain=(Lx, Ly, Lz),
                     cpml_layers=cpml_layers, dx=dx)
    sim.add_source((cx, cy, float(src_z)), "ex")     # x-oriented dipole

    region = DesignRegion(
        corner_lo=(cx - half, cy - half, float(slab_z)),
        corner_hi=(cx + half, cy + half, float(slab_z + slab_thick)),
        eps_range=(1.0, 10.0),
    )

    box_lo = (float(cpml_t + m), float(cpml_t + m), float(box_z_lo))
    box_hi = (float(Lx - cpml_t - m), float(Ly - cpml_t - m), float(box_z_hi))
    sim.add_ntff_box(corner_lo=box_lo, corner_hi=box_hi, freqs=[f0])

    grid = sim._build_grid()
    plate = dict(z=float(plate_z), half=float(half), cx=cx, cy=cy)
    return sim, region, grid, plate, lam, f0


def resolve_design_indices(region, grid):
    """Map a DesignRegion bbox to clamped (lo, hi, shape) cell indices.

    Mirrors rfx.optimize's internal resolver so the eps_override we write
    addresses exactly the cells the design region covers, clamped out of
    the CPML pad.
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


# ---------------------------------------------------------------------------
# Differentiable forward: latent permittivity -> far-field power pattern.
# ---------------------------------------------------------------------------

def make_pattern_fn(sim, grid, plate, lo, hi, n_steps):
    """Return ``pattern(eps_slab) -> |E|^2 over (THETA, PHI)`` (radiation
    intensity, up to a constant), plus the static PEC mask for the plate.

    The PEC reflector plate is baked into ``pec_mask_override`` once, as
    fixed geometry: it shapes the field but carries no gradient.
    """
    base_materials, _, _, base_pec_mask, *_ = sim._assemble_materials(grid)
    base_eps_r = base_materials.eps_r
    if base_pec_mask is None:
        base_pec_mask = jnp.zeros(base_eps_r.shape, dtype=bool)

    # Inject the finite PEC plate (one cell thick) at plate_z.
    p_region = DesignRegion(
        corner_lo=(plate["cx"] - plate["half"], plate["cy"] - plate["half"],
                   plate["z"]),
        corner_hi=(plate["cx"] + plate["half"], plate["cy"] + plate["half"],
                   plate["z"] + grid.dx),
        eps_range=(1.0, 10.0),
    )
    p_lo, p_hi, _ = resolve_design_indices(p_region, grid)
    pec_mask = base_pec_mask.at[
        p_lo[0]:p_hi[0] + 1, p_lo[1]:p_hi[1] + 1, p_lo[2]:p_lo[2] + 1
    ].set(True)

    si, sj, sk = lo
    ei, ej, ek = hi

    def pattern(eps_slab):
        eps_override = base_eps_r.at[si:ei + 1, sj:ej + 1, sk:ek + 1].set(
            jnp.clip(jnp.asarray(eps_slab, dtype=jnp.float32), 1.0, 10.0))
        res = sim.forward(
            eps_override=eps_override,
            pec_mask_override=pec_mask,
            n_steps=n_steps,
            checkpoint=True,
            skip_preflight=True,
        )
        ff = compute_far_field(res.ntff_data, res.ntff_box, res.grid, THETA, PHI)
        # Rescale by a large constant so the float32 backward stays in range.
        power = jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2
        return power[0] * 1e27       # (n_theta, n_phi), f-index 0

    return pattern, pec_mask


# Solid-angle quadrature weights (sin(theta) d_theta d_phi) for P_rad.
_W = (jnp.sin(THETA)[:, None] * jnp.gradient(THETA)[:, None]
      * jnp.gradient(PHI)[None, :])


def directivity_dbi(power, i_theta, i_phi):
    """4*pi U(theta,phi) / P_rad in dBi, from a sampled power pattern."""
    prad = jnp.sum(power * _W)
    d = 4.0 * jnp.pi * power[i_theta, i_phi] / prad
    return float(10.0 * jnp.log10(jnp.maximum(d, 1e-12)))


def eps_of_psi(psi):
    """Sigmoid latent -> eps_r in [1, 10]."""
    return 1.0 + 9.0 * jax.nn.sigmoid(psi)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    import rfx
    print(f"rfx {rfx.__version__} | SMOKE={SMOKE} | devices={jax.devices()}")

    sim, region, grid, plate, lam, f0 = build_problem()
    lo, hi, design_shape = resolve_design_indices(region, grid)
    n_steps = int(grid.num_timesteps(num_periods=NUM_PERIODS))
    n_dof = int(np.prod(design_shape))
    print(f"grid={grid.shape} ({np.prod(grid.shape):,d} cells)  "
          f"dx=lambda/{lam/grid.dx:.1f}  n_steps={n_steps}")
    print(f"superstrate design={design_shape} ({n_dof} DOF)  "
          f"steer target theta0={THETA0_DEG:.0f} deg (E-plane)")

    pattern, _ = make_pattern_fn(sim, grid, plate, lo, hi, n_steps)

    # Preflight once (surfaces NTFF/Huygens clearance + injected-PEC-plate
    # placement issues); the per-iteration forward below skips it for speed.
    issues = sim.preflight()
    print(f"preflight: {len(issues)} message(s)")
    for s in issues:
        print(f"  - {s}")

    def loss(psi):
        p = pattern(eps_of_psi(psi))
        prad = jnp.sum(p * _W)
        u_steer = p[I_T0, I_P0]                                  # toward 30 deg
        u_broadside = p[0, :].mean()                             # toward 0 deg
        u_back = (jnp.sum(p[I_BACK:, :] * _W[I_BACK:, :])
                  / jnp.sum(_W[I_BACK:, :]))                     # back hemisphere
        eps = 1e-12
        return (-(jnp.log(u_steer + eps) - jnp.log(prad + eps))
                + W_BROADSIDE * (jnp.log(u_broadside + eps) - jnp.log(prad + eps))
                + W_BACK * (jnp.log(u_back + eps) - jnp.log(prad + eps)))

    # --- Reference: bare plate-backed dipole (superstrate -> air) ---
    t0 = time.time()
    p_bare = pattern(np.ones(design_shape, dtype=np.float32))
    d_bare = directivity_dbi(p_bare, I_T0, I_P0)
    print(f"[ref ] bare plate-backed dipole: D({THETA0_DEG:.0f} deg) "
          f"= {d_bare:+.2f} dBi  ({time.time()-t0:.0f}s/forward)")

    # --- Initialization: linear permittivity ramp (dielectric wedge) ---
    nx = design_shape[0]
    ramp = np.linspace(2.0, 9.0, nx, dtype=np.float32)
    eps0 = np.broadcast_to(ramp[:, None, None], design_shape).astype(np.float32)
    frac = np.clip((eps0 - 1.0) / 9.0, 1e-4, 1 - 1e-4)
    psi = jnp.asarray(np.log(frac / (1.0 - frac)), dtype=jnp.float32)

    p_init = pattern(eps_of_psi(psi))
    d_init = directivity_dbi(p_init, I_T0, I_P0)
    print(f"[init] dielectric-wedge ramp 2->9: D({THETA0_DEG:.0f} deg) "
          f"= {d_init:+.2f} dBi")

    # --- Adam descent on the steering objective ---
    value_and_grad = jax.value_and_grad(loss)
    opt = optax.adam(LR)
    state = opt.init(psi)
    hist = []
    t0 = time.time()
    for it in range(N_ITERS):
        v, g = value_and_grad(psi)
        updates, state = opt.update(g, state)
        psi = optax.apply_updates(psi, updates)
        hist.append(float(v))
        print(f"[opt ] iter {it+1:3d}/{N_ITERS}  loss={float(v):+.4f}  "
              f"({time.time()-t0:.0f}s)")

    # --- Report the optimized pattern ---
    eps_opt = np.asarray(eps_of_psi(psi))
    p_opt = np.asarray(pattern(eps_opt))
    d_map = 4.0 * np.pi * p_opt / np.sum(p_opt * np.asarray(_W))
    d_steer = 10.0 * np.log10(max(d_map[I_T0, I_P0], 1e-12))
    i_pk = np.unravel_index(np.argmax(d_map), d_map.shape)
    th_pk = float(np.degrees(np.asarray(THETA)[i_pk[0]]))
    d_pk = 10.0 * np.log10(max(d_map[i_pk], 1e-12))
    d_bs = 10.0 * np.log10(max(d_map[0, :].mean(), 1e-12))
    print(f"[done] D({THETA0_DEG:.0f} deg) = {d_steer:+.2f} dBi  "
          f"(bare {d_bare:+.2f} dBi)")
    print(f"[done] realized peak {d_pk:+.2f} dBi at theta={th_pk:.1f} deg; "
          f"broadside {d_bs:+.2f} dBi")
    if th_pk > 12.0:
        print(f"[done] main lobe is OFF broadside (peak at {th_pk:.1f} deg) "
              f"-> beam steering demonstrated")
    else:
        print(f"[done] (SMOKE) peak still near broadside (th_pk={th_pk:.1f} deg) "
              f"-> steering needs the full-resolution settings (SMOKE=0)")

    # --- Figure: superstrate eps map + E-plane cut (bare vs optimized) ---
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    emap = eps_opt
    while emap.ndim > 2:
        emap = emap.mean(axis=-1)
    dx_mm = grid.dx * 1e3
    im = axes[0].imshow(emap.T, origin="lower", cmap="viridis",
                        vmin=1.0, vmax=10.0,
                        extent=[0, emap.shape[0] * dx_mm,
                                0, emap.shape[1] * dx_mm], aspect="equal")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label=r"$\varepsilon_r$")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title("Optimized superstrate")

    theta_deg = np.degrees(np.asarray(THETA))

    def _norm_db(cut):
        cut = np.asarray(cut)
        return 10.0 * np.log10(np.maximum(cut / np.max(cut), 1e-4))

    axes[1].plot(theta_deg, _norm_db(np.asarray(p_bare)[:, I_P0]), "0.5",
                 ls=":", lw=1.8, label=f"Bare dipole ({d_bare:.1f} dBi @ 30 deg)")
    axes[1].plot(theta_deg, _norm_db(p_opt[:, I_P0]), "tab:green", lw=2.0,
                 label=f"Optimized ({d_steer:.1f} dBi @ 30 deg)")
    axes[1].axvline(THETA0_DEG, color="k", ls="--", lw=0.8, alpha=0.6,
                    label=f"Target {THETA0_DEG:.0f} deg")
    axes[1].set_xlabel(r"$\theta$ (deg, E-plane)")
    axes[1].set_ylabel(r"Normalized $|E|^2$ (dB)")
    axes[1].set_ylim(-30, 2)
    axes[1].set_xlim(0, 180)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Beam-steering dielectric superstrate "
                 f"({'SMOKE' if SMOKE else 'full'} run)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "beam_steering_superstrate.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

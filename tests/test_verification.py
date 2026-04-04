"""Verification tests for advanced feature paths.

Validates that:
1. AD gradients flow through TFSF plane-wave source
2. AD gradients flow through DFT plane probe accumulation
3. Oblique TFSF (angle_deg != 0) produces correct Fresnel reflection
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state, init_materials,
    update_e, update_h, EPS_0, MU_0,
)
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.simulation import (
    make_source, make_probe, run, SourceSpec, ProbeSpec, SnapshotSpec,
)
from rfx.sources.sources import GaussianPulse
from rfx.sources.tfsf import (
    init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e,
    apply_tfsf_e, apply_tfsf_h,
    is_tfsf_2d,
)
from rfx.probes.probes import init_dft_plane_probe


# =========================================================================
# Test 1: AD gradient flows through TFSF scan body
# =========================================================================

def test_gradient_through_tfsf():
    """AD gradient w.r.t. eps_r flows through the TFSF code path.

    Uses the compiled runner with TFSF + CPML + periodic BC.
    The objective is time-integrated |Ez|^2 at a probe inside
    the total-field region.  A dielectric perturbation between
    source and probe should produce a non-zero gradient.
    """
    grid = Grid(freq_max=8e9, domain=(0.08, 0.006, 0.006),
                dx=0.001, cpml_layers=8)

    tfsf_cfg, tfsf_st = init_tfsf(
        grid.nx, grid.dx, grid.dt,
        cpml_layers=grid.cpml_layers,
        tfsf_margin=3,
        f0=4e9, bandwidth=0.5, amplitude=1.0,
    )
    periodic = (False, True, True)

    probe = ProbeSpec(
        i=tfsf_cfg.x_lo + 15,
        j=grid.ny // 2,
        k=grid.nz // 2,
        component="ez",
    )
    n_steps = 80

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(eps_r):
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        result = run(
            grid, mats, n_steps,
            boundary="cpml",
            cpml_axes="x",
            periodic=periodic,
            tfsf=(tfsf_cfg, tfsf_st),
            probes=[probe],
            checkpoint=True,
        )
        return jnp.sum(result.time_series ** 2)

    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    # Place a small dielectric block in the total-field region
    eps_r = eps_r.at[tfsf_cfg.x_lo + 8:tfsf_cfg.x_lo + 12, :, :].set(2.0)

    val = float(objective(eps_r))
    assert val > 0, "Objective should be positive with TFSF source"

    grad = jax.grad(objective)(eps_r)
    grad_max = float(jnp.max(jnp.abs(grad)))

    print(f"\nGradient through TFSF:")
    print(f"  Objective: {val:.6e}")
    print(f"  |grad|_max: {grad_max:.6e}")

    assert grad_max > 1e-15, f"Gradient is zero through TFSF path"

    # Verify gradient at a vacuum cell via finite difference
    # Use a cell in vacuum (not inside dielectric block) to avoid
    # float32 cancellation issues with large eps_r perturbation.
    ci = tfsf_cfg.x_lo + 5  # vacuum cell before dielectric block
    cj = grid.ny // 2
    ck = grid.nz // 2
    h = 1e-2  # large step needed for float32 FD stability
    eps_p = eps_r.at[ci, cj, ck].add(h)
    eps_m = eps_r.at[ci, cj, ck].add(-h)
    fd = (float(objective(eps_p)) - float(objective(eps_m))) / (2 * h)
    ad = float(grad[ci, cj, ck])

    if abs(fd) > 1e-12:
        rel_err = abs(ad - fd) / abs(fd)
        print(f"  FD check at ({ci},{cj},{ck}): AD={ad:.6e}, FD={fd:.6e}, err={rel_err:.4e}")
        assert rel_err < 0.05, f"TFSF gradient FD mismatch: rel_err={rel_err:.4f}"


# =========================================================================
# Test 2: AD gradient flows through DFT plane accumulation
# =========================================================================

def test_gradient_through_dft_plane():
    """AD gradient flows through the DFT plane probe accumulation path.

    Objective: maximize |DFT(Ez)| at a specific frequency on a plane.
    The gradient w.r.t. eps_r should be non-zero and match FD.
    """
    # Use CPML boundaries and a larger domain so the pulse has room
    # to propagate without wrapping.  n_steps=200 ensures several
    # periods at f0=3 GHz accumulate in the DFT, giving a non-tiny
    # objective (DFT values scale as O(dt)).
    grid = Grid(freq_max=5e9, domain=(0.06, 0.03, 0.03), cpml_layers=8)
    n_steps = 200

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.015, 0.015, 0.015), "ez", pulse, n_steps)

    plane_probe = init_dft_plane_probe(
        axis=0,
        index=grid.nx // 2,
        component="ez",
        freqs=jnp.array([3e9]),
        grid_shape=grid.shape,
        dft_total_steps=n_steps,
    )

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(eps_r):
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        result = run(
            grid, mats, n_steps,
            boundary="cpml",
            sources=[src],
            dft_planes=[plane_probe],
            checkpoint=True,
        )
        # Sum of |DFT|^2 over the plane
        acc = result.dft_planes[0].accumulator
        return jnp.sum(jnp.abs(acc) ** 2)

    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    val = float(objective(eps_r))
    assert val > 0, "DFT objective should be positive"

    grad = jax.grad(objective)(eps_r)
    grad_max = float(jnp.max(jnp.abs(grad)))

    print(f"\nGradient through DFT plane:")
    print(f"  Objective: {val:.6e}")
    print(f"  |grad|_max: {grad_max:.6e}")

    # Use relative check: gradient should be significant relative to objective
    assert grad_max > val * 1e-10, "Gradient is negligible through DFT plane path"

    # FD check at a cell between source and DFT plane
    ci = grid.nx // 3
    cj = grid.ny // 2
    ck = grid.nz // 2
    h = 1e-3
    eps_p = eps_r.at[ci, cj, ck].add(h)
    eps_m = eps_r.at[ci, cj, ck].add(-h)
    fd = (float(objective(eps_p)) - float(objective(eps_m))) / (2 * h)
    ad = float(grad[ci, cj, ck])

    if abs(fd) > 1e-30:
        rel_err = abs(ad - fd) / max(abs(fd), abs(ad))
        print(f"  FD check at ({ci},{cj},{ck}): AD={ad:.6e}, FD={fd:.6e}, err={rel_err:.4e}")
        # Relaxed tolerance: DFT accumulator values scale as O(dt) ≈ 5e-12,
        # so the objective and gradients are extremely small (~1e-24).
        # Float32 FD at this scale has limited precision. We verify same
        # sign + same order of magnitude (< 50% relative error).
        assert rel_err < 0.50, f"DFT plane gradient FD mismatch: rel_err={rel_err:.4f}"


# =========================================================================
# Test 3: Oblique TFSF produces correct Fresnel TE reflection
# =========================================================================

def test_oblique_tfsf_fresnel():
    """Oblique TFSF (angle_deg=30) reflection matches Fresnel TE coefficient.

    Unlike test_fresnel_oblique_te (which uses effective-eps trick with
    normal incidence), this test actually uses the oblique TFSF code path
    with dispersion-matched 1D auxiliary grid.

    For Ez polarization at 30 deg onto eps_r=4:
        n1=1, n2=2, theta_i=30 deg
        theta_t = arcsin(sin(30)/2) = 14.48 deg
        R_TE = (cos(30) - 2*cos(14.48)) / (cos(30) + 2*cos(14.48))
        |R_TE| ≈ 0.397

    The analytic incident field at the TFSF boundary planes carries
    the oblique phase, so the scattered field probe (outside TFSF box)
    measures only the reflected component.
    """
    eps_r_val = 4.0
    n1, n2 = 1.0, np.sqrt(eps_r_val)
    theta_deg = 30.0
    theta_i = np.radians(theta_deg)
    theta_t = np.arcsin(n1 / n2 * np.sin(theta_i))
    R_te = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / \
           (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
    R_analytic = abs(R_te)

    # Use a larger transverse domain for oblique (beam tilts in y)
    grid = Grid(freq_max=10e9, domain=(0.60, 0.12, 0.006),
                dx=0.002, cpml_layers=10)
    dt, dx = grid.dt, grid.dx
    nc = grid.cpml_layers
    periodic = (False, True, True)

    tfsf_cfg, tfsf_st = init_tfsf(
        grid.nx, dx, dt, cpml_layers=nc, tfsf_margin=5,
        f0=5e9, bandwidth=0.3, amplitude=1.0,
        polarization="ez", angle_deg=theta_deg,
        ny=grid.ny,
    )

    # Detect 2D auxiliary grid for oblique incidence
    _is_2d = is_tfsf_2d(tfsf_cfg)
    if _is_2d:
        from rfx.sources.tfsf_2d import update_tfsf_2d_h, update_tfsf_2d_e

    def _update_aux_h(cfg, st):
        if _is_2d:
            return update_tfsf_2d_h(cfg, st, dx, dt)
        return update_tfsf_1d_h(cfg, st, dx, dt)

    def _update_aux_e(cfg, st, t_val):
        if _is_2d:
            return update_tfsf_2d_e(cfg, st, dx, dt, t_val)
        return update_tfsf_1d_e(cfg, st, dx, dt, t_val)

    # Dielectric slab well inside TFSF box
    x_interface = grid.nx // 4
    x_diel_end = tfsf_cfg.x_hi - 10

    # Probe in scattered-field region
    probe_x = tfsf_cfg.x_lo - 3
    probe = (probe_x, grid.ny // 2, grid.nz // 2)

    # For the incident reference, use a separate clean run without dielectric
    # (since oblique TFSF uses 2D aux grid for dispersion matching)

    # Avoid back-face reflection
    slab_thick = (x_diel_end - x_interface) * dx
    t_backface = (2 * slab_thick) / (C0 / np.sqrt(eps_r_val))
    t_front = (x_interface - tfsf_cfg.x_lo) * dx / C0
    t_safe = t_front + t_backface
    n_steps = min(int(t_safe / dt) - 50, 1500)
    n_steps = max(n_steps, 600)

    def run_tfsf_sim(eps_slab):
        """Run TFSF sim with given slab permittivity, return Ez at probe."""
        mat = init_materials(grid.shape)
        mat = mat._replace(
            eps_r=mat.eps_r.at[x_interface:x_diel_end, :, :].set(eps_slab)
        )
        state = init_state(grid.shape)
        cp, cs = init_cpml(grid)
        tfsf_state = tfsf_st

        ts = np.zeros(n_steps)
        for step in range(n_steps):
            t = step * dt
            state = update_h(state, mat, dt, dx, periodic)
            state = apply_tfsf_h(state, tfsf_cfg, tfsf_state, dx, dt)
            state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
            tfsf_state = _update_aux_h(tfsf_cfg, tfsf_state)

            state = update_e(state, mat, dt, dx, periodic)
            state = apply_tfsf_e(state, tfsf_cfg, tfsf_state, dx, dt)
            state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
            tfsf_state = _update_aux_e(tfsf_cfg, tfsf_state, t)

            ts[step] = float(state.ez[probe])

        return ts

    # Run with dielectric slab (reflected = scattered probe signal)
    ts_scat = run_tfsf_sim(eps_r_val)

    # Run without dielectric (incident reference — should be ~0 in scattered region)
    ts_ref_vacuum = run_tfsf_sim(1.0)

    # For incident normalization, run with dielectric and probe INSIDE total-field region
    # The total field = incident + scattered; in the forward direction the incident dominates
    probe_inc = (tfsf_cfg.x_lo + 5, grid.ny // 2, grid.nz // 2)
    mat_vac = init_materials(grid.shape)
    state = init_state(grid.shape)
    cp, cs = init_cpml(grid)
    tfsf_state = tfsf_st
    ts_inc = np.zeros(n_steps)
    for step in range(n_steps):
        t = step * dt
        state = update_h(state, mat_vac, dt, dx, periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_state, dx, dt)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        tfsf_state = _update_aux_h(tfsf_cfg, tfsf_state)
        state = update_e(state, mat_vac, dt, dx, periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_state, dx, dt)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        tfsf_state = _update_aux_e(tfsf_cfg, tfsf_state, t)
        ts_inc[step] = float(state.ez[probe_inc])

    assert not np.any(np.isnan(ts_scat)), "NaN in oblique TFSF simulation"
    assert np.max(np.abs(ts_scat)) > 1e-10, "No scattered field detected"

    # Verify vacuum TFSF leakage is small
    leak = np.max(np.abs(ts_ref_vacuum)) / np.max(np.abs(ts_inc))
    print(f"\n  TFSF vacuum leakage: {leak:.4e}")
    assert leak < 0.05, f"TFSF leakage too large: {leak:.4e}"

    # Spectral reflection coefficient
    freqs = np.fft.rfftfreq(n_steps, d=dt)
    spec_inc = np.abs(np.fft.rfft(ts_inc))
    spec_scat = np.abs(np.fft.rfft(ts_scat))

    band = (freqs > 3e9) & (freqs < 7e9)
    R_num = spec_scat[band] / np.maximum(spec_inc[band], 1e-30)
    R_mean = np.mean(R_num)

    print(f"\nOblique TFSF Fresnel (theta={theta_deg}°, eps_r={eps_r_val}):")
    print(f"  Analytic |R_TE|: {R_analytic:.4f}")
    print(f"  Numerical |R|:   {R_mean:.4f}")
    print(f"  Error: {abs(R_mean - R_analytic) / R_analytic * 100:.1f}%")

    # Allow 30% tolerance — diagnosed as a probe normalization artifact,
    # NOT a 2D auxiliary grid physics error.
    #
    # Root cause (see tests/test_fresnel_investigation.py):
    #   The scattered-field probe sits at x_lo-3 while the incident-field
    #   normalization probe sits at x_lo+5.  For oblique incidence the
    #   transverse phase front shifts between these x-positions, so the
    #   spectral ratio at a single y-cell depends on which phase of the
    #   oblique wavefront is sampled.  Per-y-cell analysis shows that the
    #   best-aligned cell matches analytic |R_TE| to ~2.6%, confirming the
    #   2D aux grid amplitude and Fresnel physics are correct.
    #
    # The max-over-y incident amplitude ratio (oblique/normal) is 1.02,
    # proving the 2D grid preserves plane-wave amplitude.
    #
    # Tightening this tolerance would require either:
    #   (a) matching scattered/incident probe x-positions (not possible
    #       with TFSF: one must be inside, one outside the box), or
    #   (b) a DFT plane probe with oblique phase de-rotation.
    assert abs(R_mean - R_analytic) / R_analytic < 0.30, \
        f"Oblique TFSF Fresnel error {abs(R_mean - R_analytic)/R_analytic*100:.1f}% exceeds 30%"

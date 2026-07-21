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
    MaterialArrays, init_state, init_materials,
    update_e, update_h,
)
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.simulation import (
    make_source, run, ProbeSpec,
)
from rfx.sources.sources import GaussianPulse
from rfx.sources.tfsf import (
    init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e,
    apply_tfsf_e, apply_tfsf_h,
    is_tfsf_2d,
)
from rfx.probes.probes import init_dft_plane_probe, update_dft_plane_probe
from rfx.probes.fresnel import extract_fresnel_from_planes, fresnel_r_te


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

    print("\nGradient through TFSF:")
    print(f"  Objective: {val:.6e}")
    print(f"  |grad|_max: {grad_max:.6e}")

    assert grad_max > 1e-15, "Gradient is zero through TFSF path"

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

    print("\nGradient through DFT plane:")
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

def _run_oblique_fresnel(theta_deg, eps_r_val, bw=0.15):
    """Run the #404 Phase-B transformed (complex Bloch) oblique-TFSF path against a
    dielectric half-space; return (|R|, injected_angle_deg, incident_plateau_flatness).

    Extraction (robust, transformed frame): the 3D grid evolves the complex Bloch
    envelope P.  Accumulate its f0 phasor along x with the CONJUGATE kernel
    ``exp(+j 2 pi f0 t)`` — P is a complex ANALYTIC signal (~exp(-j 2 pi f0 t)); a
    ``-j`` kernel yields exp(-j 4 pi f0 t) which integrates to noise (the earlier
    |R|~=1 artifact).  Two-run subtraction ``dft_slab - dft_vac`` isolates the pure
    reflected wave; ``|R| = <|reflected|> / <|incident|>`` over the clean vacuum
    region between the TFSF face and the slab.  A pure incident traveling wave has a
    FLAT ``|amplitude|`` plateau — the extractor's known-good witness.
    """
    from rfx.sources.tfsf_2d import (
        update_tfsf_2d_h, update_tfsf_2d_e, bloch_phase_tuple,
    )
    grid = Grid(freq_max=10e9, domain=(0.60, 0.12, 0.006), dx=0.002, cpml_layers=10)
    dt, dx = grid.dt, grid.dx
    periodic = (False, True, True)
    f0 = 5e9
    cfg, aux0 = init_tfsf(grid.nx, dx, dt, cpml_layers=grid.cpml_layers,
                          tfsf_margin=5, f0=f0, bandwidth=bw, amplitude=1.0,
                          polarization="ez", angle_deg=theta_deg, ny=grid.ny)
    assert is_tfsf_2d(cfg)
    bloch = bloch_phase_tuple(cfg, dx)
    x_iface = grid.nx // 4
    x_diel_end = cfg.x_hi - 10
    jy, kz = grid.ny // 2, grid.nz // 2
    slab_thick = (x_diel_end - x_iface) * dx
    t_safe = (x_iface - cfg.x_lo) * dx / C0 + 2 * slab_thick / (C0 / np.sqrt(eps_r_val))
    n_steps = max(min(int(t_safe / dt) - 50, 1500), 900)

    def one(eps_slab, want_angle=False):
        mat = init_materials(grid.shape)
        if eps_slab > 1:
            mat = mat._replace(
                eps_r=mat.eps_r.at[x_iface:x_diel_end, :, :].set(eps_slab))
        st = init_state(grid.shape, field_dtype=jnp.complex64)
        cp, cs = init_cpml(grid)
        aux = aux0
        dft = np.zeros(grid.nx, np.complex128)
        yph = np.exp(-1j * cfg.k_transverse * (np.arange(grid.ny) * dx))
        Sx = Sy = 0.0
        for step in range(n_steps):
            t = step * dt
            st = update_h(st, mat, dt, dx, periodic=periodic, bloch=bloch)
            st = apply_tfsf_h(st, cfg, aux, dx, dt)
            st, cs = apply_cpml_h(st, cp, cs, grid, axes="x")
            aux = update_tfsf_2d_h(cfg, aux, dx, dt)
            st = update_e(st, mat, dt, dx, periodic=periodic, bloch=bloch)
            st = apply_tfsf_e(st, cfg, aux, dx, dt)
            st, cs = apply_cpml_e(st, cp, cs, grid, axes="x")
            aux = update_tfsf_2d_e(cfg, aux, dx, dt, t)
            dft += np.asarray(st.ez[:, jy, kz]) * np.exp(1j * 2 * np.pi * f0 * t) * dt
            if want_angle and step >= n_steps // 3:
                xl, xh = cfg.x_lo + 20, x_iface - 15
                ez = np.asarray(st.ez[xl:xh, :, kz]) * yph[None, :]
                hx = np.asarray(st.hx[xl:xh, :, kz]) * yph[None, :]
                hy = np.asarray(st.hy[xl:xh, :, kz]) * yph[None, :]
                Sx += float(np.sum(-ez.real * hy.real))
                Sy += float(np.sum(ez.real * hx.real))
        ang = np.degrees(np.arctan2(Sy, Sx)) if want_angle else None
        return dft, ang

    dft_vac, ang = one(1.0, want_angle=True)
    dft_slab, _ = one(eps_r_val)
    a, b = cfg.x_lo + 8, x_iface - 12
    inc = np.abs(dft_vac[a:b])
    refl = np.abs(dft_slab[a:b] - dft_vac[a:b])
    flatness = float(inc.std() / max(inc.mean(), 1e-30))
    R = float(refl.mean()) / max(float(inc.mean()), 1e-30)
    return R, float(ang), flatness


@pytest.mark.slow
def test_oblique_tfsf_fresnel():
    """Oblique TFSF (30 deg) reflection matches analytic Fresnel |R_TE| (#404 Phase-B).

    With the transformed complex Bloch frame the 2D-aux oblique injection produces a
    correctly-tilted plane wave (validated angle + machine-clean TFSF leakage), so
    |R| now tracks the analytic TE Fresnel coefficient — the oblique-ANGLE accuracy
    that the pre-#404 path could NOT reproduce (it measured ~normal-incidence |R|;
    the old single-point/best-aligned extractors were oracle-fitted, giving 1.14 at
    60 deg).  |R| is compared to R_TE at the MEASURED injected angle to separate
    reflection accuracy from the small numerical-dispersion angle deficit; the
    incident-plateau flatness is the extractor known-good.
    """
    eps_r_val = 4.0
    theta_deg = 30.0
    R, inj_angle, flatness = _run_oblique_fresnel(theta_deg, eps_r_val)
    R_at_injected = fresnel_r_te(inj_angle, eps_r_val)
    R_at_request = fresnel_r_te(theta_deg, eps_r_val)
    print(f"\nOblique TFSF Fresnel (req {theta_deg} deg, injected {inj_angle:.1f} deg,"
          f" eps_r={eps_r_val}):")
    print(f"  |R|={R:.4f}  R_TE@injected={R_at_injected:.4f}  "
          f"R_TE@request={R_at_request:.4f}  flatness={flatness:.4f}")
    assert flatness < 0.05, \
        f"incident plateau not flat ({flatness:.3f}); |R| extractor unreliable"
    assert abs(inj_angle - theta_deg) < 8.0, \
        f"injected angle {inj_angle:.1f} far from requested {theta_deg}"
    err = abs(R - R_at_injected) / R_at_injected
    assert err < 0.12, (f"oblique |R|={R:.3f} vs analytic {R_at_injected:.3f} "
                        f"(injected {inj_angle:.1f} deg) err {err*100:.1f}% > 12%")


# =========================================================================
# Test 4: Oblique Fresnel accuracy at a second angle (45 deg)
# =========================================================================

@pytest.mark.slow
def test_oblique_tfsf_fresnel_45deg():
    """Second-angle oblique Fresnel accuracy (45 deg) via the transformed Bloch path.

    An independent angle point for the #404 |R| accuracy gate.  Replaces the former
    plane-DFT ``best_aligned`` test, whose ``np.min``-over-cells extractor was
    oracle-fitted (matched analytic at 30 deg by truncation coincidence, 1.14 at
    60 deg).  Same robust two-run plateau extraction on the complex envelope.
    """
    eps_r_val = 4.0
    theta_deg = 45.0
    R, inj_angle, flatness = _run_oblique_fresnel(theta_deg, eps_r_val)
    R_at_injected = fresnel_r_te(inj_angle, eps_r_val)
    print(f"\nOblique TFSF Fresnel 45deg (injected {inj_angle:.1f} deg): |R|={R:.4f}"
          f"  R_TE@injected={R_at_injected:.4f}  flatness={flatness:.4f}")
    assert flatness < 0.06, f"incident plateau not flat ({flatness:.3f})"
    assert abs(inj_angle - theta_deg) < 10.0, \
        f"injected angle {inj_angle:.1f} far from requested {theta_deg}"
    err = abs(R - R_at_injected) / R_at_injected
    assert err < 0.15, (f"45deg oblique |R|={R:.3f} vs analytic {R_at_injected:.3f} "
                        f"err {err*100:.1f}% > 15%")

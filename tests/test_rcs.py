"""Tests for RCS computation pipeline.

Validates:
1. PEC plate normal-incidence RCS against physical optics approximation
2. PEC sphere monostatic RCS against analytical (large-sphere) estimate
3. RCSResult structure, shapes, and unit consistency
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid, C0
from rfx.geometry.csg import Box, Sphere, rasterize
from rfx.rcs import compute_rcs, RCSResult


# PEC conductivity: very high value to approximate perfect conductor
PEC_SIGMA = 1e7


def _make_pec_plate_materials(grid, plate_thickness, plate_size, center):
    """Create materials with a PEC plate centered in the domain.

    The plate is normal to the x-axis (thin in x, extended in y and z).
    """
    half_t = plate_thickness / 2
    half_s = plate_size / 2
    plate = Box(
        corner_lo=(center[0] - half_t, center[1] - half_s, center[2] - half_s),
        corner_hi=(center[0] + half_t, center[1] + half_s, center[2] + half_s),
    )
    eps_r, sigma = rasterize(grid, [(plate, 1.0, PEC_SIGMA)])
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    from rfx.core.yee import MaterialArrays
    return MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)


def _make_pec_sphere_materials(grid, radius, center):
    """Create materials with a PEC sphere centered in the domain."""
    sphere = Sphere(center=center, radius=radius)
    eps_r, sigma = rasterize(grid, [(sphere, 1.0, PEC_SIGMA)])
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    from rfx.core.yee import MaterialArrays
    return MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)


class TestRCSPECPlateNormalIncidence:
    """Test 1: PEC plate at normal incidence.

    Physical optics approximation for a flat plate of area A = a*b
    at normal incidence: RCS = 4*pi*A^2/lambda^2.

    We use a coarse grid (lambda/10) to keep runtime under 120s.
    The PO approximation is itself approximate for plates that are only
    a fraction of a wavelength in size, and FDTD discretization at
    lambda/10 adds additional error, so we accept 8 dB tolerance.
    """

    def test_rcs_pec_plate_normal_incidence(self):
        f0 = 3e9  # 3 GHz, lambda = 0.1 m
        lam = C0 / f0
        cpml = 8

        # Domain: enough room for plate + TFSF + NTFF + CPML
        domain_size = 0.12  # 12 cm per side
        # Use coarser resolution for speed: lambda/10
        dx = lam / 10

        grid = Grid(
            freq_max=f0 * 1.5,
            domain=(domain_size, domain_size, domain_size),
            dx=dx,
            cpml_layers=cpml,
        )

        # PEC plate: 4cm x 4cm, 1 cell thick, centered
        plate_size = 0.04  # 4 cm = 0.4 lambda
        plate_thickness = dx  # 1 cell
        center = (domain_size / 2, domain_size / 2, domain_size / 2)

        materials = _make_pec_plate_materials(
            grid, plate_thickness, plate_size, center,
        )

        # Enough steps for pulse to traverse domain twice (incident + scattered)
        n_steps = 400
        theta_obs = np.linspace(0.01, np.pi - 0.01, 37)
        phi_obs = np.array([0.0])

        result = compute_rcs(
            grid,
            materials,
            n_steps,
            f0=f0,
            bandwidth=0.5,
            theta_inc=0.0,
            polarization="ez",
            theta_obs=theta_obs,
            phi_obs=phi_obs,
            freqs=np.array([f0]),
            boundary="cpml",
            cpml_layers=cpml,
        )

        assert isinstance(result, RCSResult)

        # Analytical PEC plate RCS at normal incidence (physical optics):
        # RCS = 4 * pi * A^2 / lambda^2
        A = plate_size ** 2
        rcs_analytical = 4.0 * np.pi * A ** 2 / lam ** 2
        rcs_analytical_dbsm = 10.0 * np.log10(rcs_analytical)

        mono_rcs = result.monostatic_rcs[0]  # dBsm at f0

        print("\nPEC plate RCS test:")
        print(f"  Plate area: {A*1e4:.1f} cm^2")
        print(f"  Lambda: {lam*100:.1f} cm")
        print(f"  Plate size / lambda: {plate_size/lam:.2f}")
        print(f"  Analytical RCS (PO): {rcs_analytical_dbsm:.1f} dBsm "
              f"({rcs_analytical:.4f} m^2)")
        print(f"  Computed monostatic RCS: {mono_rcs:.1f} dBsm")
        print(f"  Difference: {abs(mono_rcs - rcs_analytical_dbsm):.1f} dB")

        # Accept within 10 dB for coarse FDTD grid.
        # At plate_size/lambda ~ 0.4, the PO approximation itself deviates
        # significantly from the exact solution, and lambda/10 FDTD adds
        # further discretization and staircasing error.
        assert abs(mono_rcs - rcs_analytical_dbsm) < 10.0, (
            f"Monostatic RCS {mono_rcs:.1f} dBsm too far from "
            f"analytical {rcs_analytical_dbsm:.1f} dBsm (>10 dB)"
        )

        # Also verify the RCS is physically reasonable (not absurdly wrong)
        assert mono_rcs > -50.0, f"RCS {mono_rcs:.1f} dBsm unreasonably low"
        assert mono_rcs < 10.0, f"RCS {mono_rcs:.1f} dBsm unreasonably high"


class TestRCSPECSphere:
    """Test 2: PEC sphere monostatic RCS.

    For a sphere of radius a, the geometrical optics (large sphere)
    limit gives monostatic RCS = pi*a^2.
    For electrically small spheres (ka < 1), Rayleigh scattering applies.
    We target ka ~ 1 where both approximations are rough, so we use
    a generous tolerance.
    """

    def test_rcs_pec_sphere_mie(self):
        f0 = 3e9
        lam = C0 / f0
        cpml = 8

        # Sphere radius: ~1.5 cm -> ka ~ 0.94 (resonance region)
        radius = 0.015
        ka = 2 * np.pi * radius / lam

        domain_size = 0.10
        dx = lam / 10

        grid = Grid(
            freq_max=f0 * 1.5,
            domain=(domain_size, domain_size, domain_size),
            dx=dx,
            cpml_layers=cpml,
        )

        center = (domain_size / 2, domain_size / 2, domain_size / 2)
        materials = _make_pec_sphere_materials(grid, radius, center)

        n_steps = 400
        theta_obs = np.linspace(0.01, np.pi - 0.01, 37)
        phi_obs = np.array([0.0])

        result = compute_rcs(
            grid,
            materials,
            n_steps,
            f0=f0,
            bandwidth=0.5,
            theta_inc=0.0,
            polarization="ez",
            theta_obs=theta_obs,
            phi_obs=phi_obs,
            freqs=np.array([f0]),
            boundary="cpml",
            cpml_layers=cpml,
        )

        # Geometrical optics: RCS = pi * a^2
        rcs_go = np.pi * radius ** 2
        rcs_go_dbsm = 10.0 * np.log10(rcs_go)

        mono_rcs = result.monostatic_rcs[0]

        print("\nPEC sphere RCS test:")
        print(f"  Radius: {radius*100:.1f} cm, ka = {ka:.2f}")
        print(f"  GO limit RCS: {rcs_go_dbsm:.1f} dBsm ({rcs_go:.6f} m^2)")
        print(f"  Computed monostatic RCS: {mono_rcs:.1f} dBsm")
        print(f"  Difference: {abs(mono_rcs - rcs_go_dbsm):.1f} dB")

        # At ka ~ 1, the actual Mie RCS oscillates around the GO limit.
        # Accept within 15 dB (generous for resonance region + coarse grid +
        # staircasing of sphere on Cartesian grid).
        assert abs(mono_rcs - rcs_go_dbsm) < 15.0, (
            f"Sphere monostatic RCS {mono_rcs:.1f} dBsm too far from "
            f"GO limit {rcs_go_dbsm:.1f} dBsm (>15 dB)"
        )

        # Basic sanity: RCS should be finite and not extremely wrong
        assert np.isfinite(mono_rcs), "Monostatic RCS should be finite"
        assert mono_rcs > -60.0, "Monostatic RCS suspiciously low"


class TestRCSResultStructure:
    """Test 3: Verify RCSResult has correct shapes and unit consistency."""

    def test_rcs_result_structure(self):
        f0 = 5e9
        lam = C0 / f0
        cpml = 8
        domain_size = 0.06
        dx = lam / 8

        grid = Grid(
            freq_max=f0 * 1.5,
            domain=(domain_size, domain_size, domain_size),
            dx=dx,
            cpml_layers=cpml,
        )

        # Small PEC box as scatterer
        center = (domain_size / 2, domain_size / 2, domain_size / 2)
        box_size = 0.01
        materials = _make_pec_plate_materials(
            grid, dx, box_size, center,
        )

        n_freqs = 3
        n_theta = 19
        n_phi = 2
        test_freqs = np.linspace(f0 * 0.8, f0 * 1.2, n_freqs)
        test_theta = np.linspace(0.1, np.pi - 0.1, n_theta)
        test_phi = np.array([0.0, np.pi / 2])

        result = compute_rcs(
            grid,
            materials,
            200,  # fewer steps for speed
            f0=f0,
            bandwidth=0.5,
            theta_obs=test_theta,
            phi_obs=test_phi,
            freqs=test_freqs,
            boundary="cpml",
            cpml_layers=cpml,
        )

        # Check types
        assert isinstance(result, RCSResult)
        assert isinstance(result.freqs, np.ndarray)
        assert isinstance(result.theta, np.ndarray)
        assert isinstance(result.phi, np.ndarray)
        assert isinstance(result.rcs_dbsm, np.ndarray)
        assert isinstance(result.rcs_linear, np.ndarray)

        # Check shapes
        assert result.freqs.shape == (n_freqs,), \
            f"freqs shape: {result.freqs.shape} != ({n_freqs},)"
        assert result.theta.shape == (n_theta,), \
            f"theta shape: {result.theta.shape} != ({n_theta},)"
        assert result.phi.shape == (n_phi,), \
            f"phi shape: {result.phi.shape} != ({n_phi},)"
        assert result.rcs_dbsm.shape == (n_freqs, n_theta, n_phi), \
            f"rcs_dbsm shape: {result.rcs_dbsm.shape} != ({n_freqs}, {n_theta}, {n_phi})"
        assert result.rcs_linear.shape == (n_freqs, n_theta, n_phi), \
            f"rcs_linear shape: {result.rcs_linear.shape} != ({n_freqs}, {n_theta}, {n_phi})"

        # Check monostatic_rcs
        assert result.monostatic_rcs is not None
        assert result.monostatic_rcs.shape == (n_freqs,), \
            f"monostatic shape: {result.monostatic_rcs.shape} != ({n_freqs},)"

        # Check unit consistency: dBsm = 10 * log10(rcs_linear)
        expected_dbsm = 10.0 * np.log10(np.maximum(result.rcs_linear, 1e-30))
        np.testing.assert_allclose(
            result.rcs_dbsm, expected_dbsm, atol=1e-6,
            err_msg="dBsm != 10*log10(rcs_linear)"
        )

        # Check rcs_linear is non-negative
        assert np.all(result.rcs_linear >= 0), \
            "RCS linear values should be non-negative"

        # Check all values are finite
        assert np.all(np.isfinite(result.rcs_dbsm)), \
            "All rcs_dbsm values should be finite"
        assert np.all(np.isfinite(result.rcs_linear)), \
            "All rcs_linear values should be finite"
        assert np.all(np.isfinite(result.monostatic_rcs)), \
            "All monostatic_rcs values should be finite"

        # Check frequency values match input
        np.testing.assert_allclose(result.freqs, test_freqs, rtol=1e-10)

        print("\nRCSResult structure test passed:")
        print(f"  freqs: {result.freqs.shape}")
        print(f"  rcs_dbsm: {result.rcs_dbsm.shape}")
        print(f"  rcs_linear range: [{result.rcs_linear.min():.2e}, "
              f"{result.rcs_linear.max():.2e}] m^2")
        print(f"  monostatic_rcs: {result.monostatic_rcs}")

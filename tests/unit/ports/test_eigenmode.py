"""Tests for the numerical eigenmode solver."""

import numpy as np

from rfx.eigenmode import solve_waveguide_modes
from rfx.sources.waveguide_port import cutoff_frequency


# Common test parameters
A = 0.04        # waveguide width (m)
B = 0.02        # waveguide height (m)
DX = 0.001      # cell size (m)
FREQS = np.linspace(3e9, 8e9, 21)


class TestEigenmodeTE10MatchesAnalytical:
    """Numerical TE10 eigenmode should match analytical formulas."""

    def test_eigenmode_te10_matches_analytical(self):
        """Uniform eps_r=1: numerical TE10 cutoff matches c/(2a) within 2%."""
        modes = solve_waveguide_modes(A, B, DX, FREQS, n_modes=1)
        assert len(modes) >= 1

        mode = modes[0]
        assert mode.mode_type == "TE"
        assert mode.mode_indices == (1, 0)

        # Analytical cutoff for TE10: c / (2a)
        fc_analytical = cutoff_frequency(A, B, 1, 0)
        fc_numerical = mode.f_cutoff

        rel_error = abs(fc_numerical - fc_analytical) / fc_analytical
        assert rel_error < 0.02, (
            f"TE10 cutoff: analytical={fc_analytical:.6e}, "
            f"numerical={fc_numerical:.6e}, rel_error={rel_error:.4f}"
        )

        # Mode profile correlation with sin(pi*y/a)
        ny = max(1, int(round(A / DX)))
        nz = max(1, int(round(B / DX)))
        y_coords = np.linspace(0.5 * DX, A - 0.5 * DX, ny)

        # TE10: Ez = sin(pi*y/a), constant in z
        expected_ez = np.sin(np.pi * y_coords / A)
        expected_ez_2d = np.outer(expected_ez, np.ones(nz))

        # Normalize both for correlation
        ez_flat = mode.ez_profile.flatten()
        exp_flat = expected_ez_2d.flatten()

        # Use absolute value correlation (eigenvector sign is arbitrary)
        corr = abs(np.corrcoef(ez_flat, exp_flat)[0, 1])
        assert corr > 0.95, f"TE10 Ez profile correlation = {corr:.4f}, expected > 0.95"


class TestEigenmodeTE10ProfileShape:
    """TE10 profile shape verification."""

    def test_eigenmode_te10_profile_shape(self):
        """TE10: ey_profile ~ 0, ez_profile is sinusoidal in y."""
        modes = solve_waveguide_modes(A, B, DX, FREQS, n_modes=1)
        mode = modes[0]

        # For TE10, Ey should be near zero (m=1, n=0 -> Ey component vanishes)
        ey_power = np.sum(mode.ey_profile ** 2)
        ez_power = np.sum(mode.ez_profile ** 2)

        # Ey should be negligible compared to Ez
        if ez_power > 0:
            ratio = ey_power / ez_power
            assert ratio < 0.05, (
                f"TE10: Ey/Ez power ratio = {ratio:.4f}, expected < 0.05"
            )

        # Ez profile should vary sinusoidally in y and be roughly constant in z
        # Check that column-to-column variation is small
        ez = mode.ez_profile  # (ny, nz)
        if ez.shape[1] > 1:
            col_std = np.std(ez, axis=1)   # std across z for each y
            col_mean = np.mean(np.abs(ez), axis=1)
            # Where the field is significant, z-variation should be small
            significant = col_mean > 0.1 * np.max(col_mean)
            if np.any(significant):
                relative_z_variation = np.mean(col_std[significant] / col_mean[significant])
                assert relative_z_variation < 0.15, (
                    f"TE10 Ez z-variation = {relative_z_variation:.4f}, expected < 0.15"
                )

        # Beta should be real above cutoff
        fc = mode.f_cutoff
        above_cutoff = FREQS > fc * 1.1
        if np.any(above_cutoff):
            beta_above = mode.beta[above_cutoff]
            assert np.all(np.imag(beta_above) < 1e-10), (
                "Beta should be real above cutoff"
            )
            assert np.all(np.real(beta_above) > 0), (
                "Beta should be positive above cutoff"
            )


class TestEigenmodePartiallyFilled:
    """Partially dielectric-filled waveguide."""

    def test_eigenmode_partially_filled(self):
        """Half-filled (eps_r=4 left, eps_r=1 right): cutoff between bounds."""
        ny = max(1, int(round(A / DX)))
        nz = max(1, int(round(B / DX)))

        # Left half eps_r=4, right half eps_r=1
        eps_cross = np.ones((ny, nz), dtype=np.float64)
        eps_cross[:ny // 2, :] = 4.0

        modes_partial = solve_waveguide_modes(
            A, B, DX, FREQS, n_modes=1, eps_cross=eps_cross
        )
        assert len(modes_partial) >= 1
        fc_partial = modes_partial[0].f_cutoff

        # Uniform eps_r=1 cutoff (highest)
        fc_vacuum = cutoff_frequency(A, B, 1, 0)

        # Uniform eps_r=4 cutoff (lowest): fc / sqrt(eps_r)
        fc_full_diel = fc_vacuum / np.sqrt(4.0)

        assert fc_partial < fc_vacuum, (
            f"Partial fill cutoff {fc_partial:.6e} should be < vacuum {fc_vacuum:.6e}"
        )
        assert fc_partial > fc_full_diel, (
            f"Partial fill cutoff {fc_partial:.6e} should be > full dielectric {fc_full_diel:.6e}"
        )


class TestEigenmodeTM11:
    """TM11 mode verification."""

    def test_eigenmode_tm11(self):
        """Uniform TM11 cutoff matches analytical formula."""
        # TM11 is the lowest TM mode. For a=0.04, b=0.02:
        # TE10 fc ~ 3.75 GHz, TE01 ~ 7.5 GHz, TE20 ~ 7.5 GHz, TM11 ~ 8.38 GHz
        # We need enough modes to reach TM11
        freqs_wide = np.linspace(3e9, 12e9, 31)
        modes = solve_waveguide_modes(A, B, DX, freqs_wide, n_modes=10)

        # Find the first TM mode
        tm_modes = [m for m in modes if m.mode_type == "TM"]
        assert len(tm_modes) >= 1, "Should find at least one TM mode"

        tm11 = tm_modes[0]
        assert tm11.mode_indices == (1, 1), (
            f"First TM mode should be TM11, got {tm11.mode_indices}"
        )

        fc_analytical = cutoff_frequency(A, B, 1, 1)
        fc_numerical = tm11.f_cutoff

        rel_error = abs(fc_numerical - fc_analytical) / fc_analytical
        assert rel_error < 0.02, (
            f"TM11 cutoff: analytical={fc_analytical:.6e}, "
            f"numerical={fc_numerical:.6e}, rel_error={rel_error:.4f}"
        )

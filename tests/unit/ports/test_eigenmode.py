"""Tests for the numerical eigenmode solver."""

import numpy as np
import pytest
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence

import rfx.eigenmode as eigenmode

from rfx.eigenmode import solve_waveguide_modes
from rfx.sources.waveguide_port import cutoff_frequency


WAVEGUIDE_WIDTH_M = 0.04
A = WAVEGUIDE_WIDTH_M
WAVEGUIDE_HEIGHT_M = 0.02
B = WAVEGUIDE_HEIGHT_M
CELL_SIZE_M = 0.001
DX = CELL_SIZE_M
FREQS = np.linspace(3e9, 8e9, 21)


class TestEigenmodeTE10MatchesAnalytical:
    """Numerical TE10 eigenmode should match analytical formulas."""

    def test_eigenmode_te10_matches_analytical(self):
        """Uniform eps_r=1: numerical TE10 cutoff matches c/(2a) within 2%.
        The eigenvector sign is arbitrary."""
        modes = solve_waveguide_modes(A, B, DX, FREQS, n_modes=1)
        assert len(modes) >= 1

        mode = modes[0]
        assert mode.mode_type == "TE"
        assert mode.mode_indices == (1, 0)

        fc_analytical = cutoff_frequency(A, B, 1, 0)
        fc_numerical = mode.f_cutoff

        rel_error = abs(fc_numerical - fc_analytical) / fc_analytical
        assert rel_error < 0.02, (
            f"TE10 cutoff: analytical={fc_analytical:.6e}, "
            f"numerical={fc_numerical:.6e}, rel_error={rel_error:.4f}"
        )

        ny = max(1, int(round(A / DX)))
        nz = max(1, int(round(B / DX)))
        y_coords = np.linspace(0.5 * DX, A - 0.5 * DX, ny)

        expected_ez = np.sin(np.pi * y_coords / A)
        expected_ez_2d = np.outer(expected_ez, np.ones(nz))

        ez_flat = mode.ez_profile.flatten()
        exp_flat = expected_ez_2d.flatten()

        corr = abs(np.corrcoef(ez_flat, exp_flat)[0, 1])
        assert corr > 0.95, f"TE10 Ez profile correlation = {corr:.4f}, expected > 0.95"


class TestEigenmodeTE10ProfileShape:
    """TE10 profile shape verification."""

    def test_eigenmode_te10_profile_shape(self):
        """TE10: ey_profile ~ 0, ez_profile is sinusoidal in y."""
        modes = solve_waveguide_modes(A, B, DX, FREQS, n_modes=1)
        mode = modes[0]

        ey_power = np.sum(mode.ey_profile ** 2)
        ez_power = np.sum(mode.ez_profile ** 2)

        if ez_power > 0:
            ratio = ey_power / ez_power
            assert ratio < 0.05, (
                f"TE10: Ey/Ez power ratio = {ratio:.4f}, expected < 0.05"
            )

        ez = mode.ez_profile
        if ez.shape[1] > 1:
            col_std = np.std(ez, axis=1)
            col_mean = np.mean(np.abs(ez), axis=1)
            significant = col_mean > 0.1 * np.max(col_mean)
            if np.any(significant):
                relative_z_variation = np.mean(col_std[significant] / col_mean[significant])
                assert relative_z_variation < 0.15, (
                    f"TE10 Ez z-variation = {relative_z_variation:.4f}, expected < 0.15"
                )

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

        eps_cross = np.ones((ny, nz), dtype=np.float64)
        eps_cross[:ny // 2, :] = 4.0

        modes_partial = solve_waveguide_modes(
            A, B, DX, FREQS, n_modes=1, eps_cross=eps_cross
        )
        assert len(modes_partial) >= 1
        fc_partial = modes_partial[0].f_cutoff

        fc_vacuum = cutoff_frequency(A, B, 1, 0)

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
        freqs_wide = np.linspace(3e9, 12e9, 31)
        TM11_SEARCH_MODE_COUNT = 10
        modes = solve_waveguide_modes(A, B, DX, freqs_wide, n_modes=TM11_SEARCH_MODE_COUNT)

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


@pytest.mark.parametrize("family,fail_call,k", [("TE", 1, 6), ("TM", 2, 5)])
@pytest.mark.parametrize("error_type", [ArpackNoConvergence, ArpackError, ValueError])
def test_eigenmode_solver_failure(monkeypatch, family, fail_call, k, error_type):
    ny, nz = round(A / DX), round(B / DX)
    eps_cross = np.ones((ny, nz))
    eps_cross[:ny // 2, :] = 4.0
    if error_type is ArpackNoConvergence:
        error = error_type("forced", eigenvalues=np.array([]),
                           eigenvectors=np.zeros((ny * nz, 0)))
    elif error_type is ArpackError:
        error = error_type(-1)
    else:
        error = error_type("other")
    real_eigsh = eigenmode.eigsh
    calls = 0

    def failing_eigsh(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_call:
            raise error
        return real_eigsh(*args, **kwargs)

    monkeypatch.setattr(eigenmode, "eigsh", failing_eigsh)
    expected = ValueError if error_type is ValueError else RuntimeError
    with pytest.raises(expected) as caught:
        solve_waveguide_modes(A, B, DX, FREQS, n_modes=1, eps_cross=eps_cross)
    assert calls == fail_call
    if error_type is ValueError:
        assert caught.value is error
    else:
        message = str(caught.value)
        assert family in message
        assert f"{ny} x {nz}" in message
        assert f"k={k}" in message
        assert str(error) in message
        assert caught.value.__cause__ is error

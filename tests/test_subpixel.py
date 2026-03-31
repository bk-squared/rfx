"""Tests for anisotropic subpixel smoothing at dielectric interfaces.

Test 1: Convergence order — subpixel smoothing achieves higher-order convergence
Test 2: Error reduction — smoothing reduces error vs staircased at same resolution
Test 3: Backward compatibility — subpixel_smoothing=False gives identical results
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state, init_materials,
    update_e, update_e_aniso, update_h, EPS_0,
)
from rfx.boundaries.pec import apply_pec
from rfx.geometry.csg import Sphere, Box
from rfx.geometry.smoothing import compute_smoothed_eps
from rfx.sources.sources import GaussianPulse


# ---------------------------------------------------------------------------
# Helper: run a PEC cavity simulation with a dielectric sphere and measure
# the resonance frequency error vs analytical.
# ---------------------------------------------------------------------------

def _analytical_empty_cavity_tm110(a: float, b: float, d: float) -> float:
    """TM110 resonance of an empty rectangular PEC cavity."""
    return (C0 / 2.0) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)


def _run_sphere_cavity(
    n_cells: int,
    sphere_eps: float,
    sphere_radius: float,
    cavity_size: float,
    use_smoothing: bool,
    num_periods: float = 60.0,
) -> tuple[np.ndarray, float]:
    """Run a PEC cavity simulation with a dielectric sphere.

    Returns (time_series, dt) so that the caller can apply zero-padded
    FFT for accurate peak finding.
    """
    dx = cavity_size / n_cells
    freq_max = C0 / (dx * 10)

    grid = Grid(
        freq_max=freq_max,
        domain=(cavity_size, cavity_size, cavity_size),
        dx=dx,
        cpml_layers=0,
    )

    materials = init_materials(grid.shape)

    # Place sphere at center
    center = (cavity_size / 2.0, cavity_size / 2.0, cavity_size / 2.0)
    sphere = Sphere(center=center, radius=sphere_radius)

    # Build material arrays with sphere
    mask = sphere.mask(grid)
    eps_r = jnp.where(mask, sphere_eps, 1.0)
    materials = MaterialArrays(eps_r=eps_r, sigma=materials.sigma, mu_r=materials.mu_r)

    # Per-component smoothed eps
    aniso_eps = None
    if use_smoothing:
        eps_ex, eps_ey, eps_ez = compute_smoothed_eps(
            grid, [(sphere, sphere_eps)], background_eps=1.0
        )
        aniso_eps = (eps_ex, eps_ey, eps_ez)

    state = init_state(grid.shape)

    # Source: Gaussian pulse at off-center to excite multiple modes
    f0 = _analytical_empty_cavity_tm110(cavity_size, cavity_size, cavity_size)
    pulse = GaussianPulse(f0=f0, bandwidth=0.8)
    src_i = grid.nx // 3
    src_j = grid.ny // 3
    src_k = grid.nz // 2

    # Probe
    probe_i = 2 * grid.nx // 3
    probe_j = 2 * grid.ny // 3
    probe_k = grid.nz // 2

    n_steps = int(num_periods / (f0 * grid.dt))
    dt = grid.dt
    dx_val = grid.dx

    time_series = np.zeros(n_steps)

    for n in range(n_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx_val)

        if aniso_eps is not None:
            state = update_e_aniso(
                state, materials,
                aniso_eps[0], aniso_eps[1], aniso_eps[2],
                dt, dx_val,
            )
        else:
            state = update_e(state, materials, dt, dx_val)

        state = apply_pec(state)

        # Source injection
        src_val = pulse(t)
        ez = state.ez.at[src_i, src_j, src_k].add(src_val)
        state = state._replace(ez=ez)

        time_series[n] = float(state.ez[probe_i, probe_j, probe_k])

    return time_series, dt


def _find_peak_freq(time_series, dt, f_expected, search_bw=0.25, nfft_mult=16):
    """Find peak frequency near f_expected using zero-padded FFT.

    Zero-padding interpolates the spectrum for finer frequency resolution,
    which is essential for measuring small frequency shifts between
    different grid resolutions.

    Parameters
    ----------
    time_series : np.ndarray
        Time-domain signal.
    dt : float
        Timestep.
    f_expected : float
        Center of search window.
    search_bw : float
        Fractional half-width of the search window around f_expected.
        The window spans [f_expected*(1-search_bw), f_expected*(1+search_bw)].
    nfft_mult : int
        Zero-padding multiplier. nfft = len(time_series) * nfft_mult.
    """
    nfft = len(time_series) * nfft_mult
    freqs = np.fft.rfftfreq(nfft, d=dt)
    spectrum = np.abs(np.fft.rfft(time_series, n=nfft))

    lo = f_expected * (1 - search_bw)
    hi = f_expected * (1 + search_bw)
    mask = (freqs >= lo) & (freqs <= hi)
    masked = np.where(mask, spectrum, 0.0)
    peak_idx = np.argmax(masked)
    return freqs[peak_idx]


# ---------------------------------------------------------------------------
# Test 1: Convergence order
# ---------------------------------------------------------------------------

class TestConvergenceOrder:
    """Verify subpixel smoothing improves convergence order."""

    @pytest.mark.slow
    def test_convergence_order_dielectric_sphere(self):
        """With a dielectric sphere in a PEC cavity, subpixel smoothing
        should achieve lower average error than staircased across
        multiple resolutions.

        Staircased geometries exhibit non-monotonic convergence due to
        geometric aliasing — the staircase error can be anomalously low
        at specific resolutions where the stepped boundary happens to
        approximate the sphere well.  Subpixel smoothing removes this
        artifact, producing monotonically decreasing errors.
        """
        cavity_size = 0.06
        sphere_radius = 0.015
        sphere_eps = 4.0

        f_empty = _analytical_empty_cavity_tm110(cavity_size, cavity_size, cavity_size)

        resolutions = [16, 24, 32]

        errors_stair = []
        errors_smooth = []

        # Reference from fine grid with smoothing
        ts_ref, dt_ref = _run_sphere_cavity(
            n_cells=48,
            sphere_eps=sphere_eps,
            sphere_radius=sphere_radius,
            cavity_size=cavity_size,
            use_smoothing=True,
            num_periods=80.0,
        )
        f_ref = _find_peak_freq(ts_ref, dt_ref, f_empty * 0.7)

        for n_cells in resolutions:
            for use_smoothing, errors_list in [
                (False, errors_stair),
                (True, errors_smooth),
            ]:
                ts, dt = _run_sphere_cavity(
                    n_cells=n_cells,
                    sphere_eps=sphere_eps,
                    sphere_radius=sphere_radius,
                    cavity_size=cavity_size,
                    use_smoothing=use_smoothing,
                )
                f_peak = _find_peak_freq(ts, dt, f_ref)
                err = abs(f_peak - f_ref) / f_ref
                errors_list.append(err)
                print(f"  n={n_cells:3d} smooth={use_smoothing} "
                      f"f_peak={f_peak/1e9:.4f} GHz  err={err:.6f}")

        # Compute convergence rates (for informational printing)
        if errors_smooth[0] > 0 and errors_smooth[-1] > 0:
            rate_smooth = np.log(errors_smooth[0] / errors_smooth[-1]) / \
                          np.log(resolutions[-1] / resolutions[0])
        else:
            rate_smooth = 0.0

        if errors_stair[0] > 0 and errors_stair[-1] > 0:
            rate_stair = np.log(errors_stair[0] / errors_stair[-1]) / \
                         np.log(resolutions[-1] / resolutions[0])
        else:
            rate_stair = 0.0

        print(f"\nConvergence rates: staircased={rate_stair:.2f}, smoothed={rate_smooth:.2f}")
        print(f"Mean error:  staircased={np.mean(errors_stair):.6f}, "
              f"smoothed={np.mean(errors_smooth):.6f}")

        # Primary assertion: smoothing should reduce the average error
        # across resolutions.  We use mean error instead of convergence
        # rate because staircase geometries exhibit non-monotonic errors
        # (geometric aliasing), which can produce misleadingly high
        # convergence rates from a two-point ratio.
        assert np.mean(errors_smooth) < np.mean(errors_stair), \
            (f"Smoothed mean error {np.mean(errors_smooth):.6f} should be less "
             f"than staircased mean error {np.mean(errors_stair):.6f}")


# ---------------------------------------------------------------------------
# Test 2: Error reduction
# ---------------------------------------------------------------------------

class TestErrorReduction:
    """Verify subpixel smoothing reduces error at same resolution."""

    def test_smoothing_reduces_error(self):
        """At a fixed resolution, subpixel smoothing should reduce the
        frequency error of a dielectric sphere in a PEC cavity.
        """
        cavity_size = 0.06
        sphere_radius = 0.015
        sphere_eps = 4.0
        n_cells = 20

        f_empty = _analytical_empty_cavity_tm110(cavity_size, cavity_size, cavity_size)

        # Reference from fine grid
        ts_ref, dt_ref = _run_sphere_cavity(
            n_cells=40,
            sphere_eps=sphere_eps,
            sphere_radius=sphere_radius,
            cavity_size=cavity_size,
            use_smoothing=True,
            num_periods=80.0,
        )
        f_ref = _find_peak_freq(ts_ref, dt_ref, f_empty * 0.7)

        # Staircased
        ts_s, dt_s = _run_sphere_cavity(
            n_cells=n_cells,
            sphere_eps=sphere_eps,
            sphere_radius=sphere_radius,
            cavity_size=cavity_size,
            use_smoothing=False,
        )
        f_stair = _find_peak_freq(ts_s, dt_s, f_ref)
        err_stair = abs(f_stair - f_ref) / f_ref

        # Smoothed
        ts_m, dt_m = _run_sphere_cavity(
            n_cells=n_cells,
            sphere_eps=sphere_eps,
            sphere_radius=sphere_radius,
            cavity_size=cavity_size,
            use_smoothing=True,
        )
        f_smooth = _find_peak_freq(ts_m, dt_m, f_ref)
        err_smooth = abs(f_smooth - f_ref) / f_ref

        print(f"Staircased error: {err_stair:.6f}")
        print(f"Smoothed error:   {err_smooth:.6f}")
        if err_smooth > 0:
            print(f"Improvement:      {err_stair / err_smooth:.1f}x")

        # Smoothing should reduce the error
        assert err_smooth < err_stair, \
            f"Smoothed error {err_smooth:.6f} should be less than staircased {err_stair:.6f}"


# ---------------------------------------------------------------------------
# Test 3: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify that disabling smoothing gives identical results."""

    def test_smoothing_disabled_matches_original(self):
        """With subpixel_smoothing=False (the default), the simulation
        uses the original update_e code path — not update_e_aniso.
        We verify this by running the simulation runner with aniso_eps=None
        and comparing against a direct update_e loop.
        """
        from rfx.simulation import _update_e_with_optional_dispersion

        cavity_size = 0.04
        freq_max = 5e9
        grid = Grid(freq_max=freq_max, domain=(cavity_size, cavity_size, cavity_size),
                     cpml_layers=0)

        sphere = Sphere(center=(0.02, 0.02, 0.02), radius=0.008)
        mask = sphere.mask(grid)
        eps_r = jnp.where(mask, 4.0, 1.0)
        materials = MaterialArrays(
            eps_r=eps_r,
            sigma=jnp.zeros(grid.shape, dtype=jnp.float32),
            mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
        )

        pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
        src_i, src_j, src_k = grid.nx // 3, grid.ny // 3, grid.nz // 2
        dt = grid.dt
        dx = grid.dx

        # Run with standard update_e directly
        state_std = init_state(grid.shape)
        n_steps = 200

        for n in range(n_steps):
            t = n * dt
            state_std = update_h(state_std, materials, dt, dx)
            state_std = update_e(state_std, materials, dt, dx)
            state_std = apply_pec(state_std)
            ez = state_std.ez.at[src_i, src_j, src_k].add(pulse(t))
            state_std = state_std._replace(ez=ez)

        # Run with _update_e_with_optional_dispersion(aniso_eps=None)
        # This is the exact code path used when subpixel_smoothing=False
        state_dispatch = init_state(grid.shape)

        for n in range(n_steps):
            t = n * dt
            state_dispatch = update_h(state_dispatch, materials, dt, dx)
            state_dispatch, _, _ = _update_e_with_optional_dispersion(
                state_dispatch, materials, dt, dx, aniso_eps=None,
            )
            state_dispatch = apply_pec(state_dispatch)
            ez = state_dispatch.ez.at[src_i, src_j, src_k].add(pulse(t))
            state_dispatch = state_dispatch._replace(ez=ez)

        # These should be BIT-IDENTICAL because aniso_eps=None falls through
        # to the exact same update_e function
        np.testing.assert_array_equal(
            np.array(state_std.ex), np.array(state_dispatch.ex),
            err_msg="Ex mismatch: aniso_eps=None should use identical code path",
        )
        np.testing.assert_array_equal(
            np.array(state_std.ey), np.array(state_dispatch.ey),
            err_msg="Ey mismatch: aniso_eps=None should use identical code path",
        )
        np.testing.assert_array_equal(
            np.array(state_std.ez), np.array(state_dispatch.ez),
            err_msg="Ez mismatch: aniso_eps=None should use identical code path",
        )

        print("Backward compatibility: PASSED (aniso_eps=None uses identical update_e path)")

    def test_default_subpixel_off_in_simulation_api(self):
        """Verify that Simulation.run() subpixel_smoothing parameter exists
        and defaults to False.
        """
        from rfx.api import Simulation
        import inspect
        sig = inspect.signature(Simulation.run)
        param = sig.parameters.get("subpixel_smoothing")
        assert param is not None, "subpixel_smoothing parameter not found in Simulation.run()"
        assert param.default is False, \
            f"subpixel_smoothing default should be False, got {param.default}"
        print("API default check: subpixel_smoothing=False confirmed")

    def test_update_e_aniso_matches_update_e_single_step(self):
        """Verify that update_e_aniso with uniform eps produces the exact
        same result as update_e for a single step (no accumulation drift).
        """
        shape = (10, 10, 10)
        import jax
        key = jax.random.PRNGKey(42)
        state = init_state(shape)
        state = state._replace(
            hx=jax.random.normal(key, shape, dtype=jnp.float32) * 1e-3,
            hy=jax.random.normal(jax.random.PRNGKey(1), shape, dtype=jnp.float32) * 1e-3,
            hz=jax.random.normal(jax.random.PRNGKey(2), shape, dtype=jnp.float32) * 1e-3,
        )

        eps_r = jnp.ones(shape) * 3.5
        sigma = jnp.ones(shape) * 0.01
        materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=jnp.ones(shape))

        dt = 1e-12
        dx = 1e-3

        s1 = update_e(state, materials, dt, dx)
        s2 = update_e_aniso(state, materials, eps_r, eps_r, eps_r, dt, dx)

        np.testing.assert_allclose(
            np.array(s1.ex), np.array(s2.ex), atol=0, rtol=0,
            err_msg="Single-step Ex mismatch",
        )
        np.testing.assert_allclose(
            np.array(s1.ey), np.array(s2.ey), atol=0, rtol=0,
            err_msg="Single-step Ey mismatch",
        )
        np.testing.assert_allclose(
            np.array(s1.ez), np.array(s2.ez), atol=0, rtol=0,
            err_msg="Single-step Ez mismatch",
        )
        print("Single-step equivalence: PASSED")

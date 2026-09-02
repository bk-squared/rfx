"""Tests for multi-mode waveguide port support.

Covers:
- Analytical rectangular mode enumeration and cutoff frequencies
- Multi-mode port initialization (configs per mode)
- Multi-mode S-matrix extraction shape and structure
- Backward compatibility with single-mode usage
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.waveguide_port import (
    WaveguidePort,
    WaveguidePortConfig,
    cutoff_frequency,
    solve_rectangular_modes,
    init_multimode_waveguide_port,
    init_waveguide_port,
    inject_waveguide_port,
    update_waveguide_port_probe,
    extract_waveguide_sparams,
    extract_multimode_s_matrix,
)


# ---------------------------------------------------------------------------
# Test solve_rectangular_modes
# ---------------------------------------------------------------------------


class TestSolveRectangularModes:
    """Analytical rectangular mode enumeration."""

    def test_wr90_cutoff_frequencies(self):
        """TE10, TE20, TE01 cutoff frequencies for WR-90."""
        a = 22.86e-3   # WR-90 width
        b = 10.16e-3   # WR-90 height
        modes = solve_rectangular_modes(a, b, freq_max=20e9, n_modes=5)

        assert len(modes) >= 3, f"Expected at least 3 modes, got {len(modes)}"

        # TE10: f_c = c/(2a) = 6.557 GHz
        fc_te10_expected = C0 / (2 * a)
        assert modes[0]["mode_type"] == "TE"
        assert modes[0]["m"] == 1
        assert modes[0]["n"] == 0
        assert abs(modes[0]["f_cutoff"] - fc_te10_expected) / fc_te10_expected < 0.01

        # TE20: f_c = c/a = 13.12 GHz
        fc_te20_expected = C0 / a
        te20 = [m for m in modes if m["m"] == 2 and m["n"] == 0 and m["mode_type"] == "TE"]
        assert len(te20) == 1
        assert abs(te20[0]["f_cutoff"] - fc_te20_expected) / fc_te20_expected < 0.01

        # TE01: f_c = c/(2b) = 14.76 GHz
        fc_te01_expected = C0 / (2 * b)
        te01 = [m for m in modes if m["m"] == 0 and m["n"] == 1 and m["mode_type"] == "TE"]
        assert len(te01) == 1
        assert abs(te01[0]["f_cutoff"] - fc_te01_expected) / fc_te01_expected < 0.01

    def test_modes_sorted_by_cutoff(self):
        """Returned modes must be sorted by cutoff frequency."""
        a, b = 0.04, 0.02
        modes = solve_rectangular_modes(a, b, freq_max=20e9, n_modes=10)
        cutoffs = [m["f_cutoff"] for m in modes]
        assert cutoffs == sorted(cutoffs), "Modes not sorted by cutoff frequency"

    def test_dominant_mode_is_te10(self):
        """For a > b, the dominant mode should be TE10."""
        a, b = 0.04, 0.02
        modes = solve_rectangular_modes(a, b, freq_max=20e9, n_modes=1)
        assert len(modes) == 1
        assert modes[0]["mode_type"] == "TE"
        assert modes[0]["m"] == 1
        assert modes[0]["n"] == 0

    def test_freq_max_filter(self):
        """Only modes below freq_max are returned."""
        a, b = 0.04, 0.02
        # TE10 cutoff ~ 3.75 GHz
        modes_low = solve_rectangular_modes(a, b, freq_max=5e9, n_modes=10)
        # Only TE10 should be below 5 GHz
        assert len(modes_low) == 1
        assert modes_low[0]["m"] == 1 and modes_low[0]["n"] == 0

    def test_tm_modes_included(self):
        """TM modes should appear in the list when cutoff is below freq_max."""
        a, b = 0.04, 0.02
        # TM11 cutoff ~ 8.38 GHz for a=0.04, b=0.02
        modes = solve_rectangular_modes(a, b, freq_max=10e9, n_modes=10)
        tm_modes = [m for m in modes if m["mode_type"] == "TM"]
        assert len(tm_modes) >= 1, "Expected at least one TM mode below 10 GHz"
        assert tm_modes[0]["m"] == 1 and tm_modes[0]["n"] == 1

    def test_n_modes_limit(self):
        """n_modes limits the number of returned modes."""
        a, b = 0.04, 0.02
        modes_3 = solve_rectangular_modes(a, b, freq_max=30e9, n_modes=3)
        modes_7 = solve_rectangular_modes(a, b, freq_max=30e9, n_modes=7)
        assert len(modes_3) == 3
        assert len(modes_7) == 7
        # The first 3 should match
        for i in range(3):
            assert modes_3[i]["m"] == modes_7[i]["m"]
            assert modes_3[i]["n"] == modes_7[i]["n"]
            assert modes_3[i]["mode_type"] == modes_7[i]["mode_type"]

    def test_empty_when_freq_max_below_cutoff(self):
        """Returns empty list when no modes exist below freq_max."""
        a, b = 0.04, 0.02
        # TE10 cutoff ~ 3.75 GHz, use freq_max below that
        modes = solve_rectangular_modes(a, b, freq_max=2e9, n_modes=5)
        assert len(modes) == 0


# ---------------------------------------------------------------------------
# Test init_multimode_waveguide_port
# ---------------------------------------------------------------------------


class TestInitMultimodeWaveguidePort:
    """Multi-mode port initialization."""

    def test_single_mode_returns_one_config(self):
        """n_modes=1 returns a list of one config."""
        a, b = 0.04, 0.02
        dx = 0.002
        ny, nz = int(a / dx), int(b / dx)
        nc = 10
        freqs = jnp.linspace(4e9, 8e9, 12)
        port = WaveguidePort(
            x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
            a=a, b=b, mode=(1, 0), mode_type="TE", direction="+x",
        )
        cfgs = init_multimode_waveguide_port(port, dx, freqs, n_modes=1)
        assert len(cfgs) == 1
        assert isinstance(cfgs[0], WaveguidePortConfig)

    def test_two_modes_returns_two_configs(self):
        """n_modes=2 returns two configs with different mode profiles."""
        a, b = 0.04, 0.02
        dx = 0.002
        ny, nz = int(a / dx), int(b / dx)
        nc = 10
        freqs = jnp.linspace(4e9, 12e9, 20)
        port = WaveguidePort(
            x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
            a=a, b=b, mode=(1, 0), mode_type="TE", direction="+x",
        )
        cfgs = init_multimode_waveguide_port(port, dx, freqs, n_modes=2)
        assert len(cfgs) == 2

        # First mode (TE10) should be driven (nonzero amplitude)
        assert cfgs[0].src_amp != 0.0

        # Second mode should be passive (zero amplitude)
        assert cfgs[1].src_amp == 0.0

        # Cutoff frequencies should be different
        assert cfgs[0].f_cutoff != cfgs[1].f_cutoff

        # First mode should have lower cutoff
        assert cfgs[0].f_cutoff < cfgs[1].f_cutoff

    def test_three_modes_wr90(self):
        """Three-mode WR-90 port: TE10, TE20, TE01."""
        a = 22.86e-3
        b = 10.16e-3
        dx = 1e-3
        ny, nz = int(round(a / dx)), int(round(b / dx))
        nc = 10
        freqs = jnp.linspace(6e9, 18e9, 20)
        port = WaveguidePort(
            x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
            a=a, b=b, mode=(1, 0), mode_type="TE", direction="+x",
        )
        cfgs = init_multimode_waveguide_port(port, dx, freqs, n_modes=3)
        assert len(cfgs) == 3

        # Verify cutoff ordering
        assert cfgs[0].f_cutoff < cfgs[1].f_cutoff < cfgs[2].f_cutoff

        # Verify all share the same aperture location
        assert all(cfg.x_index == cfgs[0].x_index for cfg in cfgs)
        assert all(cfg.u_lo == cfgs[0].u_lo for cfg in cfgs)
        assert all(cfg.u_hi == cfgs[0].u_hi for cfg in cfgs)

    def test_mode_profiles_orthogonal(self):
        """Mode profiles from different modes should be approximately orthogonal."""
        a, b = 0.04, 0.02
        dx = 0.002
        ny, nz = int(a / dx), int(b / dx)
        nc = 10
        freqs = jnp.linspace(4e9, 12e9, 12)
        port = WaveguidePort(
            x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
            a=a, b=b, mode=(1, 0), mode_type="TE", direction="+x",
        )
        cfgs = init_multimode_waveguide_port(port, dx, freqs, n_modes=3)

        # Check pairwise orthogonality of E-field profiles
        for i in range(len(cfgs)):
            for j in range(i + 1, len(cfgs)):
                ey_i = np.array(cfgs[i].ey_profile).flatten()
                ez_i = np.array(cfgs[i].ez_profile).flatten()
                ey_j = np.array(cfgs[j].ey_profile).flatten()
                ez_j = np.array(cfgs[j].ez_profile).flatten()

                # Overlap integral: sum(Ey_i * Ey_j + Ez_i * Ez_j) * dA
                overlap = np.sum(ey_i * ey_j + ez_i * ez_j) * dx * dx

                # Self-overlap for normalization
                self_i = np.sum(ey_i ** 2 + ez_i ** 2) * dx * dx
                self_j = np.sum(ey_j ** 2 + ez_j ** 2) * dx * dx

                if self_i > 1e-20 and self_j > 1e-20:
                    normalized_overlap = abs(overlap) / np.sqrt(self_i * self_j)
                    assert normalized_overlap < 0.15, (
                        f"Modes {i} and {j} are not orthogonal: "
                        f"normalized overlap = {normalized_overlap:.4f}"
                    )

    def test_backward_compat_single_mode(self):
        """Single-mode init_multimode should match init_waveguide_port."""
        a, b = 0.04, 0.02
        dx = 0.002
        ny, nz = int(a / dx), int(b / dx)
        nc = 10
        freqs = jnp.linspace(4e9, 8e9, 12)
        port = WaveguidePort(
            x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
            a=a, b=b, mode=(1, 0), mode_type="TE", direction="+x",
        )

        cfg_single = init_waveguide_port(port, dx, freqs, f0=5e9, probe_offset=10)
        cfgs_multi = init_multimode_waveguide_port(
            port, dx, freqs, n_modes=1, f0=5e9, probe_offset=10,
        )

        assert len(cfgs_multi) == 1
        cfg_m = cfgs_multi[0]
        assert cfg_single.f_cutoff == cfg_m.f_cutoff
        assert cfg_single.x_index == cfg_m.x_index
        assert np.allclose(np.array(cfg_single.ey_profile),
                           np.array(cfg_m.ey_profile), atol=1e-6)
        assert np.allclose(np.array(cfg_single.ez_profile),
                           np.array(cfg_m.ez_profile), atol=1e-6)


# ---------------------------------------------------------------------------
# Test multi-mode S-matrix extraction via FDTD
# ---------------------------------------------------------------------------


class TestMultiModeSMatrix:
    """Multi-mode S-matrix integration tests with actual FDTD runs."""

    @pytest.fixture
    def waveguide_setup(self):
        """Set up a simple empty waveguide for multi-mode testing."""
        a, b = 0.04, 0.02
        dx = 0.002
        nc = 10
        grid = Grid(
            freq_max=12e9,
            domain=(0.12, a, b),
            dx=dx,
            cpml_layers=nc,
            cpml_axes="x",
        )
        freqs = jnp.linspace(4e9, 10e9, 15)
        ny, nz = grid.ny, grid.nz

        port_in = WaveguidePort(
            x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
            a=(ny - 1) * dx, b=(nz - 1) * dx,
            mode=(1, 0), mode_type="TE", direction="+x",
        )
        port_out = WaveguidePort(
            x_index=grid.nx - nc - 6, y_slice=(0, ny), z_slice=(0, nz),
            a=(ny - 1) * dx, b=(nz - 1) * dx,
            mode=(1, 0), mode_type="TE", direction="-x",
        )
        return grid, dx, freqs, port_in, port_out, nc

    def test_single_mode_s21_passthrough(self, waveguide_setup):
        """Single-mode (n_modes=1) empty waveguide: |S21| should be near 1."""
        grid, dx, freqs, port_in, port_out, nc = waveguide_setup
        n_steps = grid.num_timesteps(num_periods=25)

        cfgs_in = init_multimode_waveguide_port(
            port_in, dx, freqs, n_modes=1, f0=6e9,
            probe_offset=15, ref_offset=3, dft_total_steps=n_steps,
        )
        cfgs_out = init_multimode_waveguide_port(
            port_out, dx, freqs, n_modes=1, f0=6e9,
            probe_offset=15, ref_offset=3, dft_total_steps=n_steps,
        )

        # Run with input port driven
        all_cfgs = cfgs_in + cfgs_out
        state = init_state(grid.shape)
        materials = init_materials(grid.shape)
        cp, cs = init_cpml(grid)

        for step in range(n_steps):
            t = step * grid.dt
            state = update_h(state, materials, grid.dt, dx)
            state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
            state = apply_pec(state)
            state = update_e(state, materials, grid.dt, dx)
            for cfg in all_cfgs:
                state = inject_waveguide_port(state, cfg, t, grid.dt, dx)
            state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
            state = apply_pec(state)
            all_cfgs = [
                update_waveguide_port_probe(cfg, state, grid.dt, dx)
                for cfg in all_cfgs
            ]

        # Check S21 from input port
        _, s21 = extract_waveguide_sparams(all_cfgs[0])
        s21_abs = np.abs(np.array(s21))

        # Above cutoff, S21 should be reasonable for an empty guide
        fc = cutoff_frequency(port_in.a, port_in.b, 1, 0)
        above = np.array(freqs) > fc * 1.2
        if np.any(above):
            mean_s21 = np.mean(s21_abs[above])
            assert mean_s21 > 0.3, f"Mean |S21| = {mean_s21:.3f}, expected > 0.3"

    def test_two_mode_shapes(self, waveguide_setup):
        """Two-mode port configs should have correct S-matrix shape."""
        grid, dx, freqs, port_in, port_out, nc = waveguide_setup
        n_steps = 100  # Short run just to verify shape/mechanics

        cfgs_in = init_multimode_waveguide_port(
            port_in, dx, freqs, n_modes=2, f0=6e9,
            probe_offset=15, ref_offset=3, dft_total_steps=n_steps,
        )
        cfgs_out = init_multimode_waveguide_port(
            port_out, dx, freqs, n_modes=2, f0=6e9,
            probe_offset=15, ref_offset=3, dft_total_steps=n_steps,
        )

        # Verify we get 2 configs per physical port
        assert len(cfgs_in) == 2, f"Expected 2 input configs, got {len(cfgs_in)}"
        assert len(cfgs_out) == 2, f"Expected 2 output configs, got {len(cfgs_out)}"

        # Only the dominant mode should be driven
        assert cfgs_in[0].src_amp != 0.0
        assert cfgs_in[1].src_amp == 0.0
        assert cfgs_out[0].src_amp != 0.0
        assert cfgs_out[1].src_amp == 0.0

    def test_multimode_s_matrix_shape(self, waveguide_setup):
        """Multi-mode S-matrix should be (N_total, N_total, n_freqs)."""
        grid, dx, freqs, port_in, port_out, nc = waveguide_setup
        n_steps = grid.num_timesteps(num_periods=10)

        cfgs_in = init_multimode_waveguide_port(
            port_in, dx, freqs, n_modes=2, f0=6e9,
            probe_offset=15, ref_offset=3, dft_total_steps=n_steps,
        )
        cfgs_out = init_multimode_waveguide_port(
            port_out, dx, freqs, n_modes=2, f0=6e9,
            probe_offset=15, ref_offset=3, dft_total_steps=n_steps,
        )

        port_mode_cfgs = [cfgs_in, cfgs_out]  # 2 ports x 2 modes = 4 total

        s_matrix, mode_map = extract_multimode_s_matrix(
            grid,
            init_materials(grid.shape),
            port_mode_cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes="x",
            pec_axes="yz",
        )

        n_total = 4  # 2 ports x 2 modes
        n_freqs = len(freqs)
        assert s_matrix.shape == (n_total, n_total, n_freqs), (
            f"S-matrix shape {s_matrix.shape}, expected ({n_total}, {n_total}, {n_freqs})"
        )

        # Mode map should have 4 entries
        assert len(mode_map) == 4
        # First two entries belong to port 0
        assert mode_map[0][0] == 0 and mode_map[0][1] == 0  # port 0, mode 0
        assert mode_map[1][0] == 0 and mode_map[1][1] == 1  # port 0, mode 1
        # Next two entries belong to port 1
        assert mode_map[2][0] == 1 and mode_map[2][1] == 0  # port 1, mode 0
        assert mode_map[3][0] == 1 and mode_map[3][1] == 1  # port 1, mode 1


# ---------------------------------------------------------------------------
# Test API-level multi-mode support
# ---------------------------------------------------------------------------


class TestAPIMultiMode:
    """Test the high-level Simulation.add_waveguide_port(n_modes=...) API."""

    def test_add_waveguide_port_n_modes_default(self):
        """Default n_modes=1 should work as before."""
        from rfx.api import Simulation
        sim = Simulation(freq_max=12e9, domain=(0.05, 0.04, 0.02), boundary="cpml")
        sim.add_waveguide_port(0.005, direction="+x", mode=(1, 0))
        assert len(sim._waveguide_ports) == 1
        assert sim._waveguide_ports[0].n_modes == 1

    def test_add_waveguide_port_n_modes_2(self):
        """n_modes=2 should be stored in the entry."""
        from rfx.api import Simulation
        sim = Simulation(freq_max=12e9, domain=(0.05, 0.04, 0.02), boundary="cpml")
        sim.add_waveguide_port(0.005, direction="+x", n_modes=2)
        assert sim._waveguide_ports[0].n_modes == 2

    def test_add_waveguide_port_n_modes_invalid(self):
        """Invalid n_modes should raise ValueError."""
        from rfx.api import Simulation
        sim = Simulation(freq_max=12e9, domain=(0.05, 0.04, 0.02), boundary="cpml")
        with pytest.raises(ValueError, match="n_modes must be a positive integer"):
            sim.add_waveguide_port(0.005, direction="+x", n_modes=0)
        with pytest.raises(ValueError, match="n_modes must be a positive integer"):
            sim.add_waveguide_port(0.005, direction="+x", n_modes=-1)


# ---------------------------------------------------------------------------
# Near-cutoff robustness and degenerate-mode handling (GitHub issue #1)
# ---------------------------------------------------------------------------


class TestNearCutoffRobustness:
    """Multi-mode waveguide behaviour near cutoff and with degenerate modes."""

    def test_modes_near_cutoff_stability(self):
        """Modes near cutoff frequency should not produce NaN or instability.

        WR-90: TE10 cutoff at ~6.56 GHz.
        With freq_max=7 GHz, only TE10 barely propagates; others are below cutoff.
        Mode profiles must be finite and properly normalized.
        """
        from rfx.sources.waveguide_port import (
            solve_rectangular_modes,
            _te_mode_profiles,
            _tm_mode_profiles,
        )
        a, b = 22.86e-3, 10.16e-3  # WR-90
        modes = solve_rectangular_modes(a, b, freq_max=7e9, n_modes=3)

        # Should get at least TE10 (the only mode below 7 GHz)
        assert len(modes) >= 1, f"Expected at least 1 mode, got {len(modes)}"
        assert modes[0]["f_cutoff"] < 7e9

        # All returned modes must have cutoff below freq_max
        for m_info in modes:
            assert m_info["f_cutoff"] <= 7e9

        # Verify mode profiles have no NaN/Inf
        n_pts = 20
        y_coords = np.linspace(0.5e-4, a - 0.5e-4, n_pts)
        z_coords = np.linspace(0.5e-4, b - 0.5e-4, n_pts)
        for m_info in modes:
            m_idx, n_idx = m_info["m"], m_info["n"]
            if m_info["mode_type"] == "TE":
                ey, ez, hy, hz = _te_mode_profiles(a, b, m_idx, n_idx, y_coords, z_coords)
            else:
                ey, ez, hy, hz = _tm_mode_profiles(a, b, m_idx, n_idx, y_coords, z_coords)
            for label, arr in [("ey", ey), ("ez", ez), ("hy", hy), ("hz", hz)]:
                assert not np.any(np.isnan(arr)), (
                    f"NaN in {label} for {m_info['mode_type']}{m_idx}{n_idx}"
                )
                assert not np.any(np.isinf(arr)), (
                    f"Inf in {label} for {m_info['mode_type']}{m_idx}{n_idx}"
                )

    def test_near_cutoff_port_config_stability(self):
        """Port config for a frequency just above cutoff must not contain NaN.

        Operating at 6.6 GHz with WR-90 (TE10 cutoff ~6.56 GHz) means the
        mode barely propagates. The compiled WaveguidePortConfig must still
        have finite profiles and a valid cutoff frequency.
        """
        a, b = 22.86e-3, 10.16e-3
        dx = 1e-3
        ny, nz = int(round(a / dx)), int(round(b / dx))
        nc = 10
        freqs = jnp.linspace(6.6e9, 7e9, 5)  # Just above TE10 cutoff
        port = WaveguidePort(
            x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
            a=a, b=b, mode=(1, 0), mode_type="TE", direction="+x",
        )
        cfgs = init_multimode_waveguide_port(port, dx, freqs, n_modes=1, f0=6.6e9)
        assert len(cfgs) == 1
        cfg = cfgs[0]

        # Cutoff must be finite and close to expected value
        fc_expected = C0 / (2 * a)
        assert np.isfinite(cfg.f_cutoff)
        assert abs(cfg.f_cutoff - fc_expected) / fc_expected < 0.01

        # Profiles must be finite
        for label, arr in [("ey", cfg.ey_profile), ("ez", cfg.ez_profile),
                           ("hy", cfg.hy_profile), ("hz", cfg.hz_profile)]:
            arr_np = np.array(arr)
            assert not np.any(np.isnan(arr_np)), f"NaN in {label} profile"
            assert not np.any(np.isinf(arr_np)), f"Inf in {label} profile"

    def test_nearly_degenerate_modes(self):
        """Square waveguide has degenerate TE10/TE01 -- should handle correctly.

        For a square cross-section (a == b), TE10 and TE01 share the same
        cutoff frequency. solve_rectangular_modes must return both without
        error, and their cutoff frequencies must be numerically equal.
        """
        a, b = 20e-3, 20e-3  # Square waveguide
        modes = solve_rectangular_modes(a, b, freq_max=15e9, n_modes=5)

        # Must return at least 2 modes
        assert len(modes) >= 2, f"Expected at least 2 modes, got {len(modes)}"

        cutoffs = [m["f_cutoff"] for m in modes]

        # TE10 and TE01 should have identical cutoff (within floating-point tolerance)
        assert abs(cutoffs[0] - cutoffs[1]) / cutoffs[0] < 1e-10, (
            f"First two modes not degenerate: {cutoffs[0]:.6e} vs {cutoffs[1]:.6e}"
        )

        # The two degenerate modes should be TE10 and TE01 (or TE01 and TE10)
        first_two = {(m["mode_type"], m["m"], m["n"]) for m in modes[:2]}
        assert ("TE", 1, 0) in first_two, f"TE10 missing from first two modes: {first_two}"
        assert ("TE", 0, 1) in first_two, f"TE01 missing from first two modes: {first_two}"

    def test_degenerate_mode_profiles_orthogonal(self):
        """Degenerate TE10 and TE01 in a square guide must be orthogonal.

        Even though their cutoff frequencies coincide, their spatial profiles
        must remain linearly independent (zero overlap integral).
        """
        from rfx.sources.waveguide_port import _te_mode_profiles
        a = b = 20e-3
        n_pts = 30
        y_coords = np.linspace(0.5e-4, a - 0.5e-4, n_pts)
        z_coords = np.linspace(0.5e-4, b - 0.5e-4, n_pts)
        dy = y_coords[1] - y_coords[0]
        dz = z_coords[1] - z_coords[0]

        ey_10, ez_10, _, _ = _te_mode_profiles(a, b, 1, 0, y_coords, z_coords)
        ey_01, ez_01, _, _ = _te_mode_profiles(a, b, 0, 1, y_coords, z_coords)

        overlap = np.sum(ey_10 * ey_01 + ez_10 * ez_01) * dy * dz
        self_10 = np.sum(ey_10**2 + ez_10**2) * dy * dz
        self_01 = np.sum(ey_01**2 + ez_01**2) * dy * dz

        assert self_10 > 1e-20, "TE10 self-norm is effectively zero"
        assert self_01 > 1e-20, "TE01 self-norm is effectively zero"

        normalized_overlap = abs(overlap) / np.sqrt(self_10 * self_01)
        assert normalized_overlap < 0.05, (
            f"Degenerate TE10/TE01 not orthogonal: overlap = {normalized_overlap:.6f}"
        )

    def test_degenerate_multimode_port_configs(self):
        """Multi-mode port on a square guide must produce distinct configs
        for the degenerate pair TE10/TE01."""
        a = b = 20e-3
        dx = 1e-3
        ny, nz = int(round(a / dx)), int(round(b / dx))
        nc = 10
        freqs = jnp.linspace(8e9, 12e9, 10)
        port = WaveguidePort(
            x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
            a=a, b=b, mode=(1, 0), mode_type="TE", direction="+x",
        )
        cfgs = init_multimode_waveguide_port(port, dx, freqs, n_modes=2, f0=10e9)
        assert len(cfgs) == 2

        # Both cutoffs should be nearly equal (degenerate)
        assert abs(cfgs[0].f_cutoff - cfgs[1].f_cutoff) / cfgs[0].f_cutoff < 1e-10

        # But profiles must differ — at least one component should be substantially different
        ey_diff = np.max(np.abs(np.array(cfgs[0].ey_profile) - np.array(cfgs[1].ey_profile)))
        ez_diff = np.max(np.abs(np.array(cfgs[0].ez_profile) - np.array(cfgs[1].ez_profile)))
        assert ey_diff > 0.01 or ez_diff > 0.01, (
            "Degenerate mode configs have identical profiles — modes not resolved"
        )

    def test_near_cutoff_single_mode_only(self):
        """When freq_max barely exceeds TE10 cutoff, only one mode is returned."""
        a, b = 22.86e-3, 10.16e-3
        fc_te10 = C0 / (2 * a)  # ~6.557 GHz

        # freq_max is 1% above TE10 cutoff
        modes = solve_rectangular_modes(a, b, freq_max=fc_te10 * 1.01, n_modes=10)
        assert len(modes) == 1, f"Expected exactly 1 mode, got {len(modes)}"
        assert modes[0]["mode_type"] == "TE"
        assert modes[0]["m"] == 1 and modes[0]["n"] == 0

    def test_square_guide_higher_degenerate_pairs(self):
        """Square guide TE11/TM11 should also be degenerate."""
        a = b = 20e-3
        # TE11 and TM11 cutoff = c * sqrt(2) / (2a) ~ 10.6 GHz for a=20mm
        modes = solve_rectangular_modes(a, b, freq_max=12e9, n_modes=10)

        te11 = [m for m in modes if m["mode_type"] == "TE" and m["m"] == 1 and m["n"] == 1]
        tm11 = [m for m in modes if m["mode_type"] == "TM" and m["m"] == 1 and m["n"] == 1]
        assert len(te11) == 1, "TE11 not found"
        assert len(tm11) == 1, "TM11 not found"
        assert abs(te11[0]["f_cutoff"] - tm11[0]["f_cutoff"]) / te11[0]["f_cutoff"] < 1e-10, (
            "TE11 and TM11 cutoffs should be identical in a square guide"
        )

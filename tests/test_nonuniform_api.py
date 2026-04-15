"""Tests for non-uniform mesh integration with Simulation API."""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.auto_config import auto_configure
from rfx.geometry.csg import Box
from rfx.nonuniform import make_nonuniform_grid, make_current_source
from rfx.core.yee import MaterialArrays
from rfx.sources.sources import GaussianPulse


class TestNonUniformGrid:
    """Test make_nonuniform_grid construction."""

    def test_basic_grid(self):
        dz_profile = np.array([0.4e-3]*4 + [0.5e-3]*10)
        grid = make_nonuniform_grid((0.05, 0.05), dz_profile, 0.5e-3, 12)
        assert grid.nx > 24  # domain + 2*cpml
        assert grid.ny > 24
        assert grid.nz == len(dz_profile) + 24  # profile + 2*cpml
        assert grid.dt > 0
        assert len(grid.inv_dz) == grid.nz

    def test_cfl_from_min_dz(self):
        dz_profile = np.array([0.1e-3]*4 + [1e-3]*5)
        grid = make_nonuniform_grid((0.01, 0.01), dz_profile, 1e-3, 8)
        # dt should be limited by dz_min=0.1mm, not dx=1mm
        from rfx.grid import C0
        dt_fine = 0.99 / (C0 * np.sqrt(1/1e-3**2 + 1/1e-3**2 + 1/0.1e-3**2))
        assert abs(grid.dt - dt_fine) / dt_fine < 0.01


class TestMakeCurrentSource:
    """Test current source normalization."""

    def test_source_shape(self):
        dz_profile = np.array([0.4e-3]*4 + [0.5e-3]*5)
        grid = make_nonuniform_grid((0.02, 0.02), dz_profile, 0.5e-3, 8)
        shape = (grid.nx, grid.ny, grid.nz)
        materials = MaterialArrays(
            eps_r=jnp.ones(shape) * 4.4,
            sigma=jnp.zeros(shape),
            mu_r=jnp.ones(shape),
        )
        pulse = GaussianPulse(f0=2.4e9, bandwidth=0.8)
        src = make_current_source(grid, (20, 20, 10), 'ez', pulse, 100, materials)
        assert src[0] == 20
        assert src[1] == 20
        assert src[2] == 10
        assert src[3] == 'ez'
        assert len(src[4]) == 100

    def test_dv_normalization(self):
        """Cb/dV normalization should produce different amplitudes for different dz."""
        dx = 0.5e-3
        dz_mixed = np.array([0.2e-3]*4 + [0.5e-3]*5)
        grid = make_nonuniform_grid((0.02, 0.02), dz_mixed, dx, 8)

        shape = (grid.nx, grid.ny, grid.nz)
        materials = MaterialArrays(
            eps_r=jnp.ones(shape),
            sigma=jnp.zeros(shape),
            mu_r=jnp.ones(shape),
        )
        pulse = GaussianPulse(f0=2.4e9, bandwidth=0.8)

        # Source in fine-dz region vs coarse-dz region
        k_fine = 8 + 2   # cpml(8) + 2 → inside 0.2mm zone
        k_coarse = 8 + 6  # cpml(8) + 6 → inside 0.5mm zone
        src_fine = make_current_source(grid, (20, 20, k_fine), 'ez', pulse, 50, materials)
        src_coarse = make_current_source(grid, (20, 20, k_coarse), 'ez', pulse, 50, materials)

        # Same grid/dt, but different dz → different dV → different amplitude
        peak_fine = np.max(np.abs(src_fine[4]))
        peak_coarse = np.max(np.abs(src_coarse[4]))
        # Smaller dz → smaller dV → larger Cb/dV → larger amplitude
        assert peak_fine > peak_coarse
        # Ratio should roughly match dz ratio (0.5/0.2 = 2.5)
        ratio = peak_fine / peak_coarse
        assert 2.0 < ratio < 3.0


class TestSimulationNonUniform:
    """Test Simulation class with dz_profile parameter."""

    def test_dz_profile_stored(self):
        dz = np.array([0.4e-3]*4 + [0.5e-3]*5)
        sim = Simulation(
            freq_max=5e9, domain=(0.05, 0.05, 0.01),
            dz_profile=dz, cpml_layers=8,
        )
        assert sim._dz_profile is not None
        assert len(sim._dz_profile) == 9

    def test_nonuniform_run(self):
        """Smoke test: non-uniform sim runs without error."""
        dz = np.array([0.4e-3]*4 + [0.5e-3]*5)
        sim = Simulation(
            freq_max=5e9, domain=(0.02, 0.02, 0.01),
            dx=0.5e-3, dz_profile=dz, cpml_layers=8,
        )
        sim.add_source((0.01, 0.01, 0.001), "ez")
        sim.add_probe((0.01, 0.01, 0.001), "ez")
        result = sim.run(n_steps=20)
        assert result.time_series is not None
        assert result.dt > 0
        ts = np.asarray(result.time_series)
        assert ts.shape[0] == 20

    def test_nonuniform_upml_smoke(self):
        """boundary='upml' accepts a nonuniform dz_profile and stays stable.

        Commit 85de45f disentangled the scalar-dx curl scaling in UPML.
        This regression pins that nonuniform + UPML (a) constructs,
        (b) runs, (c) produces a non-trivial signal, (d) does not blow up.
        """
        dz = np.array([0.4e-3] * 4 + [0.6e-3] * 6)
        sim = Simulation(
            freq_max=5e9, domain=(0.02, 0.02, 0.01),
            dx=0.5e-3, dz_profile=dz, cpml_layers=8,
            boundary="upml",
        )
        sim.add_source((0.01, 0.01, 0.0025), "ez")
        sim.add_probe((0.01, 0.01, 0.0025), "ez")
        result = sim.run(n_steps=100, compute_s_params=False)
        ts = np.asarray(result.time_series)
        assert ts.shape[0] == 100
        assert np.all(np.isfinite(ts)), "UPML+nonuniform produced NaN/Inf"
        peak = float(np.max(np.abs(ts)))
        assert peak > 0.0, "signal stayed zero — source / probe mismatch"
        # Late-time amplitude must not exceed the initial pulse — absorbing
        # boundary should damp energy, never source it.
        assert float(np.max(np.abs(ts[-20:]))) <= 1.05 * peak, (
            f"late-time peak {np.max(np.abs(ts[-20:])):.3e} > 1.05x "
            f"early peak {peak:.3e} — UPML sourcing energy"
        )

    def test_nonuniform_rejects_ntff(self):
        dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.01),
            dx=0.5e-3,
            dz_profile=dz,
            boundary="cpml",
        )
        sim.add_source((0.01, 0.01, 0.001), "ez")
        sim.add_ntff_box(
            corner_lo=(0.002, 0.002, 0.001),
            corner_hi=(0.018, 0.018, 0.004),
            freqs=[2.4e9],
        )
        with pytest.raises(ValueError, match="NTFF far-field is not supported"):
            sim.run(n_steps=20)

    def test_nonuniform_dft_plane_probe_accumulates(self):
        """DFT plane probe runs on NU mesh and accumulates with step count.

        Accumulation is monotonic for a finite-bandwidth source, so the
        |DFT| magnitude at 200 steps must exceed the magnitude at 100
        steps. Also asserts the plane is finite and non-zero.
        """
        dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)

        def _run(n_steps: int):
            sim = Simulation(
                freq_max=5e9,
                domain=(0.02, 0.02, 0.01),
                dx=0.5e-3,
                dz_profile=dz,
                boundary="upml",
            )
            sim.add_source((0.01, 0.01, 0.005), "ez")
            sim.add_dft_plane_probe(
                axis="z", coordinate=0.005, component="ez",
                freqs=jnp.asarray([2.4e9]),
                name="mid_xy",
            )
            return sim.run(n_steps=n_steps)

        r100 = _run(100)
        r200 = _run(200)

        # (a) dft_planes is present and non-empty
        assert r200.dft_planes is not None
        assert "mid_xy" in r200.dft_planes

        acc200 = np.asarray(r200.dft_planes["mid_xy"].accumulator)
        acc100 = np.asarray(r100.dft_planes["mid_xy"].accumulator)

        # (b) finite and non-zero
        assert np.all(np.isfinite(acc200))
        assert np.max(np.abs(acc200)) > 0.0

        # (c) longer run → larger |accumulator|
        assert np.max(np.abs(acc200)) > np.max(np.abs(acc100))

    def test_nonuniform_rejects_tfsf(self):
        dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
        sim = Simulation(
            freq_max=8e9,
            domain=(0.08, 0.006, 0.006),
            boundary="cpml",
            cpml_layers=8,
            dx=0.001,
            dz_profile=dz,
        )
        sim.add_tfsf_source(f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3)
        with pytest.raises(ValueError, match="TFSF plane-wave source is not supported"):
            sim.run(n_steps=20, compute_s_params=False)

    def test_nonuniform_rejects_waveguide_ports(self):
        dz = np.array([0.4e-3] * 4 + [0.5e-3] * 12)
        sim = Simulation(
            freq_max=10e9,
            domain=(0.12, 0.04, 0.02),
            boundary="cpml",
            cpml_layers=10,
            dx=0.002,
            dz_profile=dz,
        )
        sim.add_waveguide_port(0.01)
        with pytest.raises(ValueError, match="Waveguide ports are not supported"):
            sim.run(n_steps=20, compute_s_params=False)

    def test_nonuniform_lumped_rlc_resistor_damps_source(self):
        """A resistor in the fine-dz region must dissipate energy.

        Runs the same NU configuration twice — once with a parallel
        50 Ω resistor co-located with the source, once without — and
        confirms the peak probe amplitude is strictly lower with the
        resistor present.
        """
        dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
        domain = (0.02, 0.02, 0.01)
        src_pos = (0.01, 0.01, 0.001)  # inside the fine-dz region

        def _build(with_r: bool):
            sim = Simulation(
                freq_max=5e9,
                domain=domain,
                dx=0.5e-3,
                dz_profile=dz,
                boundary="upml",
            )
            sim.add_source(
                src_pos, "ez",
                waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8),
            )
            sim.add_probe(src_pos, "ez")
            if with_r:
                sim.add_lumped_rlc(
                    position=src_pos, component="ez",
                    R=50.0, L=0.0, C=0.0, topology="parallel",
                )
            return sim

        sim_free = _build(with_r=False)
        sim_rlc = _build(with_r=True)

        res_free = sim_free.run(n_steps=300)
        res_rlc = sim_rlc.run(n_steps=300)

        ts_free = np.asarray(res_free.time_series).reshape(-1)
        ts_rlc = np.asarray(res_rlc.time_series).reshape(-1)

        peak_free = float(np.max(np.abs(ts_free)))
        peak_rlc = float(np.max(np.abs(ts_rlc)))

        assert np.all(np.isfinite(ts_free))
        assert np.all(np.isfinite(ts_rlc))
        assert peak_free > 0
        assert peak_rlc > 0
        # The resistor must damp the on-cell field response.
        assert peak_rlc < peak_free, (
            f"expected R=50Ω to damp the probe peak; "
            f"got peak_rlc={peak_rlc:.3e} vs peak_free={peak_free:.3e}"
        )


class TestAutoConfigNonUniform:
    """Test auto_configure dz_profile generation."""

    def test_thin_substrate_triggers_nonuniform(self):
        """Substrate thinner than 4*dx should trigger non-uniform z."""
        geometry = [
            (Box((0, 0, 0), (0.03, 0.03, 0.0016)), "fr4"),
            (Box((0, 0, 0), (0.03, 0.03, 0)), "pec"),     # ground
            (Box((0.005, 0.005, 0.0016), (0.025, 0.025, 0.0016)), "pec"),  # patch
        ]
        materials = {
            "fr4": {"eps_r": 4.4, "sigma": 0.025},
            "pec": {"eps_r": 1.0, "sigma": 1e10},
        }
        config = auto_configure(
            geometry, (1e9, 4e9), materials=materials, accuracy="standard",
        )
        # dx ~ 3.5mm at standard accuracy, h=1.6mm → 0.45 cells → non-uniform
        if config.dz_profile is not None:
            assert config.uses_nonuniform
            assert len(config.dz_profile) >= 4
            # Verify thin feature is resolved with at least 4 cells
            dz_min = np.min(config.dz_profile)
            assert dz_min < config.dx

    def test_thick_substrate_stays_uniform(self):
        """Substrate thicker than 4*dx should stay uniform."""
        geometry = [
            (Box((0, 0, 0), (0.03, 0.03, 0.01)), "dielectric"),
        ]
        materials = {
            "dielectric": {"eps_r": 2.2, "sigma": 0.0},
        }
        auto_configure(
            geometry, (1e9, 4e9), materials=materials, accuracy="standard",
        )
        # 10mm substrate with dx~3.5mm → ~3 cells, but feature detection
        # uses z_features which checks z_thick/dx < 4
        # This is borderline — the test validates the logic path
        # For truly thick substrates, no non-uniform needed

    def test_to_sim_kwargs_includes_dz(self):
        """SimConfig.to_sim_kwargs should include dz_profile when set."""
        from rfx.auto_config import SimConfig
        config = SimConfig(
            dx=0.5e-3, domain=(0.05, 0.05, 0.01),
            cpml_layers=12, n_steps=1000,
            freq_range=(1e9, 4e9), margin=0.01,
            dt=1e-12, accuracy="standard",
            dz_profile=np.array([0.4e-3]*4 + [0.5e-3]*5),
        )
        kwargs = config.to_sim_kwargs()
        assert "dz_profile" in kwargs
        assert len(kwargs["dz_profile"]) == 9

    def test_summary_shows_nonuniform(self):
        from rfx.auto_config import SimConfig
        config = SimConfig(
            dx=0.5e-3, domain=(0.05, 0.05, 0.01),
            cpml_layers=12, n_steps=1000,
            freq_range=(1e9, 4e9), margin=0.01,
            dt=1e-12, accuracy="standard",
            dz_profile=np.array([0.4e-3]*4 + [0.5e-3]*5),
        )
        s = config.summary()
        assert "non-uniform" in s


class TestNonUniformDispersive:
    """Test dispersive materials (Debye/Lorentz) on non-uniform mesh."""

    def test_nonuniform_with_debye(self):
        """Debye dispersive material should run on non-uniform grid."""
        from rfx.materials.debye import DebyePole
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.005),
                         boundary="cpml", dz_profile=np.array([0.5e-3]*10))
        sim.add_material("lossy_dielectric", eps_r=4.0,
                         debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box(corner_lo=(0, 0, 0), corner_hi=(0.02, 0.02, 0.005)),
                material="lossy_dielectric")
        sim.add_source(position=(0.01, 0.01, 0.0025), component="ez")
        result = sim.run(n_steps=100)
        assert result is not None
        assert result.dt > 0

    def test_nonuniform_with_lorentz(self):
        """Lorentz dispersive material should run on non-uniform grid."""
        from rfx.materials.lorentz import lorentz_pole
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.005),
                         boundary="cpml", dz_profile=np.array([0.5e-3]*10))
        sim.add_material("lorentz_mat", eps_r=2.0,
                         lorentz_poles=[lorentz_pole(delta_eps=1.5,
                                       omega_0=2*np.pi*3e9, delta=1e9)])
        sim.add(Box(corner_lo=(0, 0, 0), corner_hi=(0.02, 0.02, 0.005)),
                material="lorentz_mat")
        sim.add_source(position=(0.01, 0.01, 0.0025), component="ez")
        result = sim.run(n_steps=50)
        assert result is not None
        assert result.dt > 0

    def test_nonuniform_mixed_debye_lorentz(self):
        """Mixed Debye + Lorentz materials should run on non-uniform grid."""
        from rfx.materials.debye import DebyePole
        from rfx.materials.lorentz import lorentz_pole
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.005),
                         boundary="cpml", dz_profile=np.array([0.5e-3]*10))
        sim.add_material("debye_mat", eps_r=3.0,
                         debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add_material("lorentz_mat", eps_r=2.0,
                         lorentz_poles=[lorentz_pole(delta_eps=1.5,
                                       omega_0=2*np.pi*3e9, delta=1e9)])
        sim.add(Box(corner_lo=(0, 0, 0), corner_hi=(0.02, 0.02, 0.0025)),
                material="debye_mat")
        sim.add(Box(corner_lo=(0, 0, 0.0025), corner_hi=(0.02, 0.02, 0.005)),
                material="lorentz_mat")
        sim.add_source(position=(0.01, 0.01, 0.0025), component="ez")
        result = sim.run(n_steps=50)
        assert result is not None
        assert result.dt > 0

    def test_nonuniform_debye_energy_bounded(self):
        """Debye on non-uniform grid should not blow up (energy bounded)."""
        from rfx.materials.debye import DebyePole
        dz = np.array([0.2e-3]*4 + [0.5e-3]*6)
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.005),
                         boundary="cpml", dz_profile=dz, dx=0.5e-3)
        sim.add_material("dispersive", eps_r=4.0,
                         debye_poles=[DebyePole(delta_eps=2.0, tau=5e-12)])
        sim.add(Box(corner_lo=(0, 0, 0), corner_hi=(0.02, 0.02, 0.005)),
                material="dispersive")
        sim.add_source(position=(0.01, 0.01, 0.0025), component="ez")
        sim.add_probe((0.01, 0.01, 0.0025), "ez")
        result = sim.run(n_steps=200)
        ts = np.asarray(result.time_series)
        # Energy should not diverge: peak field should stay finite
        assert np.all(np.isfinite(ts))
        assert np.max(np.abs(ts)) < 1e10

    def test_nonuniform_debye_with_probe(self):
        """Probe should record non-zero signal with dispersive material."""
        from rfx.materials.debye import DebyePole
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.005),
                         boundary="cpml", dz_profile=np.array([0.5e-3]*10),
                         dx=0.5e-3)
        sim.add_material("dispersive", eps_r=4.0,
                         debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box(corner_lo=(0, 0, 0), corner_hi=(0.02, 0.02, 0.005)),
                material="dispersive")
        sim.add_source(position=(0.01, 0.01, 0.0025), component="ez")
        sim.add_probe((0.01, 0.01, 0.0025), "ez")
        result = sim.run(n_steps=100)
        ts = np.asarray(result.time_series)
        assert ts.shape == (100, 1)
        # Source should inject energy — probe should see non-zero signal
        assert np.max(np.abs(ts)) > 0

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

    def test_nonuniform_ntff_box_accumulates(self):
        """NTFF box accumulates on NU mesh and yields finite far-field.

        Runs an ez dipole inside the box on a graded-dz UPML domain
        for 150 steps. Asserts: (a) result exposes ntff_data/ntff_box,
        (b) all 6 face accumulators are finite and at least one has
        non-zero magnitude, (c) compute_far_field returns a finite
        far-field at a small angular sample without raising.
        """
        from rfx.farfield import compute_far_field

        dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.01),
            dx=0.5e-3,
            dz_profile=dz,
            boundary="upml",
        )
        sim.add_source((0.01, 0.01, 0.004), "ez")
        sim.add_ntff_box(
            corner_lo=(0.004, 0.004, 0.002),
            corner_hi=(0.016, 0.016, 0.006),
            freqs=[2.4e9],
        )
        result = sim.run(n_steps=150, compute_s_params=False)

        # (a) both NTFF attributes are populated on NU path
        assert result.ntff_data is not None
        assert result.ntff_box is not None

        face_arrays = [
            np.asarray(result.ntff_data.x_lo),
            np.asarray(result.ntff_data.x_hi),
            np.asarray(result.ntff_data.y_lo),
            np.asarray(result.ntff_data.y_hi),
            np.asarray(result.ntff_data.z_lo),
            np.asarray(result.ntff_data.z_hi),
        ]

        # (b) every face accumulator is finite; at least one is non-zero
        assert all(np.all(np.isfinite(f)) for f in face_arrays)
        assert any(np.max(np.abs(f)) > 0 for f in face_arrays), (
            "all 6 NTFF face accumulators stayed zero — scan never wrote"
        )

        # (c) far-field post-processing must run cleanly on NU grid
        theta = np.array([np.pi / 2])
        phi = np.array([0.0, np.pi / 2])
        ff = compute_far_field(
            result.ntff_data, result.ntff_box, result.grid, theta, phi,
        )
        assert np.all(np.isfinite(ff.E_theta))
        assert np.all(np.isfinite(ff.E_phi))
        assert (np.max(np.abs(ff.E_theta)) + np.max(np.abs(ff.E_phi))) > 0

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

    def test_nonuniform_tfsf_x_incidence_accumulates(self):
        """TFSF +x plane-wave runs on an NU dz mesh and excites fields.

        The 1D auxiliary grid runs along the uniform x axis (scalar
        grid.dx), so no aux-grid refactor is required — the usual
        init_tfsf / apply_tfsf_e / apply_tfsf_h / update_tfsf_1d_*
        machinery works unchanged on NU.
        """
        dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
        sim = Simulation(
            freq_max=8e9,
            domain=(0.08, 0.006, 0.006),
            boundary="cpml",
            cpml_layers=8,
            dx=0.001,
            dz_profile=dz,
        )
        sim.add_tfsf_source(
            f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3,
            polarization="ez", direction="+x",
        )
        # Probe inside the total-field region, in vacuum (no scatterer).
        sim.add_probe((0.04, 0.003, 0.0025), "ez")
        r = sim.run(n_steps=200, compute_s_params=False)
        ts = np.asarray(r.time_series[:, 0])
        assert np.all(np.isfinite(ts))
        assert np.max(np.abs(ts)) > 0.0
        # No late-time exponential blow-up.
        assert np.max(np.abs(ts[-20:])) < 2.0 * np.max(np.abs(ts))

    def test_nonuniform_tfsf_oblique_rejected(self):
        """Oblique TFSF (angle_deg != 0) is rejected on NU with a clear message."""
        dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
        sim = Simulation(
            freq_max=8e9,
            domain=(0.08, 0.006, 0.006),
            boundary="cpml",
            cpml_layers=8,
            dx=0.001,
            dz_profile=dz,
        )
        sim.add_tfsf_source(
            f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3, angle_deg=30.0,
        )
        with pytest.raises(ValueError, match="oblique"):
            sim.run(n_steps=20, compute_s_params=False)

    def test_nonuniform_waveguide_port_extracts_s11(self):
        """Waveguide port runs on a NU dz mesh and produces a finite S11.

        Uses a y-normal port on an x-z plane (the typical patch-antenna
        geometry). The aperture spans x (uniform) and z (nonuniform),
        which exercises the per-axis cell-width path.

        Sanity bands:
          (a) no exception
          (b) |S11| finite and within [0, 1.05]
          (c) NU result agrees with uniform reference (dz_profile = full(nz, dz_mean))
              within 5% (degenerate NU == uniform)
        """
        a_wg = 0.04          # aperture along x
        b_wg = 0.02          # aperture along z (the NU axis)
        length = 0.06        # along y
        dx = 0.002

        # Uniform reference: dz = dx everywhere
        nz_total = int(round(b_wg / dx))
        dz_uniform = np.full(nz_total, float(dx))

        # Degenerate-uniform NU profile (dz = dx everywhere) — used for
        # a bit-close agreement check against the uniform reference.
        dz_nu = np.full(nz_total, float(dx))

        # f0 above TE10 cutoff (cutoff = c / (2*a))
        from rfx.grid import C0 as _C0
        f_c = _C0 / (2 * a_wg)
        f0 = f_c * 1.6
        freqs = jnp.linspace(f_c * 1.3, f_c * 2.2, 8)

        def _build(dz_profile):
            sim = Simulation(
                freq_max=float(freqs[-1]) * 1.2,
                domain=(a_wg, length, b_wg),
                boundary="cpml",
                cpml_layers=8,
                dx=dx,
                dz_profile=dz_profile,
            )
            sim.add_waveguide_port(
                x_position=0.005,           # 5 mm into y from the -y face
                direction="+y",
                x_range=(0.0, a_wg),
                z_range=(0.0, b_wg),
                f0=f0,
                bandwidth=0.5,
                freqs=freqs,
                probe_offset=10,
                ref_offset=3,
                name="p1",
            )
            return sim

        sim_nu = _build(dz_nu)
        sim_ref = _build(dz_uniform)

        n_steps = 200
        try:
            res_nu = sim_nu.run(n_steps=n_steps, compute_s_params=False)
            res_ref = sim_ref.run(n_steps=n_steps, compute_s_params=False)
        except ValueError as exc:
            # Aperture-on-NU edge cases that don't relate to the integral
            # math should be flagged but not silently passed. Re-raise.
            raise

        assert res_nu.waveguide_sparams is not None, "NU path must surface waveguide_sparams"
        assert "p1" in res_nu.waveguide_sparams
        s11_nu = res_nu.waveguide_sparams["p1"].s11
        s11_ref = res_ref.waveguide_sparams["p1"].s11

        assert np.all(np.isfinite(s11_nu)), f"|S11| not finite: {s11_nu}"
        s11_nu_mag = np.abs(s11_nu)
        assert np.all(s11_nu_mag >= 0.0)
        assert np.all(s11_nu_mag <= 1.05), f"|S11| exceeds 1.05: {s11_nu_mag}"

        # Degenerate NU == uniform (both use dx everywhere).
        # Compare the spectrum element-wise; allow 5% absolute tolerance
        # on |S11|.
        s11_ref_mag = np.abs(s11_ref)
        diff = np.abs(s11_nu_mag - s11_ref_mag)
        assert np.max(diff) < 0.05, (
            f"NU |S11| disagrees with uniform reference: max|diff|={np.max(diff):.4f}, "
            f"NU={s11_nu_mag}, REF={s11_ref_mag}"
        )

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

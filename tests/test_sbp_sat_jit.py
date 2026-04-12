"""SBP-SAT JIT runner correctness and performance tests.

Tests the jax.lax.scan-based subgridded runner that replaces
the Python-loop runner for 50-100x speedup. Validates:
1. JIT compilation runs without error (PEC and CPML boundaries)
2. Non-trivial field evolution (fields are not stuck at zero)
3. Energy stability over 1000 steps (no divergence)
4. Probe recording works correctly
5. No-probe and no-source edge cases
"""

import numpy as np
import jax.numpy as jnp


def _make_pec_sim(with_probe=True, with_source=True):
    """Small PEC cavity with 2:1 z-axis refinement."""
    from rfx import Simulation
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec")
    if with_source:
        sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_refinement(z_range=(0.015, 0.025), ratio=2)
    if with_probe:
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    return sim


def _make_cpml_sim(with_probe=True):
    """Small CPML domain with 2:1 z-axis refinement."""
    from rfx import Simulation
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="cpml")
    sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_refinement(z_range=(0.015, 0.025), ratio=2)
    if with_probe:
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    return sim


# ── 1. Basic functionality ──────────────────────────────────────

class TestJITBasic:
    """JIT subgridded runner basic functionality."""

    def test_jit_pec_runs_without_error(self):
        """PEC boundary subgridded simulation completes without error."""
        sim = _make_pec_sim()
        result = sim.run(n_steps=50)
        assert result is not None
        assert result.time_series is not None

    def test_jit_cpml_runs_without_error(self):
        """CPML boundary subgridded simulation completes without error."""
        sim = _make_cpml_sim()
        result = sim.run(n_steps=50)
        assert result is not None
        assert result.time_series is not None

    def test_jit_produces_nonzero_fields(self):
        """JIT path should produce non-trivial field evolution."""
        sim = _make_pec_sim()
        result = sim.run(n_steps=200)
        ez_max = float(jnp.max(jnp.abs(result.state.ez)))
        assert ez_max > 1e-10, f"Fields are near-zero: max|Ez|={ez_max}"

    def test_jit_time_series_shape(self):
        """Time series should have shape (n_steps, n_probes)."""
        sim = _make_pec_sim()
        n_steps = 100
        result = sim.run(n_steps=n_steps)
        ts = np.array(result.time_series)
        assert ts.shape == (n_steps, 1), f"Expected ({n_steps}, 1), got {ts.shape}"

    def test_jit_time_series_nonzero(self):
        """Probe should record non-zero values at the source location."""
        sim = _make_pec_sim()
        result = sim.run(n_steps=200)
        ts = np.array(result.time_series).ravel()
        ts_max = np.max(np.abs(ts))
        assert ts_max > 1e-10, f"Probe recorded near-zero: max={ts_max}"


# ── 2. Edge cases ───────────────────────────────────────────────

class TestJITEdgeCases:
    """Edge cases for the JIT subgridded runner."""

    def test_jit_no_probe(self):
        """Simulation without probes should still complete."""
        sim = _make_pec_sim(with_probe=False)
        result = sim.run(n_steps=50)
        assert result is not None
        # time_series should be empty or zeros
        ts = np.array(result.time_series)
        assert ts.size == 0 or np.allclose(ts, 0)

    def test_jit_no_source(self):
        """Simulation without sources should produce zero fields."""
        sim = _make_pec_sim(with_source=False, with_probe=True)
        result = sim.run(n_steps=50)
        ts = np.array(result.time_series).ravel()
        assert np.allclose(ts, 0), "No source should produce zero fields"


# ── 3. Energy stability ────────────────────────────────────────

class TestJITStability:
    """Energy stability tests for JIT subgridded runner."""

    def test_jit_fields_finite_1000_steps(self):
        """Fields must remain finite over 1000 steps."""
        sim = _make_pec_sim()
        result = sim.run(n_steps=1000)
        ts = np.array(result.time_series).ravel()

        assert not np.any(np.isnan(ts)), "NaN detected in time series"
        assert np.all(np.isfinite(ts)), "Inf detected in time series"

    def test_jit_energy_stable_1000_steps(self):
        """Energy must not grow unboundedly over 1000 steps.

        In a PEC cavity with a Gaussian pulse source, the source
        injects energy for the first ~200 steps, then stops. After
        that, the SAT coupling dissipates energy. We check that:
        1. No NaN or Inf
        2. Late-time energy does not exceed peak energy
        """
        sim = _make_pec_sim()
        result = sim.run(n_steps=1000)
        ts = np.array(result.time_series).ravel()
        n = len(ts)

        assert not np.any(np.isnan(ts)), "NaN detected"
        assert np.all(np.isfinite(ts)), "Inf detected"

        # Peak energy anywhere in the trace
        peak_energy = np.max(ts ** 2)
        # Late-time energy
        late_energy = np.max(ts[int(0.8 * n):] ** 2)

        # Late energy should not exceed peak (no unbounded growth)
        assert late_energy <= peak_energy * 1.1, (
            f"Late-time energy growth: late={late_energy:.3e} > "
            f"1.1*peak={1.1*peak_energy:.3e}"
        )

    def test_jit_cpml_fields_finite(self):
        """CPML boundary fields must remain finite over 500 steps."""
        sim = _make_cpml_sim()
        result = sim.run(n_steps=500)
        ts = np.array(result.time_series).ravel()
        assert not np.any(np.isnan(ts)), "NaN detected in CPML time series"
        assert np.all(np.isfinite(ts)), "Inf detected in CPML time series"


# ── 4. Low-level JIT runner tests ──────────────────────────────

class TestJITRunnerDirect:
    """Direct tests of the jit_runner module (bypassing Simulation API)."""

    def test_direct_jit_runner_pec(self):
        """Call run_subgridded_jit directly with PEC grid."""
        from rfx.grid import Grid
        from rfx.core.yee import MaterialArrays, EPS_0, MU_0
        from rfx.subgridding.sbp_sat_3d import SubgridConfig3D
        from rfx.subgridding.jit_runner import run_subgridded_jit

        # Small 15^3 coarse grid, no CPML
        grid_c = Grid(freq_max=5e9, domain=(0.04, 0.04, 0.04),
                       cpml_layers=0)
        nx, ny, nz = grid_c.shape
        dx_c = grid_c.dx
        ratio = 2
        dx_f = dx_c / ratio

        # Fine region in the center
        fi_lo, fi_hi = 4, nx - 4
        fj_lo, fj_hi = 4, ny - 4
        fk_lo, fk_hi = 4, nz - 4
        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
        dt = 0.45 * dx_f / (C0 * np.sqrt(3))

        config = SubgridConfig3D(
            nx_c=nx, ny_c=ny, nz_c=nz, dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(dt), ratio=ratio, tau=0.5,
        )

        shape_c = (nx, ny, nz)
        shape_f = (nx_f, ny_f, nz_f)
        mats_c = MaterialArrays(
            eps_r=jnp.ones(shape_c, dtype=jnp.float32),
            sigma=jnp.zeros(shape_c, dtype=jnp.float32),
            mu_r=jnp.ones(shape_c, dtype=jnp.float32),
        )
        mats_f = MaterialArrays(
            eps_r=jnp.ones(shape_f, dtype=jnp.float32),
            sigma=jnp.zeros(shape_f, dtype=jnp.float32),
            mu_r=jnp.ones(shape_f, dtype=jnp.float32),
        )

        # Source waveform: Gaussian pulse
        n_steps = 100
        times = np.arange(n_steps) * dt
        f0 = 3e9
        waveform = np.exp(-((times - 3 / f0) * f0) ** 2) * np.sin(
            2 * np.pi * f0 * times
        )
        si, sj, sk = nx_f // 2, ny_f // 2, nz_f // 2
        sources_f = [(si, sj, sk, "ez", waveform.astype(np.float32))]
        probe_indices_f = [(si, sj, sk)]
        probe_components = ["ez"]

        result = run_subgridded_jit(
            grid_c, mats_c, mats_f, config, n_steps,
            sources_f=sources_f,
            probe_indices_f=probe_indices_f,
            probe_components=probe_components,
        )

        assert result.time_series.shape == (n_steps, 1)
        ts = np.array(result.time_series).ravel()
        assert not np.any(np.isnan(ts)), "NaN in direct JIT runner output"
        assert np.max(np.abs(ts)) > 1e-10, "Fields should be non-zero"

    def test_direct_jit_runner_no_probes_no_sources(self):
        """JIT runner with no sources and no probes should return zeros."""
        from rfx.grid import Grid
        from rfx.core.yee import MaterialArrays, EPS_0, MU_0
        from rfx.subgridding.sbp_sat_3d import SubgridConfig3D
        from rfx.subgridding.jit_runner import run_subgridded_jit

        grid_c = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                       cpml_layers=0)
        nx, ny, nz = grid_c.shape
        dx_c = grid_c.dx
        ratio = 2
        dx_f = dx_c / ratio

        fi_lo, fi_hi = 3, nx - 3
        fj_lo, fj_hi = 3, ny - 3
        fk_lo, fk_hi = 3, nz - 3
        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
        dt = 0.45 * dx_f / (C0 * np.sqrt(3))

        config = SubgridConfig3D(
            nx_c=nx, ny_c=ny, nz_c=nz, dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(dt), ratio=ratio, tau=0.5,
        )

        shape_c = (nx, ny, nz)
        shape_f = (nx_f, ny_f, nz_f)
        mats_c = MaterialArrays(
            eps_r=jnp.ones(shape_c, dtype=jnp.float32),
            sigma=jnp.zeros(shape_c, dtype=jnp.float32),
            mu_r=jnp.ones(shape_c, dtype=jnp.float32),
        )
        mats_f = MaterialArrays(
            eps_r=jnp.ones(shape_f, dtype=jnp.float32),
            sigma=jnp.zeros(shape_f, dtype=jnp.float32),
            mu_r=jnp.ones(shape_f, dtype=jnp.float32),
        )

        result = run_subgridded_jit(
            grid_c, mats_c, mats_f, config, 20,
        )

        # No sources => all fields should be zero
        assert float(jnp.max(jnp.abs(result.state_f.ez))) == 0.0


class TestJITRunnerHCoupling:
    """Verify that the JIT runner applies H-field SAT coupling."""

    def test_jit_runner_h_coupling_energy(self):
        """JIT runner energy should track standalone stepper (which has H-coupling).

        If JIT is missing H-coupling, energy diverges from the standalone
        reference by orders of magnitude.
        """
        from rfx.subgridding.sbp_sat_3d import (
            SubgridConfig3D, init_subgrid_3d, step_subgrid_3d,
            compute_energy_3d,
        )
        from rfx.subgridding.jit_runner import run_subgridded_jit
        from rfx.grid import Grid
        from rfx.core.yee import MaterialArrays, EPS_0, MU_0

        # --- Config: 20^3 coarse, fine region 7-13, ratio=3 ---
        shape_c = (20, 20, 20)
        dx_c = 0.003
        ratio = 3
        fine_region = (7, 13, 7, 13, 7, 13)
        fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = fine_region
        dx_f = dx_c / ratio
        C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
        dt = 0.45 * dx_f / (C0 * np.sqrt(3))

        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        config = SubgridConfig3D(
            nx_c=20, ny_c=20, nz_c=20, dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(dt), ratio=ratio, tau=0.5,
        )

        n_steps = 200
        times = np.arange(n_steps) * dt
        f0 = 3e9
        waveform = np.exp(-((times - 3 / f0) * f0) ** 2) * np.sin(
            2 * np.pi * f0 * times
        )

        si, sj, sk = nx_f // 2, ny_f // 2, nz_f // 2

        # --- JIT runner ---
        shape_f = (nx_f, ny_f, nz_f)
        mats_c = MaterialArrays(
            eps_r=jnp.ones(shape_c, dtype=jnp.float32),
            sigma=jnp.zeros(shape_c, dtype=jnp.float32),
            mu_r=jnp.ones(shape_c, dtype=jnp.float32),
        )
        mats_f = MaterialArrays(
            eps_r=jnp.ones(shape_f, dtype=jnp.float32),
            sigma=jnp.zeros(shape_f, dtype=jnp.float32),
            mu_r=jnp.ones(shape_f, dtype=jnp.float32),
        )

        # Build a Grid that matches the 20^3 config
        grid_c_custom = Grid.__new__(Grid)
        grid_c_custom.__dict__.update({
            '_dx': dx_c, '_dy': dx_c, '_dz': dx_c,
            '_nx': 20, '_ny': 20, '_nz': 20,
            'cpml_layers': 0, 'dt': dt,
        })
        grid_c_custom.shape = shape_c

        sources_f = [(si, sj, sk, "ez", waveform.astype(np.float32))]

        result_jit = run_subgridded_jit(
            grid_c_custom, mats_c, mats_f, config, n_steps,
            sources_f=sources_f,
        )

        # Compute JIT final energy (sum of squared fields)
        jit_e_sq = float(
            jnp.sum(result_jit.state_f.ex ** 2 +
                     result_jit.state_f.ey ** 2 +
                     result_jit.state_f.ez ** 2 +
                     result_jit.state_f.hx ** 2 +
                     result_jit.state_f.hy ** 2 +
                     result_jit.state_f.hz ** 2)
        )

        # --- Standalone stepper (has H-coupling) ---
        config_sa, state_sa = init_subgrid_3d(
            shape_c=shape_c, dx_c=dx_c,
            fine_region=fine_region, ratio=ratio,
            courant=0.45, tau=0.5,
        )

        for step in range(n_steps):
            state_sa = step_subgrid_3d(state_sa, config_sa)
            # Inject source AFTER stepping (matches JIT injection order:
            # sources are added after E-coupling in the JIT scan body)
            state_sa = state_sa._replace(
                ez_f=state_sa.ez_f.at[si, sj, sk].add(
                    float(waveform[step]))
            )

        sa_e_sq = float(
            jnp.sum(state_sa.ex_f ** 2 + state_sa.ey_f ** 2 +
                     state_sa.ez_f ** 2 + state_sa.hx_f ** 2 +
                     state_sa.hy_f ** 2 + state_sa.hz_f ** 2)
        )

        # --- Assertions ---
        # 1. JIT energy should be finite and positive
        assert np.isfinite(jit_e_sq), f"JIT energy is not finite: {jit_e_sq}"
        assert jit_e_sq > 0, f"JIT energy should be positive: {jit_e_sq}"

        # 2. Standalone energy should also be positive
        assert sa_e_sq > 0, (
            f"Standalone energy should be positive: {sa_e_sq}"
        )

        # 3. With H-coupling in both, energies should be in the same
        #    ballpark. Without H-coupling, JIT energy diverges by
        #    orders of magnitude. Allow 100x tolerance.
        ratio_energy = max(jit_e_sq, sa_e_sq) / (min(jit_e_sq, sa_e_sq) + 1e-30)
        assert ratio_energy < 100, (
            f"JIT energy diverges from standalone: JIT={jit_e_sq:.3e}, "
            f"SA={sa_e_sq:.3e}, ratio={ratio_energy:.1f}x — "
            f"H-coupling likely missing in JIT runner"
        )


class TestSubgridMaterialTransition:
    """Validate subgridding with dielectric material crossing the coarse-fine boundary."""

    def test_dielectric_crossing_boundary_stable(self):
        """Dielectric box straddling refinement boundary should not cause NaN or divergence."""
        from rfx import Simulation, Box, GaussianPulse

        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                         boundary="pec", dx=0.003)
        sim.add_material("dielectric", eps_r=4.0)
        # Box crosses the refinement z-boundary (0.009-0.021 vs refine 0.009-0.021)
        sim.add(Box((0.005, 0.005, 0.005), (0.020, 0.020, 0.020)),
                material="dielectric")
        sim.add_source(position=(0.015, 0.015, 0.015), component="ez",
                       waveform=GaussianPulse(f0=2e9, bandwidth=0.5))
        sim.add_probe(position=(0.015, 0.015, 0.015), component="ez")
        sim.add_refinement(z_range=(0.009, 0.021), ratio=2)

        result = sim.run(n_steps=200)
        ts = np.array(result.time_series[:, 0])

        assert not np.any(np.isnan(ts)), "NaN in material-transition subgrid"
        assert np.max(np.abs(ts)) > 0, "Zero signal with dielectric"

    def test_dielectric_changes_field_amplitude(self):
        """Dielectric material should produce different field amplitudes vs vacuum.

        With eps_r=4, the wave impedance and field amplitudes change.
        A co-located probe should see a measurably different signal.
        """
        from rfx import Simulation, Box, GaussianPulse

        domain = (0.03, 0.03, 0.03)
        dx = 0.003

        def _run_with_eps(eps_r):
            sim = Simulation(freq_max=5e9, domain=domain, boundary="pec", dx=dx)
            if eps_r > 1.0:
                sim.add_material("diel", eps_r=eps_r)
                sim.add(Box((0, 0, 0), domain), material="diel")
            sim.add_source((0.015, 0.015, 0.015), "ez",
                           waveform=GaussianPulse(f0=2e9, bandwidth=0.5))
            sim.add_probe((0.015, 0.015, 0.015), "ez")
            sim.add_refinement(z_range=(0.009, 0.021), ratio=2)
            return sim.run(n_steps=100)

        res_vac = _run_with_eps(1.0)
        res_die = _run_with_eps(4.0)

        ts_vac = np.array(res_vac.time_series[:, 0])
        ts_die = np.array(res_die.time_series[:, 0])

        # Signals should differ (different wave impedance in dielectric)
        diff = np.max(np.abs(ts_vac - ts_die))
        ref = np.max(np.abs(ts_vac)) + 1e-30
        rel_diff = diff / ref

        assert rel_diff > 0.01, (
            f"Dielectric should change signal: rel_diff={rel_diff:.4f}"
        )

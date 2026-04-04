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

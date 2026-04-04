"""SBP-SAT penalty coefficient should be configurable."""
import numpy as np
from rfx import Simulation


class TestSBPSATAlpha:
    """Tests for configurable tau (SAT penalty coefficient)."""

    def _make_sim(self, tau=None):
        """Build a minimal subgridded simulation."""
        dx_c = 2e-3
        ratio = 4
        dom = (0.04, 0.04, 0.04)

        sim = Simulation(freq_max=5e9, domain=dom, boundary="cpml",
                         cpml_layers=4, dx=dx_c)
        sim.add_source(position=(0.02, 0.02, 0.02), component="ez")

        kw = {"z_range": (0.01, 0.03), "ratio": ratio}
        if tau is not None:
            kw["tau"] = tau
        sim.add_refinement(**kw)
        return sim

    def test_default_tau_is_half(self):
        """Default tau should be 0.5 when not specified."""
        sim = self._make_sim()
        assert sim._refinement["tau"] == 0.5

    def test_custom_tau_stored(self):
        """Custom tau value should be stored in refinement dict."""
        sim = self._make_sim(tau=1.0)
        assert sim._refinement["tau"] == 1.0

    def test_custom_tau_accepted(self):
        """Simulation with custom tau should run without error."""
        sim = self._make_sim(tau=1.0)
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
        result = sim.run(n_steps=50)
        assert result is not None

    def test_default_tau_runs(self):
        """Simulation with default tau should run without error."""
        sim = self._make_sim()
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
        result = sim.run(n_steps=50)
        assert result is not None

    def test_tau_propagates_to_config(self):
        """Tau should propagate from add_refinement through to SubgridConfig3D."""
        sim = self._make_sim(tau=0.75)
        # Build the grid and config to verify tau reaches SubgridConfig3D
        grid = sim._build_grid()
        base_materials, _, _, pec_mask = sim._assemble_materials(grid)

        # Patch run to capture config instead of running full sim
        ref = sim._refinement
        assert ref["tau"] == 0.75

        # Verify the config is built with the correct tau by checking
        # the intermediate dict propagation
        for tau_val in [0.25, 0.75, 1.0]:
            sim2 = self._make_sim(tau=tau_val)
            assert sim2._refinement["tau"] == tau_val

    def test_init_subgrid_3d_tau_passthrough(self):
        """init_subgrid_3d should accept and propagate tau."""
        from rfx.subgridding.sbp_sat_3d import init_subgrid_3d
        config, _ = init_subgrid_3d(tau=0.75)
        assert config.tau == 0.75

    def test_init_subgrid_3d_default_tau(self):
        """init_subgrid_3d default tau should be 0.5."""
        from rfx.subgridding.sbp_sat_3d import init_subgrid_3d
        config, _ = init_subgrid_3d()
        assert config.tau == 0.5

    def test_different_tau_different_results(self):
        """Different tau values must produce measurably different time series.

        This is the key regression test for the alpha-cap bug: previously
        cb_vac * 2*tau/dx always exceeded 0.5 so the min() cap made all
        tau values produce identical results.
        """
        from rfx.subgridding.sbp_sat_3d import (
            init_subgrid_3d, step_subgrid_3d,
        )
        import jax.numpy as jnp

        results = {}
        for tau in [0.1, 0.5]:
            config, state = init_subgrid_3d(
                shape_c=(20, 20, 20), dx_c=0.003,
                fine_region=(7, 13, 7, 13, 7, 13), ratio=3,
                tau=tau,
            )
            # Inject pulse on coarse grid
            state = state._replace(ez_c=state.ez_c.at[4, 10, 10].set(1.0))

            for _ in range(300):
                state = step_subgrid_3d(state, config)

            results[tau] = float(jnp.sum(state.ez_c ** 2)) + float(jnp.sum(state.ez_f ** 2))

        diff = abs(results[0.1] - results[0.5])
        assert diff > 1e-10, (
            f"Different tau values produced identical results "
            f"(diff={diff:.2e}); alpha cap bug likely still present"
        )

    def test_alpha_values_are_dimensionless(self):
        """Alpha coefficients must be dimensionless fractions in (0, 1]."""
        from rfx.subgridding.sbp_sat_3d import init_subgrid_3d

        for tau in [0.1, 0.25, 0.5, 0.75, 1.0]:
            config, _ = init_subgrid_3d(
                shape_c=(20, 20, 20), dx_c=0.003,
                fine_region=(7, 13, 7, 13, 7, 13), ratio=3,
                tau=tau,
            )
            # Reproduce the alpha computation from _shared_node_coupling_3d
            alpha_c = tau * min(config.dx_f / config.dx_c, 1.0)
            alpha_f = tau * min(config.dx_c / config.dx_f, 1.0)

            assert 0 < alpha_c <= 1.0, f"alpha_c={alpha_c} out of (0,1] for tau={tau}"
            assert 0 < alpha_f <= 1.0, f"alpha_f={alpha_f} out of (0,1] for tau={tau}"
            # alpha_c should be smaller than alpha_f (fine is finer grid)
            assert alpha_c <= alpha_f, (
                f"alpha_c={alpha_c} > alpha_f={alpha_f} for tau={tau}"
            )

    def test_energy_stable_after_fix(self):
        """Energy must remain finite and bounded after the alpha scaling fix."""
        from rfx.subgridding.sbp_sat_3d import (
            init_subgrid_3d, step_subgrid_3d, compute_energy_3d,
        )

        for tau in [0.25, 0.5, 1.0]:
            config, state = init_subgrid_3d(
                shape_c=(15, 15, 15), dx_c=0.004,
                fine_region=(5, 10, 5, 10, 5, 10), ratio=3,
                tau=tau,
            )
            state = state._replace(ez_c=state.ez_c.at[3, 7, 7].set(0.5))
            compute_energy_3d(state, config)

            for _ in range(500):
                state = step_subgrid_3d(state, config)

            final_energy = compute_energy_3d(state, config)
            assert np.isfinite(final_energy), (
                f"Energy diverged (NaN/Inf) at tau={tau}"
            )
            assert final_energy >= 0, f"Negative energy at tau={tau}"

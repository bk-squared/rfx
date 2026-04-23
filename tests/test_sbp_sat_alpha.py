"""SAT penalty coefficient tests for the canonical Phase-1 lane."""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.subgridding.sbp_sat_3d import (
    compute_energy_3d,
    init_subgrid_3d,
    phase1_3d_dt,
    sat_penalty_coefficients,
    step_subgrid_3d,
)


class TestSBPSATAlpha:
    def _make_sim(self, tau=None):
        sim = Simulation(
            freq_max=5e9,
            domain=(0.04, 0.04, 0.04),
            boundary="pec",
            dx=2e-3,
        )
        sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
        kw = {"z_range": (0.01, 0.03), "ratio": 2}
        if tau is not None:
            kw["tau"] = tau
        sim.add_refinement(**kw)
        return sim

    def test_default_tau_is_half(self):
        sim = self._make_sim()
        assert sim._refinement["tau"] == 0.5

    def test_custom_tau_stored(self):
        sim = self._make_sim(tau=1.0)
        assert sim._refinement["tau"] == 1.0

    def test_init_subgrid_3d_tau_passthrough(self):
        config, _ = init_subgrid_3d(tau=0.75)
        assert config.tau == 0.75

    def test_init_subgrid_3d_default_tau(self):
        config, _ = init_subgrid_3d()
        assert config.tau == 0.5

    def test_init_subgrid_3d_uses_phase1_dt_rule(self):
        config, _ = init_subgrid_3d(shape_c=(8, 8, 10), dx_c=0.004, ratio=2)
        assert np.isclose(config.dt, phase1_3d_dt(config.dx_f))

    def test_init_subgrid_3d_rejects_noncanonical_courant(self):
        with pytest.raises(ValueError, match="canonical near-fine-CFL rule"):
            init_subgrid_3d(courant=0.45)

    def test_alpha_values_follow_canonical_formula(self):
        for tau in [0.1, 0.25, 0.5, 0.75, 1.0]:
            alpha_c, alpha_f = sat_penalty_coefficients(ratio=3, tau=tau)
            assert np.isclose(alpha_c, tau / 4.0)
            assert np.isclose(alpha_f, tau * 3.0 / 4.0)
            assert 0 < alpha_c <= alpha_f <= 1.0

    def test_different_tau_different_results(self):
        results = {}
        for tau in [0.1, 0.5]:
            config, state = init_subgrid_3d(
                shape_c=(8, 8, 10),
                dx_c=0.004,
                fine_region=(0, 8, 0, 8, 3, 7),
                ratio=2,
                tau=tau,
            )
            state = state._replace(ez_c=state.ez_c.at[4, 4, 2].set(1.0))
            for _ in range(200):
                state = step_subgrid_3d(state, config)
            results[tau] = float(jnp.sum(state.ez_c ** 2)) + float(jnp.sum(state.ez_f ** 2))

        assert abs(results[0.1] - results[0.5]) > 1e-8

    def test_energy_stable_after_fix(self):
        for tau in [0.25, 0.5, 1.0]:
            config, state = init_subgrid_3d(
                shape_c=(8, 8, 10),
                dx_c=0.004,
                fine_region=(0, 8, 0, 8, 3, 7),
                ratio=2,
                tau=tau,
            )
            state = state._replace(ez_c=state.ez_c.at[3, 3, 2].set(0.5))
            for _ in range(300):
                state = step_subgrid_3d(state, config)
            final_energy = compute_energy_3d(state, config)
            assert np.isfinite(final_energy)
            assert final_energy >= 0

"""Grid selection must precede inspection, assembly and lane eligibility."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation


def _board():
    sim = Simulation(freq_max=10e9, domain=(.002, .002, .004), boundary="pec")
    sim.add(Box((0, 0, .001), (.002, .002, .002)), material="fr4")
    return sim


def test_new_uniform_consumer_cannot_silently_build_surrogate_grid():
    sim = _board()
    with pytest.raises(NotImplementedError, match="resolved non-uniform"):
        sim._build_grid()
    assert sim._build_realized_grid().shape == sim._build_nonuniform_grid().shape


def test_mesh_planning_and_export_use_solver_timestep_and_domain():
    from rfx.artifacts import _mesh_summary
    from rfx.interop import design_to_dict, simulation_from_design

    sim = _board()
    grid = sim._build_realized_grid()
    assert sim._mesh_planner_state()["dt"] == float(grid.dt)
    summary = _mesh_summary(sim, n_steps=None, available_memory_gb=None)
    assert tuple(summary["grid_shape"]) == grid.shape
    assert summary["dt"] == float(grid.dt)
    design = design_to_dict(sim)
    assert design["mesh"]["dx"] == sim._dx
    assert tuple(design["domain"]["extent"]) == sim._domain
    rebuilt = simulation_from_design(design)
    other = rebuilt._build_realized_grid()
    assert grid.shape == other.shape
    np.testing.assert_array_equal(grid.dz, other.dz)
    assert "NonUniformGrid" in repr(sim)


def test_vmap_routes_before_uniform_assembly(monkeypatch):
    module = importlib.import_module("rfx.vmap_sweep")
    sim = _board()
    calls = []

    def fallback(sim, name, values, *, n_steps):
        calls.append((sim._build_realized_grid().shape, n_steps))
        return "sequential"

    monkeypatch.setattr(module, "_sequential_fallback", fallback)
    with pytest.warns(UserWarning, match="Falling back to sequential"):
        assert module.vmap_material_sweep(
            sim, "fr4.eps_r", [3., 4.], n_steps=2) == "sequential"
    assert calls == [(sim._build_nonuniform_grid().shape, 2)]


def test_uniform_only_topology_refuses_realized_nu_before_optimization():
    from rfx.topology import topology_optimize

    with pytest.raises(NotImplementedError, match="topology_optimize.*non-uniform"):
        topology_optimize(_board(), None, None, skip_preflight=True)


def test_gradient_check_baseline_uses_forward_grid():
    from rfx.optimize import gradient_check

    sim = _board()
    sim.add_source((.001, .001, .0015), amplitude_kind="field")
    sim.add_probe((.001, .001, .0015))
    check = gradient_check(
        sim, jnp.array(0.), lambda r: jnp.sum(r.time_series ** 2),
        n_steps=3, eps=.01,
    )
    assert np.isfinite(check.relative_error)
    assert np.abs(np.asarray(check.ad_grad)) > 0
    np.testing.assert_allclose(check.ad_grad, check.fd_grad, rtol=.02)


def test_optimize_builds_design_region_on_auto_mesh():
    from rfx.optimize import DesignRegion, optimize

    sim = _board()
    sim.add_source((.001, .001, .0015), amplitude_kind="field")
    sim.add_probe((.001, .001, .0015))
    region = DesignRegion((.0005, .0005, .001), (.0015, .0015, .002))
    result = optimize(sim, region, lambda r: jnp.sum(r.time_series ** 2),
                      n_iters=1, n_steps=3, verbose=False, skip_preflight=True)
    assert len(result.loss_history) == 1
    assert np.isfinite(result.loss_history[0])
    assert all(size > 0 for size in result.eps_design.shape)

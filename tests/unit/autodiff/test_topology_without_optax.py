"""Topology's built-in Adam works without the optional optax package."""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.topology import TopologyDesignRegion, topology_optimize


def test_topology_optimizes_without_optax(monkeypatch):
    monkeypatch.setitem(sys.modules, "optax", None)
    dx = 299792458.0 / 5e9 / 20.0
    sim = Simulation(freq_max=5e9, domain=(6 * dx,) * 3, dx=dx, boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    region = TopologyDesignRegion(
        corner_lo=(0.006, 0.006, 0.006), corner_hi=(0.009, 0.009, 0.009),
        material_bg="air", material_fg="fr4",
    )
    initial = jax.random.uniform(jax.random.PRNGKey(42), (2, 2, 2), minval=0.4, maxval=0.6)
    result = topology_optimize(
        sim, region, lambda run: -jnp.sum(run.time_series ** 2),
        init_density=initial, n_iterations=5, learning_rate=0.05,
        beta_schedule=[(0, 1.0)], verbose=False, skip_preflight=True,
    )
    assert len(result.history) == 5
    assert np.all(np.isfinite(result.history))
    assert result.history[-1] < result.history[0]
    assert np.all(np.isfinite(result.density))
    assert np.all((result.density > 0) & (result.density < 1))
    assert not np.array_equal(result.density, initial)

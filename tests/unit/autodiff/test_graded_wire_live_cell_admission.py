"""A traced mesh must not reach Python truth conversion of PEC edge masks."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.optimize import DesignRegion, optimize


def _board(delta):
    dz = jnp.full(8, .001).at[3].add(delta).at[4].add(-delta)
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008),
                     dx=.001, boundary="pec", dz_profile=dz)
    sim.add(Box((.001, .001, .001), (.003, .003, .003)), material="pec")
    sim.add_port((.004, .004, .002), "ez", impedance=50., extent=.002)
    sim.add_probe((.004, .004, .005), "ez")
    return sim


def _observable(delta):
    return jnp.sum(_board(delta).forward(n_steps=4, skip_preflight=True).time_series**2)


@pytest.mark.parametrize("route", ["jit", "vmap"])
def test_traced_wire_conductor_live_cells_refuse(route):
    with pytest.raises(NotImplementedError, match="graded wire port.*conductor.*concrete mesh") as exc:
        if route == "jit":
            jax.jit(_observable)(jnp.float32(0.))
        else:
            jax.vmap(_observable)(jnp.zeros(2))
    assert isinstance(exc.value, jax.errors.TracerArrayConversionError)
    assert "eager jax.jvp/jax.grad" in str(exc.value)
    assert "optimize(jit=False)" in str(exc.value)


@pytest.mark.parametrize("route", ["jvp", "grad"])
def test_eager_wire_mesh_derivative(route):
    value = (jax.jvp(_observable, (jnp.float32(0.),), (jnp.float32(1.),))
             if route == "jvp" else jax.grad(_observable)(jnp.float32(0.)))
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(value))


@pytest.mark.parametrize("jit", [False, True])
def test_optimize_wire_conductor_mesh(jit):
    # The nested simulation's mesh is a design variable on the objective tape.
    objective = lambda result: _observable(jnp.sum(result.time_series) * 1e-6)
    kwargs = dict(n_iters=2, n_steps=4, verbose=False, skip_preflight=True, jit=jit)
    region = DesignRegion((.005, .005, .002), (.007, .007, .004))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = optimize(_board(0.), region, objective, **kwargs)
    assert len(result.loss_history) == 2
    assert np.isfinite(result.loss_history).all()
    fallback = [r for r in records if "every iteration now recompiles" in str(r.message)]
    assert len(fallback) == int(jit)
    if jit:
        assert "TracedWireLiveCellsError" in str(fallback[0].message)

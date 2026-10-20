"""PEC shape classification names unsupported traced topology decisions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere


def _shape_run(value, kind, smoothing, graded):
    sim = Simulation(
        freq_max=10e9, domain=(.008, .008, .008), dx=.001, boundary="pec",
        **({"dz_profile": np.full(8, .001)} if graded else {}))
    shape = (Box((.002, .002, .002), (.005 + value, .005, .005))
             if kind == "Box" else Sphere((.004, .004, .004), .002 + value))
    sim.add(shape, material="pec")
    return jnp.sum(sim.forward(n_steps=2, skip_preflight=True,
                              subpixel_smoothing=smoothing).time_series**2)


@pytest.mark.parametrize("kind,dimension", [("Box", "extent"), ("Sphere", "radius")])
@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("route", ["jit", "vmap"])
def test_traced_pec_dimension_refuses(kind, dimension, smoothing, graded, route):
    fn = lambda value: _shape_run(value, kind, smoothing, graded)
    with pytest.raises(NotImplementedError, match=f"{kind}.*traced {dimension}.*concrete"):
        if route == "jit":
            jax.jit(fn)(jnp.float32(0.))
        else:
            jax.vmap(fn)(jnp.zeros(2))


@pytest.mark.parametrize("kind", ["Box", "Sphere"])
@pytest.mark.parametrize("route", ["jvp", "grad"])
def test_pec_shape_derivative_keeps_existing_refusal(kind, route):
    fn = lambda value: _shape_run(value, kind, False, False)
    with pytest.raises(NotImplementedError, match="gradient.*declared shape dimension"):
        if route == "jvp":
            jax.jvp(fn, (jnp.float32(0.),), (jnp.float32(1.),))
        else:
            jax.grad(fn)(jnp.float32(0.))

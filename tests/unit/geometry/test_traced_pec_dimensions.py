"""PEC shape classification names unsupported traced topology decisions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere


def _shape_run(value, kind, smoothing, graded, material="pec"):
    sim = Simulation(
        freq_max=10e9, domain=(.008, .008, .008), dx=.001, boundary="pec",
        **({"dz_profile": np.full(8, .001)} if graded else {}))
    shape = (Box((.002, .002, .002), (.005 + value, .005, .005))
             if kind == "Box" else Sphere((.004, .004, .004), .002 + value))
    if material == "dielectric":
        sim.add_material("dielectric", eps_r=2.25)
    sim.add(shape, material=material)
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


@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("route", ["jit", "vmap"])
def test_dielectric_box_refuses_traced_bounds(smoothing, graded, route):
    fn = lambda value: _shape_run(value, "Box", smoothing, graded, material="dielectric")
    with pytest.raises(NotImplementedError, match="Box.*traced extent.*any material.*concrete Box"):
        if route == "jit":
            jax.jit(fn)(jnp.float32(0.))
        else:
            jax.vmap(fn)(jnp.zeros(2))


@pytest.mark.parametrize("kind,material", [("Box", "pec"), ("Box", "dielectric"), ("Sphere", "pec")])
def test_traced_dimension_refusal_is_a_jax_concretization_error(kind, material):
    # optimize(jit=True), observables' eval_shape guard and the distributed
    # entry catch JAX's error to fall back; the named refusal must stay one.
    fn = lambda value: _shape_run(value, kind, False, False, material=material)
    from rfx._traced_refusal import TracedStructureRefusal
    with pytest.raises(jax.errors.ConcretizationTypeError, match=f"{kind}.*traced") as exc:
        jax.jit(fn)(jnp.float32(0.))
    assert isinstance(exc.value, TracedStructureRefusal)

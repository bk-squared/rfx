"""Declared dimensions refuse tangents without blocking shape value transforms."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation, Sphere
from rfx.sources.sources import GaussianPulse
from tests.unit.runners.test_forward_subpixel_smoothing import sphere_sim


def shape_sim(dimension, shape="sphere", graded=False, amplitude=1.0):
    sim = sphere_sim(graded, radius=dimension)
    entry = sim._geometry[0]
    if shape == "cylinder":
        sim._geometry[0] = replace(entry, shape=Cylinder(entry.shape.center, dimension, .0046))
    elif shape == "box":
        sim._geometry[0] = replace(entry, shape=Box((.003, .004, .004), (dimension, .008, .008)))
    sim._ports[0] = replace(sim._ports[0], waveform=GaussianPulse(
        f0=10e9, bandwidth=.8, amplitude=amplitude))
    return sim


def objective(dimension, entry="forward", shape="sphere", graded=False,
              smoothing=False, skip_preflight=True, amplitude=1.0):
    result = getattr(shape_sim(dimension, shape, graded, amplitude), entry)(
        n_steps=20, skip_preflight=skip_preflight, subpixel_smoothing=smoothing)
    return jnp.sum(jnp.asarray(result.time_series) ** 2)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("shape,dimension", [("sphere", .0023), ("cylinder", .0023), ("box", .0077)])
def test_a_declared_dimension_gradient_refuses(entry, shape, dimension):
    with pytest.raises(NotImplementedError, match="declared shape dimension"):
        jax.grad(lambda d: objective(d, entry, shape))(dimension)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_a_graded_shape_gradient_refuses(entry):
    with pytest.raises(NotImplementedError, match="declared shape dimension"):
        jax.grad(lambda r: objective(r, entry, graded=True))(.0023)


def test_a_shape_gradient_refuses_with_preflight():
    with pytest.raises(NotImplementedError, match="declared shape dimension"):
        jax.grad(lambda r: objective(r, skip_preflight=False))(.0023)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_a_smoothed_shape_gradient_refuses(entry):
    # The #1849 tangent guard, not the traced pad-reach refusal, answers a gradient.
    with pytest.raises(NotImplementedError, match="A gradient with respect to"):
        jax.grad(lambda r: objective(r, entry, smoothing=True))(.0023)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_b_gradient_of_jitted_shape_refuses(entry):
    with pytest.raises(NotImplementedError, match="declared shape dimension"):
        jax.grad(jax.jit(lambda r: objective(r, entry)))(.0023)


def test_b_jacfwd_shape_refuses():
    with pytest.raises(NotImplementedError, match="declared shape dimension"):
        jax.jacfwd(objective)(.0023)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("shape", ["sphere", "cylinder"])
@pytest.mark.parametrize("transform", ["jit", "vmap"])
def test_c_shape_value_transforms_match_plain(entry, graded, shape, transform):
    def loss(dimension):
        return objective(dimension, entry, shape, graded)

    if transform == "jit":
        expected = loss(.0023)
        actual = jax.jit(loss)(.0023)
        # A record summed over steps: jit reassociates float32 sums (measured
        # 2.9e-7..4.9e-7 relative on run(), 0 on forward()).
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)
    else:
        dimensions = (.0023, .0027)
        expected = jnp.stack([loss(d) for d in dimensions])
        actual = jax.vmap(loss)(jnp.asarray(dimensions))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_g_amplitude_gradient_with_jit_traced_radius(entry):
    def loss(radius, amplitude):
        return objective(radius, entry, amplitude=amplitude)

    expected = jax.grad(lambda a: loss(.0023, a))(1.0)
    actual = jax.grad(jax.jit(loss), argnums=1)(.0023, 1.0)
    assert float(expected) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


def pec_box_objective(radius, smoothing):
    """The same sphere in a closed PEC box: no absorber, so no pad-reach decision."""
    sim = Simulation(freq_max=20e9, domain=(.012, .012, .012), dx=.001, boundary="pec")
    sim.add_material("sphere", eps_r=4.0)
    sim.add(Sphere((.00537, .00613, .00573), radius), material="sphere")
    sim.add_source((.0031, .0042, .0043), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8),
                   amplitude_kind="current")
    sim.add_probe((.0072, .0051, .0064), "ez")
    result = sim.forward(n_steps=20, skip_preflight=True, subpixel_smoothing=smoothing)
    return jnp.sum(jnp.asarray(result.time_series) ** 2)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("smoothing", [True, "full_tensor", "kottke_pec"])
@pytest.mark.parametrize("transform", ["jit", "vmap"])
def test_d_smoothed_shape_transform_in_cpml_refuses(entry, graded, smoothing, transform):
    # Whether a traced shape reaches the absorber has no concrete answer; this
    # was a raw ConcretizationTypeError. Graded kottke_pec keeps its own
    # earlier refusal; graded full_tensor is admitted and reaches this one.
    def loss(radius):
        return objective(radius, entry, graded=graded, smoothing=smoothing)

    match = ("continued into it" if not graded or smoothing != "kottke_pec"
             else "uniform single-device|non-uniform mesh")
    with pytest.raises(NotImplementedError, match=match):
        if transform == "jit":
            jax.jit(loss)(.0023)
        else:
            jax.vmap(loss)(jnp.asarray((.0023, .0027)))


@pytest.mark.parametrize("smoothing", [True, "full_tensor"])
def test_d_smoothed_shape_jit_in_pec_box_matches_plain(smoothing):
    expected = pec_box_objective(.0023, smoothing)
    actual = jax.jit(lambda r: pec_box_objective(r, smoothing))(.0023)
    # Measured 3e-7 / 6e-7 relative (float32 reassociation under jit).
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)

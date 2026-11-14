"""Pin named refusals for traced PEC dimensions on concrete meshes.

Dielectric dimensions and the supported PEC dimensions on a traced mesh keep
running, while the existing PEC Sphere radius refusal keeps its exact text.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Cylinder, GaussianPulse, Simulation, Sphere
from rfx.geometry.rasterize_grid import TracedShapeDimensionError

DX = 1e-3
NX = 16
BASE = np.full(NX, DX)
DT = 0.5 * DX / (3e8 * np.sqrt(3.0))
PEC_DIMENSIONS = [("Cylinder", "radius"), ("Cylinder", "height"),
                  ("Cylinder", "center"), ("Sphere", "center")]


@pytest.fixture(autouse=True)
def _ignore_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _profile(d):
    bump = np.zeros(NX)
    bump[7], bump[8] = 1.0, -1.0
    return jnp.asarray(BASE) + d * jnp.asarray(bump)


def _shape(kind, dim, value, *, center_x=.005):
    center = (value if dim == "center" else center_x, .004, .004)
    radius = value if dim in ("radius", "radius and height") else .0016
    if kind == "Cylinder":
        height = value if dim in ("height", "radius and height") else .002
        return Cylinder(center=center, radius=radius, height=height)
    return Sphere(center=center, radius=radius)


def _concrete_objective(kind, dim, value, material="pec"):
    sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001,
                     boundary="pec", cpml_layers=0)
    sim.add_material("diel", eps_r=4.0)
    sim.add(_shape(kind, dim, value), material=material)
    sim.add_source((.002, .002, .002), "ez", amplitude_kind="field")
    sim.add_probe((.002, .003, .002), "ez")
    out = sim.forward(n_steps=2, skip_preflight=True, subpixel_smoothing=False)
    return jnp.sum(out.time_series ** 2)


def _mesh_objective(kind, dim, value, *, concrete_profile=None):
    if concrete_profile is None:
        kw = dict(dx_profile=_profile(value), dt=DT, dt_min_cell=0.5e-3,
                  nominal_profiles={"x": BASE})
        radius = .0016 + (0.1 * value if dim == "radius" else 0.0)
        height = .002 + (0.1 * value if dim == "height" else 0.0)
        center = (.011 + (0.1 * value if dim == "center" else 0.0), .004, .004)
        shape = (Cylinder(center=center, radius=radius, height=height)
                 if kind == "Cylinder" else Sphere(center=center, radius=radius))
    else:
        kw = dict(dx_profile=concrete_profile, dt=DT, dt_min_cell=0.5e-3)
        shape = _shape(kind, dim, value, center_x=.011)
    sim = Simulation(freq_max=20e9, domain=(NX * DX, 8 * DX, 8 * DX),
                     dx=DX, boundary="pec", **kw)
    sim.add(shape, material="pec")
    sim.add_source((2 * DX, 4 * DX, 4 * DX), "ez",
                   waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
    sim.add_probe((5 * DX, 4 * DX, 4 * DX), "ez")
    return jnp.sum(sim.forward(n_steps=4, skip_preflight=True).time_series ** 2)


def _value(dim):
    return {"radius": .0016, "height": .0025, "center": .005}[dim]


def _assert_refusal(call, kind, fields):
    expected = (f"A PEC {kind} with a traced {fields} cannot check its resolved "
                f"volume. Use a concrete {fields} outside jit/vmap.")
    with pytest.raises(TracedShapeDimensionError) as caught:
        jax.block_until_ready(call())
    assert str(caught.value) == expected
    assert isinstance(caught.value, jax.errors.ConcretizationTypeError)


@pytest.mark.parametrize("kind,dim", PEC_DIMENSIONS)
@pytest.mark.parametrize("transform", ["jit", "vmap"])
def test_d1_concrete_pec_dimension(kind, dim, transform):
    def objective(value):
        return _concrete_objective(kind, dim, value)

    value = _value(dim)
    if transform == "jit":
        call = lambda: jax.jit(objective)(value)
    else:
        call = lambda: jax.vmap(objective)(jnp.array([value, value + 1e-4]))
    _assert_refusal(call, kind, dim)


def test_d1g_concrete_graded_pec_radius():
    profile = np.asarray(_profile(2e-4))

    def objective(radius):
        return _mesh_objective("Cylinder", "radius", radius, concrete_profile=profile)

    _assert_refusal(lambda: jax.jit(objective)(.0016), "Cylinder", "radius")


def test_d1j_joint_pec_radius_and_height():
    def objective(value):
        return _concrete_objective("Cylinder", "radius and height", value)

    _assert_refusal(lambda: jax.jit(objective)(.0016), "Cylinder", "radius and height")


def test_d2_existing_pec_sphere_radius_refusal():
    def objective(radius):
        return _concrete_objective("Sphere", "radius", radius)

    _assert_refusal(lambda: jax.jit(objective)(.0016), "Sphere", "radius")


@pytest.mark.parametrize("kind,dim", PEC_DIMENSIONS + [("Sphere", "radius")])
def test_d3_dielectric_dimension_runs(kind, dim):
    def objective(value):
        return _concrete_objective(kind, dim, value, material="diel")

    result = jax.block_until_ready(jax.jit(objective)(_value(dim)))
    assert np.isfinite(np.asarray(result)).all()


@pytest.mark.parametrize("kind,dim", PEC_DIMENSIONS)
def test_d4_pec_dimension_on_traced_mesh_runs(kind, dim):
    def objective(d):
        return _mesh_objective(kind, dim, d)

    result = jax.block_until_ready(jax.jit(objective)(0.0))
    assert np.isfinite(np.asarray(result)).all()

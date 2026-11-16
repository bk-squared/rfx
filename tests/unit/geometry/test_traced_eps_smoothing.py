"""A traced eps_r flows through subpixel smoothing (#2436).

The smoother groups shapes by medium. The key used to be the permittivity
value itself, so a JAX tracer (``jax.grad`` / ``jax.jit`` over eps_r) stopped
with ``TypeError: unhashable type`` and no gradient reached the user. The key
is now the medium's identity: the float value when concrete (today's key), the
tracer object when traced.

``subpixel_smoothing="full_tensor"`` refuses a traced eps_r by name (#2441):
its traced node-tensor build classifies grid-aligned faces in float32 and
solves a different model from the concrete build.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, Sphere
from rfx.errors import RfxUnsupported
from rfx.core.jax_utils import is_tracer
from rfx.geometry import smoothing
from rfx.geometry.rasterize_grid import coords_from_uniform_grid

DX = .001
BOX_A = Box((.0031, .0035, .0036), (.0063, .0083, .0084))
BOX_B = Box((.0063, .0035, .0036), (.0093, .0083, .0084))


def _sphere_loss(eps, mode, graded=False):
    profile = np.array([.001] * 4 + [.0008] * 5 + [.001] * 4) if graded else None
    sim = Simulation(freq_max=20e9, domain=(.012, .012, .012), dx=DX, boundary="pec",
                     dz_profile=profile)
    sim.add_material("body", eps_r=eps)
    sim.add(Sphere((.00537, .00613, .00573), .0023), material="body")
    sim.add_source((.0031, .0042, .0043), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8),
                   amplitude_kind="current")
    sim.add_probe((.0072, .0051, .0064), "ez")
    result = sim.forward(n_steps=20, skip_preflight=True, subpixel_smoothing=mode)
    return jnp.sum(jnp.asarray(result.time_series) ** 2)


def test_traced_eps_gradient_matches_central_difference():
    """Uniform mesh, smoothing True: d(loss)/d(eps_r) exists and agrees with a
    central difference to 1e-3 (the design's A3 bar at h = 0.01)."""
    eps0, h = 4.0, 0.01
    ad = float(jax.grad(lambda e: _sphere_loss(e, True))(eps0))
    fd = (float(_sphere_loss(eps0 + h, True)) - float(_sphere_loss(eps0 - h, True))) / (2 * h)
    assert np.isfinite(ad) and ad != 0.0
    assert abs(ad - fd) / max(abs(ad), abs(fd)) <= 1e-3, (ad, fd)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("transform", [jax.grad, jax.jit], ids=["grad", "jit"])
def test_traced_eps_full_tensor_is_refused_by_name(graded, transform):
    with pytest.raises(RfxUnsupported, match="full-tensor smoothing at box edges"):
        transform(lambda e: _sphere_loss(e, "full_tensor", graded))(4.0)


def _grid_and_index():
    sim = Simulation(freq_max=20e9, domain=(.012, .012, .012), dx=DX, boundary="pec")
    grid = sim._build_grid()
    coords = coords_from_uniform_grid(grid)
    node = lambda axis, value: int(np.argmin(np.abs(np.asarray(axis) - value)))
    # The A|B face is x = 6.3 mm. Ex at x = 6.5 mm (cell 6.0..7.0 mm), Ey at x = 6.0 mm;
    # transverse position 6 mm, more than 2 mm from every other face.
    i, j, k = node(coords.x, .006), node(coords.y, .006), node(coords.z, .006)
    assert abs(float(coords.x[i]) - .006) < 1e-9
    return grid, (i, j, k)


def test_touching_traced_and_concrete_bodies_average_against_each_other():
    """A traced A (eps 4) touching a concrete B (eps 2) along a flat x face.

    B is applied last and owns both interface samples. Half a cell beyond B's
    face lies A, so the Kottke averages are taken against A, never vacuum:
    Ex (normal) at x = 6.5 mm, fill f = 0.7 of B: 1 / (0.7/2 + 0.3/eps);
    Ey (tangential) at x = 6.0 mm, f = 0.2 of B: 0.2*2 + 0.8*eps.
    """
    grid, (i, j, k) = _grid_and_index()

    def smoothed(eps):
        ex, ey, _ = smoothing.compute_smoothed_eps(grid, [(BOX_A, eps), (BOX_B, 2.0)], 1.0)
        return jnp.stack([ex[i, j, k], ey[i, j, k]])

    eps0 = 4.0
    values = np.asarray(jax.jit(smoothed)(eps0))
    expected = np.array([1.0 / (0.7 / 2.0 + 0.3 / eps0), 0.2 * 2.0 + 0.8 * eps0])
    np.testing.assert_allclose(values, expected, rtol=1e-5)
    jac = np.asarray(jax.jacfwd(smoothed)(eps0))
    d_expected = np.array([(0.3 / eps0 ** 2) / (0.7 / 2.0 + 0.3 / eps0) ** 2, 0.8])
    np.testing.assert_allclose(jac, d_expected, rtol=1e-4)


def test_concrete_key_is_the_value():
    """Concrete runs keep today's grouping: the key is the float value, so two
    materials of equal eps_r are one medium and their shared face is bulk."""
    assert smoothing.medium_key(4.0) == 4.0 and type(smoothing.medium_key(4.0)) is float
    assert smoothing.medium_key(4) == 4.0 and smoothing.medium_key(np.float64(2.5)) == 2.5
    grid, (i, j, k) = _grid_and_index()
    other_four = float.fromhex("0x1p2")  # an equal value, a second object
    ex, ey, _ = smoothing.compute_smoothed_eps(grid, [(BOX_A, 4.0), (BOX_B, other_four)], 1.0)
    assert float(ex[i, j, k]) == 4.0 and float(ey[i, j, k]) == 4.0


def test_one_tracer_object_per_material(monkeypatch):
    """Inside one trace a material's shapes carry one tracer object, so they
    form one group (one medium) as a concrete material does."""
    seen = []
    real = smoothing._kottke_smooth

    def spy(xp, shapes, *args, **kwargs):
        seen.append([eps for _, eps in shapes])
        return real(xp, shapes, *args, **kwargs)

    monkeypatch.setattr(smoothing, "_kottke_smooth", spy)

    def loss(eps):
        sim = Simulation(freq_max=20e9, domain=(.012, .012, .012), dx=DX, boundary="pec")
        sim.add_material("body", eps_r=eps)
        sim.add_material("host", eps_r=2.0)
        sim.add(Sphere((.0035, .006, .006), .0015), material="body")
        sim.add(Box((.0055, .0045, .0045), (.0085, .0075, .0075)), material="host")
        sim.add(Sphere((.0095, .006, .006), .0015), material="body")
        sim.add_source((.0031, .0042, .0043), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8),
                       amplitude_kind="current")
        sim.add_probe((.0072, .0051, .0064), "ez")
        result = sim.forward(n_steps=2, skip_preflight=True, subpixel_smoothing=True)
        return jnp.sum(jnp.asarray(result.time_series) ** 2)

    jax.grad(loss)(4.0)
    assert seen
    for eps_list in seen:
        assert len(eps_list) == 3 and eps_list[1] == 2.0
        assert is_tracer(eps_list[0]) and eps_list[0] is eps_list[2]

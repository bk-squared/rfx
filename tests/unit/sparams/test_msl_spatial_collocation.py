"""MSL H-to-E-plane contracts; hand geometry and analytic waves, no FDTD.

These tests distinguish spatial sampling error from passivity: the old
same-index current also preserves unit reflection for an ideal reflector.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from rfx.grid import Grid
from rfx.nonuniform import make_nonuniform_grid
from rfx.sources.msl_port import (
    MSLPort,
    msl_collocate_h_planes,
    msl_h_plane_stencil,
)

# Dyadic metres make the independently declared nodes exact in either dtype.
U = 2.0**-12
DIRECTIONS = ("+x", "-x", "+y", "-y")


def _port(direction):
    return MSLPort(U, U, 2 * U, 0.0, U, direction, 50.0)


def _uniform(dx=U, **kwargs):
    return Grid(1e9, (12 * dx, 16 * dx, 6 * dx), dx=dx,
                cpml_layers=0, **kwargs)


def _graded(*, swapped=False):
    # x nodes: 0,1,2,4,8,9,10 U; y nodes: 0,1,4,5,7,8 U.
    x = U * np.array([1, 1, 2, 4, 1, 1], dtype=float)
    y = U * np.array([1, 3, 1, 2, 1], dtype=float)
    if swapped:
        x, y = y, x
    return make_nonuniform_grid(
        (float(x.sum()), float(y.sum())), np.full(4, U), U,
        cpml_layers=1, dx_profile=x, dy_profile=y,
    )


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_uniform_stencil_targets_the_e_node_despite_sign_and_asymmetric_pads(direction):
    grid = Grid(1e9, (12 * U, 16 * U, 6 * U), dx=U,
                cpml_layers=2, cpml_axes="xy", pec_faces={"x_lo"})
    axis = direction[-1]
    target, index = (5 * U, 5) if axis == "x" else (7 * U, 9)
    got = msl_h_plane_stencil(grid, _port(direction), target)
    assert got["axis"] == axis
    assert got["e_index"] == index
    assert tuple(got["h_indices"]) == (index - 1, index)
    np.testing.assert_allclose(got["registration_coordinates"],
                               [target - U, target], rtol=0, atol=0)
    np.testing.assert_allclose(got["sample_coordinates"],
                               [target - U / 2, target + U / 2], rtol=0, atol=0)
    assert got["voltage_coordinate"] == target
    np.testing.assert_array_equal(got["weights"], [0.5, 0.5])


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_graded_stencil_reproduces_an_affine_field_at_the_physical_e_node(direction):
    got = msl_h_plane_stencil(_graded(), _port(direction), 4 * U)
    if direction[-1] == "x":
        index, registration, samples, weights = 4, [2, 4], [3, 6], [2 / 3, 1 / 3]
    else:
        index, registration, samples, weights = 3, [1, 4], [2.5, 4.5], [1 / 4, 3 / 4]
    assert got["axis"] == direction[-1]
    assert got["e_index"] == index
    assert tuple(got["h_indices"]) == (index - 1, index)
    assert got["voltage_coordinate"] == 4 * U
    np.testing.assert_allclose(got["registration_coordinates"],
                               U * np.array(registration), rtol=0, atol=0)
    np.testing.assert_allclose(got["sample_coordinates"],
                               U * np.array(samples), rtol=0, atol=0)
    np.testing.assert_allclose(got["weights"], weights, rtol=1e-14, atol=0)
    # Samples come from the hand geometry, not from the helper's receipt.
    with enable_x64():
        slope, intercept = 2.0 - 0.75j, -3.0 + 0.5j
        left, right = [jnp.asarray([intercept + slope * p]) for p in samples]
        result = msl_collocate_h_planes(left, right, got["weights"])
        np.testing.assert_allclose(result, [intercept + slope * 4], rtol=1e-14)
        assert abs(complex(np.asarray((left + right) / 2)[0])
                   - (intercept + slope * 4)) > 0.5


def test_axis_permutation_preserves_the_same_physical_stencil_and_transverse_data():
    x = msl_h_plane_stencil(_graded(), _port("+x"), 4 * U)
    y = msl_h_plane_stencil(_graded(swapped=True), _port("-y"), 4 * U)
    for key in ("e_index", "h_indices", "registration_coordinates",
                "sample_coordinates", "voltage_coordinate", "weights"):
        np.testing.assert_array_equal(x[key], y[key])
    left = jnp.asarray([[1., 2., 3.], [4., 5., 6.]])
    right = jnp.asarray([[7., -1., 2.], [0., 3., -4.]])
    a = msl_collocate_h_planes(left, right, x["weights"])
    b = msl_collocate_h_planes(left.T, right.T, y["weights"])
    np.testing.assert_array_equal(a.T, b)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_matched_wave_reflection_changes_from_first_to_second_order(direction):
    sign = 1 if direction[0] == "+" else -1
    beta, zc = 0.2 / U, 50.0
    old_errors, new_errors = [], []
    with enable_x64():
        for dx in (U, U / 2, U / 4):
            stencil = msl_h_plane_stencil(_uniform(dx), _port(direction), 5 * dx)
            phase = sign * beta * dx / 2
            left = jnp.asarray([np.exp(1j * phase) / zc])
            right = jnp.asarray([np.exp(-1j * phase) / zc])
            current = msl_collocate_h_planes(left, right, stencil["weights"])
            gamma_old = (1 - zc * np.asarray(right)) / (1 + zc * np.asarray(right))
            gamma_new = (1 - zc * np.asarray(current)) / (1 + zc * np.asarray(current))
            np.testing.assert_allclose(gamma_old, 1j * np.tan(phase / 2), atol=1e-14)
            np.testing.assert_allclose(gamma_new, np.tan(phase / 2)**2, atol=1e-14)
            old_errors.append(abs(gamma_old[0]))
            new_errors.append(abs(gamma_new[0]))
    np.testing.assert_allclose(np.array(old_errors[:-1]) / old_errors[1:], 2, rtol=0.01)
    np.testing.assert_allclose(np.array(new_errors[:-1]) / new_errors[1:], 4, rtol=0.01)
    assert np.all(np.array(new_errors) < np.array(old_errors) / 10)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_perfect_reflector_preserves_unit_magnitude_with_either_sampling_rule(direction):
    stencil = msl_h_plane_stencil(_uniform(), _port(direction), 5 * U)
    sign = 1 if direction[0] == "+" else -1
    rho = np.exp(1j * np.array([0.7, 2.1]))
    phase, zc = sign * 0.1, 50.0
    voltage = 1 + rho
    with enable_x64():
        left = jnp.asarray((np.exp(1j * phase) - rho * np.exp(-1j * phase)) / zc)
        right = jnp.asarray((np.exp(-1j * phase) - rho * np.exp(1j * phase)) / zc)
        centered = msl_collocate_h_planes(left, right, stencil["weights"])
        for current in (np.asarray(right), np.asarray(centered)):
            gamma = (voltage - zc * current) / (voltage + zc * current)
            np.testing.assert_allclose(abs(gamma), 1, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(np.real(voltage * current.conj()), 0, atol=1e-15)


@pytest.mark.parametrize("dtype,x64", [
    ("float32", False), ("complex64", False),
    ("float32", True), ("complex64", True),
    ("float64", True), ("complex128", True),
])
def test_collocation_preserves_field_dtype_and_jitted_field_sensitivities(dtype, x64):
    # f64 weights must not promote c64/f32 fields even with x64 enabled.
    weights = (np.float64(2 / 3), np.float64(1 / 3))
    with enable_x64(x64):
        complex_field = dtype.startswith("complex")
        real_dtype = np.float64 if dtype in ("float64", "complex128") else np.float32
        left_np = np.array([[1., 2.], [-3., 4.]])
        right_np = np.array([[5., -6.], [7., 8.]])
        if complex_field:
            left_np = left_np + 0.5j * right_np
            right_np = right_np - 0.25j
        left, right = jnp.asarray(left_np, dtype=dtype), jnp.asarray(right_np, dtype=dtype)
        apply = jax.jit(lambda a, b: msl_collocate_h_planes(a, b, weights))
        result = apply(left, right)
        assert result.dtype == left.dtype
        tolerance = 8 * np.finfo(real_dtype).eps
        np.testing.assert_allclose(result, (2 * left_np + right_np) / 3,
                                   rtol=tolerance, atol=tolerance)

        def loss(a, b):
            value = msl_collocate_h_planes(a * left, b * right, weights)
            return jnp.sum(jnp.real(value) + 0.3 * jnp.imag(value))

        gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(
            jnp.asarray(1.25, dtype=real_dtype), jnp.asarray(-0.75, dtype=real_dtype))
        expected = [(2 / 3) * (left_np.real + 0.3 * left_np.imag).sum(),
                    (1 / 3) * (right_np.real + 0.3 * right_np.imag).sum()]
        np.testing.assert_allclose(gradients, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("coordinate", [0.0, 12 * U, -U, 13 * U, np.nan, np.inf])
def test_stencil_refuses_unbracketed_or_invalid_targets(coordinate):
    with pytest.raises(ValueError):
        msl_h_plane_stencil(_uniform(), _port("+x"), coordinate)


def test_stencil_refuses_a_registration_coordinate_that_maps_to_the_wrong_plane(monkeypatch):
    grid = _uniform()
    original = grid.position_to_index

    def inconsistent_registration(position):
        indices = original(position)
        if position[0] == 4 * U:
            return (5, indices[1], indices[2])
        return indices

    monkeypatch.setattr(grid, "position_to_index", inconsistent_registration)
    with pytest.raises(ValueError):
        msl_h_plane_stencil(grid, _port("+x"), 5 * U)


@pytest.mark.parametrize("right_shape", [(2, 1), (3, 2), (2,)])
def test_collocation_refuses_broadcastable_and_nonbroadcastable_shape_mismatches(right_shape):
    with pytest.raises(ValueError):
        msl_collocate_h_planes(jnp.ones((2, 2)), jnp.ones(right_shape), (0.5, 0.5))


def test_integer_samples_cannot_silently_truncate_fractional_weights():
    with pytest.raises(ValueError, match="floating or complex"):
        msl_collocate_h_planes(jnp.array([1]), jnp.array([3]), (0.5, 0.5))


@pytest.mark.parametrize("coordinate", [-1e308, 1e308])
def test_out_of_range_coordinate_is_rejected_before_index_conversion(coordinate):
    with pytest.raises(ValueError, match="inside the grid"):
        msl_h_plane_stencil(_uniform(), _port("+x"), coordinate)

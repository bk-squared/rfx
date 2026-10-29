"""Concrete source samples agree across plain and user-jitted entry points."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.core.yee import init_materials
from rfx.model.source_coefficients import (
    current_source_samples, field_source_samples, uniform_source_table,
)
from rfx.sources.port_drive import port_drive_waveform


N_STEPS = 768
_PROFILE = np.array([1.0e-3] * 7
                    + [0.9e-3, 1.1e-3, 0.95e-3, 1.05e-3, 0.9e-3, 1.1e-3, 1.05e-3, 0.95e-3]
                    + [1.0e-3] * 9)


def scene(*, port=False):
    """A 3456-cell graded block scene with the tutorial's current source."""
    sim = Simulation(freq_max=10e9, domain=(24e-3, 12e-3, 12e-3), dx=1e-3,
                     dx_profile=_PROFILE, boundary="pec", cpml_layers=0)
    if port:
        sim.add_port((6e-3, 6e-3, 6e-3), "ez", impedance=50.,
                     waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    else:
        # nonuniform_patch_demo.py uses add_source(amplitude_kind="current"),
        # not a positive-impedance lumped port.
        sim.add_source((6e-3, 6e-3, 6e-3), "ez",
                       waveform=GaussianPulse(f0=2.4e9, bandwidth=1.2),
                       amplitude_kind="current")
    sim.add_material("block", eps_r=4.0)
    sim.add(Box((-1.0, 3e-3, 3e-3), (1.0, 9e-3, 9e-3)), material="block")
    sim.add_probe((6e-3, 6e-3, 6e-3), "ez")
    sim.add_probe((13e-3, 5e-3, 7e-3), "ez")
    sim.add_probe((19e-3, 6e-3, 6e-3), "hy")
    return sim


def assert_same_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert actual.tobytes() == expected.tobytes()


def test_a_port_table_under_user_jit():
    # The tutorial board's GaussianPulse, including its default amplitude
    # and cutoff; no field solve is needed to expose the 2000-sample defect.
    excitation = GaussianPulse(f0=2.4e9, bandwidth=1.2)
    grid = SimpleNamespace(dt=7.5e-13)
    materials = init_materials((4, 4, 4))

    def table():
        return port_drive_waveform(
            grid, (2, 2, 2), "ez", excitation, 2000, materials,
            sigma_port=20., unit_field=1000.)

    expected = table()
    assert np.max(np.abs(expected)) > 0
    assert_same_bits(jax.jit(table)(), expected)


def test_b_graded_current_run_under_user_jit():
    sim = scene()
    pulse = sim._ports[0].waveform
    assert (N_STEPS - 1) * sim._build_nonuniform_grid().dt > (pulse.cutoff + 6) * pulse.tau

    def run():
        return sim.run(n_steps=N_STEPS, skip_preflight=True,
                       compute_s_params=False).time_series

    expected = np.asarray(run())
    actual = np.asarray(jax.jit(run)())
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    peak = np.max(np.abs(expected))
    assert peak > 0
    # #2241 amendment 2: identical source tables still permit the standing
    # cross-trace rounding bars for the compiled time step and record sum.
    per_step_error = np.max(np.abs(actual - expected), axis=1)
    assert np.all(per_step_error <= 9 * np.spacing(np.float32(peak)))
    summed_error = np.max(np.abs(np.sum(actual, axis=0, dtype=np.float64)
                                - np.sum(expected, axis=0, dtype=np.float64)))
    assert summed_error <= 1e-4 * peak
    assert_same_bits(sim.forward(n_steps=N_STEPS, skip_preflight=True,
                                 checkpoint=False).time_series, expected)


def test_c_forward_keeps_port_coefficient_gradient():
    sim = scene(port=True)
    grid = sim._build_nonuniform_grid()

    def first_drive(eps):
        trace = sim.forward(
            n_steps=2, eps_override=jnp.full(grid.shape, eps),
            skip_preflight=True, checkpoint=False).time_series
        # GaussianPulse(0) == 0: before sample 1 all fields are zero, so
        # this colocated probe's material derivative is the drive's Cb
        # derivative, with no propagation contribution to hide a lost one.
        return trace[1, 0]

    expected, expected_grad = jax.value_and_grad(first_drive)(4.)
    actual, gradient = jax.jit(jax.value_and_grad(first_drive))(4.)
    assert np.isfinite(expected) and float(expected) != 0
    assert np.isfinite(expected_grad) and float(expected_grad) != 0
    assert np.isfinite(actual) and float(actual) != 0
    assert np.isfinite(gradient) and float(gradient) != 0
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
    np.testing.assert_allclose(gradient, expected_grad, rtol=1e-6, atol=0)


@pytest.mark.parametrize("sampler", ["uniform", "graded", "adi"])
def test_other_concrete_source_samples_under_user_jit(sampler):
    waveform = GaussianPulse(f0=2.4e9, bandwidth=1.2)
    grid = SimpleNamespace(dt=7.5e-13, dx=1e-3)

    def table():
        if sampler == "uniform":
            return uniform_source_table(grid, (2, 2, 2), "ez", waveform,
                                        2000, None, "field")
        if sampler == "graded":
            return current_source_samples(grid, waveform, 2000)
        return field_source_samples(waveform, 2000, grid.dt)

    # The reference is the original plain evaluation, not the new guard.
    expected = jax.vmap(waveform)(jnp.arange(2000, dtype=jnp.float32) * grid.dt)
    assert_same_bits(table(), expected)
    assert_same_bits(jax.jit(table)(), expected)


@pytest.mark.parametrize("parameter", ["dt", "amplitude", "f0", "bandwidth", "cutoff"])
def test_traced_source_parameters_keep_the_traced_path(parameter):
    base = GaussianPulse(f0=2.4e9, bandwidth=1.2)

    def samples(scale, *, reference):
        dt = 7.5e-13 * scale if parameter == "dt" else 7.5e-13
        pulse = base if parameter == "dt" else replace(
            base, **{parameter: getattr(base, parameter) * scale})
        if reference:
            return jax.vmap(pulse)(jnp.arange(2000, dtype=jnp.float32) * dt)
        return field_source_samples(pulse, 2000, dt)

    def evaluate(reference):
        return jax.jit(lambda scale: jax.jvp(
            lambda value: samples(value, reference=reference),
            (scale,), (jnp.ones_like(scale),)))(1.)

    expected, expected_tangent = evaluate(True)
    actual, tangent = evaluate(False)
    assert_same_bits(actual, expected)
    assert_same_bits(tangent, expected_tangent)
    assert np.isfinite(tangent).all() and np.any(np.asarray(tangent) != 0)

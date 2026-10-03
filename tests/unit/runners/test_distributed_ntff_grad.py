"""The NTFF record of run(devices=) stays differentiable (#1441 review).

Assembling the per-device NTFF records must not leave JAX: a host copy there
broke jax.grad through the far-field accumulators."""
import math

import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation


def _ntff_power(amplitude, devices):
    sim = Simulation(freq_max=15e9, domain=(15e-3, 7e-3, 7e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=2, precision="float32")
    sim.add_source((6e-3, 3e-3, 3e-3), "ez", amplitude_kind="field",
                   waveform=lambda t: amplitude * jnp.cos(t * 2e10))
    sim.add_probe((6e-3, 3e-3, 3e-3), "ez")
    sim.add_ntff_box((3e-3, 2e-3, 2e-3), (12e-3, 5e-3, 5e-3), freqs=[5e9])
    result = sim.run(n_steps=8, devices=devices, skip_preflight=True)
    return sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in jax.tree.leaves(result.ntff_data))


def test_ntff_gradient_through_two_device_run():
    """The field is linear in the source amplitude a, so the NTFF power goes as
    a**2 and its derivative at a = 1 is twice the power."""
    devices = jax.devices("cpu")[:2]
    if len(devices) != 2:
        pytest.skip("requires two virtual CPU devices (root conftest)")
    power = float(_ntff_power(1.0, devices))
    gradient = float(jax.grad(_ntff_power)(1.0, devices))
    assert math.isfinite(gradient) and power > 0.0
    assert abs(gradient - 2.0 * power) <= 1e-4 * abs(2.0 * power), (gradient, power)

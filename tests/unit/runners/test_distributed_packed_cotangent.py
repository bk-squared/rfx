"""Packed material and source gradients retain the caller's placement."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from tests._multi_device_scene import host_devices


pytestmark = pytest.mark.distributed


def scene(eps_r, amplitude, graded, sigma=0.):
    dx = 1e-3
    profile = np.full(24, dx)
    profile[8:16] *= 1.25
    sim = Simulation(freq_max=15e9, domain=(11 * dx, 11 * dx, float(profile.sum())),
                     dx=dx, boundary='cpml', cpml_layers=6,
                     **({'dz_profile': profile} if graded else {}))
    sim.add_material('dielectric', eps_r=eps_r, sigma=sigma)
    sim.add(Box((3.7e-3, 2.8e-3, 2.8e-3), (7.3e-3, 8.3e-3, 8.3e-3)),
            material='dielectric')
    sim.add_source((3.2e-3, 5.5e-3, 5.5e-3), 'ez', amplitude_kind='field',
                   waveform=GaussianPulse(f0=7.5e9, bandwidth=.8, amplitude=amplitude))
    sim.add_probe((6.0e-3, 5.5e-3, 5.0e-3), 'ez')
    return sim


@pytest.mark.parametrize('graded', [True, False], ids=['graded', 'uniform'])
@pytest.mark.parametrize('with_sigma', [True], ids=['eps-amplitude-sigma'])
def test_packed_cotangents_match_separate_inputs(graded, with_sigma, record_property):
    devices = host_devices(2)
    values = [2., 1., .01] if with_sigma else [2., 1.]
    packed = jax.device_put(jnp.asarray(values, dtype=jnp.float32), devices[0])
    inputs = tuple(packed[i] for i in range(len(values)))

    def objective(eps_r, amplitude, sigma=0.):
        result = scene(eps_r, amplitude, graded, sigma).forward(
            n_steps=20, distributed=True, devices=devices, skip_preflight=True)
        return jnp.sum(jnp.asarray(result.time_series) ** 2)

    separate = jax.grad(objective, argnums=tuple(range(len(inputs))))(*inputs)
    actual = jax.grad(lambda p: objective(*(p[i] for i in range(len(values)))))(packed)
    assert actual.sharding == packed.sharding
    for cotangent, caller in zip(separate, inputs):
        assert cotangent.sharding == caller.sharding
    expected = np.asarray([np.asarray(g) for g in separate], dtype=np.float32)
    actual = np.asarray(actual)
    assert actual.dtype == expected.dtype == np.float32
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert np.all(expected != 0), 'each packed parameter must affect the objective'
    np.testing.assert_array_max_ulp(actual, expected, maxulp=9)
    record_property('packed_gradient', actual.tolist())
    record_property('bitwise_equal', np.array_equal(actual.view(np.uint32), expected.view(np.uint32)))

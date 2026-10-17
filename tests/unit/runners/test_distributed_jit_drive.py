"""rfx #1840: a user jax.jit must not change the graded multi-device point-source drive.

One device builds its drive samples as host constants (declaration_setup), so it is bit-identical eager and
under a user jit. The multi-device lane builds them in rfx.model.distributed_ports.assemble; when a user jit
staged them, the compiled Gaussian samples moved the source edge by 1-2 float32 ULP from step 2 on. The pin:
the multi-device run under a user jit (with eps_override traced by it, and with nothing traced) gives the
eager run's bits at the source edge and at every probe.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation

pytestmark = pytest.mark.distributed

N_STEPS = 60
_PROFILE = np.array([1.0e-3] * 7 + [0.9e-3, 1.1e-3, 0.95e-3, 1.05e-3, 0.9e-3, 1.1e-3, 1.05e-3, 0.95e-3]
                    + [1.0e-3] * 9)


def _scene(boundary, port=False):
    sim = Simulation(freq_max=10e9, domain=(24e-3, 12e-3, 12e-3), dx=1e-3, boundary=boundary,
                     cpml_layers=6 if boundary == 'cpml' else 0, dx_profile=_PROFILE)
    if port:
        sim.add_port((6e-3, 6e-3, 6e-3), 'ez', impedance=50., waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    else:
        sim.add_source((6e-3, 6e-3, 6e-3), 'ez', waveform=GaussianPulse(f0=5e9, bandwidth=.8),
                       amplitude_kind='current')
    sim.add_material('block', eps_r=4.0)
    sim.add(Box((-1.0, 3e-3, 3e-3), (1.0, 9e-3, 9e-3)), material='block')
    sim.add_probe((6e-3, 6e-3, 6e-3), 'ez')       # the source edge
    sim.add_probe((13e-3, 5e-3, 7e-3), 'ez')
    sim.add_probe((19e-3, 6e-3, 6e-3), 'hy')
    return sim


@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_user_jit_keeps_the_multi_device_drive_bits(boundary):
    sim = _scene(boundary)
    devices = jax.devices('cpu')[:2]
    eps = jnp.asarray(sim._assemble_materials_nu(sim._build_nonuniform_grid())[0].eps_r)

    def forward(e=None):
        kw = {} if e is None else dict(eps_override=e)
        return sim.forward(n_steps=N_STEPS, checkpoint=False, skip_preflight=True,
                           distributed=True, devices=devices, **kw).time_series

    eager = np.asarray(forward(eps))
    assert np.abs(eager[:, 0]).max() > 0
    np.testing.assert_array_equal(np.asarray(jax.jit(forward)(eps)), eager)
    plain = np.asarray(forward())
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda: forward())()), plain)


def test_user_jit_keeps_the_multi_device_lumped_port_drive_bits():
    """The port half of the drive block: an untraced user jit gives the eager bits (main: 3.5 ULP).
    With a traced override the port load is traced arithmetic, as on one device, so it is not pinned here."""
    sim = _scene('pec', port=True)
    devices = jax.devices('cpu')[:2]

    def forward():
        return sim.forward(n_steps=N_STEPS, checkpoint=False, skip_preflight=True,
                           distributed=True, devices=devices).time_series

    eager = np.asarray(forward())
    assert np.abs(eager[:, 0]).max() > 0
    np.testing.assert_array_equal(np.asarray(jax.jit(forward)()), eager)

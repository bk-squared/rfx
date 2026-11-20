"""A traced scalar ``dx`` is refused once, at ``Simulation(dx=...)``.

The host turns the cell size into the grid's cell counts (the wall count
``length / dx``) and into the MSL probe ladder's counts, so a traced ``dx``
leaves no grid to build. Every entry that takes a traced ``dx`` -- ``run()``
and ``forward()`` on one and on two devices, and ``add_msl_port`` -- raises
``TracedStructureRefusal`` with one message under ``jax.jit`` and under
``jax.grad``. The same models with a concrete ``dx`` record and register.
Two devices need ``XLA_FLAGS=--xla_force_host_platform_device_count=2``, as
the PR lane sets it; without them those rows skip.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx._traced_refusal import TracedStructureRefusal
from tests._multi_device_scene import host_devices

MESSAGE = ("Simulation(dx=...) is traced (under jax.jit, jax.grad or jax.vmap): the cell size "
           "sets the grid's cell counts on the host. Keep dx concrete; for a differentiable mesh "
           "trace dx_profile, dy_profile or dz_profile instead and declare its concrete widths at "
           "the primal point in nominal_profiles so positions on that axis resolve.")
TRANSFORMS = {"jit": jax.jit, "grad": jax.grad}


def devices_for(n_devices):
    return jax.devices("cpu")[:1] if n_devices == 1 else host_devices(n_devices)


def pec_box(dx):
    sim = Simulation(freq_max=15e9, domain=(16e-3, 16e-3, 16e-3), dx=dx, boundary="pec")
    sim.add_source((4e-3, 8e-3, 8e-3), "ez", amplitude_kind="current")
    sim.add_probe((9.6e-3, 8e-3, 8e-3), "ez")
    return sim


def record(sim, entry, devices):
    if entry == "forward":
        options = {"distributed": True, "devices": devices} if len(devices) > 1 else {}
        return sim.forward(n_steps=2, checkpoint=False, skip_preflight=True, **options).time_series
    options = {"devices": devices} if len(devices) > 1 else {}
    return sim.run(n_steps=2, skip_preflight=True, **options).time_series


def msl_board(dx):
    sim = Simulation(freq_max=16e9, domain=(24e-3, 12e-3, 8e-3), dx=dx, boundary="pec")
    sim.add_material("substrate", eps_r=3.0)
    sim.add(Box((2e-3, 2e-3, 1e-3), (14e-3, 10e-3, 3e-3)), material="substrate")
    sim.add(Box((2e-3, 2e-3, 0.0), (14e-3, 10e-3, 1e-3)), material="pec")
    sim.add_msl_port(position=(5e-3, 6e-3, 1e-3), width=2e-3, height=2e-3, direction="+x",
                     impedance=50.0, waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
    return sim


def refusal(transform, build):
    def objective(v):
        return jnp.sum(build(jnp.float32(1e-3) + v * 1e-5))

    with pytest.raises(Exception) as caught:
        TRANSFORMS[transform](objective)(jnp.float32(0.0))
    return caught.value


@pytest.mark.parametrize("transform", ["jit", "grad"])
@pytest.mark.parametrize("n_devices", [1, 2])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_traced_dx_is_refused_before_the_record(entry, n_devices, transform):
    devices = devices_for(n_devices)
    exc = refusal(transform, lambda dx: record(pec_box(dx), entry, devices))
    assert type(exc) is TracedStructureRefusal
    assert str(exc) == MESSAGE


@pytest.mark.parametrize("transform", ["jit", "grad"])
def test_traced_dx_is_refused_before_the_msl_probe_ladder(transform):
    def build(dx):
        msl_board(dx)
        return dx

    exc = refusal(transform, build)
    assert type(exc) is TracedStructureRefusal
    assert str(exc) == MESSAGE


@pytest.mark.parametrize("n_devices", [1, 2])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_concrete_dx_box_records(entry, n_devices):
    series = np.asarray(record(pec_box(1e-3), entry, devices_for(n_devices)))
    assert series.size > 0
    assert np.isfinite(series).all()


def test_concrete_dx_msl_port_registers():
    assert len(msl_board(jnp.float32(1e-3))._msl_ports) == 1

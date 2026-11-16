"""Vector padding is refused on the graded (non-uniform) step.

The graded step allocates its arrays at the grid's own extents. A forced
target (``"aligned"`` or ``"odd"``) is refused by name with the public
``RfxUnsupported`` on each single-device lane (run, until_decay, forward)
and on the multi-device graded runner; it is never ignored. The memory
planners estimate a graded grid at its own extents, under a forced target
and under the gate table's row alike.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.errors import RfxUnsupported
from rfx.stepping import vector_pad
from tests.unit.runners.test_vector_pad import PULSE, TARGETS, _walls

#: A z profile with four cell sizes; the PEC box has nz = 6, which a uniform
#: grid would pad on both targets (aligned to 8, odd to 7).
DZ = 1e-3 * np.array([1.1, .8, .6, .8, 1.])


def _graded():
    """A small graded PEC box; x and y lengths are not whole cells."""
    sim = Simulation(freq_max=10e9, domain=(.0113, .0071, float(DZ.sum())), dx=.001, dz_profile=DZ,
                     boundary=_walls(), cpml_layers=0, snap='declared')
    sim.add_source((.003, .004, .0021), 'ez', amplitude_kind='current', waveform=PULSE)
    sim.add_probe((.009, .005, .0015), 'ez')
    return sim


def _forward(sim):
    shape = sim._build_nonuniform_grid().shape
    return sim.forward(n_steps=8, eps_override=jnp.ones(shape, jnp.float32), skip_preflight=True)


LANES = {
    'run': lambda sim: sim.run(n_steps=8, skip_preflight=True, compute_s_params=False),
    'until_decay': lambda sim: sim.run(n_steps=8, until_decay=1e-3, decay_check_interval=4, decay_min_steps=4,
                                       decay_max_steps=8, skip_preflight=True, compute_s_params=False),
    'forward': _forward,
}


@pytest.mark.parametrize('lane', sorted(LANES))
@pytest.mark.parametrize('target', TARGETS)
def test_graded_refuses_a_forced_pad(target, lane):
    sim = _graded()
    with vector_pad.force_pad(target), vector_pad.record_plans() as plans, \
            pytest.raises(RfxUnsupported, match='vector padding: graded not padded yet'):
        LANES[lane](sim)
    assert not any(p.active for p in plans), plans
    jax.clear_caches()


def test_distributed_graded_refuses_a_forced_pad():
    """The multi-device graded runner is not padded: a forced pad is refused by name."""
    with vector_pad.force_pad('aligned'), pytest.raises(NotImplementedError, match='vector padding') as info:
        _graded().run(n_steps=8, skip_preflight=True, compute_s_params=False, devices=jax.devices('cpu')[:2])
    assert 'distributed non-uniform' in str(info.value)
    jax.clear_caches()


@pytest.mark.parametrize('target', TARGETS)
def test_memory_planners_estimate_a_graded_grid_at_its_own_extents(target, monkeypatch):
    from rfx.auto_config import SimConfig, _auto_memory_mb
    sim = _graded()
    shape = tuple(sim._build_nonuniform_grid().shape)
    # A uniform grid of this shape would be padded on this target.
    assert vector_pad.plan_pad(shape, target=target, periodic_axes=(False,) * 3).padded != shape
    with vector_pad.force_pad(target):
        accounting = sim._ad_memory_static_accounting()
    assert accounting['cells'] == math.prod(shape), (accounting['cells'], shape)

    config = SimConfig(dx=.001, domain=(.0113, .0071, float(DZ.sum())), cpml_layers=4, n_steps=10,
                       freq_range=(1e9, 10e9), margin=0., dt=1e-12, accuracy='standard', dz_profile=DZ)
    config_shape = tuple(config.grid_shape)
    assert vector_pad.plan_pad(config_shape, target=target, periodic_axes=(False,) * 3).padded != config_shape
    with vector_pad.force_pad(target):
        assert config.estimated_memory_mb == _auto_memory_mb(math.prod(config_shape), 4)

    # The gate table's row does not pad a graded grid either.
    monkeypatch.setattr(vector_pad, '_MEASURED_GAIN', {vector_pad._device_kind(): target})
    assert sim._ad_memory_static_accounting()['cells'] == math.prod(shape)
    assert config.estimated_memory_mb == _auto_memory_mb(math.prod(config_shape), 4)

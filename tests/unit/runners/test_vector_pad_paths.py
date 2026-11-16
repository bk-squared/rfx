"""Vector padding on the adjoint scan, the vmap material sweep and the memory planners.

Each path pads by the stepping layer (``rfx.stepping.vector_pad``) and must
return the unpadded run's bytes on the CPU, or, for a case listed as not
bitwise, stay within the bars of ``tests/unit/runners/test_vector_pad.py``.
The memory planners estimate from the extents the step allocates.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.stepping import vector_pad
from tests.unit.runners.test_vector_pad import TARGETS, _walls, assert_bitwise, assert_padded

ADJOINT_STEPS = 160


def _adjoint_records(mode, gradient):
    from tests.unit.autodiff.test_reciprocity_adjoint import fixture, objective
    sim, eps = fixture('float32')
    sigma = jnp.zeros_like(eps)
    loss = objective(sim, steps=ADJOINT_STEPS, mode=gradient)
    with vector_pad.force_pad(mode), vector_pad.record_plans() as plans:
        value, (grad_eps, grad_sigma) = jax.value_and_grad(loss, argnums=(0, 1))(eps, sigma)
    return {'value': np.asarray(value), 'grad_eps': np.asarray(grad_eps),
            'grad_sigma': np.asarray(grad_sigma)}, plans


@pytest.mark.parametrize('gradient', ('adjoint', 'autodiff'))
@pytest.mark.parametrize('target', TARGETS)
def test_design_box_objective_and_gradient_padded_bitwise(target, gradient):
    """The reciprocity adjoint scan, and autodiff through the same design box."""
    ref, ref_plans = _adjoint_records('off', gradient)
    got, plans = _adjoint_records(target, gradient)
    assert ref_plans and not any(p.active for p in ref_plans)
    assert_padded(plans, f'design box {gradient}/{target}')
    assert_bitwise(ref, got, f'design box {gradient}/{target}')
    jax.clear_caches()


def _sweep_scene(boundary):
    sim = Simulation(freq_max=10e9, domain=(.0113, .0071, .0052), dx=.001,
                     boundary=boundary if boundary == 'cpml' else _walls(),
                     cpml_layers=4 if boundary == 'cpml' else 0, snap='declared')
    sim.add_material('brick', eps_r=2.2)
    sim.add(Box((.004, .002, .001), (.008, .005, .004)), material='brick')
    sim.add_source((.003, .004, .003), 'ez', waveform=GaussianPulse(f0=6e9, bandwidth=.8))
    sim.add_probe((.009, .005, .002), 'ez')
    sim.add_probe((.002, .002, .004), 'hy')
    sim.add_dft_plane_probe(axis='z', coordinate=.003, component='ez',
                            freqs=jnp.array([4e9, 6e9, 8e9]), name='plane')
    return sim


@pytest.mark.parametrize('boundary', ('pec', 'cpml'))
@pytest.mark.parametrize('target', TARGETS)
def test_vmap_material_sweep_padded_bitwise(target, boundary):
    from rfx.vmap_sweep import vmap_material_sweep
    out = {}
    for mode in ('off', target):
        with vector_pad.force_pad(mode), vector_pad.record_plans() as plans:
            result = vmap_material_sweep(_sweep_scene(boundary), 'brick.eps_r',
                                         jnp.array([1.5, 2.2, 3.0]), n_steps=60)
        out[mode] = ({'time_series': np.asarray(result.time_series),
                      'plane': np.asarray(result.dft_planes['plane'])}, plans)
    assert out['off'][1] and not any(p.active for p in out['off'][1])
    assert_padded(out[target][1], f'vmap sweep {boundary}/{target}')
    assert_bitwise(out['off'][0], out[target][0], f'vmap sweep {boundary}/{target}')
    jax.clear_caches()


def _planner_scene():
    sim = Simulation(freq_max=10e9, domain=(.0113, .0071, .0052), dx=.001, boundary=_walls(),
                     cpml_layers=0, snap='declared')
    sim.add_source((.003, .004, .003), 'ez', waveform=GaussianPulse(f0=6e9, bandwidth=.8))
    sim.add_probe((.009, .005, .002), 'ez')
    return sim


def _expected_padded(shape, target):
    """The padded extents, from the target's rule written out here."""
    nx, ny, nz = shape
    if target == 'aligned':
        return nx, ny, 4 * math.ceil(nz / 4)
    return tuple(n + (n % 2 == 0) for n in shape)


@pytest.mark.parametrize('target', TARGETS)
def test_memory_planner_estimates_padded_extents(target):
    sim = _planner_scene()
    shape = tuple(sim._build_grid().shape)
    padded = _expected_padded(shape, target)
    assert padded != shape
    unpadded = sim.estimate_ad_memory(n_steps=40)
    with vector_pad.force_pad(target):
        estimate = sim.estimate_ad_memory(n_steps=40)
        accounting = sim._ad_memory_static_accounting()
    # The six float32 field arrays of the scan carry; the rest of the carry
    # (the step counter) does not depend on the extents.
    rest = unpadded.carry_bytes - 6 * math.prod(shape) * 4
    assert 0 <= rest < 64, (unpadded.carry_bytes, shape)
    assert estimate.carry_bytes == 6 * math.prod(padded) * 4 + rest, (estimate.carry_bytes, shape, padded)
    assert accounting['cells'] == math.prod(padded)
    assert accounting['field_bytes'] == 6 * math.prod(padded) * 4
    # The reported grid stays the physical one.
    assert (accounting['nx'], accounting['ny'], accounting['nz']) == shape
    jax.clear_caches()


def test_memory_planner_auto_follows_the_gate_table(monkeypatch):
    sim = _planner_scene()
    shape = tuple(sim._build_grid().shape)
    assert sim._ad_memory_static_accounting()['cells'] == math.prod(shape)
    monkeypatch.setattr(vector_pad, '_MEASURED_GAIN', {vector_pad._device_kind(): 'aligned'})
    assert sim._ad_memory_static_accounting()['cells'] == math.prod(_expected_padded(shape, 'aligned'))
    with vector_pad.record_plans() as plans:
        sim.run(n_steps=4, skip_preflight=True, compute_s_params=False)
    assert plans and all(p.padded == _expected_padded(shape, 'aligned') for p in plans)


def test_auto_config_estimates_padded_extents():
    from rfx.auto_config import SimConfig, _auto_memory_mb
    config = SimConfig(dx=.001, domain=(.0113, .0071, .0052), cpml_layers=4, n_steps=10,
                       freq_range=(1e9, 10e9), margin=0., dt=1e-12, accuracy='standard')
    shape = tuple(config.grid_shape)
    with vector_pad.force_pad('aligned'):
        assert config.estimated_memory_mb == _auto_memory_mb(
            math.prod(_expected_padded(shape, 'aligned')), 4)
    assert config.estimated_memory_mb == _auto_memory_mb(math.prod(shape), 4)

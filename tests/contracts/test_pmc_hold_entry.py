"""Observe production builders and scan entries; never replace their initial state."""
from contextlib import contextmanager
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from tests._multi_device_scene import host_devices
from tests.contracts.boundary_fields import component, field_nodes, is_aux


def scene(axis, cells, cpml=False):
    widths = np.full(cells, .001)
    widths[3:5] = (.0009, .0011)
    boundaries = dict.fromkeys('xyz', 'pec')
    if cpml:
        # Keep x unpadded so cells=10 still pins the short last slab.
        boundaries.update({a: 'cpml' for a in 'yz' if a != axis})
    boundaries[axis] = Boundary('pec', 'pmc')
    sim = Simulation(freq_max=12e9, domain=(cells * .001, .006, .006), dx=.001,
                     dx_profile=widths, boundary=BoundarySpec(**boundaries), cpml_layers=2 if cpml else 0)
    sim.add_source((.004, .003, .003), 'ez', amplitude_kind='field')
    sim.add_probe((.004, .003, .003), 'ez')
    return sim


def ghost_rows(carry, shape, devices, axis):
    """Whole held subsystem, with independent physical/packed row arithmetic."""
    a = 'xyz'.index(axis)
    indices = [slice(None)] * 3
    states = field_nodes(carry)
    if a == 0 and component(states[0], 0).shape != shape:
        per_rank = (shape[0] + devices - 1) // devices
        stride = per_rank + 2
        # Check every stored copy of the physical high ghost, including halos.
        rows = [r * stride + shape[0] - 1 - r * per_rank + 1
                for r in range(devices)
                if 0 <= shape[0] - 1 - r * per_rank + 1 < stride]
        indices[0] = np.array(rows)
    else:
        indices[a] = -1
    rows = [component(state, c)[tuple(indices)] for state in states
            for c in (a, *(c + 3 for c in range(3) if c != a))]
    for auxiliary in jax.tree_util.tree_leaves(carry, is_leaf=is_aux):
        if is_aux(auxiliary):
            for name, value in auxiliary._asdict().items():
                if (name.startswith('psi_h') and name[5] != axis
                        and name[-3] != axis and value.size):
                    rows.append(value[tuple(indices)])
    return tuple(rows)


@contextmanager
def observe(monkeypatch, shape, devices, axis, anchor, mode):
    """Identity taps inspect primal, JVP and VJP carries without modifying them."""
    records = []
    original = jax.lax.scan

    def record(label, arrays):
        records.append((label, tuple(np.asarray(a) for a in arrays)))

    def sample(label, carry):
        jax.debug.callback(lambda *a: record(label, a),
                           *ghost_rows(carry, shape, devices, axis))

    @jax.custom_jvp
    def tangent_tap(carry, parameter):
        return carry

    @tangent_tap.defjvp
    def tangent_rule(primals, tangents):
        sample('tangent', tangents[0])
        return primals[0], tangents[0]

    @jax.custom_vjp
    def cotangent_tap(carry, inputs):
        return carry

    def forward(carry, inputs):
        return carry, None

    def backward(residual, cotangent):
        sample('cotangent', cotangent)
        return cotangent, None

    cotangent_tap.defvjp(forward, backward)

    def scan(function, initial, xs=None, *args, **kwargs):
        if field_nodes(initial):
            sample('primal', initial)
            if mode == 'jvp':
                initial = tangent_tap(initial, anchor)
            elif mode == 'vjp':
                # Keep even a constant initial carry on the reverse tape via
                # the scan's own differentiable inputs. The outer scalar must
                # not become an extra jit input with a device-placed zero VJP.
                _, constants = jax.closure_convert(
                    lambda carry, values: original(function, carry, values, *args, **kwargs),
                    initial, xs)
                initial = cotangent_tap(initial, (xs, constants))
        return original(function, initial, xs, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(jax.lax, 'scan', scan)
        yield records


def assert_records(records, mode):
    labels = {label for label, _ in records}
    assert 'primal' in labels, 'no production scan entry observed'
    if mode != 'primal':
        assert {'jvp': 'tangent', 'vjp': 'cotangent'}[mode] in labels
    for label, rows in records:
        for row in rows:
            if np.any(row != 0):
                print(f'{label} PMC high ghost: nonzero={np.count_nonzero(row)}', file=sys.stderr)
            np.testing.assert_array_equal(row, np.zeros(row.shape, row.dtype))


def devices_for(count, two_devices):
    # Root conftest initializes the CPU host devices before importing JAX. The PR lane has
    # two; the three-device cases run in validation.yml's weekly three-device job.
    return two_devices if count == 2 else host_devices(count)


@pytest.mark.parametrize('axis', tuple('xyz'))
@pytest.mark.parametrize('cells', (10, 11))
@pytest.mark.parametrize('count', (1, 2, 3))
@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('cpml', (False, True), ids=('pec', 'cpml'))
def test_builder_scan_entry(monkeypatch, two_devices, axis, cells, count, entry, cpml):
    devices = devices_for(count, two_devices)
    sim = scene(axis, cells, cpml)
    shape = sim._build_nonuniform_grid().shape
    assert shape[0] == cells + 1
    if cells == 10 and count > 1:
        assert shape[0] % count, 'must exercise the padded short last slab'
    options = dict(n_steps=2, devices=devices, skip_preflight=True)
    if entry == 'forward':
        options.update(distributed=count > 1, devices=devices if count > 1 else None, checkpoint=False)
    else:
        options['compute_s_params'] = False
    with observe(monkeypatch, shape, count, axis, jnp.array(1.), 'primal') as records:
        result = getattr(sim, entry)(**options)
        jax.block_until_ready(result.time_series)
        jax.effects_barrier()
    assert_records(records, 'primal')
    jax.clear_caches()


def upml_scene(axis):
    # Uniform grid; UPML on the two transverse axes, PEC/PMC on the PMC axis.
    boundaries = {a: 'upml' for a in 'xyz'}
    boundaries[axis] = Boundary('pec', 'pmc')
    sim = Simulation(freq_max=12e9, domain=(.010, .008, .008), dx=.001,
                     boundary=BoundarySpec(**boundaries), cpml_layers=2)
    sim.add_source((.004, .003, .003), 'ez', amplitude_kind='field')
    sim.add_probe((.004, .003, .003), 'ez')
    return sim


@pytest.mark.parametrize('axis', tuple('xyz'))
@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_upml_builder_scan_entry(monkeypatch, axis, entry):
    """Uniform UPML holds its PMC ghost (db_h folded to zero), so its entry must be zero too."""
    sim = upml_scene(axis)
    shape = sim._build_grid().shape
    options = dict(n_steps=2, skip_preflight=True)
    options.update(checkpoint=False) if entry == 'forward' else options.update(compute_s_params=False)
    with observe(monkeypatch, shape, 1, axis, jnp.array(1.), 'primal') as records:
        result = getattr(sim, entry)(**options)
        jax.block_until_ready(result.time_series)
        jax.effects_barrier()
    assert_records(records, 'primal')
    jax.clear_caches()


@pytest.mark.parametrize('count', (1, 2, 3))
@pytest.mark.parametrize('mode', ('jvp', 'vjp'))
@pytest.mark.parametrize('chunk', (None, 2))
@pytest.mark.parametrize('cpml', (False, True), ids=('pec', 'cpml'))
def test_ad_scan_entry(monkeypatch, two_devices, count, mode, chunk, cpml):
    devices = devices_for(count, two_devices)
    sim = scene('x', 10, cpml)
    shape = sim._build_nonuniform_grid().shape
    records = []

    def solve(parameter):
        with observe(monkeypatch, shape, count, 'x', parameter, mode) as observed:
            result = sim.forward(eps_override=jnp.ones(shape) * parameter,
                devices=devices if count > 1 else None, distributed=count > 1, n_steps=5, checkpoint=True,
                checkpoint_every=chunk, skip_preflight=True)
            # Keep the observation list alive through traced execution/rematerialisation.
            records.append(observed)
            return result.time_series

    if mode == 'jvp':
        result = jax.jvp(solve, (jnp.array(1.),), (jnp.array(.2),))
    else:
        value, pullback = jax.vjp(solve, jnp.array(1.))
        result = pullback(jnp.ones_like(value))
    jax.block_until_ready(result)
    jax.effects_barrier()
    samples = [item for group in records for item in group]
    assert_records(samples, mode)
    if chunk is not None:
        assert sum(label == 'primal' for label, _ in samples) >= 3, 'segments and tail not reached'
    assert any(np.any(np.asarray(a) != 0) for a in jax.tree.leaves(result)), 'AD witness is vacuous'
    jax.clear_caches()

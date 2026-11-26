"""Assembly decisions and one-device point-probe reads share one measurement rule.

Multi-device probes restore payload tables and retain the general projection.
"""
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_state
from rfx.measurement import projector
from rfx.measurement.plan import Node, Owner, field_channel
from rfx.stepping.slab_measurements import local_table, table_metadata
from tests.contracts.path_equivalence.builders import build


@pytest.fixture(autouse=True)
def cpu():
    with jax.default_device(jax.devices('cpu')[0]):
        yield


def point_table(nodes=(Node('ez', 2, 3, 1, 1.0),)):
    owner = Owner('probe', 'probe', (field_channel('ez', nodes),))
    return projector.sample_table((owner,), dtype=np.float32)


def test_decision():
    table = point_table()
    assert table.direct == (('ez', 2, 3, 1),)
    assert point_table((Node('ez', 2, 3, 1, .5),)).direct is None
    assert point_table((Node('ez', 2, 3, 1, 1.), Node('ez', 3, 3, 1, 1.))).direct is None
    plane = point_table(tuple(Node('ez', i, j, 1, 1.) for i in (2, 3) for j in (2, 3)))
    assert plane.boxes and plane.direct is None

    @jax.jit
    def traced_weight(weight):
        traced = point_table((Node('ez', 2, 3, 1, weight),))
        assert traced.direct is None
        return projector.project(init_state((5, 5, 4)), traced)

    np.testing.assert_array_equal(traced_weight(jnp.float32(1.)), [0.])
    payload = jax.tree.map(lambda a: jnp.asarray(a)[None],
                           (tuple((t.indices, t.channel_ids, t.weights) for t in table.components),
                            table.order, table.segment_ids))
    restored = local_table(table_metadata(table), payload)
    assert restored.direct is None

    @jax.jit
    def traced_payload(data):
        restored = local_table(table_metadata(table), data)
        assert restored.direct is None
        return projector.project(init_state((5, 5, 4)), restored)

    np.testing.assert_array_equal(traced_payload(payload), [0.])
    mixed = Owner('probes', 'probe', (
        field_channel('hz', (Node('hz', 1, 2, 3, 1.),)),
        field_channel('ex', (Node('ex', 3, 2, 1, 1.),))))
    assert projector.sample_table((mixed,), dtype=np.float32).direct == (
        ('hz', 1, 2, 3), ('ex', 3, 2, 1))


def normalized_bits(value):
    values = np.asarray(value)
    assert values.dtype == np.float32
    return (values + np.float32(0.)).view(np.uint32)


def test_signed_zero_and_special_values():
    values = np.array([-0., 0., -1., np.inf, -np.inf, np.nan], np.float32)
    owner = Owner('probes', 'probe', tuple(
        field_channel('ez', (Node('ez', i, 0, 0, 1.),)) for i in range(len(values))))
    table = projector.sample_table((owner,), dtype=np.float32)
    state = init_state((len(values), 1, 1))._replace(ez=jnp.asarray(values[:, None, None]))
    direct = np.asarray(projector.project(state, table))
    general = np.asarray(projector.project(state, table._replace(direct=None)))
    assert direct[0].view(np.uint32) == np.uint32(0x80000000)
    assert general[0].view(np.uint32) == np.uint32(0)
    np.testing.assert_array_equal(normalized_bits(direct), normalized_bits(general))


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_forward_series_and_eps_gradient(mesh):
    sim = build(('_materials', 'eps'), 'fwd_uniform' if mesh == 'uniform' else 'fwd_nonuniform',
                graded=mesh == 'graded')
    grid = sim._build_realized_grid()
    eps = jnp.full(grid.shape, 2.2, jnp.float32)
    compile_table = projector.sample_table

    def evaluate(force_general):
        seen = []

        def compile(*args, **kwargs):
            table = compile_table(*args, **kwargs)
            if table.n_channels and all(name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
                                        for name in table.channel_names):
                seen.append(table)
            return table._replace(direct=None) if force_general else table

        def energy(material):
            series = sim.forward(eps_override=material, n_steps=12, checkpoint=False,
                                 skip_preflight=True).time_series
            return jnp.sum(series**2), series

        with patch.object(projector, 'sample_table', compile):
            (_, series), gradient = jax.value_and_grad(energy, has_aux=True)(eps)
        assert seen
        return series, gradient

    series, gradient = evaluate(False)
    general_series, general_gradient = evaluate(True)
    assert np.any(np.asarray(series) != 0) and np.any(np.asarray(gradient) != 0)
    np.testing.assert_array_equal(normalized_bits(series), normalized_bits(general_series))
    np.testing.assert_array_equal(normalized_bits(gradient), normalized_bits(general_gradient))


def test_uniform_step_probe_read_jaxpr():
    from rfx.stepping import uniform
    from rfx.stepping.sequence import HookPoint

    sim = build(('_materials', 'eps'), 'fwd_uniform')
    captured = []
    compose = uniform.compose

    def capture(kernels, attachments):
        captured.extend(attachments[HookPoint.AFTER_SOURCES])
        return compose(kernels, attachments)

    with patch.object(uniform, 'compose', capture):
        sim.forward(n_steps=2, checkpoint=False, skip_preflight=True)
    assert len(captured) == 1
    state = init_state(sim._build_realized_grid().shape)

    def read(st):
        frame = SimpleNamespace(st=st)
        captured[0](frame)
        return frame.probe_out

    jaxpr = jax.make_jaxpr(read)(state)
    names = [eq.primitive.name for eq in jaxpr.jaxpr.eqns]
    assert 'gather' not in names, names
    assert 'scatter-add' not in names, names
    assert 'slice' in names, names

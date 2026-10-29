"""Stage 1 reads boolean zero masks in the scan and enforces E = 0 in PEC."""
from collections import namedtuple
import re
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.stepping import uniform_setup
from rfx.stepping.conductors import apply_conformal_pec


class _Captured(Exception):
    pass


def _step_jaxpr(sim, **options):
    traces = []
    def capture(fn, carry, xs=None, length=None, **kwargs):
        first = jax.tree.map(lambda x: x[0], xs) if xs is not None else None
        traces.append(jax.make_jaxpr(fn)(carry, first))
        raise _Captured
    with patch.object(jax.lax, 'scan', capture), pytest.raises(_Captured):
        sim.run(n_steps=1, skip_preflight=True, **options)
    assert len(traces) == 1
    return traces[0]


def _equations(jaxpr, origin=''):
    """Inline call/shard-map jaxprs in program order, retaining call sites."""
    jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
    for eq in jaxpr.eqns:
        source = origin + '\n' + str(eq.source_info.traceback) + '\n' + str(eq.params.get('name', ''))
        children = []
        for value in eq.params.values():
            values = value if isinstance(value, (tuple, list)) else (value,)
            children.extend(v for v in values if hasattr(v, 'jaxpr') or hasattr(v, 'eqns'))
        if children:
            for child in children:
                yield from _equations(child, source)
        else:
            yield eq, source


def _called(source, name):
    return re.search(r'(?<![\w])' + name + r'(?![\w])', source) is not None


def _scene():
    sim = Simulation(30e9, (.013, .011, .009), dx=.001,
                     boundary='cpml', cpml_layers=3)
    sim.add(Box((.0062, .0023, .0017), (.0101, .0064, .0053)), material='pec')
    sim.add_source((.004, .005, .003), 'ey', amplitude_kind='current')
    sim.add_probe((.005, .006, .004), 'ey')
    return sim


def test_conformal_scan_reads_boolean_zero_masks():
    body = _step_jaxpr(_scene(), conformal_pec=True)
    selects = []
    for eq, source in _equations(body):
        # Inspect the whole body: moving the compare just outside the helper
        # must not evade the pin. This scene has no other float-grid zero test.
        if eq.primitive.name == 'eq' and any(
                hasattr(v, 'val') and np.shape(v.val) == () and v.val == 0
                for v in eq.invars):
            assert not any(getattr(v.aval, 'dtype', None) == np.dtype('float32')
                           and len(getattr(v.aval, 'shape', ())) == 3
                           for v in eq.invars), 'float32 conformal weight compared inside scan'
        if eq.primitive.name == 'select_n' and _called(source, 'apply_conformal_pec'):
            selects.append(eq)
            assert eq.invars[0].aval.dtype == np.dtype('bool'), 'conformal zero must read bool'
    assert len(selects) == 3, 'scan must apply all three conformal zero masks'


def test_conformal_zero_edges_are_exactly_zero_after_steps():
    weights, masks, final = [], [], []
    original_invariants = uniform_setup.core_step_invariants
    original_scan = jax.lax.scan

    def remember(ctx):
        invariants = original_invariants(ctx)
        weights.extend(ctx.materials.components.tensor.conformal_weights)
        masks.extend(invariants['conformal_zero_masks'])
        return invariants

    def scan(fn, carry, *args, **kwargs):
        result = original_scan(fn, carry, *args, **kwargs)
        if isinstance(carry, dict) and 'fdtd' in carry:
            final.append(result[0]['fdtd'])
        return result

    with patch.object(uniform_setup, 'core_step_invariants', remember), \
            patch.object(jax.lax, 'scan', scan):
        result = _scene().run(n_steps=20, conformal_pec=True, skip_preflight=True)
    assert len(weights) == len(masks) == 3 and len(final) == 1
    assert np.any(np.asarray(result.time_series) != 0), 'scene must have a live field'
    for component, weight, mask in zip(('ex', 'ey', 'ez'), weights, masks):
        expected = np.asarray(weight) == 0
        assert expected.any(), 'PEC body must contain fully covered edges'
        assert np.asarray(mask).dtype == np.dtype('bool')
        np.testing.assert_array_equal(np.asarray(mask), expected)
        assert np.all(np.asarray(getattr(final[0], component))[expected] == 0.0), \
            'electric field on every fully covered PEC edge must be exactly zero'


def test_conformal_zero_refuses_float_weights():
    # w = 0 inside PEC and 1 in free space: as a condition the weights would
    # zero the open edge and keep the field inside the conductor.
    weight = np.array([0.0, 1.0], np.float32)
    state = namedtuple('E', 'ex ey ez')(*(np.ones(2, np.float32),) * 3)
    with pytest.raises(TypeError, match='boolean w == 0 mask'):
        apply_conformal_pec(state, weight, weight, weight)

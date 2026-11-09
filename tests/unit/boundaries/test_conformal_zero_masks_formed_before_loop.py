"""Stage 1 reads boolean zero masks in the scan and enforces E = 0 in PEC."""
from collections import namedtuple
import re
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
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


def test_conformal_final_fields_zero_inside_pec_and_live_on_cut_edges():
    result = _scene().run(n_steps=20, conformal_pec=True, skip_preflight=True)
    grid = result.grid
    assert grid.shape == (20, 18, 16)
    lo, hi = np.array((.0062, .0023, .0017)), np.array((.0101, .0064, .0053))
    nodes = np.meshgrid(*(np.array([grid.node_of(a, i) for i in range(grid.shape[a])])
                          for a in range(3)), indexing='ij')
    cut_peak = 0.
    for axis, component in enumerate(('ex', 'ey', 'ez')):
        # A Yee edge runs from its node to the next node along its component.
        inside = np.ones(grid.shape, bool)
        intersects = np.ones(grid.shape, bool)
        for a in range(3):
            end = nodes[a] + (grid.dx if a == axis else 0.)
            inside &= (nodes[a] >= lo[a]) & (end <= hi[a])
            intersects &= (end > lo[a]) & (nodes[a] < hi[a])
        partial = intersects & ~inside
        field = np.asarray(getattr(result.state, component))
        assert inside.any() and partial.any(), f'{component}: empty geometry witness'
        error = float(np.max(np.abs(field[inside])))
        live = float(np.max(np.abs(field[partial])))
        cut_peak = max(cut_peak, live)
        print(f"{component}: mesh={grid.shape} steps=20 covered={inside.sum()} error={error:.17g} bar=0 partial_peak={live:.17g}")
        assert error == 0., f'{component}: fully covered PEC edge field {error:.17g} != 0'
        dead = int(np.count_nonzero(field[partial] == 0))
        assert dead == 0, f'{component}: {dead} of {partial.sum()} cut edges read exactly 0'
    probe_peak = float(np.max(np.abs(np.asarray(result.time_series))))
    print(f'conformal: cut_peak={cut_peak:.17g} probe_peak={probe_peak:.17g}')
    assert cut_peak > 0, 'partially covered edges must stay live'
    assert probe_peak > 0, 'scene must have a live probe record'


def test_conformal_zero_refuses_float_weights():
    # w = 0 inside PEC and 1 in free space: as a condition the weights would
    # zero the open edge and keep the field inside the conductor.
    weight = np.array([0.0, 1.0], np.float32)
    state = namedtuple('E', 'ex ey ez')(*(np.ones(2, np.float32),) * 3)
    with pytest.raises(TypeError, match='boolean w == 0 mask'):
        apply_conformal_pec(state, weight, weight, weight)

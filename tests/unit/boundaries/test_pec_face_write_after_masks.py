"""A domain-face partial write follows every whole-grid E conductor operator."""
import re
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from tests._multi_device_scene import host_devices
from tests.unit.boundaries.test_sheet_on_pec_face_wall_wins import wall_scene


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


def _assert_order(jaxpr, *, sheet=False, shape=None):
    masks, sheets, faces = [], [], []
    for index, (eq, source) in enumerate(_equations(jaxpr)):
        arrays = [v.aval for v in eq.outvars if hasattr(v, 'aval')]
        arrays = [a for a in arrays if len(getattr(a, 'shape', ())) == 3
                  and np.issubdtype(a.dtype, np.inexact)
                  and (shape is None or a.shape == shape)]
        if not arrays:
            continue
        op = eq.primitive.name
        if op in ('mul', 'select_n') and any(_called(source, n) for n in (
                'apply_pec_edges', 'apply_pec_mask_shmap', 'apply_edge_keep', 'apply_conformal_pec')):
            masks.append(index)
        if op == 'select_n' and _called(source, 'apply_sheet_impedance_e'):
            sheets.append(index)
        if op in ('scatter', 'dynamic_update_slice') and any(_called(source, n) for n in (
                'apply_pec', 'apply_pec_faces', 'apply_pec_face_shmap')):
            faces.append(index)
    assert masks, 'scene must exercise a whole-grid E PEC operator'
    assert faces, 'scene must exercise domain-face writes'
    assert max(masks) < min(faces), f'whole-grid E PEC operator {max(masks)} must precede every face write {faces}'
    if sheet:
        assert sheets, 'scene must exercise the sheet select'
        assert max(sheets) < min(faces), f'sheet select {max(sheets)} must precede every face write {faces}'


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_single_device_masks_and_sheet_precede_faces(graded):
    sim = wall_scene(graded)
    sim.add(Box((.007, .002, .001), (.009, .004, .003)), material='pec')
    _assert_order(_step_jaxpr(sim), sheet=True)


@pytest.mark.parametrize('path', ['v2-cpml', 'v2-pec', 'nu', 'nu-wrapped'])
def test_two_device_masks_precede_faces(path):
    devices = host_devices(2)
    graded = path.startswith('nu')
    mesh = {'dy_profile': np.array([1., .9, 1.1, 1., 1., 1., 1., 1., 1., 1., 1.]) * .001} if graded else {}
    boundary = 'cpml' if path == 'v2-cpml' else 'pec'
    if path == 'nu':
        # A PMC face selects the dedicated NU runner on this base; plain
        # PEC uses the canonical graded step through the slab wrapper.
        from rfx.boundaries.spec import Boundary, BoundarySpec
        boundary = BoundarySpec(x='pec', y='pec', z=Boundary(lo='pec', hi='pmc'))
    sim = Simulation(30e9, (.013, .011, .009), dx=.001,
                     boundary=boundary, cpml_layers=3, **mesh)
    sim.add(Box((.005, .002, .002), (.008, .004, .005)), material='pec')
    sim.add_source((.003, .005, .003), 'ey', amplitude_kind='current')
    sim.add_probe((.010, .006, .004), 'ey')
    _assert_order(_step_jaxpr(sim, devices=devices))


def _subgrid_scene():
    sim = Simulation(30e9, (.013, .011, .009), dx=.001, boundary='pec', cpml_layers=0)
    sim.add(Box((.002, .003, .001), (.004, .005, .005)), material='pec')
    sim.add_source((.005, .006, .004), 'ey', amplitude_kind='current')
    sim.add_probe((.008, .007, .005), 'ey')
    sim.add_refinement(z_range=(.003, .006), ratio=2, validation='research')
    return sim


def test_subgridded_coarse_masks_precede_faces():
    sim = _subgrid_scene()
    _assert_order(_step_jaxpr(sim), shape=sim._build_grid().shape)


@pytest.mark.parametrize('path', ['sbp-sat', 'coarse-reference'])
def test_reference_masks_precede_faces(path):
    # Public run assembles the coarse material and PEC volume mask. The
    # reference SBP-SAT half-step consumes those same assembled operands.
    from rfx.subgridding import jit_runner
    from rfx.subgridding.sbp_sat_3d import _update_e_only
    from rfx.core.yee import init_state
    captured = {}
    def capture(grid_c, mats_c, mats_f, config, n_steps, **kwargs):
        captured.update(grid=grid_c, mats=mats_c, fine=mats_f,
                        config=config, mask=kwargs['opts'].pec_mask_c)
        raise _Captured
    with patch.object(jit_runner, 'run_subgridded_jit', capture), pytest.raises(_Captured):
        _subgrid_scene().run(n_steps=1, skip_preflight=True)
    c = captured
    if path == 'coarse-reference':
        from rfx.subgridding import runner
        # This reference is not run()'s runner. Its legacy coupling expects
        # cell-count fine shapes, unlike the current node-count API. Trace
        # its actual E half-step up to coupling; do not change its arithmetic
        # or manufacture a second geometry/material assembly for the test.
        half_step = {}
        def at_coupling(coarse, fine, *args, **kwargs):
            half_step['e'] = coarse
            raise _Captured
        def one_step(mats):
            try:
                runner.run_subgridded(c['grid'], mats, None, c['fine'],
                                     c['config'], 1, pec_mask_c=c['mask'], cpml_axes='')
            except _Captured:
                return half_step['e']
            raise AssertionError('reference must reach the coupling boundary')
        with patch.object(runner, '_shared_node_coupling_3d', at_coupling):
            _assert_order(jax.make_jaxpr(one_step)(c['mats']), shape=c['grid'].shape)
        return
    state = init_state(c['grid'].shape)
    def half_step(st):
        return _update_e_only(st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
                              c['config'].dt, c['config'].dx_c,
                              mats=c['mats'], pec_mask=c['mask'], boundary_pec=True)
    _assert_order(jax.make_jaxpr(half_step)(state))

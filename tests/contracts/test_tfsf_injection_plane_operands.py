"""Injection operands: public declaration matrix and a closed classification.

Coordinates reproduce DESIGN F3/placement.py; expected planes come only from
production source configuration, never from the boundary rule's helpers.
"""
import ast
import inspect
import json
import sys
import warnings
from dataclasses import fields
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.boundaries.tfsf import INJECTION_OPERANDS, injection_material_rule
from rfx.core.yee import MaterialArrays
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole, drude_pole
from rfx.sources.tfsf import tfsf_from_declaration

INVENTORY = ('drude', 'kerr', 'pec_volume', 'pec_sheet', 'pec_wire',
             'pinned_sheet', 'thin_pec_sheet', 'surface_impedance', 'series_RL')
OBJECTS = INVENTORY + ('eps', 'sigma', 'mu', 'debye', 'lorentz', 'R_only')
LANES = ('run_uniform', 'fwd_uniform', 'run_distributed', 'fwd_distributed',
         'run_graded', 'fwd_graded')


def mm(*values):
    return tuple(v * 1e-3 for v in values)


def scene(obj=None, *, graded=False, interior=False, kind='slab'):
    width = 24 if interior or kind != 'slab' else 13
    boundary = ('cpml' if graded or kind != 'slab' else
                BoundarySpec(x=Boundary(lo='cpml', hi='cpml'), y='periodic', z='periodic'))
    if kind == 'methodB':
        boundary = BoundarySpec(x='cpml', y='cpml', z='periodic')
    sim = Simulation(freq_max=10e9, domain=mm(width, 24 if kind != 'slab' else 12,
                                           24 if kind == 'box' else 12),
                     dx=1e-3, boundary=boundary, cpml_layers=8,
                     **({'dz_profile': np.full(12, 1e-3)} if graded else {}))
    sim.add_probe(mm(10, 8, 8), 'ez')
    sim.add_tfsf_source(f0=5e9, bandwidth=.5, margin=6, polarization='ez', direction='+x',
                        **({'closed_box': True} if kind == 'box' else
                           {'angle_deg': 20., 'method': 'methodB'} if kind == 'methodB' else {}))
    x = 11 if interior else 6
    ys, zs = (8, 16) if kind != 'slab' else (0, 12), (8, 16) if kind == 'box' else (0, 12)
    slab = Box(mm(x-1, ys[0], zs[0]), mm(x+1, ys[1], zs[1]))
    sheet = Box(mm(x, ys[0], zs[0]), mm(x, ys[1], zs[1]))
    medium = {
        'drude': dict(lorentz_poles=[drude_pole(2*np.pi*10e9, 1e10)]),
        'kerr': dict(chi3=1e3), 'eps': dict(eps_r=2.), 'sigma': dict(sigma=.1),
        'mu': dict(mu_r=2.), 'debye': dict(debye_poles=[DebyePole(1., 1e-10)]),
        'lorentz': dict(lorentz_poles=[LorentzPole(2*np.pi*5e9, 1e10, (2*np.pi*5e9)**2)]),
    }
    if obj in medium:
        sim.add_material('blk', **medium[obj])
        sim.add(slab, material='blk')
    elif obj == 'pec_volume':
        sim.add(slab, material='pec')
    elif obj == 'pec_sheet':
        sim.add(sheet, material='pec')
    elif obj == 'pec_wire':
        sim.add(PolylineWire((mm(x, 6, 0), mm(x, 6, 12)), radius=0.), material='pec')
    elif obj == 'pinned_sheet':
        sim.add_pinned_sheet(plane_index=x, i_range=(0, 12), j_range=(0, 12), normal_axis=0)
    elif obj in ('thin_pec_sheet', 'surface_impedance'):
        sim.add_thin_conductor(sheet, **({} if obj == 'thin_pec_sheet' else
                              dict(sigma_bulk=1e3, thickness=1e-4, surface_impedance_f0=5e9)))
    elif obj in ('series_RL', 'R_only'):
        sim.add_lumped_rlc(mm(x, 6, 6), 'ez', R=10., L=1e-9 if obj == 'series_RL' else 0.)
    return sim


def execute(sim, lane):
    kwargs = dict(n_steps=1, skip_preflight=True)
    if lane.endswith('distributed'):
        devices = jax.devices('cpu')
        assert len(devices) >= 2
        kwargs['devices'] = devices[:2]
    if lane.startswith('run'):
        return sim.run(**kwargs, compute_s_params=False)
    return sim.forward(**kwargs, checkpoint=False,
                       **({'distributed': True} if lane.endswith('distributed') else {}))


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('obj', OBJECTS)
def test_refusal_matrix(obj, lane, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = scene(obj, graded=lane.endswith('graded'))
        grid = sim._build_nonuniform_grid() if lane.endswith('graded') else sim._build_grid()
        cfg = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=1)[0]
        # Independently pinned by placement.py's geometry: x=6mm is x_lo=14.
        assert cfg.x_lo == 14 and cfg.x_hi == 15
        started = []
        original = jax.lax.scan

        def scan(*args, **kwargs):
            if '/rfx/stepping/' in sys._getframe(1).f_code.co_filename:
                started.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(jax.lax, 'scan', scan)
        with pytest.raises((ValueError, NotImplementedError)) as caught:
            execute(sim, lane)
        assert not started
        message = str(caught.value)
        new = 'non-vacuum material found at' in message and ': ' in message and message.endswith(')')
        if (obj in INVENTORY and not lane.endswith('graded')) or obj == 'pec_volume':
            assert new, message
        elif not new:
            assert lane.endswith('graded') and any(s in message for s in (
                'prevents an invariant periodic replacement', 'not carried by the graded',
                'only on the uniform', 'resolve the wire as a volume',
                'on the non-uniform', 'non-uniform mesh', 'graded mesh')), message
        print(json.dumps(dict(object=obj, lane=lane,
                              outcome='refused-by-new-rule' if new else 'refused-earlier',
                              message=message)), file=sys.stderr)


@pytest.mark.parametrize('obj', INVENTORY)
def test_off_plane_controls(obj):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = scene(obj, interior=True)
        grid = sim._build_grid()
        cfg = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=1)[0]
        assert cfg.x_hi - cfg.x_lo >= 6
        # Objects now occupy x=18,19 (edges through 20), safely inside 14..26.
        assert cfg.x_lo + 2 <= 18 and 20 <= cfg.x_hi - 2
        execute(sim, 'run_uniform')


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('kind,x,smoothing', [
    ('dielectric', 6.5, True),
    ('dielectric', 7.2, 'full_tensor'),
    ('dielectric', 7.5, 'full_tensor'),
    ('series_RL', 6.5, False),
    ('series_RL', 18.5, False),
])
def test_normal_component_half_cell_admitted(entry, kind, x, smoothing):
    # Review scenes: 24 mm, 1 mm cells, 8 pad cells, margin 6. Literal
    # positions give faces +0.5/+1.2/+1.5 cells from x_lo and Ex edges
    # at x_lo+1/2 and x_hi+1/2; no slicing/placement helper sets these values.
    sim = scene(interior=True)
    cfg = tfsf_from_declaration(sim._tfsf, sim._build_grid(), sim._freq_max, n_steps=2)[0]
    assert (cfg.x_lo, cfg.x_hi) == (14, 26)
    if kind == 'dielectric':
        sim.add_material('dielectric', eps_r=4.)
        sim.add(Box(mm(x, 0, 0), mm(15, 12, 12)), material='dielectric')
    else:
        sim.add_lumped_rlc(mm(x, 6, 6), 'ex', R=10., L=1e-9)
    kwargs = {'compute_s_params': False} if entry == 'run' else {'checkpoint': False}
    result = getattr(sim, entry)(n_steps=2, skip_preflight=True,
                                subpixel_smoothing=smoothing, **kwargs)
    assert np.isfinite(np.asarray(result.time_series)).all()


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_tangential_series_rl_on_plane_refused(entry):
    sim = scene(interior=True)
    sim.add_lumped_rlc(mm(6, 6.5, 6), 'ey', R=10., L=1e-9)
    with pytest.raises(ValueError, match=r'x_lo: lumped RLC element \(lumped_rlc.ey\)'):
        getattr(sim, entry)(n_steps=2, skip_preflight=True)


@pytest.mark.parametrize('kind', ['box', 'methodB'])
@pytest.mark.parametrize('obj', ['pec_sheet', 'drude'])
def test_other_injection_kinds(kind, obj):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError) as caught:
            execute(scene(obj, kind=kind), 'run_uniform')
    message = str(caught.value)
    assert ('closed_box TFSF' if kind == 'box' else 'Method-B oblique TFSF') in message
    assert 'non-vacuum material found at' in message
    assert ': ' in message and message.endswith(')')


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('kind', ['staircase_eps', 'series_RL_ez', 'thin_pec_sheet'])
def test_uncorrected_high_e_component_admitted(entry, kind):
    sim = scene(interior=True)
    cfg = tfsf_from_declaration(sim._tfsf, sim._build_grid(), sim._freq_max, n_steps=2)[0]
    assert (cfg.x_lo, cfg.x_hi) == (14, 26)
    # 8 pad cells and 1 mm cells: node x_hi=26 is x=18 mm. The literal
    # slab [9,18) mm ends with cell 25 (center 17.5 mm), i.e. x_hi-1.
    # Its tangential E mean, the Ez RL and the sheet reach node x_hi only.
    if kind == 'staircase_eps':
        sim.add_material('dielectric', eps_r=4.)
        sim.add(Box(mm(9, 0, 0), mm(18, 12, 12)), material='dielectric')
    elif kind == 'series_RL_ez':
        sim.add_lumped_rlc(mm(18, 6, 6), 'ez', R=10., L=1e-9)
    else:
        sim.add_thin_conductor(Box(mm(18, 0, 0), mm(18, 12, 12)))
    kwargs = {'compute_s_params': False} if entry == 'run' else {'checkpoint': False}
    result = getattr(sim, entry)(n_steps=2, skip_preflight=True, **kwargs)
    assert np.isfinite(np.asarray(result.time_series)).all()


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_staircase_slab_filling_high_cell_refused(entry):
    sim = scene(interior=True)
    cfg = tfsf_from_declaration(sim._tfsf, sim._build_grid(), sim._freq_max, n_steps=2)[0]
    assert cfg.x_hi == 26
    # One extra cell: [9,19) mm fills cell x_hi=26 (center 18.5 mm).
    sim.add_material('dielectric', eps_r=4.)
    sim.add(Box(mm(9, 0, 0), mm(19, 12, 12)), material='dielectric')
    with pytest.raises(ValueError, match=r'x_hi: permittivity \(materials.eps_r\)'):
        getattr(sim, entry)(n_steps=2, skip_preflight=True)


def test_setup_parameter_classification_is_complete():
    from rfx.stepping.uniform_setup import _build_step_setup
    from rfx.stepping.graded_setup import _build_nu_scan
    for setup in (_build_step_setup, _build_nu_scan):
        parameters = set(inspect.signature(setup).parameters)
        assert parameters <= INJECTION_OPERANDS.keys()
        # Read the call itself, independently of the rule's traversal. Materials
        # has its own argument; updates/localized also carry non-operands.
        calls = [node for node in ast.walk(ast.parse(inspect.getsource(setup)))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == 'admit_setup']
        assert len(calls) == 1
        keywords = {kw.arg: kw.value for kw in calls[0].keywords}
        assert isinstance(keywords['materials'], ast.Name)
        assert keywords['materials'].id == 'materials'
        updates = keywords['updates']
        assert isinstance(updates, ast.Call) and updates.func.id == 'dict'
        passed = {kw.arg for kw in updates.keywords}
        passed.update(ast.literal_eval(pair.elts[0]) for pair in keywords['localized'].elts)
        assert passed <= INJECTION_OPERANDS.keys()
        expected = {name for name in parameters - {'materials'}
                    if INJECTION_OPERANDS[name][0] is not None}
        assert expected == {name for name in passed if INJECTION_OPERANDS[name][0] is not None}


def test_material_leaf_classification_is_complete():
    from rfx.model.materials import ComponentMaterials, ComponentDispersion, KernelComponents
    from rfx.model.tensor_operands import TensorOperands
    from rfx.geometry.node_tensor import NodeTensor
    # Independent structural enumeration, including optional leaves absent in
    # an empty scene. New fields fail even when defaulted to None in production.
    nested = {'components': ComponentMaterials, 'tensor': TensorOperands,
              'node_smoothing': NodeTensor, 'debye': ComponentDispersion,
              'lorentz': ComponentDispersion}
    nonspatial = {'periodic', 'grid_key', 'source_upml', 'inv_eps_smooth', 'guard_count', 'poles', 'dt'}

    def visit(cls, prefix):
        # Static dataclass fields are pytree metadata, never an operand array.
        names = cls._fields if hasattr(cls, '_fields') else [
            f.name for f in fields(cls) if not f.metadata.get('static')]
        for name in names:
            if name in nonspatial or name == 'cells_view':
                continue
            path = prefix + '.' + name
            if name in nested:
                visit(nested[name], path)
            else:
                # Scalar volume arrays and component / pole tuple entries.
                for candidate in (path, path + '.0'):
                    try:
                        injection_material_rule(candidate)
                        break
                    except ValueError:
                        pass
                else:
                    pytest.fail(f'unclassified spatial leaf: {path}')
    visit(MaterialArrays, 'materials')
    visit(KernelComponents, 'materials.components')


@pytest.mark.parametrize('transform', ['jit', 'grad', 'vmap'])
def test_traced_override_is_judged(transform):
    from rfx.boundaries.tfsf import refuse_injection_plane_operands
    cfg = SimpleNamespace(x_lo=2, x_hi=5, closed_box=False)
    shape = (8, 3, 3)

    def checked(value):
        eps = jnp.ones(shape).at[2].set(value)
        refuse_injection_plane_operands(cfg, SimpleNamespace(shape=shape),
                                       MaterialArrays(eps, jnp.zeros(shape), jnp.ones(shape)), {}, ())
        return value * value

    call = {'jit': jax.jit(checked), 'grad': jax.grad(checked),
            'vmap': jax.jit(jax.vmap(checked))}[transform]
    with pytest.raises(Exception, match='requires vacuum'):
        jax.block_until_ready(call(jnp.array([2.]) if transform == 'vmap' else jnp.array(2.)))

"""Whole-grid realization is structural on CPU and a value gate on GPUs."""
import re
import sys
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from rfx import Simulation, Box, DebyePole
from rfx.materials.lorentz import lorentz_pole
from rfx.model.electric_metrics import stage_forward_materials
from rfx.model.materials import realize_components
from rfx.stepping import slab_setup as setup
from rfx.stepping.slab import Slab
from rfx.stepping.slab_view import stack_and_place, uniform_slab_rows

pytestmark = pytest.mark.distributed


def scene(family='plain', scale=1):
    nx, ny, nz = 23 * scale, 12 * scale, 10 * scale
    dx = np.resize(np.array([.89, 1.07, .94, 1.13]), nx) * 1e-3
    dy = np.resize(np.array([.91, 1.09, 1.03]), ny) * 1e-3
    dx[[0, -1]] = dy[[0, -1]] = 1e-3
    xs, ys = np.r_[0., np.cumsum(dx)], np.r_[0., np.cumsum(dy)]
    sim = Simulation(freq_max=15e9, domain=(float(xs[-1]), float(ys[-1]), nz * 1e-3),
                     dx=1e-3, dx_profile=dx, dy_profile=dy, boundary='cpml', cpml_layers=3)
    pole_kw = ({'debye_poles': [DebyePole(.3, 1e-11), DebyePole(.2, 2e-11)]} if family == 'debye'
               else {'lorentz_poles': [lorentz_pole(.3, 2 * np.pi * 3e9, 1e9),
                                      lorentz_pole(.2, 2 * np.pi * 2e9, 2e9)]}
               if family == 'lorentz' else {})
    sim.add_material('board', eps_r=4.4, sigma=.015, **pole_kw)
    sim.add_material('insert', eps_r=2.2)
    corner = 10 * scale  # Neither the two- nor three-device ownership boundary.
    sim.add(Box((xs[3 * scale], ys[2 * scale], 2e-3 * scale),
                (xs[corner], ys[7 * scale], 8e-3 * scale)), material='board')
    sim.add(Box((xs[corner], ys[7 * scale], 2e-3 * scale),
                (xs[19 * scale], ys[10 * scale], 8e-3 * scale)), material='insert')
    sim.add_source((xs[5 * scale], ys[4 * scale], 4e-3 * scale), 'ez', amplitude_kind='current')
    sim.add_probe((xs[14 * scale], ys[5 * scale], 3e-3 * scale), 'ez')
    grid = sim._build_nonuniform_grid()
    materials, db, lr, _ = sim._assemble_materials_nu(grid)
    corner = int(np.flatnonzero(np.any(np.asarray(materials.eps_r) == np.float32(4.4), axis=(1, 2)))[-1]) + 1
    # Exercise every H component and occupancy as well as the dielectric corner.
    mu = np.ones(grid.shape, np.float32)
    mu[4 * scale:9 * scale, 3 * scale:8 * scale, 2 * scale:7 * scale] = 1.3
    materials = materials._replace(mu_r=jnp.asarray(mu))
    occupancy = np.zeros(grid.shape, np.float32)
    occupancy[6 * scale:9 * scale, 3 * scale:5 * scale, 3 * scale:6 * scale] = .25
    return SimpleNamespace(sim=sim, grid=grid, materials=materials, occupancy=occupancy,
                           spec=db if family == 'debye' else lr, family=family, corner=corner)


def staged_scene(n, family='plain'):
    devices = jax.devices()[:n]
    if len(devices) < n:
        pytest.skip(f'requires {n} devices')
    s = scene(family)
    s.layout = Slab(s.grid.shape[0], n)
    assert s.corner % s.layout.nx_per_rank != 0
    s.mesh = Mesh(np.asarray(devices), ('x',))
    s.inputs = stage_forward_materials(s.materials, s.layout, s.mesh, uniform=True)
    s.occ = stack_and_place(s.occupancy, uniform_slab_rows(s.layout), s.mesh, outside=0)
    return s


def products(s):
    pole = (None if s.spec is None else setup.stage_dispersion(
        s.inputs, s.grid.dt, s.spec, s.grid, s.layout, s.mesh, s.family))
    mats = setup.realize_materials(s.inputs, s.grid, s.layout, s.mesh, s.occ,
                                  cpml=True, debye=s.family == 'debye', lorentz=s.family == 'lorentz')
    return mats, pole


def build(s):
    pole = (None if s.spec is None else setup.stage_dispersion(
        s.inputs, s.grid.dt, s.spec, s.grid, s.layout, s.mesh, s.family))
    kw = {} if pole is None else {s.family: pole}
    return setup.build_setup(s.grid, s.inputs, s.layout, s.mesh, 2, [], [],
                             occupancy=s.occ, cpml=True, **kw)


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('family', ['plain', 'debye', 'lorentz'])
def test_whole_grid_structure(n, family, monkeypatch):
    s = staged_scene(n, family)
    calls = []
    original = setup.realize_components

    def observe(*args, **kwargs):
        calls.append(kwargs.get('view'))
        result = original(*args, **kwargs)
        if kwargs.get('view') is None:
            assert all(d.platform == 'cpu' for a in jax.tree.leaves(result)
                       if isinstance(a, jax.Array) for d in a.devices())
        return result

    monkeypatch.setattr(setup, 'realize_components', observe)
    result = build(s)
    assert calls and all(view is None for view in calls), 'slab view reached concrete material realization'
    assert result.materials_realized == 'whole_grid_host'
    assert result.materials_realized_reason is None


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('family', ['plain', 'debye', 'lorentz'])
def test_host_products_release_input_layout_copies(n, family):
    s = staged_scene(n, family)
    before = {id(a) for a in jax.live_arrays()}
    result = products(s)
    jax.block_until_ready(result)
    shapes = {s.inputs.eps_r.shape, (1,) + s.inputs.eps_r.shape[1:]}
    retained = [(a.shape, str(a.sharding)) for a in jax.live_arrays()
                if id(a) not in before and a.shape in shapes]
    assert not retained, retained


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('family', ['plain', 'debye', 'lorentz'])
def test_every_product_matches_per_slab(n, family, monkeypatch):
    s = staged_scene(n, family)
    if s.mesh.devices.flat[0].platform != 'cpu':
        pytest.skip('GPU per-slab reference has the corner rounding defect')
    actual = products(s)
    monkeypatch.setattr(setup, '_material_host_device', lambda *args, **kwargs: (None, 'traced inputs'))
    expected = products(s)  # Eager reference: no outer jit changes its rounding.
    a_leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
    e_leaves = jax.tree_util.tree_flatten_with_path(expected)[0]
    # grid_key describes a whole-grid view versus a slab view; array paths agree.
    assert [p for p, _ in a_leaves] == [p for p, _ in e_leaves]
    for (index, a), (_, e) in zip(a_leaves, e_leaves):
        if not hasattr(e, 'shape'):
            assert a == e, index
            continue
        assert a.shape == e.shape, index
        assert a.dtype == e.dtype, index
        assert a.sharding.spec == e.sharding.spec, index
        assert np.array_equal(np.asarray(a), np.asarray(e)), index


@pytest.mark.parametrize('n', [2, 3])
def test_owned_components_match_one_device(n):
    s = staged_scene(n)
    mu = np.asarray(s.materials.mu_r)
    assert np.all(mu[4:9, 3:8, 2:7] == np.float32(1.3))
    assert np.count_nonzero(mu != 1) == 5 * 5 * 5
    staged = products(s)[0][0]
    whole = realize_components(s.materials, s.grid, periodic=(False,) * 3)
    for name in ('eps_update', 'sigma_update', 'mu_update'):
        for actual, expected in zip(getattr(staged.components, name), getattr(whole, name)):
            actual, expected = np.asarray(actual), np.asarray(expected)
            for rank in range(n):
                lo = rank * s.layout.nx_per_rank
                hi = min((rank + 1) * s.layout.nx_per_rank, s.grid.shape[0])
                start = rank * s.layout.nx_local + s.layout.ghost_width
                a, e = actual[start:start + hi - lo], expected[lo:hi]
                if name == 'mu_update' and s.mesh.devices.flat[0].platform == 'gpu':
                    # CPU host vs GPU realization differs by 1 float32 ULP at the mu_r=1.3 box face.
                    assert a.dtype == e.dtype == np.float32
                    np.testing.assert_array_max_ulp(a, e, maxulp=1)
                else:
                    assert np.array_equal(a, e), (name, rank)
                if name == 'mu_update' and lo < 9 and hi > 4:
                    assert np.any(e != 1), 'owned comparison must read the magnetic box'


@pytest.mark.parametrize('n', [2, 3])
def test_no_cpu_fallback_warns_once(n, monkeypatch):
    s = staged_scene(n, 'debye')
    local_devices = jax.local_devices
    monkeypatch.setattr(setup, '_warned_no_cpu', False)

    def unavailable(*args, backend=None, **kwargs):
        if backend == 'cpu':
            raise RuntimeError('CPU backend unavailable')
        return local_devices(*args, backend=backend, **kwargs)

    monkeypatch.setattr(jax, 'local_devices', unavailable)
    calls = []
    original = setup.realize_components

    def observe(*args, **kwargs):
        calls.append(kwargs.get('view'))
        return original(*args, **kwargs)

    monkeypatch.setattr(setup, 'realize_components', observe)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(2):
            result = build(s)
            assert result.materials_realized == 'per_slab'
            assert result.materials_realized_reason == 'no CPU backend'
    runtime = [w for w in caught if w.category is RuntimeWarning]
    assert len(runtime) == 1
    assert runtime[0].filename == __file__
    expected_line = build.__code__.co_firstlineno + 1
    assert runtime[0].lineno == expected_line
    print(f'warning caller={runtime[0].filename}:{runtime[0].lineno}', file=sys.stderr)
    assert str(runtime[0].message) == (
        'no CPU backend in this JAX process (JAX_PLATFORMS does not list cpu), '
        'so the graded multi-device lane forms its material coefficients per slab; '
        'on a GPU they can differ from a one-device run in the last float32 bit '
        'at dielectric corner edges (within every accuracy bar)')
    assert calls and all(view is not None for view in calls)


@pytest.mark.parametrize('leaf', ['materials', 'occupancy', 'mask', 'parameter'])
def test_traced_switch_never_probes_cpu(leaf, monkeypatch):
    def forbidden(*args):
        raise AssertionError('traced switch probed CPU')
    monkeypatch.setattr(jax, 'local_devices', forbidden)

    def call(value):
        inputs = dict(materials=1., occupancy=0., mask=True, parameter=1.)
        inputs[leaf] = value
        device, reason = setup._material_host_device(inputs)
        assert device is None and reason == 'traced inputs'
        return value

    jax.make_jaxpr(call)(1.)


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('family', ['plain', 'debye', 'lorentz'])
def test_closed_over_concrete_products_under_jit(n, family, monkeypatch):
    s = staged_scene(n, family)
    calls = []
    original = setup.realize_components

    def observe(*args, **kwargs):
        calls.append(kwargs.get('view'))
        return original(*args, **kwargs)

    monkeypatch.setattr(setup, 'realize_components', observe)
    from rfx.stepping import rank
    if family == 'lorentz' and s.mesh.devices.flat[0].platform == 'gpu' and rank._command_buffers_on():
        from rfx.stepping.rank import OUTER_JIT_REFUSAL
        with pytest.raises(NotImplementedError, match=re.escape(OUTER_JIT_REFUSAL)):
            jax.jit(lambda: products(s))()
        return
    actual = jax.jit(lambda: products(s))()
    assert calls and all(view is None for view in calls)
    expected = products(s)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.array_equal(np.asarray(a), np.asarray(e))


def test_traced_pole_inputs_keep_setup_on_per_slab_route(monkeypatch):
    s = staged_scene(2, 'debye')
    coeffs, state = setup.stage_dispersion(s.inputs, s.grid.dt, s.spec, s.grid, s.layout, s.mesh, 'debye')
    calls = []
    original = setup.realize_components

    def observe(*args, **kwargs):
        calls.append(kwargs.get('view'))
        return original(*args, **kwargs)

    monkeypatch.setattr(setup, 'realize_components', observe)

    def traced(value):
        pole = coeffs._replace(cb=tuple(a * value for a in coeffs.cb))
        result = setup.build_setup(s.grid, s.inputs, s.layout, s.mesh, 2, [], [],
                                   cpml=True, debye=(pole, state))
        assert result.materials_realized == 'per_slab'
        assert result.materials_realized_reason == 'traced inputs'
        return result.invariants['materials'].components.mu_update

    jax.make_jaxpr(traced)(1.)
    assert calls and all(view is not None for view in calls)


@pytest.mark.parametrize('n', [2, 3])
def test_non_addressable_input_keeps_slab_views(n, monkeypatch):
    s = staged_scene(n, 'debye')
    target = s.inputs.eps_r
    array_type = type(target)
    addressable = array_type.is_fully_addressable
    # Real local JAX array, with only this leaf's addressability metadata overridden.
    monkeypatch.setattr(array_type, 'is_fully_addressable', property(
        lambda a: False if a is target else addressable.__get__(a, array_type)))
    calls = []
    original = setup.realize_components

    def observe(*args, **kwargs):
        calls.append(kwargs.get('view'))
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('non-addressable input reached a host read')

    monkeypatch.setattr(setup, 'realize_components', observe)
    monkeypatch.setattr(setup, '_whole_material_rows', forbidden)
    monkeypatch.setattr(jax, 'local_devices', forbidden)
    result = build(s)
    assert result.materials_realized == 'per_slab'
    assert result.materials_realized_reason == 'multi-process'
    assert calls and all(view is not None for view in calls)


def test_multi_process_mesh_never_probes_cpu(monkeypatch):
    mesh = SimpleNamespace(devices=np.asarray([
        SimpleNamespace(process_index=0), SimpleNamespace(process_index=1)]))

    def forbidden(*args, **kwargs):
        raise AssertionError('multi-process mesh probed CPU')

    monkeypatch.setattr(jax, 'local_devices', forbidden)
    assert setup._material_host_device(np.ones(1), mesh=mesh) == (None, 'multi-process')

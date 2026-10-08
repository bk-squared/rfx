"""Equal-grid realization: exact masks/specs and at most one DC float32 ULP."""
import dataclasses
import itertools
import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Cylinder, PolylineWire, Simulation
from rfx.model.conductors import realized_conductors

CASES = ('dielectric', 'pec_box', 'pec_sheet_on', 'pec_sheet_off', 'thin_pec',
         'dc_on', 'dc_off', 'f0_on', 'f0_off', 'wire', 'cylinder_dielectric',
         'cylinder_pec', 'pad_pec', 'pad_dielectric', 'pad_dc', 'pad_f0', 'mixed')


@pytest.fixture(autouse=True)
def float32_path_comparison():
    # Scalar uniform and float32 graded metrics intentionally retain their
    # original precision. The separate x64 test pins the scalar expression.
    with jax.enable_x64(False):
        yield


def build(case, nu=False, h=1/1024, order=('dc', 'pec', 'f0'), radius=2.5):
    def p(*x):
        return tuple(v*h for v in x)
    counts = (32, 32, 8) if case == 'disc' else (12, 10, 8)
    kw = dict(freq_max=15e9, domain=p(*counts), dx=h, boundary='cpml', cpml_layers=2)
    if nu:
        kw.update({f'd{a}_profile': np.full(n, h) for a, n in zip('xyz', counts)})
    sim = Simulation(**kw)
    sim.add_material('d', eps_r=2.5, sigma=.25, mu_r=1.5)
    box = Box(p(5, 2, 1), p(5, 6, 5))
    off = Box(p(5.4, 2.2, 1.3), p(5.4, 6.2, 5.3))
    vol = Box(p(4.2, 2.1, 1.3), p(7.4, 6.2, 5.3))
    pad = Box(p(4, 2, 1), p(12, 6, 5))
    cylinder = Cylinder(p(6, 5, 4), radius=1.7*h, height=4*h)

    def thin(shape, kind, *, sigma=1000, thickness=None):
        args = dict(sigma_bulk=1e7 if kind == 'pec' else sigma,
                    thickness=h/10 if thickness is None else thickness)
        if kind == 'f0':
            args['surface_impedance_f0'] = 5e9
        sim.add_thin_conductor(shape, **args)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        if case in ('dielectric', 'pec_box'):
            sim.add(vol, material='d' if case == 'dielectric' else 'pec')
        elif case.startswith('pec_sheet'):
            sim.add(off if case.endswith('off') else box, material='pec')
        elif case == 'thin_pec':
            thin(box, 'pec')
        elif case in ('dc_on', 'dc_off', 'f0_on', 'f0_off'):
            thin(off if case.endswith('off') else box, case.split('_')[0])
        elif case == 'wire':
            sim.add(PolylineWire((p(4.2, 4, 2), p(4.2, 4, 6)), radius=0), material='pec')
        elif case.startswith('cylinder'):
            sim.add(cylinder, material='pec' if case.endswith('pec') else 'd')
        elif case in ('pad_pec', 'pad_dielectric'):
            sim.add(pad, material='pec' if case == 'pad_pec' else 'd')
        elif case in ('pad_dc', 'pad_f0'):
            thin(Box(p(4, 2, 3), p(12, 6, 3)), case.split('_')[1])
        elif case == 'mixed':
            thin(box, 'dc')
            thin(Box(p(7, 2, 1), p(7, 6, 5)), 'pec')
            thin(off, 'f0')
        elif case == 'overlap':
            for kind in order:
                thin(box, kind, sigma=1234, thickness=h/7)
        elif case == 'disc':
            thin(Cylinder(p(16, 16, 4), radius=radius*h, height=0), 'dc')
        else:
            raise AssertionError(case)
    grid = sim._build_nonuniform_grid() if nu else sim._build_grid()
    return sim, grid


def products(sim, grid, nu=False):
    root = realized_conductors(sim, grid, nonuniform=nu)
    result = {}

    def flatten(value, key):
        if dataclasses.is_dataclass(value):
            for f in dataclasses.fields(value):
                flatten(getattr(value, f.name), key+'.'+f.name)
        elif isinstance(value, tuple) and hasattr(value, '_fields'):
            for f in value._fields:
                flatten(getattr(value, f), key+'.'+f)
        elif isinstance(value, (tuple, list)):
            result[key+'.length'] = np.asarray(len(value))
            for i, item in enumerate(value):
                flatten(item, key+f'[{i}]')
        elif value is None:
            result[key] = np.asarray('None')
        else:
            result[key] = np.asarray(value)

    for name in ('materials', 'pec_cells', 'pec_edges', 'sheets', 'sheet_impedance', 'wires'):
        flatten(getattr(root, name), name)
    flatten(root.sheet_context(root.pec_edges), 'context')
    return result


def assert_products_equal(a, b, *, dc_ulp=0):
    assert a.keys() == b.keys()
    for name in a:
        if name == 'materials.sigma' and dc_ulp:
            np.testing.assert_array_max_ulp(a[name], b[name], maxulp=dc_ulp)
        else:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


@pytest.mark.parametrize('case', CASES)
def test_equal_cell_paths(case):
    a, ga = build(case)
    b, gb = build(case, True)
    for axis in range(3):
        np.testing.assert_array_equal(ga.cells(axis), gb.cells(axis))
    assert_products_equal(products(a, ga), products(b, gb, True), dc_ulp=1)


def test_overlap_order_and_paths():
    outputs = []
    for nu, order in itertools.product((False, True), (('dc', 'pec', 'f0'), ('f0', 'pec', 'dc'))):
        sim, grid = build('overlap', nu, h=.001, order=order)
        outputs.append(products(sim, grid, nu))
    assert_products_equal(outputs[0], outputs[1])
    assert_products_equal(outputs[2], outputs[3])
    assert_products_equal(outputs[0], outputs[2], dc_ulp=1)


def test_dc_nonbox_admission_stays_path_specific():
    """DC admission declaration replaces the old per-path tiny-disc asymmetry."""
    for nonuniform in (False, True):
        sim, grid = build('disc', nonuniform, radius=.4)
        with pytest.raises(ValueError, match='Cylinder.*A_d=.*A_r='):
            products(sim, grid, nonuniform)
        sim._snap = 'declared'
        with pytest.warns(UserWarning, match='Cylinder.*A_d=.*A_r='):
            sigma = products(sim, grid, nonuniform)['materials.sigma']
        assert np.count_nonzero(sigma) > 0
        np.testing.assert_array_equal(sigma[sigma != 0], 1000 * .1)


@pytest.mark.parametrize('kind', ('dc', 'f0'))
def test_unequal_cells_use_normal_dual(kind):
    from rfx.materials.thin_conductor import leontovich_rs
    h = 1/1024
    profile = np.array([h]*4+[2*h]*4)
    sim = Simulation(freq_max=15e9, domain=(12*h, 10*h, 12*h), dx=h,
                     dz_profile=profile, boundary='cpml', cpml_layers=2)
    kwargs = dict(sigma_bulk=1234, thickness=h/7)
    if kind == 'f0':
        kwargs['surface_impedance_f0'] = 5e9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sim.add_thin_conductor(Box((2*h, 2*h, 4*h), (8*h, 7*h, 4*h)), **kwargs)
    grid = sim._build_nonuniform_grid()
    root = realized_conductors(sim, grid, nonuniform=True)
    arr = np.asarray(root.materials.sigma if kind == 'dc' else root.sheet_impedance[0].sigma_sheet)
    occupied = np.argwhere(arr != 0)
    assert len(occupied) > 0
    assert np.unique(occupied[:, 2]).tolist() == [6]
    conductance = 1234*(h/7) if kind == 'dc' else 1/float(leontovich_rs(5e9, 1234))
    divisor = 2*h if kind == 'dc' else 1.5*h
    np.testing.assert_allclose(arr[arr != 0], conductance/divisor, rtol=2e-7)


def test_uniform_x64_keeps_scalar_sheet_precision():
    from rfx.materials.thin_conductor import leontovich_rs
    with jax.enable_x64():
        sim, grid = build('f0_on', h=.001)
        root = realized_conductors(sim, grid)
        spec, = root.sheet_impedance
        actual = np.asarray(spec.sigma_sheet)[np.asarray(spec.mask)]
        expected = 1.0 / float(leontovich_rs(5e9, 1000)) / grid.dx
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def dc_overlap_fixture(lane, order=('a', 'b')):
    """Two distinct DC sheets: 12 shared cells, plus exclusive cells each."""
    h = 1/1024
    kw = {}
    if lane != 'uniform':
        kw['dz_profile'] = (np.full(8, h) if lane == 'equal' else
                            np.array([1, 1, .5, 1.5, .6, 1.4, 1, 1])*h)
    sim = Simulation(freq_max=15e9, domain=(12*h, 10*h, 8*h), dx=h,
                     boundary='cpml', cpml_layers=2, **kw)
    declarations = {
        'a': (Box((2*h, 2*h, 4*h), (8*h, 7*h, 4.1*h)), 1234, 2.5),
        'b': (Box((5*h, 3*h, 4*h), (10*h, 9*h, 4.1*h)), 77, 4.5),
    }
    for name in order:
        shape, sigma, eps = declarations[name]
        sim.add_thin_conductor(shape, sigma_bulk=sigma, thickness=h/7, eps_r=eps)
    grid = sim._build_grid() if lane == 'uniform' else sim._build_nonuniform_grid()
    return sim, grid


@pytest.mark.parametrize('lane', ('uniform', 'equal', 'graded'))
def test_dc_overlap_last_declaration_wins(lane):
    from rfx.model.materials import assemble_cells
    results = []
    for order in (('a', 'b'), ('b', 'a')):
        sim, grid = dc_overlap_fixture(lane, order)
        mats = assemble_cells(sim, grid)[0]
        results.append(mats)
        # User-coordinate cell names: A-only (3,3,4), B-only (9,8,4),
        # overlap (6,4,4), and vacuum (1,1,4). The graded node at z=4h
        # occupies the .6h cell above the 1.5h cell.
        primal = grid.dx * (.6 if lane == 'graded' else 1)
        last = (77, 4.5) if order[-1] == 'b' else (1234, 2.5)
        for xyz, (sigma, eps) in [((3, 3, 4), (1234, 2.5)),
                                  ((9, 8, 4), (77, 4.5)),
                                  ((6, 4, 4), last), ((1, 1, 4), (0, 1))]:
            index = tuple(i+p for i, p in zip(xyz, grid.axis_pads))
            np.testing.assert_allclose(mats.sigma[index], sigma*(grid.dx/7)/primal, rtol=2e-7)
            assert mats.eps_r[index] == eps
    assert np.count_nonzero(results[0].sigma != results[1].sigma) == 12
    assert np.count_nonzero(results[0].eps_r != results[1].eps_r) == 12


@pytest.mark.parametrize('nu', (False, True))
def test_dc_geometry_mask_records_declared_conductor(nu):
    from rfx.model.materials import assemble_cells
    sim, grid = dc_overlap_fixture('equal' if nu else 'uniform', order=('a',))
    masks = []
    mats = assemble_cells(sim, grid, geometry_masks=masks)[0]
    assert len(masks) == 1
    key, mask = masks[0]
    assert key == id(sim._thin_conductors[0])
    expected = np.zeros(grid.shape, dtype=bool)
    px, py, pz = grid.axis_pads
    expected[px+2:px+8, py+2:py+7, pz+4] = True
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(np.asarray(mats.sigma) != 0, expected)


def test_periodic_dc_cells_cross_seam():
    from rfx.boundaries.spec import BoundarySpec
    from rfx.model.materials import assemble_cells
    h = 1/1024
    sim = Simulation(freq_max=15e9, domain=(12*h, 10*h, 8*h), dx=h,
                     boundary=BoundarySpec(x='periodic', y='periodic', z='cpml'),
                     cpml_layers=2)
    # Finite subcell thickness exercises DC plane identification itself;
    # a zero-thickness Box is also wrapped by the generic shape sampler.
    sim.add_thin_conductor(Box((11.5*h, 8*h, 2*h), (11.7*h, 13*h, 6*h)),
                           sigma_bulk=1234, thickness=h/7, eps_r=3.5)
    grid = sim._build_grid()
    mats = assemble_cells(sim, grid)[0]
    expected = np.zeros(grid.shape, dtype=bool)
    expected[0, [0, 1, 2, 8, 9], grid.pad_z_lo+2:grid.pad_z_lo+6] = True
    np.testing.assert_array_equal(np.asarray(mats.sigma) != 0, expected)
    np.testing.assert_allclose(np.asarray(mats.sigma)[expected], 1234/7, rtol=1e-7)
    np.testing.assert_array_equal(np.asarray(mats.eps_r)[expected], 3.5)


def test_f0_fold_refuses_multilayer_occupancy(monkeypatch):
    # Isolate the fold's defensive occupancy gate from the rasterizer's
    # own geometry checks: a malformed producer must not multiply loss.
    import jax.numpy as jnp
    import rfx.model.thin_conductors as fold
    from types import SimpleNamespace
    from rfx.core.yee import init_materials
    from rfx.materials.thin_conductor import apply_thin_conductor
    sim, grid = build('f0_on')
    original = fold.sheet_spec_from_shape

    def multilayer(*args, **kwargs):
        spec = original(*args, **kwargs)
        mask = spec.footprint | jnp.roll(spec.footprint, 1, axis=spec.normal_axis)
        return SimpleNamespace(**{**vars(spec), 'footprint': mask})

    monkeypatch.setattr(fold, 'sheet_spec_from_shape', multilayer)
    with pytest.raises(ValueError, match='rasterizes to 2 cell layers'):
        apply_thin_conductor(grid, sim._thin_conductors[0], init_materials(grid.shape),
                             sheet_specs=[])


def test_uniform_thin_pec_shape_collector():
    from rfx.model.materials import assemble_cells
    sim, grid = build('thin_pec')
    sheets = []
    out = assemble_cells(sim, grid, pec_sheets=sheets)
    assert len(out[4]) == 1
    assert out[4][0] == sim._thin_conductors[0].shape
    assert len(sheets) == 1
    assert out[3] is None


def float64_material_sheet_fixture(nu=True):
    sim, grid = build('f0_on', nu)
    h = grid.dx
    sim.add_material('d64', eps_r=np.float64(2), sigma=np.float64(.125))
    sim.add(Box((2*h, 2*h, 2*h), (8*h, 7*h, 6*h)), material='d64')
    return sim, grid


def test_graded_x64_sheet_preserves_material_dtype():
    from rfx.materials.thin_conductor import leontovich_rs
    with jax.enable_x64():
        sim, grid = float64_material_sheet_fixture()
        root = realized_conductors(sim, grid, nonuniform=True)
        spec, = root.sheet_impedance
        assert root.materials.sigma.dtype == np.float64
        assert spec.sigma_sheet.dtype == np.float64
        expected = np.float32(1 / float(leontovich_rs(5e9, 1000)) / grid.dx)
        np.testing.assert_array_equal(np.asarray(spec.sigma_sheet)[np.asarray(spec.mask)], expected)

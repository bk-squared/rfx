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
    sim, grid = build('disc', radius=.4)
    sigma = products(sim, grid)['materials.sigma']
    assert np.count_nonzero(sigma) > 0
    np.testing.assert_array_equal(sigma[sigma != 0], 1000 * .1)
    sim, grid = build('disc', True, radius=.4)
    with pytest.raises(NotImplementedError, match="non-Box shape"):
        products(sim, grid, True)


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
    np.testing.assert_allclose(arr[arr != 0], conductance/(1.5*h), rtol=2e-7)


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

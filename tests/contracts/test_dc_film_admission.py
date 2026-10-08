"""DC film admission: independent footprint, layer and conductance references."""
from dataclasses import replace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation, Sphere
from rfx.geometry.csg import _grid_coords
from rfx.model.materials import assemble_cells, realize_components

H = 1e-3
T = 35e-6
SIGMA = 1 / (377 * T)


def model(lane='uniform', radius=10.5, offset=.27, normal=2,
          snap='strict', centre=(23.17, 24.31), height=0):
    counts = [48] * 3
    counts[normal] = 24
    kwargs = {}
    if lane == 'equal':
        kwargs[f'd{"xyz"[normal]}_profile'] = np.full(24, H)
    if lane == 'normal':
        kwargs[f'd{"xyz"[normal]}_profile'] = H*np.array([1]*10+[.9, 1.1]+[1]*12)
    if lane == 'inplane':
        for a in range(3):
            if a != normal:
                kwargs[f'd{"xyz"[a]}_profile'] = np.r_[[H, H], np.tile([.9*H, 1.1*H], 22), [H, H]]
    sim = Simulation(10e9, tuple(n*H for n in counts), dx=H,
                     boundary='pec', snap=snap, **kwargs)
    grid = sim._build_grid() if lane == 'uniform' else sim._build_nonuniform_grid()
    coords = _grid_coords(grid)
    mid = float(coords[normal][11 if lane == 'normal' else 12]) + offset*H
    xyz = [0.] * 3
    tangents = [a for a in range(3) if a != normal]
    for a, value in zip(tangents, centre):
        xyz[a] = value*H
    xyz[normal] = mid
    shape = Cylinder(tuple(xyz), radius*H, height*H, axis='xyz'[normal])
    sim.add_thin_conductor(shape, sigma_bulk=SIGMA, thickness=T)
    return sim, grid, shape


def reference(grid, shape):
    normal = 'xyz'.index(shape.axis)
    nodes = [np.asarray(c, dtype=float) for c in _grid_coords(grid)]
    tangents = [a for a in range(3) if a != normal]
    a, b = tangents
    footprint = ((nodes[a][:, None] - shape.center[a])**2
                 + (nodes[b][None, :] - shape.center[b])**2 <= shape.radius**2)
    area = float(np.sum(footprint * np.asarray(grid.cells(a))[:, None]
                        * np.asarray(grid.cells(b))[None, :]))
    k = int(np.argmin(np.abs(nodes[normal] - shape.center[normal])))
    mask = np.zeros(grid.shape, dtype=bool)
    index = [slice(None)] * 3
    index[normal] = k
    mask[tuple(index)] = footprint
    return mask, area, np.pi*shape.radius**2


def products(sim, grid):
    masks = []
    cells = assemble_cells(sim, grid, geometry_masks=masks)[0]
    return cells, np.asarray(masks[-1][1])


@pytest.mark.parametrize('normal', range(3))
@pytest.mark.parametrize('offset', [0, .27])
@pytest.mark.parametrize('radius', [10.5, 14.5])
def test_b2_equal_grids_share_box_layer_and_fold(normal, offset, radius):
    results = []
    for lane in ('uniform', 'equal'):
        sim, grid, shape = model(lane, radius, offset, normal)
        cells, mask = products(sim, grid)
        expected, _, _ = reference(grid, shape)
        np.testing.assert_array_equal(mask, expected)
        lo, hi = map(list, shape.bounding_box())
        lo[normal] = hi[normal] = shape.center[normal]
        sim._thin_conductors[0] = replace(sim._thin_conductors[0], shape=Box(tuple(lo), tuple(hi)))
        _, box = products(sim, grid)
        tangents = tuple(a for a in range(3) if a != normal)
        np.testing.assert_array_equal(mask.any(axis=tangents), box.any(axis=tangents))
        results.append((cells, mask))
    np.testing.assert_array_equal(results[0][1], results[1][1])
    np.testing.assert_array_max_ulp(results[0][0].sigma, results[1][0].sigma, maxulp=2)


@pytest.mark.parametrize('lane', ['uniform', 'equal', 'normal', 'inplane'])
@pytest.mark.parametrize('normal', range(3))
@pytest.mark.parametrize('radius', [10.5, 14.5])
def test_b3_integrated_conductance(lane, normal, radius):
    sim, grid, shape = model(lane, radius, normal=normal)
    cells, mask = products(sim, grid)
    expected, area, declared = reference(grid, shape)
    np.testing.assert_array_equal(mask, expected)
    assert abs(np.sqrt(area / declared) - 1) <= .01
    components = realize_components(cells, grid, periodic=(False, False, False))
    widths = np.asarray(grid.cells(normal), dtype=float)
    dual = np.r_[widths[0], (widths[:-1] + widths[1:])/2]
    index = [int(np.argmin(np.abs(np.asarray(c) - shape.center[a])))
             for a, c in enumerate(_grid_coords(grid))]
    index[normal] = slice(None)
    for a in range(3):
        if a != normal:
            ratio = np.sum(np.asarray(components.sigma_update[a])[tuple(index)] * dual)/(SIGMA*T)
            np.testing.assert_allclose(ratio, 1, rtol=4e-7, atol=0)


class UnknownArea:
    def __init__(self, inner):
        self.inner = inner

    def bounding_box(self):
        return self.inner.bounding_box()

    def mask_on_coords(self, *coords):
        return self.inner.mask_on_coords(*coords)

    def mask(self, grid):
        return self.inner.mask(grid)


@pytest.mark.parametrize('lane', ['uniform', 'equal', 'normal', 'inplane'])
@pytest.mark.parametrize('radius', [2.5, 5.5])
def test_b4_small_disc_refusal_and_declared_finding(lane, radius):
    # Canonical placement retains the predeclared 3.4% / 1.03% witnesses.
    # B2/B3/B5 also exercise fractional, asymmetric transverse placement.
    sim, grid, shape = model(lane, radius, centre=(24, 24))
    _, area, declared = reference(grid, shape)
    expected_refusal = abs(np.sqrt(area/declared)-1) > .01
    if expected_refusal:
        with pytest.raises(ValueError, match=r'Cylinder.*A_d=.*A_r='):
            products(sim, grid)
        findings = sim.preflight().by_code('dc_film_area')
        assert any(f.severity == 'error' for f in findings)
        diagnostic = sim.realized_geometry().entities[0]
        assert diagnostic.occupancy_role == 'diagnostic'
        assert diagnostic.realized_footprint_area_m2 == pytest.approx(area, rel=1e-7)
    else:
        products(sim, grid)
    sim._snap = 'declared'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        products(sim, grid)
    assert any(getattr(w.message, 'code', None) == 'dc_film_area' for w in caught) == expected_refusal
    findings = sim.preflight().by_code('dc_film_area')
    assert len(findings) == int(expected_refusal)
    assert all(f.severity == 'warning' for f in findings)
    entity = sim.realized_geometry().entities[0]
    assert entity.realized_footprint_area_m2 == pytest.approx(area, rel=1e-7)
    assert entity.declared_footprint_area_m2 == pytest.approx(declared, rel=1e-12)


@pytest.mark.parametrize('lane', ['uniform', 'equal', 'normal', 'inplane'])
@pytest.mark.parametrize('snap', ['strict', 'declared'])
@pytest.mark.parametrize('kind', ['thick', 'sphere', 'empty'])
def test_b4_structural_refusals(lane, snap, kind):
    sim, grid, shape = model(lane, snap=snap, height=2.5 if kind == 'thick' else 0)
    if kind == 'sphere':
        shape = Sphere(shape.center, 2.5*H)
    elif kind == 'empty':
        shape = replace(shape, center=(.2, .2, shape.center[2]))
    sim._thin_conductors[0] = replace(sim._thin_conductors[0], shape=shape)
    with pytest.raises(ValueError, match=r'(Cylinder|Sphere).*layers') as caught:
        products(sim, grid)
    assert caught.value.code == 'dc_film_layers'


@pytest.mark.parametrize('lane', ['uniform', 'equal', 'normal', 'inplane'])
def test_b4_unknown_area(lane):
    sim, grid, shape = model(lane)
    sim._thin_conductors[0] = replace(sim._thin_conductors[0], shape=UnknownArea(shape))
    with pytest.raises(ValueError, match=r'UnknownArea.*A_d=None.*A_r=.*cannot judge'):
        products(sim, grid)
    sim._snap = 'declared'
    with pytest.warns(UserWarning, match='cannot judge'):
        products(sim, grid)
    assert any(f.severity == 'warning' for f in sim.preflight().by_code('dc_film_area'))
    entity = sim.realized_geometry().entities[0]
    assert entity.declared_footprint_area_m2 is None
    assert entity.realized_footprint_area_m2 > 0


@pytest.mark.parametrize('lane', ['uniform', 'equal', 'normal', 'inplane'])
@pytest.mark.parametrize('centre', [(24, 24), (23.17, 24.31)])
def test_b5_radius_trend(lane, centre):
    for radius in np.arange(2.5, 21, 1):
        sim, grid, shape = model(lane, float(radius), centre=centre)
        expected, area, declared = reference(grid, shape)
        if abs(np.sqrt(area/declared)-1) > .01:
            with pytest.raises(ValueError, match='A_d=.*A_r='):
                products(sim, grid)
        else:
            _, mask = products(sim, grid)
            np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize('lane', ['uniform', 'normal'])
@pytest.mark.parametrize('parameter,value', [('thickness', T), ('sigma_bulk', SIGMA)])
def test_parameter_tracers_keep_geometry_checks(lane, parameter, value):
    sim, grid, _ = model(lane)
    conductor = sim._thin_conductors[0]
    baseline = float(jnp.sum(products(sim, grid)[0].sigma))

    def total(v):
        sim._thin_conductors[0] = replace(conductor, **{parameter: v})
        return jnp.sum(assemble_cells(sim, grid)[0].sigma)

    try:
        got = float(jax.grad(total)(value))
        np.testing.assert_allclose(got, baseline/value, rtol=4e-7, atol=0)
        if parameter == 'sigma_bulk':
            with pytest.raises(ValueError, match='sigma_bulk.*PEC/lossy class.*abstract.*without jit'):
                jax.jit(jax.grad(total))(value)
        else:
            got = float(jax.jit(jax.grad(total))(value))
            np.testing.assert_allclose(got, baseline/value, rtol=4e-7, atol=0)
    finally:
        sim._thin_conductors[0] = conductor


@pytest.mark.parametrize('lane', ['uniform', 'normal'])
def test_plain_grad_copper_retains_concrete_pec_products(lane):
    sim, grid, disc = model(lane)
    conductor = replace(sim._thin_conductors[0], shape=Box(*disc.bounding_box()))
    observed = []

    def total(sigma):
        sim._thin_conductors[0] = replace(conductor, sigma_bulk=sigma)
        sheets = []
        cells = assemble_cells(sim, grid, pec_sheets=sheets)[0]
        assert len(sheets) == 1
        observed.append((cells.sigma, sheets[0].footprint))
        return jnp.sum(cells.sigma)

    try:
        concrete = total(5.8e7)
        value, derivative = jax.value_and_grad(total)(5.8e7)
    finally:
        sim._thin_conductors[0] = conductor
    assert float(concrete) == float(value) == float(derivative) == 0
    for before, traced in zip(observed[0], observed[1]):
        np.testing.assert_array_equal(before, traced)


@pytest.mark.parametrize('snap', ['strict', 'declared'])
@pytest.mark.parametrize('kind', ['admitted', 'tall', 'small', 'unknown', 'outside'])
def test_coordinate_tracer_requires_declared_override(snap, kind):
    def total(profile):
        sim = Simulation(10e9, (.048, .048, .024), dx=H, boundary='pec',
                         dz_profile=profile, snap=snap)
        shape = Cylinder((.024, .024, .01227),
                         (2.5 if kind == 'small' else 10.5)*H,
                         2.5*H if kind == 'tall' else 0)
        if kind == 'unknown':
            shape = UnknownArea(shape)
        elif kind == 'outside':
            shape = replace(shape, center=(.2, .2, .01227))
        sim.add_thin_conductor(shape, sigma_bulk=SIGMA, thickness=T)
        return jnp.sum(assemble_cells(sim, sim._build_nonuniform_grid())[0].sigma)

    call = jax.jit(jax.grad(total))
    if snap == 'strict':
        with pytest.raises(ValueError, match='cannot judge layer/area on a traced mesh.*Box.*declared'):
            call(jnp.full(24, H))
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            derivative = call(jnp.full(24, H))
        findings = [w.message for w in caught if getattr(w.message, 'code', None) == 'dc_film_unchecked']
        assert len(findings) == 1
        assert 'without checking area or the one-layer rule' in str(findings[0])
        assert np.isfinite(np.asarray(derivative)).all()


@pytest.mark.parametrize('lane', ['uniform', 'normal'])
@pytest.mark.parametrize('kind', ['small', 'unknown', 'sphere', 'tall'])
def test_preflight_has_one_finding_per_film_even_after_cached_assembly(lane, kind):
    sim, _, shape = model(lane, radius=2.5, centre=(24, 24), snap='declared',
                          height=2.5 if kind == 'tall' else 0)
    if kind == 'unknown':
        shape = UnknownArea(shape)
    elif kind == 'sphere':
        shape = Sphere(shape.center, 2.5*H)
    sim._thin_conductors[0] = replace(sim._thin_conductors[0], shape=shape)
    sim.add_thin_conductor(shape, sigma_bulk=SIGMA, thickness=T)
    code = 'dc_film_layers' if kind in ('sphere', 'tall') else 'dc_film_area'
    for _ in range(2):
        findings = sim.preflight().by_code(code)
        assert len(findings) == 2
        assert {f.loc for f in findings} == {'thin_conductors[0]', 'thin_conductors[1]'}
        assert not sim.preflight().by_code('dc_film_area' if code == 'dc_film_layers' else 'dc_film_layers')


@pytest.mark.parametrize('snap', ['strict', 'declared'])
def test_traced_preflight_has_one_finding(snap):
    def check(profile):
        sim = Simulation(10e9, (.048, .048, .024), dx=H, boundary='pec',
                         dz_profile=profile, snap=snap)
        sim.add_thin_conductor(Cylinder((.024, .024, .01227), 2.5*H, 0),
                               sigma_bulk=SIGMA, thickness=T)
        findings = sim.preflight(check_ntff=False, check_resolution=False).by_code('dc_film_unchecked')
        assert len(findings) == 1
        assert findings[0].severity == ('error' if snap == 'strict' else 'warning')
        return jnp.sum(profile)

    assert np.isfinite(float(jax.jit(check)(jnp.full(24, H))))


@pytest.mark.parametrize('lane', ['uniform', 'normal'])
def test_unknown_area_remedy_names_custom_shape_method(lane):
    sim, grid, shape = model(lane)
    sim._thin_conductors[0] = replace(sim._thin_conductors[0], shape=UnknownArea(shape))
    with pytest.raises(ValueError, match='Implement footprint_area\\(normal\\).*snap="declared"') as caught:
        products(sim, grid)
    assert 'draw a Box' not in str(caught.value)


@pytest.mark.parametrize('lane', ['uniform', 'normal'])
def test_clipped_disc_message_names_grid_extent(lane):
    sim, grid, _ = model(lane, centre=(2.17, 24.31))
    with pytest.raises(ValueError, match='A_d=.*A_r=.*clipped by the grid extent') as caught:
        products(sim, grid)
    assert 'Refine' not in str(caught.value)
    sim._snap = 'declared'
    with pytest.warns(UserWarning, match='clipped by the grid extent'):
        products(sim, grid)


@pytest.mark.parametrize('lane', ['uniform', 'normal'])
@pytest.mark.parametrize('snap', ['strict', 'declared'])
def test_traced_radius_refuses_without_raw_tracer_error(lane, snap):
    sim, grid, shape = model(lane, snap=snap)
    conductor = sim._thin_conductors[0]

    def total(radius):
        sim._thin_conductors[0] = replace(conductor, shape=replace(shape, radius=radius))
        return jnp.sum(assemble_cells(sim, grid)[0].sigma)

    try:
        with pytest.raises(ValueError, match='Cylinder.*traced shape bounds.*radius.*concrete'):
            jax.jit(jax.grad(total))(shape.radius)
    finally:
        sim._thin_conductors[0] = conductor


@pytest.mark.parametrize('offset', [.27, 11.73])
def test_periodic_plane_matches_box(offset):
    from rfx.boundaries.spec import BoundarySpec
    sim, _, shape = model(offset=offset)
    sim._boundary_spec = BoundarySpec(x='pec', y='pec', z='periodic')
    sim._periodic_axes = 'z'
    grid = sim._build_grid()
    _, mask = products(sim, grid)
    lo, hi = map(list, shape.bounding_box())
    lo[2] = hi[2] = shape.center[2]
    sim._thin_conductors[0] = replace(sim._thin_conductors[0], shape=Box(tuple(lo), tuple(hi)))
    _, control = products(sim, grid)
    np.testing.assert_array_equal(mask.any(axis=(0, 1)), control.any(axis=(0, 1)))
    assert mask.any(axis=(0, 1)).sum() == 1


def test_vmap_uses_shared_fold_and_declared_policy():
    from rfx.vmap_sweep import _apply_batched_thin_conductors
    sim, grid, shape = model(radius=14.5, snap='declared')
    sim._thin_conductors[0] = replace(sim._thin_conductors[0], shape=UnknownArea(shape))
    with pytest.warns(UserWarning, match='cannot judge'):
        expected = products(sim, grid)[0]
    eps = jnp.ones((2,) + grid.shape)
    sigma = jnp.zeros_like(eps)
    with pytest.warns(UserWarning, match='cannot judge'):
        _, actual, _ = _apply_batched_thin_conductors(sim, grid, eps, sigma, eps)
    for row in actual:
        np.testing.assert_array_equal(row, expected.sigma)


def test_falsifier_a_independent_graded_mask_is_detected(monkeypatch):
    import rfx.model.thin_conductors as owner
    original = owner.admit_dc_film

    def independent(shape, grid, **kwargs):
        result = original(shape, grid, **kwargs)
        return replace(result, mask=shape.mask_on_coords(*_grid_coords(grid))) if hasattr(grid, 'dx_arr') else result

    monkeypatch.setattr(owner, 'admit_dc_film', independent)
    with pytest.raises(AssertionError):
        test_b2_equal_grids_share_box_layer_and_fold(2, .27, 10.5)


@pytest.mark.parametrize('mutation', ['self', 'rectangle'])
def test_falsifier_b_false_area_reference_is_detected(monkeypatch, mutation):
    if mutation == 'self':
        import rfx.model.thin_conductors as owner
        original = owner.dc_footprint_areas
        def corrupted(*args):
            area, _ = original(*args)
            return area, area
        monkeypatch.setattr(owner, 'dc_footprint_areas', corrupted)
    else:
        monkeypatch.setattr(Cylinder, 'footprint_area', lambda self, normal: 4*self.radius**2)
    with pytest.raises((AssertionError, pytest.fail.Exception, ValueError)):
        test_b5_radius_trend('uniform', (24, 24))


def test_falsifier_c_dropped_layer_guard_is_detected(monkeypatch):
    import rfx.model.thin_conductors as owner
    original = owner.admit_dc_film

    def no_layer_guard(shape, grid, **kwargs):
        result = original(replace(shape, height=0), grid, **kwargs)
        return replace(result, mask=shape.mask_on_coords(*_grid_coords(grid)))

    monkeypatch.setattr(owner, 'admit_dc_film', no_layer_guard)
    with pytest.raises(pytest.fail.Exception):
        test_b4_structural_refusals('uniform', 'declared', 'thick')

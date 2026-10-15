"""D1–D6: expected incident cells and weights are independent of product helpers."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.model import conductors as products


CELLS = {
    'ex': [(8, 7, 3), (8, 7, 4), (8, 8, 3), (8, 8, 4)],
    'ey': [(7, 8, 3), (7, 8, 4), (8, 8, 3), (8, 8, 4)],
    'ez': [(7, 7, 4), (7, 8, 4), (8, 7, 4), (8, 8, 4)],
}
SHEET_EDGES = {
    'ex': [(6, 6, 4), (6, 7, 4), (6, 8, 4), (7, 6, 4), (7, 7, 4), (7, 8, 4)],
    'ey': [(6, 6, 4), (6, 7, 4), (7, 6, 4), (7, 7, 4), (8, 6, 4), (8, 7, 4)],
    'ez': [],
}
SHEET_CELLS = [(5, 6, 3), (5, 6, 4), (5, 7, 3), (5, 7, 4),
               (6, 5, 3), (6, 5, 4), (6, 6, 3), (6, 6, 4),
               (6, 7, 3), (6, 7, 4), (6, 8, 3), (6, 8, 4),
               (7, 5, 3), (7, 5, 4), (7, 6, 3), (7, 6, 4),
               (7, 7, 3), (7, 7, 4), (7, 8, 3), (7, 8, 4),
               (8, 6, 3), (8, 6, 4), (8, 7, 3), (8, 7, 4)]
GRADED_SHEET_EDGES = {
    'ex': [(6, 7, 4), (6, 8, 4), (7, 7, 4), (7, 8, 4)],
    'ey': [(6, 7, 4), (7, 7, 4), (8, 7, 4)], 'ez': [],
}
GRADED_SHEET_CELLS = [(5, 7, 3), (5, 7, 4),
                      (6, 6, 3), (6, 6, 4), (6, 7, 3), (6, 7, 4), (6, 8, 3), (6, 8, 4),
                      (7, 6, 3), (7, 6, 4), (7, 7, 3), (7, 7, 4), (7, 8, 3), (7, 8, 4),
                      (8, 7, 3), (8, 7, 4)]


def model(lane='uniform', kind='r', component='ez'):
    kw = {'dy_profile': np.array([.001] + [.0009, .0011]*7 + [.001])} if lane == 'graded' else {}
    sim = Simulation(freq_max=10e9, domain=(.016, .016, .012), dx=.001, boundary='pec', **kw)
    if kind in ('r', 'rc', 'parallel'):
        sim.add_lumped_rlc((.008, .008, .004), component, R=50.,
            C=1e-13 if kind != 'r' else 0., topology='parallel' if kind == 'parallel' else 'series')
    elif kind in ('sheet', 'pec', 'film'):
        args = {'sigma_bulk': 5.8e7} if kind == 'pec' else {'sigma_bulk': 7.895683520871486, 'thickness': .0025330295910584444}
        if kind == 'sheet':
            args['surface_impedance_f0'] = 5e9
        sim.add_thin_conductor(Box((.006, .006, .004), (.008, .008, .004)), **args)
    sim.add_source((.009, .008, .004), component, amplitude_kind='field', waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    sim.add_probe((.009, .008, .004), component)
    grid = sim._build_nonuniform_grid() if lane == 'graded' else sim._build_grid()
    return sim, grid


class Captured(Exception):
    pass


def capture(monkeypatch, sim, lane, **options):
    seen = {}
    original = products.at_kernel
    def at_kernel(*a, **kw):
        result = original(*a, **kw)
        seen['root'] = result[0]
        return result
    def kernel(*a, **kw):
        seen.update(kw)
        raise Captured
    monkeypatch.setattr(products, 'at_kernel', at_kernel)
    if lane == 'graded':
        import rfx.runners.nonuniform as runner
        monkeypatch.setattr(runner, 'run_nonuniform', kernel)
    else:
        import rfx.simulation as runner
        monkeypatch.setattr(runner, 'run', kernel)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        with pytest.raises(Captured):
            sim.forward(n_steps=2, checkpoint=False, skip_preflight=True, **options)
    seen['warnings'] = [str(w.message) for w in emitted if 'Port occupancy release:' in str(w.message)]
    return seen


def keep(array, component, edge):
    i, j, k = edge
    o = np.asarray(array, dtype=np.float64)
    if component == 'ex':
        return (1-o[i,j,k])*(1-o[i,j-1,k])*(1-o[i,j,k-1])*(1-o[i,j-1,k-1])
    if component == 'ey':
        return (1-o[i,j,k])*(1-o[i-1,j,k])*(1-o[i,j,k-1])*(1-o[i-1,j,k-1])
    return (1-o[i,j,k])*(1-o[i-1,j,k])*(1-o[i,j-1,k])*(1-o[i-1,j-1,k])


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('kind', ['r', 'rc', 'parallel'])
@pytest.mark.parametrize('component', ['ex', 'ey', 'ez'])
@pytest.mark.parametrize('placement', ['interior', 'face', 'diagonal', 'away', 'zero'])
@pytest.mark.parametrize('p', [.5, 1.])
def test_k1_k2_k3_k4_lumped(monkeypatch, lane, kind, component, placement, p):
    sim, grid = model(lane, kind, component)
    occ = np.zeros(grid.shape, np.float32)
    lo, hi = [6, 6, 2], [11, 11, 7]
    transverse = {'ex': (1, 2), 'ey': (0, 2), 'ez': (0, 1)}[component]
    edge = (8, 8, 4)
    if placement == 'face':
        lo[transverse[0]] = edge[transverse[0]]
    elif placement == 'diagonal':
        for a in transverse:
            hi[a] = edge[a]
    elif placement == 'away':
        lo[transverse[0]] = edge[transverse[0]] + 1
    if placement != 'zero':
        occ[tuple(slice(a,b) for a,b in zip(lo,hi))] = p
    seen = capture(monkeypatch, sim, lane, pec_occupancy_override=jnp.asarray(occ))
    actual = np.asarray(seen['pec_occupancy'])
    assert keep(actual, component, edge) == 1., 'K1'
    np.testing.assert_array_equal(seen['root'].released_cells, CELLS[component], err_msg='K2')
    expected = occ.copy()
    for cell in CELLS[component]:
        expected[cell] = 0
    np.testing.assert_array_equal(actual, expected, err_msg='K3')
    assert len(seen['warnings']) == (placement not in ('away', 'zero')), 'K4'
    if seen['warnings']:
        assert all(s in seen['warnings'][0] for s in ('lumped_rlc[0]', component, '(8, 8, 4)'))
    meta = seen['lumped_rlc'][0] if lane == 'uniform' else seen['rlc_metas'][0]
    assert (meta.i, meta.j, meta.k, meta.component) == (*edge, component), 'R+C realized-edge falsifier'


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('p', [.5, 1.])
def test_k1_k2_k3_sheet(monkeypatch, lane, p):
    sim, grid = model(lane, 'sheet')
    occ = np.full(grid.shape, p, np.float32)
    seen = capture(monkeypatch, sim, lane, pec_occupancy_override=jnp.asarray(occ))
    actual = np.asarray(seen['pec_occupancy'])
    ctx = seen['sheet_impedance']
    for comp in ('ex', 'ey', 'ez'):
        edges = np.argwhere(np.asarray(getattr(ctx, 'mask_'+comp)))
        expected_edges = SHEET_EDGES if lane == 'uniform' else GRADED_SHEET_EDGES
        assert edges.tolist() == [list(e) for e in expected_edges[comp]], 'K2 sheet edge list'
        for edge in edges:
            assert keep(actual, comp, edge) == 1., ('K1 sheet', comp, edge)
    expected_cells = SHEET_CELLS if lane == 'uniform' else GRADED_SHEET_CELLS
    np.testing.assert_array_equal(seen['root'].released_cells, expected_cells, err_msg='K2 sheet')
    expected = occ.copy()
    for cell in expected_cells:
        expected[tuple(cell)] = 0
    np.testing.assert_array_equal(actual, expected, err_msg='K3 sheet')
    assert len(seen['warnings']) == 1 and 'thin_conductor[0]' in seen['warnings'][0]


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('kind', ['r', 'rc', 'sheet'])
def test_k5_gradient(lane, kind):
    sim, grid = model(lane, kind)
    occ = jnp.full(grid.shape, .25)
    def loss(o):
        return jnp.sum(sim.forward(n_steps=24, pec_occupancy_override=o,
            checkpoint=False, skip_preflight=True).time_series**2)
    grad = np.asarray(jax.grad(loss)(occ))
    if kind == 'sheet':
        cells = SHEET_CELLS if lane == 'uniform' else GRADED_SHEET_CELLS
    else:
        cells = CELLS['ez']
    assert all(grad[tuple(c)] == 0. for c in cells), 'K5 released gradient'
    assert np.any(grad != 0.), 'K5 live gradient'


@pytest.mark.parametrize('kind,match', [('r', 'lumped_rlc\\[0\\]'), ('rc', 'lumped_rlc\\[0\\]'), ('sheet', 'thin_conductor\\[0\\]'), ('film', 'thin conductor 0')])
@pytest.mark.parametrize('p', [0., .5])
def test_k6_refused_before_step(monkeypatch, kind, match, p):
    sim, _ = model(kind=kind)
    import rfx.simulation as runner
    monkeypatch.setattr(runner, 'run', lambda *a, **kw: pytest.fail('first step reached'))
    with pytest.raises(ValueError, match=match):
        sim.forward(n_steps=2, design_box=((.006,.006,.003),(.009,.009,.005)),
            design_occupancy_override=jnp.full((4,4,3), p), skip_preflight=True)


@pytest.mark.parametrize('kind', ['r', 'rc', 'sheet', 'pec'])
def test_k6_clear_box_runs(kind):
    sim, _ = model(kind=kind)
    result = sim.forward(n_steps=2, design_box=((.010,.010,.006),(.011,.011,.007)),
        design_occupancy_override=jnp.zeros((2,2,2)), skip_preflight=True, checkpoint=False)
    assert np.isfinite(np.asarray(result.time_series)).all()


@pytest.mark.parametrize('inside', [False, True])
@pytest.mark.parametrize('p', [0., .5])
def test_k7_source_box(inside, p):
    sim, _ = model(kind='none')
    sim.add_source((.008 if inside else .013, .008, .004), 'ez', amplitude_kind='current')
    result = sim.forward(n_steps=8, design_box=((.008,.007,.003),(.010,.009,.005)),
        design_occupancy_override=jnp.full((3,3,3), p), skip_preflight=True, checkpoint=False)
    records = [d for d in result.diagnostics if d.code == 'source_in_design_region']
    expected = ['source[0]', 'source[1]'] if inside else ['source[0]']
    assert [d.subject for d in records] == expected
    assert all(d.severity == 'advisory' and d.values['component'] == 'ez' for d in records)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('p', [None, 0., .5])
def test_k7b_whole_domain(lane, p):
    sim, grid = model(lane, 'none')
    sim.add_source((.012,.008,.004), 'ez', amplitude_kind='current')
    observed = []
    def loss(o):
        result = sim.forward(n_steps=8, skip_preflight=True, checkpoint=False,
            **({} if p is None else {'pec_occupancy_override': o}))
        observed.append(tuple(d for d in result.diagnostics if d.code.startswith('source_') and 'design' in d.code))
        return jnp.sum(result.time_series**2)
    occ = jnp.full(grid.shape, p or 0.)
    loss(occ)
    jax.grad(loss)(occ)
    assert observed[0] == observed[1]
    assert len(observed[0]) == (p is not None)
    if p is not None:
        d = observed[0][0]
        assert d.code == 'source_under_design_occupancy'
        assert 'source[0]' in d.subject and 'source[1]' in d.subject


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_k4_port_and_element_one_warning(monkeypatch, lane):
    sim, grid = model(lane, 'rc')
    sim.add_port((.010,.008,.004), 'ez', impedance=50.)
    seen = capture(monkeypatch, sim, lane, pec_occupancy_override=jnp.full(grid.shape, .5))
    assert len(seen['warnings']) == 1
    assert 'port[1]' in seen['warnings'][0] and 'lumped_rlc[0]' in seen['warnings'][0]


def test_k6_pec_sheet_inside_box_runs():
    sim, _ = model(kind='pec')
    result = sim.forward(n_steps=2, design_box=((.006,.006,.003),(.009,.009,.005)),
        design_occupancy_override=jnp.zeros((4,4,3)), skip_preflight=True, checkpoint=False)
    assert np.isfinite(np.asarray(result.time_series)).all()


@pytest.mark.parametrize('kind', ['r', 'sheet', 'none'])
def test_path_graded_box_refuses(kind):
    sim, _ = model('graded', kind)
    with pytest.raises(NotImplementedError, match='design_box'):
        sim.forward(n_steps=2, design_box=((.006,.006,.003),(.009,.009,.005)),
            design_occupancy_override=jnp.zeros((4,4,3)), skip_preflight=True)


@pytest.mark.parametrize('kind', ['r', 'sheet', 'none'])
@pytest.mark.parametrize('path', ['distributed', 'adi', 'subgrid', 'run'])
def test_path_existing_refusals(kind, path):
    sim, grid = model(kind=kind)
    options = dict(n_steps=2, pec_occupancy_override=jnp.zeros(grid.shape), skip_preflight=True)
    if path == 'run':
        with pytest.raises(TypeError, match='pec_occupancy_override'):
            sim.run(**options)
    elif path == 'distributed':
        with pytest.raises(NotImplementedError, match='pec_occupancy_override'):
            sim.forward(distributed=True, **options)
    elif path == 'adi':
        sim._solver = 'adi'
        with pytest.raises((ValueError, NotImplementedError), match='adi|ADI'):
            sim.forward(**options)
    else:
        sim.add_refinement(z_range=(.003,.007), ratio=2)
        with pytest.raises(NotImplementedError, match='refinement|subgrid'):
            sim.forward(**options)


@pytest.mark.parametrize('kind', ['r', 'sheet'])
def test_path_two_device_graded_refuses(kind):
    sim, grid = model('graded', kind)
    with pytest.raises((ValueError, NotImplementedError), match='[Rr][Ll][Cc]|surface.impedance'):
        sim.forward(n_steps=2, pec_occupancy_override=jnp.zeros(grid.shape),
            distributed=True, devices=jax.devices()[:2], skip_preflight=True)


def test_path_two_device_graded_source_not_reached():
    sim, grid = model('graded', 'none')
    result = sim.forward(n_steps=2, pec_occupancy_override=jnp.zeros(grid.shape),
        distributed=True, devices=jax.devices()[:2], skip_preflight=True)
    assert not any(d.code == 'source_under_design_occupancy' for d in result.diagnostics)


@pytest.mark.parametrize('kind', ['rc', 'sheet', 'none'])
def test_path_topology(monkeypatch, kind):
    from rfx.topology import TopologyDesignRegion, topology_optimize
    sim, _ = model(kind=kind)
    region = TopologyDesignRegion((.006,.006,.002), (.010,.010,.006))
    if kind == 'sheet':
        with pytest.raises(ValueError, match='surface.impedance'):
            topology_optimize(sim, region, lambda r: jnp.sum(r.time_series**2),
                n_iterations=1, skip_preflight=True, verbose=False)
        return
    # Stop at the first real forward result: optimizer setup is immaterial to
    # this admission contract and the optimization extra need not be installed.
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'optax', SimpleNamespace(
        adam=lambda rate: SimpleNamespace(init=lambda values: ())))
    original_forward = sim._forward_from_materials
    def forward(*a, **kw):
        return original_forward(*a, **{**kw, 'n_steps': 2})
    monkeypatch.setattr(sim, '_forward_from_materials', forward)
    seen = []
    original = products.prepare_port_occupancy
    def prepare(*a, **kw):
        result = original(*a, **kw)
        seen.append(result)
        return result
    monkeypatch.setattr(products, 'prepare_port_occupancy', prepare)
    monkeypatch.setattr(jax, 'value_and_grad', lambda fn: lambda x: (fn(x), jnp.zeros_like(x)))
    def objective(result):
        assert sum(d.code == 'source_under_design_occupancy' for d in result.diagnostics) == 1
        raise Captured
    with pytest.raises(Captured):
        topology_optimize(sim, region, objective, n_iterations=1,
            init_density=jnp.full((5,5,5), .5), skip_preflight=True, verbose=False)
    assert seen
    if kind == 'rc':
        for root, occ in seen:
            np.testing.assert_array_equal(root.released_cells, CELLS['ez'])
            assert keep(occ, 'ez', (8,8,4)) == 1.


def test_k4_k7b_whole_occupancy_with_a_disjoint_box(monkeypatch):
    sim, grid = model(kind='sheet')
    seen = capture(monkeypatch, sim, 'uniform', pec_occupancy_override=jnp.full(grid.shape, .5),
        design_box=((.010,.010,.006),(.011,.011,.007)), design_occupancy_override=jnp.zeros((2,2,2)))
    np.testing.assert_array_equal(seen['root'].released_cells, SHEET_CELLS)
    assert [d.code for d in seen['root'].design_findings] == ['source_under_design_occupancy']
    assert len(seen['warnings']) == 1


@pytest.mark.parametrize('lane,form', [('uniform', 'box'), ('uniform', 'whole'), ('graded', 'whole')])
@pytest.mark.parametrize('p', [0., .5])
def test_k7_k7b_findings_change_no_time_series(monkeypatch, lane, form, p):
    """The recorded main comparison also covers these numerical paths."""
    import rfx.model.design_admission as admission
    sim, grid = model(lane, 'none')
    sim.add_source((.008,.008,.004), 'ez', amplitude_kind='current',
                   waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    options = ({'pec_occupancy_override': jnp.full(grid.shape, p)} if form == 'whole' else
               {'design_box': ((.008,.007,.003),(.010,.009,.005)),
                'design_occupancy_override': jnp.full((3,3,3), p)})
    def run():
        return sim.forward(n_steps=80, checkpoint=False, skip_preflight=True, **options)
    told = run()
    monkeypatch.setattr(admission, 'source_findings', lambda *a, **kw: ())
    quiet = run()
    assert not quiet.diagnostics
    assert len(told.diagnostics) == (2 if form == 'box' else 1)
    a, b = np.asarray(told.time_series), np.asarray(quiet.time_series)
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def test_d6_source_clear_of_box_has_no_finding(monkeypatch):
    import rfx.model.design_admission as admission
    sim, _ = model(kind='none')
    options = dict(n_steps=8, checkpoint=False, skip_preflight=True,
                   design_box=((.003,.003,.002),(.004,.004,.003)),
                   design_occupancy_override=jnp.full((2,2,2), .5))
    actual = sim.forward(**options)
    assert not actual.diagnostics
    monkeypatch.setattr(admission, 'source_findings', lambda *a, **kw: ())
    baseline = sim.forward(**options)
    assert np.asarray(actual.time_series).tobytes() == np.asarray(baseline.time_series).tobytes()


# The box writes occupancy (and permittivity) into its own cells only; an edge
# is reached iff one of its four incident cells is a box cell. A box whose last
# cell sits one plane before a declaration's incident cells cannot reach it.
@pytest.mark.parametrize('kind', ['r', 'rc'])
def test_box_one_plane_short_of_element_runs(kind):
    sim, _ = model(kind=kind)      # Ez edge (8,8,4): incident cells x, y in {7, 8}
    result = sim.forward(n_steps=2, design_box=((.005,.007,.004),(.006,.008,.004)),
        design_occupancy_override=jnp.full((2,2,1), .5), skip_preflight=True, checkpoint=False)
    assert np.isfinite(np.asarray(result.time_series)).all()


def test_box_one_plane_short_of_element_reaches_when_extended():
    sim, _ = model(kind='r')
    with pytest.raises(ValueError, match=r'lumped_rlc\[0\]'):
        sim.forward(n_steps=2, design_box=((.005,.007,.004),(.007,.008,.004)),
            design_occupancy_override=jnp.full((3,2,1), .5), skip_preflight=True)


@pytest.mark.parametrize('form', ['occupancy', 'eps'])
def test_box_one_plane_below_f0_sheet_runs(form):
    sim, _ = model(kind='sheet')   # sheet edges at k=4: incident cells k in {3, 4}
    options = ({'design_occupancy_override': jnp.full((2,2,2), .5)} if form == 'occupancy'
               else {'design_eps_override': jnp.full((2,2,2), 2.)})
    result = sim.forward(n_steps=2, design_box=((.006,.006,.001),(.007,.007,.002)),
        skip_preflight=True, checkpoint=False, **options)
    assert np.isfinite(np.asarray(result.time_series)).all()


def test_box_reaching_f0_sheet_cell_is_refused():
    sim, _ = model(kind='sheet')
    with pytest.raises(ValueError, match=r'thin_conductor\[0\]'):
        sim.forward(n_steps=2, design_box=((.006,.006,.001),(.007,.007,.003)),
            design_occupancy_override=jnp.full((2,2,3), .5), skip_preflight=True)


def test_source_one_plane_past_box_has_no_finding():
    sim, _ = model(kind='none')    # source Ez edge (9,8,4): incident cells x in {8, 9}
    result = sim.forward(n_steps=2, design_box=((.005,.007,.003),(.007,.009,.005)),
        design_occupancy_override=jnp.full((3,3,3), .5), skip_preflight=True, checkpoint=False)
    assert not [d for d in result.diagnostics if d.code == 'source_in_design_region']
    result = sim.forward(n_steps=2, design_box=((.005,.007,.003),(.008,.009,.005)),
        design_occupancy_override=jnp.full((4,3,3), .5), skip_preflight=True, checkpoint=False)
    assert [d.subject for d in result.diagnostics if d.code == 'source_in_design_region'] == ['source[0]']

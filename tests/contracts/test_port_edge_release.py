"""Port design-metal release: independent incident-cell and kernel-boundary checks."""
from dataclasses import replace
from types import SimpleNamespace
import warnings
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, GaussianPulse
from rfx.model import conductors as products
from rfx.nonuniform import position_to_index


class Captured(Exception):
    pass


def model(lane, kind):
    graded = lane == 'graded'
    if kind == 'msl':
        from tests.unit.ports.test_msl_realized_port_contract import _model
        sim, _ = _model(nonuniform=graded)
        edge, component = (8, 12, 8), 'ez'
    else:
        kwargs = {'dy_profile': np.array([.001]+[.0009, .0011]*7+[.001])} if graded else {}
        sim = Simulation(freq_max=10e9, domain=(.016, .016, .012), dx=.001, boundary='pec', **kwargs)
        component = kind if kind in ('ex', 'ey', 'ez') else 'ez'
        sim.add_port((.008, .008, .004), component, impedance=50.,
                     waveform=GaussianPulse(f0=5e9, bandwidth=.8),
                     **({'extent': .004} if kind == 'wire' else {}))
        edge = (8, 8, 4)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    return sim, grid, edge, component


def box(edge, component, kind, placement):
    axis = 'xyz'.index(component[1])
    transverse = [a for a in range(3) if a != axis]
    lo, hi = np.array(edge)-2, np.array(edge)+3
    if kind == 'wire':
        hi[axis] = edge[axis]+6
    if placement == 'face':
        lo[transverse[0]] = edge[transverse[0]]
    elif placement == 'touch':
        for a in transverse:
            hi[a] = edge[a]
    elif placement == 'away':
        lo[transverse[0]] = edge[transverse[0]]+1
    return lo, hi


def capture(monkeypatch, sim, grid, lane, occupancy=None, topology_box=None, p=.5):
    seen = {}
    original = products.at_kernel
    def at_kernel(*args, **kwargs):
        result = original(*args, **kwargs)
        seen['conductors'] = result[0]
        return result
    def kernel(*args, **kwargs):
        seen['occupancy'] = kwargs['pec_occupancy']
        seen['sources'] = kwargs['sources']
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
            if lane == 'topology':
                from rfx.topology import topology_optimize, TopologyDesignRegion
                lo, hi = topology_box
                # Inspect the real topology forward before its first solve, without
                # introducing tracers into a host-array capture. C5 tests real AD.
                monkeypatch.setattr(jax, 'value_and_grad', lambda fn: lambda x: (fn(x), jnp.zeros_like(x)))
                region = TopologyDesignRegion(tuple(lo*.001), tuple((hi-1)*.001))
                topology_optimize(sim, region, lambda r: jnp.sum(r.time_series**2),
                    n_iterations=1, init_density=jnp.full(tuple(hi-lo), p),
                    verbose=False, skip_preflight=True)
            else:
                sim.forward(pec_occupancy_override=occupancy, n_steps=2,
                            checkpoint=False, skip_preflight=True)
    seen['warnings'] = [w for w in emitted if 'Port occupancy release:' in str(w.message)]
    return seen


def independent_cells(component, edge, shape, periodic):
    """Test-owned incidence spelling; never calls a product helper."""
    offsets = {'ex': [(0,0,0),(0,-1,0),(0,0,-1),(0,-1,-1)],
               'ey': [(0,0,0),(-1,0,0),(0,0,-1),(-1,0,-1)],
               'ez': [(0,0,0),(-1,0,0),(0,-1,0),(-1,-1,0)]}
    cells = set()
    for offset in offsets[component]:
        q = [edge[a]+offset[a] for a in range(3)]
        for a in range(3):
            if periodic[a] or shape[a] == 1:
                q[a] %= shape[a]
        if all(0 <= q[a] < shape[a] for a in range(3)):
            cells.add(tuple(q))
    return cells


@pytest.mark.parametrize('lane', ['uniform', 'graded', 'topology'])
@pytest.mark.parametrize('kind', ['ex', 'ey', 'ez', 'wire', 'msl'])
@pytest.mark.parametrize('placement', ['interior', 'face', 'touch', 'away'])
@pytest.mark.parametrize('p', [.5, 1.])
def test_c1_c2_c4_kernel_occupancy(monkeypatch, lane, kind, placement, p):
    if lane == 'topology':
        pytest.importorskip('optax')
    sim, grid, edge, component = model(lane, kind)
    lo, hi = box(edge, component, kind, placement)
    occupancy = jnp.zeros(grid.shape).at[tuple(slice(a,b) for a,b in zip(lo,hi))].set(p)
    seen = capture(monkeypatch, sim, grid, lane, occupancy, (lo,hi), p)
    root, actual = seen['conductors'], np.asarray(seen['occupancy'])
    assert root.released_port_edges
    driven = {(source[3], tuple(map(int, source[:3]))) for source in seen['sources']}
    assert {(comp, cell) for _, comp, cell in root.released_port_edges} == driven
    expected_cells = set()
    for entity, comp, driven in root.released_port_edges:
        cells = independent_cells(comp, driven, grid.shape, root.periodic)
        expected_cells.update(cells)
        weight = np.prod([1.-np.float64(actual[c]) for c in cells], dtype=np.float64)
        assert weight == 1., ('C1', lane, kind, placement, driven, weight)
    np.testing.assert_array_equal(root.released_cells, np.array(sorted(expected_cells),np.int32))
    assert root.released_cells.dtype == np.int32
    expected = np.array(occupancy)
    # Projection preserves .5 and 1; compare with float32 tolerance only on
    # topology's input density mapping, never on the release or its weight.
    if lane == 'topology':
        from rfx.topology import density_to_material_fields
        rho = jax.nn.sigmoid(jnp.log(p/(1.-p+1e-8)+1e-8))
        value = density_to_material_fields(jnp.full(tuple(hi-lo),rho), 1., 1., pec_fg=True, beta=1.).pec_occupancy
        expected[tuple(slice(a,b) for a,b in zip(lo,hi))] = np.asarray(value)
    unreleased = np.ones(grid.shape, bool)
    unreleased[tuple(root.released_cells.T)] = False
    np.testing.assert_array_equal(actual[unreleased], expected[unreleased], err_msg='C2')
    if placement == 'away':
        np.testing.assert_array_equal(actual, expected)
    assert len(seen['warnings']) == (placement != 'away'), ('C4', seen['warnings'])
    if seen['warnings']:
        warning = seen['warnings'][0]
        assert warning.category is UserWarning
        message = str(warning.message)
        for entity, comp, driven in root.released_port_edges:
            assert entity in message and comp in message and str(driven) in message
        assert ('microstrip' if kind == 'msl' else 'wire' if kind == 'wire' else 'lumped') in message
        assert 'cells' in message and 'kept free of design metal' in message


@pytest.mark.parametrize('component,edge,periodic,expected', [
    ('ex',(3,3,3),(False,False,False),[(3,2,2),(3,2,3),(3,3,2),(3,3,3)]),
    ('ey',(3,3,3),(False,False,False),[(2,3,2),(2,3,3),(3,3,2),(3,3,3)]),
    ('ez',(3,3,3),(False,False,False),[(2,2,3),(2,3,3),(3,2,3),(3,3,3)]),
    ('ez',(0,3,3),(True,False,False),[(0,2,3),(0,3,3),(16,2,3),(16,3,3)]),
    ('ez',(0,3,3),(False,False,False),[(0,2,3),(0,3,3)]),
])
def test_c3_literal_cells(component, edge, periodic, expected):
    sim, grid, _, _ = model('uniform','ez')
    root = replace(products.realized_conductors(sim,grid), periodic=periodic, pec_edges=None)
    root = products.publish_port_release(root,[edge],component=component,entity_id='port[0]',kind='lumped')
    np.testing.assert_array_equal(root.released_cells,np.asarray(expected,np.int32))
    assert root.released_port_edges == (('port[0]',component,edge),)


def test_c3_wire_literal_and_singleton():
    sim, grid, _, _ = model('uniform','wire')
    root = products.preview_port_stages(sim,products.realized_conductors(sim,grid))
    expected = [(7,7,4),(7,7,5),(7,7,6),(7,7,7),(7,8,4),(7,8,5),(7,8,6),(7,8,7),
                (8,7,4),(8,7,5),(8,7,6),(8,7,7),(8,8,4),(8,8,5),(8,8,6),(8,8,7)]
    np.testing.assert_array_equal(root.released_cells,np.array(expected,np.int32))
    assert root.released_port_edges == tuple(('port[0]','ez',(8,8,k)) for k in (4,5,6,7))
    single = replace(root,grid=SimpleNamespace(shape=(1,5,5)),released_port_edges=(),
                     released_cells=np.empty((0,3),np.int32),_released_port_details=())
    single = products.publish_port_release(single,[(0,2,2)],component='ez',entity_id='p',kind='lumped')
    np.testing.assert_array_equal(single.released_cells,[(0,1,2),(0,2,2)])


@pytest.mark.parametrize('lane',['uniform','graded'])
def test_c4_zero_and_traced_warning(monkeypatch,lane):
    sim, grid, _, _ = model(lane,'ez')
    seen = capture(monkeypatch,sim,grid,lane,jnp.zeros(grid.shape))
    assert not seen['warnings']
    root=seen['conductors']
    assert len(root.released_cells) > 0
    expected = np.array([(7,7,4), (7,8,4), (8,7,4), (8,8,4)])
    with pytest.warns(UserWarning,match='port\\[0\\]') as caught:
        grad=jax.grad(lambda x:jnp.sum(products.release_port_occupancy(x,root)))(jnp.zeros(grid.shape))
    assert len(caught)==1
    np.testing.assert_array_equal(np.asarray(grad)[tuple(expected.T)],0.)
    with pytest.warns(UserWarning,match='port\\[0\\]') as caught:
        jax.jit(lambda x:products.release_port_occupancy(x,root))(jnp.zeros(grid.shape)).block_until_ready()
    assert len(caught)==1


# Literal expectations for these fixed fixtures, independent of the publisher.
GRADIENT_CELLS = {
    'ex': [(8,7,3), (8,7,4), (8,8,3), (8,8,4)],
    'ey': [(7,8,3), (7,8,4), (8,8,3), (8,8,4)],
    'ez': [(7,7,4), (7,8,4), (8,7,4), (8,8,4)],
    'wire': [(7,7,4), (7,7,5), (7,7,6), (7,7,7),
             (7,8,4), (7,8,5), (7,8,6), (7,8,7),
             (8,7,4), (8,7,5), (8,7,6), (8,7,7),
             (8,8,4), (8,8,5), (8,8,6), (8,8,7)],
    'msl-uniform': [
        (7,5,6), (7,5,7), (7,5,8), (7,5,9),
        (7,6,6), (7,6,7), (7,6,8), (7,6,9),
        (7,7,6), (7,7,7), (7,7,8), (7,7,9),
        (7,8,6), (7,8,7), (7,8,8), (7,8,9),
        (7,9,6), (7,9,7), (7,9,8), (7,9,9),
        (7,10,6), (7,10,7), (7,10,8), (7,10,9),
        (7,11,6), (7,11,7), (7,11,8), (7,11,9),
        (7,12,6), (7,12,7), (7,12,8), (7,12,9),
        (7,13,6), (7,13,7), (7,13,8), (7,13,9),
        (7,14,6), (7,14,7), (7,14,8), (7,14,9),
        (7,15,6), (7,15,7), (7,15,8), (7,15,9),
        (7,16,6), (7,16,7), (7,16,8), (7,16,9),
        (7,17,6), (7,17,7), (7,17,8), (7,17,9),
        (7,18,6), (7,18,7), (7,18,8), (7,18,9),
        (8,5,6), (8,5,7), (8,5,8), (8,5,9),
        (8,6,6), (8,6,7), (8,6,8), (8,6,9),
        (8,7,6), (8,7,7), (8,7,8), (8,7,9),
        (8,8,6), (8,8,7), (8,8,8), (8,8,9),
        (8,9,6), (8,9,7), (8,9,8), (8,9,9),
        (8,10,6), (8,10,7), (8,10,8), (8,10,9),
        (8,11,6), (8,11,7), (8,11,8), (8,11,9),
        (8,12,6), (8,12,7), (8,12,8), (8,12,9),
        (8,13,6), (8,13,7), (8,13,8), (8,13,9),
        (8,14,6), (8,14,7), (8,14,8), (8,14,9),
        (8,15,6), (8,15,7), (8,15,8), (8,15,9),
        (8,16,6), (8,16,7), (8,16,8), (8,16,9),
        (8,17,6), (8,17,7), (8,17,8), (8,17,9),
        (8,18,6), (8,18,7), (8,18,8), (8,18,9)],
    'msl-graded': [
        (7,7,6), (7,7,7), (7,7,8), (7,7,9),
        (7,8,6), (7,8,7), (7,8,8), (7,8,9),
        (7,9,6), (7,9,7), (7,9,8), (7,9,9),
        (7,10,6), (7,10,7), (7,10,8), (7,10,9),
        (7,11,6), (7,11,7), (7,11,8), (7,11,9),
        (7,12,6), (7,12,7), (7,12,8), (7,12,9),
        (7,13,6), (7,13,7), (7,13,8), (7,13,9),
        (7,14,6), (7,14,7), (7,14,8), (7,14,9),
        (7,15,6), (7,15,7), (7,15,8), (7,15,9),
        (7,16,6), (7,16,7), (7,16,8), (7,16,9),
        (8,7,6), (8,7,7), (8,7,8), (8,7,9),
        (8,8,6), (8,8,7), (8,8,8), (8,8,9),
        (8,9,6), (8,9,7), (8,9,8), (8,9,9),
        (8,10,6), (8,10,7), (8,10,8), (8,10,9),
        (8,11,6), (8,11,7), (8,11,8), (8,11,9),
        (8,12,6), (8,12,7), (8,12,8), (8,12,9),
        (8,13,6), (8,13,7), (8,13,8), (8,13,9),
        (8,14,6), (8,14,7), (8,14,8), (8,14,9),
        (8,15,6), (8,15,7), (8,15,8), (8,15,9),
        (8,16,6), (8,16,7), (8,16,8), (8,16,9)],
}


@pytest.mark.parametrize('lane',['uniform','graded'])
@pytest.mark.parametrize('kind',['ex','ey','ez','wire','msl'])
def test_c5_result_gradient(lane,kind):
    sim,grid,edge,component=model(lane,kind)
    sim.add_probe(tuple(float(grid.node_of(a,edge[a]+(a==0))) for a in range(3)),component)
    root=products.preview_port_stages(sim,products.realized_conductors(sim,grid,nonuniform=lane=='graded'))
    assert len(root.released_cells) > 0
    expected = np.array(GRADIENT_CELLS[f'msl-{lane}' if kind == 'msl' else kind])
    np.testing.assert_array_equal(root.released_cells, expected)
    lo,hi=box(edge,component,kind,'interior')
    occupancy=jnp.zeros(grid.shape).at[tuple(slice(a,b) for a,b in zip(lo,hi))].set(.5)
    def loss(o):
        r=sim.forward(pec_occupancy_override=o,n_steps=40,checkpoint=False,skip_preflight=True)
        return jnp.sum(r.time_series**2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        grad=np.asarray(jax.grad(loss)(occupancy))
    released = [w for w in caught if 'Port occupancy release:' in str(w.message)]
    assert len(released) == 1, ('C4 forward warning count', len(released))
    assert released[0].category is UserWarning
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(grad[tuple(expected.T)],0.)
    region=np.asarray(occupancy)>0
    region[tuple(expected.T)]=False
    assert np.any(grad[region]!=0), 'dead objective'


def test_c6_coax_refusal(monkeypatch):
    sim=Simulation(freq_max=10e9,domain=(.016,.016,.012),dx=.001,boundary='cpml',cpml_layers=4)
    sim.add_coaxial_port((.008,.008,0.),face='bottom',pin_length=.005)
    import rfx.simulation as kernel
    monkeypatch.setattr(kernel,'run',lambda *a,**k:pytest.fail('stepped'))
    with pytest.raises(NotImplementedError,match=r'add_coaxial_port\(\) is not wired into Simulation.forward\(\)'):
        sim.forward(pec_occupancy_override=jnp.zeros(sim._build_grid().shape),n_steps=2,skip_preflight=True)


@pytest.mark.parametrize('kind',['ex','ey','ez','wire','msl'])
def test_c6_graded_distributed_refusal(monkeypatch,kind):
    sim,grid,_,_=model('graded',kind)
    import rfx.runners.distributed_nu as kernel
    monkeypatch.setattr(kernel,'run_nonuniform_distributed_pec',lambda *a,**k:pytest.fail('stepped'))
    message='microstrip port' if kind=='msl' else 'Lumped / wire ports'
    with pytest.raises(NotImplementedError,match=message):
        sim.forward(pec_occupancy_override=jnp.ones(grid.shape)*.5,n_steps=2,
                    distributed=True,devices=jax.devices(),skip_preflight=True)


@pytest.mark.parametrize('refusal',['distributed','adi','subgrid','run'])
def test_c6_other_refusals(monkeypatch,refusal):
    sim,grid,_,_=model('uniform','ez')
    import rfx.simulation as kernel
    monkeypatch.setattr(kernel,'run',lambda *a,**k:pytest.fail('stepped'))
    kwargs=dict(pec_occupancy_override=jnp.ones(grid.shape)*.5,n_steps=2,skip_preflight=True)
    if refusal=='run':
        with pytest.raises(TypeError,match="unexpected keyword argument 'pec_occupancy_override'"):
            sim.run(**kwargs)
    elif refusal=='distributed':
        with pytest.raises(NotImplementedError,match='only for non-uniform meshes'):
            sim.forward(distributed=True,**kwargs)
    elif refusal=='adi':
        sim._solver='adi'
        with pytest.raises(ValueError,match="solver='adi' does not support pec_occupancy_override"):
            sim.forward(**kwargs)
    else:
        sim.add_refinement(z_range=(.003,.007),ratio=2)
        with pytest.raises(NotImplementedError,match='refinement|subgrid'):
            sim.forward(**kwargs)


@pytest.mark.parametrize('kind',['ex','ey','ez','wire','msl'])
def test_c6_design_box_names_port(monkeypatch,kind):
    sim,grid,edge,_=model('uniform',kind)
    import rfx.simulation as kernel
    monkeypatch.setattr(kernel,'run',lambda *a,**k:pytest.fail('stepped'))
    point=tuple(x*.001 for x in edge)
    with pytest.raises(ValueError,match=r'(?:msl_port|port)\[0\].*port-cleared'):
        sim.forward(design_box=(point,point),design_occupancy_override=jnp.ones((1,1,1))*.5,
                    n_steps=2,skip_preflight=True)


@pytest.mark.parametrize('cell,admitted', [((9,8,4), True), ((8,8,5), True), ((7,7,4), False)])
def test_r7_design_box_cell_set(monkeypatch, cell, admitted):
    sim, grid, _, _ = model('uniform', 'ez')
    import rfx.simulation as kernel
    seen = {}
    occupancy = jnp.full((1,1,1), .5)
    def observe(*args, **kwargs):
        seen['design'] = kwargs['design_occupancy']
        raise Captured
    monkeypatch.setattr(kernel, 'run', observe)
    point = tuple(float(grid.node_of(a, i)) for a, i in enumerate(cell))
    if admitted:
        with pytest.raises(Captured):
            sim.forward(design_box=(point,point), design_occupancy_override=occupancy,
                        n_steps=2, checkpoint=False, skip_preflight=True)
        assert seen['design'].bounds == tuple(v for i in cell for v in (i,i+1))
        np.testing.assert_array_equal(seen['design'].occupancy, occupancy)
    else:
        with pytest.raises(ValueError, match=r'port\[0\].*port-cleared'):
            sim.forward(design_box=(point,point), design_occupancy_override=occupancy,
                        n_steps=2, checkpoint=False, skip_preflight=True)
        assert not seen


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_wire_dead_edge_releases_only_live_edges(monkeypatch, lane):
    sim, grid, _, _ = model(lane, 'wire')
    lo = tuple(float(grid.node_of(a, i)) for a, i in enumerate((8,8,7)))
    hi = tuple(float(grid.node_of(a, i)) for a, i in enumerate((9,9,8)))
    sim.add(Box(lo, hi), material='pec')
    occupancy = jnp.full(grid.shape, .5)
    seen = capture(monkeypatch, sim, grid, lane, occupancy)
    root, actual = seen['conductors'], np.asarray(seen['occupancy'])
    assert root.released_port_edges == (
        ('port[0]', 'ez', (8,8,4)), ('port[0]', 'ez', (8,8,5)), ('port[0]', 'ez', (8,8,6)))
    assert root.pec_edges[2][8,8,7]
    for cells in [
            [(7,7,4), (7,8,4), (8,7,4), (8,8,4)],
            [(7,7,5), (7,8,5), (8,7,5), (8,8,5)],
            [(7,7,6), (7,8,6), (8,7,6), (8,8,6)]]:
        assert np.prod([1.-np.float64(actual[c]) for c in cells], dtype=np.float64) == 1.
    for cell in [(7,7,7), (7,8,7), (8,7,7), (8,8,7)]:
        assert actual[cell] == .5


def test_all_dead_wire_has_geometry_preview():
    sim, _, _, _ = model('uniform', 'wire')
    sim.add(Box((.008,.008,.004), (.009,.009,.008)), material='pec')
    assert sim.realized_geometry() is not None


def test_release_rejects_wrong_occupancy_shape():
    sim, grid, _, _ = model('uniform', 'ez')
    root = products.preview_port_stages(sim, products.realized_conductors(sim, grid))
    wrong = jnp.zeros((16,17,13))
    with pytest.raises(ValueError) as caught:
        products.release_port_occupancy(wrong, root)
    assert '(16, 17, 13)' in str(caught.value) and '(17, 17, 13)' in str(caught.value)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_c2_former_face_cells(monkeypatch, lane):
    sim, grid, edge, component = model(lane, 'ez')
    lo, hi = box(edge, component, 'ez', 'interior')
    occupancy = jnp.zeros(grid.shape).at[tuple(slice(a,b) for a,b in zip(lo,hi))].set(.5)
    actual = np.asarray(capture(monkeypatch, sim, grid, lane, occupancy)['occupancy'])
    for cell in [(9,8,4), (8,9,4), (8,8,3), (8,8,5)]:
        assert actual[cell] == .5, ('C2', cell, actual[cell])


def test_c4_multiple_ports_warn_once(monkeypatch):
    sim, grid, _, _ = model('uniform', 'ez')
    sim.add_port((.010,.010,.006), 'ex', impedance=50., excite=False)
    seen = capture(monkeypatch, sim, grid, 'uniform', jnp.full(grid.shape,.5))
    assert len(seen['warnings']) == 1
    text = str(seen['warnings'][0].message)
    assert 'port[0]' in text and 'port[1]' in text and 'ex' in text and 'ez' in text
    assert seen['conductors'].released_port_edges == (('port[0]','ez',(8,8,4)), ('port[1]','ex',(10,10,6)))


def test_soft_source_publishes_nothing():
    sim=Simulation(freq_max=10e9, domain=(.016,.016,.012), dx=.001, boundary='pec')
    sim.add_source((.008,.008,.004),'ez',amplitude_kind='field')
    root=products.preview_port_stages(sim,products.realized_conductors(sim,sim._build_grid()))
    assert root.released_port_edges == () and root.released_cells.shape == (0,3)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('mode', ['uniform', 'laplace'])
def test_preview_matches_modal_support(monkeypatch, lane, mode):
    sim, grid, _, _ = model(lane, 'msl')
    sim._msl_ports[0] = replace(sim._msl_ports[0], mode=mode)
    preview = products.preview_port_stages(sim, products.realized_conductors(
        sim, grid, nonuniform=lane == 'graded'))
    seen = capture(monkeypatch, sim, grid, lane, jnp.zeros(grid.shape))
    assert preview.released_port_edges == seen['conductors'].released_port_edges
    np.testing.assert_array_equal(preview.released_cells, seen['conductors'].released_cells)


# W6: replaces "a waveguide-port model publishes no released cells" (shape (0, 3)).
WG_LAYERS = (12, 13, 14, 15, 16, 22, 23)
WG_CELLS = np.array([(i, j, k) for i in WG_LAYERS
                     for j in range(11) for k in range(5)], dtype=np.int32)
WG_GRADED_CELLS = np.array([(i, j, k) for i in WG_LAYERS
                            for j in range(7, 19) for k in range(7, 13)
                            if (j, k) != (7, 7)], dtype=np.int32)


def waveguide_model(lane='uniform'):
    from tests.unit.autodiff.test_waveguide_forward import _wr90_sim
    sim = _wr90_sim()
    if lane == 'graded':
        sim._dx_profile = np.array([.002]*8 + [.0019, .0021]*4 + [.002]*9)
    grid = sim._build_nonuniform_grid() if lane == 'graded' else sim._build_grid()
    return sim, grid


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
@pytest.mark.parametrize('direction', ['+', '-'])
def test_w1_literal_waveguide_cells(axis, direction):
    if axis == 'x' and direction == '+':
        sim, grid = waveguide_model()
    else:
        domain = {'x': (.05,.02286,.01016), 'y': (.02286,.05,.01016),
                  'z': (.02286,.01016,.05)}[axis]
        ranges = {'x': dict(y_range=(0.,.02286), z_range=(0.,.01016)),
                  'y': dict(x_range=(0.,.02286), z_range=(0.,.01016)),
                  'z': dict(x_range=(0.,.02286), y_range=(0.,.01016))}[axis]
        sim = Simulation(freq_max=12e9, domain=domain, dx=.002,
                         boundary='cpml', cpml_layers=8)
        sim.add_waveguide_port(.01 if direction == '+' else .04,
                               direction=direction+axis, **ranges)
        grid = sim._build_grid()
    cfg = sim._build_waveguide_port_config(sim._waveguide_ports[0], grid, jnp.array([10e9]), 40)
    # Read once from these six real public-API ports, then pinned as literals.
    planes = (13, 16, 23) if direction == '+' else (28, 25, 18)
    layers = WG_LAYERS if direction == '+' else (17, 18, 24, 25, 27, 28, 29)
    edge_planes = {13, 14, 16, 23} if direction == '+' else {18, 25, 28, 29}
    assert (cfg.x_index, cfg.ref_x, cfg.probe_x) == planes
    assert (cfg.u_lo, cfg.u_hi, cfg.v_lo, cfg.v_hi) == (0, 11, 0, 5)
    assert cfg.normal_axis == axis and cfg.direction == direction+axis
    root = products.realized_conductors(sim, grid)
    if axis == 'x' and direction == '+':
        expected = WG_CELLS
    else:
        coords = {'x': lambda p,u,v: (p,u,v), 'y': lambda p,u,v: (u,p,v),
                  'z': lambda p,u,v: (u,v,p)}[axis]
        expected = np.array(sorted(coords(p,u,v) for p in layers
                                   for u in range(11) for v in range(5)), np.int32)
    components = {'x': ('ey', 'ez'), 'y': ('ex', 'ez'), 'z': ('ex', 'ey')}[axis]
    assert (cfg.e_u_component, cfg.e_v_component) == components
    root = products.waveguide_port_stage(root, cfg, 'waveguide_port[0]')
    np.testing.assert_array_equal(root.released_cells, expected)
    assert root.released_cells.dtype == np.int32
    assert {e[1] for e in root.released_port_edges} == set(components)
    assert {e[2]['xyz'.index(axis)] for e in root.released_port_edges} == edge_planes


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('region', ['whole', 'source', 'measurement'])
@pytest.mark.parametrize('p', [.5, 1.])
def test_w2_waveguide_kernel_weights(monkeypatch, lane, region, p):
    sim, grid = waveguide_model(lane)
    expected = WG_GRADED_CELLS if lane == 'graded' else WG_CELLS
    us, vs = (range(8, 19), range(8, 13)) if lane == 'graded' else (range(11), range(5))
    occupancy = np.zeros(grid.shape, np.float32)
    if region == 'whole':
        occupancy[:] = p
    else:
        layers = (12, 13) if region == 'source' else (22, 23)
        selected = expected[np.isin(expected[:, 0], layers)]
        occupancy[tuple(selected.T)] = p
    seen = capture(monkeypatch, sim, grid, lane, jnp.asarray(occupancy))
    actual = np.asarray(seen['occupancy'])
    def factor(i, j, k):
        return 1. if min(i, j, k) < 0 else 1. - np.float64(actual[i, j, k])
    for i in (13, 14, 16, 23):
        for j in us:
            for k in vs:
                # Test-owned four-cell products for Ey and Ez, including clipping.
                ey_weight = factor(i,j,k)*factor(i-1,j,k)*factor(i,j,k-1)*factor(i-1,j,k-1)
                ez_weight = factor(i,j,k)*factor(i-1,j,k)*factor(i,j-1,k)*factor(i-1,j-1,k)
                assert ey_weight == 1. and ez_weight == 1., (i, j, k, ey_weight, ez_weight)
    np.testing.assert_array_equal(seen['conductors'].released_cells, expected)
    outside = np.ones(grid.shape, bool)
    outside[tuple(expected.T)] = False
    assert actual[outside].tobytes() == occupancy[outside].tobytes()
    assert len(seen['warnings']) == 1
    message = str(seen['warnings'][0].message)
    count = '497 cells' if lane == 'graded' else '385 cells'
    for part in ('waveguide_port[0]', 'source 13 (also 14)', 'reference 16', 'measurement 23', count):
        assert part in message
    assert len(message) < 300


def test_w3_waveguide_source_record():
    sim, grid = waveguide_model()
    zero = jnp.zeros(grid.shape)
    occupied = zero.at[12:14, :11, :5].set(1.)
    baseline = sim.forward(pec_occupancy_override=zero, n_steps=40, checkpoint=False, skip_preflight=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = sim.forward(pec_occupancy_override=occupied, n_steps=40, checkpoint=False, skip_preflight=True)
    a, b = np.asarray(result.time_series), np.asarray(baseline.time_series)
    assert np.max(np.abs(a)) > 0.
    assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), np.max(np.abs(a-b))


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_w4_waveguide_warning_and_gradient(lane):
    sim, grid = waveguide_model(lane)
    expected = WG_GRADED_CELLS if lane == 'graded' else WG_CELLS
    beside = (24, 13, 10) if lane == 'graded' else (24, 5, 2)
    def loss(o):
        result = sim.forward(pec_occupancy_override=o, n_steps=40, checkpoint=False, skip_preflight=True)
        return jnp.sum(result.time_series**2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        loss(jnp.zeros(grid.shape)).block_until_ready()
    assert not [w for w in caught if 'Port occupancy release:' in str(w.message)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gradient = np.asarray(jax.grad(loss)(jnp.zeros(grid.shape)))
    released = [w for w in caught if 'Port occupancy release:' in str(w.message)]
    assert len(released) == 1 and 'waveguide_port[0]' in str(released[0].message)
    np.testing.assert_array_equal(gradient[tuple(expected.T)], 0.)
    assert np.isfinite(gradient).all()
    assert gradient[beside] != 0., gradient[beside]


@pytest.mark.parametrize('cell,admitted', [((23,5,2), False), ((24,5,2), True)])
def test_w5_waveguide_design_box(monkeypatch, cell, admitted):
    sim, grid = waveguide_model()
    import rfx.simulation as kernel
    def stop(*args, **kwargs):
        raise Captured
    monkeypatch.setattr(kernel, 'run', stop)
    point = tuple(float(grid.node_of(a, i)) for a, i in enumerate(cell))
    error = Captured if admitted else ValueError
    kwargs = {} if admitted else {'match': r'waveguide_port\[0\].*port-cleared'}
    with pytest.raises(error, **kwargs):
        sim.forward(design_box=(point,point), design_occupancy_override=jnp.ones((1,1,1))*.5,
                    n_steps=2, checkpoint=False, skip_preflight=True)


@pytest.mark.parametrize('mutation', ['removed', 'source_only'])
def test_w7_waveguide_mutations(monkeypatch, mutation):
    if mutation == 'removed':
        monkeypatch.setattr(products, 'waveguide_port_stage', lambda root, cfg, entity: root)
    else:
        publish = products.publish_port_release
        def source_only(root, cells, **kwargs):
            if kwargs['kind'] == 'waveguide':
                cells = [cell for cell in cells if cell[0] == 13]
            return publish(root, cells, **kwargs)
        monkeypatch.setattr(products, 'publish_port_release', source_only)
    with pytest.raises(AssertionError):
        test_w1_literal_waveguide_cells('x', '+')
    with pytest.MonkeyPatch.context() as capture_patch:
        with pytest.raises(AssertionError):
            test_w2_waveguide_kernel_weights(capture_patch, 'uniform', 'measurement', .5)
    if mutation == 'removed':
        with pytest.raises(AssertionError):
            test_w3_waveguide_source_record()
    else:
        test_w3_waveguide_source_record()


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_waveguide_without_occupancy_does_not_publish(monkeypatch, lane):
    sim, grid = waveguide_model(lane)
    def forbidden(*args, **kwargs):
        pytest.fail('no-occupancy forward enumerated waveguide edges')
    monkeypatch.setattr(products, 'waveguide_port_stage', forbidden)
    seen = capture(monkeypatch, sim, grid, lane)
    assert seen['conductors'].released_cells.shape == (0, 3)
    assert not seen['conductors'].released_port_edges
    assert seen['occupancy'] is None


def test_two_large_waveguide_ports_publication_cost_and_bounded_refusal():
    sim = Simulation(freq_max=12e9, domain=(.04,.08,.04), dx=.001,
                     boundary='cpml', cpml_layers=8)
    for sign, position in (('+', .01), ('-', .03)):
        sim.add_waveguide_port(position, direction=sign+'x',
                               y_range=(0.,.08), z_range=(0.,.04))
    grid = sim._build_grid()
    configs = [sim._build_waveguide_port_config(entry, grid, jnp.array([10e9]), 2)
               for entry in sim._waveguide_ports]
    assert all((cfg.u_hi-cfg.u_lo, cfg.v_hi-cfg.v_lo) == (80,40) for cfg in configs)
    root = products.realized_conductors(sim, grid)
    start = perf_counter()
    for i, config in enumerate(configs):
        root = products.waveguide_port_stage(root, config, f'waveguide_port[{i}]')
    elapsed = perf_counter()-start
    print(f'80x40 two-port publication: {elapsed:.6f}s')
    assert elapsed < 10., elapsed
    assert len(root.released_port_edges) == 51200
    design = SimpleNamespace(bounds=tuple(v for n in grid.shape for v in (0,n-1)))
    start = perf_counter()
    with pytest.raises(ValueError) as caught:
        products.refuse_port_design_box(design, root)
    message = str(caught.value)
    print(f'80x40 two-port refusal: {perf_counter()-start:.6f}s; {len(message)} characters')
    assert message.startswith('waveguide_port[0], waveguide_port[1]:')
    sample = sorted(map(tuple, root.released_cells.tolist()))[:6]
    assert f'cell(s) {sample} and {len(root.released_cells)-6} more' in message
    assert len(message) < 600

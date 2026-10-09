"""Port design-metal release: independent incident-cell and kernel-boundary checks."""
from dataclasses import replace
from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, GaussianPulse
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
    with pytest.warns(UserWarning,match='port\\[0\\]') as caught:
        grad=jax.grad(lambda x:jnp.sum(products.release_port_occupancy(x,root)))(jnp.zeros(grid.shape))
    assert len(caught)==1
    np.testing.assert_array_equal(np.asarray(grad)[tuple(root.released_cells.T)],0.)
    with pytest.warns(UserWarning,match='port\\[0\\]') as caught:
        jax.jit(lambda x:products.release_port_occupancy(x,root))(jnp.zeros(grid.shape)).block_until_ready()
    assert len(caught)==1


@pytest.mark.parametrize('lane',['uniform','graded'])
@pytest.mark.parametrize('kind',['ex','ey','ez','wire','msl'])
def test_c5_result_gradient(lane,kind):
    sim,grid,edge,component=model(lane,kind)
    sim.add_probe(tuple(float(grid.node_of(a,edge[a]+(a==0))) for a in range(3)),component)
    root=products.preview_port_stages(sim,products.realized_conductors(sim,grid,nonuniform=lane=='graded'))
    lo,hi=box(edge,component,kind,'interior')
    occupancy=jnp.zeros(grid.shape).at[tuple(slice(a,b) for a,b in zip(lo,hi))].set(.5)
    def loss(o):
        r=sim.forward(pec_occupancy_override=o,n_steps=40,checkpoint=False,skip_preflight=True)
        return jnp.sum(r.time_series**2)
    with pytest.warns(UserWarning,match='Port occupancy release:'):
        grad=np.asarray(jax.grad(loss)(occupancy))
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(grad[tuple(root.released_cells.T)],0.)
    region=np.asarray(occupancy)>0
    region[tuple(root.released_cells.T)]=False
    assert np.any(grad[region]!=0), 'dead objective'


def test_c6_coax_refusal(monkeypatch):
    sim=Simulation(freq_max=10e9,domain=(.016,.016,.012),dx=.001,boundary='cpml',cpml_layers=4)
    sim.add_coaxial_port((.008,.008,0.),face='bottom',pin_length=.005)
    import rfx.simulation as kernel
    monkeypatch.setattr(kernel,'run',lambda *a,**k:pytest.fail('stepped'))
    with pytest.raises(NotImplementedError,match=r'add_coaxial_port\(\) is not wired into Simulation.forward\(\)'):
        sim.forward(pec_occupancy_override=jnp.zeros(sim._build_grid().shape),n_steps=2,skip_preflight=True)


def test_waveguide_empty_release(monkeypatch):
    from tests.unit.autodiff.test_waveguide_forward import _wr90_sim
    sim=_wr90_sim();grid=sim._build_grid()
    seen=capture(monkeypatch,sim,grid,'uniform',jnp.ones(grid.shape)*.5)
    assert seen['conductors'].released_cells.shape==(0,3)
    assert seen['conductors'].released_cells.dtype==np.int32


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
    with pytest.raises(ValueError,match=r'port-cleared.*(?:msl_port|port)\[0\]'):
        sim.forward(design_box=(point,point),design_occupancy_override=jnp.ones((1,1,1))*.5,
                    n_steps=2,skip_preflight=True)


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

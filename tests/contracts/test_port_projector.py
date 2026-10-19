"""PR 1a: compare sampling with the pre-table arithmetic, never with itself."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import CurlBoundary
from rfx.measurement.ports import lumped_wire_owner
from rfx.measurement.projector import sample_table, gather, combine, project
from rfx.probes.probes import _ampere_loop
from rfx.nonuniform import _build_wp_meta, wire_port_current


def grid_case(mesh, shape=(7, 6, 5)):
    widths = [np.full(n, .0013) for n in shape]
    if mesh != 'uniform':
        axis = {'dz': 2, 'dx': 0}[mesh]
        widths[axis] = np.linspace(.00037, .00173, shape[axis])
    return SimpleNamespace(shape=shape, cells=lambda a: widths[a],
                           dx_arr=jnp.asarray(widths[0]), dy_arr=jnp.asarray(widths[1]),
                           dz=jnp.asarray(widths[2]))


def owner(grid, mesh, comp, live, boundary):
    from rfx.measurement import ports
    o = ports.lumped_wire_owner(grid, comp, live, nu=mesh != 'uniform',
                               pmc_faces=boundary.pmc_faces, periodic=boundary.periodic,
                               port_id=0, kind='wire')
    return replace(o, channels=tuple(c for c in o.channels if c.sample_stage == 'post-injection'))


def today(state, grid, mesh, comp, live, boundary):
    mid = live[len(live)//2]
    f = getattr(state, comp)
    if mesh == 'uniform':
        dx = float(grid.cells(0)[0])
        return jnp.stack((-f[mid]*dx, _ampere_loop(state, mid, comp, dx, boundary.periodic, boundary=boundary),
                          -sum(f[c] for c in live)*dx))
    meta = _build_wp_meta([dict(mid_i=mid[0], mid_j=mid[1], mid_k=mid[2],
                               component=comp, impedance=50., live_cells=live)], grid)[0]
    width = meta[{'ex': 5, 'ey': 6, 'ez': 11}[comp]]
    return jnp.stack((-f[mid]*width,
                      wire_port_current(state.hx, state.hy, state.hz, comp, *mid,
                                        meta[9], meta[10], meta[12], boundary=boundary),
                      -sum(f[c]*w for c, w in zip(live, meta[14]))))


@pytest.mark.parametrize('mesh', ['uniform', 'dz', 'dx'])
@pytest.mark.parametrize('comp', ['ex', 'ey', 'ez'])
@pytest.mark.parametrize('case', ['interior', 'pec_low', 'pec_high', 'pmc_low', 'pmc_high',
                                  'periodic_low', 'periodic_high', 'length_one',
                                  'pec_last_physical', 'pmc_last_physical', 'periodic_last_physical'])
def test_random_fields_match_today(mesh, comp, case):
    """J1a, each backward axis independently, including arbitrary high ghosts."""
    axes = {'ex': (1, 2), 'ey': (2, 0), 'ez': (0, 1)}[comp]
    worst = np.zeros(3)
    for axis in axes:
        shape = [7, 6, 5]
        if case == 'length_one':
            shape[axis] = 1
        grid = grid_case(mesh, tuple(shape))
        mid = [2, 2, 2]
        if 'low' in case or case == 'length_one':
            mid[axis] = 0
        elif 'high' in case:
            mid[axis] = shape[axis]-1
        elif 'last_physical' in case:
            mid[axis] = shape[axis]-2
        along = 'xyz'.index(comp[1])
        live = []
        for q in (0, 1, 2):
            cell = mid.copy()
            cell[along] = q+1
            live.append(tuple(cell))
        live = tuple(live)
        periodic = tuple(case.startswith('periodic') and a == axis for a in range(3))
        faces = frozenset((f'{"xyz"[axis]}_{"lo" if "low" in case else "hi"}',)) if case.startswith('pmc') else frozenset()
        boundary = CurlBoundary(pmc_faces=faces, periodic=periodic)
        table = sample_table((owner(grid, mesh, comp, live, boundary),), dtype=np.float32)
        rng = np.random.default_rng(709)
        states = {c: jnp.asarray(rng.normal(size=(64, *shape)), dtype=jnp.float32)
                  for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')}
        def evaluate(fields):
            state = SimpleNamespace(**fields)
            return project(state, table), today(state, grid, mesh, comp, live, boundary)
        got, expected = jax.jit(jax.vmap(evaluate))(states)
        got, expected = np.asarray(got), np.asarray(expected)
        ratios = np.max(np.abs(got-expected), axis=0)/np.spacing(np.max(np.abs(expected), axis=0))
        worst = np.maximum(worst, ratios)
    assert np.max(worst) <= 9, worst


def test_combine_time_axis_jit_vmap_and_gradient():
    grid = grid_case('uniform')
    table = sample_table((owner(grid, 'uniform', 'ez', ((2, 2, 1), (2, 2, 2)), CurlBoundary()),), dtype=np.float32)
    state = SimpleNamespace(**{c: jnp.arange(np.prod(grid.shape), dtype=jnp.float32).reshape(grid.shape)
                               for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')})
    samples = gather(state, table)
    times = jnp.stack((samples, 2*samples), axis=1)
    np.testing.assert_array_equal(combine(times, table), jnp.stack((project(state, table), 2*project(state, table))))
    np.testing.assert_array_equal(jax.jit(lambda x: combine(x, table))(times), combine(times, table))
    gradient = jax.grad(lambda x: jnp.sum(combine(x, table)))(samples)
    np.testing.assert_array_equal(gradient, np.concatenate([t.weights for t in table.components]))
    jaxpr = jax.make_jaxpr(lambda ex, ey, ez, hx, hy, hz: project(
        SimpleNamespace(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz), table))(
            *(getattr(state, c) for c in ('ex','ey','ez','hx','hy','hz')))
    names = [e.primitive.name for e in jaxpr.jaxpr.eqns]
    # The extra gather permutes all terms once for the sorted reduction.
    assert names.count('gather') == len(table.components)+1
    assert names.count('scatter-add') == 1


@pytest.mark.parametrize('mesh', ['uniform', 'dz', 'dx'])
@pytest.mark.parametrize('comp', ['ex', 'ey', 'ez'])
def test_owner_weights_match_runner_metrics_bitwise(mesh, comp):
    grid = grid_case(mesh)
    live = ((2, 2, 1), (2, 2, 2), (2, 2, 3))
    o = owner(grid, mesh, comp, live, CurlBoundary())
    if mesh == 'uniform':
        expected = [grid.cells(0)[0]]*3
        gap = [expected[0]]*3
    else:
        m = _build_wp_meta([dict(mid_i=2, mid_j=2, mid_k=2, component=comp,
                                impedance=50., live_cells=live)], grid)[0]
        expected = [m[{'ex':5, 'ey':6, 'ez':11}[comp]], *({'ex':(m[12],m[10]), 'ey':(m[9],m[12]), 'ez':(m[10],m[9])}[comp])]
        gap = m[14]
    actual = [-o.channels[0].nodes[0].weight, o.channels[1].nodes[0].weight, -o.channels[1].nodes[2].weight]
    assert np.asarray(actual, np.float64).tobytes() == np.asarray(expected, np.float64).tobytes()
    assert np.asarray([-n.weight for n in o.channels[2].nodes], np.float64).tobytes() == np.asarray(gap, np.float64).tobytes()


def test_live_edges_and_whole_gap_are_used():
    grid = grid_case('uniform')
    # Realized PEC edge 0 is absent, even though it was in the declared span.
    live = ((2, 2, 1), (2, 2, 2), (2, 2, 3))
    o = owner(grid, 'uniform', 'ez', live, CurlBoundary())
    state = SimpleNamespace(**{c: jnp.zeros(grid.shape) for c in ('ex','ey','ez','hx','hy','hz')})
    state.ez = state.ez.at[2,2,0].set(1000).at[2,2,1].set(3).at[2,2,2].set(5).at[2,2,3].set(11)
    got = project(state, sample_table((o,), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(got)[[0,2]], np.array([-5.,-19.])*.0013, rtol=1e-7)


def test_ringdown_missing_term_names_port_and_node():
    from rfx.ringdown import _port_layout
    grid = grid_case('uniform')
    live = ((2,2,1), (2,2,2), (2,2,3))
    o = owner(grid, 'uniform', 'ez', live, CurlBoundary())
    pm = SimpleNamespace(owner=o, component='ez', live_cells=live, mid=(2,2,2))
    col = {(n.component, (n.i,n.j,n.k)): i for c in o.channels for i,n in enumerate(c.nodes)}
    del col['hy', (1,2,2)]
    with pytest.raises(KeyError, match=r'port:0:mode:0.*hy.*1, 2, 2'):
        _port_layout(pm, col)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('case', ['lumped', 'wire', 'plate', 'low_face', 'high_face'])
def test_plan_and_runner_compile_identical_nodes_and_weight_bits(graded, case, monkeypatch):
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.measurement import ports
    from rfx.measurement.plan import build_measurement_plan
    from tests.unit.ports.test_wire_port_live_mid_764 import _build
    if case == 'plate':
        sim = _build(graded)
    else:
        kw = dict(dz_profile=np.full(8, .001)) if graded else {}
        sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                         boundary=BoundarySpec(x='pmc', y='pec', z='pec') if 'face' in case else 'pec', **kw)
        x = 0. if case == 'low_face' else .008 if case == 'high_face' else .004
        sim.add_port(position=(x, .004, .003), component='ez', impedance=50.,
                     **({} if case == 'lumped' else dict(extent=.002)))
    captured = []
    original = ports.lumped_wire_owner

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ports, 'lumped_wire_owner', capture)
        sim.forward(n_steps=2, checkpoint=False, skip_preflight=True,
                    port_s11_freqs=np.array([2e9]))
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    plan = build_measurement_plan(sim, grid, n_steps=2, path='fwd_nonuniform' if graded else 'fwd_uniform',
                                 frequencies=np.array([2e9]))
    planned = next(o for o in plan.owners if o.id.startswith('port:'))
    assert captured, 'runner did not compile an owner'
    actual = captured[0]
    assert [c.name for c in planned.channels] == [c.name for c in actual.channels]
    for expected_channel, actual_channel in zip(planned.channels, actual.channels):
        assert [(n.component,n.i,n.j,n.k) for n in expected_channel.nodes] == [
            (n.component,n.i,n.j,n.k) for n in actual_channel.nodes]
        assert np.asarray([n.weight for n in expected_channel.nodes], np.float64).tobytes() == np.asarray(
            [n.weight for n in actual_channel.nodes], np.float64).tobytes()


def test_plate_owner_samples_only_the_realized_live_edge():
    from tests.unit.ports.test_wire_port_live_mid_764 import _build
    from rfx.measurement.plan import build_measurement_plan
    sim = _build(False)
    grid = sim._build_grid()
    plan = build_measurement_plan(sim, grid, n_steps=2)
    o = next(o for o in plan.owners if o.kind == 'wire')
    state = SimpleNamespace(**{c: jnp.zeros(grid.shape) for c in ('ex','ey','ez','hx','hy','hz')})
    state.ez = state.ez.at[6,6,4].set(1000).at[6,6,5].set(2000).at[6,6,6].set(7)
    table = sample_table((o,), dtype=np.float32)
    got = project(state, table)
    np.testing.assert_allclose(np.asarray(got)[[0,2]], [-.007,-.007], rtol=1e-7)


def test_mixed_components_and_port_kinds_keep_owner_channel_order():
    grid = grid_case('uniform')
    live = ((2,2,1),(2,2,2),(2,2,3))
    wires = tuple(owner(grid, 'uniform', comp, live, CurlBoundary()) for comp in ('ex','ey'))
    lumped = lumped_wire_owner(grid, 'ez', ((2,2,2),), nu=False, pmc_faces=(),
                               periodic=(False,)*3, port_id=2, kind='lumped')
    lumped = replace(lumped, channels=lumped.channels[:2])
    table = sample_table((wires[0], lumped, wires[1]), dtype=np.float32)
    assert table.channel_names == ('V','I','V_port','V','I','V','I','V_port')
    rng = np.random.default_rng(27)
    state = SimpleNamespace(**{c:jnp.asarray(rng.normal(size=grid.shape),jnp.float32)
                               for c in ('ex','ey','ez','hx','hy','hz')})
    expected = jnp.concatenate((today(state,grid,'uniform','ex',live,CurlBoundary()),
        today(state,grid,'uniform','ez',((2,2,2),),CurlBoundary())[:2],
        today(state,grid,'uniform','ey',live,CurlBoundary())))
    error = np.max(np.abs(np.asarray(project(state,table)-expected)))
    assert error <= 9*np.spacing(np.float32(np.max(np.abs(expected))))


def test_graded_run_matches_forward_lumped_sparameters():
    from rfx import Simulation
    sim = Simulation(freq_max=10e9, domain=(.008,.008,.008), dx=.001,
                     boundary='pec', dz_profile=np.full(8,.001))
    sim.add_port(position=(.004,.004,.003), component='ez', impedance=50.)
    run = sim.run(n_steps=2, compute_s_params=True, skip_preflight=True)
    forward = sim.forward(n_steps=2, port_s11_freqs=np.asarray(run.freqs),
                          checkpoint=False, skip_preflight=True)
    assert np.asarray(run.s_params).tobytes() == np.asarray(forward.s_params).tobytes()
    assert run.lumped_port_sparams and forward.lumped_port_sparams
    for actual, expected in zip(run.lumped_port_sparams[0][1], forward.lumped_port_sparams[0][1], strict=True):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_numpy_combine_preserves_float64_with_x64_off():
    from rfx.measurement.projector import ComponentTerms, SampleTable
    if jax.config.read('jax_enable_x64'):
        pytest.skip('requires the solver x64 setting to be off')
    assert jax.config.read('jax_enable_x64') is False
    rng = np.random.default_rng(1567)
    terms = []
    for component in ('ex', 'ey', 'hz'):
        n = int(rng.integers(20, 40))
        terms.append(ComponentTerms(component,
            tuple(rng.integers(0, 8, n, dtype=np.int32) for _ in range(3)),
            rng.integers(0, 7, n, dtype=np.int32), rng.normal(size=n)))
    ids = np.concatenate([t.channel_ids for t in terms])
    weights = np.concatenate([t.weights for t in terms])
    order = np.argsort(ids, kind='stable').astype(np.int32)
    table = SampleTable(tuple(terms), 8, tuple(str(i) for i in range(8)), order, ids[order])
    for trailing in ((), (31,), (3, 11)):
        samples = rng.normal(size=(len(ids),) + trailing)
        expected = np.zeros(trailing + (8,), dtype=np.float64)
        for channel in range(8):
            expected[..., channel] = np.sum(
                samples[ids == channel] * weights[ids == channel].reshape(
                    (-1,) + (1,) * len(trailing)), axis=0)
        actual = combine(samples, table)
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.float64
        for channel in range(8):
            peak = np.max(np.abs(expected[..., channel]))
            assert np.max(np.abs(actual[..., channel] - expected[..., channel])) <= 1e-12 * peak
    assert jax.config.read('jax_enable_x64') is False


@pytest.mark.parametrize('n_live', [1, 3])
def test_recorded_port_vi_rebuilds_in_float64_with_x64_off(n_live):
    """Ring-down host rebuild: float64 probe columns in, float64 out.

    Expected values are the hand-written Ampere loop on the 2x2x2 probe
    neighbourhood (I_z = dy (Hy[1] - Hy[0]) - dx (Hx[1] - Hx[0])), not
    the owner's weights.
    """
    from rfx.measurement.port_records import recorded_port_vi
    if jax.config.read('jax_enable_x64'):
        pytest.skip('requires the solver x64 setting to be off')
    assert jax.config.read('jax_enable_x64') is False
    dx = .0013
    live = tuple((2, 2, k) for k in range(2 - n_live // 2, 2 - n_live // 2 + n_live))
    o = lumped_wire_owner(grid_case('uniform'), 'ez', live, nu=False, pmc_faces=frozenset(),
                          periodic=(False, False, False), port_id=0, kind='wire')
    pm = SimpleNamespace(owner=o, component='ez', live_cells=live, mid=live[n_live // 2])
    rng = np.random.default_rng(1617)
    e = rng.normal(size=(n_live, 17))
    hx, hy, hz = (rng.normal(size=(2, 2, 2, 17)) for _ in range(3))
    v, v_port, i = recorded_port_vi(pm, e, hx, hy, hz)
    expected = (-dx * e[n_live // 2], -dx * e.sum(axis=0),
                dx * (hy[1, 1, 1] - hy[0, 1, 1]) - dx * (hx[1, 1, 1] - hx[1, 0, 1]))
    for got, want in zip((v, v_port, i), expected):
        assert isinstance(got, np.ndarray) and got.dtype == np.float64
        assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
    assert jax.config.read('jax_enable_x64') is False


@pytest.mark.parametrize('kind', ['wire', 'lumped'])
@pytest.mark.parametrize('defect', ['channel_order', 'midpoint'])
def test_runner_table_refuses_owner_metadata_disagreement(kind, defect, monkeypatch):
    from rfx.measurement import ports
    from rfx.measurement.projector import port_table
    original = ports.lumped_wire_owner

    def altered(*args, **kwargs):
        o = original(*args, **kwargs)
        channels = list(o.channels)
        if defect == 'channel_order':
            channels[0], channels[1] = channels[1], channels[0]
        else:
            n = channels[0].nodes[0]
            channels[0] = replace(channels[0], nodes=(replace(n, i=n.i+1),))
        return replace(o, channels=tuple(channels))

    monkeypatch.setattr(ports, 'lumped_wire_owner', altered)
    mid = (2, 2, 2)
    with pytest.raises(ValueError, match=f'port:0:mode:0:.*{defect.split("_")[0]}'):
        port_table(grid_case('uniform'), (('ez', (mid,), kind, mid, 0),),
                   nu=False, boundary=CurlBoundary(), dtype=np.float32)


def test_recorded_port_vi_uses_channel_names():
    from rfx.measurement.port_records import recorded_port_vi
    live = ((2, 2, 1), (2, 2, 2), (2, 2, 3))
    o = owner(grid_case('uniform'), 'uniform', 'ez', live, CurlBoundary())
    pm = SimpleNamespace(owner=replace(o, channels=tuple(reversed(o.channels))),
                         component='ez', live_cells=live, mid=live[1])
    e = np.array([[2., 3.], [5., 7.], [11., 13.]])
    h = np.zeros((2, 2, 2, 2))
    v, vp, i = recorded_port_vi(pm, e, h, h, h)
    np.testing.assert_allclose(v, -.0013*e[1], rtol=1e-12)
    np.testing.assert_allclose(vp, -.0013*e.sum(axis=0), rtol=1e-12)
    np.testing.assert_array_equal(i, 0.)


def test_ringdown_layout_rejects_node_outside_local_neighbourhood():
    from rfx.ringdown import _port_layout
    live = ((2, 2, 2),)
    o = owner(grid_case('uniform'), 'uniform', 'ez', live, CurlBoundary())
    channels = list(o.channels)
    n = replace(channels[1].nodes[0], i=4)
    channels[1] = replace(channels[1], nodes=(n,))
    o = replace(o, channels=tuple(channels))
    pm = SimpleNamespace(owner=o, component='ez', live_cells=live, mid=live[0])
    col = {(n.component, (n.i, n.j, n.k)): i
           for c in o.channels for i, n in enumerate(c.nodes)}
    with pytest.raises(ValueError, match=r'port:0:mode:0.*hy.*4, 2, 2'):
        _port_layout(pm, col)

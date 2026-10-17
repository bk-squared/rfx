"""Binary pinned-sheet layout switch: declared-layout bytes and compile reuse."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.model.occupancy import with_graded_edge_keep

DX = .001
DZ = np.linspace(.0008, .0012, 10)
BINS = np.array([2e9, 3e9], dtype=np.float32)
FULL = ((3, 7), (3, 7))
LAYOUTS = {
    'ones': (FULL,),
    'zeros': (),
    'rectangle': (((4, 6), (4, 6)),),
    'union': (((3, 4), (3, 7)), ((6, 7), (3, 7))),
}


def board(rectangles=(FULL,), *, graded=True, source_on_sheet=False):
    sim = Simulation(freq_max=10e9, domain=(.010, .010, float(DZ.sum())),
                     dx=DX, boundary='cpml', cpml_layers=2,
                     **({'dz_profile': DZ} if graded else {}))
    for ir, jr in rectangles:
        sim.add_pinned_sheet(plane_index=4, i_range=ir, j_range=jr, name='patch')
    z = float(DZ[:4].sum())
    if source_on_sheet:
        sim.add_source((.0045, .004, z), 'ex', amplitude_kind='current')
    else:
        sim.add_port((.002, .005, float(DZ[:2].sum())), 'ez', impedance=50,
                     waveform=GaussianPulse(f0=3e9, bandwidth=.9))
    sim.add_probe((.0045, .004, z), 'ex')
    sim.add_probe((.008, .005, float(DZ[:3].sum())), 'ez')
    return sim


def edge_arrays(rectangles):
    """Expected binary edges from the declared rectangles, without production helpers."""
    x = np.zeros((4, 5), np.float32)
    y = np.zeros((5, 4), np.float32)
    for (i0, i1), (j0, j1) in rectangles:
        x[i0-3:i1-3, j0-3:j1-3+1] = 1
        y[i0-3:i1-3+1, j0-3:j1-3] = 1
    return x, y


def observe(sim, pair=None, **kwargs):
    result = sim.forward(n_steps=60, skip_preflight=True, port_s11_freqs=BINS,
                         pinned_sheet_layout=None if pair is None else (pair,), **kwargs)
    return result.time_series, result.s_params


def assert_bytes(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        a, e = np.asarray(a), np.asarray(e)
        assert np.isfinite(a).all() and np.isfinite(e).all()
        assert a.shape == e.shape and a.dtype == e.dtype
        print(f'bytes shape={a.shape}: differing={np.count_nonzero(a.view(np.uint8) != e.view(np.uint8))}, '
              f'negative zeros actual={np.count_nonzero((a.real == 0) & np.signbit(a.real))}, '
              f'expected={np.count_nonzero((e.real == 0) & np.signbit(e.real))}')
        assert a.tobytes() == e.tobytes()


@pytest.mark.parametrize('name', LAYOUTS)
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_declared_layout_bytes(name, compiled):
    rectangles = LAYOUTS[name]
    switched, declared = board(), board(rectangles)
    pair = tuple(jnp.asarray(a) for a in edge_arrays(rectangles))
    def dynamic(m):
        return observe(switched, m)

    def fixed():
        return observe(declared)
    actual = (jax.jit(dynamic) if compiled else dynamic)(pair)
    expected = (jax.jit(fixed) if compiled else fixed)()
    assert_bytes(actual, expected)


def test_two_layouts_one_backend_compile():
    sim = board()
    fn = jax.jit(lambda m: observe(sim, m))
    # Lowering performs declaration setup; only the one executable compile is counted.
    ones = tuple(jnp.asarray(a) for a in edge_arrays(LAYOUTS['ones']))
    zeros = tuple(jnp.asarray(a) for a in edge_arrays(LAYOUTS['zeros']))
    lowered = fn.lower(ones)
    events = []
    counting = [True]

    def listener(event, duration, **kwargs):
        if counting[0] and event == '/jax/core/compile/backend_compile_duration':
            events.append(event)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        executable = lowered.compile()
        first = jax.block_until_ready(executable(ones))
        second = jax.block_until_ready(executable(zeros))
    finally:
        counting[0] = False
    assert len(events) == 1
    assert np.asarray(first[0]).tobytes() != np.asarray(second[0]).tobytes()
    assert_bytes(second, jax.jit(lambda: observe(board(())))())


def test_nonbinary_admission_names_sheet_and_edge():
    x, y = edge_arrays(())
    x[1, 2] = .5
    with pytest.raises(ValueError, match=r"sheet 0 \('patch'\): edge x\(1, 2\).*exactly 0 or 1"):
        observe(board(), (x, y))


def test_layout_gradient_refused():
    sim = board()
    pair = tuple(jnp.asarray(a) for a in edge_arrays(()))
    with pytest.raises(TypeError, match='pinned_sheet_layout is a layout switch.*physical sensitivity'):
        jax.grad(lambda m: jnp.sum(jnp.abs(observe(sim, m)[1]) ** 2))(pair)


def test_reports_and_original_declaration():
    sim = board()
    original = tuple(sim._pinned_sheets)
    result = sim.forward(n_steps=2, skip_preflight=True,
                         pinned_sheet_layout=(edge_arrays(()),))
    assert result.sheet_reports_skipped == tuple((0, report) for report in (
        '_with_sheet_spans', 'sheet_preflight', 'refuse_dead_soft_sources',
        'line_stub_admission', 'warn_sheet_planes_inside_dielectric'))
    assert tuple(sim._pinned_sheets) == original
    assert sim.forward(n_steps=2, skip_preflight=True,
                       pinned_sheet_layout=(None,)).sheet_reports_skipped is None


def test_source_on_addressed_edge_refused():
    with pytest.raises(ValueError, match='source or port drives addressed edge.*graded lane'):
        board(source_on_sheet=True).forward(n_steps=2, skip_preflight=True,
                                            pinned_sheet_layout=(edge_arrays(()),))


@pytest.mark.parametrize('graded,distributed', [(False, False), (False, True), (True, True)])
def test_other_forward_lanes_refuse(graded, distributed):
    with pytest.raises(NotImplementedError, match='single-device graded lane'):
        board(graded=graded).forward(n_steps=2, skip_preflight=True,
                                    distributed=distributed,
                                    pinned_sheet_layout=(edge_arrays(()),))


def test_run_argument_not_reachable():
    assert 'pinned_sheet_layout' not in inspect.signature(Simulation.run).parameters
    with pytest.raises(TypeError, match='pinned_sheet_layout'):
        board().run(n_steps=2, pinned_sheet_layout=(edge_arrays(()),))


def test_cell_keep_remains_on_released_layout_edges():
    from rfx.model.occupancy import admit_sheet_layout
    sim = board()
    grid = sim._build_nonuniform_grid()
    materials = sim._assemble_materials_nu(grid, pec_sheets=[], pec_wires=[])[0]
    occ = jnp.full(grid.shape, .5)
    layout = admit_sheet_layout(sim._pinned_sheets, (edge_arrays(()),))
    result = with_graded_edge_keep(materials, occ, layout, grid=grid, sources=[],
                                   waveguide_ports=[], dtype=jnp.float32,
                                   sheet_edge_masks=None, periodic=(False,) * 3)
    pads = grid.axis_pads
    # Four incident cells each retain 1/2: an addressed free edge retains 1/16.
    for axis in (0, 1):
        assert float(result.edge_keep[axis][4+pads[0], 4+pads[1], 4+pads[2]]) == .0625


@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_same_simulation_then_zero_layout_bytes(compiled):
    sim = board()
    # Populate diagnostic caches as well as executing the fixed-sheet solve.
    sim.realized_geometry()
    cached = {name: sim.__dict__.get(name) for name in (
        '_pf_campaign_ctx', '_realized_geometry_record', '_coax_geometry', '_mesh_resolution')}

    def fixed():
        return observe(sim)

    def switched(pair):
        return observe(sim, pair)

    def absent():
        return observe(board(()))

    jax.block_until_ready((jax.jit(fixed) if compiled else fixed)())
    zeros = tuple(jnp.asarray(a) for a in edge_arrays(()))
    actual = (jax.jit(switched) if compiled else switched)(zeros)
    expected = (jax.jit(absent) if compiled else absent)()
    assert_bytes(actual, expected)
    assert len(sim._pinned_sheets) == 1
    for name, value in cached.items():
        assert sim.__dict__.get(name) is value


@pytest.mark.parametrize('lane', ['run_adi', 'run_subgridded'])
def test_pinned_sheet_not_reachable_on_adi_or_subgrid(lane):
    from rfx.runners._admission import admit, refused
    sim = board(graded=False)
    # These production lane gates reject the declaration itself before stepping.
    assert ('_pinned_sheets', 'pec_sheet') in refused(sim, lane)
    with pytest.raises(NotImplementedError, match='pinned sheet'):
        admit(sim, lane)


def seam_board(ranges, names):
    sim = board(())
    for ir, name in zip(ranges, names, strict=True):
        sim.add_pinned_sheet(plane_index=4, i_range=ir, j_range=(3, 7), name=name)
    # An additional probe inside the addressed part of the overlap case.
    sim.add_probe((.0065, .004, float(DZ[:4].sum())), 'ex')
    return sim


def sheet_pair_values(ir, value):
    # Inclusive nodes [i0,i1] x [3,7]: x edges (di,5), y edges (di+1,4).
    di = ir[1] - ir[0]
    return (jnp.full((di, 5), value, dtype=jnp.float32),
            jnp.full((di + 1, 4), value, dtype=jnp.float32))


def seam_observe(sim, layout):
    result = sim.forward(n_steps=60, skip_preflight=True, port_s11_freqs=BINS,
                         pinned_sheet_layout=layout)
    return result.time_series, result.s_params


def assert_seam_message(call, *, index, name):
    # Nodes i=5 and i=6 are adjacent. Their cross edges start at i=5,
    # on each common j node 3..7, plane k=4; CPML adds 2 on every axis.
    hand_built_seams = {('ex', (5 + 2, j + 2, 4 + 2)) for j in range(3, 8)}
    component, edge = sorted(hand_built_seams)[0]
    with pytest.raises(ValueError) as error:
        call()
    message = str(error.value)
    assert f"sheet {index} ({name!r})" in message
    assert f"seam edge {component}{edge}" in message
    for remedy in ('overlap the addressed sheet with its neighbour by one node',
                   'leave at least one empty node between them',
                   'declare both as one addressed sheet'):
        assert remedy in message


@pytest.mark.parametrize('value', [0., 1.])
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_seam_fixed_addressed_refused(value, compiled):
    # Original index 1 is addressed; fixed index 0 starts one node after it.
    sim = seam_board(((6, 8), (3, 5)), ('fixed', 'design'))
    pair = sheet_pair_values((3, 5), value)

    def run(m):
        return seam_observe(sim, (None, m))

    assert_seam_message(lambda: (jax.jit(run) if compiled else run)(pair),
                        index=1, name='design')


@pytest.mark.parametrize('value', [0., 1.])
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_seam_two_addressed_refused(value, compiled):
    sim = seam_board(((3, 5), (6, 8)), ('left', 'right'))
    layout = (sheet_pair_values((3, 5), value), sheet_pair_values((6, 8), value))

    def run(m):
        return seam_observe(sim, m)

    assert_seam_message(lambda: (jax.jit(run) if compiled else run)(layout),
                        index=0, name='left')


@pytest.mark.parametrize('value', [0., 1.], ids=['zeros', 'ones'])
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_overlap_declared_layout_bytes(value, compiled):
    # Fixed x edges start at 3,4; addressed x edges at 5,6,7. No cross
    # edge is missing. The shared i=5 y-edge line remains fixed metal.
    switched = seam_board(((3, 5), (5, 8)), ('fixed', 'design'))
    declared = (seam_board(((3, 5), (5, 8)), ('fixed', 'design')) if value else
                seam_board(((3, 5),), ('fixed',)))
    pair = sheet_pair_values((5, 8), value)

    def run(m):
        return seam_observe(switched, (None, m))

    def fixed():
        return seam_observe(declared, None)

    actual = (jax.jit(run) if compiled else run)(pair)
    expected = (jax.jit(fixed) if compiled else fixed)()
    assert_bytes(actual, expected)


@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_one_empty_node_between_sheets_accepted(compiled):
    # i=6 is empty: no axis-adjacent node pair connects i=5 to i=7.
    sim = seam_board(((3, 5), (7, 8)), ('design', 'fixed'))
    pair = sheet_pair_values((3, 5), 1.)

    def run(m):
        return seam_observe(sim, (m, None))

    result = (jax.jit(run) if compiled else run)(pair)
    assert all(np.isfinite(np.asarray(a)).all() for a in result)

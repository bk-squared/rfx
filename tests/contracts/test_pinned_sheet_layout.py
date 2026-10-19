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


_COMPILE_COUNTER = {'on': False, 'n': 0}
_LISTENING = []


def _compile_listener(event, duration, **kwargs):
    if _COMPILE_COUNTER['on'] and event == '/jax/core/compile/backend_compile_duration':
        _COMPILE_COUNTER['n'] += 1


def _compiles(fn, *args):
    if not _LISTENING:
        jax.monitoring.register_event_duration_secs_listener(_compile_listener)
        _LISTENING.append(True)
    _COMPILE_COUNTER.update(on=True, n=0)
    try:
        result = jax.block_until_ready(fn(*args))
    finally:
        _COMPILE_COUNTER['on'] = False
    return result, _COMPILE_COUNTER['n']


@pytest.mark.parametrize('count', [2, 3])
def test_two_layouts_one_backend_compile(count):
    sim = board()
    names = ('ones', 'zeros', 'rectangle')[:count]
    pairs = [tuple(jnp.asarray(a) for a in edge_arrays(LAYOUTS[name])) for name in names]
    def forward_wrapper(m):
        return observe(sim, m)
    # Initialize trace-time declaration primitives independently of test order.
    # This traces without compiling a solve or calling the measured user jit.
    jax.make_jaxpr(forward_wrapper)(pairs[0])
    fn = jax.jit(forward_wrapper)
    assert fn._cache_size() == 0
    measured = [_compiles(fn, pair) for pair in pairs]
    counts = [n for _, n in measured]
    print(f'user jit backend compiles across {count} layouts: {counts}')
    assert counts == [1] + [0] * (count - 1)
    assert sum(counts) == 1
    assert fn._cache_size() == 1  # The user jit was called, not just lowered/compiled.
    for (actual, _), name in zip(measured, names):
        expected = jax.jit(lambda: observe(board(LAYOUTS[name])))()
        assert_bytes(actual, expected)
    assert np.asarray(measured[0][0][0]).tobytes() != np.asarray(measured[1][0][0]).tobytes()


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
        'sheet span report', 'sheet preflight advisories', 'dead soft-source check',
        'line-stub admission', 'sheet-in-dielectric warning'))
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


@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_length_one_normal_refused(compiled):
    from rfx.boundaries.spec import BoundarySpec
    sim = Simulation(10e9, (.010, .010, .001), dx=DX, cpml_layers=2,
                     boundary=BoundarySpec(x='cpml', y='cpml', z='periodic'),
                     dx_profile=np.array([.001, .0009, .0011, .001, .0009,
                                          .0011, .001, .0009, .0011, .001]))
    sim.add_pinned_sheet(plane_index=0, i_range=(3, 7), j_range=(3, 7), name='patch')
    sim.add_source((.002, .005, 0.), 'ez', amplitude_kind='current')
    sim.add_probe((.005, .005, 0.), 'ez')
    def run(m):
        return sim.forward(n_steps=2, skip_preflight=True, pinned_sheet_layout=(m,)).time_series
    with pytest.raises(ValueError, match=r"sheet 0 \('patch'\).*length-1 normal PEC stage.*ez\(5, 5, 0\)"):
        (jax.jit(run) if compiled else run)(edge_arrays(LAYOUTS['ones']))


@pytest.mark.parametrize('kind', ['lumped', 'wire'])
@pytest.mark.parametrize('excite', [False, True], ids=['passive', 'excited'])
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_port_release_refused(kind, excite, compiled):
    sim = board()
    if excite:
        sim._ports.clear()  # One driven port per record; the addressed port is that drive.
    # Lumped physical Ex centre (4.5,5,plane4) mm -> padded edge (6,7,6).
    # Wire starts at node 4 and spans edges 4,5,6 in the sheet plane.
    sim.add_port((.0045 if kind == 'lumped' else .004, .005, float(DZ[:4].sum())),
                 'ex', impedance=50, excite=excite,
                 **({'extent': .003} if kind == 'wire' else {}))
    def run(m):
        return observe(sim, m)
    with pytest.raises(ValueError, match=rf"sheet 0 \('patch'\).*{kind} port-release stage.*ex\(6, 7, 6\)"):
        (jax.jit(run) if compiled else run)(edge_arrays(LAYOUTS['ones']))


@pytest.mark.parametrize('coverage', ['cover', 'one_node'])
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_finite_impedance_overlap_refused(coverage, compiled):
    from rfx import Box
    sim = board()
    z = float(DZ[:4].sum())
    # Cover: nodes 2..9. One-node overlap: f0 nodes x=7..9, y=3..7;
    # shared Ey edges at x=7, starting y=3..6, first padded (9,5,6).
    lo, hi = ((.002, .002, z), (.009, .009, z)) if coverage == 'cover' else (
        (.007, .003, z), (.009, .007, z))
    sim.add_thin_conductor(Box(lo, hi), sigma_bulk=1e5,
                           thickness=35e-6, surface_impedance_f0=3e9)
    edge = r'ex\(5, 5, 6\)' if coverage == 'cover' else r'ey\(9, 5, 6\)'
    def run(m):
        return observe(sim, m)
    with pytest.raises(ValueError, match=rf"sheet 0 \('patch'\).*finite-impedance sheet operator.*{edge}"):
        (jax.jit(run) if compiled else run)(edge_arrays(LAYOUTS['ones']))


def oriented_board(normal, rectangles=(FULL,), *, periodic=False):
    from rfx.boundaries.spec import BoundarySpec
    sim = Simulation(10e9, (.010, .010, float(DZ.sum())), dx=DX, cpml_layers=2,
                     dz_profile=DZ, boundary=(BoundarySpec(x='periodic', y='cpml', z='cpml')
                                              if periodic else 'cpml'))
    for ir, jr in rectangles:
        sim.add_pinned_sheet(normal_axis=normal, plane_index=4, i_range=ir, j_range=jr, name='patch')
    sim.add_port((.002, .002, float(DZ[:2].sum())), 'ez', impedance=50,
                 waveform=GaussianPulse(f0=3e9, bandwidth=.9))
    tangents = [a for a in range(3) if a != normal]
    pos = [.005, .005, float(DZ[:5].sum())]
    pos[normal] = .004 if normal != 2 else float(DZ[:4].sum())
    sim.add_probe(tuple(pos), 'e' + 'xyz'[tangents[0]])
    sim.add_probe((0., .0045, float(DZ[:4].sum())), 'ey')
    sim.add_probe((.0095, .005, float(DZ[:4].sum())), 'ex')
    return sim


def range_pair(ir, jr, value):
    di, dj = ir[1]-ir[0], jr[1]-jr[0]
    return (np.full((di, dj+1), value, np.float32), np.full((di+1, dj), value, np.float32))


@pytest.mark.parametrize('normal', [0, 1])
@pytest.mark.parametrize('value', [0., 1.])
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_normal_xy_layout_bytes(normal, value, compiled):
    switched = oriented_board(normal)
    declared = oriented_board(normal, (FULL,) if value else ())
    run = lambda m: observe(switched, m)
    fixed = lambda: observe(declared)
    actual = (jax.jit(run) if compiled else run)(range_pair(*FULL, value))
    assert_bytes(actual, (jax.jit(fixed) if compiled else fixed)())


@pytest.mark.parametrize('value', [0., 1.])
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
def test_periodic_virtual_node_bytes(value, compiled):
    rect = ((6, 10), (3, 7))
    switched = oriented_board(2, (rect,), periodic=True)
    declared = oriented_board(2, (rect,) if value else (), periodic=True)
    run = lambda m: observe(switched, m)
    fixed = lambda: observe(declared)
    assert_bytes((jax.jit(run) if compiled else run)(range_pair(*rect, value)),
                 (jax.jit(fixed) if compiled else fixed)())


@pytest.mark.parametrize('axis', [0, 1], ids=['x', 'y'])
@pytest.mark.parametrize('high', [False, True], ids=['addressed-low', 'addressed-high'])
@pytest.mark.parametrize('periodic', [False, True])
def test_seam_orientations_refused(axis, high, periodic):
    low = ((3, 5), (3, 7)) if axis == 0 else ((3, 7), (3, 5))
    upper = ((6, 8), (3, 7)) if axis == 0 else ((3, 7), (6, 8))
    fixed, addressed = (low, upper) if high else (upper, low)
    sim = oriented_board(2, (fixed, addressed), periodic=periodic)
    pair = range_pair(*addressed, 1.)
    # Cross edge starts at node 5 on the split axis and 3 on the other.
    edge = (5 if axis == 0 else 3, 3 if axis == 0 else 5, 4)
    pads = (0, 2, 2) if periodic else (2, 2, 2)
    padded = tuple(a+b for a, b in zip(edge, pads))
    with pytest.raises(ValueError) as error:
        seam_observe(sim, (None, pair))
    assert "sheet 1 ('patch')" in str(error.value)
    assert f"seam edge e{'xy'[axis]}{padded}" in str(error.value)


def test_periodic_endpoint_disagreement_refused():
    sim = oriented_board(2, (((0, 10), (3, 7)),), periodic=True)
    x, y = range_pair((0, 10), (3, 7), 0.)
    y[-1, 1] = 1
    with pytest.raises(ValueError, match=r"sheet 0 \('patch'\).*node-0/node-N.*ey\(0, 6, 6\)"):
        observe(sim, (x, y))


def test_periodic_traced_endpoint_product():
    # At x=0==10 the two Ey rows multiply: either row being 1 zeros the edge.
    from rfx.model.occupancy import admit_sheet_layout
    sim = oriented_board(2, (((0, 10), (3, 7)),), periodic=True)
    grid = sim._build_nonuniform_grid()
    materials = sim._assemble_materials_nu(grid, pec_sheets=[], pec_wires=[])[0]
    def keep(pair):
        layout = admit_sheet_layout(sim._pinned_sheets, (pair,))
        return with_graded_edge_keep(materials, None, layout, grid=grid, sources=[],
                                    waveguide_ports=[], dtype=jnp.float32, sheet_edge_masks=None,
                                    periodic=(True, False, False)).edge_keep[1]
    x, y = range_pair((0, 10), (3, 7), 0.)
    y[-1, 1] = 1
    actual = np.asarray(jax.jit(keep)((x, y)))
    expected = np.ones(grid.shape, np.float32)
    expected[0, 6, 6] = 0
    assert actual.tobytes() == expected.tobytes()
    # Also exercise the full forward under jit with the differing traced rows.
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.jit(lambda m: observe(sim, m))((x, y)))


def test_traced_cells_layout_bytes_and_seam_refusal():
    pair = edge_arrays(LAYOUTS['ones'])
    sim = board()
    cells = np.zeros(sim._build_nonuniform_grid().shape, bool)
    actual = jax.jit(lambda c, m: observe(sim, m, pec_mask_override=c))(cells, pair)
    expected = jax.jit(lambda c: observe(board(), pec_mask_override=c))(cells)
    assert_bytes(actual, expected)
    adjacent = seam_board(((3, 5), (6, 8)), ('fixed', 'design'))
    # Even an all-true traced cell mask cannot prove that the seam is covered.
    with pytest.raises(ValueError, match=r"sheet 1 \('design'\).*seam edge ex\(7, 5, 6\)"):
        jax.jit(lambda c: adjacent.forward(n_steps=2, skip_preflight=True,
                    pinned_sheet_layout=(None, sheet_pair_values((6, 8), 1.)),
                    pec_mask_override=c).time_series)(np.ones(cells.shape, bool))


@pytest.mark.parametrize('stage', ['series', 'parallel_inductor', 'current_moments', 'msl_geometry'])
def test_other_metal_consumers_refused(stage):
    sim = board()
    position = (.0045, .005, float(DZ[:4].sum()))
    if stage in ('series', 'parallel_inductor'):
        sim.add_lumped_rlc(position, 'ex', R=50., L=1e-9,
                           topology='series' if stage == 'series' else 'parallel')
        message, edge = 'lumped RLC post-keep solve', 'ex(6, 7, 6)'
    elif stage == 'current_moments':
        sim.add_current_moment_monitor((.001, .001, .001), (.009, .009, .008),
                                       .002, BINS)
        message, edge = 'current-moment metal census', 'ex(5, 5, 6)'
    else:
        sim._ports.clear()
        sim.add_msl_port((.005, .005, 0), width=.002, height=float(DZ[:4].sum()),
                         direction='+x', eps_r_sub=1., excite=False)
        # Feed x=5 reads trace edge x=4; width .002 centred at y=5
        # starts at y=4, on plane 4. Pads (2,2,2) give (6,6,6).
        message, edge = 'MSL geometry-edge admission', 'ex(6, 6, 6)'
    with pytest.raises(ValueError) as error:
        sim.forward(n_steps=2, skip_preflight=True,
                    pinned_sheet_layout=(edge_arrays(LAYOUTS['ones']),))
    assert "sheet 0 ('patch')" in str(error.value)
    assert message in str(error.value)
    assert edge in str(error.value)


@pytest.mark.parametrize('kind', ['lumped', 'wire', 'waveguide', 'microstrip'])
def test_all_release_record_kinds_refused(kind):
    from types import SimpleNamespace
    from rfx.model.conductors import refuse_layout_port_releases
    from rfx.model.occupancy import admit_sheet_layout
    sim = board()
    grid = sim._build_nonuniform_grid()
    # Hand-declared Ex edge from node (4,5,4), with CPML pads (2,2,2).
    edge = (6, 7, 6)
    root = SimpleNamespace(grid=grid, released_port_edges=(('p', 'ex', edge),),
                           _released_port_details=(('p', kind, 'ex', edge, ()),))
    layout = admit_sheet_layout(sim._pinned_sheets, (edge_arrays(()),))
    with pytest.raises(ValueError, match=rf"{kind} port-release stage.*ex\(6, 7, 6\)"):
        refuse_layout_port_releases(layout, root)


@pytest.mark.parametrize('value', [0., 1.])
def test_tfsf_layout_refused(value):
    # The drawn sheet under an open plane wave on the graded mesh is refused by
    # the TF/SF invariance admission (#1816), which reads only the static metal;
    # the switched sheet must not slip past it at any layout value.
    sim = board((FULL,))
    sim._ports.clear()
    sim.add_tfsf_source(f0=3e9, margin=3, direction='+x', polarization='ez')
    with pytest.raises(ValueError, match=r"sheet 0 \('patch'\): the TF/SF invariance admission"):
        sim.forward(n_steps=4, skip_preflight=True,
                    pinned_sheet_layout=(edge_arrays((FULL,) if value else ()),))


@pytest.mark.parametrize('stage', ['parallel_rc', 'wire_away'])
@pytest.mark.parametrize('value', [0., 1.])
def test_other_stages_layout_bytes(stage, value):
    def build(rectangles):
        sim = board(rectangles)
        if stage == 'parallel_rc':
            sim.add_lumped_rlc((.0045, .005, float(DZ[:4].sum())), 'ex',
                               R=50., C=1e-13, topology='parallel')
        else:
            sim._ports.clear()
            sim.add_port((.002, .005, float(DZ[:2].sum())), 'ez',
                         extent=.001, impedance=50.,
                         waveform=GaussianPulse(f0=3e9, bandwidth=.9))
            sim.add_port((.008, .008, float(DZ[:2].sum())), 'ez',
                         extent=.002, impedance=50., excite=False)
        return sim

    def run(sim, pair=None):
        result = sim.forward(n_steps=60, skip_preflight=True,
                             pinned_sheet_layout=None if pair is None else (pair,),
                             port_s11_freqs=BINS)
        # A port record contains only its driven column.
        return result.time_series, result.s_params[:, :1, :]
    assert_bytes(run(build((FULL,)), edge_arrays((FULL,) if value else ())),
                 run(build((FULL,) if value else ())))


def test_waveguide_forward_release_refused():
    # Full PEC-walled aperture: no y/z CPML pads. Feed x=10mm on 2mm
    # cells is node 5; eight x CPML cells place it at x=13. Ey edges
    # on the declared yz sheet start at (13,3,1), inside the aperture.
    sim = Simulation(12e9, (.05, 11 * .002, 5 * .002),  dx=.002, cpml_layers=8,  # solved length; #1887
                     boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'},
                     dx_profile=np.array([.002]*8 + [.0019, .0021]*4 + [.002]*9))
    sim.add_waveguide_port(x_position=.01, direction='+x',
                           y_range=(0., 11 * .002), z_range=(0., 5 * .002), n_modes=1)
    sim.add_pinned_sheet(normal_axis=0, plane_index=5, i_range=(3, 7),
                         j_range=(1, 3), name='aperture')
    with pytest.raises(ValueError, match=r"sheet 0 \('aperture'\).*waveguide port-release stage.*ey\(13, 3, 1\)"):
        sim.forward(n_steps=2, skip_preflight=True,
                    pinned_sheet_layout=(range_pair((3, 7), (1, 3), 1.),))

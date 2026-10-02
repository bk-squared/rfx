"""A 1 mm load at a periodic seam occupies 1 mm; a sheet at L is at zero.

An interval retains its final cell, while a plane identifies the two rims.
The off-centre source and probe measure the field with the declared load.
"""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec


def model(dx=.001):
    return Simulation(20e9, (.010, .006, .004), dx=dx, cpml_layers=0,
                      boundary=BoundarySpec(x='periodic', y='pec', z='pec'))


def loaded_trace(dx, *, point_loads=False, entry='run'):
    sim = model(dx)
    sim.add_source((.007, .002, .002), 'ex', amplitude_kind='field')
    sim.add_probe((.005, .003, .002), 'ex')
    if point_loads:
        count = round(.001 / dx)
        for n in range(count):
            sim.add_port((.009 + n * dx, .003, .002), component='ex',
                         impedance=50. / count, excite=False)
    else:
        sim.add_port((.009, .003, .002), component='ex', extent=.001,
                     impedance=50., excite=False)
    kwargs = {'compute_s_params': False} if entry == 'run' else {}
    return np.asarray(getattr(sim, entry)(n_steps=256, **kwargs).time_series)


def _check_loaded_trace(actual, reference):
    assert np.max(np.abs(reference)) > 1e-3
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize('dx', [.001, .0005])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_wire_ending_at_period_loads_only_last_edges(dx, entry):
    # Independent public point loads name each edge directly. Their series
    # impedances sum to 50 ohm; they never resolve an interval endpoint.
    _check_loaded_trace(loaded_trace(dx, entry=entry),
                        loaded_trace(dx, point_loads=True, entry=entry))


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_interval_endpoint_is_exclusive_on_each_axis(axis):
    from rfx.grid import Grid
    grid = Grid(20e9, (.010, .006, .004), dx=.001, cpml_layers=0,
                periodic_axes='xyz')
    lo, hi = [.002]*3, [.002]*3
    lo[axis], hi[axis] = grid.domain[axis]-.001, grid.domain[axis]
    a, b = grid.interval_to_indices(lo, hi)
    assert (a[axis], b[axis]) == (grid.shape[axis]-1, grid.shape[axis])
    assert grid.interval_to_indices(hi, lo) == (b, a)


def test_filament_ending_at_period_owns_last_edge_under_jit():
    import jax
    from rfx import PolylineWire
    sim = model()
    sim.add(PolylineWire([(.009, .003, .002), (.010, .003, .002)], radius=0.),
            material='pec')
    sim.add_source((.007, .002, .002), 'ex', amplitude_kind='field')
    sim.add_probe((.005, .003, .002), 'ex')
    sim.run(n_steps=8, compute_s_params=False)
    grid = sim._build_grid()
    def masks():
        wires = []
        sim._assemble_materials(grid, pec_wires=wires)
        return wires[0].edges
    expected = np.zeros(grid.shape, dtype=bool)
    expected[9, 3, 2] = True
    for realized in (masks(), jax.jit(masks)()):
        np.testing.assert_array_equal(realized[0], expected)
        assert not np.asarray(realized[1]).any()
        assert not np.asarray(realized[2]).any()


@pytest.mark.parametrize('jit', [False, True])
def test_positive_subcell_filament_radius_at_period_refuses(jit):
    import jax
    from rfx import PolylineWire
    sim = model()
    sim.add(PolylineWire([(.009, .003, .002), (.010, .003, .002)], radius=.0001),
            material='pec')
    grid = sim._build_grid()

    def assemble():
        return sim._assemble_materials(grid)

    with pytest.raises(ValueError, match='resolve the wire as a volume'):
        (jax.jit(assemble) if jit else assemble)()


def test_microstrip_plane_and_substrate_endpoint_at_period():
    from rfx.grid import Grid
    from rfx.sources.msl_port import MSLPort, _msl_yz_cells, msl_h_plane_stencil
    grid = Grid(20e9, (.010, .006, .004), dx=.001, cpml_layers=0,
                periodic_axes='xyz')
    def port(feed):
        return MSLPort(feed_x=feed, y_lo=.004, y_hi=.006, z_lo=.003,
                       z_hi=.004, direction='+x', impedance=50.)
    # Width includes both nodes: 4, 5, 6 mm with 6 mm identified with zero.
    # The 3--4 mm substrate height contains one E edge, not two nodes.
    assert _msl_yz_cells(grid, port(.010)) == _msl_yz_cells(grid, port(0.)) == [(0, 4, 3), (0, 5, 3), (0, 0, 3)]
    assert msl_h_plane_stencil(grid, port(.010), .010) == msl_h_plane_stencil(grid, port(0.), 0.)


@pytest.mark.parametrize('kind', ['pec', 'surface_impedance', 'dc'])
def test_thin_sheet_at_period_has_zero_plane_edges(kind):
    from rfx.boundaries.pec import realized_pec_edge_masks
    footprints = []
    for x in (0., .010):
        sim = model()
        options = {'surface_impedance_f0': 10e9} if kind == 'surface_impedance' else {}
        if kind == 'dc':
            options = {'sigma_bulk': 1e4}
        sim.add_thin_conductor(Box((x, .001, .001), (x, .005, .003)),
                               thickness=35e-6, **options)
        sim.add_source((.003, .002, .002), 'ez', amplitude_kind='field')
        sim.add_probe((.005, .003, .002), 'ez')
        sim.run(n_steps=8, compute_s_params=False)
        sheets, impedance = [], []
        materials = sim._assemble_materials(sim._build_grid(), pec_sheets=sheets, sheet_specs=impedance)[0]
        if kind == 'pec':
            footprints.append(realized_pec_edge_masks(None, sheets, periodic=(True, False, False)))
        elif kind == 'surface_impedance':
            footprints.append((impedance[0].mask,))
        else:
            footprints.append((materials.sigma,))
    assert sum(np.asarray(m).sum() for m in footprints[0]) > 0
    for a, b in zip(*footprints):
        np.testing.assert_array_equal(a, b)


def _sheet(x):
    sim = model()
    sim.add(Box((x, .001, .001), (x, .005, .003)), material='pec')
    sim.add_source((.003, .002, .002), 'ez', amplitude_kind='field')
    sim.add_probe((.005, .003, .002), 'ez')
    trace = np.asarray(sim.run(n_steps=32, compute_s_params=False).time_series)
    sheets = []
    sim._assemble_materials(sim._build_grid(), pec_sheets=sheets)
    from rfx.boundaries.pec import realized_pec_edge_masks
    edges = realized_pec_edge_masks(None, sheets, (), periodic=(True, False, False))
    return trace, sheets[0].plane, edges


def _check_sheet(actual, reference):
    assert actual[1] == reference[1] == 0
    assert sum(np.asarray(m).sum() for m in reference[2]) > 0
    for a, b in zip(actual[2], reference[2]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[0], reference[0])


def test_sheet_at_period_is_the_zero_plane():
    _check_sheet(_sheet(.010), _sheet(0.))


def _check_sheet_extent(actual, expected):
    assert any(np.asarray(mask).any() for mask in expected)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('start', [.001, .009])
@pytest.mark.parametrize('kind', ['box', 'thin', 'surface_impedance', 'pinned'])
def test_sheet_tangential_interval_keeps_last_edge_without_wrapping_its_start(start, kind):
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx
    sim = model()
    shape = Box((start, .001, .002), (.010, .005, .002))
    if kind == 'box':
        sim.add(shape, material='pec')
    elif kind == 'pinned':
        sim.add_pinned_sheet(normal_axis=2, plane_index=2,
                             i_range=(round(start/.001), 10), j_range=(1, 5))
    else:
        options = {'surface_impedance_f0': 10e9} if kind == 'surface_impedance' else {}
        sim.add_thin_conductor(shape, **options)
    sim.add_source((.007, .002, .001), 'ex', amplitude_kind='field')
    sim.add_probe((.005, .003, .001), 'ex')
    sim.run(n_steps=8, compute_s_params=False)
    grid = sim._build_grid()
    sheets, impedance = [], []
    sim._assemble_materials(grid, pec_sheets=sheets, sheet_specs=impedance)
    if kind == 'surface_impedance':
        ctx = build_sheet_impedance_ctx(impedance, periodic=(True, False, False))
        actual = (ctx.mask_ex, ctx.mask_ey, ctx.mask_ez)
    else:
        actual = realized_pec_edge_masks(None, sheets, periodic=(True, False, False))
    expected = [np.zeros(grid.shape, dtype=bool) for _ in range(3)]
    i = round(start/.001)
    expected[0][i:10, 1:6, 2] = True
    expected[1][i:10, 1:5, 2] = True
    expected[1][0, 1:5, 2] = True  # transverse edges on endpoint plane L = 0
    _check_sheet_extent(actual, expected)


def _box_material():
    sim = model()
    sim.add_material('dielectric', eps_r=4.)
    sim.add(Box((.009, .001, .001), (.010, .005, .003)), material='dielectric')
    sim.add_source((.007, .002, .002), 'ex', amplitude_kind='field')
    sim.add_probe((.005, .003, .002), 'ex')
    sim.run(n_steps=8, compute_s_params=False)
    return np.asarray(sim._assemble_materials(sim._build_grid())[0].eps_r)


def _check_box(eps):
    expected = np.ones((10, 7, 5), dtype=np.float32)
    expected[9, 1:5, 1:3] = 4.
    np.testing.assert_array_equal(eps, expected)


def test_box_ending_at_period_keeps_last_cell():
    _check_box(_box_material())


def test_traced_volume_bounds_preserve_sampling_and_refuse_distant_images():
    import jax
    import jax.numpy as jnp
    from rfx import Sphere
    def material(radius):
        sim = model()
        sim.add_material('diel', eps_r=4.)
        sim.add(Sphere((.005, .003, .002), radius), material='diel')
        return sim._assemble_materials(sim._build_grid())[0].eps_r
    radius = jnp.float32(.0013)
    np.testing.assert_array_equal(jax.jit(material)(radius), material(radius))
    # A fill can cross the seam; traced bounds use the fixed [-L,2L] window.
    np.testing.assert_array_equal(jax.jit(material)(jnp.float32(.006)),
                                  material(jnp.float32(.006)))
    with pytest.raises(Exception, match="axis 'x'.*bounds.*L=0.01"):
        jax.jit(material)(jnp.float32(.016)).block_until_ready()


def test_forward_design_box_resolves_before_existing_seam_refusal():
    import jax.numpy as jnp
    sim = model()
    sim.add_source((.007, .002, .002), 'ex', amplitude_kind='field')
    sim.add_probe((.005, .003, .002), 'ex')
    bounds = ((.009, .001, .001), (.010, .003, .002))
    grid = sim._build_grid()
    assert sim._design_box_bounds_from_corners(grid, bounds) == (9, 10, 1, 4, 1, 3)
    # The localized adjoint already refuses a periodic-face box because
    # its write window cannot wrap. It must see the intended final cell.
    with pytest.raises(NotImplementedError, match=r'PERIODIC x face.*cells \[9, 10\)'):
        sim.forward(n_steps=8, design_box=bounds,
                    design_eps_override=jnp.full((1, 3, 2), 4.), skip_preflight=True)


@pytest.mark.parametrize('kind', ['wire', 'flux'])
@pytest.mark.parametrize('lo,hi', [(.009, .011), (-.001, .001)])
def test_crossing_interval_is_refused(kind, lo, hi):
    sim = model()
    if kind == 'wire':
        sim.add_port((lo, .003, .002), component='ex', extent=hi-lo,
                     impedance=50., excite=False)
    else:
        sim.add_flux_monitor(axis='y', coordinate=.003, freqs=np.array([10e9]),
                             size=(hi-lo, .002), center=((hi+lo)/2, .002))
    with pytest.raises(ValueError, match="axis 'x'.*interval.*L=0.01"):
        sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)


@pytest.mark.parametrize('x', [.015, -.007])
@pytest.mark.parametrize('kind', ['source', 'probe'])
def test_out_of_period_point_is_refused(x, kind):
    sim = model()
    getattr(sim, 'add_' + kind)((x, .002, .002), 'ez')
    with pytest.raises(ValueError, match='outside the grid shape'):
        sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)


@pytest.mark.parametrize('n', [9, 10])
def test_point_admission_is_half_open(n):
    from rfx.grid import Grid
    g = Grid(20e9, (n * .001, .006, .004), dx=.001, cpml_layers=0, periodic_axes='x')
    assert g.index_of('x', -.0005) == n - 1
    assert g.index_of('x', n * .001 - .0005) == n - 1
    for x in (-.0005001, n * .001 + .0005):
        with pytest.raises(ValueError, match='outside this axis'):
            g.index_of('x', x)
        with pytest.raises(ValueError, match='outside the grid shape'):
            g.position_to_index((x, .002, .002))


@pytest.mark.parametrize('check,args', [
    ('_check_loaded_trace', (np.zeros(4), np.ones(4))),
    ('_check_box', (np.ones((10, 7, 5)),)),
    ('_check_sheet', ((np.zeros(4), 1, (np.ones(4),)*3),
                    (np.zeros(4), 0, (np.ones(4),)*3))),
    ('_check_sheet_extent', ((np.zeros(4),)*3, (np.ones(4),)*3)),
])
def test_seam_checks_reject_wrong_measurements(check, args):
    with pytest.raises(AssertionError):
        globals()[check](*args)

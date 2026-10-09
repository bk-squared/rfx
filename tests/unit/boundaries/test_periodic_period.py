"""A vacuum ring repeats every declared L; its seam is an ordinary cell.

The TEM resonance is c/L up to second-order Yee dispersion. Probes and flux
planes at L sample the same physical plane as at zero, and a constant
broadside plane wave has no dependence on its transverse period.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, Box
from rfx.boundaries.spec import BoundarySpec
from rfx.core.yee import init_state
from rfx.grid import Grid
from rfx.probes.probes import flux_spectrum, update_flux_monitor
from rfx.runners.uniform import build_flux_monitor_cfgs
from tests.unit.boundaries.test_boundary_mini_battery import nearest, resonance


def _check_ring(frequencies):
    target = 299792458.0 / .024
    errors = []
    for dx, frequency in zip((.001, .0005, .00025), frequencies):
        errors.append(abs(frequency / target - 1))
        # From the 1-D Yee dispersion relation, with the 3-D Courant dt.
        dt = .99 * dx / (299792458.0 * np.sqrt(3))
        yee_frequency = np.arcsin(299792458.0 * dt / dx * np.sin(np.pi * dx / .024)) / (np.pi * dt)
        assert frequency == pytest.approx(yee_frequency, rel=2e-5)
    assert errors[0] < .003  # Yee dispersion is 0.193% at 24 cells/period.
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def _check_index(g, axis):
    n = 'xyz'.index(axis)
    length = g.domain[n]
    assert g.shape[n] == round(length / .001)
    assert getattr(g, f'pad_{axis}_lo') == getattr(g, f'pad_{axis}_hi') == 0
    assert g.index_of(axis, length) == 0
    assert 0 <= g.index_of(axis, length - .0005) < g.shape[n]
    assert g.index_of(axis, -.0005) == g.shape[n] - 1
    assert g.index_of(axis, length - .0005) == g.shape[n] - 1
    with pytest.raises(ValueError, match='outside this axis'):
        g.index_of(axis, -.001)
    position = [0., 0., 0.]
    position[n] = length
    assert g.position_to_index(position)[n] == 0
    assert sum(g.cells(axis)) == pytest.approx(length, abs=1e-14)


def _check_seam(trace, a, b):
    assert np.max(np.abs(trace[:, 0])) > 1e-6
    np.testing.assert_array_equal(trace[:, 0], trace[:, 1])
    assert np.max(np.abs(a)) > 0
    np.testing.assert_array_equal(a, b)


def _check_flux(measured):
    assert float(measured[0]) == pytest.approx((2 * 15 + 2) * 1e-6, rel=2e-6)


def _check_broadside(traces):
    assert np.max(np.abs(traces[0])) > 1e-3
    np.testing.assert_allclose(traces[0], traces[1], rtol=2e-5, atol=2e-6)


def _check_snap(g):
    assert g.dx == pytest.approx(.0103 / 11)
    assert g.dx <= .001
    assert g.nx == 11


def _check_induced_wrap(grid, actual, legacy):
    assert grid.ny == 28 and grid.pad_y_lo == grid.pad_y_hi == 8
    assert grid.nz == (1 if grid.is_2d else 21)
    np.testing.assert_array_equal(actual, legacy)


def _check_material(epsilon):
    assert epsilon == pytest.approx(5.)


def _check_half_node(position_index, axis_index, expected_position, expected_axis):
    assert position_index == expected_position
    assert axis_index == expected_axis


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_nonperiodic_half_node_takes_the_lower_node(dtype, axis):
    # Both take lower ties, with their original division arithmetic:
    # position_to_index keeps the scalar dtype; index_of divides float64.
    # float32(.00075) is above the tie when promoted before division.
    grid = Grid(60e9, (.009, .006, .0042), dx=.0003, cpml_layers=0)
    position = [0., 0., 0.]
    position[axis] = dtype(.00075)
    _check_half_node(grid.position_to_index(position)[axis],
                     grid.index_of(axis, position[axis]), 2, 3 if dtype is np.float32 else 2)


@pytest.mark.parametrize('mode', ['2d_tmz', '2d_tez'])
def test_invariant_axis_does_not_evaluate_its_coordinate(mode):
    grid = Grid(60e9, (.009, .006, .0042), dx=.0003, cpml_layers=0, mode=mode)
    assert grid.position_to_index((np.float32(.00075), 0.)) == (2, 0, 0)
    assert grid.index_of('z', object()) == 0


@pytest.mark.parametrize('entry', ['position_to_index', 'index_of'])
def test_nonperiodic_traced_coordinate_keeps_concretization_refusal(entry):
    grid = Grid(60e9, (.009, .006, .0042), dx=.0003, cpml_layers=0)
    lookup = (lambda x: grid.position_to_index((x, 0., 0.))[0]) if entry == 'position_to_index' else (
        lambda x: grid.index_of(0, x))
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.make_jaxpr(lookup)(jnp.float32(.00075))


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_ring_frequency_converges_at_second_order(entry):
    _check_ring([nearest(resonance('ring', entry, dx), 299792458.0 / .024, (8e9, 16e9))
                 for dx in (.001, .0005, .00025)])


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_periodic_index_wrap_and_full_cell_measure(axis):
    _check_index(Grid(20e9, (.024, .020, .016), dx=.001, periodic_axes=axis), axis)
    boundaries = {a: 'periodic' if a == axis else 'pec' for a in 'xyz'}
    sim = Simulation(20e9, (.024, .020, .016), dx=.001,
                     boundary=BoundarySpec(**boundaries))
    box = sim.fidelity_report(print_report=False)[0]['axes']['xyz'.index(axis)]
    assert box['n_cells'] == round(sim._domain['xyz'.index(axis)] / .001)
    assert box['realized_extent_um'] == pytest.approx(sim._domain['xyz'.index(axis)] * 1e6)


def test_period_snap_and_explicit_refusal():
    # lambda/20 requests 1 mm; 10.3 mm is deliberately noncommensurate.
    with pytest.warns(UserWarning, match='snapped'):
        g = Grid(299792458.0 / .020, (.0103, .008, .006), periodic_axes='x')
    _check_snap(g)
    with pytest.raises(ValueError, match='axis x: L=0.0103 m, dx=0.001 m.*nearest dividing dx=0.00103 m'):
        Grid(20e9, (.0103, .008, .006), dx=.001, periodic_axes='x')
    with pytest.raises(ValueError, match='common.*axis x.*axis y.*nearest common dividing dx=0.0001 m'):
        Grid(20e9, (.0103, .008, .006), dx=.001, periodic_axes='xy')
    # Their 0.1 mm common spacing is below the automatic cost bound.
    with pytest.raises(ValueError, match='common.*nearest common dividing dx=0.0001 m'):
        Grid(299792458.0 / .020, (.0103, .008, .006), periodic_axes='xy')
    common = Grid(20e9, (.0103, .008, .006), dx=.0001, periodic_axes='xy')
    assert common.shape[:2] == (103, 80)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_seam_probe_and_dft_plane(entry):
    sim = Simulation(20e9, (.024, .003, .002), dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(x='periodic', y='periodic', z='pec'))
    sim.add_source((.021, .001, .001), 'ez', amplitude_kind='field')
    for x in (0., .024):
        sim.add_probe((x, .001, .001), 'ez')
        sim.add_dft_plane_probe(axis='x', coordinate=x, component='ez', freqs=jnp.array([12.5e9]), name=str(x))
        if entry == 'run':
            sim.add_flux_monitor(axis='x', coordinate=x, freqs=jnp.array([12.5e9]), name=str(x))
    result = getattr(sim, entry)(n_steps=256, skip_preflight=True)
    trace = np.asarray(result.time_series)
    a, b = (np.asarray(result.dft_planes[str(x)].accumulator) for x in (0., .024))
    _check_seam(trace, a, b)
    if entry == 'run':
        lo, hi = (result.flux_monitors[str(x)] for x in (0., .024))
        assert np.max(np.abs(np.asarray(lo.e2_dft))) > 0
        assert np.max(np.abs(np.asarray(lo.h1_dft))) > 0
        np.testing.assert_array_equal(flux_spectrum(lo), flux_spectrum(hi))


@pytest.mark.parametrize('finite', [False, True])
def test_seam_flux_counts_every_periodic_cell(finite):
    sim = Simulation(20e9, (.024, .003, .005), dx=.001, cpml_layers=0,
                     boundary=BoundarySpec.uniform('periodic'))
    for x in (0., .024):
        sim.add_flux_monitor(axis='x', coordinate=x, freqs=jnp.array([0.]),
                             size=(.003, .005) if finite else None, name=str(x))
    grid = sim._build_grid()
    monitors = build_flux_monitor_cfgs(sim, grid, 1)
    state = init_state(grid.shape)
    # A 2 V/m, 1 A/m plane carries 2 W/m² through exactly 15 mm².
    # Deliberately mark the last cell: losing it is visible in the flux.
    ey = jnp.full(grid.shape, 2.).at[:, -1, -1].set(4.)
    state = state._replace(ey=ey, hz=jnp.ones(grid.shape))
    for monitor in monitors:
        assert monitor.index == 0
        measured = flux_spectrum(update_flux_monitor(monitor, state, 1.))
        _check_flux(measured)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_broadside_field_is_independent_of_transverse_period(entry):
    traces = []
    for ly, lz in ((.003, .005), (.007, .004)):
        sim = Simulation(20e9, (.020, ly, lz), dx=.001, cpml_layers=8,
                         boundary=BoundarySpec(x='cpml', y='periodic', z='periodic'))
        sim.add_tfsf_source(f0=10e9, margin=3)
        sim.add_probe((.011, .001, .002), 'ez')
        result = getattr(sim, entry)(n_steps=256, skip_preflight=True)
        traces.append(np.asarray(result.time_series))
        jax.clear_caches()
    _check_broadside(traces)


@pytest.mark.parametrize('mode', ['2d_tmz', '2d_tez'])
def test_invariant_axis_is_not_a_period(mode):
    g = Grid(20e9, (.0113, .0092, .00073), dx=.001, mode=mode, periodic_axes='z')
    assert g.nz == 1 and g.periodic_axes == ''
    assert g.dx == .001 and g.index_of('z', 100.) == 0


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_public_auto_resolution_snaps_and_explicit_resolution_refuses(entry):
    boundary = BoundarySpec(x='periodic', y='pec', z='pec')
    sim = Simulation(299792458.0 / .020, (.0103, .008, .006), boundary=boundary, cpml_layers=0)
    sim.add_source((.002, .003, .003), 'ez')
    sim.add_probe((.007, .004, .003), 'ez')
    with pytest.warns(UserWarning, match='snapped'):
        getattr(sim, entry)(n_steps=4, skip_preflight=True)
    bad = Simulation(20e9, (.0103, .008, .006), dx=.001, boundary=boundary, cpml_layers=0)
    with pytest.raises(ValueError, match='nearest dividing dx'):
        getattr(bad, entry)(n_steps=4, skip_preflight=True)


def test_material_averaging_uses_last_physical_cell_at_the_seam():
    from rfx.core.yee import component_e_materials
    sim = Simulation(20e9, (.010, .006, .004), dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(x='periodic', y='pec', z='pec'))
    sim.add_material('seam', eps_r=9.)
    sim.add(Box((.009, 0., 0.), (.010, .006, .004)), material='seam')
    grid = sim._build_grid()
    materials = sim._build_materials(grid)[0]
    eps, _ = component_e_materials(materials, sim._periodic_flags())
    # Ez at x=0 lies between eps=1 at x=0 and eps=9 at x=L-dx.
    _check_material(float(eps[2][0, 3, 2]))


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('mode', ['2d_tmz', '2d_tez', '3d'])
def test_tfsf_induced_wrap_keeps_legacy_box(mode, entry, monkeypatch):
    if mode != '3d':
        # The old implicit ring included both eight-cell pads and its bounding
        # node. Declare that 28-cell ring and translate the finite object/probe.
        sim = Simulation(20e9, (.020, .028, .004), dx=.001, cpml_layers=8,
                         mode=mode, boundary={'x': 'cpml', 'y': 'periodic', 'z': 'periodic'})
        component = 'ey' if mode == '2d_tez' else 'ez'
        sim.add_material('glass', eps_r=3.2)
        sim.add(Box((.007, .010, 0.), (.012, .014, .002)), material='glass')
        sim.add_tfsf_source(f0=10e9, margin=3, polarization=component)
        sim.add_probe((.015, .016, .001), component)
        actual = getattr(sim, entry)(n_steps=256, skip_preflight=True).time_series
        declared_grid = Grid(20e9, (.020, .028, .004), dx=.001, cpml_layers=8,
                             mode=mode, cpml_axes='x', periodic_axes='yz')
        monkeypatch.setattr(sim, '_build_grid', lambda **kwargs: declared_grid)
        legacy = getattr(sim, entry)(n_steps=256, skip_preflight=True).time_series
        np.testing.assert_array_equal(actual, legacy)
        return
    # The original body touches z_lo, so main extrudes it through that pad.
    # Declare the padded ring, including the extruded glass below the old face.
    sim = Simulation(20e9, (.020, .028, .021), dx=.001, cpml_layers=8, mode=mode,
                     boundary={'x': 'cpml', 'y': 'periodic', 'z': 'periodic'})
    sim.add_material('glass', eps_r=3.2)
    sim.add(Box((.007, .010, 0.), (.012, .014, .010)), material='glass')
    sim.add_tfsf_source(f0=10e9, margin=3, polarization='ez')
    sim.add_probe((.015, .016, .009), 'ez')
    actual = getattr(sim, entry)(n_steps=256, skip_preflight=True).time_series
    declared_grid = Grid(20e9, (.020, .028, .021), dx=.001, cpml_layers=8,
                         mode=mode, cpml_axes='x', periodic_axes='yz')
    monkeypatch.setattr(sim, '_build_grid', lambda **kwargs: declared_grid)
    legacy = getattr(sim, entry)(n_steps=256, skip_preflight=True).time_series
    np.testing.assert_array_equal(actual, legacy)


@pytest.mark.parametrize('judge', ['ring', 'index', 'seam', 'flux', 'broadside', 'snap', 'induced_wrap', 'material', 'half_node'])
def test_period_judges_reject_corrupted_measurements(judge):
    # These deliberately wrong observations keep each physical judge live.
    # Replacing a judge by a no-op must fail this control. The separate
    # fence-post mutation exercises the real solver with every helper kept.
    with pytest.raises(AssertionError):
        if judge == 'ring':
            _check_ring([11.970415e9, 12.230779e9, 12.361121e9])
        elif judge == 'index':
            g = Grid(20e9, (.024, .020, .016), dx=.001, cpml_layers=0)
            _check_index(g, 'x')
        elif judge == 'seam':
            _check_seam(np.array([[1., .99]]), np.ones(1), np.ones(1))
        elif judge == 'flux':
            _check_flux([28e-6])  # final 4 W/m² cell omitted
        elif judge == 'broadside':
            _check_broadside([np.ones(4), np.full(4, .99)])
        elif judge == 'snap':
            g = Grid(20e9, (.0103, .008, .006), dx=.001, cpml_layers=0)
            _check_snap(g)
        elif judge == 'induced_wrap':
            g = Grid(20e9, (.020, .011, .004), dx=.001, cpml_layers=8)
            _check_induced_wrap(g, np.ones(4), np.zeros(4))
        elif judge == 'material':
            _check_material(1.)
        else:
            _check_half_node(3, 3, 2, 2)

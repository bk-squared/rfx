"""A 50 ohm microstrip load includes the width's two endpoint nodes.

The 4--6 mm width on a 6 mm period has three parallel columns, at 4, 5,
and 0 mm. A full-period width has six unique columns on the 1 mm mesh.
These finite cavity records compare loads, not settled S-parameters.
"""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec


def microstrip_model(*, full_period=False, point_loads=False, mode='uniform', short_trace=False,
                     field_source=True, end=.006):
    sim = Simulation(20e9, (.010, .006, .004), dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(x='pec', y='periodic', z='pec'))
    sim.add(Box((0., 0., .001), (.010, .006, .001)), material='pec')
    lo, hi = (0., .006) if full_period else (.004, end)
    sim.add(Box((.001, lo, .002), (.009, .005 if short_trace else hi, .002)), material='pec')
    if point_loads:
        # Named physical nodes, independent of the aperture resolver. Parallel
        # 150 ohm (or six 300 ohm) resistors have total resistance 50 ohm.
        widths = (0, 1, 2, 3, 4, 5) if full_period else (4, 5, 0)
        for j in widths:
            sim.add_port((.002, j*.001, .001), component='ez',
                         impedance=50.*len(widths), excite=False)
    else:
        sim.add_msl_port((.002, (lo+hi)/2, .001), width=hi-lo, height=.001,
                         direction='+x', impedance=50., excite=False, mode=mode,
                         n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    if field_source:
        sim.add_source((.007, .003, .001), 'ez', amplitude_kind='field')
    sim.add_probe((.005, .002, .001), 'ez')
    return sim


def _check_microstrip_trace(actual, reference):
    peak = np.max(np.abs(reference))
    assert peak > .01
    # The same edge resistances permit only float32 evaluation residual.
    np.testing.assert_allclose(actual, reference, rtol=0,
                               atol=8*np.finfo(reference.dtype).eps*peak)


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('full_period', [False, True])
def test_periodic_microstrip_matches_parallel_point_loads(entry, full_period):
    traces = []
    for point_loads in (False, True):
        sim = microstrip_model(full_period=full_period, point_loads=point_loads)
        kwargs = {'compute_s_params': False} if entry == 'run' else {}
        traces.append(np.asarray(getattr(sim, entry)(n_steps=256, **kwargs).time_series))
    _check_microstrip_trace(*traces)


def test_microstrip_trace_check_rejects_missing_load_response():
    reference = np.array([.02, .1, -.08], dtype=np.float32)
    with pytest.raises(AssertionError):
        _check_microstrip_trace(np.zeros_like(reference), reference)


def _check_width_cells(actual, expected):
    assert actual == expected


@pytest.mark.parametrize('repeat', [False, True])
def test_width_count_check_rejects_missing_or_repeated_node(repeat):
    expected = [(2, j, 1) for j in (0, 1, 2, 3, 4, 5)]
    corrupted = expected + [expected[0]] if repeat else expected[:-1]
    with pytest.raises(AssertionError):
        _check_width_cells(corrupted, expected)


@pytest.mark.parametrize('direction', ['+x', '-x', '+y', '-y'])
@pytest.mark.parametrize('full_period', [False, True])
def test_width_nodes_are_unique_and_height_is_exclusive(direction, full_period):
    from rfx.grid import Grid
    from rfx.sources.msl_port import MSLPort, _msl_yz_cells
    grid = Grid(20e9, (.006, .006, .004), dx=.001, cpml_layers=0,
                periodic_axes='xy')
    port = MSLPort(feed_x=.002, y_lo=0. if full_period else .004, y_hi=.006,
                   z_lo=.001, z_hi=.003, direction=direction, impedance=50.)
    widths = (0, 1, 2, 3, 4, 5) if full_period else (4, 5, 0)
    expected = [(2, j, k) if direction[-1] == 'x' else (j, 2, k)
                for j in widths for k in (1, 2)]
    _check_width_cells(_msl_yz_cells(grid, port), expected)


def test_width_larger_than_period_is_refused():
    from rfx.grid import Grid
    from rfx.sources.msl_port import MSLPort, _msl_yz_cells
    grid = Grid(20e9, (.010, .006, .004), dx=.001, cpml_layers=0, periodic_axes='y')
    port = MSLPort(feed_x=.002, y_lo=0., y_hi=.007, z_lo=.001,
                   z_hi=.002, direction='+x', impedance=50.)
    with pytest.raises(ValueError, match="periodic axis 'y'.*interval.*L=0.006"):
        _msl_yz_cells(grid, port)


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('mode', ['laplace', 'eigenmode'])
def test_periodic_microstrip_mode_aperture_is_refused(entry, mode):
    sim = microstrip_model(mode=mode)
    kwargs = {'compute_s_params': False} if entry == 'run' else {}
    error, message = ((NotImplementedError, 'eigenmode') if mode == 'eigenmode'
                      else (ValueError, 'MSL mode/current aperture.*seam'))
    with pytest.raises(error, match=message):
        getattr(sim, entry)(n_steps=8, **kwargs)


@pytest.mark.parametrize('operation', ['s_matrix', 'mixed_s_matrix', 'plane_probes'])
def test_periodic_microstrip_current_aperture_is_refused(operation):
    from rfx.probes.msl_wave_decomp import register_msl_plane_probes
    sim = microstrip_model(field_source=False)
    if operation == 'mixed_s_matrix':
        sim.add_port((.007, .003, .001), component='ez', impedance=50.)
    with pytest.raises(ValueError, match='MSL mode/current aperture.*seam'):
        if operation == 's_matrix':
            sim.compute_msl_s_matrix(n_steps=8, freqs=np.array([10e9]))
        elif operation == 'mixed_s_matrix':
            sim.compute_mixed_s_matrix(n_steps=8, freqs=np.array([10e9]))
        else:
            register_msl_plane_probes(sim, port_index=0, freqs=np.array([10e9]))


@pytest.mark.parametrize('short_trace,end', [(True, .006), (False, .0057)])
def test_seam_column_requires_a_trace_conductor(short_trace, end):
    # End the three-column trace at 5 mm. The port's
    # zero-image column now has no upper conductor and must be refused.
    # A width ending at 5.7 mm also rounds to node 6 == 0, while its
    # conductor footprint ends at node 5. Check the selected load node.
    sim = microstrip_model(short_trace=short_trace, end=end)
    with pytest.raises(ValueError, match='trace'):
        sim.run(n_steps=8, compute_s_params=False)


def test_seam_aperture_names_only_incident_conductors():
    from rfx.geometry.port_termination import default_msl_terminates
    sim = microstrip_model()
    # This separate strip lies between the minimum and maximum image indices,
    # but outside the port's three physical columns (4, 5, 0 mm).
    sim.add(Box((.001, .0015, .002), (.009, .0025, .002)), material='pec')
    refs = default_msl_terminates(sim, sim._build_grid(), position=(.002, .005, .001),
                                  width=.002, height=.001, direction='+x')
    assert len(refs) == 1 and refs[0].entry is sim._geometry[1]


def test_laplace_fringe_reaching_the_seam_is_refused():
    from rfx.grid import Grid
    from rfx.sources.msl_port import MSLPort, compute_msl_mode_profile
    grid = Grid(20e9, (.010, .006, .004), dx=.001, cpml_layers=0, periodic_axes='y')
    port = MSLPort(feed_x=.002, y_lo=.003, y_hi=.005, z_lo=.001,
                   z_hi=.002, direction='+x', impedance=50.)
    with pytest.raises(ValueError, match='MSL Laplace aperture.*seam'):
        compute_msl_mode_profile(grid, port, 1., pad_y_cells=1)


@pytest.mark.parametrize('side', ['lo', 'hi'])
@pytest.mark.parametrize('consumer', ['short', 'line', 'resistor', 'voltage', 'voltage_jnp'])
def test_coaxial_node_aperture_at_periodic_seam_is_refused(side, consumer):
    from rfx.grid import Grid
    from rfx.sources import coaxial_port as coax
    grid = Grid(20e9, (.010, .006, .004), dx=.001, cpml_layers=0, periodic_axes='y')
    kwargs = dict(center_xy=(.004, .001 if side == 'lo' else .005), outer_radius=.001)
    with pytest.raises(ValueError, match="Coaxial node-inclusive aperture.*'y'.*seam"):
        if consumer == 'short':
            coax.stamp_coaxial_short_plane(grid, None, z_index=2, **kwargs)
        elif consumer == 'line':
            kwargs['outer_radius'] = .0005
            coax.stamp_coaxial_line(grid, None, z_lo_index=1, z_hi_index=2,
                                    shell_thickness_m=.0005, **kwargs)
        elif consumer == 'resistor':
            coax.stamp_coaxial_annular_resistor(grid, None, z_index=2, target_impedance=50., **kwargs)
        elif consumer == 'voltage':
            coax.coaxial_line_plane_voltage(grid, None, None, **kwargs)
        else:
            coax.coaxial_line_plane_voltage_jnp(grid, None, **kwargs)

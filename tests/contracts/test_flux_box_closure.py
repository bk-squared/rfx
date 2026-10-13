"""Closed-box power from field DFTs and independently integrated real nodes."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rfx import Simulation, GaussianPulse, make_port_source, run
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.probes import flux_spectrum

FREQS = np.array([8e9, 10e9, 12e9])
STEPS = 768
# Every C1 box is asymmetric about the source on every axis, including
# the half-cell displacement of the z-directed source edge.
OFFSETS = (((-3, 5), (-4, 3), (-3, 5)),
           ((-4, 3), (-3, 5), (-5, 3)),
           ((-5, 4), (-5, 4), (-4, 6)))


def _record(name, values):
    directory = os.environ.get('RFX_FLUX_CONTRACT_OUTPUT')
    if directory:
        Path(directory, name + '.json').write_text(json.dumps(values, indent=2) + '\n')


def _axis_integral_weights(x, lo, hi, nodal, full=False):
    # Integrate the linear interpolant on each interval, assigning half the
    # interval to each bounding node. No production area/weight helper is read.
    stop = hi-1 if full else hi
    lengths = np.diff(x[lo:stop+1])
    if not nodal:
        return np.r_[lengths, 0.] if full else lengths
    weights = np.zeros(stop-lo+1)
    weights[:-1] += lengths/2
    weights[1:] += lengths/2
    return weights


def _field_integral(mon, coords):
    axes = [a for a in range(3) if a != mon.axis]
    electric_axes = ((1, 2), (2, 0), (0, 1))[mon.axis]
    terms = []
    for e, h, component in zip((mon.e1_dft, mon.e2_dft),
                                (mon.h2_dft, mon.h1_dft), electric_axes):
        weights = [_axis_integral_weights(coords[a], lo, hi, component != a,
                                          full=hi == len(coords[a]))
                   for a, lo, hi in zip(axes, (mon.lo1, mon.lo2), (mon.hi1, mon.hi2))]
        area = weights[0][:, None] * weights[1][None, :]
        if e.shape[-2:] != area.shape:
            return None  # baseline layout witness, never accepted by the test
        product = np.asarray(e, dtype=np.complex128) * np.conj(np.asarray(h, dtype=np.complex128))
        terms.append(np.sum(product * area, axis=(-2, -1)))
    return np.real(terms[0]-terms[1])


def measure_scene(graded, model, steps=STEPS):
    half, mirror = model == 'half', model != 'asymmetric'
    profiles = [np.r_[1., np.linspace(.8, 1.2, 22), 1.]*.001,
                np.full(12 if half else 24, .001),
                np.r_[1., np.linspace(1.2, .8, 22), 1.]*.001]
    if not graded:
        profiles = [np.full(len(p), .001) for p in profiles]
    sim = Simulation(freq_max=20e9, domain=(.024, .012 if half else .024, .024),
                     dx=.001, cpml_layers=4,
                     boundary=BoundarySpec(x='cpml', y=Boundary('pmc', 'cpml') if half else 'cpml', z='cpml'),
                     **({f'd{a}_profile': p for a, p in zip('xyz', profiles)} if graded else {}))
    grid = sim._build_realized_grid()
    # Coordinates from the declared cell profiles, including explicit CPML
    # padding. These do not call the production flux builder or geometry spine.
    coords = []
    for a, p in zip('xyz', profiles):
        low, high = getattr(grid, f'pad_{a}_lo'), getattr(grid, f'pad_{a}_hi')
        cells = np.r_[np.full(low, p[0]), p, np.full(high, p[-1])]
        coords.append(np.r_[0., np.cumsum(cells)] - low*p[0])
    source_node = np.array([10, 0 if half else (12 if mirror else 11), 13])
    position = tuple(float(np.r_[0., np.cumsum(p)][i]) for p, i in zip(profiles, source_node))
    pulse = GaussianPulse(f0=10e9, bandwidth=.8)
    sim.add_port(position, 'ez', impedance=50., waveform=pulse)
    order = []
    for box, offsets in enumerate(OFFSETS):
        indices = source_node[:, None] + np.array(offsets)
        if mirror:
            # Explicitly symmetric in y only for comparison with the reflected
            # half model. x and z remain asymmetric in every box.
            reach = (3, 4, 5)[box]
            indices[1] = (0, reach) if half else (12-reach, 12+reach)
        endpoints = np.array([[np.r_[0., np.cumsum(p)][i] for i in pair]
                              for p, pair in zip(profiles, indices)])
        for axis in range(3):
            tang = [a for a in range(3) if a != axis]
            for side, sign in ((0, -1), (1, 1)):
                name = f'b{box}_{axis}_{side}'
                sim.add_flux_monitor(axis='xyz'[axis], coordinate=float(endpoints[axis, side]),
                                     freqs=FREQS, name=name,
                                     size=tuple(np.diff(endpoints, axis=1)[tang, 0]),
                                     center=tuple(endpoints.mean(axis=1)[tang]))
                order.append((box, axis, side, sign, name))
    if graded:
        from rfx.runners.nonuniform import run_nonuniform_path
        result = run_nonuniform_path(sim, n_steps=steps, s_param_freqs=FREQS,
                                     compute_s_params=True, lane='fwd_nonuniform')
        monitors = [result.flux_monitors[name] for *_, name in order]
    else:
        from rfx.runners.uniform import build_flux_monitor_cfgs
        from rfx.simulation import LumpedPortSParamSpec
        from rfx.sources.sources import LumpedPort, setup_lumped_port
        port = LumpedPort(position, 'ez', 50., pulse)
        materials = setup_lumped_port(grid, port, sim._assemble_materials(grid)[0])
        spec = LumpedPortSParamSpec(*grid.position_to_index(position), 'ez', FREQS, 50.)
        result = run(grid, materials, steps, boundary='cpml', pec_axes='',
                     sources=[make_port_source(grid, port, materials, steps)],
                     lumped_port_sparams=[spec], flux_monitors=build_flux_monitor_cfgs(sim, grid, steps))
        monitors = result.flux_monitors
    power_channels = np.asarray(result.sparam_time_records[0])
    time = np.arange(len(power_channels))
    voltage = np.exp(-2j*np.pi*FREQS[:, None]*(time+1)*grid.dt) @ power_channels[:, 0] * grid.dt
    current = np.exp(-2j*np.pi*FREQS[:, None]*(time+.5)*grid.dt) @ power_channels[:, 1] * grid.dt
    power = .5*np.real(voltage*np.conj(current))
    net, expected = np.zeros((3, 3)), np.zeros((3, 3))
    face, interior, residual = [], [], []
    for (box, axis, side, sign, name), mon in zip(order, monitors):
        actual = np.asarray(flux_spectrum(mon, exact_f64=True))
        oracle = _field_integral(mon, coords)
        net[box] += .5*sign*actual
        expected[box] += .5*sign*oracle if oracle is not None else np.nan
        if oracle is not None:
            residual.append(float(np.max(np.abs(actual-oracle))))
        if axis == 1:
            (face if side == 0 else interior).append(actual.tolist())
    values = dict(mesh='graded' if graded else 'uniform', model=model, frequencies=FREQS.tolist(),
                  flux=net.tolist(), expected=expected.tolist(), port_power=power.tolist(),
                  spread_over_peak=(np.ptp(net, axis=0)/np.max(np.abs(net))).tolist(),
                  face=face, interior=interior, quadrature_residual=residual,
                  steps=steps, dt_s=float(grid.dt),
                  port_tail_over_peak=(np.max(np.abs(power_channels[-steps//10:, :2]), axis=0)
                                       / np.max(np.abs(power_channels[:, :2]), axis=0)).tolist())
    _record(f'{values["mesh"]}-{model}-{steps}', values)
    return values


@pytest.fixture(scope='module', params=[(False, 768), (True, 768), (False, 1536), (True, 1536)],
                ids=['uniform-768', 'graded-768', 'uniform-1536', 'graded-1536'])
def scenes(request):
    graded, steps = request.param
    return {model: measure_scene(graded, model, steps) for model in ('asymmetric', 'full', 'half')}


def _independent_integral(scene):
    actual, expected = np.asarray(scene['flux']), np.asarray(scene['expected'])
    assert np.isfinite(expected).all(), 'all nodal endpoints must be accumulated'
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.)
    assert np.all(actual > 0), 'nonzero outward power in every box and bin'
    return actual


def test_c1_asymmetric_boxes(scenes):
    flux = _independent_integral(scenes['asymmetric'])
    assert np.all(np.ptp(flux, axis=0) <= 1e-4*np.max(flux)), flux
    if scenes['asymmetric']['mesh'] == 'graded':
        assert np.all(np.ptp(flux, axis=0) <= 1e-5*np.max(flux)), flux  # F2b declaration


def test_c2_magnetic_face_closed_boxes(scenes):
    full, half = (_independent_integral(scenes[m]) for m in ('full', 'half'))
    assert np.all(np.ptp(half, axis=0) <= 1e-4*np.max(half)), half
    assert np.all(np.abs(2*half-full) <= 1e-4*np.max(full)), (full, half)


def test_c3_magnetic_plane_has_zero_flux(scenes):
    half = scenes['half']
    peak = np.max(np.abs(half['interior']))
    assert peak > 0
    assert np.max(np.abs(half['face'])) <= 1e-6*peak


def test_c4_port_power(scenes):
    full, half = (_independent_integral(scenes[m]) for m in ('full', 'half'))
    p_full, p_half = (np.asarray(scenes[m]['port_power']) for m in ('full', 'half'))
    assert np.all(np.abs(p_half-2*half) <= np.abs(p_full-full) + 2e-4*np.max(p_full))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_full_plane_between_pec_walls(graded):
    profile = np.r_[1., np.linspace(.7, 1.3, 18), 1.]*.001
    sim = Simulation(freq_max=20e9, domain=(.032, .020, .006), dx=.001, cpml_layers=6,
                     boundary=BoundarySpec(x='cpml', y='pec', z='pec'),
                     **({'dy_profile': profile} if graded else {}))
    sim.add_source((.006, .007, .002), 'ez', amplitude_kind='field',
                   waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    for name, size in [('full', None), ('finite', (.020, .006))]:
        sim.add_flux_monitor(axis='x', coordinate=.021, freqs=FREQS, name=name, size=size)
    result = sim.run(n_steps=STEPS, skip_preflight=True, compute_s_params=False)
    coords = (None, np.r_[0., np.cumsum(profile if graded else np.full(20, .001))],
              np.arange(7)*.001)
    values = {}
    for name in ('full', 'finite'):
        mon = result.flux_monitors[name]
        actual = np.asarray(flux_spectrum(mon, exact_f64=True))
        expected = _field_integral(mon, coords)
        assert expected is not None
        assert np.max(np.abs(actual)) > 0
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.)
        values[name] = actual.tolist()
    np.testing.assert_allclose(values['full'], values['finite'], rtol=2e-6, atol=0.)
    _record(f'full-plane-{graded}', values)


def test_finite_periodic_window_includes_its_closing_node():
    import jax.numpy as jnp
    from rfx.core.yee import init_state
    from rfx.probes.probes import update_flux_monitor
    from rfx.runners.uniform import build_flux_monitor_cfgs
    sim = Simulation(freq_max=20e9, domain=(.004, .006, .008), dx=.001,
                     cpml_layers=0, boundary=BoundarySpec.uniform('periodic'))
    sim.add_flux_monitor(axis='x', coordinate=.002, freqs=np.array([0.]),
                         size=(.004, .003), center=(.004, .0025))
    grid = sim._build_realized_grid()
    monitor, = build_flux_monitor_cfgs(sim, grid, 1)
    state = init_state(grid.shape)
    state = state._replace(ez=jnp.broadcast_to((2+jnp.arange(6))[None, :, None], grid.shape),
                           hy=jnp.ones(grid.shape), step=jnp.asarray(1))
    actual = flux_spectrum(update_flux_monitor(monitor, state, 1.), exact_f64=True)
    # E_z on y=2,3,4,5,6 mm reads 4,5,6,7,2: the last node is the image
    # of y=0, not the final stored y=5 mm sample. Integrate over 3 mm in z.
    expected = -(.5*4+5+6+7+.5*2)*.001*.003
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.)


def test_tfsf_full_plane_covers_the_runtime_period():
    from rfx.measurement.plan import build_measurement_plan
    from rfx.measurement.flux_weights import monitor_periodic_flags
    sim = Simulation(20e9, (.032, .010, .001), dx=.001, cpml_layers=4,
                     boundary='cpml', mode='2d_tmz')
    sim.add_tfsf_source(f0=10e9, bandwidth=.8, margin=3, polarization='ez')
    for name, size in [('full', None), ('finite', (.006, .001))]:
        sim.add_flux_monitor(axis='x', coordinate=.020, freqs=FREQS,
                             name=name, size=size, center=(.005, .0005))
    grid = sim._build_realized_grid()
    assert grid.periodic_axes == ''
    result = sim.run(n_steps=512, skip_preflight=True, compute_s_params=False)
    full, finite = (result.flux_monitors[n] for n in ('full', 'finite'))
    period, window = grid.shape[1]*.001, .006
    full_flux, finite_flux = (np.asarray(flux_spectrum(m)) for m in (full, finite))
    assert np.max(np.abs(full_flux)) > 0
    np.testing.assert_allclose(full_flux/period, finite_flux/window, rtol=2e-6, atol=0.)
    for area in (full.dA, full.dA2):
        np.testing.assert_allclose(np.sum(area), period*.001, rtol=2e-7)
    plan = build_measurement_plan(sim, grid, n_steps=512)
    owner = next(o for o in plan.owners if o.id == 'flux_plane:0')
    for channel in owner.channels[:2]:
        np.testing.assert_allclose(sum(n.area for n in channel.nodes), period*.001, rtol=2e-7)
    graded = Simulation(20e9, (.032, .010, .001), dx=.001, cpml_layers=4,
                        boundary='cpml', mode='2d_tmz', dy_profile=np.full(10, .001))
    graded.add_tfsf_source(f0=10e9, margin=3)
    assert monitor_periodic_flags(graded, graded._build_realized_grid()) == (False, False, False)
    _record('tfsf-period', dict(full=full_flux.tolist(), finite=finite_flux.tolist(),
                              period=period, window=window))

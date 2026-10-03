"""B3b review witnesses: resolved image faces and high-face flux areas."""
import numpy as np
import pytest

from rfx import Simulation, GaussianPulse
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.probes import flux_spectrum
from tests.unit.boundaries.test_pmc_mirror_judges import record


@pytest.mark.parametrize('mode,component', [('2d_tez', 'ey'), ('2d_tmz', 'ez')])
@pytest.mark.parametrize('normal', ['x', 'y'])
def test_collapsed_pmc_axis_does_not_change_flux(mode, component, normal):
    flux, traces = [], []
    for pmc in (False, True):
        sim = Simulation(freq_max=15e9, domain=(.020, .020, .001), dx=.0005,
                         cpml_layers=8, mode=mode,
                         boundary=BoundarySpec(x='cpml', y='cpml', z='pmc') if pmc else 'cpml')
        sim.add_source((.009, .007, 0.), component, amplitude_kind='field',
                       waveform=GaussianPulse(f0=8e9))
        sim.add_probe((.014, .013, 0.), component)
        sim.add_flux_monitor(axis=normal, coordinate=.013, freqs=np.array([6e9, 8e9, 10e9]), name='flux')
        result = sim.run(n_steps=800, skip_preflight=True, compute_s_params=False)
        flux.append(np.asarray(flux_spectrum(result.flux_monitors['flux'], exact_f64=True)))
        traces.append(np.asarray(result.time_series))
    assert np.linalg.norm(flux[0]) > 0
    np.testing.assert_array_equal(traces[0], traces[1])
    np.testing.assert_array_equal(flux[0], flux[1])
    record(f'collapsed-{mode}-{normal}', dict(no_pmc=flux[0].tolist(), pmc=flux[1].tolist()))


@pytest.mark.parametrize('axis', range(3), ids=['x_hi', 'y_hi', 'z_hi'])
@pytest.mark.parametrize('graded', [False, True])
def test_high_face_flux_matches_full_mirror(axis, graded):
    """Rotate a y-normal flux plane through all three magnetic high faces."""
    b, c = (axis+1) % 3, (axis+2) % 3
    def rotate(values):
        out = [0., 0., 0.]
        for i, v in zip((axis, b, c), values):
            out[i] = v
        return tuple(out)
    fluxes = []
    for full in (False, True):
        profile = np.r_[1., [.9, 1.1]*5, 1.]*.001
        if full:
            profile = np.r_[profile, profile[::-1]]
        faces = {a: 'pec' for a in 'xyz'}
        faces['xyz'[axis]] = 'pec' if full else Boundary('pec', 'pmc')
        faces['xyz'[b]] = 'cpml'
        sim = Simulation(freq_max=20e9, domain=rotate((.024 if full else .012, .009, .004)),
                         dx=.001, cpml_layers=4, boundary=BoundarySpec(**faces),
                         **({f'd{"xyz"[axis]}_profile': profile} if graded else {}))
        for x in ((.008, .016) if full else (.008,)):
            sim.add_source(rotate((x, .003, .002)), 'e'+'xyz'[c], amplitude_kind='field',
                           waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        sim.add_flux_monitor(axis='xyz'[b], coordinate=.006, freqs=np.array([8e9, 10e9, 12e9]), name='flux')
        result = sim.run(n_steps=256, skip_preflight=True, compute_s_params=False)
        fluxes.append(np.asarray(flux_spectrum(result.flux_monitors['flux'], exact_f64=True)))
    relative = np.linalg.norm(2*fluxes[0]-fluxes[1])/np.linalg.norm(fluxes[1])
    record(f'high-face-flux-{axis}-{graded}', dict(relative=float(relative),
           half=fluxes[0].tolist(), full=fluxes[1].tolist()))
    assert relative <= 512*np.finfo(np.float32).eps


@pytest.mark.parametrize('mode,component', [('2d_tez', 'ey'), ('2d_tmz', 'ez')])
def test_two_d_flux_half_domain_matches_full_mirror(mode, component):
    powers = []
    for full in (False, True):
        sim = Simulation(freq_max=20e9, domain=(.024 if full else .012, .009, .001),
                         dx=.001, cpml_layers=4, mode=mode,
                         boundary=BoundarySpec(x='pec' if full else Boundary('pec', 'pmc'),
                                               y='cpml', z='pmc'))
        for x in ((.008, .016) if full else (.008,)):
            sim.add_source((x, .003, 0.), component, amplitude_kind='field',
                           waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        sim.add_flux_monitor(axis='y', coordinate=.006, freqs=np.array([8e9, 10e9, 12e9]), name='flux')
        result = sim.run(n_steps=256, skip_preflight=True, compute_s_params=False)
        powers.append(np.asarray(flux_spectrum(result.flux_monitors['flux'], exact_f64=True)))
    relative = np.linalg.norm(2*powers[0]-powers[1])/np.linalg.norm(powers[1])
    record(f'two-d-flux-mirror-{mode}', dict(relative=float(relative)))
    assert np.linalg.norm(powers[1]) > 0
    assert relative <= 512*np.finfo(np.float32).eps

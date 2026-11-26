"""Round 29: the sum of pole maxima bounds every frequency in the read band."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, LorentzPole, Simulation


MEDIA = ('census', 'probe-a', 'probe-b', 'probe-c', 'mixed', 'above-band')


def declaration(name):
    w0 = 2 * np.pi * 8e9
    strength, damping = ((.4, .15) if name in ('census', 'probe-a') else
                         (2., .1) if name == 'probe-c' else (1., .1))
    lorentz = [(w0, damping * w0, strength * w0**2)]
    debye = []
    bins = np.array([0., 20e9])
    if name == 'mixed':
        w1 = 2 * np.pi * 12e9
        lorentz.append((w1, .2 * w1, .6 * w1**2))
        debye.append((1.25, 1e-11))
    if name == 'above-band':
        bins = np.array([2e9, 4e9])
    return 4., 1.5 if name == 'mixed' else 1., lorentz, debye, bins


def numpy_bound(eps, mu, lorentz, debye, bins):
    total = abs(eps) + sum(abs(de) for de, _ in debye)
    for w0, delta, strength in lorentz:
        peak = np.sqrt(max(w0**2 - 2 * delta**2, 0.)) / (2 * np.pi)
        samples = [0., min(bins), max(bins)]
        if min(bins) <= peak <= max(bins):
            samples.append(peak)
        omega = 2 * np.pi * np.array(samples)
        total += np.max(np.abs(strength / (w0**2 - omega**2 + 2j * delta * omega)))
    return np.sqrt(abs(mu) * total)


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    jax.clear_caches()


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('medium', MEDIA)
def test_band_bound_is_strict(medium, entry):
    eps, mu, lorentz, debye, bins = declaration(medium)
    sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001, boundary='pec')
    sim.add_material('medium', eps_r=eps, mu_r=mu,
                     lorentz_poles=[LorentzPole(*p) for p in lorentz],
                     debye_poles=[DebyePole(*p) for p in debye])
    sim.add(Box((0., 0., 0.), (.008,) * 3), material='medium')
    sim.add_dft_plane_probe(axis='z', coordinate=.004, component='ez', freqs=bins, name='read')
    result = getattr(sim, entry)(n_steps=2, skip_preflight=True)
    actual = float(result.settling_echo_bound['n_max'])
    omega = 2 * np.pi * np.r_[0., np.linspace(min(bins), max(bins), 20001)]
    response = np.full(omega.shape, eps, dtype=np.complex128)
    for w0, delta, strength in lorentz:
        response += strength / (w0**2 - omega**2 + 2j * delta * omega)
    for de, tau in debye:
        response += de / (1 + 1j * omega * tau)
    dense = np.max(np.sqrt(np.abs(mu * response)))
    expected = numpy_bound(eps, mu, lorentz, debye, bins)
    assert actual >= dense, (medium, entry, actual, dense)
    before = np.nextafter(np.float32(actual), np.float32(-np.inf))
    assert abs(float(before) - expected) <= np.spacing(np.float32(expected)), (medium, entry, actual, expected)


@pytest.mark.parametrize('medium', ['lorentz', 'debye'])
@pytest.mark.parametrize('graded', [False, True])
def test_traced_pole_parameters_are_admitted(medium, graded):
    w0 = 2 * np.pi * 8e9
    bins = np.array([0., 20e9])

    def bound(scale):
        options = {'dx_profile': np.full(8, .001)} if graded else {}
        sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001, boundary='pec', **options)
        poles = ({'lorentz_poles': [LorentzPole(w0, .15 * w0, .4 * w0**2 * scale)]}
                 if medium == 'lorentz' else {'debye_poles': [DebyePole(scale, 1e-11)]})
        sim.add_material('medium', eps_r=4., **poles)
        sim.add(Box((0., 0., 0.), (.008,) * 3), material='medium')
        return sim.forward(n_steps=2, skip_preflight=True).settling_echo_bound['n_max']

    expected = numpy_bound(4., 1., [(w0, .15*w0, .4*w0**2)] if medium == 'lorentz' else [],
                           [(1., 1e-11)] if medium == 'debye' else [], bins)
    actual = float(jax.jit(bound)(jnp.float32(1.)))
    eps32 = np.finfo(np.float32).eps
    margin = (1 + eps32 * (1 + w0 / (.3 * w0) if medium == 'lorentz' else 0)) * (1 + eps32)**2
    assert expected <= actual <= expected * margin

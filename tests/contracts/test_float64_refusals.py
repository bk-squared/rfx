"""Precision combination contracts: rfx#2613 (PR A)."""
import pytest

from rfx import GaussianPulse, Simulation
from tests._x64_compat import enable_x64


def precision_scene(kind, precision):
    if kind == 'upml':
        sim = Simulation(freq_max=6e9, domain=(.04, .03, .026), dx=.0005,
                         cpml_layers=8, boundary='upml', precision=precision)
        sim.add_source((.00985, .01235, .01085), 'ez', waveform=GaussianPulse(f0=3e9))
        sim.add_probe((.03035, .01785, .01435), 'ez')
    else:
        # Existing oblique API fixture, tests/unit/api/test_api.py.
        sim = Simulation(freq_max=5e9, domain=(.12, .12, .006), dx=.004,
                         cpml_layers=8, boundary='cpml', precision=precision)
        sim.add_tfsf_source(f0=5e9, bandwidth=.2, amplitude=1., margin=3,
                            polarization='ez', angle_deg=30.)
        sim.add_probe((.04, .06, .003), 'ez')
    return sim


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('kind,words', [('upml', 'float32 UPML update'),
                                      ('oblique', 'complex64 fields')])
def test_float64_combination_refused(entry, kind, words):
    """rfx#2613: both public entries refuse before stepping, without tracker text."""
    with enable_x64():
        sim = precision_scene(kind, 'float64')
        with pytest.raises(NotImplementedError, match=words) as error:
            getattr(sim, entry)(n_steps=1)
        assert "precision='float32'" in str(error.value)
        assert '#' not in str(error.value)


@pytest.mark.parametrize('kind', ['upml', 'oblique'])
@pytest.mark.parametrize('precision', ['float32', 'mixed'])
def test_lower_precision_still_admitted(kind, precision):
    """rfx#2613: the new combination rule applies only to float64 declarations."""
    from rfx.runners._admission import admit
    with enable_x64():
        sim = precision_scene(kind, precision)
        admit(sim, 'run_uniform')
        admit(sim, 'fwd_uniform')

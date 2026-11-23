"""A closed TF/SF box admits an additive soft source in either order."""
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from tests.contracts.run_forward_census import freeze


def closed_box_with_soft_source(source_first):
    sim = Simulation(freq_max=1.5e9, domain=(.055, .055, .040), dx=.0025,
                     cpml_layers=8, snap='declared')

    def source():
        sim.add_source((.030, .026, .021), 'ez',
                       waveform=GaussianPulse(f0=.6e9, bandwidth=.95), amplitude_kind='current')

    if source_first:
        source()
    sim.add_tfsf_source(f0=.6e9, bandwidth=.95, polarization='ez', direction='+x',
                        margin=3, waveform='differentiated_gaussian', closed_box=True)
    if not source_first:
        source()
    for component in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        sim.add_probe((.022, .031, .017), component)
    sim.add_probe((.004, .031, .017), 'ez')
    return sim


@pytest.mark.parametrize('source_first', [False, True], ids=['plane-wave-first', 'source-first'])
def test_closed_box_soft_source_probe_bits(source_first):
    run = closed_box_with_soft_source(source_first).run(n_steps=300, compute_s_params=False)
    forward = closed_box_with_soft_source(source_first).forward(n_steps=300, checkpoint=False)
    a, b = np.asarray(run.time_series), np.asarray(forward.time_series)
    assert a.shape == b.shape == (300, 7) and a.dtype == b.dtype == np.float32
    assert np.isfinite(a).all() and np.all(np.max(np.abs(a), axis=0) > 0)
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize('port_first', [False, True])
@pytest.mark.parametrize('excite', [False, True])
@pytest.mark.parametrize('extent', [None, .002])
def test_plane_wave_port_refusal_preserves_declaration(port_first, excite, extent):
    sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001, cpml_layers=4)

    def port():
        sim.add_port((.011, .009, .0055), 'ez', impedance=50., excite=excite, extent=extent)

    def plane_wave():
        sim.add_tfsf_source(closed_box=True)

    (port if port_first else plane_wave)()
    before = freeze(vars(sim))
    with pytest.raises(ValueError, match='A TF/SF plane wave.*lumped or wire port'):
        (plane_wave if port_first else port)()
    assert freeze(vars(sim)) == before

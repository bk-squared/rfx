"""A PEC aperture mode cannot stand in for a magnetic half guide."""
import numpy as np
import pytest

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec


def model(graded=False, two_ports=False):
    sim = Simulation(freq_max=12e9, domain=(.040, .020, .004), dx=.001,
                     boundary=BoundarySpec(x='cpml', y=Boundary('pec', 'pmc'), z='pec'),
                     cpml_layers=4, **({'dy_profile': np.ones(20)*.001} if graded else {}))
    sim.add_waveguide_port(.010, direction='+x', mode=(1, 0), freqs=np.array([10e9]),
                           reference_plane=.014)
    if two_ports:
        sim.add_waveguide_port(.030, direction='-x', mode=(1, 0), freqs=np.array([10e9]),
                               reference_plane=.026)
    return sim


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('entry', ['run', 'forward', 'compute_waveguide_s_matrix'])
def test_pmc_waveguide_mode_solver_refuses(graded, skip, entry, monkeypatch):
    sim = model(graded, two_ports=entry == "compute_waveguide_s_matrix")
    kw = dict(n_steps=2, skip_preflight=skip)
    if entry == 'compute_waveguide_s_matrix':
        kw = dict(num_periods=1)
        if skip:
            monkeypatch.setattr(sim, '_preflight_waveguide_setup', lambda **kwargs: None)
    with pytest.raises(NotImplementedError, match='y_hi.*waveguide-port.*waveguide_port.init_waveguide_port'):
        getattr(sim, entry)(**kw)


@pytest.mark.parametrize('smoothing', [True, 'kottke_pec'])
def test_waveguide_smoothing_guard_independent_of_mode_refusal(monkeypatch, smoothing):
    # Keep the smoothing fence even if a future magnetic aperture solver
    # removes the earlier refusal. The material SDF still needs an extension.
    import rfx.boundaries.pmc as pmc
    monkeypatch.setattr(pmc, 'refuse_waveguide_pmc', lambda sim: None)
    import rfx.runners._admission as admission
    monkeypatch.setattr(admission, 'admit', lambda *args, **kwargs: None)
    sim = model(two_ports=True)
    sim.add_material('dielectric', eps_r=3.)
    sim.add(Box((.004, .005, .001), (.030, .020, .003)), material='dielectric')
    with pytest.raises(NotImplementedError, match='subpixel smoothing.*y_hi.*half a cell'):
        sim.compute_waveguide_s_matrix(num_periods=1, subpixel_smoothing=smoothing)

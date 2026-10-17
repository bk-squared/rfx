"""M1 is visible here (review B: law 2.64, spectra 1.4e-2), while scene T touches only the x absorbers."""
import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.runners import _distributed_nu_wrapped as runner
from tests._distributed_dispersive_cpml import all_e_cells, check, measure, record
from tests.unit.runners.test_distributed_edge_averaged_materials import WAVEFORM

pytestmark = [pytest.mark.distributed, pytest.mark.slow]

DX = np.array([1.0, 1.0, 0.8, 0.7, 0.7, 0.8, 1.0, 1.2, 1.2, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0,
               1.1, 1.3, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.1, 1.0]) * 1e-3
LX = float(DX.sum())


def scene(cells=True):
    """Review B scene_b2.py's full builder, including its four named probes."""
    sim = Simulation(freq_max=10e9, domain=(LX, 11e-3, 13e-3), dx=1e-3,
                     boundary='cpml', cpml_layers=6, dx_profile=DX)
    sim.add_source((7.3e-3, 5.5e-3, 6.2e-3), 'ez', waveform=WAVEFORM, amplitude_kind='current')
    for p, c in [((3.1e-3, 1.2e-3, 9.4e-3), 'ez'), ((12.2e-3, 2.3e-3, 11.6e-3), 'ex'),
                 ((LX - 1.3e-3, 4.1e-3, 6.1e-3), 'ey'), ((15.4e-3, 6.7e-3, 5.3e-3), 'hz')]:
        sim.add_probe(p, c)
    sim.add_material('P', eps_r=3., sigma=0.02,
        debye_poles=[DebyePole(delta_eps=1.5, tau=8e-12), DebyePole(delta_eps=0.7, tau=4e-11)],
        lorentz_poles=[LorentzPole(omega_0=2*np.pi*6e9, delta=2*np.pi*0.8e9, kappa=0.5*(2*np.pi*6e9)**2)])
    sim.add(Box((-1., -1., 7e-3), (1., 5e-3, 1.)), material='P')
    sim.add_material('L', eps_r=2.2, sigma=0.05, lorentz_poles=[LorentzPole(
        omega_0=2*np.pi*9e9, delta=2*np.pi*2e9, kappa=(2*np.pi*9e9)**2)])
    sim.add(Box((9e-3, 6e-3, 1e-3), (16e-3, 10e-3, 6e-3)), material='L')
    if cells:
        for pos, comp in all_e_cells(sim):
            sim.add_probe(pos, comp)
    return sim


@pytest.mark.parametrize('n', [400, 1600])
def test_full_absorber_law_c_and_bar_s(n, monkeypatch):
    sim = scene()
    dt = float(sim._build_nonuniform_grid().dt)
    original, calls = runner.build_setup, []
    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'build_setup', observe)
    for entry in ('forward', 'run'):
        def evaluate(devices=None):
            if entry == 'run':
                return np.asarray(sim.run(n_steps=n, skip_preflight=True, devices=devices).time_series)
            kw = {} if devices is None else dict(distributed=True, devices=devices)
            return np.asarray(sim.forward(n_steps=n, skip_preflight=True, checkpoint=False, **kw).time_series)
        one = evaluate()
        for count in ((2, 3) if entry == 'forward' else (2,)):
            calls.clear()
            multi = evaluate(jax.devices('cpu')[:count])
            assert calls, (entry, count, n, 'wrapped setup not reached')
            row = dict(box='review_B_full', kind='mixed_lossy', entry=entry,
                       devices=count, n=n, **measure(one, multi, dt))
            record(row)
            check(row)

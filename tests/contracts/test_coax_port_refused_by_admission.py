"""S1 admits the uniform model lanes and names every other lane's refusal."""
import pytest
import numpy as np

from rfx import Simulation
from rfx.runners._admission import admit
from tests.contracts.test_coax_port_model_and_drive import scene, capture


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_uniform_reaches_runner(monkeypatch, entry):
    model = capture(monkeypatch, scene(), entry)
    assert model['sources'] and model['mag_sources']


@pytest.mark.parametrize('lane', ['run_nonuniform', 'fwd_nonuniform', 'run_subgridded',
    'run_distributed', 'fwd_distributed', 'run_distributed_nu', 'fwd_distributed_nu'])
def test_other_lane_refuses_by_name(lane):
    with pytest.raises(NotImplementedError, match=lane + ': add_coaxial_port'):
        admit(scene(), lane)


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_without_coax_runs_one_step(mesh, entry):
    profiles = {'dx_profile': np.full(12, .001)} if mesh == 'graded' else {}
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .012), dx=.001,
                     boundary='pec', **profiles)
    sim.add_probe((.008, .006, .006), 'ez')
    result = getattr(sim, entry)(n_steps=1, skip_preflight=True)
    assert result.time_series.shape == (1, 1)
    assert np.isfinite(np.asarray(result.time_series)).all()

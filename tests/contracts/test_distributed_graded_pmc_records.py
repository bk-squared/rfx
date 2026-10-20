"""PMC graded solves refuse each requested record instead of returning empties."""
import pytest
import numpy as np

from rfx import Simulation, SnapshotSpec
from rfx.boundaries.spec import Boundary, BoundarySpec
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import record, mutation_result


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('kind', ['planes', 'flux', 'snapshots', 'moments', 'planes_flux'])
def test_pmc_record_refusal(kind, entry):
    grading = (dict(dz_profile=np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 8), 1e-3])
               if kind == 'moments' else
               dict(dx_profile=np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 13), 1e-3]))
    # main's solved lengths on the graded mesh (round): 15.3 -> 15, 8.7 -> 9 mm; #1887
    sim = Simulation(freq_max=15e9, domain=(15e-3, 9e-3, 10.1e-3), dx=1e-3,
        **grading,
        boundary=BoundarySpec(x=Boundary('pec', 'pmc'), y='pec', z='pec'))
    sim.add_source((5.1e-3, 2.7e-3, 4.3e-3), 'ez', amplitude_kind='field')
    kwargs = dict(n_steps=4, skip_preflight=True, devices=host_devices(2))
    if entry == 'forward':
        kwargs['distributed'] = True
    if kind in ('planes', 'planes_flux'):
        sim.add_dft_plane_probe(axis='y', coordinate=3.1e-3, freqs=[3e9])
    if kind in ('flux', 'planes_flux'):
        sim.add_flux_monitor(axis='y', coordinate=3.1e-3, freqs=[3e9])
    if kind == 'snapshots':
        kwargs['snapshot'] = SnapshotSpec(interval=1, components=('ez',))
    if kind == 'moments':
        sim.add_current_moment_monitor((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 8e-3),
                                      block_size=2e-3, freqs=[3e9])
    # Public forward has no snapshot keyword. Moments reject PMC because an
    # electric-current monitor cannot see its magnetic surface current.
    expected_type = TypeError if (kind, entry) == ('snapshots', 'forward') else NotImplementedError
    words = {'planes': ('DFT planes',), 'flux': ('flux monitors',),
             'snapshots': ('snapshot',), 'moments': ('current-moment',),
             'planes_flux': ('DFT planes', 'flux monitors')}
    with pytest.raises(expected_type) as failure:
        getattr(sim, entry)(**kwargs)
    message = str(failure.value)
    assert all(word in message for word in words[kind]), message
    if kind == 'moments':
        assert 'PMC' in message, message
    if kind != 'moments' and (kind, entry) != ('snapshots', 'forward'):
        assert 'wrapped step' in message and 'PMC' in message
    record('pmc_refusal', dict(record=kind, entry=entry), dict(message=message))


@pytest.mark.parametrize('kind,entry', [('planes', 'run'), ('flux', 'run'), ('snapshots', 'run'),
                                     ('planes', 'forward'), ('flux', 'forward')])
def test_pmc_record_guard_mutation(kind, entry, monkeypatch):
    from rfx.runners import _distributed_nu_execute
    monkeypatch.setattr(_distributed_nu_execute, '_require_wrapped_records', lambda *args: None)
    with pytest.raises(pytest.fail.Exception, match='DID NOT RAISE') as failure:
        test_pmc_record_refusal(kind, entry)
    mutation_result('pmc_refusal', f'drop_guard/{kind}/{entry}', failure.value)

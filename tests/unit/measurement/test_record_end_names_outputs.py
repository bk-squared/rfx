"""Every unguarded record names its outputs (DESIGN C1–C3 and Addendum 1)."""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.api._spec import Result
from rfx.errors import AccuracyWarning
from tests.unit.farfield.test_current_moment_monitor import _ringdown_box

BINS = np.array([6, 8, 10, 12.4, 14, 16]) * 1e9
SBINS = np.array([10, 12.4, 14]) * 1e9
S = ("S-parameters",)
FIELDS = ("NTFF far fields", "field-DFT planes", "flux monitors", "current-moment far fields")


def _build(scene):
    if scene == 5:
        sim = Simulation(freq_max=20e9, domain=(.012, .012, .008), dx=.001,
                         boundary='cpml', cpml_layers=6)
        sim.add_material('fill', eps_r=2.2, sigma=0)
        sim.add(Box((.003, .003, .002), (.009, .009, .004)), material='fill')
        sim.add_port(position=(.006, .006, .002), component='ez', impedance=50., extent=.001,
                     waveform=GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5))
        sim.add_current_moment_monitor((.001, .001, .001), (.011, .011, .006),
                                       block_size=.002, freqs=np.array([6e9, 8e9]))
        sim.add_probe((.004, .005, .003), component='ez')
    else:
        sim = _ringdown_box('uniform')
    if scene != 1:
        sim.add_ntff_box((.001, .001, .001), (.011, .011, .006), freqs=BINS)
    if scene in (2, 6):
        sim.add_flux_monitor(axis='z', coordinate=.006, freqs=BINS, size=(.010, .010),
                             center=(.006, .006), name='ren_flux')
    if scene == 6:
        sim.add_probe((.004, .005, .007), component='ez')
    return sim


def _messages(caught):
    return [str(w.message) for w in caught if issubclass(w.category, AccuracyWarning)
            and (str(w.message).startswith('no ring-down settling witness')
                 or str(w.message).startswith('ring-down settling witness'))]


def _assert_names(messages, expected):
    assert len(messages) == len(expected), messages
    for message, names in zip(messages, expected):
        assert message.endswith('Outputs read from this record: ' + ', '.join(names) + '.'), message
        for other in (*S, *FIELDS):
            if other not in names:
                assert other not in message, message


@pytest.mark.parametrize('scene,statuses,reads,spoken', [
    (1, ['absent'], [('current-moment far fields',)], [('current-moment far fields',)]),
    (2, ['pass', 'absent'], [S, ('NTFF far fields', 'flux monitors', 'current-moment far fields')],
     [('NTFF far fields', 'flux monitors', 'current-moment far fields')]),
    (3, ['undetermined', 'absent'], [S, ('NTFF far fields', 'current-moment far fields')],
     [S, ('NTFF far fields', 'current-moment far fields')]),
    (5, ['undetermined', 'undetermined'], [S, ('NTFF far fields', 'current-moment far fields')],
     [S, ('NTFF far fields', 'current-moment far fields')]),
    (6, ['pass', 'pass'], [S, ('NTFF far fields', 'flux monitors', 'current-moment far fields')], []),
], ids=['scene1', 'scene2', 'scene3', 'scene5_80', 'scene6'])
def test_run_names_every_unguarded_record(scene, statuses, reads, spoken, monkeypatch):
    if scene == 3:
        monkeypatch.setattr('rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES', 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim = _build(scene)
        kwargs = dict(n_steps=80 if scene == 5 else 800, compute_s_params=scene != 1,
                      skip_preflight=False)
        if scene != 1:
            kwargs['s_param_freqs'] = SBINS
        result = sim.run(**kwargs)
    rows = result.settling_witness.get('runs') or [result.settling_witness]
    assert [r['status'] for r in rows] == statuses
    assert [r['reads'] for r in rows] == reads
    _assert_names(_messages(caught), spoken)


def _row(status, reads):
    return dict(status=status, reads=reads, db={'fail': -20., 'pass': -80., 'absent': None}[status],
                reason='no probe records the fields' if status == 'absent' else '',
                worst_freq_hz=10e9, share_per_bin=np.array([.1 if status == 'fail' else 0.]),
                error_per_bin=np.zeros(1), worst_record='probe0(ez)', source_dominated=False)


@pytest.mark.parametrize('s_status,field_status,spoken', [
    ('fail', 'absent', [S, FIELDS]),
    ('pass', 'fail', [FIELDS]),
], ids=['F2', 'F3'])
def test_rows_warn_for_their_own_outputs(s_status, field_status, spoken):
    from rfx.sparams._tail_witness import combine_witness_details
    sim = Simulation(freq_max=20e9, domain=(.012, .012, .008), dx=.001)
    sim._ntff, sim._dft_planes = object(), [object()]
    sim._flux_monitors, sim._current_moments = [object()], object()
    witness = combine_witness_details(_row(s_status, S), _row(field_status, FIELDS))
    sim._run_settling_witness = lambda result: (witness['db'], witness)
    result = Result(state=None, time_series=None, s_params=np.ones((1, 1, 1)), freqs=SBINS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim._attach_run_settling_witness(result, n_steps=800)
    _assert_names(_messages(caught), spoken)


@pytest.mark.parametrize('status,reads', [
    ('fail', ()), ('absent', ()), ('fail', FIELDS)], ids=['fail', 'absent', 'fail_fields'])
def test_s_without_an_s_row_still_warns(status, reads):
    """A lane that returns S but builds no S row (review P2, Addendum 4):
    the probe or field row speaks for S too, as base did."""
    sim = Simulation(freq_max=20e9, domain=(.012, .012, .008), dx=.001)
    if reads:
        sim._ntff, sim._dft_planes = object(), [object()]
        sim._flux_monitors, sim._current_moments = [object()], object()
    row = _row(status, reads)
    witness = {**row, 'runs': [row], 'groups': [row]}
    sim._run_settling_witness = lambda result: (witness['db'], witness)
    result = Result(state=None, time_series=None, s_params=np.ones((1, 1, 1)), freqs=SBINS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim._attach_run_settling_witness(result, n_steps=800)
    _assert_names(_messages(caught), [S + reads])


def test_field_row_names_have_the_design_order():
    from rfx.sparams._record_selection import build_settling_witness
    result = SimpleNamespace(ntff_box=object(), dft_planes={'e': object()},
                             flux_monitors={'p': object()}, current_moment_monitor=object())
    _, witness = build_settling_witness(result, bins=SBINS, dt=1e-12, freq_max=20e9,
                                       probe_info=(), source_end=None, port_source_end=None)
    assert witness['status'] == 'absent'
    assert witness['reads'] == FIELDS

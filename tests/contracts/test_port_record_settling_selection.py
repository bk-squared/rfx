"""The result's S read, not record presence, admits a settling record."""
from types import SimpleNamespace

import numpy as np
import pytest

from rfx.api._spec import ForwardResult
from rfx.measurement import PortTimeRecord
from rfx.sparams._record_selection import SReadPorts
from rfx.sparams._record_selection import forward_settling_witness


@pytest.mark.parametrize('reads_s', [False, True])
def test_forward_scores_only_s_read_ports(monkeypatch, reads_s):
    import rfx.sparams._tail_witness as witness
    seen = []

    def score(records, *args, **kwargs):
        seen.extend(records)
        return -42., {'status': 'pass'}

    monkeypatch.setattr(witness, 'port_record_witness', score)
    record = PortTimeRecord('port:0:mode:0', 'lumped', ('V', 'I', 'V_ref'), np.ones((20, 3)))
    result = ForwardResult(time_series=None, dt=1., freqs=np.array([.1]),
                           s_params=np.ones(1) if reads_s else None,
                           sparam_time_records=(record,), sparam_read_ids=SReadPorts(('port:0:mode:0',)))
    db, detail = forward_settling_witness(result)
    assert len(seen) == int(reads_s)
    assert detail['scored_port_record_ids'] == (['port:0:mode:0'] if reads_s else [])
    assert detail['skipped_port_records'] == ([] if reads_s else [
        {'id': 'port:0:mode:0', 'reason': 'S not read'}])
    assert db == (-42. if reads_s else None)
    assert result.settling_db == db
    assert result.settling_witness == detail


def test_uniform_forward_skips_wire_when_result_reads_lumped_s(monkeypatch):
    import rfx.sparams._tail_witness as witness
    seen = []
    monkeypatch.setattr(witness, 'port_record_witness',
                        lambda records, *a, **k: (seen.extend(records) or -42., {}))
    wire = PortTimeRecord('port:0:mode:0', 'wire', ('V', 'I', 'V_port', 'V_ref'), np.ones((20, 4)))
    lumped = PortTimeRecord('port:1:mode:0', 'lumped', ('V', 'I', 'V_ref'), np.ones((20, 3)))
    result = ForwardResult(time_series=None, dt=1., freqs=np.array([.1]),
                           s_params=np.ones(1), sparam_time_records=(lumped, wire),
                           sparam_read_ids=SReadPorts(('port:1:mode:0',)),
                           wire_port_sparams=((None, None),),
                           lumped_port_sparams=((None, None),))
    _, detail = forward_settling_witness(result)
    assert len(seen) == 1 and seen[0] is lumped
    assert detail['scored_port_record_ids'] == ['port:1:mode:0']
    assert detail['skipped_port_records'] == [
        {'id': 'port:0:mode:0', 'reason': 'S not read'}]


def test_run_does_not_score_record_without_s(monkeypatch):
    from rfx import Simulation
    import rfx.sparams._tail_witness as witness
    sim = Simulation(freq_max=1e9, domain=(.01,)*3, dx=.001)
    monkeypatch.setattr(witness, 'port_record_witness',
                        lambda *a, **k: pytest.fail('S not read'))
    result = SimpleNamespace(s_params=None, time_series=None, dt=1.,
                             sparam_time_records=(PortTimeRecord('port:0:mode:0', 'lumped', ('V', 'I', 'V_ref'), np.ones((20, 3))),))
    db, detail = sim._run_settling_witness(result)
    assert db is None
    assert detail['skipped_port_records'] == [
        {'id': 'port:0:mode:0', 'reason': 'S not read'}]


def _lumped_then_wire(graded):
    import jax.numpy as jnp
    from rfx import Simulation
    sim = Simulation(freq_max=10e9, domain=(.008, .006, .002), dx=.001, boundary='pec',
                     **({'dz_profile': np.array([.0008, .0012])} if graded else {}))
    sim.add_port((.002, .003, 0.), 'ez', impedance=50., waveform=lambda t: jnp.cos(t * 2e10))
    sim.add_port((.006, .003, 0.), 'ez', impedance=50., extent=.002, excite=False)
    return sim


@pytest.mark.parametrize('graded, freqs, read_kinds', [
    (False, np.array([2e9, 4e9]), {'lumped'}),   # uniform forward forms S from the lumped accumulators only
    (True, np.array([2e9, 4e9]), {'lumped', 'wire'}),  # graded forward with bins: S from both kinds
    (True, None, set()),                         # graded forward without bins: no S, both ports only record
], ids=['uniform-lumped-S', 'graded-both-S', 'graded-no-S-records-only'])
def test_real_forward_scores_exactly_the_records_behind_its_s(monkeypatch, graded, freqs, read_kinds):
    import warnings
    import rfx.sparams._tail_witness as witness
    seen, score = [], witness.port_record_witness
    monkeypatch.setattr(witness, 'port_record_witness',
                        lambda records, *a, **k: (seen.extend(records), score(records, *a, **k))[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = _lumped_then_wire(graded).forward(n_steps=60, skip_preflight=True, port_s11_freqs=freqs)
        result.settling_witness
    records = result.sparam_time_records
    assert (result.s_params is not None) == bool(read_kinds) and {r.kind for r in records} == {'lumped', 'wire'}
    expected = {r.id for r in records if r.kind in read_kinds}
    assert {r.id for r in seen} == expected
    assert {s['id'] for s in result.settling_witness['skipped_port_records']} == {r.id for r in records} - expected

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

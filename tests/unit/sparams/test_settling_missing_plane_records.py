"""Dedicated settling probes are scored without retained plane time records."""

import numpy as np
import pytest


@pytest.mark.parametrize('family', ['msl', 'mixed'])
def test_dedicated_witness_probes_survive_missing_plane_records(monkeypatch, family):
    import rfx.sources.waveguide_port as scorer

    captured = []
    original_score = scorer.settling_db_from_named_records

    def score(records, **kwargs):
        records = tuple(records)
        if any(name == 'probe0' for name, _ in records):
            captured.append(records)
        return original_score(records, **kwargs)

    monkeypatch.setattr(scorer, 'settling_db_from_named_records', score)
    if family == 'msl':
        from tests.unit.sparams.test_msl_internal_probe_advisories import _thru, FREQS

        sim = _thru()
        original_run = sim.run

        def run(**kwargs):
            return original_run(**kwargs)._replace(dft_time_records=None)

        monkeypatch.setattr(sim, 'run', run)
        result = sim.compute_msl_s_matrix(freqs=FREQS, num_periods=2.)
    else:
        from tests.unit.sparams.test_mixed_port_sparam import _base_sim, _add_feed, _add_msl

        sim, y_c = _base_sim()
        _add_feed(sim, y_c, x=2e-3)
        _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)
        original_forward = sim._forward_from_materials

        def forward(*args, **kwargs):
            raw = original_forward(*args, **kwargs)
            return {**raw, 'dft_time_records': None, 'sparam_time_records': None}

        monkeypatch.setattr(sim, '_forward_from_materials', forward)
        result = sim.compute_mixed_s_matrix(
            freqs=np.linspace(1e9, 4e9, 5), num_periods=4., skip_preflight=True)
    assert len(captured) == len(result.settling_witness) == 2
    assert all(records and all(name.startswith('probe') for name, _ in records)
               for records in captured)
    assert all(row['status'] == 'undetermined' for row in result.settling_witness)


@pytest.mark.parametrize("readout", ["ntff", "dft"])
def test_far_field_or_plane_read_without_a_probe_is_not_witnessed_by_port_records(readout):
    from types import SimpleNamespace
    from rfx import Simulation
    from rfx.api._spec import Result

    dt = 0.4
    t = np.arange(1000) * dt
    port = np.exp((-0.1 + 2j * np.pi * 0.5) * t)
    fields = (dict(ntff_box=SimpleNamespace(freqs=np.array([0.5]))) if readout == "ntff"
              else dict(dft_planes={"e": SimpleNamespace(freqs=np.array([0.5]))}))
    sim = Simulation(freq_max=1.0, domain=(1.0,) * 3, dx=0.1)
    sim._settling_source_end = lambda *a, **k: 0
    result = Result(state=None, s_params=None, dt=dt, time_series=np.zeros((1000, 0)),
                    sparam_time_records=(port,), freqs=None, **fields)
    _, detail = sim._run_settling_witness(result)
    assert detail["status"] != "pass"
    assert "no probe records" in detail["reason"]

"""Public two-device run() returns implicit S and finishes probe diagnostics."""

import jax
import numpy as np
import pytest

from rfx import Simulation


N_STEPS = 40
# Measured max one/two-device difference: 0 dB for both probes and the worst.
SETTLING_ATOL_DB = 1e-5


def _devices():
    devices = jax.devices("cpu")
    assert len(devices) >= 2, "requires the root conftest's two CPU devices"
    return devices[:2]


def _model(*, port=False, probes=True):
    sim = Simulation(freq_max=15e9, domain=(24e-3, 12e-3, 12e-3),
                     dx=1e-3, boundary="pec")
    if port:
        sim.add_port((6e-3, 6e-3, 6e-3), "ez", impedance=50.0)
    else:
        sim.add_source((6e-3, 6e-3, 6e-3), "ez", amplitude_kind="field")
    if probes:
        for x in (12e-3, 22e-3):
            sim.add_probe((x, 6e-3, 6e-3), "ez")
    return sim


def test_default_lumped_port_s_request_matches_explicit():
    results = [_model(port=True).run(n_steps=N_STEPS, devices=_devices(), **kwargs)
               for kwargs in ({}, {"compute_s_params": True})]
    assert results[0].s_params is not None
    np.testing.assert_array_equal(results[0].s_params, results[1].s_params)
    np.testing.assert_array_equal(results[0].freqs, results[1].freqs)


def test_graded_lumped_port_default_is_no_s_request():
    """On a graded mesh one device (the non-uniform lane) computes S by default
    only for wire ports, so a single-cell port's default is no S request. The
    graded multi-device runner refuses the port itself, and that refusal, not
    the S-request one, is what the user must see."""
    def graded():
        sim = Simulation(freq_max=15e9, domain=(24e-3, 12e-3, 0.0),
                         dx=1e-3, boundary="pec",
                         dz_profile=np.array([1e-3] * 6 + [0.5e-3] * 12))
        sim.add_port((6e-3, 6e-3, 6e-3), "ez", impedance=50.0)
        sim.add_probe((12e-3, 6e-3, 6e-3), "ez")
        return sim
    assert graded().run(n_steps=N_STEPS).s_params is None
    with pytest.raises(NotImplementedError) as exc:
        graded().run(n_steps=N_STEPS, devices=_devices())
    assert "lumped / wire ports" in str(exc.value).lower()
    assert "compute_s_params" not in str(exc.value)


def test_lumped_port_can_opt_out_of_s_params():
    result = _model(port=True).run(n_steps=N_STEPS, devices=_devices(),
                                   compute_s_params=False)
    assert result.s_params is None
    assert np.max(np.abs(result.time_series)) > 0


def test_portless_default_run_returns_matching_settling_witness():
    single = _model().run(n_steps=N_STEPS, devices=_devices()[:1])
    multi = _model().run(n_steps=N_STEPS, devices=_devices())
    assert multi.s_params is None
    assert single.settling_db is not None and multi.settling_db is not None
    assert single.settling_witness is not None and multi.settling_witness is not None
    assert multi.settling_witness["status"] == "measured"
    assert multi.settling_db == pytest.approx(single.settling_db, rel=0,
                                            abs=SETTLING_ATOL_DB)
    single_records = single.settling_witness["per_record_db"]
    multi_records = multi.settling_witness["per_record_db"]
    assert len(single_records) == 2
    assert multi_records == pytest.approx(single_records, rel=0,
                                         abs=SETTLING_ATOL_DB)
    assert {k: v for k, v in multi.settling_witness.items() if k != "per_record_db"} == {
        k: v for k, v in single.settling_witness.items() if k != "per_record_db"}
    differences = [abs(multi.settling_db - single.settling_db)] + [
        abs(multi_records[k] - single_records[k]) for k in single_records]
    print(f"settling max absolute difference: {max(differences):.12g} dB; "
          f"tolerance: {SETTLING_ATOL_DB:g} dB (rtol=0)")


def test_probeless_run_reports_absent_witness():
    result = _model(probes=False).run(n_steps=N_STEPS, devices=_devices())
    assert result.settling_db is None
    assert result.settling_witness["status"] == "absent"
    assert result.settling_witness["reason"]

"""Public two-device run() returns implicit S and finishes probe diagnostics."""

import jax
import numpy as np
import pytest

from rfx import Simulation


N_STEPS = 40
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


def test_graded_lumped_port_default_forms_s():
    """On a graded mesh one driven single-cell port forms S by default on one
    device (#1373: graded run() lumped S); the graded multi-device runner
    carries the port (#1402 step 2) with the same default."""
    def graded():
        sim = Simulation(freq_max=15e9, domain=(24e-3, 12e-3, 0.0),
                         dx=1e-3, boundary="pec",
                         dz_profile=np.array([1e-3] * 6 + [0.5e-3] * 12))
        sim.add_port((6e-3, 6e-3, 6e-3), "ez", impedance=50.0)
        sim.add_probe((12e-3, 6e-3, 6e-3), "ez")
        return sim
    one = graded().run(n_steps=N_STEPS)
    multi = graded().run(n_steps=N_STEPS, devices=_devices())
    assert one.s_params is not None and multi.s_params is not None
    a, b = np.asarray(one.s_params), np.asarray(multi.s_params)
    assert a.shape == b.shape
    peak = float(np.max(np.abs(a)))
    assert float(np.max(np.abs(a - b))) <= 9 * float(np.spacing(np.float32(peak)))
    assert np.max(np.abs(np.asarray(multi.time_series))) > 0
    np.testing.assert_array_equal(np.asarray(multi.time_series), np.asarray(one.time_series))


def test_lumped_port_can_opt_out_of_s_params():
    result = _model(port=True).run(n_steps=N_STEPS, devices=_devices(),
                                   compute_s_params=False)
    assert result.s_params is None
    assert np.max(np.abs(result.time_series)) > 0


def test_portless_default_run_returns_matching_settling_witness():
    from dataclasses import replace
    import jax.numpy as jnp

    steps = 160
    single_sim, multi_sim = _model(), _model()
    for sim in (single_sim, multi_sim):
        sim._ports[0] = replace(sim._ports[0], waveform=lambda t: jnp.where(t == 0, 1., 0.))
    single = single_sim.run(n_steps=steps, devices=_devices()[:1])
    multi = multi_sim.run(n_steps=steps, devices=_devices())
    single_sim.add_dft_plane_probe(axis="z", coordinate=.006, component="ez", n_freqs=3)
    plane_run = single_sim.run(n_steps=steps, devices=_devices()[:1])
    single = single_sim._attach_run_settling_witness(
        single._replace(dft_planes=plane_run.dft_planes, settling_witness=None))
    multi_sim.add_dft_plane_probe(axis="z", coordinate=.006, component="ez", n_freqs=3)
    multi = multi_sim._attach_run_settling_witness(
        multi._replace(dft_planes=plane_run.dft_planes, settling_witness=None))
    assert multi.s_params is None
    assert single.settling_db is not None and multi.settling_db is not None
    assert single.settling_witness is not None and multi.settling_witness is not None
    for result in (single, multi):
        detail = result.settling_witness
        assert detail["status"] == "undetermined"
        assert "source end" not in detail["reason"]
        assert set(detail["per_record_db"]) == {"probe0(ez)", "probe1(ez)"}
        assert detail["share_per_bin"].size == 3
        assert result.time_series.shape == (steps, 2)
    np.testing.assert_equal(multi.settling_witness, single.settling_witness)
    np.testing.assert_array_equal(multi.time_series, single.time_series)


def test_probeless_run_reports_absent_witness():
    result = _model(probes=False).run(n_steps=N_STEPS, devices=_devices())
    assert result.settling_db is None
    assert result.settling_witness["status"] == "absent"
    assert result.settling_witness["reason"]

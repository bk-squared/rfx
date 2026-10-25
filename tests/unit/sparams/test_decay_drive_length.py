"""Multi-port drives use the realized main record on either mesh."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.probes import sparam_driver
from rfx.sparams import _tail_witness


def _box(mesh):
    kwargs = (
        dict(dx_profile=np.r_[np.full(6, 1e-3), np.full(4, 0.6e-3),
                              np.full(6, 1e-3)], domain=(0, 12e-3, 10e-3))
        if mesh == "graded" else dict(domain=(12.4e-3, 12e-3, 10e-3))
    )
    sim = Simulation(freq_max=12e9, dx=1e-3, boundary="cpml", cpml_layers=6,
                     **kwargs)
    for n, x in enumerate((4e-3, 8e-3)):
        sim.add_port(position=(x, 6e-3, 4e-3), component="ez", impedance=50.,
                     excite=n == 0, waveform=GaussianPulse(f0=6e9, bandwidth=0.8))
    sim.add_material("d", eps_r=4.3)
    sim.add(Box((6.3e-3, 2.3e-3, 1.3e-3), (7.7e-3, 9.7e-3, 8.7e-3)), material="d")
    return sim


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_decay_drives_match_main_record(mesh, monkeypatch):
    # Observe actual receiver samples consumed by the driver's settling witness,
    # not just the requested n_steps. Keep the production driver and witness live.
    driver = sparam_driver.compute_lumped_wire_s_matrix_via_scan
    witness = _tail_witness.port_record_witness
    drive_records = []
    active = False

    def observe_driver(*args, **kwargs):
        nonlocal active
        active = True
        try:
            return driver(*args, **kwargs)
        finally:
            active = False

    def observe_witness(records, *args, **kwargs):
        if active:
            drive_records.append(tuple(record.samples.shape[0] for record in records))
        return witness(records, *args, **kwargs)

    monkeypatch.setattr(sparam_driver, "compute_lumped_wire_s_matrix_via_scan", observe_driver)
    monkeypatch.setattr(_tail_witness, "port_record_witness", observe_witness)
    run_kwargs = dict(skip_preflight=True, compute_s_params=True,
                      s_param_freqs=np.linspace(3e9, 9e9, 7))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stopped = _box(mesh).run(until_decay=1e-3, num_periods=3, **run_kwargs)
        stop, = (d for d in stopped.diagnostics if d.code == "until_decay.stop")
        stop_step = stop.values["stop_step"]
        decay_records = list(drive_records)
        drive_records.clear()
        reference = _box(mesh).run(n_steps=stop_step, **run_kwargs)
    actual, expected = np.asarray(stopped.s_params), np.asarray(reference.s_params)
    peak = float(np.max(np.abs(expected)))
    # Four float32 epsilons of the peak |S|, across every matrix entry and bin.
    bound = 4 * np.finfo(np.float32).eps * peak
    difference = float(np.max(np.abs(actual - expected)))
    settling_warnings = [str(w.message) for w in caught
                         if issubclass(w.category, UserWarning)
                         and ("settl" in str(w.message).lower()
                              or "trunc" in str(w.message).lower())]
    assert decay_records == [(stop_step, stop_step)] * 2, "drive records must reach the decay stop"
    assert drive_records == [(stop_step, stop_step)] * 2
    assert difference <= bound, f"max |S - S(n_steps=stop)| = {difference} > {bound}"
    assert not settling_warnings
    assert stopped.settling_witness["status"] == "pass"


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("stop_step, override, expected", [
    (None, None, 132), (550, None, 550), (550, 80, 80), (None, 80, 80),
])
def test_drive_length_override_and_fixed_record(graded, stop_step, override, expected):
    from rfx.measurement.decay import drive_steps

    stop = None if stop_step is None else SimpleNamespace(values={"stop_step": stop_step})
    result = {"decay_stop": stop} if graded else SimpleNamespace(decay_stop=stop)
    assert drive_steps(result, 132, override) == expected


@pytest.mark.parametrize("graded", [False, True])
def test_forced_length_run_drives_its_record(graded):
    # until_decay=0 runs decay_max_steps with no stop record; the drives follow it.
    from rfx.measurement.decay import drive_steps

    result = {"decay_stop": None} if graded else SimpleNamespace(decay_stop=None)
    assert drive_steps(result, 132, until_decay=0.0, decay_max_steps=400) == 400
    assert drive_steps(result, 132, 80, until_decay=0.0, decay_max_steps=400) == 80


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_forced_length_drives_match_main_record(mesh, monkeypatch):
    driver = sparam_driver.compute_lumped_wire_s_matrix_via_scan
    lengths = []

    def observe_driver(*args, **kwargs):
        lengths.append(kwargs["n_steps"])
        return driver(*args, **kwargs)

    monkeypatch.setattr(sparam_driver, "compute_lumped_wire_s_matrix_via_scan", observe_driver)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _box(mesh).run(until_decay=0.0, decay_max_steps=400, num_periods=3,
                                skip_preflight=True, compute_s_params=True,
                                s_param_freqs=np.linspace(3e9, 9e9, 7))
    assert np.asarray(result.time_series).shape[0] == 400
    assert lengths == [400]

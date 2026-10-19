"""Implicit mixed-drive S resolves to the explicit fields-only request."""
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest

from rfx import Simulation

# Verbatim refusal on origin/main f0e3805d9, not derived from the implementation.
REASON = (
    "Lumped/wire S-matrix requests do not support plain sources / "
    "0-ohm ports (add_source, add_polarized_source or "
    "add_port(impedance=0)): they fire in every port drive. "
)
REFUSAL = REASON + (
    "Remove the plain source, or call run(compute_s_params=False) "
    "to get raw port waves with the source included."
)
FIX = ("Remove the plain source, or pass compute_s_params=False "
       "to silence this warning; returning no S-parameters.")


def model(graded=False, wire=False, passive=False, source=True):
    profile = {"dz_profile": np.array([.001] * 4 + [.0005] * 8)} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.016, .008, .008), dx=.001,
                     boundary="pec", **profile)
    sim.add_port((.008, .004, .003), "ez", impedance=50., excite=not passive,
                 **({"extent": .002} if wire else {}))
    sim.add_probe((.008, .004, .003), "ez")
    if source:
        sim.add_source((.004, .004, .003), "ez", amplitude_kind="field")
    return sim


def assert_bytes(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("wire", [False, True], ids=["lumped", "wire"])
@pytest.mark.parametrize("distributed", [False, True], ids=["one", "two"])
def test_default_fields_equal_explicit_false(graded, wire, distributed):
    devices = jax.devices("cpu")[:2] if distributed else None
    if distributed:
        assert len(devices) == 2
    options = dict(n_steps=12, skip_preflight=True, devices=devices)
    expected = model(graded, wire).run(compute_s_params=False, **options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = model(graded, wire).run(**options)
    mixed = [w for w in caught if "S-matrix requests do not support plain sources" in str(w.message)]
    # Main's graded lumped default does not request S at all (#1892).
    if graded and not wire:
        assert mixed == []
    else:
        assert len(mixed) == 1
        assert mixed[0].category is UserWarning
        assert str(mixed[0].message) == REASON + FIX
        assert Path(mixed[0].filename).resolve() == Path(__file__).resolve()
    assert actual.s_params is expected.s_params is None
    assert np.any(np.asarray(expected.time_series))
    assert_bytes(actual.time_series, expected.time_series)
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        assert_bytes(getattr(actual.state, component), getattr(expected.state, component))


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("wire", [False, True], ids=["lumped", "wire"])
@pytest.mark.parametrize("s_request", [
    {"compute_s_params": True}, {"s_param_freqs": [5e9]}, {"s_param_n_steps": 12},
], ids=["true", "freqs", "steps"])
def test_explicit_refusal_unchanged(graded, wire, s_request):
    sim = model(graded, wire)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError) as error:
            sim.run(n_steps=12, skip_preflight=True, **s_request)
    assert str(error.value) == REFUSAL
    assert caught == []


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_passive_main_record_remains_admitted(graded):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model(graded, wire=True, passive=True).run(n_steps=12, skip_preflight=True)
    assert result.s_params is not None
    assert not any("plain sources" in str(w.message) for w in caught)


@pytest.mark.parametrize("source", [False, True])
def test_preflight_explicit_request_result_is_unused(source):
    sim = model(wire=True, source=source)
    with warnings.catch_warnings(record=True) as caught:
        report = sim.preflight_sparameters(calculator="run")
    assert caught == []
    assert len(report) == int(source)
    if source:
        assert str(report[0]) == "NotImplementedError: " + REFUSAL
    else:
        assert sim._validate_run_sparameter_request(
            compute_s_params=True, s_param_freqs=None, s_param_n_steps=None) is True


def test_distributed_default_validation_retains_resolved_true(monkeypatch):
    calls = []
    original = Simulation._validate_run_sparameter_request

    def recorded(self, **kwargs):
        resolved = original(self, **kwargs)
        calls.append((kwargs["compute_s_params"], resolved))
        return resolved

    monkeypatch.setattr(Simulation, "_validate_run_sparameter_request", recorded)
    result = model(wire=True, source=False).run(
        n_steps=12, skip_preflight=True, devices=jax.devices("cpu")[:2])
    assert result.s_params is not None
    assert calls == [(None, None), (True, True)]


def test_distributed_fallback_keeps_resolved_false(monkeypatch):
    from rfx.runners import distributed_v2

    expected = model(wire=True).run(n_steps=12, skip_preflight=True, compute_s_params=False)
    # Exercise the full-domain fallback's argument transport without adding
    # unrelated waveguide/TFSF geometry to this synthetic port experiment.
    monkeypatch.setattr(distributed_v2, "one_device_reason", lambda sim: "test full-domain feature")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = model(wire=True).run(n_steps=12, skip_preflight=True,
                                      devices=jax.devices("cpu")[:2])
    mixed = [w for w in caught if "plain sources" in str(w.message)]
    assert len(mixed) == 1
    assert actual.s_params is None
    assert_bytes(actual.time_series, expected.time_series)
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        assert_bytes(getattr(actual.state, component), getattr(expected.state, component))

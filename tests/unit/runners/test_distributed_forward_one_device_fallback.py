"""A5: full-domain graded fallbacks preserve the public forward program."""
import gc
import inspect
import warnings
from dataclasses import fields, is_dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from tests._multi_device_scene import host_devices


MEMORY = ("the whole domain is held on one device; peak memory is that of a "
          "one-device run, not divided over the requested devices")


@pytest.fixture(autouse=True)
def release_models(monkeypatch):
    # The once-per-process experimental-lane notice is independent of fallback.
    monkeypatch.setattr("rfx.api._execute._DISTRIBUTED_FIRST_CALL_WARNED", True)
    yield
    jax.clear_caches()
    gc.collect()


def scene(kind):
    if kind == "waveguide":
        sim = Simulation(12e9, (.06, .02, .01), dx=.002, cpml_layers=4,
                         dz_profile=np.array([.002, .0018, .0022, .0018, .0022]),
                         boundary={"x": "cpml", "y": "pec", "z": "pec"})
        sim.add_waveguide_port(.010, direction="+x", freqs=np.array([10e9, 11e9]),
                               f0=11e9, bandwidth=.6, probe_offset=2, ref_offset=1)
        sim.add_probe((.030, .010, .006), "ez")
    else:
        sim = Simulation(20e9, (.030, .020, .012), dx=.001, cpml_layers=4,
                         dz_profile=.001 * np.array([1.2, 1.1, 1., .9, .8, .7, .6,
                                                     .6, .7, .8, .9, 1., 1.1, .6]))
        sim.add_tfsf_source()
        sim.add_probe((.023, .012, .007), "ez")
    shape = sim._build_nonuniform_grid().shape
    eps = jnp.ones(shape, dtype=jnp.float32)
    eps = eps.at[shape[0] // 2 - 1:shape[0] // 2 + 2, :, :].set(1.7)
    return sim, eps


def array_leaves(value, path="", active=None):
    """Include arrays inside result records and host grid objects, except diagnostics."""
    if isinstance(value, (jax.Array, np.ndarray)):
        return {path: np.asarray(value)}
    active = set() if active is None else active
    if value is None or id(value) in active or callable(value):
        return {}
    active.add(id(value))
    try:
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            items = ((name, getattr(value, name)) for name in value._fields)
        elif is_dataclass(value):
            items = ((f.name, getattr(value, f.name)) for f in fields(value))
        elif isinstance(value, dict):
            items = value.items()
        elif isinstance(value, (tuple, list)):
            items = enumerate(value)
        elif hasattr(value, "__dict__"):
            items = vars(value).items()
        else:
            return {}
        found = {}
        for name, child in items:
            if name != "diagnostics":
                found.update(array_leaves(child, f"{path}/{name}", active))
        return found
    finally:
        active.remove(id(value))


@pytest.mark.parametrize("kind", ["waveguide", "tfsf"])
@pytest.mark.parametrize("automatic_devices", [False, True])
def test_forward_fallback_equals_one_device(kind, automatic_devices, monkeypatch):
    devices = host_devices(2)
    sim, eps = scene(kind)
    options = dict(eps_override=eps, n_steps=48, num_periods=2., checkpoint=False,
                   checkpoint_every=4, n_warmup=4, skip_preflight=True,
                   exchange_interval=1)
    expected = sim.forward(**options)
    background = sim.forward(**{**options, "eps_override": None})
    assert not np.array_equal(expected.time_series, background.time_series)
    expected_arrays = array_leaves(expected)
    assert expected_arrays and np.any(np.asarray(expected.time_series) != 0)
    public_forward = sim.forward
    forwarded = []

    def single_forward(**kwargs):
        forwarded.append(kwargs)
        assert kwargs["distributed"] is False
        assert kwargs["devices"] is None
        return public_forward(**kwargs)

    monkeypatch.setattr(sim, "forward", single_forward)
    with pytest.warns(UserWarning, match="Falling back to single-device"):
        actual = public_forward(**options, distributed=True,
                                devices=None if automatic_devices else devices)
    actual_arrays = array_leaves(actual)
    assert actual_arrays.keys() == expected_arrays.keys()
    for path, value in expected_arrays.items():
        assert np.array_equal(actual_arrays[path], value), path
    # The public signature supplies the default arguments; no runner helper does.
    wanted = {name: p.default for name, p in inspect.signature(public_forward).parameters.items()
              if p.kind != inspect.Parameter.VAR_KEYWORD}
    wanted.update(options, distributed=False, devices=None)
    assert len(forwarded) == 1 and forwarded[0].keys() == wanted.keys()
    for name, value in wanted.items():
        assert forwarded[0][name] is value or np.array_equal(forwarded[0][name], value), name


@pytest.mark.parametrize("kind", ["waveguide", "tfsf"])
def test_forward_fallback_gradient_equals_one_device(kind):
    devices = host_devices(2)
    sim, eps = scene(kind)

    def objective(e, distributed):
        result = sim.forward(eps_override=e, n_steps=48, checkpoint=False,
                             n_warmup=4, skip_preflight=True, distributed=distributed,
                             devices=devices if distributed else None)
        return jnp.sum(result.time_series ** 2)

    expected = jax.grad(lambda e: objective(e, False))(eps)
    assert np.isfinite(expected).all() and np.any(np.asarray(expected) != 0)
    with pytest.warns(UserWarning, match="Falling back to single-device"):
        actual = jax.grad(lambda e: objective(e, True))(eps)
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("kind", ["waveguide", "tfsf"])
@pytest.mark.parametrize("automatic_devices", [False, True])
@pytest.mark.parametrize("skip_preflight", [True, False])
def test_fallback_record_on_both_entries(kind, automatic_devices, skip_preflight, capsys):
    devices = host_devices(2)
    sim, _ = scene(kind)
    for entry in ("forward", "run"):
        options = (dict(distributed=True, devices=None if automatic_devices else devices)
                   if entry == "forward" else dict(devices=devices, compute_s_params=False))
        native_options = {} if entry == "forward" else {"compute_s_params": False}
        with warnings.catch_warnings(record=True) as native_warnings:
            warnings.simplefilter("always")
            getattr(sim, entry)(n_steps=4, skip_preflight=skip_preflight, **native_options)
        native_printed = capsys.readouterr().out.count("[PREFLIGHT]")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = getattr(sim, entry)(n_steps=4, skip_preflight=skip_preflight, **options)
        # The fallback runs preflight once (the entry's), never again inside.
        assert capsys.readouterr().out.count("[PREFLIGHT]") == native_printed
        # The short physical record can warn about settling on either entry.
        # The distributed request adds exactly one fallback warning.
        assert len(caught) == len(native_warnings) + 1
        records = [d for d in result.diagnostics if d.code == "distributed.one_device_fallback"]
        assert len(records) == 1
        record = records[0]
        reason = "waveguide ports" if kind == "waveguide" else "TFSF plane-wave sources"
        message = (f"Distributed runner does not yet support {reason}. "
                   "Falling back to single-device execution with the same arguments. " + MEMORY)
        assert record.message == message
        if skip_preflight:
            assert [str(w.message) for w in caught] == [message] + [
                str(w.message) for w in native_warnings]
        else:
            # Preflight warns before the fallback warning; the set is the same.
            assert sorted(str(w.message) for w in caught) == sorted(
                [message] + [str(w.message) for w in native_warnings])
        assert record.subject == reason and record.severity == "warning"
        assert record.values == {"requested_devices": 2, "devices_used": 1, "memory": MEMORY}
        assert record.path == ("fwd_distributed_nu" if entry == "forward" else "run_distributed")

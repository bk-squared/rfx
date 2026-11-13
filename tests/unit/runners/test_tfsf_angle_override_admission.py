"""F3/Addendum 2: first refusal, named overrides, no field scan."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.errors import RfxRefusal, RfxUnsupported
from rfx.ringdown import RingdownSpec
from tests.unit.autodiff.test_forward_tfsf_angle_gradient import scene, emit
from tests.unit.runners.test_distributed_admission_refusals import _closed_box_scene


@pytest.fixture
def no_scan(monkeypatch):
    def stop(*args, **kwargs):
        raise AssertionError("admission judge reached a field scan")
    monkeypatch.setattr(jax.lax, "scan", stop)


ROWS = ["graded_x", "graded_out", "distributed", "graded_distributed", "adjoint",
        "ringdown_cpml", "ringdown_periodic", "closed", "closed_two_devices",
        "methodB", "normal", "concrete90"]


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("row", ROWS)
def test_first_refusal(row, skip, no_scan):
    options = dict(n_steps=10, skip_preflight=skip, tfsf_angle_override=30.)
    expected, words = RfxUnsupported, ["tfsf_angle_override"]
    if row.startswith("closed"):
        sim = _closed_box_scene()
        words += ["closed"]
    else:
        graded = "x" if row in ("graded_x", "graded_distributed") else "out" if row == "graded_out" else None
        theta = 0. if row == "normal" else 30.
        sim, facts = scene(theta=theta, boundary=None if row == "ringdown_periodic" else "cpml",
                           graded=graded, method="methodB" if row == "methodB" else "bloch")
    if row.startswith("graded"):
        if not skip:
            expected, words = RfxRefusal, ["not yet supported on a nonuniform mesh"]
        else:
            words = ["graded multi-device lane"] if row == "graded_distributed" else ["tfsf_angle_override", "fwd_nonuniform"]
    if row in ("distributed", "graded_distributed", "closed_two_devices"):
        assert len(jax.devices("cpu")) >= 2
        options.update(distributed=True, devices=jax.devices("cpu")[:2])
        if row == "distributed":
            words = ["is not supported on forward_distributed"]
    if row == "adjoint":
        options.update(gradient="adjoint", design_box=((facts["front"], 0., 0.),
                        (facts["front"]+.025, .016, .008)), design_eps_override=jnp.ones((1, 1, 1)))
        expected, words = NotImplementedError, ["does not support TFSF sources"]
    if row.startswith("ringdown"):
        options["ringdown"] = RingdownSpec()
        words = ["not supported with periodic axes" if row.endswith("periodic") else "not supported with a TFSF plane wave"]
    if row == "methodB":
        words += ["methodB"]
    if row == "concrete90":
        options["tfsf_angle_override"] = 90.
        expected, words = RfxRefusal, ["tfsf_angle_override", "< 90"]
    with pytest.raises(expected) as caught:
        sim.forward(**options)
    message = str(caught.value)
    emit("B7", row=row, skip_preflight=skip, exception=type(caught.value).__name__, message=message)
    assert all(word in message for word in words), (row, words, message)


def test_first_job_traced95_message():
    # F3 runtime wrapping is itself measured; a host-only admission mock cannot test it.
    sim, _ = scene()
    fn = jax.jit(lambda angle: sim.forward(n_steps=10, tfsf_angle_override=angle,
                                          skip_preflight=False).time_series)
    with pytest.raises(Exception) as caught:
        fn(jnp.float32(95)).block_until_ready()
    message = str(caught.value)
    emit("F3", exception=type(caught.value).__name__, message=message)
    assert "tfsf_angle_override" in message, "F3 STOP: runtime wrapping lost override name: " + message
    assert "< 90" in message


@pytest.mark.parametrize("name", ["f0", "bandwidth", "amplitude", "angle_deg", "theta_deg", "phi_deg", "direction_tol_deg"])
@pytest.mark.parametrize("closed", [False, True])
def test_traced_declaration_named(name, closed, no_scan):
    sim = Simulation(freq_max=10e9, domain=(.30, .016, .008), dx=.002, boundary="cpml")
    def fn(value):
        sim.add_tfsf_source(**{name: value}, closed_box=closed)
        return value
    with pytest.raises(RfxRefusal) as caught:
        jax.jit(fn)(jnp.float32(1))
    message = str(caught.value)
    emit("B7_declaration", name=name, closed=closed, message=message)
    assert name in message and "incident source" in message
    if "deg" in name:
        assert "tfsf_angle_override" in message


def test_run_has_no_override():
    assert "tfsf_angle_override" not in inspect.signature(Simulation.run).parameters


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("kind", ["normal", "bloch", "methodB"])
def test_first_job_dispatch_reachability(kind, graded, no_scan):
    sim, _ = scene(theta=0. if kind == "normal" else 30., boundary="cpml",
                   graded="x" if graded else None, method="methodB" if kind == "methodB" else "bloch")
    # Direct dispatch is the claimed earlier refusal site. No override check runs here.
    with pytest.raises(RfxUnsupported) as caught:
        sim._dispatch_plan(mode="forward", n_steps=10, num_periods=1,
                           distributed=True, emit_time_series=True)
    message = str(caught.value)
    emit("Addendum2_dispatch", kind=kind, graded=graded, exception=type(caught.value).__name__, message=message)
    assert ("graded multi-device lane" if graded else "is not supported on forward_distributed") in message


def test_first_job_closed_box_two_devices_returns():
    # Existing case's normal closed box and interior sheet; recheck after removal too.
    from tests.unit.runners.test_distributed_admission_refusals import test_closed_box_late_sheet_inside_runs
    test_closed_box_late_sheet_inside_runs("forward", 2)
    emit("Addendum2_closed_box", devices=2, admitted=True)

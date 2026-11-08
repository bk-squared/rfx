"""Public wire reference-plane entry contracts on the issue's 16 mm thru."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from rfx.errors import RfxRefusal
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
from tests.locks.test_refplane_port_waves import _build_thru, _DOMAIN, _DX, _FREQS


@pytest.mark.parametrize("lane", ["uniform", "graded", "distributed", "distributed_graded"])
@pytest.mark.parametrize("request_s", [False, True])
@pytest.mark.parametrize("port_index", [0, 1])
def test_forward_refuses_before_time_loop(lane, request_s, port_index, monkeypatch):
    sim = _build_thru(reference_plane_cells=3)
    if port_index == 1:
        sim._ports[0] = replace(sim._ports[0], reference_plane_cells=None)
    if "graded" in lane:
        sim._dz_profile = np.full(round(_DOMAIN[2] / _DX), _DX)
        assert sim._uses_nonuniform_mesh
    options = {}
    if "distributed" in lane:
        devices = jax.devices("cpu")
        if len(devices) < 2:
            pytest.skip("not reachable: fewer than two CPU devices")
        options.update(distributed=True, devices=devices[:2])
    if request_s:
        options["port_s11_freqs"] = _FREQS
    from rfx.stepping import prepared

    def forbidden(*args, **kwargs):
        pytest.fail("forward reached the time-loop entry")

    monkeypatch.setattr(prepared, "advance", forbidden)
    with pytest.raises(ValueError) as error:
        sim.forward(n_steps=1, skip_preflight=True, **options)
    assert str(error.value) == (
        "forward() does not form reference-plane S-parameters: "
        f"add_port(reference_plane_cells=...) is set on port {port_index}; "
        "use run(compute_s_params=True) for reference-plane S, "
        "or remove the option for forward()")


def test_forward_without_option_runs():
    result = _build_thru().forward(n_steps=64, port_s11_freqs=_FREQS,
                                  skip_preflight=True)
    assert np.isfinite(np.asarray(result.s_params)[:, 0]).all()


@pytest.mark.parametrize("compute_s_params", [True, None])
def test_run_partial_optin_refuses_before_time_loop(compute_s_params, monkeypatch):
    sim = _build_thru(reference_plane_cells=3)
    sim._ports[1] = replace(sim._ports[1], reference_plane_cells=None)
    from rfx.stepping import prepared

    def forbidden(*args, **kwargs):
        pytest.fail("partial opt-in run reached the time-loop entry")

    monkeypatch.setattr(prepared, "advance", forbidden)
    with pytest.raises(ValueError) as error:
        sim.run(n_steps=1, compute_s_params=compute_s_params,
                s_param_freqs=_FREQS, skip_preflight=True)
    assert str(error.value) == (
        "reference_plane_cells must be set on every wire port of the "
        "S-parameter solve: set on port(s) 0, not on port(s) 1; "
        "set it on all of them or on none")


@pytest.mark.parametrize("compute_s_params", [True, None])
def test_run_all_opted_reaches_time_loop(compute_s_params, monkeypatch):
    from rfx.stepping import prepared

    class ReachedLoop(Exception):
        pass

    def reached(*args, **kwargs):
        raise ReachedLoop

    monkeypatch.setattr(prepared, "advance", reached)
    with pytest.raises(ReachedLoop):
        _build_thru(reference_plane_cells=3).run(
            n_steps=1, compute_s_params=compute_s_params,
            s_param_freqs=_FREQS, skip_preflight=True)


@pytest.mark.parametrize("not_opted, message", [
    (1, "reference_plane_cells must be set on every wire port of the "
     "S-parameter solve: set on port(s) 0, not on port(s) 1; "
     "set it on all of them or on none"),
    (0, "reference_plane_cells must be set on every wire port of the "
     "S-parameter solve: set on port(s) 1, not on port(s) 0; "
     "set it on all of them or on none"),
])
def test_driver_partial_optin_refuses_before_time_loop(not_opted, message, monkeypatch):
    sim = _build_thru(reference_plane_cells=3)
    sim._ports[not_opted] = replace(sim._ports[not_opted], reference_plane_cells=None)
    from rfx.stepping import prepared

    def forbidden(*args, **kwargs):
        pytest.fail("partial opt-in driver reached the time-loop entry")

    monkeypatch.setattr(prepared, "advance", forbidden)
    with pytest.raises(RfxRefusal) as error:
        compute_lumped_wire_s_matrix_via_scan(
            sim, _FREQS, n_steps=1, skip_preflight=True)
    assert str(error.value) == message


def test_driver_all_opted_reaches_time_loop(monkeypatch):
    from rfx.stepping import prepared

    class ReachedLoop(Exception):
        pass

    def reached(*args, **kwargs):
        raise ReachedLoop

    monkeypatch.setattr(prepared, "advance", reached)
    with pytest.raises(ReachedLoop):
        compute_lumped_wire_s_matrix_via_scan(
            _build_thru(reference_plane_cells=3), _FREQS,
            n_steps=1, skip_preflight=True)

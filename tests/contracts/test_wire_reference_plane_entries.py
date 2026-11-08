"""Public wire reference-plane entry contracts on the issue's 16 mm thru."""

from dataclasses import replace

import jax
import numpy as np
import pytest

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

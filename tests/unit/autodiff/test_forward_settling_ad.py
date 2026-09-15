"""Forward diagnostics retain material derivatives and every dispatch route."""
from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._spec import ForwardResult
from rfx.api._sparams import settling_verdict
from rfx.sources.sources import GaussianPulse
from tests._x64_compat import enable_x64


def test_port_material_ad_matches_resolved_fd_with_forward_diagnostics(monkeypatch):
    with enable_x64():
        sim = Simulation(freq_max=30e9, domain=(.008,) * 3, dx=.001,
                         boundary="cpml", cpml_layers=2, precision="float64")
        sim.add_port((.002, .004, .004), "ez", impedance=50.,
                     waveform=GaussianPulse(f0=20e9, bandwidth=.8))
        sim.add_probe((.004, .004, .004), "ez")
        eps = jnp.ones(sim._build_grid().shape, dtype=jnp.float64)
        freqs = np.array([20e9])

        def forward(alpha):
            return sim.forward(eps_override=eps * alpha, n_steps=80,
                               port_s11_freqs=freqs, skip_preflight=True)

        def objective(alpha):
            result = forward(alpha)
            if isinstance(alpha, jax.core.Tracer):
                assert result.settling_db is None
                assert result.settling_witness["status"] == "absent"
            return jnp.real(jnp.sum(jnp.abs(result.s_params)**2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            alpha = jnp.float64(1.)
            value, grad = jax.jit(jax.value_and_grad(objective))(alpha)
            for h in (1e-3, 5e-4):
                plus, minus = float(objective(alpha + h)), float(objective(alpha - h))
                span = abs(plus - minus)
                assert span / np.spacing(max(abs(plus), abs(minus))) > 1e4
                fd = (plus - minus) / (2 * h)
                np.testing.assert_allclose(grad, fd, rtol=3e-4, atol=0)
            reported = forward(alpha)
            with monkeypatch.context() as patch:
                patch.setattr(sim, "_attach_run_settling_witness", lambda result, **kwargs: result)
                plain = forward(alpha)
        assert np.isfinite([value, grad]).all() and float(grad) != 0
        assert np.isfinite(reported.settling_db)
        assert reported.settling_probe_info == ((0, 2),)
        np.testing.assert_array_equal(reported.time_series, plain.time_series)
        np.testing.assert_array_equal(reported.s_params, plain.s_params)


@pytest.mark.parametrize("lane", ["fwd_uniform", "fwd_nonuniform", "fwd_distributed_nu"])
@pytest.mark.parametrize("recorded", [True, False])
def test_each_public_forward_dispatch_exposes_the_probe_witness(lane, recorded, monkeypatch):
    """Intercept the backend: this checks dispatch, not distributed physics."""
    monkeypatch.setattr("rfx.api._execute._DISTRIBUTED_FIRST_CALL_WARNED", False)
    sim = Simulation(freq_max=10e9, domain=(.008,) * 3, dx=.001,
                     boundary="cpml", cpml_layers=2)
    sim.add_source((.004,) * 3, "ez", amplitude_kind="field")
    sim.add_probe((.004, .004, .005), "ez")
    sim.add_dft_plane_probe(axis="z", coordinate=.004, component="ez", n_freqs=1)
    plan = SimpleNamespace(lane=lane, n_steps=100)
    monkeypatch.setattr(sim, "_dispatch_plan", lambda **kwargs: plan)
    records = jnp.ones((100, 1 if recorded else 0), dtype=jnp.float32)
    calls = []

    def backend(*args, **kwargs):
        calls.append(kwargs)
        return ForwardResult(time_series=records)

    target = {
        "fwd_uniform": "_forward_from_materials",
        "fwd_nonuniform": "_forward_nonuniform_from_materials",
        "fwd_distributed_nu": "_forward_distributed_nonuniform_from_materials",
    }[lane]
    monkeypatch.setattr(sim, target, backend)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.forward(n_steps=100, skip_preflight=True,
                             distributed=lane == "fwd_distributed_nu")
    assert len(calls) == 1 and calls[0]["n_steps"] == 100
    assert settling_verdict(result.settling_db) == ("fail" if recorded else "absent")
    assert result.settling_probe_info == (((0, 2),) if recorded else ())
    scoped = [str(w.message) for w in caught
              if "witness FAILED" in str(w.message) or "no ring-down settling witness" in str(w.message)]
    assert len(scoped) == 1 and "forward()" in scoped[0]
    np.testing.assert_array_equal(result.time_series, records)

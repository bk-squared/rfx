"""An accelerator query must not change a closed box's solved fields."""

import jax
import numpy as np

from rfx import Simulation
from rfx.backend import backend_class
from rfx.sources.sources import GaussianPulse


def _closed_box():
    sim = Simulation(freq_max=20e9, domain=(0.0137, 0.0113, 0.0096),
                     dx=1e-3, boundary="pec", cpml_layers=0, snap="declared")
    sim.add_source((0.0034, 0.0042, 0.0027), "ez",
                   waveform=GaussianPulse(f0=10e9), amplitude_kind="field")
    sim.add_probe((0.0092, 0.0073, 0.0061), "ez")
    return sim


def test_gpu_query_closed_box_keeps_standard_lane_and_cpu_bits(monkeypatch):
    import rfx.simulation as simulation

    with jax.default_device(jax.devices("cpu")[0]):
        assert not backend_class().accelerator
        plain = _closed_box().run(n_steps=40, compute_s_params=False,
                                  skip_preflight=True)
        trace_calls = 0
        original = simulation.update_he_fast

        def counted(*args, **kwargs):
            nonlocal trace_calls
            trace_calls += 1
            return original(*args, **kwargs)

        with monkeypatch.context() as forced:
            forced.setattr(jax, "default_backend", lambda: "gpu")
            forced.setattr(simulation, "update_he_fast", counted)
            assert backend_class().accelerator
            accelerated = _closed_box().run(n_steps=40, compute_s_params=False,
                                            skip_preflight=True)
            jax.block_until_ready(accelerated.time_series)
        assert trace_calls == 0, "accelerator query reached update_he_fast"

    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        expected = np.asarray(getattr(plain.state, component))
        actual = np.asarray(getattr(accelerated.state, component))
        assert actual.dtype == expected.dtype == np.dtype("float32")
        assert np.array_equal(actual, expected), component
    expected = np.asarray(plain.time_series)
    actual = np.asarray(accelerated.time_series)
    assert actual.dtype == expected.dtype == np.dtype("float32")
    assert np.any(expected != 0), "the physical source must reach the probe"
    assert np.array_equal(actual, expected), "probe time series"

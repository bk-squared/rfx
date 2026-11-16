"""Physics anchor for the public coax one-port settling witness."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse


def _run(n_steps, termination="short"):
    sim = Simulation(
        freq_max=10e9, domain=(0.008, 0.008, 0.030), dx=0.0004, boundary="cpml",
    )
    sim.add_coaxial_port(
        (0.004, 0.004, 0.015), face="top", pin_length=0.005,
        pin_radius=0.000635, outer_radius=0.002055,
        waveform=GaussianPulse(f0=5e9, bandwidth=1.2),
    )
    return sim.compute_coaxial_line_reflection(
        termination=termination, n_steps=n_steps, freqs=jnp.linspace(4e9, 6e9, 5),
    )


@pytest.mark.slow_physics
def test_short_line_settling_after_feed_absorbs_return():
    """A short sends the pulse back; the feed resistor absorbs it over round
    trips of 0.263 ns. At 700 steps (0.53 ns), the reflected pulse is still on
    the planes, so the witness must not pass and settling_db must be nan.
    At 3000 steps (2.29 ns), it must read at most -40 dB with status "pass".
    """
    results = {n_steps: _run(n_steps) for n_steps in (700, 3000)}
    expected_channels = {f"coax/plane{k}/V" for k in range(12)} | {"probe0"}
    for n_steps, result in results.items():
        print(
            f"short n_steps={n_steps}: settling_db={result.settling_db}, "
            f"status={result.settling_witness['status']}"
        )
    for n_steps, result in results.items():
        assert set(result.settling_witness["per_record_db"]) == expected_channels, n_steps

    in_transit = results[700]
    assert in_transit.settling_witness["status"] != "pass"
    assert np.isnan(in_transit.settling_db), in_transit.settling_db

    settled = results[3000]
    assert settled.settling_db <= -40.0, settled.settling_db
    assert settled.settling_witness["status"] == "pass"


@pytest.mark.slow_physics
def test_matched_line_settling_uses_source_end():
    """The matched load has absorbed the pulse by the 0.42 ns source end, so
    the 595-step line has settled. Its identification window [T/2, T] starts
    before the source end, so only source-end handling decides this case.
    """
    result = _run(595, termination="matched")
    print(
        f"matched n_steps=595: settling_db={result.settling_db}, "
        f"status={result.settling_witness['status']}"
    )
    expected_channels = {f"coax/plane{k}/V" for k in range(12)} | {"probe0"}
    assert set(result.settling_witness["per_record_db"]) == expected_channels
    assert result.settling_db <= -40.0, result.settling_db
    assert result.settling_witness["status"] == "pass"

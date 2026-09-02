"""Post-run advisory coverage for issues #332 and #336."""

import warnings

from rfx import Simulation
from rfx.sources.sources import GaussianPulse


def _run(*, boundary="cpml", num_periods=0.1, amplitude=1.0):
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.012, 0.012, 0.012),
        dx=1.0e-3,
        boundary=boundary,
        cpml_layers=2,
    )
    waveform = GaussianPulse(
        f0=5.0e9,
        bandwidth=0.8,
        amplitude=amplitude,
    )
    sim.add_source((0.006, 0.006, 0.006), "ez", waveform=waveform)
    sim.add_probe((0.007, 0.006, 0.006), "ez")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = sim.run(
            num_periods=num_periods,
            compute_s_params=False,
            skip_preflight=True,
        )
    return result, [str(item.message) for item in recorded]


def test_g1_warns_when_open_pulse_tail_is_hot():
    _, messages = _run(num_periods=0.1)

    assert any("truncated" in message for message in messages)


def test_g1_is_silent_after_generous_ringdown():
    _, messages = _run(num_periods=20.0)

    assert not any("truncated" in message for message in messages)


def test_g1_skips_pec_closed_boundary_with_hot_tail():
    _, messages = _run(boundary="pec", num_periods=0.1)

    assert not any("truncated" in message for message in messages)


def test_g6_warns_when_source_amplitude_is_exactly_zero():
    _, messages = _run(num_periods=1.0, amplitude=0.0)

    assert any("no field energy was recorded" in message for message in messages)


def test_g6_is_silent_for_normal_coupling():
    _, messages = _run(num_periods=20.0)

    assert not any("no field energy was recorded" in message for message in messages)

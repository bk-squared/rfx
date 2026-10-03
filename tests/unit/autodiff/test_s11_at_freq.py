"""The time-gated objective was removed in 2.0 (#1448)."""
import pytest
from rfx import optimize_objectives


def test_time_gating_removed():
    with pytest.raises(AttributeError, match="minimize_s11_at_freq_wave_decomp.*port_s11_freqs"):
        optimize_objectives.minimize_s11_at_freq(10e9)

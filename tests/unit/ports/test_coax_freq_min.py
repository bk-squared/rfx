"""Declared coaxial band defaults, explicit values and refusals."""

from __future__ import annotations

import pytest

from rfx import Simulation
from rfx.errors import RfxRefusal


def test_default_freq_min():
    sim = Simulation(freq_max=10e9, domain=(0.020, 0.020, 0.020), boundary="cpml")
    sim.add_coaxial_port((0.010, 0.010, 0.015))
    assert sim._coaxial_ports[0].freq_min == 10e9 / 10


def test_explicit_freq_min():
    sim = Simulation(freq_max=10e9, domain=(0.020, 0.020, 0.020), boundary="cpml")
    sim.add_coaxial_port((0.010, 0.010, 0.015), freq_min=1e9)
    assert sim._coaxial_ports[0].freq_min == 1e9


@pytest.mark.parametrize("freq_min", [0, -1e9, float("nan"), 10e9, 20e9])
def test_refuses_freq_min_outside_band(freq_min):
    sim = Simulation(freq_max=10e9, domain=(0.020, 0.020, 0.020), boundary="cpml")
    with pytest.raises(RfxRefusal) as excinfo:
        sim.add_coaxial_port((0.010, 0.010, 0.015), freq_min=freq_min)
    assert "freq_min=" in str(excinfo.value)

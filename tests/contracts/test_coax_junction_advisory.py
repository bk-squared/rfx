"""A plate-fed coaxial port has no false pin-short advisory."""

import warnings

import pytest

from rfx import Simulation
from rfx.geometry.csg import Box


def a1():
    sim = Simulation(freq_max=4.5e9, domain=(.040, .040, .120),
                     dx=.001, boundary="cpml", precision="float32")
    sim.add(Box((.005, .005, .109), (.035, .035, .110)), material="pec")
    sim.add_coaxial_port(position=(.020, .020, .110), face="bottom",
                         pin_radius=.001, outer_radius=.006, impedance=50.)
    return sim


@pytest.mark.parametrize("route", ["standalone", "run"])
def test_no_false_pin_short_advisory(route):
    sim = a1()
    code = "coaxial_port_junction_short"
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        if route == "standalone":
            report = sim.preflight()
        else:
            sim.run(n_steps=1, compute_s_params=False)
    assert not any(
        getattr(item.message, "code", None) == code
        or "The pin is terminated in a short by registered geometry" in str(item.message)
        for item in emitted
    )
    if route == "standalone":
        assert not any(item.code == code for item in report)
